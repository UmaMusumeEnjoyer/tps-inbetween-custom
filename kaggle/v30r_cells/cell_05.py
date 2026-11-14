# [CELL 5]: Kiến Trúc Mạng TPS-Inbetween V3.0-Remastered R2 (100% V2.6 Match + Time-Symmetric FBFC)
import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv(in_p, out_p, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_p, out_p, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_p)
    )

def deconv(in_p, out_p, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_p, out_p, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.PReLU(out_p)
    )

class Conv2(nn.Module):
    def __init__(self, in_p, out_p, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_p, out_p, 3, stride, 1)
        self.conv2 = conv(out_p, out_p, 3, 1, 1)

    def forward(self, x):
        return self.conv2(self.conv1(x))

class Contextnet(nn.Module):
    def __init__(self, indim=1, c=16):
        super(Contextnet, self).__init__()
        self.conv1 = Conv2(indim, c)
        self.conv2 = Conv2(c, 2*c)
        self.conv3 = Conv2(2*c, 4*c)
        self.conv4 = Conv2(4*c, 8*c)

    def forward(self, x, flow):
        x = self.conv1(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode='bilinear', align_corners=False) * 0.5
        f1 = warp_normalized_flow(x, flow)
        x = self.conv2(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode='bilinear', align_corners=False) * 0.5
        f2 = warp_normalized_flow(x, flow)
        x = self.conv3(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode='bilinear', align_corners=False) * 0.5
        f3 = warp_normalized_flow(x, flow)
        x = self.conv4(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode='bilinear', align_corners=False) * 0.5
        f4 = warp_normalized_flow(x, flow)
        return [f1, f2, f3, f4]

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(*[conv(c, c) for _ in range(8)])
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    def forward(self, x, flow, scale):
        if scale != 1:
            x = F.interpolate(x, scale_factor=1./scale, mode='bilinear', align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1./scale, mode='bilinear', align_corners=False) * (1./scale)
            x = torch.cat((x, flow), 1)
        x = self.conv0(x.contiguous())
        x = self.convblock(x) + x
        with torch.amp.autocast('cuda', enabled=False):
            tmp = self.lastconv(x.float().contiguous())
            tmp = F.interpolate(tmp, scale_factor=scale * 2, mode='bilinear', align_corners=False)
            flow = tmp[:, :4] * scale * 2
            mask = tmp[:, 4:5]
        return flow, mask

class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(6, c=240)
        self.block1 = IFBlock(13+4, c=150)
        self.block2 = IFBlock(13+4, c=90)

    def forward(self, I0t, I1t, scale=[4, 2, 1]):
        img0 = I0t.repeat(1, 3, 1, 1)
        img1 = I1t.repeat(1, 3, 1, 1)
        warped_i0 = I0t
        warped_i1 = I1t
        flow = None
        mask = None
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow is not None:
                flow_d, mask_d = stu[i](torch.cat((img0, img1, warped_i0.repeat(1, 3, 1, 1), warped_i1.repeat(1, 3, 1, 1), mask), 1), flow, scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            if i < 2:
                warped_i0 = warp_pixel_flow(I0t, flow[:, :2])
                warped_i1 = warp_pixel_flow(I1t, flow[:, 2:4])
        return flow, torch.sigmoid(mask)

class AdaptiveStrokeFusionHead(nn.Module):
    """
    Đầu hòa trộn nét thích ứng từ chuẩn vàng V2.6:
    Dự đoán blend_mask tối ưu giữa 2 luồng warped frame I0t và I1t.
    """
    def __init__(self):
        super(AdaptiveStrokeFusionHead, self).__init__()
        self.net = nn.Sequential(
            conv(8, 32, 3, 1, 1),
            conv(32, 32, 3, 1, 1),
            nn.Conv2d(32, 1, 3, 1, 1)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, fine_I0t, fine_I1t, flow, ifnet_mask, t_vals):
        b, _, h, w = fine_I0t.shape
        t_map = t_vals.repeat(1, 1, h, w)
        inp = torch.cat([fine_I0t, fine_I1t, flow, ifnet_mask, t_map], dim=1)
        residual_logit = self.net(inp)
        prior_logit = torch.logit(torch.clamp(1.0 - t_map, 1e-4, 1.0 - 1e-4))
        blend_mask = torch.sigmoid(prior_logit + residual_logit)
        return blend_mask

class TimeSymmetricFBFCEngine(nn.Module):
    """
    Động cơ Forward-Backward Flow Consistency (FBFC) Đối Xứng Thời Gian trên Khung t (Zero Extra Weights):
    Đo đạc độ khả nghịch hình học từ tổng trường chuyển động về 2 keyframes theo tọa độ pixel:
    flow_mismatch = (1 - t)*s_t0 + t*s_t1
    delta_flow = ||flow_mismatch||_2
    conf_fbfc = exp(-max(0, delta_flow - tau_safe)^2 / (2 * sigma^2))
    """
    def __init__(self, tau_safe=2.0, sigma=4.0):
        super(TimeSymmetricFBFCEngine, self).__init__()
        self.tau_safe = tau_safe
        self.sigma = sigma

    def forward(self, s_t0, s_t1, t_vals):
        b, _, h, w = s_t0.shape
        t_b = t_vals.view(b, 1, 1, 1)
        flow_mismatch = (1.0 - t_b) * s_t0 + t_b * s_t1
        delta_flow = torch.sqrt(torch.sum(flow_mismatch ** 2, dim=1, keepdim=True) + 1e-10)
        conf_fbfc = torch.exp(-torch.clamp(delta_flow - self.tau_safe, min=0.0)**2 / (2.0 * (self.sigma ** 2)))
        return conf_fbfc.detach(), delta_flow.detach()

class SynthesisUnet_v30R(nn.Module):
    """
    UNet tổng hợp nét tương thích 100% với V2.6:
    detail_add = sigmoid(conv_add(x)) * 0.25 bảo toàn biên độ cực tiểu phẳng.
    """
    def __init__(self, indim=9, c=16):
        super(SynthesisUnet_v30R, self).__init__()
        self.down0 = Conv2(indim, 2*c)
        self.down1 = Conv2(4*c, 4*c)
        self.down2 = Conv2(8*c, 8*c)
        self.down3 = Conv2(16*c, 16*c)
        self.up0 = deconv(32*c, 8*c)
        self.up1 = deconv(16*c, 4*c)
        self.up2 = deconv(8*c, 2*c)
        self.up3 = deconv(4*c, c)

        self.conv_erase = nn.Conv2d(c, 1, 3, 1, 1)
        self.conv_add = nn.Conv2d(c, 1, 3, 1, 1)

        nn.init.zeros_(self.conv_erase.weight)
        nn.init.constant_(self.conv_erase.bias, 2.0)

        nn.init.zeros_(self.conv_add.weight)
        nn.init.constant_(self.conv_add.bias, -2.0)

    def forward(self, img0, img1, flow0, flow1, warped0, warped1, mask, c0, c1):
        s0 = self.down0(torch.cat((img0, img1, warped0, warped1, flow0, flow1, mask), 1))
        s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
        s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
        s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
        x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))
        x = self.up1(torch.cat((x, s2), 1))
        x = self.up2(torch.cat((x, s1), 1))
        x = self.up3(torch.cat((x, s0), 1))

        gate_erase = torch.sigmoid(self.conv_erase(x))
        detail_add = torch.sigmoid(self.conv_add(x))
        return gate_erase, detail_add

class TPSInbetweenFinetune_v30R(nn.Module):
    """
    Kiến trúc TPS-Inbetween V3.0-Remastered R2:
    - Kế thừa 100% 168 weight tensors của V2.6 (Contextnet + IFNet + asf_head + unet).
    - Tích hợp Time-Symmetric FBFC Engine tính toán sự nhất quán chuyển động trên khung t.
    - Hoàn toàn tương thích với Warm-Start V2.6 Gold.
    """
    def __init__(self):
        super(TPSInbetweenFinetune_v30R, self).__init__()
        self.encoder = Contextnet(indim=1)
        self.IFNet = IFNet()
        self.asf_head = AdaptiveStrokeFusionHead()
        self.fbfc = TimeSymmetricFBFCEngine(tau_safe=2.0, sigma=4.0)
        self.unet = SynthesisUnet_v30R(indim=9)

    def forward(self, x0, x1, t_vals, tps_precomputed=None):
        if x0.dtype == torch.uint8:
            x0 = (255.0 - x0.float()) * (1.0 / 255.0)
        if x1.dtype == torch.uint8:
            x1 = (255.0 - x1.float()) * (1.0 / 255.0)

        b, c, h, w = x0.shape
        dev = x0.device
        t_b = t_vals.view(b, 1, 1, 1).to(dev)

        # 1. Coarse TPS Motion
        if tps_precomputed is not None:
            c0_b, th01_b, c1_b, th10_b, valid_b = tps_precomputed
            c0_b = c0_b.to(dev)
            th01_b = th01_b.to(dev)
            c1_b = c1_b.to(dev)
            th10_b = th10_b.to(dev)
            valid_b = valid_b.to(dev)

            coarse_flow01 = tps_flow_from_theta_batch(th01_b, c0_b, valid_b, h, w, dev)
            coarse_flow10 = tps_flow_from_theta_batch(th10_b, c1_b, valid_b, h, w, dev)
        else:
            coarse_flow01 = torch.zeros((b, 2, h, w), device=dev)
            coarse_flow10 = torch.zeros((b, 2, h, w), device=dev)

        coarse_flow0t = coarse_flow01 * (t_b ** 2) - (1.0 - t_b) * t_b * coarse_flow10
        coarse_flow1t = coarse_flow10 * ((1.0 - t_b) ** 2) - t_b * (1.0 - t_b) * coarse_flow01

        coarse_I0t = warp_normalized_flow(x0, coarse_flow0t)
        coarse_I1t = warp_normalized_flow(x1, coarse_flow1t)

        # 2. Fine Motion Residuals & Context Features
        feats0 = self.encoder(x0, coarse_flow0t)
        feats1 = self.encoder(x1, coarse_flow1t)
        flow_refine, ifnet_mask = self.IFNet(coarse_I0t, coarse_I1t)

        fine_I0t = warp_pixel_flow(coarse_I0t, flow_refine[:, :2])
        fine_I1t = warp_pixel_flow(coarse_I1t, flow_refine[:, 2:4])

        # 3. Time-Symmetric FBFC Engine (Tính theo pixel chuẩn hóa)
        u_px0 = coarse_flow0t[:, 0:1] * (w - 1.0)
        v_px0 = coarse_flow0t[:, 1:2] * (h - 1.0)
        u_px1 = coarse_flow1t[:, 0:1] * (w - 1.0)
        v_px1 = coarse_flow1t[:, 1:2] * (h - 1.0)
        s_t0_px = torch.cat([u_px0, v_px0], dim=1) + flow_refine[:, :2]
        s_t1_px = torch.cat([u_px1, v_px1], dim=1) + flow_refine[:, 2:4]
        conf_fbfc, delta_flow = self.fbfc(s_t0_px, s_t1_px, t_vals)

        # 4. Adaptive Stroke Fusion Head
        blend_mask = self.asf_head(fine_I0t, fine_I1t, flow_refine, ifnet_mask, t_b)
        base_interp = blend_mask * fine_I0t + (1.0 - blend_mask) * fine_I1t

        # 5. Synthesis UNet (Bảo toàn chuẩn vàng V2.6: Nhận coarse_flow0t, coarse_flow1t)
        gate_erase, detail_add = self.unet(
            x0, x1,
            coarse_flow0t, coarse_flow1t,
            fine_I0t, fine_I1t, ifnet_mask, feats0, feats1
        )
        pred_t = torch.clamp(gate_erase * base_interp + detail_add, 0.0, 1.0)

        if self.training:
            return pred_t, blend_mask, conf_fbfc, delta_flow, fine_I0t, fine_I1t
        return pred_t, blend_mask, conf_fbfc

# =============================================================================
# [PRODUCTION HEAD]: Soft-Knee Gamma RGBA Lineart Exporter (Zero Yellow Bleed)
# =============================================================================
def export_pure_lineart_rgba(pred_ink_tensor, tau_noise=0.020, gamma=0.78):
    """
    Chuyển đổi tensor nét vẽ thành mảng numpy RGBA trong suốt 4-kênh:
    - Triệt tiêu 100% hiện tượng ố vàng nền / sai khác màu giấy scan.
    - Điểm gãy nền tau_noise = 0.020: Mọi nhiễu quét giấy có Alpha = 0 tuyệt đối.
    - Vùng nét mảnh [0.020, 0.060]: Ánh xạ qua hàm smoothstep phi tuyến giúp ngọn tóc
      mảnh không bị tàng hình mà hiển thị sắc nét tự nhiên.
    - Nét vẽ: RGB = (0, 0, 0) Đen thuần khiết.
    """
    if isinstance(pred_ink_tensor, torch.Tensor):
        pred = pred_ink_tensor.detach().squeeze().clamp(0.0, 1.0).cpu().numpy()
    else:
        pred = np.clip(pred_ink_tensor.squeeze(), 0.0, 1.0)

    h, w = pred.shape
    alpha = np.zeros((h, w), dtype=np.uint8)

    mask = pred >= tau_noise
    if mask.any():
        p_val = pred[mask]
        # Smoothstep gate trên dải [tau_noise, 0.060]
        t_knee = np.clip((p_val - tau_noise) / 0.040, 0.0, 1.0)
        smooth_gate = t_knee * t_knee * (3.0 - 2.0 * t_knee)
        
        norm_val = np.clip((p_val - tau_noise) / max(1e-4, 1.0 - tau_noise), 0.0, 1.0)
        alpha_f = smooth_gate * np.power(norm_val, gamma) * 255.0
        alpha[mask] = np.clip(alpha_f, 0, 255).astype(np.uint8)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = 0        # B (Pure Ink)
    rgba[:, :, 1] = 0        # G
    rgba[:, :, 2] = 0        # R
    rgba[:, :, 3] = alpha    # A

    return rgba

device = globals().get('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
if 'base_model' not in globals():
    base_model = TPSInbetweenFinetune_v30R().to(device)

print("=" * 80)
print(f"[✓] CELL 5 hoàn tất: Mô hình TPSInbetweenFinetune_v30R R2 sẵn sàng trên {device}.")
print("=" * 80)
sys.stdout.flush()
