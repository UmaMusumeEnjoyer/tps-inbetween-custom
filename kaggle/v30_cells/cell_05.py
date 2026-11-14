# [CELL 5]: Kiến Trúc Mạng TPS-Inbetween V3.0 (Asymmetric Stroke & Occlusion Engine)
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
        x = torch.cat((I0t.repeat(1, 3, 1, 1), I1t.repeat(1, 3, 1, 1)), dim=1)
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        flow_list = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow is not None:
                flow_d, mask_d = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow, scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp_pixel_flow(img0, flow[:, :2])
            warped_img1 = warp_pixel_flow(img1, flow[:, 2:4])
        return flow_list[2], mask_list[2]

# =============================================================================
# [CƠ CHẾ CỐT LÕI 1 - V3.0]: Bản Đồ Bất Đối Xứng Đặc Trưng & Occlusion Confidence
# =============================================================================
class AsymmetricFeatureAnalyzer(nn.Module):
    def __init__(self, in_channels=34):  # 1 (I0) + 1 (I1) + 16 (f0[0]) + 16 (f1[0])
        super(AsymmetricFeatureAnalyzer, self).__init__()
        self.net = nn.Sequential(
            conv(in_channels, 32, 3, 1, 1),
            conv(32, 16, 3, 1, 1),
            nn.Conv2d(16, 2, 3, 1, 1)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -1.5)

    def forward(self, fine_I0t, fine_I1t, feat0_sub, feat1_sub):
        b, _, h, w = fine_I0t.shape
        f0_up = F.interpolate(feat0_sub, size=(h, w), mode='bilinear', align_corners=False)
        f1_up = F.interpolate(feat1_sub, size=(h, w), mode='bilinear', align_corners=False)
        inp = torch.cat([fine_I0t, fine_I1t, f0_up, f1_up], dim=1)
        occ_logits = self.net(inp)
        occ_confidence = torch.sigmoid(occ_logits)
        return occ_confidence

# =============================================================================
# [CƠ CHẾ CỐT LÕI 2 - V3.0]: Lan Truyền Vận Tốc Theo Đường Biên Bao (Contour-Guided Flow)
# =============================================================================
class ContourGuidedFlowPropagator(nn.Module):
    def __init__(self):
        super(ContourGuidedFlowPropagator, self).__init__()
        self.conv_in = conv(7, 32, 3, 1, 1)  # 1 (I0) + 1 (I1) + 4 (flow) + 1 (shared contour)
        self.d1 = conv(32, 32, 3, 1, 1, dilation=1)
        self.d2 = conv(32, 32, 3, 1, 2, dilation=2)
        self.d4 = conv(32, 32, 3, 1, 4, dilation=4)
        self.conv_out = nn.Conv2d(32, 4, 3, 1, 1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, coarse_I0t, coarse_I1t, flow_refine):
        shared_contour = torch.minimum(coarse_I0t, coarse_I1t)
        inp = torch.cat([coarse_I0t, coarse_I1t, flow_refine, shared_contour], dim=1)
        x = self.conv_in(inp)
        x = self.d1(x) + self.d2(x) + self.d4(x)
        delta_flow = self.conv_out(x)
        return flow_refine + delta_flow, delta_flow

# =============================================================================
# [CƠ CHẾ CỐT LÕI 3 - V3.0]: Cổng Hiển Thị Kép Độc Lập (Phá Bỏ Bẫy Tổng Bằng 1)
# =============================================================================
class AsymmetricStrokeFusionHead_v3(nn.Module):
    def __init__(self):
        super(AsymmetricStrokeFusionHead_v3, self).__init__()
        self.net = nn.Sequential(
            conv(10, 32, 3, 1, 1),
            conv(32, 32, 3, 1, 1),
            nn.Conv2d(32, 2, 3, 1, 1)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.gamma_occ = nn.Parameter(torch.full((1, 2, 1, 1), 0.35))

    def forward(self, fine_I0t, fine_I1t, flow, ifnet_mask, t_vals, occ_confidence):
        b, _, h, w = fine_I0t.shape
        t_map = t_vals.repeat(1, 1, h, w)
        inp = torch.cat([fine_I0t, fine_I1t, flow, ifnet_mask, t_map, occ_confidence], dim=1)
        residual_logits = self.net(inp)

        prior_0 = torch.logit(torch.clamp(1.0 - t_map, 1e-4, 1.0 - 1e-4))
        prior_1 = torch.logit(torch.clamp(t_map, 1e-4, 1.0 - 1e-4))
        prior_logits = torch.cat([prior_0, prior_1], dim=1)

        vis_logits = prior_logits + residual_logits + self.gamma_occ * occ_confidence
        visibility_gates = torch.sigmoid(vis_logits)
        v0 = visibility_gates[:, 0:1]
        v1 = visibility_gates[:, 1:2]
        return v0, v1

class SynthesisUnet_v3(nn.Module):
    def __init__(self, indim=9, c=16):
        super(SynthesisUnet_v3, self).__init__()
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

class TPSInbetweenFinetune_v3(nn.Module):
    """
    Kiến trúc TPS-Inbetween V3.0 (Major Architectural Upgrade).
    Tương thích 100% Warm-Start Backbone V2.6 + 3 Nhánh Đột Phá V3:
    1. AsymmetricFeatureAnalyzer (Occlusion Confidence)
    2. ContourGuidedFlowPropagator (Biên bao dẫn hướng)
    3. AsymmetricStrokeFusionHead_v3 (Decoupled Visibility Gates V0, V1)
    4. SynthesisUnet_v3 (Gated Silhouette Erase & Hair-tip Reconnection)
    """
    def __init__(self):
        super(TPSInbetweenFinetune_v3, self).__init__()
        self.encoder = Contextnet(indim=1)
        self.IFNet = IFNet()
        self.asym_analyzer = AsymmetricFeatureAnalyzer(in_channels=34)
        self.contour_propagator = ContourGuidedFlowPropagator()
        self.asf_head = AsymmetricStrokeFusionHead_v3()
        self.unet = SynthesisUnet_v3(indim=9)

        self._init_v3_weights()

    def _init_v3_weights(self):
        for m in [self.asym_analyzer, self.contour_propagator, self.asf_head]:
            for mod in m.modules():
                if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(mod.weight, a=0.25, mode='fan_out', nonlinearity='leaky_relu')
                    if mod.bias is not None:
                        nn.init.zeros_(mod.bias)
        # Small normal trên các lớp output để phá bẫy đối xứng mà vẫn giữ ổn định
        for head in [self.asf_head.net[-1], self.contour_propagator.conv_out]:
            if hasattr(head, 'weight') and head.weight is not None:
                nn.init.normal_(head.weight, mean=0.0, std=1e-3)
                if head.bias is not None:
                    nn.init.zeros_(head.bias)

    def forward(self, x0, x1, t_vals, kps_input=None, kps_counts=None, max_kps=80, tps_precomputed=None):
        if x0.dtype == torch.uint8:
            x0 = (255.0 - x0.float()) * (1.0 / 255.0)
        if x1.dtype == torch.uint8:
            x1 = (255.0 - x1.float()) * (1.0 / 255.0)

        b, c, h, w = x0.shape
        dev = x0.device
        t_vals = t_vals.view(b, 1, 1, 1).to(dev)

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
            flow01_list, flow10_list = [], []
            with torch.no_grad():
                for i in range(b):
                    kps = kps_input[i] if kps_input is not None else None
                    c0_p, th01_p, c1_p, th10_p, v = solve_tps_pair_cpu(kps, max_kps=max_kps, lambd=0.5)
                    c0_p = c0_p.unsqueeze(0).to(dev)
                    th01_p = th01_p.unsqueeze(0).to(dev)
                    c1_p = c1_p.unsqueeze(0).to(dev)
                    th10_p = th10_p.unsqueeze(0).to(dev)
                    v_t = v.view(1, 1, 1, 1).to(dev)

                    f01 = tps_flow_from_theta_batch(th01_p, c0_p, v_t, h, w, dev)
                    f10 = tps_flow_from_theta_batch(th10_p, c1_p, v_t, h, w, dev)
                    flow01_list.append(f01)
                    flow10_list.append(f10)

            coarse_flow01 = torch.cat(flow01_list, dim=0)
            coarse_flow10 = torch.cat(flow10_list, dim=0)

        coarse_flow0t = coarse_flow01 * (t_vals ** 2) - (1.0 - t_vals) * t_vals * coarse_flow10
        coarse_flow1t = coarse_flow10 * ((1.0 - t_vals) ** 2) - t_vals * (1.0 - t_vals) * coarse_flow01

        # 2. Coarse Warping
        coarse_I0t = warp_normalized_flow(x0, coarse_flow0t)
        coarse_I1t = warp_normalized_flow(x1, coarse_flow1t)

        # 3. Fine Motion & Feature Extraction
        feats0 = self.encoder(x0, coarse_flow0t)
        feats1 = self.encoder(x1, coarse_flow1t)
        flow_raw, ifnet_mask = self.IFNet(coarse_I0t, coarse_I1t)

        # [V3.0 UPGRADE 1]: Lan truyền vận tốc theo đường biên bao (Contour-Guided Flow)
        flow_guided, delta_flow = self.contour_propagator(coarse_I0t, coarse_I1t, flow_raw)

        # 4. Fine Warping với Flow được dẫn hướng
        fine_I0t = warp_pixel_flow(coarse_I0t, flow_guided[:, :2])
        fine_I1t = warp_pixel_flow(coarse_I1t, flow_guided[:, 2:4])

        # [V3.0 UPGRADE 2]: Phân tích bất đối xứng đặc trưng & Occlusion Confidence
        occ_conf = self.asym_analyzer(fine_I0t, fine_I1t, feats0[0], feats1[0])

        # [V3.0 UPGRADE 3]: Cổng hiển thị kép độc lập (V0, V1) in [0, 1]^2
        v0, v1 = self.asf_head(fine_I0t, fine_I1t, flow_guided, ifnet_mask, t_vals, occ_conf)
        base_interp = torch.clamp(v0 * fine_I0t + v1 * fine_I1t, 0.0, 1.0)

        # 5. Gated Silhouette Erase & Detail Reconnection (UNet v3.0)
        gate_erase, detail_add = self.unet(
            x0, x1,
            coarse_flow0t, coarse_flow1t,
            fine_I0t, fine_I1t, ifnet_mask, feats0, feats1
        )
        pred_t = torch.clamp(gate_erase * base_interp + detail_add, 0.0, 1.0)
        if self.training:
            return pred_t, (v0, v1), occ_conf, delta_flow, fine_I0t, fine_I1t
        return pred_t, (v0, v1), occ_conf

# Alias tương thích
base_model = TPSInbetweenFinetune_v3().to(device)
print("=" * 80)
print("[✓] Mô hình TPSInbetweenFinetune_v3 (Asymmetric Stroke & Occlusion Engine) sẵn sàng.")
print("=" * 80)
sys.stdout.flush()
