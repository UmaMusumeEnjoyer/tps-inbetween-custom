# [CELL 5]: Kiến Trúc Mạng TPS-Inbetween V3.2 (Asymmetric Stroke & Flat-Minima Gated Head)
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

# =============================================================================
# [MODULE 1 - V3.2]: Phân Tích Bất Đối Xứng Đặc Trưng & Occlusion Confidence
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
        hs, ws = feat0_sub.shape[-2], feat0_sub.shape[-1]
        i0_sub = F.interpolate(fine_I0t, size=(hs, ws), mode='bilinear', align_corners=False)
        i1_sub = F.interpolate(fine_I1t, size=(hs, ws), mode='bilinear', align_corners=False)
        inp = torch.cat([i0_sub, i1_sub, feat0_sub, feat1_sub], dim=1)
        occ_logits_sub = self.net(inp)
        occ_logits = F.interpolate(occ_logits_sub, size=(h, w), mode='bilinear', align_corners=False)
        occ_confidence = torch.sigmoid(occ_logits)
        return occ_confidence

# =============================================================================
# [MODULE 2 - V3.2]: Lightweight Contour-Guided Flow Propagator (Depthwise-Dilated)
# =============================================================================
class ContourGuidedFlowPropagator_v31(nn.Module):
    def __init__(self):
        super(ContourGuidedFlowPropagator_v31, self).__init__()
        self.conv_in = conv(7, 24, 3, 1, 1)  # 1 (I0) + 1 (I1) + 4 (flow) + 1 (shared contour)
        self.dw_d1 = nn.Sequential(
            nn.Conv2d(24, 24, 3, 1, 1, dilation=1, groups=24, bias=False),
            nn.Conv2d(24, 24, 1, bias=True),
            nn.PReLU(24)
        )
        self.dw_d2 = nn.Sequential(
            nn.Conv2d(24, 24, 3, 1, 2, dilation=2, groups=24, bias=False),
            nn.Conv2d(24, 24, 1, bias=True),
            nn.PReLU(24)
        )
        self.conv_out = nn.Conv2d(24, 4, 3, 1, 1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, coarse_I0t, coarse_I1t, flow_refine):
        b, _, h, w = coarse_I0t.shape
        c0_s = F.avg_pool2d(coarse_I0t, 2, 2)
        c1_s = F.avg_pool2d(coarse_I1t, 2, 2)
        fl_s = F.avg_pool2d(flow_refine, 2, 2) * 0.5
        shared_contour = torch.minimum(c0_s, c1_s)
        inp = torch.cat([c0_s, c1_s, fl_s, shared_contour], dim=1)
        x = self.conv_in(inp)
        x = x + self.dw_d1(x) + self.dw_d2(x)
        delta_flow_sub = self.conv_out(x)
        delta_flow = F.interpolate(delta_flow_sub, size=(h, w), mode='bilinear', align_corners=False) * 2.0
        return flow_refine + delta_flow, delta_flow

ContourGuidedFlowPropagator_v32 = ContourGuidedFlowPropagator_v31

# =============================================================================
# [MODULE 3 - V3.2]: Cổng Hiển Thị Kép Bất Đối Xứng (Asymmetric Stroke Fusion Head)
# =============================================================================
class AsymmetricStrokeFusionHead_v31(nn.Module):
    def __init__(self):
        super(AsymmetricStrokeFusionHead_v31, self).__init__()
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
        hs, ws = h // 2, w // 2
        t_sub = t_vals.expand(b, 1, hs, ws)
        inp_full = torch.cat([fine_I0t, fine_I1t, flow, ifnet_mask, occ_confidence], dim=1)
        inp_sub = F.avg_pool2d(inp_full, 2, 2)
        inp = torch.cat([inp_sub[:, :7], t_sub, inp_sub[:, 7:]], dim=1)
        residual_logits_sub = self.net(inp)
        residual_logits = F.interpolate(residual_logits_sub, size=(h, w), mode='bilinear', align_corners=False)

        prior_0 = torch.logit(torch.clamp(1.0 - t_vals, 1e-4, 1.0 - 1e-4))
        prior_1 = torch.logit(torch.clamp(t_vals, 1e-4, 1.0 - 1e-4))
        prior_logits = torch.cat([prior_0, prior_1], dim=1)

        vis_logits = prior_logits + residual_logits + self.gamma_occ * occ_confidence
        visibility_gates = torch.sigmoid(vis_logits)
        v0 = visibility_gates[:, 0:1]
        v1 = visibility_gates[:, 1:2]
        return v0, v1

AsymmetricStrokeFusionHead_v32 = AsymmetricStrokeFusionHead_v31

# =============================================================================
# [MODULE 4 - V3.2]: Synthesis UNet Kế Thừa Trực Tiếp Chuẩn Vàng V2.6
# =============================================================================
class SynthesisUnet_v31(nn.Module):
    def __init__(self, indim=9, c=16):
        super(SynthesisUnet_v31, self).__init__()
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

        # Khởi tạo Bias cân bằng chuẩn tương thích 100% V2.6 (khử sương mù nền trắng tuyệt đối)
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

SynthesisUnet_v32 = SynthesisUnet_v31

# =============================================================================
# [KIẾN TRÚC TỔNG HỢP]: TPSInbetweenFinetune_v32
# =============================================================================
class TPSInbetweenFinetune_v32(nn.Module):
    """
    Kiến trúc TPS-Inbetween V3.2 (Flat-Minima Retention & Native 540p Integrity).
    - Backbone Contextnet + IFNet kế thừa 100% từ V2.6 Gold.
    - AsymmetricFeatureAnalyzer & AsymmetricStrokeFusionHead_v32: cổng hiển thị kép bất đối xứng.
    - Lightweight ContourGuidedFlowPropagator_v32: giảm 60% FLOPs.
    - SynthesisUnet_v32: Kế thừa 100% Gating Structure & Bias chuẩn V2.6 (Khử sương mù nền trắng tuyệt đối).
    """
    def __init__(self):
        super(TPSInbetweenFinetune_v32, self).__init__()
        self.encoder = Contextnet(indim=1)
        self.IFNet = IFNet()
        self.asym_analyzer = AsymmetricFeatureAnalyzer(in_channels=34)
        self.contour_propagator = ContourGuidedFlowPropagator_v32()
        self.asf_head = AsymmetricStrokeFusionHead_v32()
        self.unet = SynthesisUnet_v32(indim=9)

        self._init_v32_weights()

    def _init_v32_weights(self):
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

        # [V3.2]: Lan truyền vận tốc theo đường biên bao siêu nhẹ
        flow_guided, delta_flow = self.contour_propagator(coarse_I0t, coarse_I1t, flow_raw)

        # 4. Fine Warping với Flow được dẫn hướng
        fine_I0t = warp_pixel_flow(coarse_I0t, flow_guided[:, :2])
        fine_I1t = warp_pixel_flow(coarse_I1t, flow_guided[:, 2:4])

        # [V3.2]: Phân tích bất đối xứng đặc trưng & Occlusion Confidence
        occ_conf = self.asym_analyzer(fine_I0t, fine_I1t, feats0[0], feats1[0])

        # [V3.2]: Cổng hiển thị kép bất đối xứng (V0, V1) in [0, 1]^2
        v0, v1 = self.asf_head(fine_I0t, fine_I1t, flow_guided, ifnet_mask, t_vals, occ_conf)
        base_interp = torch.clamp(v0 * fine_I0t + v1 * fine_I1t, 0.0, 1.0)

        # 5. Gated Silhouette Erase & Detail Reconnection (UNet v3.2 với Pristine Background + Positive Bias)
        gate_erase, detail_add = self.unet(
            x0, x1,
            coarse_flow0t, coarse_flow1t,
            fine_I0t, fine_I1t, ifnet_mask, feats0, feats1
        )
        pred_t = torch.clamp(gate_erase * base_interp + detail_add, 0.0, 1.0)
        if self.training:
            return pred_t, (v0, v1), occ_conf, delta_flow, fine_I0t, fine_I1t
        return pred_t, (v0, v1), occ_conf

TPSInbetweenFinetune_v31 = TPSInbetweenFinetune_v32

base_model = TPSInbetweenFinetune_v32().to(device)
print("=" * 80)
print("[✓] Mô hình TPSInbetweenFinetune_v32 (Pristine Background + Positive Bias + Lightweight Propagator) sẵn sàng.")
print("=" * 80)
sys.stdout.flush()
