#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_occlusion.py
--------------------------------------------------------------------------------
KIẾN TRÚC MÔ HÌNH PHA 4: COARSE TPS + OCCLUSION-AWARE HEAD (TPSInbetweenOcclusion)
Thử nghiệm Cách ly: Phase 4 Occlusion Head (M4_Occ)
Không ảnh hưởng 100% tới Phase 3 Remake (Phase3_Remake / M3_Best).

Các nâng cấp cốt lõi:
1. Occlusion Head trong IFNet.block2:
   Lớp ConvTranspose2d cuối cùng xuất 6 kênh:
     - 4 kênh luồng quang học tinh chỉnh: [F0->t (2), F1->t (2)]
     - 2 kênh mặt nạ che khuất: [M0->t (1), M1->t (1)] qua Sigmoid in [0.0, 1.0].
2. Bộ Trộn Trọng số Chuẩn hóa (Normalized Visibility Blending):
   W0 = (1 - t) * M0->t,  W1 = t * M1->t
   I_fused = (W0 * Warp(I0, F0->t) + W1 * Warp(I1, F1->t)) / (W0 + W1 + eps)
   res = unet(...) * 2.0 - 1.0
   It = clamp(I_fused + res, 0.0, 1.0)
3. Khởi tạo Trọng số (Warm-Start):
   Kế thừa toàn bộ 158 tensors từ checkpoint Phase 3 Remake (model_tps_finetuned_best.pt).
   2 kênh mặt nạ mới được khởi tạo bias = +2.0 (Sigmoid ≈ 0.88), weights = 0.0,
   đảm bảo tại bước 0 mô hình hội tụ mượt mà, zero loss shock.
4. OcclusionAwareLoss:
   - Photometric loss có trọng số hiển thị: L_photo = M0*rho(I0t - GT) + M1*rho(I1t - GT)
   - Coverage constraint: L_cov = mean(ReLU(0.95 - (M0 + M1))^2)
   - Edge-aware smoothness: L_smooth giữ biên mặt nạ bám sát nét vẽ GT.
--------------------------------------------------------------------------------
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

_GRID_CACHE = {}
_PIXEL_GRID_CACHE = {}


def warp_normalized_flow(img, flow):
    """Backward-warp theo luồng Coarse TPS (tọa độ chuẩn hóa [0, 1] -> [-1, 1])."""
    _, _, h, w = img.shape
    key = (h, w, flow.device)
    base_grid = _GRID_CACHE.get(key)
    if base_grid is None:
        gy, gx = torch.meshgrid(
            torch.linspace(0.0, 1.0, h, device=flow.device, dtype=torch.float32),
            torch.linspace(0.0, 1.0, w, device=flow.device, dtype=torch.float32),
            indexing='ij'
        )
        base_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).contiguous()
        _GRID_CACHE[key] = base_grid

    with torch.amp.autocast('cuda', enabled=False):
        grid = (base_grid + flow.float().permute(0, 2, 3, 1)) * 2.0 - 1.0
        out = F.grid_sample(
            img.float().contiguous(), grid,
            mode='bilinear', padding_mode='border', align_corners=True
        )
    return out.to(img.dtype)


def warp_pixel_flow(img, flow):
    """Backward-warp theo luồng quang học IFNet (đơn vị pixel)."""
    _, _, h, w = img.shape
    key = (h, w, flow.device)
    base_grid = _PIXEL_GRID_CACHE.get(key)
    if base_grid is None:
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=flow.device, dtype=torch.float32),
            torch.linspace(-1.0, 1.0, w, device=flow.device, dtype=torch.float32),
            indexing='ij'
        )
        base_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).contiguous()
        _PIXEL_GRID_CACHE[key] = base_grid

    with torch.amp.autocast('cuda', enabled=False):
        flow_f32 = flow.float()
        flow_norm = torch.cat([
            flow_f32[:, 0:1, :, :] / max(1.0, (w - 1.0) / 2.0),
            flow_f32[:, 1:2, :, :] / max(1.0, (h - 1.0) / 2.0)
        ], dim=1)
        grid = (base_grid + flow_norm.permute(0, 2, 3, 1)).contiguous()
        out = F.grid_sample(
            img.float().contiguous(), grid,
            mode='bilinear', padding_mode='border', align_corners=True
        )
    return out.to(img.dtype)


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
        self.conv2 = Conv2(c, 2 * c)
        self.conv3 = Conv2(2 * c, 4 * c)
        self.conv4 = Conv2(4 * c, 8 * c)

    def forward(self, x, flow):
        x = self.conv1(x)
        flow = F.interpolate(flow, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False) * 0.5
        f1 = warp_normalized_flow(x, flow)
        x = self.conv2(x)
        flow = F.interpolate(flow, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False) * 0.5
        f2 = warp_normalized_flow(x, flow)
        x = self.conv3(x)
        flow = F.interpolate(flow, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False) * 0.5
        f3 = warp_normalized_flow(x, flow)
        x = self.conv4(x)
        flow = F.interpolate(flow, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False) * 0.5
        f4 = warp_normalized_flow(x, flow)
        return [f1, f2, f3, f4]


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64, out_planes=5):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(*[conv(c, c) for _ in range(8)])
        self.lastconv = nn.ConvTranspose2d(c, out_planes, 4, 2, 1)

    def forward(self, x, flow, scale):
        orig_h, orig_w = x.shape[2], x.shape[3]
        if scale != 1:
            x = F.interpolate(x, scale_factor=1.0 / scale, mode='bilinear', align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False) * (1.0 / scale)
            x = torch.cat((x, flow), 1)
        x = self.conv0(x.contiguous())
        x = self.convblock(x) + x
        with torch.amp.autocast('cuda', enabled=False):
            tmp = self.lastconv(x.float().contiguous())
            tmp = F.interpolate(tmp, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            flow = tmp[:, :4] * scale * 2
            extra = tmp[:, 4:]
        return flow, extra


class IFNetOcclusion(nn.Module):
    """
    IFNet tích hợp Occlusion Head tại Block 2:
    - block0: xuất [F0->t (2), F1->t (2), mask (1)] tại scale 4 (5 channels)
    - block1: xuất [dF0->t (2), dF1->t (2), dmask (1)] tại scale 2 (5 channels)
    - block2 (Occlusion Head): xuất [dF0->t (2), dF1->t (2), M0->t (1), M1->t (1)] tại scale 1 (6 channels)
    """
    def __init__(self):
        super(IFNetOcclusion, self).__init__()
        self.block0 = IFBlock(6, c=240, out_planes=5)
        self.block1 = IFBlock(13 + 4, c=150, out_planes=5)
        self.block2 = IFBlock(13 + 4, c=90, out_planes=6)

    def forward(self, I0t, I1t, scale=(4, 2, 1)):
        img0 = I0t.repeat(1, 3, 1, 1)
        img1 = I1t.repeat(1, 3, 1, 1)

        # Scale 4: Block 0
        flow, mask = self.block0(torch.cat((img0, img1), 1), None, scale=scale[0])
        warped_i0 = warp_pixel_flow(I0t, flow[:, :2])
        warped_i1 = warp_pixel_flow(I1t, flow[:, 2:4])

        # Scale 2: Block 1
        if flow.shape[2:] != I0t.shape[2:]:
            flow = F.interpolate(flow, size=I0t.shape[2:], mode='bilinear', align_corners=False)
        if mask.shape[2:] != I0t.shape[2:]:
            mask = F.interpolate(mask, size=I0t.shape[2:], mode='bilinear', align_corners=False)
        inp1 = torch.cat((img0, img1, warped_i0.repeat(1, 3, 1, 1), warped_i1.repeat(1, 3, 1, 1), mask), 1)
        flow_d1, mask_d1 = self.block1(inp1, flow, scale=scale[1])
        flow = flow + flow_d1
        mask = mask + mask_d1
        warped_i0 = warp_pixel_flow(I0t, flow[:, :2])
        warped_i1 = warp_pixel_flow(I1t, flow[:, 2:4])

        # Scale 1: Block 2 (Occlusion Head: outputs 4 flow channels + 2 occlusion mask channels)
        if flow.shape[2:] != I0t.shape[2:]:
            flow = F.interpolate(flow, size=I0t.shape[2:], mode='bilinear', align_corners=False)
        if mask.shape[2:] != I0t.shape[2:]:
            mask = F.interpolate(mask, size=I0t.shape[2:], mode='bilinear', align_corners=False)
        inp2 = torch.cat((img0, img1, warped_i0.repeat(1, 3, 1, 1), warped_i1.repeat(1, 3, 1, 1), mask), 1)
        flow_d2, occ_logits = self.block2(inp2, flow, scale=scale[2])
        flow = flow + flow_d2

        M0 = torch.sigmoid(occ_logits[:, 0:1])
        M1 = torch.sigmoid(occ_logits[:, 1:2])
        unet_mask = torch.sigmoid(mask)
        return flow, M0, M1, unet_mask


class OriginalUnet(nn.Module):
    """UNet gốc của bài báo (indim=9, outdim=1, đầu ra residual [-1, 1])."""
    def __init__(self, indim=9, outdim=1, c=16):
        super(OriginalUnet, self).__init__()
        self.down0 = Conv2(indim, 2 * c)
        self.down1 = Conv2(4 * c, 4 * c)
        self.down2 = Conv2(8 * c, 8 * c)
        self.down3 = Conv2(16 * c, 16 * c)
        self.up0 = deconv(32 * c, 8 * c)
        self.up1 = deconv(16 * c, 4 * c)
        self.up2 = deconv(8 * c, 2 * c)
        self.up3 = deconv(4 * c, c)
        self.conv = nn.Conv2d(c, outdim, 3, 1, 1)

    def forward(self, img0, img1, flow0, flow1, warped0, warped1, mask, c0, c1):
        s0 = self.down0(torch.cat((img0, img1, warped0, warped1, flow0, flow1, mask), 1))
        if s0.shape[2:] != c0[0].shape[2:]:
            s0 = F.interpolate(s0, size=c0[0].shape[2:], mode='bilinear', align_corners=False)
        s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
        if s1.shape[2:] != c0[1].shape[2:]:
            s1 = F.interpolate(s1, size=c0[1].shape[2:], mode='bilinear', align_corners=False)
        s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
        if s2.shape[2:] != c0[2].shape[2:]:
            s2 = F.interpolate(s2, size=c0[2].shape[2:], mode='bilinear', align_corners=False)
        s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
        if s3.shape[2:] != c0[3].shape[2:]:
            s3 = F.interpolate(s3, size=c0[3].shape[2:], mode='bilinear', align_corners=False)
        x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))
        if x.shape[2:] != s2.shape[2:]:
            x = F.interpolate(x, size=s2.shape[2:], mode='bilinear', align_corners=False)
        x = self.up1(torch.cat((x, s2), 1))
        if x.shape[2:] != s1.shape[2:]:
            x = F.interpolate(x, size=s1.shape[2:], mode='bilinear', align_corners=False)
        x = self.up2(torch.cat((x, s1), 1))
        if x.shape[2:] != s0.shape[2:]:
            x = F.interpolate(x, size=s0.shape[2:], mode='bilinear', align_corners=False)
        x = self.up3(torch.cat((x, s0), 1))
        if x.shape[2:] != img0.shape[2:]:
            x = F.interpolate(x, size=img0.shape[2:], mode='bilinear', align_corners=False)
        x = self.conv(x)
        return torch.sigmoid(x)


class TPSInbetweenOcclusion(nn.Module):
    """
    Kiến trúc TPS-Inbetween nâng cấp Occlusion Head (Phase 4):
      - Giữ nguyên toàn bộ 156 tensors phụ trợ (encoder + unet + block0 + block1).
      - Thay đổi duy nhất tại IFNet.block2.lastconv: xuất 6 kênh [F(4), M0(1), M1(1)].
      - Bộ Trộn Trọng số Chuẩn hóa (Normalized Visibility Blending):
        W0 = (1 - t) * M0->t
        W1 = t * M1->t
        I_fused = (W0 * fine_I0t + W1 * fine_I1t) / (W0 + W1 + eps)
    """
    def __init__(self):
        super(TPSInbetweenOcclusion, self).__init__()
        self.encoder = Contextnet(indim=1, c=16)
        self.IFNet = IFNetOcclusion()
        self.unet = OriginalUnet(indim=9, outdim=1, c=16)

    def forward(self, x0, x1, t_val, coarse_flow01, coarse_flow10, direct_t_flows=False):
        t = t_val
        if direct_t_flows:
            coarse_flow0t = coarse_flow01
            coarse_flow1t = coarse_flow10
        else:
            coarse_flow0t = coarse_flow01 * (t ** 2) - (1.0 - t) * t * coarse_flow10
            coarse_flow1t = coarse_flow10 * ((1.0 - t) ** 2) - t * (1.0 - t) * coarse_flow01

        coarse_I0t = warp_normalized_flow(x0, coarse_flow0t)
        coarse_I1t = warp_normalized_flow(x1, coarse_flow1t)
        coarse_warp = torch.clamp((1.0 - t) * coarse_I0t + t * coarse_I1t, 0.0, 1.0)

        feats0 = self.encoder(x0, coarse_flow0t)
        feats1 = self.encoder(x1, coarse_flow1t)

        flow_bet_coar, M0, M1, unet_mask = self.IFNet(coarse_I0t, coarse_I1t)

        fine_I0t = warp_pixel_flow(coarse_I0t, flow_bet_coar[:, :2])
        fine_I1t = warp_pixel_flow(coarse_I1t, flow_bet_coar[:, 2:4])

        # Normalized Visibility Blending
        w0 = (1.0 - t) * M0
        w1 = t * M1
        w_sum = w0 + w1 + 1e-6
        I_fused = (w0 * fine_I0t + w1 * fine_I1t) / w_sum

        refine_out = self.unet(
            x0, x1,
            coarse_flow0t + flow_bet_coar[:, :2],
            coarse_flow1t + flow_bet_coar[:, 2:4],
            fine_I0t, fine_I1t, unet_mask, feats0, feats1
        )
        res = refine_out[:, 0:1, :, :] * 2.0 - 1.0
        It = torch.clamp(I_fused + res, 0.0, 1.0)
        return It, coarse_warp, fine_I0t, fine_I1t, flow_bet_coar, M0, M1


def load_tps_occlusion_weights(model, ckpt_path, device='cpu', prefer_ema=True, is_rank0=True):
    """
    Nạp trọng số Warm-Start từ Phase 3 Remake (158 tensors) sang Phase 4 Occlusion Model:
      - 156 tensors khớp tuyệt đối 100%.
      - IFNet.block2.lastconv (4 kênh flow + 2 kênh occlusion):
        + Kế thừa 4 kênh flow từ Phase 3.
        + Khởi tạo 2 kênh occlusion mới: weights = 0.0, bias = +2.0 (Sigmoid ≈ 0.88).
    Nếu checkpoint đã là Phase 4 hoàn chỉnh, nạp khớp 100% 158/158 tensors.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint tại: {ckpt_path}")

    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(raw, dict):
        if prefer_ema and raw.get('ema_state') is not None:
            ema_obj = raw['ema_state']
            sd = ema_obj['shadow'] if isinstance(ema_obj, dict) and 'shadow' in ema_obj else ema_obj
        elif 'model_state' in raw:
            sd = raw['model_state']
        elif 'state_dict' in raw:
            sd = raw['state_dict']
        elif 'model' in raw:
            sd = raw['model']
        else:
            sd = raw
    else:
        sd = raw

    model_sd = model.state_dict()
    matched_sd = {}
    warm_started_keys = []

    for k, v in sd.items():
        clean_k = k.replace('module.', '').replace('base_model.', '')
        if clean_k in model_sd:
            tgt_shape = model_sd[clean_k].shape
            if tgt_shape == v.shape:
                matched_sd[clean_k] = v
            elif clean_k == "IFNet.block2.lastconv.weight" and v.shape == torch.Size([90, 5, 4, 4]) and tgt_shape == torch.Size([90, 6, 4, 4]):
                # Warm-start block2 lastconv weight (copy 4 flow channels, 2 mask channels = 0)
                new_w = torch.zeros(tgt_shape, dtype=v.dtype, device=v.device)
                new_w[:, :4, :, :] = v[:, :4, :, :]
                matched_sd[clean_k] = new_w
                warm_started_keys.append(clean_k)
            elif clean_k == "IFNet.block2.lastconv.bias" and v.shape == torch.Size([5]) and tgt_shape == torch.Size([6]):
                # Warm-start block2 lastconv bias (copy 4 flow biases, 2 mask biases = +2.0)
                new_b = torch.zeros(tgt_shape, dtype=v.dtype, device=v.device)
                new_b[:4] = v[:4]
                new_b[4:] = 2.0  # Sigmoid(+2.0) ≈ 0.8808
                matched_sd[clean_k] = new_b
                warm_started_keys.append(clean_k)

    missing, unexpected = model.load_state_dict(matched_sd, strict=False)
    if len(missing) > 0:
        raise RuntimeError(f"Thiếu {len(missing)} tensors khi nạp {ckpt_path}: {missing[:5]}")

    if is_rank0:
        tag = os.path.basename(ckpt_path)
        if warm_started_keys:
            print(f"[✓] Warm-Start Phase 4 từ Phase 3 ({tag}): Khớp 158/158 tensors!"
                  f" (block2.lastconv mở rộng 5->6 kênh với bias=+2.0, Sigmoid≈0.88)", flush=True)
        else:
            print(f"[✓] Nạp trọng số Phase 4 ({tag}): Khớp 100% {len(matched_sd)}/{len(model_sd)} active tensors!", flush=True)
    return model


class OcclusionAwareLoss(nn.Module):
    """
    Hàm Mất mát Nhận thức Che khuất (Occlusion-Aware Loss):
      1. Photometric Loss có trọng số hiển thị:
         L_photo = M0 * rho(fine_I0t - GT) + M1 * rho(fine_I1t - GT)
         với rho(x) = sqrt(x^2 + eps^2) (Charbonnier Loss).
      2. Ràng buộc Nhất quán Bao phủ (Coverage Constraint):
         L_cov = mean(ReLU(0.95 - (M0 + M1))^2)
         (Ngăn chặn mạng triệt tiêu M -> 0 để trốn phạt lỗi tái tạo).
      3. Làm mượt Mặt nạ Che khuất có hướng dẫn nét vẽ (Edge-aware Smoothness):
         L_smooth = sum_{c} |d_x M_c| * exp(-alpha * |d_x GT|) + |d_y M_c| * exp(-alpha * |d_y GT|)
         (Giữ biên của M bám sát theo các đường nét viền vật thể, không lốm đốm nhiễu hạt).
    """
    def __init__(self, w_photo=0.25, w_cov=0.50, w_smooth=0.10, cov_margin=0.95, smooth_alpha=10.0, eps=1e-3):
        super(OcclusionAwareLoss, self).__init__()
        self.w_photo = float(w_photo)
        self.w_cov = float(w_cov)
        self.w_smooth = float(w_smooth)
        self.cov_margin = float(cov_margin)
        self.smooth_alpha = float(smooth_alpha)
        self.eps = float(eps)

    def forward(self, fine_I0t, fine_I1t, M0, M1, gt_t):
        # 1. Photometric Loss có trọng số hiển thị
        diff0 = fine_I0t - gt_t
        diff1 = fine_I1t - gt_t
        rho0 = torch.sqrt(diff0 ** 2 + self.eps ** 2)
        rho1 = torch.sqrt(diff1 ** 2 + self.eps ** 2)
        loss_photo = torch.mean(M0 * rho0 + M1 * rho1)

        # 2. Coverage Constraint (Chống lười biếng M -> 0)
        cov_deficit = F.relu(self.cov_margin - (M0 + M1))
        loss_cov = torch.mean(cov_deficit ** 2)

        # 3. Edge-aware Smoothness (Bám sát viền nét GT, phẳng mịn trên nền)
        def _edge_smooth(m_map, img_ref):
            dx_m = torch.abs(m_map[:, :, :, 1:] - m_map[:, :, :, :-1])
            dy_m = torch.abs(m_map[:, :, 1:, :] - m_map[:, :, :-1, :])
            with torch.no_grad():
                dx_img = torch.abs(img_ref[:, :, :, 1:] - img_ref[:, :, :, :-1])
                dy_img = torch.abs(img_ref[:, :, 1:, :] - img_ref[:, :, :-1, :])
                wx = torch.exp(-self.smooth_alpha * dx_img)
                wy = torch.exp(-self.smooth_alpha * dy_img)
            return torch.mean(dx_m * wx) + torch.mean(dy_m * wy)

        loss_smooth = _edge_smooth(M0, gt_t) + _edge_smooth(M1, gt_t)

        total_occ_loss = (
            self.w_photo * loss_photo
            + self.w_cov * loss_cov
            + self.w_smooth * loss_smooth
        )
        details = {
            'loss_occ_photo': float(loss_photo.detach().item()),
            'loss_occ_cov': float(loss_cov.detach().item()),
            'loss_occ_smooth': float(loss_smooth.detach().item()),
            'total_occ_loss': float(total_occ_loss.detach().item()),
        }
        return total_occ_loss, details
