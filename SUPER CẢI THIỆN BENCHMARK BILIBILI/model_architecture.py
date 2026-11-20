#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_architecture.py
--------------------------------------------------------------------------------
KIẾN TRÚC MẠNG NGUYÊN BẢN TPS-INBETWEEN (ZHU ET AL., HIT 2024)
Dùng cho inference cách ly hoàn toàn, không phụ thuộc vào mã nguồn huấn luyện.
Active parameters: 10,068,668 (158 active tensors).
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
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(*[conv(c, c) for _ in range(8)])
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

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
            mask = tmp[:, 4:5]
        return flow, mask


class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(6, c=240)
        self.block1 = IFBlock(13 + 4, c=150)
        self.block2 = IFBlock(13 + 4, c=90)

    def forward(self, I0t, I1t, scale=(4, 2, 1)):
        img0 = I0t.repeat(1, 3, 1, 1)
        img1 = I1t.repeat(1, 3, 1, 1)
        warped_i0 = I0t
        warped_i1 = I1t
        flow = None
        mask = None
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow is not None:
                if flow.shape[2:] != I0t.shape[2:]:
                    flow = F.interpolate(flow, size=I0t.shape[2:], mode='bilinear', align_corners=False)
                if mask.shape[2:] != I0t.shape[2:]:
                    mask = F.interpolate(mask, size=I0t.shape[2:], mode='bilinear', align_corners=False)
                inp = torch.cat((img0, img1, warped_i0.repeat(1, 3, 1, 1), warped_i1.repeat(1, 3, 1, 1), mask), 1)
                flow_d, mask_d = stu[i](inp, flow, scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            if i < 2:
                warped_i0 = warp_pixel_flow(I0t, flow[:, :2])
                warped_i1 = warp_pixel_flow(I1t, flow[:, 2:4])
        return flow, torch.sigmoid(mask)


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


class TPSInbetweenOriginal(nn.Module):
    """
    Kiến trúc nguyên bản TPS-Inbetween (Zhu et al., HIT 2024).
    10,068,668 active parameters — CAM KẾT 0% THAY ĐỔI KIẾN TRÚC.
    """
    def __init__(self):
        super(TPSInbetweenOriginal, self).__init__()
        self.encoder = Contextnet(indim=1, c=16)
        self.IFNet = IFNet()
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
        flow_bet_coar, m = self.IFNet(coarse_I0t, coarse_I1t)

        fine_I0t = warp_pixel_flow(coarse_I0t, flow_bet_coar[:, :2])
        fine_I1t = warp_pixel_flow(coarse_I1t, flow_bet_coar[:, 2:4])

        refine_out = self.unet(
            x0, x1,
            coarse_flow0t + flow_bet_coar[:, :2],
            coarse_flow1t + flow_bet_coar[:, 2:4],
            fine_I0t, fine_I1t, m, feats0, feats1
        )
        res = refine_out[:, 0:1, :, :] * 2.0 - 1.0
        It = torch.clamp((1.0 - t) * fine_I0t + t * fine_I1t + res, 0.0, 1.0)
        return It, coarse_warp, fine_I0t, fine_I1t, flow_bet_coar, m


def load_tps_weights(model, ckpt_path, device='cpu', prefer_ema=False, is_rank0=True):
    """
    Nạp trọng số từ checkpoint finetuned hoặc baseline.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Không tìm thấy file checkpoint tại: {ckpt_path}")

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
    skipped_keys = []
    for k, v in sd.items():
        clean_k = k.replace('module.', '').replace('base_model.', '')
        if clean_k in model_sd and model_sd[clean_k].shape == v.shape:
            matched_sd[clean_k] = v
        else:
            skipped_keys.append(clean_k)

    missing, _ = model.load_state_dict(matched_sd, strict=False)
    if len(missing) > 0:
        raise RuntimeError(f"Thiếu {len(missing)} active tensors khi nạp {ckpt_path}: {missing[:5]}")

    if is_rank0:
        tag = os.path.basename(ckpt_path)
        print(f"[✓] Nạp trọng số ({tag}): Khớp {len(matched_sd)}/{len(model_sd)} active tensors 100%!"
              + (f" (Bỏ qua {len(skipped_keys)} unused/dead keys)" if skipped_keys else ""), flush=True)
    return model
