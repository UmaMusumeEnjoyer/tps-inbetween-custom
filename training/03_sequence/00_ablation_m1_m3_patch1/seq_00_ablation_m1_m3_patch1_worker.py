#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seq_00_ablation_m1_m3_patch1_worker.py
--------------------------------------------------------------------------------
Standalone DDP Worker for Kaggle 2x Tesla T4 GPUs:
Ablation Study: M1 (TPS-Inbetween Original Baseline) -> M3 (Ours Sequence V1 Patch 1)

Key Technical Requirements:
1. Strict 100% Architecture Identity: M1 and M3 share the exact same TPSInbetweenOriginal_M1_M3 (M_goc).
2. Zero Host RAM Leak: glibc malloc_trim, worker memory cleanup, garbage collection.
3. 100% Dual-GPU Utilization: Precomputed TPS mmap cache, GPU hierarchical TPS flow, NVMe SSD reading.
4. Lineart Loss Suite for M3: Charbonnier + DT + Sobel + Soft-clDice.
5. Strict Checkpoint Policy: ONLY 3 files (best.pt, latest.pt, resume.pt).
6. Auto-Resume from resume.pt after kernel reset or Watchdog RAM exit (code 42).
7. Baseline Evaluation at Epoch 0: Measures both M0 (TPS Coarse) and M1 on full validation set.
--------------------------------------------------------------------------------
"""

import os
import sys
import gc
import time
import math
import json
import glob
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# ==============================================================================
# 0. HỆ THỐNG & TỐI ƯU HÓA BỘ NHỚ (ZERO HOST RAM LEAK & FAST OPENCV)
# ==============================================================================
os.environ["MALLOC_TRIM_THRESHOLD_"] = "65536"
os.environ["MALLOC_ARENA_MAX"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

try:
    import cv2
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
except ImportError:
    pass

torch.backends.cudnn.benchmark = True

def trim_memory():
    """Giải phóng rác Python và ép giải phóng bộ nhớ heap C (glibc malloc_trim)."""
    gc.collect()
    if sys.platform.startswith('linux'):
        try:
            import ctypes
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        except Exception:
            pass

# ==============================================================================
# 1. CÁC TOÁN TỬ HÌNH HỌC VÀ LƯỚI BIẾN DẠNG (WARP & TPS ENGINE)
# ==============================================================================
_PIXEL_GRID_CACHE = {}
_GRID_CACHE = {}

def warp_normalized_flow(img, flow):
    """
    Biến dạng lưới theo luồng Coarse TPS (tọa độ chuẩn hóa [0, 1] sang [-1, 1]).
    """
    b, _, h, w = img.shape
    key = (h, w, flow.device)
    base_grid = _GRID_CACHE.get(key)
    if base_grid is None:
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, h, device=flow.device, dtype=torch.float32),
            torch.linspace(0, 1, w, device=flow.device, dtype=torch.float32),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).contiguous()
        _GRID_CACHE[key] = base_grid

    with torch.amp.autocast('cuda', enabled=False):
        grid = (base_grid + flow.float().permute(0, 2, 3, 1)) * 2.0 - 1.0
        out = F.grid_sample(img.float().contiguous(), grid, padding_mode='border', align_corners=True)
    return out.to(img.dtype)

def warp_pixel_flow(img, flow):
    """
    Biến dạng lưới theo luồng quang học IFNet (tọa độ pixel thực tế).
    """
    b, _, h, w = img.shape
    key = (h, w, flow.device)
    base_grid = _PIXEL_GRID_CACHE.get(key)
    if base_grid is None:
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=flow.device, dtype=torch.float32),
            torch.linspace(-1, 1, w, device=flow.device, dtype=torch.float32),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).contiguous()
        _PIXEL_GRID_CACHE[key] = base_grid

    with torch.amp.autocast('cuda', enabled=False):
        flow_f32 = flow.float()
        flow_norm = torch.cat([
            flow_f32[:, 0:1, :, :] / max(1.0, (w - 1.0) / 2.0),
            flow_f32[:, 1:2, :, :] / max(1.0, (h - 1.0) / 2.0)
        ], dim=1)
        grid = (base_grid + flow_norm.permute(0, 2, 3, 1)).contiguous()
        out = F.grid_sample(img.float().contiguous(), grid, padding_mode='border', align_corners=True)
    return out.to(img.dtype)

def tps_rbf(r):
    r = torch.clamp(r, min=1e-6)
    return (r ** 2) * torch.log(r)

def get_grid_pts(h, w, device):
    key = (h, w, device, 'pts')
    pts = _GRID_CACHE.get(key)
    if pts is None:
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, h, device=device, dtype=torch.float32),
            torch.linspace(0, 1, w, device=device, dtype=torch.float32),
            indexing='ij'
        )
        pts = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2).contiguous()
        _GRID_CACHE[key] = pts
    return pts

@torch.no_grad()
def tps_flow_from_theta_batch(theta_batch, c_src_batch, valid_mask, h, w, device, downscale_factor=4):
    """
    Nội suy trường dòng chảy TPS trực tiếp trên GPU từ ma trận theta đã tiền tính toán.
    Tận dụng downscale để tính toán cực nhanh (<1ms) rồi phóng đại song tuyến tính.
    """
    with torch.amp.autocast('cuda', enabled=False):
        theta_f32 = theta_batch.float()
        c_src_f32 = c_src_batch.float()
        valid_f32 = valid_mask.float()
        b = theta_f32.shape[0]

        if downscale_factor > 1:
            h_sub = h // downscale_factor
            w_sub = w // downscale_factor
        else:
            h_sub, w_sub = h, w

        pts = get_grid_pts(h_sub, w_sub, device)
        total_pts = pts.shape[0]
        pts_exp = pts.unsqueeze(0).expand(b, -1, -1)

        dist = torch.cdist(pts_exp, c_src_f32, p=2.0)
        U = tps_rbf(dist)

        ones = torch.ones((b, total_pts, 1), device=device, dtype=pts.dtype)
        basis = torch.cat([U, ones, pts_exp], dim=-1)

        mapped = torch.bmm(basis, theta_f32)
        flow = (mapped - pts_exp).view(b, h_sub, w_sub, 2).permute(0, 3, 1, 2).contiguous()

        if downscale_factor > 1:
            flow = F.interpolate(flow, size=(h, w), mode='bilinear', align_corners=True)

        flow.mul_(valid_f32.view(b, 1, 1, 1))
        return flow

# ==============================================================================
# 2. KIẾN TRÚC MÔ HÌNH M_gốc (TPSInbetweenOriginal_M1_M3: KHÓA CHẶT 100% KIẾN TRÚC)
# ==============================================================================
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
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(*[conv(c, c) for _ in range(8)])
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    def forward(self, x, flow, scale):
        orig_h, orig_w = x.shape[2], x.shape[3]
        if scale != 1:
            x = F.interpolate(x, scale_factor=1./scale, mode='bilinear', align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False) * (1./scale)
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
                if flow.shape[2:] != I0t.shape[2:]:
                    flow = F.interpolate(flow, size=I0t.shape[2:], mode='bilinear', align_corners=False)
                if mask.shape[2:] != I0t.shape[2:]:
                    mask = F.interpolate(mask, size=I0t.shape[2:], mode='bilinear', align_corners=False)
                flow_d, mask_d = stu[i](torch.cat((img0, img1, warped_i0.repeat(1, 3, 1, 1), warped_i1.repeat(1, 3, 1, 1), mask), 1), flow, scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            if i < 2:
                warped_i0 = warp_pixel_flow(I0t, flow[:, :2])
                warped_i1 = warp_pixel_flow(I1t, flow[:, 2:4])
        return flow, torch.sigmoid(mask)

class OriginalUnet(nn.Module):
    """UNet gốc của TPS-Inbetween (indim=9, outdim=1, sinh ra residual res in [-1, 1])."""
    def __init__(self, indim=9, outdim=1, c=16):
        super(OriginalUnet, self).__init__()
        self.down0 = Conv2(indim, 2*c)
        self.down1 = Conv2(4*c, 4*c)
        self.down2 = Conv2(8*c, 8*c)
        self.down3 = Conv2(16*c, 16*c)
        self.up0 = deconv(32*c, 8*c)
        self.up1 = deconv(16*c, 4*c)
        self.up2 = deconv(8*c, 2*c)
        self.up3 = deconv(4*c, c)
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

class TPSInbetweenOriginal_M1_M3(nn.Module):
    """
    Kiến trúc M_gốc (TPS-Inbetween nguyên bản của bài báo Zhu et al., HIT 2024).
    DÙNG CHUNG DUY NHẤT 1 CLASS NÀY CHO CẢ M1 (Baseline A) VÀ M3 (Ours Sequence V1).
    Cam kết 100% Zero Architectural Difference!
    """
    def __init__(self):
        super(TPSInbetweenOriginal_M1_M3, self).__init__()
        self.encoder = Contextnet(indim=1, c=16)
        self.IFNet = IFNet()
        self.unet = OriginalUnet(indim=9, outdim=1, c=16)

    def forward(self, x0, x1, t_val, coarse_flow01, coarse_flow10):
        """
        x0, x1: [B, 1, H, W] trong không gian nét vẽ Ink Space [0, 1] (1.0 = mực đen, 0.0 = nền trắng)
        t_val: [B, 1, 1, 1]
        coarse_flow01, coarse_flow10: [B, 2, H, W]
        """
        t = t_val
        coarse_flow0t = coarse_flow01 * (t ** 2) - (1.0 - t) * t * coarse_flow10
        coarse_flow1t = coarse_flow10 * ((1.0 - t) ** 2) - t * (1.0 - t) * coarse_flow01

        coarse_I0t = warp_normalized_flow(x0, coarse_flow0t)
        coarse_I1t = warp_normalized_flow(x1, coarse_flow1t)
        # M0 (TPS Coarse Warp)
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
        # Phương trình tổng hợp nguyên bản của TPS-Inbetween
        It = torch.clamp((1.0 - t) * fine_I0t + t * fine_I1t + res, 0.0, 1.0)

        return It, coarse_warp, fine_I0t, fine_I1t, flow_bet_coar, m

def load_m1_weights(model, ckpt_path, device='cpu', is_rank0=True):
    """
    Nạp trọng số gốc model_latest.pt (M1 - Pretrained trên ML240) vào mô hình.
    Loại bỏ các key dư thừa (dead keys như IFNet.block_tea, IFNet.contextnet)
    và nạp 100% active keys của encoder, IFNet.block0..2, unet.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Không tìm thấy file checkpoint M1 gốc tại: {ckpt_path}")

    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and 'model_state' in sd:
        sd = sd['model_state']
    elif isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']

    model_sd = model.state_dict()
    matched_sd = {}
    skipped_dead_keys = []

    for k, v in sd.items():
        clean_k = k.replace('module.', '')
        if clean_k in model_sd and model_sd[clean_k].shape == v.shape:
            matched_sd[clean_k] = v
        else:
            skipped_dead_keys.append(clean_k)

    missing, unexpected = model.load_state_dict(matched_sd, strict=False)

    if is_rank0:
        print(f"[✓] Nạp trọng số M1 (model_latest.pt): Khớp {len(matched_sd)}/{len(model_sd)} active tensors 100%!", flush=True)
        if len(missing) > 0:
            print(f"[!] Cảnh báo missing keys: {missing}", flush=True)
        if len(skipped_dead_keys) > 0:
            print(f"[*] Bỏ qua {len(skipped_dead_keys)} dead keys thừa từ codebase gốc (block_tea/contextnet)", flush=True)

    return model

# ==============================================================================
# 3. BỘ HÀM MẤT MÁT CHUẨN LINEART CHO M3 (HÀM MỤC TIÊU THEO MA TRẬN ABLATION)
# ==============================================================================
def soft_erode(img):
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)

def soft_dilate(img):
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))

def soft_open(img):
    return soft_dilate(soft_erode(img))

def soft_skeletonize(img, iters=2):
    img = torch.clamp(img, 0.0, 1.0)
    skel = F.relu(img - soft_open(img))
    for _ in range(iters - 1):
        img = soft_erode(img)
        skel = skel + F.relu(img - soft_open(img))
    return torch.clamp(skel, 0.0, 1.0)

def _approx_gpu_distance_map(bin_mask, steps=6):
    """
    Xấp xỉ bản đồ khoảng cách Euclidean khả vi trên GPU (Truncated Distance Transform):
    Giá trị pixel = khoảng cách từ điểm ảnh đến nét vẽ gần nhất [0.0 = nằm trên nét, 1.0 = xa nét].
    """
    d_acc = 1.0 - bin_mask
    curr = bin_mask
    for s in range(1, steps + 1):
        k = 2 * s + 1
        dil = F.max_pool2d(curr, kernel_size=k, stride=1, padding=s)
        weight = 1.0 / float(s)
        d_acc = d_acc - weight * (dil - curr)
        curr = dil
    return torch.clamp(d_acc, 0.0, 1.0)

class SoftClDiceLoss(nn.Module):
    def __init__(self, iters=2, smooth=1.0):
        super().__init__()
        self.iters = iters
        self.smooth = smooth

    def forward(self, pred, target, t_skel=None):
        p_skel = soft_skeletonize(pred, iters=self.iters)
        if t_skel is None:
            t_skel = soft_skeletonize(target, iters=self.iters)
        tprec = (torch.sum(p_skel * target) + self.smooth) / (torch.sum(p_skel) + self.smooth)
        tsens = (torch.sum(t_skel * pred) + self.smooth) / (torch.sum(t_skel) + self.smooth)
        cldice = 2.0 * (tprec * tsens) / (tprec + tsens + 1e-6)
        return 1.0 - cldice

class LineartLoss_M3(nn.Module):
    """
    Bộ Loss Chuẩn Lineart cho M3:
    L = 1.0 * L_Charb + 0.50 * L_DT + 0.25 * L_Sobel + 0.20 * L_Soft_clDice
    """
    def __init__(self):
        super(LineartLoss_M3, self).__init__()
        self.cldice = SoftClDiceLoss(iters=2)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, gt):
        # 1. Charbonnier Reconstruction Loss
        loss_charb = torch.mean(torch.sqrt((pred - gt) ** 2 + 1e-6))

        # 2. Symmetric Differentiable Distance Transform Loss
        dt_g = _approx_gpu_distance_map((gt > 0.15).float(), steps=6)
        dt_p = _approx_gpu_distance_map((pred > 0.15).float(), steps=6)
        loss_dt = torch.mean(pred * dt_g) + torch.mean(gt * dt_p)

        # 3. Sobel Stroke Edge Loss
        gx_p = F.conv2d(pred, self.sobel_x, padding=1)
        gy_p = F.conv2d(pred, self.sobel_y, padding=1)
        gx_g = F.conv2d(gt, self.sobel_x, padding=1)
        gy_g = F.conv2d(gt, self.sobel_y, padding=1)
        loss_sobel = F.l1_loss(gx_p, gx_g) + F.l1_loss(gy_p, gy_g)

        # 4. Soft-clDice Topological Skeleton Loss
        with torch.no_grad():
            skel_gt = soft_skeletonize(gt, iters=2)
        loss_cldice = self.cldice(pred, gt, t_skel=skel_gt)

        total_loss = 1.0 * loss_charb + 0.50 * loss_dt + 0.25 * loss_sobel + 0.20 * loss_cldice
        return total_loss, {
            'loss_charb': loss_charb.item(),
            'loss_dt': loss_dt.item(),
            'loss_sobel': loss_sobel.item(),
            'loss_cldice': loss_cldice.item(),
            'total_loss': total_loss.item()
        }

# ==============================================================================
# 4. HÀM ĐO LƯỜNG CHỈ SỐ THẨM ĐỊNH KHOA HỌC (WCD, S-EMD, RECALL, GHOST)
# ==============================================================================
def compute_wcd_batch_gpu(p_bin, g_bin):
    """
    Weighted Chamfer Distance (WCD) chuẩn bài báo TPS (Zhu et al., HIT 2024):
    G(.) = Softplus, kết hợp hệ số phạt diện tích nét H(b0, b1).
    """
    dist_g = _approx_gpu_distance_map(g_bin, steps=8)
    dist_p = _approx_gpu_distance_map(p_bin, steps=8)
    d_p_to_g = torch.sum(p_bin * F.softplus(dist_g * 10.0), dim=[-2, -1]) / torch.clamp(torch.sum(p_bin, dim=[-2, -1]), min=1.0)
    d_g_to_p = torch.sum(g_bin * F.softplus(dist_p * 10.0), dim=[-2, -1]) / torch.clamp(torch.sum(g_bin, dim=[-2, -1]), min=1.0)
    area_p = torch.sum(p_bin, dim=[-2, -1])
    area_g = torch.sum(g_bin, dim=[-2, -1])
    h_weight = 0.5 * (1.0 + torch.minimum(area_p, area_g) / torch.clamp(torch.maximum(area_p, area_g), min=1.0))
    wcd = h_weight * (d_p_to_g + d_g_to_p)
    return wcd

def compute_1d_sliced_emd_batch_gpu(p_f, g_f, num_proj=8):
    """
    1D-Sliced Earth Mover's Distance trên GPU cho phân bố nét vẽ.
    """
    b, _, h, w = p_f.shape
    p_norm = p_f / torch.clamp(torch.sum(p_f, dim=[-2, -1], keepdim=True), min=1e-6)
    g_norm = g_f / torch.clamp(torch.sum(g_f, dim=[-2, -1], keepdim=True), min=1e-6)
    proj_p_h = torch.sum(p_norm, dim=-1)
    proj_g_h = torch.sum(g_norm, dim=-1)
    cdf_p_h = torch.cumsum(proj_p_h, dim=-1)
    cdf_g_h = torch.cumsum(proj_g_h, dim=-1)
    emd_h = torch.mean(torch.abs(cdf_p_h - cdf_g_h), dim=[-2, -1])
    proj_p_w = torch.sum(p_norm, dim=-2)
    proj_g_w = torch.sum(g_norm, dim=-2)
    cdf_p_w = torch.cumsum(proj_p_w, dim=-1)
    cdf_g_w = torch.cumsum(proj_g_w, dim=-1)
    emd_w = torch.mean(torch.abs(cdf_p_w - cdf_g_w), dim=[-2, -1])
    return 0.5 * (emd_h + emd_w)

@torch.no_grad()
def evaluate_metrics_batch_gpu(pred, target):
    """
    Đo lường toàn diện các chỉ số chất lượng nét vẽ trực tiếp trên GPU.
    pred, target: [B, 1, H, W] trong không gian nét vẽ (ink space).
    """
    p = torch.clamp(pred, 0.0, 1.0)
    g = torch.clamp(target, 0.0, 1.0)
    p_bin = (p > 0.15).float()
    g_bin = (g > 0.15).float()

    # 1. WCD & S-EMD
    wcd = compute_wcd_batch_gpu(p_bin, g_bin)
    sliced_emd = compute_1d_sliced_emd_batch_gpu(p, g)

    # 2. Stroke Recall, Precision, F1
    g_dil = soft_dilate(soft_dilate(g_bin))
    tp = torch.sum(p_bin * g_dil, dim=[-3, -2, -1])
    fp = torch.sum(p_bin * (1.0 - g_dil), dim=[-3, -2, -1])
    fn = torch.sum(g_bin * (1.0 - soft_dilate(soft_dilate(p_bin))), dim=[-3, -2, -1])
    recall = tp / torch.clamp(tp + fn, min=1.0)
    precision = tp / torch.clamp(tp + fp, min=1.0)
    f1 = 2.0 * recall * precision / torch.clamp(recall + precision, min=1e-6)

    # 3. Skeleton Recall
    skel_g = soft_skeletonize(g, iters=2)
    skel_recall = torch.sum(skel_g * p, dim=[-3, -2, -1]) / torch.clamp(torch.sum(skel_g, dim=[-3, -2, -1]), min=1.0)

    # 4. Ghost Pixels (Bóng ma ngoài vùng nét GT)
    gt_safe = soft_dilate(soft_dilate((g > 0.08).float()))
    outside_gt = 1.0 - gt_safe
    ghost_px = torch.sum((p > 0.15).float() * outside_gt, dim=[-3, -2, -1])
    dark_ghost_px = torch.sum((p > 0.35).float() * outside_gt, dim=[-3, -2, -1])

    # 5. PSNR & SSIM
    mse = torch.mean((p - g) ** 2, dim=[-3, -2, -1])
    psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-8))

    mu_p = F.avg_pool2d(p, 11, stride=1, padding=5)
    mu_g = F.avg_pool2d(g, 11, stride=1, padding=5)
    sigma_p = F.avg_pool2d(p*p, 11, stride=1, padding=5) - mu_p**2
    sigma_g = F.avg_pool2d(g*g, 11, stride=1, padding=5) - mu_g**2
    sigma_pg = F.avg_pool2d(p*g, 11, stride=1, padding=5) - mu_p*mu_g
    ssim_map = ((2*mu_p*mu_g + 1e-4)*(2*sigma_pg + 9e-4)) / ((mu_p**2 + mu_g**2 + 1e-4)*(sigma_p + sigma_g + 9e-4))
    ssim = torch.mean(ssim_map, dim=[-3, -2, -1])

    val_l1 = torch.mean(torch.abs(p - g), dim=[-3, -2, -1])

    return {
        'wcd': wcd,
        'sliced_emd': sliced_emd,
        'skel_recall': skel_recall,
        'recall': recall,
        'f1': f1,
        'ghost': ghost_px,
        'dark_ghost': dark_ghost_px,
        'val_l1': val_l1,
        'psnr': psnr,
        'ssim': ssim
    }

# ==============================================================================
# 5. DATASET SEQUENCE V1 (PATCH 1) CHO M1 -> M3
# ==============================================================================
class SakugaSequenceM1ToM3Dataset(Dataset):
    """
    Dataset nạp các chuỗi đa khung hình (N = 3, 4, 5) từ Sakuga Golden Dataset V1 Patch 1.
    Mỗi chuỗi clip_XXXXXX (N khung hình) được mở rộng thành (N - 2) mẫu huấn luyện (I0, I1, Ik, tk).
    Các mẫu trong cùng 1 clip dùng chung ma trận nghiệm TPS đầu mút (c0, th01, c1, th10).
    """
    def __init__(self, metadata_csv, clips_dir, tps_cache_path, is_train=True):
        super().__init__()
        self.is_train = is_train
        self.clips_dir = clips_dir
        self.df = pd.read_csv(metadata_csv)

        # Nạp bộ nhớ đệm TPS nghiệm sẵn (Mmap)
        cache_data = np.load(tps_cache_path)
        self.c0_arr = cache_data['c0']
        self.th01_arr = cache_data['th01']
        self.c1_arr = cache_data['c1']
        self.th10_arr = cache_data['th10']
        self.valid_arr = cache_data['valid']

        # Xây dựng danh sách các mẫu frame trung gian
        self.samples = []
        for idx, row in self.df.iterrows():
            clip_id = str(row['clip_id'])
            seq_len = int(row['seq_len'])
            cache_idx = int(row.get('cache_idx', idx))

            # Lấy danh sách hệ số nội suy t (ưu tiên t_spatial_str, fallback t_time_str)
            t_str = str(row.get('t_spatial_str', ''))
            if not t_str or t_str == 'nan' or ';' not in t_str:
                t_str = str(row.get('t_time_str', ''))

            if ';' in t_str:
                t_list = [float(x.strip()) for x in t_str.split(';') if x.strip()]
            else:
                t_list = [float(k) / float(seq_len - 1) for k in range(1, seq_len - 1)]

            # Mỗi frame trung gian từ 2 đến seq_len - 1
            for k_idx, t_val in enumerate(t_list):
                frame_k = k_idx + 2 # Frame index (1-based: 1=I0, 2..N-1=Ik, N=I1)
                self.samples.append({
                    'clip_id': clip_id,
                    'frame_0': 1,
                    'frame_1': seq_len,
                    'frame_k': frame_k,
                    't_val': float(t_val),
                    'cache_idx': cache_idx,
                    'seq_len': seq_len
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        clip_id = sample['clip_id']
        c_dir = os.path.join(self.clips_dir, clip_id)

        p0 = os.path.join(c_dir, f"{sample['frame_0']:06d}.png")
        p1 = os.path.join(c_dir, f"{sample['frame_1']:06d}.png")
        pk = os.path.join(c_dir, f"{sample['frame_k']:06d}.png")

        # Nạp ảnh mức xám [0, 255]
        im0 = Image.open(p0).convert('L')
        im1 = Image.open(p1).convert('L')
        imk = Image.open(pk).convert('L')

        # Chuyển đổi sang Ink Space [0.0, 1.0] (1.0 = mực đen, 0.0 = nền trắng)
        # im_array in [0, 255]: 255 là trắng, 0 là đen -> ink = (255 - im) / 255.0
        x0 = (255.0 - np.array(im0, dtype=np.float32)) / 255.0
        x1 = (255.0 - np.array(im1, dtype=np.float32)) / 255.0
        xk = (255.0 - np.array(imk, dtype=np.float32)) / 255.0

        t_val = sample['t_val']
        cache_idx = sample['cache_idx']

        c0 = self.c0_arr[cache_idx].copy()
        th01 = self.th01_arr[cache_idx].copy()
        c1 = self.c1_arr[cache_idx].copy()
        th10 = self.th10_arr[cache_idx].copy()
        valid = float(self.valid_arr[cache_idx])

        # Data Augmentation khi train: Đảo chiều thời gian (Temporal Reversal: I0 <-> I1, t <-> 1-t)
        if self.is_train and random.random() < 0.5:
            x0, x1 = x1, x0
            c0, c1 = c1, c0
            th01, th10 = th10, th01
            t_val = 1.0 - t_val

        t_x0 = torch.from_numpy(x0).unsqueeze(0)
        t_x1 = torch.from_numpy(x1).unsqueeze(0)
        t_xk = torch.from_numpy(xk).unsqueeze(0)

        return (
            t_x0,
            t_x1,
            t_xk,
            torch.tensor([t_val], dtype=torch.float32),
            torch.from_numpy(c0),
            torch.from_numpy(th01),
            torch.from_numpy(c1),
            torch.from_numpy(th10),
            torch.tensor(valid, dtype=torch.float32),
            sample['seq_len'],
            clip_id
        )

# ==============================================================================
# 6. QUẢN LÝ CHECKPOINT TINH GỌN TUYỆT ĐỐI (MINIMAL CHECKPOINT MANAGER)
# ==============================================================================
class MinimalCheckpointManager_M3:
    """
    Chính sách lưu Checkpoint tinh gọn tuyệt đối:
    CHỈ DUY TRÌ ĐÚNG 3 FILE DUY NHẤT trong output directory:
      1. model_m3_seq_patch1_best.pt   : Checkpoint điểm cao nhất
      2. model_m3_seq_patch1_latest.pt : Checkpoint epoch gần nhất
      3. model_m3_seq_patch1_resume.pt : Checkpoint trạng thái đầy đủ để resume sau khi reset kernel
    TUYỆT ĐỐI KHÔNG LƯU RỜI TỪNG EPOCH (ep01.pt, ep02.pt...) GÂY ĐẦY Ổ CỨNG KAGGLE!
    """
    def __init__(self, output_dir, is_rank0=True):
        self.output_dir = output_dir
        self.is_rank0 = is_rank0
        self.best_score = float('inf')
        self.best_epoch = 0
        os.makedirs(output_dir, exist_ok=True)

        self.best_path = os.path.join(output_dir, "model_m3_seq_patch1_best.pt")
        self.latest_path = os.path.join(output_dir, "model_m3_seq_patch1_latest.pt")
        self.resume_path = os.path.join(output_dir, "model_m3_seq_patch1_resume.pt")

    def save_checkpoint(self, epoch, model, optimizer, scheduler, scaler, score, history, is_best=False):
        if not self.is_rank0:
            return

        state = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'best_score': self.best_score,
            'history': history
        }

        # 1. Lưu atomic latest.pt
        tmp_latest = self.latest_path + ".tmp"
        torch.save(state, tmp_latest)
        if os.path.exists(self.latest_path):
            os.remove(self.latest_path)
        os.rename(tmp_latest, self.latest_path)

        # 2. Nếu đạt kỷ lục mới -> Lưu atomic best.pt
        if is_best:
            self.best_score = score
            self.best_epoch = epoch
            tmp_best = self.best_path + ".tmp"
            torch.save(state, tmp_best)
            if os.path.exists(self.best_path):
                os.remove(self.best_path)
            os.rename(tmp_best, self.best_path)
            print(f"🌟 [KỶ LỤC MỚI M3]: Đã cập nhật {self.best_path} tại Epoch {epoch:02d} (Score = {score:.4f})!", flush=True)

    def save_resume(self, epoch, model, optimizer, scheduler, scaler, history):
        """Lưu đầy đủ trạng thái huấn luyện để tiếp tục train liền mạch sau khi reset kernel."""
        if not self.is_rank0:
            return
        full_state = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
            'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
            'scaler_state': scaler.state_dict() if scaler is not None else None,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'history': history
        }
        tmp_resume = self.resume_path + ".tmp"
        torch.save(full_state, tmp_resume)
        if os.path.exists(self.resume_path):
            os.remove(self.resume_path)
        os.rename(tmp_resume, self.resume_path)
        print(f"[💾 RESUME] Đã lưu trạng thái huấn luyện Epoch {epoch:02d} vào {self.resume_path}", flush=True)

    def load_resume(self, model, optimizer=None, scheduler=None, scaler=None, map_location='cpu'):
        """Kiểm tra và nạp lại trạng thái nếu file resume.pt tồn tại."""
        if not os.path.exists(self.resume_path):
            return 0, []

        print(f"[🔍 RESUME] Phát hiện {self.resume_path}! Đang tự động khôi phục phiên huấn luyện...", flush=True)
        ckpt = torch.load(self.resume_path, map_location=map_location, weights_only=False)
        last_epoch = ckpt.get('epoch', 0)
        self.best_score = ckpt.get('best_score', float('inf'))
        self.best_epoch = ckpt.get('best_epoch', 0)
        history = ckpt.get('history', [])

        # Khôi phục model
        model_sd = ckpt['model_state']
        model.load_state_dict(model_sd)

        # Khôi phục optimizer & scheduler
        if optimizer is not None and ckpt.get('optimizer_state') is not None:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        if scheduler is not None and ckpt.get('scheduler_state') is not None:
            scheduler.load_state_dict(ckpt['scheduler_state'])
        if scaler is not None and ckpt.get('scaler_state') is not None:
            scaler.load_state_dict(ckpt['scaler_state'])

        print(f"[✓ RESUME THÀNH CÔNG]: Sẽ tiếp tục huấn luyện từ Epoch {last_epoch + 1:02d} (Best Epoch trước đó: {self.best_epoch:02d})!", flush=True)
        return last_epoch, history

# ==============================================================================
# 7. QUÁ TRÌNH VALIDATION & ĐO LƯỜNG ĐỐI CHUẨN KHOA HỌC (M0, M1, M3)
# ==============================================================================
@torch.no_grad()
def run_validation_suite(model, val_loader, device, is_ddp=False, evaluate_m0=False):
    """
    Chạy bộ kiểm định trên toàn bộ tập validation:
    - Nếu evaluate_m0=True: đo lường cả M0 (TPS Coarse Warp) và mô hình nạp vào.
    - Phân tách chỉ số theo từng độ dài chuỗi N = 3, 4, 5.
    """
    model.eval()
    local_metrics = {
        'count': 0,
        'val_l1': 0.0,
        'psnr': 0.0,
        'ssim': 0.0,
        'wcd': 0.0,
        'sliced_emd': 0.0,
        'skel_recall': 0.0,
        'recall': 0.0,
        'ghost': 0.0,
        'dark_ghost': 0.0
    }

    # Bảng phân tách theo N=3, 4, 5
    breakdown = {
        3: {'count': 0, 'psnr': 0.0, 'ssim': 0.0, 'wcd': 0.0, 'skel_recall': 0.0, 'ghost': 0.0},
        4: {'count': 0, 'psnr': 0.0, 'ssim': 0.0, 'wcd': 0.0, 'skel_recall': 0.0, 'ghost': 0.0},
        5: {'count': 0, 'psnr': 0.0, 'ssim': 0.0, 'wcd': 0.0, 'skel_recall': 0.0, 'ghost': 0.0}
    }

    # M0 metrics (nếu có yêu cầu)
    m0_metrics = {
        'count': 0, 'val_l1': 0.0, 'psnr': 0.0, 'ssim': 0.0, 'wcd': 0.0, 'sliced_emd': 0.0, 'skel_recall': 0.0, 'recall': 0.0, 'ghost': 0.0
    } if evaluate_m0 else None

    for batch in val_loader:
        t_x0, t_x1, t_xk, t_val, c0, th01, c1, th10, valid, seq_lens, _ = batch
        t_x0 = t_x0.to(device, non_blocking=True)
        t_x1 = t_x1.to(device, non_blocking=True)
        t_xk = t_xk.to(device, non_blocking=True)
        t_val = t_val.to(device, non_blocking=True).view(-1, 1, 1, 1)
        c0 = c0.to(device, non_blocking=True)
        th01 = th01.to(device, non_blocking=True)
        c1 = c1.to(device, non_blocking=True)
        th10 = th10.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)

        b, _, h, w = t_x0.shape

        # Tính toán Coarse Flow từ TPS matrix
        cf01 = tps_flow_from_theta_batch(th01, c0, valid, h, w, device, downscale_factor=4)
        cf10 = tps_flow_from_theta_batch(th10, c1, valid, h, w, device, downscale_factor=4)

        # Forward mô hình
        with torch.amp.autocast('cuda', enabled=True):
            pred_it, coarse_warp, fine_i0, fine_i1, _, _ = model(t_x0, t_x1, t_val, cf01, cf10)

        # Đo lường cho Model chính
        m_dict = evaluate_metrics_batch_gpu(pred_it, t_xk)
        for i in range(b):
            local_metrics['count'] += 1
            local_metrics['val_l1'] += float(m_dict['val_l1'][i])
            local_metrics['psnr'] += float(m_dict['psnr'][i])
            local_metrics['ssim'] += float(m_dict['ssim'][i])
            local_metrics['wcd'] += float(m_dict['wcd'][i])
            local_metrics['sliced_emd'] += float(m_dict['sliced_emd'][i])
            local_metrics['skel_recall'] += float(m_dict['skel_recall'][i])
            local_metrics['recall'] += float(m_dict['recall'][i])
            local_metrics['ghost'] += float(m_dict['ghost'][i])
            local_metrics['dark_ghost'] += float(m_dict['dark_ghost'][i])

            sl = int(seq_lens[i])
            if sl in breakdown:
                breakdown[sl]['count'] += 1
                breakdown[sl]['psnr'] += float(m_dict['psnr'][i])
                breakdown[sl]['ssim'] += float(m_dict['ssim'][i])
                breakdown[sl]['wcd'] += float(m_dict['wcd'][i])
                breakdown[sl]['skel_recall'] += float(m_dict['skel_recall'][i])
                breakdown[sl]['ghost'] += float(m_dict['ghost'][i])

        # Đo lường cho M0 (Coarse Warp)
        if evaluate_m0:
            m0_dict = evaluate_metrics_batch_gpu(coarse_warp, t_xk)
            for i in range(b):
                m0_metrics['count'] += 1
                m0_metrics['val_l1'] += float(m0_dict['val_l1'][i])
                m0_metrics['psnr'] += float(m0_dict['psnr'][i])
                m0_metrics['ssim'] += float(m0_dict['ssim'][i])
                m0_metrics['wcd'] += float(m0_dict['wcd'][i])
                m0_metrics['sliced_emd'] += float(m0_dict['sliced_emd'][i])
                m0_metrics['skel_recall'] += float(m0_dict['skel_recall'][i])
                m0_metrics['recall'] += float(m0_dict['recall'][i])
                m0_metrics['ghost'] += float(m0_dict['ghost'][i])

    # Gom kết quả phân tán qua all_reduce
    if is_ddp:
        t_buf = torch.tensor([
            local_metrics['count'],
            local_metrics['val_l1'],
            local_metrics['psnr'],
            local_metrics['ssim'],
            local_metrics['wcd'],
            local_metrics['sliced_emd'],
            local_metrics['skel_recall'],
            local_metrics['recall'],
            local_metrics['ghost'],
            local_metrics['dark_ghost']
        ], dtype=torch.float64, device=device)
        dist.all_reduce(t_buf, op=dist.ReduceOp.SUM)

        cnt = max(1.0, t_buf[0].item())
        res = {
            'val_l1': t_buf[1].item() / cnt,
            'psnr': t_buf[2].item() / cnt,
            'ssim': t_buf[3].item() / cnt,
            'wcd': t_buf[4].item() / cnt,
            'sliced_emd': t_buf[5].item() / cnt,
            'skel_recall': t_buf[6].item() / cnt,
            'recall': t_buf[7].item() / cnt,
            'ghost': t_buf[8].item() / cnt,
            'dark_ghost': t_buf[9].item() / cnt,
            'total_samples': int(cnt)
        }

        # Reduce breakdown
        for sl in [3, 4, 5]:
            b_buf = torch.tensor([
                breakdown[sl]['count'],
                breakdown[sl]['psnr'],
                breakdown[sl]['ssim'],
                breakdown[sl]['wcd'],
                breakdown[sl]['skel_recall'],
                breakdown[sl]['ghost']
            ], dtype=torch.float64, device=device)
            dist.all_reduce(b_buf, op=dist.ReduceOp.SUM)
            b_cnt = max(1.0, b_buf[0].item())
            breakdown[sl] = {
                'count': int(b_cnt),
                'psnr': b_buf[1].item() / b_cnt,
                'ssim': b_buf[2].item() / b_cnt,
                'wcd': b_buf[3].item() / b_cnt,
                'skel_recall': b_buf[4].item() / b_cnt,
                'ghost': b_buf[5].item() / b_cnt
            }

        res_m0 = None
        if evaluate_m0:
            m0_buf = torch.tensor([
                m0_metrics['count'],
                m0_metrics['val_l1'],
                m0_metrics['psnr'],
                m0_metrics['ssim'],
                m0_metrics['wcd'],
                m0_metrics['sliced_emd'],
                m0_metrics['skel_recall'],
                m0_metrics['recall'],
                m0_metrics['ghost']
            ], dtype=torch.float64, device=device)
            dist.all_reduce(m0_buf, op=dist.ReduceOp.SUM)
            m0_cnt = max(1.0, m0_buf[0].item())
            res_m0 = {
                'val_l1': m0_buf[1].item() / m0_cnt,
                'psnr': m0_buf[2].item() / m0_cnt,
                'ssim': m0_buf[3].item() / m0_cnt,
                'wcd': m0_buf[4].item() / m0_cnt,
                'sliced_emd': m0_buf[5].item() / m0_cnt,
                'skel_recall': m0_buf[6].item() / m0_cnt,
                'recall': m0_buf[7].item() / m0_cnt,
                'ghost': m0_buf[8].item() / m0_cnt,
                'total_samples': int(m0_cnt)
            }
    else:
        cnt = max(1, local_metrics['count'])
        res = {k: v / cnt for k, v in local_metrics.items() if k != 'count'}
        res['total_samples'] = cnt
        for sl in [3, 4, 5]:
            b_cnt = max(1, breakdown[sl]['count'])
            breakdown[sl] = {k: v / b_cnt for k, v in breakdown[sl].items() if k != 'count'}
            breakdown[sl]['count'] = b_cnt
        res_m0 = None
        if evaluate_m0:
            m0_cnt = max(1, m0_metrics['count'])
            res_m0 = {k: v / m0_cnt for k, v in m0_metrics.items() if k != 'count'}
            res_m0['total_samples'] = m0_cnt

    res['breakdown'] = breakdown
    if evaluate_m0:
        return res, res_m0
    return res

# ==============================================================================
# 8. HÀM CHÍNH THỰC THI DDP HUẤN LUYỆN & BENCHMARK M1 -> M3
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Huấn luyện & Benchmark M1 -> M3 trên Kaggle 2x Tesla T4")
    parser.add_argument('--train_csv', type=str, required=True, help="Đường dẫn train_sequences.csv")
    parser.add_argument('--val_csv', type=str, required=True, help="Đường dẫn val_sequences.csv")
    parser.add_argument('--clips_dir', type=str, required=True, help="Thư mục clips trên SSD")
    parser.add_argument('--tps_cache', type=str, required=True, help="File mmap tps_cache.npz")
    parser.add_argument('--m1_checkpoint', type=str, required=True, help="Đường dẫn model_latest.pt gốc")
    parser.add_argument('--output_dir', type=str, default="/kaggle/working/checkpoints_m3_patch1", help="Thư mục checkpoint")
    parser.add_argument('--epochs', type=int, default=15, help="Số epoch huấn luyện")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size trên mỗi GPU (Tổng 16 cho 2 GPU)")
    parser.add_argument('--lr_backbone', type=float, default=1.0e-5, help="Learning rate cho Backbone (Contextnet + IFNet)")
    parser.add_argument('--lr_unet', type=float, default=2.5e-5, help="Learning rate cho UNet")
    parser.add_argument('--num_workers', type=int, default=2, help="Số worker nạp dữ liệu mỗi GPU")
    parser.add_argument('--ram_limit_gb', type=float, default=23.5, help="Ngưỡng Watchdog RAM kích hoạt thoát an toàn")
    args = parser.parse_args()

    # Thiết lập DDP
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp = world_size > 1

    if is_ddp:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    is_rank0 = (local_rank == 0)

    if is_rank0:
        print("=" * 80)
        print("🚀 KHỞI ĐỘNG TIẾN TRÌNH HUẤN LUYỆN & BENCHMARK: M1 -> M3 (BẰNG CHỨNG THÉP #1)")
        print(f"🔧 Cấu hình: World Size = {world_size} GPUs | Device = {device} | Batch Size/GPU = {args.batch_size}")
        print(f"📁 Output Checkpoints = {args.output_dir} (Chính sách lưu ĐÚNG 3 FILE)")
        print("=" * 80)
        sys.stdout.flush()

    # Khởi tạo Checkpoint Manager
    ckpt_manager = MinimalCheckpointManager_M3(args.output_dir, is_rank0=is_rank0)

    # 1. Khởi tạo Mô hình M_gốc
    base_model = TPSInbetweenOriginal_M1_M3().to(device)

    # Nạp trọng số M1 (model_latest.pt)
    load_m1_weights(base_model, args.m1_checkpoint, device=device, is_rank0=is_rank0)

    # 2. Khởi tạo Dataset & DataLoader
    train_dataset = SakugaSequenceM1ToM3Dataset(
        metadata_csv=args.train_csv,
        clips_dir=args.clips_dir,
        tps_cache_path=args.tps_cache,
        is_train=True
    )
    val_dataset = SakugaSequenceM1ToM3Dataset(
        metadata_csv=args.val_csv,
        clips_dir=args.clips_dir,
        tps_cache_path=args.tps_cache,
        is_train=False
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_ddp else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=False
    )

    if is_rank0:
        print(f"[✓] Dataset Ready: Train = {len(train_dataset):,} samples | Val = {len(val_dataset):,} samples", flush=True)

    # 3. Optimizer & Scaler
    optimizer = torch.optim.AdamW([
        {'params': base_model.encoder.parameters(), 'lr': args.lr_backbone, 'weight_decay': 1e-4},
        {'params': base_model.IFNet.parameters(), 'lr': args.lr_backbone, 'weight_decay': 1e-4},
        {'params': base_model.unet.parameters(), 'lr': args.lr_unet, 'weight_decay': 1e-4}
    ])
    scaler = torch.amp.GradScaler('cuda')
    criterion = LineartLoss_M3().to(device)

    # 4. Kiểm tra Cơ chế Auto-Resume trước khi vào vòng lặp
    start_epoch = 1
    history = []
    last_ep, hist = ckpt_manager.load_resume(base_model, optimizer=optimizer, scaler=scaler, map_location=device)
    if last_ep > 0:
        start_epoch = last_ep + 1
        history = hist

    # 5. BENCHMARK BASELINE TẠI EPOCH 0 (M0 & M1) NẾU CHẠY TỪ ĐẦU
    baseline_metrics_path = os.path.join(os.path.dirname(args.output_dir), "baseline_m0_m1_metrics.json")
    if start_epoch == 1 and not os.path.exists(baseline_metrics_path):
        if is_rank0:
            print("\n" + "=" * 80)
            print("🔬 ĐANG ĐO LƯỜNG BASELINE BAN ĐẦU TẠI EPOCH 0 CHO M0 VÀ M1...")
            print("   M0: TPS Coarse Warp (Toán tử uốn lưới thuần, không có mạng nơ-ron)")
            print("   M1: TPS-Inbetween Original (Trọng số gốc bài báo huấn luyện trên ML240)")
            print("=" * 80, flush=True)

        m1_res, m0_res = run_validation_suite(base_model, val_loader, device, is_ddp=is_ddp, evaluate_m0=True)

        if is_rank0:
            baseline_data = {
                'M0_TPS_Coarse_Warp': m0_res,
                'M1_TPS_Original_ML240': m1_res
            }
            with open(baseline_metrics_path, 'w', encoding='utf-8') as f:
                json.dump(baseline_data, f, indent=2, ensure_ascii=False)
            print(f"[✓] ĐÃ LƯU KẾT QUẢ ĐỐI CHUẨN EPOCH 0 SANG: {baseline_metrics_path}")
            print(f"📊 M0 (Coarse Warp) : PSNR = {m0_res['psnr']:.2f}dB | SSIM = {m0_res['ssim']:.4f} | WCD = {m0_res['wcd']:.4f} | Recall = {m0_res['recall']*100:.1f}%")
            print(f"📊 M1 (Original TPS): PSNR = {m1_res['psnr']:.2f}dB | SSIM = {m1_res['ssim']:.4f} | WCD = {m1_res['wcd']:.4f} | Recall = {m1_res['recall']*100:.1f}%")
            print("=" * 80 + "\n", flush=True)

    # 6. Bọc mô hình DDP
    if is_ddp:
        ddp_model = DDP(base_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    else:
        ddp_model = base_model

    history_csv_path = os.path.join(os.path.dirname(args.output_dir), "training_history_m3_patch1.csv")
    total_epochs = args.epochs
    watchdog_paused = False
    train_start_time = time.time()

    if is_rank0:
        print(f"🚀 BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN M3: TỪ EPOCH {start_epoch:02d} ĐẾN {total_epochs:02d}\n", flush=True)

    for epoch in range(start_epoch, total_epochs + 1):
        t_ep_start = time.time()
        t_train_start = time.time()

        if is_ddp and hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)

        # Cập nhật Learning Rate với Cosine Decay
        if epoch == 1:
            cur_lr_bb = args.lr_backbone * 0.5
            cur_lr_un = args.lr_unet * 0.6
        elif epoch <= 3:
            cur_lr_bb = args.lr_backbone
            cur_lr_un = args.lr_unet
        else:
            progress = float(epoch - 3) / float(max(1, total_epochs - 3))
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            cur_lr_bb = args.lr_backbone * max(0.1, cosine_factor)
            cur_lr_un = args.lr_unet * max(0.1, cosine_factor)

        optimizer.param_groups[0]['lr'] = cur_lr_bb
        optimizer.param_groups[1]['lr'] = cur_lr_bb
        optimizer.param_groups[2]['lr'] = cur_lr_un

        ddp_model.train()
        train_loss_acc = 0.0
        train_steps = 0

        for batch_idx, batch in enumerate(train_loader):
            t_x0, t_x1, t_xk, t_val, c0, th01, c1, th10, valid, _, _ = batch
            t_x0 = t_x0.to(device, non_blocking=True)
            t_x1 = t_x1.to(device, non_blocking=True)
            t_xk = t_xk.to(device, non_blocking=True)
            t_val = t_val.to(device, non_blocking=True).view(-1, 1, 1, 1)
            c0 = c0.to(device, non_blocking=True)
            th01 = th01.to(device, non_blocking=True)
            c1 = c1.to(device, non_blocking=True)
            th10 = th10.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)

            b, _, h, w = t_x0.shape

            optimizer.zero_grad(set_to_none=True)

            # Tính toán Coarse Flow từ ma trận TPS trên GPU
            cf01 = tps_flow_from_theta_batch(th01, c0, valid, h, w, device, downscale_factor=4)
            cf10 = tps_flow_from_theta_batch(th10, c1, valid, h, w, device, downscale_factor=4)

            with torch.amp.autocast('cuda', enabled=True):
                pred_it, _, _, _, _, _ = ddp_model(t_x0, t_x1, t_val, cf01, cf10)
                loss, loss_details = criterion(pred_it, t_xk)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_acc += loss.item()
            train_steps += 1

        train_time_m = (time.time() - t_train_start) / 60.0

        # Vòng kiểm định Validation
        t_val_start = time.time()
        val_metrics = run_validation_suite(base_model, val_loader, device, is_ddp=is_ddp, evaluate_m0=False)
        val_time_m = (time.time() - t_val_start) / 60.0
        ep_time_m = (time.time() - t_ep_start) / 60.0

        # Điểm kiểm định tổng hợp (Score càng nhỏ càng tốt, kết hợp WCD và L1)
        composite_score = val_metrics['wcd'] * 0.6 + val_metrics['val_l1'] * 0.4
        is_best = composite_score < ckpt_manager.best_score

        # Ghi log và Checkpoint tại Rank 0
        if is_rank0:
            avg_train_loss = train_loss_acc / max(1, train_steps)
            throughput = len(train_dataset) / max(1.0, (time.time() - t_train_start))

            row = {
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_l1': val_metrics['val_l1'],
                'psnr': val_metrics['psnr'],
                'ssim': val_metrics['ssim'],
                'wcd': val_metrics['wcd'],
                'sliced_emd': val_metrics['sliced_emd'],
                'skel_recall': val_metrics['skel_recall'],
                'recall': val_metrics['recall'],
                'ghost_pixels': val_metrics['ghost'],
                'score': composite_score,
                'is_best': is_best,
                'train_time_min': train_time_m,
                'val_time_min': val_time_m,
                'total_time_min': ep_time_m
            }
            # Thêm breakdown theo N vào row
            for sl in [3, 4, 5]:
                if sl in val_metrics['breakdown']:
                    row[f'psnr_N{sl}'] = val_metrics['breakdown'][sl]['psnr']
                    row[f'wcd_N{sl}'] = val_metrics['breakdown'][sl]['wcd']
                    row[f'skel_recall_N{sl}'] = val_metrics['breakdown'][sl]['skel_recall']

            history.append(row)
            pd.DataFrame(history).to_csv(history_csv_path, index=False)

            # Lưu Checkpoint (Chỉ 3 file: latest, best, resume)
            ckpt_manager.save_checkpoint(epoch, base_model, optimizer, None, scaler, composite_score, history, is_best=is_best)
            ckpt_manager.save_resume(epoch, base_model, optimizer, None, scaler, history)

            print(f"📌 EPOCH {epoch:02d}/{total_epochs:02d} HOÀN TẤT | Train: {train_time_m:.2f} min | Val: {val_time_m:.2f} min | Tốc độ: {throughput:.1f} samples/s")
            print(f"   🔍 VAL M3: L1={val_metrics['val_l1']:.4f} | SSIM={val_metrics['ssim']:.4f} | PSNR={val_metrics['psnr']:.2f}dB | WCD={val_metrics['wcd']:.4f} | SkelRecall={val_metrics['skel_recall']*100:.1f}%")
            print(f"   🎯 Composite Score: {composite_score:.4f} (Kỷ lục: {ckpt_manager.best_score:.4f})")
            sys.stdout.flush()

        # Kiểm tra Host RAM Watchdog để bảo vệ phiên Kaggle
        trim_memory()
        try:
            import psutil
            mem_gb = psutil.virtual_memory().used / (1024**3)
            if mem_gb >= args.ram_limit_gb:
                if is_rank0:
                    print(f"\n⚠️ [WATCHDOG RAM CẢNH BÁO]: Host RAM đạt {mem_gb:.2f} GB (Ngưỡng {args.ram_limit_gb} GB)!")
                    print("   Đã lưu trạng thái Resume an toàn. Tiến trình sẽ dừng với exit code 42 để Kernel có thể restart.", flush=True)
                watchdog_paused = True
                break
        except Exception:
            pass

        if is_ddp:
            dist.barrier()

    if is_ddp:
        dist.destroy_process_group()

    if is_rank0:
        if watchdog_paused:
            print("\n⏸️ HUẤN LUYỆN TẠM DỪNG DO WATCHDOG RAM. Chạy Cell 6 để restart kernel rồi Run All để tiếp tục!", flush=True)
            sys.exit(42)
        else:
            total_train_m = (time.time() - train_start_time) / 60.0
            print(f"\n🎉 HOÀN TẤT TOÀN BỘ {total_epochs} EPOCHS HUẤN LUYỆN M3 TRONG {total_train_m:.1f} PHÚT!", flush=True)
            sys.exit(0)

if __name__ == '__main__':
    main()
