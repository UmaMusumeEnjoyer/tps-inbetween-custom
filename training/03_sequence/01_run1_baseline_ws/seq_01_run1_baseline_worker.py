#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seq_p1_v32_ghost_temp_scale_worker.py
--------------------------------------------------------------------------------
Động cơ Huấn luyện & Kiểm định DDP 2x GPU Tesla T4 (Kaggle Dual-GPU) cho:
Fine-tune Mô hình TPS-Inbetween Gốc (model_latest.pt) trên Sakuga Sequence Final Dataset
để kiểm chứng khách quan (Zero-Bias Hypothesis Testing):
  - M0: Coarse TPS Warp (Toán tử uốn lưới hình học thuần túy, không mạng nơ-ron)
  - M1: Original TPS-Inbetween (model_latest.pt — Baseline gốc ML240)
  - M3: Finetuned TPS-Inbetween (100% cùng kiến trúc TPSInbetweenOriginal với M1)

Các chuẩn mực kỹ thuật cốt lõi được đảm bảo trong file này:
1. 100% ZERO ARCHITECTURAL CHANGE:
   - M1 và M3 dùng chung duy nhất class TPSInbetweenOriginal (13,026,543 active params).
2. SỬA TRIỆT ĐỂ LỖI NGƯỢC DẤU BACKWARD-WARP TPS & DYNAMIC MULT32 PADDING:
   - Khôi phục đúng quy ước tpsinbet.py: flow01 (kps1 -> kps0), flow10 (kps0 -> kps1).
   - Xử lý chính xác mọi chiều rộng khung hình W in [720, 1480] khi đệm mult-32.
3. HỖ TRỢ 3 TÙY CHỌN TỐC ĐỘ HUẤN LUYỆN (SHORT / MEDIUM / LONG):
   - Mặc định theo duyệt Plan: TRAIN_MODE = "LONG" (Option 3 — Full-Trajectory
     540p, 12-step Symmetric DT + Charb + Sobel + Soft-clDice + VGG/Perceptual,
     EMA với adaptive warmup, Cosine LR Warmup).
   - DDP Length & Width-Bucketed Batch Sampler giữ cả 2 GPU ở ~98-100% tải.
   - Chunked Multi-frame Gradient Accumulation giữ VRAM an toàn (~5-6 GB / 15 GB T4).
4. CHÍNH SÁCH CHECKPOINT TINH GỌN TUYỆT ĐỐI (CHỈ 2 FILE):
   - model_tps_finetuned_best.pt : Checkpoint tốt nhất theo Balanced Geometry Score.
   - model_tps_finetuned_last.pt : Checkpoint epoch gần nhất + trạng thái Auto-Resume.
   - Không lưu rời từng epoch! Tích hợp RAM Watchdog + glibc malloc_trim().
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
from torch.utils.data import Dataset, DataLoader, Sampler

# ==============================================================================
# 0. TỐI ƯU HỆ THỐNG, KHÓA PHÂN MẢNH RAM C & CẤU HÌNH MÔI TRƯỜNG
# ==============================================================================
os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "65536")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

try:
    import cv2
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
except ImportError:
    cv2 = None

torch.backends.cudnn.benchmark = True


def trim_memory():
    """Giải phóng bộ nhớ Python và ép glibc trả lại vùng nhớ trống cho OS (Linux/Kaggle)."""
    gc.collect()
    if sys.platform.startswith('linux'):
        try:
            import ctypes
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        except Exception:
            pass


# ==============================================================================
# 1. TOÁN TỬ BIẾN DẠNG LƯỚI & GIẢI TPS CHUẨN BACKWARD-WARP (KHỚP 100% TPSINBET.PY)
# ==============================================================================
FIXED_TPS_ANCHORS = np.array([
    [0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
    [0.0, 0.5],             [1.0, 0.5],
    [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]
], dtype=np.float32)

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


def tps_rbf_torch(r):
    r = torch.clamp(r, min=1e-6)
    return (r ** 2) * torch.log(r)


def solve_tps_backward_pair(kps0, kps1, max_pts=128, lambd=0.01):
    """
    Giải hệ phương trình TPS theo đúng chiều Backward-Warp của tpsinbet.py:
      - flow01 được nội suy từ các điểm điều khiển tại kps1 nhắm tới kps0 (c0=kps1 -> kps0)
      - flow10 được nội suy từ các điểm điều khiển tại kps0 nhắm tới kps1 (c1=kps0 -> kps1)
    Tự động giảm mẫu đều xuống (max_pts - 8) điểm + 8 điểm neo biên cố định [0, 1]^2.
    Trả về: c0, th01, c1, th10, valid_flag.
    """
    def _solve_single(c_src, c_dst):
        if c_src is None or c_dst is None or len(c_src) < 4:
            return (
                np.zeros((max_pts, 2), dtype=np.float32),
                np.zeros((max_pts + 3, 2), dtype=np.float32),
                0.0
            )
        n_kp_keep = max_pts - 8
        if len(c_src) > n_kp_keep:
            idx = np.linspace(0, len(c_src) - 1, n_kp_keep, dtype=int)
            cs_sub = np.asarray(c_src[idx], dtype=np.float64)
            cd_sub = np.asarray(c_dst[idx], dtype=np.float64)
        else:
            cs_sub = np.asarray(c_src, dtype=np.float64)
            cd_sub = np.asarray(c_dst, dtype=np.float64)

        anchors64 = FIXED_TPS_ANCHORS.astype(np.float64)
        cs_all = np.vstack([cs_sub, anchors64])
        cd_all = np.vstack([cd_sub, anchors64])
        m = len(cs_all)

        diff = cs_all[:, None, :] - cs_all[None, :, :]
        r = np.linalg.norm(diff, axis=-1)
        K = (r ** 2) * np.log(np.maximum(r, 1e-6))
        np.fill_diagonal(K, float(lambd))

        P = np.hstack([np.ones((m, 1), dtype=np.float64), cs_all])
        A = np.zeros((m + 3, m + 3), dtype=np.float64)
        A[:m, :m] = K
        A[:m, m:] = P
        A[m:, :m] = P.T

        Y = np.zeros((m + 3, 2), dtype=np.float64)
        Y[:m, :] = cd_all

        try:
            theta = np.linalg.solve(A, Y)
        except np.linalg.LinAlgError:
            theta, _, _, _ = np.linalg.lstsq(A, Y, rcond=1e-4)

        valid = 0.0 if (
            np.isnan(theta).any() or np.isinf(theta).any() or np.max(np.abs(theta)) > 50.0
        ) else 1.0

        c_pad = np.zeros((max_pts, 2), dtype=np.float32)
        th_pad = np.zeros((max_pts + 3, 2), dtype=np.float32)
        if valid > 0.5:
            c_pad[:m] = cs_all.astype(np.float32)
            th_pad[:m] = theta[:m].astype(np.float32)
            th_pad[max_pts:max_pts + 3] = theta[m:m + 3].astype(np.float32)
        return c_pad, th_pad, valid

    # Quy ước tpsinbet.py: flow01 tại kps1 -> kps0; flow10 tại kps0 -> kps1
    c0, th01, v01 = _solve_single(kps1, kps0)
    c1, th10, v10 = _solve_single(kps0, kps1)
    v = 1.0 if (v01 > 0.5 and v10 > 0.5) else 0.0
    return c0, th01, c1, th10, v


def _get_norm_grid_pts(h, w, device):
    key = (h, w, device, 'tps_pts_2d')
    pts = _GRID_CACHE.get(key)
    if pts is None:
        gy, gx = torch.meshgrid(
            torch.linspace(0.0, 1.0, h, device=device, dtype=torch.float32),
            torch.linspace(0.0, 1.0, w, device=device, dtype=torch.float32),
            indexing='ij'
        )
        pts = torch.stack([gx, gy], dim=-1).view(-1, 2).contiguous()
        _GRID_CACHE[key] = pts
    return pts


@torch.no_grad()
def tps_flow_from_theta_batch(theta_batch, c_src_batch, valid_mask, orig_hw, h_pad, w_pad, device, downscale_factor=2):
    """
    Nội suy chính xác trường dòng chảy Coarse TPS trên GPU từ nghiệm đã tiền tính toán,
    tự động co giãn chính xác theo tỷ lệ khung ảnh gốc (orig_h, orig_w) bên trong canvas đệm (h_pad, w_pad).
    Dùng chunking (chunk_size=32768 điểm) để giữ mức đỉnh VRAM thấp (<35 MB).
    """
    with torch.amp.autocast('cuda', enabled=False):
        theta_f32 = theta_batch.float()
        c_src_f32 = c_src_batch.float()
        valid_f32 = valid_mask.float()
        b = theta_f32.shape[0]

        h_sub = max(16, h_pad // max(1, downscale_factor))
        w_sub = max(16, w_pad // max(1, downscale_factor))

        pts_pad = _get_norm_grid_pts(h_sub, w_sub, device)  # [P, 2]
        total_pts = pts_pad.shape[0]
        pts_pad_exp = pts_pad.unsqueeze(0).expand(b, -1, -1)  # [B, P, 2]

        # Tỷ lệ quy đổi giữa tọa độ chuẩn hóa của canvas (h_pad, w_pad) và ảnh gốc (orig_h, orig_w)
        orig_hw_f = orig_hw.to(device=device, dtype=torch.float32)  # [B, 2] -> (H_orig, W_orig)
        sx = (float(max(1, w_pad - 1)) / torch.clamp(orig_hw_f[:, 1] - 1.0, min=1.0)).view(b, 1, 1)
        sy = (float(max(1, h_pad - 1)) / torch.clamp(orig_hw_f[:, 0] - 1.0, min=1.0)).view(b, 1, 1)
        scale_xy = torch.cat([sx, sy], dim=-1)  # [B, 1, 2]

        # Quy về tọa độ gốc [0, 1] và giữ dấu vùng biên hợp lệ
        pts_orig_raw = pts_pad_exp * scale_xy
        in_bounds = ((pts_orig_raw[..., 0:1] <= 1.001) & (pts_orig_raw[..., 1:2] <= 1.001)).float()
        pts_orig = torch.clamp(pts_orig_raw, 0.0, 1.0)

        flow_chunks = []
        chunk_size = 32768
        for i in range(0, total_pts, chunk_size):
            end_i = min(i + chunk_size, total_pts)
            q = pts_orig[:, i:end_i, :]  # [B, C, 2]
            dist = torch.cdist(q, c_src_f32, p=2.0)  # [B, C, M]
            U = tps_rbf_torch(dist)
            ones = torch.ones((b, q.shape[1], 1), device=device, dtype=torch.float32)
            basis = torch.cat([U, ones, q], dim=-1)  # [B, C, M+3]
            mapped = torch.bmm(basis, theta_f32)  # [B, C, 2]
            disp_orig = mapped - q
            disp_pad = (disp_orig / scale_xy) * in_bounds[:, i:end_i, :]
            flow_chunks.append(disp_pad)

        flow = torch.cat(flow_chunks, dim=1).view(b, h_sub, w_sub, 2).permute(0, 3, 1, 2).contiguous()
        if h_sub != h_pad or w_sub != w_pad:
            flow = F.interpolate(flow, size=(h_pad, w_pad), mode='bilinear', align_corners=True)

        flow.mul_(valid_f32.view(b, 1, 1, 1))
        return flow


# ==============================================================================
# 2. KIẾN TRÚC MÔ HÌNH NGUYÊN BẢN 100% (TPSInbetweenOriginal: ZHU ET AL., 2024)
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
    Dùng chung DUY NHẤT 1 class này cho M1 (model_latest.pt) và M3 (Finetuned):
    13,026,543 active parameters — CAM KẾT 0% THAY ĐỔI KIẾN TRÚC.
    """
    def __init__(self):
        super(TPSInbetweenOriginal, self).__init__()
        self.encoder = Contextnet(indim=1, c=16)
        self.IFNet = IFNet()
        self.unet = OriginalUnet(indim=9, outdim=1, c=16)

    def forward(self, x0, x1, t_val, coarse_flow01, coarse_flow10):
        t = t_val
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
    Nạp trọng số từ model_latest.pt gốc (252 keys gồm 94 dead keys) hoặc checkpoint finetune.
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


# ==============================================================================
# 3. HÀM MẤT MÁT CHUYÊN DỤNG CHO LINEART (OPTION 1 / 2 / 3 - LONG 5-TERM LOSS)
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


def approx_gpu_distance_map(bin_mask, steps=12):
    """
    Bản đồ khoảng cách Euclidean xấp xỉ siêu nhanh bằng phép giãn 3x3 lặp trên GPU [0, 1].
    Giá trị 0.0 tại chính pixel nét và tăng tuyến tính tới 1.0 ngoài bán kính `steps` pixel.
    """
    curr = bin_mask
    d_sum = curr
    for _ in range(steps):
        curr = F.max_pool2d(curr, kernel_size=3, stride=1, padding=1)
        d_sum = d_sum + curr
    # pixel thuộc bin_mask có d_sum = steps + 1 -> dt = 0; pixel ngoài bán kính có d_sum = 0 -> dt = 1
    dt = 1.0 - (d_sum / float(steps + 1))
    return torch.clamp(dt, 0.0, 1.0)


def soft_inverse_dt_coverage(pred_ink, gt_bin, steps=12):
    """
    Phần kéo thuận khả vi 100%: đo xem quanh mỗi pixel nét GT trong bán kính 1..steps,
    dự đoán `pred_ink` đạt mực gần nhất bao nhiêu (tránh hoàn toàn gradient chết của hard threshold!).
    """
    curr = pred_ink
    loss_acc = 0.0
    gt_count = torch.clamp(gt_bin.sum(dim=(-3, -2, -1)), min=1.0)
    for s in range(steps):
        curr = F.max_pool2d(curr, kernel_size=3, stride=1, padding=1)
        miss_at_s = torch.sum(gt_bin * (1.0 - curr), dim=(-3, -2, -1)) / gt_count
        loss_acc = loss_acc + miss_at_s
    return (loss_acc / float(steps)).mean()


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


class VGGPerceptualOrLaplacianLoss(nn.Module):
    """
    Perceptual Loss dựa trên VGG16 đông băng (khớp tinh thần bài báo gốc TPS-Inbetween).
    Tự động dự phòng sang Multi-Scale Laplacian Pyramid Loss nếu chạy ở môi trường offline
    chưa cache sẵn weights VGG16 (không bao giờ gây crash).
    """
    def __init__(self, is_rank0=True):
        super().__init__()
        self.use_vgg = False
        self.blocks = None
        try:
            import torchvision.models as tv_models
            weights_enum = getattr(tv_models, 'VGG16_Weights', None)
            if weights_enum is not None:
                vgg = tv_models.vgg16(weights=weights_enum.DEFAULT).features
            else:
                vgg = tv_models.vgg16(pretrained=True).features
            vgg.eval()
            for p in vgg.parameters():
                p.requires_grad = False
            # Lấy 4 tầng relu1_2(4), relu2_2(9), relu3_3(16), relu4_3(23)
            self.slice1 = nn.Sequential(*list(vgg[:4]))
            self.slice2 = nn.Sequential(*list(vgg[4:9]))
            self.slice3 = nn.Sequential(*list(vgg[9:16]))
            self.slice4 = nn.Sequential(*list(vgg[16:23]))
            self.use_vgg = True
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            if is_rank0:
                print("[✓] VGG16 Perceptual Loss đã kích hoạt (Option 3 LONG Mode).", flush=True)
        except Exception as e:
            if is_rank0:
                print(f"[*] VGG16 chưa tải được ({e}) -> Fallback sang 4-Scale Laplacian Perceptual Loss.", flush=True)

    def _lap_pyr_loss(self, pred, gt):
        loss = 0.0
        cur_p, cur_g = pred, gt
        for w in [0.4, 0.3, 0.2, 0.1]:
            blur_p = F.avg_pool2d(F.pad(cur_p, (1, 1, 1, 1), mode='replicate'), 3, stride=1)
            blur_g = F.avg_pool2d(F.pad(cur_g, (1, 1, 1, 1), mode='replicate'), 3, stride=1)
            loss = loss + w * F.l1_loss(cur_p - blur_p, cur_g - blur_g)
            cur_p = F.avg_pool2d(cur_p, 2, stride=2)
            cur_g = F.avg_pool2d(cur_g, 2, stride=2)
        return loss

    def forward(self, pred_ink, gt_ink):
        if not self.use_vgg:
            return self._lap_pyr_loss(pred_ink, gt_ink)

        # Chuyển về không gian ảnh trắng nền (1 - ink) cho ImageNet VGG
        p_rgb = (1.0 - pred_ink).repeat(1, 3, 1, 1)
        g_rgb = (1.0 - gt_ink).repeat(1, 3, 1, 1)
        p_norm = (p_rgb - self.mean) / self.std
        g_norm = (g_rgb - self.mean) / self.std

        weights = [0.20, 0.30, 0.30, 0.20]
        total = 0.0
        h_p, h_g = p_norm, g_norm
        for blk, w in zip([self.slice1, self.slice2, self.slice3, self.slice4], weights):
            h_p = blk(h_p)
            with torch.no_grad():
                h_g = blk(h_g)
            total = total + w * F.l1_loss(h_p, h_g)
        return total


class AdaptiveLineartLoss(nn.Module):
    """
    Hàm mục tiêu thích ứng theo chế độ huấn luyện:
      - SHORT : 1.0*Charb + 0.50*DT(6)  + 0.25*Sobel
      - MEDIUM: 1.0*Charb + 0.50*DT(8)  + 0.25*Sobel + 0.20*Soft-clDice
      - LONG  : 1.0*Charb + 0.60*DT(12) + 0.25*Sobel + 0.25*Soft-clDice + 0.15*Perceptual(VGG)
    Hỗ trợ cắt biên chính xác (valid_hw) để không tính loss lên phần pad biên phải/dưới!
    """
    def __init__(self, train_mode="LONG", is_rank0=True):
        super().__init__()
        self.train_mode = train_mode.upper()
        if self.train_mode == "SHORT":
            self.dt_steps = 6
            self.w_dt = 0.50
            self.w_cldice = 0.0
            self.w_perc = 0.0
        elif self.train_mode == "MEDIUM":
            self.dt_steps = 8
            self.w_dt = 0.50
            self.w_cldice = 0.20
            self.w_perc = 0.0
        else:
            self.dt_steps = 12
            self.w_dt = 0.60
            self.w_cldice = 0.25
            self.w_perc = 0.15

        self.cldice = SoftClDiceLoss(iters=2) if self.w_cldice > 0 else None
        self.perceptual = VGGPerceptualOrLaplacianLoss(is_rank0=is_rank0) if self.w_perc > 0 else None

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, gt, valid_hw=None):
        if valid_hw is not None:
            if torch.all(valid_hw == valid_hw[0:1]):
                hi, wi = int(valid_hw[0, 0].item()), int(valid_hw[0, 1].item())
                return self._compute_single(pred[:, :, :hi, :wi], gt[:, :, :hi, :wi])
            # Cắt từng mẫu theo kích thước thực tế khi batch có các clip hơi lệch W
            total_l = 0.0
            details_acc = {'loss_charb': 0.0, 'loss_dt': 0.0, 'loss_sobel': 0.0, 'loss_cldice': 0.0, 'loss_perc': 0.0}
            b = pred.shape[0]
            for i in range(b):
                hi, wi = int(valid_hw[i, 0].item()), int(valid_hw[i, 1].item())
                p_i = pred[i:i + 1, :, :hi, :wi]
                g_i = gt[i:i + 1, :, :hi, :wi]
                l_i, d_i = self._compute_single(p_i, g_i)
                total_l = total_l + l_i / float(b)
                for k in details_acc:
                    details_acc[k] += d_i[k] / float(b)
            details_acc['total_loss'] = float(total_l.detach().item())
            return total_l, details_acc
        return self._compute_single(pred, gt)

    def _compute_single(self, pred, gt):
        # 1. Charbonnier
        loss_charb = torch.mean(torch.sqrt((pred - gt) ** 2 + 1e-6))

        # 2. Symmetric Differentiable Distance Transform Loss (2 chiều đều khả vi!)
        with torch.no_grad():
            gt_bin = (gt > 0.15).float()
            dt_g = approx_gpu_distance_map(gt_bin, steps=self.dt_steps)
            if self.cldice is not None:
                skel_gt = soft_skeletonize(gt, iters=2)

        # Phạt nét vẽ giả xa GT + kéo nét vẽ tới đúng nét GT còn thiếu
        loss_dt_p2g = torch.mean(pred * dt_g)
        loss_dt_g2p = soft_inverse_dt_coverage(pred, gt_bin, steps=self.dt_steps)
        loss_dt = loss_dt_p2g + 0.15 * loss_dt_g2p

        # 3. Sobel Edge Gradient
        gx_p = F.conv2d(pred, self.sobel_x, padding=1)
        gy_p = F.conv2d(pred, self.sobel_y, padding=1)
        gx_g = F.conv2d(gt, self.sobel_x, padding=1)
        gy_g = F.conv2d(gt, self.sobel_y, padding=1)
        loss_sobel = F.l1_loss(gx_p, gx_g) + F.l1_loss(gy_p, gy_g)

        # 4. Soft-clDice
        loss_cldice = self.cldice(pred, gt, t_skel=skel_gt) if self.cldice is not None else pred.new_zeros(())

        # 5. Perceptual / VGG
        loss_perc = self.perceptual(pred, gt) if self.perceptual is not None else pred.new_zeros(())

        total_loss = (
            1.0 * loss_charb
            + self.w_dt * loss_dt
            + 0.25 * loss_sobel
            + self.w_cldice * loss_cldice
            + self.w_perc * loss_perc
        )
        return total_loss, {
            'loss_charb': float(loss_charb.detach().item()),
            'loss_dt': float(loss_dt.detach().item()),
            'loss_sobel': float(loss_sobel.detach().item()),
            'loss_cldice': float(loss_cldice.detach().item()),
            'loss_perc': float(loss_perc.detach().item()),
            'total_loss': float(total_loss.detach().item()),
        }


# ==============================================================================
# 4. HỆ THỐNG ĐO LƯỜNG CHỈ SỐ KHÁCH QUAN KÉP (DUAL-REGIME ZERO-BIAS METRICS)
# ==============================================================================
def compute_wcd_batch_gpu(p_bin, g_bin):
    """Weighted Chamfer Distance trên mặt nạ nhị phân [B, 1, H, W]."""
    dist_g = approx_gpu_distance_map(g_bin, steps=10)
    dist_p = approx_gpu_distance_map(p_bin, steps=10)
    d_p_to_g = torch.sum(p_bin * F.softplus(dist_g * 10.0), dim=(-2, -1)) / torch.clamp(torch.sum(p_bin, dim=(-2, -1)), min=1.0)
    d_g_to_p = torch.sum(g_bin * F.softplus(dist_p * 10.0), dim=(-2, -1)) / torch.clamp(torch.sum(g_bin, dim=(-2, -1)), min=1.0)
    area_p = torch.sum(p_bin, dim=(-2, -1))
    area_g = torch.sum(g_bin, dim=(-2, -1))
    h_weight = 0.5 * (1.0 + torch.minimum(area_p, area_g) / torch.clamp(torch.maximum(area_p, area_g), min=1.0))
    return h_weight * (d_p_to_g + d_g_to_p)


def compute_semd_batch_gpu(p_ink, g_ink):
    """1D-Sliced Wasserstein EMD trên phân bố nét vẽ dọc hai trục X/Y."""
    p_norm = p_ink / torch.clamp(torch.sum(p_ink, dim=(-2, -1), keepdim=True), min=1e-6)
    g_norm = g_ink / torch.clamp(torch.sum(g_ink, dim=(-2, -1), keepdim=True), min=1e-6)
    emd_h = torch.mean(torch.abs(torch.cumsum(torch.sum(p_norm, dim=-1), dim=-1) - torch.cumsum(torch.sum(g_norm, dim=-1), dim=-1)), dim=(-2, -1))
    emd_w = torch.mean(torch.abs(torch.cumsum(torch.sum(p_norm, dim=-2), dim=-1) - torch.cumsum(torch.sum(g_norm, dim=-2), dim=-1)), dim=(-2, -1))
    return 0.5 * (emd_h + emd_w)


_DOG_KERNEL_CACHE = {}


def _gaussian_blur_2d(img, sigma):
    key = (round(float(sigma), 4), img.device, img.dtype)
    k2d = _DOG_KERNEL_CACHE.get(key)
    if k2d is None:
        rad = max(1, int(round(3.0 * sigma)))
        coords = torch.arange(-rad, rad + 1, device=img.device, dtype=torch.float32)
        g1 = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
        g1 = g1 / g1.sum()
        k2d = torch.outer(g1, g1).view(1, 1, 2 * rad + 1, 2 * rad + 1).to(img.dtype)
        _DOG_KERNEL_CACHE[key] = k2d
    rad = k2d.shape[-1] // 2
    return F.conv2d(F.pad(img, (rad, rad, rad, rad), mode='replicate'), k2d)


def compute_dog_bin_mask(ink_img, sigma=1.0, k=1.6, tau=0.02):
    """
    Bộ lọc Difference-of-Gaussians (DoG) trích xuất nét thuần túy,
    triệt tiêu hoàn toàn độ xám nền (nền xám nhạt vs nền trắng tinh) để đánh giá công bằng M1 vs M3.
    """
    gray = 1.0 - torch.clamp(ink_img, 0.0, 1.0)
    g1 = _gaussian_blur_2d(gray, sigma)
    g2 = _gaussian_blur_2d(gray, sigma * k)
    dog = g1 - g2
    return (dog < -tau).float()


@torch.no_grad()
def evaluate_metrics_dual_regime(pred_ink, gt_ink):
    """
    Đo lường 10+ chỉ số khách quan trên cùng 1 cặp ảnh (pred_ink, gt_ink) shape [B, 1, H, W]:
      - Regime A (Raw Continuous): PSNR, SSIM, IE_MAE (0..255), PWCR, RGD, Ghost_Px, Dark_Ghost_Px, Val_L1
      - Regime B (Thresholded tau=0.15 & DoG): WCD, WCD_DoG, S-EMD, Skeleton Recall, Hair-Tip Recall,
                                               Stroke Recall, Stroke Precision, Stroke F1, Density Ratio
    """
    p = torch.clamp(pred_ink.float(), 0.0, 1.0)
    g = torch.clamp(gt_ink.float(), 0.0, 1.0)

    # ---- REGIME A: RAW CONTINUOUS ----
    mse = torch.mean((p - g) ** 2, dim=(-3, -2, -1))
    psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-8))

    mu_p = F.avg_pool2d(p, 11, stride=1, padding=5)
    mu_g = F.avg_pool2d(g, 11, stride=1, padding=5)
    sig_p = F.avg_pool2d(p * p, 11, stride=1, padding=5) - mu_p ** 2
    sig_g = F.avg_pool2d(g * g, 11, stride=1, padding=5) - mu_g ** 2
    sig_pg = F.avg_pool2d(p * g, 11, stride=1, padding=5) - mu_p * mu_g
    ssim_map = ((2 * mu_p * mu_g + 1e-4) * (2 * sig_pg + 9e-4)) / ((mu_p ** 2 + mu_g ** 2 + 1e-4) * (sig_p + sig_g + 9e-4))
    ssim = torch.mean(ssim_map, dim=(-3, -2, -1))

    val_l1 = torch.mean(torch.abs(p - g), dim=(-3, -2, -1))
    ie_mae = val_l1 * 255.0

    # Pixel brightness domain [0, 255]
    p_255 = (1.0 - p) * 255.0
    g_255 = (1.0 - g) * 255.0
    pwcr = torch.mean((p_255 >= 250.0).float(), dim=(-3, -2, -1))
    rgd = torch.mean(((p_255 >= 180.0) & (p_255 < 250.0)).float(), dim=(-3, -2, -1))

    gt_white_safe = (g_255 > 250.0).float()
    gt_stroke_safe = soft_dilate(soft_dilate((g > 0.08).float()))
    outside_gt = (1.0 - gt_stroke_safe) * gt_white_safe
    ghost_px = torch.sum(((p_255 >= 180.0) & (p_255 <= 250.0)).float() * gt_white_safe, dim=(-3, -2, -1))
    dark_ghost_px = torch.sum((p > 0.35).float() * (1.0 - gt_stroke_safe), dim=(-3, -2, -1))

    # ---- REGIME B: BINARIZED TOPOLOGY (TH=0.15 + DoG) ----
    p_bin = (p > 0.15).float()
    g_bin = (g > 0.15).float()
    p_dog = compute_dog_bin_mask(p)
    g_dog = compute_dog_bin_mask(g)

    wcd = compute_wcd_batch_gpu(p_bin, g_bin)
    wcd_dog = compute_wcd_batch_gpu(p_dog, g_dog)
    semd = compute_semd_batch_gpu(p, g)

    # Stroke Recall / Precision / F1 (dung sai 2px dilation cho nét mảnh 1px)
    g_dil = soft_dilate(soft_dilate(g_bin))
    p_dil = soft_dilate(soft_dilate(p_bin))
    tp_r = torch.sum(g_bin * p_dil, dim=(-3, -2, -1))
    tp_p = torch.sum(p_bin * g_dil, dim=(-3, -2, -1))
    cnt_p = torch.sum(p_bin, dim=(-3, -2, -1))
    cnt_g = torch.sum(g_bin, dim=(-3, -2, -1))
    recall = tp_r / torch.clamp(cnt_g, min=1.0)
    precision = tp_p / torch.clamp(cnt_p, min=1.0)
    f1 = 2.0 * recall * precision / torch.clamp(recall + precision, min=1e-6)
    density_ratio = cnt_p / torch.clamp(cnt_g, min=1.0)

    # Skeleton Recall & Hair-Tip Recall
    skel_g = soft_skeletonize(g, iters=2)
    skel_mask = (skel_g > 0.20).float()
    p_dil1 = soft_dilate(p_bin)
    skel_recall = torch.sum(p_dil1 * skel_mask, dim=(-3, -2, -1)) / torch.clamp(torch.sum(skel_mask, dim=(-3, -2, -1)), min=1.0)

    nbrs = F.avg_pool2d(skel_g, kernel_size=3, stride=1, padding=1) * 9.0 - skel_g
    endpoints = ((nbrs <= 1.8) & (skel_g > 0.20)).float()
    tip_recall = torch.sum(p_dil1 * endpoints, dim=(-3, -2, -1)) / torch.clamp(torch.sum(endpoints, dim=(-3, -2, -1)), min=1.0)

    return {
        'psnr': psnr.squeeze(-1) if psnr.ndim > 1 else psnr,
        'ssim': ssim.squeeze(-1) if ssim.ndim > 1 else ssim,
        'val_l1': val_l1.squeeze(-1) if val_l1.ndim > 1 else val_l1,
        'ie_mae': ie_mae.squeeze(-1) if ie_mae.ndim > 1 else ie_mae,
        'pwcr': pwcr.squeeze(-1) if pwcr.ndim > 1 else pwcr,
        'rgd': rgd.squeeze(-1) if rgd.ndim > 1 else rgd,
        'ghost': ghost_px.squeeze(-1) if ghost_px.ndim > 1 else ghost_px,
        'dark_ghost': dark_ghost_px.squeeze(-1) if dark_ghost_px.ndim > 1 else dark_ghost_px,
        'wcd': wcd.squeeze(-1) if wcd.ndim > 1 else wcd,
        'wcd_dog': wcd_dog.squeeze(-1) if wcd_dog.ndim > 1 else wcd_dog,
        'semd': semd.squeeze(-1) if semd.ndim > 1 else semd,
        'recall': recall.squeeze(-1) if recall.ndim > 1 else recall,
        'precision': precision.squeeze(-1) if precision.ndim > 1 else precision,
        'f1': f1.squeeze(-1) if f1.ndim > 1 else f1,
        'density_ratio': density_ratio.squeeze(-1) if density_ratio.ndim > 1 else density_ratio,
        'skel_recall': skel_recall.squeeze(-1) if skel_recall.ndim > 1 else skel_recall,
        'tip_recall': tip_recall.squeeze(-1) if tip_recall.ndim > 1 else tip_recall,
    }


# ==============================================================================
# 5. DATASET CHUỖI SEQUENCE ĐA KHUNG HÌNH N={3,5,7} & DDP BUCKET SAMPLER
# ==============================================================================
def pad_to_canvas(arr_2d, target_h, target_w):
    """Đệm ảnh ink [H, W] bằng nền trắng (0.0) ở đáy/phải tới (target_h, target_w)."""
    h, w = arr_2d.shape
    if h == target_h and w == target_w:
        return arr_2d
    out = np.zeros((target_h, target_w), dtype=arr_2d.dtype)
    out[:h, :w] = arr_2d
    return out


class SakugaSequenceDataset(Dataset):
    """
    Dataset Sequence cho cả 3 chế độ (SHORT / MEDIUM / LONG) và Validation:
      - `mode == 'LONG'`: trả về NGUYÊN CHUỖI (x0, x1, xk_all[M], t_vals[M]) ở độ phân giải gốc 540p.
      - `mode == 'SHORT'`: mỗi epoch lấy 1 frame trung gian ngẫu nhiên/chuỗi + Ink Crop 384x384.
      - `mode == 'MEDIUM'`: mở rộng thành toàn bộ 3,169 mẫu đơn + Motion-BBox Crop 512x512.
    """
    def __init__(self, metadata_csv, clips_dir, tps_cache_path, train_mode="LONG", is_train=True, seed=42):
        super().__init__()
        self.clips_dir = clips_dir
        self.train_mode = train_mode.upper()
        self.is_train = is_train
        self.df = pd.read_csv(metadata_csv).reset_index(drop=True)

        # Mmap nhẹ trên CPU RAM (~ vài chục MB dùng chung)
        self.tps_cache_path = tps_cache_path
        cache_data = np.load(tps_cache_path, mmap_mode='r')
        self.c0_arr = cache_data['c0']
        self.th01_arr = cache_data['th01']
        self.c1_arr = cache_data['c1']
        self.th10_arr = cache_data['th10']
        self.valid_arr = cache_data['valid']
        self.orig_hw_arr = cache_data['orig_hw']

        self.samples = []
        if self.is_train and self.train_mode == "LONG":
            # 1 item = toàn bộ 1 sequence (x0, x1, toàn bộ N-2 frames trung gian)
            for idx, row in self.df.iterrows():
                cid = str(row['clip_id'])
                seq_len = int(row['seq_len'])
                cache_idx = int(row['cache_idx']) if 'cache_idx' in row else idx
                hw = self.orig_hw_arr[cache_idx]
                t_list = [float(k) / float(seq_len - 1) for k in range(1, seq_len - 1)]
                self.samples.append({
                    'clip_id': cid,
                    'scene_id': str(row.get('scene_id', '')),
                    'seq_len': seq_len,
                    'cache_idx': cache_idx,
                    'orig_h': int(hw[0]),
                    'orig_w': int(hw[1]),
                    't_list': t_list,
                    'mean_disp': float(row.get('mean_disp', 0.0)),
                })
        elif self.is_train and self.train_mode == "SHORT":
            # 1 item = 1 chuỗi (tự chọn 1 frame ngẫu nhiên mỗi epoch trong __getitem__)
            for idx, row in self.df.iterrows():
                cid = str(row['clip_id'])
                seq_len = int(row['seq_len'])
                cache_idx = int(row['cache_idx']) if 'cache_idx' in row else idx
                hw = self.orig_hw_arr[cache_idx]
                self.samples.append({
                    'clip_id': cid,
                    'scene_id': str(row.get('scene_id', '')),
                    'seq_len': seq_len,
                    'cache_idx': cache_idx,
                    'orig_h': int(hw[0]),
                    'orig_w': int(hw[1]),
                    'frame_k': None,
                    'mean_disp': float(row.get('mean_disp', 0.0)),
                })
        else:
            # Validation hoặc MEDIUM Train: mở rộng thành mọi frame trung gian (t_k, I_k)
            for idx, row in self.df.iterrows():
                cid = str(row['clip_id'])
                seq_len = int(row['seq_len'])
                cache_idx = int(row['cache_idx']) if 'cache_idx' in row else idx
                hw = self.orig_hw_arr[cache_idx]
                for k in range(1, seq_len - 1):
                    self.samples.append({
                        'clip_id': cid,
                        'scene_id': str(row.get('scene_id', '')),
                        'seq_len': seq_len,
                        'cache_idx': cache_idx,
                        'orig_h': int(hw[0]),
                        'orig_w': int(hw[1]),
                        'frame_k': k + 1,  # 1-based file index: 000002..00000(N-1)
                        't_val': float(k) / float(seq_len - 1),
                        'mean_disp': float(row.get('mean_disp', 0.0)),
                    })

    def __len__(self):
        return len(self.samples)

    def _read_ink(self, clip_id, frame_idx_1b):
        p = os.path.join(self.clips_dir, clip_id, f"{frame_idx_1b:06d}.png")
        with Image.open(p) as im:
            arr = np.asarray(im.convert('L'), dtype=np.float32)
        return (255.0 - arr) / 255.0

    def __getitem__(self, idx):
        s = self.samples[idx]
        cid = s['clip_id']
        seq_len = s['seq_len']
        cache_idx = s['cache_idx']

        x0 = self._read_ink(cid, 1)
        x1 = self._read_ink(cid, seq_len)

        c0 = np.array(self.c0_arr[cache_idx], copy=True)
        th01 = np.array(self.th01_arr[cache_idx], copy=True)
        c1 = np.array(self.c1_arr[cache_idx], copy=True)
        th10 = np.array(self.th10_arr[cache_idx], copy=True)
        valid = float(self.valid_arr[cache_idx])
        orig_h, orig_w = x0.shape

        if self.is_train and self.train_mode == "LONG":
            # Tải đồng thời toàn bộ N-2 khung hình trung gian của chuỗi
            xk_list = [self._read_ink(cid, f_idx) for f_idx in range(2, seq_len)]
            xk_all = np.stack(xk_list, axis=0)  # [M, H, W]
            t_arr = np.array(s['t_list'], dtype=np.float32)

            # Data Augmentation: Temporal Reversal (đảo chiều I0 <-> I1, xk[::-1], t <-> 1-t)
            if random.random() < 0.5:
                x0, x1 = x1, x0
                c0, c1 = c1, c0
                th01, th10 = th10, th01
                xk_all = xk_all[::-1].copy()
                t_arr = (1.0 - t_arr[::-1]).copy()

            return {
                'is_sequence': True,
                'x0': torch.from_numpy(x0).unsqueeze(0),               # [1, H, W]
                'x1': torch.from_numpy(x1).unsqueeze(0),               # [1, H, W]
                'xk_all': torch.from_numpy(xk_all).unsqueeze(1),       # [M, 1, H, W]
                't_vals': torch.from_numpy(t_arr),                     # [M]
                'c0': torch.from_numpy(c0),
                'th01': torch.from_numpy(th01),
                'c1': torch.from_numpy(c1),
                'th10': torch.from_numpy(th10),
                'valid': torch.tensor(valid, dtype=torch.float32),
                'orig_hw': torch.tensor([orig_h, orig_w], dtype=torch.int32),
                'seq_len': seq_len,
                'clip_id': cid,
                'scene_id': s['scene_id'],
                'mean_disp': s['mean_disp'],
            }

        # SINGLE FRAME (SHORT / MEDIUM / VAL)
        if s['frame_k'] is None:
            k_rel = random.randint(1, seq_len - 2)
            fk = k_rel + 1
            t_val = float(k_rel) / float(seq_len - 1)
        else:
            fk = s['frame_k']
            t_val = s['t_val']

        xk = self._read_ink(cid, fk)
        if self.is_train and random.random() < 0.5:
            x0, x1 = x1, x0
            c0, c1 = c1, c0
            th01, th10 = th10, th01
            t_val = 1.0 - t_val

        return {
            'is_sequence': False,
            'x0': torch.from_numpy(x0).unsqueeze(0),
            'x1': torch.from_numpy(x1).unsqueeze(0),
            'xk': torch.from_numpy(xk).unsqueeze(0),
            't_val': torch.tensor([t_val], dtype=torch.float32),
            'c0': torch.from_numpy(c0),
            'th01': torch.from_numpy(th01),
            'c1': torch.from_numpy(c1),
            'th10': torch.from_numpy(th10),
            'valid': torch.tensor(valid, dtype=torch.float32),
            'orig_hw': torch.tensor([orig_h, orig_w], dtype=torch.int32),
            'seq_len': seq_len,
            'frame_k': fk,
            'clip_id': cid,
            'scene_id': s['scene_id'],
            'mean_disp': s['mean_disp'],
        }


def pad_collate_fn(batch):
    """
    Collate tự động đệm các ảnh trong cùng micro-batch lên kích thước chia hết cho 32
    mà KHÔNG làm biến dạng tỷ lệ nét vẽ. Giữ nguyên `orig_hw` để TPS tính chuẩn xác.
    """
    max_h = max(item['orig_hw'][0].item() for item in batch)
    max_w = max(item['orig_hw'][1].item() for item in batch)
    h_pad = ((max_h + 31) // 32) * 32
    w_pad = ((max_w + 31) // 32) * 32

    is_seq = batch[0]['is_sequence']

    def _pad_t(t, th, tw):
        dh = th - t.shape[-2]
        dw = tw - t.shape[-1]
        if dh == 0 and dw == 0:
            return t
        return F.pad(t, (0, dw, 0, dh), mode='constant', value=0.0)

    out = {
        'is_sequence': is_seq,
        'h_pad': h_pad,
        'w_pad': w_pad,
        'x0': torch.stack([_pad_t(b['x0'], h_pad, w_pad) for b in batch], dim=0),
        'x1': torch.stack([_pad_t(b['x1'], h_pad, w_pad) for b in batch], dim=0),
        'c0': torch.stack([b['c0'] for b in batch], dim=0),
        'th01': torch.stack([b['th01'] for b in batch], dim=0),
        'c1': torch.stack([b['c1'] for b in batch], dim=0),
        'th10': torch.stack([b['th10'] for b in batch], dim=0),
        'valid': torch.stack([b['valid'] for b in batch], dim=0),
        'orig_hw': torch.stack([b['orig_hw'] for b in batch], dim=0),
        'seq_len': torch.tensor([b['seq_len'] for b in batch], dtype=torch.int32),
        'clip_id': [b['clip_id'] for b in batch],
        'scene_id': [b['scene_id'] for b in batch],
        'mean_disp': [b['mean_disp'] for b in batch],
    }
    if is_seq:
        # Lưu ý: Sampler đảm bảo tất cả item trong cùng sequence batch có cùng seq_len -> cùng M!
        out['xk_all'] = torch.stack([_pad_t(b['xk_all'], h_pad, w_pad) for b in batch], dim=0)
        out['t_vals'] = torch.stack([b['t_vals'] for b in batch], dim=0)
    else:
        out['xk'] = torch.stack([_pad_t(b['xk'], h_pad, w_pad) for b in batch], dim=0)
        out['t_val'] = torch.stack([b['t_val'] for b in batch], dim=0)
        out['frame_k'] = [b['frame_k'] for b in batch]
    return out


class DDPLengthWidthBatchSampler(Sampler):
    """
    Sampler cân bằng tải tuyệt đối giữa 2 GPU (97-100% GPU Utilization):
      1. Nhóm các sequence theo `seq_len` (N=3, 5, 7) và sắp xếp theo chiều rộng `orig_w`
         để không lãng phí padding.
      2. Tạo micro-batch kích thước chuẩn cho từng N:
         - N=3 (1 frame giữa): batch = 4 seqs (4 frames / GPU forward)
         - N=5 (3 frames giữa): batch = 2 seqs (6 frames / GPU) [hoặc 1 nếu W > 1120]
         - N=7 (5 frames giữa): batch = 1 seq  (5 frames / GPU)
      3. Ghép từng cặp batch có cùng `seq_len` cho (Rank 0, Rank 1) để hai GPU không bao giờ
         lệch tải khi đồng bộ NCCL all_reduce!
    """
    def __init__(self, samples, world_size=1, rank=0, batch_n3=4, batch_n5=2, batch_n7=1, seed=42):
        self.samples = samples
        self.world_size = max(1, world_size)
        self.rank = rank
        self.bs_map = {3: batch_n3, 5: batch_n5, 7: batch_n7}
        self.seed = seed
        self.epoch = 0
        self._rank_batches = []
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.epoch = epoch
        rng = random.Random(self.seed + epoch * 997)

        # Phân loại theo seq_len
        by_len = {}
        for idx, s in enumerate(self.samples):
            sl = s['seq_len']
            by_len.setdefault(sl, []).append((s['orig_w'], rng.random(), idx))

        # Gom thành các micro-batch gần tương đồng chiều rộng cho từng seq_len
        global_micros = {}
        for sl, items in sorted(by_len.items()):
            # Sắp xếp gần theo width nhưng có nhiễu nhẹ giữa các epoch để thứ tự ghép batch phong phú
            items_sorted = sorted(items, key=lambda x: (((x[0] + 31) // 32) * 32, x[1]))
            indices = [x[2] for x in items_sorted]
            bs_base = self.bs_map.get(sl, 1)

            batches_for_len = []
            cur = []
            for idx in indices:
                w = self.samples[idx]['orig_w']
                bs_eff = 1 if (sl >= 5 and w > 1120 and bs_base > 1) else bs_base
                cur.append(idx)
                if len(cur) >= bs_eff:
                    batches_for_len.append((sl, self.samples[idx]['orig_w'], cur))
                    cur = []
            if cur:
                # Chèn phần dư vào micro-batch cùng độ dài hoặc giữ lại nếu hợp lệ
                if batches_for_len and len(cur) < max(1, bs_base // 2):
                    pass
                else:
                    batches_for_len.append((sl, self.samples[cur[-1]]['orig_w'], cur))
            global_micros[sl] = batches_for_len

        # Ghép cặp micro-batches theo world_size để mọi rank có cùng seq_len ở mỗi step
        sync_steps = []
        for sl, b_list in global_micros.items():
            # Đảm bảo số batch chia hết cho world_size bằng cách xoay nhẹ 1 phần tử
            rem = len(b_list) % self.world_size
            if rem != 0 and len(b_list) >= self.world_size:
                b_list = b_list[:-rem]
            for j in range(0, len(b_list), self.world_size):
                tuple_batches = [b_list[j + r][2] for r in range(self.world_size)]
                sync_steps.append(tuple_batches)

        rng.shuffle(sync_steps)
        self._rank_batches = [step[self.rank] for step in sync_steps]

    def __iter__(self):
        return iter(self._rank_batches)

    def __len__(self):
        return len(self._rank_batches)


# ==============================================================================
# 6. EMA SHADOW & CHECKPOINT MANAGER (TUYỆT ĐỐI CHỈ LƯU 2 FILE: BEST VÀ LAST)
# ==============================================================================
class AdaptiveEMA:
    """
    Exponential Moving Average nhẹ trên GPU với Adaptive Warmup Decay:
      decay(step) = min(max_decay, (1 + step) / (10 + step))
    Giúp trọng số thích ứng nhanh ở 3 epoch đầu rồi làm mượt cực tốt ở cuối kỳ.
    """
    def __init__(self, model, max_decay=0.995):
        self.max_decay = max_decay
        self.num_updates = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self._backup = None

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        d = min(self.max_decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=(1.0 - d))
            else:
                self.shadow[k].copy_(v)

    def apply_shadow(self, model):
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model):
        if self._backup is not None:
            model.load_state_dict(self._backup, strict=True)
            self._backup = None

    def state_dict(self):
        return {'num_updates': self.num_updates, 'shadow': self.shadow}

    def load_state_dict(self, state, device):
        if not state:
            return
        self.num_updates = int(state.get('num_updates', 0))
        loaded_sh = state.get('shadow', {})
        for k in self.shadow:
            if k in loaded_sh and self.shadow[k].shape == loaded_sh[k].shape:
                self.shadow[k].copy_(loaded_sh[k].to(device))


class TwoFileCheckpointManager:
    """
    Chính sách lưu trữ tối giản tuyệt đối theo yêu cầu đồ án:
    CHỈ DUY TRÌ ĐÚNG 2 FILE TRONG `output_dir`:
      1. model_tps_finetuned_best.pt : Lưu trọng số tốt nhất trên tập Val (EMA weights).
      2. model_tps_finetuned_last.pt : Lưu trọng số Epoch gần nhất KÈM trạng thái Resume đầy đủ.
    Tuyệt đối KHÔNG sinh thêm file thứ 3 hoặc các file rời từng epoch!
    """
    def __init__(self, output_dir, is_rank0=True):
        self.output_dir = output_dir
        self.is_rank0 = is_rank0
        self.best_score = float('inf')
        self.best_epoch = 0
        os.makedirs(output_dir, exist_ok=True)
        self.best_path = os.path.join(output_dir, "model_tps_finetuned_best.pt")
        self.last_path = os.path.join(output_dir, "model_tps_finetuned_last.pt")

    def _atomic_save(self, obj, dst_path):
        tmp_path = dst_path + ".tmp"
        torch.save(obj, tmp_path)
        if os.path.exists(dst_path):
            os.remove(dst_path)
        os.rename(tmp_path, dst_path)

    def save_epoch(self, epoch, base_model, ema, optimizer, scaler, score, history, is_best):
        if not self.is_rank0:
            return

        if is_best:
            self.best_score = float(score)
            self.best_epoch = int(epoch)
            best_payload = {
                'epoch': int(epoch),
                'best_score': float(score),
                'model_state': {k: v.detach().cpu() for k, v in ema.shadow.items()},
                'raw_model_state': {k: v.detach().cpu() for k, v in base_model.state_dict().items()},
                'history': history,
            }
            self._atomic_save(best_payload, self.best_path)
            print(f"🌟 [KỶ LỤC MỚI]: Cập nhật {os.path.basename(self.best_path)} tại Epoch {epoch:02d} (Composite Score = {score:.4f})!", flush=True)

        # Lưu gộp Epoch gần nhất + trạng thái Resume vào duy nhất file last.pt
        last_payload = {
            'epoch': int(epoch),
            'model_state': {k: v.detach().cpu() for k, v in base_model.state_dict().items()},
            'ema_state': {
                'num_updates': ema.num_updates,
                'shadow': {k: v.detach().cpu() for k, v in ema.shadow.items()},
            },
            'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
            'scaler_state': scaler.state_dict() if scaler is not None else None,
            'best_score': float(self.best_score),
            'best_epoch': int(self.best_epoch),
            'history': history,
        }
        self._atomic_save(last_payload, self.last_path)

    def try_resume(self, base_model, ema, optimizer, scaler, map_location):
        if not os.path.exists(self.last_path):
            return 0, []
        ckpt = torch.load(self.last_path, map_location=map_location, weights_only=False)
        last_ep = int(ckpt.get('epoch', 0))
        self.best_score = float(ckpt.get('best_score', float('inf')))
        self.best_epoch = int(ckpt.get('best_epoch', 0))
        history = ckpt.get('history', [])

        if 'model_state' in ckpt:
            base_model.load_state_dict(ckpt['model_state'], strict=True)
        if 'ema_state' in ckpt and ckpt['ema_state'] is not None:
            ema.load_state_dict(ckpt['ema_state'], map_location)
        if optimizer is not None and ckpt.get('optimizer_state') is not None:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        if scaler is not None and ckpt.get('scaler_state') is not None:
            scaler.load_state_dict(ckpt['scaler_state'])
        return last_ep, history


# ==============================================================================
# 7. VALIDATION ĐỐI CHỨNG & CHỌN CHECKPOINT (KHÔNG BIAS)
# ==============================================================================
@torch.no_grad()
def run_validation_suite(model, val_loader, device, downscale_factor=2, is_ddp=False, evaluate_m0=False):
    model.eval()
    keys_sum = ['psnr', 'ssim', 'val_l1', 'ie_mae', 'pwcr', 'rgd', 'ghost', 'dark_ghost',
                'wcd', 'wcd_dog', 'semd', 'recall', 'precision', 'f1', 'density_ratio', 'skel_recall', 'tip_recall']
    acc_main = {k: 0.0 for k in keys_sum}
    acc_main['count'] = 0.0
    acc_m0 = {k: 0.0 for k in keys_sum}
    acc_m0['count'] = 0.0

    brk_keys = ['count', 'psnr', 'ssim', 'wcd', 'wcd_dog', 'semd', 'skel_recall', 'f1', 'ghost']
    breakdown = {n: {k: 0.0 for k in brk_keys} for n in (3, 5, 7)}

    for batch in val_loader:
        h_pad, w_pad = batch['h_pad'], batch['w_pad']
        x0 = batch['x0'].to(device, non_blocking=True)
        x1 = batch['x1'].to(device, non_blocking=True)
        xk = batch['xk'].to(device, non_blocking=True)
        t_val = batch['t_val'].to(device, non_blocking=True).view(-1, 1, 1, 1)
        c0 = batch['c0'].to(device, non_blocking=True)
        th01 = batch['th01'].to(device, non_blocking=True)
        c1 = batch['c1'].to(device, non_blocking=True)
        th10 = batch['th10'].to(device, non_blocking=True)
        valid = batch['valid'].to(device, non_blocking=True)
        orig_hw = batch['orig_hw']
        seq_lens = batch['seq_len']

        cf01 = tps_flow_from_theta_batch(th01, c0, valid, orig_hw, h_pad, w_pad, device, downscale_factor=downscale_factor)
        cf10 = tps_flow_from_theta_batch(th10, c1, valid, orig_hw, h_pad, w_pad, device, downscale_factor=downscale_factor)

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            pred_it, coarse_warp, *_ = model(x0, x1, t_val, cf01, cf10)

        b = x0.shape[0]
        for i in range(b):
            hi, wi = int(orig_hw[i, 0].item()), int(orig_hw[i, 1].item())
            sl = int(seq_lens[i].item())
            gt_i = xk[i:i + 1, :, :hi, :wi]
            pr_i = pred_it[i:i + 1, :, :hi, :wi]

            m = evaluate_metrics_dual_regime(pr_i, gt_i)
            acc_main['count'] += 1.0
            for k in keys_sum:
                acc_main[k] += float(m[k].item())

            if sl in breakdown:
                breakdown[sl]['count'] += 1.0
                for bk in brk_keys:
                    if bk != 'count':
                        breakdown[sl][bk] += float(m[bk].item())

            if evaluate_m0:
                m0_pr = coarse_warp[i:i + 1, :, :hi, :wi]
                mm0 = evaluate_metrics_dual_regime(m0_pr, gt_i)
                acc_m0['count'] += 1.0
                for k in keys_sum:
                    acc_m0[k] += float(mm0[k].item())

    def _finalize(acc, brk=None):
        if is_ddp:
            vec = torch.tensor([acc['count']] + [acc[k] for k in keys_sum], dtype=torch.float64, device=device)
            dist.all_reduce(vec, op=dist.ReduceOp.SUM)
            cnt = max(1.0, float(vec[0].item()))
            out = {k: float(vec[idx + 1].item()) / cnt for idx, k in enumerate(keys_sum)}
            out['total_samples'] = int(cnt)
        else:
            cnt = max(1.0, acc['count'])
            out = {k: acc[k] / cnt for k in keys_sum}
            out['total_samples'] = int(cnt)

        if brk is not None:
            brk_out = {}
            for sl in (3, 5, 7):
                if is_ddp:
                    bvec = torch.tensor([brk[sl][k] for k in brk_keys], dtype=torch.float64, device=device)
                    dist.all_reduce(bvec, op=dist.ReduceOp.SUM)
                    bcnt = max(1.0, float(bvec[0].item()))
                    brk_out[sl] = {k: float(bvec[idx].item()) / bcnt for idx, k in enumerate(brk_keys) if k != 'count'}
                    brk_out[sl]['count'] = int(bvec[0].item())
                else:
                    bcnt = max(1.0, brk[sl]['count'])
                    brk_out[sl] = {k: brk[sl][k] / bcnt for k in brk_keys if k != 'count'}
                    brk_out[sl]['count'] = int(brk[sl]['count'])
            out['breakdown'] = brk_out
        return out

    res_main = _finalize(acc_main, breakdown)
    if evaluate_m0:
        return res_main, _finalize(acc_m0)
    return res_main


# ==============================================================================
# 8. ĐỌC TẬP BENCHMARK ĐỘC LẬP SONG SONG TRÊN 2 GPU
# ==============================================================================
@torch.no_grad()
def evaluate_per_sample_benchmark(model, bench_loader, device, downscale_factor=2):
    """Chạy mô hình trên dataloader Benchmark và trả về dict keyed by (clip_id, frame_k)."""
    model.eval()
    results = {}
    for batch in bench_loader:
        h_pad, w_pad = batch['h_pad'], batch['w_pad']
        x0 = batch['x0'].to(device, non_blocking=True)
        x1 = batch['x1'].to(device, non_blocking=True)
        xk = batch['xk'].to(device, non_blocking=True)
        t_val = batch['t_val'].to(device, non_blocking=True).view(-1, 1, 1, 1)
        c0 = batch['c0'].to(device, non_blocking=True)
        th01 = batch['th01'].to(device, non_blocking=True)
        c1 = batch['c1'].to(device, non_blocking=True)
        th10 = batch['th10'].to(device, non_blocking=True)
        valid = batch['valid'].to(device, non_blocking=True)
        orig_hw = batch['orig_hw']

        cf01 = tps_flow_from_theta_batch(th01, c0, valid, orig_hw, h_pad, w_pad, device, downscale_factor=downscale_factor)
        cf10 = tps_flow_from_theta_batch(th10, c1, valid, orig_hw, h_pad, w_pad, device, downscale_factor=downscale_factor)

        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        t_start = time.time()
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            pred_it, coarse_warp, *_ = model(x0, x1, t_val, cf01, cf10)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        lat_ms = ((time.time() - t_start) * 1000.0) / max(1, x0.shape[0])

        for i in range(x0.shape[0]):
            hi, wi = int(orig_hw[i, 0].item()), int(orig_hw[i, 1].item())
            gt_i = xk[i:i + 1, :, :hi, :wi]
            pr_i = pred_it[i:i + 1, :, :hi, :wi]
            m0_i = coarse_warp[i:i + 1, :, :hi, :wi]

            m_pred = evaluate_metrics_dual_regime(pr_i, gt_i)
            m_coarse = evaluate_metrics_dual_regime(m0_i, gt_i)

            rec = {
                'clip_id': batch['clip_id'][i],
                'scene_id': batch['scene_id'][i],
                'seq_len': int(batch['seq_len'][i].item()),
                'frame_k': int(batch['frame_k'][i]),
                't_val': float(batch['t_val'][i].item()),
                'mean_disp': float(batch['mean_disp'][i]),
                'latency_ms': float(lat_ms),
            }
            for k, v in m_pred.items():
                rec[k] = float(v.item())
            for k, v in m_coarse.items():
                rec[f"m0_{k}"] = float(v.item())
            results[(rec['clip_id'], rec['frame_k'])] = rec
    return results


# ==============================================================================
# 9. CHƯƠNG TRÌNH HUẤN LUYỆN CHÍNH (DDP WORKER)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="DDP Worker Fine-tune TPS-Inbetween (Option 3 LONG mặc định)")
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--train_clips_dir', type=str, required=True, help="Thư mục train/")
    parser.add_argument('--val_clips_dir', type=str, required=True, help="Thư mục val/")
    parser.add_argument('--tps_cache', type=str, required=True)
    parser.add_argument('--m1_checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default="/kaggle/working/checkpoints_tps_finetune")
    parser.add_argument('--train_mode', type=str, default="LONG", choices=["SHORT", "MEDIUM", "LONG"])
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--grad_accum', type=int, default=2)
    parser.add_argument('--max_chunk_frames', type=int, default=2, help="Số frame tối đa mỗi forward micro-chunk giữ VRAM <=6GB")
    parser.add_argument('--lr_backbone', type=float, default=1.0e-5)
    parser.add_argument('--lr_unet', type=float, default=2.5e-5)
    parser.add_argument('--ema_decay', type=float, default=0.995)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--ram_limit_gb', type=float, default=23.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry_run', action='store_true', help="Chạy kiểm tra 2 bước khởi động nhanh rồi thoát")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp = world_size > 1

    if is_ddp:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend, init_method='env://')
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    is_rank0 = (local_rank == 0)
    random.seed(args.seed + local_rank)
    np.random.seed(args.seed + local_rank)
    torch.manual_seed(args.seed + local_rank)

    mode = args.train_mode.upper()
    tps_downscale = 2 if mode == "LONG" else 4

    if is_rank0:
        print("=" * 84)
        print(f"🚀 KHỞI ĐỘNG HUẤN LUYỆN DDP TPS-INBETWEEN (100% ZERO ARCH CHANGE | MODE={mode})")
        print(f"   World Size = {world_size} GPU(s) | Device = {device} | TPS Downscale = 1/{tps_downscale}")
        print(f"   Epochs = {args.epochs} | Grad Accum = {args.grad_accum} | Max Chunk Frames = {args.max_chunk_frames}")
        print(f"   Checkpoint Output = {args.output_dir} (Chỉ lưu 2 file: best.pt & last.pt)")
        print("=" * 84, flush=True)

    ckpt_mgr = TwoFileCheckpointManager(args.output_dir, is_rank0=is_rank0)
    base_model = TPSInbetweenOriginal().to(device)
    load_tps_weights(base_model, args.m1_checkpoint, device=device, prefer_ema=False, is_rank0=is_rank0)

    if mode == "SHORT":
        for p in base_model.IFNet.block0.parameters():
            p.requires_grad = False

    # Dataset & Loader
    train_ds = SakugaSequenceDataset(args.train_csv, args.train_clips_dir, args.tps_cache, train_mode=mode, is_train=True, seed=args.seed)
    val_ds = SakugaSequenceDataset(args.val_csv, args.val_clips_dir, args.tps_cache, train_mode="VAL", is_train=False, seed=args.seed)

    if mode == "LONG":
        train_sampler = DDPLengthWidthBatchSampler(
            train_ds.samples, world_size=world_size, rank=local_rank,
            batch_n3=4, batch_n5=2, batch_n7=1, seed=args.seed
        )
        train_loader = DataLoader(
            train_ds, batch_sampler=train_sampler, collate_fn=pad_collate_fn,
            num_workers=args.num_workers, pin_memory=(device.type == 'cuda'),
            persistent_workers=(args.num_workers > 0)
        )
    else:
        bs_single = 12 if mode == "SHORT" else 8
        train_sampler = DistributedSampler(train_ds, shuffle=True) if is_ddp else None
        train_loader = DataLoader(
            train_ds, batch_size=bs_single, shuffle=(train_sampler is None), sampler=train_sampler,
            collate_fn=pad_collate_fn, num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'), persistent_workers=(args.num_workers > 0), drop_last=True
        )

    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_ddp else None
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False, sampler=val_sampler,
        collate_fn=pad_collate_fn, num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'), persistent_workers=(args.num_workers > 0), drop_last=False
    )

    if is_rank0:
        print(f"[✓] Dữ liệu sẵn sàng: Train Items = {len(train_ds):,} ({len(train_loader):,} batches/GPU) | Val Samples = {len(val_ds):,}", flush=True)

    # Optimizer, EMA, Loss & Scaler
    ema = AdaptiveEMA(base_model, max_decay=args.ema_decay)
    optimizer = torch.optim.AdamW([
        {'params': [p for p in base_model.encoder.parameters() if p.requires_grad], 'lr': args.lr_backbone, 'weight_decay': 1e-4},
        {'params': [p for p in base_model.IFNet.parameters() if p.requires_grad], 'lr': args.lr_backbone, 'weight_decay': 1e-4},
        {'params': [p for p in base_model.unet.parameters() if p.requires_grad], 'lr': args.lr_unet, 'weight_decay': 1e-4},
    ])
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    criterion = AdaptiveLineartLoss(train_mode=mode, is_rank0=is_rank0).to(device)

    start_epoch = 1
    history = []
    if not args.dry_run:
        last_ep, hist = ckpt_mgr.try_resume(base_model, ema, optimizer, scaler, map_location=device)
        if last_ep > 0:
            start_epoch = last_ep + 1
            history = hist
            if is_rank0:
                print(f"[♻️ AUTO-RESUME] Khôi phục thành công Epoch {last_ep:02d}. Tiếp tục từ Epoch {start_epoch:02d} (Best Ep={ckpt_mgr.best_epoch:02d}).", flush=True)

    # Đo Baseline M0 & M1 tại Epoch 0 trước khi học
    baseline_json_path = os.path.join(os.path.dirname(args.output_dir), "val_baseline_m0_m1.json")
    if start_epoch == 1 and not args.dry_run and not os.path.exists(baseline_json_path):
        if is_rank0:
            print("\n🔬 ĐANG ĐO BASELINE EPOCH 0 TRÊN TẬP VAL (M0 COARSE TPS & M1 ORIGINAL)...", flush=True)
        m1_val, m0_val = run_validation_suite(base_model, val_loader, device, downscale_factor=tps_downscale, is_ddp=is_ddp, evaluate_m0=True)
        if is_rank0:
            with open(baseline_json_path, 'w', encoding='utf-8') as f:
                json.dump({'M0_Coarse': m0_val, 'M1_Original': m1_val}, f, indent=2, ensure_ascii=False)
            print(f"📊 M0 (Coarse TPS): PSNR={m0_val['psnr']:.2f}dB | SSIM={m0_val['ssim']:.4f} | WCD={m0_val['wcd']:.4f} | SkelRec={m0_val['skel_recall']*100:.1f}%", flush=True)
            print(f"📊 M1 (Orig ML240): PSNR={m1_val['psnr']:.2f}dB | SSIM={m1_val['ssim']:.4f} | WCD={m1_val['wcd']:.4f} | SkelRec={m1_val['skel_recall']*100:.1f}%\n", flush=True)

    ddp_model = DDP(base_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=(mode == "SHORT")) if is_ddp else base_model

    history_csv_path = os.path.join(os.path.dirname(args.output_dir), "training_history_finetune.csv")
    watchdog_paused = False
    t_total_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        t_ep0 = time.time()
        if hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)

        # Lịch Cosine Warmup 2 epochs
        if epoch == 1:
            lr_scale = 0.35
        elif epoch == 2:
            lr_scale = 0.70
        elif epoch == 3:
            lr_scale = 1.00
        else:
            prog = float(epoch - 3) / float(max(1, args.epochs - 3))
            lr_scale = max(0.10, 0.5 * (1.0 + math.cos(math.pi * prog)))

        optimizer.param_groups[0]['lr'] = args.lr_backbone * lr_scale
        optimizer.param_groups[1]['lr'] = args.lr_backbone * lr_scale
        optimizer.param_groups[2]['lr'] = args.lr_unet * lr_scale

        ddp_model.train()
        optimizer.zero_grad(set_to_none=True)
        acc_loss = 0.0
        acc_frames = 0
        num_batches = len(train_loader)
        grad_accum = max(1, args.grad_accum)

        for b_idx, batch in enumerate(train_loader):
            h_pad, w_pad = batch['h_pad'], batch['w_pad']
            x0 = batch['x0'].to(device, non_blocking=True)
            x1 = batch['x1'].to(device, non_blocking=True)
            c0 = batch['c0'].to(device, non_blocking=True)
            th01 = batch['th01'].to(device, non_blocking=True)
            c1 = batch['c1'].to(device, non_blocking=True)
            th10 = batch['th10'].to(device, non_blocking=True)
            valid = batch['valid'].to(device, non_blocking=True)
            orig_hw = batch['orig_hw']

            cf01 = tps_flow_from_theta_batch(th01, c0, valid, orig_hw, h_pad, w_pad, device, downscale_factor=tps_downscale)
            cf10 = tps_flow_from_theta_batch(th10, c1, valid, orig_hw, h_pad, w_pad, device, downscale_factor=tps_downscale)

            if batch['is_sequence']:
                # Option 3 (LONG): Chuỗi đa khung hình M = seq_len - 2 frames [B, M, 1, H, W]
                xk_all = batch['xk_all'].to(device, non_blocking=True)
                t_vals = batch['t_vals'].to(device, non_blocking=True)  # [B, M]
                bsz, m_steps = xk_all.shape[0], xk_all.shape[1]

                # Dựng danh sách cặp mở rộng [B*M, ...] rồi chia nhỏ micro-chunk <= max_chunk_frames
                x0_f = x0.unsqueeze(1).expand(-1, m_steps, -1, -1, -1).reshape(bsz * m_steps, 1, h_pad, w_pad)
                x1_f = x1.unsqueeze(1).expand(-1, m_steps, -1, -1, -1).reshape(bsz * m_steps, 1, h_pad, w_pad)
                cf01_f = cf01.unsqueeze(1).expand(-1, m_steps, -1, -1, -1).reshape(bsz * m_steps, 2, h_pad, w_pad)
                cf10_f = cf10.unsqueeze(1).expand(-1, m_steps, -1, -1, -1).reshape(bsz * m_steps, 2, h_pad, w_pad)
                xk_f = xk_all.reshape(bsz * m_steps, 1, h_pad, w_pad)
                t_f = t_vals.reshape(bsz * m_steps, 1, 1, 1)
                hw_f = orig_hw.unsqueeze(1).expand(-1, m_steps, -1).reshape(bsz * m_steps, 2)
            else:
                xk_f = batch['xk'].to(device, non_blocking=True)
                t_f = batch['t_val'].to(device, non_blocking=True).view(-1, 1, 1, 1)
                x0_f, x1_f, cf01_f, cf10_f, hw_f = x0, x1, cf01, cf10, orig_hw

            total_f = x0_f.shape[0]
            do_opt_step = ((b_idx + 1) % grad_accum == 0) or ((b_idx + 1) == num_batches)
            chunk_sz = max(1, args.max_chunk_frames)

            batch_loss_sum = 0.0
            for c_start in range(0, total_f, chunk_sz):
                c_end = min(c_start + chunk_sz, total_f)
                is_last_chunk = (c_end == total_f)
                need_sync = (do_opt_step and is_last_chunk)

                # Quản lý no_sync() để DDP chỉ đồng bộ gradient ở bước cuối cùng của accum
                sync_ctx = (ddp_model.no_sync() if (is_ddp and not need_sync) else torch.enable_grad())
                with sync_ctx:
                    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        pred_c, *_ = ddp_model(
                            x0_f[c_start:c_end], x1_f[c_start:c_end],
                            t_f[c_start:c_end], cf01_f[c_start:c_end], cf10_f[c_start:c_end]
                        )
                        l_c, d_c = criterion(pred_c, xk_f[c_start:c_end], valid_hw=hw_f[c_start:c_end])
                        weight_scale = float(c_end - c_start) / float(total_f * grad_accum)
                        scaled_l = l_c * weight_scale
                    scaler.scale(scaled_l).backward()
                    batch_loss_sum += d_c['total_loss'] * float(c_end - c_start)

            acc_loss += batch_loss_sum
            acc_frames += total_f

            if do_opt_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(base_model)

            if args.dry_run and b_idx >= 1:
                break

        train_min = (time.time() - t_ep0) / 60.0
        if args.dry_run:
            if is_rank0:
                print(f"[✓ DRY-RUN THÀNH CÔNG] Đã kiểm chứng {acc_frames} frames qua DDP forward/backward + EMA! Avg Loss={acc_loss/max(1,acc_frames):.4f}", flush=True)
            break

        # Kiểm định bằng trọng số EMA trên toàn bộ tập Val (tạm tráo EMA -> restore, 0 tốn VRAM)
        t_val0 = time.time()
        ema.apply_shadow(base_model)
        val_m = run_validation_suite(base_model, val_loader, device, downscale_factor=tps_downscale, is_ddp=is_ddp, evaluate_m0=False)
        ema.restore(base_model)
        val_min = (time.time() - t_val0) / 60.0

        # Thước đo cân bằng không bias nền trắng để chọn Best Checkpoint
        comp_score = (
            0.45 * val_m['wcd']
            + 0.25 * (val_m['semd'] * 3.0)
            + 0.20 * (1.0 - val_m['skel_recall'])
            + 0.10 * val_m['val_l1']
        )
        is_best = comp_score < ckpt_mgr.best_score

        if is_rank0:
            row = {
                'epoch': epoch,
                'lr_scale': round(lr_scale, 4),
                'train_loss': acc_loss / max(1, acc_frames),
                'val_l1': val_m['val_l1'],
                'psnr': val_m['psnr'],
                'ssim': val_m['ssim'],
                'wcd': val_m['wcd'],
                'wcd_dog': val_m['wcd_dog'],
                'semd': val_m['semd'],
                'skel_recall': val_m['skel_recall'],
                'tip_recall': val_m['tip_recall'],
                'recall': val_m['recall'],
                'f1': val_m['f1'],
                'density_ratio': val_m['density_ratio'],
                'pwcr': val_m['pwcr'],
                'rgd': val_m['rgd'],
                'ghost': val_m['ghost'],
                'score': comp_score,
                'is_best': is_best,
                'train_time_min': train_min,
                'val_time_min': val_min,
            }
            for sl in (3, 5, 7):
                if sl in val_m.get('breakdown', {}):
                    row[f'psnr_N{sl}'] = val_m['breakdown'][sl]['psnr']
                    row[f'wcd_N{sl}'] = val_m['breakdown'][sl]['wcd']
                    row[f'wcd_dog_N{sl}'] = val_m['breakdown'][sl]['wcd_dog']
                    row[f'skel_recall_N{sl}'] = val_m['breakdown'][sl]['skel_recall']

            history.append(row)
            pd.DataFrame(history).to_csv(history_csv_path, index=False)
            ckpt_mgr.save_epoch(epoch, base_model, ema, optimizer, scaler, comp_score, history, is_best=is_best)

            print(
                f"📌 EP {epoch:02d}/{args.epochs:02d} [{mode}] | Train={train_min:.2f}m | Val={val_min:.2f}m | "
                f"Loss={row['train_loss']:.4f} | PSNR={val_m['psnr']:.2f}dB | "
                f"WCD={val_m['wcd']:.4f} (DoG={val_m['wcd_dog']:.4f}) | SkelRec={val_m['skel_recall']*100:.1f}% | "
                f"F1={val_m['f1']*100:.1f}% | Score={comp_score:.4f}",
                flush=True
            )

        trim_memory()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        stop_flag = torch.tensor([0], dtype=torch.int32, device=device)
        if is_rank0:
            try:
                import psutil
                mem_gb = psutil.virtual_memory().used / (1024 ** 3)
                if mem_gb >= args.ram_limit_gb:
                    print(f"\n⚠️ [RAM WATCHDOG]: Host RAM đạt {mem_gb:.2f} GB >= {args.ram_limit_gb} GB. Thoát an toàn mã 42!", flush=True)
                    stop_flag.fill_(1)
            except Exception:
                pass

        if is_ddp:
            dist.broadcast(stop_flag, src=0)
            dist.barrier()

        if int(stop_flag.item()) == 1:
            watchdog_paused = True
            break

    if is_ddp:
        dist.destroy_process_group()

    if watchdog_paused:
        sys.exit(42)
    if is_rank0 and not args.dry_run:
        print(f"\n🎉 HOÀN TẤT HUẤN LUYỆN TRONG {(time.time() - t_total_start)/60.0:.1f} PHÚT!", flush=True)


if __name__ == '__main__':
    main()
