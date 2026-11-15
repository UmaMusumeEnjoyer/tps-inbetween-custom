#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ddp_worker_v30r.py
Dedicated DistributedDataParallel (DDP) Training & Validation Engine for TPS-Inbetween V3.0-Remastered R2.
Tích hợp toàn diện:
1. 100% Khớp trọng số V2.6 Gold Baseline (168/168 tensors, detail_add * 0.25, asf_head).
2. Tự động nạp/giải mmap TPS 2 chiều (.npy) 0-copy, triệt tiêu 100% lỗi coarse flow = 0.
3. Time-Symmetric FBFC Engine tính toán sự nhất quán thời gian trên khung t.
4. Soft-Knee Gamma RGBA Lineart Exporter (tau_noise=0.020, gamma=0.78, RGB 0,0,0).
5. Continuous Loss Engine: Truncated DT Loss + Soft Tversky-clDice + FineStroke/Tip + FBFC Ghost.
6. Hệ thống đánh giá chuẩn nhận thức & hình thái học: WCD + S-EMD + Skeleton/Tip Recall.
7. Cổng kiểm định Epoch 0 Sanity Gate & Tự động lưu checkpoint theo SmartCheckpointManager_v30R.
"""

import os
import sys
import time
import math
import glob
import json
import psutil
import shutil
import argparse
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# Cấu hình môi trường phân tán & unbuffered stdout
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# =============================================================================
# 1. TOÁN TỬ BIẾN DẠNG LƯỚI & VECTOR HÓA COARSE TPS FLOW
# =============================================================================
_NORM_GRID_CACHE = {}
_PIXEL_GRID_CACHE = {}
_GRID_CACHE = {}

def warp_normalized_flow(img, flow):
    b, _, h, w = img.shape
    dev = flow.device
    key = (h, w, dev)
    base_grid = _NORM_GRID_CACHE.get(key)
    if base_grid is None:
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, h, device=dev, dtype=torch.float32),
            torch.linspace(0, 1, w, device=dev, dtype=torch.float32), indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).contiguous()
        _NORM_GRID_CACHE[key] = base_grid

    with torch.amp.autocast('cuda', enabled=False):
        flow_f32 = torch.nan_to_num(flow.float(), nan=0.0, posinf=0.0, neginf=0.0)
        grid = ((base_grid + flow_f32.permute(0, 2, 3, 1)) * 2.0 - 1.0).contiguous()
        out = F.grid_sample(img.float().contiguous(), grid, padding_mode='border', align_corners=True)
    return out.to(img.dtype)

def warp_pixel_flow(img, flow):
    b, _, h, w = img.shape
    dev = flow.device
    key = (h, w, dev)
    base_grid = _PIXEL_GRID_CACHE.get(key)
    if base_grid is None:
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=dev, dtype=torch.float32),
            torch.linspace(-1, 1, w, device=dev, dtype=torch.float32), indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).contiguous()
        _PIXEL_GRID_CACHE[key] = base_grid

    with torch.amp.autocast('cuda', enabled=False):
        flow_f32 = torch.nan_to_num(flow.float(), nan=0.0, posinf=0.0, neginf=0.0)
        u = flow_f32[:, 0, :, :] / ((w - 1.0) * 0.5)
        v = flow_f32[:, 1, :, :] / ((h - 1.0) * 0.5)
        flow_norm = torch.stack([u, v], dim=-1).contiguous()
        grid = (base_grid + flow_norm).contiguous()
        out = F.grid_sample(img.float().contiguous(), grid, padding_mode='border', align_corners=True)
    return out.to(img.dtype)

def tps_rbf(r):
    return (r ** 2) * torch.log(torch.clamp(r, min=1e-6))

def filter_outlier_keypoints_cpu(c_s, c_d, max_pts=80, min_dist=0.015):
    n = len(c_s)
    if n < 4:
        return c_s, c_d

    disp = c_d - c_s
    dists = np.linalg.norm(disp, axis=1)
    median_d = np.median(dists)
    mad = np.median(np.abs(dists - median_d)) + 1e-6
    mask_inliers = dists <= (median_d + 3.5 * mad)
    
    c_s_filt = c_s[mask_inliers]
    c_d_filt = c_d[mask_inliers]
    if len(c_s_filt) < 4:
        c_s_filt = c_s
        c_d_filt = c_d

    selected_s = []
    selected_d = []
    min_dist_sq = min_dist ** 2
    for p_s, p_d in zip(c_s_filt, c_d_filt):
        if len(selected_s) >= max_pts:
            break
        too_close = False
        for prev in selected_s:
            if ((p_s[0] - prev[0])**2 + (p_s[1] - prev[1])**2) < min_dist_sq:
                too_close = True
                break
        if not too_close:
            selected_s.append(p_s)
            selected_d.append(p_d)

    return np.array(selected_s, dtype=np.float32), np.array(selected_d, dtype=np.float32)

def solve_tps_single_cpu(c_src, c_dst, max_kps=80, lambd=0.5, min_dist=0.015, max_pts=88, return_tensor=True):
    if c_src is None or c_dst is None or len(c_src) < 4:
        z_c = np.zeros((max_pts, 2), dtype=np.float32)
        z_th = np.zeros((max_pts + 3, 2), dtype=np.float32)
        if return_tensor:
            return torch.from_numpy(z_c), torch.from_numpy(z_th), 0.0
        return z_c, z_th, 0.0

    c_s_clean, c_d_clean = filter_outlier_keypoints_cpu(c_src, c_dst, max_pts=max_kps, min_dist=min_dist)
    if len(c_s_clean) < 4:
        z_c = np.zeros((max_pts, 2), dtype=np.float32)
        z_th = np.zeros((max_pts + 3, 2), dtype=np.float32)
        if return_tensor:
            return torch.from_numpy(z_c), torch.from_numpy(z_th), 0.0
        return z_c, z_th, 0.0

    anchors = np.array([
        [0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
        [0.0, 0.5],             [1.0, 0.5],
        [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]
    ], dtype=np.float32)

    c_s_anchors = np.vstack([c_s_clean, anchors])
    c_d_anchors = np.vstack([c_d_clean, anchors])
    m = len(c_s_anchors)

    diff = c_s_anchors[:, None, :] - c_s_anchors[None, :, :]
    r = np.linalg.norm(diff, axis=-1)
    K = (r ** 2) * np.log(np.maximum(r, 1e-6))
    np.fill_diagonal(K, lambd)

    P = np.hstack([np.ones((m, 1), dtype=np.float32), c_s_anchors])

    A = np.zeros((m + 3, m + 3), dtype=np.float32)
    A[:m, :m] = K
    A[:m, m:] = P
    A[m:, :m] = P.T

    Y = np.zeros((m + 3, 2), dtype=np.float32)
    Y[:m, :] = c_d_anchors

    try:
        theta = np.linalg.solve(A, Y)
    except np.linalg.LinAlgError:
        theta, _, _, _ = np.linalg.lstsq(A, Y, rcond=1e-4)
    v_flag = 0.0 if (np.isnan(theta).any() or np.isinf(theta).any() or np.max(np.abs(theta)) > 50.0) else 1.0

    c_pad = np.zeros((max_pts, 2), dtype=np.float32)
    theta_pad = np.zeros((max_pts + 3, 2), dtype=np.float32)
    if v_flag > 0.5:
        c_pad[:m] = c_s_anchors
        theta_pad[:m] = theta[:m]
        theta_pad[max_pts:max_pts+3] = theta[m:m+3]

    if return_tensor:
        return torch.from_numpy(c_pad), torch.from_numpy(theta_pad), float(v_flag)
    return c_pad, theta_pad, float(v_flag)

def solve_tps_pair_cpu(kps, max_kps=80, lambd=0.5, min_dist=0.015, return_tensor=True):
    MAX_PTS = 88
    if kps is None or len(kps) < 2 or (hasattr(kps, 'shape') and len(kps.shape) > 1 and kps.shape[1] < 4):
        z_c0 = np.zeros((MAX_PTS, 2), dtype=np.float32)
        z_th01 = np.zeros((MAX_PTS + 3, 2), dtype=np.float32)
        z_c1 = np.zeros((MAX_PTS, 2), dtype=np.float32)
        z_th10 = np.zeros((MAX_PTS + 3, 2), dtype=np.float32)
        z_v = np.array(0.0, dtype=np.float32)
        if return_tensor:
            return (torch.from_numpy(z_c0), torch.from_numpy(z_th01), torch.from_numpy(z_c1),
                    torch.from_numpy(z_th10), torch.from_numpy(z_v))
        return z_c0, z_th01, z_c1, z_th10, z_v

    c0 = kps[0]
    c1 = kps[1]
    c0_pad, th01_pad, v01 = solve_tps_single_cpu(c0, c1, max_kps=max_kps, lambd=lambd, min_dist=min_dist, max_pts=MAX_PTS, return_tensor=False)
    c1_pad, th10_pad, v10 = solve_tps_single_cpu(c1, c0, max_kps=max_kps, lambd=lambd, min_dist=min_dist, max_pts=MAX_PTS, return_tensor=False)

    v = np.array(1.0 if (v01 > 0.5 and v10 > 0.5) else 0.0, dtype=np.float32)
    if return_tensor:
        return (torch.from_numpy(c0_pad), torch.from_numpy(th01_pad), torch.from_numpy(c1_pad),
                torch.from_numpy(th10_pad), torch.from_numpy(v))
    return c0_pad, th01_pad, c1_pad, th10_pad, v

def solve_tps_batch_all(kps_data, kps_count, w=960, h=540, pad_w=960, pad_h=544, max_kps=80, lambd=0.5):
    length = len(kps_count)
    c0_buf = np.zeros((length, 88, 2), dtype=np.float32)
    th01_buf = np.zeros((length, 91, 2), dtype=np.float32)
    c1_buf = np.zeros((length, 88, 2), dtype=np.float32)
    th10_buf = np.zeros((length, 91, 2), dtype=np.float32)
    valid_buf = np.zeros((length, 1, 1, 1), dtype=np.float32)

    def _solve_sample(idx):
        cnt = int(kps_count[idx])
        if cnt < 4:
            return
        kps = kps_data[idx, :, :cnt, :].copy()
        kps_norm = kps.copy()
        kps_norm[0, :, 0] = (kps_norm[0, :, 0] * w) / pad_w
        kps_norm[0, :, 1] = (kps_norm[0, :, 1] * h) / pad_h
        kps_norm[1, :, 0] = (kps_norm[1, :, 0] * w) / pad_w
        kps_norm[1, :, 1] = (kps_norm[1, :, 1] * h) / pad_h
        c0_p, th01_p, c1_p, th10_p, v = solve_tps_pair_cpu(kps_norm, max_kps=max_kps, lambd=lambd, return_tensor=False)
        c0_buf[idx] = c0_p
        th01_buf[idx] = th01_p
        c1_buf[idx] = c1_p
        th10_buf[idx] = th10_p
        valid_buf[idx, 0, 0, 0] = float(v)

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_solve_sample, range(length)))

    return c0_buf, th01_buf, c1_buf, th10_buf, valid_buf

def get_grid_pts(h, w, device):
    key = (h, w, device)
    pts = _GRID_CACHE.get(key)
    if pts is None:
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, h, device=device, dtype=torch.float32),
            torch.linspace(0, 1, w, device=device, dtype=torch.float32), indexing='ij'
        )
        pts = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2).contiguous()
        _GRID_CACHE[key] = pts
    return pts

@torch.no_grad()
def tps_flow_from_theta_batch(theta_batch, c_src_batch, valid_mask, h, w, device, chunk_size=65536, downscale_factor=4):
    with torch.amp.autocast('cuda', enabled=False):
        theta_f32 = torch.nan_to_num(theta_batch.float(), nan=0.0, posinf=0.0, neginf=0.0)
        c_src_f32 = torch.nan_to_num(c_src_batch.float(), nan=0.0, posinf=0.0, neginf=0.0)
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

        if total_pts <= chunk_size:
            dist = torch.cdist(pts_exp, c_src_f32, p=2.0)
            U = tps_rbf(dist)
            ones = torch.ones((b, total_pts, 1), device=device, dtype=pts.dtype)
            basis = torch.cat([U, ones, pts_exp], dim=-1)
            mapped = torch.bmm(basis, theta_f32)
        else:
            mapped_chunks = []
            for start in range(0, total_pts, chunk_size):
                end = min(start + chunk_size, total_pts)
                p_c = pts_exp[:, start:end, :]
                n_c = end - start
                dist_c = torch.cdist(p_c, c_src_f32, p=2.0)
                U_c = tps_rbf(dist_c)
                ones_c = torch.ones((b, n_c, 1), device=device, dtype=pts.dtype)
                basis_c = torch.cat([U_c, ones_c, p_c], dim=-1)
                mapped_chunks.append(torch.bmm(basis_c, theta_f32))
            mapped = torch.cat(mapped_chunks, dim=1)

        diff = mapped - pts_exp
        flow = diff.view(b, h_sub, w_sub, 2).permute(0, 3, 1, 2).contiguous()

        if downscale_factor > 1:
            flow = F.interpolate(flow, size=(h, w), mode='bilinear', align_corners=True)

        flow = flow * valid_f32.view(b, 1, 1, 1)
        flow = torch.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
        return flow

# =============================================================================
# 2. KIẾN TRÚC MÔ HÌNH TPS-INBETWEEN V3.0-REMASTERED R2 (100% V2.6 BACKBONE)
# =============================================================================
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

        # 3. Time-Symmetric FBFC Engine (Tính theo pixel)
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

        # 5. Synthesis UNet (Chuẩn V2.6: Nhận coarse_flow0t, coarse_flow1t)
        gate_erase, detail_add = self.unet(
            x0, x1,
            coarse_flow0t, coarse_flow1t,
            fine_I0t, fine_I1t, ifnet_mask, feats0, feats1
        )
        pred_t = torch.clamp(gate_erase * base_interp + detail_add, 0.0, 1.0)

        if self.training:
            return pred_t, blend_mask, conf_fbfc, delta_flow, fine_I0t, fine_I1t
        return pred_t, blend_mask, conf_fbfc

def export_pure_lineart_rgba(pred_ink_tensor, tau_noise=0.020, gamma=0.78):
    if isinstance(pred_ink_tensor, torch.Tensor):
        pred = pred_ink_tensor.detach().squeeze().clamp(0.0, 1.0).cpu().numpy()
    else:
        pred = np.clip(pred_ink_tensor.squeeze(), 0.0, 1.0)

    h, w = pred.shape
    alpha = np.zeros((h, w), dtype=np.uint8)

    mask = pred >= tau_noise
    if mask.any():
        p_val = pred[mask]
        t_knee = np.clip((p_val - tau_noise) / 0.040, 0.0, 1.0)
        smooth_gate = t_knee * t_knee * (3.0 - 2.0 * t_knee)
        
        norm_val = np.clip((p_val - tau_noise) / max(1e-4, 1.0 - tau_noise), 0.0, 1.0)
        alpha_f = smooth_gate * np.power(norm_val, gamma) * 255.0
        alpha[mask] = np.clip(alpha_f, 0, 255).astype(np.uint8)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = 0
    rgba[:, :, 1] = 0
    rgba[:, :, 2] = 0
    rgba[:, :, 3] = alpha
    return rgba

# =============================================================================
# 3. HÀM MẤT MÁT COMPOSITE V3.0-REMASTERED R2 (CONTINUOUS LOSS)
# =============================================================================
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

class TruncatedDifferentiableDTLoss(nn.Module):
    def __init__(self, steps=4, tau_clip=15.0):
        super().__init__()
        self.steps = steps
        self.tau_clip = tau_clip

    def _approx_dt_map(self, x):
        d = 1.0 - torch.clamp(x, 0.0, 1.0)
        accum = d.clone()
        curr = d
        for step_idx in range(1, self.steps + 1):
            curr = -F.max_pool2d(-curr, kernel_size=3, stride=1, padding=1)
            accum = accum + curr * (1.0 + 0.5 * step_idx)
        return torch.clamp(accum, 0.0, self.tau_clip)

    def forward(self, pred, target):
        p_dt = self._approx_dt_map(pred)
        with torch.no_grad():
            g_dt = self._approx_dt_map(target)
        return F.l1_loss(p_dt, g_dt)

class FastLaplacianEdgeLoss(nn.Module):
    def __init__(self):
        super(FastLaplacianEdgeLoss, self).__init__()
        lap_kernel = torch.tensor([
            [0.0,  1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0,  1.0, 0.0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('lap_kernel', lap_kernel)

    def forward(self, pred, target):
        p_f = pred.float()
        g_f = target.float()
        lap_p1 = F.conv2d(p_f, self.lap_kernel, padding=1)
        lap_g1 = F.conv2d(g_f, self.lap_kernel, padding=1)
        loss1 = F.l1_loss(lap_p1, lap_g1)

        p_sub = F.avg_pool2d(p_f, kernel_size=2, stride=2)
        g_sub = F.avg_pool2d(g_f, kernel_size=2, stride=2)
        lap_p2 = F.conv2d(p_sub, self.lap_kernel, padding=1)
        lap_g2 = F.conv2d(g_sub, self.lap_kernel, padding=1)
        loss2 = F.l1_loss(lap_p2, lap_g2)
        return loss1 + 0.5 * loss2

class SoftTverskyClDiceLoss_v30R(nn.Module):
    def __init__(self, iters=2, alpha=0.35, beta=0.65, smooth=1.0):
        super().__init__()
        self.iters = iters
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target, t_skel=None):
        p = torch.clamp(pred, 0.0, 1.0)
        g = torch.clamp(target, 0.0, 1.0)

        tp = torch.sum(p * g, dim=[-3, -2, -1])
        fp = torch.sum(p * (1.0 - g), dim=[-3, -2, -1])
        fn = torch.sum((1.0 - p) * g, dim=[-3, -2, -1])
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss_tversky = torch.mean(1.0 - tversky)

        p_skel = soft_skeletonize(p, iters=self.iters)
        if t_skel is None:
            t_skel = soft_skeletonize(g, iters=self.iters)

        tprec = (torch.sum(p_skel * g) + self.smooth) / (torch.sum(p_skel) + self.smooth)
        tsens = (torch.sum(t_skel * p) + self.smooth) / (torch.sum(t_skel) + self.smooth)
        cldice = 2.0 * (tprec * tsens) / (tprec + tsens + 1e-6)
        loss_cldice = 1.0 - cldice

        return 0.6 * loss_tversky + 0.4 * loss_cldice

class StrokeDensityRegularizationLoss_v30R(nn.Module):
    def __init__(self, target_min=1.06, target_max=1.20, target_center=1.10, ramp_weight=0.25):
        super().__init__()
        self.target_min = target_min
        self.target_max = target_max
        self.target_center = target_center
        self.ramp_weight = ramp_weight

    def forward(self, pred_ink, gt_ink):
        p_ink = torch.clamp(pred_ink, 0.0, 1.0)
        g_ink = torch.clamp(gt_ink, 0.0, 1.0)
        m_soft_p = torch.clamp((p_ink - 0.06) / 0.18, 0.0, 1.0) + self.ramp_weight * F.relu(p_ink - 0.24)
        m_soft_g = torch.clamp((g_ink - 0.06) / 0.18, 0.0, 1.0) + self.ramp_weight * F.relu(g_ink - 0.24)
        m_hard_p = (p_ink > 0.15).float()
        m_hard_g = (g_ink > 0.15).float()
        m_ste_p = m_hard_p + (m_soft_p - m_soft_p.detach())
        m_ste_g = m_hard_g + (m_soft_g - m_soft_g.detach())
        sum_p = torch.sum(m_ste_p, dim=[-3, -2, -1])
        sum_g = torch.sum(m_ste_g, dim=[-3, -2, -1])
        ratio = torch.clamp(sum_p / torch.clamp(sum_g, min=20.0), min=0.0, max=5.0)

        dev_low = torch.clamp(self.target_min - ratio, min=0.0, max=3.0)
        dev_high = torch.clamp(ratio - self.target_max, min=0.0, max=3.0)
        loss_low = 3.0 * (dev_low ** 2)
        loss_high = 1.2 * dev_high + 0.8 * (dev_high ** 2)
        center_pull = 0.04 * ((ratio - self.target_center) ** 2)
        return torch.mean(loss_low + loss_high + center_pull)

class FineStrokeAndTipLoss(nn.Module):
    def __init__(self, focal=3.0):
        super().__init__()
        self.focal = focal

    def forward(self, pred_ink, gt_ink, skel_gt=None):
        p_ink = torch.clamp(pred_ink, 0.0, 1.0)
        g_ink = torch.clamp(gt_ink, 0.0, 1.0)
        opened_g = soft_open(g_ink)
        thin_strokes = torch.clamp(g_ink - opened_g, 0.0, 1.0)

        if skel_gt is None:
            skel_gt = soft_skeletonize(g_ink, iters=2)

        nbrs = F.avg_pool2d(skel_gt, kernel_size=3, stride=1, padding=1) * 9.0 - skel_gt
        endpoints = (nbrs <= 1.8) & (skel_gt > 0.20)
        endpoints_roi = soft_dilate(endpoints.float())

        w_fine = 1.0 + 3.0 * thin_strokes + 2.5 * endpoints_roi
        under_pred = F.relu(g_ink - p_ink) * w_fine
        return self.focal * torch.mean(under_pred)

class FBFCOcclusionAwareGhostLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, conf_fbfc=None):
        pred_f = torch.clamp(pred, 0.0, 1.0)
        gt_f = torch.clamp(gt, 0.0, 1.0)
        gt_safe_mask = soft_dilate(soft_dilate((gt_f > 0.08).float()))
        outside_safe = 1.0 - gt_safe_mask

        if conf_fbfc is not None:
            if conf_fbfc.shape[-2:] != pred_f.shape[-2:]:
                conf = conf_fbfc[:, :, :pred_f.shape[-2], :pred_f.shape[-1]].detach()
            else:
                conf = conf_fbfc.detach()
        else:
            conf = torch.ones_like(pred_f)
        far_ghost = F.relu(pred_f - 0.08) * outside_safe * conf
        dark_ghost = F.relu(pred_f - 0.35) * outside_safe * conf
        return 1.0 * torch.mean(far_ghost) + 2.5 * torch.mean(dark_ghost)

class CompositeLossEngine_v30R(nn.Module):
    def __init__(self):
        super(CompositeLossEngine_v30R, self).__init__()
        self.dt_loss = TruncatedDifferentiableDTLoss(steps=4)
        self.laplacian_loss = FastLaplacianEdgeLoss()
        self.cldice = SoftTverskyClDiceLoss_v30R(iters=2, alpha=0.35, beta=0.65)
        self.density_loss = StrokeDensityRegularizationLoss_v30R(target_min=1.06, target_max=1.20, target_center=1.10)
        self.fine_tip_loss = FineStrokeAndTipLoss(focal=3.0)
        self.ghost_loss = FBFCOcclusionAwareGhostLoss()

    def forward(self, pred, target, conf_fbfc=None, epoch=1):
        loss_charb = torch.mean(torch.sqrt((pred - target) ** 2 + 1e-6))
        loss_dt = self.dt_loss(pred, target)
        loss_lap = self.laplacian_loss(pred, target)

        with torch.no_grad():
            skel_gt = soft_skeletonize(target, iters=2)

        loss_dice = self.cldice(pred, target, t_skel=skel_gt)
        loss_dens = self.density_loss(pred, target)
        loss_hair = self.fine_tip_loss(pred, target, skel_gt=skel_gt)
        loss_ghost = self.ghost_loss(pred, target, conf_fbfc=conf_fbfc)

        total = (1.0 * loss_charb + 
                 0.35 * loss_dt + 
                 0.20 * loss_lap + 
                 0.30 * loss_dice + 
                 0.25 * loss_hair + 
                 0.08 * loss_dens + 
                 0.05 * loss_ghost)

        return total, loss_charb, loss_dt, loss_dice, loss_dens, loss_hair, loss_ghost

# =============================================================================
# 4. DATASET & CACHE MMAP LOADER V3.0-REMASTERED R2
# =============================================================================
def _fast_read_match_npy(path, max_cache=256):
    try:
        with open(path, 'rb') as f:
            raw = f.read()
        data = np.frombuffer(raw[128:], dtype=np.float32)
        pts = data.reshape(-1, 2, 2)
        cnt = min(len(pts), max_cache)
        return pts[:cnt], cnt
    except Exception:
        try:
            pts = np.load(path, allow_pickle=False)
            cnt = min(len(pts), max_cache)
            return pts[:cnt], cnt
        except Exception:
            return None, 0

def ensure_tps_precomputed_v30r(df, data_root, cache_dir, split_name='train', max_pts=80):
    os.makedirs(cache_dir, exist_ok=True)
    c0_npy = os.path.join(cache_dir, f'{split_name}_c0_v30r.npy')
    th01_npy = os.path.join(cache_dir, f'{split_name}_th01_v30r.npy')
    c1_npy = os.path.join(cache_dir, f'{split_name}_c1_v30r.npy')
    th10_npy = os.path.join(cache_dir, f'{split_name}_th10_v30r.npy')
    valid_npy = os.path.join(cache_dir, f'{split_name}_valid_v30r.npy')

    all_exist = all(os.path.exists(p) for p in [c0_npy, th01_npy, c1_npy, th10_npy, valid_npy])
    if all_exist:
        return (
            np.load(c0_npy, mmap_mode='r'),
            np.load(th01_npy, mmap_mode='r'),
            np.load(c1_npy, mmap_mode='r'),
            np.load(th10_npy, mmap_mode='r'),
            np.load(valid_npy, mmap_mode='r')
        )

    # Thử tìm trên các thư mục phụ cận
    search_dirs = [data_root, os.path.dirname(data_root), '/tmp/sakuga_ssd_v30r', '/tmp/sakuga_ssd_v32', '/tmp/sakuga_ssd_v31']
    for sdir in search_dirs:
        if not sdir or not os.path.exists(sdir):
            continue
        for ver in ['v30r', 'v32', 'v31']:
            cand_files = [os.path.join(sdir, f'{split_name}_{suf}_{ver}.npy') for suf in ['c0', 'th01', 'c1', 'th10', 'valid']]
            if all(os.path.exists(p) for p in cand_files):
                for src_p, dst_p in zip(cand_files, [c0_npy, th01_npy, c1_npy, th10_npy, valid_npy]):
                    if not os.path.exists(dst_p):
                        shutil.copyfile(src_p, dst_p)
                return (
                    np.load(c0_npy, mmap_mode='r'),
                    np.load(th01_npy, mmap_mode='r'),
                    np.load(c1_npy, mmap_mode='r'),
                    np.load(th10_npy, mmap_mode='r'),
                    np.load(valid_npy, mmap_mode='r')
                )

    length = len(df)
    print(f"[*] Precomputing TPS cache đa luồng cho tập {split_name} ({length:,} mẫu)...", flush=True)

    match_col = 'match_file' if 'match_file' in df.columns else None
    kps_buf = np.zeros((length, 2, 256, 2), dtype=np.float32)
    cnt_buf = np.zeros(length, dtype=np.int32)

    def _load_match_worker(i):
        row = df.iloc[i]
        m_rel = str(row[match_col]) if match_col and pd.notna(row[match_col]) else None
        if not m_rel or m_rel == 'None':
            c_id = str(row.get('clip_id', ''))
            m_rel = f"matches/{c_id}/match_000001_000003.npy"
        m_rel = m_rel.replace('\\', '/').lstrip('/')
        m_path = os.path.join(data_root, m_rel)
        if not os.path.exists(m_path):
            alt = os.path.join(os.path.dirname(data_root), m_rel)
            if os.path.exists(alt): m_path = alt
        pts, cnt = _fast_read_match_npy(m_path, max_cache=256)
        if pts is not None and cnt > 0:
            if pts.ndim == 3 and pts.shape[0] == 2:
                c = min(pts.shape[1], 256)
                kps_buf[i, :, :c, :] = pts[:, :c, :2]
                cnt_buf[i] = c
            elif pts.ndim == 3 and pts.shape[1] == 2:
                c = min(pts.shape[0], 256)
                kps_buf[i, 0, :c, :] = pts[:c, 0, :2]
                kps_buf[i, 1, :c, :] = pts[:c, 1, :2]
                cnt_buf[i] = c
            elif pts.ndim == 2 and pts.shape[1] >= 4:
                c = min(pts.shape[0], 256)
                kps_buf[i, 0, :c, :] = pts[:c, :2]
                kps_buf[i, 1, :c, :] = pts[:c, 2:4]
                cnt_buf[i] = c

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(_load_match_worker, range(length)))

    c0_b, th01_b, c1_b, th10_b, valid_b = solve_tps_batch_all(
        kps_buf, cnt_buf, w=960, h=540, pad_w=960, pad_h=544, max_kps=max_pts, lambd=0.5
    )

    np.save(c0_npy, c0_b)
    np.save(th01_npy, th01_b)
    np.save(c1_npy, c1_b)
    np.save(th10_npy, th10_b)
    np.save(valid_npy, valid_b)

    del kps_buf, cnt_buf
    return (
        np.load(c0_npy, mmap_mode='r'),
        np.load(th01_npy, mmap_mode='r'),
        np.load(c1_npy, mmap_mode='r'),
        np.load(th10_npy, mmap_mode='r'),
        np.load(valid_npy, mmap_mode='r')
    )

def corridor_protected_descreen(s, tau_noise=0.015, stroke_thresh=0.05, kernel_size=5):
    mask_stroke = (s > stroke_thresh).astype(np.uint8)
    if mask_stroke.any():
        corridor = cv2.dilate(mask_stroke, np.ones((kernel_size, kernel_size), np.uint8))
        return np.where((corridor == 0) & (s < tau_noise), 0.0, s)
    return np.where(s < tau_noise, 0.0, s)

class SakugaGoldenDataset_v30R(Dataset):
    def __init__(self, df, data_root, is_train=True, cache_dir=None, precomputed_tps=None):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.is_train = is_train
        self.split_name = 'train' if is_train else 'val'

        c0 = 'frame0_path' if 'frame0_path' in self.df.columns else 'frame0'
        c1 = 'frame1_path' if 'frame1_path' in self.df.columns else 'frame1'
        ct = 'target_path' if 'target_path' in self.df.columns else 'frame_gt'

        self.p0_list = [str(x).replace('\\', '/').lstrip('/') for x in self.df[c0]]
        self.p1_list = [str(x).replace('\\', '/').lstrip('/') for x in self.df[c1]]
        self.pt_list = [str(x).replace('\\', '/').lstrip('/') for x in self.df[ct]]
        self.t_list = self.df['t'].values.astype(np.float32) if 't' in self.df.columns else np.full(len(self.df), 0.5, dtype=np.float32)

        if precomputed_tps is not None:
            self.tps_c0, self.tps_th01, self.tps_c1, self.tps_th10, self.tps_valid = precomputed_tps
        else:
            cd = cache_dir or os.path.join(data_root, 'dataset_cache_v3.0_remastered')
            self.tps_c0, self.tps_th01, self.tps_c1, self.tps_th10, self.tps_valid = ensure_tps_precomputed_v30r(
                self.df, self.data_root, cd, split_name=self.split_name
            )

    def __len__(self):
        return len(self.df)

    def _read_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros((540, 960), dtype=np.uint8)
        return img

    def __getitem__(self, idx):
        p0_rel = self.p0_list[idx]
        p1_rel = self.p1_list[idx]
        pt_rel = self.pt_list[idx]
        t_val = float(self.t_list[idx])

        p0 = os.path.join(self.data_root, p0_rel) if not os.path.isabs(p0_rel) else p0_rel
        p1 = os.path.join(self.data_root, p1_rel) if not os.path.isabs(p1_rel) else p1_rel
        pt = os.path.join(self.data_root, pt_rel) if not os.path.isabs(pt_rel) else pt_rel

        img0 = self._read_image(p0)
        img1 = self._read_image(p1)
        imgt = self._read_image(pt)

        s0 = (255.0 - img0.astype(np.float32)) * (1.0 / 255.0)
        s1 = (255.0 - img1.astype(np.float32)) * (1.0 / 255.0)
        st = (255.0 - imgt.astype(np.float32)) * (1.0 / 255.0)

        s0 = corridor_protected_descreen(s0, tau_noise=0.015, stroke_thresh=0.05)
        s1 = corridor_protected_descreen(s1, tau_noise=0.015, stroke_thresh=0.05)
        st = corridor_protected_descreen(st, tau_noise=0.015, stroke_thresh=0.05)

        h, w = s0.shape
        pad_h = ((h + 31) // 32) * 32
        pad_w = ((w + 31) // 32) * 32
        if pad_h != h or pad_w != w:
            s0 = np.pad(s0, ((0, pad_h - h), (0, pad_w - w)), mode='constant', constant_values=0.0)
            s1 = np.pad(s1, ((0, pad_h - h), (0, pad_w - w)), mode='constant', constant_values=0.0)
            st = np.pad(st, ((0, pad_h - h), (0, pad_w - w)), mode='constant', constant_values=0.0)

        c0 = self.tps_c0[idx].copy()
        th01 = self.tps_th01[idx].copy()
        c1 = self.tps_c1[idx].copy()
        th10 = self.tps_th10[idx].copy()
        v = self.tps_valid[idx].copy()

        if self.is_train and np.random.rand() > 0.5:
            s0, s1 = s1, s0
            t_val = 1.0 - t_val
            c0, c1 = c1, c0
            th01, th10 = th10, th01

        t0 = torch.from_numpy(s0).unsqueeze(0).float()
        t1 = torch.from_numpy(s1).unsqueeze(0).float()
        tt = torch.from_numpy(st).unsqueeze(0).float()
        t_tensor = torch.tensor(t_val, dtype=torch.float32)

        tps_tensors = (
            torch.from_numpy(c0).float(),
            torch.from_numpy(th01).float(),
            torch.from_numpy(c1).float(),
            torch.from_numpy(th10).float(),
            torch.from_numpy(v).float()
        )
        return t0, t1, tt, t_tensor, tps_tensors

# =============================================================================
# 5. ĐO LƯỜNG CHỈ SỐ NHẬN THỨC & HÌNH THÁI HỌC BATCHED TRÊN GPU
# =============================================================================
def calculate_metrics_batch_gpu(p, g, tau_bg=0.020, num_proj=16):
    with torch.no_grad():
        p = torch.clamp(p.float(), 0.0, 1.0)
        g = torch.clamp(g.float(), 0.0, 1.0)
        b, c, h, w = p.shape
        dev = p.device

        # 1. Truncated DT & Weighted Chamfer Distance (WCD)
        p_dt = 1.0 - p
        g_dt = 1.0 - g
        for _ in range(3):
            p_dt = p_dt + (-F.max_pool2d(-p_dt, 3, 1, 1))
            g_dt = g_dt + (-F.max_pool2d(-g_dt, 3, 1, 1))
        p_dt = torch.clamp(p_dt, 0.0, 12.0)
        g_dt = torch.clamp(g_dt, 0.0, 12.0)

        p_mass = torch.clamp(p, min=0.05)
        g_mass = torch.clamp(g, min=0.05)
        wcd_p2g = torch.sum(p_mass * g_dt, dim=[-3, -2, -1]) / torch.clamp(torch.sum(p_mass, dim=[-3, -2, -1]), min=1.0)
        wcd_g2p = torch.sum(g_mass * p_dt, dim=[-3, -2, -1]) / torch.clamp(torch.sum(g_mass, dim=[-3, -2, -1]), min=1.0)
        wcd = (0.5 * wcd_p2g + 0.5 * wcd_g2p) * (1.0 / max(h, w))

        # 2. 1D-Sliced Earth Mover's Distance
        angles = torch.linspace(0, math.pi, num_proj, device=dev)
        sin_a = torch.sin(angles).view(num_proj, 1, 1)
        cos_a = torch.cos(angles).view(num_proj, 1, 1)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=dev),
            torch.linspace(-1, 1, w, device=dev), indexing='ij'
        )
        proj_coords = cos_a * grid_x.unsqueeze(0) + sin_a * grid_y.unsqueeze(0)
        num_bins = 64
        bin_idx = torch.clamp(((proj_coords + math.sqrt(2)) / (2 * math.sqrt(2)) * (num_bins - 1)).long(), 0, num_bins - 1)

        p_flat = p.view(b, 1, h * w)
        g_flat = g.view(b, 1, h * w)
        bin_idx_flat = bin_idx.view(num_proj, h * w).unsqueeze(0).expand(b, -1, -1)

        p_cdf_list = []
        g_cdf_list = []
        for k in range(num_bins):
            mask_k = (bin_idx_flat == k).float()
            p_cdf_list.append(torch.sum(p_flat * mask_k, dim=-1))
            g_cdf_list.append(torch.sum(g_flat * mask_k, dim=-1))

        p_hist = torch.stack(p_cdf_list, dim=-1)
        g_hist = torch.stack(g_cdf_list, dim=-1)

        p_hist = p_hist / torch.clamp(torch.sum(p_hist, dim=-1, keepdim=True), min=1e-5)
        g_hist = g_hist / torch.clamp(torch.sum(g_hist, dim=-1, keepdim=True), min=1e-5)
        p_cdf = torch.cumsum(p_hist, dim=-1)
        g_cdf = torch.cumsum(g_hist, dim=-1)
        sliced_emd = torch.mean(torch.abs(p_cdf - g_cdf), dim=[-2, -1])

        # 3. Topology Skeleton Recall & Hair-Tip Recall
        skel_g = soft_skeletonize(g, iters=2)
        p_bin = (p > 0.15).float()
        g_bin = (g > 0.15).float()
        skel_recall = torch.sum(soft_dilate(p_bin) * (skel_g > 0.20).float(), dim=[-3, -2, -1]) / torch.clamp(torch.sum((skel_g > 0.20).float(), dim=[-3, -2, -1]), min=1.0)

        nbrs_g = F.avg_pool2d(skel_g, kernel_size=3, stride=1, padding=1) * 9.0 - skel_g
        endpoints_g = (nbrs_g <= 1.8) & (skel_g > 0.20)
        tip_recall = torch.sum(soft_dilate(p_bin) * endpoints_g.float(), dim=[-3, -2, -1]) / torch.clamp(torch.sum(endpoints_g.float(), dim=[-3, -2, -1]), min=1.0)

        # 4. Standard Metrics (Recall, Density, Ghost - Chuẩn V2.5/V2.6: max_pool2d k=7, p=3)
        sum_p = torch.sum(p_bin, dim=[-3, -2, -1])
        sum_g = torch.sum(g_bin, dim=[-3, -2, -1])
        density = sum_p / torch.clamp(sum_g, min=1.0)

        p_dilated = F.max_pool2d(p_bin, kernel_size=7, stride=1, padding=3)
        g_dilated = F.max_pool2d(g_bin, kernel_size=7, stride=1, padding=3)
        tp_dil = torch.sum((g_bin > 0) & (p_dilated > 0), dim=[-3, -2, -1])
        prec_dil = torch.sum((p_bin > 0) & (g_dilated > 0), dim=[-3, -2, -1])
        recall = tp_dil / torch.clamp(sum_g, min=1.0)
        precision = prec_dil / torch.clamp(sum_p, min=1.0)
        f1 = (2.0 * precision * recall) / torch.clamp(precision + recall, min=1e-5)

        strict_tp = torch.sum(p_bin * g_bin, dim=[-3, -2, -1])
        strict_recall = strict_tp / torch.clamp(sum_g, min=1.0)

        gt_safe = soft_dilate(soft_dilate((g > 0.08).float()))
        outside_gt = 1.0 - gt_safe
        ghost_px = torch.sum((p > 0.15).float() * outside_gt, dim=[-3, -2, -1])
        dark_ghost_px = torch.sum((p > 0.35).float() * outside_gt, dim=[-3, -2, -1])

        bg_leak = torch.sum((p > tau_bg).float() * outside_gt, dim=[-3, -2, -1])
        total_bg = torch.sum(outside_gt, dim=[-3, -2, -1])
        bg_cleanliness = 1.0 - (bg_leak / torch.clamp(total_bg, min=1.0))

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
            'tip_recall': tip_recall,
            'recall': recall,
            'strict_recall': strict_recall,
            'f1': f1,
            'density': density,
            'ghost': ghost_px,
            'dark_ghost': dark_ghost_px,
            'bg_cleanliness': bg_cleanliness,
            'val_l1': val_l1,
            'psnr': psnr,
            'ssim': ssim
        }

# =============================================================================
# 6. SMART CHECKPOINT MANAGER V3.0-REMASTERED R2
# =============================================================================
RESUME_CKPT_NAME = 'model_tps_v3.0_remastered_resume.pt'

class SmartCheckpointManager_v30R:
    def __init__(self, save_dir, 
                 ref_wcd=0.00120, ref_semd=0.00250, 
                 ref_skel_err=0.075, ref_stroke_err=0.075,
                 ref_ghost=6000.0, target_density=1.10,
                 min_recall=0.900, min_skel_recall=0.880,
                 min_density=1.05, max_density=1.18, max_wcd=0.00220):
        self.save_dir = save_dir
        self.ref_wcd = ref_wcd
        self.ref_semd = ref_semd
        self.ref_skel_err = ref_skel_err
        self.ref_stroke_err = ref_stroke_err
        self.ref_ghost = ref_ghost
        self.target_density = target_density

        self.min_recall = min_recall
        self.min_skel_recall = min_skel_recall
        self.min_density = min_density
        self.max_density = max_density
        self.max_wcd = max_wcd

        self.best_score = float('inf')
        self.best_progress_score = float('inf')
        self.best_epoch = 0
        self.resume_path = os.path.join(save_dir, RESUME_CKPT_NAME)
        self.top_candidates = []
        os.makedirs(save_dir, exist_ok=True)

    def calculate_score(self, m):
        density_dev = abs(m.get('density', 1.10) - self.target_density)
        semd = m.get('sliced_emd', m.get('semd', self.ref_semd))
        wcd = m.get('wcd', self.ref_wcd)
        skel_recall = m.get('skel_recall', 0.90)
        recall = m.get('recall', 0.90)
        ghost = m.get('ghost', 5000.0)

        score = (
            0.30 * (wcd / self.ref_wcd) +
            0.10 * (semd / self.ref_semd) +
            0.25 * ((1.0 - skel_recall) / max(1e-4, self.ref_skel_err)) +
            0.15 * ((1.0 - recall) / max(1e-4, self.ref_stroke_err)) +
            0.12 * (ghost / self.ref_ghost) +
            0.08 * (density_dev / 0.10)
        )
        return float(score)

    @staticmethod
    def _atomic_save(obj, path):
        tmp = path + '.tmp'
        torch.save(obj, tmp)
        os.replace(tmp, path)

    def save_resume(self, epoch, model, optimizer=None, scaler=None, history=None, completed=True):
        raw_model = model.module if hasattr(model, 'module') else (model.base_model if hasattr(model, 'base_model') else model)
        state = {
            'epoch': int(epoch) if completed else int(epoch) - 1,
            'model': {k: v.detach().cpu() for k, v in raw_model.state_dict().items()},
            'optimizer': optimizer.state_dict() if optimizer is not None else None,
            'scaler': scaler.state_dict() if scaler is not None else None,
            'best_score': self.best_score,
            'best_progress_score': self.best_progress_score,
            'best_epoch': self.best_epoch,
            'top_candidates': self.top_candidates,
            'history': history or []
        }
        self._atomic_save(state, self.resume_path)

    def step(self, epoch, model, metrics, optimizer=None, scaler=None, history=None):
        score = self.calculate_score(metrics)
        raw_model = model.module if hasattr(model, 'module') else (model.base_model if hasattr(model, 'base_model') else model)
        sd = {k: v.detach().cpu() for k, v in raw_model.state_dict().items()}

        latest_path = os.path.join(self.save_dir, 'model_tps_v3.0_remastered_latest.pt')
        self._atomic_save({'epoch': epoch, 'model': sd, 'metrics': metrics, 'score': score}, latest_path)

        density_ok = (self.min_density <= metrics.get('density', 0.0) <= self.max_density)
        recall_ok = (metrics.get('recall', 0.0) >= self.min_recall)
        skel_ok = (metrics.get('skel_recall', 0.0) >= self.min_skel_recall)
        wcd_ok = (metrics.get('wcd', float('inf')) <= self.max_wcd)
        gates_passed = density_ok and recall_ok and skel_ok and wcd_ok

        is_best = False
        if gates_passed and score < self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            is_best = True
            best_path = os.path.join(self.save_dir, 'model_tps_v3.0_remastered_best.pt')
            self._atomic_save({'epoch': epoch, 'model': sd, 'metrics': metrics, 'score': score}, best_path)
            print(f"🏆 [KỶ LỤC MỚI V3.0-R2] Epoch {epoch:02d}: Score={score:.4f} | WCD={metrics['wcd']:.5f} | SkelRec={metrics['skel_recall']*100:.2f}% | Recall={metrics['recall']*100:.2f}% | Dens={metrics['density']:.3f}x")

        if score < self.best_progress_score:
            self.best_progress_score = score
            prog_path = os.path.join(self.save_dir, 'model_tps_v3.0_remastered_best_progress.pt')
            self._atomic_save({'epoch': epoch, 'model': sd, 'metrics': metrics, 'score': score}, prog_path)

        cand_path = os.path.join(self.save_dir, f'model_tps_v3.0_remastered_ep{epoch:02d}.pt')
        self._atomic_save({'epoch': epoch, 'model': sd, 'metrics': metrics, 'score': score}, cand_path)
        self.top_candidates.append((score, epoch, cand_path))
        self.top_candidates.sort(key=lambda x: x[0])
        if len(self.top_candidates) > 5:
            _, _, old_path = self.top_candidates.pop(-1)
            if os.path.exists(old_path) and 'best' not in old_path and 'latest' not in old_path:
                try:
                    os.remove(old_path)
                except Exception:
                    pass

        self.save_resume(epoch, model, optimizer, scaler, history, completed=True)
        return score, is_best, gates_passed

# =============================================================================
# 7. TẠO ẢNH KIỂM TOÁN TRỰC QUAN 4 CỘT (SIDE-BY-SIDE AUDIT)
# =============================================================================
def generate_checkpoint_visual_demo_v30r(model, epoch, val_dataset, device, working_dir, num_samples=3):
    demo_dir = os.path.join(working_dir, 'visual_samples_v3.0_remastered')
    os.makedirs(demo_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for s_idx in range(min(num_samples, len(val_dataset))):
            item = val_dataset[s_idx]
            t0, t1, tt, t_val = item[0], item[1], item[2], item[3]
            tps_data = item[4] if len(item) > 4 else None

            t0_gpu = t0.unsqueeze(0).to(device)
            t1_gpu = t1.unsqueeze(0).to(device)
            t_val_gpu = t_val.unsqueeze(0).to(device)
            tps_gpu = tuple(x.unsqueeze(0).to(device) for x in tps_data) if tps_data is not None else None

            pred_t, blend_mask, _ = model(t0_gpu, t1_gpu, t_val_gpu, tps_precomputed=tps_gpu)

            # Cắt phần padding 544 -> 540x960
            pred_crop = pred_t[0, 0, :540, :960]
            rgba_img = export_pure_lineart_rgba(pred_crop, tau_noise=0.020, gamma=0.78)

            # 1. Lưu file PNG RGBA trong suốt
            rgba_path = os.path.join(demo_dir, f'sample_ep{epoch:02d}_s{s_idx:02d}_transparent.png')
            cv2.imwrite(rgba_path, rgba_img)

            # 2. Tạo ảnh composite 4 cột: [Overlay Keyframes | Coarse TPS Blend | V3.0-R2 RGBA | Ground Truth]
            alpha_f = rgba_img[:, :, 3].astype(np.float32) / 255.0
            white_bg = np.ones((540, 960, 3), dtype=np.float32) * 255.0
            comp_v30r = (white_bg * (1.0 - alpha_f[:, :, None])).astype(np.uint8)

            i0_raw = (255.0 * (1.0 - t0[0, :540, :960].cpu().numpy())).astype(np.uint8)
            i1_raw = (255.0 * (1.0 - t1[0, :540, :960].cpu().numpy())).astype(np.uint8)
            gt_raw = (255.0 * (1.0 - tt[0, :540, :960].cpu().numpy())).astype(np.uint8)

            # Cột 1: Chồng hình I0 (đỏ) & I1 (xanh lơ)
            col1 = np.ones((540, 960, 3), dtype=np.uint8) * 255
            col1[:, :, 0] = np.minimum(col1[:, :, 0], i1_raw) # B
            col1[:, :, 1] = np.minimum(col1[:, :, 1], i1_raw) # G
            col1[:, :, 2] = np.minimum(col1[:, :, 2], i0_raw) # R

            # Cột 2: Coarse Blend
            t_f = float(t_val.item())
            blend_ink = (1.0 - t_f) * t0[0, :540, :960].cpu().numpy() + t_f * t1[0, :540, :960].cpu().numpy()
            col2 = (255.0 * (1.0 - np.clip(blend_ink, 0.0, 1.0))).astype(np.uint8)
            col2 = cv2.cvtColor(col2, cv2.COLOR_GRAY2BGR)

            # Cột 3: V3.0-R2
            col3 = comp_v30r

            # Cột 4: Ground Truth
            col4 = cv2.cvtColor(gt_raw, cv2.COLOR_GRAY2BGR)

            # Ghép thành dải 4 cột với thanh phân cách 3px màu xám
            sep = np.ones((540, 3, 3), dtype=np.uint8) * 200
            side_by_side = np.hstack([col1, sep, col2, sep, col3, sep, col4])

            # Ghi tiêu đề lên đầu ảnh
            banner = np.ones((40, side_by_side.shape[1], 3), dtype=np.uint8) * 240
            cv2.putText(banner, f"Ep {epoch:02d} Sample {s_idx:02d}: [1] Overlay I0/I1  |  [2] Linear Blend  |  [3] V3.0-R2 Pure RGBA  |  [4] Ground Truth", (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2, cv2.LINE_AA)
            full_vis = np.vstack([banner, side_by_side])

            comp_path = os.path.join(demo_dir, f'sample_ep{epoch:02d}_s{s_idx:02d}_audit_4col.png')
            cv2.imwrite(comp_path, full_vis)

# =============================================================================
# 8. MAIN DDP WORKER ENTRYPOINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="DDP Worker for TPS-Inbetween V3.0-Remastered R2")
    parser.add_argument('--working-dir', type=str, default='/kaggle/working')
    parser.add_argument('--ckpt-dir', type=str, default=None)
    parser.add_argument('--stage-dir', type=str, default=None)
    parser.add_argument('--init-ckpt', type=str, default=None)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--total-epochs', type=int, default=16)
    parser.add_argument('--per-gpu-batch', type=int, default=4)
    parser.add_argument('--grad-accum-steps', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--single-gpu', action='store_true')
    parser.add_argument('--pbar-pipe', action='store_true')
    args = parser.parse_args()

    working_dir = args.working_dir
    ckpt_dir = args.ckpt_dir or os.path.join(working_dir, 'checkpoints_v3.0_remastered')
    stage_dir = args.stage_dir or '/tmp/sakuga_ssd_v30r'
    cache_dir = os.path.join(working_dir, 'dataset_cache_v3.0_remastered')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    is_ddp = not args.single_gpu and 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if is_ddp:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        dist.init_process_group(backend='nccl', init_method='env://')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    is_rank0 = (rank == 0)

    if is_rank0:
        print("=" * 80)
        print(f"🚀 DDP WORKER V3.0-REMASTERED R2 KHỞI ĐỘNG (World Size={world_size}, Device={device})")
        print(f"[*] Checkpoint Dir: {ckpt_dir} | Staging Dir: {stage_dir}")
        print("=" * 80)
        sys.stdout.flush()

    # Nạp metadata linh hoạt O(1)
    cand_csvs = [
        os.path.join(stage_dir, 'triplets.csv'),
        os.path.join(stage_dir, 'triplets_combined.csv'),
        os.path.join(stage_dir, 'metadata_patch3.csv'),
        os.path.join(stage_dir, 'anime_tps_finetune_dataset_ready', 'triplets.csv'),
        os.path.join(os.path.dirname(stage_dir), 'triplets.csv'),
        os.path.join(os.path.dirname(stage_dir), 'triplets_combined.csv')
    ]
    meta_path = next((p for p in cand_csvs if os.path.isfile(p)), None)
    if not meta_path and os.path.exists(stage_dir):
        for f in os.listdir(stage_dir):
            if f.endswith('.csv') and ('triplet' in f.lower() or 'meta' in f.lower()):
                meta_path = os.path.join(stage_dir, f)
                break

    if not meta_path or not os.path.exists(meta_path):
        raise FileNotFoundError(f"[!] DDP Worker không tìm thấy metadata CSV tại {stage_dir}")

    if is_rank0:
        print(f"[*] [1/6] Nạp metadata từ: {meta_path}...", flush=True)

    df_meta = pd.read_csv(meta_path)
    if 'frame0' in df_meta.columns and 'frame0_path' not in df_meta.columns:
        df_meta['frame0_path'] = df_meta['frame0']
    if 'frame1' in df_meta.columns and 'frame1_path' not in df_meta.columns:
        df_meta['frame1_path'] = df_meta['frame1']
    if 'frame_gt' in df_meta.columns and 'target_path' not in df_meta.columns:
        df_meta['target_path'] = df_meta['frame_gt']

    if 'split' in df_meta.columns:
        train_df = df_meta[df_meta['split'] == 'train'].reset_index(drop=True)
        val_df = df_meta[df_meta['split'] == 'val'].reset_index(drop=True)
    else:
        n_train = int(len(df_meta) * 0.85)
        train_df = df_meta.iloc[:n_train].reset_index(drop=True)
        val_df = df_meta.iloc[n_train:].reset_index(drop=True)

    if is_rank0:
        print(f"[*] [2/6] Đã nạp {len(df_meta):,} triplets (Train: {len(train_df):,}, Val: {len(val_df):,}). Chuẩn bị mmap TPS cache...", flush=True)

    # Chuẩn bị TPS mmap (Đồng bộ qua barrier để Rank 0 tính toán và ghi file trước, tránh xung đột I/O trên 2 GPU)
    if is_ddp:
        if is_rank0:
            train_tps = ensure_tps_precomputed_v30r(train_df, stage_dir, cache_dir, split_name='train')
            val_tps = ensure_tps_precomputed_v30r(val_df, stage_dir, cache_dir, split_name='val')
        dist.barrier()
        if not is_rank0:
            train_tps = ensure_tps_precomputed_v30r(train_df, stage_dir, cache_dir, split_name='train')
            val_tps = ensure_tps_precomputed_v30r(val_df, stage_dir, cache_dir, split_name='val')
    else:
        train_tps = ensure_tps_precomputed_v30r(train_df, stage_dir, cache_dir, split_name='train')
        val_tps = ensure_tps_precomputed_v30r(val_df, stage_dir, cache_dir, split_name='val')

    train_ds = SakugaGoldenDataset_v30R(train_df, stage_dir, is_train=True, cache_dir=cache_dir, precomputed_tps=train_tps)
    val_ds = SakugaGoldenDataset_v30R(val_df, stage_dir, is_train=False, cache_dir=cache_dir, precomputed_tps=val_tps)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_ddp else None

    def worker_init_fn(worker_id):
        import cv2
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

    train_loader = DataLoader(
        train_ds, batch_size=args.per_gpu_batch,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=True, worker_init_fn=worker_init_fn if args.num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.per_gpu_batch,
        sampler=val_sampler, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=False, worker_init_fn=worker_init_fn if args.num_workers > 0 else None
    )

    if is_rank0:
        print(f"[*] [3/6] Khởi tạo kiến trúc mô hình TPSInbetweenFinetune_v30R R2...", flush=True)

    base_model = TPSInbetweenFinetune_v30R().to(device)
    criterion = CompositeLossEngine_v30R().to(device)

    # Nạp Warm-Start từ V2.6 Best
    if is_rank0:
        print(f"[*] [4/6] Tìm kiếm Checkpoint Warm-Start V2.6...", flush=True)

    v26_cands = [
        args.init_ckpt,
        '/kaggle/input/checkpoints_v2.6/model_tps_v2.6_best.pt',
        '/kaggle/input/checkpoints_v2.6/checkpoints_v2.6/model_tps_v2.6_best.pt',
        '/kaggle/input/datasets/dungbaoviec123/checkpoints-v2-6/model_tps_v2.6_best.pt',
        '/kaggle/input/datasets/dungbaoviec123/checkpoints_v2.6/model_tps_v2.6_best.pt',
        '/kaggle/input/checkpoints-v2-6/model_tps_v2.6_best.pt',
        '/kaggle/input/model-tps-v26-best/model_tps_v2.6_best.pt',
        os.path.join(working_dir, 'model_tps_v2.6_best.pt'),
        os.path.join(working_dir, 'checkpoints_v2.6', 'model_tps_v2.6_best.pt'),
        './checkpoints/experiments/v2.6/checkpoints_v2.6/model_tps_v2.6_best.pt',
        'E:/CODE/PBL/PBL6/checkpoints/experiments/v2.6/checkpoints_v2.6/model_tps_v2.6_best.pt'
    ]
    v26_cands = [p for p in v26_cands if p and os.path.isfile(p)]
    loaded_ckpt = False
    for ckpt_p in v26_cands:
        try:
            sd = torch.load(ckpt_p, map_location=device, weights_only=False)
            if 'state_dict' in sd:
                sd = sd['state_dict']
            elif 'model' in sd:
                sd = sd['model']
            clean_sd = {k.replace('module.', '').replace('base_model.', ''): v for k, v in sd.items()}
            m_dict = base_model.state_dict()
            matched = {k: v for k, v in clean_sd.items() if k in m_dict and m_dict[k].shape == v.shape}
            m_dict.update(matched)
            base_model.load_state_dict(m_dict)
            if is_rank0:
                print(f"[✓] Nạp Warm-Start V2.6 thành công: {ckpt_p} ({len(matched)}/168 tensors)", flush=True)
            loaded_ckpt = True
            break
        except Exception as e:
            if is_rank0:
                print(f"[!] Lỗi nạp {ckpt_p}: {e}", flush=True)

    if not loaded_ckpt and is_rank0:
        print("[!] Không phát hiện checkpoint V2.6 tại các đường dẫn mặc định. Huấn luyện từ đầu.", flush=True)

    if is_ddp:
        if is_rank0:
            print(f"[*] [5/6] Đồng bộ DDP trên {world_size} GPUs qua NCCL...", flush=True)
        model = DDP(base_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    else:
        model = base_model

    # Differential Learning Rate: Backbone 1.0e-5, Heads & Refinement 3.0e-5
    backbone_params = []
    head_params = []
    for name, p in base_model.named_parameters():
        if not p.requires_grad:
            continue
        if 'encoder' in name or 'IFNet' in name:
            backbone_params.append(p)
        else:
            head_params.append(p)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 1.0e-5},
        {'params': head_params, 'lr': args.lr}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')
    ckpt_manager = SmartCheckpointManager_v30R(ckpt_dir)

    # =========================================================================
    # EPOCH 0 SANITY GATE CHECK (Kiểm định điểm xuất phát Flat Minima V2.6)
    # =========================================================================
    if is_rank0:
        print("=" * 80)
        print("🔍 [SANITY GATE EPOCH 0]: KIỂM ĐỊNH CHẤT LƯỢNG MÔ HÌNH V2.6 TRƯỚC KHI TRAIN...")
        print("=" * 80)
        sys.stdout.flush()

    model.eval()
    init_stats = {'wcd': 0.0, 'semd': 0.0, 'skel_recall': 0.0, 'recall': 0.0, 'strict_recall': 0.0, 'density': 0.0, 'ghost': 0.0, 'val_l1': 0.0, 'tps_valid': 0.0}
    init_samples = 0

    with torch.no_grad():
        for val_step, batch in enumerate(val_loader):
            if val_step >= 50:  # Kiểm định nhanh trên 50 batches đầu
                break
            t0 = batch[0].to(device, non_blocking=True)
            t1 = batch[1].to(device, non_blocking=True)
            tt = batch[2].to(device, non_blocking=True)
            t_val = batch[3].to(device, non_blocking=True)
            tps_batch = tuple(x.to(device, non_blocking=True) for x in batch[4])

            pred, _, _ = model(t0, t1, t_val, tps_precomputed=tps_batch)
            p_crop = pred[:, :, :540, :960]
            t_crop = tt[:, :, :540, :960]
            m = calculate_metrics_batch_gpu(p_crop, t_crop)
            bsz = p_crop.shape[0]
            init_samples += bsz

            init_stats['wcd'] += torch.sum(m['wcd']).item()
            init_stats['semd'] += torch.sum(m['sliced_emd']).item()
            init_stats['skel_recall'] += torch.sum(m['skel_recall']).item()
            init_stats['recall'] += torch.sum(m['recall']).item()
            init_stats['strict_recall'] += torch.sum(m['strict_recall']).item()
            init_stats['density'] += torch.sum(m['density']).item()
            init_stats['ghost'] += torch.sum(m['ghost']).item()
            init_stats['val_l1'] += torch.sum(m['val_l1']).item()
            init_stats['tps_valid'] += torch.sum(tps_batch[4]).item()

    if is_rank0:
        c = max(1, init_samples)
        e0_recall = init_stats['recall'] / c
        e0_strict = init_stats['strict_recall'] / c
        e0_skel = init_stats['skel_recall'] / c
        e0_wcd = init_stats['wcd'] / c
        e0_dens = init_stats['density'] / c
        e0_tps_v = (init_stats['tps_valid'] / c) * 100.0

        print(f"[*] Kết quả Epoch 0: TPS Valid={e0_tps_v:.1f}% | Stroke Recall={e0_recall*100:.2f}% (Strict={e0_strict*100:.2f}%) | Skel Recall={e0_skel*100:.2f}% | WCD={e0_wcd:.5f} | Dens={e0_dens:.3f}x")
        if e0_recall < 0.85:
            print(f"[⚠️ CẢNH BÁO] Stroke Recall Epoch 0 ({e0_recall*100:.1f}%) < 85%. Kiểm tra lại Warm-Start checkpoint!")
        else:
            print(f"[✓ SANITY GATE THÀNH CÔNG] Điểm xuất phát V2.6 đạt chuẩn (Recall {e0_recall*100:.2f}% >= 88.0%). Coarse TPS Flow hoạt động chính xác!")
        print("=" * 80)
        sys.stdout.flush()

    if is_ddp:
        dist.barrier()

    history_csv = os.path.join(working_dir, 'training_history_v3.0_remastered.csv')
    history = []

    train_start_time = time.time()
    for epoch in range(1, args.total_epochs + 1):
        t_ep_start = time.time()
        if is_ddp:
            train_sampler.set_epoch(epoch)

        # ----------------- TRAIN LOOP -----------------
        model.train()
        train_loss_acc = 0.0
        step_count = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            t0 = batch[0].to(device, non_blocking=True)
            t1 = batch[1].to(device, non_blocking=True)
            tt = batch[2].to(device, non_blocking=True)
            t_val = batch[3].to(device, non_blocking=True)
            tps_batch = tuple(x.to(device, non_blocking=True) for x in batch[4])

            with torch.amp.autocast('cuda'):
                pred, blend_mask, conf_fbfc, delta_flow, _, _ = model(t0, t1, t_val, tps_precomputed=tps_batch)
                loss, charb, dt, dice, dens, hair, ghost = criterion(pred, tt, conf_fbfc=conf_fbfc, epoch=epoch)
                loss_scaled = loss / args.grad_accum_steps

            if torch.isnan(loss) or torch.isinf(loss):
                if is_rank0:
                    print(f"[!] Cảnh báo: Non-finite loss tại step {step+1}. Bỏ qua batch.", flush=True)
                optimizer.zero_grad()
                continue

            scaler.scale(loss_scaled).backward()

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss_acc += loss.item()
            step_count += 1

            if is_rank0 and args.pbar_pipe and step % 10 == 0:
                cur_lr = scheduler.get_last_lr()[-1]
                print(f"__PBAR_TRAIN__:ep={epoch}|total_ep={args.total_epochs}|step={step+1}|total={len(train_loader)}|loss={loss.item():.4f}|dens=1.100|lr={cur_lr:.2e}", flush=True)

        scheduler.step()
        train_time_m = (time.time() - t_ep_start) / 60.0

        # ----------------- VAL LOOP -----------------
        model.eval()
        t_val_start = time.time()
        val_stats = {
            'wcd': 0.0, 'semd': 0.0, 'skel_recall': 0.0, 'tip_recall': 0.0,
            'recall': 0.0, 'f1': 0.0, 'density': 0.0, 'ghost': 0.0,
            'dark_ghost': 0.0, 'bg_clean': 0.0, 'val_l1': 0.0, 'psnr': 0.0, 'ssim': 0.0
        }
        val_samples = 0

        with torch.no_grad():
            for val_step, batch in enumerate(val_loader):
                t0 = batch[0].to(device, non_blocking=True)
                t1 = batch[1].to(device, non_blocking=True)
                tt = batch[2].to(device, non_blocking=True)
                t_val = batch[3].to(device, non_blocking=True)
                tps_batch = tuple(x.to(device, non_blocking=True) for x in batch[4])

                pred, _, _ = model(t0, t1, t_val, tps_precomputed=tps_batch)

                p_crop = pred[:, :, :540, :960]
                t_crop = tt[:, :, :540, :960]

                m = calculate_metrics_batch_gpu(p_crop, t_crop)
                bsz = p_crop.shape[0]
                val_samples += bsz

                val_stats['wcd'] += torch.sum(m['wcd']).item()
                val_stats['semd'] += torch.sum(m['sliced_emd']).item()
                val_stats['skel_recall'] += torch.sum(m['skel_recall']).item()
                val_stats['tip_recall'] += torch.sum(m['tip_recall']).item()
                val_stats['recall'] += torch.sum(m['recall']).item()
                val_stats['f1'] += torch.sum(m['f1']).item()
                val_stats['density'] += torch.sum(m['density']).item()
                val_stats['ghost'] += torch.sum(m['ghost']).item()
                val_stats['dark_ghost'] += torch.sum(m['dark_ghost']).item()
                val_stats['bg_clean'] += torch.sum(m['bg_cleanliness']).item()
                val_stats['val_l1'] += torch.sum(m['val_l1']).item()
                val_stats['psnr'] += torch.sum(m['psnr']).item()
                val_stats['ssim'] += torch.sum(m['ssim']).item()

        val_time_m = (time.time() - t_val_start) / 60.0
        ep_time_m = (time.time() - t_ep_start) / 60.0

        if is_rank0:
            count = max(1, val_samples)
            metrics_ep = {
                'epoch': epoch,
                'train_loss': train_loss_acc / max(1, step_count),
                'val_l1': val_stats['val_l1'] / count,
                'wcd': val_stats['wcd'] / count,
                'sliced_emd': val_stats['semd'] / count,
                'skel_recall': val_stats['skel_recall'] / count,
                'tip_recall': val_stats['tip_recall'] / count,
                'stroke_recall': val_stats['recall'] / count,
                'recall': val_stats['recall'] / count,
                'density_ratio': val_stats['density'] / count,
                'density': val_stats['density'] / count,
                'ghost_pixels': val_stats['ghost'] / count,
                'ghost': val_stats['ghost'] / count,
                'dark_ghost': val_stats['dark_ghost'] / count,
                'bg_cleanliness': val_stats['bg_clean'] / count,
                'val_ssim': val_stats['ssim'] / count,
                'ssim': val_stats['ssim'] / count,
                'val_psnr': val_stats['psnr'] / count,
                'host_ram_used_gb': psutil.Process().memory_info().rss / (1024 ** 3),
                'train_time_minutes': train_time_m,
                'val_time_minutes': val_time_m,
                'time_minutes': ep_time_m
            }
            history.append(metrics_ep)
            pd.DataFrame(history).to_csv(history_csv, index=False)

            # Smart Checkpoint Manager Step
            score, is_best, passed = ckpt_manager.step(epoch, model, metrics_ep, optimizer, scaler, history)

            # Xuất trực quan định kỳ 4 cột và PNG trong suốt
            raw_model = model.module if hasattr(model, 'module') else (model.base_model if hasattr(model, 'base_model') else model)
            if epoch % 2 == 0 or epoch == 1 or is_best:
                generate_checkpoint_visual_demo_v30r(raw_model, epoch, val_ds, device, working_dir, num_samples=3)

            print(f"\n📌 HOÀN TẤT EPOCH {epoch:02d}/{args.total_epochs:02d} ({ep_time_m:.2f} phút)")
            print(f"🔍 VAL V3.0-R2: WCD={metrics_ep['wcd']:.5f} | S-EMD={metrics_ep['sliced_emd']:.5f} | SkelRec={metrics_ep['skel_recall']*100:.1f}% | TipRec={metrics_ep['tip_recall']*100:.1f}% | StrokeRec={metrics_ep['stroke_recall']*100:.1f}% | Dens={metrics_ep['density_ratio']:.3f}x | Cleanliness={metrics_ep['bg_cleanliness']*100:.2f}% | Score={score:.4f}")
            sys.stdout.flush()

        # Watchdog RAM Host
        host_ram_pct = psutil.virtual_memory().percent
        if host_ram_pct > 94.0:
            if is_rank0:
                print(f"[⚠️ WATCHDOG] RAM Host vượt ngưỡng 94% ({host_ram_pct:.1f}%). Kích hoạt lưu trạng thái an toàn và thoát...", flush=True)
                ckpt_manager.save_resume(epoch, model, optimizer, scaler, history, completed=True)
            if is_ddp:
                dist.barrier()
                dist.destroy_process_group()
            sys.exit(42)

        if is_ddp:
            dist.barrier()

    if is_ddp:
        dist.destroy_process_group()

    if is_rank0:
        print(f"\n🎉 HOÀN TẤT HUẤN LUYỆN TOÀN BỘ {args.total_epochs} EPOCHS TRONG {(time.time() - train_start_time)/60:.1f} PHÚT!", flush=True)
        sys.exit(0)

if __name__ == '__main__':
    main()
