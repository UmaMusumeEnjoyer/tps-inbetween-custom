#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_benchmark_suite.py
--------------------------------------------------------------------------------
BỘ CÔNG CỤ BENCHMARK ĐỐI CHỨNG KHÁCH QUAN KÉP (ZERO-BIAS DUAL-BENCHMARK SUITE)
Kiểm chứng đóng góp thực tế của Dataset Custom đối với TPS-Inbetween (M1 gốc vs M3 finetune)
với CAM KẾT 100% ZERO ARCHITECTURAL MODIFICATION.

Hai bài kiểm định chuẩn hóa:
  [Benchmark #1] Tập Chuỗi Độc Lập `sakuga_golden_benchmark_sequences_v2`
                 (198 sequences | 149 scenes không trùng lặp | 412 frames N=3,5,7)
    - So sánh: M0 (Coarse TPS Warp) | M1 (Original model_latest.pt) | M3_best | M3_last
    - Dual-Regime Metrics:
        + Regime A (Raw Continuous): PSNR, SSIM, IE-MAE, PWCR, RGD, Ghost Pixels
        + Regime B (Thresholded tau=0.15 & DoG): WCD, WCD_DoG, S-EMD,
          Skeleton Recall, Hair-Tip Recall, Stroke Recall/Precision/F1, Density Ratio
    - Phân tầng theo N={3, 5, 7} & Biên độ chuyển động (mean_disp <6, 6..14, >=14 px)
    - Thống kê chống thiên vị: Head-to-Head Win Rate (%) & Wilcoxon Signed-Rank Test
      gom nhóm theo 149 scenes độc lập (tránh pseudoreplication).
    - Tự động xuất Top-3 Mẫu Cải Thiện Mạnh Nhất & Top-3 Failure Cases (5 cột HUD).

  [Benchmark #2] Tập Thực Tế Liên Tục `data/lineart_experiments/test`
                 (24 frames 540p | 7 Keyframes -> 17 Inbetween frames | delta_f = 3,4,5)
    - Dùng đúng điểm neo GlueStick chính thống (hoặc precomputed cache .npz) + pad 540->544.
    - Hỗ trợ đối chiếu SIFT Reference tương thích bảng lịch sử evolution_timeline_master.csv.
    - Xuất bảng CSV từng frame, từng chặng delta_f, ảnh Side-by-Side 4 cột,
      Error Heatmap và Video đối chiếu MP4/GIF 8 FPS.
--------------------------------------------------------------------------------
"""

import os
import sys
import time
import json
import glob
import math
import types
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

try:
    import cv2
    cv2.setNumThreads(0)
except ImportError:
    cv2 = None

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

try:
    import imageio
except ImportError:
    imageio = None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_repo_root(start_dir):
    cur = start_dir
    for _ in range(7):
        if os.path.isdir(os.path.join(cur, "checkpoints")) and os.path.isdir(os.path.join(cur, "models")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.abspath(os.path.join(start_dir, "..", "..", "..", ".."))


REPO_ROOT = _find_repo_root(SCRIPT_DIR)
DEFAULT_ORIG_CKPT = os.path.join(REPO_ROOT, "checkpoints", "pretrained_baselines", "tps-original", "model_latest.pt")
DEFAULT_LINEART_TEST = os.path.join(REPO_ROOT, "data", "lineart_experiments", "test")
DEFAULT_GLUESTICK_CACHE = os.path.join(REPO_ROOT, "kaggle_seq_p1_v32_ghost_temp_scale", "lineart_test_gluestick_matches.npz")

# Tái sử dụng trực tiếp toán tử từ ddp_worker nếu nằm trong cùng repo để tránh trùng lặp,
# kèm fallback độc lập khi script chạy tách rời trên Kaggle/server khác.
_WORKER_DIR = os.path.join(REPO_ROOT, "kaggle_seq_p1_v32_ghost_temp_scale")
if os.path.isdir(_WORKER_DIR) and _WORKER_DIR not in sys.path:
    sys.path.insert(0, _WORKER_DIR)

try:
    from seq_p1_v32_ghost_temp_scale_worker import (
        TPSInbetweenOriginal,
        load_tps_weights,
        solve_tps_backward_pair,
        tps_flow_from_theta_batch,
        evaluate_metrics_dual_regime,
    )
    _IMPORTED_CORE = True
except ImportError:
    _IMPORTED_CORE = False


if not _IMPORTED_CORE:
    FIXED_TPS_ANCHORS = np.array([
        [0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
        [0.0, 0.5],             [1.0, 0.5],
        [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]
    ], dtype=np.float32)
    _GRID_CACHE, _PIXEL_GRID_CACHE, _DOG_KERNEL_CACHE = {}, {}, {}

    def warp_normalized_flow(img, flow):
        _, _, h, w = img.shape
        key = (h, w, flow.device)
        base_grid = _GRID_CACHE.get(key)
        if base_grid is None:
            gy, gx = torch.meshgrid(
                torch.linspace(0.0, 1.0, h, device=flow.device),
                torch.linspace(0.0, 1.0, w, device=flow.device), indexing='ij'
            )
            base_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).contiguous()
            _GRID_CACHE[key] = base_grid
        grid = (base_grid + flow.float().permute(0, 2, 3, 1)) * 2.0 - 1.0
        return F.grid_sample(img.float().contiguous(), grid, mode='bilinear', padding_mode='border', align_corners=True).to(img.dtype)

    def warp_pixel_flow(img, flow):
        _, _, h, w = img.shape
        key = (h, w, flow.device)
        base_grid = _PIXEL_GRID_CACHE.get(key)
        if base_grid is None:
            gy, gx = torch.meshgrid(
                torch.linspace(-1.0, 1.0, h, device=flow.device),
                torch.linspace(-1.0, 1.0, w, device=flow.device), indexing='ij'
            )
            base_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).contiguous()
            _PIXEL_GRID_CACHE[key] = base_grid
        flow_f = flow.float()
        fn = torch.cat([flow_f[:, 0:1] / max(1.0, (w - 1.0) / 2.0), flow_f[:, 1:2] / max(1.0, (h - 1.0) / 2.0)], dim=1)
        grid = (base_grid + fn.permute(0, 2, 3, 1)).contiguous()
        return F.grid_sample(img.float().contiguous(), grid, mode='bilinear', padding_mode='border', align_corners=True).to(img.dtype)

    def solve_tps_backward_pair(kps0, kps1, max_pts=128, lambd=0.01):
        def _solve_single(c_src, c_dst):
            if c_src is None or c_dst is None or len(c_src) < 4:
                return np.zeros((max_pts, 2), np.float32), np.zeros((max_pts + 3, 2), np.float32), 0.0
            n_keep = max_pts - 8
            if len(c_src) > n_keep:
                idx = np.linspace(0, len(c_src) - 1, n_keep, dtype=int)
                cs, cd = np.asarray(c_src[idx], np.float64), np.asarray(c_dst[idx], np.float64)
            else:
                cs, cd = np.asarray(c_src, np.float64), np.asarray(c_dst, np.float64)
            anc = FIXED_TPS_ANCHORS.astype(np.float64)
            cs_all, cd_all = np.vstack([cs, anc]), np.vstack([cd, anc])
            m = len(cs_all)
            r = np.linalg.norm(cs_all[:, None, :] - cs_all[None, :, :], axis=-1)
            K = (r ** 2) * np.log(np.maximum(r, 1e-6))
            np.fill_diagonal(K, float(lambd))
            P = np.hstack([np.ones((m, 1), np.float64), cs_all])
            A = np.zeros((m + 3, m + 3), np.float64)
            A[:m, :m], A[:m, m:], A[m:, :m] = K, P, P.T
            Y = np.zeros((m + 3, 2), np.float64)
            Y[:m, :] = cd_all
            try:
                th = np.linalg.solve(A, Y)
            except np.linalg.LinAlgError:
                th = np.linalg.lstsq(A, Y, rcond=1e-4)[0]
            valid = 0.0 if (np.isnan(th).any() or np.isinf(th).any() or np.max(np.abs(th)) > 50.0) else 1.0
            cp, tp = np.zeros((max_pts, 2), np.float32), np.zeros((max_pts + 3, 2), np.float32)
            if valid > 0.5:
                cp[:m], tp[:m], tp[max_pts:] = cs_all.astype(np.float32), th[:m].astype(np.float32), th[m:].astype(np.float32)
            return cp, tp, valid
        c0, th01, v01 = _solve_single(kps1, kps0)
        c1, th10, v10 = _solve_single(kps0, kps1)
        return c0, th01, c1, th10, (1.0 if (v01 > 0.5 and v10 > 0.5) else 0.0)

    @torch.no_grad()
    def tps_flow_from_theta_batch(theta_batch, c_src_batch, valid_mask, orig_hw, h_pad, w_pad, device, downscale_factor=2):
        th_f, cs_f, v_f = theta_batch.float(), c_src_batch.float(), valid_mask.float()
        b = th_f.shape[0]
        h_sub, w_sub = max(16, h_pad // max(1, downscale_factor)), max(16, w_pad // max(1, downscale_factor))
        gy, gx = torch.meshgrid(torch.linspace(0, 1, h_sub, device=device), torch.linspace(0, 1, w_sub, device=device), indexing='ij')
        pts_pad = torch.stack([gx, gy], dim=-1).view(1, -1, 2).expand(b, -1, -1)
        hw_f = orig_hw.to(device=device, dtype=torch.float32)
        sx = (float(max(1, w_pad - 1)) / torch.clamp(hw_f[:, 1] - 1.0, min=1.0)).view(b, 1, 1)
        sy = (float(max(1, h_pad - 1)) / torch.clamp(hw_f[:, 0] - 1.0, min=1.0)).view(b, 1, 1)
        scale_xy = torch.cat([sx, sy], dim=-1)
        pts_raw = pts_pad * scale_xy
        in_b = ((pts_raw[..., 0:1] <= 1.001) & (pts_raw[..., 1:2] <= 1.001)).float()
        pts_q = torch.clamp(pts_raw, 0.0, 1.0)
        chunks = []
        for i in range(0, pts_q.shape[1], 32768):
            q = pts_q[:, i:i + 32768]
            r = torch.clamp(torch.cdist(q, cs_f, p=2.0), min=1e-6)
            U = (r ** 2) * torch.log(r)
            basis = torch.cat([U, torch.ones((b, q.shape[1], 1), device=device), q], dim=-1)
            chunks.append(((torch.bmm(basis, th_f) - q) / scale_xy) * in_b[:, i:i + 32768])
        flow = torch.cat(chunks, dim=1).view(b, h_sub, w_sub, 2).permute(0, 3, 1, 2).contiguous()
        if h_sub != h_pad or w_sub != w_pad:
            flow = F.interpolate(flow, size=(h_pad, w_pad), mode='bilinear', align_corners=True)
        return flow * v_f.view(b, 1, 1, 1)

    def _conv(i, o, k=3, s=1, p=1):
        return nn.Sequential(nn.Conv2d(i, o, k, s, p, bias=True), nn.PReLU(o))

    def _deconv(i, o):
        return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1, bias=True), nn.PReLU(o))

    class _Conv2(nn.Module):
        def __init__(self, i, o, s=2):
            super().__init__()
            self.conv1, self.conv2 = _conv(i, o, 3, s, 1), _conv(o, o, 3, 1, 1)
        def forward(self, x):
            return self.conv2(self.conv1(x))

    class _Contextnet(nn.Module):
        def __init__(self, indim=1, c=16):
            super().__init__()
            self.conv1, self.conv2 = _Conv2(indim, c), _Conv2(c, 2 * c)
            self.conv3, self.conv4 = _Conv2(2 * c, 4 * c), _Conv2(4 * c, 8 * c)
        def forward(self, x, flow):
            feats = []
            for cv_blk in (self.conv1, self.conv2, self.conv3, self.conv4):
                x = cv_blk(x)
                flow = F.interpolate(flow, size=x.shape[2:], mode='bilinear', align_corners=False) * 0.5
                feats.append(warp_normalized_flow(x, flow))
            return feats

    class _IFBlock(nn.Module):
        def __init__(self, in_planes, c=64):
            super().__init__()
            self.conv0 = nn.Sequential(_conv(in_planes, c // 2, 3, 2, 1), _conv(c // 2, c, 3, 2, 1))
            self.convblock = nn.Sequential(*[_conv(c, c) for _ in range(8)])
            self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)
        def forward(self, x, flow, scale):
            oh, ow = x.shape[2], x.shape[3]
            if scale != 1:
                x = F.interpolate(x, scale_factor=1.0 / scale, mode='bilinear', align_corners=False)
            if flow is not None:
                flow = F.interpolate(flow, size=x.shape[2:], mode='bilinear', align_corners=False) * (1.0 / scale)
                x = torch.cat((x, flow), 1)
            x = self.conv0(x.contiguous())
            x = self.convblock(x) + x
            tmp = F.interpolate(self.lastconv(x.float().contiguous()), size=(oh, ow), mode='bilinear', align_corners=False)
            return tmp[:, :4] * scale * 2, tmp[:, 4:5]

    class _IFNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.block0, self.block1, self.block2 = _IFBlock(6, 240), _IFBlock(17, 150), _IFBlock(17, 90)
        def forward(self, i0, i1, scale=(4, 2, 1)):
            r0, r1 = i0.repeat(1, 3, 1, 1), i1.repeat(1, 3, 1, 1)
            w0, w1, flow, mask = i0, i1, None, None
            for i, blk in enumerate((self.block0, self.block1, self.block2)):
                if flow is not None:
                    fd, md = blk(torch.cat((r0, r1, w0.repeat(1, 3, 1, 1), w1.repeat(1, 3, 1, 1), mask), 1), flow, scale[i])
                    flow, mask = flow + fd, mask + md
                else:
                    flow, mask = blk(torch.cat((r0, r1), 1), None, scale[i])
                if i < 2:
                    w0, w1 = warp_pixel_flow(i0, flow[:, :2]), warp_pixel_flow(i1, flow[:, 2:4])
            return flow, torch.sigmoid(mask)

    class _OriginalUnet(nn.Module):
        def __init__(self, indim=9, outdim=1, c=16):
            super().__init__()
            self.down0, self.down1 = _Conv2(indim, 2 * c), _Conv2(4 * c, 4 * c)
            self.down2, self.down3 = _Conv2(8 * c, 8 * c), _Conv2(16 * c, 16 * c)
            self.up0, self.up1 = _deconv(32 * c, 8 * c), _deconv(16 * c, 4 * c)
            self.up2, self.up3 = _deconv(8 * c, 2 * c), _deconv(4 * c, c)
            self.conv = nn.Conv2d(c, outdim, 3, 1, 1)
        def forward(self, img0, img1, flow0, flow1, w0, w1, mask, c0, c1):
            s0 = self.down0(torch.cat((img0, img1, w0, w1, flow0, flow1, mask), 1))
            s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
            s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
            s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
            x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))
            x = self.up1(torch.cat((x, s2), 1))
            x = self.up2(torch.cat((x, s1), 1))
            x = self.up3(torch.cat((x, s0), 1))
            return torch.sigmoid(self.conv(x))

    class TPSInbetweenOriginal(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = _Contextnet(1, 16)
            self.IFNet = _IFNet()
            self.unet = _OriginalUnet(9, 1, 16)
        def forward(self, x0, x1, t_val, cf01, cf10):
            t = t_val
            cf0t = cf01 * (t ** 2) - (1.0 - t) * t * cf10
            cf1t = cf10 * ((1.0 - t) ** 2) - t * (1.0 - t) * cf01
            ci0, ci1 = warp_normalized_flow(x0, cf0t), warp_normalized_flow(x1, cf1t)
            coarse_w = torch.clamp((1.0 - t) * ci0 + t * ci1, 0.0, 1.0)
            f0, f1 = self.encoder(x0, cf0t), self.encoder(x1, cf1t)
            fbc, m = self.IFNet(ci0, ci1)
            fi0, fi1 = warp_pixel_flow(ci0, fbc[:, :2]), warp_pixel_flow(ci1, fbc[:, 2:4])
            res = self.unet(x0, x1, cf0t + fbc[:, :2], cf1t + fbc[:, 2:4], fi0, fi1, m, f0, f1)[:, 0:1] * 2.0 - 1.0
            It = torch.clamp((1.0 - t) * fi0 + t * fi1 + res, 0.0, 1.0)
            return It, coarse_w, fi0, fi1, fbc, m

    def load_tps_weights(model, ckpt_path, device='cpu', prefer_ema=False, is_rank0=True):
        raw = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(raw, dict):
            if prefer_ema and raw.get('ema_state') is not None:
                eo = raw['ema_state']
                sd = eo['shadow'] if isinstance(eo, dict) and 'shadow' in eo else eo
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
        msd = model.state_dict()
        matched = {k.replace('module.', '').replace('base_model.', ''): v for k, v in sd.items()
                   if k.replace('module.', '').replace('base_model.', '') in msd and msd[k.replace('module.', '').replace('base_model.', '')].shape == v.shape}
        missing, _ = model.load_state_dict(matched, strict=False)
        if missing:
            raise RuntimeError(f"Missing keys: {missing[:5]}")
        if is_rank0:
            print(f"[✓] Nạp {os.path.basename(ckpt_path)}: {len(matched)}/{len(msd)} active tensors!")
        return model

    @torch.no_grad()
    def evaluate_metrics_dual_regime(pred_ink, gt_ink):
        def _dil(x): return F.max_pool2d(x, (3, 3), (1, 1), (1, 1))
        def _ero(x): return torch.min(-F.max_pool2d(-x, (3, 1), (1, 1), (1, 0)), -F.max_pool2d(-x, (1, 3), (1, 1), (0, 1)))
        def _skel(img):
            im = torch.clamp(img, 0.0, 1.0)
            s = F.relu(im - _dil(_ero(im)))
            im2 = _ero(im)
            return torch.clamp(s + F.relu(im2 - _dil(_ero(im2))), 0.0, 1.0)
        def _dt(b_m, steps=10):
            c = b_m; acc = c
            for _ in range(steps):
                c = F.max_pool2d(c, 3, 1, 1); acc = acc + c
            return torch.clamp(1.0 - acc / float(steps + 1), 0.0, 1.0)
        def _blur(im, sig):
            r = max(1, int(round(3 * sig))); x = torch.arange(-r, r + 1, device=im.device, dtype=torch.float32)
            g = torch.exp(-(x ** 2) / (2 * sig * sig)); g = g / g.sum()
            k = torch.outer(g, g).view(1, 1, 2 * r + 1, 2 * r + 1).to(im.dtype)
            return F.conv2d(F.pad(im, (r, r, r, r), mode='replicate'), k)
        def _wcd(pb, gb):
            dg, dp = _dt(gb), _dt(pb)
            d1 = (pb * F.softplus(dg * 10)).sum((-2, -1)) / pb.sum((-2, -1)).clamp(min=1.0)
            d2 = (gb * F.softplus(dp * 10)).sum((-2, -1)) / gb.sum((-2, -1)).clamp(min=1.0)
            ap, ag = pb.sum((-2, -1)), gb.sum((-2, -1))
            return 0.5 * (1.0 + torch.minimum(ap, ag) / torch.maximum(ap, ag).clamp(min=1.0)) * (d1 + d2)
        p, g = pred_ink.float().clamp(0, 1), gt_ink.float().clamp(0, 1)
        mse = ((p - g) ** 2).mean((-3, -2, -1))
        psnr = 10.0 * torch.log10(1.0 / mse.clamp(min=1e-8))
        mp, mg = F.avg_pool2d(p, 11, 1, 5), F.avg_pool2d(g, 11, 1, 5)
        sp, sg = F.avg_pool2d(p * p, 11, 1, 5) - mp ** 2, F.avg_pool2d(g * g, 11, 1, 5) - mg ** 2
        spg = F.avg_pool2d(p * g, 11, 1, 5) - mp * mg
        ssim = (((2 * mp * mg + 1e-4) * (2 * spg + 9e-4)) / ((mp ** 2 + mg ** 2 + 1e-4) * (sp + sg + 9e-4))).mean((-3, -2, -1))
        l1 = (p - g).abs().mean((-3, -2, -1))
        p255, g255 = (1.0 - p) * 255.0, (1.0 - g) * 255.0
        pwcr = (p255 >= 250.0).float().mean((-3, -2, -1))
        rgd = ((p255 >= 180.0) & (p255 < 250.0)).float().mean((-3, -2, -1))
        ghost = (((p255 >= 180.0) & (p255 <= 250.0)).float() * (g255 > 250.0).float()).sum((-3, -2, -1))
        dark_g = ((p > 0.35).float() * (1.0 - _dil(_dil((g > 0.08).float())))).sum((-3, -2, -1))
        pb, gb = (p > 0.15).float(), (g > 0.15).float()
        pd_m = ((_blur(1 - p, 1.0) - _blur(1 - p, 1.6)) < -0.02).float()
        gd_m = ((_blur(1 - g, 1.0) - _blur(1 - g, 1.6)) < -0.02).float()
        pn = p / p.sum((-2, -1), keepdim=True).clamp(min=1e-6); gn = g / g.sum((-2, -1), keepdim=True).clamp(min=1e-6)
        semd = 0.5 * ((pn.sum(-1).cumsum(-1) - gn.sum(-1).cumsum(-1)).abs().mean((-2, -1)) + (pn.sum(-2).cumsum(-1) - gn.sum(-2).cumsum(-1)).abs().mean((-2, -1)))
        gdil, pdil = _dil(_dil(gb)), _dil(_dil(pb))
        rec = (gb * pdil).sum((-3, -2, -1)) / gb.sum((-3, -2, -1)).clamp(min=1.0)
        prec = (pb * gdil).sum((-3, -2, -1)) / pb.sum((-3, -2, -1)).clamp(min=1.0)
        f1 = 2 * rec * prec / (rec + prec).clamp(min=1e-6)
        sk_g = _skel(g); sk_m = (sk_g > 0.20).float(); pd1 = _dil(pb)
        sk_r = (pd1 * sk_m).sum((-3, -2, -1)) / sk_m.sum((-3, -2, -1)).clamp(min=1.0)
        nbr = F.avg_pool2d(sk_g, 3, 1, 1) * 9.0 - sk_g
        ep = ((nbr <= 1.8) & (sk_g > 0.20)).float()
        tip_r = (pd1 * ep).sum((-3, -2, -1)) / ep.sum((-3, -2, -1)).clamp(min=1.0)
        dens = pb.sum((-3, -2, -1)) / gb.sum((-3, -2, -1)).clamp(min=1.0)
        return {k: v.squeeze(-1) if v.ndim > 1 else v for k, v in {
            'psnr': psnr, 'ssim': ssim, 'val_l1': l1, 'ie_mae': l1 * 255.0,
            'pwcr': pwcr, 'rgd': rgd, 'ghost': ghost, 'dark_ghost': dark_g,
            'wcd': _wcd(pb, gb), 'wcd_dog': _wcd(pd_m, gd_m), 'semd': semd,
            'recall': rec, 'precision': prec, 'f1': f1, 'density_ratio': dens,
            'skel_recall': sk_r, 'tip_recall': tip_r,
        }.items()}


# ==============================================================================
# CÁC HÀM ĐỌC ẢNH, TRÍCH ĐIỂM NEO GLUESTICK/SIFT & VẼ BÁO CÁO TRỰC QUAN
# ==============================================================================
def load_ink_image(path):
    with Image.open(path) as im:
        gray = np.asarray(im.convert('L'), dtype=np.float32)
    return (255.0 - gray) / 255.0


def ink_to_u8_gray(ink_t):
    arr = (1.0 - ink_t.squeeze().detach().cpu().numpy().clip(0.0, 1.0)) * 255.0
    return np.clip(np.round(arr), 0, 255).astype(np.uint8)


def extract_sift_matches(im0_u8, im1_u8, max_pts=120):
    if cv2 is None:
        return np.zeros((2, 0, 2), dtype=np.float32)
    h, w = im0_u8.shape
    sift = cv2.SIFT_create(nfeatures=1000)
    k0, d0 = sift.detectAndCompute(im0_u8, None)
    k1, d1 = sift.detectAndCompute(im1_u8, None)
    if d0 is None or d1 is None or len(k0) < 4 or len(k1) < 4:
        return np.zeros((2, 0, 2), dtype=np.float32)
    matches = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True).match(d0, d1)
    if len(matches) < 4:
        return np.zeros((2, 0, 2), dtype=np.float32)
    p0 = np.float32([k0[m.queryIdx].pt for m in matches])
    p1 = np.float32([k1[m.trainIdx].pt for m in matches])
    _, mask = cv2.findHomography(p0, p1, cv2.RANSAC, 15.0)
    if mask is not None:
        p0, p1 = p0[mask.ravel() == 1], p1[mask.ravel() == 1]
    p0[:, 0] /= float(w); p0[:, 1] /= float(h)
    p1[:, 0] /= float(w); p1[:, 1] /= float(h)
    if len(p0) > max_pts:
        idx = np.linspace(0, len(p0) - 1, max_pts, dtype=int)
        p0, p1 = p0[idx], p1[idx]
    return np.stack([p0, p1], axis=0).astype(np.float32)


def try_init_gluestick(device):
    tps_repo = os.path.join(REPO_ROOT, "models", "tps-inbetween", "tps-inbetween")
    weights_p = os.path.join(tps_repo, "model", "resources", "weights", "checkpoint_GlueStick_MD.tar")
    if not os.path.exists(weights_p):
        return None
    if cv2 is not None and 'pytlsd' not in sys.modules:
        def _cv2_lsd(img, scale=1.0):
            if not isinstance(img, np.ndarray):
                img = np.array(img)
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img.dtype != np.uint8:
                img = (img * 255.0).clip(0, 255).astype(np.uint8) if img.max() <= 1.01 and img.max() > 0 else np.clip(img, 0, 255).astype(np.uint8)
            img_s = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA) if scale != 1.0 else img
            det = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
            lines, _, _, _ = det.detect(img_s)
            if lines is None or len(lines) == 0:
                return np.zeros((0, 5), dtype=np.float32)
            lines = lines.reshape(-1, 4).astype(np.float32)
            if scale != 1.0:
                lines /= float(scale)
            return np.hstack([lines, np.ones((len(lines), 1), dtype=np.float32)])
        fake = types.ModuleType('pytlsd')
        fake.lsd = _cv2_lsd
        sys.modules['pytlsd'] = fake
    if tps_repo not in sys.path:
        sys.path.insert(0, tps_repo)
    try:
        from model.gluestick.models.two_view_pipeline import TwoViewPipeline
        conf = {
            'name': 'two_view_pipeline', 'use_lines': True,
            'extractor': {
                'name': 'wireframe',
                'sp_params': {'force_num_keypoints': False, 'max_num_keypoints': 1000},
                'wireframe_params': {'merge_points': True, 'merge_line_endpoints': True},
                'max_n_lines': 300,
            },
            'matcher': {'name': 'gluestick', 'weights': weights_p, 'trainable': False},
            'ground_truth': {'from_pose_depth': False},
        }
        return TwoViewPipeline(conf).to(device).eval()
    except Exception:
        return None


def compute_wilcoxon_p(a_vals, b_vals):
    if wilcoxon is None or len(a_vals) < 10:
        return float('nan')
    diff = np.asarray(a_vals, dtype=np.float64) - np.asarray(b_vals, dtype=np.float64)
    if np.allclose(diff, 0.0):
        return 1.0
    try:
        _, p = wilcoxon(diff, alternative='two-sided', zero_method='wilcox')
        return float(p)
    except Exception:
        return float('nan')


def render_audit_row_panel(im0_u8, im1_u8, m0_u8, m1_u8, m3_u8, gt_u8, title_str, stats_m1, stats_m3):
    """Vẽ dải 5 cột chuẩn: [Overlay I0/I1 | M0 Coarse | M1 Orig | M3 Finetune | Ground Truth]."""
    h, w = gt_u8.shape
    overlay = (0.5 * im0_u8.astype(np.float32) + 0.5 * im1_u8.astype(np.float32)).astype(np.uint8)
    cols = [overlay, m0_u8, m1_u8, m3_u8, gt_u8]
    titles = [
        ("Overlay I0/I1", (210, 210, 210)),
        ("M0 Coarse TPS", (180, 180, 180)),
        (f"M1 Orig ({stats_m1})", (100, 150, 255)),
        (f"M3 Fine ({stats_m3})", (90, 240, 120)),
        ("Ground Truth", (245, 245, 245)),
    ]
    head_h, foot_h = 34, 28
    canvas = np.full((head_h + h + foot_h, w * 5, 3), 26, dtype=np.uint8)
    for i, (arr, (lbl, clr)) in enumerate(zip(cols, titles)):
        bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR) if cv2 is not None else np.repeat(arr[:, :, None], 3, axis=-1)
        canvas[head_h:head_h + h, i * w:(i + 1) * w] = bgr
        if cv2 is not None:
            cv2.putText(canvas, lbl, (i * w + 10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, clr, 2, cv2.LINE_AA)
            if i > 0:
                cv2.line(canvas, (i * w, 0), (i * w, head_h + h + foot_h), (70, 70, 70), 2)
    if cv2 is not None:
        cv2.putText(canvas, title_str, (12, head_h + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


# ==============================================================================
# BENCHMARK #1: TẬP CHUỖI CÁCH LY `sakuga_golden_benchmark_sequences_v2`
# ==============================================================================
def run_benchmark_isolated_sequences(bench_dir, models_dict, device, out_dir, downscale_factor=2, save_visuals=True):
    csv_path = os.path.join(bench_dir, "benchmark_sequences.csv")
    clips_dir = os.path.join(bench_dir, "clips")
    matches_dir = os.path.join(bench_dir, "matches")
    df_bench = pd.read_csv(csv_path)

    print("\n" + "=" * 86)
    print(f"🔬 BENCHMARK #1: TẬP CHUỖI CÁCH LY ({len(df_bench)} sequences / {df_bench['scene_id'].nunique()} scenes)")
    print(f"   Các mô hình đối chứng: {['M0_Coarse'] + list(models_dict.keys())}")
    print("=" * 86, flush=True)

    per_sample_rows = []
    # Lưu thông tin nhẹ để sau chọn Top-3 cải thiện & Top-3 tệ nhất (chỉ lưu path + metric)
    sample_index = []

    t_start = time.time()
    for row_idx, row in df_bench.iterrows():
        cid = str(row['clip_id'])
        sid = str(row['scene_id'])
        seq_len = int(row['seq_len'])
        disp = float(row.get('mean_disp', 0.0))

        # Đọc matches chuẩn Backward-Warp
        m_cands = sorted(glob.glob(os.path.join(matches_dir, cid, "*.npy")))
        if m_cands:
            kps = np.load(m_cands[0]).astype(np.float32)
            c0_np, th01_np, c1_np, th10_np, v_flag = solve_tps_backward_pair(kps[0], kps[1], max_pts=128, lambd=0.01)
        else:
            c0_np = c1_np = np.zeros((128, 2), dtype=np.float32)
            th01_np = th10_np = np.zeros((131, 2), dtype=np.float32)
            v_flag = 0.0

        im0 = load_ink_image(os.path.join(clips_dir, cid, "000001.png"))
        im1 = load_ink_image(os.path.join(clips_dir, cid, f"{seq_len:06d}.png"))
        h, w = im0.shape
        h_pad = ((h + 31) // 32) * 32
        w_pad = ((w + 31) // 32) * 32

        x0 = F.pad(torch.from_numpy(im0)[None, None], (0, w_pad - w, 0, h_pad - h)).to(device)
        x1 = F.pad(torch.from_numpy(im1)[None, None], (0, w_pad - w, 0, h_pad - h)).to(device)
        orig_hw = torch.tensor([[h, w]], dtype=torch.int32, device=device)
        c0_t = torch.from_numpy(c0_np)[None].to(device)
        th01_t = torch.from_numpy(th01_np)[None].to(device)
        c1_t = torch.from_numpy(c1_np)[None].to(device)
        th10_t = torch.from_numpy(th10_np)[None].to(device)
        v_t = torch.tensor([v_flag], dtype=torch.float32, device=device)

        cf01 = tps_flow_from_theta_batch(th01_t, c0_t, v_t, orig_hw, h_pad, w_pad, device, downscale_factor=downscale_factor)
        cf10 = tps_flow_from_theta_batch(th10_t, c1_t, v_t, orig_hw, h_pad, w_pad, device, downscale_factor=downscale_factor)

        for k in range(1, seq_len - 1):
            fk = k + 1
            t_val = float(k) / float(seq_len - 1)
            t_tensor = torch.tensor([[[[t_val]]]], dtype=torch.float32, device=device)
            gt_ink = torch.from_numpy(load_ink_image(os.path.join(clips_dir, cid, f"{fk:06d}.png")))[None, None].to(device)

            sample_rec = {
                'clip_id': cid, 'scene_id': sid, 'seq_len': seq_len, 'frame_k': fk,
                't_val': round(t_val, 4), 'mean_disp': disp, 'orig_w': w, 'orig_h': h
            }
            coarse_ref = None
            for m_name, mdl in models_dict.items():
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                ts = time.time()
                with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    pred_p, cw_p, *_ = mdl(x0, x1, t_tensor, cf01, cf10)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                lat_ms = (time.time() - ts) * 1000.0

                pred_crop = pred_p[:, :, :h, :w]
                if coarse_ref is None:
                    coarse_ref = cw_p[:, :, :h, :w]
                    mm0 = evaluate_metrics_dual_regime(coarse_ref, gt_ink)
                    for mk, mv in mm0.items():
                        sample_rec[f"M0_{mk}"] = float(mv.item())

                mm = evaluate_metrics_dual_regime(pred_crop, gt_ink)
                for mk, mv in mm.items():
                    sample_rec[f"{m_name}_{mk}"] = float(mv.item())
                sample_rec[f"{m_name}_latency_ms"] = float(lat_ms)

            per_sample_rows.append(sample_rec)
            sample_index.append(sample_rec)

        if (row_idx + 1) % 50 == 0 or (row_idx + 1) == len(df_bench):
            print(f"   ... Hoàn thành {row_idx + 1}/{len(df_bench)} sequences ({time.time() - t_start:.1f}s)", flush=True)

    df_res = pd.DataFrame(per_sample_rows)
    os.makedirs(out_dir, exist_ok=True)
    df_res.to_csv(os.path.join(out_dir, "benchmark_per_sample_comparison.csv"), index=False)

    # Tổng hợp bảng so sánh toàn cục cho tất cả model tags
    all_tags = ['M0'] + list(models_dict.keys())
    metric_cols = ['psnr', 'ssim', 'ie_mae', 'pwcr', 'rgd', 'ghost', 'wcd', 'wcd_dog',
                   'semd', 'skel_recall', 'tip_recall', 'recall', 'precision', 'f1', 'density_ratio']
    summary_rows = []
    for tag in all_tags:
        srow = {'model_tag': tag, 'total_frames': len(df_res), 'total_scenes': df_res['scene_id'].nunique()}
        for mc in metric_cols:
            col = f"{tag}_{mc}"
            if col in df_res.columns:
                srow[mc] = float(df_res[col].mean())
        lat_col = f"{tag}_latency_ms"
        srow['latency_ms'] = float(df_res[lat_col].mean()) if lat_col in df_res.columns else 0.0
        summary_rows.append(srow)
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(out_dir, "benchmark_overall_summary.csv"), index=False)

    # Phân tầng theo seq_len N={3,5,7} và Motion Tiers (<6, 6..14, >=14)
    strat_rows = []
    for sl in (3, 5, 7):
        sub = df_res[df_res['seq_len'] == sl]
        if len(sub) == 0:
            continue
        for tag in all_tags:
            r = {'group_type': 'seq_len', 'group_val': f"N={sl}", 'model_tag': tag, 'frames': len(sub)}
            for mc in ['psnr', 'ssim', 'wcd', 'wcd_dog', 'semd', 'skel_recall', 'tip_recall', 'f1', 'ghost', 'density_ratio']:
                r[mc] = float(sub[f"{tag}_{mc}"].mean())
            strat_rows.append(r)

    disp_bins = [('Small_<6px', df_res['mean_disp'] < 6.0),
                 ('Med_6_14px', (df_res['mean_disp'] >= 6.0) & (df_res['mean_disp'] < 14.0)),
                 ('Large_>=14px', df_res['mean_disp'] >= 14.0)]
    for dlabel, mask in disp_bins:
        sub = df_res[mask]
        if len(sub) == 0:
            continue
        for tag in all_tags:
            r = {'group_type': 'motion_disp', 'group_val': dlabel, 'model_tag': tag, 'frames': len(sub)}
            for mc in ['psnr', 'ssim', 'wcd', 'wcd_dog', 'semd', 'skel_recall', 'tip_recall', 'f1', 'ghost']:
                r[mc] = float(sub[f"{tag}_{mc}"].mean())
            strat_rows.append(r)
    pd.DataFrame(strat_rows).to_csv(os.path.join(out_dir, "benchmark_stratified_breakdown.csv"), index=False)

    # Kiểm định thống kê không thiên vị M3_best (hoặc model thứ 2) vs M1_Original
    stat_audit = {'overall': summary_rows}
    if 'M1_Original' in models_dict:
        compare_tag = 'M3_Best' if 'M3_Best' in models_dict else next((k for k in models_dict if k != 'M1_Original'), None)
        if compare_tag is not None:
            # Gom nhóm trung bình theo từng scene độc lập (149 scenes) trước Wilcoxon test!
            scene_df = df_res.groupby('scene_id').mean(numeric_only=True)
            h2h = {}
            for mc, better_higher in [('wcd', False), ('wcd_dog', False), ('semd', False),
                                      ('skel_recall', True), ('tip_recall', True), ('f1', True), ('psnr', True), ('ssim', True)]:
                c_m1, c_m3 = f"M1_Original_{mc}", f"{compare_tag}_{mc}"
                if better_higher:
                    win_rate = float(np.mean(df_res[c_m3].values > df_res[c_m1].values))
                else:
                    win_rate = float(np.mean(df_res[c_m3].values < df_res[c_m1].values))
                pval = compute_wilcoxon_p(scene_df[c_m3].values, scene_df[c_m1].values)
                h2h[mc] = {
                    'm1_mean': float(df_res[c_m1].mean()),
                    'm3_mean': float(df_res[c_m3].mean()),
                    'delta': float(df_res[c_m3].mean() - df_res[c_m1].mean()),
                    'frame_win_rate_pct': round(win_rate * 100.0, 2),
                    'scene_wilcoxon_p': pval,
                    'significant_p001': (not math.isnan(pval)) and (pval < 0.01),
                }
            # Kết luận H0 vs H1 tự động dựa trên cả 2 trụ cột Hình học (WCD_DoG/S-EMD) + Bảo toàn nét (Skel_Recall/F1)
            geom_pass = (h2h['wcd_dog']['delta'] < 0 and h2h['wcd_dog']['significant_p001']) or (h2h['wcd']['delta'] < 0 and h2h['wcd']['significant_p001'])
            topo_pass = h2h['skel_recall']['delta'] > 0 and h2h['skel_recall']['significant_p001']
            stat_audit['head_to_head_M3_vs_M1'] = h2h
            stat_audit['hypothesis_verdict'] = (
                "H1_SUPPORTED (Cải thiện đồng thời cả Hình học + Tô-pô nét vẽ với p < 0.01)"
                if (geom_pass and topo_pass) else
                "MIXED_OR_H0 (Cần xem xét kỹ phân tách Raw vs DoG và chỉ số Skel/WCD)"
            )

            # Xuất Top-3 Cải Thiện & Top-3 Failure Cases theo delta WCD_DoG
            if save_visuals and len(df_res) >= 6:
                df_res['delta_wcd_dog'] = df_res[f"{compare_tag}_wcd_dog"] - df_res["M1_Original_wcd_dog"]
                best3 = df_res.sort_values('delta_wcd_dog', ascending=True).head(3)
                worst3 = df_res.sort_values('delta_wcd_dog', ascending=False).head(3)
                vis_dir = os.path.join(out_dir, "audit_cases_top3")
                os.makedirs(vis_dir, exist_ok=True)
                for tag_grp, sub_df in [("TOP3_IMPROVED", best3), ("TOP3_FAILURE_OR_HARD", worst3)]:
                    panels = []
                    for _, sr in sub_df.iterrows():
                        cid, fk, sl = sr['clip_id'], int(sr['frame_k']), int(sr['seq_len'])
                        im0_l = load_ink_image(os.path.join(clips_dir, cid, "000001.png"))
                        im1_l = load_ink_image(os.path.join(clips_dir, cid, f"{sl:06d}.png"))
                        gt_l = load_ink_image(os.path.join(clips_dir, cid, f"{fk:06d}.png"))
                        hh, ww = im0_l.shape
                        hp, wp = ((hh + 31) // 32) * 32, ((ww + 31) // 32) * 32
                        mc = sorted(glob.glob(os.path.join(matches_dir, cid, "*.npy")))
                        if mc:
                            kps = np.load(mc[0]).astype(np.float32)
                            c0n, t01n, c1n, t10n, vf = solve_tps_backward_pair(kps[0], kps[1], 128, 0.01)
                        else:
                            c0n = c1n = np.zeros((128, 2), np.float32); t01n = t10n = np.zeros((131, 2), np.float32); vf = 0.0
                        x0p = F.pad(torch.from_numpy(im0_l)[None, None], (0, wp - ww, 0, hp - hh)).to(device)
                        x1p = F.pad(torch.from_numpy(im1_l)[None, None], (0, wp - ww, 0, hp - hh)).to(device)
                        hw_t = torch.tensor([[hh, ww]], dtype=torch.int32, device=device)
                        cf01_ = tps_flow_from_theta_batch(torch.from_numpy(t01n)[None].to(device), torch.from_numpy(c0n)[None].to(device), torch.tensor([vf], device=device), hw_t, hp, wp, device, downscale_factor)
                        cf10_ = tps_flow_from_theta_batch(torch.from_numpy(t10n)[None].to(device), torch.from_numpy(c1n)[None].to(device), torch.tensor([vf], device=device), hw_t, hp, wp, device, downscale_factor)
                        tt = torch.tensor([[[[float(sr['t_val'])]]]], dtype=torch.float32, device=device)
                        with torch.no_grad():
                            pr1, cw_, *_ = models_dict['M1_Original'](x0p, x1p, tt, cf01_, cf10_)
                            pr3, *_ = models_dict[compare_tag](x0p, x1p, tt, cf01_, cf10_)
                        panel = render_audit_row_panel(
                            ink_to_u8_gray(x0p[:, :, :hh, :ww]), ink_to_u8_gray(x1p[:, :, :hh, :ww]),
                            ink_to_u8_gray(cw_[:, :, :hh, :ww]), ink_to_u8_gray(pr1[:, :, :hh, :ww]),
                            ink_to_u8_gray(pr3[:, :, :hh, :ww]), (255.0 - gt_l * 255.0).clip(0, 255).astype(np.uint8),
                            f"[{tag_grp}] {cid} (N={sl}, k={fk}, t={sr['t_val']:.2f}, disp={sr['mean_disp']:.1f}px) | dWCD_DoG={sr['delta_wcd_dog']:+.4f}",
                            f"WCD={sr['M1_Original_wcd_dog']:.3f}|Sk={sr['M1_Original_skel_recall']*100:.0f}%",
                            f"WCD={sr[f'{compare_tag}_wcd_dog']:.3f}|Sk={sr[f'{compare_tag}_skel_recall']*100:.0f}%"
                        )
                        panels.append(panel)
                    # Ghép 3 hàng có thể hơi khác chiều rộng -> đệm ngang về max_w
                    max_pw = max(p.shape[1] for p in panels)
                    norm_panels = [np.pad(p, ((0, 0), (0, max_pw - p.shape[1]), (0, 0)), constant_values=26) for p in panels]
                    stacked = np.vstack(norm_panels)
                    Image.fromarray(stacked[:, :, ::-1] if cv2 is not None else stacked).save(os.path.join(vis_dir, f"{tag_grp}.png"))

    with open(os.path.join(out_dir, "benchmark_statistical_audit.json"), 'w', encoding='utf-8') as f:
        json.dump(stat_audit, f, indent=2, ensure_ascii=False)

    print("\n📊 KẾT QUẢ BENCHMARK #1 (ISOLATED 198 CLIPS / 412 FRAMES):")
    print(df_summary.round(4).to_string(index=False))
    if 'head_to_head_M3_vs_M1' in stat_audit:
        print(f"\n⚖️ KIỂM ĐỊNH ĐỐI ĐẦU THEO SCENE (149 SCENES) | Kết luận: {stat_audit['hypothesis_verdict']}")
        for mc, st in stat_audit['head_to_head_M3_vs_M1'].items():
            print(f"   - {mc:12s}: M1={st['m1_mean']:.4f} -> M3={st['m3_mean']:.4f} (Delta={st['delta']:+.4f} | WinRate={st['frame_win_rate_pct']:.1f}% | p={st['scene_wilcoxon_p']:.2e})")
    return df_summary, stat_audit


# ==============================================================================
# BENCHMARK #2: TẠO FRAME THỰC TẾ LIÊN TỤC (`data/lineart_experiments/test`)
# ==============================================================================
def run_benchmark_continuous_lineart_test(test_dir, models_dict, device, out_dir, gluestick_npz=None, downscale_factor=2):
    spec_path = os.path.join(test_dir, "continuous_keyframes_spec.json")
    if not os.path.exists(spec_path):
        print(f"[*] Không tìm thấy {spec_path}, bỏ qua Benchmark #2.")
        return None

    with open(spec_path, 'r', encoding='utf-8') as f:
        spec = json.load(f)
    frames_spec = spec['frames']
    key_indices = sorted(spec['key_indices'])

    print("\n" + "=" * 86)
    print(f"🎬 BENCHMARK #2: TẠO FRAME THỰC TẾ LIÊN TỤC ({len(frames_spec)} frames: {len(key_indices)} keyframes, {len(frames_spec)-len(key_indices)} inbetweens)")
    print("=" * 86, flush=True)

    # Nạp ảnh GT 960x540
    gt_imgs_u8 = [np.asarray(Image.open(os.path.join(test_dir, item['filename'])).convert('L'), dtype=np.uint8) for item in frames_spec]
    h, w = gt_imgs_u8[0].shape
    h_pad, w_pad = ((h + 31) // 32) * 32, ((w + 31) // 32) * 32  # 540 -> 544, 960 -> 960

    # Chuẩn bị cặp điểm neo GlueStick (ưu tiên precomputed .npz -> chạy GlueStick trực tiếp -> SIFT fallback)
    gs_cache = dict(np.load(gluestick_npz)) if (gluestick_npz and os.path.exists(gluestick_npz)) else None
    gs_model = None if gs_cache is not None else try_init_gluestick(device)
    match_source = "GlueStick_MD_Precomputed" if gs_cache is not None else ("GlueStick_MD_Live" if gs_model is not None else "SIFT_Fallback")
    print(f"[*] Bộ trích xuất điểm neo hình học: {match_source} | Pad: {w}x{h} -> {w_pad}x{h_pad}", flush=True)

    seg_data = {}
    for i in range(len(key_indices) - 1):
        ka, kb = key_indices[i], key_indices[i + 1]
        seg_key = f"seg_{ka}_{kb}"
        ima_u8, imb_u8 = gt_imgs_u8[ka], gt_imgs_u8[kb]
        if gs_cache is not None and seg_key in gs_cache:
            kps = gs_cache[seg_key].astype(np.float32)
        elif gs_model is not None:
            t0_g = torch.from_numpy(ima_u8.astype(np.float32) / 255.0)[None, None].to(device)
            t1_g = torch.from_numpy(imb_u8.astype(np.float32) / 255.0)[None, None].to(device)
            with torch.no_grad():
                pred_g = gs_model({'image0': t0_g, 'image1': t1_g})
            m0 = pred_g['matches0'][0].cpu().numpy()
            valid_g = m0 != -1
            kp0 = pred_g['keypoints0'][0].cpu().numpy()[valid_g].astype(np.float32)
            kp1 = pred_g['keypoints1'][0].cpu().numpy()[m0[valid_g]].astype(np.float32)
            kp0[:, 0] /= float(w); kp0[:, 1] /= float(h)
            kp1[:, 0] /= float(w); kp1[:, 1] /= float(h)
            kps = np.stack([kp0, kp1], axis=0)
        else:
            kps = extract_sift_matches(ima_u8, imb_u8, max_pts=120)

        c0_np, t01_np, c1_np, t10_np, vf = solve_tps_backward_pair(kps[0], kps[1], max_pts=128, lambd=0.01)
        x0_p = F.pad(torch.from_numpy((255.0 - ima_u8.astype(np.float32)) / 255.0)[None, None], (0, w_pad - w, 0, h_pad - h)).to(device)
        x1_p = F.pad(torch.from_numpy((255.0 - imb_u8.astype(np.float32)) / 255.0)[None, None], (0, w_pad - w, 0, h_pad - h)).to(device)
        hw_t = torch.tensor([[h, w]], dtype=torch.int32, device=device)
        cf01 = tps_flow_from_theta_batch(torch.from_numpy(t01_np)[None].to(device), torch.from_numpy(c0_np)[None].to(device), torch.tensor([vf], device=device), hw_t, h_pad, w_pad, device, downscale_factor)
        cf10 = tps_flow_from_theta_batch(torch.from_numpy(t10_np)[None].to(device), torch.from_numpy(c1_np)[None].to(device), torch.tensor([vf], device=device), hw_t, h_pad, w_pad, device, downscale_factor)
        seg_data[(ka, kb)] = {'x0': x0_p, 'x1': x1_p, 'cf01': cf01, 'cf10': cf10, 'num_matches': kps.shape[1]}

    os.makedirs(out_dir, exist_ok=True)
    reconstructed = {tag: [img.copy() for img in gt_imgs_u8] for tag in ['M0'] + list(models_dict.keys())}
    per_frame_rows = []

    for item in frames_spec:
        f_idx = int(item['frame_idx'])
        if item['type'] != 'INBETWEEN':
            continue
        ka, kb, tv = int(item['ref_key_a']), int(item['ref_key_b']), float(item['t_val'])
        delta_f = kb - ka
        seg = seg_data[(ka, kb)]
        gt_ink = torch.from_numpy((255.0 - gt_imgs_u8[f_idx].astype(np.float32)) / 255.0)[None, None].to(device)
        tt = torch.tensor([[[[tv]]]], dtype=torch.float32, device=device)

        row_base = {'frame_idx': f_idx, 'time_sec': item['time_sec'], 'filename': item['filename'],
                    'segment': f"{ka}->{kb}", 'delta_f': delta_f, 't_val': round(tv, 4), 'matches': seg['num_matches']}
        cw_done = False
        for m_name, mdl in models_dict.items():
            ts = time.time()
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                pr_p, cw_p, *_ = mdl(seg['x0'], seg['x1'], tt, seg['cf01'], seg['cf10'])
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            lat = (time.time() - ts) * 1000.0
            pr_crop = pr_p[:, :, :h, :w]
            reconstructed[m_name][f_idx] = ink_to_u8_gray(pr_crop)
            mm = evaluate_metrics_dual_regime(pr_crop, gt_ink)

            if not cw_done:
                cw_crop = cw_p[:, :, :h, :w]
                reconstructed['M0'][f_idx] = ink_to_u8_gray(cw_crop)
                mm0 = evaluate_metrics_dual_regime(cw_crop, gt_ink)
                for mk, mv in mm0.items():
                    row_base[f"M0_{mk}"] = float(mv.item())
                cw_done = True

            for mk, mv in mm.items():
                row_base[f"{m_name}_{mk}"] = float(mv.item())
            row_base[f"{m_name}_latency_ms"] = float(lat)

        per_frame_rows.append(row_base)

    df_frames = pd.DataFrame(per_frame_rows)
    df_frames.to_csv(os.path.join(out_dir, "lineart_test_per_frame_metrics.csv"), index=False)

    # Tổng hợp theo toàn chuỗi & theo từng delta_f (3, 4, 5)
    summary_list = []
    tags = ['M0'] + list(models_dict.keys())
    for grp_name, sub_df in [("ALL_17_INBETWEENS", df_frames)] + [(f"delta_f={dfv}", df_frames[df_frames['delta_f'] == dfv]) for dfv in sorted(df_frames['delta_f'].unique())]:
        for t in tags:
            sr = {'group': grp_name, 'model_tag': t, 'frames': len(sub_df)}
            for mc in ['psnr', 'ssim', 'ie_mae', 'wcd', 'wcd_dog', 'semd', 'skel_recall', 'tip_recall', 'f1', 'density_ratio', 'ghost', 'pwcr', 'rgd']:
                sr[mc] = float(sub_df[f"{t}_{mc}"].mean())
            summary_list.append(sr)
    df_sum2 = pd.DataFrame(summary_list)
    df_sum2.to_csv(os.path.join(out_dir, "lineart_test_summary.csv"), index=False)

    print("\n📊 KẾT QUẢ BENCHMARK #2 (ALL 17 INBETWEEN FRAMES - GLUESTICK):")
    print(df_sum2[df_sum2['group'] == 'ALL_17_INBETWEENS'].round(4).to_string(index=False))

    # Xuất Video / GIF 24 frames, Error Heatmaps & Side-by-Side (nếu có cv2 hoặc imageio)
    m1_tag = 'M1_Original' if 'M1_Original' in models_dict else None
    m3_tag = 'M3_Best' if 'M3_Best' in models_dict else next((k for k in models_dict if k != 'M1_Original'), None)
    if m1_tag and m3_tag and cv2 is not None:
        sbs_dir = os.path.join(out_dir, "frames_side_by_side")
        heat_dir = os.path.join(out_dir, "error_heatmaps")
        os.makedirs(sbs_dir, exist_ok=True)
        os.makedirs(heat_dir, exist_ok=True)
        gif_frames = []
        montage_rows = []
        # Vẽ thanh Side-by-Side 4 cột [Overlay Keyframes | Original M1 | Finetuned M3 | Ground Truth]
        for item in frames_spec:
            idx = int(item['frame_idx'])
            is_k = (item['type'] == 'KEYFRAME')
            gt_u8 = gt_imgs_u8[idx]
            m1_u8 = reconstructed[m1_tag][idx]
            m3_u8 = reconstructed[m3_tag][idx]
            if is_k:
                ov_u8 = gt_u8
                s1, s3 = "KEYFRAME", "KEYFRAME"
            else:
                ka, kb = int(item['ref_key_a']), int(item['ref_key_b'])
                ov_u8 = (0.5 * gt_imgs_u8[ka].astype(np.float32) + 0.5 * gt_imgs_u8[kb].astype(np.float32)).astype(np.uint8)
                fr_row = df_frames[df_frames['frame_idx'] == idx].iloc[0]
                s1 = f"PSNR:{fr_row[f'{m1_tag}_psnr']:.1f}|Sk:{fr_row[f'{m1_tag}_skel_recall']*100:.0f}%"
                s3 = f"PSNR:{fr_row[f'{m3_tag}_psnr']:.1f}|Sk:{fr_row[f'{m3_tag}_skel_recall']*100:.0f}%"

                # Bản đồ nhiệt sai số: [M1 Error Heatmap | M3 Error Heatmap]
                d1 = np.clip(np.abs(m1_u8.astype(np.float32) - gt_u8.astype(np.float32)) * 2.5, 0, 255).astype(np.uint8)
                d3 = np.clip(np.abs(m3_u8.astype(np.float32) - gt_u8.astype(np.float32)) * 2.5, 0, 255).astype(np.uint8)
                hm_pair = np.hstack([cv2.applyColorMap(d1, cv2.COLORMAP_JET), cv2.applyColorMap(d3, cv2.COLORMAP_JET)])
                hm_head = np.full((32, w * 2, 3), 24, dtype=np.uint8)
                cv2.putText(hm_head, f"M1 ORIG ERROR [MAE={fr_row[f'{m1_tag}_ie_mae']:.2f}]", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 150, 255), 2, cv2.LINE_AA)
                cv2.putText(hm_head, f"M3 FINE ERROR [MAE={fr_row[f'{m3_tag}_ie_mae']:.2f}]", (w + 12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 240, 120), 2, cv2.LINE_AA)
                cv2.imwrite(os.path.join(heat_dir, f"heatmap_{idx:02d}_{item['filename']}"), np.vstack([hm_head, hm_pair]))

            head_h = 34
            canvas = np.full((h + head_h, w * 4, 3), 24, dtype=np.uint8)
            cols = [(ov_u8, f"INPUT [{item['filename']}]", (210, 210, 210)),
                    (m1_u8, f"M1 ORIG ({s1})", (100, 150, 255)),
                    (m3_u8, f"M3 FINE ({s3})", (80, 240, 120)),
                    (gt_u8, "GROUND TRUTH", (245, 245, 245))]
            for ci, (arr, lbl, colr) in enumerate(cols):
                canvas[head_h:, ci * w:(ci + 1) * w] = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                cv2.putText(canvas, lbl, (ci * w + 10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, colr, 2, cv2.LINE_AA)
                if ci > 0:
                    cv2.line(canvas, (ci * w, 0), (ci * w, h + head_h), (70, 70, 70), 2)
            out_png = os.path.join(sbs_dir, f"frame_{idx:02d}_{item['filename']}")
            cv2.imwrite(out_png, canvas)
            half_frame = cv2.resize(canvas, (w * 2, (h + head_h) // 2))
            gif_frames.append(half_frame[:, :, ::-1])
            if (not is_k) and (idx in {2, 5, 9, 12, 17, 21}):
                montage_rows.append(half_frame)

        if montage_rows:
            cv2.imwrite(os.path.join(out_dir, "montage_6segments_M1_vs_M3.png"), np.vstack(montage_rows))

        if imageio is not None:
            gif_path = os.path.join(out_dir, "lineart_test_comparison_M1_vs_M3.gif")
            imageio.mimsave(gif_path, gif_frames, duration=0.14, loop=0)
            print(f"[✓] Đã xuất ảnh động đối đầu GIF: {gif_path}")

        mp4_path = os.path.join(out_dir, "lineart_test_comparison_M1_vs_M3.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(mp4_path, fourcc, 8.0, (gif_frames[0].shape[1], gif_frames[0].shape[0]))
        for gf in gif_frames:
            vw.write(gf[:, :, ::-1])
        vw.release()
        print(f"[✓] Đã xuất video đối đầu MP4: {mp4_path}")

    return df_sum2


def main():
    parser = argparse.ArgumentParser(description="Chạy Dual-Benchmark đối chứng khách quan (M0 vs M1 vs M3)")
    parser.add_argument('--bench_dir', type=str, default=SCRIPT_DIR)
    parser.add_argument('--orig_ckpt', type=str, default=DEFAULT_ORIG_CKPT)
    parser.add_argument('--finetuned_ckpt', type=str, default=None, help="Đường dẫn model_tps_finetuned_best.pt")
    parser.add_argument('--last_ckpt', type=str, default=None, help="Đường dẫn model_tps_finetuned_last.pt (tùy chọn)")
    parser.add_argument('--lineart_test_dir', type=str, default=DEFAULT_LINEART_TEST)
    parser.add_argument('--gluestick_npz', type=str, default=DEFAULT_GLUESTICK_CACHE)
    parser.add_argument('--out_dir', type=str, default=os.path.join(SCRIPT_DIR, "benchmark_results"))
    parser.add_argument('--downscale', type=int, default=2)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    models_dict = {}

    if os.path.exists(args.orig_ckpt):
        m1 = TPSInbetweenOriginal().to(device)
        load_tps_weights(m1, args.orig_ckpt, device=device, prefer_ema=False)
        models_dict['M1_Original'] = m1.eval()
    else:
        print(f"[!] Không tìm thấy Original Checkpoint: {args.orig_ckpt}")

    if args.finetuned_ckpt and os.path.exists(args.finetuned_ckpt):
        m3_b = TPSInbetweenOriginal().to(device)
        load_tps_weights(m3_b, args.finetuned_ckpt, device=device, prefer_ema=True)
        models_dict['M3_Best'] = m3_b.eval()

    if args.last_ckpt and os.path.exists(args.last_ckpt):
        m3_l = TPSInbetweenOriginal().to(device)
        load_tps_weights(m3_l, args.last_ckpt, device=device, prefer_ema=True)
        models_dict['M3_Last'] = m3_l.eval()

    if not models_dict:
        print("[!] Chưa nạp được mô hình nào để đối chuẩn!")
        sys.exit(1)

    run_benchmark_isolated_sequences(
        args.bench_dir, models_dict, device,
        os.path.join(args.out_dir, "bench1_isolated_sequences"),
        downscale_factor=args.downscale, save_visuals=True
    )
    run_benchmark_continuous_lineart_test(
        args.lineart_test_dir, models_dict, device,
        os.path.join(args.out_dir, "bench2_continuous_lineart_test"),
        gluestick_npz=args.gluestick_npz, downscale_factor=args.downscale
    )
    print(f"\n🎉 HOÀN TẤT BÁO CÁO ĐỐI CHUẨN TẠI: {args.out_dir}")


if __name__ == '__main__':
    main()
