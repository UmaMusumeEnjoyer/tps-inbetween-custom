#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seq_p3_v2_multilayer_tps_worker.py
--------------------------------------------------------------------------------
ĐỘNG CƠ HUẤN LUYỆN DDP 2x GPU PHA 3 — COARSE TPS ĐA LỚP (MULTI-LAYER CLUSTERED TPS)
Kế thừa 100% kiến trúc `TPSInbetweenOriginal` (10,068,668 active params — 0% sửa kiến trúc)
và tinh chỉnh nhẹ (2-3 Epochs) từ `model_tps_blended.pt` để mạng neural thích nghi hoàn toàn
với đặc tính trường dòng chảy đa lớp (Static Layer + Clustered Dynamic Layers + Voronoi Mask).

Các nâng cấp cốt lõi của Pha 3 (Bước 4):
  1. Nạp trực tiếp bộ cache `tps_cache_multilayer.npz` (hoặc tương thích fallback single-layer)
     và tái tạo batch flow đa lớp siêu tốc trên GPU (`reconstruct_multilayer_flow_batch_from_cache`).
  2. Làm giàu dữ liệu có chủ đích tại chỗ (Zero-cost Compositional Multi-Layer Enrichment):
     - Static-Background Compositing (`--static_aug_prob 0.25`) trên chuỗi Sakuga thực tế:
       trộn nền kiến trúc/đường chân trời tĩnh vào (x0, x1, xk_all) kèm static keypoints.
     - Nâng cấp 100% chuỗi hình học tổng hợp (`--synth_count 180`) sang Multi-Layer TPS với 6 Mode,
       bổ sung Mode Occlusion Crossing (vật thể động lướt qua nét tĩnh) & Multi-Cluster Crossing.
  3. Cân bằng trọng số Loss Pha 3 (`w_ghost=0.65, w_temp=0.55`) + LR phẫu thuật (`1.0e-5 / 2.5e-5`)
     để vừa làm sạch ranh giới giao thoa, vừa giữ vững mật độ nét chuẩn ~1.00 của Blended Model.
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
import contextlib
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
for _p in (SCRIPT_DIR, os.path.join(REPO_ROOT, "src", "sequence_p3_p4")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from seq_p3_multi_layer_tps import (
    FIXED_TPS_ANCHORS,
    solve_multilayer_cache_entry,
    reconstruct_multilayer_flow_batch_from_cache,
)

# Import các thành phần lõi không thay đổi (TPSInbetweenOriginal, Loss, Metrics, Sampler, EMA, Manager)
_P2_DIR = os.path.join(REPO_ROOT, "training", "03_sequence", "p2_v33_ghost_purge")
if os.path.isdir(_P2_DIR) and _P2_DIR not in sys.path:
    sys.path.insert(0, _P2_DIR)

try:
    from seq_p2_v33_ghost_purge_worker import (
        trim_memory,
        TPSInbetweenOriginal,
        load_tps_weights,
        solve_tps_backward_pair,
        tps_flow_from_theta_batch,
        AdaptiveLineartLoss,
        BidirectionalStrokeWidthLoss,
        SoftClDiceLoss,
        VGGPerceptualOrLaplacianLoss,
        approx_gpu_distance_map,
        soft_dilate,
        soft_skeletonize,
        evaluate_metrics_dual_regime,
        DDPLengthWidthBatchSampler,
        AdaptiveEMA,
        TwoFileCheckpointManager,
        _apply_paired_spatial_aug_gpu,
    )
    _IMPORTED_P2 = True
except ImportError:
    _IMPORTED_P2 = False

if not _IMPORTED_P2:
    # Khi chạy độc lập trên Kaggle, seq_p2_v33_ghost_purge_worker.py được Cell 1 ghi chung cùng thư mục
    raise ImportError("Không tìm thấy module nền tảng `seq_p2_v33_ghost_purge_worker.py`!")


# ==============================================================================
# 0b. HÀM MẤT MÁT PHA 3 CÂN BẰNG: BẢO TOÀN SKEL / DENSITY / WCD_DOG & SẠCH NỀN XA
# ==============================================================================
class Phase3ExactDensityLoss(nn.Module):
    """
    Siết chặt sàn mật độ nét (dead_zone_low=0.99) và tăng phạt làm nhạt/mảnh nét (3.2*dev_low + 1.5*dev_low^2)
    để ngăn chặn hoàn toàn hiện tượng trôi mảnh nét khi fine-tune từ `model_tps_blended.pt`.
    """
    def __init__(self, tau=0.15, dead_zone_low=0.99, dead_zone_high=1.04):
        super().__init__()
        self.tau = tau
        self.low = dead_zone_low
        self.high = dead_zone_high

    def forward(self, pred, gt):
        s_p = torch.clamp((pred - 0.06) / 0.18, 0.0, 1.0) + 0.25 * F.relu(pred - 0.24)
        h_p = (pred >= self.tau).float()
        m_p = h_p - s_p.detach() + s_p

        h_g = (gt >= self.tau).float()
        area_p = torch.sum(m_p, dim=(-2, -1))
        area_g = torch.sum(h_g, dim=(-2, -1)).detach()
        r = area_p / torch.clamp(area_g, min=1.0)

        dev_low = F.relu(self.low - r)
        dev_high = F.relu(r - self.high)
        dev_center = r - 1.0
        loss = (
            3.2 * dev_low + 1.5 * (dev_low ** 2)
            + 2.8 * dev_high + 1.5 * (dev_high ** 2)
            + 0.12 * (dev_center ** 2)
        )
        return loss.mean()


class Phase3ZoneCleanlinessLoss(nn.Module):
    """
    Làm sạch có chọn lọc (Surgical Background Cleanliness):
      - Nền xa (> 2px khỏi nét GT): dọn sương mờ và đặc biệt phạt mạnh mực đậm sai vị trí (Dark Ghost >= 0.20).
      - Vành sát nét (near_halo): chỉ phạt mực >= 0.12 (nới từ 0.06 -> 0.12) để bảo toàn 100% tóc mảnh anti-aliased!
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        with torch.no_grad():
            gt_stroke_wide = soft_dilate(soft_dilate((gt > 0.04).float()))
            far_bg = torch.clamp(1.0 - gt_stroke_wide, 0.0, 1.0)
            near_halo = torch.clamp(gt_stroke_wide - (gt > 0.12).float(), 0.0, 1.0)
            gt_area = torch.clamp(torch.sum((gt > 0.15).float(), dim=(-3, -2, -1)), min=500.0)

        near_count = torch.clamp(torch.sum(near_halo, dim=(-3, -2, -1)), min=200.0)
        pen_far_p045 = F.relu(pred - 0.045)
        pen_far = (
            1.0 * F.relu(pred - 0.018)
            + 2.5 * pen_far_p045
            + 3.0 * (pen_far_p045 ** 2)
            + 9.0 * F.relu(pred - 0.20)
        )
        loss_far = torch.sum(far_bg * pen_far, dim=(-3, -2, -1)) / gt_area

        pen_halo = F.relu(pred - 0.12)
        loss_halo = 0.35 * torch.sum(near_halo * pen_halo, dim=(-3, -2, -1)) / near_count
        return (loss_far + loss_halo).mean()


class Phase3BalancedLineartLoss(AdaptiveLineartLoss):
    """
    Kế thừa `AdaptiveLineartLoss` nhưng thay thế các thành phần gây tẩy mảnh nét bằng cấu hình cân bằng Pha 3 v2:
      - w_dt = 0.40 (tối ưu WCD_DoG)
      - w_dens = 0.70 (giữ chặt Density Ratio ~ 0.98 - 1.02)
      - w_width = 0.35 (bảo vệ lõi skeleton)
      - w_cldice = 0.28 (bảo vệ ngọn tóc mảnh & Skeleton Recall)
      - w_ghost = 0.32 (vừa dọn Dark Ghost, không làm mờ nét)
    """
    def __init__(self, train_mode="LONG", is_rank0=True, w_ghost=0.32, w_temp=0.55):
        super().__init__(train_mode=train_mode, is_rank0=is_rank0, w_ghost=w_ghost, w_temp=w_temp)
        self.w_dt = 0.40
        self.w_dens = 0.70
        self.w_width = 0.35
        self.w_cldice = 0.28 if self.train_mode == "LONG" else self.w_cldice
        self.density_loss = Phase3ExactDensityLoss(tau=0.15, dead_zone_low=0.99, dead_zone_high=1.04)
        self.ghost_loss = Phase3ZoneCleanlinessLoss()
        self.stroke_width_loss = BidirectionalStrokeWidthLoss()


# ==============================================================================
# 1. LÀM GIÀU DỮ LIỆU ĐA LỚP TẠI CHỖ (ON-THE-FLY MULTI-LAYER ENRICHMENT)
# ==============================================================================
def _inject_static_background_on_sequence(x0, x1, xk_all, orig_h, orig_w, cache_entry):
    """
    Trộn lớp nền đường thẳng tĩnh mảnh/thưa (sparse static background lineart, 1.5-2.5px)
    vào chuỗi Sakuga thực tế để không làm loãng mật độ mực nhân vật mà vẫn rèn luyện
    xử lý ranh giới giao thoa che khuất (Occlusion Crossing).
    """
    if cv2 is None or cache_entry['num_clusters'] <= 0:
        return x0, x1, xk_all, cache_entry

    bg = np.zeros((orig_h, orig_w), dtype=np.float32)
    pts_st = []
    style = random.choice(["ground_horizon", "corner_frame", "single_pillar"])
    th = int(round(random.uniform(1.5, 2.5)))
    intensity = random.uniform(0.80, 1.0)

    if style == "ground_horizon":
        y_l = random.uniform(0.68 * orig_h, 0.88 * orig_h)
        x_a, x_b = random.uniform(0.08 * orig_w, 0.20 * orig_w), random.uniform(0.80 * orig_w, 0.92 * orig_w)
        cv2.line(bg, (int(x_a), int(y_l)), (int(x_b), int(y_l)), intensity, th, lineType=cv2.LINE_AA)
        for xv in np.linspace(x_a, x_b, 20):
            pts_st.append([xv / max(1.0, orig_w - 1), y_l / max(1.0, orig_h - 1)])
    elif style == "corner_frame":
        bx0 = random.uniform(0.10 * orig_w, 0.25 * orig_w)
        by0 = random.uniform(0.60 * orig_h, 0.78 * orig_h)
        bx1 = random.uniform(0.75 * orig_w, 0.90 * orig_w)
        by1 = random.uniform(0.14 * orig_h, 0.35 * orig_h)
        cv2.line(bg, (int(bx0), int(by0)), (int(bx1), int(by0)), intensity, th, lineType=cv2.LINE_AA)
        cv2.line(bg, (int(bx0), int(by0)), (int(bx0), int(by1)), intensity, th, lineType=cv2.LINE_AA)
        for xv in np.linspace(bx0, bx1, 12):
            pts_st.append([xv / max(1.0, orig_w - 1), by0 / max(1.0, orig_h - 1)])
        for yv in np.linspace(by1, by0, 10):
            pts_st.append([bx0 / max(1.0, orig_w - 1), yv / max(1.0, orig_h - 1)])
    else:
        xv = random.uniform(0.18 * orig_w, 0.82 * orig_w)
        ya, yb = random.uniform(0.12 * orig_h, 0.24 * orig_h), random.uniform(0.76 * orig_h, 0.88 * orig_h)
        cv2.line(bg, (int(xv), int(ya)), (int(xv), int(yb)), intensity, th, lineType=cv2.LINE_AA)
        for yv in np.linspace(ya, yb, 16):
            pts_st.append([xv / max(1.0, orig_w - 1), yv / max(1.0, orig_h - 1)])

    x0_aug = np.maximum(x0, bg)
    x1_aug = np.maximum(x1, bg)
    xk_aug = np.maximum(xk_all, bg[None, :, :])

    pts_st_np = np.asarray(pts_st, dtype=np.float32)
    max_s = cache_entry['static_pts0'].shape[0]
    cur_ns = int(cache_entry['num_static'])
    sp0 = cache_entry['static_pts0'].copy()
    sp1 = cache_entry['static_pts1'].copy()
    avail = max(0, max_s - cur_ns)
    n_add = min(avail, len(pts_st_np))
    if n_add > 0:
        sp0[cur_ns:cur_ns + n_add] = pts_st_np[:n_add]
        sp1[cur_ns:cur_ns + n_add] = pts_st_np[:n_add]
        cache_entry = dict(cache_entry)
        cache_entry['static_pts0'] = sp0
        cache_entry['static_pts1'] = sp1
        cache_entry['num_static'] = cur_ns + n_add

    return x0_aug, x1_aug, xk_aug, cache_entry


def _generate_multilayer_synthetic_sequence(seq_len, orig_h=540, orig_w=960, max_pts=128, max_clusters=4, lambd=0.01):
    """
    Sinh chuỗi hình học tổng hợp 540p sử dụng 100% Multi-Layer Clustered TPS (Direct-t) với 6 kịch bản:
      - Mode 1 (25%): Hình tròn/elip thu phóng & dịch xa
      - Mode 2 (25%): Hình tròn/elip bay + đường nền / khung phòng tĩnh (Case 3)
      - Mode 3 (15%): Cung cong mở xoay & biến dạng
      - Mode 4 (15%): Hai hình tròn chuyển động song song độc lập
      - Mode 5 (10%): Occlusion Crossing — vật thể chuyển động cắt ngang trực diện qua đường tĩnh
      - Mode 6 (10%): Multi-Cluster Crossing — hai vật thể chuyển động ngược/chéo chiều nhau
    """
    def _ellipse_pts(cx, cy, rx, ry, rot_rad, n_pts=32):
        ang = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
        lx, ly = rx * np.cos(ang), ry * np.sin(ang)
        ca, sa = math.cos(rot_rad), math.sin(rot_rad)
        px = cx + lx * ca - ly * sa
        py = cy + lx * sa + ly * ca
        return np.stack([px / float(orig_w - 1), py / float(orig_h - 1)], axis=1).astype(np.float32)

    mode_roll = random.random()
    frames = []
    kps_per_t = []
    t_list = [float(k) / float(seq_len - 1) for k in range(seq_len)]

    if mode_roll < 0.50:
        # Mode 1 (25%) & Mode 2 (25%): Circle/oval + optional static ground/box
        has_static = (mode_roll >= 0.25)
        y_max_c = 340 if has_static else 420
        rx0 = random.uniform(26.0, 92.0)
        ry0 = rx0 * random.uniform(0.75, 1.25)
        scale_r = random.uniform(0.42, 2.20)
        rx1 = float(np.clip(rx0 * scale_r, 18.0, 96.0))
        ry1 = float(np.clip(ry0 * scale_r * random.uniform(0.85, 1.15), 18.0, 96.0))
        cx0 = random.uniform(110.0, 420.0)
        cx1 = float(np.clip(cx0 + random.choice([-1.0, 1.0]) * random.uniform(80.0, 420.0), 100.0, orig_w - 100.0))
        if random.random() < 0.5:
            cx0, cx1 = cx1, cx0
        cy0 = random.uniform(115.0, y_max_c - 60.0)
        cy1 = float(np.clip(cy0 + random.uniform(-90.0, 90.0), 105.0, y_max_c - 40.0))
        rot0 = random.uniform(-0.35, 0.35)
        rot1 = rot0 + random.uniform(-0.35, 0.35)
        th0, th1 = random.uniform(2.0, 5.0), random.uniform(2.0, 5.0)

        st_type = random.choice(["line", "rect"])
        st_y = random.uniform(410.0, 490.0)
        st_x0, st_x1 = random.uniform(80.0, 200.0), random.uniform(orig_w - 200.0, orig_w - 80.0)
        st_th = int(round(random.uniform(2.5, 4.5)))
        if st_type == "line":
            xs_st = np.linspace(st_x0, st_x1, 24)
            pts_st = np.stack([xs_st / float(orig_w - 1), np.full_like(xs_st, st_y) / float(orig_h - 1)], axis=1).astype(np.float32)
        else:
            bx0, by0 = random.uniform(120.0, 740.0), st_y - 45.0
            bw, bh = random.uniform(70.0, 140.0), random.uniform(40.0, 65.0)
            pts_st = np.vstack([
                np.stack([np.linspace(bx0, bx0 + bw, 7)[:-1], np.full(6, by0)], axis=1),
                np.stack([np.full(6, bx0 + bw), np.linspace(by0, by0 + bh, 7)[:-1]], axis=1),
                np.stack([np.linspace(bx0 + bw, bx0, 7)[:-1], np.full(6, by0 + bh)], axis=1),
                np.stack([np.full(6, bx0), np.linspace(by0 + bh, by0, 7)[:-1]], axis=1),
            ])
            pts_st[:, 0] /= float(orig_w - 1)
            pts_st[:, 1] /= float(orig_h - 1)
            pts_st = pts_st.astype(np.float32)

        for t in t_list:
            im = np.zeros((orig_h, orig_w), dtype=np.float32)
            cx, cy = (1.0 - t) * cx0 + t * cx1, (1.0 - t) * cy0 + t * cy1
            rx, ry = (1.0 - t) * rx0 + t * rx1, (1.0 - t) * ry0 + t * ry1
            rot = (1.0 - t) * rot0 + t * rot1
            th = max(2, int(round((1.0 - t) * th0 + t * th1)))
            if cv2 is not None:
                cv2.ellipse(im, (int(round(cx)), int(round(cy))), (int(round(rx)), int(round(ry))), math.degrees(rot), 0, 360, 1.0, th, lineType=cv2.LINE_AA)
                if has_static:
                    if st_type == "line":
                        cv2.line(im, (int(st_x0), int(st_y)), (int(st_x1), int(st_y)), 1.0, st_th, lineType=cv2.LINE_AA)
                    else:
                        cv2.rectangle(im, (int(bx0), int(by0)), (int(bx0 + bw), int(by0 + bh)), 1.0, st_th, lineType=cv2.LINE_AA)
            pts_mv = _ellipse_pts(cx, cy, rx, ry, rot, n_pts=32)
            frames.append(im)
            kps_per_t.append(np.vstack([pts_mv, pts_st]) if has_static else pts_mv)
    elif mode_roll < 0.65:
        # Mode 3 (15%): Smooth curved arc
        n_ctrl = 24
        s_u = np.linspace(-1.0, 1.0, n_ctrl)
        base_len = random.uniform(60.0, 140.0)
        bend0, bend1 = random.uniform(-55.0, 55.0), random.uniform(-55.0, 55.0)
        cx0, cy0 = random.uniform(160.0, 440.0), random.uniform(140.0, 400.0)
        cx1 = float(np.clip(cx0 + random.choice([-1.0, 1.0]) * random.uniform(60.0, 320.0), 140.0, orig_w - 140.0))
        cy1 = float(np.clip(cy0 + random.uniform(-100.0, 100.0), 120.0, orig_h - 120.0))
        sc0, sc1 = random.uniform(0.75, 1.25), random.uniform(0.65, 1.45)
        ang0 = random.uniform(-0.7, 0.7)
        ang1 = ang0 + random.uniform(-0.6, 0.6)
        th = int(round(random.uniform(2.0, 4.5)))

        def _arc_px(t):
            cx, cy = (1.0 - t) * cx0 + t * cx1, (1.0 - t) * cy0 + t * cy1
            sc, ang, bend = (1.0 - t) * sc0 + t * sc1, (1.0 - t) * ang0 + t * ang1, (1.0 - t) * bend0 + t * bend1
            lx, ly = s_u * base_len * sc, bend * sc * (1.0 - s_u ** 2)
            ca, sa = math.cos(ang), math.sin(ang)
            return np.stack([cx + lx * ca - ly * sa, cy + lx * sa + ly * ca], axis=1)

        for t in t_list:
            im = np.zeros((orig_h, orig_w), dtype=np.float32)
            p_px = _arc_px(t)
            if cv2 is not None:
                cv2.polylines(im, [np.round(p_px).astype(np.int32)], False, 1.0, th, lineType=cv2.LINE_AA)
            frames.append(im)
            kps_per_t.append(np.stack([p_px[:, 0] / float(orig_w - 1), p_px[:, 1] / float(orig_h - 1)], axis=1).astype(np.float32))
    elif mode_roll < 0.80:
        # Mode 4 (15%): Two independent moving circles/ovals
        lanes = [(145.0, 40.0), (385.0, 40.0)]
        params = []
        for cy_base, cy_jit in lanes:
            rx0 = random.uniform(24.0, 62.0)
            ry0 = rx0 * random.uniform(0.8, 1.2)
            sc = random.uniform(0.55, 1.80)
            rx1, ry1 = float(np.clip(rx0 * sc, 18.0, 72.0)), float(np.clip(ry0 * sc, 18.0, 72.0))
            cx0 = random.uniform(120.0, 450.0)
            cx1 = float(np.clip(cx0 + random.choice([-1.0, 1.0]) * random.uniform(80.0, 350.0), 100.0, orig_w - 100.0))
            params.append((cx0, cy_base + random.uniform(-cy_jit, cy_jit), rx0, ry0, cx1, cy_base + random.uniform(-cy_jit, cy_jit), rx1, ry1))
        th = int(round(random.uniform(2.5, 4.5)))
        for t in t_list:
            im = np.zeros((orig_h, orig_w), dtype=np.float32)
            pts_two = []
            for (cx0, cy0, rx0, ry0, cx1, cy1, rx1, ry1) in params:
                cx, cy = (1.0 - t) * cx0 + t * cx1, (1.0 - t) * cy0 + t * cy1
                rx, ry = (1.0 - t) * rx0 + t * rx1, (1.0 - t) * ry0 + t * ry1
                if cv2 is not None:
                    cv2.ellipse(im, (int(round(cx)), int(round(cy))), (int(round(rx)), int(round(ry))), 0, 0, 360, 1.0, th, lineType=cv2.LINE_AA)
                pts_two.append(_ellipse_pts(cx, cy, rx, ry, 0.0, n_pts=24))
            frames.append(im)
            kps_per_t.append(np.vstack(pts_two))
    elif mode_roll < 0.90:
        # Mode 5 (10%): Occlusion Crossing — hình tròn/elip chuyển động cắt qua đường kẻ tĩnh
        rx0 = random.uniform(35.0, 70.0)
        ry0 = rx0 * random.uniform(0.85, 1.15)
        rx1, ry1 = rx0 * random.uniform(0.65, 1.35), ry0 * random.uniform(0.65, 1.35)
        cx0, cx1 = random.uniform(140.0, 260.0), random.uniform(680.0, 820.0)
        if random.random() < 0.5:
            cx0, cx1 = cx1, cx0
        st_y = random.uniform(210.0, 330.0)
        cy0 = st_y + random.choice([-1.0, 1.0]) * random.uniform(60.0, 120.0)
        cy1 = st_y - (cy0 - st_y) * random.uniform(0.6, 1.1)
        xs_st = np.linspace(100.0, orig_w - 100.0, 28)
        pts_st = np.stack([xs_st / float(orig_w - 1), np.full_like(xs_st, st_y) / float(orig_h - 1)], axis=1).astype(np.float32)
        th = int(round(random.uniform(2.5, 4.0)))
        for t in t_list:
            im = np.zeros((orig_h, orig_w), dtype=np.float32)
            cx, cy = (1.0 - t) * cx0 + t * cx1, (1.0 - t) * cy0 + t * cy1
            rx, ry = (1.0 - t) * rx0 + t * rx1, (1.0 - t) * ry0 + t * ry1
            if cv2 is not None:
                cv2.line(im, (100, int(round(st_y))), (orig_w - 100, int(round(st_y))), 1.0, th, lineType=cv2.LINE_AA)
                cv2.ellipse(im, (int(round(cx)), int(round(cy))), (int(round(rx)), int(round(ry))), 0, 0, 360, 1.0, th, lineType=cv2.LINE_AA)
            frames.append(im)
            kps_per_t.append(np.vstack([_ellipse_pts(cx, cy, rx, ry, 0.0, n_pts=32), pts_st]))
    else:
        # Mode 6 (10%): Multi-Cluster Crossing — hai hình tròn chạy ngược/chéo chiều
        rx_a, ry_a = random.uniform(30.0, 55.0), random.uniform(30.0, 55.0)
        rx_b, ry_b = random.uniform(28.0, 50.0), random.uniform(28.0, 50.0)
        p_a = (150.0, random.uniform(130.0, 220.0), 800.0, random.uniform(310.0, 410.0))
        p_b = (800.0, random.uniform(130.0, 220.0), 150.0, random.uniform(310.0, 410.0))
        th = int(round(random.uniform(2.5, 4.0)))
        for t in t_list:
            im = np.zeros((orig_h, orig_w), dtype=np.float32)
            cxa, cya = (1.0 - t) * p_a[0] + t * p_a[2], (1.0 - t) * p_a[1] + t * p_a[3]
            cxb, cyb = (1.0 - t) * p_b[0] + t * p_b[2], (1.0 - t) * p_b[1] + t * p_b[3]
            if cv2 is not None:
                cv2.ellipse(im, (int(round(cxa)), int(round(cya))), (int(round(rx_a)), int(round(ry_a))), 0, 0, 360, 1.0, th, lineType=cv2.LINE_AA)
                cv2.ellipse(im, (int(round(cxb)), int(round(cyb))), (int(round(rx_b)), int(round(ry_b))), 0, 0, 360, 1.0, th, lineType=cv2.LINE_AA)
            pts_ab = np.vstack([
                _ellipse_pts(cxa, cya, rx_a, ry_a, 0.0, n_pts=24),
                _ellipse_pts(cxb, cyb, rx_b, ry_b, 0.0, n_pts=24),
            ])
            frames.append(im)
            kps_per_t.append(pts_ab)

    x0, x1 = frames[0], frames[-1]
    xk_all = np.stack(frames[1:-1], axis=0)
    kps0, kps1 = kps_per_t[0], kps_per_t[-1]
    t_mid = np.array(t_list[1:-1], dtype=np.float32)

    # Giải Multi-Layer TPS Cache Entry (Direct-t) cho từng khung hình trung gian m_idx
    m_steps = seq_len - 2
    max_mask_pts, max_static_pts = 64, 96
    c0_m = np.zeros((m_steps, max_clusters, max_pts, 2), dtype=np.float32)
    th01_m = np.zeros((m_steps, max_clusters, max_pts + 3, 2), dtype=np.float32)
    c1_m = np.zeros((m_steps, max_clusters, max_pts, 2), dtype=np.float32)
    th10_m = np.zeros((m_steps, max_clusters, max_pts + 3, 2), dtype=np.float32)
    val_m = np.zeros((m_steps, max_clusters), dtype=np.float32)
    dpts0_m = np.zeros((m_steps, max_clusters, max_mask_pts, 2), dtype=np.float32)
    dpts1_m = np.zeros((m_steps, max_clusters, max_mask_pts, 2), dtype=np.float32)
    dlen_m = np.zeros((m_steps, max_clusters), dtype=np.int32)
    spts0_m = np.zeros((m_steps, max_static_pts, 2), dtype=np.float32)
    spts1_m = np.zeros((m_steps, max_static_pts, 2), dtype=np.float32)
    nstat_m = np.zeros((m_steps,), dtype=np.int32)
    nclus_m = np.zeros((m_steps,), dtype=np.int32)

    from seq_p3_multi_layer_tps import segment_motion_keypoints, solve_tps_dynamic_layer, _stratified_spatial_subsample
    seg = segment_motion_keypoints(kps0, kps1, img_hw=(orig_h, orig_w), eps_static_px=1.5, filter_outliers=False)
    clusters = seg['dynamic_clusters'][:max_clusters]
    if seg['num_static'] > 0:
        if seg['num_static'] > max_static_pts:
            s0_sub, s1_sub = _stratified_spatial_subsample(seg['static_kps0'], seg['static_kps1'], max_static_pts)
        else:
            s0_sub, s1_sub = seg['static_kps0'], seg['static_kps1']
        n_stat = len(s0_sub)
    else:
        s0_sub = s1_sub = np.zeros((0, 2), dtype=np.float32)
        n_stat = 0

    for m_idx in range(m_steps):
        tv = float(t_mid[m_idx])
        if n_stat > 0:
            spts0_m[m_idx, :n_stat] = s0_sub.astype(np.float32)
            spts1_m[m_idx, :n_stat] = s1_sub.astype(np.float32)
        nstat_m[m_idx] = n_stat
        nclus_m[m_idx] = len(clusters)

        for k, dc in enumerate(clusters):
            p0_k, p1_k = dc['kps0'], dc['kps1']
            pt_k = (1.0 - tv) * p0_k + tv * p1_k
            ct0, tht0, v0 = solve_tps_dynamic_layer(pt_k, p0_k, max_pts=max_pts, lambd=lambd, use_anchors=False)
            ct1, tht1, v1 = solve_tps_dynamic_layer(pt_k, p1_k, max_pts=max_pts, lambd=lambd, use_anchors=False)
            c0_m[m_idx, k], th01_m[m_idx, k] = ct0, tht0
            c1_m[m_idx, k], th10_m[m_idx, k] = ct1, tht1
            val_m[m_idx, k] = 1.0 if (v0 > 0.5 and v1 > 0.5) else 0.0

            # Đưa pt_k vào dyn_pts0/dyn_pts1 để mặt nạ Voronoi đặt tâm tại đúng quỹ đạo t
            if len(pt_k) > max_mask_pts:
                pt_sub, _ = _stratified_spatial_subsample(pt_k, pt_k, max_mask_pts)
            else:
                pt_sub = pt_k
            n_mk = len(pt_sub)
            dpts0_m[m_idx, k, :n_mk] = pt_sub.astype(np.float32)
            dpts1_m[m_idx, k, :n_mk] = pt_sub.astype(np.float32)
            dlen_m[m_idx, k] = n_mk

    return {
        'is_sequence': True,
        'is_direct_t_flows': True,
        'is_multilayer_cache': True,
        'x0': torch.from_numpy(x0).unsqueeze(0),
        'x1': torch.from_numpy(x1).unsqueeze(0),
        'xk_all': torch.from_numpy(xk_all).unsqueeze(1),
        't_vals': torch.from_numpy(t_mid),
        'c0_dyn_all': torch.from_numpy(c0_m),
        'th01_dyn_all': torch.from_numpy(th01_m),
        'c1_dyn_all': torch.from_numpy(c1_m),
        'th10_dyn_all': torch.from_numpy(th10_m),
        'valid_dyn_all': torch.from_numpy(val_m),
        'dyn_pts0_all': torch.from_numpy(dpts0_m),
        'dyn_pts1_all': torch.from_numpy(dpts1_m),
        'dyn_pts_len_all': torch.from_numpy(dlen_m),
        'static_pts0_all': torch.from_numpy(spts0_m),
        'static_pts1_all': torch.from_numpy(spts1_m),
        'num_static_all': torch.from_numpy(nstat_m),
        'num_clusters_all': torch.from_numpy(nclus_m),
        'orig_hw': torch.tensor([orig_h, orig_w], dtype=torch.int32),
        'seq_len': seq_len,
        'clip_id': f"synth_ml_N{seq_len}",
        'scene_id': "synth_geom_ml",
        'mean_disp': 120.0,
    }


# ==============================================================================
# 2. DATASET ĐA LỚP (ĐỌC TPS_CACHE_MULTILAYER.NPZ + STATIC BG AUGMENTATION)
# ==============================================================================
class SakugaMultiLayerDataset(Dataset):
    def __init__(self, metadata_csv, clips_dir, tps_cache_path, train_mode="LONG", is_train=True, seed=42, synth_count=0, static_aug_prob=0.25):
        super().__init__()
        self.clips_dir = clips_dir
        self.train_mode = train_mode.upper()
        self.is_train = is_train
        self.static_aug_prob = float(static_aug_prob) if is_train else 0.0
        self.df = pd.read_csv(metadata_csv).reset_index(drop=True)

        self.tps_cache_path = tps_cache_path
        with np.load(tps_cache_path) as cache_data:
            self.is_multilayer = ('c0_dyn' in cache_data.files)
            self.orig_hw_arr = np.array(cache_data['orig_hw'])
            if self.is_multilayer:
                self.c0_dyn = np.array(cache_data['c0_dyn'])
                self.th01_dyn = np.array(cache_data['th01_dyn'])
                self.c1_dyn = np.array(cache_data['c1_dyn'])
                self.th10_dyn = np.array(cache_data['th10_dyn'])
                self.valid_dyn = np.array(cache_data['valid_dyn'])
                self.dyn_pts0 = np.array(cache_data['dyn_pts0'])
                self.dyn_pts1 = np.array(cache_data['dyn_pts1'])
                self.dyn_pts_len = np.array(cache_data['dyn_pts_len'])
                self.static_pts0 = np.array(cache_data['static_pts0'])
                self.static_pts1 = np.array(cache_data['static_pts1'])
                self.num_static = np.array(cache_data['num_static'])
                self.num_clusters = np.array(cache_data['num_clusters'])
            else:
                self.c0_arr = np.array(cache_data['c0'])
                self.th01_arr = np.array(cache_data['th01'])
                self.c1_arr = np.array(cache_data['c1'])
                self.th10_arr = np.array(cache_data['th10'])
                self.valid_arr = np.array(cache_data['valid'])

        self.samples = []
        if self.is_train and self.train_mode == "LONG":
            for idx, row in self.df.iterrows():
                cid = str(row['clip_id'])
                seq_len = int(row['seq_len'])
                cache_idx = int(row['cache_idx']) if 'cache_idx' in row else idx
                hw = self.orig_hw_arr[cache_idx]
                t_list = [float(k) / float(seq_len - 1) for k in range(1, seq_len - 1)]
                self.samples.append({
                    'is_synth': False,
                    'clip_id': cid,
                    'scene_id': str(row.get('scene_id', '')),
                    'seq_len': seq_len,
                    'cache_idx': cache_idx,
                    'orig_h': int(hw[0]),
                    'orig_w': int(hw[1]),
                    't_list': t_list,
                    'mean_disp': float(row.get('mean_disp', 0.0)),
                })
            n_synth = int(synth_count) if (cv2 is not None and synth_count > 0) else 0
            for s_i in range(n_synth):
                sl = (3, 5, 7)[s_i % 3]
                self.samples.append({
                    'is_synth': True,
                    'clip_id': f"synth_ml_{s_i:04d}_N{sl}",
                    'scene_id': "synth_geom_ml",
                    'seq_len': sl,
                    'cache_idx': -1,
                    'orig_h': 540,
                    'orig_w': 960,
                    't_list': [float(k) / float(sl - 1) for k in range(1, sl - 1)],
                    'mean_disp': 120.0,
                })
        else:
            for idx, row in self.df.iterrows():
                cid = str(row['clip_id'])
                seq_len = int(row['seq_len'])
                cache_idx = int(row['cache_idx']) if 'cache_idx' in row else idx
                hw = self.orig_hw_arr[cache_idx]
                for k in range(1, seq_len - 1):
                    self.samples.append({
                        'is_synth': False,
                        'clip_id': cid,
                        'scene_id': str(row.get('scene_id', '')),
                        'seq_len': seq_len,
                        'cache_idx': cache_idx,
                        'orig_h': int(hw[0]),
                        'orig_w': int(hw[1]),
                        'frame_k': k + 1,
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
        if s.get('is_synth', False):
            return _generate_multilayer_synthetic_sequence(s['seq_len'], orig_h=s['orig_h'], orig_w=s['orig_w'])

        cid = s['clip_id']
        seq_len = s['seq_len']
        cache_idx = s['cache_idx']

        x0 = self._read_ink(cid, 1)
        x1 = self._read_ink(cid, seq_len)
        orig_h, orig_w = x0.shape

        if self.is_multilayer:
            c_entry = {
                'c0_dyn': np.array(self.c0_dyn[cache_idx], copy=True),
                'th01_dyn': np.array(self.th01_dyn[cache_idx], copy=True),
                'c1_dyn': np.array(self.c1_dyn[cache_idx], copy=True),
                'th10_dyn': np.array(self.th10_dyn[cache_idx], copy=True),
                'valid_dyn': np.array(self.valid_dyn[cache_idx], copy=True),
                'dyn_pts0': np.array(self.dyn_pts0[cache_idx], copy=True),
                'dyn_pts1': np.array(self.dyn_pts1[cache_idx], copy=True),
                'dyn_pts_len': np.array(self.dyn_pts_len[cache_idx], copy=True),
                'static_pts0': np.array(self.static_pts0[cache_idx], copy=True),
                'static_pts1': np.array(self.static_pts1[cache_idx], copy=True),
                'num_static': int(self.num_static[cache_idx]),
                'num_clusters': int(self.num_clusters[cache_idx]),
            }

            if self.is_train and self.train_mode == "LONG":
                xk_list = [self._read_ink(cid, f_idx) for f_idx in range(2, seq_len)]
                xk_all = np.stack(xk_list, axis=0)
                t_arr = np.array(s['t_list'], dtype=np.float32)

                # Làm giàu có chủ đích: trộn nền tĩnh nhân tạo vào chuỗi Sakuga thực tế
                if self.static_aug_prob > 0.0 and random.random() < self.static_aug_prob:
                    x0, x1, xk_all, c_entry = _inject_static_background_on_sequence(
                        x0, x1, xk_all, orig_h, orig_w, c_entry
                    )

                # Temporal Reversal (đảo chiều I0 <-> I1)
                if random.random() < 0.5:
                    x0, x1 = x1, x0
                    c_entry['c0_dyn'], c_entry['c1_dyn'] = c_entry['c1_dyn'], c_entry['c0_dyn']
                    c_entry['th01_dyn'], c_entry['th10_dyn'] = c_entry['th10_dyn'], c_entry['th01_dyn']
                    c_entry['dyn_pts0'], c_entry['dyn_pts1'] = c_entry['dyn_pts1'], c_entry['dyn_pts0']
                    c_entry['static_pts0'], c_entry['static_pts1'] = c_entry['static_pts1'], c_entry['static_pts0']
                    xk_all = xk_all[::-1].copy()
                    t_arr = (1.0 - t_arr[::-1]).copy()

                return {
                    'is_sequence': True,
                    'is_direct_t_flows': False,
                    'is_multilayer_cache': True,
                    'x0': torch.from_numpy(x0).unsqueeze(0),
                    'x1': torch.from_numpy(x1).unsqueeze(0),
                    'xk_all': torch.from_numpy(xk_all).unsqueeze(1),
                    't_vals': torch.from_numpy(t_arr),
                    'c0_dyn': torch.from_numpy(c_entry['c0_dyn']),
                    'th01_dyn': torch.from_numpy(c_entry['th01_dyn']),
                    'c1_dyn': torch.from_numpy(c_entry['c1_dyn']),
                    'th10_dyn': torch.from_numpy(c_entry['th10_dyn']),
                    'valid_dyn': torch.from_numpy(c_entry['valid_dyn']),
                    'dyn_pts0': torch.from_numpy(c_entry['dyn_pts0']),
                    'dyn_pts1': torch.from_numpy(c_entry['dyn_pts1']),
                    'dyn_pts_len': torch.from_numpy(c_entry['dyn_pts_len']),
                    'static_pts0': torch.from_numpy(c_entry['static_pts0']),
                    'static_pts1': torch.from_numpy(c_entry['static_pts1']),
                    'num_static': torch.tensor(c_entry['num_static'], dtype=torch.int32),
                    'num_clusters': torch.tensor(c_entry['num_clusters'], dtype=torch.int32),
                    'orig_hw': torch.tensor([orig_h, orig_w], dtype=torch.int32),
                    'seq_len': seq_len,
                    'clip_id': cid,
                    'scene_id': s['scene_id'],
                    'mean_disp': s['mean_disp'],
                }

            fk = s['frame_k']
            t_val = s['t_val']
            xk = self._read_ink(cid, fk)
            return {
                'is_sequence': False,
                'is_direct_t_flows': False,
                'is_multilayer_cache': True,
                'x0': torch.from_numpy(x0).unsqueeze(0),
                'x1': torch.from_numpy(x1).unsqueeze(0),
                'xk': torch.from_numpy(xk).unsqueeze(0),
                't_val': torch.tensor([t_val], dtype=torch.float32),
                'c0_dyn': torch.from_numpy(c_entry['c0_dyn']),
                'th01_dyn': torch.from_numpy(c_entry['th01_dyn']),
                'c1_dyn': torch.from_numpy(c_entry['c1_dyn']),
                'th10_dyn': torch.from_numpy(c_entry['th10_dyn']),
                'valid_dyn': torch.from_numpy(c_entry['valid_dyn']),
                'dyn_pts0': torch.from_numpy(c_entry['dyn_pts0']),
                'dyn_pts1': torch.from_numpy(c_entry['dyn_pts1']),
                'dyn_pts_len': torch.from_numpy(c_entry['dyn_pts_len']),
                'static_pts0': torch.from_numpy(c_entry['static_pts0']),
                'static_pts1': torch.from_numpy(c_entry['static_pts1']),
                'num_static': torch.tensor(c_entry['num_static'], dtype=torch.int32),
                'num_clusters': torch.tensor(c_entry['num_clusters'], dtype=torch.int32),
                'orig_hw': torch.tensor([orig_h, orig_w], dtype=torch.int32),
                'seq_len': seq_len,
                'frame_k': fk,
                'clip_id': cid,
                'scene_id': s['scene_id'],
                'mean_disp': s['mean_disp'],
            }

        raise RuntimeError("Dataset Pha 3 yêu cầu bộ cache đa lớp `tps_cache_multilayer.npz`!")


def pad_collate_fn_multilayer(batch):
    max_h = max(item['orig_hw'][0].item() for item in batch)
    max_w = max(item['orig_hw'][1].item() for item in batch)
    h_pad = ((max_h + 31) // 32) * 32
    w_pad = ((max_w + 31) // 32) * 32

    is_seq = batch[0]['is_sequence']
    is_direct = bool(batch[0].get('is_direct_t_flows', False))

    def _pad_t(t, th, tw):
        dh = th - t.shape[-2]
        dw = tw - t.shape[-1]
        if dh == 0 and dw == 0:
            return t
        return F.pad(t, (0, dw, 0, dh), mode='constant', value=0.0)

    out = {
        'is_sequence': is_seq,
        'is_direct_t_flows': is_direct,
        'is_multilayer_cache': True,
        'h_pad': h_pad,
        'w_pad': w_pad,
        'x0': torch.stack([_pad_t(b['x0'], h_pad, w_pad) for b in batch], dim=0),
        'x1': torch.stack([_pad_t(b['x1'], h_pad, w_pad) for b in batch], dim=0),
        'orig_hw': torch.stack([b['orig_hw'] for b in batch], dim=0),
        'seq_len': torch.tensor([b['seq_len'] for b in batch], dtype=torch.int32),
        'clip_id': [b['clip_id'] for b in batch],
        'scene_id': [b['scene_id'] for b in batch],
        'mean_disp': [b['mean_disp'] for b in batch],
    }

    if is_direct:
        for k in ['c0_dyn_all', 'th01_dyn_all', 'c1_dyn_all', 'th10_dyn_all', 'valid_dyn_all',
                  'dyn_pts0_all', 'dyn_pts1_all', 'dyn_pts_len_all',
                  'static_pts0_all', 'static_pts1_all', 'num_static_all', 'num_clusters_all']:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    else:
        for k in ['c0_dyn', 'th01_dyn', 'c1_dyn', 'th10_dyn', 'valid_dyn',
                  'dyn_pts0', 'dyn_pts1', 'dyn_pts_len',
                  'static_pts0', 'static_pts1', 'num_static', 'num_clusters']:
            out[k] = torch.stack([b[k] for b in batch], dim=0)

    if is_seq:
        out['xk_all'] = torch.stack([_pad_t(b['xk_all'], h_pad, w_pad) for b in batch], dim=0)
        out['t_vals'] = torch.stack([b['t_vals'] for b in batch], dim=0)
    else:
        out['xk'] = torch.stack([_pad_t(b['xk'], h_pad, w_pad) for b in batch], dim=0)
        out['t_val'] = torch.stack([b['t_val'] for b in batch], dim=0)
        out['frame_k'] = [b['frame_k'] for b in batch]
    return out


# ==============================================================================
# 3. VALIDATION ĐA LỚP (THUẦN FP32)
# ==============================================================================
@torch.no_grad()
def run_validation_suite_multilayer(model, val_loader, device, downscale_factor=2, is_ddp=False, evaluate_m0=False):
    model.eval()
    keys_sum = ['psnr', 'ssim', 'val_l1', 'ie_mae', 'pwcr', 'rgd', 'ghost', 'dark_ghost',
                'wcd', 'wcd_legacy', 'wcd_dog', 'wcd_dog_legacy', 'semd',
                'recall', 'precision', 'f1', 'density_ratio', 'skel_recall', 'tip_recall']
    acc_main = {k: 0.0 for k in keys_sum}
    acc_main['count'] = 0.0
    acc_m0 = {k: 0.0 for k in keys_sum}
    acc_m0['count'] = 0.0

    brk_keys = ['count', 'psnr', 'ssim', 'wcd', 'wcd_legacy', 'wcd_dog', 'wcd_dog_legacy',
                'semd', 'skel_recall', 'tip_recall', 'f1', 'ghost', 'density_ratio']
    breakdown = {n: {k: 0.0 for k in brk_keys} for n in (3, 5, 7)}

    for batch in val_loader:
        h_pad, w_pad = batch['h_pad'], batch['w_pad']
        x0 = batch['x0'].to(device, non_blocking=True)
        x1 = batch['x1'].to(device, non_blocking=True)
        xk = batch['xk'].to(device, non_blocking=True)
        t_val = batch['t_val'].to(device, non_blocking=True).view(-1, 1, 1, 1)
        orig_hw = batch['orig_hw']
        seq_lens = batch['seq_len']

        cf01, cf10 = reconstruct_multilayer_flow_batch_from_cache(
            batch['c0_dyn'].to(device, non_blocking=True),
            batch['th01_dyn'].to(device, non_blocking=True),
            batch['c1_dyn'].to(device, non_blocking=True),
            batch['th10_dyn'].to(device, non_blocking=True),
            batch['valid_dyn'].to(device, non_blocking=True),
            batch['dyn_pts0'].to(device, non_blocking=True),
            batch['dyn_pts1'].to(device, non_blocking=True),
            batch['dyn_pts_len'].to(device, non_blocking=True),
            batch['static_pts0'].to(device, non_blocking=True),
            batch['static_pts1'].to(device, non_blocking=True),
            batch['num_static'].to(device, non_blocking=True),
            batch['num_clusters'].to(device, non_blocking=True),
            orig_hw, h_pad, w_pad, device, downscale_factor=downscale_factor,
        )

        with torch.amp.autocast('cuda', enabled=False):
            pred_it, coarse_warp, *_ = model(x0.float(), x1.float(), t_val.float(), cf01.float(), cf10.float())

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
# 4. CHƯƠNG TRÌNH HUẤN LUYỆN CHÍNH PHA 3 (DDP WORKER)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="DDP Worker Pha 3: Fine-tune nhẹ trên Coarse TPS Đa lớp")
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--train_clips_dir', type=str, required=True)
    parser.add_argument('--val_clips_dir', type=str, required=True)
    parser.add_argument('--tps_cache', type=str, required=True, help="Đường dẫn tới tps_cache_multilayer.npz")
    parser.add_argument('--m1_checkpoint', type=str, required=True)
    parser.add_argument('--init_checkpoint', type=str, default="", help="Ưu tiên model_tps_blended.pt")
    parser.add_argument('--output_dir', type=str, default="/kaggle/working/checkpoints_tps_finetune")
    parser.add_argument('--train_mode', type=str, default="LONG", choices=["LONG"])
    parser.add_argument('--epochs', type=int, default=4, help="Số epoch nhẹ cho Pha 3 v2 (mặc định: 4)")
    parser.add_argument('--grad_accum', type=int, default=2)
    parser.add_argument('--max_chunk_frames', type=int, default=3)
    parser.add_argument('--lr_backbone', type=float, default=6.0e-6)
    parser.add_argument('--lr_unet', type=float, default=1.8e-5)
    parser.add_argument('--ema_decay', type=float, default=0.996)
    parser.add_argument('--synth_count', type=int, default=150)
    parser.add_argument('--static_aug_prob', type=float, default=0.12)
    parser.add_argument('--w_ghost', type=float, default=0.32)
    parser.add_argument('--w_temp', type=float, default=0.55)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--ram_limit_gb', type=float, default=23.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp = world_size > 1

    if is_ddp:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
            backend = 'nccl' if dist.is_nccl_available() else 'gloo'
        else:
            device = torch.device('cpu')
            backend = 'gloo'
        dist.init_process_group(backend=backend, init_method=os.environ.get('DDP_INIT_METHOD', 'env://'))
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    is_rank0 = (local_rank == 0)
    random.seed(args.seed + local_rank)
    np.random.seed(args.seed + local_rank)
    torch.manual_seed(args.seed + local_rank)

    tps_downscale = 2
    if is_rank0:
        init_tag = os.path.basename(args.init_checkpoint) if (args.init_checkpoint and os.path.exists(args.init_checkpoint)) else os.path.basename(args.m1_checkpoint)
        print("=" * 84)
        print(f"🚀 KHỞI ĐỘNG HUẤN LUYỆN PHA 3 DDP MULTI-LAYER TPS (INIT={init_tag})")
        print(f"   World Size = {world_size} GPU(s) | Device = {device} | TPS Downscale = 1/{tps_downscale}")
        print(f"   Epochs = {args.epochs} | Synth_ML = {args.synth_count} | StaticBG_Aug = {args.static_aug_prob:.2f}")
        print(f"   w_ghost={args.w_ghost} | w_temp={args.w_temp} | LR=({args.lr_backbone:.1e}, {args.lr_unet:.1e})")
        print("=" * 84, flush=True)

    ckpt_mgr = TwoFileCheckpointManager(args.output_dir, is_rank0=is_rank0)
    base_model = TPSInbetweenOriginal().to(device)
    load_tps_weights(base_model, args.m1_checkpoint, device=device, prefer_ema=False, is_rank0=is_rank0)

    train_ds = SakugaMultiLayerDataset(
        args.train_csv, args.train_clips_dir, args.tps_cache,
        train_mode="LONG", is_train=True, seed=args.seed,
        synth_count=args.synth_count, static_aug_prob=args.static_aug_prob
    )
    val_ds = SakugaMultiLayerDataset(
        args.val_csv, args.val_clips_dir, args.tps_cache,
        train_mode="VAL", is_train=False, seed=args.seed, synth_count=0, static_aug_prob=0.0
    )

    train_sampler = DDPLengthWidthBatchSampler(
        train_ds.samples, world_size=world_size, rank=local_rank,
        batch_n3=4, batch_n5=2, batch_n7=1, seed=args.seed
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler, collate_fn=pad_collate_fn_multilayer,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0)
    )
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_ddp else None
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False, sampler=val_sampler,
        collate_fn=pad_collate_fn_multilayer, num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'), persistent_workers=(args.num_workers > 0), drop_last=False
    )

    if is_rank0:
        print(f"[✓] Dữ liệu Pha 3 sẵn sàng: Train={len(train_ds):,} ({len(train_loader):,} batches/GPU) | Val={len(val_ds):,}", flush=True)

    baseline_json_path = os.path.join(os.path.dirname(args.output_dir), "val_baseline_m0_m1.json")
    has_resume = os.path.exists(ckpt_mgr.last_path)
    need_baseline = (not has_resume and not args.dry_run and not os.path.exists(baseline_json_path))
    if is_ddp:
        base_flag = torch.tensor([1 if need_baseline else 0], dtype=torch.int32, device=device)
        dist.broadcast(base_flag, src=0)
        need_baseline = bool(base_flag.item())

    m0_val_cache, m1_val_cache = None, None
    if need_baseline:
        if is_rank0:
            print("\n🔬 ĐANG ĐO BASELINE EPOCH 0 TRÊN TẬP VAL VỚI MULTI-LAYER TPS...", flush=True)
        m1_val, m0_val = run_validation_suite_multilayer(base_model, val_loader, device, downscale_factor=tps_downscale, is_ddp=is_ddp, evaluate_m0=True)
        m0_val_cache, m1_val_cache = m0_val, m1_val
        if is_rank0:
            with open(baseline_json_path, 'w', encoding='utf-8') as f:
                json.dump({'M0_MultiLayer': m0_val, 'M1_Original': m1_val}, f, indent=2, ensure_ascii=False)
            print(f"📊 M0_Multi (Coarse): PSNR={m0_val['psnr']:.2f}dB | WCD_DoG={m0_val['wcd_dog']:.4f} | Ghost={m0_val['ghost']:.0f}", flush=True)
            print(f"📊 M1_Multi (Orig)  : PSNR={m1_val['psnr']:.2f}dB | WCD_DoG={m1_val['wcd_dog']:.4f} | Ghost={m1_val['ghost']:.0f}\n", flush=True)
        if is_ddp:
            dist.barrier()
    elif os.path.exists(baseline_json_path):
        try:
            with open(baseline_json_path, 'r', encoding='utf-8') as f:
                b_all = json.load(f)
                m0_val_cache = b_all.get('M0_MultiLayer', b_all.get('M0_Coarse', {}))
                m1_val_cache = b_all.get('M1_Original', {})
        except Exception:
            m0_val_cache, m1_val_cache = None, None

    # Nạp Warm-Start (model_tps_blended.pt) và đo mốc Init chuẩn để chống tụt nét
    init_val_ghost = 9200.0
    init_val_skel = 0.850
    init_val_wcd_dog = 1.395
    init_val_dens = 0.960
    if args.init_checkpoint and os.path.exists(args.init_checkpoint):
        if os.path.abspath(args.init_checkpoint) != os.path.abspath(args.m1_checkpoint):
            load_tps_weights(base_model, args.init_checkpoint, device=device, prefer_ema=True, is_rank0=is_rank0)
            if not has_resume and not args.dry_run:
                init_val = run_validation_suite_multilayer(base_model, val_loader, device, downscale_factor=tps_downscale, is_ddp=is_ddp, evaluate_m0=False)
                init_val_ghost = float(init_val['ghost'])
                init_val_skel = float(init_val['skel_recall'])
                init_val_wcd_dog = float(init_val['wcd_dog'])
                init_val_dens = float(init_val['density_ratio'])
                if is_rank0:
                    print(f"📊 Init Warm-Start (Blended + Multi-Layer): PSNR={init_val['psnr']:.2f}dB | WCD_DoG={init_val_wcd_dog:.4f} | Skel={init_val_skel*100:.1f}% | Dens={init_val_dens:.3f} | Ghost={init_val_ghost:.0f}\n", flush=True)

    ema = AdaptiveEMA(base_model, max_decay=args.ema_decay)
    optimizer = torch.optim.AdamW([
        {'params': [p for p in base_model.encoder.parameters() if p.requires_grad], 'lr': args.lr_backbone, 'weight_decay': 1e-4},
        {'params': [p for p in base_model.IFNet.parameters() if p.requires_grad], 'lr': args.lr_backbone, 'weight_decay': 1e-4},
        {'params': [p for p in base_model.unet.parameters() if p.requires_grad], 'lr': args.lr_unet, 'weight_decay': 1e-4},
    ])
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    if is_ddp:
        if is_rank0:
            criterion = Phase3BalancedLineartLoss(train_mode="LONG", is_rank0=True, w_ghost=args.w_ghost, w_temp=args.w_temp).to(device)
        dist.barrier()
        if not is_rank0:
            criterion = Phase3BalancedLineartLoss(train_mode="LONG", is_rank0=False, w_ghost=args.w_ghost, w_temp=args.w_temp).to(device)
        dist.barrier()
    else:
        criterion = Phase3BalancedLineartLoss(train_mode="LONG", is_rank0=True, w_ghost=args.w_ghost, w_temp=args.w_temp).to(device)

    start_epoch = 1
    history = []
    if not args.dry_run:
        last_ep, hist = ckpt_mgr.try_resume(base_model, ema, optimizer, scaler, map_location=device)
        if last_ep > 0:
            start_epoch = last_ep + 1
            history = hist
            if is_rank0:
                print(f"[♻️ AUTO-RESUME] Khôi phục thành công Epoch {last_ep:02d}. Tiếp tục Epoch {start_epoch:02d}.", flush=True)

    m1_ref = m1_val_cache or {}
    base_wcd_dog = float(m1_ref.get('wcd_dog', 1.460))
    base_semd = float(m1_ref.get('semd', 0.0131))
    base_skel = float(m1_ref.get('skel_recall', 0.816))
    base_tip = float(m1_ref.get('tip_recall', 0.820))
    base_ghost = float(m1_ref.get('ghost', 1409.0))
    best_ghost_gate = init_val_ghost * 1.02

    if is_ddp:
        ddp_kwargs = {'device_ids': [local_rank], 'output_device': local_rank} if device.type == 'cuda' else {}
        ddp_model = DDP(base_model, find_unused_parameters=False, **ddp_kwargs)
    else:
        ddp_model = base_model

    history_csv_path = os.path.join(os.path.dirname(args.output_dir), "training_history_finetune.csv")
    watchdog_paused = False
    t_total_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        t_ep0 = time.time()
        if hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)

        # Lịch LR phẫu thuật 4 Epochs: Warmup -> Peak -> Cosine Decay
        if epoch == 1:
            lr_scale = 0.45
        elif epoch == 2:
            lr_scale = 1.00
        else:
            prog = float(epoch - 2) / float(max(1, args.epochs - 2))
            lr_scale = max(0.15, 0.5 * (1.0 + math.cos(math.pi * prog)))

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
            is_direct = bool(batch.get('is_direct_t_flows', False))
            x0 = batch['x0'].to(device, non_blocking=True)
            x1 = batch['x1'].to(device, non_blocking=True)
            xk_all = batch['xk_all'].to(device, non_blocking=True)
            t_vals = batch['t_vals'].to(device, non_blocking=True)
            orig_hw = batch['orig_hw'].to(device, non_blocking=True)
            bsz, m_steps = xk_all.shape[0], xk_all.shape[1]

            if is_direct:
                # Tổng hợp hình học đa lớp (Direct-t per step)
                cf01_steps, cf10_steps = [], []
                for m_i in range(m_steps):
                    cf0_i, cf1_i = reconstruct_multilayer_flow_batch_from_cache(
                        batch['c0_dyn_all'][:, m_i].to(device, non_blocking=True),
                        batch['th01_dyn_all'][:, m_i].to(device, non_blocking=True),
                        batch['c1_dyn_all'][:, m_i].to(device, non_blocking=True),
                        batch['th10_dyn_all'][:, m_i].to(device, non_blocking=True),
                        batch['valid_dyn_all'][:, m_i].to(device, non_blocking=True),
                        batch['dyn_pts0_all'][:, m_i].to(device, non_blocking=True),
                        batch['dyn_pts1_all'][:, m_i].to(device, non_blocking=True),
                        batch['dyn_pts_len_all'][:, m_i].to(device, non_blocking=True),
                        batch['static_pts0_all'][:, m_i].to(device, non_blocking=True),
                        batch['static_pts1_all'][:, m_i].to(device, non_blocking=True),
                        batch['num_static_all'][:, m_i].to(device, non_blocking=True),
                        batch['num_clusters_all'][:, m_i].to(device, non_blocking=True),
                        orig_hw, h_pad, w_pad, device, downscale_factor=tps_downscale,
                        use_dyn_anchors=False, transition_width_px=18.0, falloff_radius_px=160.0
                    )
                    cf01_steps.append(cf0_i)
                    cf10_steps.append(cf1_i)
                cf01_f = torch.stack(cf01_steps, dim=1).reshape(bsz * m_steps, 2, h_pad, w_pad)
                cf10_f = torch.stack(cf10_steps, dim=1).reshape(bsz * m_steps, 2, h_pad, w_pad)
            else:
                cf01, cf10 = reconstruct_multilayer_flow_batch_from_cache(
                    batch['c0_dyn'].to(device, non_blocking=True),
                    batch['th01_dyn'].to(device, non_blocking=True),
                    batch['c1_dyn'].to(device, non_blocking=True),
                    batch['th10_dyn'].to(device, non_blocking=True),
                    batch['valid_dyn'].to(device, non_blocking=True),
                    batch['dyn_pts0'].to(device, non_blocking=True),
                    batch['dyn_pts1'].to(device, non_blocking=True),
                    batch['dyn_pts_len'].to(device, non_blocking=True),
                    batch['static_pts0'].to(device, non_blocking=True),
                    batch['static_pts1'].to(device, non_blocking=True),
                    batch['num_static'].to(device, non_blocking=True),
                    batch['num_clusters'].to(device, non_blocking=True),
                    orig_hw, h_pad, w_pad, device, downscale_factor=tps_downscale,
                    use_dyn_anchors=True, transition_width_px=22.0, falloff_radius_px=105.0
                )
                _apply_paired_spatial_aug_gpu(x0, x1, xk_all, cf01, cf10, orig_hw, is_sequence=True)
                cf01_f = cf01.unsqueeze(1).expand(-1, m_steps, -1, -1, -1).reshape(bsz * m_steps, 2, h_pad, w_pad)
                cf10_f = cf10.unsqueeze(1).expand(-1, m_steps, -1, -1, -1).reshape(bsz * m_steps, 2, h_pad, w_pad)

            x0_f = x0.unsqueeze(1).expand(-1, m_steps, -1, -1, -1).reshape(bsz * m_steps, 1, h_pad, w_pad)
            x1_f = x1.unsqueeze(1).expand(-1, m_steps, -1, -1, -1).reshape(bsz * m_steps, 1, h_pad, w_pad)
            xk_f = xk_all.reshape(bsz * m_steps, 1, h_pad, w_pad)
            t_f = t_vals.reshape(bsz * m_steps, 1, 1, 1)
            hw_f = orig_hw.unsqueeze(1).expand(-1, m_steps, -1).reshape(bsz * m_steps, 2)
            seq_ids_f = torch.arange(bsz, device=device).unsqueeze(1).expand(-1, m_steps).reshape(-1)

            total_f = x0_f.shape[0]
            total_pairs = int((seq_ids_f[1:] == seq_ids_f[:-1]).sum().item()) if total_f >= 2 else 0
            do_opt_step = ((b_idx + 1) % grad_accum == 0) or ((b_idx + 1) == num_batches)
            chunk_sz = max(1, args.max_chunk_frames)

            batch_loss_sum = 0.0
            prev_pred_last, prev_gt_last, prev_seq_id = None, None, None

            for c_start in range(0, total_f, chunk_sz):
                c_end = min(c_start + chunk_sz, total_f)
                need_sync = (do_opt_step and c_end == total_f)
                sync_ctx = (ddp_model.no_sync() if (is_ddp and not need_sync) else contextlib.nullcontext())

                with sync_ctx:
                    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        pred_c, *_ = ddp_model(
                            x0_f[c_start:c_end], x1_f[c_start:c_end],
                            t_f[c_start:c_end], cf01_f[c_start:c_end], cf10_f[c_start:c_end],
                            direct_t_flows=is_direct
                        )
                        gt_c = xk_f[c_start:c_end]
                        hw_c = hw_f[c_start:c_end]
                        seq_c = seq_ids_f[c_start:c_end]
                        l_c, d_c = criterion(pred_c, gt_c, valid_hw=hw_c)
                        scaled_l = l_c * (float(c_end - c_start) / float(total_f * grad_accum))

                        if total_pairs > 0 and criterion.w_temp > 0.0:
                            l_temp_sum = pred_c.new_zeros(())
                            if pred_c.shape[0] >= 2:
                                same_m = (seq_c[1:] == seq_c[:-1])
                                if same_m.any():
                                    l_temp_sum = l_temp_sum + criterion.compute_temporal_pair_loss(
                                        pred_c[:-1][same_m], pred_c[1:][same_m],
                                        gt_c[:-1][same_m], gt_c[1:][same_m],
                                        valid_hw=hw_c[1:][same_m]
                                    )
                            if prev_pred_last is not None and int(seq_c[0].item()) == prev_seq_id:
                                l_temp_sum = l_temp_sum + criterion.compute_temporal_pair_loss(
                                    prev_pred_last, pred_c[0:1], prev_gt_last, gt_c[0:1], valid_hw=hw_c[0:1]
                                )
                            scaled_l = scaled_l + (l_temp_sum / float(total_pairs * grad_accum))

                    scaler.scale(scaled_l).backward()
                    batch_loss_sum += d_c['total_loss'] * float(c_end - c_start)
                    prev_pred_last = pred_c[-1:].detach()
                    prev_gt_last = gt_c[-1:].detach()
                    prev_seq_id = int(seq_c[-1].item())

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
                print(f"[✓ DRY-RUN PHA 3 THÀNH CÔNG] {acc_frames} frames qua Multi-Layer DDP + StaticBG Aug + EMA! Avg Loss={acc_loss/max(1,acc_frames):.4f}", flush=True)
            break

        t_val0 = time.time()
        ema.apply_shadow(base_model)
        val_m = run_validation_suite_multilayer(base_model, val_loader, device, downscale_factor=tps_downscale, is_ddp=is_ddp, evaluate_m0=False)
        ema.restore(base_model)
        val_min = (time.time() - t_val0) / 60.0

        dens = float(val_m['density_ratio'])
        skel = float(val_m['skel_recall'])
        tip = float(val_m['tip_recall'])
        wcd_dog = float(val_m['wcd_dog'])
        semd = float(val_m['semd'])
        ghost = float(val_m['ghost'])

        # Cổng kiểm duyệt Strict Non-Regression so với Init Warm-Start (Blended):
        # Không cho phép tụt Density, Skel Recall hay phình WCD_DoG!
        gate_density = (0.955 <= dens <= 1.055)
        gate_skel = (skel >= init_val_skel - 0.002)
        gate_wcd = (wcd_dog <= init_val_wcd_dog * 1.005)
        gate_ghost = (ghost <= best_ghost_gate)
        tier1_passed = bool(gate_density and gate_skel and gate_wcd and gate_ghost)

        term_wcd = 0.25 * (wcd_dog / max(1e-4, base_wcd_dog))
        term_semd = 0.08 * (semd / max(1e-4, base_semd))
        term_skel = 0.22 * ((1.0 - skel) / max(1e-4, 1.0 - base_skel))
        term_tip = 0.09 * ((1.0 - tip) / max(1e-4, 1.0 - base_tip))
        term_dens = 0.16 * (abs(dens - 1.00) / 0.10)
        term_ghost = 0.20 * (ghost / max(1500.0, base_ghost * 2.0))
        s_val = term_wcd + term_semd + term_skel + term_tip + term_dens + term_ghost

        if tier1_passed:
            comp_score = s_val
            best_ghost_gate = min(best_ghost_gate, ghost)
        else:
            over_dens = max(0.0, dens - 1.055)
            under_dens = max(0.0, 0.955 - dens)
            under_skel = max(0.0, (init_val_skel - 0.002 - skel) * 50.0)
            over_wcd = max(0.0, (wcd_dog - init_val_wcd_dog * 1.005) * 15.0)
            over_ghost = max(0.0, (ghost - best_ghost_gate) / 1000.0)
            comp_score = s_val + 10.0 + 5.0 * over_dens + 4.0 * under_dens + under_skel + over_wcd + 1.0 * over_ghost

        is_best = comp_score < ckpt_mgr.best_score

        if is_rank0:
            row = {
                'epoch': epoch, 'lr_scale': round(lr_scale, 4),
                'train_loss': acc_loss / max(1, acc_frames),
                'val_l1': val_m['val_l1'], 'psnr': val_m['psnr'], 'ssim': val_m['ssim'],
                'wcd': val_m['wcd'], 'wcd_legacy': val_m['wcd_legacy'],
                'wcd_dog': val_m['wcd_dog'], 'wcd_dog_legacy': val_m['wcd_dog_legacy'],
                'semd': val_m['semd'], 'skel_recall': val_m['skel_recall'],
                'tip_recall': val_m['tip_recall'], 'recall': val_m['recall'],
                'precision': val_m['precision'], 'f1': val_m['f1'],
                'density_ratio': val_m['density_ratio'], 'pwcr': val_m['pwcr'],
                'rgd': val_m['rgd'], 'ghost': val_m['ghost'],
                's_val_norm': s_val, 'score': comp_score,
                'gate_passed': tier1_passed, 'is_best': is_best,
                'train_time_min': train_min, 'val_time_min': val_min,
            }
            for sl in (3, 5, 7):
                if sl in val_m.get('breakdown', {}):
                    row[f'psnr_N{sl}'] = val_m['breakdown'][sl]['psnr']
                    row[f'wcd_N{sl}'] = val_m['breakdown'][sl]['wcd']
                    row[f'wcd_dog_N{sl}'] = val_m['breakdown'][sl]['wcd_dog']
                    row[f'skel_recall_N{sl}'] = val_m['breakdown'][sl]['skel_recall']
                    row[f'tip_recall_N{sl}'] = val_m['breakdown'][sl]['tip_recall']
                    row[f'ghost_N{sl}'] = val_m['breakdown'][sl]['ghost']
                    row[f'density_N{sl}'] = val_m['breakdown'][sl]['density_ratio']

            history.append(row)
            pd.DataFrame(history).to_csv(history_csv_path, index=False)
            ckpt_mgr.save_epoch(epoch, base_model, ema, optimizer, scaler, comp_score, history, is_best=is_best)

            gate_badge = "✅ PASS" if tier1_passed else "⚠️ GATE_BREACH"
            print(
                f"📌 EP {epoch:02d}/{args.epochs:02d} [PHA3_ML] | Train={train_min:.2f}m | Val={val_min:.2f}m | "
                f"Loss={row['train_loss']:.4f} | PSNR={val_m['psnr']:.2f}dB | "
                f"WCD_DoG={val_m['wcd_dog']:.4f} | SkelRec={val_m['skel_recall']*100:.1f}% | "
                f"Dens={dens:.3f} | Ghost={ghost:.0f} | Gate={gate_badge} | Score={comp_score:.4f}",
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
        print(f"\n🎉 HOÀN TẤT HUẤN LUYỆN PHA 3 TRONG {(time.time() - t_total_start)/60.0:.1f} PHÚT!", flush=True)


if __name__ == '__main__':
    main()
