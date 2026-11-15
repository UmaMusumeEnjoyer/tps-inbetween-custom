#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_orig_vs_pha3.py
--------------------------------------------------------------------------------
BỘ BENCHMARK ĐỐI ĐẦU:
    1. TPS-Inbetween Original (Pipeline Gốc):
       - Checkpoint: checkpoints/pretrained_baselines/tps-original/model_latest.pt
       - Flow: Original Backward TPS (solve_tps_backward_pair + tps_flow_from_theta_batch)
       - Post-processing: None (Raw output)
    2. Sequence 2 Pha 3 Remake (Pipeline Mới):
       - Checkpoint: checkpoints/Re-train/Sequence 2 pha 3 remake/checkpoints/model_tps_finetuned_best.pt
       - Flow: Multi-Layer Clustered TPS (solve_multi_layer_tps)
       - Post-processing: Balanced Adaptive Sharpener (apply_adaptive_sharpener)
    3. Biến thể trung gian: Pha 3 Remake Raw (để quan sát hiệu quả đơn thuần của việc Finetune)

4 BÀI TEST TIÊU CHUẨN:
    - Case 1: Chuỗi Lineart thực tế 24 frames (17 in-betweens)
    - Case 2: Squash & Stretch Moving Circle (360px)
    - Case 3: Rotating Deforming Hollow Cross (360px - 30°)
    - Case 4: Rotating Deforming Solid Cross (360px - 30°)

XUẤT ĐẦY ĐỦ FRAME / MASKS / STRIPS / GIFS / CSV VÀ ẢNH BẢNG BIỂU ĐỒ HỌA TẠI:
    bench_outputs_seq_p3_vs_orig/
--------------------------------------------------------------------------------
"""

import os
import sys
import time
import glob
import math
import json
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from skimage.morphology import skeletonize
except ImportError:
    skeletonize = None

try:
    import imageio
except ImportError:
    imageio = None

import matplotlib.pyplot as plt
import matplotlib.patches as patches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PHASE3_DIR = os.path.join(REPO_ROOT, "kaggle_seq_p3_multilayer_tps")
PHASE2_DIR = os.path.join(REPO_ROOT, "kaggle_seq_p2_v33_ghost_purge")

for _d in (SCRIPT_DIR, PHASE3_DIR, PHASE2_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from seq_p3_multi_layer_tps import solve_multi_layer_tps
from seq_p3_adaptive_sharpener import apply_adaptive_sharpener
from seq_p2_v33_ghost_purge_worker import (
    TPSInbetweenOriginal,
    load_tps_weights,
    solve_tps_backward_pair,
    tps_flow_from_theta_batch,
    evaluate_metrics_dual_regime,
)
from seq_p3_inference_pipeline import (
    load_ink_from_input,
    ink_tensor_to_u8,
)

BASE_OUT_DIR = os.path.join(REPO_ROOT, "bench_outputs_seq_p3_vs_orig")
TABLE_IMG_DIR = os.path.join(BASE_OUT_DIR, "table_images")
LINEART_TEST_DIR = os.path.join(REPO_ROOT, "data", "lineart_experiments", "test")
GLUESTICK_NPZ = os.path.join(REPO_ROOT, "kaggle_seq_p1_v32_ghost_temp_scale", "lineart_test_gluestick_matches.npz")

ORIG_CKPT = os.path.join(REPO_ROOT, "checkpoints", "pretrained_baselines", "tps-original", "model_latest.pt")
PHA3_CKPT = os.path.join(REPO_ROOT, "checkpoints", "Re-train", "Sequence 2 pha 3 remake", "checkpoints", "model_tps_finetuned_best.pt")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Segoe UI', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
# HÀM HÌNH HỌC PHÒNG THỦ (CASE 2, 3, 4)
# ==============================================================================
def draw_deforming_circle(t, H=384, W=640):
    im = np.zeros((H, W), dtype=np.float32)
    cx = int(round((1.0 - t) * 140 + t * 490))
    cy = int(round((1.0 - t) * 192 + t * 192))
    rx = int(round((1.0 - t) * 90 + t * 45))
    ry = int(round((1.0 - t) * 40 + t * 45))
    cv2.ellipse(im, (cx, cy), (rx, ry), 0, 0, 360, 1.0, 4, lineType=cv2.LINE_AA)
    return im


def draw_deforming_cross(t, H=384, W=640):
    im = np.zeros((H, W), dtype=np.float32)
    cx = int(round((1.0 - t) * 140 + t * 490))
    cy = int(round((1.0 - t) * 192 + t * 192))
    lx = (1.0 - t) * 80.0 + t * 160.0
    ly = (1.0 - t) * 120.0 + t * 45.0
    angle_rad = np.radians(t * 30.0)

    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    def draw_bar(w_len, h_len):
        half_w, half_h = w_len / 2.0, h_len / 2.0
        corners = np.array([[-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_w, half_h]], dtype=np.float32)
        rot_pts = (rot_mat @ corners.T).T + np.array([cx, cy], dtype=np.float32)
        pts_i32 = np.int32(np.round(rot_pts))
        cv2.polylines(im, [pts_i32], isClosed=True, color=1.0, thickness=3, lineType=cv2.LINE_AA)

    draw_bar(lx, 10.0)
    draw_bar(10.0, ly)
    return np.clip(im, 0.0, 1.0)


def draw_deforming_solid_cross(t, H=384, W=640):
    im = np.zeros((H, W), dtype=np.float32)
    cx = int(round((1.0 - t) * 140 + t * 490))
    cy = int(round((1.0 - t) * 192 + t * 192))
    lx = (1.0 - t) * 80.0 + t * 160.0
    ly = (1.0 - t) * 120.0 + t * 45.0
    angle_rad = np.radians(t * 30.0)

    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    def draw_bar(w_len, h_len):
        half_w, half_h = w_len / 2.0, h_len / 2.0
        corners = np.array([[-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_w, half_h]], dtype=np.float32)
        rot_pts = (rot_mat @ corners.T).T + np.array([cx, cy], dtype=np.float32)
        pts_i32 = np.int32(np.round(rot_pts))
        cv2.fillConvexPoly(im, pts_i32, color=1.0, lineType=cv2.LINE_AA)

    draw_bar(lx, 10.0)
    draw_bar(10.0, ly)
    return np.clip(im, 0.0, 1.0)


def extract_shape_keypoints(im0, im1, max_pts=64):
    u0 = np.clip(im0 * 255.0, 0, 255).astype(np.uint8)
    u1 = np.clip(im1 * 255.0, 0, 255).astype(np.uint8)
    c0, _ = cv2.findContours(u0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c1, _ = cv2.findContours(u1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not c0 or not c1:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    pts0 = c0[0].reshape(-1, 2).astype(np.float32)
    pts1 = c1[0].reshape(-1, 2).astype(np.float32)
    n = min(len(pts0), len(pts1), max_pts)
    if n < 8:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    idx0 = np.linspace(0, len(pts0) - 1, n, dtype=int)
    idx1 = np.linspace(0, len(pts1) - 1, n, dtype=int)
    p0 = pts0[idx0]
    p1 = pts1[idx1]
    p0[:, 0] /= im0.shape[1]
    p0[:, 1] /= im0.shape[0]
    p1[:, 0] /= im1.shape[1]
    p1[:, 1] /= im1.shape[0]
    return p0, p1


def compute_exact_morphological_metrics(pred_ink_np, gt_ink_np):
    p_bin = pred_ink_np > 0.15
    g_bin = gt_ink_np > 0.15
    if skeletonize is not None:
        g_skel = skeletonize(g_bin)
        skel_total = g_skel.sum()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        p_dil = cv2.dilate(p_bin.astype(np.uint8), kernel) > 0
        skel_rec = float((p_dil & g_skel).sum() / max(1.0, float(skel_total)) * 100.0)
    else:
        skel_rec = 100.0
    p_cnt = float(p_bin.sum())
    g_cnt = float(g_bin.sum())
    dens_ratio = float(p_cnt / max(1.0, g_cnt))
    return skel_rec, dens_ratio


def render_multicolumn_strip(columns_with_headers, header_height=32):
    h_max = max(col[0].shape[0] for col in columns_with_headers)
    rendered_cols = []
    for u8_img, text, color in columns_with_headers:
        if len(u8_img.shape) == 2:
            bgr = cv2.cvtColor(u8_img, cv2.COLOR_GRAY2BGR)
        else:
            bgr = u8_img.copy()
        h, w = bgr.shape[:2]
        if h < h_max:
            bgr = cv2.copyMakeBorder(bgr, 0, h_max - h, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
        hdr = np.full((header_height, w, 3), 24, dtype=np.uint8)
        cv2.putText(hdr, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        col_full = np.vstack([hdr, bgr])
        rendered_cols.append(col_full)
    return np.hstack(rendered_cols)


# ==============================================================================
# HÀM VẼ BẢNG ĐỒ HỌA (MATPLOTLIB HIGH DPI)
# ==============================================================================
def find_best_idx(values, direction):
    arr = np.array(values, dtype=float)
    if direction == 'up':
        return int(np.nanargmax(arr))
    elif direction == 'down':
        return int(np.nanargmin(arr))
    elif direction == 'near_one':
        diff = np.abs(arr - 1.0)
        return int(np.nanargmin(diff))
    return -1


def draw_styled_table(
    title,
    subtitle,
    headers,
    col_widths,
    rows_data,
    col_directions,
    out_path,
    figsize=(14, 5),
    dpi=220,
    header_bg="#1e293b",
    header_fg="#ffffff",
    row_bg_even="#ffffff",
    row_bg_odd="#f8fafc",
    best_bg="#dcfce7",
    best_fg="#15803d",
    group_col_idx=None
):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.axis('off')

    num_rows = len(rows_data)
    num_cols = len(headers)

    top_y = 0.82
    row_height = (top_y - 0.12) / max(1, num_rows + 1)
    header_y = top_y

    fig.text(0.04, 0.94, title, fontsize=15, fontweight='bold', color="#0f172a", ha='left', va='top')
    fig.text(0.04, 0.88, subtitle, fontsize=10, color="#475569", ha='left', va='top')

    legend_str = "[↑] Cao hơn là tốt hơn    [↓] Thấp hơn là tốt hơn    [≈1.0] Tiệm cận 1.0 là tốt hơn    |    ■ Ô nền xanh lá = Giá trị tốt nhất"
    fig.text(0.96, 0.88, legend_str, fontsize=8.5, color="#059669", fontweight='semibold', ha='right', va='top')

    total_w = sum(col_widths)
    norm_widths = [w / total_w * 0.92 for w in col_widths]
    x_starts = [0.04]
    for w in norm_widths[:-1]:
        x_starts.append(x_starts[-1] + w)

    # 1. Vẽ Header
    for j in range(num_cols):
        rect = patches.FancyBboxPatch(
            (x_starts[j], header_y - row_height),
            norm_widths[j],
            row_height,
            boxstyle="square,pad=0",
            ec="#cbd5e1",
            fc=header_bg,
            lw=1.0
        )
        ax.add_patch(rect)
        hdr_text = headers[j]
        ha = 'left' if j < (2 if group_col_idx is not None else 1) else 'center'
        tx = x_starts[j] + 0.015 if ha == 'left' else x_starts[j] + norm_widths[j] / 2.0
        ax.text(
            tx,
            header_y - row_height / 2.0,
            hdr_text,
            fontsize=9.5,
            fontweight='bold',
            color=header_fg,
            ha=ha,
            va='center'
        )

    # Xác định các ô tốt nhất theo từng cột số liệu
    best_cells = set()
    start_metric_col = 2 if group_col_idx is not None else 1

    if group_col_idx is not None:
        groups = {}
        for r_i, r in enumerate(rows_data):
            g_key = r[group_col_idx]
            groups.setdefault(g_key, []).append(r_i)

        for g_key, r_indices in groups.items():
            for c_j in range(start_metric_col, num_cols):
                dir_type = col_directions[c_j]
                if dir_type is None:
                    continue
                vals = [float(rows_data[ri][c_j].replace(',', '').replace('%', '')) for ri in r_indices]
                b_idx = find_best_idx(vals, dir_type)
                if b_idx >= 0:
                    best_cells.add((r_indices[b_idx], c_j))
    else:
        for c_j in range(start_metric_col, num_cols):
            dir_type = col_directions[c_j]
            if dir_type is None:
                continue
            vals = [float(rows_data[ri][c_j].replace(',', '').replace('%', '')) for ri in range(num_rows)]
            b_idx = find_best_idx(vals, dir_type)
            if b_idx >= 0:
                best_cells.add((b_idx, c_j))

    # 2. Vẽ Data Rows
    for i in range(num_rows):
        cur_y = header_y - (i + 2) * row_height
        is_even = (i % 2 == 0)
        row_def_bg = row_bg_even if is_even else row_bg_odd

        for j in range(num_cols):
            is_best = (i, j) in best_cells
            cell_bg = best_bg if is_best else row_def_bg
            cell_fg = best_fg if is_best else "#1e293b"
            cell_weight = 'bold' if is_best else 'normal'

            rect = patches.FancyBboxPatch(
                (x_starts[j], cur_y),
                norm_widths[j],
                row_height,
                boxstyle="square,pad=0",
                ec="#e2e8f0",
                fc=cell_bg,
                lw=0.8
            )
            ax.add_patch(rect)

            cell_val = str(rows_data[i][j])
            ha = 'left' if j < start_metric_col else 'center'
            tx = x_starts[j] + 0.015 if ha == 'left' else x_starts[j] + norm_widths[j] / 2.0
            ax.text(
                tx,
                cur_y + row_height / 2.0,
                cell_val,
                fontsize=9.2,
                fontweight=cell_weight,
                color=cell_fg,
                ha=ha,
                va='center'
            )

    fig.text(0.04, 0.04, "Antigravity Sequence Inbetweening Benchmark Suite  |  Verified outputs at: bench_outputs_seq_p3_vs_orig/", fontsize=8, color="#94a3b8", ha='left', va='bottom')

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  [✓] Đã xuất ảnh bảng chỉ số: {out_path}")


# ==============================================================================
# BÀI TEST 1: CHUỖI 24 FRAMES THỰC TẾ (17 IN-BETWEENS)
# ==============================================================================
def run_test_case_1(models, out_dir):
    print("\n" + "=" * 90)
    print("▶ CHẠY CASE 1: CHUỖI 24 FRAMES THỰC TẾ (17 IN-BETWEENS)")
    print(f"  Thư mục dữ liệu: {LINEART_TEST_DIR}")
    print(f"  Thư mục đích:    {out_dir}")
    print("=" * 90)

    clean_dir_orig = os.path.join(out_dir, "clean_frames_original_pipeline")
    clean_dir_p3_raw = os.path.join(out_dir, "clean_frames_phase3_raw")
    clean_dir_p3_new = os.path.join(out_dir, "clean_frames_phase3_newpipeline")
    sbs_dir = os.path.join(out_dir, "side_by_side_comparison")

    for d in (clean_dir_orig, clean_dir_p3_raw, clean_dir_p3_new, sbs_dir):
        os.makedirs(d, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(LINEART_TEST_DIR, "*.png")))
    total_frames = len(paths)
    kf_idx = [0, 3, 7, 11, 14, 19, 23]
    kf_set = set(kf_idx)

    gs_cache = {}
    if os.path.exists(GLUESTICK_NPZ):
        data = np.load(GLUESTICK_NPZ)
        for k in data.files:
            gs_cache[k] = data[k]
        print(f"  [✓] Đã nạp {len(gs_cache)} cặp GlueStick cache.")

    frames_gt = [None] * total_frames
    frames_orig = [None] * total_frames
    frames_p3_raw = [None] * total_frames
    frames_p3_new = [None] * total_frames

    for i in range(total_frames):
        ink_gt = load_ink_from_input(paths[i])
        u8_gt = np.clip((1.0 - ink_gt) * 255.0, 0, 255).astype(np.uint8)
        frames_gt[i] = u8_gt
        if i in kf_set:
            frames_orig[i] = u8_gt
            frames_p3_raw[i] = u8_gt
            frames_p3_new[i] = u8_gt
            fname = os.path.basename(paths[i])
            cv2.imwrite(os.path.join(clean_dir_orig, fname), u8_gt)
            cv2.imwrite(os.path.join(clean_dir_p3_raw, fname), u8_gt)
            cv2.imwrite(os.path.join(clean_dir_p3_new, fname), u8_gt)

    rows = []

    for s_idx in range(len(kf_idx) - 1):
        i0, i1 = kf_idx[s_idx], kf_idx[s_idx + 1]
        delta_f = i1 - i0
        ink0 = load_ink_from_input(paths[i0])
        ink1 = load_ink_from_input(paths[i1])
        h, w = ink0.shape
        h_pad, w_pad = ((h + 31) // 32) * 32, ((w + 31) // 32) * 32

        kps = gs_cache[f"seg_{i0}_{i1}"].astype(np.float32)

        # 1. Luồng gốc của tác giả (Backward Single-Layer TPS với 8 anchors biên cố định)
        hw_t = torch.tensor([[h, w]], dtype=torch.float32, device=DEVICE)
        c0_np, t01_np, c1_np, t10_np, vf = solve_tps_backward_pair(kps[0], kps[1], max_pts=128, lambd=0.01)
        orig_cf01 = tps_flow_from_theta_batch(torch.from_numpy(t01_np)[None].to(DEVICE), torch.from_numpy(c0_np)[None].to(DEVICE), torch.tensor([vf], device=DEVICE), hw_t, h_pad, w_pad, DEVICE, 2)
        orig_cf10 = tps_flow_from_theta_batch(torch.from_numpy(t10_np)[None].to(DEVICE), torch.from_numpy(c1_np)[None].to(DEVICE), torch.tensor([vf], device=DEVICE), hw_t, h_pad, w_pad, DEVICE, 2)

        # 2. Luồng mới của Pha 3 (Multi-Layer Clustered TPS với C^2 blending)
        cf01_ml, cf10_ml, _ = solve_multi_layer_tps(
            kps[0], kps[1], (h, w), h_pad, w_pad, DEVICE,
            direct_t_flows=False, max_pts=128, lambd=0.01,
            downscale_factor=2, eps_static_px=1.5,
            use_dyn_anchors=True, transition_width_px=22.0, falloff_radius_px=105.0
        )

        x0 = F.pad(torch.from_numpy(ink0)[None, None], (0, w_pad - w, 0, h_pad - h)).to(DEVICE)
        x1 = F.pad(torch.from_numpy(ink1)[None, None], (0, w_pad - w, 0, h_pad - h)).to(DEVICE)

        for f_idx in range(i0 + 1, i1):
            tv = float(f_idx - i0) / float(delta_f)
            gt_ink_np = load_ink_from_input(paths[f_idx])
            gt_t = torch.from_numpy(gt_ink_np)[None, None].to(DEVICE)
            tt = torch.tensor([[[[tv]]]], dtype=torch.float32, device=DEVICE)

            # Mô hình 1: TPS-Inbetween Original (Pipeline gốc)
            with torch.no_grad():
                pr_orig, *_ = models['orig'](x0.float(), x1.float(), tt, orig_cf01.float(), orig_cf10.float())
            t_orig = pr_orig[:, :, :h, :w].clamp(0.0, 1.0)
            u8_orig = ink_tensor_to_u8(t_orig)
            frames_orig[f_idx] = u8_orig

            # Mô hình 2: Pha 3 Remake Raw (Finetune + Multi-Layer TPS, không Sharpener)
            with torch.no_grad():
                pr_p3, *_ = models['pha3'](x0.float(), x1.float(), tt, cf01_ml.float(), cf10_ml.float())
            t_p3_raw = pr_p3[:, :, :h, :w].clamp(0.0, 1.0)
            u8_p3_raw = ink_tensor_to_u8(t_p3_raw)
            frames_p3_raw[f_idx] = u8_p3_raw

            # Mô hình 3: Pha 3 Remake Newpipeline (Finetune + Multi-Layer TPS + Balanced Sharpener)
            t_p3_new = apply_adaptive_sharpener(t_p3_raw, mode="balanced", input_is_ink=True)
            u8_p3_new = ink_tensor_to_u8(t_p3_new)
            frames_p3_new[f_idx] = u8_p3_new

            fname = os.path.basename(paths[f_idx])
            cv2.imwrite(os.path.join(clean_dir_orig, fname), u8_orig)
            cv2.imwrite(os.path.join(clean_dir_p3_raw, fname), u8_p3_raw)
            cv2.imwrite(os.path.join(clean_dir_p3_new, fname), u8_p3_new)

            rec = {
                "frame": fname,
                "segment": f"{i0}->{i1}",
                "delta_f": delta_f,
                "t": round(tv, 3),
            }
            version_tensors = [
                ("Original_Pipeline", t_orig),
                ("Pha3_Remake_Raw", t_p3_raw),
                ("Pha3_Remake_Newpipeline", t_p3_new),
            ]
            for tag, tens in version_tensors:
                mm = evaluate_metrics_dual_regime(tens, gt_t)
                for mk, mv in mm.items():
                    rec[f"{tag}_{mk}"] = float(mv.item())
                rec[f"{tag}_ie_mae"] = float(torch.abs(tens - gt_t).mean().item() * 255.0)
            rows.append(rec)
            print(f"  Frame {f_idx:02d} (t={tv:.2f}) | Orig_Pipe: PSNR={rec['Original_Pipeline_psnr']:.2f}dB, Ghost={rec['Original_Pipeline_ghost']:.0f} | P3_Newpipe: PSNR={rec['Pha3_Remake_Newpipeline_psnr']:.2f}dB, Ghost={rec['Pha3_Remake_Newpipeline_ghost']:.0f}")

    # Render Side-by-Side 4 cột: [GT | Original Pipeline | Pha 3 Raw | Pha 3 Newpipeline]
    sbs_rgb_list = []
    for i in range(total_frames):
        c_gt = frames_gt[i]
        c_orig = frames_orig[i]
        c_p3_raw = frames_p3_raw[i]
        c_p3_new = frames_p3_new[i]

        lbl_gt = f"GT #{i:02d}" if i not in kf_set else f"Keyframe #{i:02d}"
        cols_cfg = [
            (c_gt, lbl_gt, (230, 230, 230)),
            (c_orig, "TPS-Inbetween Orig (Goc)", (255, 130, 130)),
            (c_p3_raw, "Pha 3 Remake (Raw)", (120, 180, 255)),
            (c_p3_new, "Pha 3 Remake (Newpipeline)", (80, 235, 110)),
        ]
        sbs_row = render_multicolumn_strip(cols_cfg, header_height=30)
        sbs_scaled = cv2.resize(sbs_row, (int(sbs_row.shape[1] * 0.45), int(sbs_row.shape[0] * 0.45)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(sbs_dir, f"sbs_comparison_{i:03d}.png"), sbs_scaled)
        sbs_rgb_list.append(sbs_scaled[:, :, ::-1])

    if imageio is not None and sbs_rgb_list:
        gif_path = os.path.join(out_dir, "benchmark_lineart_test_comparison.gif")
        imageio.mimsave(gif_path, sbs_rgb_list, fps=8, loop=0)
        print(f"  [✓] Đã xuất GIF đối đầu Case 1: {gif_path}")

    df_res = pd.DataFrame(rows)
    df_res.to_csv(os.path.join(out_dir, "benchmark_case1_per_frame.csv"), index=False)
    return df_res


# ==============================================================================
# BÀI TEST 2, 3, 4: DEFENSE GEOMETRY SUITE
# ==============================================================================
def run_single_geometry_case(cname, draw_fn, models, out_case_dir):
    print("\n" + "=" * 90)
    print(f"▶ CHẠY BÀI TEST HÌNH HỌC: {cname}")
    print(f"  Thư mục đích: {out_case_dir}")
    print("=" * 90)
    os.makedirs(out_case_dir, exist_ok=True)

    clean_dir = os.path.join(out_case_dir, "clean_frames")
    os.makedirs(clean_dir, exist_ok=True)

    H, W = 384, 640
    im0 = draw_fn(0.0, H, W)
    im1 = draw_fn(1.0, H, W)
    t_vals = [0.25, 0.50, 0.75]

    k0, k1 = extract_shape_keypoints(im0, im1)
    x0 = torch.from_numpy(im0)[None, None].to(DEVICE)
    x1 = torch.from_numpy(im1)[None, None].to(DEVICE)

    f0_u8 = np.clip((1.0 - im0) * 255.0, 0, 255).astype(np.uint8)
    f1_u8 = np.clip((1.0 - im1) * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(clean_dir, f"{cname}_t0.00_Keyframe0.png"), f0_u8)
    cv2.imwrite(os.path.join(clean_dir, f"{cname}_t1.00_Keyframe1.png"), f1_u8)

    strips = []
    zoom_crops = []
    gif_frames_orig = [np.stack([f0_u8] * 3, axis=-1)]
    gif_frames_p3_raw = [np.stack([f0_u8] * 3, axis=-1)]
    gif_frames_p3_new = [np.stack([f0_u8] * 3, axis=-1)]

    all_rows = []

    for tv in t_vals:
        gt_im = draw_fn(tv, H, W)
        gt_t = torch.from_numpy(gt_im)[None, None].to(DEVICE)
        tt = torch.tensor([[[[tv]]]], dtype=torch.float32, device=DEVICE)

        if len(k0) >= 8:
            # 1. Luồng gốc của tác giả (Direct Single-layer TPS không phân cụm)
            cf0t_orig, cf1t_orig, _ = solve_multi_layer_tps(
                k0, k1, (H, W), H, W, DEVICE,
                t_val=tv, direct_t_flows=True, max_pts=128, lambd=0.01,
                downscale_factor=2, eps_static_px=1.5, use_dyn_anchors=False,
                transition_width_px=18.0, falloff_radius_px=160.0
            )
            # 2. Luồng mới của Pha 3 (Multi-layer Clustered TPS)
            cf0t_ml, cf1t_ml = cf0t_orig, cf1t_orig
        else:
            cf0t_orig = torch.zeros((1, 2, H, W), device=DEVICE)
            cf1t_orig = torch.zeros((1, 2, H, W), device=DEVICE)
            cf0t_ml, cf1t_ml = cf0t_orig, cf1t_orig

        with torch.no_grad():
            # Mô hình gốc
            pr_orig, *_ = models['orig'](x0.float(), x1.float(), tt, cf0t_orig.float(), cf1t_orig.float(), direct_t_flows=True)
            t_orig = pr_orig.clamp(0.0, 1.0)

            # Pha 3 Remake Raw
            pr_p3, *_ = models['pha3'](x0.float(), x1.float(), tt, cf0t_ml.float(), cf1t_ml.float(), direct_t_flows=True)
            t_p3_raw = pr_p3.clamp(0.0, 1.0)

            # Pha 3 Remake Newpipeline
            t_p3_new = apply_adaptive_sharpener(t_p3_raw, mode="balanced", input_is_ink=True)

        gt_u8 = np.clip((1.0 - gt_im) * 255.0, 0, 255).astype(np.uint8)
        u8_orig = ink_tensor_to_u8(t_orig)
        u8_p3_raw = ink_tensor_to_u8(t_p3_raw)
        u8_p3_new = ink_tensor_to_u8(t_p3_new)

        cv2.imwrite(os.path.join(clean_dir, f"{cname}_t{tv:.2f}_GT.png"), gt_u8)
        cv2.imwrite(os.path.join(clean_dir, f"{cname}_t{tv:.2f}_Original_Pipeline.png"), u8_orig)
        cv2.imwrite(os.path.join(clean_dir, f"{cname}_t{tv:.2f}_Pha3_Raw.png"), u8_p3_raw)
        cv2.imwrite(os.path.join(clean_dir, f"{cname}_t{tv:.2f}_Pha3_Newpipeline.png"), u8_p3_new)

        gif_frames_orig.append(np.stack([u8_orig] * 3, axis=-1))
        gif_frames_p3_raw.append(np.stack([u8_p3_raw] * 3, axis=-1))
        gif_frames_p3_new.append(np.stack([u8_p3_new] * 3, axis=-1))

        rec = {"case": cname, "t_val": tv}
        version_tensors = [
            ("Original_Pipeline", t_orig),
            ("Pha3_Remake_Raw", t_p3_raw),
            ("Pha3_Remake_Newpipeline", t_p3_new),
        ]
        for tag, tens in version_tensors:
            mm = evaluate_metrics_dual_regime(tens, gt_t)
            for mk, mv in mm.items():
                rec[f"{tag}_{mk}"] = float(mv.item())
            rec[f"{tag}_ie_mae"] = float(torch.abs(tens - gt_t).mean().item() * 255.0)
            sk_rec, d_rat = compute_exact_morphological_metrics(tens[0, 0].detach().cpu().numpy(), gt_im)
            rec[f"{tag}_exact_skel_recall"] = sk_rec
            rec[f"{tag}_exact_density"] = d_rat

        all_rows.append(rec)
        print(f"  t={tv:.2f} | Orig_Pipe: SkelRec={rec['Original_Pipeline_exact_skel_recall']:.2f}%, Ghost={rec['Original_Pipeline_ghost']:.0f} | P3_Newpipe: SkelRec={rec['Pha3_Remake_Newpipeline_exact_skel_recall']:.2f}%, Ghost={rec['Pha3_Remake_Newpipeline_ghost']:.0f}")

        # Strip 4 cột đối đầu
        cols_cfg = [
            (gt_u8, f"GT (t={tv:.2f})", (230, 230, 230)),
            (u8_orig, "TPS-Inbetween Orig (Goc)", (255, 130, 130)),
            (u8_p3_raw, "Pha 3 Remake (Raw)", (120, 180, 255)),
            (u8_p3_new, "Pha 3 Remake (Newpipeline)", (80, 235, 110)),
        ]
        strip_row = render_multicolumn_strip(cols_cfg, header_height=30)
        strips.append(strip_row)

        # Crop zoom audit
        cx_mid = int(round((1.0 - tv) * 140 + tv * 490))
        cy_mid = 192
        rad_box = 85
        y0, y1 = max(0, cy_mid - rad_box), min(H, cy_mid + rad_box)
        x0_b, x1_b = max(0, cx_mid - rad_box), min(W, cx_mid + rad_box)

        z_cols = []
        for u_img, clr, lbl in [
            (gt_u8, (230, 230, 230), "GT"),
            (u8_orig, (255, 130, 130), "Orig Pipeline"),
            (u8_p3_raw, (120, 180, 255), "Pha 3 Raw"),
            (u8_p3_new, (80, 235, 110), "Pha 3 Newpipe"),
        ]:
            sub = u_img[y0:y1, x0_b:x1_b]
            sub_z = cv2.resize(sub, (220, 220), interpolation=cv2.INTER_NEAREST)
            bgr_z = cv2.cvtColor(sub_z, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(bgr_z, (0, 0), (219, 219), clr, 2)
            cv2.putText(bgr_z, f"{lbl} t={tv:.2f}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr, 1, cv2.LINE_AA)
            z_cols.append(bgr_z)
        zoom_crops.append(np.hstack(z_cols))

    gif_frames_orig.append(np.stack([f1_u8] * 3, axis=-1))
    gif_frames_p3_raw.append(np.stack([f1_u8] * 3, axis=-1))
    gif_frames_p3_new.append(np.stack([f1_u8] * 3, axis=-1))

    out_strip = os.path.join(out_case_dir, f"{cname}_comparison_4col.png")
    out_zoom = os.path.join(out_case_dir, f"{cname}_zoom_audit.png")
    cv2.imwrite(out_strip, np.vstack(strips))
    cv2.imwrite(out_zoom, np.vstack(zoom_crops))

    if imageio is not None:
        imageio.mimsave(os.path.join(out_case_dir, f"{cname}_motion_orig_pipe.gif"), gif_frames_orig, fps=4, loop=0)
        imageio.mimsave(os.path.join(out_case_dir, f"{cname}_motion_p3_raw.gif"), gif_frames_p3_raw, fps=4, loop=0)
        imageio.mimsave(os.path.join(out_case_dir, f"{cname}_motion_p3_newpipeline.gif"), gif_frames_p3_new, fps=4, loop=0)
        print(f"  [✓] Đã xuất demo GIFs cho {cname}.")

    df_case = pd.DataFrame(all_rows)
    df_case.to_csv(os.path.join(out_case_dir, f"{cname}_metrics.csv"), index=False)
    return df_case


# ==============================================================================
# HÀM XUẤT BẢNG BIỂU ĐỒ HỌA
# ==============================================================================
def render_all_table_images(df_case1, df_c2, df_c3, df_c4):
    os.makedirs(TABLE_IMG_DIR, exist_ok=True)

    # 1. Bảng Case 1
    metrics_c1 = [
        ("PSNR (dB)\n[↑]", "psnr"),
        ("SSIM\n[↑]", "ssim"),
        ("Skeleton Recall\n[↑]", "skel_recall"),
        ("Tip Recall\n[↑]", "tip_recall"),
        ("F1 Score\n[↑]", "f1"),
        ("WCD_DoG (px)\n[↓]", "wcd_dog"),
        ("Ghost Pixels\n[↓]", "ghost"),
        ("IE_MAE\n[↓]", "ie_mae"),
        ("Density Ratio\n[≈1.0]", "density_ratio"),
    ]
    versions = [
        ("TPS-Inbetween Orig (Gốc)", "Original_Pipeline"),
        ("Pha 3 Remake (Raw)", "Pha3_Remake_Raw"),
        ("Pha 3 Remake (Newpipeline)", "Pha3_Remake_Newpipeline"),
    ]

    headers_c1 = ["Phiên Bản Mô Hình"] + [m[0] for m in metrics_c1]
    widths_c1 = [3.2, 1.4, 1.3, 1.7, 1.5, 1.3, 1.6, 1.6, 1.4, 1.6]
    dirs_c1 = [None, 'up', 'up', 'up', 'up', 'up', 'down', 'down', 'down', 'near_one']

    rows_c1 = []
    summary_rows_c1 = []
    for v_label, v_prefix in versions:
        row_str = [v_label]
        row_dict = {"Phiên Bản": v_label}
        for m_label, m_key in metrics_c1:
            col = f"{v_prefix}_{m_key}"
            val = float(df_case1[col].mean())
            row_dict[m_label.replace('\n', ' ')] = round(val, 4)
            if "Recall" in m_label:
                row_str.append(f"{val * 100.0:.2f}%" if val <= 1.0 else f"{val:.2f}%")
            elif "Ghost" in m_label:
                row_str.append(f"{val:,.1f}")
            else:
                row_str.append(f"{val:.4f}")
        rows_c1.append(row_str)
        summary_rows_c1.append(row_dict)

    pd.DataFrame(summary_rows_c1).to_csv(os.path.join(BASE_OUT_DIR, "summary_case1_orig_vs_pha3.csv"), index=False)

    img_c1 = os.path.join(TABLE_IMG_DIR, "table_case1_orig_vs_pha3.png")
    draw_styled_table(
        "CASE 1: SO SÁNH CHUỖI LINEART THỰC TẾ 24 FRAMES (17 IN-BETWEENS)",
        "TPS-Inbetween Original (Pipeline Gốc) vs Sequence 2 Pha 3 Remake (Pipeline Mới)",
        headers_c1, widths_c1, rows_c1, dirs_c1, img_c1, figsize=(15, 3.8), dpi=220
    )

    # 2. Bảng Cases 2, 3, 4 tại t=0.50
    geom_cases = [
        ("Case 2: Deforming Circle", df_c2),
        ("Case 3: Deforming Hollow Cross", df_c3),
        ("Case 4: Deforming Solid Cross", df_c4),
    ]
    geom_metrics = [
        ("PSNR (dB)\n[↑]", "psnr"),
        ("SSIM\n[↑]", "ssim"),
        ("Exact Skel Rec\n[↑]", "exact_skel_recall"),
        ("Exact Density\n[≈1.0]", "exact_density"),
        ("WCD_DoG (px)\n[↓]", "wcd_dog"),
        ("Ghost Px\n[↓]", "ghost"),
        ("IE_MAE\n[↓]", "ie_mae"),
    ]
    headers_g = ["Bài Test Hình Học (Case)", "Phiên Bản"] + [m[0] for m in geom_metrics]
    widths_g = [3.2, 3.0, 1.4, 1.3, 1.6, 1.6, 1.6, 1.4, 1.4]
    dirs_g = [None, None, 'up', 'up', 'up', 'near_one', 'down', 'down', 'down']

    rows_g = []
    summary_rows_g = []
    for c_title, df_c in geom_cases:
        row_t05 = df_c[df_c["t_val"] == 0.50].iloc[0]
        for v_label, v_prefix in versions:
            row_str = [c_title, v_label]
            row_dict = {"Case": c_title + " (t=0.50)", "Phiên Bản": v_label}
            for m_label, m_key in geom_metrics:
                col = f"{v_prefix}_{m_key}"
                val = float(row_t05[col])
                row_dict[m_label.replace('\n', ' ')] = round(val, 4)
                if "Skel" in m_label:
                    row_str.append(f"{val:.2f}%")
                elif "Ghost" in m_label:
                    row_str.append(f"{val:,.1f}")
                else:
                    row_str.append(f"{val:.4f}")
            rows_g.append(row_str)
            summary_rows_g.append(row_dict)

    pd.DataFrame(summary_rows_g).to_csv(os.path.join(BASE_OUT_DIR, "summary_cases2_3_4_geometry_t05.csv"), index=False)

    img_g05 = os.path.join(TABLE_IMG_DIR, "table_cases2_3_4_orig_vs_pha3_t05.png")
    draw_styled_table(
        "CASES 2, 3, 4: DEFENSE GEOMETRY SUITE TẠI THỜI ĐIỂM t=0.50",
        "Đánh giá đối đầu tại điểm biến dạng và dịch chuyển 360px cực đại",
        headers_g, widths_g, rows_g, dirs_g, img_g05, figsize=(15.5, 6.2), dpi=220, group_col_idx=0
    )

    # 3. Bảng phân rã chi tiết toàn bộ mốc t=0.25, 0.50, 0.75
    rows_all_t = []
    for c_title, df_c in geom_cases:
        for tv in [0.25, 0.50, 0.75]:
            sub = df_c[df_c["t_val"] == tv].iloc[0]
            for v_label, v_prefix in versions:
                group_id = f"{c_title} (t={tv:.2f})"
                r_s = [group_id, v_label]
                for m_label, m_key in geom_metrics:
                    val = float(sub[f"{v_prefix}_{m_key}"])
                    if "Skel" in m_label:
                        r_s.append(f"{val:.2f}%")
                    elif "Ghost" in m_label:
                        r_s.append(f"{val:,.1f}")
                    else:
                        r_s.append(f"{val:.4f}")
                rows_all_t.append(r_s)

    img_all_t = os.path.join(TABLE_IMG_DIR, "table_geometry_all_t_breakdown.png")
    draw_styled_table(
        "DEFENSE GEOMETRY SUITE: PHÂN RÃ CHI TIẾT THEO CÁC MỐC THỜI GIAN t",
        "Đánh giá chi tiết tại t=0.25, t=0.50 và t=0.75",
        headers_g, widths_g, rows_all_t, dirs_g, img_all_t, figsize=(16.5, 14.5), dpi=220, group_col_idx=0
    )

    # 4. Master Dashboard (Ghép Case 1 & Cases 2,3,4)
    im1 = cv2.imread(img_c1)
    im2 = cv2.imread(img_g05)
    if im1 is not None and im2 is not None:
        w_target = max(im1.shape[1], im2.shape[1])
        h1, w1 = im1.shape[:2]
        h2, w2 = im2.shape[:2]
        im1_s = cv2.resize(im1, (w_target, int(h1 * (w_target / w1))), interpolation=cv2.INTER_LANCZOS4)
        im2_s = cv2.resize(im2, (w_target, int(h2 * (w_target / w2))), interpolation=cv2.INTER_LANCZOS4)
        sep = np.full((32, w_target, 3), 245, dtype=np.uint8)
        dashboard = np.vstack([im1_s, sep, im2_s])
        out_dash = os.path.join(TABLE_IMG_DIR, "table_master_orig_vs_pha3_dashboard.png")
        cv2.imwrite(out_dash, dashboard)
        print(f"  [✓] Đã xuất Master Dashboard: {out_dash}")


# ==============================================================================
# HÀM MAIN
# ==============================================================================
def main():
    print("=" * 90)
    print("🔬 BẮT ĐẦU BENCHMARK ĐỐI ĐẦU: TPS-INBETWEEN ORIGINAL vs PHA 3 REMAKE")
    print(f"  Thiết bị: {DEVICE}")
    print(f"  Original Checkpoint: {ORIG_CKPT}")
    print(f"  Pha 3 Checkpoint:    {PHA3_CKPT}")
    print(f"  Thư mục kết quả:     {BASE_OUT_DIR}")
    print("=" * 90)
    os.makedirs(BASE_OUT_DIR, exist_ok=True)

    # Nạp 2 mô hình
    print("\n📦 Nạp trọng số các mô hình...")
    m_orig = TPSInbetweenOriginal().to(DEVICE).eval()
    load_tps_weights(m_orig, ORIG_CKPT, device=DEVICE, prefer_ema=False, is_rank0=True)

    m_pha3 = TPSInbetweenOriginal().to(DEVICE).eval()
    load_tps_weights(m_pha3, PHA3_CKPT, device=DEVICE, prefer_ema=True, is_rank0=True)

    models = {'orig': m_orig, 'pha3': m_pha3}

    # Case 1: Lineart 24 frames
    out_c1 = os.path.join(BASE_OUT_DIR, "case1_lineart_test_24f")
    df_c1 = run_test_case_1(models, out_c1)

    # Case 2: Circle
    out_c2 = os.path.join(BASE_OUT_DIR, "case2_deforming_circle")
    df_c2 = run_single_geometry_case("case2_deforming_circle", draw_deforming_circle, models, out_c2)

    # Case 3: Hollow Cross
    out_c3 = os.path.join(BASE_OUT_DIR, "case3_deforming_cross")
    df_c3 = run_single_geometry_case("case3_deforming_cross", draw_deforming_cross, models, out_c3)

    # Case 4: Solid Cross
    out_c4 = os.path.join(BASE_OUT_DIR, "case4_deforming_solid_cross")
    df_c4 = run_single_geometry_case("case4_deforming_solid_cross", draw_deforming_solid_cross, models, out_c4)

    # Xuất toàn bộ ảnh bảng đồ họa và CSVs
    print("\n🎨 Tiến hành xuất các bảng đồ họa High DPI...")
    render_all_table_images(df_c1, df_c2, df_c3, df_c4)

    print("\n" + "=" * 90)
    print("🎉 HOÀN TẤT TOÀN BỘ BENCHMARK VÀ TẠO BẢNG ẢNH ĐỐI ĐẦU.")
    print(f"   Thư mục lưu trữ: {BASE_OUT_DIR}")
    print("=" * 90)


if __name__ == "__main__":
    main()
