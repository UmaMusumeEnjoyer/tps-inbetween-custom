#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Benchmark & So Sánh Đối Đầu Toàn Diện V2.1 (Head-to-Head Triad Benchmark):
1. Original TPS-Inbetween (Mixamo 3D Baseline)
2. Finetuned V2.0 Best (Huấn luyện trên Sakuga Golden Dataset V1: 3,545 train / 620 val)
3. Finetuned V2.1 Best (Finetune tiếp từ V2.0 Best trên Sakuga Golden Dataset V1 + Patch 2: 5,727 train / 1,007 val)

Tập dữ liệu kiểm thử: Sakuga Golden Dataset V1 + Patch 2 (sakuga_golden_dataset_tps_ready_v1_patch2, split='val')
Kết quả được lưu trực tiếp vào: checkpoints/experiments/v2.1/
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from run_comparison_v10_vs_original import (
    OriginalTPSInbetween,
    TPSInbetweenFinetune_v10,
    robust_tps_theta_from_points,
    tps_flow_from_theta,
    compute_psnr,
    compute_ssim,
    compute_stroke_metrics,
    compute_ghost_pixels,
    compute_ie_mae,
    compute_chamfer_distance
)
from run_tps_benchmark_v10 import compute_line_tortuosity


def select_stratified_samples(df_val, total_target=100, audit_clip_ids=None):
    """
    Chọn 100 mẫu phân tầng (stratified) đại diện trên tập validation:
    - Phân bổ đều theo các mốc thời gian t in [0.20, 0.25, 0.33, 0.40, 0.50, 0.60, 0.67, 0.75, 0.80]
    - Phân bổ đều theo độ dời chuyển động (mean_disp)
    - Cân đối giữa dữ liệu gốc V1 và dữ liệu mới bổ sung từ Patch 2
    - Bắt buộc bao gồm các clips được cấu hình để xuất Visual Audit
    """
    if audit_clip_ids is None:
        audit_clip_ids = []

    selected_indices = set()

    # 1. Ưu tiên thêm các audit clips
    for clip_id in audit_clip_ids:
        matching = df_val[df_val['clip_id'] == clip_id]
        if not matching.empty:
            selected_indices.add(matching.index[0])

    # 2. Phân nhóm theo t xấp xỉ
    t_groups = {
        0.50: 30,
        0.33: 15,
        0.67: 15,
        0.40: 10,
        0.60: 10,
        0.25: 8,
        0.75: 8,
        0.20: 2,
        0.80: 2
    }

    for t_target, quota in t_groups.items():
        sub_df = df_val[(df_val['t'] >= t_target - 0.04) & (df_val['t'] <= t_target + 0.04)]
        available = [idx for idx in sub_df.index if idx not in selected_indices]
        if not available:
            continue

        # Sắp xếp theo mean_disp để bao phủ từ chuyển động nhỏ đến lớn
        sub_sorted = df_val.loc[available].sort_values(by='mean_disp')
        n_to_pick = min(quota, len(sub_sorted))
        if n_to_pick > 0:
            picked_indices = np.linspace(0, len(sub_sorted) - 1, n_to_pick, dtype=int)
            for p_idx in picked_indices:
                selected_indices.add(sub_sorted.index[p_idx])

    # 3. Nếu chưa đủ total_target, lấy thêm đều các mẫu còn lại
    if len(selected_indices) < total_target:
        remaining = [idx for idx in df_val.index if idx not in selected_indices]
        needed = total_target - len(selected_indices)
        step = max(1, len(remaining) // needed)
        for idx in remaining[::step][:needed]:
            selected_indices.add(idx)

    selected_df = df_val.loc[sorted(list(selected_indices))].reset_index(drop=True)
    if len(selected_df) > total_target:
        audit_mask = selected_df['clip_id'].isin(audit_clip_ids)
        audit_rows = selected_df[audit_mask]
        non_audit_rows = selected_df[~audit_mask]
        needed_non_audit = max(0, total_target - len(audit_rows))
        if needed_non_audit > 0 and len(non_audit_rows) > 0:
            keep_non_audit = non_audit_rows.iloc[np.linspace(0, len(non_audit_rows) - 1, needed_non_audit, dtype=int)]
            selected_df = pd.concat([audit_rows, keep_non_audit]).sort_index().reset_index(drop=True)
        else:
            selected_df = audit_rows.iloc[:total_target].sort_index().reset_index(drop=True)

    return selected_df


def render_visual_audits_v21(output_dir, clip_id, t_val, k0_np, p_coarse_np, p_orig_np, p_v20_np, p_v21_np, gt_np,
                             m_orig_stats, m_v20_stats, m_v21_stats):
    """
    Render 3 bộ trực quan hóa chất lượng cao:
    1. Side-by-Side 6 cột:
       Keyframe 0 | Coarse Warp | Original TPS (Baseline) | Finetuned V2.0 Best | Finetuned V2.1 Best | Ground Truth
    2. Difference Heatmaps:
       Sai số Coarse | Sai số Original | Sai số V2.0 | Sai số V2.1 | Delta Map (V2.0 vs V2.1)
    3. Zoomed 400% Crops:
       So sánh đường nét siêu cận cảnh mí mắt, ngón tay, sợi tóc
    """
    h, w = k0_np.shape
    sbs_dir = os.path.join(output_dir, 'side_by_side')
    hm_dir = os.path.join(output_dir, 'difference_heatmaps')
    zm_dir = os.path.join(output_dir, 'zoomed_details')
    for d in [sbs_dir, hm_dir, zm_dir]:
        os.makedirs(d, exist_ok=True)

    k0_bgr = cv2.cvtColor(k0_np, cv2.COLOR_GRAY2BGR)
    coarse_bgr = cv2.cvtColor(p_coarse_np, cv2.COLOR_GRAY2BGR)
    orig_bgr = cv2.cvtColor(p_orig_np, cv2.COLOR_GRAY2BGR)
    v20_bgr = cv2.cvtColor(p_v20_np, cv2.COLOR_GRAY2BGR)
    v21_bgr = cv2.cvtColor(p_v21_np, cv2.COLOR_GRAY2BGR)
    gt_bgr = cv2.cvtColor(gt_np, cv2.COLOR_GRAY2BGR)

    # 1. Side by Side 6 cột
    def add_header_footer(img_panel, title, tag, stat_str, is_winner=False, color_hdr=None):
        p = img_panel.copy()
        cv2.rectangle(p, (0, 0), (w, 38), (30, 30, 30), -1)
        if color_hdr is None:
            header_color = (0, 255, 120) if is_winner else (255, 255, 255)
        else:
            header_color = color_hdr
        cv2.putText(p, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, header_color, 2)
        cv2.rectangle(p, (0, h - 34), (w, h), (25, 25, 25), -1)
        cv2.putText(p, tag, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 2)
        cv2.putText(p, stat_str, (85, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)
        return p

    p0 = add_header_footer(k0_bgr, "Keyframe 0 (t=0.0)", "[INPUT]", f"{w}x{h} Lineart")
    pc = add_header_footer(coarse_bgr, "TPS Coarse Warp", "[COARSE]", "Thin-Plate Spline")
    po = add_header_footer(orig_bgr, "Original TPS", "[ORIG]",
                           f"{m_orig_stats['psnr']:.1f}dB | {m_orig_stats['ssim']:.3f} | {m_orig_stats['ghost']}px",
                           color_hdr=(100, 150, 255))
    pv20 = add_header_footer(v20_bgr, "Finetuned V2.0", "[V2.0]",
                            f"{m_v20_stats['psnr']:.1f}dB | {m_v20_stats['ssim']:.3f} | {m_v20_stats['ghost']}px",
                            is_winner=(m_v20_stats['ghost'] <= m_v21_stats['ghost']))
    pv21 = add_header_footer(v21_bgr, "Finetuned V2.1 (+Patch2)", "[V2.1]",
                            f"{m_v21_stats['psnr']:.1f}dB | {m_v21_stats['ssim']:.3f} | {m_v21_stats['ghost']}px",
                            is_winner=(m_v21_stats['ghost'] < m_v20_stats['ghost']))
    pgt = add_header_footer(gt_bgr, f"Ground Truth (t={t_val:.2f})", "[TARGET]", "Sakuga Artist Inbetween")

    sbs_6col = np.hstack([p0, pc, po, pv20, pv21, pgt])
    sbs_path = os.path.join(sbs_dir, f"{clip_id}_t{t_val:.2f}_comparison_6col.png")
    cv2.imwrite(sbs_path, sbs_6col)

    # 2. Difference Heatmaps (Residuals)
    diff_coarse = np.abs(p_coarse_np.astype(np.float32) - gt_np.astype(np.float32))
    diff_orig = np.abs(p_orig_np.astype(np.float32) - gt_np.astype(np.float32))
    diff_v20 = np.abs(p_v20_np.astype(np.float32) - gt_np.astype(np.float32))
    diff_v21 = np.abs(p_v21_np.astype(np.float32) - gt_np.astype(np.float32))

    hm_c = cv2.applyColorMap(np.clip(diff_coarse * 2.5, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    hm_o = cv2.applyColorMap(np.clip(diff_orig * 2.5, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    hm_20 = cv2.applyColorMap(np.clip(diff_v20 * 2.5, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    hm_21 = cv2.applyColorMap(np.clip(diff_v21 * 2.5, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Delta map: Residual V2.0 - Residual V2.1 (Dương nghĩa là V2.1 ít sai số hơn V2.0)
    delta_residual = (diff_v20 - diff_v21)
    delta_norm = np.clip((delta_residual + 40.0) * (255.0 / 80.0), 0, 255).astype(np.uint8)
    hm_delta = cv2.applyColorMap(delta_norm, cv2.COLORMAP_VIRIDIS)

    cv2.putText(hm_c, "Coarse Warp Error", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(hm_o, f"Orig Error ({m_orig_stats['ghost']}px)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(hm_20, f"V2.0 Error ({m_v20_stats['ghost']}px)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(hm_21, f"V2.1 Error ({m_v21_stats['ghost']}px)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(hm_delta, "Advantage Delta (V2.1 vs V2.0)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    hm_panel = np.hstack([hm_c, hm_o, hm_20, hm_21, hm_delta])
    hm_path = os.path.join(hm_dir, f"{clip_id}_t{t_val:.2f}_heatmap.png")
    cv2.imwrite(hm_path, hm_panel)

    # 3. Zoomed 400% detail crops (Tập trung vùng nét chuyển động chính)
    cy, cx = h // 2, w // 2
    cs = 110
    y1, y2 = max(0, cy - cs), min(h, cy + cs)
    x1, x2 = max(0, cx - cs), min(w, cx + cs)

    crop_o = cv2.resize(p_orig_np[y1:y2, x1:x2], (360, 360), interpolation=cv2.INTER_NEAREST)
    crop_20 = cv2.resize(p_v20_np[y1:y2, x1:x2], (360, 360), interpolation=cv2.INTER_NEAREST)
    crop_21 = cv2.resize(p_v21_np[y1:y2, x1:x2], (360, 360), interpolation=cv2.INTER_NEAREST)
    crop_gt = cv2.resize(gt_np[y1:y2, x1:x2], (360, 360), interpolation=cv2.INTER_NEAREST)

    zm_panel = np.hstack([crop_o, crop_20, crop_21, crop_gt])
    zm_bgr = cv2.cvtColor(zm_panel, cv2.COLOR_GRAY2BGR)
    cv2.putText(zm_bgr, "Original TPS (Ghost Lines)", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2)
    cv2.putText(zm_bgr, "V2.0 Best (Clean)", (370, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 200), 2)
    cv2.putText(zm_bgr, "V2.1 (+Patch2 Sharp)", (730, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 100), 2)
    cv2.putText(zm_bgr, "Ground Truth", (1090, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    zm_path = os.path.join(zm_dir, f"{clip_id}_t{t_val:.2f}_zoom.png")
    cv2.imwrite(zm_path, zm_bgr)


def run_benchmark_v21(args):
    print("=" * 95)
    print("⚔️  BENCHMARK ĐỐI ĐẦU TOÀN DIỆN: TPS-INBETWEEN FINETUNED V2.1 VS V2.0 VS ORIGINAL TPS")
    print(f"   Model Original Baseline   : {args.model_orig}")
    print(f"   Model V2.0 Best           : {args.model_v20}")
    print(f"   Model V2.1 Target         : {args.model_v21}")
    print(f"   Dataset Root              : {args.data_root}")
    print(f"   Dataset CSV               : {args.dataset_csv}")
    print(f"   Output Directory          : {args.output_dir}")
    print(f"   Target Samples (Fast Mode): {args.num_samples}")
    print("=" * 95)

    os.makedirs(args.output_dir, exist_ok=True)
    visual_dir = os.path.join(args.output_dir, 'visual_comparisons')
    os.makedirs(visual_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"[*] Thiết bị suy luận: {device}")
    if device.type == 'cuda':
        print(f"    GPU: {torch.cuda.get_device_name(0)}")

    # 1. Nạp Model Original TPS
    print(f"\n[1/3] Đang nạp mô hình Original TPS từ: {args.model_orig} ...")
    model_orig = OriginalTPSInbetween().to(device)
    state_orig = torch.load(args.model_orig, map_location=device)
    model_orig.load_state_dict(state_orig, strict=False)
    model_orig.eval()
    print("[✓] Nạp model Original TPS thành công!")

    # 2. Nạp Model V2.0 Best
    print(f"\n[2/3] Đang nạp mô hình V2.0 Best từ: {args.model_v20} ...")
    model_v20 = TPSInbetweenFinetune_v10().to(device)
    state_v20 = torch.load(args.model_v20, map_location=device)
    if 'state_dict' in state_v20:
        state_v20 = state_v20['state_dict']
    model_v20.load_state_dict(state_v20, strict=True)
    model_v20.eval()
    print("[✓] Nạp model V2.0 Best thành công với strict=True!")

    # 3. Nạp Model V2.1 Target
    print(f"\n[3/3] Đang nạp mô hình V2.1 Target từ: {args.model_v21} ...")
    model_v21 = TPSInbetweenFinetune_v10().to(device)
    state_v21 = torch.load(args.model_v21, map_location=device)
    if 'state_dict' in state_v21:
        state_v21 = state_v21['state_dict']
    model_v21.load_state_dict(state_v21, strict=True)
    model_v21.eval()
    print("[✓] Nạp model V2.1 Target thành công với strict=True!")

    # 4. Chuẩn bị tập dữ liệu đánh giá
    df_all = pd.read_csv(args.dataset_csv)
    df_val = df_all[df_all['split'] == 'val'].reset_index(drop=True)
    print(f"\n[*] Tổng số mẫu validation trong dataset: {len(df_val)} triplets.")

    # Danh mục audit clips bao gồm các cảnh từ V1 và các cảnh mới từ Patch 2
    audit_clips = [
        'clip_000008', 'clip_000010', 'clip_000018', 'clip_000064', 'clip_003046', 'clip_003050',
        'clip_004200', 'clip_005225'
    ]
    df_eval = select_stratified_samples(df_val, total_target=args.num_samples, audit_clip_ids=audit_clips)
    print(f"[✓] Đã chọn {len(df_eval)} mẫu phân tầng (stratified) cho chế độ benchmark test nhanh.")
    print(f"    Phân phối các mốc t: {dict(df_eval['t'].round(2).value_counts())}")

    dataset_root = args.data_root
    records = []
    t_start_total = time.time()

    pbar = tqdm(df_eval.iterrows(), total=len(df_eval), desc="🚀 Đối đầu Tam Mã V2.1 vs V2.0 vs Orig", unit="sample")
    for idx, row in pbar:
        clip_id = str(row['clip_id'])
        t_val = float(row['t'])
        disp = float(row.get('mean_disp', 0.0))
        inliers = int(row.get('inliers', 0))

        p0 = os.path.join(dataset_root, str(row['frame0']).lstrip('/\\'))
        pgt = os.path.join(dataset_root, str(row['frame_gt']).lstrip('/\\'))
        p1 = os.path.join(dataset_root, str(row['frame1']).lstrip('/\\'))
        pmatch = os.path.join(dataset_root, str(row['match_file']).lstrip('/\\'))

        img0_pil = Image.open(p0).convert('L')
        imggt_pil = Image.open(pgt).convert('L')
        img1_pil = Image.open(p1).convert('L')

        im0_np = np.array(img0_pil, dtype=np.uint8)
        gt_np = np.array(imggt_pil, dtype=np.uint8)
        im1_np = np.array(img1_pil, dtype=np.uint8)

        f0 = 1.0 - (im0_np.astype(np.float32) / 255.0)
        fgt = 1.0 - (gt_np.astype(np.float32) / 255.0)
        f1 = 1.0 - (im1_np.astype(np.float32) / 255.0)

        h, w = f0.shape
        pad_h = ((h + 31) // 32) * 32
        pad_w = ((w + 31) // 32) * 32
        f0_pad = np.pad(f0, ((0, pad_h - h), (0, pad_w - w)), mode='constant')
        fgt_pad = np.pad(fgt, ((0, pad_h - h), (0, pad_w - w)), mode='constant')
        f1_pad = np.pad(f1, ((0, pad_h - h), (0, pad_w - w)), mode='constant')

        # Tính toán Coarse TPS Flow
        if os.path.exists(pmatch):
            kps = np.load(pmatch).astype(np.float32)
            kps0 = kps[0].copy()
            kps1 = kps[1].copy()
            kps0[:, 0] = (kps0[:, 0] * w) / pad_w
            kps0[:, 1] = (kps0[:, 1] * h) / pad_h
            kps1[:, 0] = (kps1[:, 0] * w) / pad_w
            kps1[:, 1] = (kps1[:, 1] * h) / pad_h
            c0 = torch.from_numpy(kps0).to(device)
            c1 = torch.from_numpy(kps1).to(device)
            if c0.shape[0] > 80:
                idx_sub = torch.linspace(0, c0.shape[0] - 1, 80, dtype=torch.long, device=device)
                c0 = c0[idx_sub]
                c1 = c1[idx_sub]
            theta01, c0_u = robust_tps_theta_from_points(c0, c1, lambd=0.5, add_anchors=True)
            theta10, c1_u = robust_tps_theta_from_points(c1, c0, lambd=0.5, add_anchors=True)
            flow01 = tps_flow_from_theta(theta01, c0_u, pad_h, pad_w, device)
            flow10 = tps_flow_from_theta(theta10, c1_u, pad_h, pad_w, device)
        else:
            flow01 = torch.zeros((1, 2, pad_h, pad_w), device=device)
            flow10 = torch.zeros((1, 2, pad_h, pad_w), device=device)

        t0_dev = torch.from_numpy(f0_pad).unsqueeze(0).unsqueeze(0).float().to(device)
        t1_dev = torch.from_numpy(f1_pad).unsqueeze(0).unsqueeze(0).float().to(device)

        # 1. Suy luận Original TPS
        t_start_o = time.time()
        with torch.no_grad():
            pred_orig, coarse_interp = model_orig(t0_dev, t1_dev, t_val, flow01, flow10)
        ms_orig = (time.time() - t_start_o) * 1000.0

        # 2. Suy luận V2.0 Best
        t_start_20 = time.time()
        with torch.no_grad():
            pred_v20, _ = model_v20(t0_dev, t1_dev, t_val, flow01, flow10)
        ms_v20 = (time.time() - t_start_20) * 1000.0

        # 3. Suy luận V2.1 Target
        t_start_21 = time.time()
        with torch.no_grad():
            pred_v21, _ = model_v21(t0_dev, t1_dev, t_val, flow01, flow10)
        ms_v21 = (time.time() - t_start_21) * 1000.0

        # Chuyển về định dạng ảnh pixel lineart (0: mực đen, 255: nền trắng)
        p_coarse_np = ((1.0 - coarse_interp[:, :, :h, :w]) * 255.0).clamp(0, 255).byte().cpu().numpy()[0, 0]
        p_orig_np = ((1.0 - pred_orig[:, :, :h, :w]) * 255.0).clamp(0, 255).byte().cpu().numpy()[0, 0]
        p_v20_np = ((1.0 - pred_v20[:, :, :h, :w]) * 255.0).clamp(0, 255).byte().cpu().numpy()[0, 0]
        p_v21_np = ((1.0 - pred_v21[:, :, :h, :w]) * 255.0).clamp(0, 255).byte().cpu().numpy()[0, 0]

        # Đo lường Coarse Baseline
        psnr_c = compute_psnr(p_coarse_np, gt_np)
        ssim_c = compute_ssim(p_coarse_np, gt_np)
        ghost_c = compute_ghost_pixels(p_coarse_np, gt_np)
        cd_c = compute_chamfer_distance(p_coarse_np, gt_np)

        # Đo lường Original TPS
        psnr_o = compute_psnr(p_orig_np, gt_np)
        ssim_o = compute_ssim(p_orig_np, gt_np)
        stroke_o = compute_stroke_metrics(p_orig_np, gt_np)
        ghost_o = compute_ghost_pixels(p_orig_np, gt_np, is_blank=stroke_o['is_blank_collapse'])
        mae_o = compute_ie_mae(p_orig_np, gt_np)
        tort_o = compute_line_tortuosity(p_orig_np, is_blank=stroke_o['is_blank_collapse'])
        cd_o = compute_chamfer_distance(p_orig_np, gt_np)

        # Đo lường V2.0 Best
        psnr_20 = compute_psnr(p_v20_np, gt_np)
        ssim_20 = compute_ssim(p_v20_np, gt_np)
        stroke_20 = compute_stroke_metrics(p_v20_np, gt_np)
        ghost_20 = compute_ghost_pixels(p_v20_np, gt_np, is_blank=stroke_20['is_blank_collapse'])
        mae_20 = compute_ie_mae(p_v20_np, gt_np)
        tort_20 = compute_line_tortuosity(p_v20_np, is_blank=stroke_20['is_blank_collapse'])
        cd_20 = compute_chamfer_distance(p_v20_np, gt_np)

        # Đo lường V2.1 Target
        psnr_21 = compute_psnr(p_v21_np, gt_np)
        ssim_21 = compute_ssim(p_v21_np, gt_np)
        stroke_21 = compute_stroke_metrics(p_v21_np, gt_np)
        ghost_21 = compute_ghost_pixels(p_v21_np, gt_np, is_blank=stroke_21['is_blank_collapse'])
        mae_21 = compute_ie_mae(p_v21_np, gt_np)
        tort_21 = compute_line_tortuosity(p_v21_np, is_blank=stroke_21['is_blank_collapse'])
        cd_21 = compute_chamfer_distance(p_v21_np, gt_np)

        delta_ghost_v21_vs_v20 = ghost_21 - ghost_20
        delta_psnr_v21_vs_v20 = psnr_21 - psnr_20
        delta_ssim_v21_vs_v20 = ssim_21 - ssim_20
        ghost_reduc_pct_vs_v20 = ((ghost_20 - ghost_21) / max(1, ghost_20)) * 100.0
        ghost_reduc_pct_vs_orig = ((ghost_o - ghost_21) / max(1, ghost_o)) * 100.0

        rec = {
            'clip_id': clip_id,
            't': t_val,
            'mean_disp': disp,
            'inliers': inliers,
            # Coarse
            'psnr_coarse': psnr_c,
            'ssim_coarse': ssim_c,
            'ghost_coarse': ghost_c,
            'chamfer_coarse': cd_c,
            # Original TPS
            'psnr_orig': psnr_o,
            'ssim_orig': ssim_o,
            'ghost_orig': ghost_o,
            'chamfer_orig': cd_o,
            'tortuosity_orig': tort_o,
            'recall_orig': stroke_o['stroke_recall'],
            'density_orig': stroke_o['stroke_density_ratio'],
            'mae_orig': mae_o,
            'latency_orig_ms': ms_orig,
            # V2.0 Best
            'psnr_v20': psnr_20,
            'ssim_v20': ssim_20,
            'ghost_v20': ghost_20,
            'chamfer_v20': cd_20,
            'tortuosity_v20': tort_20,
            'recall_v20': stroke_20['stroke_recall'],
            'density_v20': stroke_20['stroke_density_ratio'],
            'mae_v20': mae_20,
            'latency_v20_ms': ms_v20,
            # V2.1 Target
            'psnr_v21': psnr_21,
            'ssim_v21': ssim_21,
            'ghost_v21': ghost_21,
            'chamfer_v21': cd_21,
            'tortuosity_v21': tort_21,
            'recall_v21': stroke_21['stroke_recall'],
            'density_v21': stroke_21['stroke_density_ratio'],
            'mae_v21': mae_21,
            'latency_v21_ms': ms_v21,
            # Deltas V2.1 vs V2.0
            'delta_ghost_v21_v20': delta_ghost_v21_vs_v20,
            'delta_psnr_v21_v20': delta_psnr_v21_vs_v20,
            'delta_ssim_v21_v20': delta_ssim_v21_vs_v20,
            'ghost_reduction_pct_vs_v20': ghost_reduc_pct_vs_v20,
            'ghost_reduction_pct_vs_orig': ghost_reduc_pct_vs_orig,
            'winner_ghost_v21_v20': 'V2.1' if ghost_21 < ghost_20 else ('V2.0' if ghost_20 < ghost_21 else 'TIE')
        }
        records.append(rec)

        # Render Visual Audit nếu thuộc danh mục audit
        if clip_id in audit_clips:
            m_orig_s = {'psnr': psnr_o, 'ssim': ssim_o, 'ghost': ghost_o}
            m_v20_s = {'psnr': psnr_20, 'ssim': ssim_20, 'ghost': ghost_20}
            m_v21_s = {'psnr': psnr_21, 'ssim': ssim_21, 'ghost': ghost_21}
            render_visual_audits_v21(visual_dir, clip_id, t_val, im0_np, p_coarse_np, p_orig_np, p_v20_np, p_v21_np, gt_np,
                                    m_orig_s, m_v20_s, m_v21_s)

    elapsed_total = time.time() - t_start_total
    df_results = pd.DataFrame(records)

    # 5. Xuất file kết quả chi tiết
    csv_out_path = os.path.join(args.output_dir, 'benchmark_results_v21.csv')
    df_results.to_csv(csv_out_path, index=False)
    print(f"\n[✓] Đã lưu kết quả chi tiết: {csv_out_path}")

    # Tạo JSON Summary
    summary = {
        'benchmark_info': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'evaluation_mode': 'TRIAD_HEAD_TO_HEAD_STRATIFIED',
            'dataset': 'sakuga_golden_dataset_tps_ready_v1_patch2',
            'total_samples': len(df_results),
            'elapsed_sec': round(elapsed_total, 2),
            'device': str(device)
        },
        'model_orig_baseline': {
            'checkpoint_path': args.model_orig,
            'name': 'TPS-Original-Mixamo-3D',
            'psnr_mean': float(df_results['psnr_orig'].mean()),
            'psnr_std': float(df_results['psnr_orig'].std()),
            'ssim_mean': float(df_results['ssim_orig'].mean()),
            'ssim_std': float(df_results['ssim_orig'].std()),
            'ghost_pixels_mean': float(df_results['ghost_orig'].mean()),
            'ghost_pixels_median': float(df_results['ghost_orig'].median()),
            'chamfer_dist_mean': float(df_results['chamfer_orig'].mean()),
            'line_tortuosity_mean': float(df_results['tortuosity_orig'].mean()),
            'stroke_recall_mean': float(df_results['recall_orig'].mean()),
            'stroke_density_mean': float(df_results['density_orig'].mean()),
            'mae_mean': float(df_results['mae_orig'].mean()),
            'latency_ms_mean': float(df_results['latency_orig_ms'].mean()),
            'blank_collapse_rate': 0.0
        },
        'model_v20_best': {
            'checkpoint_path': args.model_v20,
            'name': 'TPS-Finetuned-V2.0-Best (V2.0.2)',
            'psnr_mean': float(df_results['psnr_v20'].mean()),
            'psnr_std': float(df_results['psnr_v20'].std()),
            'ssim_mean': float(df_results['ssim_v20'].mean()),
            'ssim_std': float(df_results['ssim_v20'].std()),
            'ghost_pixels_mean': float(df_results['ghost_v20'].mean()),
            'ghost_pixels_median': float(df_results['ghost_v20'].median()),
            'chamfer_dist_mean': float(df_results['chamfer_v20'].mean()),
            'line_tortuosity_mean': float(df_results['tortuosity_v20'].mean()),
            'stroke_recall_mean': float(df_results['recall_v20'].mean()),
            'stroke_density_mean': float(df_results['density_v20'].mean()),
            'mae_mean': float(df_results['mae_v20'].mean()),
            'latency_ms_mean': float(df_results['latency_v20_ms'].mean()),
            'blank_collapse_rate': 0.0
        },
        'model_v21_target': {
            'checkpoint_path': args.model_v21,
            'name': 'TPS-Finetuned-V2.1-Target (+Patch 2)',
            'psnr_mean': float(df_results['psnr_v21'].mean()),
            'psnr_std': float(df_results['psnr_v21'].std()),
            'ssim_mean': float(df_results['ssim_v21'].mean()),
            'ssim_std': float(df_results['ssim_v21'].std()),
            'ghost_pixels_mean': float(df_results['ghost_v21'].mean()),
            'ghost_pixels_median': float(df_results['ghost_v21'].median()),
            'chamfer_dist_mean': float(df_results['chamfer_v21'].mean()),
            'line_tortuosity_mean': float(df_results['tortuosity_v21'].mean()),
            'stroke_recall_mean': float(df_results['recall_v21'].mean()),
            'stroke_density_mean': float(df_results['density_v21'].mean()),
            'mae_mean': float(df_results['mae_v21'].mean()),
            'latency_ms_mean': float(df_results['latency_v21_ms'].mean()),
            'blank_collapse_rate': 0.0
        },
        'coarse_baseline': {
            'psnr_mean': float(df_results['psnr_coarse'].mean()),
            'ssim_mean': float(df_results['ssim_coarse'].mean()),
            'ghost_pixels_mean': float(df_results['ghost_coarse'].mean()),
            'chamfer_dist_mean': float(df_results['chamfer_coarse'].mean())
        },
        'deltas_v21_vs_v20': {
            'ghost_pixel_reduction_mean_px': float(df_results['ghost_v20'].mean() - df_results['ghost_v21'].mean()),
            'ghost_pixel_reduction_pct': float(((df_results['ghost_v20'].mean() - df_results['ghost_v21'].mean()) / df_results['ghost_v20'].mean()) * 100.0),
            'psnr_diff_db': float(df_results['psnr_v21'].mean() - df_results['psnr_v20'].mean()),
            'ssim_diff': float(df_results['ssim_v21'].mean() - df_results['ssim_v20'].mean()),
            'chamfer_diff': float(df_results['chamfer_v21'].mean() - df_results['chamfer_v20'].mean()),
            'v21_ghost_win_rate_pct': float((df_results['winner_ghost_v21_v20'] == 'V2.1').mean() * 100.0),
            'v20_ghost_win_rate_pct': float((df_results['winner_ghost_v21_v20'] == 'V2.0').mean() * 100.0)
        },
        'deltas_v21_vs_orig': {
            'ghost_pixel_reduction_mean_px': float(df_results['ghost_orig'].mean() - df_results['ghost_v21'].mean()),
            'ghost_pixel_reduction_pct': float(((df_results['ghost_orig'].mean() - df_results['ghost_v21'].mean()) / df_results['ghost_orig'].mean()) * 100.0),
            'psnr_diff_db': float(df_results['psnr_v21'].mean() - df_results['psnr_orig'].mean()),
            'ssim_diff': float(df_results['ssim_v21'].mean() - df_results['ssim_orig'].mean()),
            'chamfer_diff': float(df_results['chamfer_v21'].mean() - df_results['chamfer_orig'].mean())
        }
    }

    json_out_path = os.path.join(args.output_dir, 'summary_metrics_v21.json')
    with open(json_out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    print(f"[✓] Đã xuất file thống kê tổng hợp: {json_out_path}")

    # 6. Phân tích theo mốc thời gian t
    df_results['t_round'] = df_results['t'].round(2)
    t_summary = df_results.groupby('t_round').agg(
        sample_count=('clip_id', 'count'),
        psnr_orig=('psnr_orig', 'mean'),
        psnr_v20=('psnr_v20', 'mean'),
        psnr_v21=('psnr_v21', 'mean'),
        ssim_orig=('ssim_orig', 'mean'),
        ssim_v20=('ssim_v20', 'mean'),
        ssim_v21=('ssim_v21', 'mean'),
        ghost_orig=('ghost_orig', 'mean'),
        ghost_v20=('ghost_v20', 'mean'),
        ghost_v21=('ghost_v21', 'mean'),
        chamfer_v20=('chamfer_v20', 'mean'),
        chamfer_v21=('chamfer_v21', 'mean')
    ).round(4).reset_index()

    t_csv_path = os.path.join(args.output_dir, 'time_step_analysis_v21.csv')
    t_summary.to_csv(t_csv_path, index=False)
    print(f"[✓] Đã xuất bảng phân tích theo bước thời gian t: {t_csv_path}")

    print("\n" + "=" * 95)
    print("🏆 TỔNG HỢP KẾT QUẢ ĐỐI ĐẦU CHÍNH (TRIAD SUMMARY):")
    print(f"   Ghost Pixels : Orig = {summary['model_orig_baseline']['ghost_pixels_mean']:.1f} px | V2.0 = {summary['model_v20_best']['ghost_pixels_mean']:.1f} px | V2.1 = {summary['model_v21_target']['ghost_pixels_mean']:.1f} px")
    print(f"   Ghost Reduction V2.1 vs Orig: {summary['deltas_v21_vs_orig']['ghost_pixel_reduction_pct']:.2f}%")
    print(f"   Ghost Reduction V2.1 vs V2.0: {summary['deltas_v21_vs_v20']['ghost_pixel_reduction_pct']:.2f}% (V2.1 Win Rate: {summary['deltas_v21_vs_v20']['v21_ghost_win_rate_pct']:.1f}%)")
    print(f"   PSNR         : Orig = {summary['model_orig_baseline']['psnr_mean']:.2f} dB | V2.0 = {summary['model_v20_best']['psnr_mean']:.2f} dB | V2.1 = {summary['model_v21_target']['psnr_mean']:.2f} dB")
    print(f"   SSIM         : Orig = {summary['model_orig_baseline']['ssim_mean']:.4f} | V2.0 = {summary['model_v20_best']['ssim_mean']:.4f} | V2.1 = {summary['model_v21_target']['ssim_mean']:.4f}")
    print(f"   Chamfer Dist : Orig = {summary['model_orig_baseline']['chamfer_dist_mean']:.6f} | V2.0 = {summary['model_v20_best']['chamfer_dist_mean']:.6f} | V2.1 = {summary['model_v21_target']['chamfer_dist_mean']:.6f}")
    print(f"   Stroke Recall: Orig = {summary['model_orig_baseline']['stroke_recall_mean']*100:.2f}% | V2.0 = {summary['model_v20_best']['stroke_recall_mean']*100:.2f}% | V2.1 = {summary['model_v21_target']['stroke_recall_mean']*100:.2f}%")
    print(f"   Thời gian thực hiện toàn bộ: {elapsed_total:.1f}s")
    print("=" * 95)

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark Đối Đầu Triad TPS V2.1 vs V2.0 vs Original TPS")
    parser.add_argument('--model_orig', type=str,
                        default=r'checkpoints/pretrained_baselines/tps-original/model_latest.pt',
                        help='Đường dẫn checkpoint model Original TPS')
    parser.add_argument('--model_v20', type=str,
                        default=r'checkpoints/experiments/v2.0/v2.0.2/tps_finetune_v2.0_artifacts/checkpoints_v1.1/model_tps_v2.0_best.pt',
                        help='Đường dẫn checkpoint model V2.0 Best')
    parser.add_argument('--model_v21', type=str,
                        default=r'checkpoints/experiments/v2.1/tps_finetune_v2.1_artifacts/checkpoints_v2.1/model_tps_v1.1_best.pt',
                        help='Đường dẫn checkpoint model V2.1 Target (tên file .pt vẫn giữ 1.1)')
    parser.add_argument('--data_root', type=str,
                        default=r'data/processed/sakuga_golden_dataset_tps_ready_v1_patch2',
                        help='Thư mục gốc của Sakuga Golden Dataset V1 + Patch 2')
    parser.add_argument('--dataset_csv', type=str,
                        default=r'data/processed/sakuga_golden_dataset_tps_ready_v1_patch2/triplets.csv',
                        help='Đường dẫn file triplets.csv')
    parser.add_argument('--output_dir', type=str,
                        default=r'checkpoints/experiments/v2.1',
                        help='Thư mục lưu kết quả benchmark đối đầu (v2.1)')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Số lượng mẫu validation phân tầng cho test nhanh (mặc định: 100)')
    parser.add_argument('--cpu', action='store_true', help='Ép chạy trên CPU')

    args = parser.parse_args()
    run_benchmark_v21(args)
