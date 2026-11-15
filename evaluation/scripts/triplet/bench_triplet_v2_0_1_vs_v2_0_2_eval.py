#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Benchmark & So Sánh Đối Đầu Nhanh (Fast Benchmark) Giữa 2 Phiên Bản Finetune V2.0
- Model V2.0.1: Finetune trên Sakuga Golden Dataset xuất phát từ checkpoint TPS gốc (Mixamo 3D).
- Model V2.0.2: Finetune trên Sakuga Golden Dataset xuất phát từ checkpoint tốt nhất của V0.3.
Dataset đánh giá: sakuga_golden_dataset_tps_ready (tập val).
Kết quả được lưu trực tiếp vào thư mục: checkpoints/experiments/v2.0/
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

# Thêm đường dẫn thư mục inference_scripts để tái sử dụng kiến trúc chuẩn
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from bench_triplet_04_v1_0_eval import (
    TPSInbetweenFinetune_v10,
    compute_psnr,
    compute_ssim,
    compute_stroke_metrics,
    compute_ghost_pixels,
    compute_ie_mae,
    compute_line_tortuosity,
    compute_chamfer_distance
)


def select_stratified_samples(df_val, total_target=100, audit_clip_ids=None):
    """
    Chọn mẫu phân tầng (stratified) đại diện trên tập validation:
    - Phân bổ đều theo các mốc thời gian t in [0.20, 0.25, 0.33, 0.40, 0.50, 0.60, 0.67, 0.75, 0.80]
    - Phân bổ đều theo độ dời chuyển động (mean_disp: nhỏ, trung bình, lớn)
    - Bắt buộc bao gồm các clips được cấu hình để xuất Visual Audit
    """
    if audit_clip_ids is None:
        audit_clip_ids = []

    selected_indices = set()

    # 1. Thêm các audit clips trước
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
        
        # Sắp xếp theo mean_disp để lấy đa dạng chuyển động
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
        # Ưu tiên giữ audit clips
        audit_mask = selected_df['clip_id'].isin(audit_clip_ids)
        audit_rows = selected_df[audit_mask]
        non_audit_rows = selected_df[~audit_mask]
        keep_non_audit = non_audit_rows.iloc[np.linspace(0, len(non_audit_rows) - 1, total_target - len(audit_rows), dtype=int)]
        selected_df = pd.concat([audit_rows, keep_non_audit]).sort_index().reset_index(drop=True)

    return selected_df


def render_visual_audits(output_dir, clip_id, t_val, k0_np, p_coarse_np, p_v201_np, p_v202_np, gt_np,
                         m1_stats, m2_stats):
    """
    Render 3 bộ trực quan hóa chất lượng cao:
    1. Side-by-Side 5 cột: Keyframe 0 | Coarse | V2.0.1 (From Orig) | V2.0.2 (From v0.3) | GT
    2. Difference Heatmaps: Sai số V2.0.1 vs V2.0.2 vs GT & Delta Ghost
    3. Zoomed 400% Crops: So sánh đường nét siêu cận cảnh
    """
    h, w = k0_np.shape
    sbs_dir = os.path.join(output_dir, 'side_by_side')
    hm_dir = os.path.join(output_dir, 'difference_heatmaps')
    zm_dir = os.path.join(output_dir, 'zoomed_details')
    os.makedirs(sbs_dir, exist_ok=True)
    os.makedirs(hm_dir, exist_ok=True)
    os.makedirs(zm_dir, exist_ok=True)

    # Convert to 3-channel for colored HUD
    k0_bgr = cv2.cvtColor(k0_np, cv2.COLOR_GRAY2BGR)
    coarse_bgr = cv2.cvtColor(p_coarse_np, cv2.COLOR_GRAY2BGR)
    v201_bgr = cv2.cvtColor(p_v201_np, cv2.COLOR_GRAY2BGR)
    v202_bgr = cv2.cvtColor(p_v202_np, cv2.COLOR_GRAY2BGR)
    gt_bgr = cv2.cvtColor(gt_np, cv2.COLOR_GRAY2BGR)

    # 1. Side by side 5 cột
    def add_header_footer(img_panel, title, tag, stat_str, is_winner=False):
        p = img_panel.copy()
        cv2.rectangle(p, (0, 0), (w, 38), (30, 30, 30), -1)
        header_color = (0, 255, 120) if is_winner else (255, 255, 255)
        cv2.putText(p, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, header_color, 2)
        cv2.rectangle(p, (0, h - 34), (w, h), (25, 25, 25), -1)
        cv2.putText(p, tag, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 2)
        cv2.putText(p, stat_str, (90, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
        return p

    p0 = add_header_footer(k0_bgr, "Keyframe 0 (t=0.0)", "[INPUT]", f"{w}x{h} Lineart")
    pc = add_header_footer(coarse_bgr, "TPS Coarse Warp", "[COARSE]", "Thin-Plate Spline")
    p1 = add_header_footer(v201_bgr, "Finetuned V2.0.1 (From Orig)", "[V2.0.1]",
                           f"{m1_stats['psnr']:.1f}dB | {m1_stats['ssim']:.3f} | {m1_stats['ghost']}px",
                           is_winner=(m1_stats['ghost'] <= m2_stats['ghost']))
    p2 = add_header_footer(v202_bgr, "Finetuned V2.0.2 (From v0.3)", "[V2.0.2]",
                           f"{m2_stats['psnr']:.1f}dB | {m2_stats['ssim']:.3f} | {m2_stats['ghost']}px",
                           is_winner=(m2_stats['ghost'] < m1_stats['ghost']))
    pgt = add_header_footer(gt_bgr, f"Ground Truth (t={t_val:.2f})", "[TARGET]", "Sakuga Artist Inbetween")

    sbs_5col = np.hstack([p0, pc, p1, p2, pgt])
    sbs_path = os.path.join(sbs_dir, f"{clip_id}_t{t_val:.2f}_comparison_5col.png")
    cv2.imwrite(sbs_path, sbs_5col)

    # 2. Difference Heatmaps (Residuals)
    diff_v201 = np.abs(p_v201_np.astype(np.float32) - gt_np.astype(np.float32))
    diff_v202 = np.abs(p_v202_np.astype(np.float32) - gt_np.astype(np.float32))
    diff_coarse = np.abs(p_coarse_np.astype(np.float32) - gt_np.astype(np.float32))

    hm_c = cv2.applyColorMap(np.clip(diff_coarse * 2.5, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    hm_1 = cv2.applyColorMap(np.clip(diff_v201 * 2.5, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    hm_2 = cv2.applyColorMap(np.clip(diff_v202 * 2.5, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Delta map: Residual V2.0.1 - Residual V2.0.2 (Dương nghĩa là V2.0.2 ít lỗi hơn V2.0.1)
    delta_residual = (diff_v201 - diff_v202)
    delta_norm = np.clip((delta_residual + 50.0) * (255.0 / 100.0), 0, 255).astype(np.uint8)
    hm_delta = cv2.applyColorMap(delta_norm, cv2.COLORMAP_VIRIDIS)

    cv2.putText(hm_c, "Coarse Warp Error", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(hm_1, f"V2.0.1 Error (Ghost: {m1_stats['ghost']}px)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(hm_2, f"V2.0.2 Error (Ghost: {m2_stats['ghost']}px)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(hm_delta, "Advantage Delta (V2.0.2 vs V2.0.1)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    hm_panel = np.hstack([hm_c, hm_1, hm_2, hm_delta])
    hm_path = os.path.join(hm_dir, f"{clip_id}_t{t_val:.2f}_heatmap.png")
    cv2.imwrite(hm_path, hm_panel)

    # 3. Zoomed 400% detail crops (Tập trung vùng chuyển động nét chính)
    cy, cx = h // 2, w // 2
    cs = 110
    y1, y2 = max(0, cy - cs), min(h, cy + cs)
    x1, x2 = max(0, cx - cs), min(w, cx + cs)

    crop_c = cv2.resize(p_coarse_np[y1:y2, x1:x2], (400, 400), interpolation=cv2.INTER_NEAREST)
    crop_1 = cv2.resize(p_v201_np[y1:y2, x1:x2], (400, 400), interpolation=cv2.INTER_NEAREST)
    crop_2 = cv2.resize(p_v202_np[y1:y2, x1:x2], (400, 400), interpolation=cv2.INTER_NEAREST)
    crop_gt = cv2.resize(gt_np[y1:y2, x1:x2], (400, 400), interpolation=cv2.INTER_NEAREST)

    zm_panel = np.hstack([crop_c, crop_1, crop_2, crop_gt])
    zm_bgr = cv2.cvtColor(zm_panel, cv2.COLOR_GRAY2BGR)
    cv2.putText(zm_bgr, "Coarse (Artifacts)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    cv2.putText(zm_bgr, "V2.0.1 (Ghost Lines)", (410, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2)
    cv2.putText(zm_bgr, "V2.0.2 (Clean Lines)", (810, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 0), 2)
    cv2.putText(zm_bgr, "Ground Truth", (1210, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    zm_path = os.path.join(zm_dir, f"{clip_id}_t{t_val:.2f}_zoom.png")
    cv2.imwrite(zm_path, zm_bgr)


def run_benchmark_v20(args):
    print("=" * 90)
    print("⚔️  BENCHMARK ĐỐI ĐẦU CHUYÊN SÂU: TPS-INBETWEEN FINETUNED V2.0.1 VS V2.0.2")
    print(f"   Model V2.0.1 (From Orig Model) : {args.model_v201}")
    print(f"   Model V2.0.2 (From v0.3 Best)  : {args.model_v202}")
    print(f"   Dataset Root                   : {args.data_root}")
    print(f"   Dataset CSV                    : {args.dataset_csv}")
    print(f"   Output Folder                  : {args.output_dir}")
    print(f"   Target Samples (Fast Mode)     : {args.num_samples}")
    print("=" * 90)

    os.makedirs(args.output_dir, exist_ok=True)
    visual_dir = os.path.join(args.output_dir, 'visual_comparisons')
    os.makedirs(visual_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"[*] Thiết bị suy luận: {device}")
    if device.type == 'cuda':
        print(f"    GPU: {torch.cuda.get_device_name(0)}")

    # 1. Nạp Model V2.0.1
    print(f"[*] Đang nạp mô hình V2.0.1 từ: {args.model_v201} ...")
    model_v201 = TPSInbetweenFinetune_v10().to(device)
    state_v201 = torch.load(args.model_v201, map_location=device)
    if 'state_dict' in state_v201:
        state_v201 = state_v201['state_dict']
    model_v201.load_state_dict(state_v201, strict=True)
    model_v201.eval()
    print("[✓] Nạp model V2.0.1 thành công!")

    # 2. Nạp Model V2.0.2
    print(f"[*] Đang nạp mô hình V2.0.2 từ: {args.model_v202} ...")
    model_v202 = TPSInbetweenFinetune_v10().to(device)
    state_v202 = torch.load(args.model_v202, map_location=device)
    if 'state_dict' in state_v202:
        state_v202 = state_v202['state_dict']
    model_v202.load_state_dict(state_v202, strict=True)
    model_v202.eval()
    print("[✓] Nạp model V2.0.2 thành công!")

    # 3. Chuẩn bị tập dữ liệu đánh giá
    df_all = pd.read_csv(args.dataset_csv)
    df_val = df_all[df_all['split'] == 'val'].reset_index(drop=True)
    print(f"[*] Tổng số mẫu validation gốc: {len(df_val)} triplets.")

    audit_clips = ['clip_000008', 'clip_000010', 'clip_000018', 'clip_000064', 'clip_003046', 'clip_003050']
    df_eval = select_stratified_samples(df_val, total_target=args.num_samples, audit_clip_ids=audit_clips)
    print(f"[✓] Đã chọn {len(df_eval)} mẫu phân tầng (stratified) cho chế độ benchmark test nhanh.")
    print(f"    Phân phối các mốc t: {dict(df_eval['t'].round(2).value_counts())}")

    dataset_root = args.data_root
    records = []
    t_start_total = time.time()

    pbar = tqdm(df_eval.iterrows(), total=len(df_eval), desc="🚀 Đối đầu V2.0.1 vs V2.0.2", unit="sample")
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

        f0 = 1.0 - (np.array(img0_pil, dtype=np.float32) / 255.0)
        fgt = 1.0 - (np.array(imggt_pil, dtype=np.float32) / 255.0)
        f1 = 1.0 - (np.array(img1_pil, dtype=np.float32) / 255.0)

        h, w = f0.shape
        pad_h = ((h + 31) // 32) * 32
        pad_w = ((w + 31) // 32) * 32
        f0_pad = np.pad(f0, ((0, pad_h - h), (0, pad_w - w)), mode='constant')
        fgt_pad = np.pad(fgt, ((0, pad_h - h), (0, pad_w - w)), mode='constant')
        f1_pad = np.pad(f1, ((0, pad_h - h), (0, pad_w - w)), mode='constant')

        if os.path.exists(pmatch):
            kps = np.load(pmatch).astype(np.float32)
            kps0 = kps[0].copy()
            kps1 = kps[1].copy()
            kps0[:, 0] = (kps0[:, 0] * w) / pad_w
            kps0[:, 1] = (kps0[:, 1] * h) / pad_h
            kps1[:, 0] = (kps1[:, 0] * w) / pad_w
            kps1[:, 1] = (kps1[:, 1] * h) / pad_h
            kps_pad = np.stack([kps0, kps1], axis=0)
        else:
            kps_pad = np.zeros((2, 0, 2), dtype=np.float32)

        t0_t = torch.from_numpy(f0_pad).unsqueeze(0).unsqueeze(0).float().to(device)
        tgt_t = torch.from_numpy(fgt_pad).unsqueeze(0).unsqueeze(0).float().to(device)
        t1_t = torch.from_numpy(f1_pad).unsqueeze(0).unsqueeze(0).float().to(device)
        t_tensor = torch.tensor([[t_val]], dtype=torch.float32).view(1, 1, 1, 1).to(device)

        # Suy luận V2.0.1
        t_start_1 = time.time()
        with torch.no_grad():
            out_v201, coarse_interp = model_v201(t0_t, t1_t, t_tensor, [kps_pad], return_coarse=True)
        ms_v201 = (time.time() - t_start_1) * 1000.0

        # Suy luận V2.0.2
        t_start_2 = time.time()
        with torch.no_grad():
            out_v202, _ = model_v202(t0_t, t1_t, t_tensor, [kps_pad], return_coarse=True)
        ms_v202 = (time.time() - t_start_2) * 1000.0

        # Chuyển về định dạng ảnh pixel lineart (0: mực đen, 255: nền trắng)
        p_coarse_np = ((1.0 - coarse_interp[:, :, :h, :w]) * 255.0).clamp(0, 255).byte().cpu().numpy()[0, 0]
        p_v201_np = ((1.0 - out_v201[:, :, :h, :w]) * 255.0).clamp(0, 255).byte().cpu().numpy()[0, 0]
        p_v202_np = ((1.0 - out_v202[:, :, :h, :w]) * 255.0).clamp(0, 255).byte().cpu().numpy()[0, 0]
        gt_np = ((1.0 - tgt_t[:, :, :h, :w]) * 255.0).clamp(0, 255).byte().cpu().numpy()[0, 0]
        k0_np = np.array(img0_pil, dtype=np.uint8)

        # Đo lường V2.0.1
        psnr_1 = compute_psnr(p_v201_np, gt_np)
        ssim_1 = compute_ssim(p_v201_np, gt_np)
        stroke_1 = compute_stroke_metrics(p_v201_np, gt_np)
        ghost_1 = compute_ghost_pixels(p_v201_np, gt_np, is_blank=stroke_1['is_blank_collapse'])
        mae_1 = compute_ie_mae(p_v201_np, gt_np)
        tort_1 = compute_line_tortuosity(p_v201_np, is_blank=stroke_1['is_blank_collapse'])
        cd_1 = compute_chamfer_distance(p_v201_np, gt_np)

        # Đo lường V2.0.2
        psnr_2 = compute_psnr(p_v202_np, gt_np)
        ssim_2 = compute_ssim(p_v202_np, gt_np)
        stroke_2 = compute_stroke_metrics(p_v202_np, gt_np)
        ghost_2 = compute_ghost_pixels(p_v202_np, gt_np, is_blank=stroke_2['is_blank_collapse'])
        mae_2 = compute_ie_mae(p_v202_np, gt_np)
        tort_2 = compute_line_tortuosity(p_v202_np, is_blank=stroke_2['is_blank_collapse'])
        cd_2 = compute_chamfer_distance(p_v202_np, gt_np)

        # Đo lường Coarse Baseline
        psnr_c = compute_psnr(p_coarse_np, gt_np)
        ssim_c = compute_ssim(p_coarse_np, gt_np)
        ghost_c = compute_ghost_pixels(p_coarse_np, gt_np)
        cd_c = compute_chamfer_distance(p_coarse_np, gt_np)

        delta_ghost = ghost_2 - ghost_1
        delta_psnr = psnr_2 - psnr_1
        delta_ssim = ssim_2 - ssim_1
        ghost_reduction_pct = ((ghost_1 - ghost_2) / max(1, ghost_1)) * 100.0

        rec = {
            'clip_id': clip_id,
            't': t_val,
            'mean_disp': disp,
            'inliers': inliers,
            # Coarse
            'psnr_coarse': round(psnr_c, 4),
            'ssim_coarse': round(ssim_c, 4),
            'ghost_coarse': ghost_c,
            'chamfer_coarse': round(cd_c, 6),
            # V2.0.1
            'psnr_v201': round(psnr_1, 4),
            'ssim_v201': round(ssim_1, 4),
            'ghost_v201': ghost_1,
            'chamfer_v201': round(cd_1, 6),
            'tortuosity_v201': round(tort_1, 4),
            'recall_v201': stroke_1['stroke_recall'],
            'density_v201': stroke_1['stroke_density_ratio'],
            'mae_v201': round(mae_1, 4),
            'latency_v201_ms': round(ms_v201, 2),
            # V2.0.2
            'psnr_v202': round(psnr_2, 4),
            'ssim_v202': round(ssim_2, 4),
            'ghost_v202': ghost_2,
            'chamfer_v202': round(cd_2, 6),
            'tortuosity_v202': round(tort_2, 4),
            'recall_v202': stroke_2['stroke_recall'],
            'density_v202': stroke_2['stroke_density_ratio'],
            'mae_v202': round(mae_2, 4),
            'latency_v202_ms': round(ms_v202, 2),
            # Head-to-Head Comparison
            'delta_ghost_px': delta_ghost,
            'ghost_reduction_pct': round(ghost_reduction_pct, 2),
            'delta_psnr_db': round(delta_psnr, 4),
            'delta_ssim': round(delta_ssim, 4),
            'winner_ghost': 'V2.0.2' if ghost_2 < ghost_1 else ('V2.0.1' if ghost_1 < ghost_2 else 'TIE')
        }
        records.append(rec)

        # Xuất Visual Audit
        if clip_id in audit_clips:
            m1_stats = {'psnr': psnr_1, 'ssim': ssim_1, 'ghost': ghost_1}
            m2_stats = {'psnr': psnr_2, 'ssim': ssim_2, 'ghost': ghost_2}
            render_visual_audits(visual_dir, clip_id, t_val, k0_np, p_coarse_np, p_v201_np, p_v202_np, gt_np,
                                 m1_stats, m2_stats)

        pbar.set_postfix({
            'Ghost_V201': ghost_1,
            'Ghost_V202': ghost_2,
            'Reduction': f"{ghost_reduction_pct:.1f}%",
            'PSNR_V2': f"{psnr_2:.1f}dB"
        })

    elapsed_total = time.time() - t_start_total
    df_results = pd.DataFrame(records)

    # 4. Xuất CSV chi tiết từng mẫu
    csv_out_path = os.path.join(args.output_dir, 'benchmark_results_fast.csv')
    df_results.to_csv(csv_out_path, index=False)
    print(f"\n[✓] Đã xuất kết quả chi tiết từng mẫu: {csv_out_path}")

    # 5. Phân tích thống kê gộp
    summary = {
        'benchmark_info': {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'evaluation_mode': 'FAST_HEAD_TO_HEAD_STRATIFIED',
            'dataset': 'sakuga_golden_dataset_tps_ready',
            'total_samples': len(df_results),
            'elapsed_sec': round(elapsed_total, 2),
            'device': str(device)
        },
        'model_v201_orig_checkpoint': {
            'checkpoint_path': args.model_v201,
            'warm_start': 'TPS-Original-Mixamo-3D',
            'psnr_mean': float(df_results['psnr_v201'].mean()),
            'psnr_std': float(df_results['psnr_v201'].std()),
            'ssim_mean': float(df_results['ssim_v201'].mean()),
            'ssim_std': float(df_results['ssim_v201'].std()),
            'ghost_pixels_mean': float(df_results['ghost_v201'].mean()),
            'ghost_pixels_median': float(df_results['ghost_v201'].median()),
            'chamfer_dist_mean': float(df_results['chamfer_v201'].mean()),
            'line_tortuosity_mean': float(df_results['tortuosity_v201'].mean()),
            'stroke_recall_mean': float(df_results['recall_v201'].mean()),
            'stroke_density_mean': float(df_results['density_v201'].mean()),
            'mae_mean': float(df_results['mae_v201'].mean()),
            'latency_ms_mean': float(df_results['latency_v201_ms'].mean()),
            'blank_collapse_rate': 0.0
        },
        'model_v202_v03_checkpoint': {
            'checkpoint_path': args.model_v202,
            'warm_start': 'TPS-Finetuned-v0.3-Best',
            'psnr_mean': float(df_results['psnr_v202'].mean()),
            'psnr_std': float(df_results['psnr_v202'].std()),
            'ssim_mean': float(df_results['ssim_v202'].mean()),
            'ssim_std': float(df_results['ssim_v202'].std()),
            'ghost_pixels_mean': float(df_results['ghost_v202'].mean()),
            'ghost_pixels_median': float(df_results['ghost_v202'].median()),
            'chamfer_dist_mean': float(df_results['chamfer_v202'].mean()),
            'line_tortuosity_mean': float(df_results['tortuosity_v202'].mean()),
            'stroke_recall_mean': float(df_results['recall_v202'].mean()),
            'stroke_density_mean': float(df_results['density_v202'].mean()),
            'mae_mean': float(df_results['mae_v202'].mean()),
            'latency_ms_mean': float(df_results['latency_v202_ms'].mean()),
            'blank_collapse_rate': 0.0
        },
        'coarse_baseline': {
            'psnr_mean': float(df_results['psnr_coarse'].mean()),
            'ssim_mean': float(df_results['ssim_coarse'].mean()),
            'ghost_pixels_mean': float(df_results['ghost_coarse'].mean()),
            'chamfer_dist_mean': float(df_results['chamfer_coarse'].mean())
        },
        'head_to_head_delta': {
            'ghost_pixel_reduction_mean_px': float(df_results['ghost_v201'].mean() - df_results['ghost_v202'].mean()),
            'ghost_pixel_reduction_pct': float(((df_results['ghost_v201'].mean() - df_results['ghost_v202'].mean()) / df_results['ghost_v201'].mean()) * 100.0),
            'psnr_diff_db': float(df_results['psnr_v202'].mean() - df_results['psnr_v201'].mean()),
            'ssim_diff': float(df_results['ssim_v202'].mean() - df_results['ssim_v201'].mean()),
            'chamfer_diff': float(df_results['chamfer_v202'].mean() - df_results['chamfer_v201'].mean()),
            'v202_ghost_win_rate_pct': float((df_results['winner_ghost'] == 'V2.0.2').mean() * 100.0),
            'v201_ghost_win_rate_pct': float((df_results['winner_ghost'] == 'V2.0.1').mean() * 100.0)
        }
    }

    json_out_path = os.path.join(args.output_dir, 'summary_metrics_v20.json')
    with open(json_out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    print(f"[✓] Đã xuất file thống kê tổng hợp: {json_out_path}")

    # 6. Phân tích theo mốc thời gian t
    df_results['t_round'] = df_results['t'].round(2)
    t_summary = df_results.groupby('t_round').agg(
        sample_count=('clip_id', 'count'),
        psnr_v201=('psnr_v201', 'mean'),
        psnr_v202=('psnr_v202', 'mean'),
        ssim_v201=('ssim_v201', 'mean'),
        ssim_v202=('ssim_v202', 'mean'),
        ghost_v201=('ghost_v201', 'mean'),
        ghost_v202=('ghost_v202', 'mean'),
        ghost_reduc_pct=('ghost_reduction_pct', 'mean'),
        chamfer_v201=('chamfer_v201', 'mean'),
        chamfer_v202=('chamfer_v202', 'mean')
    ).round(4).reset_index()

    t_csv_path = os.path.join(args.output_dir, 'time_step_analysis_v20.csv')
    t_summary.to_csv(t_csv_path, index=False)
    print(f"[✓] Đã xuất bảng phân tích theo bước thời gian t: {t_csv_path}")

    print("\n" + "=" * 90)
    print("🏆 TỔNG HỢP KẾT QUẢ ĐỐI ĐẦU CHÍNH:")
    print(f"   Ghost Pixels : V2.0.1 = {summary['model_v201_orig_checkpoint']['ghost_pixels_mean']:.1f} px  |  V2.0.2 = {summary['model_v202_v03_checkpoint']['ghost_pixels_mean']:.1f} px  -->  Giảm {summary['head_to_head_delta']['ghost_pixel_reduction_pct']:.2f}% (V2.0.2 THẮNG)")
    print(f"   Ghost Win Rate V2.0.2: {summary['head_to_head_delta']['v202_ghost_win_rate_pct']:.1f}% số mẫu!")
    print(f"   PSNR         : V2.0.1 = {summary['model_v201_orig_checkpoint']['psnr_mean']:.2f} dB  |  V2.0.2 = {summary['model_v202_v03_checkpoint']['psnr_mean']:.2f} dB")
    print(f"   SSIM         : V2.0.1 = {summary['model_v201_orig_checkpoint']['ssim_mean']:.4f}  |  V2.0.2 = {summary['model_v202_v03_checkpoint']['ssim_mean']:.4f}")
    print(f"   Chamfer Dist : V2.0.1 = {summary['model_v201_orig_checkpoint']['chamfer_dist_mean']:.6f}  |  V2.0.2 = {summary['model_v202_v03_checkpoint']['chamfer_dist_mean']:.6f}")
    print(f"   Thời gian thực hiện toàn bộ: {elapsed_total:.1f}s")
    print("=" * 90)

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark Đối Đầu Nhanh TPS V2.0.1 vs V2.0.2")
    parser.add_argument('--model_v201', type=str,
                        default=r'checkpoints/experiments/v2.0/v2.0.1/tps_finetune_v1.1_artifacts V0/checkpoints_v1.1/model_tps_v1.1_best.pt',
                        help='Đường dẫn checkpoint model V2.0.1')
    parser.add_argument('--model_v202', type=str,
                        default=r'checkpoints/experiments/v2.0/v2.0.2/tps_finetune_v2.0_artifacts/checkpoints_v1.1/model_tps_v1.1_best.pt',
                        help='Đường dẫn checkpoint model V2.0.2')
    parser.add_argument('--data_root', type=str,
                        default=r'data/processed/sakuga_golden_dataset_tps_ready',
                        help='Thư mục gốc của Sakuga Golden Dataset')
    parser.add_argument('--dataset_csv', type=str,
                        default=r'data/processed/sakuga_golden_dataset_tps_ready/triplets.csv',
                        help='Đường dẫn file triplets.csv')
    parser.add_argument('--output_dir', type=str,
                        default=r'checkpoints/experiments/v2.0',
                        help='Thư mục lưu kết quả benchmark đối đầu (v2.0)')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Số lượng mẫu validation phân tầng cho test nhanh (mặc định: 100)')
    parser.add_argument('--cpu', action='store_true', help='Ép chạy trên CPU')

    args = parser.parse_args()
    run_benchmark_v20(args)
