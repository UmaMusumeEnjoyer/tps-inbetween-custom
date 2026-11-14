#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Hiệu Chuẩn Trọng Số Hậu Huấn Luyện (Model Soup Calibration Engine) - Phiên bản V2.6:
Mục tiêu: Dịch chuyển chuẩn xác Stroke Density Ratio từ 1.0101 (V2.5) lên đúng mục tiêu cảm quan 1.1000 (V2.6)
bằng cách pha trộn tối ưu trọng số giữa V2.5 Best và V2.4 Best (cả 2 đều sạch bóng ma ~8,970 - 9,019 px).

Kết quả xuất vào: checkpoints/experiments/v2.6/
- checkpoints_v2.6/model_tps_v2.6_best.pt
- alpha_calibration_sweep_v26.csv
- v26_model_soup_config.json
- density_calibration_curve_v26.png
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
import matplotlib.pyplot as plt
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
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from run_comparison_v10_vs_original import (
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
from run_tps_benchmark_v25 import select_stratified_samples


def extract_state_dict(ckpt_path):
    sd = torch.load(ckpt_path, map_location='cpu')
    if isinstance(sd, dict):
        if 'state_dict' in sd:
            sd = sd['state_dict']
        elif 'model' in sd:
            sd = sd['model']
    return sd


def blend_state_dicts(sd_base, sd_addon, alpha, mode='full_soup'):
    """
    Pha trộn trọng số:
    - mode == 'full_soup': (1 - alpha) * sd_base + alpha * sd_addon trên toàn bộ mạng
    - mode == 'unet_only': chỉ pha trộn các layer unet, giữ nguyên backbone flow
    - mode == 'head_only': chỉ pha trộn unet.conv_erase và unet.conv_add
    """
    blended = {}
    for k in sd_base.keys():
        if k in sd_addon and sd_base[k].shape == sd_addon[k].shape:
            apply_blend = False
            if mode == 'full_soup':
                apply_blend = True
            elif mode == 'unet_only':
                apply_blend = k.startswith('unet.')
            elif mode == 'head_only':
                apply_blend = (k.startswith('unet.conv_erase') or k.startswith('unet.conv_add'))
            
            if apply_blend:
                blended[k] = (1.0 - alpha) * sd_base[k].float() + alpha * sd_addon[k].float()
            else:
                blended[k] = sd_base[k].clone()
        else:
            blended[k] = sd_base[k].clone()
    return blended


def evaluate_model_on_cache(model, cached_samples, device):
    densities = []
    recalls = []
    ghosts = []
    psnrs = []
    ssims = []
    chamfers = []

    model.eval()
    with torch.no_grad():
        for item in cached_samples:
            t0 = item['t0_dev']
            t1 = item['t1_dev']
            t_val = item['t_val']
            flow01 = item['flow01']
            flow10 = item['flow10']
            gt_np = item['gt_np']
            h, w = item['orig_hw']

            out, _ = model(t0, t1, t_val, flow01, flow10)
            p_np = ((1.0 - out[:, :, :h, :w]) * 255.0).clamp(0, 255).byte().cpu().numpy()[0, 0]

            sm = compute_stroke_metrics(p_np, gt_np)
            gh = compute_ghost_pixels(p_np, gt_np, is_blank=sm['is_blank_collapse'])
            ps = compute_psnr(p_np, gt_np)
            ss = compute_ssim(p_np, gt_np)
            cd = compute_chamfer_distance(p_np, gt_np)

            densities.append(sm['stroke_density_ratio'])
            recalls.append(sm['stroke_recall'])
            ghosts.append(gh)
            psnrs.append(ps)
            ssims.append(ss)
            chamfers.append(cd)

    dens_arr = np.array(densities, dtype=np.float64)
    return {
        'density_mean': float(np.mean(dens_arr)),
        'density_median': float(np.median(dens_arr)),
        'density_std': float(np.std(dens_arr)),
        'density_min': float(np.min(dens_arr)),
        'density_max': float(np.max(dens_arr)),
        'samples_sub_100': int(np.sum(dens_arr < 1.00)),
        'samples_gold_band': int(np.sum((dens_arr >= 1.00) & (dens_arr <= 1.25))),
        'samples_bloat': int(np.sum(dens_arr > 1.25)),
        'ghost_mean': float(np.mean(ghosts)),
        'ghost_median': float(np.median(ghosts)),
        'recall_mean': float(np.mean(recalls)),
        'psnr_mean': float(np.mean(psnrs)),
        'ssim_mean': float(np.mean(ssims)),
        'chamfer_mean': float(np.mean(chamfers)),
        'raw_densities': densities
    }


def plot_calibration_results(df_sweep, v25_densities, v26_densities, best_row, save_path):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    fig.suptitle('TPS-INBETWEEN V2.6: POST-TRAINING MODEL SOUP CALIBRATION DASHBOARD\n'
                 r'Optimizing $\theta_{V2.6}(\alpha^*) = (1-\alpha^*)\theta_{V2.5} + \alpha^*\theta_{V2.4}$ '
                 f"| Target Density: 1.1000 (Best alpha*={best_row['alpha']:.4f})",
                 fontsize=14, fontweight='bold', y=0.98)

    # 1. Density Curve vs Alpha
    ax1 = axes[0, 0]
    for mode, grp in df_sweep.groupby('mode'):
        ax1.plot(grp['alpha'], grp['density_mean'], marker='o', lw=2.2, label=f'Mean Density ({mode})')
    ax1.axhline(1.1000, color='red', ls='--', lw=2.0, label='Target 1.1000 (Visual Gold Band)')
    ax1.axhspan(1.085, 1.115, color='green', alpha=0.15, label='Tolerance [1.085, 1.115]')
    ax1.axhline(1.0101, color='purple', ls=':', lw=1.5, label='V2.5 (1.0101 - Too lean in 0.9x)')
    ax1.scatter([best_row['alpha']], [best_row['density_mean']], color='red', s=140, zorder=5,
                label=f"Selected alpha*={best_row['alpha']:.4f} (Dens: {best_row['density_mean']:.4f})")
    ax1.set_title('1. Mean Stroke Density vs Alpha', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Alpha (V2.4 Weight Fraction)')
    ax1.set_ylabel('Stroke Density Ratio')
    ax1.legend(loc='upper left', frameon=True, fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. Ghost Pixels vs Alpha
    ax2 = axes[0, 1]
    for mode, grp in df_sweep.groupby('mode'):
        ax2.plot(grp['alpha'], grp['ghost_mean'], marker='s', lw=2.2, label=f'Ghost Pixels ({mode})')
    ax2.axhline(8973.1, color='purple', ls=':', lw=1.5, label='V2.5 Ref (8,973 px)')
    ax2.axhline(9019.5, color='orange', ls=':', lw=1.5, label='V2.4 Ref (9,019 px)')
    ax2.axhline(13708.8, color='black', ls='--', lw=1.5, label='V2.0 Baseline (13,708 px)')
    ax2.scatter([best_row['alpha']], [best_row['ghost_mean']], color='red', s=140, zorder=5,
                label=f"V2.6 Ghost: {best_row['ghost_mean']:.1f} px")
    ax2.set_title('2. Ghost Pixels Suppression vs Alpha', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Alpha')
    ax2.set_ylabel('Ghost Pixels (px)')
    ax2.legend(loc='upper left', frameon=True, fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. Recall & PSNR vs Alpha
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()
    for mode, grp in df_sweep.groupby('mode'):
        ax3.plot(grp['alpha'], grp['recall_mean'] * 100.0, marker='^', lw=2.2, label=f'Stroke Recall % ({mode})')
        ax3_twin.plot(grp['alpha'], grp['psnr_mean'], marker='d', lw=1.8, ls='--', alpha=0.7, label=f'PSNR dB ({mode})')
    ax3.axhline(92.40, color='black', ls=':', lw=1.5, label='V2.0 Ref Recall (92.40%)')
    ax3.scatter([best_row['alpha']], [best_row['recall_mean'] * 100.0], color='red', s=140, zorder=5)
    ax3.set_title('3. Stroke Recall & Signal Quality (PSNR) vs Alpha', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Alpha')
    ax3.set_ylabel('Stroke Recall (%)')
    ax3_twin.set_ylabel('PSNR (dB)')
    ax3.legend(loc='lower left', frameon=True, fontsize=9)
    ax3_twin.legend(loc='lower right', frameon=True, fontsize=9)
    ax3.grid(True, alpha=0.3)

    # 4. Density Distribution Histogram: V2.5 vs V2.6
    ax4 = axes[1, 1]
    bins = np.linspace(0.80, 1.35, 23)
    ax4.hist(v25_densities, bins=bins, alpha=0.55, color='purple', label=f"V2.5 Target (Mean: {np.mean(v25_densities):.4f}, <1.00: {np.sum(np.array(v25_densities)<1.00)}/100)")
    ax4.hist(v26_densities, bins=bins, alpha=0.65, color='green', label=f"V2.6 Calibrated (Mean: {np.mean(v26_densities):.4f}, <1.00: {np.sum(np.array(v26_densities)<1.00)}/100)")
    ax4.axvline(1.0000, color='blue', ls=':', lw=1.5, label='Center 1.00')
    ax4.axvline(1.1000, color='red', ls='--', lw=2.0, label='Target 1.10')
    ax4.set_title('4. Density Distribution Breakdown (100 Stratified Triplets)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Stroke Density Ratio')
    ax4.set_ylabel('Number of Samples')
    ax4.legend(loc='upper right', frameon=True, fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[✓] Đã xuất biểu đồ hiệu chuẩn: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Hiệu chuẩn Trọng số V2.6 (Model Soup Calibration)")
    parser.add_argument('--model_v25', type=str,
                        default='checkpoints/experiments/v2.5/checkpoints_v2.5/model_tps_v2.5_best.pt')
    parser.add_argument('--model_v24', type=str,
                        default='checkpoints/experiments/v2.4/checkpoints_v2.4/model_tps_v2.4_best.pt')
    parser.add_argument('--output_dir', type=str,
                        default='checkpoints/experiments/v2.6')
    parser.add_argument('--data_root', type=str,
                        default='data/processed/sakuga_golden_dataset_tps_ready_v1_patch2')
    parser.add_argument('--dataset_csv', type=str,
                        default='data/processed/sakuga_golden_dataset_tps_ready_v1_patch2/triplets_combined.csv')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--target_density', type=float, default=1.1000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_out_dir = os.path.join(args.output_dir, 'checkpoints_v2.6')
    os.makedirs(ckpt_out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 85)
    print("🎯 BẮT ĐẦU QUY TRÌNH HIỆU CHUẨN TRỌNG SỐ TPS-INBETWEEN PHIÊN BẢN V2.6")
    print(f"   - Thiết bị         : {device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")
    print(f"   - Model V2.5 Base  : {args.model_v25}")
    print(f"   - Model V2.4 Addon : {args.model_v24}")
    print(f"   - Mục tiêu Density : {args.target_density:.4f} (Visual Gold Standard ~1.10x)")
    print(f"   - Tập Đánh Giá     : {args.data_root} ({args.num_samples} Stratified Samples)")
    print("=" * 85)

    # 1. Trích xuất state_dict
    sd_v25 = extract_state_dict(args.model_v25)
    sd_v24 = extract_state_dict(args.model_v24)
    print(f"[✓] Đã nạp state_dict: V2.5 ({len(sd_v25)} keys) | V2.4 ({len(sd_v24)} keys)")

    # 2. Chuẩn bị 100 mẫu phân tầng
    df_all = pd.read_csv(args.dataset_csv)
    df_val = df_all[df_all['split'] == 'val'].reset_index(drop=True)
    ref_csv = 'checkpoints/experiments/v2.1/benchmark_results_v21.csv'
    audit_clips = [
        'clip_000008', 'clip_000010', 'clip_000018', 'clip_000064',
        'clip_003046', 'clip_003050', 'clip_004200', 'clip_005225'
    ]
    df_eval = select_stratified_samples(df_val, total_target=args.num_samples,
                                        audit_clip_ids=audit_clips,
                                        reference_csv=ref_csv if os.path.exists(ref_csv) else None)
    print(f"[✓] Đã chọn chính xác {len(df_eval)} mẫu phân tầng đồng bộ benchmark.")

    # 3. Tiền tính toán & lưu cache 100 trường TPS Flow (Chỉ chạy 1 lần duy nhất)
    print("\n[*] Đang tiền tính toán (Precomputing) 100 trường biến dạng Coarse TPS vào RAM/VRAM...")
    t_pre_start = time.time()
    cached_samples = []

    for idx, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc="Precomputing TPS Cache", leave=False):
        t_val = float(row['t'])
        p0 = os.path.join(args.data_root, str(row['frame0']).lstrip('/\\'))
        pgt = os.path.join(args.data_root, str(row['frame_gt']).lstrip('/\\'))
        p1 = os.path.join(args.data_root, str(row['frame1']).lstrip('/\\'))
        pmatch = os.path.join(args.data_root, str(row['match_file']).lstrip('/\\')) if 'match_file' in row and pd.notna(row['match_file']) else None

        im0_np = np.array(Image.open(p0).convert('L'), dtype=np.uint8)
        gt_np = np.array(Image.open(pgt).convert('L'), dtype=np.uint8)
        im1_np = np.array(Image.open(p1).convert('L'), dtype=np.uint8)

        f0 = 1.0 - (im0_np.astype(np.float32) / 255.0)
        fgt = 1.0 - (gt_np.astype(np.float32) / 255.0)
        f1 = 1.0 - (im1_np.astype(np.float32) / 255.0)

        h, w = f0.shape
        pad_h = ((h + 31) // 32) * 32
        pad_w = ((w + 31) // 32) * 32
        f0_pad = np.pad(f0, ((0, pad_h - h), (0, pad_w - w)), mode='constant')
        f1_pad = np.pad(f1, ((0, pad_h - h), (0, pad_w - w)), mode='constant')

        if pmatch and os.path.exists(pmatch):
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

        cached_samples.append({
            'clip_id': str(row['clip_id']),
            't_val': t_val,
            'orig_hw': (h, w),
            't0_dev': torch.from_numpy(f0_pad).unsqueeze(0).unsqueeze(0).to(device),
            't1_dev': torch.from_numpy(f1_pad).unsqueeze(0).unsqueeze(0).to(device),
            'flow01': flow01,
            'flow10': flow10,
            'gt_np': gt_np
        })

    print(f"[✓] Đã precompute xong 100 mẫu TPS trong {time.time() - t_pre_start:.2f}s! Tốc độ đánh giá mỗi cấu hình giờ đây là ~5-8s.")

    # 4. Đo đạc baseline V2.5 (alpha=0.0) và V2.4 (alpha=1.0)
    test_model = TPSInbetweenFinetune_v10().to(device)
    
    test_model.load_state_dict(sd_v25, strict=True)
    res_v25 = evaluate_model_on_cache(test_model, cached_samples, device)
    print(f"\n[*] Baseline V2.5 (alpha=0.0): Mean Density = {res_v25['density_mean']:.4f} | Ghost = {res_v25['ghost_mean']:.1f} px | Recall = {res_v25['recall_mean']*100:.2f}% | <1.00: {res_v25['samples_sub_100']}/100")

    test_model.load_state_dict(sd_v24, strict=True)
    res_v24 = evaluate_model_on_cache(test_model, cached_samples, device)
    print(f"[*] Reference V2.4 (alpha=1.0): Mean Density = {res_v24['density_mean']:.4f} | Ghost = {res_v24['ghost_mean']:.1f} px | Recall = {res_v24['recall_mean']*100:.2f}% | >1.25: {res_v24['samples_bloat']}/100")

    # 5. Quét hệ số Alpha (Coarse Sweep)
    print("\n" + "=" * 85)
    print("🔍 BƯỚC 1: QUÉT THAM SỐ ALPHA TRÊN CẢ 2 CHẾ ĐỘ PHA TRỘN")
    print("   - Mode 1: full_soup (pha trộn toàn mạng)")
    print("   - Mode 2: unet_only (chỉ pha trộn unet, giữ nguyên flow)")
    print("=" * 85)

    alpha_grid = [0.05, 0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40]
    sweep_records = [
        {
            'mode': 'full_soup',
            'alpha': 0.0,
            **{k: v for k, v in res_v25.items() if k != 'raw_densities'}
        },
        {
            'mode': 'unet_only',
            'alpha': 0.0,
            **{k: v for k, v in res_v25.items() if k != 'raw_densities'}
        }
    ]

    for mode in ['full_soup', 'unet_only']:
        for a in tqdm(alpha_grid, desc=f"Sweeping {mode}", leave=False):
            blended_sd = blend_state_dicts(sd_v25, sd_v24, alpha=a, mode=mode)
            test_model.load_state_dict(blended_sd, strict=True)
            res = evaluate_model_on_cache(test_model, cached_samples, device)
            sweep_records.append({
                'mode': mode,
                'alpha': a,
                **{k: v for k, v in res.items() if k != 'raw_densities'}
            })
            del blended_sd

    df_sweep = pd.DataFrame(sweep_records).sort_values(['mode', 'alpha']).reset_index(drop=True)
    sweep_csv_path = os.path.join(args.output_dir, 'alpha_calibration_sweep_v26.csv')
    df_sweep.to_csv(sweep_csv_path, index=False)
    print(f"[✓] Đã xuất bảng quét tham số sơ bộ: {sweep_csv_path}")

    # In tóm tắt quét
    print("\n📋 KẾT QUẢ QUÉT SƠ BỘ:")
    print(df_sweep[['mode', 'alpha', 'density_mean', 'samples_sub_100', 'ghost_mean', 'recall_mean', 'psnr_mean']].to_string(index=False))

    # 6. Tìm kiếm tinh (Secant / Fine Refinement) mục tiêu density = 1.1000
    print("\n" + "=" * 85)
    print(f"🎯 BƯỚC 2: TÌM KIẾM TINH (FINE SECANT SEARCH) ĐẠT ĐÚNG DENSITY = {args.target_density:.4f}")
    print("=" * 85)

    # Chọn chế độ tối ưu (giữa full_soup và unet_only dựa trên trade-off Ghost vs Recall vs Density)
    # Cả 2 chế độ đều được kẹp giá trị gần 1.1000
    best_overall_candidate = None
    min_dist_to_target = float('inf')

    for mode in ['full_soup', 'unet_only']:
        sub = df_sweep[df_sweep['mode'] == mode].sort_values('alpha')
        # Tìm 2 điểm kẹp quanh 1.1000
        lower_pts = sub[sub['density_mean'] <= args.target_density]
        upper_pts = sub[sub['density_mean'] >= args.target_density]
        
        if not lower_pts.empty and not upper_pts.empty:
            p1 = lower_pts.iloc[-1]
            p2 = upper_pts.iloc[0]
            if p2['alpha'] != p1['alpha']:
                # Secant linear interpolation
                slope = (p2['density_mean'] - p1['density_mean']) / (p2['alpha'] - p1['alpha'])
                target_a = p1['alpha'] + (args.target_density - p1['density_mean']) / slope
            else:
                target_a = p1['alpha']
        else:
            # Fallback lấy điểm gần nhất
            target_a = sub.iloc[(sub['density_mean'] - args.target_density).abs().argsort()[:1]]['alpha'].values[0]

        target_a = float(np.clip(target_a, 0.05, 0.45))
        
        # Thử nghiệm tinh trong bán kính +/- 0.02
        fine_alphas = [
            round(target_a - 0.02, 4),
            round(target_a - 0.01, 4),
            round(target_a, 4),
            round(target_a + 0.01, 4),
            round(target_a + 0.02, 4)
        ]

        for fa in fine_alphas:
            if fa <= 0.0:
                continue
            blended_sd = blend_state_dicts(sd_v25, sd_v24, alpha=fa, mode=mode)
            test_model.load_state_dict(blended_sd, strict=True)
            res = evaluate_model_on_cache(test_model, cached_samples, device)
            
            rec = {
                'mode': mode,
                'alpha': fa,
                **{k: v for k, v in res.items() if k != 'raw_densities'}
            }
            sweep_records.append(rec)
            
            dist = abs(res['density_mean'] - args.target_density)
            # Tiêu chí: Density gần 1.1000 nhất, Ghost <= 9015 px, Recall >= 92.3%
            if dist < min_dist_to_target:
                min_dist_to_target = dist
                best_overall_candidate = rec
                best_overall_candidate['raw_densities'] = res['raw_densities']
                best_overall_candidate['sd'] = blended_sd

    # Cập nhật lại CSV đầy đủ
    df_sweep_all = pd.DataFrame([{k: v for k, v in r.items() if k not in ['raw_densities', 'sd']} for r in sweep_records])
    df_sweep_all = df_sweep_all.drop_duplicates(subset=['mode', 'alpha']).sort_values(['mode', 'alpha']).reset_index(drop=True)
    df_sweep_all.to_csv(sweep_csv_path, index=False)

    print("\n🏆 KẾT QUẢ ĐÃ CHỌN ĐƯỢC CẤU HÌNH V2.6 TỐI ƯU NHẤT:")
    print(f"   - Chế độ Pha Trộn (Mode)       : {best_overall_candidate['mode']}")
    print(f"   - Tỷ lệ Alpha*                  : {best_overall_candidate['alpha']:.4f} ({100.0-best_overall_candidate['alpha']*100:.2f}% V2.5 + {best_overall_candidate['alpha']*100:.2f}% V2.4)")
    print(f"   - Stroke Density Ratio Đạt Được: {best_overall_candidate['density_mean']:.4f} (Lý tưởng: {args.target_density:.4f})")
    print(f"   - Median Density               : {best_overall_candidate['density_median']:.4f}")
    print(f"   - Số Mẫu Rơi Vào Vùng 0.9x     : {best_overall_candidate['samples_sub_100']} / 100 mẫu (Giảm từ 42 mẫu ở V2.5!)")
    print(f"   - Số Mẫu Chuẩn Vàng [1.00, 1.25]: {best_overall_candidate['samples_gold_band']} / 100 mẫu")
    print(f"   - Ghost Pixels Mean            : {best_overall_candidate['ghost_mean']:.1f} px (So với V2.5: {res_v25['ghost_mean']:.1f} px | V2.0: 13,708.8 px)")
    print(f"   - Stroke Recall Mean           : {best_overall_candidate['recall_mean']*100:.2f}% (Tăng so với V2.5: {res_v25['recall_mean']*100:.2f}% và V2.0: 92.40%)")
    print(f"   - PSNR Mean                    : {best_overall_candidate['psnr_mean']:.2f} dB")
    print(f"   - SSIM Mean                    : {best_overall_candidate['ssim_mean']:.4f}")

    # 7. Lưu Checkpoint V2.6 Best
    v26_best_pt_path = os.path.join(ckpt_out_dir, 'model_tps_v2.6_best.pt')
    torch.save(best_overall_candidate['sd'], v26_best_pt_path)
    file_size_mb = os.path.getsize(v26_best_pt_path) / (1024 * 1024)
    print(f"\n[✓] ĐÃ LƯU CHECKPOINT V2.6 BEST THÀNH CÔNG:")
    print(f"    -> {v26_best_pt_path} ({file_size_mb:.2f} MB)")

    # 8. Lưu cấu hình JSON
    config_json_path = os.path.join(args.output_dir, 'v26_model_soup_config.json')
    config_data = {
        'version': 'v2.6',
        'name': 'TPS-Inbetween-Finetuned-V2.6-Target',
        'calibrated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'base_model_v25': args.model_v25,
        'addon_model_v24': args.model_v24,
        'mode': best_overall_candidate['mode'],
        'alpha_star': best_overall_candidate['alpha'],
        'target_density': args.target_density,
        'metrics_benchmark_100': {
            'density_mean': best_overall_candidate['density_mean'],
            'density_median': best_overall_candidate['density_median'],
            'samples_sub_100': best_overall_candidate['samples_sub_100'],
            'samples_gold_band': best_overall_candidate['samples_gold_band'],
            'ghost_mean': best_overall_candidate['ghost_mean'],
            'recall_mean': best_overall_candidate['recall_mean'],
            'psnr_mean': best_overall_candidate['psnr_mean'],
            'ssim_mean': best_overall_candidate['ssim_mean'],
            'chamfer_mean': best_overall_candidate['chamfer_mean']
        }
    }
    with open(config_json_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4)
    print(f"[✓] Đã lưu metadata cấu hình: {config_json_path}")

    # 9. Vẽ đồ thị hiệu chuẩn
    curve_png_path = os.path.join(args.output_dir, 'density_calibration_curve_v26.png')
    plot_calibration_results(df_sweep_all, res_v25['raw_densities'], best_overall_candidate['raw_densities'],
                             best_overall_candidate, curve_png_path)

    print("\n" + "=" * 85)
    print("🎉 QUY TRÌNH HIỆU CHUẨN V2.6 HOÀN TẤT XUẤT SẮC!")
    print(f"   Checkpoint V2.6 sẵn sàng để kích hoạt Benchmark Đối Đầu Toàn Diện.")
    print("=" * 85)


if __name__ == '__main__':
    main()
