#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_v305_vs_v26_objective.py
--------------------------------------------------------------------------------
BỘ ĐỐI CHUẨN ĐỐI ĐẦU KHÁCH QUAN, KHOA HỌC (HEAD-TO-HEAD AUDIT):
MÔ HÌNH V3.0.5 (EPOCH 25) VS MÔ HÌNH V2.6 BASELINE (GOLD SOUP)

Nguyên tắc:
- 100% Trung thực khoa học, không thiên vị (unbiased).
- Đánh giá trên cùng tập mẫu chuẩn Core Audit Clips (12 clips phân tầng).
- Đo đạc toàn bộ: PSNR, SSIM, MAE, Recall, Precision, F1, Density Ratio, Ghost Pixels, Chamfer, Latency.
- Xuất các bảng ảnh so sánh Side-by-Side (Frame 0 | Coarse | V2.6 | V3.0.5 | Ground Truth)
  và ảnh Zoom 400% các chi tiết nét mảnh (ngọn tóc, mắt, vùng xoay người).
"""

import os
import sys
import time
import json
import math
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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'continuous_benchmark', 'inference_scripts')
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
SCRATCH_DIR = os.path.join(PROJECT_ROOT, 'scratch')
if SCRATCH_DIR not in sys.path:
    sys.path.insert(0, SCRATCH_DIR)

from run_comparison_v10_vs_original import (
    conv, deconv, Conv2, Contextnet, IFNet,
    TPSInbetweenFinetune_v10,
    robust_tps_theta_from_points,
    tps_flow_from_theta,
    warp_normalized_flow,
    warp_pixel_flow,
    compute_psnr,
    compute_ssim,
    compute_stroke_metrics,
    compute_ghost_pixels,
    compute_ie_mae,
    compute_chamfer_distance
)

import v3_engine as v3

# ==============================================================================
# 1. HÀM ĐO DARK GHOST PIXELS
# ==============================================================================
def compute_dark_ghost_pixels(pred_np, gt_np, is_blank=False, dil_radius=7, dark_thresh=60):
    if is_blank:
        return 0.0
    p_dark = (pred_np < dark_thresh).astype(np.float32)
    g_bin = (gt_np < 180).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_radius, dil_radius))
    g_dil = cv2.dilate(g_bin, kernel, iterations=1).astype(np.float32)
    dark_ghost = np.sum(p_dark * (1.0 - g_dil))
    return float(dark_ghost)

# ==============================================================================
# 2. HÀM TẠO ẢNH SIDE-BY-SIDE VÀ ZOOM 400%
# ==============================================================================
def create_comparison_panels(clip_id, t_val, f0_np, f1_np, coarse_np, v26_np, v305_np, gt_np, output_dir):
    """
    Tạo 2 ảnh so sánh chất lượng cao:
    1. Bảng 5 cột: Keyframe 0 | Coarse TPS | V2.6 Baseline | V3.0.5 (Ep 25) | Ground Truth
    2. Bảng Zoom 400% tập trung vào vùng chuyển động hoặc chi tiết nét mảnh.
    """
    os.makedirs(output_dir, exist_ok=True)
    h, w = f0_np.shape
    
    # 1. Bảng 5 cột Side-by-Side
    # Chuyển grayscale sang BGR để vẽ nhãn
    def to_bgr(gray):
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    p0 = to_bgr(f0_np)
    pc = to_bgr(coarse_np)
    p26 = to_bgr(v26_np)
    p305 = to_bgr(v305_np)
    pgt = to_bgr(gt_np)
    
    # Vẽ nhãn lên từng panel
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(p0, "Keyframe 0", (20, 40), font, 0.9, (0, 0, 180), 2)
    cv2.putText(pc, f"Coarse TPS (t={t_val:.2f})", (20, 40), font, 0.9, (120, 60, 0), 2)
    cv2.putText(p26, "V2.6 Gold Baseline", (20, 40), font, 0.9, (180, 0, 0), 2)
    cv2.putText(p305, "V3.0.5 (Epoch 25)", (20, 40), font, 0.9, (0, 140, 0), 2)
    cv2.putText(pgt, f"Ground Truth (t={t_val:.2f})", (20, 40), font, 0.9, (0, 0, 0), 2)
    
    side_by_side = np.hstack([p0, pc, p26, p305, pgt])
    sbs_path = os.path.join(output_dir, f"{clip_id}_side_by_side.png")
    cv2.imwrite(sbs_path, side_by_side)
    
    # 2. Tạo Crop Zoom 400% ở vùng nét quan trọng nhất
    # Tìm vùng nét vẽ có sự khác biệt lớn nhất giữa V2.6 và V3.0.5
    diff_map = np.abs(v26_np.astype(float) - v305_np.astype(float))
    # Gaussian blur để tìm tâm cụm khác biệt
    diff_blur = cv2.GaussianBlur(diff_map, (31, 31), 0)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(diff_blur)
    
    crop_size = min(140, h // 2, w // 2)
    cx, cy = max_loc
    x1 = max(0, min(w - crop_size, cx - crop_size // 2))
    y1 = max(0, min(h - crop_size, cy - crop_size // 2))
    x2 = x1 + crop_size
    y2 = y1 + crop_size
    
    # Vẽ ô chữ nhật màu đỏ trên panel gốc
    cv2.rectangle(p26, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.rectangle(p305, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.rectangle(pgt, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Crop và phóng đại 400%
    zoom_scale = 4
    z_f0 = cv2.resize(f0_np[y1:y2, x1:x2], (crop_size * zoom_scale, crop_size * zoom_scale), interpolation=cv2.INTER_NEAREST)
    z_26 = cv2.resize(v26_np[y1:y2, x1:x2], (crop_size * zoom_scale, crop_size * zoom_scale), interpolation=cv2.INTER_NEAREST)
    z_305 = cv2.resize(v305_np[y1:y2, x1:x2], (crop_size * zoom_scale, crop_size * zoom_scale), interpolation=cv2.INTER_NEAREST)
    z_gt = cv2.resize(gt_np[y1:y2, x1:x2], (crop_size * zoom_scale, crop_size * zoom_scale), interpolation=cv2.INTER_NEAREST)
    
    z_f0_b = to_bgr(z_f0)
    z_26_b = to_bgr(z_26)
    z_305_b = to_bgr(z_305)
    z_gt_b = to_bgr(z_gt)
    
    cv2.putText(z_f0_b, "Keyframe 0", (15, 35), font, 0.7, (0, 0, 180), 2)
    cv2.putText(z_26_b, "V2.6 Crop", (15, 35), font, 0.7, (180, 0, 0), 2)
    cv2.putText(z_305_b, "V3.0.5 Crop", (15, 35), font, 0.7, (0, 140, 0), 2)
    cv2.putText(z_gt_b, "Ground Truth", (15, 35), font, 0.7, (0, 0, 0), 2)
    
    zoom_panel = np.hstack([z_f0_b, z_26_b, z_305_b, z_gt_b])
    zoom_path = os.path.join(output_dir, f"{clip_id}_zoomed_400pct.png")
    cv2.imwrite(zoom_path, zoom_panel)
    
    return sbs_path, zoom_path

# ==============================================================================
# 3. CHƯƠNG TRÌNH ĐỐI SOÁT VÀ ĐO ĐẠC ĐỐI ĐẦU CHÍNH
# ==============================================================================
def main():
    print("=" * 85)
    print("⚖️  TIẾN TRÌNH ĐỐI SOÁT ĐỐI ĐẦU KHOA HỌC: V3.0.5 (EP 25) VS V2.6 BASELINE (GOLD)")
    print("   Tiêu chí: 100% Khách quan, Không thiên vị, Đánh giá đa chiều định lượng & định tính")
    print("=" * 85)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Thiết bị kiểm định: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    v26_ckpt_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'experiments', 'v2.6', 'checkpoints_v2.6', 'model_tps_v2.6_best.pt')
    v305_ckpt_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'experiments', 'v3.0 (2.6)', 'v3.0.5 (25epoch)', 'checkpoints_v3.0', 'model_tps_v3.0_best_progress.pt')
    dataset_csv = os.path.join(PROJECT_ROOT, 'data', 'processed', 'sakuga_golden_dataset_tps_ready_v1_patch2', 'triplets_combined.csv')
    dataset_root = os.path.join(PROJECT_ROOT, 'data', 'processed', 'sakuga_golden_dataset_tps_ready_v1_patch2')
    output_dir = os.path.join(PROJECT_ROOT, 'continuous_benchmark', 'visual_audit', 'comparison_v26_vs_v305')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Nạp Model V2.6
    print(f"\n[1/4] Nạp Model Baseline V2.6 Gold từ: {v26_ckpt_path} ...")
    model_v26 = TPSInbetweenFinetune_v10().to(device)
    sd_v26 = torch.load(v26_ckpt_path, map_location=device)
    if 'state_dict' in sd_v26:
        sd_v26 = sd_v26['state_dict']
    clean_sd26 = {k.replace('module.', '').replace('base_model.', ''): v for k, v in sd_v26.items()}
    model_v26.load_state_dict(clean_sd26, strict=True)
    model_v26.eval()
    params_v26 = sum(p.numel() for p in model_v26.parameters())
    print(f"  [✓] V2.6 nạp thành công! Tham số: {params_v26:,}")
    
    # 2. Nạp Model V3.0.5
    print(f"\n[2/4] Nạp Model V3.0.5 Epoch 25 từ: {v305_ckpt_path} ...")
    model_v305 = v3.TPSInbetweenFinetune_v3().to(device)
    sd_v305 = torch.load(v305_ckpt_path, map_location=device)
    clean_sd305 = {k.replace('module.', '').replace('base_model.', ''): v for k, v in sd_v305.items()}
    model_v305.load_state_dict(clean_sd305, strict=True)
    model_v305.eval()
    params_v305 = sum(p.numel() for p in model_v305.parameters())
    print(f"  [✓] V3.0.5 nạp thành công! Tham số: {params_v305:,} (+{params_v305 - params_v26:,} tham số so với V2.6)")
    
    # 3. Lựa chọn các Core Audit Clips (12 clips)
    df_all = pd.read_csv(dataset_csv)
    target_clips = [
        'clip_000008', 'clip_000010', 'clip_000018', 'clip_000064',
        'clip_000179', 'clip_000400', 'clip_000672', 'clip_000761',
        'clip_003046', 'clip_003050', 'clip_004200', 'clip_005225'
    ]
    df_eval = df_all[df_all['clip_id'].isin(target_clips)].copy().reset_index(drop=True)
    print(f"\n[3/4] Bắt đầu chạy đối đầu trên {len(df_eval)} Core Audit Clips...")
    
    records = []
    generated_images = []
    
    for idx, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc="⚖️ Đo đạc V2.6 vs V3.0.5"):
        clip_id = str(row['clip_id'])
        t_val = float(row['t'])
        disp = float(row.get('mean_disp', 0.0))
        
        p0 = os.path.join(dataset_root, str(row['frame0']).replace('/', '\\').lstrip('\\'))
        pgt = os.path.join(dataset_root, str(row['frame_gt']).replace('/', '\\').lstrip('\\'))
        p1 = os.path.join(dataset_root, str(row['frame1']).replace('/', '\\').lstrip('\\'))
        pmatch = os.path.join(dataset_root, str(row['match_file']).replace('/', '\\').lstrip('\\')) if 'match_file' in row and pd.notna(row['match_file']) else None
        
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
            
        t0_dev = torch.from_numpy(f0_pad).unsqueeze(0).unsqueeze(0).to(device)
        t1_dev = torch.from_numpy(f1_pad).unsqueeze(0).unsqueeze(0).to(device)
        
        # --- Suy luận V2.6 ---
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_s26 = time.perf_counter()
        with torch.no_grad():
            out_v26, coarse_interp = model_v26(t0_dev, t1_dev, t_val, flow01, flow10)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        lat_v26 = (time.perf_counter() - t_s26) * 1000.0
        
        pred_v26_np = ((1.0 - out_v26[:, :, :h, :w]) * 255.0).clamp(0, 255).byte().cpu().numpy()[0, 0]
        coarse_np = ((1.0 - coarse_interp[:, :, :h, :w]) * 255.0).clamp(0, 255).byte().cpu().numpy()[0, 0]
        
        # --- Suy luận V3.0.5 ---
        # Chuẩn bị coarse flow cho V3
        t_tensor = torch.tensor(t_val, device=device, dtype=torch.float32).view(1, 1, 1, 1)
        coarse_flow0t = flow01 * (t_tensor ** 2) - (1.0 - t_tensor) * t_tensor * flow10
        coarse_flow1t = flow10 * ((1.0 - t_tensor) ** 2) - t_tensor * (1.0 - t_tensor) * flow01
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_s305 = time.perf_counter()
        with torch.no_grad():
            c_I0t = warp_normalized_flow(t0_dev, coarse_flow0t)
            c_I1t = warp_normalized_flow(t1_dev, coarse_flow1t)
            feats0 = model_v305.encoder(t0_dev, coarse_flow0t)
            feats1 = model_v305.encoder(t1_dev, coarse_flow1t)
            flow_raw, ifnet_mask = model_v305.IFNet(c_I0t, c_I1t)
            flow_guided, _ = model_v305.contour_propagator(c_I0t, c_I1t, flow_raw)
            fine_I0t = warp_pixel_flow(c_I0t, flow_guided[:, :2])
            fine_I1t = warp_pixel_flow(c_I1t, flow_guided[:, 2:4])
            occ_conf = model_v305.asym_analyzer(fine_I0t, fine_I1t, feats0[0], feats1[0])
            v0, v1 = model_v305.asf_head(fine_I0t, fine_I1t, flow_guided, ifnet_mask, t_tensor, occ_conf)
            base_interp = torch.clamp(v0 * fine_I0t + v1 * fine_I1t, 0.0, 1.0)
            gate_erase, detail_add = model_v305.unet(
                t0_dev, t1_dev, coarse_flow0t, coarse_flow1t,
                fine_I0t, fine_I1t, ifnet_mask, feats0, feats1
            )
            out_v305 = torch.clamp(gate_erase * base_interp + detail_add, 0.0, 1.0)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        lat_v305 = (time.perf_counter() - t_s305) * 1000.0
        
        pred_v305_np = ((1.0 - out_v305[:, :, :h, :w]) * 255.0).clamp(0, 255).byte().cpu().numpy()[0, 0]
        
        # Tạo ảnh trực quan Side-by-Side và Zoom
        sbs_p, zoom_p = create_comparison_panels(clip_id, t_val, im0_np, im1_np, coarse_np, pred_v26_np, pred_v305_np, gt_np, output_dir)
        generated_images.append({'clip_id': clip_id, 'sbs': sbs_p, 'zoom': zoom_p})
        
        # Đo đạc chỉ số V2.6
        psnr_26 = compute_psnr(pred_v26_np, gt_np)
        ssim_26 = compute_ssim(pred_v26_np, gt_np)
        mae_26 = compute_ie_mae(pred_v26_np, gt_np)
        sm_26 = compute_stroke_metrics(pred_v26_np, gt_np)
        ghost_26 = compute_ghost_pixels(pred_v26_np, gt_np, sm_26['is_blank_collapse'])
        dghost_26 = compute_dark_ghost_pixels(pred_v26_np, gt_np, sm_26['is_blank_collapse'])
        cd_26 = compute_chamfer_distance(pred_v26_np, gt_np)
        
        # Tính F1 V2.6
        p26_bin = (pred_v26_np < 180).astype(float)
        gt_bin = (gt_np < 180).astype(float)
        tp26 = np.sum(p26_bin * gt_bin)
        prec26 = tp26 / max(1.0, np.sum(p26_bin))
        rec26 = sm_26['stroke_recall']
        f1_26 = (2.0 * prec26 * rec26 / (prec26 + rec26)) if (prec26 + rec26) > 1e-6 else 0.0
        
        # Đo đạc chỉ số V3.0.5
        psnr_305 = compute_psnr(pred_v305_np, gt_np)
        ssim_305 = compute_ssim(pred_v305_np, gt_np)
        mae_305 = compute_ie_mae(pred_v305_np, gt_np)
        sm_305 = compute_stroke_metrics(pred_v305_np, gt_np)
        ghost_305 = compute_ghost_pixels(pred_v305_np, gt_np, sm_305['is_blank_collapse'])
        dghost_305 = compute_dark_ghost_pixels(pred_v305_np, gt_np, sm_305['is_blank_collapse'])
        cd_305 = compute_chamfer_distance(pred_v305_np, gt_np)
        
        # Tính F1 V3.0.5
        p305_bin = (pred_v305_np < 180).astype(float)
        tp305 = np.sum(p305_bin * gt_bin)
        prec305 = tp305 / max(1.0, np.sum(p305_bin))
        rec305 = sm_305['stroke_recall']
        f1_305 = (2.0 * prec305 * rec305 / (prec305 + rec305)) if (prec305 + rec305) > 1e-6 else 0.0
        
        records.append({
            'clip_id': clip_id,
            'disp': disp,
            't_val': t_val,
            # V2.6
            'ssim_v26': ssim_26,
            'psnr_v26': psnr_26,
            'mae_v26': mae_26,
            'recall_v26': sm_26['stroke_recall'],
            'density_v26': sm_26['stroke_density_ratio'],
            'f1_v26': f1_26,
            'ghost_v26': ghost_26,
            'dark_ghost_v26': dghost_26,
            'chamfer_v26': cd_26,
            'latency_v26': lat_v26,
            # V3.0.5
            'ssim_v305': ssim_305,
            'psnr_v305': psnr_305,
            'mae_v305': mae_305,
            'recall_v305': sm_305['stroke_recall'],
            'density_v305': sm_305['stroke_density_ratio'],
            'f1_v305': f1_305,
            'ghost_v305': ghost_305,
            'dark_ghost_v305': dghost_305,
            'chamfer_v305': cd_305,
            'latency_v305': lat_v305,
        })
        
    df_res = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, 'objective_head_to_head_v26_vs_v305.csv')
    df_res.to_csv(csv_path, index=False)
    
    # [4/4] Tổng kết khoa học
    print("\n" + "=" * 85)
    print("📊 BẢNG TỔNG KẾT ĐỐI ĐẦU ĐỊNH LƯỢNG TRUNG THỰC (V2.6 VS V3.0.5):")
    print("=" * 85)
    
    metrics_summary = [
        ('SSIM (Càng cao càng tốt)', df_res['ssim_v26'].mean(), df_res['ssim_v305'].mean(), df_res['ssim_v305'].mean() - df_res['ssim_v26'].mean()),
        ('PSNR (dB, Càng cao càng tốt)', df_res['psnr_v26'].mean(), df_res['psnr_v305'].mean(), df_res['psnr_v305'].mean() - df_res['psnr_v26'].mean()),
        ('MAE / L1 Error (Càng thấp càng tốt)', df_res['mae_v26'].mean(), df_res['mae_v305'].mean(), df_res['mae_v305'].mean() - df_res['mae_v26'].mean()),
        ('Stroke Recall (%)', df_res['recall_v26'].mean() * 100, df_res['recall_v305'].mean() * 100, (df_res['recall_v305'].mean() - df_res['recall_v26'].mean()) * 100),
        ('Stroke F1-Score (%)', df_res['f1_v26'].mean() * 100, df_res['f1_v305'].mean() * 100, (df_res['f1_v305'].mean() - df_res['f1_v26'].mean()) * 100),
        ('Density Ratio (Chuẩn vàng 1.05x-1.10x)', df_res['density_v26'].mean(), df_res['density_v305'].mean(), df_res['density_v305'].mean() - df_res['density_v26'].mean()),
        ('Ghost Pixels (px, Càng thấp càng sạch)', df_res['ghost_v26'].mean(), df_res['ghost_v305'].mean(), df_res['ghost_v305'].mean() - df_res['ghost_v26'].mean()),
        ('Dark Ghost Pixels (px, Đen sâu)', df_res['dark_ghost_v26'].mean(), df_res['dark_ghost_v305'].mean(), df_res['dark_ghost_v305'].mean() - df_res['dark_ghost_v26'].mean()),
        ('Chamfer Distance (px)', df_res['chamfer_v26'].mean(), df_res['chamfer_v305'].mean(), df_res['chamfer_v305'].mean() - df_res['chamfer_v26'].mean()),
        ('Inference Latency (ms)', df_res['latency_v26'].mean(), df_res['latency_v305'].mean(), df_res['latency_v305'].mean() - df_res['latency_v26'].mean()),
    ]
    
    for label, val26, val305, diff in metrics_summary:
        winner = "🏆 V3.0.5 Thắng" if (diff > 0 if any(k in label for k in ['SSIM', 'PSNR', 'Recall', 'F1']) else diff < 0) else "🏆 V2.6 Thắng"
        if 'Density' in label:
            # Gần 1.08x hơn
            dist26 = abs(val26 - 1.08)
            dist305 = abs(val305 - 1.08)
            winner = "🏆 V3.0.5 Chuẩn hơn" if dist305 < dist26 else "🏆 V2.6 Chuẩn hơn"
        print(f"  * {label:<35}: V2.6 = {val26:>9.4f} | V3.0.5 = {val305:>9.4f} | Chênh lệch: {diff:>+9.4f} | {winner}")
        
    print(f"\n[✓] Đã xuất bảng CSV chi tiết tại: {csv_path}")
    print(f"[✓] Đã tạo toàn bộ ảnh Side-by-Side & Zoom tại: {output_dir}")
    print("=" * 85)

if __name__ == '__main__':
    main()
