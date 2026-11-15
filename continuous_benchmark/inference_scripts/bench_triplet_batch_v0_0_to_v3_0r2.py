#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_batch_evolution_benchmarks.py
--------------------------------------------------------------------------------
Batch Runner Benchmark Continuous Toàn Diện:
Từ Pretrained Baseline -> V0.1 -> V1.x -> V2.x -> V3.0 (2.6) -> V3.0-Remastered R2
Thực thi trên chuỗi chuẩn hóa 540p: 'data/lineart_experiments/test'
Toàn bộ kết quả xuất tự động vào: 'checkpoints/experiments/continuous_benchmarks/'
--------------------------------------------------------------------------------
"""

import os
import sys
import time
import json
import math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
INFERENCE_SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'continuous_benchmark', 'inference_scripts')
if INFERENCE_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, INFERENCE_SCRIPTS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bench_triplet_utils_and_v1_0_vs_orig import (
    OriginalTPSInbetween, TPSInbetweenFinetune_v10,
    extract_sift_keypoints, robust_tps_theta_from_points, tps_flow_from_theta,
    compute_psnr, compute_ssim, compute_stroke_metrics,
    compute_ghost_pixels, compute_ie_mae
)
from bench_triplet_01_v0_1_eval import TPSInbetweenFinetune
from bench_triplet_02_v0_2_eval import TPSInbetweenFinetune_v2
from bench_triplet_03_v0_3_eval import TPSInbetweenFinetune_v3 as TPSInbetweenFinetune_v03
from bench_triplet_10_v3_0_asym_fast import TPSInbetweenFinetune_v3 as TPSInbetweenFinetune_v305

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SPEC_FILE = os.path.join(PROJECT_ROOT, 'data', 'lineart_experiments', 'test', 'continuous_keyframes_spec.json')
INPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'lineart_experiments', 'test')
BENCHMARK_BASE_DIR = os.path.join(PROJECT_ROOT, 'checkpoints', 'experiments', 'continuous_benchmarks')
MASTER_DIR = os.path.join(BENCHMARK_BASE_DIR, 'MASTER_SUMMARY')
os.makedirs(BENCHMARK_BASE_DIR, exist_ok=True)
os.makedirs(MASTER_DIR, exist_ok=True)


# ==============================================================================
# HÀM ĐO LƯỜNG HÌNH THÁI VÀ TOPOLOGY TRÊN GPU
# ==============================================================================
def soft_dilate(img):
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))

def soft_erode(img):
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)

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
    d_acc = 1.0 - bin_mask
    curr = bin_mask
    for s in range(1, steps + 1):
        k = 2 * s + 1
        dil = F.max_pool2d(curr, kernel_size=k, stride=1, padding=s)
        weight = 1.0 / float(s)
        d_acc = d_acc - weight * (dil - curr)
        curr = dil
    return torch.clamp(d_acc, 0.0, 1.0)

def compute_wcd_torch(p_bin, g_bin):
    b, c, h, w = p_bin.shape
    sum_p = torch.sum(p_bin, dim=[-3, -2, -1])
    sum_g = torch.sum(g_bin, dim=[-3, -2, -1])
    min_sum = torch.clamp(torch.min(sum_p, sum_g), min=1.0)
    h_factor = torch.sigmoid(torch.abs(sum_p - sum_g) / min_sum)
    d_p = _approx_gpu_distance_map(p_bin, steps=6) * 16.0
    d_g = _approx_gpu_distance_map(g_bin, steps=6) * 16.0
    dist_term = F.softplus(p_bin * d_g) + F.softplus(g_bin * d_p)
    wcd = h_factor * (torch.sum(dist_term, dim=[-3, -2, -1]) / (h * w * 16.0))
    return float(wcd.mean().item())

def compute_semd_torch(p_ink, g_ink):
    px = torch.sum(p_ink, dim=-2)
    gx = torch.sum(g_ink, dim=-2)
    px_norm = px / torch.clamp(torch.sum(px, dim=-1, keepdim=True), min=1e-5)
    gx_norm = gx / torch.clamp(torch.sum(gx, dim=-1, keepdim=True), min=1e-5)
    emd_x = torch.mean(torch.abs(torch.cumsum(px_norm, dim=-1) - torch.cumsum(gx_norm, dim=-1)), dim=[-2, -1])

    py = torch.sum(p_ink, dim=-1)
    gy = torch.sum(g_ink, dim=-1)
    py_norm = py / torch.clamp(torch.sum(py, dim=-1, keepdim=True), min=1e-5)
    gy_norm = gy / torch.clamp(torch.sum(gy, dim=-1, keepdim=True), min=1e-5)
    emd_y = torch.mean(torch.abs(torch.cumsum(py_norm, dim=-1) - torch.cumsum(gy_norm, dim=-1)), dim=[-2, -1])
    return float(((emd_x + emd_y) * 0.5).mean().item())

def compute_topology_metrics(pred_np, gt_np):
    p_ink = 1.0 - (torch.from_numpy(pred_np).float().to(DEVICE).unsqueeze(0).unsqueeze(0) / 255.0)
    g_ink = 1.0 - (torch.from_numpy(gt_np).float().to(DEVICE).unsqueeze(0).unsqueeze(0) / 255.0)
    p_bin = (p_ink > 0.12).float()
    g_bin = (g_ink > 0.12).float()

    skel_g = soft_skeletonize(g_ink, iters=2)
    skel_mask = (skel_g > 0.20).float()
    tp_skel = torch.sum(soft_dilate(p_bin) * skel_mask, dim=[-3, -2, -1])
    skel_recall = tp_skel / torch.clamp(torch.sum(skel_mask, dim=[-3, -2, -1]), min=1.0)

    nbrs = F.avg_pool2d(skel_g, kernel_size=3, stride=1, padding=1) * 9.0 - skel_g
    endpoints = (nbrs <= 1.8) & (skel_g > 0.20)
    p_dilated = soft_dilate(p_bin)
    tp_tip = torch.sum(p_dilated * endpoints.float(), dim=[-3, -2, -1])
    tip_recall = tp_tip / torch.clamp(torch.sum(endpoints.float(), dim=[-3, -2, -1]), min=1.0)

    wcd_val = compute_wcd_torch(p_bin, g_bin)
    semd_val = compute_semd_torch(p_ink, g_ink)

    return {
        'skel_recall': float(skel_recall.mean().item()),
        'tip_recall': float(tip_recall.mean().item()),
        'wcd': wcd_val,
        'semd': semd_val
    }


# ==============================================================================
# DANH MỤC 17 CHECKPOINT TOÀN BỘ TIẾN HÓA MÔ HÌNH
# ==============================================================================
CATALOG = [
    {
        'tag': 'v0.0_baseline',
        'name': 'Pretrained Baseline',
        'cls': OriginalTPSInbetween,
        'ckpt': os.path.join(PROJECT_ROOT, 'checkpoints/pretrained_baselines/tps-original/model_latest.pt')
    },
    {
        'tag': 'v0.1',
        'name': 'TPS Finetune V0.1',
        'cls': TPSInbetweenFinetune,
        'ckpt': os.path.join(PROJECT_ROOT, 'checkpoints/experiments/v0.1/model_tps_v0.1_best.pt')
    },
    {
        'tag': 'v0.2',
        'name': 'TPS Finetune V0.2',
        'cls': TPSInbetweenFinetune_v2,
        'ckpt': os.path.join(PROJECT_ROOT, 'checkpoints/experiments/v0.2/model_tps_v0.2_best.pt')
    },
    {
        'tag': 'v0.3',
        'name': 'TPS Finetune V0.3',
        'cls': TPSInbetweenFinetune_v03,
        'ckpt': os.path.join(PROJECT_ROOT, 'checkpoints/experiments/v0.3/model_tps_v0.3_best.pt')
    },
    {
        'tag': 'v1.0',
        'name': 'TPS Finetune V1.0',
        'cls': TPSInbetweenFinetune_v10,
        'ckpt': os.path.join(PROJECT_ROOT, 'checkpoints/experiments/v1.0/model_tps_v1.0_best.pt')
    },
    {
        'tag': 'v1.1',
        'name': 'TPS Finetune V1.1',
        'cls': TPSInbetweenFinetune_v10,
        'ckpt': os.path.join(PROJECT_ROOT, 'checkpoints/experiments/v1.1/model_tps_v1.1_best.pt')
    },
    {
        'tag': 'v2.0',
        'name': 'TPS Finetune V2.0',
        'cls': TPSInbetweenFinetune_v10,
        'ckpt': os.path.join(PROJECT_ROOT, 'checkpoints/experiments/v2.0/v2.0.2/tps_finetune_v2.0_artifacts/checkpoints_v1.1/model_tps_v2.0_best.pt')
    },
    {
        'tag': 'v2.1',
        'name': 'TPS Finetune V2.1',
        'cls': TPSInbetweenFinetune_v10,
        'ckpt': os.path.join(PROJECT_ROOT, 'checkpoints/experiments/v2.1/tps_finetune_v2.1_artifacts/checkpoints_v2.1/model_tps_v1.1_best.pt')
    },
    {
        'tag': 'v2.2',
        'name': 'TPS Finetune V2.2',
        'cls': TPSInbetweenFinetune_v10,
        'ckpt': os.path.join(PROJECT_ROOT, 'checkpoints/experiments/v2.2/tps_finetune_v2.2_artifacts/checkpoints_v2.2/model_tps_v2.2_best.pt')
    },
    {
        'tag': 'v2.3',
        'name': 'TPS Finetune V2.3',
        'cls': TPSInbetweenFinetune_v10,
        'ckpt': os.path.join(PROJECT_ROOT, 'checkpoints/experiments/v2.3/tps_finetune_v2.3_artifacts/checkpoints_v2.3/model_tps_v2.3_best.pt')
    },
    {
        'tag': 'v2.4',
        'name': 'TPS Finetune V2.4',
        'cls': TPSInbetweenFinetune_v10,
        'ckpt': os.path.join(PROJECT_ROOT, 'checkpoints/experiments/v2.4/checkpoints_v2.4/model_tps_v2.4_best.pt')
    },
    {
        'tag': 'v2.5',
        'name': 'TPS Finetune V2.5',
        'cls': TPSInbetweenFinetune_v10,
        'ckpt': os.path.join(PROJECT_ROOT, 'checkpoints/experiments/v2.5/checkpoints_v2.5/model_tps_v2.5_best.pt')
    },
    {
        'tag': 'v2.6',
        'name': 'TPS Finetune V2.6',
        'cls': TPSInbetweenFinetune_v10,
        'ckpt': os.path.join(PROJECT_ROOT, 'checkpoints/experiments/v2.6/checkpoints_v2.6/model_tps_v2.6_best.pt')
    },
    {
        'tag': 'v3.0.5',
        'name': 'TPS Finetune V3.0.5',
        'cls': TPSInbetweenFinetune_v305,
        'ckpt': os.path.join(PROJECT_ROOT, 'checkpoints/experiments/v3.0 (2.6)/v3.0.5 (25epoch)/checkpoints_v3.0/model_tps_v3.0_best_progress.pt')
    },
    {
        'tag': 'v3.0_re1',
        'name': 'TPS V3.0-Remastered R1',
        'cls': TPSInbetweenFinetune_v10,
        'ckpt': os.path.join(PROJECT_ROOT, 'checkpoints/experiments/v3.0 re/v3.0 re1/checkpoints_v3.0_remastered/model_tps_v3.0_remastered_ep21.pt')
    },
    {
        'tag': 'v3.0_re2_ep12',
        'name': 'TPS V3.0-R2 Ep12 Best',
        'cls': TPSInbetweenFinetune_v10,
        'ckpt': os.path.join(PROJECT_ROOT, 'checkpoints/experiments/v3.0 re/v3.0 re2/checkpoints_v3.0_remastered/model_tps_v3.0_remastered_ep12.pt')
    },
    {
        'tag': 'v3.0_re2_soup',
        'name': 'TPS V3.0-R2 Soup (Winner)',
        'cls': TPSInbetweenFinetune_v10,
        'ckpt': os.path.join(PROJECT_ROOT, 'checkpoints/experiments/v3.0 re/v3.0 re2/checkpoints_v3.0_remastered/model_tps_v3.0_remastered_soup.pt')
    }
]


# ==============================================================================
# HUD VẼ KHUNG HÌNH SIDE-BY-SIDE
# ==============================================================================
def draw_hud(img_gt, img_pred, frame_idx, time_sec, is_keyframe, model_tag, t_val=0.0,
             psnr=50.0, ssim=1.0, recall=1.0, density=1.0, skel_rec=1.0):
    h, w = img_gt.shape[:2]
    c_gt = cv2.cvtColor(img_gt, cv2.COLOR_GRAY2BGR) if len(img_gt.shape) == 2 else img_gt.copy()
    c_pred = cv2.cvtColor(img_pred, cv2.COLOR_GRAY2BGR) if len(img_pred.shape) == 2 else img_pred.copy()

    banner_h = 36
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.rectangle(c_gt, (0, 0), (w, banner_h), (25, 25, 25), -1)
    cv2.putText(c_gt, f"GROUND TRUTH [#{frame_idx:02d} | {time_sec:.1f}s | 540p]",
                (12, 24), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.rectangle(c_pred, (0, 0), (w, banner_h), (25, 25, 25), -1)
    if is_keyframe:
        title_pred = f"KEYFRAME [INPUT #{frame_idx:02d} | {time_sec:.1f}s]"
        color_pred = (0, 220, 255)
    else:
        title_pred = f"{model_tag.upper()} [t={t_val:.2f} | #{frame_idx:02d}]"
        color_pred = (0, 255, 120)
    cv2.putText(c_pred, title_pred, (12, 24), font, 0.55, color_pred, 2, cv2.LINE_AA)

    stat_h = 28
    cv2.rectangle(c_pred, (0, h - stat_h), (w, h), (20, 20, 20), -1)
    if is_keyframe:
        stat_text = "Input Keyframe Reference (Exact 100%)"
    else:
        stat_text = f"PSNR: {psnr:.2f}dB | SSIM: {ssim:.4f} | Rec: {recall*100:.1f}% | Skel: {skel_rec*100:.1f}% | Dens: {density:.2f}x"
    cv2.putText(c_pred, stat_text, (12, h - 9), font, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

    return np.hstack([c_gt, c_pred])


# ==============================================================================
# HÀM CHÍNH BENCHMARK BATCH
# ==============================================================================
def main():
    print("=" * 85)
    print("🚀 BẮT ĐẦU CHẠY BATCH BENCHMARK TIẾN HÓA MÔ HÌNH (V0.0 BASELINE ──► V3.0-RE2)")
    print(f"   - Thiết bị tăng tốc GPU: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"   - Chuỗi test chuẩn hóa : {INPUT_DIR}")
    print(f"   - Thư mục lưu kết quả  : {BENCHMARK_BASE_DIR}")
    print("=" * 85)

    with open(SPEC_FILE, 'r', encoding='utf-8') as f:
        spec_data = json.load(f)

    frames_spec = spec_data['frames']
    total_frames = len(frames_spec)
    key_indices = set(spec_data['key_indices'])

    # 1. Nạp toàn bộ ảnh Ground Truth
    gt_images = []
    for item in frames_spec:
        fp = os.path.join(INPUT_DIR, item['filename'])
        im = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        gt_images.append(im)

    h, w = gt_images[0].shape
    pad_h = ((h + 31) // 32) * 32
    pad_w = ((w + 31) // 32) * 32

    # 2. Precompute SIFT Matches & TPS Flow cho 6 phân đoạn keyframe (TIẾT KIỆM THỜI GIAN KHỔNG LỒ)
    print("\n[*] Đang tính toán và cache trước SIFT Keypoint Matches & TPS Coarse Flow cho 6 đoạn keyframes...")
    sorted_keys = sorted(list(key_indices))
    segments_cache = []

    for i in range(len(sorted_keys) - 1):
        k_a = sorted_keys[i]
        k_b = sorted_keys[i + 1]
        im_a = gt_images[k_a]
        im_b = gt_images[k_b]

        f0 = 1.0 - (im_a.astype(np.float32) / 255.0)
        f1 = 1.0 - (im_b.astype(np.float32) / 255.0)
        f0_pad = np.pad(f0, ((0, pad_h - h), (0, pad_w - w)), mode='constant')
        f1_pad = np.pad(f1, ((0, pad_h - h), (0, pad_w - w)), mode='constant')
        t0 = torch.from_numpy(f0_pad).unsqueeze(0).unsqueeze(0).to(DEVICE)
        t1 = torch.from_numpy(f1_pad).unsqueeze(0).unsqueeze(0).to(DEVICE)

        kps_raw = extract_sift_keypoints(im_a, im_b, max_pts=80)
        num_matches = kps_raw.shape[1]

        if num_matches >= 4:
            kps0 = kps_raw[0].copy()
            kps1 = kps_raw[1].copy()
            kps0[:, 0] = (kps0[:, 0] * w) / pad_w
            kps0[:, 1] = (kps0[:, 1] * h) / pad_h
            kps1[:, 0] = (kps1[:, 0] * w) / pad_w
            kps1[:, 1] = (kps1[:, 1] * h) / pad_h
            kps_pad = np.stack([kps0, kps1], axis=0)

            c0 = torch.from_numpy(kps0).to(DEVICE)
            c1 = torch.from_numpy(kps1).to(DEVICE)
            theta01, c0_u = robust_tps_theta_from_points(c0, c1, lambd=0.5, add_anchors=True)
            theta10, c1_u = robust_tps_theta_from_points(c1, c0, lambd=0.5, add_anchors=True)
            flow01 = tps_flow_from_theta(theta01, c0_u, pad_h, pad_w, DEVICE)
            flow10 = tps_flow_from_theta(theta10, c1_u, pad_h, pad_w, DEVICE)
        else:
            kps_pad = np.zeros((2, 0, 2), dtype=np.float32)
            flow01 = torch.zeros((1, 2, pad_h, pad_w), device=DEVICE)
            flow10 = torch.zeros((1, 2, pad_h, pad_w), device=DEVICE)

        inbetweens = [item for item in frames_spec if item['type'] == 'INBETWEEN' and item['ref_key_a'] == k_a and item['ref_key_b'] == k_b]
        segments_cache.append({
            'idx_a': k_a,
            'idx_b': k_b,
            't0': t0,
            't1': t1,
            'flow01': flow01,
            'flow10': flow10,
            'kps_pad': kps_pad,
            'num_matches': num_matches,
            'inbetweens': inbetweens
        })
    print(f"[✓] Cache hoàn tất 6 phân đoạn TPS Flow. Sẵn sàng tăng tốc suy luận cho tất cả mô hình!")

    # 3. Vòng lặp duyệt qua tất cả checkpoint trong CATALOG
    master_records = []
    total_models = len(CATALOG)

    for m_idx, entry in enumerate(CATALOG):
        tag = entry['tag']
        name = entry['name']
        cls = entry['cls']
        ckpt = entry['ckpt']
        target_dir = os.path.join(BENCHMARK_BASE_DIR, tag)
        os.makedirs(target_dir, exist_ok=True)
        rec_dir = os.path.join(target_dir, 'frames_reconstructed')
        gt_dir = os.path.join(target_dir, 'frames_ground_truth')
        sbs_dir = os.path.join(target_dir, 'frames_side_by_side')
        os.makedirs(rec_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(sbs_dir, exist_ok=True)

        print(f"\n[{m_idx+1:02d}/{total_models:02d}] ──► Đang thực hiện benchmark: {name} ({tag})")
        if not os.path.exists(ckpt):
            print(f"      [!] Bỏ qua: Không tìm thấy checkpoint tại {ckpt}")
            continue

        try:
            # Nạp mô hình
            model = cls().to(DEVICE)
            sd = torch.load(ckpt, map_location=DEVICE, weights_only=False)
            if isinstance(sd, dict):
                if 'model' in sd: sd = sd['model']
                elif 'state_dict' in sd: sd = sd['state_dict']
            clean_sd = {k.replace('module.', '').replace('base_model.', ''): v for k, v in sd.items()}
            model.load_state_dict(clean_sd, strict=False)
            model.eval()

            # Thực hiện suy luận 17 frame inbetween
            t_start = time.time()
            inbetween_records = []
            reconstructed_frames = [None] * total_frames

            # Gán keyframes
            for item in frames_spec:
                idx = item['frame_idx']
                if item['type'] == 'KEYFRAME':
                    reconstructed_frames[idx] = gt_images[idx].copy()
                    cur_sec = item['time_sec']
                    cv2.imwrite(os.path.join(rec_dir, f"frame_{idx:02d}_{cur_sec:.1f}s_KEY.png"), gt_images[idx])
                    cv2.imwrite(os.path.join(gt_dir, f"gt_{idx:02d}_{cur_sec:.1f}s.png"), gt_images[idx])

            for seg in segments_cache:
                t0 = seg['t0']
                t1 = seg['t1']
                flow01 = seg['flow01']
                flow10 = seg['flow10']
                kps_pad = seg['kps_pad']
                idx_a = seg['idx_a']
                idx_b = seg['idx_b']

                for row in seg['inbetweens']:
                    cur_idx = row['frame_idx']
                    t_val = float(row['t_val'])
                    cur_sec = float(row['time_sec'])
                    gt_frame = gt_images[cur_idx]

                    t_infer_s = time.time()
                    with torch.no_grad():
                        t_tensor = torch.tensor([[t_val]], dtype=torch.float32, device=DEVICE).view(1, 1, 1, 1)
                        if tag in ['v0.1', 'v0.2', 'v0.3']:
                            out = model(t0, t1, t_tensor, [kps_pad], max_kps=80, return_coarse=True)
                        else:
                            out = model(t0, t1, float(t_val), flow01, flow10)

                        if isinstance(out, tuple):
                            out_tensor = out[0]
                        else:
                            out_tensor = out
                    lat_ms = (time.time() - t_infer_s) * 1000.0

                    pred_np = ((1.0 - out_tensor[0, 0, :h, :w].clamp(0.0, 1.0)) * 255.0).byte().cpu().numpy()
                    reconstructed_frames[cur_idx] = pred_np

                    cv2.imwrite(os.path.join(rec_dir, f"frame_{cur_idx:02d}_{cur_sec:.1f}s_INBETWEEN.png"), pred_np)
                    cv2.imwrite(os.path.join(gt_dir, f"gt_{cur_idx:02d}_{cur_sec:.1f}s.png"), gt_frame)

                    psnr = compute_psnr(pred_np, gt_frame)
                    ssim = compute_ssim(pred_np, gt_frame)
                    stroke_metrics = compute_stroke_metrics(pred_np, gt_frame)
                    ghost_px = compute_ghost_pixels(pred_np, gt_frame, is_blank=stroke_metrics['is_blank_collapse'])
                    mae = compute_ie_mae(pred_np, gt_frame)
                    topo = compute_topology_metrics(pred_np, gt_frame)

                    inbetween_records.append({
                        'frame_idx': cur_idx,
                        'time_sec': cur_sec,
                        'type': 'INBETWEEN',
                        'ref_key_a': idx_a,
                        'ref_key_b': idx_b,
                        't_val': round(t_val, 4),
                        'psnr': round(psnr, 2),
                        'ssim': round(ssim, 4),
                        'stroke_recall': round(stroke_metrics['stroke_recall'], 4),
                        'stroke_density': round(stroke_metrics['stroke_density_ratio'], 4),
                        'skel_recall': round(topo['skel_recall'], 4),
                        'tip_recall': round(topo['tip_recall'], 4),
                        'wcd': round(topo['wcd'], 6),
                        'semd': round(topo['semd'], 6),
                        'ghost_pixels': ghost_px,
                        'ie_mae': round(mae, 2),
                        'latency_ms': round(lat_ms, 1)
                    })

            total_sec = time.time() - t_start

            # Đóng gói frames Side-by-Side
            side_by_side_frames = []
            final_records = []
            for item in frames_spec:
                i = item['frame_idx']
                cur_sec = item['time_sec']
                gt_frame = gt_images[i]
                rec_frame = reconstructed_frames[i]
                is_key = (i in key_indices)

                if is_key:
                    rec_entry = {
                        'frame_idx': i,
                        'time_sec': cur_sec,
                        'type': 'KEYFRAME',
                        'ref_key_a': i,
                        'ref_key_b': i,
                        't_val': 0.0,
                        'psnr': 50.0,
                        'ssim': 1.0,
                        'stroke_recall': 1.0,
                        'stroke_density': 1.0,
                        'skel_recall': 1.0,
                        'tip_recall': 1.0,
                        'wcd': 0.0,
                        'semd': 0.0,
                        'ghost_pixels': 0,
                        'ie_mae': 0.0,
                        'latency_ms': 0.0
                    }
                else:
                    rec_entry = next(r for r in inbetween_records if r['frame_idx'] == i)

                final_records.append(rec_entry)
                sbs = draw_hud(
                    gt_frame, rec_frame, i, cur_sec, is_key, tag,
                    t_val=rec_entry['t_val'], psnr=rec_entry['psnr'], ssim=rec_entry['ssim'],
                    recall=rec_entry['stroke_recall'], density=rec_entry['stroke_density'],
                    skel_rec=rec_entry['skel_recall']
                )
                side_by_side_frames.append(sbs)
                cv2.imwrite(os.path.join(sbs_dir, f"sbs_{i:02d}_{cur_sec:.1f}s.png"), sbs)

            # Xuất CSV và JSON
            df_out = pd.DataFrame(final_records).sort_values('frame_idx').reset_index(drop=True)
            csv_path = os.path.join(target_dir, f"benchmark_results_{tag}.csv")
            df_out.to_csv(csv_path, index=False)

            ib_df = df_out[df_out['type'] == 'INBETWEEN']
            summary = {
                'version_tag': tag,
                'model_name': name,
                'model_ckpt': os.path.basename(ckpt),
                'mean_psnr_inbetween': round(float(ib_df['psnr'].mean()), 2),
                'mean_ssim_inbetween': round(float(ib_df['ssim'].mean()), 4),
                'mean_stroke_recall_inbetween': round(float(ib_df['stroke_recall'].mean()), 4),
                'mean_stroke_density_inbetween': round(float(ib_df['stroke_density'].mean()), 4),
                'mean_skel_recall_inbetween': round(float(ib_df['skel_recall'].mean()), 4),
                'mean_tip_recall_inbetween': round(float(ib_df['tip_recall'].mean()), 4),
                'mean_wcd_inbetween': round(float(ib_df['wcd'].mean()), 6),
                'mean_semd_inbetween': round(float(ib_df['semd'].mean()), 6),
                'mean_ghost_pixels_inbetween': round(float(ib_df['ghost_pixels'].mean()), 1),
                'mean_ie_mae_inbetween': round(float(ib_df['ie_mae'].mean()), 2),
                'mean_latency_ms': round(float(ib_df['latency_ms'].mean()), 1),
                'total_time_sec': round(total_sec, 2)
            }

            with open(os.path.join(target_dir, f"summary_{tag}.json"), 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            # Xuất Video MP4 & Montage
            sbs_h, sbs_w = side_by_side_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(os.path.join(target_dir, f"side_by_side_{tag}.mp4"), fourcc, 8.0, (sbs_w, sbs_h))
            for sbs in side_by_side_frames:
                out_video.write(sbs)
            out_video.release()

            # Montage 24 frame
            montage_rows = []
            for row_idx in range(4):
                cols = []
                for col_idx in range(6):
                    f_i = row_idx * 6 + col_idx
                    f_img = cv2.resize(reconstructed_frames[f_i], (320, 180))
                    f_bgr = cv2.cvtColor(f_img, cv2.COLOR_GRAY2BGR)
                    is_k = (f_i in key_indices)
                    border_col = (0, 200, 255) if is_k else (0, 255, 100)
                    cv2.rectangle(f_bgr, (0, 0), (320, 180), border_col, 2)
                    lbl = f"KEY #{f_i:02d}" if is_k else f"{tag.upper()} #{f_i:02d}"
                    cv2.putText(f_bgr, lbl, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, border_col, 2)
                    cols.append(f_bgr)
                montage_rows.append(np.hstack(cols))
            montage_full = np.vstack(montage_rows)
            cv2.imwrite(os.path.join(target_dir, f"montage_24frames_{tag}.png"), montage_full)

            master_records.append(summary)
            print(f"      [✓] Hoàn tất {name} ({total_sec:.2f}s) | PSNR: {summary['mean_psnr_inbetween']}dB | SSIM: {summary['mean_ssim_inbetween']} | Recall: {summary['mean_stroke_recall_inbetween']*100:.2f}% | Dens: {summary['mean_stroke_density_inbetween']:.2f}x | Ghost: {summary['mean_ghost_pixels_inbetween']}")

        except Exception as e:
            print(f"      [X] LỖI khi chạy {name}: {e}")
            import traceback
            traceback.print_exc()

    # 4. Xuất Bảng Master Summary CSV
    master_df = pd.DataFrame(master_records)
    master_csv_path = os.path.join(MASTER_DIR, 'evolution_timeline_master.csv')
    master_df.to_csv(master_csv_path, index=False)
    print("\n" + "=" * 85)
    print(f"🏆 ĐÃ XUẤT MASTER SUMMARY CSV: {master_csv_path}")

    # 5. Vẽ Biểu Đồ Đường Tiến Hóa Toàn Cảnh (Evolution Timeline Chart)
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        plt.subplots_adjust(hspace=0.35, wspace=0.25)
        fig.suptitle('TPS-Inbetween Model Evolution Timeline (V0.0 Baseline ──► V3.0-Remastered R2)', fontsize=16, fontweight='bold')

        versions = [r['version_tag'] for r in master_records]
        x_indices = range(len(versions))

        # Subplot 1: Stroke Recall & Skeleton Recall
        ax1 = axes[0, 0]
        ax1.plot(x_indices, [r['mean_stroke_recall_inbetween']*100 for r in master_records], marker='o', color='#2ecc71', linewidth=2, label='Stroke Recall (%)')
        ax1.plot(x_indices, [r['mean_skel_recall_inbetween']*100 for r in master_records], marker='s', color='#9b59b6', linewidth=2, label='Skeleton Recall (%)')
        ax1.plot(x_indices, [r['mean_tip_recall_inbetween']*100 for r in master_records], marker='^', color='#3498db', linewidth=2, linestyle='--', label='Hair-Tip Recall (%)')
        ax1.set_title('Topology & Stroke Retention', fontweight='bold')
        ax1.set_xticks(x_indices)
        ax1.set_xticklabels(versions, rotation=45, ha='right', fontsize=9)
        ax1.set_ylabel('Recall (%)')
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend()

        # Subplot 2: SSIM & PSNR
        ax2 = axes[0, 1]
        ax2_twin = ax2.twinx()
        l1 = ax2.plot(x_indices, [r['mean_ssim_inbetween'] for r in master_records], marker='o', color='#e67e22', linewidth=2, label='SSIM')
        l2 = ax2_twin.plot(x_indices, [r['mean_psnr_inbetween'] for r in master_records], marker='d', color='#e74c3c', linewidth=2, linestyle='--', label='PSNR (dB)')
        ax2.set_title('Structural Similarity & Fidelity', fontweight='bold')
        ax2.set_xticks(x_indices)
        ax2.set_xticklabels(versions, rotation=45, ha='right', fontsize=9)
        ax2.set_ylabel('SSIM', color='#e67e22')
        ax2_twin.set_ylabel('PSNR (dB)', color='#e74c3c')
        ax2.grid(True, linestyle=':', alpha=0.6)
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='lower right')

        # Subplot 3: Stroke Density Ratio
        ax3 = axes[1, 0]
        densities = [r['mean_stroke_density_inbetween'] for r in master_records]
        bars = ax3.bar(x_indices, densities, color='#1abc9c', alpha=0.7, edgecolor='#16a085', width=0.55)
        ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='Target Density 1.0x')
        ax3.set_title('Stroke Density Ratio (Line Thickness)', fontweight='bold')
        ax3.set_xticks(x_indices)
        ax3.set_xticklabels(versions, rotation=45, ha='right', fontsize=9)
        ax3.set_ylabel('Ratio (x)')
        ax3.grid(True, linestyle=':', alpha=0.6)
        ax3.legend()

        # Subplot 4: Ghost Pixels & WCD
        ax4 = axes[1, 1]
        ax4.plot(x_indices, [r['mean_ghost_pixels_inbetween'] for r in master_records], marker='v', color='#e74c3c', linewidth=2, label='Ghost Pixels (px)')
        ax4.set_title('Background Ghosting Artifacts (Lower is Better)', fontweight='bold')
        ax4.set_xticks(x_indices)
        ax4.set_xticklabels(versions, rotation=45, ha='right', fontsize=9)
        ax4.set_ylabel('Ghost Pixels')
        ax4.grid(True, linestyle=':', alpha=0.6)
        ax4.legend()

        chart_path = os.path.join(MASTER_DIR, 'evolution_metrics_comparison.png')
        plt.savefig(chart_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"📊 ĐÃ XUẤT BIỂU ĐỒ ĐƯỜNG TIẾN HÓA TIÊU CHUẨN: {chart_path}")
    except Exception as e:
        print(f"[!] Lỗi khi vẽ biểu đồ: {e}")

    print("=" * 85)
    print("🎉 HOÀN TẤT TOÀN DIỆN CONTINUOUS BENCHMARK CHO TẤT CẢ CÁC PHIÊN BẢN MÔ HÌNH!")
    print("=" * 85)


if __name__ == '__main__':
    main()
