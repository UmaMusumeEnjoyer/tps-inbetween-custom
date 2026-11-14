#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_continuous_test_v30r.py
--------------------------------------------------------------------------------
Thực hiện Continuous Test trên tập chuỗi anime 540p:
'data/lineart_experiments/test'
Cấu hình Keyframe & Inbetween đọc từ:
'data/lineart_experiments/test/continuous_keyframes_spec.json'

Hỗ trợ thẩm định toàn diện:
1. PSNR & SSIM
2. Stroke Recall & Density Ratio
3. Ghost Pixels & IE-MAE
4. Weighted Chamfer Distance (WCD)
5. 1D-Sliced Earth Mover's Distance (S-EMD)
6. Skeleton Recall & Hair-Tip Recall

Mặc định sử dụng Checkpoint thành phẩm:
'checkpoints/experiments/v3.0 re/v3.0 re2/checkpoints_v3.0_remastered/model_tps_v3.0_remastered_soup.pt'
--------------------------------------------------------------------------------
"""

import os
import sys
import re
import time
import json
import math
import argparse
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
INFERENCE_SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'continuous_benchmark', 'inference_scripts')
if INFERENCE_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, INFERENCE_SCRIPTS_DIR)

from run_comparison_v10_vs_original import (
    TPSInbetweenFinetune_v10, extract_sift_keypoints,
    robust_tps_theta_from_points, tps_flow_from_theta,
    compute_psnr, compute_ssim, compute_stroke_metrics,
    compute_ghost_pixels, compute_ie_mae
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_SPEC = os.path.join(PROJECT_ROOT, 'data', 'lineart_experiments', 'test', 'continuous_keyframes_spec.json')
DEFAULT_INPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'lineart_experiments', 'test')
DEFAULT_CKPT = os.path.join(
    PROJECT_ROOT, 'checkpoints', 'experiments', 'v3.0 re', 'v3.0 re2',
    'checkpoints_v3.0_remastered', 'model_tps_v3.0_remastered_soup.pt'
)
OUTPUT_BASE_DIR = os.path.join(PROJECT_ROOT, 'continuous_benchmark', 'visual_audit', 'continuous_test_v30r')


# ==============================================================================
# HÀM TÍNH TOÁN METRIC KHOA HỌC V3.0-R2 (WCD, S-EMD, SKELETON RECALL, TIP RECALL)
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

def compute_topology_metrics_torch(p_ink, g_ink):
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
# HÀM VẼ GIAO DIỆN HEAD-UP DISPLAY (HUD) SIDE-BY-SIDE
# ==============================================================================
def draw_comparison_hud(img_gt, img_pred, frame_idx, time_sec, is_keyframe, t_val=0.0,
                        psnr=50.0, ssim=1.0, recall=1.0, density=1.0, skel_rec=1.0):
    h, w = img_gt.shape[:2]
    c_gt = cv2.cvtColor(img_gt, cv2.COLOR_GRAY2BGR) if len(img_gt.shape) == 2 else img_gt.copy()
    c_pred = cv2.cvtColor(img_pred, cv2.COLOR_GRAY2BGR) if len(img_pred.shape) == 2 else img_pred.copy()

    banner_h = 36
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Banner Trái: Ground Truth
    cv2.rectangle(c_gt, (0, 0), (w, banner_h), (25, 25, 25), -1)
    cv2.putText(c_gt, f"GROUND TRUTH [#{frame_idx:02d} | {time_sec:.1f}s | 540p]",
                (12, 24), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # Banner Phải: Reconstructed
    cv2.rectangle(c_pred, (0, 0), (w, banner_h), (25, 25, 25), -1)
    if is_keyframe:
        title_pred = f"KEYFRAME [INPUT #{frame_idx:02d} | {time_sec:.1f}s]"
        color_pred = (0, 220, 255)
    else:
        title_pred = f"TPS V3.0-R2 SOUP [t={t_val:.2f} | #{frame_idx:02d}]"
        color_pred = (0, 255, 120)
    cv2.putText(c_pred, title_pred, (12, 24), font, 0.55, color_pred, 2, cv2.LINE_AA)

    # Thanh thông số dưới chân panel phải
    stat_h = 28
    cv2.rectangle(c_pred, (0, h - stat_h), (w, h), (20, 20, 20), -1)
    if is_keyframe:
        stat_text = "Input Keyframe Reference (Exact 100%)"
    else:
        stat_text = f"PSNR: {psnr:.2f}dB | SSIM: {ssim:.4f} | Rec: {recall*100:.1f}% | Skel: {skel_rec*100:.1f}% | Dens: {density:.2f}x"
    cv2.putText(c_pred, stat_text, (12, h - 9), font, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

    return np.hstack([c_gt, c_pred])


# ==============================================================================
# HÀM NẠP MÔ HÌNH THÔNG MINH
# ==============================================================================
def load_tps_model(ckpt_path):
    print(f"[*] Đang nạp mô hình TPS từ: {ckpt_path}")
    print(f"    - Thiết bị: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Không tìm thấy file checkpoint: {ckpt_path}")

    model = TPSInbetweenFinetune_v10().to(DEVICE)
    raw = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    if isinstance(raw, dict):
        if 'model' in raw:
            sd = raw['model']
        elif 'state_dict' in raw:
            sd = raw['state_dict']
        else:
            sd = raw
    else:
        sd = raw

    clean_sd = {k.replace('module.', '').replace('base_model.', ''): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    print(f"[✓] Nạp trọng số thành công! (Missing keys: {len(missing)}, Unexpected: {len(unexpected)})")
    model.eval()
    return model


# ==============================================================================
# QUY TRÌNH CHẠY CONTINUOUS BENCHMARK CHÍNH
# ==============================================================================
def run_continuous_test(spec_file=DEFAULT_SPEC, input_dir=DEFAULT_INPUT_DIR, ckpt_path=DEFAULT_CKPT):
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    rec_dir = os.path.join(OUTPUT_BASE_DIR, 'frames_reconstructed')
    gt_dir = os.path.join(OUTPUT_BASE_DIR, 'frames_ground_truth')
    sbs_dir = os.path.join(OUTPUT_BASE_DIR, 'frames_side_by_side')
    os.makedirs(rec_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(sbs_dir, exist_ok=True)

    csv_out = os.path.join(OUTPUT_BASE_DIR, 'test_v30r_benchmark_results.csv')
    json_out = os.path.join(OUTPUT_BASE_DIR, 'test_v30r_summary.json')
    mp4_out = os.path.join(OUTPUT_BASE_DIR, 'test_v30r_side_by_side.mp4')
    gif_out = os.path.join(OUTPUT_BASE_DIR, 'test_v30r_comparison_preview.gif')
    montage_out = os.path.join(OUTPUT_BASE_DIR, 'test_v30r_montage_24frames.png')

    print("=" * 85)
    print("🚀 BẮT ĐẦU CONTINUOUS TEST TRÊN CHUỖI 540p VỚI TPS V3.0-REMASTERED R2")
    print(f"   - File cấu hình Keyframe/Inbetween : {spec_file}")
    print(f"   - Thư mục ảnh nguồn               : {input_dir}")
    print(f"   - Checkpoint mô hình               : {ckpt_path}")
    print(f"   - Thư mục xuất thành phẩm          : {OUTPUT_BASE_DIR}")
    print("=" * 85)

    with open(spec_file, 'r', encoding='utf-8') as f:
        spec_data = json.load(f)

    frames_spec = spec_data['frames']
    total_frames = len(frames_spec)
    key_indices = set(spec_data['key_indices'])

    # 1. Nạp toàn bộ ảnh từ thư mục test
    gt_images = []
    time_files = []
    for item in frames_spec:
        fn = item['filename']
        fp = os.path.join(input_dir, fn)
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Không tìm thấy file ảnh: {fp}")
        im = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        gt_images.append(im)
        time_files.append((item['time_sec'], fn))

    h, w = gt_images[0].shape
    print(f"[*] Kích thước ảnh gốc: {w}x{h} (540p)")
    pad_h = ((h + 31) // 32) * 32
    pad_w = ((w + 31) // 32) * 32
    print(f"    - Kích thước sau padding bội số 32: {pad_w}x{pad_h}")

    # 2. Khởi tạo mô hình
    model = load_tps_model(ckpt_path)

    reconstructed_frames = [None] * total_frames
    records = []

    # 3. Gán trước các Keyframe
    for item in frames_spec:
        idx = item['frame_idx']
        if item['type'] == 'KEYFRAME':
            reconstructed_frames[idx] = gt_images[idx].copy()
            cur_sec = item['time_sec']
            cv2.imwrite(os.path.join(rec_dir, f"frame_{idx:02d}_{cur_sec:.1f}s_KEY.png"), gt_images[idx])
            cv2.imwrite(os.path.join(gt_dir, f"gt_{idx:02d}_{cur_sec:.1f}s.png"), gt_images[idx])

    # 4. Gom nhóm các đoạn nội suy giữa 2 keyframe liên tiếp
    sorted_keys = sorted(list(key_indices))
    segments = []
    for i in range(len(sorted_keys) - 1):
        k_a = sorted_keys[i]
        k_b = sorted_keys[i + 1]
        inbetweens = [item for item in frames_spec if item['type'] == 'INBETWEEN' and item['ref_key_a'] == k_a and item['ref_key_b'] == k_b]
        segments.append((k_a, k_b, inbetweens))

    t_start_total = time.time()
    for seg_i, (idx_a, idx_b, inbetweens) in enumerate(segments):
        t_sec_a = frames_spec[idx_a]['time_sec']
        t_sec_b = frames_spec[idx_b]['time_sec']
        M = len(inbetweens)
        print(f"\n[>>>] Đoạn #{seg_i+1}: Key #{idx_a:02d} ({t_sec_a:.1f}s) ──► Key #{idx_b:02d} ({t_sec_b:.1f}s) | Sinh {M} frame inbetween")

        im_a = gt_images[idx_a]
        im_b = gt_images[idx_b]

        f0 = 1.0 - (im_a.astype(np.float32) / 255.0)
        f1 = 1.0 - (im_b.astype(np.float32) / 255.0)
        f0_pad = np.pad(f0, ((0, pad_h - h), (0, pad_w - w)), mode='constant')
        f1_pad = np.pad(f1, ((0, pad_h - h), (0, pad_w - w)), mode='constant')
        t0 = torch.from_numpy(f0_pad).unsqueeze(0).unsqueeze(0).to(DEVICE)
        t1 = torch.from_numpy(f1_pad).unsqueeze(0).unsqueeze(0).to(DEVICE)

        kps_raw = extract_sift_keypoints(im_a, im_b, max_pts=80)
        num_matches = kps_raw.shape[1]
        print(f"      - SIFT Keypoint Matches: {num_matches} điểm")

        if num_matches >= 4:
            kps0 = kps_raw[0].copy()
            kps1 = kps_raw[1].copy()
            kps0[:, 0] = (kps0[:, 0] * w) / pad_w
            kps0[:, 1] = (kps0[:, 1] * h) / pad_h
            kps1[:, 0] = (kps1[:, 0] * w) / pad_w
            kps1[:, 1] = (kps1[:, 1] * h) / pad_h
            c0 = torch.from_numpy(kps0).to(DEVICE)
            c1 = torch.from_numpy(kps1).to(DEVICE)
            theta01, c0_u = robust_tps_theta_from_points(c0, c1, lambd=0.5, add_anchors=True)
            theta10, c1_u = robust_tps_theta_from_points(c1, c0, lambd=0.5, add_anchors=True)
            flow01 = tps_flow_from_theta(theta01, c0_u, pad_h, pad_w, DEVICE)
            flow10 = tps_flow_from_theta(theta10, c1_u, pad_h, pad_w, DEVICE)
        else:
            flow01 = torch.zeros((1, 2, pad_h, pad_w), device=DEVICE)
            flow10 = torch.zeros((1, 2, pad_h, pad_w), device=DEVICE)

        # Suy luận từng inbetween frame
        for row in inbetweens:
            cur_idx = row['frame_idx']
            t_val = float(row['t_val'])
            cur_sec = float(row['time_sec'])
            gt_frame = gt_images[cur_idx]

            t_step_start = time.time()
            with torch.no_grad():
                out_tensor, _ = model(t0, t1, float(t_val), flow01, flow10)
            step_lat = (time.time() - t_step_start) * 1000.0

            pred_np = ((1.0 - out_tensor[0, 0, :h, :w].clamp(0.0, 1.0)) * 255.0).byte().cpu().numpy()
            reconstructed_frames[cur_idx] = pred_np

            cv2.imwrite(os.path.join(rec_dir, f"frame_{cur_idx:02d}_{cur_sec:.1f}s_INBETWEEN.png"), pred_np)
            cv2.imwrite(os.path.join(gt_dir, f"gt_{cur_idx:02d}_{cur_sec:.1f}s.png"), gt_frame)

            # Tính các chỉ số chất lượng
            psnr = compute_psnr(pred_np, gt_frame)
            ssim = compute_ssim(pred_np, gt_frame)
            stroke_metrics = compute_stroke_metrics(pred_np, gt_frame)
            ghost_px = compute_ghost_pixels(pred_np, gt_frame, is_blank=stroke_metrics['is_blank_collapse'])
            mae = compute_ie_mae(pred_np, gt_frame)

            # Tính các chỉ số Topology chuẩn hóa Torch GPU
            p_ink = 1.0 - (torch.from_numpy(pred_np).float().to(DEVICE).unsqueeze(0).unsqueeze(0) / 255.0)
            g_ink = 1.0 - (torch.from_numpy(gt_frame).float().to(DEVICE).unsqueeze(0).unsqueeze(0) / 255.0)
            topo = compute_topology_metrics_torch(p_ink, g_ink)

            print(f"        [✓] Inbetween #{cur_idx:02d} ({cur_sec:.1f}s, t={t_val:.3f}): PSNR={psnr:.2f}dB | SSIM={ssim:.4f} | Recall={stroke_metrics['stroke_recall']*100:.1f}% | SkelRec={topo['skel_recall']*100:.1f}% | TipRec={topo['tip_recall']*100:.1f}% | Dens={stroke_metrics['stroke_density_ratio']:.2f}x | Lat={step_lat:.1f}ms")

            records.append({
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
                'latency_ms': round(step_lat, 1)
            })

    total_infer_time = time.time() - t_start_total
    num_inbetweens = len(records)
    print(f"\n[✓] Hoàn tất suy luận {num_inbetweens} frame inbetween trong {total_infer_time:.2f}s (Trung bình: {total_infer_time/num_inbetweens*1000:.1f}ms/frame)!")

    # 5. Đóng gói bảng dữ liệu và vẽ HUD Side-by-Side
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
            rec_entry = next(r for r in records if r['frame_idx'] == i)

        final_records.append(rec_entry)
        sbs = draw_comparison_hud(
            gt_frame, rec_frame, i, cur_sec, is_key,
            t_val=rec_entry['t_val'], psnr=rec_entry['psnr'], ssim=rec_entry['ssim'],
            recall=rec_entry['stroke_recall'], density=rec_entry['stroke_density'],
            skel_rec=rec_entry['skel_recall']
        )
        side_by_side_frames.append(sbs)
        cv2.imwrite(os.path.join(sbs_dir, f"sbs_{i:02d}_{cur_sec:.1f}s.png"), sbs)

    # 6. Xuất bảng CSV & JSON
    df_out = pd.DataFrame(final_records).sort_values('frame_idx').reset_index(drop=True)
    df_out.to_csv(csv_out, index=False)
    # Sao chép thêm một bản ra thư mục gốc để dễ đối chiếu
    df_out.to_csv(os.path.join(PROJECT_ROOT, 'test_v30r_benchmark_results.csv'), index=False)
    print(f"[✓] Đã xuất bảng CSV kết quả: {csv_out}")

    inbetween_df = df_out[df_out['type'] == 'INBETWEEN']
    summary = {
        'model_ckpt': os.path.basename(ckpt_path),
        'sequence': 'lineart_experiments/test (540p)',
        'resolution': f"{w}x{h}",
        'total_frames': total_frames,
        'keyframes_count': len(key_indices),
        'keyframes_pct': round(len(key_indices) / total_frames * 100, 2),
        'inbetweens_count': num_inbetweens,
        'inbetweens_pct': round(num_inbetweens / total_frames * 100, 2),
        'mean_psnr_inbetween': round(float(inbetween_df['psnr'].mean()), 2),
        'mean_ssim_inbetween': round(float(inbetween_df['ssim'].mean()), 4),
        'mean_stroke_recall_inbetween': round(float(inbetween_df['stroke_recall'].mean()), 4),
        'mean_stroke_density_inbetween': round(float(inbetween_df['stroke_density'].mean()), 4),
        'mean_skel_recall_inbetween': round(float(inbetween_df['skel_recall'].mean()), 4),
        'mean_tip_recall_inbetween': round(float(inbetween_df['tip_recall'].mean()), 4),
        'mean_wcd_inbetween': round(float(inbetween_df['wcd'].mean()), 6),
        'mean_semd_inbetween': round(float(inbetween_df['semd'].mean()), 6),
        'mean_ghost_pixels_inbetween': round(float(inbetween_df['ghost_pixels'].mean()), 1),
        'mean_ie_mae_inbetween': round(float(inbetween_df['ie_mae'].mean()), 2),
        'mean_latency_ms': round(float(inbetween_df['latency_ms'].mean()), 1)
    }

    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(PROJECT_ROOT, 'test_v30r_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[✓] Đã xuất bản JSON tóm tắt: {json_out}")

    # 7. Xuất Video MP4 & GIF
    sbs_h, sbs_w = side_by_side_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(mp4_out, fourcc, 8.0, (sbs_w, sbs_h))
    for sbs in side_by_side_frames:
        out_video.write(sbs)
    out_video.release()
    print(f"[✓] Đã tạo video MP4 liên tục: {mp4_out}")

    # GIF Preview
    pil_frames = []
    for sbs in side_by_side_frames:
        sbs_small = cv2.resize(sbs, (sbs_w // 2, sbs_h // 2), interpolation=cv2.INTER_AREA)
        sbs_rgb = cv2.cvtColor(sbs_small, cv2.COLOR_BGR2RGB)
        pil_frames.append(Image.fromarray(sbs_rgb))

    pil_frames[0].save(
        gif_out, save_all=True, append_images=pil_frames[1:],
        duration=125, loop=0 # ~8 FPS
    )
    print(f"[✓] Đã tạo ảnh GIF động preview: {gif_out}")

    # 8. Bảng Montage 24 frame (6 cột x 4 hàng)
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
            lbl = f"KEY #{f_i:02d}" if is_k else f"AI #{f_i:02d}"
            cv2.putText(f_bgr, lbl, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, border_col, 2)
            cols.append(f_bgr)
        montage_rows.append(np.hstack(cols))
    montage_full = np.vstack(montage_rows)
    cv2.imwrite(montage_out, montage_full)
    cv2.imwrite(os.path.join(PROJECT_ROOT, 'test_v30r_montage_24frames.png'), montage_full)
    print(f"[✓] Đã tạo ảnh Montage 24 frame: {montage_out}")

    print("\n" + "=" * 85)
    print("🎉 HOÀN TẤT TOÀN DIỆN CONTINUOUS TEST TRÊN TẬP LINEART_EXPERIMENTS/TEST!")
    print(f"   - Model Checkpoint       : {summary['model_ckpt']}")
    print(f"   - PSNR trung bình        : {summary['mean_psnr_inbetween']} dB")
    print(f"   - SSIM trung bình        : {summary['mean_ssim_inbetween']}")
    print(f"   - Stroke Recall          : {summary['mean_stroke_recall_inbetween']*100:.2f}%")
    print(f"   - Skeleton Recall        : {summary['mean_skel_recall_inbetween']*100:.2f}%")
    print(f"   - Hair-Tip Recall        : {summary['mean_tip_recall_inbetween']*100:.2f}%")
    print(f"   - Stroke Density Ratio   : {summary['mean_stroke_density_inbetween']:.2f}x")
    print(f"   - Weighted Chamfer (WCD) : {summary['mean_wcd_inbetween']}")
    print(f"   - Latency trung bình     : {summary['mean_latency_ms']} ms/frame")
    print("=" * 85)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run continuous test on test sequence")
    parser.add_argument('--spec', type=str, default=DEFAULT_SPEC, help="Path to continuous spec JSON")
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT_DIR, help="Path to input frames folder")
    parser.add_argument('--ckpt', type=str, default=DEFAULT_CKPT, help="Path to model checkpoint")
    args = parser.parse_args()

    run_continuous_test(args.spec, args.input, args.ckpt)
