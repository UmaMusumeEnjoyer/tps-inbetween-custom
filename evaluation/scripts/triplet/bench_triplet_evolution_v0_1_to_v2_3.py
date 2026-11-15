#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Benchmark & Trực Quan Hóa So Sánh Đối Đầu Side-by-Side:
Tiến Trình Phát Triển Mô Hình Finetune TPS-Inbetween Từ V0.1 Đến V2.3
Tập Dữ Liệu: sakuga_golden_dataset_tps_ready_v1_patch2 (Trích xuất ngẫu nhiên từ tập Validation)

Bố cục trình bày:
- Trải dài theo hàng dọc từ trên xuống dưới theo thứ tự lịch sử phiên bản: v0.1 -> v2.3.
- Mỗi hàng đại diện cho 1 thế hệ mô hình gồm 6 cột chuẩn benchmark đối sánh:
  Col 1: Keyframe 0 (t = 0.0) [INPUT POSE]
  Col 2: TPS Coarse Warp [GEOMETRIC BASELINE]
  Col 3: Finetuned Prediction vX.Y [MODEL INBETWEEN]
  Col 4: Ground Truth (t = t_val) [SAKUGA ARTIST TARGET]
  Col 5: Difference Residual Heatmap vs GT [GHOST & ERROR HEATMAP]
  Col 6: Zoomed 400% Split View [MODEL VS GT LINE QUALITY]
"""

import os
import sys
import time
import json
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import cv2

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
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from bench_triplet_01_v0_1_eval import TPSInbetweenFinetune
from bench_triplet_02_v0_2_eval import TPSInbetweenFinetune_v2
from bench_triplet_03_v0_3_eval import TPSInbetweenFinetune_v3
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

MODEL_CATALOG = [
    {
        'version': 'v0.1',
        'name': 'Finetuned V0.1',
        'tag': '[V0.1]',
        'codename': 'Kaggle Baseline (Mode Collapse)',
        'class': TPSInbetweenFinetune,
        'ckpt': os.path.join(PROJECT_ROOT, 'checkpoints/experiments/v0.1/model_tps_v0.1_best.pt'),
        'color': (70, 70, 255),
        'desc': 'Thử nghiệm ban đầu trên Kaggle; bị sụp đổ chế độ ảnh trắng (Blank Collapse).'
    },
    {
        'version': 'v0.2',
        'name': 'Finetuned V0.2',
        'tag': '[V0.2]',
        'codename': 'ASF Head Prototype (Thickening)',
        'class': TPSInbetweenFinetune_v2,
        'ckpt': os.path.join(PROJECT_ROOT, 'checkpoints/experiments/v0.2/model_tps_v0.2_best.pt'),
        'color': (0, 165, 255),
        'desc': 'Khắc phục ảnh trắng, bổ sung ASF Head sơ khai; còn phình nét và quầng mờ.'
    },
    {
        'version': 'v0.3',
        'name': 'Finetuned V0.3',
        'tag': '[V0.3]',
        'codename': 'ASF-GR Engine & Robust TPS',
        'class': TPSInbetweenFinetune_v3,
        'ckpt': os.path.join(PROJECT_ROOT, 'checkpoints/experiments/v0.3/model_tps_v0.3_best.pt'),
        'color': (0, 215, 255),
        'desc': 'ASF-GR Engine với Ghost Erasure Gate; giải quyết triệt để quầng mờ diện rộng.'
    },
    {
        'version': 'v1.0',
        'name': 'Finetuned V1.0',
        'tag': '[V1.0]',
        'codename': 'Production ASF-GR Standard',
        'class': TPSInbetweenFinetune_v10,
        'ckpt': os.path.join(PROJECT_ROOT, 'checkpoints/experiments/v1.0/model_tps_v1.0_best.pt'),
        'color': (255, 180, 0),
        'desc': 'Kiến trúc hoàn thiện chuyển giao sản xuất; đạt độ ổn định 100% trên Sakuga Golden.'
    },
    {
        'version': 'v1.1',
        'name': 'Finetuned V1.1',
        'tag': '[V1.1]',
        'codename': 'Boundary Anchor TPS Warp',
        'class': TPSInbetweenFinetune_v10,
        'ckpt': os.path.join(PROJECT_ROOT, 'checkpoints/experiments/v1.1/model_tps_v1.1_best.pt'),
        'color': (230, 200, 50),
        'desc': 'Bổ sung 8 điểm neo biên (Boundary Anchors); triệt tiêu vỡ khung và trôi viền.'
    },
    {
        'version': 'v2.0',
        'name': 'Finetuned V2.0',
        'tag': '[V2.0]',
        'codename': 'Stroke Regularization V2',
        'class': TPSInbetweenFinetune_v10,
        'ckpt': os.path.join(PROJECT_ROOT, 'checkpoints/experiments/v2.0/v2.0.2/tps_finetune_v2.0_artifacts/checkpoints_v1.1/model_tps_v2.0_best.pt'),
        'color': (200, 220, 100),
        'desc': 'Mở rộng tập dữ liệu Sakuga Golden V1; đạt độ tương đồng cấu trúc SSIM cao.'
    },
    {
        'version': 'v2.1',
        'name': 'Finetuned V2.1',
        'tag': '[V2.1]',
        'codename': 'Patch2 Stratified Golden Dataset',
        'class': TPSInbetweenFinetune_v10,
        'ckpt': os.path.join(PROJECT_ROOT, 'checkpoints/experiments/v2.1/tps_finetune_v2.1_artifacts/checkpoints_v2.1/model_tps_v1.1_best.pt'),
        'color': (150, 255, 120),
        'desc': 'Huấn luyện trên Golden Dataset Patch2; giảm bớt ghosting tại các pha chuyển động lớn.'
    },
    {
        'version': 'v2.2',
        'name': 'Finetuned V2.2',
        'tag': '[V2.2]',
        'codename': 'Fast GPU Chamfer & Temp Reversal',
        'class': TPSInbetweenFinetune_v10,
        'ckpt': os.path.join(PROJECT_ROOT, 'checkpoints/experiments/v2.2/tps_finetune_v2.2_artifacts/checkpoints_v2.2/model_tps_v2.2_best.pt'),
        'color': (100, 255, 180),
        'desc': 'Tối ưu xấp xỉ Chamfer GPU siêu tốc và Temporal Reversal cân bằng chiều thời gian.'
    },
    {
        'version': 'v2.3',
        'name': 'Finetuned V2.3',
        'tag': '[V2.3]',
        'codename': 'Anti-Thickening & FP32 (Target Winner)',
        'class': TPSInbetweenFinetune_v10,
        'ckpt': os.path.join(PROJECT_ROOT, 'checkpoints/experiments/v2.3/tps_finetune_v2.3_artifacts/checkpoints_v2.3/model_tps_v2.3_best.pt'),
        'color': (0, 255, 120),
        'desc': 'Anti-Thickening Stroke Density Loss; giảm -34.3% bóng ma, phục hồi nét thanh mảnh nguyên bản.'
    }
]


def add_hud(panel_bgr, header_title, footer_tag, footer_stat, header_color=(255, 255, 255), tag_color=(0, 220, 255)):
    h, w = panel_bgr.shape[:2]
    out = panel_bgr.copy()

    # Header Bar
    cv2.rectangle(out, (0, 0), (w, 38), (24, 24, 28), -1)
    cv2.line(out, (0, 38), (w, 38), (50, 50, 60), 1)
    cv2.putText(out, header_title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, header_color, 2, cv2.LINE_AA)

    # Footer Bar
    cv2.rectangle(out, (0, h - 34), (w, h), (20, 20, 24), -1)
    cv2.line(out, (0, h - 34), (w, h - 34), (50, 50, 60), 1)
    cv2.putText(out, footer_tag, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.46, tag_color, 2, cv2.LINE_AA)
    cv2.putText(out, footer_stat, (85, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA)

    return out


def create_zoom_split_crop(pred_np, gt_np, cy, cx, crop_h=100, crop_w=178, out_h=540, out_w=960, is_blank=False):
    h_orig, w_orig = pred_np.shape
    y1 = max(0, cy - crop_h // 2)
    y2 = min(h_orig, y1 + crop_h)
    x1 = max(0, cx - crop_w // 2)
    x2 = min(w_orig, x1 + crop_w)

    crop_p = pred_np[y1:y2, x1:x2]
    crop_g = gt_np[y1:y2, x1:x2]

    zoom_p = cv2.resize(crop_p, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    zoom_g = cv2.resize(crop_g, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    half_w = out_w // 2
    split_bgr = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    split_bgr[:, :half_w] = cv2.cvtColor(zoom_p[:, :half_w], cv2.COLOR_GRAY2BGR)
    split_bgr[:, half_w:] = cv2.cvtColor(zoom_g[:, half_w:], cv2.COLOR_GRAY2BGR)

    cv2.line(split_bgr, (half_w, 0), (half_w, out_h), (0, 220, 255), 2)
    cv2.putText(split_bgr, "MODEL (400% ZOOM)", (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(split_bgr, "ARTIST GROUND TRUTH", (half_w + 16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 120), 2, cv2.LINE_AA)

    if is_blank:
        cv2.rectangle(split_bgr, (10, out_h // 2 - 25), (half_w - 10, out_h // 2 + 25), (0, 0, 180), -1)
        cv2.putText(split_bgr, "NO STROKES (BLANK)", (20, out_h // 2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return split_bgr


def load_all_models(device):
    """Nạp trước toàn bộ 9 mô hình vào bộ nhớ để tái sử dụng siêu tốc."""
    print("📦 Đang nạp trước toàn bộ 9 checkpoints mô hình vào GPU...")
    loaded = []
    for meta in MODEL_CATALOG:
        v_cls = meta['class']
        v_ckpt = meta['ckpt']
        model = v_cls().to(device)
        ckpt = torch.load(v_ckpt, map_location=device, weights_only=False)
        sd = ckpt['model_state_dict'] if (isinstance(ckpt, dict) and 'model_state_dict' in ckpt) else (ckpt['state_dict'] if (isinstance(ckpt, dict) and 'state_dict' in ckpt) else ckpt)
        cleaned_sd = {k.replace('module.', ''): v for k, v in sd.items()}
        model.load_state_dict(cleaned_sd, strict=False)
        model.eval()
        loaded.append((meta, model))
    print(f"✓ Đã nạp thành công 9/9 mô hình vào {device}!")
    return loaded


def process_single_clip(selected_row, loaded_models, dataset_dir, output_dir, device, pw=640, ph=360):
    clip_id = str(selected_row['clip_id'])
    t_val = float(selected_row['t'])
    disp_val = float(selected_row.get('mean_disp', 0.0))
    inliers_val = int(selected_row.get('inliers', 0))

    print("\n" + "=" * 90)
    print(f"🎬 ĐANG XỬ LÝ CLIP: {clip_id} | t={t_val:.2f} | disp={disp_val:.1f}px | inliers={inliers_val}")
    print("=" * 90)

    p0 = os.path.join(dataset_dir, selected_row['frame0'])
    pgt = os.path.join(dataset_dir, selected_row['frame_gt'])
    p1 = os.path.join(dataset_dir, selected_row['frame1'])
    pmatch = os.path.join(dataset_dir, selected_row['match_file'])

    img0_pil = Image.open(p0).convert('L')
    imggt_pil = Image.open(pgt).convert('L')
    img1_pil = Image.open(p1).convert('L')

    f0 = 1.0 - (np.array(img0_pil, dtype=np.float32) / 255.0)
    fgt = 1.0 - (np.array(imggt_pil, dtype=np.float32) / 255.0)
    f1 = 1.0 - (np.array(img1_pil, dtype=np.float32) / 255.0)

    orig_h, orig_w = f0.shape
    pad_h = ((orig_h + 31) // 32) * 32
    pad_w = ((orig_w + 31) // 32) * 32

    f0_pad = np.pad(f0, ((0, pad_h - orig_h), (0, pad_w - orig_w)), mode='constant')
    fgt_pad = np.pad(fgt, ((0, pad_h - orig_h), (0, pad_w - orig_w)), mode='constant')
    f1_pad = np.pad(f1, ((0, pad_h - orig_h), (0, pad_w - orig_w)), mode='constant')

    if os.path.exists(pmatch):
        kps = np.load(pmatch).astype(np.float32)
        kps0 = kps[0].copy()
        kps1 = kps[1].copy()
        kps0[:, 0] = (kps0[:, 0] * orig_w) / pad_w
        kps0[:, 1] = (kps0[:, 1] * orig_h) / pad_h
        kps1[:, 0] = (kps1[:, 0] * orig_w) / pad_w
        kps1[:, 1] = (kps1[:, 1] * orig_h) / pad_h
        kps_pad = np.stack([kps0, kps1], axis=0)
    else:
        kps_pad = np.zeros((2, 0, 2), dtype=np.float32)

    t0_dev = torch.from_numpy(f0_pad).unsqueeze(0).unsqueeze(0).float().to(device)
    tgt_dev = torch.from_numpy(fgt_pad).unsqueeze(0).unsqueeze(0).float().to(device)
    t1_dev = torch.from_numpy(f1_pad).unsqueeze(0).unsqueeze(0).float().to(device)
    t_tensor = torch.tensor([[t_val]], dtype=torch.float32).view(1, 1, 1, 1).to(device)

    gt_np = ((1.0 - tgt_dev[:, :, :orig_h, :orig_w]) * 255.0).clamp(0, 255).byte().cpu().numpy()[0, 0]
    k0_np = np.array(img0_pil, dtype=np.uint8)

    stroke_mask = (gt_np < 200)
    if np.any(stroke_mask):
        ys, xs = np.where(stroke_mask)
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
    else:
        cy, cx = orig_h // 2, orig_w // 2

    k0_bgr = cv2.resize(cv2.cvtColor(k0_np, cv2.COLOR_GRAY2BGR), (pw, ph), interpolation=cv2.INTER_AREA)
    k0_panel = add_hud(k0_bgr, "Keyframe 0 (Input)", "[INPUT]", f"{orig_w}x{orig_h} Lineart | t=0.00",
                       header_color=(255, 255, 255), tag_color=(0, 220, 255))

    gt_bgr = cv2.resize(cv2.cvtColor(gt_np, cv2.COLOR_GRAY2BGR), (pw, ph), interpolation=cv2.INTER_AREA)
    gt_panel = add_hud(gt_bgr, f"Ground Truth (Target t={t_val:.2f})", "[TARGET]", "Sakuga Artist Inbetween",
                       header_color=(255, 255, 255), tag_color=(0, 255, 120))

    rows_panels = []
    records = []
    coarse_baseline_record = None
    p_coarse_cached = None

    for m_idx, (meta, model) in enumerate(loaded_models):
        v_name = meta['version']
        v_tag = meta['tag']
        v_title = f"{meta['name']} - {meta['codename']}"
        v_hdr_color = meta['color']

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        with torch.no_grad():
            out, coarse = model(t0_dev, t1_dev, t_tensor, [kps_pad], max_kps=80, return_coarse=True)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - t_start) * 1000.0

        p_np = ((1.0 - out[:, :, :orig_h, :orig_w]) * 255.0).clamp(0, 255).byte().cpu().numpy()[0, 0]
        if p_coarse_cached is None:
            p_coarse_cached = ((1.0 - coarse[:, :, :orig_h, :orig_w]) * 255.0).clamp(0, 255).byte().cpu().numpy()[0, 0]

        stroke_res = compute_stroke_metrics(p_np, gt_np)
        is_blank = stroke_res['is_blank_collapse']
        psnr_val = compute_psnr(p_np, gt_np)
        ssim_val = compute_ssim(p_np, gt_np)
        ghost_val = compute_ghost_pixels(p_np, gt_np, is_blank=is_blank)
        dens_val = stroke_res['stroke_density_ratio']
        recall_val = stroke_res['stroke_recall']
        mae_val = compute_ie_mae(p_np, gt_np)
        cd_val = compute_chamfer_distance(p_np, gt_np)
        tort_val = compute_line_tortuosity(p_np, is_blank=is_blank)

        print(f"[{m_idx + 1}/9] {v_name:<5} | PSNR: {psnr_val:5.2f}dB | Ghost: {ghost_val:6}px | Dens: {dens_val:5.3f} | Blank: {str(is_blank):<5}")

        rec = {
            'clip_id': clip_id,
            't': t_val,
            'version': v_name,
            'name': meta['name'],
            'codename': meta['codename'],
            'psnr': round(psnr_val, 2),
            'ssim': round(ssim_val, 4),
            'ghost_pixels': int(ghost_val),
            'density_ratio': round(dens_val, 4),
            'stroke_recall': round(recall_val, 4),
            'chamfer_distance': round(cd_val, 6),
            'ie_mae': round(mae_val, 2),
            'tortuosity': round(tort_val, 3),
            'is_blank_collapse': bool(is_blank),
            'infer_ms': round(infer_ms, 1)
        }
        records.append(rec)

        if coarse_baseline_record is None:
            c_psnr = compute_psnr(p_coarse_cached, gt_np)
            c_ssim = compute_ssim(p_coarse_cached, gt_np)
            c_stroke = compute_stroke_metrics(p_coarse_cached, gt_np)
            c_ghost = compute_ghost_pixels(p_coarse_cached, gt_np)
            coarse_baseline_record = {
                'psnr': c_psnr,
                'ssim': c_ssim,
                'ghost': c_ghost,
                'density': c_stroke['stroke_density_ratio']
            }

        coarse_bgr = cv2.resize(cv2.cvtColor(p_coarse_cached, cv2.COLOR_GRAY2BGR), (pw, ph), interpolation=cv2.INTER_AREA)
        coarse_panel = add_hud(coarse_bgr, "TPS Coarse Warp Baseline", "[COARSE]",
                               f"{coarse_baseline_record['psnr']:.1f}dB | Ghost: {coarse_baseline_record['ghost']}px | Dens: {coarse_baseline_record['density']:.2f}",
                               header_color=(200, 200, 200), tag_color=(0, 220, 255))

        col_pred = p_np.copy()
        if is_blank:
            cv2.rectangle(col_pred, (30, orig_h // 2 - 45), (orig_w - 30, orig_h // 2 + 45), (0, 0, 0), -1)
            cv2.putText(col_pred, "BLANK COLLAPSE (0 STROKES)", (60, orig_h // 2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)

        pred_bgr = cv2.resize(cv2.cvtColor(col_pred, cv2.COLOR_GRAY2BGR), (pw, ph), interpolation=cv2.INTER_AREA)
        footer_stat_str = f"{psnr_val:.1f}dB | SSIM: {ssim_val:.3f} | Ghost: {ghost_val}px | Dens: {dens_val:.2f}"
        pred_panel = add_hud(pred_bgr, v_title, v_tag, footer_stat_str,
                             header_color=v_hdr_color,
                             tag_color=(0, 255, 120) if v_name == 'v2.3' else ((0, 0, 255) if is_blank else (0, 220, 255)))

        diff = np.abs(p_np.astype(np.float32) - gt_np.astype(np.float32))
        heatmap_raw = cv2.applyColorMap(np.clip(diff * 2.5, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_bgr = cv2.resize(heatmap_raw, (pw, ph), interpolation=cv2.INTER_AREA)
        hm_panel = add_hud(heatmap_bgr, f"Error Heatmap ({v_name} vs GT)", "[HEATMAP]",
                           f"Residual Error | Ghost: {ghost_val}px | MAE: {mae_val:.1f}",
                           header_color=(255, 255, 255), tag_color=(0, 220, 255))

        zoom_split = create_zoom_split_crop(p_np, gt_np, cy, cx, crop_h=100, crop_w=178, out_h=ph, out_w=pw, is_blank=is_blank)
        zoom_panel = add_hud(zoom_split, f"Zoom 400% Stroke Quality ({v_name})", "[ZOOM 4X]",
                             f"Left: {v_name} | Right: Artist GT | Dens: {dens_val:.2f}",
                             header_color=v_hdr_color, tag_color=(0, 220, 255))

        row_img = np.hstack([k0_panel, coarse_panel, pred_panel, gt_panel, hm_panel, zoom_panel])
        rows_panels.append(row_img)

    banner_w = pw * 6
    banner_h = 130
    banner = np.zeros((banner_h, banner_w, 3), dtype=np.uint8)
    for y in range(banner_h):
        alpha = y / banner_h
        color_val = int(18 + alpha * 10)
        banner[y, :] = (color_val + 2, color_val, color_val)
    cv2.line(banner, (0, banner_h - 2), (banner_w, banner_h - 2), (0, 255, 120), 2)

    main_title = f"TPS-INBETWEEN BENCHMARK: FINETUNE EVOLUTION V0.1 -> V2.3 ({clip_id})"
    cv2.putText(banner, main_title, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
    sub_title = (f"Bộ Dữ Liệu: sakuga_golden_dataset_tps_ready_v1_patch2 | Mẫu: {clip_id} "
                 f"| t = {t_val:.2f} | Độ Dời: {disp_val:.1f} px | Keypoints Inliers: {inliers_val}")
    cv2.putText(banner, sub_title, (24, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 1, cv2.LINE_AA)
    col_guide = ("BỐ CỤC 6 CỘT: [Col 1: Keyframe 0]  |  [Col 2: TPS Coarse Warp]  |  [Col 3: Finetuned Prediction (v0.1 -> v2.3)]  |  "
                 "[Col 4: Ground Truth Artist]  |  [Col 5: Sai Số Heatmap (JET)]  |  [Col 6: So Sánh Zoom 400% (Model Trái vs GT Phải)]")
    cv2.putText(banner, col_guide, (24, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

    all_rows = [banner] + rows_panels
    full_poster = np.vstack(all_rows)
    poster_path = os.path.join(output_dir, f"{clip_id}_t{t_val:.2f}_evolution_benchmark_v01_to_v23_6col.png")
    cv2.imwrite(poster_path, full_poster)

    compact_rows = []
    compact_banner_w = pw * 4
    compact_banner = np.zeros((100, compact_banner_w, 3), dtype=np.uint8)
    compact_banner[:] = (22, 22, 26)
    cv2.line(compact_banner, (0, 98), (compact_banner_w, 98), (0, 255, 120), 2)
    cv2.putText(compact_banner, f"TPS-INBETWEEN FINETUNING EVOLUTION (V0.1 -> V2.3) - {clip_id} (t={t_val:.2f})",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(compact_banner, "CỘT 1: FINETUNED OUTPUT  |  CỘT 2: GROUND TRUTH TARGET  |  CỘT 3: RESIDUAL HEATMAP  |  CỘT 4: 400% ZOOM CROP",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 1, cv2.LINE_AA)
    compact_rows.append(compact_banner)

    for r_img in rows_panels:
        p_pred = r_img[:, pw*2 : pw*3]
        p_gt = r_img[:, pw*3 : pw*4]
        p_hm = r_img[:, pw*4 : pw*5]
        p_zm = r_img[:, pw*5 : pw*6]
        compact_rows.append(np.hstack([p_pred, p_gt, p_hm, p_zm]))

    compact_poster = np.vstack(compact_rows)
    compact_path = os.path.join(output_dir, f"{clip_id}_t{t_val:.2f}_evolution_compact_4col.png")
    cv2.imwrite(compact_path, compact_poster)

    df_metrics = pd.DataFrame(records)
    csv_out = os.path.join(output_dir, f"{clip_id}_t{t_val:.2f}_evolution_metrics.csv")
    df_metrics.to_csv(csv_out, index=False, encoding='utf-8')

    json_data = {
        'benchmark_info': {
            'dataset': 'sakuga_golden_dataset_tps_ready_v1_patch2',
            'split': 'val',
            'clip_id': clip_id,
            't': t_val,
            'mean_disp': disp_val,
            'inliers': inliers_val,
            'device': str(device),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'poster_master_path': poster_path,
            'poster_compact_path': compact_path
        },
        'coarse_baseline': coarse_baseline_record,
        'models_evolution': records
    }
    json_out = os.path.join(output_dir, f"{clip_id}_t{t_val:.2f}_evolution_metrics.json")
    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

    print(f"✓ Đã hoàn tất {clip_id}: Master Poster={poster_path}")
    return poster_path, compact_path, json_data, records


def main():
    parser = argparse.ArgumentParser(description="Tạo ảnh benchmark đối sánh side-by-side v0.1 đến v2.3 trên Sakuga Golden Dataset")
    parser.add_argument('--dataset_dir', default='data/processed/sakuga_golden_dataset_tps_ready_v1_patch2',
                        help='Thư mục dataset Sakuga Golden Patch2')
    parser.add_argument('--clip_id', default=None,
                        help='Mã clip đơn lẻ chỉ định')
    parser.add_argument('--clip_ids', default=None,
                        help='Danh sách mã clip phân cách bằng dấu phẩy (vd: clip_002078,clip_003719,clip_003206)')
    parser.add_argument('--num_random', type=int, default=1,
                        help='Số lượng clip ngẫu nhiên cần benchmark')
    parser.add_argument('--seed', type=int, default=None,
                        help='Hạt giống sinh ngẫu nhiên')
    parser.add_argument('--output_dir', default='evaluation/outputs/triplet/comparison_v0.1_to_v2.3_evolution',
                        help='Thư mục lưu ảnh benchmark')
    parser.add_argument('--panel_w', type=int, default=640,
                        help='Chiều rộng mỗi ô hiển thị (640 tạo ra ảnh poster 3840px 4K)')
    parser.add_argument('--panel_h', type=int, default=360,
                        help='Chiều cao mỗi ô hiển thị')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 90)
    print("🚀 KHỞI ĐỘNG HỆ THỐNG BENCHMARK SIDE-BY-SIDE: FINETUNE V0.1 -> V2.3")
    print(f"   - Thiết bị suy luận : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"   - Thư mục dữ liệu   : {args.dataset_dir}")
    print(f"   - Thư mục kết xuất  : {args.output_dir}")
    print("=" * 90)

    csv_path = os.path.join(args.dataset_dir, 'triplets.csv')
    df = pd.read_csv(csv_path)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)

    selected_rows = []
    if args.clip_ids:
        c_list = [c.strip() for c in args.clip_ids.split(',') if c.strip()]
        for cid in c_list:
            matching = val_df[val_df['clip_id'] == cid]
            if not matching.empty:
                selected_rows.append(matching.iloc[0])
            else:
                print(f"[!] Cảnh báo: Không tìm thấy {cid} trong tập validation!")
    elif args.clip_id:
        matching = val_df[val_df['clip_id'] == args.clip_id]
        if matching.empty:
            raise ValueError(f"Không tìm thấy clip_id {args.clip_id} trong tập val!")
        selected_rows.append(matching.iloc[0])
    else:
        # Lọc các clip có chuyển động rõ nét (inliers >= 50, mean_disp >= 3.0)
        qualified = val_df[(val_df['inliers'] >= 50) & (val_df['mean_disp'] >= 3.0)].reset_index(drop=True)
        pool = qualified if len(qualified) >= args.num_random else val_df
        sampled_indices = random.sample(range(len(pool)), min(args.num_random, len(pool)))
        for s_idx in sampled_indices:
            selected_rows.append(pool.iloc[s_idx])

    print(f"📋 Danh sách {len(selected_rows)} clips được chọn để thực hiện benchmark đối sánh:")
    for i, r in enumerate(selected_rows, 1):
        print(f"   {i}. {r['clip_id']} | t={r['t']:.2f} | disp={r['mean_disp']:.1f}px | inliers={r['inliers']}")

    loaded_models = load_all_models(device)

    all_records = []
    generated_files = []
    for r in selected_rows:
        p_path, c_path, j_data, recs = process_single_clip(
            r, loaded_models, args.dataset_dir, args.output_dir, device, pw=args.panel_w, ph=args.panel_h
        )
        generated_files.append((p_path, c_path))
        all_records.extend(recs)

    # Xuất file tổng hợp đa clip
    df_all = pd.DataFrame(all_records)
    summary_csv = os.path.join(args.output_dir, 'multi_clip_evolution_summary.csv')
    df_all.to_csv(summary_csv, index=False, encoding='utf-8')
    print("\n" + "=" * 90)
    print(f"🎉 HOÀN TẤT TẤT CẢ! Đã lưu báo cáo tổng hợp đa clip tại: {summary_csv}")
    print("=" * 90)


if __name__ == '__main__':
    main()
