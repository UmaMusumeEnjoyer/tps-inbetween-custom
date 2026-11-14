#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_pha4_vs_pha3.py
--------------------------------------------------------------------------------
BỘ BENCHMARK ĐỐI ĐẦU 4 PHIÊN BẢN (HEAD-TO-HEAD BENCHMARK):
    1. Pha 2 Blended (Mốc cũ: model_tps_blended.pt)
    2. Pha 3 Remake Raw (M3_Best không Sharpener)
    3. Pha 3 Remake + Balanced Sharpener (Mô hình tốt nhất hiện tại)
    4. Pha 4 Occlusion Head (Mô hình thử nghiệm mới: TPSInbetweenOcclusion)

TẬP KIỂM CHỨNG:
    - Case 1: Chuỗi 24 frames thực tế (data/lineart_experiments/test - 17 in-betweens)
              Kiểm tra tóc wiggle, ghost tại các đoạn nét mới xuất hiện/mất đi.
    - Case 2: Hình tròn biến dạng di chuyển (Squash & Stretch Moving Circle)
    - Case 3: Chữ thập rỗng xoay biến dạng (Rotating Deforming Hollow Cross)
    - Case 4: Chữ thập đặc xoay biến dạng (Rotating Deforming Solid Cross)

LƯU TRỮ KẾT QUẢ CÁCH LY TẠI:
    benchmark_outputs_phase4_occlusion/
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
    import imageio
except ImportError:
    imageio = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PHASE4_DIR = os.path.join(REPO_ROOT, "kaggle_experiment_occlusion")
PHASE3_DIR = os.path.join(REPO_ROOT, "kaggle_finetune_sequence_phase3")
PHASE2_DIR = os.path.join(REPO_ROOT, "kaggle_finetune_sequence_phase2")

for _d in (SCRIPT_DIR, PHASE4_DIR, PHASE3_DIR, PHASE2_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from multi_layer_tps import solve_multi_layer_tps
from adaptive_sharpener import apply_adaptive_sharpener
from ddp_worker_tps_sequence_phase2 import (
    TPSInbetweenOriginal,
    load_tps_weights,
    evaluate_metrics_dual_regime,
)
from model_occlusion import (
    TPSInbetweenOcclusion,
    load_tps_occlusion_weights,
)
from phase4_occlusion_pipeline import (
    load_ink_from_input,
    ink_tensor_to_u8,
    mask_tensor_to_u8,
)

BASE_OUT_DIR = os.path.join(REPO_ROOT, "benchmark_outputs_phase4_occlusion")
LINEART_TEST_DIR = os.path.join(REPO_ROOT, "data", "lineart_experiments", "test")
GLUESTICK_NPZ = os.path.join(REPO_ROOT, "kaggle_finetune_sequence_final", "lineart_test_gluestick_matches.npz")

PHA2_CKPT = os.path.join(
    REPO_ROOT, "checkpoints", "Re-train", "Sequence 2 Blended",
    "checkpoints", "model_tps_blended.pt"
)
PHA3_CKPT = os.path.join(
    REPO_ROOT, "checkpoints", "Re-train", "Sequence 2 pha 3 remake",
    "checkpoints", "model_tps_finetuned_best.pt"
)
PHA4_CKPT = os.path.join(
    REPO_ROOT, "checkpoints", "Re-train", "Sequence 4 Occlusion Head",
    "model_tps_occlusion_best.pt"
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


# ==============================================================================
# BÀI TEST 1: CHUỖI 24 FRAMES THỰC TẾ
# ==============================================================================
def run_test_case_1(models, out_dir):
    print("\n" + "=" * 90)
    print("▶ CHẠY CASE 1: CHUỖI 24 FRAMES THỰC TẾ (17 IN-BETWEENS)")
    print(f"  Thư mục dữ liệu: {LINEART_TEST_DIR}")
    print(f"  Thư mục đích:    {out_dir}")
    print("=" * 90)

    clean_dir = os.path.join(out_dir, "clean_frames_phase4")
    masks_dir = os.path.join(out_dir, "occlusion_masks_phase4")
    sbs_dir = os.path.join(out_dir, "side_by_side_6col")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(sbs_dir, exist_ok=True)

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
    frames_p2 = [None] * total_frames
    frames_p3_raw = [None] * total_frames
    frames_p3_sharp = [None] * total_frames
    frames_p4_occ = [None] * total_frames
    masks_p4_m0 = [None] * total_frames
    masks_p4_m1 = [None] * total_frames

    for i in range(total_frames):
        ink_gt = load_ink_from_input(paths[i])
        u8_gt = np.clip((1.0 - ink_gt) * 255.0, 0, 255).astype(np.uint8)
        frames_gt[i] = u8_gt
        if i in kf_set:
            frames_p2[i] = u8_gt
            frames_p3_raw[i] = u8_gt
            frames_p3_sharp[i] = u8_gt
            frames_p4_occ[i] = u8_gt
            cv2.imwrite(os.path.join(clean_dir, os.path.basename(paths[i])), u8_gt)

    rows = []

    for s_idx in range(len(kf_idx) - 1):
        i0, i1 = kf_idx[s_idx], kf_idx[s_idx + 1]
        delta_f = i1 - i0
        ink0 = load_ink_from_input(paths[i0])
        ink1 = load_ink_from_input(paths[i1])
        h, w = ink0.shape
        h_pad, w_pad = ((h + 31) // 32) * 32, ((w + 31) // 32) * 32

        kps = gs_cache[f"seg_{i0}_{i1}"].astype(np.float32)
        cf01, cf10, _ = solve_multi_layer_tps(
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

            # 1. Pha 2 Blended
            with torch.no_grad():
                pr_p2, *_ = models['p2'](x0.float(), x1.float(), tt, cf01.float(), cf10.float())
            t_p2 = pr_p2[:, :, :h, :w].clamp(0.0, 1.0)
            frames_p2[f_idx] = ink_tensor_to_u8(t_p2)

            # 2. Pha 3 Remake Raw
            with torch.no_grad():
                pr_p3, *_ = models['p3'](x0.float(), x1.float(), tt, cf01.float(), cf10.float())
            t_p3_raw = pr_p3[:, :, :h, :w].clamp(0.0, 1.0)
            frames_p3_raw[f_idx] = ink_tensor_to_u8(t_p3_raw)

            # 3. Pha 3 Remake + Balanced Sharpener
            t_p3_sharp = apply_adaptive_sharpener(t_p3_raw, mode="balanced", input_is_ink=True)
            frames_p3_sharp[f_idx] = ink_tensor_to_u8(t_p3_sharp)

            # 4. Pha 4 Occlusion Head
            with torch.no_grad():
                pr_p4, cw_p4, f0_p4, f1_p4, fl_p4, M0_p4, M1_p4 = models['p4'](
                    x0.float(), x1.float(), tt, cf01.float(), cf10.float()
                )
            t_p4_occ = pr_p4[:, :, :h, :w].clamp(0.0, 1.0)
            m0_crop = M0_p4[:, :, :h, :w].clamp(0.0, 1.0)
            m1_crop = M1_p4[:, :, :h, :w].clamp(0.0, 1.0)
            frames_p4_occ[f_idx] = ink_tensor_to_u8(t_p4_occ)
            masks_p4_m0[f_idx] = mask_tensor_to_u8(m0_crop)
            masks_p4_m1[f_idx] = mask_tensor_to_u8(m1_crop)

            stem = os.path.splitext(os.path.basename(paths[f_idx]))[0]
            cv2.imwrite(os.path.join(clean_dir, os.path.basename(paths[f_idx])), frames_p4_occ[f_idx])
            cv2.imwrite(os.path.join(masks_dir, f"{stem}_M0.png"), masks_p4_m0[f_idx])
            cv2.imwrite(os.path.join(masks_dir, f"{stem}_M1.png"), masks_p4_m1[f_idx])

            # Tính toán metrics
            rec = {
                "frame": os.path.basename(paths[f_idx]),
                "segment": f"{i0}->{i1}",
                "delta_f": delta_f,
                "t": round(tv, 3),
            }
            version_tensors = [
                ("Pha2_Blended", t_p2),
                ("Pha3_Remake_Raw", t_p3_raw),
                ("Pha3_Remake_Sharp", t_p3_sharp),
                ("Pha4_Occlusion", t_p4_occ),
            ]
            for tag, tens in version_tensors:
                mm = evaluate_metrics_dual_regime(tens, gt_t)
                for mk, mv in mm.items():
                    rec[f"{tag}_{mk}"] = float(mv.item())

            rec['Pha4_M0_mean'] = float(m0_crop.mean().item())
            rec['Pha4_M1_mean'] = float(m1_crop.mean().item())
            rows.append(rec)
            print(f"  Frame {f_idx:02d} (t={tv:.2f}) -> P4_Occ: PSNR={rec['Pha4_Occlusion_psnr']:.2f}dB | Ghost={rec['Pha4_Occlusion_ghost']} | M0={rec['Pha4_M0_mean']:.3f}")

    # Xuất Side-by-Side 6 cột [GT | Overlay | Pha2 | Pha3 Raw | Pha3 Sharp | Pha4 Occ]
    sbs_rgb_list = []
    for i in range(total_frames):
        c_gt = cv2.cvtColor(frames_gt[i], cv2.COLOR_GRAY2BGR)
        c_p2 = cv2.cvtColor(frames_p2[i], cv2.COLOR_GRAY2BGR)
        c_p3_raw = cv2.cvtColor(frames_p3_raw[i], cv2.COLOR_GRAY2BGR)
        c_p3_shp = cv2.cvtColor(frames_p3_sharp[i], cv2.COLOR_GRAY2BGR)
        c_p4_occ = cv2.cvtColor(frames_p4_occ[i], cv2.COLOR_GRAY2BGR)

        w_col = c_gt.shape[1]
        def _make_hdr(text, clr):
            hdr = np.full((32, w_col, 3), 22, dtype=np.uint8)
            cv2.putText(hdr, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, clr, 1, cv2.LINE_AA)
            return hdr

        lbl_gt = f"GT #{i:02d}" if i not in kf_set else f"Keyframe #{i:02d}"
        p_gt = np.vstack([_make_hdr(lbl_gt, (230, 230, 230)), c_gt])
        p_p2 = np.vstack([_make_hdr("Pha 2 Blended", (100, 175, 255)), c_p2])
        p_p3_raw = np.vstack([_make_hdr("Pha 3 Raw", (120, 120, 255)), c_p3_raw])
        p_p3_shp = np.vstack([_make_hdr("Pha 3 + Sharp", (80, 235, 110)), c_p3_shp])
        p_p4_occ = np.vstack([_make_hdr("Pha 4 Occlusion Head", (255, 160, 40)), c_p4_occ])

        sbs_row = np.hstack([p_gt, p_p2, p_p3_raw, p_p3_shp, p_p4_occ])
        sbs_scaled = cv2.resize(sbs_row, (int(sbs_row.shape[1] * 0.45), int(sbs_row.shape[0] * 0.45)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(sbs_dir, f"sbs_6col_{i:03d}.png"), sbs_scaled)
        sbs_rgb_list.append(sbs_scaled[:, :, ::-1])

    if imageio is not None and sbs_rgb_list:
        gif_path = os.path.join(out_dir, "benchmark_lineart_test_4versions.gif")
        imageio.mimsave(gif_path, sbs_rgb_list, fps=8, loop=0)
        print(f"  [✓] Đã xuất GIF đối đầu: {gif_path}")

    df_res = pd.DataFrame(rows)
    df_res.to_csv(os.path.join(out_dir, "benchmark_summary_24f.csv"), index=False)
    return df_res


# ==============================================================================
# BÀI TEST 2, 3, 4: DEFENSE GEOMETRY SUITE
# ==============================================================================
def run_test_defense_geometry(models, out_dir):
    print("\n" + "=" * 90)
    print("▶ CHẠY CASE 2, 3, 4: DEFENSE GEOMETRY SUITE (CIRCLE, HOLLOW CROSS, SOLID CROSS)")
    print("=" * 90)
    os.makedirs(out_dir, exist_ok=True)

    cases = [
        ("Case2_DeformingCircle", draw_deforming_circle),
        ("Case3_DeformingHollowCross", draw_deforming_cross),
        ("Case4_DeformingSolidCross", draw_deforming_solid_cross),
    ]

    H, W = 384, 640
    t_vals = [0.25, 0.50, 0.75]
    all_rows = []

    for cname, draw_fn in cases:
        im0 = draw_fn(0.0, H, W)
        im1 = draw_fn(1.0, H, W)
        k0, k1 = extract_shape_keypoints(im0, im1)

        x0 = torch.from_numpy(im0)[None, None].to(DEVICE)
        x1 = torch.from_numpy(im1)[None, None].to(DEVICE)

        for tv in t_vals:
            gt_im = draw_fn(tv, H, W)
            gt_t = torch.from_numpy(gt_im)[None, None].to(DEVICE)
            tt = torch.tensor([[[[tv]]]], dtype=torch.float32, device=DEVICE)

            if len(k0) >= 8:
                cf0t, cf1t, _ = solve_multi_layer_tps(
                    k0, k1, (H, W), H, W, DEVICE,
                    t_val=tv, direct_t_flows=True, max_pts=128, lambd=0.01,
                    downscale_factor=2, eps_static_px=1.5, use_dyn_anchors=False,
                    transition_width_px=18.0, falloff_radius_px=160.0
                )
            else:
                cf0t = torch.zeros((1, 2, H, W), device=DEVICE)
                cf1t = torch.zeros((1, 2, H, W), device=DEVICE)

            with torch.no_grad():
                pr_p2, *_ = models['p2'](x0.float(), x1.float(), tt, cf0t.float(), cf1t.float(), direct_t_flows=True)
                pr_p3, *_ = models['p3'](x0.float(), x1.float(), tt, cf0t.float(), cf1t.float(), direct_t_flows=True)
                t_p3_raw = pr_p3.clamp(0.0, 1.0)
                t_p3_shp = apply_adaptive_sharpener(t_p3_raw, mode="balanced", input_is_ink=True)
                pr_p4, _, _, _, _, M0_p4, M1_p4 = models['p4'](x0.float(), x1.float(), tt, cf0t.float(), cf1t.float(), direct_t_flows=True)

            rec = {"case": cname, "t": tv}
            for tag, tens in [
                ("Pha2_Blended", pr_p2),
                ("Pha3_Remake_Raw", t_p3_raw),
                ("Pha3_Remake_Sharp", t_p3_shp),
                ("Pha4_Occlusion", pr_p4),
            ]:
                mm = evaluate_metrics_dual_regime(tens, gt_t)
                for mk, mv in mm.items():
                    rec[f"{tag}_{mk}"] = float(mv.item())
            all_rows.append(rec)

    df_geom = pd.DataFrame(all_rows)
    df_geom.to_csv(os.path.join(out_dir, "defense_geometry_metrics.csv"), index=False)
    return df_geom


# ==============================================================================
# HÀM MAIN THỰC THI BENCHMARK
# ==============================================================================
def main():
    print("=" * 90)
    print("🔬 BẮT ĐẦU BENCHMARK ĐỐI ĐẦU 4 PHIÊN BẢN (PHA 2 vs PHA 3 vs PHA 4 OCCLUSION)")
    print(f"  Thiết bị: {DEVICE}")
    print(f"  Thư mục kết quả cách ly: {BASE_OUT_DIR}")
    print("=" * 90)
    os.makedirs(BASE_OUT_DIR, exist_ok=True)

    # 1. Nạp 4 mô hình
    print("\n📦 Nạp trọng số các mô hình...")
    # Pha 2 Blended
    m_p2 = TPSInbetweenOriginal().to(DEVICE).eval()
    load_tps_weights(m_p2, PHA2_CKPT, device=DEVICE, prefer_ema=True, is_rank0=True)

    # Pha 3 Remake Raw / Sharp
    m_p3 = TPSInbetweenOriginal().to(DEVICE).eval()
    load_tps_weights(m_p3, PHA3_CKPT, device=DEVICE, prefer_ema=True, is_rank0=True)

    # Pha 4 Occlusion Head
    m_p4 = TPSInbetweenOcclusion().to(DEVICE).eval()
    if os.path.exists(PHA4_CKPT):
        print(f"  [✓] Nạp checkpoint Phase 4 đã hoàn tất: {PHA4_CKPT}")
        load_tps_occlusion_weights(m_p4, PHA4_CKPT, device=DEVICE, prefer_ema=True, is_rank0=True)
    else:
        print(f"  [*] Checkpoint Phase 4 chưa hoàn tất trên Kaggle -> Warm-Start từ Phase 3: {PHA3_CKPT}")
        load_tps_occlusion_weights(m_p4, PHA3_CKPT, device=DEVICE, prefer_ema=True, is_rank0=True)

    models = {'p2': m_p2, 'p3': m_p3, 'p4': m_p4}

    # 2. Chạy Case 1: Chuỗi 24 frames thực tế
    out_case1 = os.path.join(BASE_OUT_DIR, "case1_lineart_test_24f")
    df_case1 = run_test_case_1(models, out_case1)

    # 3. Chạy Case 2, 3, 4: Defense Geometry Suite
    out_geom = os.path.join(BASE_OUT_DIR, "case2_3_4_defense_geometry")
    df_geom = run_test_defense_geometry(models, out_geom)

    # 4. In bảng tổng kết đối đầu
    print("\n" + "=" * 90)
    print("📊 BẢNG TỔNG KẾT ĐỐI ĐẦU 4 PHIÊN BẢN (TRUNG BÌNH 17 KHUNG HÌNH TRUNG GIAN CASE 1)")
    print("=" * 90)

    metrics_list = ["psnr", "ssim", "skel_recall", "density_ratio", "tip_recall", "wcd_dog", "ghost"]
    versions = [
        ("Pha 2 Blended", "Pha2_Blended"),
        ("Pha 3 Remake Raw", "Pha3_Remake_Raw"),
        ("Pha 3 + Balanced Sharp", "Pha3_Remake_Sharp"),
        ("Pha 4 Occlusion Head", "Pha4_Occlusion"),
    ]

    summary_rows = []
    for label, prefix in versions:
        row = {"Phiên Bản": label}
        for m in metrics_list:
            col = f"{prefix}_{m}"
            if col in df_case1.columns:
                row[m.upper()] = round(float(df_case1[col].mean()), 4)
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    try:
        print(df_summary.to_markdown(index=False))
    except Exception:
        print(df_summary.to_string(index=False))

    summary_csv = os.path.join(BASE_OUT_DIR, "benchmark_4versions_summary.csv")
    df_summary.to_csv(summary_csv, index=False)
    print(f"\n[✓] Đã xuất bảng tổng kết đối đầu: {summary_csv}")


if __name__ == "__main__":
    main()
