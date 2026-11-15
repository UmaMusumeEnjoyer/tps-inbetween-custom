#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_pha4_vs_pha3.py
--------------------------------------------------------------------------------
BỘ BENCHMARK ĐỐI ĐẦU 4 BÀI TEST CHUẨN:
    - Target: Sequence 4 Occlusion Head (model_tps_occlusion_best.pt)
    - Baseline: Sequence 2 Pha 3 Remake (newpipeline) (model_tps_finetuned_best.pt + Balanced Sharpener)
    - Các biến thể:
        + Pha 3 Remake (newpipeline) [Pha 3 + Balanced Sharpener]
        + Pha 4 Occlusion Head (Raw)
        + Pha 4 Occlusion Head + Balanced Sharpener

4 TẬP KIỂM CHỨNG:
    - Case 1: Chuỗi Lineart thực tế 24 frames (data/lineart_experiments/test - 17 in-betweens)
    - Case 2: Hình tròn biến dạng di chuyển (Squash & Stretch Moving Circle - 360px motion)
    - Case 3: Chữ thập rỗng xoay biến dạng (Rotating Deforming Hollow Cross - 360px motion)
    - Case 4: Chữ thập đặc xoay biến dạng (Rotating Deforming Solid Cross - 360px motion)

TẠO VÀ LƯU TRỮ ĐẦY ĐỦ FRAME / MASKS / STRIPS / GIFS TẠI:
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
    from skimage.morphology import skeletonize
except ImportError:
    skeletonize = None

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
PHA4_CKPT_PRIMARY = os.path.join(
    REPO_ROOT, "checkpoints", "Re-train", "Sequence 2 pha 3 remake occulision test",
    "checkpoints", "model_tps_occlusion_best.pt"
)
PHA4_CKPT_FALLBACK = os.path.join(
    REPO_ROOT, "checkpoints", "Re-train", "Sequence 4 Occlusion Head",
    "model_tps_occlusion_best.pt"
)
PHA4_CKPT = PHA4_CKPT_PRIMARY if os.path.exists(PHA4_CKPT_PRIMARY) else PHA4_CKPT_FALLBACK

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
    """
    Ghép ngang nhiều cột hình ảnh kèm dải tiêu đề header riêng biệt cho mỗi cột.
    columns_with_headers: list of tuples: (u8_image_2d, header_text, bgr_color)
    """
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
# BÀI TEST 1: CHUỖI 24 FRAMES THỰC TẾ (17 IN-BETWEENS)
# ==============================================================================
def run_test_case_1(models, out_dir):
    print("\n" + "=" * 90)
    print("▶ CHẠY CASE 1: CHUỖI 24 FRAMES THỰC TẾ (17 IN-BETWEENS)")
    print(f"  Thư mục dữ liệu: {LINEART_TEST_DIR}")
    print(f"  Thư mục đích:    {out_dir}")
    print("=" * 90)

    clean_dir_p4_raw = os.path.join(out_dir, "clean_frames_phase4_raw")
    clean_dir_p4_shp = os.path.join(out_dir, "clean_frames_phase4_sharp")
    clean_dir_p3_shp = os.path.join(out_dir, "clean_frames_phase3_newpipeline")
    masks_dir_m0 = os.path.join(out_dir, "masks_phase4_M0")
    masks_dir_m1 = os.path.join(out_dir, "masks_phase4_M1")
    sbs_dir = os.path.join(out_dir, "side_by_side_comparison")

    for d in (clean_dir_p4_raw, clean_dir_p4_shp, clean_dir_p3_shp, masks_dir_m0, masks_dir_m1, sbs_dir):
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
    frames_p3_sharp = [None] * total_frames
    frames_p4_raw = [None] * total_frames
    frames_p4_sharp = [None] * total_frames
    masks_p4_m0 = [None] * total_frames
    masks_p4_m1 = [None] * total_frames

    for i in range(total_frames):
        ink_gt = load_ink_from_input(paths[i])
        u8_gt = np.clip((1.0 - ink_gt) * 255.0, 0, 255).astype(np.uint8)
        frames_gt[i] = u8_gt
        if i in kf_set:
            frames_p3_sharp[i] = u8_gt
            frames_p4_raw[i] = u8_gt
            frames_p4_sharp[i] = u8_gt
            fname = os.path.basename(paths[i])
            cv2.imwrite(os.path.join(clean_dir_p3_shp, fname), u8_gt)
            cv2.imwrite(os.path.join(clean_dir_p4_raw, fname), u8_gt)
            cv2.imwrite(os.path.join(clean_dir_p4_shp, fname), u8_gt)

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

            # 1. Pha 3 Remake + Balanced Sharpener (Sequence 2 pha 3 remake newpipeline)
            with torch.no_grad():
                pr_p3, *_ = models['p3'](x0.float(), x1.float(), tt, cf01.float(), cf10.float())
            t_p3_raw = pr_p3[:, :, :h, :w].clamp(0.0, 1.0)
            t_p3_sharp = apply_adaptive_sharpener(t_p3_raw, mode="balanced", input_is_ink=True)
            u8_p3_sharp = ink_tensor_to_u8(t_p3_sharp)
            frames_p3_sharp[f_idx] = u8_p3_sharp

            # 2. Pha 4 Occlusion Head
            with torch.no_grad():
                pr_p4, cw_p4, f0_p4, f1_p4, fl_p4, M0_p4, M1_p4 = models['p4'](
                    x0.float(), x1.float(), tt, cf01.float(), cf10.float()
                )
            t_p4_raw = pr_p4[:, :, :h, :w].clamp(0.0, 1.0)
            m0_crop = M0_p4[:, :, :h, :w].clamp(0.0, 1.0)
            m1_crop = M1_p4[:, :, :h, :w].clamp(0.0, 1.0)
            u8_p4_raw = ink_tensor_to_u8(t_p4_raw)
            u8_m0 = mask_tensor_to_u8(m0_crop)
            u8_m1 = mask_tensor_to_u8(m1_crop)

            # 3. Pha 4 Occlusion Head + Balanced Sharpener
            t_p4_sharp = apply_adaptive_sharpener(t_p4_raw, mode="balanced", input_is_ink=True)
            u8_p4_sharp = ink_tensor_to_u8(t_p4_sharp)

            frames_p4_raw[f_idx] = u8_p4_raw
            frames_p4_sharp[f_idx] = u8_p4_sharp
            masks_p4_m0[f_idx] = u8_m0
            masks_p4_m1[f_idx] = u8_m1

            fname = os.path.basename(paths[f_idx])
            stem = os.path.splitext(fname)[0]
            cv2.imwrite(os.path.join(clean_dir_p3_shp, fname), u8_p3_sharp)
            cv2.imwrite(os.path.join(clean_dir_p4_raw, fname), u8_p4_raw)
            cv2.imwrite(os.path.join(clean_dir_p4_shp, fname), u8_p4_sharp)
            cv2.imwrite(os.path.join(masks_dir_m0, f"{stem}_M0.png"), u8_m0)
            cv2.imwrite(os.path.join(masks_dir_m1, f"{stem}_M1.png"), u8_m1)

            # Tính toán metrics
            rec = {
                "frame": fname,
                "segment": f"{i0}->{i1}",
                "delta_f": delta_f,
                "t": round(tv, 3),
            }
            version_tensors = [
                ("Pha3_Remake_Sharp", t_p3_sharp),
                ("Pha4_Occlusion_Raw", t_p4_raw),
                ("Pha4_Occlusion_Sharp", t_p4_sharp),
            ]
            for tag, tens in version_tensors:
                mm = evaluate_metrics_dual_regime(tens, gt_t)
                for mk, mv in mm.items():
                    rec[f"{tag}_{mk}"] = float(mv.item())
                # Tính bổ sung IE_MAE
                ie_mae = float(torch.abs(tens - gt_t).mean().item() * 255.0)
                rec[f"{tag}_ie_mae"] = ie_mae

            rec['Pha4_M0_mean'] = float(m0_crop.mean().item())
            rec['Pha4_M1_mean'] = float(m1_crop.mean().item())
            rows.append(rec)
            print(f"  Frame {f_idx:02d} (t={tv:.2f}) | P3_Sharp: PSNR={rec['Pha3_Remake_Sharp_psnr']:.2f}dB, Ghost={rec['Pha3_Remake_Sharp_ghost']} | P4_Occ_Raw: PSNR={rec['Pha4_Occlusion_Raw_psnr']:.2f}dB, Ghost={rec['Pha4_Occlusion_Raw_ghost']} | P4_Occ_Sharp: PSNR={rec['Pha4_Occlusion_Sharp_psnr']:.2f}dB, Ghost={rec['Pha4_Occlusion_Sharp_ghost']}")

    # Render Side-by-Side 5 cột: [GT | Pha3 Newpipeline (Sharp) | Pha4 Occ Raw | Pha4 Occ Sharp | M0 Mask]
    sbs_rgb_list = []
    for i in range(total_frames):
        c_gt = frames_gt[i]
        c_p3_shp = frames_p3_sharp[i]
        c_p4_raw = frames_p4_raw[i]
        c_p4_shp = frames_p4_sharp[i]
        c_m0 = masks_p4_m0[i] if masks_p4_m0[i] is not None else np.full_like(c_gt, 255)

        lbl_gt = f"GT #{i:02d}" if i not in kf_set else f"Keyframe #{i:02d}"
        cols_cfg = [
            (c_gt, lbl_gt, (230, 230, 230)),
            (c_p3_shp, "P3 Newpipeline (Sharp)", (80, 235, 110)),
            (c_p4_raw, "P4 Occlusion Raw", (255, 170, 50)),
            (c_p4_shp, "P4 Occlusion + Sharp", (50, 210, 255)),
            (c_m0, "P4 Mask M0 (Vis)", (180, 180, 255)),
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
    gif_frames_p3 = [np.stack([f0_u8] * 3, axis=-1)]
    gif_frames_p4_raw = [np.stack([f0_u8] * 3, axis=-1)]
    gif_frames_p4_shp = [np.stack([f0_u8] * 3, axis=-1)]

    all_rows = []

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
            # Pha 3 Remake Raw + Sharp
            pr_p3, cw_p3, *_ = models['p3'](x0.float(), x1.float(), tt, cf0t.float(), cf1t.float(), direct_t_flows=True)
            t_p3_raw = pr_p3.clamp(0.0, 1.0)
            t_p3_shp = apply_adaptive_sharpener(t_p3_raw, mode="balanced", input_is_ink=True)

            # Pha 4 Occlusion Head
            pr_p4, cw_p4, _, _, _, M0_p4, M1_p4 = models['p4'](x0.float(), x1.float(), tt, cf0t.float(), cf1t.float(), direct_t_flows=True)
            t_p4_raw = pr_p4.clamp(0.0, 1.0)
            m0_t = M0_p4.clamp(0.0, 1.0)
            m1_t = M1_p4.clamp(0.0, 1.0)
            t_p4_shp = apply_adaptive_sharpener(t_p4_raw, mode="balanced", input_is_ink=True)

        gt_u8 = np.clip((1.0 - gt_im) * 255.0, 0, 255).astype(np.uint8)
        p3_shp_u8 = ink_tensor_to_u8(t_p3_shp)
        p4_raw_u8 = ink_tensor_to_u8(t_p4_raw)
        p4_shp_u8 = ink_tensor_to_u8(t_p4_shp)
        m0_u8 = mask_tensor_to_u8(m0_t)
        m1_u8 = mask_tensor_to_u8(m1_t)

        # Lưu các frames sạch
        cv2.imwrite(os.path.join(clean_dir, f"{cname}_t{tv:.2f}_GT.png"), gt_u8)
        cv2.imwrite(os.path.join(clean_dir, f"{cname}_t{tv:.2f}_P3_Newpipeline.png"), p3_shp_u8)
        cv2.imwrite(os.path.join(clean_dir, f"{cname}_t{tv:.2f}_P4_Occlusion_Raw.png"), p4_raw_u8)
        cv2.imwrite(os.path.join(clean_dir, f"{cname}_t{tv:.2f}_P4_Occlusion_Sharp.png"), p4_shp_u8)
        cv2.imwrite(os.path.join(clean_dir, f"{cname}_t{tv:.2f}_P4_Mask_M0.png"), m0_u8)
        cv2.imwrite(os.path.join(clean_dir, f"{cname}_t{tv:.2f}_P4_Mask_M1.png"), m1_u8)

        gif_frames_p3.append(np.stack([p3_shp_u8] * 3, axis=-1))
        gif_frames_p4_raw.append(np.stack([p4_raw_u8] * 3, axis=-1))
        gif_frames_p4_shp.append(np.stack([p4_shp_u8] * 3, axis=-1))

        # Đánh giá Metrics
        rec = {"case": cname, "t_val": tv}
        version_tensors = [
            ("Pha3_Remake_Sharp", t_p3_shp),
            ("Pha4_Occlusion_Raw", t_p4_raw),
            ("Pha4_Occlusion_Sharp", t_p4_shp),
        ]
        for tag, tens in version_tensors:
            mm = evaluate_metrics_dual_regime(tens, gt_t)
            for mk, mv in mm.items():
                rec[f"{tag}_{mk}"] = float(mv.item())
            rec[f"{tag}_ie_mae"] = float(torch.abs(tens - gt_t).mean().item() * 255.0)
            sk_rec, d_rat = compute_exact_morphological_metrics(tens[0, 0].detach().cpu().numpy(), gt_im)
            rec[f"{tag}_exact_skel_recall"] = sk_rec
            rec[f"{tag}_exact_density"] = d_rat

        rec['Pha4_M0_mean'] = float(m0_t.mean().item())
        rec['Pha4_M1_mean'] = float(m1_t.mean().item())
        all_rows.append(rec)

        print(f"  t={tv:.2f} | P3_Sharp: SkelRec={rec['Pha3_Remake_Sharp_exact_skel_recall']:.2f}%, Ghost={rec['Pha3_Remake_Sharp_ghost']} | P4_Occ_Raw: SkelRec={rec['Pha4_Occlusion_Raw_exact_skel_recall']:.2f}%, Ghost={rec['Pha4_Occlusion_Raw_ghost']} | P4_Occ_Sharp: SkelRec={rec['Pha4_Occlusion_Sharp_exact_skel_recall']:.2f}%, Ghost={rec['Pha4_Occlusion_Sharp_ghost']}")

        # Tạo Strip so sánh đối đầu
        cols_cfg = [
            (gt_u8, f"GT (t={tv:.2f})", (230, 230, 230)),
            (p3_shp_u8, "P3 Newpipeline", (80, 235, 110)),
            (p4_raw_u8, "P4 Occlusion Raw", (255, 170, 50)),
            (p4_shp_u8, "P4 Occlusion + Sharp", (50, 210, 255)),
            (m0_u8, "P4 M0 Mask", (180, 180, 255)),
            (m1_u8, "P4 M1 Mask", (255, 180, 180)),
        ]
        strip_row = render_multicolumn_strip(cols_cfg, header_height=30)
        strips.append(strip_row)

        # Crop zoom audit tại tâm vật thể
        cx_mid = int(round((1.0 - tv) * 140 + tv * 490))
        cy_mid = 192
        rad_box = 85
        y0, y1 = max(0, cy_mid - rad_box), min(H, cy_mid + rad_box)
        x0_b, x1_b = max(0, cx_mid - rad_box), min(W, cx_mid + rad_box)

        z_cols = []
        for u_img, clr, lbl in [
            (gt_u8, (230, 230, 230), "GT"),
            (p3_shp_u8, (80, 235, 110), "P3 Newpipeline"),
            (p4_raw_u8, (255, 170, 50), "P4 Occ Raw"),
            (p4_shp_u8, (50, 210, 255), "P4 Occ + Sharp"),
        ]:
            sub = u_img[y0:y1, x0_b:x1_b]
            sub_z = cv2.resize(sub, (220, 220), interpolation=cv2.INTER_NEAREST)
            bgr_z = cv2.cvtColor(sub_z, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(bgr_z, (0, 0), (219, 219), clr, 2)
            cv2.putText(bgr_z, f"{lbl} t={tv:.2f}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr, 1, cv2.LINE_AA)
            z_cols.append(bgr_z)
        zoom_crops.append(np.hstack(z_cols))

    gif_frames_p3.append(np.stack([f1_u8] * 3, axis=-1))
    gif_frames_p4_raw.append(np.stack([f1_u8] * 3, axis=-1))
    gif_frames_p4_shp.append(np.stack([f1_u8] * 3, axis=-1))

    # Lưu Strip 6 cột và Zoom Audit
    out_strip = os.path.join(out_case_dir, f"{cname}_comparison_6col.png")
    out_zoom = os.path.join(out_case_dir, f"{cname}_zoom_audit.png")
    cv2.imwrite(out_strip, np.vstack(strips))
    cv2.imwrite(out_zoom, np.vstack(zoom_crops))

    if imageio is not None:
        imageio.mimsave(os.path.join(out_case_dir, f"{cname}_motion_p3_sharp.gif"), gif_frames_p3, fps=4, loop=0)
        imageio.mimsave(os.path.join(out_case_dir, f"{cname}_motion_p4_raw.gif"), gif_frames_p4_raw, fps=4, loop=0)
        imageio.mimsave(os.path.join(out_case_dir, f"{cname}_motion_p4_sharp.gif"), gif_frames_p4_shp, fps=4, loop=0)
        print(f"  [✓] Đã xuất demo GIFs cho {cname}.")

    df_case = pd.DataFrame(all_rows)
    df_case.to_csv(os.path.join(out_case_dir, f"{cname}_metrics.csv"), index=False)
    return df_case


# ==============================================================================
# HÀM MAIN THỰC THI BENCHMARK
# ==============================================================================
def main():
    print("=" * 90)
    print("🔬 BẮT ĐẦU BENCHMARK 4 BÀI TEST: PHA 4 OCCLUSION vs PHA 3 REMAKE (NEWPIPELINE)")
    print(f"  Thiết bị: {DEVICE}")
    print(f"  Pha 3 Checkpoint: {PHA3_CKPT}")
    print(f"  Pha 4 Checkpoint: {PHA4_CKPT}")
    print(f"  Thư mục kết quả cách ly: {BASE_OUT_DIR}")
    print("=" * 90)
    os.makedirs(BASE_OUT_DIR, exist_ok=True)

    # 1. Nạp mô hình
    print("\n📦 Nạp trọng số các mô hình...")
    # Pha 3 Remake
    m_p3 = TPSInbetweenOriginal().to(DEVICE).eval()
    load_tps_weights(m_p3, PHA3_CKPT, device=DEVICE, prefer_ema=True, is_rank0=True)

    # Pha 4 Occlusion Head
    m_p4 = TPSInbetweenOcclusion().to(DEVICE).eval()
    load_tps_occlusion_weights(m_p4, PHA4_CKPT, device=DEVICE, prefer_ema=True, is_rank0=True)

    models = {'p3': m_p3, 'p4': m_p4}

    # 2. Chạy Case 1: Chuỗi 24 frames thực tế
    out_case1 = os.path.join(BASE_OUT_DIR, "case1_lineart_test_24f")
    df_case1 = run_test_case_1(models, out_case1)

    # 3. Chạy Case 2, 3, 4: Defense Geometry Suite
    dir_c2 = os.path.join(BASE_OUT_DIR, "case2_deforming_circle")
    df_c2 = run_single_geometry_case("case2_deforming_circle", draw_deforming_circle, models, dir_c2)

    dir_c3 = os.path.join(BASE_OUT_DIR, "case3_deforming_cross")
    df_c3 = run_single_geometry_case("case3_deforming_cross", draw_deforming_cross, models, dir_c3)

    dir_c4 = os.path.join(BASE_OUT_DIR, "case4_deforming_solid_cross")
    df_c4 = run_single_geometry_case("case4_deforming_solid_cross", draw_deforming_solid_cross, models, dir_c4)

    # 4. Xuất Báo Cáo Tổng Hợp
    print("\n" + "=" * 90)
    print("📊 XUẤT TỔNG HỢP KẾT QUẢ ĐỐI ĐẦU TẤT CẢ 4 BÀI TEST")
    print("=" * 90)

    # Tổng kết Case 1
    metrics_c1 = [
        ("PSNR (dB) (↑)", "psnr"),
        ("SSIM (↑)", "ssim"),
        ("Skeleton Recall (%) (↑)", "skel_recall"),
        ("Tip Recall (%) (↑)", "tip_recall"),
        ("F1 Score (↑)", "f1"),
        ("WCD_DoG (px) (↓)", "wcd_dog"),
        ("Ghost Pixels (px) (↓)", "ghost"),
        ("IE_MAE (↓)", "ie_mae"),
        ("Density Ratio (≈1.0)", "density_ratio"),
    ]
    versions = [
        ("Pha 3 Remake (Newpipeline)", "Pha3_Remake_Sharp"),
        ("Pha 4 Occlusion Head (Raw)", "Pha4_Occlusion_Raw"),
        ("Pha 4 Occlusion Head + Sharp", "Pha4_Occlusion_Sharp"),
    ]

    summary_rows_c1 = []
    for v_label, v_prefix in versions:
        row = {"Phiên Bản": v_label}
        for m_label, m_key in metrics_c1:
            col = f"{v_prefix}_{m_key}"
            if col in df_case1.columns:
                row[m_label] = round(float(df_case1[col].mean()), 4)
        summary_rows_c1.append(row)

    df_sum_c1 = pd.DataFrame(summary_rows_c1)
    df_sum_c1.to_csv(os.path.join(BASE_OUT_DIR, "summary_case1_lineart.csv"), index=False)
    print("\n[Case 1: Chuỗi Lineart Thực Tế - Trung bình 17 frames trung gian]")
    print(df_sum_c1.to_string(index=False))

    # Tổng kết Case 2, 3, 4 tại t=0.50
    geom_cases = [
        ("Case 2: Deforming Circle (t=0.50)", df_c2),
        ("Case 3: Deforming Hollow Cross (t=0.50)", df_c3),
        ("Case 4: Deforming Solid Cross (t=0.50)", df_c4),
    ]
    geom_metrics = [
        ("PSNR (dB) (↑)", "psnr"),
        ("SSIM (↑)", "ssim"),
        ("Exact Skel Recall (%) (↑)", "exact_skel_recall"),
        ("Exact Density Ratio (≈1.0)", "exact_density"),
        ("WCD_DoG (px) (↓)", "wcd_dog"),
        ("Ghost Pixels (px) (↓)", "ghost"),
        ("IE_MAE (↓)", "ie_mae"),
    ]

    all_geom_summaries = []
    for c_title, df_c in geom_cases:
        row_t05 = df_c[df_c["t_val"] == 0.50].iloc[0]
        for v_label, v_prefix in versions:
            row = {"Case": c_title, "Phiên Bản": v_label}
            for m_label, m_key in geom_metrics:
                col = f"{v_prefix}_{m_key}"
                if col in row_t05:
                    row[m_label] = round(float(row_t05[col]), 4)
            all_geom_summaries.append(row)

    df_sum_geom = pd.DataFrame(all_geom_summaries)
    df_sum_geom.to_csv(os.path.join(BASE_OUT_DIR, "summary_cases2_3_4_geometry_t05.csv"), index=False)
    print("\n[Cases 2, 3, 4: Defense Geometry Suite tại t=0.50]")
    print(df_sum_geom.to_string(index=False))

    print("\n" + "=" * 90)
    print(f"🎉 HOÀN THÀNH TOÀN BỘ BENCHMARK 4 TEST CASES.")
    print(f"   Toàn bộ frames, masks, strips, gifs và csvs đã được lưu tại: {BASE_OUT_DIR}")
    print("=" * 90)


if __name__ == "__main__":
    main()
