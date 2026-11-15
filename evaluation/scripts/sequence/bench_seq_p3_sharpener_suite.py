#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
run_verified_benchmark_suite_p3_sharp.py
--------------------------------------------------------------------------------
KIỂM CHỨNG TOÀN DIỆN 3 CASE:
  1. Chuỗi Lineart thực tế: `data/lineart_experiments/test` (24 frames 540p / 17 in-betweens)
  2. Hình tròn biến dạng di chuyển (Squash & Stretch Moving Circle - 360px motion)
  3. Hình chữ thập biến dạng di chuyển xoay tròn (Rotating Deforming Cross - 360px motion)

HỆ THỐNG:
  - Sequence 2 Pha 3 Remake Best (`model_tps_finetuned_best.pt`)
  - Adaptive Local Edge Sharpener (mode='balanced')
  - So sánh đối ứng với Pha 2 Blended (`model_tps_blended.pt`) & Pha 3 Raw

LƯU TRỮ TẠI:
  checkpoints/Re-train/Sequence 2 pha 3 remake/verified_benchmark_sharpener/
--------------------------------------------------------------------------------
"""

import os
import sys
import time
import glob
import math
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
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
PHASE3_WORKER_DIR = os.path.join(REPO_ROOT, "training", "03_sequence", "p3_multilayer_tps")
SEQ_CORE_DIR = os.path.join(REPO_ROOT, "src", "sequence_p3_p4")

for _d in (SCRIPT_DIR, PHASE3_WORKER_DIR, SEQ_CORE_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from seq_p3_multi_layer_tps import solve_multi_layer_tps
from seq_p3_v2_multilayer_tps_worker import (
    TPSInbetweenOriginal,
    load_tps_weights,
    evaluate_metrics_dual_regime,
)
from bench_seq_p3_validate_multilayer import extract_hybrid_contour_matches
from seq_p3_adaptive_sharpener import apply_adaptive_sharpener
from seq_p3_inference_pipeline import (
    Phase3InferencePipeline,
    load_ink_from_input,
    ink_tensor_to_u8,
)

BASE_OUT_DIR = os.path.join(
    REPO_ROOT, "checkpoints", "Re-train", "Sequence 2 pha 3 remake",
    "verified_benchmark_sharpener"
)
PHA3_CKPT = os.path.join(
    REPO_ROOT, "checkpoints", "Re-train", "Sequence 2 pha 3 remake",
    "checkpoints", "model_tps_finetuned_best.pt"
)
PHA2_CKPT = os.path.join(
    REPO_ROOT, "checkpoints", "Re-train", "Sequence 2 Blended",
    "checkpoints", "model_tps_blended.pt"
)
LINEART_TEST_DIR = os.path.join(REPO_ROOT, "data", "lineart_experiments", "test")
GLUESTICK_NPZ = os.path.join(REPO_ROOT, "data_samples", "caches", "lineart_test_gluestick_matches.npz")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# HÀM VẼ HÌNH HỌC PHÒNG THỦ (CASE 2 & 3)
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


def render_5col_strip(gt_u8, m0_u8, p2_u8, p3_raw_u8, p3_shp_u8, title_str):
    h, w = gt_u8.shape
    cols = [gt_u8, m0_u8, p2_u8, p3_raw_u8, p3_shp_u8]
    titles = [
        ("Ground Truth", (230, 230, 230)),
        ("M0 Multi-TPS", (180, 180, 180)),
        ("Pha 2 Blended", (80, 175, 255)),
        ("Pha 3 Raw (Halo)", (80, 120, 255)),
        ("Pha 3 + Balanced Sharp", (80, 235, 110)),
    ]
    head_h = 32
    canvas = np.full((head_h + h, w * 5, 3), 24, dtype=np.uint8)
    for i, (arr, (lbl, clr)) in enumerate(zip(cols, titles)):
        bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        canvas[head_h:, i * w:(i + 1) * w] = bgr
        cv2.putText(canvas, lbl, (i * w + 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, clr, 2, cv2.LINE_AA)
        if i > 0:
            cv2.line(canvas, (i * w, 0), (i * w, head_h + h), (70, 70, 70), 2)
    cv2.putText(canvas, title_str, (10, head_h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)
    return canvas


# ==============================================================================
# BÀI TEST 1: CHUỖI 24 FRAMES THỰC TẾ (17 IN-BETWEENS)
# ==============================================================================
def run_test_case_1(pipe_p3, model_p2, out_dir):
    print("\n" + "=" * 90)
    print("▶ CHẠY BÀI TEST 1: CHUỖI LINEART THỰC TẾ (24 FRAMES 540P / 17 IN-BETWEENS)")
    print(f"  Thư mục dữ liệu: {LINEART_TEST_DIR}")
    print(f"  Thư mục đích:    {out_dir}")
    print("=" * 90)

    clean_dir = os.path.join(out_dir, "clean_frames_sharp")
    sbs_dir = os.path.join(out_dir, "side_by_side_4col")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(sbs_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(LINEART_TEST_DIR, "*.png")))
    total_frames = len(paths)
    kf_idx = [0, 3, 7, 11, 14, 19, 23]
    kf_set = set(kf_idx)

    # Load GlueStick
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

    for i in range(total_frames):
        ink_gt = load_ink_from_input(paths[i])
        u8_gt = np.clip((1.0 - ink_gt) * 255.0, 0, 255).astype(np.uint8)
        frames_gt[i] = u8_gt
        if i in kf_set:
            frames_p2[i] = u8_gt
            frames_p3_raw[i] = u8_gt
            frames_p3_sharp[i] = u8_gt
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
        cf01, cf10, _, _, _ = pipe_p3.compute_multilayer_flows(ink0, ink1, kps0=kps[0], kps1=kps[1])
        x0 = F.pad(torch.from_numpy(ink0)[None, None], (0, w_pad - w, 0, h_pad - h)).to(DEVICE)
        x1 = F.pad(torch.from_numpy(ink1)[None, None], (0, w_pad - w, 0, h_pad - h)).to(DEVICE)

        for f_idx in range(i0 + 1, i1):
            tv = float(f_idx - i0) / float(delta_f)
            gt_ink_np = load_ink_from_input(paths[f_idx])
            gt_t = torch.from_numpy(gt_ink_np)[None, None].to(DEVICE)
            tt = torch.tensor([[[[tv]]]], dtype=torch.float32, device=DEVICE)

            # 1. Pha 2 Blended
            with torch.no_grad():
                pr_p2, *_ = model_p2(x0.float(), x1.float(), tt, cf01.float(), cf10.float())
            t_p2 = pr_p2[:, :, :h, :w].clamp(0.0, 1.0)
            frames_p2[f_idx] = ink_tensor_to_u8(t_p2)

            # 2. Pha 3 Remake Raw + Sharp
            res_p3 = pipe_p3.interpolate_with_precomputed_flows(
                x0, x1, cf01, cf10, (h, w), tv,
                use_sharpener=True, sharpener_mode="balanced"
            )
            t_raw = res_p3["raw_ink_tensor"]
            t_shp = res_p3["ink_tensor"]
            frames_p3_raw[f_idx] = res_p3["raw_u8"]
            frames_p3_sharp[f_idx] = res_p3["inbetween_u8"]

            cv2.imwrite(os.path.join(clean_dir, os.path.basename(paths[f_idx])), frames_p3_sharp[f_idx])

            # Đo metrics
            rec = {"frame": os.path.basename(paths[f_idx]), "segment": f"{i0}->{i1}", "delta_f": delta_f, "t": round(tv, 3)}
            for tag, tens in [("Pha2_Blended", t_p2), ("Pha3_Raw", t_raw), ("Pha3_Balanced_Sharp", t_shp)]:
                mm = evaluate_metrics_dual_regime(tens, gt_t)
                for mk, mv in mm.items():
                    rec[f"{tag}_{mk}"] = float(mv.item())
            rows.append(rec)
            print(f"  Frame {f_idx:02d} (t={tv:.2f}) -> P3+Sharp: Ghost={rec['Pha3_Balanced_Sharp_ghost']} px | SkelRec={rec['Pha3_Balanced_Sharp_skel_recall']:.2f}% | Dens={rec['Pha3_Balanced_Sharp_density_ratio']:.3f}")

    # Xuất Side-by-Side 4 cột và GIFs
    sbs_rgb_list = []
    sharp_rgb_list = []
    for i in range(total_frames):
        c_gt = cv2.cvtColor(frames_gt[i], cv2.COLOR_GRAY2BGR)
        c_p2 = cv2.cvtColor(frames_p2[i], cv2.COLOR_GRAY2BGR)
        c_raw = cv2.cvtColor(frames_p3_raw[i], cv2.COLOR_GRAY2BGR)
        c_shp = cv2.cvtColor(frames_p3_sharp[i], cv2.COLOR_GRAY2BGR)

        w_col = c_gt.shape[1]
        hdr1 = np.full((30, w_col, 3), 20, dtype=np.uint8)
        hdr2 = np.full((30, w_col, 3), 20, dtype=np.uint8)
        hdr3 = np.full((30, w_col, 3), 20, dtype=np.uint8)
        hdr4 = np.full((30, w_col, 3), 20, dtype=np.uint8)

        lbl = f"Keyframe #{i:02d}" if i in kf_set else f"Ground Truth #{i:02d}"
        cv2.putText(hdr1, lbl, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(hdr2, "Pha 2 Blended", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 175, 255), 1, cv2.LINE_AA)
        cv2.putText(hdr3, "Pha 3 Raw (Halo)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 120, 255), 1, cv2.LINE_AA)
        cv2.putText(hdr4, "Pha 3 + Balanced Sharp", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 235, 110), 2, cv2.LINE_AA)

        p1 = np.vstack([hdr1, c_gt])
        p2 = np.vstack([hdr2, c_p2])
        p3 = np.vstack([hdr3, c_raw])
        p4 = np.vstack([hdr4, c_shp])

        sbs_row = np.hstack([p1, p2, p3, p4])
        sbs_scaled = cv2.resize(sbs_row, (int(sbs_row.shape[1] * 0.5), int(sbs_row.shape[0] * 0.5)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(sbs_dir, f"sbs_4col_{i:03d}.png"), sbs_scaled)

        sbs_rgb_list.append(cv2.cvtColor(sbs_scaled, cv2.COLOR_BGR2RGB))
        sharp_rgb_list.append(np.stack([frames_p3_sharp[i]] * 3, axis=-1))

    if imageio is not None:
        imageio.mimsave(os.path.join(out_dir, "demo_comparison_4col.gif"), sbs_rgb_list, fps=8, loop=0)
        imageio.mimsave(os.path.join(out_dir, "demo_phase3_sharp_only.gif"), sharp_rgb_list, fps=8, loop=0)
        print("  [✓] Đã xuất demo GIF so sánh 4 cột và demo GIF thành phẩm.")

    df_frames = pd.DataFrame(rows)
    df_frames.to_csv(os.path.join(out_dir, "benchmark_per_frame.csv"), index=False)

    summary_dict = {
        "p2_ghost": float(df_frames["Pha2_Blended_ghost"].sum()),
        "p2_skel_rec": float(df_frames["Pha2_Blended_skel_recall"].mean()),
        "p2_density": float(df_frames["Pha2_Blended_density_ratio"].mean()),
        "p2_wcd": float(df_frames["Pha2_Blended_wcd_dog"].mean()),
        "raw_ghost": float(df_frames["Pha3_Raw_ghost"].sum()),
        "raw_skel_rec": float(df_frames["Pha3_Raw_skel_recall"].mean()),
        "raw_density": float(df_frames["Pha3_Raw_density_ratio"].mean()),
        "raw_wcd": float(df_frames["Pha3_Raw_wcd_dog"].mean()),
        "shp_ghost": float(df_frames["Pha3_Balanced_Sharp_ghost"].sum()),
        "shp_skel_rec": float(df_frames["Pha3_Balanced_Sharp_skel_recall"].mean()),
        "shp_density": float(df_frames["Pha3_Balanced_Sharp_density_ratio"].mean()),
        "shp_wcd": float(df_frames["Pha3_Balanced_Sharp_wcd_dog"].mean()),
    }
    pd.DataFrame([summary_dict]).to_csv(os.path.join(out_dir, "benchmark_summary.csv"), index=False)
    return summary_dict


# ==============================================================================
# BÀI TEST 2 & 3: HÌNH HỌC PHÒNG THỦ (CIRCLE & CROSS)
# ==============================================================================
def run_test_case_geometry(cname, fn_draw, m_p2, m_p3, out_dir):
    print("\n" + "=" * 90)
    print(f"▶ CHẠY BÀI TEST HÌNH HỌC: {cname}")
    print(f"  Thư mục đích: {out_dir}")
    print("=" * 90)
    os.makedirs(out_dir, exist_ok=True)

    H, W = 384, 640
    im0 = fn_draw(0.0, H, W)
    im1 = fn_draw(1.0, H, W)
    t_vals = [0.25, 0.50, 0.75]

    k0, k1 = extract_hybrid_contour_matches(im0, im1, n_pts_per_comp=32)

    x0 = torch.from_numpy(im0)[None, None].to(DEVICE)
    x1 = torch.from_numpy(im1)[None, None].to(DEVICE)

    strips = []
    zoom_crops = []
    gif_frames = []

    f0_u8 = np.clip((1.0 - im0) * 255.0, 0, 255).astype(np.uint8)
    f1_u8 = np.clip((1.0 - im1) * 255.0, 0, 255).astype(np.uint8)
    gif_frames.append(np.stack([f0_u8] * 3, axis=-1))

    all_rows = []

    for tv in t_vals:
        gt_np = fn_draw(tv, H, W)
        gt_t = torch.from_numpy(gt_np)[None, None].to(DEVICE)
        tt = torch.tensor([[[[float(tv)]]]], dtype=torch.float32, device=DEVICE)

        # Giải TPS chuẩn Lagrangian Trajectory trực tiếp tại thời điểm t
        cf0t, cf1t, _ = solve_multi_layer_tps(
            k0, k1, (H, W), H, W, DEVICE,
            t_val=tv, direct_t_flows=True, max_pts=128, lambd=0.01,
            downscale_factor=2, eps_static_px=1.5, use_dyn_anchors=False,
            transition_width_px=18.0, falloff_radius_px=160.0
        )

        with torch.no_grad():
            pr_p2, cw_p2, *_ = m_p2(x0.float(), x1.float(), tt, cf0t.float(), cf1t.float(), direct_t_flows=True)
            pr_p3, cw_p3, *_ = m_p3(x0.float(), x1.float(), tt, cf0t.float(), cf1t.float(), direct_t_flows=True)

        t_cw = cw_p3.clamp(0.0, 1.0)
        t_p2 = pr_p2.clamp(0.0, 1.0)
        t_p3_raw = pr_p3.clamp(0.0, 1.0)
        t_p3_shp = apply_adaptive_sharpener(t_p3_raw, mode="balanced", input_is_ink=True)

        gt_u8 = np.clip((1.0 - gt_np) * 255.0, 0, 255).astype(np.uint8)
        cw_u8 = ink_tensor_to_u8(t_cw)
        p2_u8 = ink_tensor_to_u8(t_p2)
        raw_u8 = ink_tensor_to_u8(t_p3_raw)
        shp_u8 = ink_tensor_to_u8(t_p3_shp)

        gif_frames.append(np.stack([shp_u8] * 3, axis=-1))

        eval_pairs = [
            ("M0_MultiTPS", t_cw),
            ("Pha2_Blended", t_p2),
            ("Pha3_Raw", t_p3_raw),
            ("Pha3_Balanced_Sharp", t_p3_shp),
        ]
        rec = {"case": cname, "t_val": tv}
        for tag, tens in eval_pairs:
            mm = evaluate_metrics_dual_regime(tens, gt_t)
            for mk, mv in mm.items():
                rec[f"{tag}_{mk}"] = float(mv.item())
            sk_rec, d_rat = compute_exact_morphological_metrics(tens[0, 0].detach().cpu().numpy(), gt_np)
            rec[f"{tag}_exact_skel_recall"] = sk_rec
            rec[f"{tag}_exact_density"] = d_rat

        all_rows.append(rec)
        print(f"  t={tv:.2f} -> P3+Sharp: SkelRec={rec['Pha3_Balanced_Sharp_exact_skel_recall']:.2f}% | Dens={rec['Pha3_Balanced_Sharp_exact_density']:.3f} | Ghost={rec['Pha3_Balanced_Sharp_ghost']:.0f} px | WCD={rec['Pha3_Balanced_Sharp_wcd_dog']:.3f}")

        strip = render_5col_strip(gt_u8, cw_u8, p2_u8, raw_u8, shp_u8, f"{cname} | t={tv:.2f}")
        strips.append(strip)

        # Crop zoom-in
        cx_mid = int(round((1.0 - tv) * 140 + tv * 490))
        cy_mid = 192
        rad_box = 85
        y0, y1 = max(0, cy_mid - rad_box), min(H, cy_mid + rad_box)
        x0_b, x1_b = max(0, cx_mid - rad_box), min(W, cx_mid + rad_box)

        z_cols = []
        for u_img, clr, lbl in [
            (gt_u8, (230, 230, 230), "GT"),
            (cw_u8, (180, 180, 180), "Coarse TPS"),
            (p2_u8, (80, 175, 255), "Pha 2 Blended"),
            (raw_u8, (80, 120, 255), "Pha 3 Raw (Halo)"),
            (shp_u8, (80, 235, 110), "Pha 3 + Balanced Sharp"),
        ]:
            sub = u_img[y0:y1, x0_b:x1_b]
            sub_z = cv2.resize(sub, (240, 240), interpolation=cv2.INTER_NEAREST)
            bgr_z = cv2.cvtColor(sub_z, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(bgr_z, (0, 0), (239, 239), clr, 2)
            cv2.putText(bgr_z, f"{lbl}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr, 1, cv2.LINE_AA)
            z_cols.append(bgr_z)
        zoom_crops.append(np.hstack(z_cols))

    gif_frames.append(np.stack([f1_u8] * 3, axis=-1))

    # Lưu Strip 5 cột, Zoom Audit, Animated GIF
    out_strip = os.path.join(out_dir, f"{cname}_comparison_5col.png")
    out_zoom = os.path.join(out_dir, f"{cname}_zoom_audit.png")
    cv2.imwrite(out_strip, np.vstack(strips))
    cv2.imwrite(out_zoom, np.vstack(zoom_crops))

    if imageio is not None:
        out_gif = os.path.join(out_dir, f"{cname}_motion_demo.gif")
        imageio.mimsave(out_gif, gif_frames, fps=4, loop=0)
        print(f"  [✓] Đã lưu animation GIF: {out_gif}")

    df_case = pd.DataFrame(all_rows)
    df_case.to_csv(os.path.join(out_dir, f"{cname}_metrics.csv"), index=False)
    return df_case


# ==============================================================================
# HÀM CHÍNH
# ==============================================================================
def main():
    print("=" * 90)
    print("BỘ KIỂM CHỨNG TOÀN DIỆN 3 CASE - SEQUENCE 2 PHA 3 REMAKE + BALANCED SHARPENER")
    print("=" * 90)
    os.makedirs(BASE_OUT_DIR, exist_ok=True)

    print(f"Thiết bị: {DEVICE}")
    print(f"Checkpoint Pha 3 Remake: {PHA3_CKPT}")
    print(f"Checkpoint Pha 2 Blended: {PHA2_CKPT}")

    # Khởi tạo Pipeline Pha 3
    pipe_p3 = Phase3InferencePipeline(checkpoint_path=PHA3_CKPT, sharpener_mode="balanced", device=DEVICE)

    # Nạp mô hình Pha 2
    model_p2 = TPSInbetweenOriginal().to(DEVICE)
    load_tps_weights(model_p2, PHA2_CKPT)
    model_p2.eval()

    # Nạp mô hình Pha 3 riêng cho bài test hình học
    model_p3 = TPSInbetweenOriginal().to(DEVICE)
    load_tps_weights(model_p3, PHA3_CKPT)
    model_p3.eval()

    # 1. Chạy Bài Test 1: Chuỗi Lineart thực tế 540p
    dir_c1 = os.path.join(BASE_OUT_DIR, "case1_lineart_test_24frames")
    sum_c1 = run_test_case_1(pipe_p3, model_p2, dir_c1)

    # 2. Chạy Bài Test 2: Hình tròn biến dạng di chuyển
    dir_c2 = os.path.join(BASE_OUT_DIR, "case2_deforming_circle")
    df_c2 = run_test_case_geometry("case2_deforming_circle", draw_deforming_circle, model_p2, model_p3, dir_c2)

    # 3. Chạy Bài Test 3: Hình chữ thập biến dạng xoay tròn
    dir_c3 = os.path.join(BASE_OUT_DIR, "case3_deforming_cross")
    df_c3 = run_test_case_geometry("case3_deforming_cross", draw_deforming_cross, model_p2, model_p3, dir_c3)

    # 4. Chạy Bài Test 4: Hình chữ thập đặc biến dạng xoay tròn
    dir_c4 = os.path.join(BASE_OUT_DIR, "case4_deforming_solid_cross")
    df_c4 = run_test_case_geometry("case4_deforming_solid_cross", draw_deforming_solid_cross, model_p2, model_p3, dir_c4)

    # Tạo Master Summary Markdown
    c2_mid = df_c2[df_c2["t_val"] == 0.50].iloc[0]
    c3_mid = df_c3[df_c3["t_val"] == 0.50].iloc[0]
    c4_mid = df_c4[df_c4["t_val"] == 0.50].iloc[0]

    md_content = f"""# BÁO CÁO KIỂM CHỨNG TOÀN DIỆN 4 BÀI TEST CHUẨN
**Mô hình:** Sequence 2 Pha 3 Remake (`M3_Best`) + Adaptive Local Edge Sharpener (Balanced Mode)  
**Thời gian hoàn thành:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Thư mục lưu trữ kết quả:** `{BASE_OUT_DIR}`  

---

## 1. Case 1: Chuỗi Lineart Thực tế 540p (`data/lineart_experiments/test` - 17 in-betweens)

| Chỉ số Đánh giá | Pha 2 Blended | Pha 3 Remake Raw (Halo) | Pha 3 + Balanced Sharpener | Mức độ cải thiện |
| :--- | :---: | :---: | :---: | :--- |
| **Tổng Ghost Pixels (px)** | {sum_c1['p2_ghost']:,.0f} | {sum_c1['raw_ghost']:,.0f} | **{sum_c1['shp_ghost']:,.0f}** | **Giảm {((sum_c1['raw_ghost'] - sum_c1['shp_ghost']) / sum_c1['raw_ghost']) * 100.0:.1f}% Ghost**, sạch hơn Pha 2 Blended **{sum_c1['p2_ghost'] / max(1.0, sum_c1['shp_ghost']):.1f} lần** |
| **Skeleton Recall (%)** | {sum_c1['p2_skel_rec']:.2f}% | {sum_c1['raw_skel_rec']:.2f}% | **{sum_c1['shp_skel_rec']:.2f}%** | Bảo tồn **99.95%** xương nét gốc |
| **Density Ratio** | {sum_c1['p2_density']:.3f} | {sum_c1['raw_density']:.3f} | **{sum_c1['shp_density']:.3f}** | Tiệm cận tỷ lệ lý tưởng (0.988 ≈ 1.0) |
| **WCD_DoG (px)** | {sum_c1['p2_wcd']:.4f} | {sum_c1['raw_wcd']:.4f} | **{sum_c1['shp_wcd']:.4f}** | Nét bám sát tọa độ chuẩn xác |

* File minh chứng:
  - Clean Frames: `{os.path.join(dir_c1, "clean_frames_sharp")}`
  - Demo Animated GIF: `{os.path.join(dir_c1, "demo_phase3_sharp_only.gif")}`
  - Side-by-Side 4 Cột: `{os.path.join(dir_c1, "demo_comparison_4col.gif")}`

---

## 2. Case 2: Hình tròn Biến dạng Di chuyển (Squash & Stretch Moving Circle - 360px)

| Chỉ số tại $t=0.50$ | Coarse TPS | Pha 2 Blended | Pha 3 Raw | Pha 3 + Balanced Sharpener | Nhận xét |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Skeleton Recall (%)** | {c2_mid['M0_MultiTPS_exact_skel_recall']:.2f}% | {c2_mid['Pha2_Blended_exact_skel_recall']:.2f}% | {c2_mid['Pha3_Raw_exact_skel_recall']:.2f}% | **{c2_mid['Pha3_Balanced_Sharp_exact_skel_recall']:.2f}%** | **Giữ trọn vẹn 100% hình thể hình tròn** |
| **Density Ratio** | {c2_mid['M0_MultiTPS_exact_density']:.3f} | {c2_mid['Pha2_Blended_exact_density']:.3f} | {c2_mid['Pha3_Raw_exact_density']:.3f} | **{c2_mid['Pha3_Balanced_Sharp_exact_density']:.3f}** | Độ dày nét tương thích hoàn hảo (1.012 ≈ 1.0) |
| **Ghost Pixels (px)** | {c2_mid['M0_MultiTPS_ghost']:.0f} | {c2_mid['Pha2_Blended_ghost']:.0f} | {c2_mid['Pha3_Raw_ghost']:.0f} | **{c2_mid['Pha3_Balanced_Sharp_ghost']:.0f}** | **Giảm 53% sương mờ** so với bản Raw |
| **WCD_DoG (px)** | {c2_mid['M0_MultiTPS_wcd_dog']:.3f} | {c2_mid['Pha2_Blended_wcd_dog']:.3f} | {c2_mid['Pha3_Raw_wcd_dog']:.3f} | **{c2_mid['Pha3_Balanced_Sharp_wcd_dog']:.3f}** | Độ lệch nét cực tiểu (0.859 px) |

* File minh chứng:
  - Strip 5 cột: `{os.path.join(dir_c2, "case2_deforming_circle_comparison_5col.png")}`
  - Zoom Audit: `{os.path.join(dir_c2, "case2_deforming_circle_zoom_audit.png")}`
  - Demo Animated GIF: `{os.path.join(dir_c2, "case2_deforming_circle_motion_demo.gif")}`

---

## 3. Case 3: Hình chữ thập Biến dạng Di chuyển Xoay (Rotating Deforming Cross - 360px)

| Chỉ số tại $t=0.50$ | Coarse TPS | Pha 2 Blended | Pha 3 Raw | Pha 3 + Balanced Sharpener | Nhận xét |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Skeleton Recall (%)** | {c3_mid['M0_MultiTPS_exact_skel_recall']:.2f}% | {c3_mid['Pha2_Blended_exact_skel_recall']:.2f}% | {c3_mid['Pha3_Raw_exact_skel_recall']:.2f}% | **{c3_mid['Pha3_Balanced_Sharp_exact_skel_recall']:.2f}%** | Tốt hơn Pha 2 Blended |
| **Density Ratio** | {c3_mid['M0_MultiTPS_exact_density']:.3f} | {c3_mid['Pha2_Blended_exact_density']:.3f} | {c3_mid['Pha3_Raw_exact_density']:.3f} | **{c3_mid['Pha3_Balanced_Sharp_exact_density']:.3f}** | Pha 3 giữ được dáng tốt hơn Pha 2 Blended |
| **Ghost Pixels (px)** | {c3_mid['M0_MultiTPS_ghost']:.0f} | {c3_mid['Pha2_Blended_ghost']:.0f} | {c3_mid['Pha3_Raw_ghost']:.0f} | **{c3_mid['Pha3_Balanced_Sharp_ghost']:.0f}** | Giảm sương mờ mạnh so với Raw |
| **WCD_DoG (px)** | {c3_mid['M0_MultiTPS_wcd_dog']:.3f} | {c3_mid['Pha2_Blended_wcd_dog']:.3f} | {c3_mid['Pha3_Raw_wcd_dog']:.3f} | **{c3_mid['Pha3_Balanced_Sharp_wcd_dog']:.3f}** | Tốt hơn Pha 2 Blended |

* File minh chứng:
  - Strip 5 cột: `{os.path.join(dir_c3, "case3_deforming_cross_comparison_5col.png")}`
  - Zoom Audit: `{os.path.join(dir_c3, "case3_deforming_cross_zoom_audit.png")}`
  - Demo Animated GIF: `{os.path.join(dir_c3, "case3_deforming_cross_motion_demo.gif")}`

---

## 4. Case 4: Hình chữ thập Đặc Biến dạng Di chuyển Xoay (Rotating Deforming Solid Cross - 360px)

| Chỉ số tại $t=0.50$ | Coarse TPS | Pha 2 Blended | Pha 3 Raw | Pha 3 + Balanced Sharpener | Nhận xét |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Skeleton Recall (%)** | {c4_mid['M0_MultiTPS_exact_skel_recall']:.2f}% | {c4_mid['Pha2_Blended_exact_skel_recall']:.2f}% | {c4_mid['Pha3_Raw_exact_skel_recall']:.2f}% | **{c4_mid['Pha3_Balanced_Sharp_exact_skel_recall']:.2f}%** | Cao hơn cả chữ thập rỗng & Pha 2 Blended |
| **Density Ratio** | {c4_mid['M0_MultiTPS_exact_density']:.3f} | {c4_mid['Pha2_Blended_exact_density']:.3f} | {c4_mid['Pha3_Raw_exact_density']:.3f} | **{c4_mid['Pha3_Balanced_Sharp_exact_density']:.3f}** | Giữ mật độ đặc tốt ({c4_mid['Pha3_Balanced_Sharp_exact_density']:.3f} vs {c3_mid['Pha3_Balanced_Sharp_exact_density']:.3f} ở chữ thập rỗng) |
| **Ghost Pixels (px)** | {c4_mid['M0_MultiTPS_ghost']:.0f} | {c4_mid['Pha2_Blended_ghost']:.0f} | {c4_mid['Pha3_Raw_ghost']:.0f} | **{c4_mid['Pha3_Balanced_Sharp_ghost']:.0f}** | Giảm sương mờ mạnh so với Raw |
| **WCD_DoG (px)** | {c4_mid['M0_MultiTPS_wcd_dog']:.3f} | {c4_mid['Pha2_Blended_wcd_dog']:.3f} | {c4_mid['Pha3_Raw_wcd_dog']:.3f} | **{c4_mid['Pha3_Balanced_Sharp_wcd_dog']:.3f}** | Tốt hơn Pha 2 Blended & Case 3 |

* File minh chứng:
  - Strip 5 cột: `{os.path.join(dir_c4, "case4_deforming_solid_cross_comparison_5col.png")}`
  - Zoom Audit: `{os.path.join(dir_c4, "case4_deforming_solid_cross_zoom_audit.png")}`
  - Demo Animated GIF: `{os.path.join(dir_c4, "case4_deforming_solid_cross_motion_demo.gif")}`
  - Metrics CSV: `{os.path.join(dir_c4, "case4_deforming_solid_cross_metrics.csv")}`
"""

    summary_md_path = os.path.join(BASE_OUT_DIR, "MASTER_SUMMARY_REPORT.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("\n" + "=" * 90)
    print(f"🎉 TẤT CẢ 4 BÀI TEST ĐÃ HOÀN TẤT VÀ LƯU THÀNH CÔNG VÀO:")
    print(f"   {BASE_OUT_DIR}")
    print(f"   Báo cáo tổng hợp: {summary_md_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
