#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_deforming_shapes.py
--------------------------------------------------------------------------------
BENCHMARK ĐỐI ĐẦU HÌNH HỌC PHÒNG THỦ (CHUẨN LAGRANGIAN TPS BENCHMARK #3):
  Case 1: Hình tròn di chuyển & biến dạng co giãn (Squash-Stretch Moving Circle)
  Case 2: Hình chữ thập di chuyển & biến dạng xoay/duỗi (Rotating Deforming Cross)

Sử dụng đúng cơ chế giải chuẩn hóa của Benchmark #3:
  - direct_t_flows = True (Lagrangian Trajectory TPS trực tiếp tại thời điểm t)
  - use_dyn_anchors = False (Không dùng 8 điểm neo cố định ở biên cho vật thể đơn lẻ)
--------------------------------------------------------------------------------
"""

import os
import sys
import time
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PHASE3_WORKER_DIR = os.path.join(REPO_ROOT, "kaggle_seq_p3_multilayer_tps")

for _d in (SCRIPT_DIR, PHASE3_WORKER_DIR):
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

DEFAULT_PHA2_CKPT = os.path.join(REPO_ROOT, "checkpoints", "Re-train", "Sequence 2 Blended", "checkpoints", "model_tps_blended.pt")
DEFAULT_PHA3_CKPT = os.path.join(REPO_ROOT, "checkpoints", "Re-train", "Sequence 2 pha 3 remake", "checkpoints", "model_tps_finetuned_best.pt")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "demo_02_seq_p3_sharp_outputs", "benchmark_deforming_shapes")


def draw_deforming_circle(t, H=384, W=640):
    im = np.zeros((H, W), dtype=np.float32)
    cx = int(round((1.0 - t) * 140 + t * 490))
    cy = int(round((1.0 - t) * 192 + t * 192))
    # Biến dạng chuẩn Bench 3: từ Oval 90x40 (t=0) thành Tròn R=45 (t=1)
    rx = int(round((1.0 - t) * 90 + t * 45))
    ry = int(round((1.0 - t) * 40 + t * 45))
    cv2.ellipse(im, (cx, cy), (rx, ry), 0, 0, 360, 1.0, 4, lineType=cv2.LINE_AA)
    return im


def draw_deforming_cross(t, H=384, W=640):
    im = np.zeros((H, W), dtype=np.float32)
    cx = int(round((1.0 - t) * 140 + t * 490))
    cy = int(round((1.0 - t) * 192 + t * 192))
    # Biến dạng chữ thập: thanh ngang duỗi dài (80 -> 160), thanh dọc co ngắn (120 -> 45), xoay nghiêng 30 độ
    lx = (1.0 - t) * 80.0 + t * 160.0
    ly = (1.0 - t) * 120.0 + t * 45.0
    angle_rad = np.radians(t * 30.0)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    p1 = (int(round(cx - (lx / 2) * cos_a)), int(round(cy - (lx / 2) * sin_a)))
    p2 = (int(round(cx + (lx / 2) * cos_a)), int(round(cy + (lx / 2) * sin_a)))
    cv2.line(im, p1, p2, 1.0, 4, lineType=cv2.LINE_AA)

    q1 = (int(round(cx + (ly / 2) * sin_a)), int(round(cy - (ly / 2) * cos_a)))
    q2 = (int(round(cx - (ly / 2) * sin_a)), int(round(cy + (ly / 2) * cos_a)))
    cv2.line(im, q1, q2, 1.0, 4, lineType=cv2.LINE_AA)
    return im


def ink_tensor_to_u8(t):
    arr = (1.0 - t.squeeze().detach().cpu().numpy().clip(0.0, 1.0)) * 255.0
    return np.clip(np.round(arr), 0, 255).astype(np.uint8)


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


def _render_5col_strip(gt_u8, m0_u8, p2_u8, p3_raw_u8, p3_shp_u8, title_str):
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


def main():
    os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Chạy Benchmark Biến dạng Hình học trên Device: {device}")

    m_p2 = TPSInbetweenOriginal().to(device).eval()
    m_p3 = TPSInbetweenOriginal().to(device).eval()
    load_tps_weights(m_p2, DEFAULT_PHA2_CKPT, device=device, prefer_ema=True)
    load_tps_weights(m_p3, DEFAULT_PHA3_CKPT, device=device, prefer_ema=True)

    H, W = 384, 640
    t_vals = [0.25, 0.50, 0.75]

    cases = [
        ("Case1_Deforming_Circle", draw_deforming_circle),
        ("Case2_Deforming_Cross", draw_deforming_cross),
    ]

    all_rows = []

    for cname, fn_draw in cases:
        print(f"\n▶ Đang kiểm định: {cname} (Chuẩn Lagrangian direct_t_flows=True) ...")
        im0 = fn_draw(0.0, H, W)
        im1 = fn_draw(1.0, H, W)

        k0, k1 = extract_hybrid_contour_matches(im0, im1, n_pts_per_comp=32)

        x0 = torch.from_numpy(im0)[None, None].to(device)
        x1 = torch.from_numpy(im1)[None, None].to(device)

        strips = []
        zoom_crops = []

        for tv in t_vals:
            gt_np = fn_draw(tv, H, W)
            gt_t = torch.from_numpy(gt_np)[None, None].to(device)
            tt = torch.tensor([[[[float(tv)]]]], dtype=torch.float32, device=device)

            # Giải TPS chuẩn Lagrangian Trajectory trực tiếp tại thời điểm t
            cf0t, cf1t, _ = solve_multi_layer_tps(
                k0, k1, (H, W), H, W, device,
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

            strip = _render_5col_strip(gt_u8, cw_u8, p2_u8, raw_u8, shp_u8, f"{cname} | t={tv:.2f}")
            strips.append(strip)

            # Crop zoom-in vào vùng trọng tâm của hình
            cx_mid = int(round((1.0 - tv) * 140 + tv * 490))
            cy_mid = 192
            rad_box = 85
            y0, y1 = max(0, cy_mid - rad_box), min(H, cy_mid + rad_box)
            x0_b, x1_b = max(0, cx_mid - rad_box), min(W, cx_mid + rad_box)

            z_cols = []
            for u_img, clr, lbl in [
                (gt_u8, (230, 230, 230), "GT"),
                (cw_u8, (180, 180, 180), "M0 MultiTPS"),
                (p2_u8, (80, 175, 255), "Pha2 Blended"),
                (raw_u8, (80, 120, 255), "Pha3 Raw (Halo)"),
                (shp_u8, (80, 235, 110), "Pha3 + Balanced Sharp"),
            ]:
                sub = u_img[y0:y1, x0_b:x1_b]
                sub_z = cv2.resize(sub, (240, 240), interpolation=cv2.INTER_NEAREST)
                bgr_z = cv2.cvtColor(sub_z, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(bgr_z, (0, 0), (239, 239), clr, 2)
                cv2.putText(bgr_z, f"{lbl} (t={tv:.2f})", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr, 1, cv2.LINE_AA)
                z_cols.append(bgr_z)
            zoom_crops.append(np.hstack(z_cols))

        out_strip_path = os.path.join(DEFAULT_OUT_DIR, f"{cname}_comparison_5col.png")
        cv2.imwrite(out_strip_path, np.vstack(strips))
        print(f"  [✓] Đã xuất ảnh đối chiếu: {out_strip_path}")

        out_zoom_path = os.path.join(DEFAULT_OUT_DIR, f"{cname}_zoom_audit.png")
        cv2.imwrite(out_zoom_path, np.vstack(zoom_crops))
        print(f"  [✓] Đã xuất ảnh zoom-in: {out_zoom_path}")

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(os.path.join(DEFAULT_OUT_DIR, "deforming_shapes_per_case.csv"), index=False)

    tags = ["M0_MultiTPS", "Pha2_Blended", "Pha3_Raw", "Pha3_Balanced_Sharp"]
    metric_cols = ["psnr", "ssim", "ghost", "wcd_dog", "exact_skel_recall", "exact_density", "f1"]
    sum_rows = []
    for tg in tags:
        r = {"model_tag": tg, "cases_eval": len(df_all)}
        for mc in metric_cols:
            r[mc] = float(df_all[f"{tg}_{mc}"].mean())
        sum_rows.append(r)
    df_sum = pd.DataFrame(sum_rows)
    df_sum.to_csv(os.path.join(DEFAULT_OUT_DIR, "deforming_shapes_summary.csv"), index=False)

    print("\n" + "=" * 90)
    print("📊 BẢNG TỔNG HỢP BENCHMARK BIẾN DẠNG HÌNH HỌC (CHUẨN DIRECT LAGRANGIAN TPS):")
    print("=" * 90)
    print(df_sum[["model_tag", "psnr", "ssim", "ghost", "wcd_dog", "exact_skel_recall", "exact_density", "f1"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
