#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_demo_phase3_sharp.py
--------------------------------------------------------------------------------
BENCHMARK & TẠO DEMO TRỰC QUAN TRÊN TẬP THỰC TẾ `data/lineart_experiments/test`
Đối đầu trực tiếp 3 phiên bản trên 17 khung hình nội suy (24 frames 540p):
  1. Pha 2 Blended (`model_tps_blended.pt`)
  2. Pha 3 Remake Raw (`model_tps_finetuned_best.pt` - chưa qua lọc sắc nét)
  3. Pha 3 + Balanced Adaptive Sharpener (`Phase3InferencePipeline` chính thức)

Xuất ra:
  - 24 frames PNG sạch thành phẩm (`clean_frames_sharp/`)
  - 24 ảnh đối chiếu Side-by-Side 4 cột (`side_by_side_4col/`)
  - Video GIF & MP4 đối chiếu 4 cột (`demo_comparison.gif` / `.mp4`)
  - Video GIF & MP4 riêng thành phẩm (`demo_sharp_only.gif` / `.mp4`)
  - Ảnh Montage Zoom-in chi tiết các frame khó (`montage_zoom_audit.png`)
  - Bảng số liệu CSV chi tiết từng frame và tổng hợp (`benchmark_per_frame.csv`, `benchmark_summary.csv`)
--------------------------------------------------------------------------------
"""

import os
import sys
import time
import glob
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
PHASE3_WORKER_DIR = os.path.join(REPO_ROOT, "kaggle_finetune_sequence_phase3")

for _d in (SCRIPT_DIR, PHASE3_WORKER_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from phase3_inference_pipeline import (
    Phase3InferencePipeline,
    load_ink_from_input,
    ink_tensor_to_u8,
    DEFAULT_PHASE3_CKPT,
)
from ddp_worker_tps_sequence_phase3 import (
    TPSInbetweenOriginal,
    load_tps_weights,
    evaluate_metrics_dual_regime,
)

DEFAULT_TEST_DIR = os.path.join(REPO_ROOT, "data", "lineart_experiments", "test")
DEFAULT_GS_NPZ = os.path.join(REPO_ROOT, "kaggle_finetune_sequence_final", "lineart_test_gluestick_matches.npz")
DEFAULT_PHA2_CKPT = os.path.join(REPO_ROOT, "checkpoints", "Re-train", "Sequence 2 Blended", "checkpoints", "model_tps_blended.pt")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "demo_outputs", "phase3_sharp_demo")


def _make_panel(u8_img, title, color_bgr, is_keyframe=False):
    h, w = u8_img.shape
    head_h = 34
    canvas = np.full((head_h + h, w, 3), 24, dtype=np.uint8)
    canvas[head_h:, :] = cv2.cvtColor(u8_img, cv2.COLOR_GRAY2BGR)
    cv2.putText(canvas, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color_bgr, 2, cv2.LINE_AA)
    if is_keyframe:
        cv2.rectangle(canvas, (0, head_h), (w - 1, head_h + h - 1), (0, 200, 255), 3)
    return canvas


def main():
    os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
    clean_dir = os.path.join(DEFAULT_OUT_DIR, "clean_frames_sharp")
    sbs_dir = os.path.join(DEFAULT_OUT_DIR, "side_by_side_4col")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(sbs_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(DEFAULT_TEST_DIR, "*.png")))
    kf_idx = [0, 3, 7, 11, 14, 19, 23]
    kf_set = set(kf_idx)
    gs_cache = np.load(DEFAULT_GS_NPZ)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Khởi tạo Benchmark & Demo trên Device: {device}")

    # Nạp Pha 3 Pipeline chính thức (tích hợp Balanced Adaptive Sharpener)
    pipe_p3 = Phase3InferencePipeline(
        checkpoint_path=DEFAULT_PHASE3_CKPT,
        device=device,
        use_sharpener=True,
        sharpener_mode="balanced",
    )

    # Nạp Pha 2 Blended để đối chứng trực tiếp
    model_p2 = TPSInbetweenOriginal().to(device).eval()
    load_tps_weights(model_p2, DEFAULT_PHA2_CKPT, device=device, prefer_ema=True)

    total_frames = len(paths)
    frames_p2 = [None] * total_frames
    frames_p3_raw = [None] * total_frames
    frames_p3_sharp = [None] * total_frames
    frames_gt = [None] * total_frames

    # Keyframes gốc
    for i in range(total_frames):
        ink_gt = load_ink_from_input(paths[i])
        u8_gt = np.clip((1.0 - ink_gt) * 255.0, 0, 255).astype(np.uint8)
        frames_gt[i] = u8_gt
        if i in kf_set:
            frames_p2[i] = u8_gt
            frames_p3_raw[i] = u8_gt
            frames_p3_sharp[i] = u8_gt
            Image.fromarray(u8_gt).save(os.path.join(clean_dir, os.path.basename(paths[i])))

    rows = []
    prev_t = {"Pha2_Blended": None, "M3_Raw": None, "M3_Balanced_Sharp": None}

    t_start = time.time()
    for s_idx in range(len(kf_idx) - 1):
        i0, i1 = kf_idx[s_idx], kf_idx[s_idx + 1]
        delta_f = i1 - i0
        ink0 = load_ink_from_input(paths[i0])
        ink1 = load_ink_from_input(paths[i1])
        h, w = ink0.shape
        h_pad, w_pad = ((h + 31) // 32) * 32, ((w + 31) // 32) * 32

        kps = gs_cache[f"seg_{i0}_{i1}"].astype(np.float32)
        cf01, cf10, _, _, _ = pipe_p3.compute_multilayer_flows(ink0, ink1, kps0=kps[0], kps1=kps[1])
        x0 = F.pad(torch.from_numpy(ink0)[None, None], (0, w_pad - w, 0, h_pad - h)).to(device)
        x1 = F.pad(torch.from_numpy(ink1)[None, None], (0, w_pad - w, 0, h_pad - h)).to(device)

        for f_idx in range(i0 + 1, i1):
            tv = float(f_idx - i0) / float(delta_f)
            gt_ink_np = load_ink_from_input(paths[f_idx])
            gt_t = torch.from_numpy(gt_ink_np)[None, None].to(device)
            tt = torch.tensor([[[[tv]]]], dtype=torch.float32, device=device)

            # 1. Pha 2 Blended
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
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

            Image.fromarray(frames_p3_sharp[f_idx]).save(
                os.path.join(clean_dir, os.path.basename(paths[f_idx]))
            )

            # Đo lường từng frame
            rec = {"frame_idx": f_idx, "filename": os.path.basename(paths[f_idx]), "delta_f": delta_f, "t_val": round(tv, 4)}
            bg_mask = (gt_t < 0.05).float()
            for tag, tens in [("Pha2_Blended", t_p2), ("M3_Raw", t_raw), ("M3_Balanced_Sharp", t_shp)]:
                mm = evaluate_metrics_dual_regime(tens, gt_t)
                for mk, mv in mm.items():
                    rec[f"{tag}_{mk}"] = float(mv.item())
                if prev_t[tag] is not None:
                    rec[f"{tag}_temp_bg_flicker"] = float(((tens - prev_t[tag]).abs() * bg_mask).sum().item())
                else:
                    rec[f"{tag}_temp_bg_flicker"] = 0.0
                prev_t[tag] = tens
            rows.append(rec)

    infer_time = time.time() - t_start
    print(f"⏱️ Hoàn tất suy luận 17 in-betweens (3 mô hình/chế độ) trong {infer_time:.2f}s")

    # Lưu bảng CSV
    df_per = pd.DataFrame(rows)
    df_per.to_csv(os.path.join(DEFAULT_OUT_DIR, "benchmark_per_frame.csv"), index=False)

    metrics_list = ["psnr", "ssim", "ie_mae", "ghost", "dark_ghost", "wcd", "wcd_dog", "semd", "skel_recall", "tip_recall", "f1", "density_ratio", "temp_bg_flicker"]
    sum_rows = []
    for tag in ["Pha2_Blended", "M3_Raw", "M3_Balanced_Sharp"]:
        sr = {"model_tag": tag, "inbetween_frames": len(df_per)}
        for mc in metrics_list:
            sr[mc] = float(df_per[f"{tag}_{mc}"].mean())
        sum_rows.append(sr)
    df_sum = pd.DataFrame(sum_rows)
    df_sum.to_csv(os.path.join(DEFAULT_OUT_DIR, "benchmark_summary.csv"), index=False)

    print("\n📊 BẢNG TỔNG HỢP BENCHMARK DEMO (17 IN-BETWEENS THỰC TẾ 540p):")
    print(df_sum[["model_tag", "psnr", "ssim", "ghost", "wcd", "wcd_dog", "skel_recall", "tip_recall", "f1", "density_ratio"]].round(4).to_string(index=False))

    # Xuất ảnh Side-by-Side 4 cột & video
    sbs_rgb_list = []
    sharp_rgb_list = []
    for i in range(total_frames):
        is_kf = i in kf_set
        tag_kf = " [KEYFRAME]" if is_kf else ""
        p_gt = _make_panel(frames_gt[i], f"Ground Truth (F{i:02d}){tag_kf}", (230, 230, 230), is_kf)
        p_p2 = _make_panel(frames_p2[i], "Pha 2 Blended", (80, 175, 255), is_kf)
        p_raw = _make_panel(frames_p3_raw[i], "Pha 3 Remake (Raw - Mo/Halo)", (80, 120, 255), is_kf)
        p_shp = _make_panel(frames_p3_sharp[i], "Pha 3 + Balanced Sharp (Moi)", (80, 235, 110), is_kf)

        sbs_bgr = np.hstack([p_gt, p_p2, p_raw, p_shp])
        # Thu nhỏ 50% (1920px wide) cho dễ xem mượt
        sbs_resized = cv2.resize(sbs_bgr, (sbs_bgr.shape[1] // 2, sbs_bgr.shape[0] // 2), interpolation=cv2.INTER_AREA)
        out_sbs_path = os.path.join(sbs_dir, f"compare_{i:02d}_{os.path.basename(paths[i])}")
        cv2.imwrite(out_sbs_path, sbs_resized)

        sbs_rgb_list.append(cv2.cvtColor(sbs_resized, cv2.COLOR_BGR2RGB))
        sharp_rgb_list.append(np.stack([frames_p3_sharp[i]] * 3, axis=-1))

    # Tạo Montage Zoom 3x vào 3 frame chuyển động nhanh (F05, F09, F17)
    zoom_cases = [
        (5, (120, 320, 260, 460), "Frame 05 (delta_f=4 - Chuyen dong toc/tay)"),
        (9, (160, 380, 300, 520), "Frame 09 (delta_f=4 - Bien dang xoay)"),
        (17, (140, 350, 280, 490), "Frame 17 (delta_f=5 - Khoang trong xa nhat)"),
    ]
    montage_rows = []
    for fidx, (y0, x0, y1, x1), lbl in zoom_cases:
        z_panels = []
        for u8_img, title, clr in [
            (frames_gt[fidx], "Ground Truth", (220, 220, 220)),
            (frames_p2[fidx], "Pha 2 Blended", (80, 175, 255)),
            (frames_p3_raw[fidx], "Pha 3 Raw (Halo)", (80, 120, 255)),
            (frames_p3_sharp[fidx], "Pha 3 + Balanced Sharp", (80, 235, 110)),
        ]:
            crop = u8_img[y0:y1, x0:x1]
            zoom = cv2.resize(crop, (360, 360), interpolation=cv2.INTER_NEAREST)
            bgr = cv2.cvtColor(zoom, cv2.COLOR_GRAY2BGR)
            hdr = np.full((30, 360, 3), 25, dtype=np.uint8)
            cv2.putText(hdr, title, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, clr, 2, cv2.LINE_AA)
            cv2.rectangle(bgr, (0, 0), (359, 359), clr, 2)
            z_panels.append(np.vstack([hdr, bgr]))
        row_strip = np.hstack(z_panels)
        ftr = np.full((26, row_strip.shape[1], 3), 18, dtype=np.uint8)
        cv2.putText(ftr, lbl, (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)
        montage_rows.append(np.vstack([row_strip, ftr]))

    montage_img = np.vstack(montage_rows)
    montage_path = os.path.join(DEFAULT_OUT_DIR, "montage_zoom_audit.png")
    cv2.imwrite(montage_path, montage_img)
    print(f"[✓] Đã xuất Montage Zoom 3x: {montage_path}")

    # Xuất GIF và MP4
    if imageio is not None:
        gif_cmp = os.path.join(DEFAULT_OUT_DIR, "demo_comparison_4col.gif")
        gif_shp = os.path.join(DEFAULT_OUT_DIR, "demo_phase3_sharp_only.gif")
        imageio.mimsave(gif_cmp, sbs_rgb_list, fps=8, loop=0)
        imageio.mimsave(gif_shp, sharp_rgb_list, fps=8, loop=0)
        print(f"[✓] Đã xuất GIF so sánh: {gif_cmp}")
        print(f"[✓] Đã xuất GIF thành phẩm: {gif_shp}")
        try:
            mp4_cmp = os.path.join(DEFAULT_OUT_DIR, "demo_comparison_4col.mp4")
            mp4_shp = os.path.join(DEFAULT_OUT_DIR, "demo_phase3_sharp_only.mp4")
            imageio.mimsave(mp4_cmp, sbs_rgb_list, fps=8, macro_block_size=1)
            imageio.mimsave(mp4_shp, sharp_rgb_list, fps=8, macro_block_size=1)
            print(f"[✓] Đã xuất MP4 so sánh: {mp4_cmp}")
            print(f"[✓] Đã xuất MP4 thành phẩm: {mp4_shp}")
        except Exception as e:
            print(f"[!] Bỏ qua xuất MP4: {e}")

    print(f"\n🎉 HOÀN TẤT DEMO & BENCHMARK TẠI: {DEFAULT_OUT_DIR}")


if __name__ == "__main__":
    main()
