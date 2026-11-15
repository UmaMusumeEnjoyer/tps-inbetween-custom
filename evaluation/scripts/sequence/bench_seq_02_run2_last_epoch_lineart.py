#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
test_last_epoch_lineart.py
--------------------------------------------------------------------------------
Thực hiện test chi tiết checkpoint Last Epoch (Epoch 15: model_tps_finetuned_last.pt)
trên tập dữ liệu kiểm thử thực tế: data/lineart_experiments/test

Bao gồm:
1. Tải mô hình M1 Original (model_latest.pt) và M3 Last (model_tps_finetuned_last.pt).
2. Tái tạo 17 inbetween frames từ 7 keyframes với điểm neo GlueStick.
3. Xuất toàn bộ 24 frames sạch (Clean Line Art) cho M3 Last để kiểm tra nét vẽ trực quan.
4. Xuất các bảng đối đầu Side-by-Side 4 cột (Input | M1 Original | M3 Last | Ground Truth).
5. Xuất bản đồ nhiệt sai số Error Heatmap (M1 vs M3 Last).
6. Xuất Video/GIF đối đầu so sánh và Video/GIF chuyển động mượt mà của riêng M3 Last.
7. Đo đạc định lượng chuẩn xác: PSNR, SSIM, WCD, WCD_DoG, Skel Recall, Density Ratio, Ghost.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import cv2
import imageio
from PIL import Image

import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
BENCH_SUITE_DIR = os.path.join(REPO_ROOT, "Dataset", "sakuga_pipeline_sequence", "final_dataset", "sakuga_golden_benchmark_sequences_v2")
PIPELINES_DIR = os.path.join(REPO_ROOT, "src", "sequence_p3_p4")
for _p in (SCRIPT_DIR, BENCH_SUITE_DIR, PIPELINES_DIR):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from seq_dual_benchmark_suite import (
    TPSInbetweenOriginal,
    load_tps_weights,
    ink_to_u8_gray,
    evaluate_metrics_dual_regime,
    solve_tps_backward_pair,
    tps_flow_from_theta_batch
)

ORIG_CKPT = os.path.join(REPO_ROOT, "checkpoints", "pretrained_baselines", "tps-original", "model_latest.pt")
LAST_CKPT = os.path.join(REPO_ROOT, "checkpoints", "Re-train", "Sequence 2", "checkpoints", "model_tps_finetuned_last.pt")
TEST_DIR = os.path.join(REPO_ROOT, "data", "lineart_experiments", "test")
GLUESTICK_NPZ = os.path.join(REPO_ROOT, "data_samples", "caches", "lineart_test_gluestick_matches.npz")
OUT_DIR = os.path.join(REPO_ROOT, "checkpoints", "Re-train", "Sequence 2", "test_last_epoch_lineart")


def main():
    print("=" * 86)
    print("🚀 BẮT ĐẦU TEST CHÚNG CHECKPOINT LAST EPOCH TRÊN TẬP LINEART TEST")
    print(f"[*] M1 Baseline Checkpoint: {ORIG_CKPT}")
    print(f"[*] M3 Last Epoch Checkpoint: {LAST_CKPT}")
    print(f"[*] Dataset Test: {TEST_DIR}")
    print(f"[*] Output Directory: {OUT_DIR}")
    print("=" * 86)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Thiết bị thực thi: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    os.makedirs(OUT_DIR, exist_ok=True)
    clean_frames_dir = os.path.join(OUT_DIR, "m3_last_generated_frames_clean")
    sbs_dir = os.path.join(OUT_DIR, "frames_side_by_side_M1_vs_M3_Last")
    heat_dir = os.path.join(OUT_DIR, "error_heatmaps_M1_vs_M3_Last")
    os.makedirs(clean_frames_dir, exist_ok=True)
    os.makedirs(sbs_dir, exist_ok=True)
    os.makedirs(heat_dir, exist_ok=True)

    # 1. Nạp mô hình
    print("\n📦 Đang nạp mô hình M1 Original...")
    m1 = TPSInbetweenOriginal().to(device)
    load_tps_weights(m1, ORIG_CKPT, device=device, prefer_ema=False)
    m1.eval()

    print("📦 Đang nạp mô hình M3 Last (Epoch 15)...")
    m3_last = TPSInbetweenOriginal().to(device)
    load_tps_weights(m3_last, LAST_CKPT, device=device, prefer_ema=True)
    m3_last.eval()

    models_dict = {'M1_Original': m1, 'M3_Last': m3_last}

    # 2. Đọc cấu hình sequence
    spec_path = os.path.join(TEST_DIR, "continuous_keyframes_spec.json")
    with open(spec_path, 'r', encoding='utf-8') as f:
        spec = json.load(f)
    frames_spec = spec['frames']
    key_indices = sorted(spec['key_indices'])

    print(f"\n🎬 Chuỗi kiểm thử: {len(frames_spec)} frames ({len(key_indices)} keyframes, {len(frames_spec) - len(key_indices)} inbetween frames)")

    gt_imgs_u8 = [np.asarray(Image.open(os.path.join(TEST_DIR, item['filename'])).convert('L'), dtype=np.uint8) for item in frames_spec]
    h, w = gt_imgs_u8[0].shape
    h_pad, w_pad = ((h + 31) // 32) * 32, ((w + 31) // 32) * 32
    downscale_factor = 2

    # 3. Chuẩn bị biến dạng TPS từ GlueStick Matches
    gs_cache = dict(np.load(GLUESTICK_NPZ)) if os.path.exists(GLUESTICK_NPZ) else None
    if gs_cache is None:
        raise FileNotFoundError(f"Không tìm thấy file cache GlueStick: {GLUESTICK_NPZ}")

    print(f"[*] Đã tải thành công GlueStick keypoints cache: {len(gs_cache)} segments.")

    seg_data = {}
    for i in range(len(key_indices) - 1):
        ka, kb = key_indices[i], key_indices[i + 1]
        seg_key = f"seg_{ka}_{kb}"
        ima_u8, imb_u8 = gt_imgs_u8[ka], gt_imgs_u8[kb]
        kps = gs_cache[seg_key].astype(np.float32)

        c0_np, t01_np, c1_np, t10_np, vf = solve_tps_backward_pair(kps[0], kps[1], max_pts=128, lambd=0.01)
        x0_p = F.pad(torch.from_numpy((255.0 - ima_u8.astype(np.float32)) / 255.0)[None, None], (0, w_pad - w, 0, h_pad - h)).to(device)
        x1_p = F.pad(torch.from_numpy((255.0 - imb_u8.astype(np.float32)) / 255.0)[None, None], (0, w_pad - w, 0, h_pad - h)).to(device)
        hw_t = torch.tensor([[h, w]], dtype=torch.int32, device=device)
        cf01 = tps_flow_from_theta_batch(torch.from_numpy(t01_np)[None].to(device), torch.from_numpy(c0_np)[None].to(device), torch.tensor([vf], device=device), hw_t, h_pad, w_pad, device, downscale_factor)
        cf10 = tps_flow_from_theta_batch(torch.from_numpy(t10_np)[None].to(device), torch.from_numpy(c1_np)[None].to(device), torch.tensor([vf], device=device), hw_t, h_pad, w_pad, device, downscale_factor)
        seg_data[(ka, kb)] = {'x0': x0_p, 'x1': x1_p, 'cf01': cf01, 'cf10': cf10, 'num_matches': kps.shape[1]}

    reconstructed = {tag: [img.copy() for img in gt_imgs_u8] for tag in ['M0', 'M1_Original', 'M3_Last']}
    per_frame_rows = []

    print("\n⚡ Đang thực thi suy luận tạo từng frame cho M1 và M3 Last...")
    for item in frames_spec:
        f_idx = int(item['frame_idx'])
        if item['type'] != 'INBETWEEN':
            continue
        ka, kb, tv = int(item['ref_key_a']), int(item['ref_key_b']), float(item['t_val'])
        delta_f = kb - ka
        seg = seg_data[(ka, kb)]
        gt_ink = torch.from_numpy((255.0 - gt_imgs_u8[f_idx].astype(np.float32)) / 255.0)[None, None].to(device)
        tt = torch.tensor([[[[tv]]]], dtype=torch.float32, device=device)

        row_base = {'frame_idx': f_idx, 'time_sec': item['time_sec'], 'filename': item['filename'],
                    'segment': f"{ka}->{kb}", 'delta_f': delta_f, 't_val': round(tv, 4), 'matches': seg['num_matches']}

        cw_done = False
        for m_name, mdl in models_dict.items():
            ts = time.time()
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                pr_p, cw_p, *_ = mdl(seg['x0'], seg['x1'], tt, seg['cf01'], seg['cf10'])
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            lat = (time.time() - ts) * 1000.0
            pr_crop = pr_p[:, :, :h, :w]
            reconstructed[m_name][f_idx] = ink_to_u8_gray(pr_crop)
            mm = evaluate_metrics_dual_regime(pr_crop, gt_ink)

            if not cw_done:
                cw_crop = cw_p[:, :, :h, :w]
                reconstructed['M0'][f_idx] = ink_to_u8_gray(cw_crop)
                mm0 = evaluate_metrics_dual_regime(cw_crop, gt_ink)
                for mk, mv in mm0.items():
                    row_base[f"M0_{mk}"] = float(mv.item())
                cw_done = True

            for mk, mv in mm.items():
                row_base[f"{m_name}_{mk}"] = float(mv.item())
            row_base[f"{m_name}_latency_ms"] = float(lat)

        per_frame_rows.append(row_base)
        print(f"  Frame {f_idx:02d} ({item['filename']}) [{ka}->{kb}, t={tv:.2f}]: "
              f"M1 [PSNR={row_base['M1_Original_psnr']:.2f}dB, WCD={row_base['M1_Original_wcd_dog']:.3f}, Skel={row_base['M1_Original_skel_recall']*100:.1f}%, Dens={row_base['M1_Original_density_ratio']:.3f}] | "
              f"M3_Last [PSNR={row_base['M3_Last_psnr']:.2f}dB, WCD={row_base['M3_Last_wcd_dog']:.3f}, Skel={row_base['M3_Last_skel_recall']*100:.1f}%, Dens={row_base['M3_Last_density_ratio']:.3f}]")

    df_frames = pd.DataFrame(per_frame_rows)
    df_frames.to_csv(os.path.join(OUT_DIR, "lineart_test_per_frame_metrics_M3_Last.csv"), index=False)

    # Tổng hợp metrics
    summary_list = []
    tags = ['M0', 'M1_Original', 'M3_Last']
    for grp_name, sub_df in [("ALL_17_INBETWEENS", df_frames)] + [(f"delta_f={dfv}", df_frames[df_frames['delta_f'] == dfv]) for dfv in sorted(df_frames['delta_f'].unique())]:
        for t in tags:
            sr = {'group': grp_name, 'model_tag': t, 'frames': len(sub_df)}
            for mc in ['psnr', 'ssim', 'ie_mae', 'wcd', 'wcd_dog', 'semd', 'skel_recall', 'tip_recall', 'f1', 'density_ratio', 'ghost', 'pwcr', 'rgd']:
                sr[mc] = float(sub_df[f"{t}_{mc}"].mean())
            summary_list.append(sr)
    df_sum = pd.DataFrame(summary_list)
    df_sum.to_csv(os.path.join(OUT_DIR, "lineart_test_summary_M3_Last.csv"), index=False)

    print("\n" + "=" * 86)
    print("📊 BẢNG TỔNG HỢP TOÀN BỘ 17 FRAMES SUY LUẬN LIÊN TỤC (M1 vs M3 LAST):")
    print("=" * 86)
    print(df_sum[df_sum['group'] == 'ALL_17_INBETWEENS'].round(4).to_string(index=False))

    # 4. Xuất ảnh Clean Frames và Visual Panels
    print("\n🖼️ Đang xuất toàn bộ 24 frames sạch và các visual panel...")
    sbs_gif_frames = []
    pure_m3_frames = []
    pure_m1_frames = []
    montage_rows = []

    for item in frames_spec:
        idx = int(item['frame_idx'])
        is_k = (item['type'] == 'KEYFRAME')
        gt_u8 = gt_imgs_u8[idx]
        m1_u8 = reconstructed['M1_Original'][idx]
        m3_u8 = reconstructed['M3_Last'][idx]

        # Lưu ảnh clean M3 Last
        clean_img_path = os.path.join(clean_frames_dir, f"frame_{idx:02d}_{item['filename']}")
        cv2.imwrite(clean_img_path, m3_u8)

        pure_m3_frames.append(cv2.cvtColor(m3_u8, cv2.COLOR_GRAY2RGB))
        pure_m1_frames.append(cv2.cvtColor(m1_u8, cv2.COLOR_GRAY2RGB))

        if is_k:
            ov_u8 = gt_u8
            s1, s3 = "KEYFRAME", "KEYFRAME"
        else:
            ka, kb = int(item['ref_key_a']), int(item['ref_key_b'])
            ov_u8 = (0.5 * gt_imgs_u8[ka].astype(np.float32) + 0.5 * gt_imgs_u8[kb].astype(np.float32)).astype(np.uint8)
            fr_row = df_frames[df_frames['frame_idx'] == idx].iloc[0]
            s1 = f"PSNR:{fr_row['M1_Original_psnr']:.1f}|WCD:{fr_row['M1_Original_wcd_dog']:.2f}|Sk:{fr_row['M1_Original_skel_recall']*100:.0f}%|Dens:{fr_row['M1_Original_density_ratio']:.2f}"
            s3 = f"PSNR:{fr_row['M3_Last_psnr']:.1f}|WCD:{fr_row['M3_Last_wcd_dog']:.2f}|Sk:{fr_row['M3_Last_skel_recall']*100:.0f}%|Dens:{fr_row['M3_Last_density_ratio']:.2f}"

            # Bản đồ nhiệt sai số: [M1 Error Heatmap | M3 Last Error Heatmap]
            d1 = np.clip(np.abs(m1_u8.astype(np.float32) - gt_u8.astype(np.float32)) * 2.5, 0, 255).astype(np.uint8)
            d3 = np.clip(np.abs(m3_u8.astype(np.float32) - gt_u8.astype(np.float32)) * 2.5, 0, 255).astype(np.uint8)
            hm_pair = np.hstack([cv2.applyColorMap(d1, cv2.COLORMAP_JET), cv2.applyColorMap(d3, cv2.COLORMAP_JET)])
            hm_head = np.full((32, w * 2, 3), 24, dtype=np.uint8)
            cv2.putText(hm_head, f"M1 ORIG ERROR [MAE={fr_row['M1_Original_ie_mae']:.2f}]", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 150, 255), 2, cv2.LINE_AA)
            cv2.putText(hm_head, f"M3 LAST (EP 15) ERROR [MAE={fr_row['M3_Last_ie_mae']:.2f}]", (w + 12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 240, 120), 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(heat_dir, f"heatmap_{idx:02d}_{item['filename']}"), np.vstack([hm_head, hm_pair]))

        head_h = 34
        canvas = np.full((h + head_h, w * 4, 3), 24, dtype=np.uint8)
        cols = [
            (ov_u8, f"INPUT [{item['filename']}]", (210, 210, 210)),
            (m1_u8, f"M1 ORIG ({s1})", (100, 150, 255)),
            (m3_u8, f"M3 LAST EP15 ({s3})", (80, 240, 120)),
            (gt_u8, "GROUND TRUTH", (245, 245, 245))
        ]
        for ci, (arr, lbl, colr) in enumerate(cols):
            canvas[head_h:, ci * w:(ci + 1) * w] = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            cv2.putText(canvas, lbl, (ci * w + 10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, colr, 2, cv2.LINE_AA)
            if ci > 0:
                cv2.line(canvas, (ci * w, 0), (ci * w, h + head_h), (70, 70, 70), 2)

        out_png = os.path.join(sbs_dir, f"frame_{idx:02d}_{item['filename']}")
        cv2.imwrite(out_png, canvas)
        half_frame = cv2.resize(canvas, (w * 2, (h + head_h) // 2))
        sbs_gif_frames.append(half_frame[:, :, ::-1])
        if (not is_k) and (idx in {2, 5, 9, 12, 17, 21}):
            montage_rows.append(half_frame)

    if montage_rows:
        montage_path = os.path.join(OUT_DIR, "montage_6segments_M1_vs_M3_Last.png")
        cv2.imwrite(montage_path, np.vstack(montage_rows))
        print(f"[✓] Đã xuất Montage 6 segments: {montage_path}")

    # Xuất GIF và Video Side-by-Side đối đầu
    sbs_gif_path = os.path.join(OUT_DIR, "lineart_test_comparison_M1_vs_M3_Last.gif")
    imageio.mimsave(sbs_gif_path, sbs_gif_frames, duration=0.14, loop=0)
    print(f"[✓] Đã xuất GIF Side-by-Side đối đầu: {sbs_gif_path}")

    sbs_mp4_path = os.path.join(OUT_DIR, "lineart_test_comparison_M1_vs_M3_Last.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(sbs_mp4_path, fourcc, 8.0, (sbs_gif_frames[0].shape[1], sbs_gif_frames[0].shape[0]))
    for gf in sbs_gif_frames:
        vw.write(gf[:, :, ::-1])
    vw.release()
    print(f"[✓] Đã xuất MP4 Side-by-Side đối đầu: {sbs_mp4_path}")

    # Xuất GIF và Video chuyển động mượt mà của riêng M3 Last
    pure_m3_gif_path = os.path.join(OUT_DIR, "sequence_m3_last_pure_animation.gif")
    imageio.mimsave(pure_m3_gif_path, pure_m3_frames, duration=0.14, loop=0)
    print(f"[✓] Đã xuất GIF thuần hoạt hình M3 Last: {pure_m3_gif_path}")

    pure_m3_mp4_path = os.path.join(OUT_DIR, "sequence_m3_last_pure_animation.mp4")
    vw_m3 = cv2.VideoWriter(pure_m3_mp4_path, fourcc, 8.0, (w, h))
    for mf in pure_m3_frames:
        vw_m3.write(cv2.cvtColor(mf, cv2.COLOR_RGB2BGR))
    vw_m3.release()
    print(f"[✓] Đã xuất MP4 thuần hoạt hình M3 Last: {pure_m3_mp4_path}")

    print("\n🎉 HOÀN TẤT THỬ NGHIỆM ĐỐI CHUẨN M3 LAST!")


if __name__ == '__main__':
    main()
