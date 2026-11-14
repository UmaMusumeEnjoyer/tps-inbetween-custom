#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script So Sánh Đối Đầu Phục Dựng Phân Cảnh Chuyển Động Liên Tục V1.0 vs V1.1 (ASF-GR Engine)
trên các Triplets trích xuất từ Sakuga Golden Dataset (không dùng ChayGo và TestImage).

Tính năng:
- Nạp Checkpoint V1.0 (checkpoints/experiments/v1.0/model_tps_v1.0_best.pt)
- Nạp Checkpoint V1.1 (checkpoints/experiments/v1.1/model_tps_v1.1_best.pt)
- Phục dựng inbetweens trên:
  1. sakuga_composite_sequence (Master Reel 18 frames)
  2. Các clips riêng lẻ: clip_000008, clip_000010, clip_000018, clip_000064, clip_003046, clip_003050
- Kết xuất Video 4-Panel Side-by-Side:
  [Ground Truth | Coarse TPS Warp | Finetuned V1.0 | Finetuned V1.1] kèm HUD realtime
- Kết xuất preview GIF (0.5x slow motion, 12 FPS)
- Kết xuất Difference Heatmaps (Sai số V1.0 vs V1.1)
- Xuất CSV đo lường chi tiết từng frame và Báo cáo so sánh Markdown tổng hợp.
"""

import os
import sys
import time
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

# Import kiến trúc model chuẩn đã dùng trong benchmark
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from run_tps_benchmark_v10 import (
    TPSInbetweenFinetune_v10,
    compute_psnr,
    compute_ssim,
    compute_ghost_pixels,
    extract_sift_keypoints
)


def draw_hud(panel_img, title, frame_idx, total_frames, is_keyframe, psnr_val, ssim_val, ghost_val, speed_label="0.5x"):
    h, w = panel_img.shape[:2]
    out = panel_img.copy()
    if len(out.shape) == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    # Header Bar
    cv2.rectangle(out, (0, 0), (w, 36), (25, 25, 25), -1)
    cv2.putText(out, f"{title}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Footer Bar
    cv2.rectangle(out, (0, h - 34), (w, h), (25, 25, 25), -1)
    tag_color = (0, 200, 255) if is_keyframe else (0, 255, 100)
    tag_text = "[KEY]" if is_keyframe else "[INBETWEEN]"
    cv2.putText(out, tag_text, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.50, tag_color, 2)

    hud_stat = f"{frame_idx:02d}/{total_frames:02d} | {psnr_val:.1f}dB | {ssim_val:.3f} | {ghost_val}px"
    cv2.putText(out, hud_stat, (90, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    return out


def compare_single_scene(scene_name, frames_dir, model_v10, model_v11, device, output_base_dir, fps=12.0):
    scene_output_dir = os.path.join(output_base_dir, scene_name)
    os.makedirs(scene_output_dir, exist_ok=True)
    frames_v10_dir = os.path.join(scene_output_dir, "frames_v10")
    frames_v11_dir = os.path.join(scene_output_dir, "frames_v11")
    frames_gt_dir = os.path.join(scene_output_dir, "frames_gt")
    heatmaps_dir = os.path.join(scene_output_dir, "difference_heatmaps")
    os.makedirs(frames_v10_dir, exist_ok=True)
    os.makedirs(frames_v11_dir, exist_ok=True)
    os.makedirs(frames_gt_dir, exist_ok=True)
    os.makedirs(heatmaps_dir, exist_ok=True)

    print("\n" + "=" * 85)
    print(f"🎬 ĐỐI ĐẦU PHỤC DỰNG SAKUGA: {scene_name} (V1.0 vs V1.1)")
    print(f"   - Nguồn: {frames_dir}")
    print(f"   - Đích : {scene_output_dir}")
    print("=" * 85)

    img_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.png', '.jpg'))])
    raw_frames = []
    for fn in img_files:
        im = cv2.imread(os.path.join(frames_dir, fn), cv2.IMREAD_GRAYSCALE)
        if im is not None:
            raw_frames.append(im)

    total_frames = len(raw_frames)
    print(f"[✓] Đã nạp {total_frames} frames Sakuga liên tục!")
    assert total_frames >= 3, "Cần tối thiểu 3 frames để phục dựng!"

    h, w = raw_frames[0].shape
    pad_h = ((h + 31) // 32) * 32
    pad_w = ((w + 31) // 32) * 32

    rec_v10_frames, rec_v11_frames, coarse_frames = [], [], []
    records = []

    for i in range(total_frames):
        gt_f = raw_frames[i]
        cv2.imwrite(os.path.join(frames_gt_dir, f"gt_{i:04d}.png"), gt_f)

        if i % 2 == 0:
            # Keyframe gốc
            rec_v10_frames.append(gt_f)
            rec_v11_frames.append(gt_f)
            coarse_frames.append(gt_f)
            cv2.imwrite(os.path.join(frames_v10_dir, f"rec_v10_{i:04d}.png"), gt_f)
            cv2.imwrite(os.path.join(frames_v11_dir, f"rec_v11_{i:04d}.png"), gt_f)

            rec = {
                'frame_idx': i,
                'frame_type': 'KEYFRAME',
                'psnr_v10': 50.0,
                'ssim_v10': 1.0,
                'ghost_v10': 0,
                'infer_ms_v10': 0.0,
                'psnr_v11': 50.0,
                'ssim_v11': 1.0,
                'ghost_v11': 0,
                'infer_ms_v11': 0.0,
                'psnr_gain_v11_vs_v10': 0.0,
                'ghost_diff_v11_vs_v10': 0
            }
            records.append(rec)
        else:
            # Inbetween frame t=0.5
            prev_idx = i - 1
            next_idx = min(total_frames - 1, i + 1)
            im0_raw = raw_frames[prev_idx]
            im1_raw = raw_frames[next_idx]

            kps_raw = extract_sift_keypoints(im0_raw, im1_raw, max_pts=80)
            num_matches = kps_raw.shape[1]

            f0 = 1.0 - (im0_raw.astype(np.float32) / 255.0)
            f1 = 1.0 - (im1_raw.astype(np.float32) / 255.0)

            f0_pad = np.pad(f0, ((0, pad_h - h), (0, pad_w - w)), mode='constant')
            f1_pad = np.pad(f1, ((0, pad_h - h), (0, pad_w - w)), mode='constant')

            kps_pad = kps_raw.copy()
            if num_matches > 0:
                kps_pad[0, :, 0] = (kps_pad[0, :, 0] * w) / pad_w
                kps_pad[0, :, 1] = (kps_pad[0, :, 1] * h) / pad_h
                kps_pad[1, :, 0] = (kps_pad[1, :, 0] * w) / pad_w
                kps_pad[1, :, 1] = (kps_pad[1, :, 1] * h) / pad_h

            t0_dev = torch.from_numpy(f0_pad).unsqueeze(0).unsqueeze(0).float().to(device)
            t1_dev = torch.from_numpy(f1_pad).unsqueeze(0).unsqueeze(0).float().to(device)
            t_tensor = torch.tensor([[0.5]], dtype=torch.float32).view(1, 1, 1, 1).to(device)

            # Suy luận V1.0
            t_start = time.time()
            with torch.no_grad():
                pred_v10, coarse_interp = model_v10(t0_dev, t1_dev, t_tensor, [kps_pad], max_kps=80, return_coarse=True)
            ms_v10 = (time.time() - t_start) * 1000.0

            # Suy luận V1.1
            t_start = time.time()
            with torch.no_grad():
                pred_v11, _ = model_v11(t0_dev, t1_dev, t_tensor, [kps_pad], max_kps=80, return_coarse=True)
            ms_v11 = (time.time() - t_start) * 1000.0

            p_v10_np = ((1.0 - pred_v10[:, :, :h, :w]) * 255.0).clamp(0, 255).byte().cpu().numpy()[0, 0]
            p_v11_np = ((1.0 - pred_v11[:, :, :h, :w]) * 255.0).clamp(0, 255).byte().cpu().numpy()[0, 0]
            p_coarse_np = ((1.0 - coarse_interp[:, :, :h, :w]) * 255.0).clamp(0, 255).byte().cpu().numpy()[0, 0]

            rec_v10_frames.append(p_v10_np)
            rec_v11_frames.append(p_v11_np)
            coarse_frames.append(p_coarse_np)

            cv2.imwrite(os.path.join(frames_v10_dir, f"rec_v10_{i:04d}.png"), p_v10_np)
            cv2.imwrite(os.path.join(frames_v11_dir, f"rec_v11_{i:04d}.png"), p_v11_np)

            psnr10 = compute_psnr(p_v10_np, gt_f)
            ssim10 = compute_ssim(p_v10_np, gt_f)
            ghost10 = compute_ghost_pixels(p_v10_np, gt_f)

            psnr11 = compute_psnr(p_v11_np, gt_f)
            ssim11 = compute_ssim(p_v11_np, gt_f)
            ghost11 = compute_ghost_pixels(p_v11_np, gt_f)

            # Difference Heatmap (V1.1 vs V1.0)
            diff_v11_v10 = cv2.absdiff(p_v11_np, p_v10_np)
            diff_color = cv2.applyColorMap(diff_v11_v10 * 3, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(heatmaps_dir, f"diff_v11_vs_v10_{i:04d}.png"), diff_color)

            rec = {
                'frame_idx': i,
                'frame_type': 'INBETWEEN',
                'psnr_v10': round(psnr10, 4),
                'ssim_v10': round(ssim10, 4),
                'ghost_v10': ghost10,
                'infer_ms_v10': round(ms_v10, 2),
                'psnr_v11': round(psnr11, 4),
                'ssim_v11': round(ssim11, 4),
                'ghost_v11': ghost11,
                'infer_ms_v11': round(ms_v11, 2),
                'psnr_gain_v11_vs_v10': round(psnr11 - psnr10, 4),
                'ghost_diff_v11_vs_v10': ghost11 - ghost10
            }
            records.append(rec)
            print(f"   [✓] Frame {i:02d} (Inbetween): V1.0 PSNR={psnr10:.2f}dB, Ghost={ghost10} | V1.1 PSNR={psnr11:.2f}dB, Ghost={ghost11} | Gain={psnr11-psnr10:+.2f}dB")

    df_records = pd.DataFrame(records)
    csv_path = os.path.join(scene_output_dir, "frame_metrics_comparison.csv")
    df_records.to_csv(csv_path, index=False)

    # --------------------------------------------------------------------------
    # KẾT XUẤT VIDEO 4-PANEL: GT | COARSE | V1.0 | V1.1
    # --------------------------------------------------------------------------
    print("\n[*] Đang kết xuất video 4 cột đồng bộ (12 FPS, 0.5x Slow Motion)...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vid_4way_path = os.path.join(scene_output_dir, "01_four_way_comparison_hud.mp4")
    out_4way = cv2.VideoWriter(vid_4way_path, fourcc, fps, (w * 4, h), isColor=True)
    gif_frames = []

    for idx in range(total_frames):
        gt_f = raw_frames[idx]
        crs_f = coarse_frames[idx]
        v10_f = rec_v10_frames[idx]
        v11_f = rec_v11_frames[idx]
        info = records[idx]
        is_kf = (info['frame_type'] == 'KEYFRAME')

        p_gt = draw_hud(gt_f, "Ground Truth", idx, total_frames, is_kf, 50.0, 1.0, 0)
        p_crs = draw_hud(crs_f, "Coarse TPS Warp", idx, total_frames, is_kf, compute_psnr(crs_f, gt_f), compute_ssim(crs_f, gt_f), compute_ghost_pixels(crs_f, gt_f))
        p_v10 = draw_hud(v10_f, "Finetuned V1.0", idx, total_frames, is_kf, info['psnr_v10'], info['ssim_v10'], info['ghost_v10'])
        p_v11 = draw_hud(v11_f, "Finetuned V1.1", idx, total_frames, is_kf, info['psnr_v11'], info['ssim_v11'], info['ghost_v11'])

        four_panel = np.hstack([p_gt, p_crs, p_v10, p_v11])
        out_4way.write(four_panel)

        thumb = cv2.resize(four_panel, (w * 2, h // 2))
        gif_frames.append(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))

    out_4way.release()

    # GIF preview
    gif_path = os.path.join(scene_output_dir, "02_comparison_preview.gif")
    try:
        import imageio
        imageio.mimsave(gif_path, gif_frames, fps=fps, loop=0)
        print(f"[✓] Đã lưu GIF preview tại: {gif_path}")
    except Exception as e:
        print(f"⚠️ Không tạo được GIF: {e}")

    # Tính toán chỉ số tổng hợp
    inb_df = df_records[df_records['frame_type'] == 'INBETWEEN']
    summary = {
        'scene': scene_name,
        'total_frames': total_frames,
        'inbetween_frames': len(inb_df),
        'psnr_v10_mean': float(inb_df['psnr_v10'].mean()),
        'psnr_v11_mean': float(inb_df['psnr_v11'].mean()),
        'psnr_gain_mean': float(inb_df['psnr_gain_v11_vs_v10'].mean()),
        'ssim_v10_mean': float(inb_df['ssim_v10'].mean()),
        'ssim_v11_mean': float(inb_df['ssim_v11'].mean()),
        'ghost_v10_mean': float(inb_df['ghost_v10'].mean()),
        'ghost_v11_mean': float(inb_df['ghost_v11'].mean()),
        'tsi_v10': float(inb_df['psnr_v10'].std()) if len(inb_df) > 1 else 0.0,
        'tsi_v11': float(inb_df['psnr_v11'].std()) if len(inb_df) > 1 else 0.0,
        'latency_v10_ms': float(inb_df['infer_ms_v10'].mean()),
        'latency_v11_ms': float(inb_df['infer_ms_v11'].mean()),
        'video_4way': vid_4way_path,
        'gif_preview': gif_path
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Chạy đối chuẩn phục dựng chuyển động liên tục V1.0 vs V1.1 trên Sakuga Triplets")
    parser.add_argument('--v10_model', default='checkpoints/experiments/v1.0/model_tps_v1.0_best.pt', help='Đường dẫn checkpoint v1.0')
    parser.add_argument('--v11_model', default='checkpoints/experiments/v1.1/model_tps_v1.1_best.pt', help='Đường dẫn checkpoint v1.1')
    parser.add_argument('--triplets_dir', default='data/test_sets/sakuga_continuous_triplets', help='Thư mục chứa các triplets trích xuất')
    parser.add_argument('--output_dir', default='continuous_benchmark/visual_audit/comparison_v1.0_vs_v1.1/continuous_reconstruction', help='Thư mục xuất kết quả')
    parser.add_argument('--device', default='cuda:0', help='Thiết bị tính toán')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Thiết bị suy luận: {device}")
    if device.type == 'cuda':
        print(f"    GPU: {torch.cuda.get_device_name(0)}")

    print(f"[*] Đang tải mô hình V1.0: {args.v10_model} ...")
    model_v10 = TPSInbetweenFinetune_v10().to(device)
    sd10 = torch.load(args.v10_model, map_location=device)
    model_v10.load_state_dict(sd10.get('state_dict', sd10), strict=True)
    model_v10.eval()

    print(f"[*] Đang tải mô hình V1.1: {args.v11_model} ...")
    model_v11 = TPSInbetweenFinetune_v10().to(device)
    sd11 = torch.load(args.v11_model, map_location=device)
    model_v11.load_state_dict(sd11.get('state_dict', sd11), strict=True)
    model_v11.eval()
    print("[✓] Cả hai mô hình V1.0 và V1.1 đã sẵn sàng đối đầu!")

    scenes = ['sakuga_composite_sequence', 'clip_000008', 'clip_000010', 'clip_000018', 'clip_000064', 'clip_003046', 'clip_003050']
    summary_list = []

    for sc in scenes:
        sc_dir = os.path.join(args.triplets_dir, sc)
        if os.path.exists(sc_dir):
            res = compare_single_scene(sc, sc_dir, model_v10, model_v11, device, args.output_dir)
            summary_list.append(res)

    # Viết Báo Cáo Markdown Đối Chuẩn Nghiệm Thu
    report_path = os.path.join(args.output_dir, "CONTINUOUS_RECONSTRUCTION_COMPARISON_REPORT.md")
    report_content = f"""# BÁO CÁO ĐỐI ĐẦU PHỤC DỰNG CHUYỂN ĐỘNG LIÊN TỤC V1.0 VS V1.1 (SAKUGA TRIPLETS)
> **Ngày thực hiện:** {time.strftime('%d/%m/%Y %H:%M:%S')}  
> **Tập Dữ Liệu:** Chuỗi Triplets Trích Xuất Từ Sakuga Golden Dataset (Không sử dụng ChayGo / TestImage)  
> **Model V1.0:** `{os.path.basename(args.v10_model)}`  
> **Model V1.1:** `{os.path.basename(args.v11_model)}`  
> **Tốc độ kết xuất:** **0.5x Slow Motion (12 FPS)**  

---

## 1. BẢNG TỔNG HỢP SO SÁNH ĐỐI ĐẦU TOÀN DIỆN

| Phân Cảnh Thử Nghiệm | Tổng Frames | PSNR V1.0 | **PSNR V1.1** | **$\Delta$ PSNR** | SSIM V1.0 | **SSIM V1.1** | Ghost V1.0 | **Ghost V1.1** | TSI V1.0 | **TSI V1.1** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for s in summary_list:
        report_content += f"| **{s['scene']}** | {s['total_frames']} | {s['psnr_v10_mean']:.2f} dB | **{s['psnr_v11_mean']:.2f} dB** | **{s['psnr_gain_mean']:+.2f} dB** | {s['ssim_v10_mean']:.4f} | **{s['ssim_v11_mean']:.4f}** | {s['ghost_v10_mean']:.0f} px | **{s['ghost_v11_mean']:.0f} px** | ±{s['tsi_v10']:.2f} dB | **±{s['tsi_v11']:.2f} dB** |\n"

    report_content += f"""
---

## 2. PHÂN TÍCH KỸ THUẬT CHUYÊN SÂU

1. **Chất Lượng Tái Tạo Pixel (PSNR & SSIM):**
   - Đánh giá khả năng bám sát đường nét nguyên bản của họa sĩ Sakugabooru giữa hai thế hệ checkpoint.
   - So sánh chi tiết trên chuỗi chuyển động liên tục `sakuga_composite_sequence`.
2. **Khả Năng Khử Bóng Ma (Ghosting Eradication):**
   - Đánh giá lượng Ghost Pixels dư thừa trên nền trắng khi thực hiện nội suy bước nhảy liên tiếp.
3. **Độ Ổn Định Thời Gian (Temporal Stability Index - TSI):**
   - Độ lệch chuẩn $\\sigma_{{\\text{{PSNR}}}}$ giữa các inbetweens chứng minh mô hình không bị nhấp nháy chất lượng hình ảnh qua từng frame.

---

## 3. DANH MỤC VIDEO & SẢN PHẨM TRỰC QUAN

Toàn bộ video 4 cột đồng bộ (`01_four_way_comparison_hud.mp4`) và preview GIF (`02_comparison_preview.gif`) cùng heatmaps đã được lưu trữ trong từng thư mục con tại:
`{args.output_dir}/`.
"""
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    # Master CSV
    summary_df = pd.DataFrame(summary_list)
    summary_df.to_csv(os.path.join(args.output_dir, "summary_continuous_comparison.csv"), index=False)

    print("\n" + "=" * 85)
    print("🎉 HOÀN THÀNH TOÀN DIỆN SO SÁNH PHỤC DỰNG LIÊN TỤC V1.0 VS V1.1!")
    print(f"   Báo cáo tổng hợp: {report_path}")
    print("=" * 85)

if __name__ == '__main__':
    main()
