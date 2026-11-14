#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trích xuất 100 bộ triplet mẫu phân tầng (Stratified Sampling) từ Sakuga Golden Dataset
kèm các chỉ số định lượng: PSNR, SSIM, Stroke Recall, Ghost Pixels.
"""

import os
import sys
import shutil
import json
import pandas as pd
import numpy as np

# Đảm bảo in UTF-8 trên Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

METRICS_CSV = os.path.join(PROJECT_ROOT, 'continuous_benchmark', 'metrics', 'tps-inbetween-finetuned-v1.0', 'val_620_metrics.csv')
VAL_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'sakuga_golden_dataset_tps_ready', 'val')
OUTPUT_IMAGES_DIR = os.path.join(SCRIPT_DIR, 'images')
OUTPUT_CSV = os.path.join(SCRIPT_DIR, 'dataset_100_triplets.csv')
OUTPUT_JSON = os.path.join(SCRIPT_DIR, 'dataset_100_triplets.json')

def main():
    print("=" * 70)
    print("🚀 [GIAI ĐOẠN 1]: Trích xuất 100 Triplets & Chỉ số Định lượng")
    print("=" * 70)

    if not os.path.exists(METRICS_CSV):
        raise FileNotFoundError(f"Không tìm thấy file metrics: {METRICS_CSV}")

    df = pd.read_csv(METRICS_CSV)
    print(f"[*] Đã tải {len(df)} clips từ {METRICS_CSV}")

    # Lọc các clip có đủ 3 ảnh
    valid_mask = []
    for cid in df['clip_id']:
        p0 = os.path.join(VAL_DIR, cid, '000001.png')
        p1 = os.path.join(VAL_DIR, cid, '000002.png')
        p2 = os.path.join(VAL_DIR, cid, '000003.png')
        valid_mask.append(os.path.exists(p0) and os.path.exists(p1) and os.path.exists(p2))

    df = df[valid_mask].copy()
    print(f"[*] Số clip khả dụng có đầy đủ 3 khung hình: {len(df)}")

    # Tính score v1.0 hiện tại
    df['score_v10'] = (
        df['psnr'] +
        20.0 * df['ssim'] +
        30.0 * df['stroke_recall'] -
        0.0005 * df['ghost_pixels']
    )

    # Phân tầng (Stratified Sampling) theo biên độ chuyển động (mean_disp) và độ khó
    # Nhóm 1: Dễ / Chuyển động nhỏ (disp < 4.0) -> lấy 25 clips
    # Nhóm 2: Vừa / Chuyển động điển hình (4.0 <= disp < 10.0) -> lấy 50 clips
    # Nhóm 3: Khó / Chuyển động lớn (disp >= 10.0) -> lấy 25 clips
    np.random.seed(42)

    df_easy = df[df['mean_disp'] < 4.0]
    df_med = df[(df['mean_disp'] >= 4.0) & (df['mean_disp'] < 10.0)]
    df_hard = df[df['mean_disp'] >= 10.0]

    print(f"   - Phân bố gốc: Nhóm dễ (disp < 4px): {len(df_easy)} clips")
    print(f"   - Phân bố gốc: Nhóm vừa (4px <= disp < 10px): {len(df_med)} clips")
    print(f"   - Phân bố gốc: Nhóm khó (disp >= 10px): {len(df_hard)} clips")

    s_easy = df_easy.sample(n=min(25, len(df_easy)), random_state=42)
    s_med = df_med.sample(n=min(50, len(df_med)), random_state=42)
    s_hard = df_hard.sample(n=min(25, len(df_hard)), random_state=42)

    sampled_df = pd.concat([s_easy, s_med, s_hard]).reset_index(drop=True)
    # Xáo trộn ngẫu nhiên để khi người dùng đánh giá sẽ xen kẽ đa dạng các độ khó
    sampled_df = sampled_df.sample(frac=1.0, random_state=123).reset_index(drop=True)

    print(f"\n[+] Đã lấy mẫu thành công {len(sampled_df)} triplets:")
    print(f"    - Điểm Score v1.0 trung bình: {sampled_df['score_v10'].mean():.2f} (Min: {sampled_df['score_v10'].min():.2f}, Max: {sampled_df['score_v10'].max():.2f})")
    print(f"    - PSNR trung bình: {sampled_df['psnr'].mean():.2f} dB")
    print(f"    - SSIM trung bình: {sampled_df['ssim'].mean():.4f}")
    print(f"    - Stroke Recall trung bình: {sampled_df['stroke_recall'].mean()*100:.1f}%")
    print(f"    - Ghost Pixels trung bình: {sampled_df['ghost_pixels'].mean():.0f} px")

    # Sao chép ảnh vào thư mục images của ứng dụng
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    json_data = []

    print("\n[*] Đang đồng bộ ảnh vào thư mục web app...")
    for idx, row in sampled_df.iterrows():
        cid = row['clip_id']
        clip_out_dir = os.path.join(OUTPUT_IMAGES_DIR, cid)
        os.makedirs(clip_out_dir, exist_ok=True)

        src0 = os.path.join(VAL_DIR, cid, '000001.png')
        src1 = os.path.join(VAL_DIR, cid, '000002.png')
        src2 = os.path.join(VAL_DIR, cid, '000003.png')

        dst0 = os.path.join(clip_out_dir, '000001.png')
        dst1 = os.path.join(clip_out_dir, '000002.png')
        dst2 = os.path.join(clip_out_dir, '000003.png')

        if not os.path.exists(dst0):
            shutil.copy2(src0, dst0)
        if not os.path.exists(dst1):
            shutil.copy2(src1, dst1)
        if not os.path.exists(dst2):
            shutil.copy2(src2, dst2)

        item = {
            'index': int(idx + 1),
            'clip_id': cid,
            't': float(row['t']),
            'mean_disp': round(float(row['mean_disp']), 2),
            'inliers': int(row['inliers']),
            'psnr': round(float(row['psnr']), 2),
            'ssim': round(float(row['ssim']), 4),
            'stroke_recall': round(float(row['stroke_recall']), 4),
            'ghost_pixels': int(row['ghost_pixels']),
            'score_v10': round(float(row['score_v10']), 2),
            'frame0_url': f"images/{cid}/000001.png",
            'frame_gt_url': f"images/{cid}/000002.png",
            'frame1_url': f"images/{cid}/000003.png"
        }
        json_data.append(item)

    # Xuất CSV
    out_df = pd.DataFrame(json_data)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"[✓] Đã xuất CSV: {OUTPUT_CSV} ({len(out_df)} dòng)")

    # Xuất JSON phục vụ nhúng trực tiếp vào Web App
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"[✓] Đã xuất JSON: {OUTPUT_JSON}")
    print("\n🎉 Hoàn thành Giai đoạn 1 chuẩn bị dữ liệu!")

if __name__ == '__main__':
    main()
