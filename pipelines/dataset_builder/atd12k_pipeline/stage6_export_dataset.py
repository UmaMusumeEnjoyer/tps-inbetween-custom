#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Giai đoạn 6: Xuất bản bộ dữ liệu hoàn chỉnh tương thích với `ml240.py` của TPS-Inbetween.
- Cấu trúc: train/clip_001, val/clip_001, matches/clip_001, triplets.csv
- Lưu đầy đủ thông tin: đường dẫn ảnh, nhãn t, độ dời mean_disp, số inliers.
"""

import os
import sys
import json
import shutil
import csv
from config import (
    SCREENED_TRIPLETS_FILE, LINEART_CACHE_DIR, MATCHES_CACHE_DIR,
    OUTPUT_DATASET_DIR, OUTPUT_TRAIN_DIR, OUTPUT_VAL_DIR,
    OUTPUT_MATCHES_DIR, TRIPLETS_CSV
)

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def export_dataset(screened_file=SCREENED_TRIPLETS_FILE,
                   lineart_dir=LINEART_CACHE_DIR,
                   matches_dir=MATCHES_CACHE_DIR,
                   val_ratio=0.15):
    print("=" * 70)
    print("🚀 [GIAI ĐOẠN 6]: Đóng gói và xuất bản Dataset chuẩn ML240")
    print(f"   - Thư mục đích: {OUTPUT_DATASET_DIR}")
    print(f"   - Tỷ lệ Validation: {val_ratio * 100:.0f}%")
    print("=" * 70)

    if not os.path.exists(screened_file):
        raise FileNotFoundError(f"Không tìm thấy file: {screened_file}")

    with open(screened_file, 'r', encoding='utf-8') as f:
        triplets = json.load(f)

    if len(triplets) == 0:
        print("[!] Không có triplet nào để xuất bản.")
        return 0

    os.makedirs(OUTPUT_TRAIN_DIR, exist_ok=True)
    os.makedirs(OUTPUT_VAL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MATCHES_DIR, exist_ok=True)

    # Chia train/val theo phân đoạn (hoặc theo chỉ số)
    num_triplets = len(triplets)
    val_count = max(1, int(num_triplets * val_ratio))
    val_indices = set(range(num_triplets - val_count, num_triplets))

    csv_rows = []

    for idx, tr in enumerate(triplets):
        split = "val" if idx in val_indices else "train"
        target_clip_dir = OUTPUT_VAL_DIR if split == "val" else OUTPUT_TRAIN_DIR

        # 1. Copy và chuẩn hóa đường dẫn ảnh lineart
        f0_base = os.path.splitext(tr["i0_file"])[0] + "_smooth.png"
        fgt_base = os.path.splitext(tr["igt_file"])[0] + "_smooth.png"
        f1_base = os.path.splitext(tr["i1_file"])[0] + "_smooth.png"

        src0 = os.path.join(lineart_dir, f0_base)
        src_gt = os.path.join(lineart_dir, fgt_base)
        src1 = os.path.join(lineart_dir, f1_base)

        dst0 = os.path.join(target_clip_dir, f0_base)
        dst_gt = os.path.join(target_clip_dir, fgt_base)
        dst1 = os.path.join(target_clip_dir, f1_base)

        if os.path.exists(src0) and not os.path.exists(dst0):
            shutil.copy2(src0, dst0)
        if os.path.exists(src_gt) and not os.path.exists(dst_gt):
            shutil.copy2(src_gt, dst_gt)
        if os.path.exists(src1) and not os.path.exists(dst1):
            shutil.copy2(src1, dst1)

        # 2. Copy file matching keypoints .npy
        match_src = tr["match_npy_path"]
        match_dst_name = f"{idx}.npy"
        match_dst = os.path.join(OUTPUT_MATCHES_DIR, match_dst_name)
        if os.path.exists(match_src):
            shutil.copy2(match_src, match_dst)

        # 3. Ghi thông tin vào triplets.csv
        rel_i0 = os.path.relpath(dst0, OUTPUT_DATASET_DIR).replace('\\', '/')
        rel_igt = os.path.relpath(dst_gt, OUTPUT_DATASET_DIR).replace('\\', '/')
        rel_i1 = os.path.relpath(dst1, OUTPUT_DATASET_DIR).replace('\\', '/')
        rel_match = os.path.relpath(match_dst, OUTPUT_DATASET_DIR).replace('\\', '/')

        csv_rows.append({
            "triplet_id": idx,
            "split": split,
            "i0_path": rel_i0,
            "igt_path": rel_igt,
            "i1_path": rel_i1,
            "match_npy": rel_match,
            "t": tr["t"],
            "delta_frames": tr["delta_frames"],
            "mean_disp": tr["mean_disp"],
            "max_disp": tr["max_disp"],
            "inliers": tr["inliers"]
        })

    # Lưu file triplets.csv
    with open(TRIPLETS_CSV, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            "triplet_id", "split", "i0_path", "igt_path", "i1_path",
            "match_npy", "t", "delta_frames", "mean_disp", "max_disp", "inliers"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    train_c = num_triplets - val_count
    print("\n" + "=" * 70)
    print("🎉 XUẤT BẢN DATASET HOÀN TẤT:")
    print(f"   - Tổng số triplet đóng gói: {num_triplets}")
    print(f"   - Tập Train: {train_c} triplets (trong {OUTPUT_TRAIN_DIR})")
    print(f"   - Tập Val: {val_count} triplets (trong {OUTPUT_VAL_DIR})")
    print(f"   - Thư mục Matches: {OUTPUT_MATCHES_DIR} ({len(csv_rows)} file .npy)")
    print(f"   - Bảng nhãn: {TRIPLETS_CSV}")
    print("=" * 70 + "\n")

    return num_triplets


if __name__ == '__main__':
    export_dataset()
