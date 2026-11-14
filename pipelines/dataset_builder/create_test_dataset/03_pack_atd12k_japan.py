#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03_pack_atd12k_japan.py
Thu thập và chuẩn hóa tập dữ liệu ATD-12K Anime Nhật Bản (Japan_* và Hayao_*).

Quy trình:
1. Quét thư mục gốc ATD-12K tại E:\\Dataset\\datasets\\train_10k.
2. Lọc chính xác 4,543 thư mục `Japan_*` và 338 thư mục `Hayao_*` (bỏ qua hoàn toàn 5,119 thư mục `Disney_*`).
3. Đọc 3 frame (frame1.jpg, frame2.jpg, frame3.jpg) và resize từ 1080p về chuẩn 540p (960x540).
4. Lưu vào thư mục đệm `work/candidates_atd12k/` với nhãn cố định t = 0.5.
5. Xuất file manifest JSON.
"""

import os
import sys
import glob
import json
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ATD12K_TRAIN_DIR = r"E:\Dataset\datasets\train_10k"
TARGET_WIDTH = 960
TARGET_HEIGHT = 540


def process_single_folder(item):
    src_folder, out_dir, triplet_id = item
    f1_src = os.path.join(src_folder, "frame1.jpg")
    f2_src = os.path.join(src_folder, "frame2.jpg")
    f3_src = os.path.join(src_folder, "frame3.jpg")

    if not (os.path.exists(f1_src) and os.path.exists(f2_src) and os.path.exists(f3_src)):
        return None

    target_folder = os.path.join(out_dir, f"atd12k_{triplet_id:06d}")
    os.makedirs(target_folder, exist_ok=True)

    img1 = cv2.imread(f1_src)
    img2 = cv2.imread(f2_src)
    img3 = cv2.imread(f3_src)

    if img1 is None or img2 is None or img3 is None:
        return None

    # Resize về 540p
    img1_540p = cv2.resize(img1, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
    img2_540p = cv2.resize(img2, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
    img3_540p = cv2.resize(img3, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

    cv2.imwrite(os.path.join(target_folder, "frame1.jpg"), img1_540p, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(target_folder, "frame2.jpg"), img2_540p, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(target_folder, "frame3.jpg"), img3_540p, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return {
        'triplet_id': f"atd12k_{triplet_id:06d}",
        'source': os.path.basename(src_folder),
        't': 0.5,
        'delta_frames': 2
    }


def pack_atd12k_japan(src_dir=ATD12K_TRAIN_DIR, output_dir=None, max_samples=None):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "work", "candidates_atd12k")

    os.makedirs(output_dir, exist_ok=True)
    meta_file = os.path.join(output_dir, "atd12k_manifest.json")
    if os.path.exists(meta_file) and '--force' not in sys.argv:
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            if len(existing_data) > 4000:
                print(f"[✓] Đã tìm thấy {len(existing_data)} triplets ATD-12K sẵn có trong '{output_dir}'. Bỏ qua xử lý lại.")
                print("    (Nếu muốn chạy lại, hãy xóa thư mục hoặc thêm cờ --force)")
                return len(existing_data)
        except Exception:
            pass

    print("=" * 70)
    print("🚀 [03_PACK_ATD12K_JAPAN]: Thu thập Anime Nhật Bản từ ATD-12K")
    print(f"   - Thư mục nguồn: {src_dir}")
    print(f"   - Thư mục đích: {output_dir}")
    print("=" * 70)

    all_folders = [f.path for f in os.scandir(src_dir) if f.is_dir()]
    
    # Chỉ giữ lại Japan_* và Hayao_*
    selected_folders = [
        f for f in all_folders 
        if os.path.basename(f).startswith("Japan_") or os.path.basename(f).startswith("Hayao_")
    ]

    print(f"[*] Tổng số thư mục trong train_10k: {len(all_folders)}")
    print(f"[*] Số thư mục Anime Nhật Bản đạt chuẩn (Japan + Hayao): {len(selected_folders)}")
    print(f"[*] (Đã loại bỏ {len(all_folders) - len(selected_folders)} thư mục Disney)")

    if max_samples and max_samples < len(selected_folders):
        selected_folders = selected_folders[:max_samples]
        print(f"[*] Giới hạn xử lý {max_samples} mẫu.")

    tasks = [(fpath, output_dir, idx + 1) for idx, fpath in enumerate(selected_folders)]

    results = []
    print("[*] Đang resize 1080p -> 540p đa luồng (Multi-threading)...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        for res in tqdm(executor.map(process_single_folder, tasks), total=len(tasks), desc="Chuẩn hóa ATD-12K"):
            if res is not None:
                results.append(res)

    meta_file = os.path.join(output_dir, "atd12k_manifest.json")
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"[✓] Đã xử lý thành công {len(results)} Triplets Anime từ ATD-12K!")
    print(f"    - Metadata lưu tại: {meta_file}")
    print("=" * 70 + "\n")
    return len(results)


if __name__ == '__main__':
    pack_atd12k_japan()
