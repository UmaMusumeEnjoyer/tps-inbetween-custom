#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04_merge_and_build_kaggle_input.py
Hợp nhất dữ liệu Candidate Triplets từ AnimeGoc và ATD-12K Anime,
đóng gói thành 1 file ZIP duy nhất sẵn sàng upload lên Kaggle Dataset.

Quy trình:
1. Đọc metadata từ `work/candidates_animegoc/animegoc_candidates_manifest.json`.
2. Đọc metadata từ `work/candidates_atd12k/atd12k_manifest.json`.
3. Chuẩn hóa tên thư mục thành `triplet_000001`, `triplet_000002`,...
4. Tạo `dataset_manifest.json` thống nhất.
5. Nén toàn bộ thư mục thành `candidate_triplets_anime_540p.zip`.
"""

import os
import sys
import json
import shutil
import zipfile
from tqdm import tqdm

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def merge_and_zip_dataset(base_dir=None):
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    animegoc_dir = os.path.join(base_dir, "work", "candidates_animegoc")
    atd12k_dir = os.path.join(base_dir, "work", "candidates_atd12k")
    merged_output_dir = os.path.join(base_dir, "work", "merged_candidates")
    zip_output_path = os.path.join(base_dir, "candidate_triplets_anime_540p.zip")

    os.makedirs(merged_output_dir, exist_ok=True)

    animegoc_meta_file = os.path.join(animegoc_dir, "animegoc_candidates_manifest.json")
    atd12k_meta_file = os.path.join(atd12k_dir, "atd12k_manifest.json")

    animegoc_list = []
    if os.path.exists(animegoc_meta_file):
        with open(animegoc_meta_file, 'r', encoding='utf-8') as f:
            animegoc_list = json.load(f)

    atd12k_list = []
    if os.path.exists(atd12k_meta_file):
        with open(atd12k_meta_file, 'r', encoding='utf-8') as f:
            atd12k_list = json.load(f)

    print("=" * 70)
    print("🚀 [04_MERGE_AND_BUILD_KAGGLE_INPUT]: Gom chung và Đóng gói Dataset")
    print(f"   - Số lượng Candidate từ AnimeGoc (đa mốc t): {len(animegoc_list)}")
    print(f"   - Số lượng Candidate từ ATD-12K (t = 0.5):   {len(atd12k_list)}")
    print(f"   - Tổng số Candidate Triplets:                {len(animegoc_list) + len(atd12k_list)}")
    print("=" * 70)

    if len(animegoc_list) == 0 and len(atd12k_list) == 0:
        print("[!] Chưa có dữ liệu candidate nào. Hãy chạy script 02 và 03 trước!")
        return

    merged_manifest = []
    current_id = 0

    triplets_dir = os.path.join(merged_output_dir, "triplets")
    os.makedirs(triplets_dir, exist_ok=True)

    # 1. Sao chép và đánh số từ AnimeGoc
    print("\n[*] Đang hợp nhất Candidate từ AnimeGoc...")
    for item in tqdm(animegoc_list, desc="AnimeGoc"):
        current_id += 1
        new_name = f"triplet_{current_id:06d}"
        src_path = os.path.join(animegoc_dir, item['triplet_id'])
        dst_path = os.path.join(triplets_dir, new_name)
        
        if os.path.exists(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            merged_manifest.append({
                'triplet_id': new_name,
                'source': item.get('source', 'AnimeGoc'),
                't': item.get('t', 0.5),
                'delta_frames': item.get('delta_frames', 2),
                'frame1': f"triplets/{new_name}/frame1.jpg",
                'frame2': f"triplets/{new_name}/frame2.jpg",
                'frame3': f"triplets/{new_name}/frame3.jpg"
            })

    # 2. Sao chép và đánh số từ ATD-12K
    print("\n[*] Đang hợp nhất Candidate từ ATD-12K...")
    for item in tqdm(atd12k_list, desc="ATD-12K"):
        current_id += 1
        new_name = f"triplet_{current_id:06d}"
        src_path = os.path.join(atd12k_dir, item['triplet_id'])
        dst_path = os.path.join(triplets_dir, new_name)

        if os.path.exists(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            merged_manifest.append({
                'triplet_id': new_name,
                'source': item.get('source', 'ATD12K'),
                't': item.get('t', 0.5),
                'delta_frames': item.get('delta_frames', 2),
                'frame1': f"triplets/{new_name}/frame1.jpg",
                'frame2': f"triplets/{new_name}/frame2.jpg",
                'frame3': f"triplets/{new_name}/frame3.jpg"
            })

    # Lưu file manifest chung
    manifest_path = os.path.join(merged_output_dir, "dataset_manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(merged_manifest, f, indent=2, ensure_ascii=False)

    print(f"\n[✓] Đã tạo manifest tổng hợp với {len(merged_manifest)} triplets.")

    # 3. Nén thành file ZIP
    print(f"\n[*] Đang nén thành file ZIP: {zip_output_path} (Sẽ mất vài phút)...")
    with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(manifest_path, arcname="dataset_manifest.json")
        for root, dirs, files in os.walk(triplets_dir):
            for file in files:
                full_p = os.path.join(root, file)
                rel_p = os.path.relpath(full_p, merged_output_dir)
                zipf.write(full_p, arcname=rel_p)

    file_size_gb = os.path.getsize(zip_output_path) / (1024 ** 3)
    print("\n" + "=" * 70)
    print(f"🎉 HOÀN TẤT ĐÓNG GÓI DATASET CHO KAGGLE!")
    print(f"   - File nén: {zip_output_path}")
    print(f"   - Dung lượng: {file_size_gb:.2f} GB")
    print(f"   - Tổng số triplets: {len(merged_manifest)}")
    print(f"   - Bước tiếp theo: Upload file ZIP này lên Kaggle Datasets để chạy Notebook!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    merge_and_zip_dataset()
