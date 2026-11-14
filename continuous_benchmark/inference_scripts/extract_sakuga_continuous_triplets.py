#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trích xuất các triplets đại diện từ sakuga_golden_dataset_tps_ready
để phục vụ bài test Continuous Animation Reconstruction (thay thế ChayGo và TestImage).
"""

import os
import sys
import shutil
import pandas as pd
import cv2

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

DATA_ROOT = 'data/processed/sakuga_golden_dataset_tps_ready_v1_patch2'
TRIPLETS_CSV = 'data/processed/sakuga_golden_dataset_tps_ready_v1_patch2/triplets.csv'
TARGET_DIR = 'data/test_sets/sakuga_continuous_triplets'

# Chọn 6 triplets validation đa dạng:
# - clip_000008: Chuyển động cơ thể biên độ lớn (delta_f=8, disp=9.05)
# - clip_000010: Chuyển động nhanh nét dày (delta_f=2, disp=7.28)
# - clip_000018: Sakuga hành động đặc trưng (delta_f=4, disp=8.49)
# - clip_000064: Nét vẽ phức tạp, nhiều chi tiết (delta_f=8, disp=7.64)
# - clip_003046: Hoạt họa tốc độ cao (delta_f=6, disp=10.21)
# - clip_003050: Diễn xuất nhân vật tinh tế (delta_f=4, disp=4.95)
TARGET_CLIPS = [
    'clip_000008',
    'clip_000010',
    'clip_000018',
    'clip_000064',
    'clip_003046',
    'clip_003050'
]

def main():
    os.makedirs(TARGET_DIR, exist_ok=True)
    df = pd.read_csv(TRIPLETS_CSV)
    
    extracted_manifest = []
    
    print("[*] Đang trích xuất các triplets từ Sakuga Golden Dataset...")
    for clip_id in TARGET_CLIPS:
        row = df[df['clip_id'] == clip_id]
        if row.empty:
            print(f"⚠️ Không tìm thấy clip: {clip_id}")
            continue
        row = row.iloc[0]
        
        src_clip_dir = os.path.join(DATA_ROOT, 'val', clip_id)
        if not os.path.exists(src_clip_dir):
            print(f"⚠️ Thư mục không tồn tại: {src_clip_dir}")
            continue
            
        dst_clip_dir = os.path.join(TARGET_DIR, clip_id)
        os.makedirs(dst_clip_dir, exist_ok=True)
        
        # Sao chép 3 khung hình: F0, F_gt, F1
        f0_src = os.path.join(src_clip_dir, '000001.png')
        fgt_src = os.path.join(src_clip_dir, '000002.png')
        f1_src = os.path.join(src_clip_dir, '000003.png')
        
        shutil.copy2(f0_src, os.path.join(dst_clip_dir, 'frame_00.png'))
        shutil.copy2(fgt_src, os.path.join(dst_clip_dir, 'frame_01.png'))
        shutil.copy2(f1_src, os.path.join(dst_clip_dir, 'frame_02.png'))
        
        info = {
            'clip_id': clip_id,
            'video_name': row['video_name'],
            't': row['t'],
            'delta_f': row['delta_f'],
            'inliers': row['inliers'],
            'mean_disp': row['mean_disp'],
            'local_dir': dst_clip_dir
        }
        extracted_manifest.append(info)
        print(f" [✓] Đã trích xuất {clip_id} -> {dst_clip_dir} (Disp: {row['mean_disp']}px, Inliers: {row['inliers']})")
        
    # Tạo chuỗi composite kết hợp (Master Sakuga Reel)
    composite_dir = os.path.join(TARGET_DIR, 'sakuga_composite_sequence')
    os.makedirs(composite_dir, exist_ok=True)
    
    comp_idx = 0
    for clip_id in TARGET_CLIPS:
        src_clip_dir = os.path.join(TARGET_DIR, clip_id)
        for fn in ['frame_00.png', 'frame_01.png', 'frame_02.png']:
            src_f = os.path.join(src_clip_dir, fn)
            dst_f = os.path.join(composite_dir, f"frame_{comp_idx:04d}.png")
            shutil.copy2(src_f, dst_f)
            comp_idx += 1
            
    print(f"[✓] Đã tạo chuỗi tổng hợp {comp_idx} frames tại: {composite_dir}")
    
    # Lưu metadata manifest
    manifest_df = pd.DataFrame(extracted_manifest)
    manifest_csv = os.path.join(TARGET_DIR, 'extracted_triplets_manifest.csv')
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"[✓] Đã lưu manifest tại: {manifest_csv}")

if __name__ == '__main__':
    main()
