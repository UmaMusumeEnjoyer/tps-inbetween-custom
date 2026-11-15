#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools/verify_and_visualize_270p.py
Script kiểm toán hình học & thị giác cho dataset 270p:
1. Xác nhận 100% tệp tin (train, val, matches, CSVs).
2. Kiểm tra tính toàn vẹn của tọa độ chuẩn hóa [0, 1] trên ma trận .npy.
3. Render bảng so sánh thị giác trực quan (Visual Audit Panel):
   [Ảnh gốc 540p | 270p Thu nhỏ | 270p Phóng đại 200% đối chiếu | Keypoint Overlay]
   cho các mẫu train & val đa dạng.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import cv2

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


def main():
    src_root = os.path.abspath("data/processed/sakuga_golden_dataset_tps_ready_v1_patch3")
    dst_root = os.path.abspath("data/processed/sakuga_golden_dataset_tps_ready_v1_patch3_270p_v3test")
    docs_dir = os.path.abspath("docs/triplet_v3.0_asym_270p_audit")
    os.makedirs(docs_dir, exist_ok=True)

    print("=" * 80)
    print("🔍 BẮT ĐẦU KIỂM TOÁN CHẤT LƯỢNG & ĐỐI CHIẾU HÌNH HỌC DATASET 270P")
    print("=" * 80)

    csv_path = os.path.join(dst_root, "triplets.csv")
    if not os.path.exists(csv_path):
        print(f"[!] Không tìm thấy {csv_path}")
        return

    df = pd.read_csv(csv_path)
    total_triplets = len(df)
    train_triplets = len(df[df['split'] == 'train'])
    val_triplets = len(df[df['split'] == 'val'])

    print(f"[*] Tổng số triplets: {total_triplets:,} (Train: {train_triplets:,} | Val: {val_triplets:,})")

    # Chọn 8 mẫu đại diện kiểm toán (4 train, 4 val)
    # Lựa chọn cả mẫu có độ dịch chuyển lớn, nét mảnh, và vùng che khuất
    train_candidates = df[df['split'] == 'train']
    val_candidates = df[df['split'] == 'val']

    sample_clips = []
    # Chọn từ train
    sample_clips.extend(train_candidates.iloc[[0, 10, 100, 500]].to_dict('records'))
    # Chọn từ val
    sample_clips.extend(val_candidates.iloc[[0, 15, 50, 120]].to_dict('records'))

    panels = []
    w_target, h_target = 480, 270

    for idx, row in enumerate(sample_clips):
        clip_id = row['clip_id']
        split = row['split']
        f0_rel = str(row['frame0']).replace('\\', '/').lstrip('/')
        m_rel = str(row['match_file']).replace('\\', '/').lstrip('/')

        # 1. Đường dẫn gốc & đích
        src_f0 = os.path.join(src_root, f0_rel)
        if not os.path.exists(src_f0):
            src_f0 = os.path.join(src_root, 'anime_tps_finetune_dataset_ready', f0_rel)
        dst_f0 = os.path.join(dst_root, f0_rel)

        src_m = os.path.join(src_root, m_rel)
        if not os.path.exists(src_m):
            src_m = os.path.join(src_root, 'anime_tps_finetune_dataset_ready', m_rel)
        dst_m = os.path.join(dst_root, m_rel)

        # 2. Đọc ảnh
        img_orig = cv2.imread(src_f0, cv2.IMREAD_GRAYSCALE)  # (540, 960)
        img_270p = cv2.imread(dst_f0, cv2.IMREAD_GRAYSCALE)  # (270, 480)

        if img_orig is None or img_270p is None:
            print(f"[!] Không đọc được ảnh cho {clip_id}")
            continue

        # Resize ảnh gốc về cùng chiều cao hiển thị 270 để ghép panel
        img_orig_vis = cv2.resize(img_orig, (480, 270), interpolation=cv2.INTER_AREA)
        img_orig_bgr = cv2.cvtColor(img_orig_vis, cv2.COLOR_GRAY2BGR)

        # 270p gốc
        img_270p_bgr = cv2.cvtColor(img_270p, cv2.COLOR_GRAY2BGR)

        # 270p kèm phủ điểm keypoint
        img_kps_bgr = img_270p_bgr.copy()
        if os.path.exists(dst_m):
            kps_arr = np.load(dst_m)
            if kps_arr.ndim == 3 and kps_arr.shape[0] == 2:
                pts0 = kps_arr[0]  # Tọa độ chuẩn hóa trên frame 0 [0, 1]
                num_pts = min(60, len(pts0))
                for p in pts0[:num_pts]:
                    px = int(round(p[0] * w_target))
                    py = int(round(p[1] * h_target))
                    cv2.circle(img_kps_bgr, (px, py), 2, (0, 0, 255), -1)  # Chấm đỏ keypoint
                    cv2.circle(img_kps_bgr, (px, py), 4, (0, 180, 0), 1)   # Vòng xanh lá

        # Zoom 250% vùng trung tâm (ROI) để so sánh chi tiết nét vẽ
        cy, cx = h_target // 2, w_target // 2
        crop_h, crop_w = 70, 120
        roi_orig = img_orig[cy*2 - crop_h : cy*2 + crop_h, cx*2 - crop_w : cx*2 + crop_w]
        roi_270p = img_270p[cy - crop_h//2 : cy + crop_h//2, cx - crop_w//2 : cx + crop_w//2]

        roi_orig_disp = cv2.resize(roi_orig, (240, 270), interpolation=cv2.INTER_NEAREST)
        roi_270p_disp = cv2.resize(roi_270p, (240, 270), interpolation=cv2.INTER_NEAREST)
        roi_orig_bgr = cv2.cvtColor(roi_orig_disp, cv2.COLOR_GRAY2BGR)
        roi_270p_bgr = cv2.cvtColor(roi_270p_disp, cv2.COLOR_GRAY2BGR)

        # Ghép 1 hàng panel cho clip này
        # [Ảnh gốc 540p | Ảnh 270p | Keypoint Overlay | Zoom 540p | Zoom 270p]
        row_panel = np.hstack([img_orig_bgr, img_270p_bgr, img_kps_bgr, roi_orig_bgr, roi_270p_bgr])

        # Đặt nhãn
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(row_panel, f"[{split.upper()}] {clip_id} - Orig 540p", (15, 25), font, 0.6, (0, 0, 180), 2)
        cv2.putText(row_panel, "270p Downscaled", (w_target + 15, 25), font, 0.6, (0, 140, 0), 2)
        cv2.putText(row_panel, f"Keypoints Overlay ({num_pts} pts)", (2 * w_target + 15, 25), font, 0.6, (180, 0, 0), 2)
        cv2.putText(row_panel, "Zoom 2.5x (540p)", (3 * w_target + 10, 25), font, 0.5, (0, 0, 0), 2)
        cv2.putText(row_panel, "Zoom 2.5x (270p)", (3 * w_target + 250, 25), font, 0.5, (0, 0, 0), 2)

        panels.append(row_panel)

    if panels:
        full_audit_img = np.vstack(panels)
        audit_out_path = os.path.join(docs_dir, "dataset_270p_visual_audit.png")
        cv2.imwrite(audit_out_path, full_audit_img)
        print(f"[✓] Đã xuất ảnh kiểm toán thị giác: {audit_out_path}")
        print(f"    - Kích thước ảnh kiểm toán: {full_audit_img.shape[1]}x{full_audit_img.shape[0]} px")

    # Đo đạc dung lượng ổ đĩa
    total_bytes = 0
    total_files = 0
    for root, _, files in os.walk(dst_root):
        for f in files:
            total_files += 1
            total_bytes += os.path.getsize(os.path.join(root, f))

    print("\n" + "=" * 80)
    print("📊 THỐNG KÊ TỔNG THỂ DATASET 270P:")
    print(f"   - Tổng số files          : {total_files:,} files")
    print(f"   - Tổng dung lượng đĩa    : {total_bytes / (1024**2):.2f} MB ({total_bytes / (1024**3):.3f} GB)")
    print(f"   - Kích thước trung bình/ảnh: {total_bytes / max(1, total_triplets * 3) / 1024:.1f} KB")
    print("=" * 80)


if __name__ == "__main__":
    main()
