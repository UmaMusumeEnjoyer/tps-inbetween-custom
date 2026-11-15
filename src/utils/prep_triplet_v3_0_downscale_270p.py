#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools/downscale_sakuga_dataset_v3test.py
Script hạ độ phân giải dataset Sakuga Golden TPS-Ready từ 540p (960x540) xuống 270p (480x270)
dành cho phiên bản thử nghiệm training v3.0test.

Mục tiêu kỹ thuật:
1. Giảm 75% số điểm ảnh thô (129,600 px vs 518,400 px), triệt tiêu hiện tượng phình RAM do
   phân mảnh heap cấp phát I/O buffer và hàng đợi IPC của DataLoader.
2. Giữ nguyên tỷ lệ 16:9 đồng dạng, bảo toàn 100% tọa độ chuẩn hóa [0, 1] của ma trận TPS .npy.
3. Sử dụng cv2.INTER_AREA giữ nguyên độ liên tục và sắc nét của nét vẽ anime lineart.
4. Đa luồng tốc độ cao (ThreadPoolExecutor), cơ chế resume an toàn (idempotent),
   và tự động kiểm toán toàn vẹn sau khi hoàn tất.
"""

import os
import sys
import time
import shutil
import json
import argparse
import numpy as np
import pandas as pd
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Downscale Sakuga Golden Dataset for v3.0test")
    parser.add_argument(
        "--src_root",
        type=str,
        default="data/processed/sakuga_golden_dataset_tps_ready_v1_patch3",
        help="Path to source dataset"
    )
    parser.add_argument(
        "--dst_root",
        type=str,
        default="data/processed/sakuga_golden_dataset_tps_ready_v1_patch3_270p_v3test",
        help="Path to target downscaled dataset"
    )
    parser.add_argument("--width", type=int, default=480, help="Target image width")
    parser.add_argument("--height", type=int, default=270, help="Target image height")
    parser.add_argument("--workers", type=int, default=16, help="Worker threads for parallel I/O & resize")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of triplets for quick test")
    parser.add_argument("--skip_verify", action="store_true", help="Skip post-processing audit")
    return parser.parse_args()


def process_triplet(row, src_root, dst_root, target_size):
    """
    Xử lý một triplet:
    1. Đọc và resize 3 frame (0, gt, 1) xuống target_size bằng cv2.INTER_AREA.
    2. Sao chép match_file .npy tương ứng.
    """
    target_w, target_h = target_size
    f0_rel = str(row['frame0']).replace('\\', '/').lstrip('/')
    fgt_rel = str(row['frame_gt']).replace('\\', '/').lstrip('/')
    f1_rel = str(row['frame1']).replace('\\', '/').lstrip('/')
    m_rel = str(row['match_file']).replace('\\', '/').lstrip('/')

    # Xử lý frame 0, gt, 1
    frames = [f0_rel, fgt_rel, f1_rel]
    for rel_f in frames:
        src_path = os.path.join(src_root, rel_f)
        if not os.path.exists(src_path):
            # Fallback nếu cấu trúc nằm trong anime_tps_finetune_dataset_ready
            alt_path = os.path.join(src_root, 'anime_tps_finetune_dataset_ready', rel_f)
            if os.path.exists(alt_path):
                src_path = alt_path
            else:
                return False, f"Missing frame: {rel_f}"

        dst_path = os.path.join(dst_root, rel_f)
        # Kiểm tra nếu đã tồn tại và kích thước > 0 thì bỏ qua (resume)
        if not (os.path.exists(dst_path) and os.path.getsize(dst_path) > 1024):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return False, f"Failed to read image: {src_path}"
            
            # Resize với INTER_AREA
            if (img.shape[1], img.shape[0]) != (target_w, target_h):
                img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            else:
                img_resized = img

            # Lưu PNG với mức nén 4 (cân bằng tốc độ và dung lượng)
            cv2.imwrite(dst_path, img_resized, [cv2.IMWRITE_PNG_COMPRESSION, 4])

    # Sao chép file match .npy
    src_m = os.path.join(src_root, m_rel)
    if not os.path.exists(src_m):
        alt_m = os.path.join(src_root, 'anime_tps_finetune_dataset_ready', m_rel)
        if os.path.exists(alt_m):
            src_m = alt_m
        else:
            return False, f"Missing match file: {m_rel}"

    dst_m = os.path.join(dst_root, m_rel)
    if not (os.path.exists(dst_m) and os.path.getsize(dst_m) > 128):
        os.makedirs(os.path.dirname(dst_m), exist_ok=True)
        shutil.copy2(src_m, dst_m)

    return True, None


def main():
    args = parse_args()
    t_start = time.time()
    src_root = os.path.abspath(args.src_root)
    dst_root = os.path.abspath(args.dst_root)
    target_size = (args.width, args.height)

    print("=" * 80)
    print("🚀 BẮT ĐẦU TÁC VỤ HẠ ĐỘ PHÂN GIẢI DATASET SAKUGA CHO PHIÊN BẢN V3.0TEST")
    print(f"   - Thư mục nguồn      : {src_root}")
    print(f"   - Thư mục đích       : {dst_root}")
    print(f"   - Độ phân giải đích  : {args.width}x{args.height} (270p, Tỉ lệ 0.5x, INTER_AREA)")
    print(f"   - Số luồng xử lý     : {args.workers} threads")
    print("=" * 80)
    sys.stdout.flush()

    if not os.path.exists(src_root):
        print(f"[!] Lỗi: Không tìm thấy thư mục nguồn {src_root}!")
        sys.exit(1)

    os.makedirs(dst_root, exist_ok=True)

    # 1. Định vị và sao chép các tệp metadata CSV
    meta_files = [
        'triplets.csv',
        'triplets_combined.csv',
        'quality_audit_report.csv',
        'quality_audit_report_combined.csv'
    ]
    for mf in meta_files:
        src_mf = os.path.join(src_root, mf)
        if not os.path.exists(src_mf):
            alt_mf = os.path.join(src_root, 'anime_tps_finetune_dataset_ready', mf)
            if os.path.exists(alt_mf):
                src_mf = alt_mf
        if os.path.exists(src_mf):
            dst_mf = os.path.join(dst_root, mf)
            shutil.copy2(src_mf, dst_mf)
            print(f"[✓] Đã sao chép siêu dữ liệu: {mf}")

    # Đọc danh sách triplets
    csv_path = os.path.join(dst_root, 'triplets.csv')
    if not os.path.exists(csv_path):
        print(f"[!] Lỗi: Không tìm thấy file {csv_path}!")
        sys.exit(1)

    df_triplets = pd.read_csv(csv_path)
    total_samples = len(df_triplets)
    if args.limit:
        df_triplets = df_triplets.iloc[:args.limit].copy()
        total_samples = len(df_triplets)
        print(f"[*] Chế độ giới hạn thử nghiệm: Chỉ xử lý {total_samples} triplets.")

    print(f"[*] Tổng số triplets cần xử lý: {total_samples:,} (Train: {len(df_triplets[df_triplets['split']=='train']):,} | Val: {len(df_triplets[df_triplets['split']=='val']):,})")
    print(f"[*] Tổng số frame ảnh cần chuyển đổi: {total_samples * 3:,} ảnh.")
    sys.stdout.flush()

    # 2. Xử lý đa luồng với ThreadPoolExecutor
    errors = []
    success_count = 0

    rows = [row for _, row in df_triplets.iterrows()]
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_triplet, row, src_root, dst_root, target_size): idx
            for idx, row in enumerate(rows)
        }
        
        pbar = tqdm(
            as_completed(futures),
            total=total_samples,
            desc="🔄 Downscaling 270p",
            unit="triplet",
            ncols=90
        )
        for fut in pbar:
            ok, err = fut.result()
            if ok:
                success_count += 1
            else:
                errors.append(err)
                if len(errors) <= 5:
                    print(f"\n[!] Lỗi: {err}")

    elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"[✓] HOÀN TẤT XỬ LÝ {success_count:,}/{total_samples:,} TRIPLETS TRONG {elapsed:.1f} GIÂY!")
    print(f"    - Tốc độ trung bình: {success_count / max(0.1, elapsed):.1f} triplets/giây ({success_count * 3 / max(0.1, elapsed):.1f} ảnh/giây)")
    if errors:
        print(f"    - Cảnh báo: Có {len(errors)} lỗi phát sinh trong quá trình chuyển đổi.")
    print("=" * 80)

    # 3. Tạo DATASET_INFO.json mới
    info_dict = {
        "dataset_name": "sakuga_golden_dataset_tps_ready_v1_patch3_270p_v3test",
        "description": "Sakuga Golden Dataset TPS-Ready V3.0 Testbed (270p Downscaled Edition)",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_source": "sakuga_golden_dataset_tps_ready_v1_patch3",
        "resolution": f"{args.width}x{args.height}",
        "aspect_ratio": "16:9",
        "downscale_ratio": 0.5,
        "interpolation_algorithm": "cv2.INTER_AREA",
        "color_format": "Grayscale 8-bit Lineart (PNG)",
        "keypoints_format": "GlueStick SIFT/SuperPoint .npy (Normalized [0, 1] - Invariant)",
        "total_triplets": total_samples,
        "train_triplets": int(len(df_triplets[df_triplets['split'] == 'train'])),
        "val_triplets": int(len(df_triplets[df_triplets['split'] == 'val'])),
        "total_frames": total_samples * 3,
        "total_matches": total_samples,
        "target_purpose": "Resolve per-epoch RAM accumulation & accelerate v3.0 training iteration"
    }

    info_path = os.path.join(dst_root, "DATASET_INFO.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info_dict, f, indent=2, ensure_ascii=False)
    print(f"[✓] Đã tạo tệp thông tin dataset mới: {info_path}")

    # 4. Kiểm toán Hậu Xử Lý (Post-Processing Verification)
    if not args.skip_verify and not args.limit:
        print("\n" + "=" * 80)
        print("🔍 ĐANG THỰC HIỆN KIỂM TOÁN TÍNH TOÀN VẸN (POST-PROCESSING AUDIT)...")
        print("=" * 80)
        sys.stdout.flush()

        audit_errors = 0
        checked_images = 0
        sample_indices = np.random.choice(total_samples, size=min(300, total_samples), replace=False)
        for s_idx in sample_indices:
            row = df_triplets.iloc[s_idx]
            for f_col in ['frame0', 'frame_gt', 'frame1']:
                p = os.path.join(dst_root, str(row[f_col]).replace('\\', '/').lstrip('/'))
                if not os.path.exists(p):
                    audit_errors += 1
                    continue
                checked_images += 1
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if img is None or img.shape != (args.height, args.width):
                    audit_errors += 1
                    print(f"[!] Sai kích thước tại: {p} (Shape: {img.shape if img is not None else 'None'})")

            # Kiểm tra file npy
            mp = os.path.join(dst_root, str(row['match_file']).replace('\\', '/').lstrip('/'))
            if not os.path.exists(mp):
                audit_errors += 1
            else:
                try:
                    kps = np.load(mp)
                    if kps.shape[0] != 2 or kps.ndim != 3:
                        audit_errors += 1
                except Exception:
                    audit_errors += 1

        print(f"[✓] Đã kiểm tra ngẫu nhiên {checked_images} ảnh và {len(sample_indices)} file npy.")
        if audit_errors == 0:
            print(f"[🌟 THÀNH CÔNG RỰC RỠ] 100% mẫu kiểm tra khớp chuẩn kích thước ({args.height}, {args.width}) và toàn vẹn ma trận TPS!")
        else:
            print(f"[!] Phát hiện {audit_errors} lỗi trong quá trình kiểm toán!")

    print(f"\n[DONE] Hoàn tất toàn bộ quy trình trong {time.time() - t_start:.1f}s.")


if __name__ == "__main__":
    main()
