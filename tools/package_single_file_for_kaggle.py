#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools/package_single_file_for_kaggle.py
Đóng gói sakuga_golden_dataset_tps_ready.tar.gz thành 1 file ZIP đơn nhất
(Zip-in-Zip với ZIP_STORED) để tải lên Kaggle Dataset.
Kaggle sẽ giải nén file ZIP ngoài và chỉ thấy 1 file .tar.gz duy nhất!
"""

import os
import sys
import time
import zipfile

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

def package_for_kaggle():
    pbl6_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_tar = os.path.join(pbl6_root, "data", "processed", "sakuga_golden_dataset_tps_ready.tar.gz")
    out_zip = os.path.join(pbl6_root, "data", "processed", "sakuga_golden_dataset_kaggle_upload.zip")

    if not os.path.exists(src_tar):
        print(f"❌ Không tìm thấy file nguồn: {src_tar}")
        sys.exit(1)

    src_size_mb = os.path.getsize(src_tar) / (1024 * 1024)
    print("=" * 80)
    print("📦 BẮT ĐẦU ĐÓNG GÓI SAKUGA GOLDEN DATASET CHO KAGGLE (SINGLE FILE CONTAINER)")
    print(f"[*] File nguồn: {src_tar} ({src_size_mb:.2f} MB)")
    print(f"[*] File đích  : {out_zip}")
    print("=" * 80)

    t0 = time.time()
    # Sử dụng ZIP_STORED (không nén lại vì file .tar.gz đã được nén tối ưu) -> Tốc độ cực nhanh
    with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_STORED) as zf:
        print("[*] Đang đóng gói vào file ZIP container...")
        zf.write(src_tar, arcname=os.path.basename(src_tar))

    t_elapsed = time.time() - t0
    out_size_mb = os.path.getsize(out_zip) / (1024 * 1024)

    # Thẩm định lại nội dung file ZIP
    print("\n🔍 Đang thẩm định file ZIP vừa tạo...")
    with zipfile.ZipFile(out_zip, 'r') as zf:
        namelist = zf.namelist()
        print(f"   - Số lượng file bên trong ZIP: {len(namelist)} file")
        for name in namelist:
            info = zf.getinfo(name)
            print(f"     + Entry: '{name}' | Kích thước: {info.file_size / (1024*1024):.2f} MB")

    if len(namelist) == 1 and namelist[0] == "sakuga_golden_dataset_tps_ready.tar.gz":
        print(f"\n🎉 HOÀN THÀNH XUẤT SẮC TRONG {t_elapsed:.1f} GIÂY!")
        print(f"📁 Đường dẫn file tải lên Kaggle: {out_zip} ({out_size_mb:.2f} MB)")
        print("\n👉 HƯỚNG DẪN TẢI LÊN KAGGLE:")
        print("   1. Mở Kaggle -> Datasets -> 'New Dataset' (hoặc cập nhật dataset hiện tại).")
        print(f"   2. Kéo thả file '{os.path.basename(out_zip)}' lên Kaggle.")
        print("   3. Đặt tên dataset: 'sakuga-golden-dataset-tps-ready'.")
        print("   4. Khi hoàn tất, Kaggle chỉ hiển thị DUY NHẤT 1 FILE 'sakuga_golden_dataset_tps_ready.tar.gz'.")
        print("   5. Notebook sẽ khởi động ngay lập tức trong 1-2 giây!")
    else:
        print("❌ Lỗi: Cấu trúc file ZIP không đúng!")
        sys.exit(1)

if __name__ == "__main__":
    package_for_kaggle()
