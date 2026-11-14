#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chương trình chuyển đổi và làm sạch nét vẽ Line Art cho mô hình tps-inbetween:
- Đọc tất cả các ảnh từ thư mục đầu vào (mặc định: ChayGoAnimeLinesDetail_NoBG).
- Loại bỏ hoàn toàn các pixel màu xám (1-254), nhiễu mờ, gradient anti-aliasing.
- Chỉ giữ lại nét vẽ đen tuyền (pixel = 0) trên nền trắng thuần (pixel = 255).
- Sử dụng thuật toán Hysteresis Thresholding kết hợp Connected Components để:
  + Giữ nguyên vẹn độ liên tục của các nét mảnh (tóc, viền mắt, nếp gấp áo).
  + Khử triệt để các hạt nhiễu xám cô lập trên nền ảnh.
  + Nhận diện an toàn các frame trống (blank transition frames).
- Lưu kết quả vào thư mục đầu ra (mặc định: ChayGoAnimeLinesDetail_NoBG_NoGay).

Tác giả: PBL6 - Antigravity
"""

import os
import sys
import argparse
import glob
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# Đảm bảo hiển thị Unicode tiếng Việt trên terminal Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass


def binarize_lineart_hysteresis(
    img: np.ndarray,
    strong_thresh: int = 140,
    weak_thresh: int = 205,
    min_size: int = 2,
    blank_cutoff: int = 220,
) -> np.ndarray:
    """
    Binarize ảnh line art sử dụng phương pháp Hysteresis kết hợp lọc thành phần liên thông.

    Args:
        img: Ảnh grayscale 2D dạng uint8 (0-255).
        strong_thresh: Ngưỡng nét chính (giá trị < strong_thresh chắc chắn là nét vẽ).
        weak_thresh: Ngưỡng nét phụ / chóp nét (giá trị < weak_thresh nối với nét chính).
        min_size: Diện tích tối thiểu (pixel) của thành phần để loại bỏ hạt bụi cô lập.
        blank_cutoff: Ngưỡng phát hiện frame trắng hoàn toàn.

    Returns:
        Ảnh nhị phân uint8 chỉ gồm 2 giá trị {0, 255} (0: nét đen, 255: nền trắng).
    """
    # 1. Kiểm tra frame trắng / rỗng (ví dụ: các frame chuyển cảnh)
    if img.min() > blank_cutoff:
        return np.full_like(img, 255, dtype=np.uint8)

    # 2. Tạo mặt nạ nét chính (core) và mặt nạ ứng viên (candidate)
    strong_mask = img < strong_thresh
    weak_mask = img < weak_thresh

    # Nếu không có bất kỳ pixel nét chính nào, kiểm tra xem có ứng viên không
    if not np.any(strong_mask):
        if not np.any(weak_mask):
            return np.full_like(img, 255, dtype=np.uint8)

    # 3. Phân tích thành phần liên thông trên mặt nạ ứng viên (weak_mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        weak_mask.astype(np.uint8), connectivity=8
    )

    # Nếu chỉ có background (label = 0)
    if num_labels <= 1:
        return np.full_like(img, 255, dtype=np.uint8)

    # 4. Xác định các thành phần chứa ít nhất 1 pixel nét chính
    has_strong = np.zeros(num_labels, dtype=bool)
    np.bitwise_or.at(has_strong, labels, strong_mask)
    has_strong[0] = False  # Bỏ qua nhãn background

    # 5. Lọc kích thước tối thiểu để loại bỏ nhiễu hạt bụi cực nhỏ
    areas = stats[:, cv2.CC_STAT_AREA]
    valid_components = has_strong & (areas >= min_size)

    # 6. Tạo ảnh nhị phân đầu ra: 0 cho nét đen, 255 cho nền trắng
    line_mask = valid_components[labels]
    result = np.where(line_mask, 0, 255).astype(np.uint8)

    return result


def binarize_lineart_fixed(img: np.ndarray, thresh: int = 180) -> np.ndarray:
    """Binarize ảnh bằng ngưỡng cố định đơn giản."""
    return np.where(img < thresh, 0, 255).astype(np.uint8)


def binarize_lineart_otsu(img: np.ndarray, blank_cutoff: int = 220) -> np.ndarray:
    """Binarize ảnh bằng thuật toán Otsu với cơ chế bảo vệ frame trắng."""
    if img.min() > blank_cutoff:
        return np.full_like(img, 255, dtype=np.uint8)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def process_image(
    input_path: str,
    output_path: str,
    mode: str = "hysteresis",
    strong_thresh: int = 140,
    weak_thresh: int = 205,
    min_size: int = 2,
    fixed_thresh: int = 180,
) -> dict:
    """Đọc ảnh, binarize và ghi file đầu ra."""
    # Đọc ảnh gốc giữ nguyên cấu trúc kênh
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Không thể đọc file ảnh: {input_path}")

    # Chuẩn hóa về Grayscale để xử lý
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3 and img.shape[2] == 4:
        # Nếu có alpha, pha nền trắng
        alpha = img[:, :, 3].astype(np.float32) / 255.0
        bgr = img[:, :, :3].astype(np.float32)
        white_bg = np.ones_like(bgr) * 255.0
        blended = (bgr * alpha[:, :, None] + white_bg * (1.0 - alpha[:, :, None])).astype(np.uint8)
        gray = cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Định dạng kênh không hỗ trợ: {img.shape}")

    # Áp dụng thuật toán binarize tương ứng
    if mode == "hysteresis":
        bin_img = binarize_lineart_hysteresis(
            gray,
            strong_thresh=strong_thresh,
            weak_thresh=weak_thresh,
            min_size=min_size,
        )
    elif mode == "fixed":
        bin_img = binarize_lineart_fixed(gray, thresh=fixed_thresh)
    elif mode == "otsu":
        bin_img = binarize_lineart_otsu(gray)
    else:
        raise ValueError(f"Chế độ không hợp lệ: {mode}")

    # Ghi file kết quả dưới định dạng PNG nén chất lượng cao
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    success = cv2.imwrite(output_path, bin_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not success:
        raise IOError(f"Không thể ghi ảnh ra: {output_path}")

    # Thống kê kết quả
    black_pixels = int((bin_img == 0).sum())
    total_pixels = int(bin_img.size)
    unique_vals = np.unique(bin_img).tolist()

    return {
        "file": os.path.basename(input_path),
        "black_pixels": black_pixels,
        "total_pixels": total_pixels,
        "black_pct": (black_pixels / total_pixels) * 100.0,
        "unique_vals": unique_vals,
    }


def verify_output_directory(output_dir: str) -> bool:
    """Quét toàn bộ thư mục kết quả để đảm bảo 100% ảnh không chứa pixel xám."""
    print("\n" + "=" * 65)
    print("   KIỂM ĐỊNH CHẤT LƯỢNG KẾT QUẢ ĐẦU RA (ZERO GRAY PIXEL CHECK)")
    print("=" * 65)

    png_files = sorted(glob.glob(os.path.join(output_dir, "*.png")))
    if not png_files:
        print("[!] LỖI: Không tìm thấy file png nào trong thư mục đầu ra!")
        return False

    valid_count = 0
    invalid_files = []

    for f in png_files:
        im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if im is None:
            invalid_files.append((os.path.basename(f), "Không đọc được file"))
            continue

        unique_vals = set(np.unique(im).tolist())
        # Chỉ chấp nhận tập con của {0, 255}
        if not unique_vals.issubset({0, 255}):
            invalid_files.append((os.path.basename(f), f"Có pixel xám: {unique_vals}"))
        else:
            valid_count += 1

    print(f"[*] Tổng số ảnh đã kiểm tra: {len(png_files)}")
    print(f"[*] Số ảnh đạt chuẩn 100% nhị phân {0, 255}: {valid_count}/{len(png_files)}")

    if invalid_files:
        print(f"[!] PHÁT HIỆN {len(invalid_files)} FILE CHƯA ĐẠT CHUẨN:")
        for name, err in invalid_files[:10]:
            print(f"    - {name}: {err}")
        return False

    print("[OK] TẤT CẢ CÁC ẢNH ĐỀU ĐẠT CHUẨN TUYỆT ĐỐI (0 PIXEL XÁM)!")
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Binarize Anime Line Art cho tps-inbetween (Loại bỏ 100% pixel xám)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="ChayGoAnimeLinesDetail_NoBG",
        help="Thư mục chứa ảnh gốc đầu vào (mặc định: ChayGoAnimeLinesDetail_NoBG)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ChayGoAnimeLinesDetail_NoBG_NoGay",
        help="Thư mục lưu ảnh sau khi binarize (mặc định: ChayGoAnimeLinesDetail_NoBG_NoGay)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["hysteresis", "fixed", "otsu"],
        default="hysteresis",
        help="Thuật toán binarize: hysteresis (mặc định & tốt nhất), fixed, otsu",
    )
    parser.add_argument(
        "--strong-thresh",
        type=int,
        default=140,
        help="Ngưỡng nét vẽ chính cho Hysteresis (mặc định: 140)",
    )
    parser.add_argument(
        "--weak-thresh",
        type=int,
        default=205,
        help="Ngưỡng nét vẽ phụ/chóp nét cho Hysteresis (mặc định: 205)",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=2,
        help="Kích thước tối thiểu (pixel) của thành phần nét vẽ để khử bụi (mặc định: 2)",
    )
    parser.add_argument(
        "--fixed-thresh",
        type=int,
        default=180,
        help="Ngưỡng cắt khi chọn mode fixed (mặc định: 180)",
    )
    parser.add_argument(
        "--skip-smooth",
        action="store_true",
        help="Bỏ qua các file có đuôi _smooth.png nếu có",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Bỏ qua bước tự động kiểm định sau khi chạy",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = script_dir / input_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir

    if not input_dir.exists():
        print(f"[!] LỖI: Thư mục đầu vào không tồn tại: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Tìm tất cả các file ảnh hỗ trợ
    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    all_files = [
        f
        for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in valid_exts
    ]

    if args.skip_smooth:
        all_files = [f for f in all_files if not f.name.endswith("_smooth.png")]

    all_files.sort(key=lambda p: p.name)

    print("=" * 65)
    print("   BINARIZE LINE ART FOR TPS-INBETWEEN MODEL")
    print("=" * 65)
    print(f"[*] Thư mục đầu vào : {input_dir}")
    print(f"[*] Thư mục đầu ra  : {output_dir}")
    print(f"[*] Số lượng file   : {len(all_files)}")
    print(f"[*] Thuật toán      : {args.mode}")
    if args.mode == "hysteresis":
        print(f"    - Nét chính (< strong): {args.strong_thresh}")
        print(f"    - Nét phụ (< weak)    : {args.weak_thresh}")
        print(f"    - Khử bụi (< min_size): {args.min_size} px")
    elif args.mode == "fixed":
        print(f"    - Ngưỡng cố định      : {args.fixed_thresh}")
    print("=" * 65)

    stats_list = []
    pbar = tqdm(all_files, desc="Đang chuyển đổi", unit="ảnh", ncols=80)
    for file_path in pbar:
        out_path = output_dir / file_path.name
        res = process_image(
            str(file_path),
            str(out_path),
            mode=args.mode,
            strong_thresh=args.strong_thresh,
            weak_thresh=args.weak_thresh,
            min_size=args.min_size,
            fixed_thresh=args.fixed_thresh,
        )
        stats_list.append(res)
        pbar.set_postfix({"file": file_path.name, "lines%": f"{res['black_pct']:.1f}%"})

    print("\n[OK] Đã hoàn thành chuyển đổi toàn bộ ảnh thành công!")

    # Kiểm định chất lượng
    if not args.no_verify:
        is_valid = verify_output_directory(str(output_dir))
        if not is_valid:
            sys.exit(1)


if __name__ == "__main__":
    main()
