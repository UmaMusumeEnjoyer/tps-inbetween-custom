import os
import sys
import argparse
import time

# Đảm bảo in tiếng Việt không bị lỗi encoding trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np
import torch

# Đảm bảo đường dẫn import hoạt động đúng
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from upcunet_v3 import RealWaifuUpScaler
import config


def read_image(path):
    """Đọc ảnh hỗ trợ tốt cả đường dẫn chứa ký tự Unicode/Tiếng Việt"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {path}")
    return img[:, :, [2, 1, 0]]  # BGR to RGB


def save_image(path, img_rgb):
    """Lưu ảnh RGB sang file hỗ trợ đường dẫn Unicode/Tiếng Việt"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    img_bgr = img_rgb[:, :, [2, 1, 0]]
    _, ext = os.path.splitext(path)
    ext = ext if ext else ".png"
    success, encoded_img = cv2.imencode(ext, img_bgr)
    if success:
        with open(path, "wb") as f:
            f.write(encoded_img.tobytes())
    else:
        raise ValueError(f"Không thể ghi ảnh vào đường dẫn: {path}")


def process_single_image(upscaler, input_path, output_path, tile=2, cache_mode=1, alpha=1.0):
    print(f"-> Đang xử lý: {input_path}")
    img_rgb = read_image(input_path)
    h, w, c = img_rgb.shape
    print(f"   Kích thước gốc: {w}x{h}")
    
    t0 = time.time()
    result_rgb = upscaler(img_rgb, tile_mode=tile, cache_mode=cache_mode, alpha=alpha)
    t1 = time.time()
    
    out_h, out_w, _ = result_rgb.shape
    print(f"   Kích thước sau upscale: {out_w}x{out_h} (Thời gian: {t1 - t0:.2f}s)")
    save_image(output_path, result_rgb)
    print(f"   [Thành công] Đã lưu kết quả tại: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Chạy Upscale ảnh đơn hoặc thư mục ảnh với Real-CUGAN")
    parser.add_argument("--input", "-i", type=str, default=None, help="Đường dẫn file ảnh đầu vào hoặc thư mục ảnh")
    parser.add_argument("--output", "-o", type=str, default=None, help="Đường dẫn file ảnh kết quả hoặc thư mục kết quả")
    parser.add_argument("--scale", "-s", type=int, default=getattr(config, "scale", 2), help="Tỉ lệ phóng to (2, 3, 4)")
    parser.add_argument("--weights", "-w", type=str, default=getattr(config, "model_path2", "weights_v3/up2x-latest-no-denoise.pth"), help="Đường dẫn file weights (.pth)")
    parser.add_argument("--alpha", "-a", type=float, default=getattr(config, "alpha", 1.0), help="Độ can thiệp làm nét (mặc định 1.0)")
    parser.add_argument("--tile", "-t", type=int, default=getattr(config, "tile", 2), help="Tile mode (0, 1, 2...)")
    parser.add_argument("--cache_mode", "-c", type=int, default=getattr(config, "cache_mode", 1), help="Cache mode (0, 1, 2, 3)")
    parser.add_argument("--device", "-d", type=str, default=getattr(config, "device", "cuda:0"), help="cuda:0 hoặc cpu")
    parser.add_argument("--half", action="store_true", default=getattr(config, "half", True), help="Bật nửa độ chính xác FP16")

    args = parser.parse_args()

    # Chuẩn hóa đường dẫn trọng số
    weight_path = args.weights
    if not os.path.isabs(weight_path):
        weight_path = os.path.join(current_dir, weight_path)

    if not os.path.exists(weight_path):
        print(f"LỖI: Không tìm thấy file trọng số tại: {weight_path}")
        print("Vui lòng tải file trọng số và đặt vào thư mục weights_v3/")
        sys.exit(1)

    print("==================================================")
    print("      REAL-CUGAN ANIME IMAGE RESTORATION          ")
    print("==================================================")
    print(f"Model Weights : {os.path.basename(weight_path)}")
    print(f"Scale         : {args.scale}x")
    print(f"Alpha         : {args.alpha}")
    print(f"Device        : {args.device} (FP16: {args.half})")
    print(f"Tile Mode     : {args.tile} | Cache Mode: {args.cache_mode}")
    print("==================================================")

    upscaler = RealWaifuUpScaler(
        scale=args.scale,
        weight_path=weight_path,
        half=args.half,
        device=args.device
    )

    # 1. Nếu truyền qua tham số --input là một file ảnh cụ thể
    if args.input and os.path.isfile(args.input):
        input_file = args.input
        if args.output:
            output_file = args.output
        else:
            base, ext = os.path.splitext(input_file)
            output_file = f"{base}_{args.scale}x_restored{ext}"
        process_single_image(upscaler, input_file, output_file, args.tile, args.cache_mode, args.alpha)

    # 2. Nếu là thư mục hoặc đọc từ config.py
    else:
        in_dir = args.input if args.input else os.path.join(current_dir, getattr(config, "input_dir", "input_dir1"))
        out_dir = args.output if args.output else os.path.join(current_dir, getattr(config, "output_dir", "output_dir"))
        
        if not os.path.exists(in_dir):
            os.makedirs(in_dir, exist_ok=True)
            print(f"Thư mục '{in_dir}' đang trống. Vui lòng copy ảnh cần xử lý vào đây!")
            return

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = [f for f in os.listdir(in_dir) if os.path.splitext(f)[1].lower() in valid_extensions]

        if not image_files:
            print(f"Không tìm thấy ảnh hợp lệ nào trong '{in_dir}'.")
            return

        print(f"Tìm thấy {len(image_files)} ảnh cần xử lý.")
        for img_name in image_files:
            in_path = os.path.join(in_dir, img_name)
            out_path = os.path.join(out_dir, f"{os.path.splitext(img_name)[0]}_{args.scale}x.png")
            process_single_image(upscaler, in_path, out_path, args.tile, args.cache_mode, args.alpha)

    print("\nHoàn tất quá trình upscale!")


if __name__ == "__main__":
    main()
