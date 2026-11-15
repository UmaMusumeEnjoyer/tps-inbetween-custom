import os
import sys
import shutil
import cv2
import torch
import argparse
import numpy as np
import math
from torch.nn import functional as F

# Đảm bảo thư mục gốc của uniVIP nằm trong sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Hỗ trợ hiển thị ký tự Unicode tiếng Việt trên Windows terminal
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from core.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_time_steps(times_args, num_frames):
    """
    Phân tích danh sách các mốc thời gian t từ tham số truyền vào CLI.
    Hỗ trợ cả danh sách cách nhau bởi dấu cách hoặc dấu phẩy.
    """
    t_list = []
    if times_args is not None and len(times_args) > 0:
        for item in times_args:
            parts = str(item).replace(',', ' ').split()
            for p in parts:
                try:
                    val = float(p)
                    if 0.0 <= val <= 1.0:
                        t_list.append(val)
                    else:
                        print(f"[Cảnh báo] Mốc thời gian t={val} nằm ngoài khoảng [0.0, 1.0], bỏ qua.")
                except ValueError:
                    print(f"[Cảnh báo] Không thể đọc mốc thời gian '{p}', bỏ qua.")
    elif num_frames is not None and num_frames > 0:
        # Tự động chia đều các mốc t trong khoảng (0, 1)
        step = 1.0 / (num_frames + 1)
        t_list = [(i + 1) * step for i in range(num_frames)]
    else:
        # Mặc định lấy mốc chính giữa t = 0.5
        t_list = [0.5]

    # Loại bỏ trùng lặp và sắp xếp tăng dần
    t_list = sorted(list(set(t_list)))
    return t_list


def format_inbetween_filename(prefix_1, t, digits=3):
    """
    Định dạng tên file theo quy chuẩn: {prefix_1}.{xxx}.png
    Ví dụ: prefix_1 = 1, t = 0.50  -> 1.500.png
           prefix_1 = 1, t = 0.125 -> 1.125.png
           prefix_1 = 1, t = 0.05  -> 1.050.png
    """
    multiplier = 10 ** digits
    t_part = int(round(t * multiplier))
    return f"{prefix_1}.{t_part:0{digits}d}.png"


def run_interpolation(args):
    # 1. Tạo thư mục lưu kết quả đầu ra
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # 2. Đọc và kiểm tra 2 ảnh đầu vào
    if not os.path.exists(args.frame0):
        raise FileNotFoundError(f"Không tìm thấy ảnh frame0 tại: {args.frame0}")
    if not os.path.exists(args.frame1):
        raise FileNotFoundError(f"Không tìm thấy ảnh frame1 tại: {args.frame1}")

    ori_img0 = cv2.imread(args.frame0)
    ori_img1 = cv2.imread(args.frame1)

    if ori_img0 is None:
        raise ValueError(f"Không thể đọc file ảnh Frame 1 tại: {args.frame0}")
    if ori_img1 is None:
        raise ValueError(f"Không thể đọc file ảnh Frame 2 tại: {args.frame1}")

    if ori_img0.shape != ori_img1.shape:
        raise ValueError(
            f"Kích thước 2 ảnh không khớp! Frame0: {ori_img0.shape}, Frame1: {ori_img1.shape}. "
            f"Vui lòng đảm bảo 2 ảnh có cùng chiều cao, chiều rộng và số kênh màu."
        )

    h, w, _ = ori_img0.shape
    print(f"[*] Đã tải 2 ảnh đầu vào thành công. Kích thước gốc: {w}x{h} (WxH)")

    # 3. Phân tích chuỗi mốc thời gian t
    t_list = parse_time_steps(args.times, args.num_frames)
    print(f"[*] Danh sách mốc thời gian t cần nội suy ({len(t_list)} frames): {t_list}")

    # 4. Copy và lưu lại Frame 1 & Frame 2 vào thư mục đầu ra
    frame1_name = f"{args.prefix_1}.{'0' * args.digits}.png"
    frame2_name = f"{args.prefix_2}.{'0' * args.digits}.png"

    frame1_path = os.path.join(save_dir, frame1_name)
    frame2_path = os.path.join(save_dir, frame2_name)

    cv2.imwrite(frame1_path, ori_img0)
    cv2.imwrite(frame2_path, ori_img1)
    print(f"[+] Đã lưu Frame 1 gốc -> {frame1_path}")
    print(f"[+] Đã lưu Frame 2 gốc -> {frame2_path}")

    # 5. Xác định cấu hình mô hình (Base hay Large) và file checkpoint
    model_size_lower = args.model_size.lower()
    is_large = model_size_lower in ["large", "l", "llarge"]

    model_file = args.model_file
    if model_file is None or model_file == "":
        model_file = "./checkpoints/uniVIP-L.pkl" if is_large else "./checkpoints/uniVIP-B.pkl"
    elif model_file == "./checkpoints/uniVIP-B.pkl" and is_large:
        # Nếu người dùng chọn --model_size large nhưng vẫn để default path base
        model_file = "./checkpoints/uniVIP-L.pkl"

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Không tìm thấy file checkpoint trọng số tại: {model_file}")

    print(f"[*] Thiết bị sử dụng: {DEVICE}")
    print(f"[*] Đang tải mô hình uniVIP [{'uniVIP-L (Large)' if is_large else 'uniVIP-B (Base)'}] từ: {model_file}...")

    # 6. Khởi tạo mô hình Pipeline
    model_cfg_dict = dict(
        load_pretrain=True,
        large_model=is_large,
        model_file=model_file
    )
    ppl = Pipeline(model_cfg_dict)
    ppl.eval()

    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.demo = True

    # Chuẩn bị tensor ảnh đầu vào (1, 3, H, W) trong khoảng [0, 1]
    img0 = (torch.tensor(ori_img0.transpose(2, 0, 1), dtype=torch.float32).to(DEVICE) / 255.0).unsqueeze(0)
    img1 = (torch.tensor(ori_img1.transpose(2, 0, 1), dtype=torch.float32).to(DEVICE) / 255.0).unsqueeze(0)

    # 7. Tiến hành nội suy từng mốc thời gian t
    print(f"[*] Bắt đầu quá trình nội suy các khung hình trung gian (In-between frames)...")
    for idx, t in enumerate(t_list):
        if t == 0.0 or t == 1.0:
            # Nếu t = 0 hoặc t = 1 thì chính là 2 frame gốc đã được lưu trước đó
            continue

        with torch.no_grad():
            pred_img, _ = ppl.model(
                img0,
                img1,
                time_period=float(t),
                skip_pad=False,
                fixed_pyr_level=args.fixed_pyr_level
            )

        # Cắt về kích thước gốc ban đầu (crop padding nếu có)
        pred_img = pred_img[:, :, :h, :w]
        pred_np = (pred_img[0] * 255.0).clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)

        # Định dạng tên file: {prefix_1}.{xxx}.png
        out_filename = format_inbetween_filename(args.prefix_1, t, digits=args.digits)
        out_filepath = os.path.join(save_dir, out_filename)
        cv2.imwrite(out_filepath, pred_np)
        print(f"    [{idx+1}/{len(t_list)}] Hoàn thành t={t:.4f} -> {out_filename}")

    print(f"\n[✓] Hoàn tất toàn bộ quá trình! Tất cả kết quả đã được lưu tại: {os.path.abspath(save_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="uniVIP Custom Arbitrary Time-step Video Frame Interpolation Demo"
    )
    parser.add_argument(
        "--frame0", type=str, required=True,
        help="Đường dẫn đến file ảnh Frame 1 (I_0)"
    )
    parser.add_argument(
        "--frame1", type=str, required=True,
        help="Đường dẫn đến file ảnh Frame 2 (I_1)"
    )
    parser.add_argument(
        "--times", type=str, nargs="*", default=None,
        help="Danh sách mốc thời gian t bất kỳ trong khoảng (0.0, 1.0). Ví dụ: --times 0.25 0.5 0.75 hoặc --times 0.2,0.4,0.6"
    )
    parser.add_argument(
        "--num_frames", type=int, default=None,
        help="Số lượng frame in-between muốn chia đều giữa Frame 1 và Frame 2 (nếu không nhập --times)"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./demo/output",
        help="Thư mục lưu các khung hình kết quả (mặc định: ./demo/output)"
    )
    parser.add_argument(
        "--prefix_1", type=int, default=1,
        help="Tiền tố đánh số cho Frame 1 (mặc định: 1 để tạo tên 1.000.png, 1.xxx.png)"
    )
    parser.add_argument(
        "--prefix_2", type=int, default=2,
        help="Tiền tố đánh số cho Frame 2 (mặc định: 2 để tạo tên 2.000.png)"
    )
    parser.add_argument(
        "--digits", type=int, default=3,
        help="Số chữ số thập phân cho tên file (mặc định: 3 -> 1.500.png, 1.250.png)"
    )
    parser.add_argument(
        "--model_size", type=str, default="base", choices=["base", "B", "large", "L", "LARGE"],
        help="Kích thước mô hình uniVIP: 'base'/'B' (uniVIP-B) hoặc 'large'/'L'/'LARGE' (uniVIP-L)"
    )
    parser.add_argument(
        "--model_file", "--checkpoint", dest="model_file", type=str, default="./checkpoints/uniVIP-B.pkl",
        help="Đường dẫn đến file trọng số checkpoint (.pkl)"
    )
    parser.add_argument(
        "--fixed_pyr_level", type=int, default=None,
        help="Cố định số tầng kim tự tháp (nếu để None sẽ tự động tính toán theo độ phân giải ảnh)"
    )

    args = parser.parse_args()
    run_interpolation(args)
