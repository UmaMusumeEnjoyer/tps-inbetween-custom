import os
import sys
import shutil
import cv2
import torch
import argparse
import numpy as np
import math
from torch.nn import functional as F

# Đảm bảo thư mục gốc của AnimeInterp nằm trong sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Hỗ trợ hiển thị ký tự Unicode tiếng Việt trên Windows terminal
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
        sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass

import warnings
warnings.filterwarnings("ignore")

from models.AnimeInterp_no_cupy import AnimeInterpNoCupy
try:
    from models.AnimeInterp import AnimeInterp
    CUPY_AVAILABLE = True
except Exception:
    CUPY_AVAILABLE = False


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
                        print(f"[Cảnh báo] Mốc thời gian t={val} nằm ngoài khoảng [0.0, 1.0], bỏ qua.", flush=True)
                except ValueError:
                    print(f"[Cảnh báo] Không thể đọc mốc thời gian '{p}', bỏ qua.", flush=True)
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


def load_animeinterp_model(checkpoint_path, device, prefer_cupy=True):
    """
    Khởi tạo mô hình AnimeInterp và nạp checkpoint trọng số.
    Tự động ưu tiên CuPy SoftSplat nếu có CUDA để tăng tốc độ xử lý,
    fallback về pure PyTorch ForwardWarp nếu CuPy không khả dụng hoặc khi chạy trên CPU.
    """
    model = None
    model_name = ""

    if prefer_cupy and CUPY_AVAILABLE and device.type == 'cuda':
        try:
            model = AnimeInterp(path=None)
            model_name = "AnimeInterp (CuPy SoftSplat Tăng Tốc)"
        except Exception as e:
            print(f"[!] Không thể khởi tạo CuPy SoftSplat ({e}). Chuyển sang PyTorch ForwardWarp...", flush=True)
            model = None

    if model is None:
        model = AnimeInterpNoCupy(path=None)
        model_name = "AnimeInterp (PyTorch ForwardWarp)"

    model = model.to(device)
    model.eval()

    if os.path.exists(checkpoint_path):
        print(f"[*] Đang tải trọng số từ: {checkpoint_path}...", flush=True)
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint_data, dict):
            if 'model_state_dict' in checkpoint_data:
                state_dict = checkpoint_data['model_state_dict']
            elif 'state_dict' in checkpoint_data:
                state_dict = checkpoint_data['state_dict']
            else:
                state_dict = checkpoint_data
        else:
            state_dict = checkpoint_data

        # Loại bỏ tiền tố 'module.' nếu checkpoint được lưu từ DataParallel
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                clean_state_dict[k[7:]] = v
            else:
                clean_state_dict[k] = v

        model.load_state_dict(clean_state_dict, strict=False)
        print(f"[+] Tải trọng số mô hình thành công.", flush=True)
    else:
        print(f"[!] CẢNH BÁO: Không tìm thấy checkpoint tại '{checkpoint_path}'.", flush=True)
        print(f"    Mô hình sẽ chạy với trọng số khởi tạo ngẫu nhiên.", flush=True)
        print(f"    Để có kết quả tốt nhất, hãy tải 'anime_interp_full.ckpt' về thư mục '{os.path.dirname(checkpoint_path)}'.", flush=True)

    return model, model_name


def run_interpolation(args):
    # 1. Xác định thiết bị tính toán
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Thiết bị sử dụng: {device}", flush=True)

    # 2. Tạo thư mục lưu kết quả đầu ra
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # 3. Đọc và kiểm tra 2 ảnh đầu vào
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
    print(f"[*] Đã tải 2 ảnh đầu vào thành công. Kích thước gốc: {w}x{h} (WxH)", flush=True)

    # 4. Phân tích chuỗi mốc thời gian t
    t_list = parse_time_steps(args.times, args.num_frames)
    print(f"[*] Danh sách mốc thời gian t cần nội suy ({len(t_list)} frames): {t_list}", flush=True)

    # 5. Copy và lưu lại Frame 1 & Frame 2 vào thư mục đầu ra
    frame1_name = f"{args.prefix_1}.{'0' * args.digits}.png"
    frame2_name = f"{args.prefix_2}.{'0' * args.digits}.png"

    frame1_path = os.path.join(save_dir, frame1_name)
    frame2_path = os.path.join(save_dir, frame2_name)

    cv2.imwrite(frame1_path, ori_img0)
    cv2.imwrite(frame2_path, ori_img1)
    print(f"[+] Đã lưu Frame 1 gốc -> {frame1_path}", flush=True)
    print(f"[+] Đã lưu Frame 2 gốc -> {frame2_path}", flush=True)

    # 6. Khởi tạo mô hình
    checkpoint_file = args.checkpoint if args.checkpoint is not None else args.model_file
    prefer_cupy = not args.no_cupy
    model, model_name = load_animeinterp_model(checkpoint_file, device, prefer_cupy=prefer_cupy)
    print(f"[*] Kiến trúc mô hình: {model_name}", flush=True)

    # 7. Chuẩn bị tensor ảnh đầu vào (chuyển từ BGR sang RGB và scale về [0, 1])
    img0_rgb = cv2.cvtColor(ori_img0, cv2.COLOR_BGR2RGB)
    img1_rgb = cv2.cvtColor(ori_img1, cv2.COLOR_BGR2RGB)

    img0_tensor = (torch.tensor(img0_rgb.transpose(2, 0, 1), dtype=torch.float32) / 255.0).unsqueeze(0).to(device)
    img1_tensor = (torch.tensor(img1_rgb.transpose(2, 0, 1), dtype=torch.float32) / 255.0).unsqueeze(0).to(device)

    # Padding kích thước ảnh lên bội số của 32 (phù hợp với RFR và GridNet)
    divisor = 32
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor

    if pad_h > 0 or pad_w > 0:
        # F.pad format: (left, right, top, bottom)
        img0_tensor = F.pad(img0_tensor, (0, pad_w, 0, pad_h), mode='replicate')
        img1_tensor = F.pad(img1_tensor, (0, pad_w, 0, pad_h), mode='replicate')

    # 8. Tiến hành nội suy từng mốc thời gian t
    print(f"[*] Bắt đầu quá trình nội suy các khung hình trung gian (In-between frames)...", flush=True)
    for idx, t in enumerate(t_list):
        if t == 0.0 or t == 1.0:
            # t = 0.0 hoặc 1.0 tương ứng Frame gốc đã lưu
            continue

        with torch.no_grad():
            outputs = model(img0_tensor, img1_tensor, None, None, float(t))
            pred_tensor = outputs[0]

        # Cắt về kích thước gốc ban đầu (crop padding)
        pred_tensor = pred_tensor[:, :, :h, :w]
        pred_np = (pred_tensor[0] * 255.0).clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
        pred_bgr = cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)

        # Định dạng tên file: {prefix_1}.{xxx}.png
        out_filename = format_inbetween_filename(args.prefix_1, t, digits=args.digits)
        out_filepath = os.path.join(save_dir, out_filename)
        cv2.imwrite(out_filepath, pred_bgr)
        print(f"    [{idx+1}/{len(t_list)}] Hoàn thành t={t:.4f} -> {out_filename}", flush=True)

    print(f"\n[✓] Hoàn tất toàn bộ quá trình! Tất cả kết quả đã được lưu tại: {os.path.abspath(save_dir)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AnimeInterp Custom Arbitrary Time-step Video Frame Interpolation Demo"
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
        help="Danh sách mốc thời gian t bất kỳ trong khoảng (0.0, 1.0). Ví dụ: --times 0.2 0.25 0.32 0.5 hoặc --times 0.2,0.4,0.6"
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
        help="Số chữ số thập phân cho tên file (mặc định: 3 -> 1.500.png, 1.250.png, 1.067.png)"
    )
    parser.add_argument(
        "--model_size", "--model_type", dest="model_size", type=str, default="anime_interp",
        help="Kiểu mô hình AnimeInterp (mặc định: anime_interp)"
    )
    parser.add_argument(
        "--checkpoint", "--model_file", dest="checkpoint", type=str, default="./checkpoints/anime_interp_full.ckpt",
        help="Đường dẫn đến file checkpoint trọng số (.ckpt)"
    )
    parser.add_argument(
        "--no_cupy", action="store_true",
        help="Tắt CuPy SoftSplat và chỉ sử dụng PyTorch ForwardWarp thuần"
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Bắt buộc chạy trên CPU (mặc định tự động nhận diện GPU nếu có)"
    )

    args = parser.parse_args()
    args.model_file = args.checkpoint
    run_interpolation(args)
