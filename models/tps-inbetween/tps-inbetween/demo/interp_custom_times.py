import os
import sys
import shutil
import tempfile
import cv2
import torch
import argparse
import numpy as np
from torch.nn import functional as F

# Đảm bảo thư mục gốc của tps-inbetween nằm trong sys.path
TPS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if TPS_ROOT not in sys.path:
    sys.path.insert(0, TPS_ROOT)

# Hỗ trợ hiển thị ký tự Unicode tiếng Việt trên Windows terminal
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import warnings
warnings.filterwarnings("ignore")

from model.gluestick import batch_to_np, GLUESTICK_ROOT
from model.gluestick.models.two_view_pipeline import TwoViewPipeline
from model.tpsinbet import TPS_inbet


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


def norm_flow(flow, h, w):
    """
    Chuẩn hóa tọa độ keypoints về tỉ lệ [0, 1]
    """
    flow_norm = flow.copy()
    flow_norm[:, 0] = flow_norm[:, 0] / float(w)
    flow_norm[:, 1] = flow_norm[:, 1] / float(h)
    return flow_norm


def run_interpolation(args):
    # 1. Thiết bị xử lý (GPU / CPU)
    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

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

    h, w = ori_img0.shape[:2]
    print(f"[*] Đã tải 2 ảnh đầu vào thành công. Kích thước gốc: {w}x{h} (WxH)")
    print(f"[*] Thiết bị sử dụng: {device}")

    # 4. Phân tích chuỗi mốc thời gian t
    t_list = parse_time_steps(args.times, args.num_frames)
    # Lọc bỏ t=0.0 và t=1.0 nếu có trong danh sách in-between
    inbetween_t_list = [t for t in t_list if 0.0 < t < 1.0]
    print(f"[*] Danh sách mốc thời gian t cần nội suy ({len(inbetween_t_list)} frames): {inbetween_t_list}")

    # 5. Copy và lưu lại Frame 1 & Frame 2 vào thư mục đầu ra
    frame1_name = f"{args.prefix_1}.{'0' * args.digits}.png"
    frame2_name = f"{args.prefix_2}.{'0' * args.digits}.png"

    frame1_path = os.path.join(save_dir, frame1_name)
    frame2_path = os.path.join(save_dir, frame2_name)

    cv2.imwrite(frame1_path, ori_img0)
    cv2.imwrite(frame2_path, ori_img1)
    print(f"[+] Đã lưu Frame 1 gốc -> {frame1_path}")
    print(f"[+] Đã lưu Frame 2 gốc -> {frame2_path}")

    if len(inbetween_t_list) == 0:
        print("[!] Không có mốc thời gian in-between nào nằm trong khoảng (0.0, 1.0). Kết thúc.")
        return

    # 6. Chuẩn bị ảnh xám (grayscale [0, 1]) cho mô hình
    if len(ori_img0.shape) == 3 and ori_img0.shape[2] == 3:
        gray0 = cv2.cvtColor(ori_img0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(ori_img1, cv2.COLOR_BGR2GRAY)
    else:
        gray0 = ori_img0
        gray1 = ori_img1

    torch_gray0 = (torch.from_numpy(gray0.astype(np.float32) / 255.0).to(device)).unsqueeze(0).unsqueeze(0)
    torch_gray1 = (torch.from_numpy(gray1.astype(np.float32) / 255.0).to(device)).unsqueeze(0).unsqueeze(0)

    # 7. Xử lý padding kích thước ảnh (bội số của 32 cho feature matching và Unet)
    divisor = 32
    pad_h = ((h - 1) // divisor + 1) * divisor
    pad_w = ((w - 1) // divisor + 1) * divisor
    pad_bottom = pad_h - h
    pad_right = pad_w - w

    if pad_bottom > 0 or pad_right > 0:
        torch_gray0_pad = F.pad(torch_gray0, (0, pad_right, 0, pad_bottom), mode="replicate")
        torch_gray1_pad = F.pad(torch_gray1, (0, pad_right, 0, pad_bottom), mode="replicate")
    else:
        torch_gray0_pad = torch_gray0
        torch_gray1_pad = torch_gray1

    # 8. Chạy mô hình GlueStick để tìm khớp điểm (Feature Matching)
    print("====== [1/2] Bắt đầu tìm khớp điểm (GlueStick Matching) ======")
    matcher_weights = args.matcher_weights
    if matcher_weights is None or not os.path.exists(matcher_weights):
        matcher_weights = str(GLUESTICK_ROOT / 'resources' / 'weights' / 'checkpoint_GlueStick_MD.tar')

    if not os.path.exists(matcher_weights):
        # Thử tìm theo đường dẫn tương đối từ TPS_ROOT
        alt_path = os.path.join(TPS_ROOT, 'model', 'resources', 'weights', 'checkpoint_GlueStick_MD.tar')
        if os.path.exists(alt_path):
            matcher_weights = alt_path
        else:
            raise FileNotFoundError(f"Không tìm thấy file trọng số GlueStick tại: {matcher_weights}")

    matching_conf = {
        'name': 'two_view_pipeline',
        'use_lines': True,
        'extractor': {
            'name': 'wireframe',
            'sp_params': {
                'force_num_keypoints': False,
                'max_num_keypoints': 1000,
            },
            'wireframe_params': {
                'merge_points': True,
                'merge_line_endpoints': True,
            },
            'max_n_lines': 300,
        },
        'matcher': {
            'name': 'gluestick',
            'weights': matcher_weights,
            'trainable': False,
        },
        'ground_truth': {
            'from_pose_depth': False,
        }
    }

    matching_model = TwoViewPipeline(matching_conf).to(device).eval()

    with torch.no_grad():
        match_pred = matching_model({'image0': torch_gray0_pad, 'image1': torch_gray1_pad})
        match_pred = batch_to_np(match_pred)

    kp0, kp1 = match_pred["keypoints0"], match_pred["keypoints1"]
    m0 = match_pred["matches0"]

    valid_matches = m0 != -1
    match_indices = m0[valid_matches]
    matched_kps0 = kp0[valid_matches]
    matched_kps1 = kp1[match_indices]

    n_kps0 = norm_flow(matched_kps0, pad_h, pad_w)
    n_kps1 = norm_flow(matched_kps1, pad_h, pad_w)
    kps_stack = np.stack((n_kps0, n_kps1), axis=0).astype(np.float32)
    print(f"[*] Tìm thấy {len(matched_kps0)} cặp điểm khớp đặc trưng.")

    # Sử dụng thư mục tạm để lưu npy chứa keypoints (tự động xóa sau khi xong)
    temp_dir = tempfile.mkdtemp()
    temp_npy_path = os.path.join(temp_dir, 'matching_kps.npy')
    np.save(temp_npy_path, kps_stack)

    try:
        # 9. Tải mô hình TPS-Inbetween
        print("====== [2/2] Bắt đầu nội suy khung hình (TPS-Inbetween) ======")
        checkpoint_path = args.checkpoint if args.checkpoint else args.model_path
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(TPS_ROOT, checkpoint_path)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Không tìm thấy file checkpoint TPS-Inbetween tại: {checkpoint_path}")

        print(f"[*] Đang tải mô hình TPS-Inbetween từ: {checkpoint_path}...")

        class ModelArgs:
            def __init__(self, cpu, xN):
                self.cpu = cpu
                self.xN = xN

        tps_args = ModelArgs(cpu=(device.type == 'cpu'), xN=len(inbetween_t_list) + 1)
        model = TPS_inbet(tps_args)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model = model.to(device).eval()

        # Thiết lập danh sách các mốc thời gian t tùy biến
        model.times = torch.tensor(inbetween_t_list, dtype=torch.float32, device=device)

        # 10. Tiến hành nội suy các frame
        print(f"[*] Bắt đầu tổng hợp {len(inbetween_t_list)} khung hình trung gian...")
        with torch.no_grad():
            preds, _ = model(1.0 - torch_gray0_pad, 1.0 - torch_gray1_pad, [temp_npy_path])

        for idx, (t, pred_tensor) in enumerate(zip(inbetween_t_list, preds)):
            # Đảo ngược màu về ảnh gốc (line drawing)
            pred_img = 1.0 - pred_tensor
            # Crop padding về lại kích thước gốc (h, w)
            pred_img = pred_img[:, :, :h, :w]
            pred_np = (pred_img[0, 0] * 255.0).clamp(0, 255).byte().cpu().numpy()

            if len(ori_img0.shape) == 3 and ori_img0.shape[2] == 3:
                pred_out = cv2.cvtColor(pred_np, cv2.COLOR_GRAY2BGR)
            else:
                pred_out = pred_np

            out_filename = format_inbetween_filename(args.prefix_1, t, digits=args.digits)
            out_filepath = os.path.join(save_dir, out_filename)
            cv2.imwrite(out_filepath, pred_out)
            print(f"    [{idx+1}/{len(inbetween_t_list)}] Hoàn thành t={t:.4f} -> {out_filename}")

    finally:
        # Dọn dẹp thư mục tạm
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\n[✓] Hoàn tất toàn bộ quá trình! Tất cả kết quả đã được lưu tại: {os.path.abspath(save_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TPS-Inbetween Custom Arbitrary Time-step Animation Frame Interpolation Demo"
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
        "--model_size", type=str, default="default",
        help="Kích thước mô hình hoặc biến thể (mặc định: default)"
    )
    parser.add_argument(
        "--checkpoint", "--model_path", dest="checkpoint", type=str, default="./ckpt/model_latest.pt",
        help="Đường dẫn đến file checkpoint TPS-Inbetween (.pt)"
    )
    parser.add_argument(
        "--matcher_weights", type=str, default="./model/resources/weights/checkpoint_GlueStick_MD.tar",
        help="Đường dẫn đến file trọng số của GlueStick matcher"
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Bắt buộc sử dụng CPU thay vì GPU"
    )

    args = parser.parse_args()
    # Hỗ trợ backward compatibility nếu model_path được truyền
    if not hasattr(args, 'model_path'):
        args.model_path = args.checkpoint

    run_interpolation(args)
