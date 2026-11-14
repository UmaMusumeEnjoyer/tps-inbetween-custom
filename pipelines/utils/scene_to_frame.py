from concurrent.futures import ThreadPoolExecutor
import os
import sys
import cv2
import numpy as np

# Kiểm tra PyTorch và GPU CUDA
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False

# Đảm bảo console Windows hiển thị đúng UTF-8 (tiếng Việt)
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def compute_frame_difference_cpu(prev_frame, curr_frame, resize_dim=(320, 180), grid=(4, 4)):
    """Tính toán sai khác giữa 2 frame trên CPU (NumPy)."""
    p_small = cv2.resize(prev_frame, resize_dim, interpolation=cv2.INTER_AREA)
    c_small = cv2.resize(curr_frame, resize_dim, interpolation=cv2.INTER_AREA)
    
    diff = cv2.absdiff(p_small, c_small)
    global_mae = float(np.mean(diff))
    
    h, w, _ = diff.shape
    gh, gw = grid
    bh, bw = h // gh, w // gw
    
    max_block_mae = 0.0
    for r in range(gh):
        for c in range(gw):
            block = diff[r * bh : (r + 1) * bh, c * bw : (c + 1) * bw]
            b_mean = float(np.mean(block))
            if b_mean > max_block_mae:
                max_block_mae = b_mean
                
    combined_score = global_mae * 0.4 + max_block_mae * 0.6
    return combined_score, global_mae, max_block_mae


def compute_frame_difference_gpu(prev_tensor, curr_frame, device, resize_dim=(320, 180), grid=(4, 4)):
    """
    Tính toán sai khác giữa 2 frame cực nhanh trên GPU (PyTorch CUDA Tensor).
    - Toàn bộ phép tính MAE và Adaptive Pooling Grid chạy song song trên CUDA cores.
    """
    # Resize nhanh trên CPU trước khi nạp GPU tensor (hoặc interpolate)
    c_small = cv2.resize(curr_frame, resize_dim, interpolation=cv2.INTER_AREA)
    
    # Chuyển BGR (H, W, C) -> Tensor (1, C, H, W) trên GPU
    curr_tensor = torch.from_numpy(c_small).to(device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    
    # 1. Sai khác tuyệt đối giữa 2 tensor trên GPU
    diff = torch.abs(curr_tensor - prev_tensor)
    global_mae = diff.mean().item()
    
    # 2. Sai khác cục bộ theo Grid bằng Adaptive Avg Pooling trên GPU
    diff_gray = diff.mean(dim=1, keepdim=True)  # (1, 1, H, W)
    pooled = F.adaptive_avg_pool2d(diff_gray, grid)  # (1, 1, gh, gw)
    max_block_mae = pooled.max().item()
    
    combined_score = global_mae * 0.4 + max_block_mae * 0.6
    return combined_score, global_mae, max_block_mae, curr_tensor


def analyze_and_extract_frames(video_path, threshold=8.0, output_dir=None, min_interval_s=0.1, img_ext=".png", device_mode="auto"):
    """
    Trích xuất các frame khác nhau từ video và lưu vào thư mục với hỗ trợ tăng tốc GPU.
    
    Args:
        video_path (str): Đường dẫn tới file video.
        threshold (float): Ngưỡng sai khác tổng hợp (mặc định 8.0). Càng nhỏ càng nhạy.
        output_dir (str, optional): Thư mục lưu các frame. Mặc định là thư mục mang tên video.
        min_interval_s (float): Khoảng cách thời gian tối thiểu giữa các frame được ghi nhận (mặc định 0.1s).
        img_ext (str): Đuôi mở rộng của ảnh cần lưu (vd: '.png', '.jpg').
        device_mode (str): Thiết bị tính toán ('auto', 'cuda', 'cpu').
    """
    if not os.path.exists(video_path):
        print(f"Lỗi: Không tìm thấy file video tại '{video_path}'")
        return

    # Xác định thiết bị tính toán (GPU / CPU)
    use_gpu = False
    device = None
    if device_mode.lower() in ["auto", "cuda"] and HAS_CUDA:
        use_gpu = True
        device = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(0)
        print(f"-> Sử dụng GPU: {device_name} (CUDA)")
    else:
        print("-> Sử dụng CPU (NumPy)")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Lỗi: Không thể mở video.")
        return

    # Xác định thư mục đầu ra
    if output_dir is None:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(os.path.dirname(video_path) or ".", video_name)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Thư mục lưu kết quả: {os.path.abspath(output_dir)}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video dài {duration:.2f}s | FPS: {fps:.2f} | Tổng số frame video: {total_frames} | Ngưỡng nhạy: {threshold}")

    # Đọc frame đầu tiên làm gốc (Frame 0.0s)
    ret, prev_frame = cap.read()
    if not ret:
        print("Lỗi: Không thể đọc frame đầu tiên của video.")
        cap.release()
        return
        
    # Khởi tạo ThreadPoolExecutor để ghi file ảnh nền bất đồng bộ (không nghẽn CPU/GPU)
    io_executor = ThreadPoolExecutor(max_workers=4)

    first_frame_name = f"0.0s{img_ext}"
    first_frame_path = os.path.join(output_dir, first_frame_name)
    io_executor.submit(cv2.imwrite, first_frame_path, prev_frame.copy())
    saved_count = 1
    print(f"[Thời gian: 0.0s] Đã lưu Frame Gốc đầu tiên -> {first_frame_name}")

    # Chuẩn bị frame gốc trên GPU nếu có
    prev_tensor = None
    if use_gpu:
        p_small = cv2.resize(prev_frame, (320, 180), interpolation=cv2.INTER_AREA)
        prev_tensor = torch.from_numpy(p_small).to(device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    # Quản lý timeline
    timeline = []
    current_frame_name = first_frame_name
    current_frame_start_s = 0.0
    last_saved_time_s = 0.0

    frame_idx = 0
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        current_time_s = frame_idx / fps if fps > 0 else 0.0
        
        # Chỉ kiểm tra nếu cách frame đã lưu trước đó ít nhất min_interval_s (0.1s)
        if (current_time_s - last_saved_time_s) < (min_interval_s - 0.02):
            continue

        # Tính độ sai khác
        if use_gpu:
            score, g_mae, b_mae, curr_tensor = compute_frame_difference_gpu(
                prev_tensor, curr_frame, device=device
            )
        else:
            score, g_mae, b_mae = compute_frame_difference_cpu(prev_frame, curr_frame)
        
        if score >= threshold:
            # Phát hiện frame mới có sự khác biệt
            timestamp_sec_rounded = round(current_time_s, 1)
            
            # Đảm bảo không trùng mốc thời gian với frame trước
            if timestamp_sec_rounded <= current_frame_start_s:
                timestamp_sec_rounded = round(current_frame_start_s + min_interval_s, 1)

            prev_duration_s = round(timestamp_sec_rounded - current_frame_start_s, 1)
            
            # Ghi nhận frame trước đó
            timeline.append({
                "frame_name": current_frame_name,
                "start_s": current_frame_start_s,
                "end_s": timestamp_sec_rounded,
                "duration_s": prev_duration_s
            })

            # Lưu frame mới bằng thread pool bất đồng bộ
            frame_filename = f"{timestamp_sec_rounded:.1f}s{img_ext}"
            frame_filepath = os.path.join(output_dir, frame_filename)
            io_executor.submit(cv2.imwrite, frame_filepath, curr_frame.copy())
            saved_count += 1
            print(f"[Thời gian: {timestamp_sec_rounded:.1f}s] Đã lưu Frame MỚI (Điểm lệch: {score:.2f}, Vùng lớn nhất: {b_mae:.2f}) -> {frame_filename}")

            # Cập nhật frame gốc mới để so sánh
            if use_gpu:
                prev_tensor = curr_tensor
            else:
                prev_frame = curr_frame.copy()
                
            current_frame_name = frame_filename
            current_frame_start_s = timestamp_sec_rounded
            last_saved_time_s = current_time_s

    # Ghi nhận frame cuối cùng đến hết thời lượng video
    final_end_s = round(max(duration, current_frame_start_s + min_interval_s), 1)
    final_duration_s = round(max(min_interval_s, final_end_s - current_frame_start_s), 1)
    timeline.append({
        "frame_name": current_frame_name,
        "start_s": current_frame_start_s,
        "end_s": final_end_s,
        "duration_s": final_duration_s
    })

    cap.release()

    # Đợi tất cả các tác vụ lưu ảnh hoàn tất
    io_executor.shutdown(wait=True)

    # Tạo và lưu file .txt tổng hợp timeline
    txt_path = os.path.join(output_dir, "timestamps.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("                   TỔNG HỢP TIMELINE CÁC FRAME TRONG VIDEO\n")
        f.write("=" * 80 + "\n")
        f.write(f"File video nguồn     : {os.path.basename(video_path)}\n")
        f.write(f"Tổng thời lượng      : {duration:.2f}s (~{final_end_s:.1f}s)\n")
        f.write(f"FPS                  : {fps:.2f}\n")
        f.write(f"Tổng số frame video  : {total_frames}\n")
        f.write(f"Thiết bị tính toán   : {'GPU (' + torch.cuda.get_device_name(0) + ')' if use_gpu else 'CPU'}\n")
        f.write(f"Bước chia thời gian  : {min_interval_s:.1f}s\n")
        f.write(f"Ngưỡng nhạy          : {threshold}\n")
        f.write(f"Tổng số frame độc lập: {len(timeline)}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"{'STT':<6}{'Tên Frame':<16}{'Bắt đầu':<12}{'Kết thúc':<12}{'Thời lượng lặp lại':<20}\n")
        f.write("-" * 80 + "\n")
        for idx, item in enumerate(timeline, start=1):
            start_str = f"{item['start_s']:.1f}s"
            end_str = f"{item['end_s']:.1f}s"
            dur_str = f"{item['duration_s']:.1f}s"
            f.write(f"{idx:<6}{item['frame_name']:<16}{start_str:<12}{end_str:<12}{dur_str:<20}\n")
        
        f.write("-" * 80 + "\n\n")
        f.write("DANH SÁCH CHI TIẾT:\n")
        for idx, item in enumerate(timeline, start=1):
            f.write(f"  {idx}. {item['frame_name']}: {item['start_s']:.1f}s -> {item['end_s']:.1f}s (tồn tại/lặp lại trong {item['duration_s']:.1f}s)\n")

    print(f"\nHoàn tất! Tổng cộng đã lưu {saved_count} frame vào thư mục '{output_dir}'.")
    print(f"Đã tạo file tổng hợp timestamp: '{txt_path}'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trích xuất các frame khác nhau từ video theo thời gian thực tế với GPU.")
    parser.add_argument("video_path", nargs="?", default=None, help="Đường dẫn đến file video")
    parser.add_argument("--threshold", type=float, default=8.0, help="Ngưỡng sai khác để phân biệt frame mới (mặc định: 8.0)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Thiết bị tính toán (mặc định: auto - tự động nhận diện GPU)")
    parser.add_argument("--output_dir", type=str, default=None, help="Thư mục lưu frame (mặc định: tạo thư mục trùng tên video)")
    parser.add_argument("--step", type=float, default=0.1, help="Độ chia thời gian tối thiểu tính bằng giây (mặc định: 0.1s)")
    parser.add_argument("--ext", type=str, default=".png", choices=[".png", ".jpg", ".jpeg"], help="Định dạng ảnh lưu (mặc định: .png)")

    args = parser.parse_args()

    if args.video_path:
        analyze_and_extract_frames(
            video_path=args.video_path,
            threshold=args.threshold,
            output_dir=args.output_dir,
            min_interval_s=args.step,
            img_ext=args.ext,
            device_mode=args.device
        )
    else:
        print("Cách sử dụng:")
        print("  python scene_to_frame.py <duong_dan_video> [--threshold 8.0] [--device auto|cuda|cpu] [--output_dir <thu_muc>] [--ext .png]")
        print("Ví dụ:")
        print("  python scene_to_frame.py video_mau.mp4 --device cuda")

