import cv2
import os
import glob
import re
import sys
import time

# Đảm bảo console Windows hiển thị đúng UTF-8 (tiếng Việt)
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def get_timestamp(filepath):
    """
    Trích xuất timestamp từ tên file ảnh.
    Hỗ trợ các định dạng:
      - '0.0.png', '12.5.jpg'
      - '0.0s.png', '10.3s.png'
      - 'frame_0.5s.png', 'img_12.png'
    """
    filename = os.path.basename(filepath)
    name_without_ext = os.path.splitext(filename)[0]

    # 1. Thử parse trực tiếp dạng float ("0.0", "12.5")
    try:
        return float(name_without_ext)
    except ValueError:
        pass

    # 2. Thử bỏ hậu tố 's' / 'sec' ("0.0s", "10.3s")
    cleaned = name_without_ext.rstrip("sS").strip()
    try:
        return float(cleaned)
    except ValueError:
        pass

    # 3. Dùng regex tìm số thực hoặc số nguyên đầu tiên
    match = re.search(r"[-+]?\d*\.?\d+", name_without_ext)
    if match:
        try:
            return float(match.group())
        except ValueError:
            pass

    return -1.0


def create_video_from_timestamped_images(image_folder, output_video, fps=30):
    """
    Tạo video từ danh sách ảnh có timestamp trong thư mục.
    """
    image_folder = os.path.abspath(image_folder)
    if not os.path.exists(image_folder):
        print(f"❌ Lỗi: Thư mục '{image_folder}' không tồn tại!")
        return False

    # 1. Lấy danh sách các file ảnh (hỗ trợ png, jpg, jpeg, webp, bmp)
    raw_files = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.webp', '*.bmp', '*.PNG', '*.JPG', '*.JPEG'):
        raw_files.extend(glob.glob(os.path.join(image_folder, ext)))

    # Loại bỏ file trùng lặp (tránh bị double do Windows case-insensitive)
    image_files = list({os.path.normpath(f): f for f in raw_files}.values())

    if not image_files:
        print(f"❌ Không tìm thấy ảnh nào trong thư mục: '{image_folder}'")
        return False

    # 2. Trích xuất thời gian và sắp xếp file từ nhỏ đến lớn
    valid_files = []
    for f in image_files:
        ts = get_timestamp(f)
        if ts >= 0:
            valid_files.append((ts, f))

    # Sắp xếp dựa trên giá trị thời gian timestamp
    valid_files.sort(key=lambda x: x[0])

    if not valid_files:
        print("❌ Không có file ảnh nào có tên chứa timestamp hợp lệ.")
        return False

    print(f"\n📂 Thư mục nguồn      : {image_folder}")
    print(f"🖼️  Số lượng frame hợp lệ: {len(valid_files)}")
    print(f"⏱️  Timestamp đầu - cuối : {valid_files[0][0]:.2f}s -> {valid_files[-1][0]:.2f}s")
    print(f"🎯 Video đầu ra        : {output_video}")
    print(f"⚡ Tốc độ khung hình   : {fps} FPS")

    # 3. Khởi tạo VideoWriter với kích thước của ảnh đầu tiên
    first_image = cv2.imread(valid_files[0][1])
    if first_image is None:
        print(f"❌ Không thể đọc ảnh đầu tiên: {valid_files[0][1]}")
        return False

    height, width, layers = first_image.shape
    size = (width, height)

    # Tạo thư mục cha cho file output nếu chưa có
    out_dir = os.path.dirname(os.path.abspath(output_video))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, size)

    if not out.isOpened():
        print(f"❌ Không thể khởi tạo VideoWriter để ghi file '{output_video}'!")
        return False

    print(f"\n🚀 Bắt đầu tạo video ({width}x{height}, {fps} FPS)...")
    start_proc_time = time.time()
    total_frames_written = 0

    # 4. Xử lý từng ảnh và lặp lại frame dựa trên độ trễ
    for i in range(len(valid_files)):
        current_timestamp, img_path = valid_files[i]

        # Tính toán thời gian tồn tại của ảnh này trên video
        if i < len(valid_files) - 1:
            next_timestamp = valid_files[i + 1][0]
            duration = max(0.0, next_timestamp - current_timestamp)
        else:
            # Ảnh cuối cùng mặc định hiển thị 0.1 giây (hoặc 1/fps)
            duration = 0.1

        # Quy đổi thời lượng sang số frame (làm tròn)
        frames_to_write = int(round(duration * fps))

        # Đảm bảo ít nhất 1 frame cho ảnh nếu thời lượng quá ngắn nhưng > 0
        if frames_to_write <= 0 and duration > 0:
            frames_to_write = 1

        if frames_to_write <= 0:
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Cảnh báo: Không thể đọc ảnh {img_path}, bỏ qua.")
            continue

        # Đảm bảo ảnh cùng kích thước
        if (img.shape[1], img.shape[0]) != size:
            img = cv2.resize(img, size)

        # Ghi ảnh vào video
        for _ in range(frames_to_write):
            out.write(img)

        total_frames_written += frames_to_write

        file_name = os.path.basename(img_path)
        print(f"  [{i+1}/{len(valid_files)}] {file_name:<16} -> giữ trong {duration:.2f}s ({frames_to_write} frames)")

    # 5. Giải phóng bộ nhớ
    out.release()
    elapsed = time.time() - start_proc_time
    video_duration = total_frames_written / fps if fps > 0 else 0

    print("=" * 60)
    print(f"✅ HOÀN TẤT!")
    print(f"📁 Video đã lưu tại : {os.path.abspath(output_video)}")
    print(f"🎞️ Tổng số frame ghi : {total_frames_written} frames (~{video_duration:.2f}s)")
    print(f"⏳ Thời gian xử lý   : {elapsed:.2f}s")
    print("=" * 60)
    return True


def parse_arguments_and_run():
    """
    Xử lý nhận đường dẫn từ tham số dòng lệnh hoặc nhập trực tiếp từ bàn phím.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Ghép các ảnh frame có timestamp thành video hoàn chỉnh."
    )
    parser.add_argument("folder", nargs="?", default=None, help="Đường dẫn thư mục chứa ảnh")
    parser.add_argument("-o", "--output", default=None, help="Đường dẫn file video đầu ra (.mp4)")
    parser.add_argument("--fps", type=int, default=30, help="Tốc độ khung hình FPS (mặc định: 30)")

    args = parser.parse_args()

    folder = args.folder

    # Nếu không truyền đường dẫn qua CLI, nhập trực tiếp từ bàn phím
    if not folder:
        print("=" * 60)
        print("       CÔNG CỤ GHÉP FRAME THÀNH VIDEO (FRAME TO VIDEO)")
        print("=" * 60)
        while True:
            try:
                user_in = input("\n👉 Nhập đường dẫn thư mục chứa ảnh (hoặc kéo thả folder vào đây): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nĐã hủy thao tác.")
                sys.exit(0)

            # Xóa dấu ngoặc kép / đơn nếu kéo thả trên Windows
            user_in = user_in.strip('"\'')
            if not user_in:
                print("⚠️ Vui lòng không để trống đường dẫn!")
                continue
            if not os.path.isdir(user_in):
                print(f"❌ Thư mục '{user_in}' không tồn tại. Vui lòng nhập lại!")
                continue
            folder = user_in
            break

    # Xử lý đường dẫn file video đầu ra
    output = args.output
    if not output:
        folder_clean = os.path.normpath(folder)
        folder_name = os.path.basename(folder_clean) or "output"
        default_output = f"{folder_name}_reconstructed.mp4"

        # Nếu đang ở chế độ nhập từ bàn phím (không truyền qua CLI)
        if not args.folder:
            try:
                out_in = input(f"👉 Nhập tên/đường dẫn file video đầu ra (nhấn Enter để dùng mặc định '{default_output}'): ").strip().strip('"\'')
            except (EOFError, KeyboardInterrupt):
                out_in = ""
            output = out_in if out_in else default_output
        else:
            output = default_output

    # Đảm bảo đuôi mở rộng là .mp4
    if not os.path.splitext(output)[1]:
        output += ".mp4"

    # Xử lý FPS
    fps = args.fps
    if not args.folder:
        try:
            fps_in = input(f"👉 Nhập FPS video (nhấn Enter để dùng mặc định {fps}): ").strip()
            if fps_in.isdigit() and int(fps_in) > 0:
                fps = int(fps_in)
        except (EOFError, KeyboardInterrupt):
            pass

    create_video_from_timestamped_images(folder, output, fps=fps)


if __name__ == "__main__":
    parse_arguments_and_run()