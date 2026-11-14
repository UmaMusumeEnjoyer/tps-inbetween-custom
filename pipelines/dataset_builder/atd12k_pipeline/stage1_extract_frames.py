#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Giai đoạn 1: Trích xuất toàn bộ chuỗi frame 24fps liên tục bằng ffmpeg.
Tên file chuẩn hóa 6 chữ số: 000001.png, 000002.png,...
"""

import os
import sys
import subprocess
import glob
from config import INPUT_VIDEO, RAW_FRAMES_DIR, FPS

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def extract_frames(video_path=INPUT_VIDEO, output_dir=RAW_FRAMES_DIR, fps=FPS):
    print("=" * 70)
    print("🚀 [GIAI ĐOẠN 1]: Trích xuất chuỗi frame 24fps liên tục bằng FFmpeg")
    print(f"   - Video nguồn: {video_path}")
    print(f"   - Thư mục đích: {output_dir}")
    print(f"   - Frame rate: {fps} fps")
    print("=" * 70)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Không tìm thấy video tại: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Kiểm tra xem đã có frame chưa để tránh chạy lại không cần thiết
    existing_frames = glob.glob(os.path.join(output_dir, "*.png"))
    if len(existing_frames) > 500:
        print(f"[*] Thư mục đã có {len(existing_frames)} frame sẵn có. Bỏ qua trích xuất lại.")
        return len(existing_frames)

    output_pattern = os.path.join(output_dir, "%06d.png")
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-vsync", "0",
        output_pattern
    ]

    print(f"[*] Thực thi: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[!] Lỗi FFmpeg:\n{res.stderr}")
        raise RuntimeError("Trích xuất frame bằng FFmpeg thất bại.")

    total_extracted = len(glob.glob(os.path.join(output_dir, "*.png")))
    print(f"[✓] Hoàn thành! Đã trích xuất thành công {total_extracted} frame 24fps.\n")
    return total_extracted


if __name__ == '__main__':
    extract_frames()
