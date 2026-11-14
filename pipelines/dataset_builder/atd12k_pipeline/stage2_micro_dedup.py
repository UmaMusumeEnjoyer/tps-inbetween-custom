#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Giai đoạn 2: Lọc trùng lặp vi mô (Micro-Deduplication - Chuẩn ATD-12K) tăng tốc bằng CUDA.
- Đo PSNR và Sobel Edge Difference trên GPU.
- Phân biệt chính xác giữa frame bị lặp lại (hold/freeze) và frame inbetween có cử động nhỏ.
- Xuất danh sách Unique Drawings kèm Timeline chi tiết.
"""

import os
import sys
import glob
import json
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from config import (
    RAW_FRAMES_DIR, DRAWINGS_DIR, DEVICE,
    DEDUP_PSNR_THRESH, DEDUP_EDGE_DIFF_THRESH, DEDUP_MAX_HOLD_FRAMES
)

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


class CudaDedupEngine:
    def __init__(self, device=DEVICE):
        self.device = torch.device(device)
        print(f"[*] Khởi tạo CudaDedupEngine trên thiết bị: {self.device}")

        # Sobel kernel chuẩn 3x3 để phát hiện biên nét
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]], dtype=torch.float32, device=self.device).view(1, 1, 3, 3) / 4.0
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                                [ 0.0,  0.0,  0.0],
                                [ 1.0,  2.0,  1.0]], dtype=torch.float32, device=self.device).view(1, 1, 3, 3) / 4.0

        self.sobel_x = sobel_x
        self.sobel_y = sobel_y

    def image_to_tensor(self, img_path):
        """Đọc ảnh từ đĩa, chuyển thành Grayscale Tensor [1, 1, H, W] trên GPU chuẩn hóa [0, 1]."""
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Không thể đọc ảnh: {img_path}")
        tensor = torch.from_numpy(img).float().to(self.device).unsqueeze(0).unsqueeze(0) / 255.0
        return tensor

    def compute_edge_map(self, gray_tensor):
        """Tính ma trận biên nét Sobel trên GPU bằng conv2d phản chiếu."""
        pad = F.pad(gray_tensor, (1, 1, 1, 1), mode='replicate')
        gx = F.conv2d(pad, self.sobel_x)
        gy = F.conv2d(pad, self.sobel_y)
        edge = torch.sqrt(gx * gx + gy * gy + 1e-8)
        return edge

    def compare_frames(self, tensor1, edge1, tensor2):
        """
        So sánh sai khác giữa 2 frame trên GPU:
        - PSNR toàn khung hình
        - Độ lệch biên nét Sobel (Delta Edge)
        """
        # 1. PSNR
        mse = torch.mean((tensor1 - tensor2) ** 2)
        if mse.item() < 1e-10:
            psnr = 100.0
        else:
            psnr = (10.0 * torch.log10(1.0 / mse)).item()

        # 2. Edge Difference
        edge2 = self.compute_edge_map(tensor2)
        delta_edge = (torch.mean(torch.abs(edge1 - edge2)) * 255.0).item()

        return psnr, delta_edge, edge2


def run_micro_dedup(raw_dir=RAW_FRAMES_DIR, output_dir=DRAWINGS_DIR):
    print("=" * 70)
    print("🚀 [GIAI ĐOẠN 2]: Lọc trùng lặp vi mô (Micro-Deduplication - Chuẩn ATD-12K) trên CUDA")
    print(f"   - Ngưỡng PSNR trùng lặp: >= {DEDUP_PSNR_THRESH} dB")
    print(f"   - Ngưỡng lệch biên Sobel: < {DEDUP_EDGE_DIFF_THRESH}")
    print(f"   - Giới hạn hold tối đa: {DEDUP_MAX_HOLD_FRAMES} frames")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)
    frame_files = sorted(glob.glob(os.path.join(raw_dir, "*.png")))
    total_frames = len(frame_files)
    if total_frames < 3:
        raise ValueError(f"Số lượng frame trong '{raw_dir}' quá ít ({total_frames})!")

    engine = CudaDedupEngine(device=DEVICE)

    unique_drawings = []
    
    # Khởi tạo bản vẽ đầu tiên D_0
    anchor_idx = 1
    anchor_file = os.path.basename(frame_files[0])
    anchor_tensor = engine.image_to_tensor(frame_files[0])
    anchor_edge = engine.compute_edge_map(anchor_tensor)
    current_duration = 1

    unique_drawings.append({
        "drawing_id": 0,
        "frame_idx": anchor_idx,
        "frame_file": anchor_file,
        "duration": 1
    })

    print(f"[*] Bắt đầu quét qua {total_frames} frame video trên GPU CUDA...")

    for i in tqdm(range(1, total_frames), desc="Micro-Deduplication (CUDA)"):
        curr_file = os.path.basename(frame_files[i])
        curr_idx = i + 1  # 1-indexed
        curr_tensor = engine.image_to_tensor(frame_files[i])

        # So sánh frame hiện tại với anchor frame của bản vẽ đang giữ
        psnr, delta_edge, curr_edge = engine.compare_frames(anchor_tensor, anchor_edge, curr_tensor)

        # Điều kiện coi là trùng lặp (duplicate):
        # PSNR rất cao (ít thay đổi) VÀ độ lệch nét Sobel rất nhỏ (không có chuyển động nét mới)
        is_duplicate = (psnr >= DEDUP_PSNR_THRESH and delta_edge < DEDUP_EDGE_DIFF_THRESH) or (psnr >= 48.0)

        # Tránh giữ một hình vẽ quá lâu (> DEDUP_MAX_HOLD_FRAMES)
        if is_duplicate and current_duration < DEDUP_MAX_HOLD_FRAMES:
            # Cùng một bản vẽ được giữ thêm 1 frame
            current_duration += 1
            unique_drawings[-1]["duration"] = current_duration
        else:
            # Phát hiện bản vẽ mới (Unique Drawing)
            anchor_idx = curr_idx
            anchor_file = curr_file
            anchor_tensor = curr_tensor
            anchor_edge = curr_edge
            current_duration = 1

            new_id = len(unique_drawings)
            unique_drawings.append({
                "drawing_id": new_id,
                "frame_idx": anchor_idx,
                "frame_file": anchor_file,
                "duration": 1
            })

    output_json = os.path.join(output_dir, "timeline_drawings.json")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(unique_drawings, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("📊 KẾT QUẢ MICRO-DEDUPLICATION (ATD-12K):")
    print(f"   - Tổng số frame video 24fps gốc: {total_frames}")
    print(f"   - Số lượng bản vẽ thực tế (Unique Drawings): {len(unique_drawings)}")
    dedup_ratio = (1.0 - len(unique_drawings) / total_frames) * 100.0
    print(f"   - Tỷ lệ trùng lặp đã loại bỏ: {dedup_ratio:.2f}%")
    avg_hold = sum(d['duration'] for d in unique_drawings) / len(unique_drawings)
    print(f"   - Thời lượng giữ bản vẽ trung bình: {avg_hold:.2f} frame (~{avg_hold/24.0:.3f}s)")
    print(f"   - Đã lưu metadata vào: {output_json}")
    print("=" * 70 + "\n")

    return unique_drawings


if __name__ == '__main__':
    run_micro_dedup()
