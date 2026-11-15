#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Giai đoạn 5: Tiền xử lý Line Art chuẩn ml240 (Segmentation -> AniLines -> Smooth).
- Tách nền bằng ISNetDIS (anime-segmentation) trên CUDA.
- Trích xuất line art bằng AniLines (mode detail) trên CUDA.
- Làm mượt nét bằng MS-SSAA Lanczos-4 + Bilateral Filter (smooth_lines.py).
- Chỉ xử lý các frame xuất hiện trong tập triplet đã đạt chuẩn để tối ưu thời gian tính toán.
"""

import os
import sys
import json
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from config import (
    SCREENED_TRIPLETS_FILE, LINEART_CACHE_DIR,
    PBL6_DIR, DEVICE, ANILINES_WEIGHTS, ANIMESEG_CKPT
)

# Thêm đường dẫn AniLines và anime-segmentation vào sys.path
ANILINES_DIR = os.path.join(PBL6_DIR, 'third_party', 'lineart_tools', 'AniLines-Anime-Lineart-Extractor')
if ANILINES_DIR not in sys.path:
    sys.path.insert(0, ANILINES_DIR)

ANIMESEG_DIR = os.path.join(PBL6_DIR, 'third_party', 'lineart_tools', 'anime-segmentation')
if ANIMESEG_DIR not in sys.path:
    sys.path.insert(0, ANIMESEG_DIR)

SMOOTH_DIR = os.path.join(PBL6_DIR, 'third_party', 'lineart_tools', 'ChayGoAnimeLinesDetail_NoBG')
if os.path.isdir(SMOOTH_DIR) and SMOOTH_DIR not in sys.path:
    sys.path.insert(0, SMOOTH_DIR)

from network.line_extractor import LineExtractor
from network.anime_seg import AnimeBgRemover
from smooth_lines import smooth_lineart_channel

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


class AnimeLinePipelineGPU:
    def __init__(self, device=DEVICE):
        self.device = torch.device(device)
        print(f"[*] Khởi tạo AnimeLinePipelineGPU trên thiết bị: {self.device}")

        # 1. Tải mô hình Anime Segmentation
        print("    [1/2] Tải mô hình ISNetDIS (Tách nền anime)...")
        self.bg_remover = AnimeBgRemover(ckpt_path=ANIMESEG_CKPT, device=device, img_size=1024)

        # 2. Tải mô hình AniLines (Mode Detail)
        print("    [2/2] Tải mô hình AniLines Detail (Trích xuất nét vẽ)...")
        self.line_extractor = LineExtractor(2, 1, True).to(self.device)
        self.line_extractor.load_state_dict(torch.load(ANILINES_WEIGHTS, map_location=self.device, weights_only=True))
        self.line_extractor.eval()
        for p in self.line_extractor.parameters():
            p.requires_grad = False

    def extract_lineart_from_white_bg(self, img_white_bg):
        """Trích xuất line art grayscale từ ảnh đã xóa phông nền trắng."""
        img_gray = cv2.cvtColor(img_white_bg, cv2.COLOR_BGR2GRAY)

        sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = cv2.magnitude(sobelx, sobely)
        sobel = 255 - cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

        t_img = torch.from_numpy(img_gray).unsqueeze(0).unsqueeze(0).float().to(self.device) / 255.0
        t_sobel = torch.from_numpy(sobel).unsqueeze(0).unsqueeze(0).float().to(self.device) / 255.0

        x_in = torch.cat([t_img, t_sobel], dim=1)

        B, C, H, W = x_in.shape
        pad_h = 8 - (H % 8) if (H % 8) != 0 else 0
        pad_w = 8 - (W % 8) if (W % 8) != 0 else 0
        x_in = torch.nn.functional.pad(x_in, (0, pad_w, 0, pad_h), mode='reflect')

        with torch.no_grad():
            if 'cuda' in self.device.type:
                with torch.amp.autocast(device_type='cuda'):
                    pred = self.line_extractor(x_in)
            else:
                pred = self.line_extractor(x_in)

        pred = pred[:, :, :H, :W]
        lineart_np = (pred.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        return lineart_np

    def process_frame(self, raw_img_path, output_smooth_path):
        """Xử lý toàn bộ pipeline cho 1 frame: BGR -> NoBG -> Detail Line -> Smooth."""
        if os.path.exists(output_smooth_path):
            return True

        img_bgr = cv2.imread(raw_img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Không thể đọc ảnh: {raw_img_path}")

        # Bước 1: Tách nền nhân vật, thay bằng nền trắng
        img_white_bg = self.bg_remover.remove_background(img_bgr)

        # Bước 2: Trích xuất nét vẽ AniLines Detail
        lineart_gray = self.extract_lineart_from_white_bg(img_white_bg)

        # Bước 3: Làm mượt nét sub-pixel anti-aliasing
        smooth_gray = smooth_lineart_channel(
            lineart_gray,
            scale=4,
            sigma_color=38.0,
            sigma_space=5.5,
            smooth_iters=1,
            preserve_core=True,
            vector_curve=False
        )

        cv2.imwrite(output_smooth_path, smooth_gray)
        return True


def run_lineart_processing(screened_file=SCREENED_TRIPLETS_FILE,
                           output_cache_dir=LINEART_CACHE_DIR):
    print("=" * 70)
    print("🚀 [GIAI ĐOẠN 5]: Tiền xử lý Line Art chuẩn ml240 (Segmentation + AniLines + Smooth) trên CUDA")
    print("=" * 70)

    if not os.path.exists(screened_file):
        raise FileNotFoundError(f"Không tìm thấy file: {screened_file}")

    with open(screened_file, 'r', encoding='utf-8') as f:
        triplets = json.load(f)

    if len(triplets) == 0:
        print("[!] Không có triplet nào đạt chuẩn. Dừng xử lý.")
        return 0

    os.makedirs(output_cache_dir, exist_ok=True)

    # Thu thập tất cả các file ảnh độc lập cần xử lý
    unique_frame_paths = set()
    for tr in triplets:
        unique_frame_paths.add(tr["i0_path"])
        unique_frame_paths.add(tr["igt_path"])
        unique_frame_paths.add(tr["i1_path"])

    sorted_paths = sorted(list(unique_frame_paths))
    print(f"[*] Cần xử lý Line Art cho {len(sorted_paths)} frame độc lập (được sử dụng bởi {len(triplets)} triplets)...")

    pipeline = AnimeLinePipelineGPU(device=DEVICE)

    for p in tqdm(sorted_paths, desc="Line Art Processing (CUDA)"):
        base_name = os.path.splitext(os.path.basename(p))[0]
        out_smooth = os.path.join(output_cache_dir, f"{base_name}_smooth.png")
        pipeline.process_frame(p, out_smooth)

    print(f"\n[✓] Hoàn thành! Toàn bộ {len(sorted_paths)} frame đã được xử lý và lưu tại: {output_cache_dir}\n")
    return len(sorted_paths)


if __name__ == '__main__':
    run_lineart_processing()
