#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
create_extract_mix_notebook.py
Sinh file notebook Jupyter: `00_extract_and_mix_sakugabooru_atd12k_kaggle.ipynb`
Hỗ trợ chế độ DUAL-MODE thông minh:
- Chạy trên MÁY TÍNH CỤC BỘ (Local Windows PC: E:\\Dataset\\datasets và sakugabooru_character_acting).
- HOẶC chạy trên KAGGLE CLOUD (/kaggle/input và /kaggle/working).
- Tích hợp 100% thanh tiến độ thời gian thực (Elapsed, ETA, Speed, %) cho toàn bộ các bước.
"""

import json
import os
import sys

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def build_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3 (.venv)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    def add_md(source):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    def add_code(source):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    # =========================================================================
    # Header Markdown
    # =========================================================================
    add_md("""# 🎬 PIPELINE EXTRACT & TRỘN FRAME ANIME CHƯA QUA CHỈNH SỬA (RAW CANDIDATES)
## Kết Hợp Video Sakugabooru Character Acting & Dataset ATD-12K Anime Nhật Bản
**Đồ án PBL6 - Hệ Thống Nội Suy Khung Hình Anime Bằng Thin-Plate Spline & Deep Learning**

---

### 📌 Mục Tiêu Của Notebook Này:
1. **Đầu vào 1:** Video `sakugabooru_character_acting` (tại thư mục cục bộ hoặc upload lên Kaggle).
2. **Đầu vào 2:** Tập dữ liệu ATD-12K Anime (tại `E:\\Dataset\\datasets` hoặc dataset Kaggle).
3. **Nhiệm vụ xử lý:**
   - **Xử lý Sakugabooru:** Quét video character acting, áp dụng **CUDA Micro-Dedup (Sobel Edge Difference + PSNR)** để lọc bỏ frame đứng hình (holds / on-twos / on-threes) và nhận diện cắt cảnh. Sau đó áp dụng **Step-based Triplet Mining** để khai thác các triplet có bước nhảy thời gian phong phú: $t \\in \\{0.25, 0.333, 0.5, 0.667, 0.75\\}$.
   - **Xử lý ATD-12K Anime Nhật:** Lọc chính xác các thư mục `Japan_*` và `Hayao_*` (loại bỏ hoàn toàn `Disney_*`), resize từ 1080p về chuẩn 540p ($960 \\times 540$).
   - **Trộn & Đánh số chuẩn hóa:** Hợp nhất 2 nguồn thành cấu trúc thư mục chuẩn `triplets/triplet_XXXXXX/frame1.jpg, frame2.jpg, frame3.jpg` kèm `dataset_manifest.json`.
   - **Đóng gói file nén:** Xuất ra file `candidate_triplets_anime_540p.zip` (~4 - 5 GB) sẵn sàng để đưa vào pipeline Deep Learning tiếp theo.

---

### 💻 Hỗ Trợ Chạy Linh Hoạt (Dual-Mode: Local PC & Kaggle Cloud):
- **Chế độ Local PC:** Tự động phát hiện đường dẫn `E:\\Dataset\\datasets` và `sakugabooru_character_acting`.
- **Chế độ Kaggle Cloud:** Tự động nhận diện `/kaggle/input` và lưu tại `/kaggle/working`.
- **Thanh tiến độ thời gian thực (Real-time ETA):** Hiển thị % hoàn thành, thời gian đã chạy (`Elapsed`), thời gian còn lại (`ETA`) và tốc độ mẫu/giây ở 100% tất cả các bước.""")

    # =========================================================================
    # Cell 1: Environment & Settings
    # =========================================================================
    add_md("""---
### ⚙️ Bước 1: Kiểm Tra Tài Nguyên Phần Cứng & Thiết Lập Cấu Hình""")
    add_code("""import os
import sys
import glob
import json
import time
import math
import shutil
import random
import zipfile
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

# Hàm định dạng giây sang HH:MM:SS
def format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

# Thiết lập Seed ngẫu nhiên
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"[*] Thiết bị tăng tốc tính toán: {DEVICE}")
if torch.cuda.is_available():
    print(f"[*] Tên GPU: {torch.cuda.get_device_name(0)}")

# Tự động nhận diện môi trường: Kaggle Cloud hay Local PC
IS_KAGGLE = os.path.exists("/kaggle")
print(f"[*] Môi trường thực thi: {'KAGGLE CLOUD' if IS_KAGGLE else 'LOCAL PC (Windows)'}")

# Thư mục gốc làm việc
if IS_KAGGLE:
    BASE_WORK_DIR = "/kaggle/working"
else:
    BASE_WORK_DIR = os.path.abspath(os.getcwd())

# Kiểm tra dung lượng đĩa
total, used, free = shutil.disk_usage(BASE_WORK_DIR)
print(f"[*] Dung lượng ổ đĩa tại '{BASE_WORK_DIR}': Tổng {total/(1024**3):.1f} GB | Đã dùng {used/(1024**3):.1f} GB | Trống {free/(1024**3):.1f} GB")

# ==============================================================================
# BẢNG THAM SỐ CẤU HÌNH (CÓ THỂ ĐIỀU CHỈNH THEO NHU CẦU)
# ==============================================================================
TARGET_WIDTH = 960            # Chuẩn chiều rộng 540p (16:9)
TARGET_HEIGHT = 540           # Chuẩn chiều cao 540p
JPEG_QUALITY = 95             # Chất lượng nén JPG

# Giới hạn số lượng triplets mục tiêu (theo bảng phân tích hybrid)
MAX_SAKUGA_TRIPLETS = 3500    # Số triplet mục tiêu từ Sakugabooru (đa mốc t)
MAX_TRIPLETS_PER_CLIP = 20    # Giới hạn tối đa số triplet trên mỗi video clip (tránh thiên vị 1 clip)
MAX_ATD12K_TRIPLETS = 4500    # Số triplet mục tiêu từ ATD-12K Anime Nhật (t = 0.5)

# Ngưỡng lọc Micro-Dedup và phát hiện chuyển cảnh
MIN_DELTA_F = 2               # Bước nhảy frame tối thiểu (On-twos: 2 frame)
MAX_DELTA_F = 6               # Bước nhảy frame tối đa (<= 0.25 giây ở 24fps)
PSNR_DEDUP_THRESH = 40.0      # Ngưỡng PSNR coi là frame trùng lặp (đứng hình)
EDGE_DIFF_THRESH = 0.35       # Ngưỡng biến thiên nét Sobel coi là trùng nét
CUT_SCENE_PSNR = 14.0         # Ngưỡng PSNR phát hiện cắt cảnh (Cut Scene)
CUT_SCENE_EDGE = 4.5          # Ngưỡng chênh lệch nét phát hiện cắt cảnh
MAX_HOLD_FRAMES = 8           # Giới hạn frame đứng hình liên tiếp trước khi reset""")

    # =========================================================================
    # Cell 2: Smart Path Auto-Discovery (Local PC & Kaggle)
    # =========================================================================
    add_md("""---
### 🔎 Bước 2: Tự Động Định Vị Dataset (Local PC `E:\\Dataset\\datasets` & Sakugabooru)
Tự động tìm kiếm nguồn dữ liệu phù hợp trên cả Local PC lẫn Kaggle:
- **ATD-12K:** Tự động quét `E:\\Dataset\\datasets\\train_10k` hoặc `E:\\Dataset\\datasets`.
- **Sakugabooru:** Tự động tìm thư mục `sakugabooru_character_acting` hoặc file zip kèm thanh tiến độ giải nén.""")
    add_code("""# ==============================================================================
# CẤU HÌNH ĐƯỜNG DẪN ĐẦU VÀO
# ==============================================================================
if IS_KAGGLE:
    CUSTOM_SAKUGA_PATH = None
    CUSTOM_ATD12K_PATH = None
else:
    # ĐƯỜNG DẪN CỤC BỘ TRÊN MÁY TÍNH CỦA BẠN:
    CUSTOM_SAKUGA_PATH = r"E:\\CODE\\PBL\\PBL6\\sakugabooru_character_acting"
    CUSTOM_ATD12K_PATH = r"E:\\Dataset\\datasets"

# 1. Tìm video Sakugabooru
sakuga_video_files = []
sakuga_zip_files = []

if CUSTOM_SAKUGA_PATH and os.path.exists(CUSTOM_SAKUGA_PATH):
    print(f"[*] Đang quét video Sakugabooru tại: {CUSTOM_SAKUGA_PATH}")
    for root, dirs, files in os.walk(CUSTOM_SAKUGA_PATH):
        for f in files:
            if f.lower().endswith(('.mp4', '.mkv', '.webm', '.avi')):
                sakuga_video_files.append(os.path.join(root, f))
else:
    # Quét đệ quy trong thư mục làm việc / Kaggle input
    search_root = "/kaggle/input" if IS_KAGGLE else os.path.abspath("..")
    for root, dirs, files in os.walk(search_root):
        for f in files:
            low_f = f.lower()
            if low_f.endswith(('.mp4', '.mkv', '.webm', '.avi')):
                if "sakuga" in root.lower() or "character_acting" in root.lower() or "sakuga" in low_f:
                    sakuga_video_files.append(os.path.join(root, f))
            elif low_f.endswith('.zip') and ("sakuga" in low_f or "character_acting" in low_f):
                sakuga_zip_files.append(os.path.join(root, f))

# Nếu video đang nằm trong file zip và chưa có thư mục video
if not sakuga_video_files and sakuga_zip_files:
    zip_src = sakuga_zip_files[0]
    print(f"[*] Phát hiện file nén Sakugabooru: {zip_src}")
    SAKUGA_EXTRACT_DIR = os.path.join(BASE_WORK_DIR, "temp_sakuga_videos")
    os.makedirs(SAKUGA_EXTRACT_DIR, exist_ok=True)
    
    with zipfile.ZipFile(zip_src, 'r') as zf:
        members = zf.infolist()
        total_uncompressed_bytes = sum(m.file_size for m in members)
        print(f"[*] Đang giải nén {len(members)} files ({total_uncompressed_bytes / (1024**3):.2f} GB) vào '{SAKUGA_EXTRACT_DIR}'...")
        with tqdm(total=len(members), desc="📦 Giải nén Sakugabooru",
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} files [{elapsed}<{remaining}, {rate_fmt}]') as pbar_unzip:
            for member in members:
                zf.extract(member, SAKUGA_EXTRACT_DIR)
                pbar_unzip.update(1)

    sakuga_video_files = glob.glob(f"{SAKUGA_EXTRACT_DIR}/**/*.mp4", recursive=True)

sakuga_video_files.sort()

# 2. Tìm thư mục chứa ATD-12K Anime (Japan_* và Hayao_*)
atd12k_japan_folders = []
atd12k_disney_count = 0

atd_search_roots = []
if CUSTOM_ATD12K_PATH and os.path.exists(CUSTOM_ATD12K_PATH):
    atd_search_roots.append(CUSTOM_ATD12K_PATH)
    train_10k_candidate = os.path.join(CUSTOM_ATD12K_PATH, "train_10k")
    if os.path.exists(train_10k_candidate):
        atd_search_roots.append(train_10k_candidate)
else:
    atd_search_roots.append("/kaggle/input" if IS_KAGGLE else r"E:\\Dataset\\datasets")

for s_root in atd_search_roots:
    if os.path.exists(s_root):
        print(f"[*] Đang quét ATD-12K Anime tại: {s_root}")
        for root, dirs, files in os.walk(s_root):
            for d in dirs:
                if d.startswith("Japan_") or d.startswith("Hayao_"):
                    atd12k_japan_folders.append(os.path.join(root, d))
                elif d.startswith("Disney_"):
                    atd12k_disney_count += 1
        if atd12k_japan_folders:
            break

# Bỏ trùng lặp
atd12k_japan_folders = sorted(list(set(atd12k_japan_folders)))

print("-" * 75)
print("📊 BÁO CÁO NHẬN DIỆN DỮ LIỆU ĐẦU VÀO:")
print(f"   - Video Sakugabooru tìm thấy:         {len(sakuga_video_files):,} files")
print(f"   - ATD-12K Anime Nhật (Japan + Hayao): {len(atd12k_japan_folders):,} thư mục đạt chuẩn")
print(f"   - ATD-12K Disney (được loại bỏ 100%): {atd12k_disney_count:,} thư mục")
print("-" * 75)

if len(sakuga_video_files) == 0:
    print("[!] CẢNH BÁO: Chưa tìm thấy video Sakugabooru nào! Hãy kiểm tra lại CUSTOM_SAKUGA_PATH.")
if len(atd12k_japan_folders) == 0:
    print("[!] CẢNH BÁO: Chưa tìm thấy thư mục ATD-12K Anime Nhật nào! Hãy kiểm tra lại CUSTOM_ATD12K_PATH.")""")

    # =========================================================================
    # Cell 3: Micro-Dedup Worker & Step-Based Mining
    # =========================================================================
    add_md("""---
### 🧠 Bước 3: Định Nghĩa Bộ Lọc Micro-Dedup CUDA & Thuật Toán Khai Thác Đa Mốc Thời Gian $t$
- **Micro-Dedup trên GPU:** Dùng tích chập Sobel 2D trên PyTorch CUDA để tính toán độ biến thiên nét vẽ vi mô và PSNR giữa các khung hình, loại bỏ các frame trùng do nhịp giữ hình (on-twos/on-threes).
- **Phát hiện cắt cảnh:** Tự động ngắt đoạn khi PSNR < 14.0 hoặc chênh lệch nét đột ngột > 4.5.
- **Khai thác mốc thời gian linh hoạt ($t$-distribution):**
  - Nhịp Step-2: $(I_0, I_1, I_2) \\implies t = 0.5$
  - Nhịp Step-3: $(I_0, I_1, I_3) \\implies t \\approx 0.333$ và $(I_0, I_2, I_3) \\implies t \\approx 0.667$
  - Nhịp Step-4: $(I_0, I_1, I_4) \\implies t \\approx 0.250$ và $(I_0, I_3, I_4) \\implies t \\approx 0.750$""")
    add_code("""class CudaDedupWorker:
    \"\"\"Tính toán Sobel Edge & PSNR trực tiếp trên CUDA Tensor để lọc frame cực nhanh.\"\"\"
    def __init__(self, device=DEVICE):
        self.device = torch.device(device)
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]], dtype=torch.float32, device=self.device).view(1, 1, 3, 3) / 4.0
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                                [ 0.0,  0.0,  0.0],
                                [ 1.0,  2.0,  1.0]], dtype=torch.float32, device=self.device).view(1, 1, 3, 3) / 4.0
        self.sobel_x = sobel_x
        self.sobel_y = sobel_y

    def compute_edge(self, gray_tensor):
        pad = F.pad(gray_tensor, (1, 1, 1, 1), mode='replicate')
        gx = F.conv2d(pad, self.sobel_x)
        gy = F.conv2d(pad, self.sobel_y)
        return torch.sqrt(gx * gx + gy * gy + 1e-8)

    def is_duplicate(self, t1, edge1, t2):
        mse = torch.mean((t1 - t2) ** 2)
        psnr = 100.0 if mse.item() < 1e-10 else (10.0 * torch.log10(1.0 / mse)).item()
        edge2 = self.compute_edge(t2)
        delta_edge = (torch.mean(torch.abs(edge1 - edge2)) * 255.0).item()
        
        is_dup = (psnr >= PSNR_DEDUP_THRESH) and (delta_edge < EDGE_DIFF_THRESH)
        is_cut = (psnr < CUT_SCENE_PSNR) or (delta_edge > CUT_SCENE_EDGE)
        return is_dup, is_cut, edge2


def mine_diverse_triplets(drawings):
    \"\"\"
    Khai thác các Candidate Triplets từ chuỗi Unique Drawings với bước nhảy đa dạng.
    Bổ sung các mốc t = 0.25, 0.333, 0.5, 0.667, 0.75 chuẩn theo phân tích thiết kế dataset.
    \"\"\"
    triplets = []
    n = len(drawings)
    if n < 3:
        return triplets

    for i in range(n - 2):
        f0, img0 = drawings[i]

        # 1. Step-2: Khai thác mốc t = 0.5 (kề cận trực tiếp)
        f_gt = drawings[i + 1][0]
        img_gt = drawings[i + 1][1]
        f1 = drawings[i + 2][0]
        img1 = drawings[i + 2][1]
        delta_f = f1 - f0
        if MIN_DELTA_F <= delta_f <= MAX_DELTA_F:
            t = round((f_gt - f0) / float(delta_f), 4)
            triplets.append({
                'f0': f0, 'f_gt': f_gt, 'f1': f1,
                'delta_f': delta_f, 't': t,
                'img0': img0, 'img_gt': img_gt, 'img1': img1
            })

        # 2. Step-3: Khai thác mốc t = 1/3 (0.333) và t = 2/3 (0.667)
        if i + 3 < n:
            f3 = drawings[i + 3][0]
            img3 = drawings[i + 3][1]
            delta_f3 = f3 - f0
            if MIN_DELTA_F <= delta_f3 <= MAX_DELTA_F:
                t_third_1 = round((drawings[i + 1][0] - f0) / float(delta_f3), 4)
                triplets.append({
                    'f0': f0, 'f_gt': drawings[i + 1][0], 'f1': f3,
                    'delta_f': delta_f3, 't': t_third_1,
                    'img0': img0, 'img_gt': drawings[i + 1][1], 'img1': img3
                })
                t_third_2 = round((drawings[i + 2][0] - f0) / float(delta_f3), 4)
                triplets.append({
                    'f0': f0, 'f_gt': drawings[i + 2][0], 'f1': f3,
                    'delta_f': delta_f3, 't': t_third_2,
                    'img0': img0, 'img_gt': drawings[i + 2][1], 'img1': img3
                })

        # 3. Step-4: Khai thác mốc t = 1/4 (0.25) và t = 3/4 (0.75)
        if i + 4 < n:
            f4 = drawings[i + 4][0]
            img4 = drawings[i + 4][1]
            delta_f4 = f4 - f0
            if MIN_DELTA_F <= delta_f4 <= MAX_DELTA_F:
                t_quarter_1 = round((drawings[i + 1][0] - f0) / float(delta_f4), 4)
                triplets.append({
                    'f0': f0, 'f_gt': drawings[i + 1][0], 'f1': f4,
                    'delta_f': delta_f4, 't': t_quarter_1,
                    'img0': img0, 'img_gt': drawings[i + 1][1], 'img1': img4
                })
                t_quarter_3 = round((drawings[i + 3][0] - f0) / float(delta_f4), 4)
                triplets.append({
                    'f0': f0, 'f_gt': drawings[i + 3][0], 'f1': f4,
                    'delta_f': delta_f4, 't': t_quarter_3,
                    'img0': img0, 'img_gt': drawings[i + 3][1], 'img1': img4
                })

    return triplets


def extract_triplets_from_sakuga_clip(video_path, dedup_worker, max_triplets=MAX_TRIPLETS_PER_CLIP):
    \"\"\"Đọc 1 video Sakugabooru và trích xuất các Candidate Triplets sạch.\"\"\"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    unique_drawings = []
    prev_tensor = None
    prev_edge = None
    hold_count = 0
    all_triplets = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Resize về chuẩn 540p (960x540)
        h, w = frame.shape[:2]
        if w != TARGET_WIDTH or h != TARGET_HEIGHT:
            frame_540p = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
        else:
            frame_540p = frame

        # Chuyển đổi Grayscale Tensor trên CUDA
        gray = cv2.cvtColor(frame_540p, cv2.COLOR_BGR2GRAY)
        tensor = torch.from_numpy(gray).float().to(dedup_worker.device).unsqueeze(0).unsqueeze(0) / 255.0

        if prev_tensor is None:
            prev_tensor = tensor
            prev_edge = dedup_worker.compute_edge(tensor)
            unique_drawings.append((frame_idx, frame_540p))
            frame_idx += 1
            continue

        is_dup, is_cut, edge2 = dedup_worker.is_duplicate(prev_tensor, prev_edge, tensor)

        if is_cut:
            all_triplets.extend(mine_diverse_triplets(unique_drawings))
            unique_drawings = [(frame_idx, frame_540p)]
            prev_tensor = tensor
            prev_edge = edge2
            hold_count = 0
            if len(all_triplets) >= max_triplets * 2:
                break
            frame_idx += 1
            continue

        if is_dup:
            hold_count += 1
            if hold_count > MAX_HOLD_FRAMES:
                all_triplets.extend(mine_diverse_triplets(unique_drawings))
                unique_drawings = [(frame_idx, frame_540p)]
                hold_count = 0
        else:
            unique_drawings.append((frame_idx, frame_540p))
            prev_tensor = tensor
            prev_edge = edge2
            hold_count = 0

        frame_idx += 1

    cap.release()

    if len(unique_drawings) >= 3:
        all_triplets.extend(mine_diverse_triplets(unique_drawings))

    random.shuffle(all_triplets)
    return all_triplets[:max_triplets]

print("[✓] Đã khởi tạo hoàn tất Worker CUDA Dedup & Triplet Mining Logic!")""")

    # =========================================================================
    # Cell 4: Extract Sakugabooru Triplets (Rich Progress & ETA)
    # =========================================================================
    add_md("""---
### 🚀 Bước 4: Khai Thác Candidate Triplets Từ Video Sakugabooru
Thực thi trích xuất frame từ các clip Sakugabooru, lưu trực tiếp vào thư mục đệm `work_candidates/sakuga/`.
**Hiển thị thanh tiến độ thời gian thực:**
- Tiến độ duyệt video clips (`clip_index / total_clips` kèm `%`)
- Thời gian đã trôi qua (`Elapsed`)
- Thời gian ước tính còn lại (`ETA`)
- Số lượng Triplets thu hoạch thực tế so với mục tiêu (`X / 3500`)
- Tốc độ xử lý (`triplets/giây`)""")
    add_code("""WORK_DIR = os.path.join(BASE_WORK_DIR, "work_candidates")
SAKUGA_OUT_DIR = os.path.join(WORK_DIR, "sakuga")
os.makedirs(SAKUGA_OUT_DIR, exist_ok=True)

dedup_worker = CudaDedupWorker(device=DEVICE)

sakuga_candidates_metadata = []
total_sakuga_extracted = 0

print("=" * 75)
print(f"🚀 BẮT ĐẦU KHAI THÁC VIDEO SAKUGABOORU (Mục tiêu: ~{MAX_SAKUGA_TRIPLETS} triplets)")
print("=" * 75)

start_sakuga_time = time.time()

with tqdm(total=len(sakuga_video_files), desc="🎬 Trích xuất Sakugabooru",
          bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} clips [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
    for v_path in sakuga_video_files:
        clip_name = os.path.splitext(os.path.basename(v_path))[0]
        clip_triplets = extract_triplets_from_sakuga_clip(v_path, dedup_worker, max_triplets=MAX_TRIPLETS_PER_CLIP)
        
        for tr in clip_triplets:
            total_sakuga_extracted += 1
            folder_id = f"sakuga_{total_sakuga_extracted:06d}"
            target_dir = os.path.join(SAKUGA_OUT_DIR, folder_id)
            os.makedirs(target_dir, exist_ok=True)

            cv2.imwrite(os.path.join(target_dir, "frame1.jpg"), tr['img0'], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            cv2.imwrite(os.path.join(target_dir, "frame2.jpg"), tr['img_gt'], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            cv2.imwrite(os.path.join(target_dir, "frame3.jpg"), tr['img1'], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

            sakuga_candidates_metadata.append({
                'source_type': 'sakugabooru',
                'local_id': folder_id,
                'source_clip': clip_name,
                't': tr['t'],
                'delta_frames': tr['delta_f'],
                'folder_path': target_dir
            })

            if total_sakuga_extracted >= MAX_SAKUGA_TRIPLETS:
                break

        elapsed_sec = time.time() - start_sakuga_time
        trip_rate = (total_sakuga_extracted / elapsed_sec) if elapsed_sec > 0 else 0
        rem_triplets = max(0, MAX_SAKUGA_TRIPLETS - total_sakuga_extracted)
        eta_sec = (rem_triplets / trip_rate) if trip_rate > 0 else 0

        pbar.set_postfix({
            'Triplets': f"{total_sakuga_extracted}/{MAX_SAKUGA_TRIPLETS}",
            'Tốc độ': f"{trip_rate:.1f} tr/s",
            'ETA mục tiêu': format_time(eta_sec)
        })
        pbar.update(1)

        if total_sakuga_extracted >= MAX_SAKUGA_TRIPLETS:
            print()
            print(f"[*] Đã đạt số lượng mục tiêu Sakugabooru ({MAX_SAKUGA_TRIPLETS} triplets). Dừng trích xuất.")
            break

total_sakuga_time = time.time() - start_sakuga_time
print()
print("=" * 75)
print(f"[✓] Khai thác thành công {len(sakuga_candidates_metadata)} Candidate Triplets từ Sakugabooru!")
print(f"    - Thời gian chạy: {format_time(total_sakuga_time)}")
if total_sakuga_time > 0:
    print(f"    - Tốc độ trung bình: {len(sakuga_candidates_metadata) / total_sakuga_time:.1f} triplets/giây")
print("=" * 75)""")

    # =========================================================================
    # Cell 5: Standardize ATD-12K Anime (Rich Progress & ETA)
    # =========================================================================
    add_md("""---
### 🎌 Bước 5: Chuẩn Hóa Candidate Triplets Từ ATD-12K Anime Nhật Bản
- Lọc lấy các thư mục `Japan_*` và `Hayao_*`.
- Resize từ 1080p về 540p ($960 \\times 540$) bằng đa luồng CPU (Multi-threading).
- Ghi nhận với mốc thời gian chuẩn $t = 0.5$.
- Hiển thị thanh tiến độ thời gian thực với % hoàn thành, thời gian đã chạy, thời gian còn lại (ETA) và tốc độ mẫu/giây.""")
    add_code("""ATD_OUT_DIR = os.path.join(WORK_DIR, "atd12k")
os.makedirs(ATD_OUT_DIR, exist_ok=True)

# Lựa chọn số lượng mẫu ATD-12K theo giới hạn
selected_atd_folders = atd12k_japan_folders[:MAX_ATD12K_TRIPLETS]

def process_single_atd_folder(item):
    folder_path, out_base_dir, out_index = item
    f1_src = os.path.join(folder_path, "frame1.jpg")
    f2_src = os.path.join(folder_path, "frame2.jpg")
    f3_src = os.path.join(folder_path, "frame3.jpg")

    if not (os.path.exists(f1_src) and os.path.exists(f2_src) and os.path.exists(f3_src)):
        return None

    img1 = cv2.imread(f1_src)
    img2 = cv2.imread(f2_src)
    img3 = cv2.imread(f3_src)
    if img1 is None or img2 is None or img3 is None:
        return None

    folder_id = f"atd12k_{out_index:06d}"
    target_dir = os.path.join(out_base_dir, folder_id)
    os.makedirs(target_dir, exist_ok=True)

    h, w = img1.shape[:2]
    if w != TARGET_WIDTH or h != TARGET_HEIGHT:
        img1 = cv2.resize(img1, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
        img3 = cv2.resize(img3, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

    cv2.imwrite(os.path.join(target_dir, "frame1.jpg"), img1, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    cv2.imwrite(os.path.join(target_dir, "frame2.jpg"), img2, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    cv2.imwrite(os.path.join(target_dir, "frame3.jpg"), img3, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    return {
        'source_type': 'atd12k_japan',
        'local_id': folder_id,
        'source_clip': os.path.basename(folder_path),
        't': 0.5,
        'delta_frames': 2,
        'folder_path': target_dir
    }

print("=" * 75)
print(f"🚀 BẮT ĐẦU CHUẨN HÓA ATD-12K ANIME NHẬT (Tổng số: {len(selected_atd_folders)} triplets)")
print("=" * 75)

tasks = [(fpath, ATD_OUT_DIR, idx + 1) for idx, fpath in enumerate(selected_atd_folders)]
atd12k_candidates_metadata = []

start_atd_time = time.time()
with ThreadPoolExecutor(max_workers=8) as executor:
    pbar_atd = tqdm(executor.map(process_single_atd_folder, tasks), total=len(tasks),
                    desc="🎌 Chuẩn hóa ATD-12K",
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} triplets [{elapsed}<{remaining}, {rate_fmt}]')
    for res in pbar_atd:
        if res is not None:
            atd12k_candidates_metadata.append(res)
            pbar_atd.set_postfix({'Hợp lệ': len(atd12k_candidates_metadata)})

total_atd_time = time.time() - start_atd_time
print()
print("=" * 75)
print(f"[✓] Chuẩn hóa thành công {len(atd12k_candidates_metadata)} Candidate Triplets từ ATD-12K Anime Nhật Bản!")
print(f"    - Thời gian chạy: {format_time(total_atd_time)}")
if total_atd_time > 0:
    print(f"    - Tốc độ trung bình: {len(atd12k_candidates_metadata) / total_atd_time:.1f} mẫu/giây")
print("=" * 75)""")

    # =========================================================================
    # Cell 6: Merge & Build Unified Dataset (Rich Progress & ETA)
    # =========================================================================
    add_md("""---
### 🔀 Bước 6: Hợp Nhất & Trộn Dữ Liệu (Interleaving Merge)
- Trộn đều các mẫu từ Sakugabooru ($t \\ne 0.5$ và $t = 0.5$) với ATD-12K ($t = 0.5$).
- Đánh số đồng bộ thứ tự chuẩn: `triplets/triplet_000001`, `triplets/triplet_000002`, ...
- Hiển thị thanh tiến độ hợp nhất, sao chép file kèm % và ETA thời gian thực.""")
    add_code("""MERGED_DIR = os.path.join(BASE_WORK_DIR, "candidate_triplets_anime_540p")
MERGED_TRIPLETS_DIR = os.path.join(MERGED_DIR, "triplets")
os.makedirs(MERGED_TRIPLETS_DIR, exist_ok=True)

# Gom toàn bộ metadata từ 2 nguồn
all_pool = []
all_pool.extend(sakuga_candidates_metadata)
all_pool.extend(atd12k_candidates_metadata)

random.seed(RANDOM_SEED)
random.shuffle(all_pool)

print("=" * 75)
print(f"🔀 TIẾN HÀNH HỢP NHẤT VÀ ĐÁNH SỐ TOÀN BỘ DATASET ({len(all_pool)} Triplets)")
print(f"   - Từ Sakugabooru (đa mốc t): {len(sakuga_candidates_metadata)}")
print(f"   - Từ ATD-12K Japan (t = 0.5): {len(atd12k_candidates_metadata)}")
print("=" * 75)

final_manifest = []
global_id = 0

start_merge_time = time.time()
with tqdm(total=len(all_pool), desc="🔀 Hợp nhất & Đánh số",
          bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} triplets [{elapsed}<{remaining}, {rate_fmt}]') as pbar_merge:
    for item in all_pool:
        global_id += 1
        new_triplet_id = f"triplet_{global_id:06d}"
        dst_folder = os.path.join(MERGED_TRIPLETS_DIR, new_triplet_id)
        os.makedirs(dst_folder, exist_ok=True)

        src_folder = item['folder_path']
        shutil.copy2(os.path.join(src_folder, "frame1.jpg"), os.path.join(dst_folder, "frame1.jpg"))
        shutil.copy2(os.path.join(src_folder, "frame2.jpg"), os.path.join(dst_folder, "frame2.jpg"))
        shutil.copy2(os.path.join(src_folder, "frame3.jpg"), os.path.join(dst_folder, "frame3.jpg"))

        final_manifest.append({
            'triplet_id': new_triplet_id,
            'source': item['source_clip'],
            'source_type': item['source_type'],
            't': item['t'],
            'delta_frames': item['delta_frames'],
            'frame1': f"triplets/{new_triplet_id}/frame1.jpg",
            'frame2': f"triplets/{new_triplet_id}/frame2.jpg",
            'frame3': f"triplets/{new_triplet_id}/frame3.jpg"
        })
        pbar_merge.update(1)

# Lưu dataset_manifest.json
manifest_path = os.path.join(MERGED_DIR, "dataset_manifest.json")
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(final_manifest, f, indent=2, ensure_ascii=False)

total_merge_time = time.time() - start_merge_time
print()
print("=" * 75)
print(f"[✓] Đã tạo thành công 'dataset_manifest.json' với {len(final_manifest)} Triplets hoàn chỉnh!")
print(f"    - Thời gian hợp nhất: {format_time(total_merge_time)}")
print("=" * 75)""")

    # =========================================================================
    # Cell 7: Pack to ZIP with Real-time Progress & ETA
    # =========================================================================
    add_md("""---
### 📦 Bước 7: Dọn Dẹp Thư Mục Tạm & Đóng Gói File ZIP (Kèm Thanh Tiến Độ Nén & ETA)
- Xóa bỏ các thư mục đệm tạm thời để tiết kiệm tối đa ổ đĩa.
- Quét trước toàn bộ danh sách file ảnh cần nén (~24,000 files).
- Nén toàn bộ vào `candidate_triplets_anime_540p.zip` **với thanh tiến độ thời gian thực, % hoàn thành, thời gian đã chạy và thời gian còn lại (ETA)**.""")
    add_code("""# 1. Dọn dẹp thư mục tạm để giải phóng ổ cứng trước khi nén
print("[*] Đang dọn dẹp thư mục tạm work_candidates...")
shutil.rmtree(WORK_DIR, ignore_errors=True)
temp_sakuga_dir = os.path.join(BASE_WORK_DIR, "temp_sakuga_videos")
if os.path.exists(temp_sakuga_dir):
    shutil.rmtree(temp_sakuga_dir, ignore_errors=True)

# Kiểm tra dung lượng hiện tại
t, u, f = shutil.disk_usage(BASE_WORK_DIR)
print(f"[*] Dung lượng đĩa trống trước khi nén: {f/(1024**3):.2f} GB")

# 2. Quét trước toàn bộ file để hiển thị thanh tiến độ nén chính xác
print()
print("[*] Đang quét danh sách file cần nén...")
all_files_to_zip = []
for root, dirs, files in os.walk(MERGED_TRIPLETS_DIR):
    for file in files:
        full_path = os.path.join(root, file)
        rel_path = os.path.relpath(full_path, MERGED_DIR)
        all_files_to_zip.append((full_path, rel_path))

total_files_to_compress = len(all_files_to_zip) + 1  # Cộng thêm dataset_manifest.json
ZIP_OUTPUT_PATH = os.path.join(BASE_WORK_DIR, "candidate_triplets_anime_540p.zip")

print(f"[*] Tổng số file cần nén vào ZIP: {total_files_to_compress:,} files.")
print(f"[*] Đang nén toàn bộ dataset vào '{ZIP_OUTPUT_PATH}'...")

start_zip_time = time.time()
with zipfile.ZipFile(ZIP_OUTPUT_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(manifest_path, arcname="dataset_manifest.json")
    
    with tqdm(total=len(all_files_to_zip), desc="📦 Nén ZIP Dataset",
              bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} files [{elapsed}<{remaining}, {rate_fmt}]') as pbar_zip:
        for full_path, rel_path in all_files_to_zip:
            zipf.write(full_path, arcname=rel_path)
            pbar_zip.update(1)

zip_duration = time.time() - start_zip_time
zip_size_gb = os.path.getsize(ZIP_OUTPUT_PATH) / (1024 ** 3)

print()
print("=" * 75)
print(f"🎉 HOÀN TẤT ĐÓNG GÓI DATASET!")
print(f"   - Tên file zip:      {ZIP_OUTPUT_PATH}")
print(f"   - Dung lượng nén:    {zip_size_gb:.2f} GB")
print(f"   - Thời gian nén:     {format_time(zip_duration)}")
if zip_duration > 0:
    print(f"   - Tốc độ nén:        {total_files_to_compress / zip_duration:.1f} files/giây")
print(f"   - Tổng số candidate: {len(final_manifest):,} triplets")
print("=" * 75)

if IS_KAGGLE:
    from IPython.display import FileLink, display
    print()
    print("👉 Bấm vào link bên dưới để tải trực tiếp file ZIP về máy tính của bạn:")
    display(FileLink("candidate_triplets_anime_540p.zip"))
else:
    print()
    print(f"[✓] File ZIP đã được lưu cục bộ tại: {ZIP_OUTPUT_PATH}")""")

    # =========================================================================
    # Cell 8: Statistics & Verification Collage (Pure Python + Optional Plot)
    # =========================================================================
    add_md("""---
### 📊 Bước 8: Thống Kê Chi Tiết & Trực Quan Hóa Mẫu Dữ Liệu
Kiểm tra tính cân bằng phân phối thời gian $t$, nguồn dữ liệu và hiển thị trực quan các frame.""")
    add_code("""from collections import Counter

print("=" * 65)
print(f"📈 THỐNG KÊ CHI TIẾT TẬP DATASET MỚI TẠO ({len(final_manifest):,} Triplets):")
print("=" * 65)

# 1. Thống kê nguồn
source_counts = Counter(x['source_type'] for x in final_manifest)
print()
print("Phân bố theo nguồn dữ liệu:")
for src, cnt in source_counts.items():
    print(f"   - {src:20s}: {cnt:,} triplets ({cnt/len(final_manifest)*100:.1f}%)")

# 2. Thống kê mốc thời gian t
t_counts = Counter(round(x['t'], 3) for x in final_manifest)
print()
print("Phân bố chi tiết theo mốc thời gian t:")
for t_val in sorted(t_counts.keys()):
    print(f"   - t = {t_val:.3f}: {t_counts[t_val]:,} triplets ({t_counts[t_val]/len(final_manifest)*100:.1f}%)")

# 3. Vẽ biểu đồ nếu môi trường có sẵn matplotlib & pandas
try:
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.DataFrame(final_manifest)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    df['source_type'].value_counts().plot(kind='bar', ax=axes[0], color=['#3b82f6', '#10b981'], edgecolor='black')
    axes[0].set_title("Phân Bố Nguồn Dữ Liệu", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Số lượng mẫu")
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    df['t'].hist(bins=20, ax=axes[1], color='#8b5cf6', edgecolor='black')
    axes[1].set_title("Phân Bố Bước Nhảy Thời Gian (t-distribution)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Giá trị t")
    axes[1].set_ylabel("Tần suất")
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
except ImportError:
    print()
    print("[i] Gợi ý: Cài thêm 'pandas' và 'matplotlib' nếu muốn vẽ biểu đồ đồ họa.")

print()
print("=" * 65)
print("🎉 Xin chúc mừng! Dataset Candidate Triplets Anime đã hoàn tất và sẵn sàng!")
print("=" * 65)""")

    # =========================================================================
    # Cell 9: Markdown Instructions for Next Steps
    # =========================================================================
    add_md("""---
### 🎯 BƯỚC TIẾP THEO SAU KHI CHẠY XONG NOTEBOOK NÀY:

1. **File ZIP đầu ra:**
   - File `candidate_triplets_anime_540p.zip` đã được lưu sẵn ngay tại thư mục hiện tại của bạn.

2. **Upload lên Kaggle làm Dataset:**
   - Truy cập [Kaggle Datasets](https://www.kaggle.com/datasets) -> Chọn **New Dataset**.
   - Đặt tên Dataset: `anime-candidate-triplets-540p`.
   - Upload file `candidate_triplets_anime_540p.zip` lên -> Bấm **Create**.

3. **Chạy Pipeline Deep Learning (`generate_tps_dataset_kaggle.ipynb` trên Kaggle):**
   - Mở file `generate_tps_dataset_kaggle.ipynb` trên Kaggle.
   - Bấm **+ Add Input** và chọn Dataset `anime-candidate-triplets-540p` vừa tạo.
   - Bật GPU T4 x2 và bấm **Run All** để hoàn thành trích xuất nét vẽ và tọa độ neo TPS!""")

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "00_extract_and_mix_sakugabooru_atd12k_kaggle.ipynb")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)

    print(f"[✓] Đã tạo thành công file Jupyter Notebook: {output_path}")
    return output_path


if __name__ == '__main__':
    build_notebook()
