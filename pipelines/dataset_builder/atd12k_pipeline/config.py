#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cấu hình tập trung cho Pipeline Tạo và Kiểm Chứng Dataset ATD-12K cho TPS-Inbetween.
"""

import os
import torch

# Đường dẫn thư mục gốc
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATASET_DIR = os.path.abspath(os.path.join(PIPELINE_DIR, '..'))
PBL6_DIR = os.path.abspath(os.path.join(TEST_DATASET_DIR, '..'))
TEST_DIR = os.path.join(PBL6_DIR, 'Test')
MODELS_DIR = os.path.join(TEST_DIR, 'TestModels')

# Python executable
PYTHON_EXE = os.path.join(TEST_DATASET_DIR, '.venv', 'Scripts', 'python.exe')

# Thiết bị tính toán
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Video đầu vào thử nghiệm
INPUT_VIDEO = os.path.join(TEST_DIR, 'ChayGo.mp4')
FPS = 24

# Thư mục đầu ra của từng giai đoạn
RAW_FRAMES_DIR = os.path.join(PIPELINE_DIR, 'work', '01_raw_frames')
DRAWINGS_DIR = os.path.join(PIPELINE_DIR, 'work', '02_unique_drawings')
CANDIDATE_TRIPLETS_FILE = os.path.join(PIPELINE_DIR, 'work', '03_candidate_triplets.json')
MATCHES_CACHE_DIR = os.path.join(PIPELINE_DIR, 'work', '04_matches_cache')
SCREENED_TRIPLETS_FILE = os.path.join(PIPELINE_DIR, 'work', '04_screened_triplets.json')
LINEART_CACHE_DIR = os.path.join(PIPELINE_DIR, 'work', '05_lineart_cache')

# Thư mục Dataset hoàn chỉnh
OUTPUT_DATASET_DIR = os.path.join(PIPELINE_DIR, 'output_dataset')
OUTPUT_TRAIN_DIR = os.path.join(OUTPUT_DATASET_DIR, 'train', 'clip_001')
OUTPUT_VAL_DIR = os.path.join(OUTPUT_DATASET_DIR, 'val', 'clip_001')
OUTPUT_MATCHES_DIR = os.path.join(OUTPUT_DATASET_DIR, 'matches', 'clip_001')
TRIPLETS_CSV = os.path.join(OUTPUT_DATASET_DIR, 'triplets.csv')

# Trọng số mô hình
GLUESTICK_WEIGHTS = os.path.join(MODELS_DIR, 'tps-inbetween', 'model', 'resources', 'weights', 'checkpoint_GlueStick_MD.tar')
TPS_CKPT = os.path.join(MODELS_DIR, 'tps-inbetween', 'ckpt', 'model_latest.pt')
ANILINES_WEIGHTS = os.path.join(TEST_DATASET_DIR, 'AniLines-Anime-Lineart-Extractor', 'weights', 'detail.pth')
ANIMESEG_CKPT = os.path.join(TEST_DATASET_DIR, 'anime-segmentation', 'saved_models', 'isnetis.ckpt')

# Tham số Micro-Deduplication (GPU)
DEDUP_PSNR_THRESH = 40.0         # Frame có PSNR > 40dB so với frame trước bị coi là có khả năng trùng
DEDUP_EDGE_DIFF_THRESH = 0.35    # Độ lệch biên cạnh Sobel nhỏ hơn 0.35
DEDUP_MAX_HOLD_FRAMES = 8        # Nếu giữ nguyên quá 8 frame thì tách để tránh pause quá dài

# Tham số Triplet Mining
MIN_VIDEO_GAP = 2                # Bước nhảy tối thiểu: 2 frames (On 2s)
MAX_VIDEO_GAP = 6                # Bước nhảy tối đa: 6 frames (<= 0.25s)

# Tham số GlueStick & RANSAC Screening
SCREEN_MIN_INLIERS = 20          # Tối thiểu 20 inlier sạch
SCREEN_MAX_MEAN_DISP = 15.0      # Độ dời trung bình <= 15.0px (an toàn cho TPS)
SCREEN_MAX_OUTLIER_DISP = 50.0   # Không có điểm nào lệch > 50px
SCREEN_RANSAC_THRESH = 5.0       # Ngưỡng RANSAC Affine 5.0px
