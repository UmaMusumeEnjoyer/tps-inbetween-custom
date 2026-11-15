# [CELL 1]: Cấu hình Môi trường, Phần cứng Kaggle (2x T4 / 1x T4 / A100) & Trình Dọn Rác Bộ Nhớ V3.0-Remastered
import os
import sys
import ctypes

os.environ["MALLOC_TRIM_THRESHOLD_"] = "65536"
os.environ["MALLOC_ARENA_MAX"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Tắt monitor thread của tqdm để tránh xung đột luồng IOPub của Jupyter làm treo "Run All"
import tqdm
tqdm.monitor_interval = 0

import gc
import time
import math
import random
import json
import shutil
import numpy as np
import pandas as pd
from PIL import Image
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torch.utils.data.dataloader import default_collate
import torchvision.models as models

try:
    mp.set_sharing_strategy('file_descriptor')
except Exception:
    pass

try:
    import resource
    _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (_hard, _hard))
except Exception:
    pass

def configure_glibc_allocator():
    """Ép glibc giải phóng bộ nhớ heap không sử dụng về OS cgroup ngay lập tức."""
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.mallopt(-100, 65536)  # M_TRIM_THRESHOLD = 64KB
        libc.mallopt(-102, 0)      # M_MMAP_THRESHOLD = 0
        libc.malloc_trim(0)
        return True
    except Exception:
        return False

def drop_linux_pagecache(file_path):
    """Báo cho kernel Linux xả PageCache của file đã đọc xong để tránh OOM cgroup."""
    if sys.platform.startswith('linux') and os.path.exists(file_path):
        try:
            fd = os.open(file_path, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except Exception:
            pass

def cleanup_memory():
    """Dọn dẹp triệt để bộ nhớ cả trên GPU VRAM lẫn Host CPU RAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

def get_ram_usage_gb():
    """Đo lường lượng Host RAM vật lý (RSS) mà tiến trình Python đang chiếm dụng."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    except Exception:
        return 0.0

# Khởi tạo hạt giống ngẫu nhiên đảm bảo tính tái lập (Reproducibility)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# Phát hiện GPU & Cấu hình thiết bị
NUM_GPUS = torch.cuda.device_count()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Thiết lập thư mục làm việc tiêu chuẩn
WORKING_DIR = '/kaggle/working' if os.path.exists('/kaggle') else './workspace'
os.makedirs(WORKING_DIR, exist_ok=True)

CKPT_DIR = os.path.join(WORKING_DIR, 'checkpoints_v3.0_remastered')
DEMO_DIR = os.path.join(WORKING_DIR, 'visual_samples_v3.0_remastered')
CACHE_DIR = os.path.join(WORKING_DIR, 'dataset_cache_v3.0_remastered')
for d in [CKPT_DIR, DEMO_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

configure_glibc_allocator()
cleanup_memory()

print("=" * 80)
print(f"[*] THIẾT LẬP MÔI TRƯỜNG HUẤN LUYỆN TPS-INBETWEEN V3.0 REMASTERED")
print(f"[*] Thiết bị chính: {device} | Số GPU khả dụng: {NUM_GPUS}")
if NUM_GPUS > 0:
    for g in range(NUM_GPUS):
        p = torch.cuda.get_device_properties(g)
        print(f"    - GPU {g}: {p.name} ({p.total_memory / (1024**3):.1f} GB VRAM)")
print(f"[*] Host RAM ban đầu: {get_ram_usage_gb():.2f} GB")
print(f"[*] Thư mục Checkpoint: {CKPT_DIR}")
print(f"[*] Thư mục Visual Samples: {DEMO_DIR}")
print(f"[*] Định dạng xuất: PNG RGBA Trong Suốt (Alpha Matte, Pure Black Ink)")
print("=" * 80)
sys.stdout.flush()
