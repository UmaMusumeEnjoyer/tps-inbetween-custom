# [CELL 1]: Cấu hình Môi trường, Phần cứng Kaggle (2x T4 / 1x T4 / A100) & Trình Dọn Rác Bộ Nhớ Chủ Động V3.2
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
    try:
        libc = ctypes.CDLL('libc.so.6')
        libc.mallopt(-1, 65536)
        libc.mallopt(-8, 2)
    except Exception:
        pass

configure_glibc_allocator()

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
NUM_GPUS = torch.cuda.device_count()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# [CẤU HÌNH CUDNN BENCHMARK V3.2]:
torch.backends.cudnn.benchmark = (NUM_GPUS <= 1)
print("=" * 80)
print(f"[*] THIẾT LẬP MÔI TRƯỜNG & PHẦN CỨNG HUẤN LUYỆN V3.2 (PROGRESSIVE STROKE PRESERVATION)")
print(f"[*] Thiết bị chính: {device} | Số lượng GPU khả dụng: {NUM_GPUS}")
for i in range(NUM_GPUS):
    prop = torch.cuda.get_device_properties(i)
    print(f"    - GPU {i}: {prop.name} (VRAM: {prop.total_memory / 1e9:.2f} GB)")

# Ngưỡng watchdog RAM (working set container). Vượt ngưỡng => lưu resume & dừng êm.
RAM_SOFT_LIMIT_GB = 28.0
RAM_WARN_GB = 24.0

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass

def _read_cgroup_memory_bytes():
    for p in ('/sys/fs/cgroup/memory.current', '/sys/fs/cgroup/memory/memory.usage_in_bytes'):
        try:
            with open(p, 'r') as f:
                return int(f.read().strip())
        except Exception:
            continue
    return None

def _read_cgroup_memory_stat():
    out = {}
    for p in ('/sys/fs/cgroup/memory.stat', '/sys/fs/cgroup/memory/memory.stat'):
        if not os.path.exists(p):
            continue
        try:
            with open(p, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2 and parts[1].isdigit():
                        out[parts[0]] = int(parts[1])
            if out:
                return out
        except Exception:
            continue
    return out

def get_container_ram_info():
    usage = _read_cgroup_memory_bytes()
    stats = _read_cgroup_memory_stat()
    if usage is not None:
        inactive_file = stats.get('inactive_file', 0)
        active_file = stats.get('active_file', 0)
        cache_total = stats.get('file', inactive_file + active_file)
        working_set = max(0, usage - inactive_file)
        return {
            'source': 'cgroup',
            'usage_gb': usage / (1024 ** 3),
            'working_set_gb': working_set / (1024 ** 3),
            'cache_gb': cache_total / (1024 ** 3),
        }
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {
            'source': 'psutil',
            'usage_gb': (vm.total - vm.available) / (1024 ** 3),
            'working_set_gb': (vm.total - vm.available - getattr(vm, 'cached', 0)) / (1024 ** 3),
            'cache_gb': getattr(vm, 'cached', 0) / (1024 ** 3),
        }
    except Exception:
        return {'source': 'unknown', 'usage_gb': 0.0, 'working_set_gb': 0.0, 'cache_gb': 0.0}

def get_host_ram_used_gb():
    info = get_container_ram_info()
    return info['working_set_gb'] if info['working_set_gb'] > 0 else info['usage_gb']

def format_host_ram():
    info = get_container_ram_info()
    return f"{info['working_set_gb']:.2f} GB (cached: {info['cache_gb']:.2f} GB, total: {info['usage_gb']:.2f} GB)"

POSIX_FADV_DONTNEED = 4

def drop_file_page_cache(file_path):
    if sys.platform != 'linux':
        return
    try:
        libc = ctypes.CDLL('libc.so.6')
        fd = os.open(file_path, os.O_RDONLY)
        try:
            libc.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except Exception:
        pass

def imread_gray(path):
    if not path:
        return None
    try:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is not None:
            drop_file_page_cache(path)
            return im
    except Exception:
        pass
    try:
        with Image.open(path) as img:
            im = np.array(img.convert('L'), dtype=np.uint8)
            drop_file_page_cache(path)
            return im
    except Exception:
        return None

WORKING_DIR = '/kaggle/working' if os.path.exists('/kaggle') else './workspace'
os.makedirs(WORKING_DIR, exist_ok=True)
CACHE_DIR = os.path.join(WORKING_DIR, 'dataset_cache_v3.2')
os.makedirs(CACHE_DIR, exist_ok=True)
CKPT_DIR = os.path.join(WORKING_DIR, 'checkpoints_v3.2')
os.makedirs(CKPT_DIR, exist_ok=True)

print(f"[*] Thư mục làm việc: {WORKING_DIR}")
print(f"[*] RAM Host ban đầu : {format_host_ram()}")
print(f"[*] Watchdog RAM     : cảnh báo > {RAM_WARN_GB:.0f} GB | lưu resume & dừng êm > {RAM_SOFT_LIMIT_GB:.0f} GB")
print("=" * 80)
sys.stdout.flush()
