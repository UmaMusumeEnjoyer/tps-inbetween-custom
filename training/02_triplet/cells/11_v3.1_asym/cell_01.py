# [CELL 1]: Cấu hình Môi trường, Phần cứng Kaggle (2x T4 / 1x T4 / A100) & Trình Dọn Rác Bộ Nhớ Chủ Động V3.1
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
# [CẤU HÌNH CUDNN BENCHMARK V3.1]:
# Bật benchmark khi chạy đơn GPU hoặc trên các tiến trình độc lập DDP (mỗi GPU 1 tiến trình riêng).
# Trên DataParallel đơn luồng chia sẻ context, cudnn benchmark bị tắt để tránh lock autotuner.
torch.backends.cudnn.benchmark = (NUM_GPUS <= 1)
print("=" * 80)
print(f"[*] THIẾT LẬP MÔI TRƯỜNG & PHẦN CỨNG HUẤN LUYỆN V3.1 (14 CELLS)")
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
    for p, mapping in (
        ('/sys/fs/cgroup/memory.stat', {'anon': 'anon', 'file': 'file', 'shmem': 'shmem',
                                        'active_file': 'active_file', 'inactive_file': 'inactive_file'}),
        ('/sys/fs/cgroup/memory/memory.stat', {'total_rss': 'anon', 'total_cache': 'file', 'total_shmem': 'shmem',
                                               'total_active_file': 'active_file', 'total_inactive_file': 'inactive_file'}),
    ):
        try:
            with open(p, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) == 2 and parts[0] in mapping:
                        out[mapping[parts[0]]] = int(parts[1])
            if out:
                return out
        except Exception:
            continue
    return out

def _read_pss_bytes(pid):
    try:
        with open(f'/proc/{pid}/smaps_rollup', 'r') as f:
            for line in f:
                if line.startswith('Pss:'):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0

def get_memory_stats():
    G = 1024 ** 3
    s = {'cgroup': 0.0, 'working_set': 0.0, 'anon': 0.0, 'file': 0.0, 'shmem': 0.0,
         'devshm': 0.0, 'pss': 0.0, 'children': 0}
    cg = _read_cgroup_memory_bytes()
    st = _read_cgroup_memory_stat()
    if cg is not None:
        s['cgroup'] = cg / G
        s['working_set'] = max(0, cg - st.get('inactive_file', 0)) / G
    for k in ('anon', 'file', 'shmem'):
        if k in st:
            s[k] = st[k] / G
    try:
        if os.path.exists('/dev/shm'):
            s['devshm'] = shutil.disk_usage('/dev/shm').used / G
    except Exception:
        pass
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        children = proc.children(recursive=True)
        s['children'] = len(children)
        pss = _read_pss_bytes(proc.pid) + sum(_read_pss_bytes(c.pid) for c in children)
        if pss == 0:
            pss = proc.memory_info().rss
        s['pss'] = pss / G
        if cg is None:
            s['working_set'] = s['pss']
    except Exception:
        pass
    return s

def get_host_ram_gb():
    return get_memory_stats()['working_set']

def format_host_ram(s=None):
    s = s or get_memory_stats()
    return (f"WS={s['working_set']:.2f} GB | anon={s['anon']:.2f} | file={s['file']:.2f} | "
            f"shmem={s['shmem']:.2f} | /dev/shm={s['devshm']:.2f} | PSS={s['pss']:.2f} | children={s['children']}")

def drop_file_page_cache(path_or_fd):
    if not hasattr(os, 'posix_fadvise') or not hasattr(os, 'POSIX_FADV_DONTNEED'):
        return
    try:
        if isinstance(path_or_fd, int):
            os.posix_fadvise(path_or_fd, 0, 0, os.POSIX_FADV_DONTNEED)
        elif isinstance(path_or_fd, str) and os.path.exists(path_or_fd) and os.path.isfile(path_or_fd):
            fd = os.open(path_or_fd, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
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
CACHE_DIR = os.path.join(WORKING_DIR, 'dataset_cache_v3.1')
os.makedirs(CACHE_DIR, exist_ok=True)
CKPT_DIR = os.path.join(WORKING_DIR, 'checkpoints_v3.1')
os.makedirs(CKPT_DIR, exist_ok=True)

print(f"[*] Thư mục làm việc: {WORKING_DIR}")
print(f"[*] RAM Host ban đầu : {format_host_ram()}")
print(f"[*] Watchdog RAM     : cảnh báo > {RAM_WARN_GB:.0f} GB | lưu resume & dừng êm > {RAM_SOFT_LIMIT_GB:.0f} GB")
print("=" * 80)
sys.stdout.flush()
