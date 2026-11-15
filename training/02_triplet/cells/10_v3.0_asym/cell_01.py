# [CELL 1]: Cấu hình Môi trường, Phần cứng Kaggle (2x T4 GPU) & Trình Dọn Rác Bộ Nhớ Chủ Động
import os
import sys
import ctypes

os.environ["MALLOC_TRIM_THRESHOLD_"] = "65536"
os.environ["MALLOC_ARENA_MAX"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# [AN TOÀN MULTI-GPU]: Không bật expandable_segments:True khi dùng nn.DataParallel trên 2x T4
# để tránh lỗi cuMemMap / illegal memory access khi cấp phát chéo stream giữa cuda:0 và cuda:1

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

# [CHIẾN LƯỢC CHIA SẺ TENSOR GIỮA WORKER]: 'file_descriptor' (mặc định PyTorch trên Linux).
# Lưu ý: 'file_system' KHÔNG ghi xuống ổ đĩa - nó vẫn tạo segment trong /dev/shm (RAM tmpfs) và
# chỉ được dọn khi torch_shm_manager thu gom => dễ để lại rác. 'file_descriptor' shm_unlink ngay,
# kernel tự thu hồi khi fd đóng.
try:
    mp.set_sharing_strategy('file_descriptor')
except Exception:
    pass

# Nâng giới hạn số file descriptor (file_descriptor strategy cần nhiều fd)
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
# [CỰC KỲ QUAN TRỌNG TRÊN 2x T4]: Tắt cudnn.benchmark để ngăn xung đột cuDNN autotuner đa luồng
torch.backends.cudnn.benchmark = False

NUM_GPUS = torch.cuda.device_count()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("=" * 80)
print(f"[*] THIẾT LẬP MÔI TRƯỜNG & PHẦN CỨNG HUẤN LUYỆN V3.0 (14 CELLS)")
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
    """Đọc memory.stat (cgroup v2 hoặc v1) -> dict {anon, file, shmem, active_file, inactive_file} (bytes)."""
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
    """Thống kê RAM chính xác (không cộng trùng RSS):
    working_set (= thanh RAM Kaggle), anon, file (PageCache), shmem, /dev/shm, PSS tổng, số tiến trình con."""
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
            pss = proc.memory_info().rss  # fallback (Windows/không có smaps_rollup)
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
    """Giải phóng các trang file đơn lẻ khỏi Linux OS Page Cache (TUYỆT ĐỐI không walk thư mục FUSE)."""
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

_HAS_FADVISE = hasattr(os, 'posix_fadvise') and hasattr(os, 'POSIX_FADV_DONTNEED')

def imread_gray(path):
    """Đọc ảnh xám: os.read -> cv2.imdecode, sau đó POSIX_FADV_DONTNEED để bỏ file khỏi PageCache ngay
    (chống PageCache phình theo epoch khi WeightedRandomSampler chạm dần file mới)."""
    if not path:
        return None
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            chunks = []
            while True:
                b = os.read(fd, 1 << 20)
                if not b:
                    break
                chunks.append(b)
            buf = b''.join(chunks)
            if _HAS_FADVISE:
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                except Exception:
                    pass
        finally:
            os.close(fd)
        im = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        del buf, chunks
        if im is not None:
            return im
    except Exception:
        pass
    try:
        with Image.open(path) as img:
            return np.array(img.convert('L'), dtype=np.uint8)
    except Exception:
        return None

WORKING_DIR = '/kaggle/working' if os.path.exists('/kaggle') else './workspace'
os.makedirs(WORKING_DIR, exist_ok=True)
CACHE_DIR = os.path.join(WORKING_DIR, 'dataset_cache')
os.makedirs(CACHE_DIR, exist_ok=True)
CKPT_DIR = os.path.join(WORKING_DIR, 'checkpoints_v3.0')
os.makedirs(CKPT_DIR, exist_ok=True)

print(f"[*] Thư mục làm việc: {WORKING_DIR}")
print(f"[*] RAM Host ban đầu : {format_host_ram()}")
print(f"[*] Watchdog RAM     : cảnh báo > {RAM_WARN_GB:.0f} GB | lưu resume & dừng êm > {RAM_SOFT_LIMIT_GB:.0f} GB")
print("=" * 80)
sys.stdout.flush()
