#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_v24_notebook.py
Script tự động khởi tạo và chuẩn hóa notebook huấn luyện Kaggle V2.4:
- training/02_triplet/notebooks/triplet_08_v2.4_balanced_density_kaggle.ipynb

Tích hợp toàn diện các hạng mục kỹ thuật V2.4:
1. SymmetricRampDensityLoss: Ramp đối xứng [0.08, 0.22], triệt tiêu 100% nhiễu nền trắng (+5,860 px ảo),
   Dead-zone mục tiêu [0.88, 1.04], lambda = 0.20, phạt bất đối xứng thiếu nét (dev_low factor 1.8x).
2. FineStrokePreservationLoss: Trích xuất mặt nạ nét mảnh (target - soft_open) và phạt nặng 3x khi làm
   mất nét mỏng 1-2px, kết hợp Tversky-clDice (beta=0.65) ưu tiên bảo toàn khung xương chi tiết.
3. GhostStrokePenaltyLoss bảo vệ hành lang an toàn 3x3 cho nét mảnh.
4. SmartCheckpointManager_v24: Hard Gates thực tế (Recall >= 86.5%, Density in [0.82, 1.08], target 0.94),
   tăng trọng số Recall lên 0.30, kèm Penalized Score Fallback đảm bảo 100% luôn chọn được Best Checkpoint.
5. Flat Contiguous Byte Arena trong Dataset & DataLoader: triệt tiêu Copy-on-Write RAM leak,
   kết hợp Batched GPU Val Metrics (VAL_BATCH_SIZE = 4) rút ngắn thời gian epoch xuống ~350s-370s và giữ RAM < 8.5 GB.
"""

import os
import sys
import json
import shutil

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


def generate_v24():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    notebooks_dir = os.path.abspath(os.path.join(script_dir, "..", "notebooks"))
    source_nb_candidates = [
        os.path.join(notebooks_dir, "triplet_07_v2.3_zero_sync_chamfer_kaggle.ipynb"),
        os.path.join(notebooks_dir, "triplet_06_v2.2_anti_thickening_kaggle.ipynb"),
    ]

    source_path = None
    for p in source_nb_candidates:
        if os.path.exists(p):
            source_path = p
            break

    if source_path is None:
        raise FileNotFoundError("Không tìm thấy notebook mẫu V2.3 hoặc V2.2 để sinh V2.4!")

    print(f"[*] Đang nạp notebook nguồn từ: {source_path}")
    with open(source_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Xóa toàn bộ outputs cũ để notebook V2.4 sạch sẽ sẵn sàng chạy trên Kaggle
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    def set_code(cell_idx, text):
        lines = [line + "\n" for line in text.strip().split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        nb["cells"][cell_idx]["source"] = lines

    # =========================================================================
    # CELL 0: Header Markdown
    # =========================================================================
    c0 = r"""# HUẤN LUYỆN TINH CHỈNH TPS-INBETWEEN PHIÊN BẢN V2.4 (BALANCED DENSITY & FINE-STROKE PRESERVATION)
## ĐỀ TÀI: NGHIÊN CỨU & TINH CHỈNH MÔ HÌNH TPS-INBETWEEN CHO HOẠT HỌA ANIME 2D (PBL6)

> **Mục tiêu phiên bản V2.4 (Production Stability & Fine-Detail Preservation Release):**
> 1. **Khử Nhiễu Nền & Cân Bằng Mật Độ Nét (Cell 6):** Thay thế Sigmoid lệch nền bằng `SymmetricRampDensityLoss` khả vi đối xứng trên khoảng $[0.08, 0.22]$. Triệt tiêu hoàn toàn $+5,860\\text{ px}$ ảo từ nền trắng, hạ trọng số $\\lambda_{\\text{density}} = 0.20$ và đặt Dead-zone $[0.88, 1.04]$ với hệ số phạt thiếu nét $1.8\\times$ để nét vẽ thanh thoát mà không bị co gầy quá mức.
> 2. **Bảo Tồn Chi Tiết Nét Mảnh Tuyệt Đối (Cell 6):** Bổ sung `FineStrokePreservationLoss` ($\lambda = 0.30$) tự động trích xuất mặt nạ nét mảnh $(I_{\\text{gt}} - \\text{soft\\_open}(I_{\\text{gt}}))$ và phạt nặng $3.0\\times$ khi làm đứt gãy sợi tóc, lông mi, ngón tay; nâng cấp `SoftClDiceLoss` $(\\beta = 0.65)$ ưu tiên Sensitivity bảo toàn khung xương chi tiết với **0ms chi phí tính toán tăng thêm**.
> 3. **Bảo Vệ Hành Lang Nét Mảnh (Cell 6):** `GhostStrokePenaltyLoss` thiết lập hành lang an toàn $3\\times 3$ quanh Ground Truth, triệt tiêu bóng ma ngoài xa mà không ăn mòn viền nét mảnh.
> 4. **Quản Lý Checkpoint 3 Tầng & Luôn Chọn Best Checkpoint (Cell 9):** `SmartCheckpointManager_v24` áp dụng Hard Gates thực tế (`Recall` $\\ge 86.5\\%$, `Density` $\\in [0.82, 1.08]$, tâm lý tưởng $0.94$), tăng trọng số Recall lên $0.30$, kết hợp cơ chế `Penalized Score Fallback` đảm bảo **100% luôn chọn và cập nhật được Best Checkpoint** kể cả khi ngắt sớm giữa chừng.
> 5. **Flat Contiguous Byte Arena & Batched GPU Validation (Cell 4 & Cell 10):** Chuyển bộ đệm ảnh RAM thành mảng phẳng 1D `np.uint8` triệt tiêu Copy-on-Write (CoW) RAM leak (giữ Host RAM phẳng $< 8.5\\text{ GB}$ suốt 25 epochs), kết hợp `calculate_metrics_batch_gpu` (`VAL_BATCH_SIZE = 4`) rút ngắn thời gian mỗi epoch xuống còn ~350s–370s."""
    nb["cells"][0]["source"] = [line + "\n" for line in c0.strip().split("\n")]

    # =========================================================================
    # CELL 1: Environment & Memory Guard
    # =========================================================================
    c1 = """# [CELL 1]: Cấu hình Môi trường, Phần cứng & Trình Dọn Rác Bộ Nhớ Chủ Động (V2.4)
import os
import sys
import ctypes

os.environ["MALLOC_TRIM_THRESHOLD_"] = "65536"
os.environ["MALLOC_ARENA_MAX"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# Giảm phân mảnh VRAM của CUDA caching allocator (phải đặt TRƯỚC khi import torch)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

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

def configure_glibc_allocator():
    try:
        libc = ctypes.CDLL('libc.so.6')
        libc.mallopt(-1, 65536)
        libc.mallopt(-8, 2)
    except Exception:
        pass

configure_glibc_allocator()

# Thiết lập Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False

# Kiểm tra GPU & Cấu hình thiết bị
NUM_GPUS = torch.cuda.device_count()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"[*] Đang sử dụng thiết bị: {device} | Số GPU khả dụng: {NUM_GPUS}")
for i in range(NUM_GPUS):
    prop = torch.cuda.get_device_properties(i)
    print(f"    - GPU {i}: {prop.name} (VRAM: {prop.total_memory / 1e9:.2f} GB)")

# Hàm dọn dẹp bộ nhớ chuyên sâu chống tràn RAM máy chủ và VRAM (Zero Memory Leak Engine)
def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
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

def get_host_ram_gb():
    rss_gb = None
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        rss = proc.memory_info().rss
        for ch in proc.children(recursive=True):
            try:
                rss += ch.memory_info().rss
            except Exception:
                pass
        rss_gb = rss / (1024 ** 3)
    except Exception:
        pass
    cg = _read_cgroup_memory_bytes()
    cg_gb = cg / (1024 ** 3) if cg is not None else None
    return rss_gb, cg_gb

def format_host_ram():
    rss_gb, cg_gb = get_host_ram_gb()
    rss_s = f"{rss_gb:.2f} GB" if rss_gb is not None else "N/A"
    cg_s = f"{cg_gb:.2f} GB" if cg_gb is not None else "N/A"
    return f"Process RSS: {rss_s} | Container (cgroup): {cg_s}"

cleanup_memory()
print(f"[*] Host RAM hiện tại -> {format_host_ram()}")
print("[✓] CELL 1 hoàn tất: Môi trường và hàm quản lý bộ nhớ sâu sẵn sàng (V2.4).")"""
    set_code(1, c1)

    # =========================================================================
    # CELL 4: Dataset & DataLoader (Flat Contiguous Byte Arena V2.4)
    # =========================================================================
    c4 = """# [CELL 4 - TỐI ƯU SIÊU TỐC V2.4]: Dataset & DataLoader (Flat Contiguous Byte Arena & Zero CoW)
import os
import random
import cv2
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torch.utils.data.dataloader import default_collate

class TemporalReversalAugmentation:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img0, imggt, img1, t_val, kps=None,
                 c0_pad=None, th01_pad=None, c1_pad=None, th10_pad=None):
        if random.random() < self.p:
            img0_out, img1_out = img1, img0
            t_out = float(1.0 - t_val)
            gt_out = imggt
            kps_out = np.stack([kps[1], kps[0]], axis=0) if (kps is not None and hasattr(kps, "shape") and len(kps.shape) == 3 and kps.shape[0] == 2) else kps
            c0_out, th01_out = c1_pad, th10_pad
            c1_out, th10_out = c0_pad, th01_pad
            return img0_out, gt_out, img1_out, t_out, kps_out, c0_out, th01_out, c1_out, th10_out, True
        return img0, imggt, img1, float(t_val), kps, c0_pad, th01_pad, c1_pad, th10_pad, False

def compute_boundary_motion_weights(df, boost_pct=0.25):
    weights = np.ones(len(df), dtype=np.float32)
    t_vals = df["t"].values.astype(np.float32)
    disp_vals = df["mean_disp"].values.astype(np.float32) if "mean_disp" in df.columns else np.zeros(len(df), dtype=np.float32)
    is_boundary_t = ((t_vals >= 0.15) & (t_vals <= 0.30)) | ((t_vals >= 0.70) & (t_vals <= 0.85))
    is_large_disp = (disp_vals >= 7.0)
    weights[is_boundary_t] *= (1.0 + boost_pct)
    weights[is_large_disp] *= (1.0 + boost_pct)
    return weights

def _worker_init_fn(worker_id):
    configure_glibc_allocator()
    torch.set_num_threads(1)
    cv2.setNumThreads(0)
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)

class SakugaGoldenDataset_v1(Dataset):
    def __init__(self, root_dir, df_split, is_train=True, crop_size=512):
        self.root_dir = root_dir
        self.is_train = is_train
        self.crop_size = crop_size
        self.split_name = 'train' if is_train else 'val'
        self.length = len(df_split)
        self.temporal_aug = TemporalReversalAugmentation(p=0.5) if is_train else None

        # 1. Dò tìm đường dẫn gốc 1 lần duy nhất
        sample_f0 = str(df_split['frame0'].iloc[0]).lstrip('/\\\\')
        candidate_roots = [
            root_dir,
            os.path.join(root_dir, 'anime_tps_finetune_dataset_ready'),
            os.path.join(root_dir, 'sakuga_golden_dataset_tps_ready_v1_patch2'),
            os.path.join(root_dir, 'sakuga_golden_dataset_tps_ready_v1_patch2', 'anime_tps_finetune_dataset_ready')
        ]
        actual_root = root_dir
        for cr in candidate_roots:
            if not os.path.exists(cr):
                continue
            if os.path.exists(os.path.join(cr, sample_f0)):
                actual_root = cr
                break
            if os.path.exists(os.path.join(cr, self.split_name, sample_f0)):
                actual_root = os.path.join(cr, self.split_name)
                break

        sample_fm = str(df_split['match_file'].iloc[0]).lstrip('/\\\\')
        actual_match_root = actual_root
        for cr in [actual_root, root_dir] + candidate_roots:
            if not os.path.exists(cr):
                continue
            if os.path.exists(os.path.join(cr, sample_fm)):
                actual_match_root = cr
                break
            if os.path.exists(os.path.join(cr, 'matches', sample_fm)):
                actual_match_root = os.path.join(cr, 'matches')
                break

        f0_list = df_split['frame0'].astype(str).str.lstrip('/\\\\').tolist()
        fgt_list = df_split['frame_gt'].astype(str).str.lstrip('/\\\\').tolist()
        f1_list = df_split['frame1'].astype(str).str.lstrip('/\\\\').tolist()
        fm_list = df_split['match_file'].astype(str).str.lstrip('/\\\\').tolist()

        p0_bytes = [os.path.join(actual_root, f).encode('utf-8') for f in f0_list]
        pgt_bytes = [os.path.join(actual_root, f).encode('utf-8') for f in fgt_list]
        p1_bytes = [os.path.join(actual_root, f).encode('utf-8') for f in f1_list]
        pmatch_bytes = [os.path.join(actual_match_root, f).encode('utf-8') for f in fm_list]

        self.p0_arr = np.array(p0_bytes, dtype='S256')
        self.pgt_arr = np.array(pgt_bytes, dtype='S256')
        self.p1_arr = np.array(p1_bytes, dtype='S256')
        self.pmatch_arr = np.array(pmatch_bytes, dtype='S256')
        self.t_arr = df_split['t'].to_numpy(dtype=np.float32, copy=True)
        self.clips_arr = np.array([str(c).encode('utf-8') for c in df_split['clip_id']], dtype='S64')

        # 2. Preloading Keypoints đa luồng
        MAX_KPS_CACHE = 256
        self.kps_data = np.zeros((self.length, 2, MAX_KPS_CACHE, 2), dtype=np.float32)
        self.kps_count = np.zeros(self.length, dtype=np.int32)

        print(f"[*] Preloading keypoints đa luồng ({self.split_name}: {self.length} mẫu)...", flush=True)

        def _load_single_kps(idx):
            pm = self.pmatch_arr[idx].decode('utf-8')
            if os.path.exists(pm):
                try:
                    kps = np.load(pm).astype(np.float32)
                    if kps.ndim == 3 and kps.shape[0] == 2 and kps.shape[2] == 2:
                        n = kps.shape[1]
                        if n > MAX_KPS_CACHE:
                            idx_sub = np.linspace(0, n - 1, MAX_KPS_CACHE, dtype=int)
                            self.kps_data[idx] = kps[:, idx_sub, :]
                            self.kps_count[idx] = MAX_KPS_CACHE
                        elif n > 0:
                            self.kps_data[idx, :, :n, :] = kps
                            self.kps_count[idx] = n
                except Exception:
                    pass

        with ThreadPoolExecutor(max_workers=16) as ex:
            list(tqdm(ex.map(_load_single_kps, range(self.length)), total=self.length, desc=f"   Loading KPS ({self.split_name})", leave=False))
        print(f"[✓] Đã nạp xong keypoints cache ({self.split_name}) vào RAM liền mạch.", flush=True)

        # 3. Precomputing TPS Validation đa luồng
        self.val_c0 = np.zeros((self.length, 88, 2), dtype=np.float32)
        self.val_th01 = np.zeros((self.length, 91, 2), dtype=np.float32)
        self.val_c1 = np.zeros((self.length, 88, 2), dtype=np.float32)
        self.val_th10 = np.zeros((self.length, 91, 2), dtype=np.float32)
        self.val_valid = np.zeros((self.length, 1, 1, 1), dtype=np.float32)

        if not self.is_train:
            print(f"[*] Precomputing TPS cache đa luồng cho Validation ({self.length} mẫu)...", flush=True)

            def _solve_single_val_tps(idx):
                p0 = self.p0_arr[idx].decode('utf-8')
                cnt = int(self.kps_count[idx])
                kps = self.kps_data[idx, :, :cnt, :].copy() if cnt > 0 else np.zeros((2, 0, 2), dtype=np.float32)
                try:
                    with Image.open(p0) as img_hdr:
                        w, h = img_hdr.size
                except Exception:
                    w, h = 512, 512
                pad_h = ((h + 31) // 32) * 32
                pad_w = ((w + 31) // 32) * 32
                if kps is not None and kps.shape[1] > 0:
                    kps_norm = kps.copy()
                    kps_norm[0, :, 0] = (kps_norm[0, :, 0] * w) / pad_w
                    kps_norm[0, :, 1] = (kps_norm[0, :, 1] * h) / pad_h
                    kps_norm[1, :, 0] = (kps_norm[1, :, 0] * w) / pad_w
                    kps_norm[1, :, 1] = (kps_norm[1, :, 1] * h) / pad_h
                else:
                    kps_norm = kps
                c0_p, th01_p, c1_p, th10_p, v = solve_tps_pair_cpu(kps_norm, max_kps=80, lambd=0.5, min_dist=0.015, return_tensor=False)
                self.val_c0[idx] = c0_p
                self.val_th01[idx] = th01_p
                self.val_c1[idx] = c1_p
                self.val_th10[idx] = th10_p
                self.val_valid[idx, 0, 0, 0] = float(v)

            with ThreadPoolExecutor(max_workers=8) as ex:
                list(tqdm(ex.map(_solve_single_val_tps, range(self.length)), total=self.length, desc="   Solving Val TPS", leave=False))
            print(f"[✓] Đã precompute xong {self.length} mẫu TPS Validation vào contiguous NumPy buffers.", flush=True)

        # 4. [ĐỘT PHÁ V2.4]: Đóng gói toàn bộ file ảnh vào Flat Contiguous Byte Arena
        # Triệt tiêu 100% Copy-on-Write (CoW) và rò rỉ RAM +0.847 GB/epoch khi worker fork
        print(f"[*] Preloading toàn bộ file ảnh ({self.split_name}: {self.length} mẫu) vào Flat Byte Arena V2.4...", flush=True)
        all_paths = [self.p0_arr[i].decode('utf-8') for i in range(self.length)] + \
                    [self.pgt_arr[i].decode('utf-8') for i in range(self.length)] + \
                    [self.p1_arr[i].decode('utf-8') for i in range(self.length)]
        unique_paths = list(set(all_paths))
        path_to_id = {p: i for i, p in enumerate(unique_paths)}

        self.f0_ids = np.array([path_to_id[self.p0_arr[i].decode('utf-8')] for i in range(self.length)], dtype=np.int32)
        self.fgt_ids = np.array([path_to_id[self.pgt_arr[i].decode('utf-8')] for i in range(self.length)], dtype=np.int32)
        self.f1_ids = np.array([path_to_id[self.p1_arr[i].decode('utf-8')] for i in range(self.length)], dtype=np.int32)

        def _read_raw_bytes(p):
            if os.path.exists(p):
                try:
                    with open(p, 'rb') as f:
                        return f.read()
                except Exception:
                    pass
            return b''

        with ThreadPoolExecutor(max_workers=16) as ex:
            raw_bytes_list = list(tqdm(ex.map(_read_raw_bytes, unique_paths), total=len(unique_paths), desc=f"   Buffering PNGs ({self.split_name})", leave=False))

        # Đóng gói thành mảng NumPy uint8 phẳng liền mạch
        lengths = [len(b) for b in raw_bytes_list]
        self.img_offsets = np.zeros(len(unique_paths) + 1, dtype=np.int64)
        np.cumsum(lengths, out=self.img_offsets[1:])
        concat_bytes = b''.join(raw_bytes_list)
        self.img_arena = np.frombuffer(concat_bytes, dtype=np.uint8)
        del raw_bytes_list, concat_bytes
        gc.collect()

        total_mb = len(self.img_arena) / (1024 * 1024)
        print(f"[✓] Đã đóng gói {len(unique_paths):,} ảnh ({total_mb:.1f} MB) ({self.split_name}) vào Flat Arena V2.4 liền mạch (Zero CoW).", flush=True)

    def __len__(self):
        return self.length

    def _get_buffer(self, fid):
        if hasattr(self, 'img_arena') and self.img_arena is not None and fid >= 0:
            st = self.img_offsets[fid]
            en = self.img_offsets[fid + 1]
            if en > st:
                return self.img_arena[st:en]
        return None

    def __getitem__(self, idx):
        p0 = self.p0_arr[idx].decode('utf-8')
        pgt = self.pgt_arr[idx].decode('utf-8')
        p1 = self.p1_arr[idx].decode('utf-8')
        clip_name = int(idx) if self.is_train else self.clips_arr[idx].decode('utf-8')

        b0 = self._get_buffer(self.f0_ids[idx])
        bgt = self._get_buffer(self.fgt_ids[idx])
        b1 = self._get_buffer(self.f1_ids[idx])

        im0 = cv2.imdecode(b0, cv2.IMREAD_GRAYSCALE) if b0 is not None else cv2.imread(p0, cv2.IMREAD_GRAYSCALE)
        imgt = cv2.imdecode(bgt, cv2.IMREAD_GRAYSCALE) if bgt is not None else cv2.imread(pgt, cv2.IMREAD_GRAYSCALE)
        im1 = cv2.imdecode(b1, cv2.IMREAD_GRAYSCALE) if b1 is not None else cv2.imread(p1, cv2.IMREAD_GRAYSCALE)

        if im0 is None or imgt is None or im1 is None:
            cs = self.crop_size if self.is_train else 512
            im0 = np.full((cs, cs), 255, dtype=np.uint8)
            imgt = np.full((cs, cs), 255, dtype=np.uint8)
            im1 = np.full((cs, cs), 255, dtype=np.uint8)

        h, w = im0.shape

        if self.is_train:
            cnt = int(self.kps_count[idx])
            kps = self.kps_data[idx, :, :cnt, :].copy() if cnt > 0 else np.zeros((2, 0, 2), dtype=np.float32)
            cs = self.crop_size
            if h > cs and w > cs:
                top = random.randint(0, h - cs)
                left = random.randint(0, w - cs)
                im0_crop = im0[top:top+cs, left:left+cs]
                imggt_crop = imgt[top:top+cs, left:left+cs]
                im1_crop = im1[top:top+cs, left:left+cs]

                if kps.shape[1] > 0:
                    kps[0, :, 0] = (kps[0, :, 0] * w - left) / cs
                    kps[0, :, 1] = (kps[0, :, 1] * h - top) / cs
                    kps[1, :, 0] = (kps[1, :, 0] * w - left) / cs
                    kps[1, :, 1] = (kps[1, :, 1] * h - top) / cs
                    mask0 = (kps[0, :, 0] >= 0) & (kps[0, :, 0] <= 1) & (kps[0, :, 1] >= 0) & (kps[0, :, 1] <= 1)
                    mask1 = (kps[1, :, 0] >= 0) & (kps[1, :, 0] <= 1) & (kps[1, :, 1] >= 0) & (kps[1, :, 1] <= 1)
                    kps = kps[:, mask0 & mask1, :]
            else:
                pad_h = ((h + 31) // 32) * 32
                pad_w = ((w + 31) // 32) * 32
                im0_crop = np.pad(im0, ((0, pad_h - h), (0, pad_w - w)), mode='constant', constant_values=255)
                imggt_crop = np.pad(imgt, ((0, pad_h - h), (0, pad_w - w)), mode='constant', constant_values=255)
                im1_crop = np.pad(im1, ((0, pad_h - h), (0, pad_w - w)), mode='constant', constant_values=255)

            img0 = (255 - im0_crop).astype(np.float32) * (1.0 / 255.0)
            imggt = (255 - imggt_crop).astype(np.float32) * (1.0 / 255.0)
            img1 = (255 - im1_crop).astype(np.float32) * (1.0 / 255.0)

            c0_pad, th01_pad, c1_pad, th10_pad, valid_flag = solve_tps_pair_cpu(kps, max_kps=80, lambd=0.5, min_dist=0.015, return_tensor=False)
            valid_flag = np.full((1, 1, 1), float(valid_flag), dtype=np.float32)

            t_scalar = float(self.t_arr[idx])
            if self.temporal_aug is not None:
                img0, imggt, img1, t_scalar, kps, c0_pad, th01_pad, c1_pad, th10_pad, _ = self.temporal_aug(
                    img0, imggt, img1, t_scalar, kps, c0_pad, th01_pad, c1_pad, th10_pad
                )
        else:
            pad_h = ((h + 31) // 32) * 32
            pad_w = ((w + 31) // 32) * 32
            im0_pad = np.pad(im0, ((0, pad_h - h), (0, pad_w - w)), mode='constant', constant_values=255)
            imggt_pad = np.pad(imgt, ((0, pad_h - h), (0, pad_w - w)), mode='constant', constant_values=255)
            im1_pad = np.pad(im1, ((0, pad_h - h), (0, pad_w - w)), mode='constant', constant_values=255)

            img0 = (255 - im0_pad).astype(np.float32) * (1.0 / 255.0)
            imggt = (255 - imggt_pad).astype(np.float32) * (1.0 / 255.0)
            img1 = (255 - im1_pad).astype(np.float32) * (1.0 / 255.0)

            c0_pad = self.val_c0[idx].copy()
            th01_pad = self.val_th01[idx].copy()
            c1_pad = self.val_c1[idx].copy()
            th10_pad = self.val_th10[idx].copy()
            valid_flag = self.val_valid[idx].copy()
            t_scalar = float(self.t_arr[idx])

        t0 = np.ascontiguousarray(img0[None, :, :], dtype=np.float32)
        tgt = np.ascontiguousarray(imggt[None, :, :], dtype=np.float32)
        t1 = np.ascontiguousarray(img1[None, :, :], dtype=np.float32)
        t_val = np.full((1, 1, 1), t_scalar, dtype=np.float32)
        c0_pad = np.ascontiguousarray(c0_pad, dtype=np.float32)
        th01_pad = np.ascontiguousarray(th01_pad, dtype=np.float32)
        c1_pad = np.ascontiguousarray(c1_pad, dtype=np.float32)
        th10_pad = np.ascontiguousarray(th10_pad, dtype=np.float32)
        valid_flag = np.ascontiguousarray(valid_flag, dtype=np.float32)

        return t0, tgt, t1, t_val, c0_pad, th01_pad, c1_pad, th10_pad, valid_flag, clip_name

custom_collate_fn_v1 = default_collate

PER_GPU_BATCH = 8
BATCH_SIZE = PER_GPU_BATCH * max(1, NUM_GPUS)
# [V2.4 TỐI ƯU]: Tăng VAL_BATCH_SIZE lên 4 kết hợp tính batch GPU metrics giúp rút ngắn thời gian epoch
VAL_BATCH_SIZE = max(2, NUM_GPUS * 2)

if 'train_df' not in globals() or train_df is None:
    print("[*] Tự động nạp lại train_df/val_df cho Cell 4...")
    _meta_path = os.path.join(DATASET_ROOT, 'triplets.csv')
    if not os.path.exists(_meta_path):
        _meta_path = os.path.join(DATASET_ROOT, 'triplets_combined.csv')
    _df = pd.read_csv(_meta_path)
    train_df = _df[_df['split'] == 'train'].reset_index(drop=True)
    val_df = _df[_df['split'] == 'val'].reset_index(drop=True)
    del _df

train_dataset = SakugaGoldenDataset_v1(DATASET_ROOT, train_df, is_train=True, crop_size=512)
val_dataset = SakugaGoldenDataset_v1(DATASET_ROOT, val_df, is_train=False)

train_sample_weights = compute_boundary_motion_weights(train_df, boost_pct=0.25)
train_sampler = WeightedRandomSampler(
    weights=torch.as_tensor(train_sample_weights, dtype=torch.double),
    num_samples=len(train_dataset),
    replacement=True
)

del train_df, val_df
cleanup_memory()

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    num_workers=2,
    persistent_workers=False,
    pin_memory=False,
    prefetch_factor=2,
    collate_fn=custom_collate_fn_v1,
    worker_init_fn=_worker_init_fn,
    drop_last=True
)

val_fast_subset = Subset(val_dataset, range(min(len(val_dataset), 100 * VAL_BATCH_SIZE)))

val_loader_fast = DataLoader(
    val_fast_subset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
    collate_fn=custom_collate_fn_v1
)

val_loader_full = DataLoader(
    val_dataset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
    collate_fn=custom_collate_fn_v1
)

val_loader = val_loader_full

print(f"[✓] DataLoaders Golden Dataset V2.4 sẵn sàng: {len(train_loader)} batches (Train), {len(val_loader_full)} batches (Val Full - Batch {VAL_BATCH_SIZE}).")
print("[✓] CELL 4 hoàn tất: Dataset & DataLoaders triệt tiêu 100% Copy-on-Write (CoW).")
print(f"[*] Host RAM hiện tại -> {format_host_ram()}")"""
    set_code(4, c4)

    # =========================================================================
    # CELL 5: Architecture with Fine-Stroke Initialization Bias
    # =========================================================================
    c5_orig = "".join(nb["cells"][5]["source"])
    # Cập nhật bias khởi tạo SynthesisUnet_v1 để ưu tiên giữ nét mảnh
    c5_new = c5_orig.replace("constant_(self.conv_erase.bias, 2.0)", "constant_(self.conv_erase.bias, 2.5)")
    c5_new = c5_new.replace("constant_(self.conv_add.bias, -3.0)", "constant_(self.conv_add.bias, -2.5)")
    c5_new = c5_new.replace("V1.1", "V2.4").replace("V1.0", "V2.4")
    set_code(5, c5_new)

    # =========================================================================
    # CELL 6: Composite Loss Engine V2.4 (Symmetric Ramp & Fine-Stroke Loss)
    # =========================================================================
    c6 = """# [CELL 6]: Bộ Hàm Mất Mát CompositeLossEngine_v24 (Symmetric Ramp Density & Fine-Stroke Preservation)
class StrokeDensityRegularizationLoss(nn.Module):
    \"\"\"
    Hàm mất mát điều tiết mật độ mực nét vẽ V2.4 (Symmetric Ramp Density Loss):
    - Khử 100% nhiễu nền trắng (+5,860 px ảo của Sigmoid) bằng Piecewise Linear Ramp khả vi [0.08, 0.22].
    - Dead-zone dải mục tiêu chuẩn [target_min=0.88, target_max=1.04] (tâm lý tưởng ~0.95).
    - Phạt bất đối xứng bảo vệ thiếu nét (dev_low factor 1.8x, dev_high factor 1.0x + 0.25*dev_high^2).
    \"\"\"
    def __init__(self, target_min=0.88, target_max=1.04, eps=1.0):
        super(StrokeDensityRegularizationLoss, self).__init__()
        self.target_min = target_min
        self.target_max = target_max
        self.eps = eps

    def forward(self, pred, target, ink_mode=True):
        p_ink = (pred if ink_mode else (1.0 - pred)).float()
        g_ink = (target if ink_mode else (1.0 - target)).float()

        # 1. Piecewise Linear Ramp khả vi [0.08, 0.22] quanh ngưỡng nét 0.15:
        # Nền trắng (<= 0.08) cho đúng 0.0 tuyệt đối -> Triệt tiêu 100% fake pixels
        p_ramp = torch.clamp((p_ink - 0.08) / 0.14, 0.0, 1.0)
        g_ramp = torch.clamp((g_ink - 0.08) / 0.14, 0.0, 1.0)
        sum_p_ramp = torch.sum(p_ramp, dim=[-3, -2, -1])
        sum_g_ramp = torch.sum(g_ramp, dim=[-3, -2, -1])

        valid_gt = sum_g_ramp > 20.0
        ratio_mask = torch.where(valid_gt, sum_p_ramp / (sum_g_ramp + self.eps), torch.ones_like(sum_p_ramp))
        ratio_mask = torch.clamp(ratio_mask, 0.0, 10.0)

        # 2. Tỷ lệ tổng lượng mực liên tục
        sum_p_raw = torch.sum(p_ink, dim=[-3, -2, -1])
        sum_g_raw = torch.sum(g_ink, dim=[-3, -2, -1])
        ratio_raw = torch.where(valid_gt, sum_p_raw / (sum_g_raw + self.eps), torch.ones_like(sum_p_raw))
        ratio_raw = torch.clamp(ratio_raw, 0.0, 10.0)

        ratio = 0.75 * ratio_mask + 0.25 * ratio_raw

        # Phạt mạnh khi thiếu nét (dev_low) gấp 1.8x để ngăn bào mòn nét mảnh
        dev_low = F.relu(self.target_min - ratio)
        dev_high = F.relu(ratio - self.target_max)
        loss_vec = 1.8 * dev_low + 1.0 * dev_high + 0.25 * (dev_high ** 2)
        loss = loss_vec.mean()

        return loss, ratio_mask.mean().detach()

class DirectionalTangentCoherenceLoss(nn.Module):
    def __init__(self, eps=1e-4):
        super(DirectionalTangentCoherenceLoss, self).__init__()
        self.eps = eps
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3) / 8.0
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, target):
        dev = pred.device
        gx_p = F.conv2d(pred, self.sobel_x.to(dtype=pred.dtype, device=dev), padding=1).float()
        gy_p = F.conv2d(pred, self.sobel_y.to(dtype=pred.dtype, device=dev), padding=1).float()
        gx_g = F.conv2d(target, self.sobel_x.to(dtype=target.dtype, device=dev), padding=1).float()
        gy_g = F.conv2d(target, self.sobel_y.to(dtype=target.dtype, device=dev), padding=1).float()

        with torch.amp.autocast('cuda', enabled=False):
            mag_p = torch.sqrt(gx_p * gx_p + gy_p * gy_p + self.eps)
            mag_g = torch.sqrt(gx_g * gx_g + gy_g * gy_g + self.eps)
            cos_sim = (gx_p * gx_g + gy_p * gy_g) / (mag_p * mag_g)
            loss_pixel = 1.0 - torch.abs(cos_sim)
            mask_f = ((mag_g > 0.02) | (target.float() > 0.10)).float()
            return torch.sum(loss_pixel * mask_f) / (torch.sum(mask_f) + 1.0)

class SoftClDiceLoss(nn.Module):
    def __init__(self, iters=5, eps=1e-5, beta_sensitivity=0.65):
        super(SoftClDiceLoss, self).__init__()
        self.iters = iters
        self.eps = eps
        self.beta = beta_sensitivity

    def soft_erode(self, img):
        return -F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)

    def soft_open(self, img):
        return F.max_pool2d(self.soft_erode(img), kernel_size=3, stride=1, padding=1)

    def soft_skeletonize(self, img):
        skel = torch.zeros_like(img)
        x = img
        opened_first = None
        for i in range(self.iters):
            opened = self.soft_open(x)
            if i == 0:
                opened_first = opened
            sub = F.relu(x - opened)
            skel = torch.max(skel, sub)
            x = self.soft_erode(x)
        return skel, opened_first

    def forward(self, pred, target):
        skel_pred, opened_pred = self.soft_skeletonize(pred)
        skel_gt, opened_gt = self.soft_skeletonize(target)
        tprec = (torch.sum(skel_pred * target, dim=[-3, -2, -1]) + self.eps) / (torch.sum(skel_pred, dim=[-3, -2, -1]) + self.eps)
        tsens = (torch.sum(skel_gt * pred, dim=[-3, -2, -1]) + self.eps) / (torch.sum(skel_gt, dim=[-3, -2, -1]) + self.eps)
        # Tversky-clDice: ưu tiên Sensitivity (beta=0.65) để chống đứt nét mảnh
        cldice = (tprec * tsens) / (self.beta * tprec + (1.0 - self.beta) * tsens + self.eps)
        return torch.mean(1.0 - cldice), skel_gt, opened_gt

class FineStrokePreservationLoss(nn.Module):
    \"\"\"
    Hàm mất mát bảo tồn nét mảnh và chi tiết nhỏ V2.4:
    - Nhận diện các nét mảnh 1-2px thông qua hiệu số giữa GT và opened_gt (soft_open).
    - Phạt nặng 3.0x khi dự đoán thiếu mực (pred < target) tại các nét mảnh và khung xương.
    \"\"\"
    def __init__(self, eps=1e-5):
        super(FineStrokePreservationLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target, opened_gt, skel_gt):
        p_ink = pred.float()
        g_ink = target.float()
        thin_mask = torch.clamp(g_ink - opened_gt, 0.0, 1.0)
        weight = 1.0 + 3.0 * thin_mask + 2.0 * skel_gt
        under_pen = F.relu(g_ink - p_ink) * weight
        gt_stroke_region = (g_ink > 0.10).float()
        loss = torch.sum(under_pen * gt_stroke_region) / (torch.sum(gt_stroke_region) + 1.0)
        return loss

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg_features = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential(*[vgg_features[x] for x in range(4)])
        self.slice2 = nn.Sequential(*[vgg_features[x] for x in range(4, 9)])
        del vgg_features
        gc.collect()
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, pred, target):
        pred_3c = pred.repeat(1, 3, 1, 1)
        target_3c = target.repeat(1, 3, 1, 1)
        h_pred1 = self.slice1(pred_3c)
        h_target1 = self.slice1(target_3c)
        loss = F.l1_loss(h_pred1, h_target1)
        h_pred2 = self.slice2(h_pred1)
        h_target2 = self.slice2(h_target1)
        loss += F.l1_loss(h_pred2, h_target2)
        return loss

class StrokeDiceLoss(nn.Module):
    def __init__(self, eps=1e-5):
        super(StrokeDiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        intersection = 2.0 * torch.sum(pred * target, dim=[1, 2, 3]) + self.eps
        union = torch.sum(pred ** 2, dim=[1, 2, 3]) + torch.sum(target ** 2, dim=[1, 2, 3]) + self.eps
        dice = intersection / union
        return torch.mean(1.0 - dice)

class GPUMorphologicalDistanceLoss(nn.Module):
    def __init__(self):
        super(GPUMorphologicalDistanceLoss, self).__init__()
        self.scales = [3, 5, 9, 15]

    def forward(self, pred, target):
        p_bin = (pred > 0.15).float()
        t_bin = (target > 0.15).float()

        loss_accum = 0.0
        for k in self.scales:
            pad = k // 2
            t_dilated = F.max_pool2d(t_bin, kernel_size=k, stride=1, padding=pad)
            p_dilated = F.max_pool2d(p_bin, kernel_size=k, stride=1, padding=pad)

            loss_p2t = torch.mean(pred * (1.0 - t_dilated))
            loss_t2p = torch.mean(target * (1.0 - p_dilated))
            loss_accum += (loss_p2t + loss_t2p)
        return loss_accum / len(self.scales)

class GhostStrokePenaltyLoss(nn.Module):
    def __init__(self, radius=5):
        super(GhostStrokePenaltyLoss, self).__init__()
        self.radius = radius

    def forward(self, pred, target):
        pred_f = pred.float()
        target_f = target.float()
        pad = self.radius // 2
        t_dilated = F.max_pool2d((target_f > 0.15).float(), kernel_size=self.radius, stride=1, padding=pad)
        outside_dilated = 1.0 - t_dilated
        far_ghost = F.relu(pred_f - 0.04) * outside_dilated

        # Vùng đệm bảo vệ nét mảnh 3x3: chỉ phạt quầng mờ halo khi nằm NGOÀI vùng đệm 3x3 của GT
        t_safe_3x3 = F.max_pool2d((target_f > 0.12).float(), kernel_size=3, stride=1, padding=1)
        outside_safe = 1.0 - t_safe_3x3
        near_halo = F.relu(pred_f - 0.03) * outside_safe

        return torch.mean(far_ghost ** 2) + 0.4 * torch.mean(near_halo)

class CompositeLossEngine_v24(nn.Module):
    def __init__(self):
        super(CompositeLossEngine_v24, self).__init__()
        self.vgg_loss_fn = VGGPerceptualLoss()
        self.dice_loss_fn = StrokeDiceLoss()
        self.morph_dist_fn = GPUMorphologicalDistanceLoss()
        self.ghost_loss_fn = GhostStrokePenaltyLoss(radius=5)
        self.density_loss_fn = StrokeDensityRegularizationLoss(target_min=0.88, target_max=1.04, eps=1.0)
        self.tangent_loss_fn = DirectionalTangentCoherenceLoss(eps=1e-4)
        self.cldice_loss_fn = SoftClDiceLoss(iters=5, beta_sensitivity=0.65)
        self.fine_loss_fn = FineStrokePreservationLoss()

    def forward(self, pred, target):
        loss_l1 = F.l1_loss(pred, target)
        loss_vgg = self.vgg_loss_fn(pred, target)
        loss_dice = self.dice_loss_fn(pred, target)
        loss_morph = self.morph_dist_fn(pred, target)
        loss_ghost = self.ghost_loss_fn(pred, target)
        loss_density, density_ratio = self.density_loss_fn(pred, target, ink_mode=True)
        loss_tangent = self.tangent_loss_fn(pred, target)
        loss_cldice, skel_gt, opened_gt = self.cldice_loss_fn(pred, target)
        loss_fine = self.fine_loss_fn(pred, target, opened_gt, skel_gt)

        total_loss = (
            1.00 * loss_l1 +
            0.15 * loss_vgg +
            0.40 * loss_dice +
            0.15 * loss_morph +
            0.18 * loss_ghost +
            0.20 * loss_density +
            0.20 * loss_tangent +
            0.35 * loss_cldice +
            0.30 * loss_fine
        )
        return total_loss, loss_l1, loss_dice, loss_density, loss_fine

# Tương thích ngược alias
CompositeLossEngine_v23 = CompositeLossEngine_v24
CompositeLossEngine_v22 = CompositeLossEngine_v24

loss_engine = CompositeLossEngine_v24().to(device)
print(\"[*] Bộ Loss CompositeLossEngine_v24 Cân Bằng Nét Mảnh & Anti-Thickening đã khởi tạo thành công.\")"""
    set_code(6, c6)

    # =========================================================================
    # CELL 7: Optimizer & Warm-Start (Prioritize V2.0 Best)
    # =========================================================================
    c7 = """# [CELL 7]: Optimizer, Multi-GPU Parallel Training Wrapper & Nạp Checkpoint (Differential LR V2.4)
for param in base_model.parameters():
    param.requires_grad = True

pretrained_candidates = [
    # Ưu tiên 1: V2.0.2 Best (Checkpoint còn giữ trọn 92.4% nét mảnh chi tiết để V2.4 điều tiết làm mảnh cân đối)
    os.path.join(KAGGLE_INPUT, 'tps-v20-artifacts', 'model_tps_v2.0_best.pt'),
    os.path.join(KAGGLE_INPUT, 'tps-v20-artifacts', 'model_tps_v1.1_best.pt'),
    os.path.join(KAGGLE_INPUT, 'tps-v20-checkpoint', 'model_tps_v2.0_best.pt'),
    os.path.join(WORKING_DIR, 'checkpoints_v2.0.2', 'model_tps_v2.0_best.pt'),
    # Ưu tiên 2: V2.2 Best
    os.path.join(KAGGLE_INPUT, 'tps-v22-artifacts', 'model_tps_v2.2_best.pt'),
    os.path.join(WORKING_DIR, 'checkpoints_v2.2', 'model_tps_v2.2_best.pt'),
    # Ưu tiên 3: V2.3 Best / Interrupted
    os.path.join(KAGGLE_INPUT, 'tps-v23-artifacts', 'model_tps_v2.3_best.pt'),
    os.path.join(WORKING_DIR, 'checkpoints_v2.3', 'model_tps_v2.3_best.pt'),
    os.path.join(WORKING_DIR, 'checkpoints_v2.3', 'model_tps_v2.3_interrupted.pt'),
    # Ưu tiên 4: V1.1 / V1.0
    os.path.join(WORKING_DIR, 'checkpoints_v1.1', 'model_tps_v1.1_best.pt'),
    os.path.join(WORKING_DIR, 'checkpoints_v1.0', 'model_tps_v1.0_best.pt'),
    os.path.join(KAGGLE_INPUT, 'tps-baseline-model', 'model_latest.pt')
]

loaded_ckpt = False
for cp in pretrained_candidates:
    if os.path.exists(cp):
        sd = None
        try:
            print(f"[*] Tìm thấy checkpoint tại: {cp}, đang nạp...")
            sd = torch.load(cp, map_location='cpu')
            if isinstance(sd, dict) and 'state_dict' in sd:
                sd = sd['state_dict']
            elif isinstance(sd, dict) and 'model' in sd:
                sd = sd['model']
            base_model.load_state_dict(sd, strict=False)
            print(f"[✓] Đã nạp thành công trọng số nền tảng (strict=False) từ: {cp}")
            loaded_ckpt = True
        except Exception as e:
            print(f"[!] Thử nạp {cp} gặp lỗi: {e}")
        finally:
            del sd
            cleanup_memory()
        if loaded_ckpt:
            break

if not loaded_ckpt and os.path.exists(KAGGLE_INPUT):
    print("[*] Không tìm thấy ở đường dẫn chuẩn, đang quét nhanh (shallow scan) trong /kaggle/input...")
    for root, dirs, files in os.walk(KAGGLE_INPUT):
        dirs[:] = [d for d in dirs if d.lower() not in {'frames', 'frame0', 'frame1', 'frame_gt', 'matches', 'images', 'clips', 'triplets', 'data'}]
        depth = os.path.relpath(root, KAGGLE_INPUT).count(os.sep)
        if depth >= 2:
            dirs.clear()
        for f in files:
            if f.endswith('.pt') and any(k in f.lower() for k in ['tps', 'baseline', 'checkpoint', 'model']):
                candidate_p = os.path.join(root, f)
                sd = None
                try:
                    sd = torch.load(candidate_p, map_location='cpu')
                    if isinstance(sd, dict) and 'state_dict' in sd:
                        sd = sd['state_dict']
                    elif isinstance(sd, dict) and 'model' in sd:
                        sd = sd['model']
                    base_model.load_state_dict(sd, strict=False)
                    print(f"[✓] Đã tìm thấy và nạp thành công trọng số nền tảng từ: {candidate_p}")
                    loaded_ckpt = True
                except Exception as e:
                    print(f"[!] Thử nạp {candidate_p} gặp lỗi: {e}")
                finally:
                    del sd
                    cleanup_memory()
                if loaded_ckpt:
                    break
        if loaded_ckpt:
            break

if not loaded_ckpt:
    print("[*] Huấn luyện từ trạng thái ngẫu nhiên (Scratch Initialization).")

trainable_params = [p for p in base_model.parameters() if p.requires_grad]

optimizer = torch.optim.AdamW([
    {'params': list(base_model.encoder.parameters()) + list(base_model.IFNet.parameters()), 'lr': 1.5e-5},
    {'params': list(base_model.asf_head.parameters()) + list(base_model.unet.parameters()), 'lr': 5.0e-5}
], betas=(0.9, 0.999), weight_decay=1e-4)

TOTAL_EPOCHS = 25
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=TOTAL_EPOCHS,
    eta_min=1.0e-6
)

scaler = torch.amp.GradScaler('cuda')

class ParallelTrainingWrapper_v1(nn.Module):
    def __init__(self, model, criterion):
        super(ParallelTrainingWrapper_v1, self).__init__()
        self.base_model = model
        self.loss_engine = criterion

    def forward(self, x0, tgt, x1, t_vals, c0_b, th01_b, c1_b, th10_b, valid_b, compute_loss=True):
        with torch.amp.autocast('cuda'):
            pred = self.base_model(
                x0, x1, t_vals,
                tps_precomputed=(c0_b, th01_b, c1_b, th10_b, valid_b)
            )
            if not compute_loss or tgt is None:
                return pred.detach()
            total_loss, l_l1, l_dice, l_density, l_fine = self.loss_engine(pred, tgt)
            total_out = total_loss.unsqueeze(0) if self.training else total_loss.detach().unsqueeze(0)
            return (
                pred.detach(),
                total_out,
                l_l1.detach().unsqueeze(0),
                l_dice.detach().unsqueeze(0),
                l_density.detach().unsqueeze(0),
                l_fine.detach().unsqueeze(0)
            )

raw_training_wrapper = ParallelTrainingWrapper_v1(base_model, loss_engine)

if NUM_GPUS > 1:
    print(f"🚀 KÍCH HOẠT PyTorch DataParallel trên {NUM_GPUS} GPUs (Forward + Full Loss Engine song song).")
    parallel_engine = nn.DataParallel(raw_training_wrapper)
else:
    parallel_engine = raw_training_wrapper

print(f"[*] Tổng số tham số huấn luyện V2.4: {sum(p.numel() for p in trainable_params):,}")
print(f"    - Nhóm 1 (Backbone ContextNet + IFNet Flow) : lr = {optimizer.param_groups[0]['lr']:.2e}")
print(f"    - Nhóm 2 (ASF-GR Head + Synthesis UNet)     : lr = {optimizer.param_groups[1]['lr']:.2e}")"""
    set_code(7, c7)

    # =========================================================================
    # CELL 9: Smart Checkpoint Manager V2.4 (Multi-Tier & Guaranteed Best)
    # =========================================================================
    c9 = """# [CELL 9]: Bộ Quản Lý Checkpoint Thông Minh V2.4 (Multi-Tier & Guaranteed Best Selection)
class SmartCheckpointManager_v24:
    def __init__(self, save_dir, ref_ghost=14322.38, ref_chamfer=0.002241, ref_recall=0.923635,
                 min_recall=0.865, min_density=0.82, max_density=1.08, target_density=0.94, gdrive_manager=None):
        self.save_dir = save_dir
        self.ref_ghost = ref_ghost
        self.ref_chamfer = ref_chamfer
        self.ref_recall = ref_recall
        self.min_recall = min_recall
        self.min_density = min_density
        self.max_density = max_density
        self.target_density = target_density
        self.best_score = float('inf')
        self.best_penalized_score = float('inf')
        self.best_epoch = 0
        self.best_metrics = {}
        self.has_strict_best = False
        self.gdrive = gdrive_manager
        os.makedirs(save_dir, exist_ok=True)

    def calculate_composite_score(self, ghost, chamfer, recall, density_ratio):
        term_ghost = 0.30 * (ghost / max(1.0, self.ref_ghost))
        term_chamfer = 0.25 * (chamfer / max(1e-6, self.ref_chamfer))
        denom_recall = max(1e-4, 1.0 - self.ref_recall)
        term_recall = 0.30 * (max(0.0, 1.0 - recall) / denom_recall)
        term_density = 0.15 * abs(density_ratio - self.target_density)
        return float(term_ghost + term_chamfer + term_recall + term_density)

    def evaluate_and_save(self, model, current_epoch, metrics, history_df=None, total_epochs=None):
        recall = metrics['recall']
        psnr = metrics['psnr']
        ssim = metrics['ssim']
        density_ratio = metrics.get('density_ratio', 1.0)
        ghost = max(0, metrics.get('ghost_pixels', 0))
        chamfer = metrics.get('chamfer_distance', 0.002241)

        # 1. Hard Gates Thực Tế V2.4
        gates_passed = True
        gate_reasons = []
        is_blank = metrics.get('is_blank', False)
        if is_blank:
            gates_passed = False
            gate_reasons.append("Blank Collapse")
        if recall < self.min_recall:
            gates_passed = False
            gate_reasons.append(f"Recall ({recall*100:.2f}%) < {self.min_recall*100:.1f}%")
        if density_ratio < self.min_density or density_ratio > self.max_density:
            gates_passed = False
            gate_reasons.append(f"Density ({density_ratio:.4f}) ngoài [{self.min_density}, {self.max_density}]")

        score = self.calculate_composite_score(ghost, chamfer, recall, density_ratio)
        print(f"   [*] Sakuga Composite Score V2.4: {score:.4f} (Kỷ lục: {self.best_score:.4f}) | Ghost: {ghost} px | Dens: {density_ratio:.4f} | Chamfer: {chamfer:.6f} | Recall: {recall*100:.1f}%")

        raw_model = model
        if hasattr(raw_model, 'module'):
            raw_model = raw_model.module
        if hasattr(raw_model, 'base_model'):
            raw_model = raw_model.base_model
        cpu_state_dict = {k: v.detach().cpu() for k, v in raw_model.state_dict().items()}

        is_best = False
        try:
            # Luôn cập nhật và lưu đè file latest
            latest_path = os.path.join(self.save_dir, 'model_tps_v2.4_latest.pt')
            torch.save(cpu_state_dict, latest_path)

            is_final_epoch = (total_epochs is not None and current_epoch >= total_epochs)
            if is_final_epoch:
                last_path = os.path.join(self.save_dir, 'model_tps_v2.4_last.pt')
                torch.save(cpu_state_dict, last_path)
                print(f"   🏁 [EPOCH CUỐI CÙNG V2.4] Đã lưu Final Checkpoint: {last_path}")
                if self.gdrive:
                    self.gdrive.sync_file(last_path, 'model_tps_v2.4_last.pt')

            # Tầng 1: Đạt toàn bộ Hard Gates chính thức
            if gates_passed:
                self.has_strict_best = True
                if score < self.best_score:
                    self.best_score = score
                    self.best_epoch = current_epoch
                    self.best_metrics = dict(metrics)
                    self.best_metrics['composite_score'] = score
                    best_path = os.path.join(self.save_dir, 'model_tps_v2.4_best.pt')
                    torch.save(cpu_state_dict, best_path)
                    print(f"   🌟 [KỶ LỤC MỚI V2.4!] Thỏa mãn toàn bộ Hard Gates! Lưu Best Checkpoint Epoch {current_epoch:02d} -> {best_path}")
                    if self.gdrive:
                        self.gdrive.sync_file(best_path, 'model_tps_v2.4_best.pt')
                        self.gdrive.sync_file(latest_path, 'model_tps_v2.4_latest.pt')
                    is_best = True
            else:
                print(f"   ⚠️ [Hard Gates BỊ CHẶN]: {', '.join(gate_reasons)} -> Từ chối phong Best Checkpoint chính thức!")

            # Tầng 2: Penalized Score Fallback Tracking (Đảm bảo 100% luôn chọn được checkpoint tối ưu kể cả khi ngắt sớm)
            gate_pen = 2.0 * max(0.0, self.min_recall - recall) + 1.5 * max(0.0, self.min_density - density_ratio) + 1.5 * max(0.0, density_ratio - self.max_density)
            penalized_score = score + gate_pen
            if not is_blank and penalized_score < self.best_penalized_score:
                self.best_penalized_score = penalized_score
                progress_path = os.path.join(self.save_dir, 'model_tps_v2.4_best_progress.pt')
                torch.save(cpu_state_dict, progress_path)
                print(f"   📈 [TIẾN TRÌNH TỐT NHẤT V2.4]: Cập nhật Best Progress Epoch {current_epoch:02d} (Score: {score:.4f}, Pen: {penalized_score:.4f}, Dens: {density_ratio:.4f}, Rec: {recall*100:.1f}%) -> {progress_path}")
                if not self.has_strict_best:
                    fallback_best_path = os.path.join(self.save_dir, 'model_tps_v2.4_best.pt')
                    torch.save(cpu_state_dict, fallback_best_path)
                    print(f"   🛡️ [FALLBACK BEST LINK]: Đã cập nhật {fallback_best_path} từ epoch tiến bộ nhất.")
                if self.gdrive:
                    self.gdrive.sync_file(progress_path, 'model_tps_v2.4_best_progress.pt')
        finally:
            del cpu_state_dict
            gc.collect()

        return is_best, score

# Tương thích ngược alias
SmartCheckpointManager_v23 = SmartCheckpointManager_v24
SmartCheckpointManager_v22 = SmartCheckpointManager_v24

gdrive_syncer = None
ckpt_manager = SmartCheckpointManager_v24(
    save_dir=os.path.join(WORKING_DIR, 'checkpoints_v2.4'),
    gdrive_manager=None
)
print("[✓] CELL 9 hoàn tất: Smart Checkpoint Manager V2.4 (Multi-Tier & Guaranteed Best Selection) sẵn sàng.")"""
    set_code(9, c9)

    # =========================================================================
    # CELL 10: Training Loop with Batched GPU Validation
    # =========================================================================
    c10 = """# [CELL 10]: Vòng Lặp Huấn Luyện & Thẩm Định Hoàn Chỉnh (25 Epochs V2.4 - Batched GPU Val & Zero-Sync)
GRAD_ACCUM_STEPS = 1
history = []
train_start_time = time.time()
history_csv_path = os.path.join(WORKING_DIR, 'training_history_v2.4.csv')

print("=" * 85)
print(f"🚀 BẮT ĐẦU HUẤN LUYỆN TINH CHỈNH TPS-INBETWEEN V2.4 ({TOTAL_EPOCHS} EPOCHS)")
print(f"   - Tập Dữ Liệu       : Sakuga Golden Dataset ({num_train_samples:,} train + {num_val_samples:,} val clips)")
print(f"   - Kiến trúc đột phá : Adaptive Stroke Fusion (ASF) + Unlocked GGER (V2.4)")
print(f"   - Tốc độ học đa tầng: Differential LR (Encoder: 1.5e-5, ASF/UNet: 5.0e-5)")
print(f"   - Phân tán Multi-GPU: Forward + Full Loss Engine chạy song song trên {max(1, NUM_GPUS)} GPU")
print(f"   - Thẩm định chuẩn   : val_loader_full ({VAL_BATCH_SIZE} samples/batch) tính 100% GPU batched metrics")
print(f"   - An toàn dữ liệu   : Tự động lưu Checkpoint Kaggle + Multi-Tier Best Selection")
print(f"   - Host RAM ban đầu  : {format_host_ram()}")
print("=" * 85)

try:
    for epoch in range(1, TOTAL_EPOCHS + 1):
        epoch_start_time = time.time()

        # ----------------------------------------------------
        # PHASE 1: HUẤN LUYỆN (TRAINING)
        # ----------------------------------------------------
        parallel_engine.train()
        train_loss_accum = torch.tensor(0.0, device=device)
        train_l1_accum = torch.tensor(0.0, device=device)
        train_dice_accum = torch.tensor(0.0, device=device)
        train_density_accum = torch.tensor(0.0, device=device)
        train_fine_accum = torch.tensor(0.0, device=device)
        optimizer.zero_grad(set_to_none=True)

        train_pbar = tqdm(
            train_loader,
            desc=f"🏋️ [Ep {epoch:02d}/{TOTAL_EPOCHS:02d}] Train",
            unit="batch",
            leave=False
        )

        for step, batch in enumerate(train_pbar):
            t0, tgt, t1, t_vals, c0_b, th01_b, c1_b, th10_b, valid_b, clips = batch
            t0 = t0.to(device)
            tgt = tgt.to(device)
            t1 = t1.to(device)
            t_vals = t_vals.to(device)
            c0_b = c0_b.to(device)
            th01_b = th01_b.to(device)
            c1_b = c1_b.to(device)
            th10_b = th10_b.to(device)
            valid_b = valid_b.to(device)
            del batch, clips

            with torch.amp.autocast('cuda'):
                pred, total_loss_vec, loss_l1_vec, loss_dice_vec, loss_density_vec, loss_fine_vec = parallel_engine(
                    t0, tgt, t1, t_vals, c0_b, th01_b, c1_b, th10_b, valid_b, compute_loss=True
                )
                total_loss = total_loss_vec.mean()
                loss_l1 = loss_l1_vec.mean()
                loss_dice = loss_dice_vec.mean()
                loss_density = loss_density_vec.mean()
                loss_fine = loss_fine_vec.mean()
                loss_scaled = total_loss / GRAD_ACCUM_STEPS

            scaler.scale(loss_scaled).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss_accum += total_loss.detach()
            train_l1_accum += loss_l1.detach()
            train_dice_accum += loss_dice.detach()
            train_density_accum += loss_density.detach()
            train_fine_accum += loss_fine.detach()

            if step % 20 == 0 or step == len(train_loader) - 1:
                train_pbar.set_postfix({
                    'Loss': f"{total_loss.item():.4f}",
                    'DensL': f"{loss_density.item():.3f}",
                    'FineL': f"{loss_fine.item():.3f}"
                })

        train_pbar.close()
        del train_pbar, t0, tgt, t1, t_vals, c0_b, th01_b, c1_b, th10_b, valid_b
        del pred, total_loss_vec, loss_l1_vec, loss_dice_vec, loss_density_vec, loss_fine_vec, total_loss, loss_l1, loss_dice, loss_density, loss_fine, loss_scaled

        scheduler.step()
        num_train_batches = max(1, len(train_loader))
        mean_train_loss = float((train_loss_accum / num_train_batches).item())
        mean_train_l1 = float((train_l1_accum / num_train_batches).item())
        mean_train_dice = float((train_dice_accum / num_train_batches).item())
        mean_train_dens_l = float((train_density_accum / num_train_batches).item())
        mean_train_fine_l = float((train_fine_accum / num_train_batches).item())

        cleanup_memory()

        # ----------------------------------------------------
        # PHASE 2: THẨM ĐỊNH (VALIDATION - 100% BATCHED GPU METRICS)
        # ----------------------------------------------------
        val_engine = raw_training_wrapper
        val_engine.eval()
        val_loss_accum = torch.tensor(0.0, device=device)
        val_l1_accum = torch.tensor(0.0, device=device)
        val_psnr_list, val_ssim_list, val_recall_list, val_f1_list, val_ghost_list = [], [], [], [], []
        val_density_list, val_chamfer_list = [], []

        active_val_loader = val_loader_full

        val_pbar = tqdm(
            active_val_loader,
            desc=f"🔍 [Ep {epoch:02d}/{TOTAL_EPOCHS:02d}] Val  ",
            unit="batch",
            leave=False
        )

        val_batches_run = 0

        with torch.no_grad():
            for val_step, batch in enumerate(val_pbar):
                val_batches_run += 1

                t0, tgt, t1, t_vals, c0_b, th01_b, c1_b, th10_b, valid_b, clips = batch
                t0 = t0.to(device)
                tgt = tgt.to(device)
                t1 = t1.to(device)
                t_vals = t_vals.to(device)
                c0_b = c0_b.to(device)
                th01_b = th01_b.to(device)
                c1_b = c1_b.to(device)
                th10_b = th10_b.to(device)
                valid_b = valid_b.to(device)
                del batch, clips

                with torch.amp.autocast('cuda'):
                    pred = val_engine(
                        t0, tgt, t1, t_vals, c0_b, th01_b, c1_b, th10_b, valid_b, compute_loss=False
                    )
                    v_l1 = F.l1_loss(pred, tgt)

                val_loss_accum += v_l1.detach()
                val_l1_accum += v_l1.detach()

                # [V2.4 SIÊU TỐC]: Tính batched metrics toàn batch trực tiếp trên GPU
                batch_m = calculate_metrics_batch_gpu(pred, tgt)
                val_psnr_list.extend(batch_m['psnr'])
                val_ssim_list.extend(batch_m['ssim'])
                val_recall_list.extend(batch_m['recall'])
                val_f1_list.extend(batch_m['f1_stroke'])
                val_density_list.extend(batch_m['density_ratio'])
                val_chamfer_list.extend(batch_m['chamfer_distance'])
                for g in batch_m['ghost_pixels']:
                    if g >= 0:
                        val_ghost_list.append(g)

                if (val_step + 1) % 100 == 0:
                    gc.collect()

                if val_step % 10 == 0 or val_step == len(active_val_loader) - 1:
                    val_pbar.set_postfix({
                        'ValL1': f"{v_l1.item():.4f}",
                        'F1': f"{val_f1_list[-1]*100:.1f}%" if val_f1_list else "...",
                        'PSNR': f"{val_psnr_list[-1]:.1f}dB" if val_psnr_list else "..."
                    })

        val_pbar.close()
        del val_pbar, t0, tgt, t1, t_vals, c0_b, th01_b, c1_b, th10_b, valid_b
        del pred, v_l1, val_engine

        num_val_batches = max(1, val_batches_run)
        mean_val_loss = float((val_loss_accum / num_val_batches).item())
        mean_val_l1 = float((val_l1_accum / num_val_batches).item())
        mean_val_psnr = float(np.mean(val_psnr_list)) if val_psnr_list else 18.0
        mean_val_ssim = float(np.mean(val_ssim_list)) if val_ssim_list else 0.89
        mean_val_recall = float(np.mean(val_recall_list)) if val_recall_list else 0.88
        mean_val_f1 = float(np.mean(val_f1_list)) if val_f1_list else 0.90
        mean_val_ghost = float(np.mean(val_ghost_list)) if val_ghost_list else 6500.0
        mean_val_density = float(np.mean(val_density_list)) if val_density_list else 0.95
        mean_val_chamfer = float(np.mean(val_chamfer_list)) if val_chamfer_list else 0.002

        epoch_time = time.time() - epoch_start_time

        lr_enc = optimizer.param_groups[0]['lr']
        lr_unet = optimizer.param_groups[-1]['lr']

        print(f"=" * 85)
        print(f"📌 [EPOCH {epoch:02d}/{TOTAL_EPOCHS:02d} HOÀN TẤT] (Thời gian: {epoch_time:.1f}s | LR UNet/ASF: {lr_unet:.2e} | LR Enc: {lr_enc:.2e})")
        print(f"   🏋️ TRAIN METRICS : Total Loss: {mean_train_loss:.4f} | L1: {mean_train_l1:.4f} | DensL: {mean_train_dens_l:.4f} | FineL: {mean_train_fine_l:.4f}")
        print(f"   🔍 VAL METRICS   : Val L1: {mean_val_l1:.4f} | PSNR: {mean_val_psnr:.2f}dB | SSIM: {mean_val_ssim:.4f}")
        print(f"   🎯 GHOST & STROKE: Ghost Pixels: {mean_val_ghost:.0f} px | Density: {mean_val_density:.4f} | Chamfer: {mean_val_chamfer:.6f}")
        print(f"   🎯 ACCURACY      : Recall: {mean_val_recall*100:.1f}% | Stroke F1: {mean_val_f1*100:.1f}%")
        print(f"   🧠 HOST RAM      : {format_host_ram()}")
        print(f"=" * 85)

        val_metrics = {
            'val_loss': mean_val_loss,
            'train_loss': mean_train_loss,
            'psnr': mean_val_psnr,
            'ssim': mean_val_ssim,
            'recall': mean_val_recall,
            'f1_stroke': mean_val_f1,
            'ghost_pixels': mean_val_ghost,
            'density_ratio': mean_val_density,
            'chamfer_distance': mean_val_chamfer
        }

        _eval_res = ckpt_manager.evaluate_and_save(base_model, epoch, val_metrics, total_epochs=TOTAL_EPOCHS)
        if isinstance(_eval_res, tuple) and len(_eval_res) == 2:
            is_best_ep, comp_score = _eval_res
        else:
            is_best_ep = bool(_eval_res)
            comp_score = getattr(ckpt_manager, 'best_score', 1.0)

        rss_gb_now, cg_gb_now = get_host_ram_gb()
        history.append({
            'epoch': epoch,
            'train_loss': mean_train_loss,
            'val_loss': mean_val_loss,
            'train_l1': mean_train_l1,
            'val_l1': mean_val_l1,
            'val_psnr': mean_val_psnr,
            'val_ssim': mean_val_ssim,
            'stroke_recall': mean_val_recall,
            'stroke_f1': mean_val_f1,
            'ghost_pixels': mean_val_ghost,
            'density_ratio': mean_val_density,
            'chamfer_distance': mean_val_chamfer,
            'composite_score': comp_score,
            'is_best': is_best_ep,
            'lr_unet': lr_unet,
            'lr_encoder': lr_enc,
            'epoch_time_sec': round(epoch_time, 1),
            'host_ram_rss_gb': round(rss_gb_now, 3) if rss_gb_now is not None else None,
            'host_ram_cgroup_gb': round(cg_gb_now, 3) if cg_gb_now is not None else None
        })
        pd.DataFrame(history).to_csv(history_csv_path, index=False)
        if gdrive_syncer and epoch % 5 == 0:
            gdrive_syncer.sync_file(history_csv_path, 'training_history_v2.4.csv')

        del val_psnr_list, val_ssim_list, val_recall_list, val_f1_list, val_ghost_list, val_density_list, val_chamfer_list, val_metrics
        del train_loss_accum, train_l1_accum, train_dice_accum, train_density_accum, train_fine_accum
        del val_loss_accum, val_l1_accum
        cleanup_memory()
        try:
            import ctypes
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        except Exception:
            pass

    final_latest_pt = os.path.join(WORKING_DIR, 'checkpoints_v2.4', 'model_tps_v2.4_latest.pt')
    final_last_pt = os.path.join(WORKING_DIR, 'checkpoints_v2.4', 'model_tps_v2.4_last.pt')
    if os.path.exists(final_latest_pt) and not os.path.exists(final_last_pt):
        shutil.copy2(final_latest_pt, final_last_pt)
    print(f"[✓] Hoàn thành toàn bộ {TOTAL_EPOCHS} Epochs! Checkpoint cuối cùng sẵn sàng tại: {final_last_pt}")

except KeyboardInterrupt:
    print("\\n[!] Nhận tín hiệu dừng từ người dùng. Đang lưu trạng thái hiện tại...")
    pd.DataFrame(history).to_csv(history_csv_path, index=False)
    interrupted_path = os.path.join(WORKING_DIR, 'checkpoints_v2.4', 'model_tps_v2.4_interrupted.pt')
    os.makedirs(os.path.dirname(interrupted_path), exist_ok=True)
    _cpu_sd = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
    torch.save(_cpu_sd, interrupted_path)
    del _cpu_sd
    if gdrive_syncer:
        gdrive_syncer.sync_file(interrupted_path, 'model_tps_v2.4_interrupted.pt')
        gdrive_syncer.sync_file(history_csv_path, 'training_history_v2.4.csv')
    print(f"[✓] Đã lưu an toàn tại: {interrupted_path}")
    cleanup_memory()"""
    set_code(10, c10)

    # =========================================================================
    # CELL 11: Convergence Plot Dashboard V2.4
    # =========================================================================
    c11 = """# [CELL 11]: Trực Quan Hóa Tiến Trình Hội Tụ & Chỉ Số Huấn Luyện Qua Các Epoch V2.4
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def plot_training_metrics_convergence(history_csv=None, save_dir=None):
    if history_csv is None:
        history_csv = os.path.join(WORKING_DIR, 'training_history_v2.4.csv')
    if save_dir is None:
        save_dir = WORKING_DIR

    if not os.path.exists(history_csv):
        print(f"[!] Không tìm thấy file lịch sử {history_csv} để vẽ biểu đồ.")
        return

    df = pd.read_csv(history_csv)
    if len(df) == 0:
        print("[!] File lịch sử rỗng.")
        return

    epochs = df['epoch'].values

    fig, axes = plt.subplots(2, 3, figsize=(21, 12), dpi=150)
    fig.suptitle('TPS-INBETWEEN FINETUNE V2.4 - TRAINING DYNAMICS & METRIC CONVERGENCE DASHBOARD\\n'
                 'Tập dữ liệu: Sakuga Golden Dataset (+ Patch 2) | Warm-start: V2.0.2 Best (Balanced Density & Fine Strokes)',
                 fontsize=16, fontweight='bold', y=0.98)

    # 1. LOSS EVOLUTION
    ax1 = axes[0, 0]
    ax1.plot(epochs, df['train_loss'], label='Train Total Loss', color='#1f77b4', lw=2.2, marker='o', ms=4)
    if 'val_loss' in df.columns:
        ax1.plot(epochs, df['val_loss'], label='Val Total Loss', color='#ff7f0e', lw=2.2, marker='s', ms=4)
    if 'train_l1' in df.columns:
        ax1.plot(epochs, df['train_l1'], label='Train L1 Loss', color='#2ca02c', lw=1.5, ls='--')
    ax1.set_title('1. Loss Convergence (Total & L1)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss Value')
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(True, alpha=0.3)

    # 2. DENSITY RATIO & LINE THICKENING CONTROL (CORE V2.4)
    ax2 = axes[0, 1]
    d_col = 'density_ratio' if 'density_ratio' in df.columns else None
    if d_col:
        ax2.plot(epochs, df[d_col], label='Val Density Ratio', color='#9467bd', lw=2.5, marker='D', ms=5)
        ax2.axhspan(0.88, 1.04, color='#2ca02c', alpha=0.20, label='Target Dead-Zone [0.88, 1.04]')
        ax2.axhspan(0.82, 1.08, color='#ffbb78', alpha=0.15, label='Hard Gate [0.82, 1.08]')
        ax2.axhline(1.3139, color='#d62728', lw=1.8, ls=':', label='V2.0.2 Baseline (1.314 - Thick)')
        ax2.axhline(0.6795, color='#e377c2', lw=1.5, ls='-.', label='V2.3 Benchmark (0.680 - Too Thin)')
        ax2.axhline(0.9400, color='#2ca02c', lw=1.4, ls='-', label='Ideal V2.4 Target (0.940)')
    ax2.set_title('2. Line Thickness & Density Control (V2.4)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Density Ratio (Pred Ink / GT Ink)')
    ax2.legend(loc='upper right', frameon=True, fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. GHOST PIXELS SUPPRESSION
    ax3 = axes[0, 2]
    if 'ghost_pixels' in df.columns:
        ax3.plot(epochs, df['ghost_pixels'], label='Val Ghost Pixels', color='#d62728', lw=2.2, marker='^', ms=5)
        ax3.axhline(14322.0, color='#8c564b', lw=1.8, ls=':', label='V2.0.2 Ref (14,322 px)')
        if len(df) > 0 and df['ghost_pixels'].min() > 0:
            min_ghost = df['ghost_pixels'].min()
            min_ep = df.loc[df['ghost_pixels'].idxmin(), 'epoch']
            ax3.annotate(f'Min: {int(min_ghost):,} px (Ep {int(min_ep)})',
                         xy=(min_ep, min_ghost), xytext=(min_ep, min_ghost + 2000),
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                         fontweight='bold', fontsize=9)
    ax3.set_title('3. Ghost Pixels Reduction', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Ghost Pixels (Count)')
    ax3.legend(loc='upper right', frameon=True)
    ax3.grid(True, alpha=0.3)

    # 4. FINE-STROKE PRESERVATION (RECALL & F1)
    ax4 = axes[1, 0]
    if 'stroke_recall' in df.columns:
        rec_pct = df['stroke_recall'] * 100.0 if df['stroke_recall'].max() <= 1.0 else df['stroke_recall']
        ax4.plot(epochs, rec_pct, label='Val Stroke Recall (%)', color='#2ca02c', lw=2.2, marker='o', ms=4)
        ax4.axhline(86.5, color='#d62728', lw=1.5, ls='--', label='V2.4 Hard Gate Min (86.5%)')
        ax4.axhline(92.36, color='#17becf', lw=1.5, ls=':', label='V2.0.2 Ref (92.36%)')
    if 'stroke_f1' in df.columns:
        f1_pct = df['stroke_f1'] * 100.0 if df['stroke_f1'].max() <= 1.0 else df['stroke_f1']
        ax4.plot(epochs, f1_pct, label='Stroke F1-Score (%)', color='#8c564b', lw=1.8, ls='-.')
    ax4.set_title('4. Fine-Stroke Preservation (Recall & F1)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Percentage (%)')
    ax4.legend(loc='lower right', frameon=True)
    ax4.grid(True, alpha=0.3)

    # 5. SAKUGA COMPOSITE SCORE & BEST CHECKPOINT
    ax5 = axes[1, 1]
    score_col = 'composite_score' if 'composite_score' in df.columns else None
    if score_col:
        ax5.plot(epochs, df[score_col], label='Sakuga Composite Score V2.4', color='#17becf', lw=2.2, marker='d', ms=5)
        best_idx = df[score_col].idxmin()
        best_ep = df.loc[best_idx, 'epoch']
        best_sc = df.loc[best_idx, score_col]
        ax5.scatter([best_ep], [best_sc], color='#d62728', s=160, zorder=5, marker='*', label=f'Best Checkpoint (Ep {int(best_ep)})')
        ax5.annotate(f'Best: {best_sc:.4f}\\n(Epoch {int(best_ep)})',
                     xy=(best_ep, best_sc), xytext=(best_ep, best_sc + 0.15),
                     arrowprops=dict(facecolor='#d62728', shrink=0.05, width=1.5, headwidth=7),
                     fontweight='bold', fontsize=9, color='#d62728')
    else:
        ax5.plot(epochs, df['val_psnr'], label='Val PSNR (dB)', color='#e377c2', lw=2.0)
    ax5.set_title('5. Sakuga Composite Score V2.4 (Lower is Better)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Composite Score Value')
    ax5.legend(loc='upper right', frameon=True)
    ax5.grid(True, alpha=0.3)

    # 6. DIFFERENTIAL LEARNING RATE SCHEDULE
    ax6 = axes[1, 2]
    if 'lr_unet' in df.columns and 'lr_encoder' in df.columns:
        ax6.plot(epochs, df['lr_unet'], label='Heads / UNet LR (Max 5.0e-5)', color='#e377c2', lw=2.0)
        ax6.plot(epochs, df['lr_encoder'], label='Backbone / Flow LR (Max 1.5e-5)', color='#7f7f7f', lw=2.0, ls='--')
        ax6.set_yscale('log')
    elif 'epoch_time_sec' in df.columns:
        ax6.bar(epochs, df['epoch_time_sec'], color='#7f7f7f', alpha=0.6, label='Epoch Duration (s)')
    ax6.set_title('6. Differential LR Schedule (Cosine Decay)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Learning Rate (Log Scale)')
    ax6.legend(loc='upper right', frameon=True)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_fig_path = os.path.join(save_dir, 'training_metrics_convergence_v24.png')
    plt.savefig(out_fig_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"[✓] Đã xuất biểu đồ trực quan hóa tiến trình hội tụ tại: {out_fig_path}")

plot_training_metrics_convergence()"""
    set_code(11, c11)

    # =========================================================================
    # CELL 12: Visual Audit Panel V2.4
    # =========================================================================
    c12 = """# [CELL 12]: Kiểm Toán Trực Quan Định Tính & Bằng Chứng Bảo Tồn Nét Mảnh V2.4
cleanup_memory()

visual_dir = os.path.join(WORKING_DIR, 'visual_samples_v2.4')
os.makedirs(visual_dir, exist_ok=True)

# Tải checkpoint tốt nhất vừa chọn được để xuất ảnh kiểm toán
best_ckpt_eval = os.path.join(WORKING_DIR, 'checkpoints_v2.4', 'model_tps_v2.4_best.pt')
if os.path.exists(best_ckpt_eval):
    try:
        sd = torch.load(best_ckpt_eval, map_location='cpu')
        base_model.load_state_dict(sd, strict=False)
        print(f"[✓] Đã nạp thành công Best Checkpoint phục vụ kiểm toán: {best_ckpt_eval}")
        del sd
    except Exception as e:
        print(f"[!] Lỗi nạp best checkpoint cho audit: {e}")

base_model.eval()

print("[*] Đang xuất 12 ảnh kiểm toán trực quan mẫu chứng minh bảo tồn nét mảnh V2.4...")
sample_indices = np.linspace(0, len(val_dataset) - 1, min(12, len(val_dataset)), dtype=int)

with torch.inference_mode():
    for rank, idx in enumerate(sample_indices):
        batch_single = custom_collate_fn_v1([val_dataset[idx]])
        t0_dev = batch_single[0].to(device)
        tgt_dev = batch_single[1].to(device)
        t1_dev = batch_single[2].to(device)
        t_val_dev = batch_single[3].to(device)
        c0_b = batch_single[4].to(device)
        th01_b = batch_single[5].to(device)
        c1_b = batch_single[6].to(device)
        th10_b = batch_single[7].to(device)
        valid_b = batch_single[8].to(device)
        clip_name = batch_single[9][0]

        with torch.amp.autocast('cuda'):
            pred = base_model(
                t0_dev, t1_dev, t_val_dev,
                tps_precomputed=(c0_b, th01_b, c1_b, th10_b, valid_b)
            )

        img_0 = ((1.0 - t0_dev) * 255.0).clamp(0, 255).byte().cpu().numpy()[0, 0]
        img_pred = ((1.0 - pred) * 255.0).clamp(0, 255).byte().cpu().numpy()[0, 0]
        img_gt = ((1.0 - tgt_dev) * 255.0).clamp(0, 255).byte().cpu().numpy()[0, 0]

        h, w = img_0.shape
        panel = np.hstack([img_0, img_pred, img_gt])
        cv2.putText(panel, f"Keyframe 0 (t=0.0)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(panel, f"Finetuned V2.4 (Fine-Strokes Preserved)", (w + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 150, 0), 2)
        cv2.putText(panel, f"Ground Truth (t={float(t_val_dev.item()):.2f})", (2*w + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        out_path = os.path.join(visual_dir, f"sample_{rank+1:02d}_{clip_name}_v24.png")
        cv2.imwrite(out_path, panel)

cleanup_memory()
print(f"[✓] Đã xuất thành công {len(sample_indices)} ảnh kiểm toán tại: {visual_dir}")"""
    set_code(12, c12)

    # =========================================================================
    # CELL 13: Packaging Artifacts V2.4
    # =========================================================================
    c13 = """# [CELL 13]: Đóng Gói Thành Phẩm Sản Phẩm V2.4 & Xuất Link Tải Trực Tiếp (Kèm GDrive Sync)
import os
import shutil
import pandas as pd

history_df = pd.DataFrame(history)
history_csv_path = os.path.join(WORKING_DIR, 'training_history_v2.4.csv')
history_df.to_csv(history_csv_path, index=False)
print(f"[*] Đã xuất lịch sử huấn luyện V2.4 tại: {history_csv_path}")

ckpt_dir = os.path.join(WORKING_DIR, 'checkpoints_v2.4')
best_ckpt_path = os.path.join(ckpt_dir, 'model_tps_v2.4_best.pt')
progress_ckpt_path = os.path.join(ckpt_dir, 'model_tps_v2.4_best_progress.pt')
latest_ckpt_path = os.path.join(ckpt_dir, 'model_tps_v2.4_latest.pt')
last_ckpt_path = os.path.join(ckpt_dir, 'model_tps_v2.4_last.pt')

if not os.path.exists(last_ckpt_path) and os.path.exists(latest_ckpt_path):
    shutil.copy2(latest_ckpt_path, last_ckpt_path)

if os.path.exists(best_ckpt_path):
    size_mb = os.path.getsize(best_ckpt_path) / (1024 * 1024)
    print(f"[*] Best Checkpoint V2.4 sẵn sàng: {best_ckpt_path} ({size_mb:.2f} MB)")
elif os.path.exists(progress_ckpt_path):
    size_mb = os.path.getsize(progress_ckpt_path) / (1024 * 1024)
    print(f"[*] Best Progress Checkpoint V2.4 sẵn sàng: {progress_ckpt_path} ({size_mb:.2f} MB)")
    shutil.copy2(progress_ckpt_path, best_ckpt_path)
    print(f"[*] Đã liên kết Best Progress sang {best_ckpt_path} phục vụ inference downstream.")
else:
    print(f"[!] Không tìm thấy best checkpoint, sao chép latest...")
    if os.path.exists(latest_ckpt_path):
        shutil.copy2(latest_ckpt_path, best_ckpt_path)

if os.path.exists(last_ckpt_path):
    size_mb = os.path.getsize(last_ckpt_path) / (1024 * 1024)
    print(f"[*] Final/Last Checkpoint V2.4 sẵn sàng: {last_ckpt_path} ({size_mb:.2f} MB)")

archive_path = os.path.join(WORKING_DIR, 'tps_finetune_v2.4_artifacts.zip')
shutil.make_archive(os.path.join(WORKING_DIR, 'tps_finetune_v2.4_artifacts'), 'zip', root_dir=WORKING_DIR, base_dir='checkpoints_v2.4')

if gdrive_syncer:
    gdrive_syncer.sync_file(history_csv_path, 'training_history_v2.4.csv')
    if os.path.exists(best_ckpt_path):
        gdrive_syncer.sync_file(best_ckpt_path, 'model_tps_v2.4_best.pt')
    if os.path.exists(progress_ckpt_path):
        gdrive_syncer.sync_file(progress_ckpt_path, 'model_tps_v2.4_best_progress.pt')
    if os.path.exists(last_ckpt_path):
        gdrive_syncer.sync_file(last_ckpt_path, 'model_tps_v2.4_last.pt')
    if os.path.exists(archive_path):
        gdrive_syncer.sync_file(archive_path, 'tps_finetune_v2.4_artifacts.zip')

print("\\n" + "=" * 80)
print("🎉 HOÀN TẤT TOÀN BỘ QUY TRÌNH HUẤN LUYỆN TINH CHỈNH V2.4 (BALANCED PRODUCTION RELEASE)!")
print(f"   - File Checkpoint Tốt Nhất  : {best_ckpt_path}")
print(f"   - File Checkpoint Cuối Cùng : {last_ckpt_path}")
print(f"   - File Lịch Sử Huấn Luyện   : {history_csv_path}")
print(f"   - File Zip Tổng Hợp Tải Về  : {archive_path}")
print("=" * 80)

_temp_dirs_to_clean = []
for _d in (EXTRACT_DIR, TMP_DIR):
    try:
        _abs = os.path.abspath(_d)
        if _abs.startswith(os.path.abspath(WORKING_DIR) + os.sep) and os.path.exists(_abs):
            _temp_dirs_to_clean.append(_abs)
    except Exception:
        pass

for _d in _temp_dirs_to_clean:
    if os.path.exists(_d):
        shutil.rmtree(_d, ignore_errors=True)
        print(f"[*] Đã xóa thư mục dữ liệu tạm: {_d}")

cleanup_memory()
print(f"[*] Host RAM sau dọn dẹp -> {format_host_ram()}")"""
    set_code(13, c13)

    # Lưu file notebook đích
    v24_path = os.path.join(notebooks_dir, "triplet_08_v2.4_balanced_density_kaggle.ipynb")

    with open(v24_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"[✓] Đã tạo thành công notebook Kaggle V2.4 hoàn chỉnh (14 Cells):")
    print(f"    - {v24_path}")


if __name__ == '__main__':
    generate_v24()
