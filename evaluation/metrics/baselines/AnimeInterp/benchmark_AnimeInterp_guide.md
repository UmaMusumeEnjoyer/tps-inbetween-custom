# 📖 Hướng Dẫn Benchmark AnimeInterp trên Kaggle GPU

> **Model:** AnimeInterp (CVPR 2021) — *Deep Animation Video Interpolation in the Wild*  
> **Paper:** [arxiv.org/abs/2104.02495](https://arxiv.org/abs/2104.02495)  
> **Môi trường:** Kaggle Notebook · GPU T4 x2 hoặc P100 · Python 3.10+ · PyTorch ≥ 1.7  
> **Tác giả pipeline benchmark:** Hệ thống PBL6

---

## 📋 Mục Lục

1. [Tổng Quan Model](#1-tổng-quan-model)
2. [Bảng 17 Chỉ Số Áp Dụng](#2-bảng-17-chỉ-số-áp-dụng)
3. [Cấu Trúc File Output](#3-cấu-trúc-file-output)
4. [Bước 0: Chuẩn Bị Trước Khi Upload Kaggle](#4-bước-0-chuẩn-bị-trước-khi-upload-kaggle)
5. [Cell 1 – Thiết Lập Môi Trường](#5-cell-1--thiết-lập-môi-trường)
6. [Cell 2 – Nạp Model & Checkpoints](#6-cell-2--nạp-model--checkpoints)
7. [Cell 3 – Dataloader Triplet ATD-12K](#7-cell-3--dataloader-triplet-atd-12k)
8. [Cell 4 – Module Tính Các Metrics](#8-cell-4--module-tính-các-metrics)
9. [Cell 5 – Pipeline Benchmark & Xuất CSV](#9-cell-5--pipeline-benchmark--xuất-csv)
10. [Kết Quả Mong Đợi](#10-kết-quả-mong-đợi)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Tổng Quan Model

**AnimeInterp** là mô hình frame interpolation chuyên biệt cho **anime/cartoon video** (CVPR 2021). Kiến trúc gồm 3 giai đoạn:

```
frame1.png (t=0) ─┐
                   ├─► [Segment-Guided Matching (SGM)] ─► F12_init, F21_init
frame3.png (t=1) ─┘                                              │
                                                                  ▼
                   ┌─────────────── [RFR: Recurrent Flow Refinement] ──────────┐
                   │  (Transformer-like, RAFT-based, iters=12)                  │
                   │  Refined: F12, F21  →  F1t = t·F12,  F2t = (1-t)·F21      │
                   └───────────────────────────────────────────────────────────┘
                                          │
                   ┌──────────────────────▼──────────────────────────────────────┐
                   │  [ForwardWarp + FeatureExtractor (3 scales)]                 │
                   │  I1t, I2t, feat1t{1,2,3}, feat2t{1,2,3}                      │
                   └──────────────────────┬──────────────────────────────────────┘
                                          │
                   ┌──────────────────────▼──────────────────────────────────────┐
                   │  [GridNet Synthesis (6-ch input, 3-ch output)]               │
                   └──────────────────────┬──────────────────────────────────────┘
                                          │
                               frame_pred.png (t=0.5)
```

**Lưu ý quan trọng:**
- Model mặc định dùng **`softsplat` (cupy-based)** cho ForwardWarp. Trên Kaggle, nên dùng **`AnimeInterp_no_cupy`** để tránh lỗi compile CUDA extension.
- SGM flow (`guide_flo13.npy`, `guide_flo31.npy`) là **optional** — nếu không có, dùng zero-flow khởi tạo.
- Input tensor phải ở dạng `[B, 3, H, W]` float [0, 1], không cần normalize thêm (model tự normalize nội bộ).

---

## 2. Bảng 17 Chỉ Số Áp Dụng

| STT | Chỉ Số | Trạng Thái | Triển Khai |
|:---:|--------|:----------:|-----------|
| 1 | **Chamfer Distance (CD)** | ❌ | `N/A` — Model output RGB dày đặc, không phải point cloud |
| 2 | **Weighted CD (WCD)** | ❌ | `N/A` |
| 3 | **LPIPS** | ✅ | `lpips` với backbone AlexNet, input chuẩn hóa [-1, 1] |
| 4 | **EMD** | ❌ | `N/A` |
| 5 | **RRLD** | ❌ | `N/A` |
| 6 | **PSNR** | ✅ | `skimage.metrics.peak_signal_noise_ratio` |
| 7 | **SSIM** | ✅ | `skimage.metrics.structural_similarity` (multichannel, gaussian) |
| 8 | **IE (Interpolation Error)** | ⚠️ | MAE pixel-level: `mean(|pred - gt| × 255)` |
| 9 | **FVD** | ⚠️ | Chỉ tính khi chạy sequence ≥ 9 frame (disabled mặc định) |
| 10 | **Warping Error** | ⚠️ | Cần optical flow giữa các frame liên tiếp (disabled mặc định) |
| 11 | **Hausdorff Distance** | ❌ | `N/A` |
| 12 | **Line IoU** | ❌ | `N/A` |
| 13 | **Disconnection Rate** | ❌ | `N/A` |
| 14 | **Arbitrary t Support** | ❌ | `N/A` — mặc định `t=0.5` cố định |
| 15 | **Inference Speed** | ✅ | `time.perf_counter()` sau warmup 3 lần, tính ms/frame & FPS |
| 16 | **VRAM Peak** | ✅ | `torch.cuda.max_memory_allocated()` (MB) |
| 17 | **Fine-tuning Speed** | ⚠️ | Ghi nhận `N/A` (không có training script trong scope này) |

> **Tóm tắt chỉ số bắt buộc đo:** LPIPS ✅, PSNR ✅, SSIM ✅, Inference Speed ✅, VRAM Peak ✅  
> **Tùy chọn:** IE ⚠️  
> **Ghi N/A:** CD, WCD, EMD, RRLD, FVD*, Warping Error*, Hausdorff, Line IoU, Disconnection Rate, Arbitrary t, Fine-tuning Speed

---

## 3. Cấu Trúc File Output

```
Benchmark/AnimeInterp/
├── benchmark_AnimeInterp_guide.md      ← Tài liệu này
├── benchmark_AnimeInterp.ipynb         ← Notebook upload lên Kaggle
└── benchmark_results_AnimeInterp.csv   ← Sinh ra sau khi chạy notebook
```

**Format CSV đầu ra** (17 cột chuẩn hóa):
```
clip_name, CD, WCD, LPIPS, EMD, RRLD, PSNR, SSIM, IE, FVD, Warping_Error,
Hausdorff, Line_IoU, Disconnection_Rate, Arbitrary_t, Infer_ms, FPS, VRAM_MB, Finetune_Speed
```

---

## 4. Bước 0: Chuẩn Bị Trước Khi Upload Kaggle

### 4.1 Đóng gói code AnimeInterp thành Kaggle Dataset

```bash
# Trên máy local, zip thư mục model:
cd TestModels/
zip -r AnimeInterp_code.zip AnimeInterp/
```

Sau đó lên **kaggle.com → Datasets → New Dataset** → Upload file `AnimeInterp_code.zip`.

### 4.2 Đóng gói dataset ATD-12K

```bash
# Chỉ cần test_2k_540p/ cho benchmark cơ bản
zip -r ATD12K_test_540p.zip ATD-12K/test_2k_540p/
# (Tùy chọn) Thêm SGM flows nếu muốn dùng:
zip -r ATD12K_flows.zip ATD-12K/test_2k_pre_calc_sgm_flows/
```

Upload tương tự lên Kaggle Dataset.

### 4.3 Cài đặt Kaggle Notebook

1. Vào **kaggle.com → Code → New Notebook**
2. Chọn **Settings → Accelerator → GPU T4 x2** (hoặc P100)
3. Thêm 2 dataset vừa tạo vào notebook (Add Data)
4. Upload file `benchmark_AnimeInterp.ipynb` hoặc copy-paste từng cell

---

## 5. Cell 1 – Thiết Lập Môi Trường

```python
# ============================================================
# CELL 1: Cài đặt thư viện và kiểm tra môi trường
# ============================================================
import subprocess, sys

# 1a. Cài đặt packages cần thiết
packages = [
    "lpips",
    "opencv-python-headless",
    "scikit-image>=0.19",
    "scipy",
    "tqdm",
    "pandas",
    "matplotlib",
    "Pillow",
]
for pkg in packages:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

# 1b. Kiểm tra GPU
import torch
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"GPU count       : {torch.cuda.device_count()}")
    print(f"VRAM total      : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Kiểm tra bổ sung
import subprocess
subprocess.run(["nvidia-smi"], check=False)
```

**Lưu ý:**
- `lpips` là thư viện chính để tính LPIPS perceptual metric.
- `opencv-python-headless` không cần display (phù hợp server headless Kaggle).

---

## 6. Cell 2 – Nạp Model & Checkpoints

```python
# ============================================================
# CELL 2: Chuẩn bị source code và load model
# ============================================================
import os, sys, zipfile

# Đường dẫn Kaggle (thay đổi theo tên dataset của bạn)
KAGGLE_INPUT = "/kaggle/input"

# Giải nén code AnimeInterp (nếu upload dạng zip)
CODE_ZIP = os.path.join(KAGGLE_INPUT, "animeinterp-code/AnimeInterp_code.zip")
CODE_DIR = "/kaggle/working/AnimeInterp"

if os.path.exists(CODE_ZIP):
    with zipfile.ZipFile(CODE_ZIP, 'r') as z:
        z.extractall("/kaggle/working/")
    print(f"[OK] Giải nén code -> {CODE_DIR}")
else:
    # Nếu dataset đã là thư mục, dùng trực tiếp
    CODE_DIR = os.path.join(KAGGLE_INPUT, "animeinterp-code/AnimeInterp")
    print(f"[OK] Dùng trực tiếp từ {CODE_DIR}")

# Thêm vào sys.path
sys.path.insert(0, CODE_DIR)
print(f"sys.path[0] = {sys.path[0]}")

# Đường dẫn checkpoint
CKPT_PATH = os.path.join(CODE_DIR, "checkpoints/anime_interp_full.ckpt")
if not os.path.exists(CKPT_PATH):
    # Thử tìm trong Kaggle input
    for root, dirs, files in os.walk(KAGGLE_INPUT):
        for f in files:
            if "anime_interp_full" in f:
                CKPT_PATH = os.path.join(root, f)
                break
print(f"Checkpoint: {CKPT_PATH}")
assert os.path.exists(CKPT_PATH), f"❌ Không tìm thấy checkpoint tại: {CKPT_PATH}"

# Load model (dùng no_cupy để tránh lỗi compile extension trên Kaggle)
import torch
import torch.nn as nn

# Import model không dùng cupy (an toàn nhất trên Kaggle)
from models.AnimeInterp_no_cupy import AnimeInterpNoCupy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(ckpt_path, device):
    """Load AnimeInterp model từ checkpoint."""
    model = AnimeInterpNoCupy(path=None)  # path=None để không load flow từ RFR riêng
    
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Checkpoint có thể là dict với key 'model_state_dict' hoặc trực tiếp state_dict
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
    
    # Strip 'module.' prefix nếu checkpoint từ DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        key = k[7:] if k.startswith("module.") else k
        new_state_dict[key] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print(f"[OK] Model loaded: AnimeInterpNoCupy → {device}")
    return model

model = load_model(CKPT_PATH, DEVICE)
```

---

## 7. Cell 3 – Dataloader Triplet ATD-12K

```python
# ============================================================
# CELL 3: Dataset và DataLoader chuẩn triplet ATD-12K
# ============================================================
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

# Đường dẫn dataset
DATASET_ROOT = "/kaggle/input/atd12k-test/test_2k_540p"
# Nếu có SGM flows (tùy chọn):
FLOW_ROOT = "/kaggle/input/atd12k-flows/test_2k_pre_calc_sgm_flows"
USE_SGM_FLOW = os.path.exists(FLOW_ROOT)

# Kích thước resize (phải chia hết cho 32 để tránh lỗi với convolutional strides)
# Original: 960x540, nhưng để tiết kiệm VRAM dùng 480x270 hoặc giữ nguyên
TARGET_SIZE = (960, 540)  # (W, H) — giữ nguyên 540p


class ATD12KTripletDataset(Dataset):
    """
    Dataset đọc triplet (frame1, frame2, frame3) từ cấu trúc ATD-12K.
    
    frame1.png = Input I0 (t=0)
    frame2.png = Ground Truth (t=0.5)  
    frame3.png = Input I1 (t=1.0)
    """
    def __init__(self, root, flow_root=None, target_size=(960, 540)):
        self.root = root
        self.flow_root = flow_root
        self.target_size = target_size  # (W, H)
        
        self.clips = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])
        print(f"[Dataset] Tổng số clip: {len(self.clips)}")
        
        # Kiểm tra các clip hợp lệ (có đủ 3 frame)
        valid_clips = []
        for clip in self.clips:
            clip_dir = os.path.join(root, clip)
            f1 = os.path.join(clip_dir, "frame1.png")
            f2 = os.path.join(clip_dir, "frame2.png")
            f3 = os.path.join(clip_dir, "frame3.png")
            if all(os.path.exists(f) for f in [f1, f2, f3]):
                valid_clips.append(clip)
        
        self.clips = valid_clips
        print(f"[Dataset] Clip hợp lệ (đủ 3 frame): {len(self.clips)}")
    
    def _load_image(self, path):
        """Load ảnh PNG → Tensor [3, H, W] float [0, 1]."""
        img = Image.open(path).convert("RGB")
        if self.target_size is not None:
            img = img.resize(self.target_size, Image.LANCZOS)
        return TF.to_tensor(img)  # [3, H, W], range [0, 1]
    
    def _load_flow(self, npy_path, target_size):
        """Load optical flow từ .npy file."""
        if not os.path.exists(npy_path):
            # Nếu không có flow, trả về zero flow
            H, W = target_size[1], target_size[0]
            return torch.zeros(2, H, W, dtype=torch.float32)
        
        flow_np = np.load(npy_path)  # [2, H_orig, W_orig]
        flow = torch.from_numpy(flow_np.astype(np.float32))
        
        # Interpolate về target size
        import torch.nn.functional as F
        h_orig, w_orig = flow.shape[1], flow.shape[2]
        flow = F.interpolate(flow.unsqueeze(0), size=(target_size[1], target_size[0]))[0]
        
        # Scale flow values
        flow[0] *= target_size[0] / w_orig  # x-component
        flow[1] *= target_size[1] / h_orig  # y-component
        
        return flow
    
    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, idx):
        clip_name = self.clips[idx]
        clip_dir = os.path.join(self.root, clip_name)
        
        # Load ảnh
        I0 = self._load_image(os.path.join(clip_dir, "frame1.png"))  # [3, H, W]
        GT = self._load_image(os.path.join(clip_dir, "frame2.png"))  # [3, H, W]
        I1 = self._load_image(os.path.join(clip_dir, "frame3.png"))  # [3, H, W]
        
        # Load SGM flow nếu có
        if self.flow_root is not None:
            flow_dir = os.path.join(self.flow_root, clip_name)
            F12i = self._load_flow(os.path.join(flow_dir, "guide_flo13.npy"), self.target_size)
            F21i = self._load_flow(os.path.join(flow_dir, "guide_flo31.npy"), self.target_size)
        else:
            H, W = I0.shape[1], I0.shape[2]
            F12i = torch.zeros(2, H, W, dtype=torch.float32)
            F21i = torch.zeros(2, H, W, dtype=torch.float32)
        
        return {
            "I0": I0,
            "I1": I1,
            "GT": GT,
            "F12i": F12i,
            "F21i": F21i,
            "clip_name": clip_name,
        }


# Khởi tạo dataset
dataset = ATD12KTripletDataset(
    root=DATASET_ROOT,
    flow_root=FLOW_ROOT if USE_SGM_FLOW else None,
    target_size=TARGET_SIZE,
)

# DataLoader batch_size=1 (model thiết kế cho từng frame riêng lẻ)
dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

print(f"[OK] DataLoader sẵn sàng: {len(dataset)} mẫu")
```

---

## 8. Cell 4 – Module Tính Các Metrics

```python
# ============================================================
# CELL 4: Module tính metrics chuẩn hóa
# ============================================================
import torch
import numpy as np
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import time

# Khởi tạo LPIPS (chỉ cần làm 1 lần)
lpips_fn = lpips.LPIPS(net='alex').to(DEVICE)
lpips_fn.eval()
print("[OK] LPIPS (AlexNet) initialized")


def tensor_to_numpy_uint8(tensor):
    """Chuyển tensor [3, H, W] float [0,1] → numpy uint8 [H, W, 3] RGB."""
    arr = tensor.detach().cpu().clamp(0, 1).numpy()
    arr = (arr * 255).astype(np.uint8)
    return arr.transpose(1, 2, 0)  # [H, W, 3]


def compute_psnr(pred_np, gt_np):
    """PSNR giữa 2 ảnh numpy uint8 [H, W, 3]."""
    pred_f = pred_np.astype(np.float64) / 255.0
    gt_f   = gt_np.astype(np.float64) / 255.0
    return peak_signal_noise_ratio(gt_f, pred_f, data_range=1.0)


def compute_ssim(pred_np, gt_np):
    """SSIM (multichannel) giữa 2 ảnh numpy uint8 [H, W, 3]."""
    pred_f = pred_np.astype(np.float64) / 255.0
    gt_f   = gt_np.astype(np.float64) / 255.0
    return structural_similarity(gt_f, pred_f, channel_axis=2, gaussian_weights=True, data_range=1.0)


def compute_lpips(pred_tensor, gt_tensor, lpips_fn, device):
    """
    LPIPS perceptual similarity.
    Input: tensor [3, H, W] float [0, 1]
    Output: scalar float (thấp hơn = tốt hơn)
    """
    # LPIPS yêu cầu input [-1, 1]
    pred_t = (pred_tensor.unsqueeze(0).to(device) * 2 - 1).clamp(-1, 1)
    gt_t   = (gt_tensor.unsqueeze(0).to(device) * 2 - 1).clamp(-1, 1)
    with torch.no_grad():
        score = lpips_fn(pred_t, gt_t)
    return score.item()


def compute_ie(pred_np, gt_np):
    """
    IE (Interpolation Error): MAE theo chuẩn ATD-12K paper.
    IE = mean(sqrt(sum((pred*255 - gt*255)^2, axis=channel)))
    Tương đương Euclidean distance per-pixel, averaged.
    """
    pred_f = pred_np.astype(np.float64)
    gt_f   = gt_np.astype(np.float64)
    per_pixel_error = np.sqrt(np.sum((pred_f - gt_f) ** 2, axis=2))  # [H, W]
    return float(np.mean(per_pixel_error))


@torch.no_grad()
def run_inference_timed(model, I0, I1, F12i, F21i, device, t=0.5, n_warmup=3):
    """
    Chạy inference và đo thời gian (loại bỏ GPU warmup).
    
    Returns:
        pred_tensor: [3, H, W] float [0,1]
        latency_ms: thời gian inference (ms)
        vram_mb: VRAM peak (MB)
    """
    I0 = I0.unsqueeze(0).to(device)   # [1, 3, H, W]
    I1 = I1.unsqueeze(0).to(device)   # [1, 3, H, W]
    F12i = F12i.unsqueeze(0).to(device)  # [1, 2, H, W]
    F21i = F21i.unsqueeze(0).to(device)  # [1, 2, H, W]
    
    # GPU Warmup (loại bỏ overhead lần đầu)
    torch.cuda.synchronize()
    for _ in range(n_warmup):
        _ = model(I0, I1, F12i, F21i, t)
    torch.cuda.synchronize()
    
    # Reset VRAM counter
    torch.cuda.reset_peak_memory_stats(device)
    
    # Đo thời gian thực sự
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    
    outputs = model(I0, I1, F12i, F21i, t)
    
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    
    latency_ms = (t_end - t_start) * 1000.0
    vram_mb    = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    
    # outputs = (It_warp, F12, F21, F12in, F21in)
    pred = outputs[0].squeeze(0).cpu().clamp(0, 1)  # [3, H, W]
    
    return pred, latency_ms, vram_mb
```

---

## 9. Cell 5 – Pipeline Benchmark & Xuất CSV

```python
# ============================================================
# CELL 5: Pipeline Benchmark tự động + Xuất báo cáo 17 chỉ số
# ============================================================
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import os

OUTPUT_DIR = "/kaggle/working/benchmark_AnimeInterp"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUTPUT_DIR, "benchmark_results_AnimeInterp.csv")

# Cấu hình
T_INTERP = 0.5     # Thời điểm nội suy (cố định t=0.5 cho ATD-12K)
N_WARMUP  = 3      # Số lần warmup GPU trước khi đo

# Cột kết quả theo chuẩn 17 chỉ số
COLUMNS = [
    "clip_name",
    "CD", "WCD",                          # ❌ → N/A
    "LPIPS",                               # ✅
    "EMD", "RRLD",                         # ❌ → N/A
    "PSNR", "SSIM",                        # ✅
    "IE",                                  # ⚠️
    "FVD", "Warping_Error",                # ⚠️ → N/A (single-frame mode)
    "Hausdorff", "Line_IoU",               # ❌ → N/A
    "Disconnection_Rate",                  # ❌ → N/A
    "Arbitrary_t",                         # ❌ → N/A
    "Infer_ms", "FPS",                     # ✅
    "VRAM_MB",                             # ✅
    "Finetune_Speed",                      # ⚠️ → N/A
]

results = []
preview_items = []  # Lưu ảnh để visualize

print("=" * 60)
print("  AnimeInterp Benchmark — ATD-12K test_2k_540p")
print("=" * 60)
print(f"  Dataset  : {DATASET_ROOT}")
print(f"  Samples  : {len(dataset)}")
print(f"  Device   : {DEVICE}")
print(f"  SGM Flow : {'Enabled' if USE_SGM_FLOW else 'Disabled (zero-init)'}")
print("=" * 60)

model.eval()

for batch_idx, batch in enumerate(tqdm(dataloader, desc="Benchmarking")):
    clip_name = batch["clip_name"][0]
    I0   = batch["I0"][0]    # [3, H, W]
    I1   = batch["I1"][0]    # [3, H, W]
    GT   = batch["GT"][0]    # [3, H, W]
    F12i = batch["F12i"][0]  # [2, H, W]
    F21i = batch["F21i"][0]  # [2, H, W]
    
    try:
        # ─── Inference có đo thời gian ───────────────────────────
        pred, latency_ms, vram_mb = run_inference_timed(
            model, I0, I1, F12i, F21i, DEVICE, t=T_INTERP, n_warmup=N_WARMUP
        )
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
        
        # ─── Chuyển sang numpy ───────────────────────────────────
        pred_np = tensor_to_numpy_uint8(pred)   # [H, W, 3] uint8
        gt_np   = tensor_to_numpy_uint8(GT)     # [H, W, 3] uint8
        
        # ─── Tính metrics ✅ ─────────────────────────────────────
        psnr  = compute_psnr(pred_np, gt_np)
        ssim  = compute_ssim(pred_np, gt_np)
        lp    = compute_lpips(pred, GT, lpips_fn, DEVICE)
        ie    = compute_ie(pred_np, gt_np)
        
        # Lưu sample đầu tiên để visualize
        if len(preview_items) < 6:
            preview_items.append({
                "clip": clip_name,
                "I0_np": tensor_to_numpy_uint8(I0),
                "pred_np": pred_np,
                "GT_np": gt_np,
                "I1_np": tensor_to_numpy_uint8(I1),
                "psnr": psnr,
                "ssim": ssim,
                "lpips": lp,
            })
        
        row = {
            "clip_name"        : clip_name,
            "CD"               : "N/A",
            "WCD"              : "N/A",
            "LPIPS"            : round(lp, 6),
            "EMD"              : "N/A",
            "RRLD"             : "N/A",
            "PSNR"             : round(psnr, 4),
            "SSIM"             : round(ssim, 6),
            "IE"               : round(ie, 4),
            "FVD"              : "N/A",
            "Warping_Error"    : "N/A",
            "Hausdorff"        : "N/A",
            "Line_IoU"         : "N/A",
            "Disconnection_Rate": "N/A",
            "Arbitrary_t"      : "N/A",
            "Infer_ms"         : round(latency_ms, 2),
            "FPS"              : round(fps, 2),
            "VRAM_MB"          : round(vram_mb, 2),
            "Finetune_Speed"   : "N/A",
        }
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n⚠️  OOM tại clip '{clip_name}' — bỏ qua. Thử giảm TARGET_SIZE.")
            torch.cuda.empty_cache()
            row = {col: "OOM" for col in COLUMNS}
            row["clip_name"] = clip_name
        else:
            print(f"\n❌ Lỗi tại clip '{clip_name}': {e}")
            row = {col: "ERROR" for col in COLUMNS}
            row["clip_name"] = clip_name
    
    results.append(row)
    
    # Giải phóng cache VRAM định kỳ
    if (batch_idx + 1) % 50 == 0:
        torch.cuda.empty_cache()


# ─── Tạo DataFrame và tính statistics ─────────────────────────
df = pd.DataFrame(results, columns=COLUMNS)

# Chuyển N/A về np.nan để tính toán
numeric_cols = ["LPIPS", "PSNR", "SSIM", "IE", "Infer_ms", "FPS", "VRAM_MB"]
df_num = df[numeric_cols].copy()
for col in numeric_cols:
    df_num[col] = pd.to_numeric(df[col], errors='coerce')

print("\n" + "=" * 60)
print("  📊 BENCHMARK SUMMARY — AnimeInterp (CVPR 2021)")
print("=" * 60)

summary = df_num.agg(['mean', 'std', 'min', 'max']).round(4)
print(summary.to_markdown())

# ─── Thêm hàng SUMMARY vào cuối DataFrame ──────────────────────
summary_row = {"clip_name": "*** MEAN ***"}
for col in numeric_cols:
    summary_row[col] = round(df_num[col].mean(), 4)
for col in COLUMNS:
    if col not in summary_row:
        summary_row[col] = "—"
df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

# ─── Xuất CSV ───────────────────────────────────────────────────
df.to_csv(CSV_PATH, index=False)
print(f"\n[OK] CSV saved → {CSV_PATH}")

# ─── In bảng tóm tắt định dạng Markdown ─────────────────────────
print("\n## 📋 Bảng Kết Quả Tóm Tắt (17 Chỉ Số)\n")
print("| Chỉ số | Mean | Std | Min | Max |")
print("|--------|------|-----|-----|-----|")
for col in numeric_cols:
    v = df_num[col].dropna()
    print(f"| **{col}** | {v.mean():.4f} | {v.std():.4f} | {v.min():.4f} | {v.max():.4f} |")

na_metrics = ["CD", "WCD", "EMD", "RRLD", "FVD", "Warping_Error",
               "Hausdorff", "Line_IoU", "Disconnection_Rate", "Arbitrary_t", "Finetune_Speed"]
for m in na_metrics:
    print(f"| **{m}** | N/A | — | — | — |")


# ─── Visualization: Preview frames ─────────────────────────────
if preview_items:
    n_show = min(len(preview_items), 3)
    fig, axes = plt.subplots(n_show, 4, figsize=(20, 5 * n_show))
    if n_show == 1:
        axes = [axes]
    
    fig.suptitle("AnimeInterp Benchmark Preview\n(Input I0 | Predicted | Ground Truth | Input I1)",
                 fontsize=14, fontweight='bold')
    
    for i, item in enumerate(preview_items[:n_show]):
        axes[i][0].imshow(item["I0_np"]);    axes[i][0].set_title(f"I0 (t=0)\n{item['clip']}", fontsize=9)
        axes[i][1].imshow(item["pred_np"]); axes[i][1].set_title(f"Pred (t=0.5)\nPSNR={item['psnr']:.2f}dB  SSIM={item['ssim']:.4f}", fontsize=9)
        axes[i][2].imshow(item["GT_np"]);   axes[i][2].set_title(f"GT (t=0.5)\nLPIPS={item['lpips']:.4f}", fontsize=9)
        axes[i][3].imshow(item["I1_np"]);   axes[i][3].set_title("I1 (t=1)", fontsize=9)
        for ax in axes[i]:
            ax.axis("off")
    
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "preview_results.png")
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"[OK] Preview saved → {fig_path}")

# ─── Distribution plots ─────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
for ax, col, color in zip(axes2, ["PSNR", "SSIM", "LPIPS"], ["steelblue", "green", "coral"]):
    data = df_num[col].dropna()
    ax.hist(data, bins=30, color=color, alpha=0.7, edgecolor='white')
    ax.axvline(data.mean(), color='red', linestyle='--', label=f'Mean={data.mean():.3f}')
    ax.set_title(f"{col} Distribution", fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.legend()
fig2.suptitle("AnimeInterp — Metric Distributions", fontsize=13, fontweight='bold')
plt.tight_layout()
fig2_path = os.path.join(OUTPUT_DIR, "metric_distributions.png")
plt.savefig(fig2_path, dpi=100, bbox_inches='tight')
plt.show()
print(f"[OK] Distribution plot saved → {fig2_path}")

print("\n✅ BENCHMARK HOÀN TẤT!")
print(f"   Tổng số clip đã xử lý : {len(results)}")
print(f"   Kết quả lưu tại        : {CSV_PATH}")
```

---

## 10. Kết Quả Mong Đợi

Dựa trên kết quả paper gốc trên ATD-12K test set (960×540):

| Chỉ Số | Giá Trị Mong Đợi | Ghi Chú |
|--------|:---:|---------|
| **PSNR** | ~30–33 dB | Higher is better |
| **SSIM** | ~0.92–0.96 | Higher is better |
| **LPIPS** | ~0.03–0.08 | Lower is better |
| **IE** | ~3–8 | Lower is better |
| **Infer_ms** | ~150–400 ms | Phụ thuộc GPU (P100 nhanh hơn T4) |
| **VRAM_MB** | ~2000–5000 MB | Phụ thuộc độ phân giải |

> **Nếu kết quả lệch đáng kể:** Kiểm tra xem SGM flow có được load đúng không — model dùng zero-flow sẽ kém hơn ~1-2 dB PSNR so với dùng SGM flow đầy đủ.

---

## 11. Troubleshooting

### 🔴 Lỗi: `CUDA Out of Memory`
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```
**Giải pháp:**
1. Giảm `TARGET_SIZE`:
   ```python
   TARGET_SIZE = (480, 272)  # Thay vì (960, 540)
   ```
2. Thêm `torch.cuda.empty_cache()` sau mỗi iteration.
3. Tắt gradient tracking: đảm bảo `@torch.no_grad()` được áp dụng đúng.
4. Với T4 x2, chỉ dùng 1 GPU (batch_size=1 không cần DataParallel).

### 🔴 Lỗi: `module 'cupy' not found`
**Giải pháp:** Dùng `AnimeInterp_no_cupy.py` thay vì `AnimeInterp.py`:
```python
# Thay dòng import:
from models.AnimeInterp_no_cupy import AnimeInterpNoCupy as Model
```

### 🔴 Lỗi: `size of tensor must be divisible by 32`
**Giải pháp:** Pad ảnh trước khi inference:
```python
import torch.nn.functional as F

def pad_to_multiple(tensor, multiple=32):
    """Pad tensor [B, C, H, W] để H và W chia hết cho multiple."""
    _, _, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect'), H, W

def unpad(tensor, orig_H, orig_W):
    return tensor[:, :, :orig_H, :orig_W]
```

### 🔴 Lỗi: PSNR/SSIM = `nan` hoặc rất thấp
**Nguyên nhân:** Ảnh output bị clamp sai hoặc range không đúng.
**Kiểm tra:**
```python
print(pred.min().item(), pred.max().item())  # Phải trong [0, 1]
```

### 🔴 Lỗi: `strict=False` load model thiếu key
**Nguyên nhân:** Checkpoint được save với `DataParallel` hoặc prefix khác.
**Giải pháp:** In ra các key không match để debug:
```python
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("Missing:", missing[:5])
print("Unexpected:", unexpected[:5])
```

### 🟡 LPIPS rất cao (>0.5)
**Nguyên nhân thường gặp:** Ảnh màu bị lệch RGB/BGR.
- AnimeInterp xử lý ảnh RGB (PIL load là RGB). Không cần đảo channel.
- Nếu dùng OpenCV để load, nhớ `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`.

### 🟡 Inference chậm hơn mong đợi
- Trên T4: ~400–600ms/frame (bình thường).
- Trên P100: ~200–350ms/frame (nhanh hơn).
- `iters=12` trong RFR là cố định — không thể giảm mà không re-train.

---

## Phụ Lục: Cấu Trúc Checkpoint

```
anime_interp_full.ckpt
├── model_state_dict:
│   ├── module.flownet.*     (RFR - Recurrent Flow Refinement)
│   ├── module.feat_ext.*    (FeatureExtractor - 3 scales)
│   ├── module.fwarp.*       (ForwardWarp)
│   └── module.synnet.*      (GridNet synthesis)
└── epoch / optimizer_state_dict (nếu có)
```

Khi load, cần strip prefix `module.` (từ DataParallel training):
```python
new_dict = {k.replace("module.", ""): v for k, v in ckpt.items()}
```

---

*Benchmark được tạo bởi hệ thống PBL6 — AnimeInterp CVPR 2021 evaluation pipeline*
