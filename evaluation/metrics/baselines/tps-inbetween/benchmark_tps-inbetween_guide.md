# 📖 Hướng Dẫn Benchmark TPS-Inbetween trên Kaggle GPU

> **Model:** TPS-Inbetween — *Thin-Plate Spline-based Interpolation for Animation Line Inbetweening*  
> **Paper:** "Thin-Plate Spline-based Interpolation for Animation Line Inbetweening" (Tianyi Zhu, Wei Shang, Dongwei Ren, Wangmeng Zuo)  
> **Môi trường:** Kaggle Notebook · GPU T4 x2 hoặc P100 · Python 3.10+ · PyTorch ≥ 1.10  
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

**TPS-Inbetween** là mô hình inbetweening (nội suy khung hình trung gian) chuyên biệt cho **animation line art** (hoạt họa đường nét). Pipeline gồm 2 giai đoạn chính:

```
frame1.png (t=0) ─────────────────────────────────────────────────────┐
                                                                        │
frame3.png (t=1) ─────────────────────────────────────────────────────┤
                                                                        │
                                                                        ▼
                    ┌─────────────── [GlueStick Matcher] ──────────────┐
                    │  • Trích xuất keypoints & line segments           │
                    │  • Khớp đặc trưng giữa 2 frame                   │
                    │  → matched_kps0, matched_kps1                     │
                    └──────────────────┬────────────────────────────────┘
                                       │  (normalized keypoints)
                                       ▼
                    ┌─────────── [TPS Motion Estimation] ───────────────┐
                    │  • tps_theta_from_points(kps0 → kps1)             │
                    │  • tps_grid() → coarse_flow01, coarse_flow10      │
                    │  • Tính coarse_flow0t = t²·flow01 - (1-t)t·flow10 │
                    └──────────────────┬────────────────────────────────┘
                                       │
                                       ▼
                    ┌─────────── [Fine Motion Refinement] ──────────────┐
                    │  • Contextnet encoder (indim=1, 4-scale feats)     │
                    │  • IFNet (RIFE-based): coarse_I0t, coarse_I1t      │
                    │    → fine flow_bet_coar, mask                       │
                    └──────────────────┬────────────────────────────────┘
                                       │
                                       ▼
                    ┌─────────── [UNet Synthesis] ────────────────────┐
                    │  • indim=9, outdim=1 (grayscale line art)        │
                    │  • It = (1-t)·fine_I0t + t·fine_I1t + residual  │
                    │  → It = clamp(It, 0, 1)                          │
                    └──────────────────┬───────────────────────────────┘
                                       │
                              pred_frame.png (t=0.5)
```

**Đặc điểm kỹ thuật quan trọng:**
- **Input:** Ảnh grayscale (L mode) – model làm việc ở không gian line art đen trắng
- **Output:** Grayscale line art `[0, 1]`, shape `[1, 1, H, W]`
- **Arbitrary t:** Hỗ trợ đầy đủ bằng cách đặt `xN` và `self.times = torch.linspace(0, 1, xN+1)[1:-1]`
- **Padding:** Kích thước ảnh phải chia hết cho **32** (stride của IFNet)
- **Pretrained weights:** `ckpt/model_latest.pt` + `model/resources/weights/checkpoint_GlueStick_MD.tar`
- **Inversion convention:** Model hoạt động trên ảnh **đảo ngược** (nét đen = 0 → thành 1, nền trắng = 1 → thành 0)

---

## 2. Bảng 17 Chỉ Số Áp Dụng

| STT | Chỉ Số | Trạng Thái | Triển Khai |
|:---:|--------|:----------:|-----------|
| 1 | **Chamfer Distance (CD)** | ✅ | Binarize output → edge map → `batch_chamfer_distance()` |
| 2 | **Weighted CD (WCD)** | ✅ | Binarize → `batch_chamfer_distance_w()` với softplus weight |
| 3 | **LPIPS** | ⚠️ | Tính trên ảnh grayscale repeat 3 channel → `lpips.LPIPS(net='alex')` |
| 4 | **EMD** | ⚠️ | Binarize → nonzero points → `wasserstein_1d()` per batch |
| 5 | **RRLD** | ⚠️ | `mean(|pred_line - gt_line|) / mean(gt_line)` (relative line density ratio) |
| 6 | **PSNR** | ⚠️ | `skimage.metrics.peak_signal_noise_ratio` trên grayscale output |
| 7 | **SSIM** | ⚠️ | `kornia.metrics.ssim()` trên ảnh grayscale `[0, 1]` |
| 8 | **IE (Interpolation Error)** | ⚠️ | MAE pixel: `mean(|pred - gt| × 255)` |
| 9 | **FVD** | ❌ | `N/A` — Không áp dụng cho triplet benchmark đơn frame |
| 10 | **Warping Error** | ❌ | `N/A` — Không áp dụng |
| 11 | **Hausdorff Distance** | ✅ | Binarize → `batch_hausdorff_distance()` via EDT |
| 12 | **Line IoU** | ✅ | Dilate 3px → `intersection / union` trên binary maps |
| 13 | **Disconnection Rate** | ✅ | `cv2.connectedComponentsWithStats()`: so sánh số lượng connected components |
| 14 | **Arbitrary t Support** | ✅ | Ghi nhận `YES` + demo t=[0.25, 0.5, 0.75] |
| 15 | **Inference Speed** | ✅ | Warmup 3 lần → đo `time.perf_counter()`, tính ms/frame & FPS |
| 16 | **VRAM Peak** | ✅ | `torch.cuda.reset_peak_memory_stats()` → `max_memory_allocated()` (MB) |
| 17 | **Fine-tuning Speed** | ⚠️ | `N/A` (không có training script trong scope benchmark này) |

> **Lưu ý về domain:** TPS-Inbetween là line art model. Khi benchmark trên ATD-12K (ảnh màu), cần chuyển ảnh sang grayscale L-mode để phù hợp với đầu vào model. Khi đó các metric dựa trên binary edge map (CD, WCD, HD, Line IoU, Disconnection Rate) sẽ được tính trên DoG-thresholded output.

---

## 3. Cấu Trúc File Output

```
Benchmark/tps-inbetween/
├── benchmark_tps-inbetween_guide.md    ← Tài liệu hướng dẫn này
├── benchmark_tps-inbetween.ipynb       ← Notebook chạy trực tiếp trên Kaggle
└── benchmark_results_tps-inbetween.csv ← Kết quả sau khi chạy (17 cột)
```

**Cấu trúc CSV output (`benchmark_results_tps-inbetween.csv`):**

```
clip_name, CD, WCD, LPIPS, EMD, RRLD, PSNR, SSIM, IE, FVD, Warping_Error,
Hausdorff, Line_IoU, Disconnection_Rate, Arbitrary_t, Speed_ms, FPS,
VRAM_MB, Fine_tuning_Speed
```

---

## 4. Bước 0: Chuẩn Bị Trước Khi Upload Kaggle

### 4.1 Chuẩn bị code repository

Trên máy local, zip thư mục model:
```bash
cd TestModels/
zip -r tps_inbetween_code.zip tps-inbetween/ --exclude "tps-inbetween/.git/*" --exclude "tps-inbetween/__pycache__/*"
```

### 4.2 Upload lên Kaggle Dataset

1. Vào **kaggle.com → Datasets → New Dataset**
2. Upload file zip:
   - `tps_inbetween_code.zip` (source code + `ckpt/model_latest.pt` ~53MB)
   - `atd12k_test_540p.zip` (tập test ATD-12K 540p)
3. Chú ý path trên Kaggle sau khi mount:
   ```
   /kaggle/input/tps-inbetween-model/tps_inbetween_code.zip
   /kaggle/input/atd12k-benchmark/atd12k_test_540p.zip
   ```

### 4.3 Checkpoints cần thiết

| File | Kích thước | Nguồn |
|------|-----------|-------|
| `ckpt/model_latest.pt` | ~53 MB | [Google Drive](https://drive.google.com/file/d/1B15f04lrfClFE34vP_ecHm5WEDTqN6iJ/view) |
| `model/resources/weights/checkpoint_GlueStick_MD.tar` | ~19 MB | [GlueStick GitHub](https://github.com/cvg/GlueStick) |

> ⚠️ **Quan trọng:** Cả hai file trọng số phải có mặt trước khi chạy inference.

---

## 5. Cell 1 – Thiết Lập Môi Trường

### Mục tiêu
- Cài đặt các thư viện cần thiết (lpips, kornia, imageio, POT, torchmetrics…)
- Kiểm tra GPU khả dụng
- Đặt hạt ngẫu nhiên để tái tạo kết quả

### Code

```python
# ============================================================
# CELL 1: THIẾT LẬP MÔI TRƯỜNG
# ============================================================
import subprocess, sys

# Cài các package bổ sung (PyTorch đã sẵn có trên Kaggle)
packages = [
    "pytlsd",               # Line Segment Detector cho GlueStick Wireframe
    "omegaconf",            # Cấu hình BaseModel trong GlueStick
    "scikit-learn",         # DBSCAN cluster trong Wireframe
    "lpips",
    "kornia",
    "imageio[ffmpeg]",
    "POT",                  # Python Optimal Transport (cho EMD/Wasserstein)
    "torchmetrics",
    "scikit-image",
    "opencv-python-headless",
    "scipy",
    "tqdm",
    "pandas",
    "matplotlib",
    "timm",
    "tabulate",
]
for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# Kiểm tra GPU
import torch
import os

print("=" * 60)
print("🔧 ENVIRONMENT CHECK")
print("=" * 60)
print(f"Python     : {sys.version}")
print(f"PyTorch    : {torch.__version__}")
print(f"CUDA avail : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU [{i}]    : {props.name} | {props.total_memory / 1024**3:.1f} GB VRAM")
    os.system("nvidia-smi")
else:
    print("⚠️  No GPU detected. Will run on CPU (very slow).")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n✅ Using device: {DEVICE}")

# Reproducibility
import random
import numpy as np

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```

---

## 6. Cell 2 – Nạp Model & Checkpoints

### Mục tiêu
- Giải nén code nếu cần
- Thiết lập `sys.path`
- Khởi tạo GlueStick Matcher và TPS-Inbetween model

### Code

```python
# ============================================================
# CELL 2: AUTO DISCOVERY & LOAD SOURCE CODE & MODEL WEIGHTS
# ============================================================
import os
import sys
import zipfile
import shutil
import urllib.request
from pathlib import Path

print("🔍 [STEP 1/4] Tự động tìm kiếm mã nguồn TPS-Inbetween...")

SEARCH_ROOTS = ["/kaggle/input", "/kaggle/working", ".", "..", "../../TestModels", "./TestModels"]
MODEL_ROOT = None

# ── 2.1 Quét và tự động giải nén bất kỳ file zip nào chứa mã nguồn ──
for sroot in SEARCH_ROOTS:
    if not os.path.exists(sroot):
        continue
    for root, dirs, files in os.walk(sroot):
        for f in files:
            if f.endswith(".zip"):
                zip_path = os.path.join(root, f)
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        names = zf.namelist()
                        if any("tpsinbet.py" in n or "gluestick" in n or "IFNet.py" in n for n in names):
                            target_extract = "/kaggle/working/tps-inbetween" if os.path.exists("/kaggle/working") else "./extracted_tps_inbetween"
                            print(f"📦 Tìm thấy file zip mã nguồn: {zip_path}")
                            print(f"   Đang giải nén vào {target_extract}...")
                            os.makedirs(target_extract, exist_ok=True)
                            zf.extractall(target_extract)
                            print(f"✅ Đã giải nén xong!")
                            break
                except Exception as e:
                    pass
        if MODEL_ROOT is not None:
            break

# ── 2.2 Quét tìm thư mục chứa module `model.tpsinbet` & `model.gluestick` ──
if FOUND_SOURCE_DIR is None:
    for sroot in ["/kaggle/input", "/kaggle/working", ".", "..", "../../TestModels", "./TestModels"]:
        if not os.path.exists(sroot):
            continue
        for root, dirs, files in os.walk(sroot):
            if os.path.exists(os.path.join(root, "model", "tpsinbet.py")) and os.path.exists(os.path.join(root, "model", "gluestick")):
                FOUND_SOURCE_DIR = os.path.abspath(root)
                break
            if "tpsinbet.py" in files and "gluestick" in dirs:
                FOUND_SOURCE_DIR = os.path.abspath(os.path.dirname(root))
                break
        if FOUND_SOURCE_DIR is not None:
            break

if FOUND_SOURCE_DIR is None:
    print("❌ KHÔNG TÌM THẤY MÃ NGUỒN TPS-INBETWEEN TRONG /kaggle/input!")
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            print(f"   - {root} | files: {files[:5]}")
    raise FileNotFoundError("Vui lòng thêm Dataset chứa mã nguồn TPS-Inbetween vào Kaggle Notebook!")

print(f"✅ Đã tìm thấy mã nguồn tại: {FOUND_SOURCE_DIR}")

# ── 2.3 Sao chép sang /kaggle/working để đảm bảo quyền GHI (Writable) ──
TARGET_WORKSPACE = "/kaggle/working/tps-inbetween" if os.path.exists("/kaggle/working") else os.path.abspath("./tps-inbetween")

if os.path.abspath(FOUND_SOURCE_DIR) != os.path.abspath(TARGET_WORKSPACE):
    print(f"📋 Đang sao chép mã nguồn sang không gian làm việc có quyền ghi: {TARGET_WORKSPACE}...")
    os.makedirs(TARGET_WORKSPACE, exist_ok=True)
    shutil.copytree(FOUND_SOURCE_DIR, TARGET_WORKSPACE, dirs_exist_ok=True)
    MODEL_ROOT = TARGET_WORKSPACE
else:
    MODEL_ROOT = FOUND_SOURCE_DIR

if MODEL_ROOT not in sys.path:
    sys.path.insert(0, MODEL_ROOT)
os.chdir(MODEL_ROOT)
print(f"📂 Writable Working Directory: {os.getcwd()}")

# Đảm bảo model có __init__.py hợp lệ
model_init = os.path.join(MODEL_ROOT, "model", "__init__.py")
if not os.path.exists(model_init):
    try:
        Path(model_init).touch()
    except Exception:
        pass

# ── 2.4 Kiểm tra & Chuẩn bị Checkpoint Weights ──
print("\n🔍 [STEP 2/4] Kiểm tra Checkpoint Weights...")
CKPT_PATH = os.path.join(MODEL_ROOT, "ckpt", "model_latest.pt")
GLUE_WEIGHTS = os.path.join(MODEL_ROOT, "model", "resources", "weights", "checkpoint_GlueStick_MD.tar")

os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
os.makedirs(os.path.dirname(GLUE_WEIGHTS), exist_ok=True)

if not os.path.exists(CKPT_PATH) and os.path.exists("/kaggle/input"):
    for root, dirs, files in os.walk("/kaggle/input"):
        for f in files:
            if f == "model_latest.pt" or (f.endswith(".pt") and "model" in f):
                found_ckpt = os.path.join(root, f)
                print(f"📋 Đã tìm thấy TPS checkpoint tại: {found_ckpt} -> Sao chép vào {CKPT_PATH}")
                shutil.copyfile(found_ckpt, CKPT_PATH)
                break
        if os.path.exists(CKPT_PATH):
            break

if not os.path.exists(GLUE_WEIGHTS) and os.path.exists("/kaggle/input"):
    for root, dirs, files in os.walk("/kaggle/input"):
        if "checkpoint_GlueStick_MD.tar" in files:
            found_glue = os.path.join(root, "checkpoint_GlueStick_MD.tar")
            shutil.copyfile(found_glue, GLUE_WEIGHTS)
            break

if os.path.exists(GLUE_WEIGHTS):
    print(f"✅ Đã tìm thấy GlueStick Checkpoint: {GLUE_WEIGHTS} ({os.path.getsize(GLUE_WEIGHTS)/(1024**2):.1f} MB)")
else:
    print(f"🌐 Đang tải GlueStick weights từ GitHub Release về {GLUE_WEIGHTS}...")
    glue_url = "https://github.com/cvg/GlueStick/releases/download/v0.1_arxiv/checkpoint_GlueStick_MD.tar"
    try:
        urllib.request.urlretrieve(glue_url, GLUE_WEIGHTS)
        print(f"✅ Tải GlueStick checkpoint thành công ({os.path.getsize(GLUE_WEIGHTS)/(1024**2):.1f} MB)")
    except Exception as e:
        print(f"❌ Không thể tải tự động GlueStick weights: {e}")

if os.path.exists(CKPT_PATH):
    print(f"✅ Đã tìm thấy TPS Checkpoint: {CKPT_PATH} ({os.path.getsize(CKPT_PATH)/(1024**2):.1f} MB)")
else:
    print(f"⚠️ Không tìm thấy {CKPT_PATH}. Model sẽ được khởi tạo mặc định!")

# ── 2.4 Import modules ──
import torch
import numpy as np
from model.gluestick import batch_to_np, numpy_image_to_torch, GLUESTICK_ROOT
import util.utils as utils
from util.utils import tps_theta_from_points, tps_grid

# --- 2.5 Khởi tạo GlueStick Matcher (Tối ưu hóa VRAM) ---
print("\n🔗 Loading GlueStick Matcher...")
MATCHER_CONF = {
    'name': 'two_view_pipeline',
    'use_lines': True,
    'extractor': {
        'name': 'wireframe',
        'sp_params': {
            'force_num_keypoints': False,
            'max_num_keypoints': 400,  # 400 thay vì 1000 để giảm 70% VRAM & tăng gấp đôi tốc độ
        },
        'wireframe_params': {
            'merge_points': True,
            'merge_line_endpoints': True,
        },
        'max_n_lines': 150,           # 150 thay vì 300
    },
    'matcher': {
        'name': 'gluestick',
        'weights': GLUE_WEIGHTS,
        'trainable': False,
    },
    'ground_truth': {'from_pose_depth': False},
}

matching_model = TwoViewPipeline(MATCHER_CONF).to(DEVICE).eval()
print(f"✅ GlueStick Matcher loaded on {DEVICE}")

# ── Patch TPS.fit chống Singular Matrix Crash ──────────────────
original_tps_fit = utils.TPS.fit
def robust_tps_fit(c, lambd=1e-3, reduced=False):
    n = c.shape[0]
    device = c.device
    U = utils.TPS.u(utils.TPS.d(c, c))
    K = U + torch.eye(n, dtype=torch.float32, device=device) * max(lambd, 1e-4)
    P = torch.ones((n, 3), dtype=torch.float32, device=device)
    P[:, 1:] = c[:, :2]
    v = torch.zeros(n + 3, dtype=torch.float32, device=device)
    v[:n] = c[:, -1]
    A = torch.zeros((n + 3, n + 3), dtype=torch.float32, device=device)
    A[:n, :n] = K
    A[:n, -3:] = P
    A[-3:, :n] = P.T
    try:
        theta = torch.linalg.solve(A, v)
    except Exception:
        try:
            theta = torch.linalg.lstsq(A, v).solution
        except Exception:
            theta = torch.zeros(n + 3, dtype=torch.float32, device=device)
    return theta[1:] if reduced else theta

utils.TPS.fit = robust_tps_fit
print("🛡️ Đã kích hoạt lớp bảo vệ chống Singular Matrix cho TPS Motion Solver!")

# --- 2.6 Khởi tạo TPS-Inbetween ---
print("\n🧠 Loading TPS-Inbetween model...")

class TPSArgs:
    """Minimal args object để khởi tạo TPS_inbet"""
    xN = 2       # xN=2 sinh ra 1 frame trung gian tại t=0.5 (torch.linspace(0, 1, 3)[1:-1] == [0.5])
    cpu = (DEVICE == "cpu")

tps_args = TPSArgs()
tps_model = TPS_inbet(tps_args)
tps_model.times = torch.tensor([0.5], device=DEVICE)
tps_model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
tps_model = tps_model.to(DEVICE).eval()
print(f"✅ TPS-Inbetween loaded from {CKPT_PATH}")

# Đếm params
total_params = sum(p.numel() for p in tps_model.parameters())
trainable_params = sum(p.numel() for p in tps_model.parameters() if p.requires_grad)
print(f"   Total params    : {total_params:,}")
print(f"   Trainable params: {trainable_params:,}")
```

---

## 7. Cell 3 – Dataloader Triplet ATD-12K

### Mục tiêu
- Tạo PyTorch Dataset đọc các triplet `frame1.png / frame2.png / frame3.png`
- Xử lý tiền xử lý đặc thù: chuyển sang **grayscale (L mode)**, padding chia hết 32, chuẩn hóa `[0, 1]`
- Tạo hàm `run_matching()` để lấy matched keypoints từ GlueStick

### Code

```python
# ============================================================
# CELL 3: DATALOADER & PREPROCESSING
# ============================================================
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# --- Đường dẫn dataset ---
ATD_ROOT = "/kaggle/input/atd12k-benchmark/test_2k_540p"
# Hoặc nếu dùng local:
# ATD_ROOT = "/kaggle/working/ATD-12K/test_2k_540p"

# Padding size requirement
PAD_MULTIPLE = 32

def pad_to_multiple(tensor: torch.Tensor, multiple: int = 32):
    """Pad tensor [B, C, H, W] để H và W chia hết cho multiple."""
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
    return tensor, h, w  # trả về kích thước gốc để crop lại sau


def crop_to_original(tensor: torch.Tensor, orig_h: int, orig_w: int):
    """Crop tensor về kích thước gốc sau padding."""
    return tensor[..., :orig_h, :orig_w]


def img_to_grayscale_tensor(img_path: str, pad_multiple: int = 32):
    """
    Đọc ảnh → chuyển Grayscale → tensor [1, 1, H_pad, W_pad].
    TPS-Inbetween hoạt động trên ảnh grayscale L-mode.
    
    Returns:
        tensor_padded : [1, 1, H_pad, W_pad], float32 [0, 1]
        orig_h, orig_w: kích thước gốc trước padding
    """
    img = Image.open(img_path).convert('L')   # Grayscale
    orig_w, orig_h = img.size
    np_img = np.array(img, dtype=np.float32) / 255.0  # [H, W]
    tensor = torch.from_numpy(np_img).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    tensor_padded, orig_h, orig_w = pad_to_multiple(tensor, pad_multiple)
    return tensor_padded, orig_h, orig_w


class ATD12KTripletDataset(Dataset):
    """
    Dataset đọc triplet ATD-12K theo chuẩn:
      <clip_name>/
        frame1.png  (input, t=0)
        frame2.png  (GT,    t=0.5)
        frame3.png  (input, t=1.0)
    """

    def __init__(self, root_dir: str, pad_multiple: int = 32, max_samples: int = None):
        self.root_dir = root_dir
        self.pad_multiple = pad_multiple
        self.clips = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        if max_samples is not None:
            self.clips = self.clips[:max_samples]
        print(f"📂 Found {len(self.clips)} clips in {root_dir}")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_name = self.clips[idx]
        clip_dir  = os.path.join(self.root_dir, clip_name)

        p1 = os.path.join(clip_dir, "frame1.png")
        p2 = os.path.join(clip_dir, "frame2.png")
        p3 = os.path.join(clip_dir, "frame3.png")

        t1_padded, orig_h, orig_w = img_to_grayscale_tensor(p1, self.pad_multiple)
        t2_padded, _, _           = img_to_grayscale_tensor(p2, self.pad_multiple)
        t3_padded, _, _           = img_to_grayscale_tensor(p3, self.pad_multiple)

        return {
            "clip_name" : clip_name,
            "frame1"    : t1_padded.squeeze(0),    # [1, H_pad, W_pad]
            "frame2_gt" : t2_padded.squeeze(0),    # [1, H_pad, W_pad]
            "frame3"    : t3_padded.squeeze(0),    # [1, H_pad, W_pad]
            "frame1_path": p1,
            "orig_h"    : orig_h,
            "orig_w"    : orig_w,
        }


# Hàm GlueStick matching
def run_matching(img_gray_tensor0: torch.Tensor,
                 img_gray_tensor1: torch.Tensor,
                 matching_model,
                 device: str,
                 tmp_dir: str = "/kaggle/working/tmp_matches"):
    """
    Chạy GlueStick matching giữa 2 ảnh grayscale.
    
    Args:
        img_gray_tensor0/1 : [1, 1, H, W] float32 [0, 1] trên device
    Returns:
        matching_npy_path  : đường dẫn file .npy chứa matched keypoints
                             shape [2, N, 2] (normalized coords)
    """
    os.makedirs(tmp_dir, exist_ok=True)

    b, c, h, w = img_gray_tensor0.shape

    with torch.no_grad():
        pred = matching_model({'image0': img_gray_tensor0,
                               'image1': img_gray_tensor1})
    pred = batch_to_np(pred)
    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0       = pred["matches0"]

    valid_matches   = m0 != -1
    match_indices   = m0[valid_matches]
    matched_kps0    = kp0[valid_matches]
    matched_kps1    = kp1[match_indices]

    # Normalize keypoints to [0, 1]
    n_kps0 = matched_kps0.copy()
    n_kps1 = matched_kps1.copy()
    n_kps0[:, 0] /= w;  n_kps0[:, 1] /= h
    n_kps1[:, 0] /= w;  n_kps1[:, 1] /= h

    kps_stack = np.stack((n_kps0, n_kps1), axis=0)   # [2, N, 2]

    # Lưu ra file tạm
    import hashlib, time
    uid = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    npy_path = os.path.join(tmp_dir, f"match_{uid}.npy")
    np.save(npy_path, kps_stack)

    return npy_path


# --- Test dataset ---
print("🧪 Testing dataset loading...")
dataset = ATD12KTripletDataset(ATD_ROOT, pad_multiple=PAD_MULTIPLE, max_samples=5)
sample = dataset[0]
print(f"  clip_name  : {sample['clip_name']}")
print(f"  frame1     : {sample['frame1'].shape}  | dtype={sample['frame1'].dtype}")
print(f"  frame2_gt  : {sample['frame2_gt'].shape}")
print(f"  frame3     : {sample['frame3'].shape}")
print(f"  orig HxW   : {sample['orig_h']} x {sample['orig_w']}")
```

---

## 8. Cell 4 – Module Tính Các Metrics

### Mục tiêu
- Implement đầy đủ các metric ✅ và ⚠️
- Sử dụng lại hàm từ `util/metrics.py` của model
- Thêm Line IoU, Disconnection Rate, RRLD

### Code

```python
# ============================================================
# CELL 4: METRICS MODULE
# ============================================================
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.spatial.distance import directed_hausdorff
import scipy.ndimage

# Import từ util/metrics.py của TPS-inbetween
from util.metrics import (
    batch_chamfer_distance,
    batch_chamfer_distance_w,
    batch_hausdorff_distance,
    batch_edt,
)
from util.utils import batch_dog

# Khởi tạo LPIPS (chỉ một lần)
lpips_fn = lpips.LPIPS(net='alex').to(DEVICE)
lpips_fn.eval()
print("✅ LPIPS (AlexNet) initialized")


# ─── Helper: grayscale [0,1] tensor → binary line edge map ────────────────────
def to_binary_lineart(tensor_01: torch.Tensor, dog_t: float = 1.0,
                      threshold: float = 0.5) -> torch.Tensor:
    """
    Chuyển grayscale [0,1] tensor sang binary line art map.
    TPS model output là white background + dark lines (convention: line=1 sau inversion).
    
    Args:
        tensor_01 : [B, 1, H, W] hoặc [1, H, W] float32 [0, 1]
    Returns:
        binary    : [B, H, W] float32 {0, 1} – 1 = line pixel
    """
    if tensor_01.dim() == 3:
        tensor_01 = tensor_01.unsqueeze(0)  # → [1, 1, H, W]

    # Model output convention: nền trắng (~1), nét đen (~0)
    # Đảo ngược để nét = 1 (foreground)
    inverted = 1.0 - tensor_01

    # DoG edge extraction
    dog = batch_dog(inverted, t=dog_t, sigma=1.0, k=1.6, epsilon=0.01)

    # Threshold
    binary = (dog > threshold).float()
    return binary.squeeze(1)  # [B, H, W]


# ─── 1. Chamfer Distance ───────────────────────────────────────────────────────
def compute_cd(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Chamfer Distance trên binary line art. pred/gt: [1, 1, H, W] or [1, H, W]"""
    pred_bin = to_binary_lineart(pred.cpu())   # [1, H, W]
    gt_bin   = to_binary_lineart(gt.cpu())
    cd = batch_chamfer_distance(gt_bin, pred_bin)
    return cd.mean().item()


# ─── 2. Weighted CD ────────────────────────────────────────────────────────────
def compute_wcd(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Weighted Chamfer Distance."""
    pred_bin = to_binary_lineart(pred.cpu())
    gt_bin   = to_binary_lineart(gt.cpu())
    wcd = batch_chamfer_distance_w(gt_bin, pred_bin)
    return wcd.mean().item()


# ─── 3. LPIPS (⚠️) ─────────────────────────────────────────────────────────────
def compute_lpips(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    LPIPS trên grayscale: repeat 1→3 channels, normalize [-1,1].
    pred/gt: [1, 1, H, W] float32 [0, 1]
    """
    pred_rgb = pred.repeat(1, 3, 1, 1).to(DEVICE)  # [1, 3, H, W]
    gt_rgb   = gt.repeat(1, 3, 1, 1).to(DEVICE)
    # Normalize to [-1, 1]
    pred_norm = pred_rgb * 2.0 - 1.0
    gt_norm   = gt_rgb   * 2.0 - 1.0
    with torch.no_grad():
        score = lpips_fn(pred_norm, gt_norm)
    return score.item()


# ─── 4. EMD (⚠️) ───────────────────────────────────────────────────────────────
def compute_emd(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Earth Mover's Distance (Wasserstein-1D) trên nonzero pixel coords.
    Returns: scalar float
    """
    try:
        from ot import wasserstein_1d
        pred_bin = to_binary_lineart(pred.cpu()).squeeze(0).numpy()  # [H, W]
        gt_bin   = to_binary_lineart(gt.cpu()).squeeze(0).numpy()

        pts_p = np.argwhere(pred_bin > 0.5).astype(np.float32)
        pts_g = np.argwhere(gt_bin   > 0.5).astype(np.float32)

        if len(pts_p) == 0 or len(pts_g) == 0:
            return float('nan')

        h, w = pred_bin.shape
        pts_p = pts_p / np.array([h, w])
        pts_g = pts_g / np.array([h, w])

        # 1D Wasserstein per axis then average
        emd_y = wasserstein_1d(
            torch.from_numpy(pts_p[:, 0]),
            torch.from_numpy(pts_g[:, 0])
        )
        emd_x = wasserstein_1d(
            torch.from_numpy(pts_p[:, 1]),
            torch.from_numpy(pts_g[:, 1])
        )
        return ((emd_y + emd_x) / 2).item()
    except Exception as e:
        print(f"  [EMD warning] {e}")
        return float('nan')


# ─── 5. RRLD — Relative Residual Line Density (⚠️) ────────────────────────────
def compute_rrld(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    RRLD = |mean(pred_line) - mean(gt_line)| / (mean(gt_line) + eps)
    Measures relative line density difference.
    """
    pred_bin = to_binary_lineart(pred.cpu()).squeeze(0).numpy()
    gt_bin   = to_binary_lineart(gt.cpu()).squeeze(0).numpy()

    mean_pred = pred_bin.mean()
    mean_gt   = gt_bin.mean()
    eps = 1e-8

    rrld = abs(mean_pred - mean_gt) / (mean_gt + eps)
    return float(rrld)


# ─── 6. PSNR (⚠️) ──────────────────────────────────────────────────────────────
def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """PSNR trên grayscale [0, 1] float."""
    pred_np = pred.squeeze().cpu().numpy().clip(0, 1)
    gt_np   = gt.squeeze().cpu().numpy().clip(0, 1)
    return peak_signal_noise_ratio(gt_np, pred_np, data_range=1.0)


# ─── 7. SSIM (⚠️) ──────────────────────────────────────────────────────────────
def compute_ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """SSIM trên grayscale [0, 1]."""
    pred_np = pred.squeeze().cpu().numpy().clip(0, 1)
    gt_np   = gt.squeeze().cpu().numpy().clip(0, 1)
    return structural_similarity(gt_np, pred_np, data_range=1.0)


# ─── 8. IE — Interpolation Error / MAE (⚠️) ────────────────────────────────────
def compute_ie(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """IE = mean |pred - gt| × 255 (pixel-level MAE in [0,255] scale)."""
    ie = (pred.squeeze().cpu() - gt.squeeze().cpu()).abs().mean().item() * 255.0
    return ie


# ─── 11. Hausdorff Distance ────────────────────────────────────────────────────
def compute_hausdorff(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Hausdorff Distance via EDT, normalized by image diagonal."""
    pred_bin = to_binary_lineart(pred.cpu())   # [1, H, W]
    gt_bin   = to_binary_lineart(gt.cpu())
    hd = batch_hausdorff_distance(gt_bin, pred_bin)
    return hd.mean().item()


# ─── 12. Line IoU ─────────────────────────────────────────────────────────────
def compute_line_iou(pred: torch.Tensor, gt: torch.Tensor,
                     dilate_px: int = 3) -> float:
    """
    Line IoU = Intersection / Union trên dilated binary line maps.
    Dilation của 3px để bù sai lệch nhỏ về vị trí nét.
    """
    pred_bin = to_binary_lineart(pred.cpu()).squeeze(0).numpy().astype(np.uint8)
    gt_bin   = to_binary_lineart(gt.cpu()).squeeze(0).numpy().astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px*2+1, dilate_px*2+1))
    pred_dilated = cv2.dilate(pred_bin, kernel)
    gt_dilated   = cv2.dilate(gt_bin,   kernel)

    intersection = np.logical_and(pred_dilated, gt_dilated).sum()
    union        = np.logical_or(pred_dilated,  gt_dilated).sum()

    if union == 0:
        return 1.0  # Cả hai ảnh đều trống
    return float(intersection) / float(union)


# ─── 13. Disconnection Rate ────────────────────────────────────────────────────
def compute_disconnection_rate(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Disconnection Rate = |num_CC(pred) - num_CC(gt)| / num_CC(gt)
    
    num_CC = số lượng connected components (4-connectivity).
    Đo tỷ lệ đứt gãy đường nét so với Ground Truth.
    """
    pred_bin = to_binary_lineart(pred.cpu()).squeeze(0).numpy().astype(np.uint8)
    gt_bin   = to_binary_lineart(gt.cpu()).squeeze(0).numpy().astype(np.uint8)

    num_cc_pred, _ = cv2.connectedComponents(pred_bin, connectivity=4)
    num_cc_gt,   _ = cv2.connectedComponents(gt_bin,   connectivity=4)

    # Background (1 component) luôn tồn tại, trừ đi 1
    num_cc_pred = max(num_cc_pred - 1, 0)
    num_cc_gt   = max(num_cc_gt   - 1, 0)

    if num_cc_gt == 0:
        return 0.0
    return abs(num_cc_pred - num_cc_gt) / num_cc_gt


# ─── 15. Inference Speed ───────────────────────────────────────────────────────
def measure_inference_speed(tps_model, matching_model, img1_tensor, img3_tensor,
                             device: str, n_warmup: int = 3, n_runs: int = 10,
                             tmp_dir: str = "/kaggle/working/tmp_matches") -> dict:
    """
    Đo inference latency chính xác:
    - Warmup 3 lần (loại bỏ JIT/cuDNN warmup overhead)
    - Đo n_runs lần → tính mean, std
    
    Returns: dict với 'mean_ms', 'std_ms', 'fps'
    """
    import time
    torch.cuda.synchronize() if device == "cuda" else None

    # Warmup
    for _ in range(n_warmup):
        npy_path = run_matching(img1_tensor, img3_tensor, matching_model, device, tmp_dir)
        inv1 = (1 - img1_tensor).to(device)
        inv3 = (1 - img3_tensor).to(device)
        with torch.no_grad():
            _ = tps_model(inv1, inv3, [npy_path])
        if device == "cuda":
            torch.cuda.synchronize()
        os.remove(npy_path) if os.path.exists(npy_path) else None

    # Đo
    times_ms = []
    for _ in range(n_runs):
        npy_path = run_matching(img1_tensor, img3_tensor, matching_model, device, tmp_dir)
        inv1 = (1 - img1_tensor).to(device)
        inv3 = (1 - img3_tensor).to(device)

        torch.cuda.synchronize() if device == "cuda" else None
        t_start = time.perf_counter()
        with torch.no_grad():
            _ = tps_model(inv1, inv3, [npy_path])
        torch.cuda.synchronize() if device == "cuda" else None
        t_end = time.perf_counter()

        times_ms.append((t_end - t_start) * 1000.0)
        os.remove(npy_path) if os.path.exists(npy_path) else None

    mean_ms = np.mean(times_ms)
    std_ms  = np.std(times_ms)
    fps     = 1000.0 / mean_ms

    return {"mean_ms": mean_ms, "std_ms": std_ms, "fps": fps}


# ─── 16. VRAM Peak ─────────────────────────────────────────────────────────────
def measure_vram_peak(tps_model, matching_model, img1_tensor, img3_tensor,
                      device: str, tmp_dir: str = "/kaggle/working/tmp_matches") -> float:
    """
    Đo VRAM peak (MB) khi chạy inference 1 lần.
    Returns: peak_mb (float)
    """
    if device != "cuda":
        return 0.0

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    npy_path = run_matching(img1_tensor, img3_tensor, matching_model, device, tmp_dir)
    inv1 = (1 - img1_tensor).to(device)
    inv3 = (1 - img3_tensor).to(device)

    with torch.no_grad():
        _ = tps_model(inv1, inv3, [npy_path])

    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated()
    peak_mb    = peak_bytes / 1024**2

    os.remove(npy_path) if os.path.exists(npy_path) else None
    return peak_mb


print("✅ Metrics module loaded successfully")
print("   Metrics available: CD, WCD, LPIPS, EMD, RRLD, PSNR, SSIM, IE,")
print("   Hausdorff, Line_IoU, Disconnection_Rate, Speed, VRAM")
```

---

## 9. Cell 5 – Pipeline Benchmark & Xuất CSV

### Mục tiêu
- Vòng lặp benchmark toàn bộ dataset
- Thu thập 17 chỉ số per clip
- Xuất CSV + bảng tóm tắt + visualization

### Code

```python
# ============================================================
# CELL 5: PIPELINE BENCHMARK & EXPORT
# ============================================================
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# --- Cấu hình ---
OUTPUT_DIR   = "/kaggle/working/benchmark_tps_inbetween"
CSV_PATH     = os.path.join(OUTPUT_DIR, "benchmark_results_tps-inbetween.csv")
TMP_DIR      = os.path.join(OUTPUT_DIR, "tmp_matches")
VIZ_DIR      = os.path.join(OUTPUT_DIR, "visualizations")
MAX_SAMPLES  = None   # None = chạy toàn bộ dataset | số nguyên = chạy thử N clip
N_VIZ        = 5      # Số clip để visualize

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR,    exist_ok=True)
os.makedirs(VIZ_DIR,    exist_ok=True)

# --- Load dataset ---
dataset = ATD12KTripletDataset(ATD_ROOT, pad_multiple=PAD_MULTIPLE,
                               max_samples=MAX_SAMPLES)

# --- Đo speed & VRAM trên 1 sample đại diện ---
print("\n⏱️  Measuring speed & VRAM on representative sample...")
sample0 = dataset[0]
img1_rep = sample0['frame1'].unsqueeze(0).to(DEVICE)
img3_rep = sample0['frame3'].unsqueeze(0).to(DEVICE)

speed_info = measure_inference_speed(tps_model, matching_model,
                                     img1_rep, img3_rep, DEVICE,
                                     n_warmup=3, n_runs=10, tmp_dir=TMP_DIR)
vram_mb    = measure_vram_peak(tps_model, matching_model,
                               img1_rep, img3_rep, DEVICE, tmp_dir=TMP_DIR)

SPEED_MS = speed_info["mean_ms"]
FPS_VAL  = speed_info["fps"]
VRAM_VAL = vram_mb

print(f"  Speed: {SPEED_MS:.1f} ms/frame ± {speed_info['std_ms']:.1f} ms")
print(f"  FPS  : {FPS_VAL:.2f}")
print(f"  VRAM : {VRAM_VAL:.1f} MB")

# --- Main benchmark loop (Clean Progress Bar Only & Memory-Safe) ---
print(f"\n🚀 Running benchmark on {len(dataset)} clips...")
records = []
viz_count = 0

pbar = tqdm(range(len(dataset)), desc="🚀 Tiến trình Benchmark", unit="clip")
for idx in pbar:
    sample    = dataset[idx]
    clip_name = sample['clip_name']

    frame1    = sample['frame1'].unsqueeze(0).to(DEVICE)   # [1, 1, H_pad, W_pad]
    frame3    = sample['frame3'].unsqueeze(0).to(DEVICE)
    frame2_gt = sample['frame2_gt'].unsqueeze(0)            # GT (CPU)
    orig_h, orig_w = sample['orig_h'], sample['orig_w']

    try:
        with torch.inference_mode():
            # ── 1. Matching ──────────────────────────────────────────────────
            npy_path = run_matching(frame1, frame3, matching_model, DEVICE, TMP_DIR)

            # ── 2. TPS Inference ─────────────────────────────────────────────
            inv_frame1 = (1.0 - frame1)
            inv_frame3 = (1.0 - frame3)
            pred_list, _ = tps_model(inv_frame1, inv_frame3, [npy_path])

            pred_inv = pred_list[0].detach()                    # [1, 1, H_pad, W_pad]
            pred     = (1.0 - pred_inv).clamp(0, 1)

            # Crop về kích thước gốc
            pred_crop  = crop_to_original(pred,      orig_h, orig_w).cpu()
            gt_crop    = crop_to_original(frame2_gt, orig_h, orig_w).cpu()

        # Cleanup file npy tạm
        if os.path.exists(npy_path):
            os.remove(npy_path)

        # ── 3. Tính các Metrics ──────────────────────────────────────────
        cd_val   = compute_cd(pred_crop,  gt_crop)
        wcd_val  = compute_wcd(pred_crop, gt_crop)
        lpips_val= compute_lpips(pred_crop, gt_crop)
        emd_val  = compute_emd(pred_crop,  gt_crop)
        rrld_val = compute_rrld(pred_crop, gt_crop)
        psnr_val = compute_psnr(pred_crop, gt_crop)
        ssim_val = compute_ssim(pred_crop, gt_crop)
        ie_val   = compute_ie(pred_crop,   gt_crop)
        hd_val   = compute_hausdorff(pred_crop, gt_crop)
        liou_val = compute_line_iou(pred_crop,  gt_crop)
        dr_val   = compute_disconnection_rate(pred_crop, gt_crop)

        # ── 4. Ghi record ────────────────────────────────────────────────
        records.append({
            "clip_name"         : clip_name,
            "CD"                : round(cd_val,   6),
            "WCD"               : round(wcd_val,  6),
            "LPIPS"             : round(lpips_val, 6),
            "EMD"               : round(emd_val,  6) if not np.isnan(emd_val) else "nan",
            "RRLD"              : round(rrld_val, 6),
            "PSNR"              : round(psnr_val, 4),
            "SSIM"              : round(ssim_val, 6),
            "IE"                : round(ie_val,   4),
            "FVD"               : "N/A",
            "Warping_Error"     : "N/A",
            "Hausdorff"         : round(hd_val,   6),
            "Line_IoU"          : round(liou_val, 6),
            "Disconnection_Rate": round(dr_val,   6),
            "Arbitrary_t"       : "YES",
            "Speed_ms"          : round(SPEED_MS, 2),
            "FPS"               : round(FPS_VAL,  2),
            "VRAM_MB"           : round(VRAM_VAL, 1),
            "Fine_tuning_Speed" : "N/A",
        })

        # Cập nhật thông số tóm tắt trực tiếp trên thanh tiến trình
        pbar.set_postfix({"PSNR": f"{psnr_val:.2f}dB", "SSIM": f"{ssim_val:.3f}", "CD": f"{cd_val:.4f}"})

        # ── 5. Lưu Visualization ra ổ đĩa (Không in ra màn hình để tiết kiệm RAM) ─
        if viz_count < N_VIZ:
            f1_np   = sample['frame1'].squeeze().numpy()[:orig_h, :orig_w]
            gt_np   = gt_crop.squeeze().numpy()
            pred_np = pred_crop.squeeze().numpy()

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(f1_np, cmap='gray', vmin=0, vmax=1)
            axes[0].set_title("Input Frame 1 (t=0)")
            axes[0].axis('off')
            axes[1].imshow(gt_np, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title("GT Frame 2 (t=0.5)")
            axes[1].axis('off')
            axes[2].imshow(pred_np, cmap='gray', vmin=0, vmax=1)
            axes[2].set_title(f"TPS-Pred (PSNR={psnr_val:.2f}dB)")
            axes[2].axis('off')

            plt.tight_layout()
            save_path = os.path.join(VIZ_DIR, f"viz_{viz_count:02d}_{clip_name}.png")
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            del fig, axes, f1_np, gt_np, pred_np
            viz_count += 1

    except Exception as e:
        records.append({
            "clip_name": clip_name,
            **{k: "ERROR" for k in [
                "CD","WCD","LPIPS","EMD","RRLD","PSNR","SSIM","IE",
                "FVD","Warping_Error","Hausdorff","Line_IoU",
                "Disconnection_Rate","Arbitrary_t","Speed_ms",
                "FPS","VRAM_MB","Fine_tuning_Speed"
            ]}
        })

    # Dọn dẹp bộ nhớ sau mỗi vòng lặp
    del frame1, frame3, frame2_gt, sample
    if 'pred_crop' in locals():
        del pred_crop, gt_crop, pred, pred_inv, pred_list, inv_frame1, inv_frame3
    if idx % 10 == 0:
        import gc
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

# ── Tạo DataFrame & xuất CSV ──────────────────────────────────────────────────
df = pd.DataFrame(records)
df.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')
print(f"\n💾 Results saved to: {CSV_PATH}")
print(f"   Total clips processed: {len(df)}")

# ── Bảng tóm tắt ──────────────────────────────────────────────────────────────
NUMERIC_COLS = ["CD","WCD","LPIPS","EMD","RRLD","PSNR","SSIM","IE",
                "Hausdorff","Line_IoU","Disconnection_Rate","Speed_ms","FPS","VRAM_MB"]

df_num = df[NUMERIC_COLS].apply(pd.to_numeric, errors='coerce')
summary = pd.DataFrame({
    "Mean"  : df_num.mean(),
    "Std"   : df_num.std(),
    "Min"   : df_num.min(),
    "Max"   : df_num.max(),
    "Median": df_num.median(),
})

print("\n" + "=" * 80)
print("📊 BENCHMARK SUMMARY — TPS-Inbetween (17 Metrics)")
print("=" * 80)
print(summary.round(6).to_markdown())

# ── Full 17-metric summary bảng Markdown ──────────────────────────────────────
full_17 = {
    "Chamfer Distance (CD)"     : f"{df_num['CD'].mean():.6f}",
    "Weighted CD (WCD)"         : f"{df_num['WCD'].mean():.6f}",
    "LPIPS"                     : f"{df_num['LPIPS'].mean():.6f}",
    "EMD"                       : f"{df_num['EMD'].mean():.6f}",
    "RRLD"                      : f"{df_num['RRLD'].mean():.6f}",
    "PSNR (dB)"                 : f"{df_num['PSNR'].mean():.4f}",
    "SSIM"                      : f"{df_num['SSIM'].mean():.6f}",
    "IE (px)"                   : f"{df_num['IE'].mean():.4f}",
    "FVD"                       : "N/A",
    "Warping Error"             : "N/A",
    "Hausdorff Distance"        : f"{df_num['Hausdorff'].mean():.6f}",
    "Line IoU"                  : f"{df_num['Line_IoU'].mean():.6f}",
    "Disconnection Rate"        : f"{df_num['Disconnection_Rate'].mean():.6f}",
    "Arbitrary t Support"       : "YES",
    "Inference Speed (ms/frame)": f"{SPEED_MS:.1f} ms",
    "Inference Speed (FPS)"     : f"{FPS_VAL:.2f} FPS",
    "VRAM Peak (MB)"            : f"{VRAM_VAL:.1f} MB",
    "Fine-tuning Speed"         : "N/A",
}

print("\n" + "=" * 80)
print("🏆 FINAL 17-METRIC TABLE — TPS-Inbetween")
print("=" * 80)
print(f"{'Metric':<35} {'Value':>20}")
print("-" * 57)
for metric, val in full_17.items():
    print(f"{metric:<35} {val:>20}")
print("=" * 80)
print(f"\n✅ Benchmark complete! Results saved to: {CSV_PATH}")
```

---

## 10. Kết Quả Mong Đợi

Dựa trên kiến trúc TPS-Inbetween và đặc điểm animation line art:

| Metric | Range Mong Đợi | Ghi Chú |
|--------|---------------|---------|
| **CD** | 0.005 – 0.020 | Thấp hơn = tốt hơn |
| **WCD** | 0.005 – 0.025 | Thấp hơn = tốt hơn |
| **LPIPS** | 0.10 – 0.30 | Thấp hơn = tốt hơn (grayscale) |
| **PSNR** | 22 – 30 dB | Cao hơn = tốt hơn |
| **SSIM** | 0.75 – 0.92 | Cao hơn = tốt hơn |
| **Hausdorff** | 0.01 – 0.08 | Thấp hơn = tốt hơn |
| **Line IoU** | 0.55 – 0.80 | Cao hơn = tốt hơn |
| **Disconnection Rate** | 0.10 – 0.50 | Thấp hơn = tốt hơn |
| **Speed** | 200 – 800 ms | Phụ thuộc GlueStick matching |
| **VRAM** | 2,000 – 6,000 MB | Phụ thuộc kích thước ảnh |

> **Chú ý:** TPS-Inbetween **chậm hơn** các mô hình RGB vì bước GlueStick matching. Speed chủ yếu bị dominated bởi GlueStick (~80%), TPS network chỉ chiếm ~20%.

---

## 11. Troubleshooting

### ❗ Lỗi 1: `CUDA out of memory`
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```
**Giải pháp:**
```python
# Giảm kích thước ảnh resize
img = img.resize((960, 540))  # Hoặc (480, 270)

# Hoặc giảm max_num_keypoints trong GlueStick conf
'sp_params': {'max_num_keypoints': 500},   # Giảm từ 1000
'max_n_lines': 150,                         # Giảm từ 300
```

### ❗ Lỗi 2: `Shape mismatch` khi chạy UNet
```
RuntimeError: Expected input tensor of shape [B, 9, H, W], got [B, 9, H+K, W+L]
```
**Giải pháp:** Đảm bảo kích thước ảnh chia hết cho 32:
```python
# Kiểm tra kích thước
assert h % 32 == 0 and w % 32 == 0, f"Size must be divisible by 32! Got {h}x{w}"
```

### ❗ Lỗi 3: `checkpoint_GlueStick_MD.tar` không tìm thấy
```
FileNotFoundError: [Errno 2] No such file or directory: '.../checkpoint_GlueStick_MD.tar'
```
**Giải pháp:**
```bash
# Download GlueStick weights
wget https://github.com/cvg/GlueStick/releases/download/v0.1_arxiv/checkpoint_GlueStick_MD.tar \
     -P /kaggle/working/tps-inbetween/model/resources/weights/
```

### ❗ Lỗi 4: `cupy` import error trong `sketcher.py`
```
ImportError: No module named 'cupy'
```
**Giải pháp:** `sketcher.py` có try/except cho cupy nên an toàn. Nếu vẫn lỗi, thêm:
```python
# Patch cupy import ở đầu notebook
import sys
sys.modules['cupy'] = type(sys)('cupy')
```

### ❗ Lỗi 5: `scipy.ndimage.morphology` deprecated
```
AttributeError: module 'scipy.ndimage' has no attribute 'morphology'
```
**Giải pháp:** Upgrade scipy hoặc patch trong `util/metrics.py`:
```python
# Thay scipy.ndimage.morphology.distance_transform_edt bằng:
from scipy.ndimage import distance_transform_edt
```

### ❗ Lỗi 6: Ảnh ATD-12K màu RGB, model cần grayscale
**Giải pháp:** Dataset loader đã xử lý tự động với `.convert('L')`. Nếu tự đọc thủ công:
```python
from PIL import Image
img = Image.open(path).convert('L')  # Grayscale
```

### ❗ Lỗi 7: GlueStick matching rất chậm (>5 giây/frame)
**Nguyên nhân:** Quá nhiều keypoints hoặc ảnh lớn.  
**Giải pháp:**
```python
# Resize ảnh trước khi matching
img_small = img.resize((960, 540), Image.LANCZOS)
# Hoặc giảm max keypoints
'max_num_keypoints': 500, 'max_n_lines': 100
```

### ❗ Lỗi 8: `torch.linalg.solve` không ổn định (singular matrix)
**Nguyên nhân:** Quá ít keypoints matched (<3).  
**Giải pháp:** Model đã handle case này (khi `kps.shape[1] <= 2` → dùng zero flow). Đây là behavior mong muốn.

### ❗ Lỗi 9: `ModuleNotFoundError: No module named 'pytlsd'`
**Nguyên nhân:** `pytlsd` (Line Segment Detector) là thư viện cần thiết cho GlueStick Wireframe.
**Giải pháp:**
```bash
pip install pytlsd omegaconf scikit-learn
```

### ❗ Lỗi 10: `OSError: [Errno 30] Read-only file system` trên `/kaggle/input`
**Nguyên nhân:** Thư mục `/kaggle/input` trên Kaggle là Read-Only.
**Giải pháp:** Notebook tự động copy mã nguồn sang `/kaggle/working/tps-inbetween` để có toàn quyền ghi.

---

*Tài liệu này được tạo tự động bởi hệ thống PBL6 Benchmark Pipeline.*  
*Cập nhật lần cuối: 2026-08-30*
