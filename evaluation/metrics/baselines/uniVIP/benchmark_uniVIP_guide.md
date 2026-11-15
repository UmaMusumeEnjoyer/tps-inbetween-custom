# Hướng Dẫn Benchmark uniVIP trên Kaggle Notebook

> **Mô hình:** uniVIP — *Unified Arbitrary-Time Video Frame Interpolation and Prediction* (ICASSP 2025)  
> **Paper:** [arxiv.org/pdf/2503.02316](https://arxiv.org/pdf/2503.02316)  
> **Môi trường:** Kaggle Notebook · GPU T4 x2 hoặc P100 · Linux · Python 3.10+

---

## Mục Lục

1. [Tổng Quan Mô Hình](#1-tổng-quan-mô-hình)
2. [Bảng 17 Chỉ Số Benchmark](#2-bảng-17-chỉ-số-benchmark)
3. [Chuẩn Bị Trước Khi Chạy](#3-chuẩn-bị-trước-khi-chạy)
4. [Cell-by-Cell Hướng Dẫn Chi Tiết](#4-cell-by-cell-hướng-dẫn-chi-tiết)
5. [Cấu Trúc Output & Kết Quả](#5-cấu-trúc-output--kết-quả)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Tổng Quan Mô Hình

uniVIP là mô hình frame interpolation/prediction theo kiến trúc **UPR-Net** (forward-warping pyramid recurrent), mở rộng để hỗ trợ:
- **Arbitrary time t**: Nội suy tại bất kỳ thời điểm `t ∈ (0, 1)`.
- **Frame prediction**: Dự đoán frame tương lai với `t > 1`.
- Sử dụng **softmax splatting** (CuPy) cho forward warping.

### Biến thể model:
| Variant | Checkpoint | Tham số | VRAM (540p) |
|---------|-----------|---------|------------|
| uniVIP-B (Base) | `uniVIP-B.pkl` | ~6.3 MB | ~2.2 GB |
| uniVIP-L (Large) | `uniVIP-L.pkl` | ~25 MB | ~4.5 GB |

### Pipeline Inference:
```
img0 (BGR, [0,255]) → normalize [0,1] → pad(div64) → Model(img0, img1, t=0.5) → pred → crop → output (BGR, [0,1])
```

> **QUAN TRỌNG:** uniVIP đọc ảnh bằng `cv2.imread()` (BGR format). Khi tính metrics LPIPS và SSIM, phải chuyển sang **RGB**.

---

## 2. Bảng 17 Chỉ Số Benchmark

| STT | Chỉ Số | Trạng Thái | Ghi Chú |
|:---:|--------|:----------:|--------|
| 1 | Chamfer Distance (CD) | ❌ | N/A — model RGB dense |
| 2 | Weighted CD (WCD) | ❌ | N/A |
| 3 | LPIPS | ✅ | Đo perceptual similarity qua AlexNet |
| 4 | EMD | ❌ | N/A |
| 5 | RRLD | ❌ | N/A |
| 6 | PSNR | ✅ | Tỷ lệ tín hiệu/nhiễu pixel RGB (dB) |
| 7 | SSIM | ✅ | Độ tương đồng cấu trúc hình ảnh |
| 8 | IE (Interpolation Error) | ⚠️ | MAE pixel giữa pred và GT |
| 9 | FVD | ⚠️ | Cần sinh sequence video liên tiếp |
| 10 | Warping Error | ⚠️ | Cần chuỗi frame để đo temporal |
| 11 | Hausdorff Distance | ❌ | N/A |
| 12 | Line IoU | ❌ | N/A |
| 13 | Disconnection Rate | ❌ | N/A |
| 14 | Arbitrary t Support | ✅ | Tính năng cốt lõi của uniVIP |
| 15 | Inference Speed | ✅ | ms/frame và FPS trên GPU |
| 16 | VRAM Peak | ✅ | MB, đo bằng torch.cuda.max_memory_allocated() |
| 17 | Fine-tuning Speed | ⚠️ | s/epoch nếu chạy fine-tune |

**Chú thích:** ✅ = bắt buộc đo | ⚠️ = đo nếu có điều kiện | ❌ = ghi `N/A`

---

## 3. Chuẩn Bị Trước Khi Chạy

### 3.1 Upload lên Kaggle Dataset

Cần tạo **2 Kaggle Dataset** riêng biệt:

#### Dataset 1: `univip-model` (Source Code + Checkpoints)

Cấu trúc thư mục khi upload:
```
univip-model/
└── uniVIP/
    ├── core/
    │   ├── models/
    │   │   ├── softsplat/
    │   │   ├── univip_base.py
    │   │   └── univip_llarge.py
    │   ├── utils/
    │   ├── dataset.py
    │   ├── loss.py
    │   └── pipeline.py
    ├── demo/
    ├── tools/
    ├── checkpoints/
    │   ├── uniVIP-B.pkl    (<-- 6.3 MB)
    │   └── uniVIP-L.pkl    (<-- 25 MB)
    └── requirements.txt
```

**Cách tạo Kaggle Dataset:**
1. Nén thư mục `TestModels/uniVIP/` thành file zip
2. Vào https://www.kaggle.com/datasets/new
3. Upload file zip → đặt title `univip-model` → Publish (Public hoặc Private)
4. Ghi nhớ slug: `your-username/univip-model`

#### Dataset 2: `atd12k-benchmark` (Dữ liệu test)

```
atd12k-benchmark/
└── test_2k_540p/
    ├── clip_001/
    │   ├── frame1.png    (input, t=0)
    │   ├── frame2.png    (GT, t=0.5)
    │   └── frame3.png    (input, t=1)
    └── clip_002/
        └── ...
```

> **Mẹo:** Nếu chỉ muốn test nhanh, tạo 5-10 clip mẫu thay vì toàn bộ dataset để tiết kiệm thời gian upload.

### 3.2 Tạo Kaggle Notebook

1. Vào https://www.kaggle.com/code → tạo notebook mới
2. Click **"+ Add data"** → Thêm 2 dataset ở trên
3. Bật **Accelerator: GPU T4 x2** (hoặc P100)
4. Upload file `benchmark_uniVIP.ipynb` → Run all

### 3.3 Lưu ý về CuPy (BẮT BUỘC)

uniVIP sử dụng **CuPy** cho softmax-splatting forward warping.
Phải cài đúng phiên bản tương ứng với CUDA version:

```bash
# Kiểm tra CUDA version
nvcc --version

# CUDA 11.x → cài:
pip install cupy-cuda11x

# CUDA 12.x → cài:
pip install cupy-cuda12x

# CUDA 10.2 (cũ) → cài:
pip install cupy-cuda102==9.4.0
```

---

## 4. Cell-by-Cell Hướng Dẫn Chi Tiết

### CELL 1 — Thiết Lập Môi Trường

**Mục tiêu:** Kiểm tra GPU, cài đặt thư viện phụ thuộc.

```bash
# Check GPU
!nvidia-smi
!nvcc --version

# Cài thư viện
!pip install -q lpips scikit-image tqdm pandas matplotlib
!pip install -q cupy-cuda11x   # Thay theo CUDA version của bạn
```

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### CELL 2 — Nạp Source Code & Model

**Mục tiêu:** Thiết lập sys.path và load Pipeline từ checkpoint.

```python
import sys, os

MODEL_ROOT   = "/kaggle/input/univip-model/uniVIP"
DATASET_ROOT = "/kaggle/input/atd12k-benchmark/test_2k_540p"
OUTPUT_DIR   = "/kaggle/working/benchmark_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Thêm vào Python path
sys.path.insert(0, MODEL_ROOT)

import torch
from core.pipeline import Pipeline

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chọn model variant
USE_LARGE = False   # True để dùng uniVIP-L
variant   = "uniVIP-L" if USE_LARGE else "uniVIP-B"
ckpt_path = f"{MODEL_ROOT}/checkpoints/{variant}.pkl"

torch.set_grad_enabled(False)
torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True

ppl = Pipeline(dict(load_pretrain=True, large_model=USE_LARGE, model_file=ckpt_path))
ppl.eval()
print(f"✅ Loaded {variant}")
```

### CELL 3 — Dataloader ATD-12K

**Mục tiêu:** Đọc triplet dữ liệu, tiền xử lý đúng format uniVIP.

Xem chi tiết trong file notebook `benchmark_uniVIP.ipynb`.

Lưu ý quan trọng:
- `frame1.png` → img0 (t=0, input đầu)
- `frame3.png` → img1 (t=1, input cuối)
- `frame2.png` → GT (t=0.5, ground truth)
- Phải **pad** ảnh sao cho H và W chia hết cho **64** trước khi inference
- Model nhận ảnh **BGR float32 [0,1]**, output cũng là **BGR float32 [0,1]**

### CELL 4 — Tính Metrics

**Mục tiêu:** Tính toán tất cả 17 chỉ số theo quy chuẩn.

| Metric | Hàm | Ghi chú |
|--------|-----|---------|
| PSNR | `compute_psnr(pred, gt)` | Trả về dB, càng cao càng tốt |
| SSIM | `compute_ssim(pred, gt)` | Chuyển BGR→RGB trước khi tính |
| LPIPS | `compute_lpips(pred, gt, fn)` | Chuẩn hóa [-1,1], càng thấp càng tốt |
| IE (MAE/RMSE) | `compute_ie(pred, gt)` | Pixel-level error |
| Speed | `measure_inference_speed(ppl, ...)` | Warmup 5 lần, đo 20-30 lần |
| VRAM | `torch.cuda.max_memory_allocated()` | Đo sau mỗi inference |

### CELL 5 — Chạy Benchmark & Xuất CSV

**Mục tiêu:** Vòng lặp toàn bộ dataset, xuất kết quả 17 chỉ số.

Xem chi tiết trong file notebook. Output là:
- `benchmark_results_uniVIP.csv` — Bảng chi tiết mỗi clip
- `benchmark_uniVIP_visualization.png` — So sánh pred vs GT

---

## 5. Cấu Trúc Output & Kết Quả

### File CSV — 17 cột chuẩn hóa:

| Nhóm | Các cột |
|------|---------|
| Metadata | clip_name, model, resolution |
| N/A | CD, WCD, EMD, RRLD, Hausdorff_Distance, Line_IoU, Disconnection_Rate |
| Core (✅) | LPIPS, PSNR, SSIM |
| Optional (⚠️) | IE_MAE, IE_RMSE, FVD, Warping_Error |
| Special (✅) | Arbitrary_t_Support, Inference_ms, Inference_FPS, VRAM_Peak_MB |
| Fine-tune (⚠️) | Finetuning_Speed |

### Kết Quả Tham Khảo (uniVIP-B, ATD-12K 540p):

| Metric | Khoảng Điển Hình |
|--------|----------------|
| PSNR ↑ | 32–38 dB |
| SSIM ↑ | 0.92–0.97 |
| LPIPS ↓ | 0.04–0.12 |
| Speed | 80–150 ms/frame (T4) |
| VRAM Peak | 1,500–2,500 MB |

---

## 6. Troubleshooting

### Lỗi: `ImportError` khi import CuPy / softsplat

```
ImportError: cannot import name 'softsplat'
```
→ CuPy chưa được cài hoặc sai version CUDA. Cài lại: `pip install cupy-cuda11x`

---

### Lỗi: CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```
→ Giảm resolution: `img0 = F.interpolate(img0, scale_factor=0.5)`  
→ Dùng uniVIP-B thay vì uniVIP-L (`USE_LARGE = False`)  
→ Giảm `fixed_pyr_level` từ 5 xuống 3

---

### Lỗi: Màu sắc bị lệch (đỏ ↔ xanh)

→ OpenCV đọc BGR, LPIPS/SSIM cần RGB. Luôn đảo channel:
```python
tensor_rgb = tensor[[2, 1, 0], :, :]   # BGR → RGB
```

---

### Lỗi: `fixed_pyr_level` not recognized

→ Version cũ không hỗ trợ tham số này. Bỏ tham số:
```python
pred, _ = ppl.model(img0_padded, img1_padded, time_period=0.5)
```

---

### Lỗi: Kích thước không chia hết cho 64

```
RuntimeError: size mismatch
```
→ Luôn pad trước inference:
```python
h, w = img0.shape[-2:]
ph = ((h - 1) // 64 + 1) * 64
pw = ((w - 1) // 64 + 1) * 64
img0 = F.pad(img0, (0, pw-w, 0, ph-h), "constant", 0.5)
```

---

### Lỗi: `ModuleNotFoundError: No module named 'core'`

→ Kiểm tra sys.path:
```python
import sys
sys.path.insert(0, "/kaggle/input/univip-model/uniVIP")
import core  # Test import
```

---

## Tài Liệu Tham Khảo

- **Paper uniVIP:** https://arxiv.org/pdf/2503.02316
- **UPR-Net (base model):** https://github.com/srcn-ivl/UPR-Net
- **Softmax-Splatting:** https://github.com/sniklaus/softmax-splatting
- **LPIPS:** https://github.com/richzhang/PerceptualSimilarity
- **ATD-12K:** https://github.com/lisiyao21/AnimeInterp

