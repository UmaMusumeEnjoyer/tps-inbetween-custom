# Hướng Dẫn Benchmark UPR-Net trên Kaggle GPU

> **Model:** UPR-Net (CVPR 2023) — Unified Pyramid-based Recurrent Network for Video Frame Interpolation  
> **Paper:** *A Unified Pyramid Recurrent Network for Video Frame Interpolation*  
> **Output type:** RGB anime/video chất lượng cao, hỗ trợ arbitrary t ∈ (0, 1)  
> **Checkpoints:** `upr-base.pkl` (~6.3 MB) | `upr-large.pkl` (~14.1 MB) | `upr-llarge.pkl` (~25.1 MB)

---

## 📋 Bảng Chỉ Số Benchmark (17 Metrics)

| STT | Tên Chỉ Số | Trạng Thái | Ghi Chú |
|:---:|---|:---:|---|
| 1 | Chamfer Distance (CD) | ❌ | `N/A` — Model xử lý ảnh RGB dày đặc |
| 2 | Weighted CD (WCD) | ❌ | `N/A` |
| 3 | **LPIPS** | ✅ | Đo perceptual similarity qua VGG/AlexNet |
| 4 | EMD | ❌ | `N/A` |
| 5 | RRLD | ❌ | `N/A` |
| 6 | **PSNR** | ✅ | Chỉ số tái tạo pixel RGB (dB) |
| 7 | **SSIM** | ✅ | Chỉ số cấu trúc hình ảnh |
| 8 | IE (Interpolation Error) | ⚠️ | Đo MAE pixel nếu muốn mở rộng |
| 9 | **FVD** | ✅ | Áp dụng với sequence video output |
| 10 | **Warping Error** | ✅ | Đo độ ổn định thời gian qua optical flow |
| 11 | Hausdorff Distance | ❌ | `N/A` |
| 12 | Line IoU | ❌ | `N/A` |
| 13 | Disconnection Rate | ❌ | `N/A` |
| 14 | **Arbitrary t Support** | ✅ | Tham số `time_period` tùy biến |
| 15 | **Inference Speed** | ✅ | ms/frame và FPS trên GPU |
| 16 | **VRAM Peak** | ✅ | `torch.cuda.max_memory_allocated()` |
| 17 | Fine-tuning Speed | ⚠️ | Đo thời gian train/epoch nếu cần |

> ✅ = Bắt buộc đo | ⚠️ = Tùy chọn mở rộng | ❌ = Ghi `N/A`

---

## 🚀 Hướng Dẫn Từng Bước (Kaggle Notebook)

### Bước 0: Chuẩn Bị Trước Khi Lên Kaggle

#### 0.1 Cấu Trúc Thư Mục Cần Upload

```
Kaggle Dataset 1: upr-net-source
└── UPR-Net/
    ├── core/
    │   ├── models/ (upr_base.py, upr_large.py, upr_llarge.py, softsplat/)
    │   ├── pipeline.py, dataset.py, loss.py
    │   └── utils/
    ├── demo/
    └── checkpoints/ (upr-base.pkl, upr-large.pkl, upr-llarge.pkl)

Kaggle Dataset 2: atd12k-benchmark
└── test_2k_540p/
    ├── clip_001/ (frame1.png, frame2.png, frame3.png)
    └── clip_002/ ...
```

#### 0.2 Cách Tạo Kaggle Dataset

1. Vào kaggle.com/datasets → **New Dataset**
2. Upload thư mục `TestModels/UPR-Net/` nén thành `.zip`
3. Upload thư mục `ATD-12K/test_2k_540p/` nén thành `.zip`

---

### Bước 1: Thiết Lập Môi Trường

```python
!nvidia-smi
!pip install -q lpips scikit-image scipy timm einops

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

---

### Bước 2: Load Model

```python
import sys, os, math
sys.path.insert(0, "/kaggle/input/upr-net-source/UPR-Net")
from core.pipeline import Pipeline

DEVICE = torch.device("cuda")
MODEL_SIZE = "large"
CKPT_PATH = "/kaggle/input/upr-net-source/UPR-Net/checkpoints/upr-large.pkl"
WIDTH = 960  # 540p
PYR_LEVEL = math.ceil(math.log2(WIDTH / 448) + 3)

ppl = Pipeline(dict(load_pretrain=True, model_size=MODEL_SIZE, model_file=CKPT_PATH))
ppl.eval()
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True
```

---

### Bước 3: Dataloader Chuẩn ATD-12K

Dataset quét toàn bộ clip folder có đủ `frame1.png`, `frame2.png`, `frame3.png`. Đọc ảnh BGR bằng OpenCV → chuyển RGB → padding chia hết cho `2^(pyr_level+1)`.

---

### Bước 4: Tính Metrics

- **PSNR/SSIM:** dùng `skimage.metrics`
- **LPIPS:** dùng `lpips.LPIPS(net='alex')`, chuẩn hóa về [-1, 1]
- **IE:** MAE pixel-level
- **Warping Error:** frame difference ước lượng
- **FVD:** cần sequence 16+ frames, xem Cell mở rộng
- **Inference Speed:** GPU warmup 3 lần → đo 10 lần → lấy mean
- **VRAM:** `torch.cuda.max_memory_allocated()`

---

### Bước 5: Pipeline & Xuất CSV

Chạy vòng lặp qua toàn bộ dataset, tính 17 metrics, lưu ra:
- `benchmark_results_UPR-Net.csv`
- Visualization ảnh pred vs GT

---

## ⚠️ Troubleshooting

| Lỗi | Nguyên Nhân | Giải Pháp |
|---|---|---|
| CUDA OOM | Ảnh quá lớn | Dùng model `base`, clear cache |
| BGR/RGB lệch | Quên `cvtColor` | Luôn `COLOR_BGR2RGB` trước khi tensor |
| Padding error | H/W không chia hết `divisor` | Dùng công thức padding chuẩn |
| ModuleNotFoundError | Thiếu `sys.path` | Insert path trước mọi import |
| Checkpoint load lỗi | CUDA map issue | Load với `map_location='cpu'` |

---

## 📝 Tài Liệu Tham Khảo

- **Paper:** UPR-Net CVPR 2023 — https://arxiv.org/abs/2211.03456
- **ATD-12K:** https://github.com/lisiyao21/AnimeInterp
- **LPIPS:** https://github.com/richzhang/PerceptualSimilarity

