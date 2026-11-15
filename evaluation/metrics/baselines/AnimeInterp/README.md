# 📁 Benchmark/AnimeInterp

**AnimeInterp (CVPR 2021)** — *Deep Animation Video Interpolation in the Wild*

## 📂 Nội Dung Thư Mục

| File | Mô Tả |
|------|-------|
| [`benchmark_AnimeInterp_guide.md`](./benchmark_AnimeInterp_guide.md) | 📖 Hướng dẫn chi tiết từng bước (setup, cells, troubleshooting) |
| [`benchmark_AnimeInterp.ipynb`](./benchmark_AnimeInterp.ipynb) | 🎯 Jupyter Notebook upload lên Kaggle để chạy trực tiếp |
| `benchmark_results_AnimeInterp.csv` | 📊 Sinh ra sau khi chạy notebook (17 chỉ số chuẩn hóa) |

## ⚡ Quick Start

1. **Upload lên Kaggle:**  
   - `benchmark_AnimeInterp.ipynb` → New Notebook → Upload  
   - Add Datasets: `AnimeInterp_code.zip` + `ATD-12K/test_2k_540p`  
   - Settings → Accelerator → **GPU T4 x2** hoặc **P100**

2. **Cập nhật 3 đường dẫn trong Cell 2A:**
   ```python
   ANIMEINTERP_ZIP  = "/kaggle/input/<tên-dataset>/AnimeInterp_code.zip"
   DATASET_ROOT     = "/kaggle/input/<tên-dataset>/test_2k_540p"
   CHECKPOINT       = "/kaggle/input/<tên-dataset>/anime_interp_full.ckpt"
   ```

3. **Run All** → Kết quả xuất ra `benchmark_results_AnimeInterp.csv`

## 📊 17 Chỉ Số Chuẩn Hóa

| Chỉ Số | Trạng Thái | Ghi Chú |
|--------|:----------:|---------|
| CD, WCD | ❌ | N/A — model RGB |
| **LPIPS** | ✅ | AlexNet backbone |
| EMD, RRLD | ❌ | N/A |
| **PSNR** | ✅ | skimage |
| **SSIM** | ✅ | multichannel, gaussian |
| **IE** | ⚠️ | L2 per-pixel MAE |
| FVD, Warping Error | ⚠️ | N/A (single-frame) |
| Hausdorff, Line IoU, Disconnection | ❌ | N/A |
| Arbitrary t | ❌ | Fixed t=0.5 |
| **Inference Speed** | ✅ | ms/frame & FPS |
| **VRAM Peak** | ✅ | torch.cuda.max_memory_allocated |
| Fine-tuning Speed | ⚠️ | N/A |

## 🎯 Kết Quả Mong Đợi

| Metric | Expected Range |
|--------|---------------|
| PSNR | 30–33 dB |
| SSIM | 0.92–0.96 |
| LPIPS | 0.03–0.08 |
| Infer Time (T4) | 400–600 ms/frame |
| VRAM (960×544) | 2000–5000 MB |

---
*Model: [lisiyao21/AnimeInterp](https://github.com/lisiyao21/AnimeInterp) | CVPR 2021*
