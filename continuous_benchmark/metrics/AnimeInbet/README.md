# 🚀 AnimeInbet Benchmark Package (PBL6 System)

> **Mô hình:** AnimeInbet (ICCV 2023) — *Deep Geometrized Cartoon Line Inbetweening*  
> **Mục tiêu:** Benchmark hiệu năng và chất lượng hình học của AnimeInbet trên tập dữ liệu Triplet ATD-12K / Line Art trên môi trường **Kaggle GPU (T4 x2 / P100)** theo chuẩn hóa **17 chỉ số**.

---

## 📁 Cấu Trúc Thư Mục

```
Benchmark/AnimeInbet/
├── README.md                           ← File này (Tóm tắt nhanh quy trình)
├── benchmark_AnimeInbet_guide.md       ← Tài liệu hướng dẫn chi tiết từng bước & phân tích domain
├── benchmark_AnimeInbet.ipynb          ← Jupyter Notebook hoàn chỉnh để upload chạy trực tiếp trên Kaggle
└── benchmark_results_AnimeInbet.csv    ← Bảng kết quả chuẩn hóa 17 chỉ số (sinh ra sau khi chạy)
```

---

## 📊 Bảng 17 Chỉ Số Chuẩn Hóa Cho AnimeInbet

| STT | Chỉ Số | Trạng Thái | Mô Tả Thực Hiện |
|:---:|---|:---:|---|
| 1 | **Chamfer Distance (CD)** | ✅ | Đo khoảng cách trung bình đối xứng hai chiều giữa 2 tập điểm nét qua Euclidean Distance Transform (EDT) |
| 2 | **Weighted CD (WCD)** | ✅ | Gán trọng số cao hơn cho các điểm giao nhau (Junction/Corner points) trên nét vẽ |
| 3 | **LPIPS** | ❌ | `N/A` (AlexNet pretrain không tương thích cho nét vẽ nhị phân thưa) |
| 4 | **EMD (Earth Mover's Distance)** | ⚠️ | Đo khoảng cách Wasserstein-1D giữa phân phối tọa độ điểm nét |
| 5 | **RRLD** | ⚠️ | Đo sai lệch tỷ lệ mật độ nét vẽ tương đối giữa Pred và GT |
| 6 | **PSNR** | ❌ | `N/A` (Output là nét vẽ thưa, so sánh pixel nền trắng/đen không có ý nghĩa) |
| 7 | **SSIM** | ❌ | `N/A` (Không áp dụng cho đồ thị nét vẽ nhị phân thưa) |
| 8 | **IE (Interpolation Error)** | ❌ | `N/A` (Không áp dụng so sánh sai số pixel RGB dày đặc) |
| 9 | **FVD** | ❌ | `N/A` (I3D video metric không áp dụng cho domain nhị phân) |
| 10 | **Warping Error** | ❌ | `N/A` (Optical flow không áp dụng cho chuỗi nét vector rời rạc) |
| 11 | **Hausdorff Distance** | ✅ | Đo khoảng cách sai lệch cực đại (worst-case distance) phát hiện nét bị lệch xa |
| 12 | **Line IoU** | ✅ | Đo tỷ lệ diện tích giao trên hợp (IoU) giữa 2 bản vẽ nét sau khi giãn nở 3px |
| 13 | **Disconnection Rate** | ✅ | Đo tỷ lệ chênh lệch số lượng thành phần liên thông phát hiện đứt gãy nét |
| 14 | **Arbitrary t Support** | ❌ | `N/A` (Cố định $t=0.5$ theo chuẩn hóa đánh giá Triplet) |
| 15 | **Inference Speed** | ✅ | Đo thời gian xử lý trung bình (`ms/frame`) và `FPS` sau warmup GPU |
| 16 | **VRAM Peak** | ✅ | Đo bộ nhớ GPU đỉnh tiêu thụ (`MB`) qua `torch.cuda.max_memory_allocated()` |
| 17 | **Fine-tuning Speed** | ⚠️ | `N/A` (Không có vòng lặp huấn luyện trong inference benchmark) |

---

## ⚡ Hướng Dẫn Nhanh Chạy Trên Kaggle

1. **Chuẩn bị Dataset:** Nén thư mục `TestModels/AnimeInbet` thành file `.zip` và upload lên Kaggle Dataset (ví dụ: `animeinbet-code`).
2. **Tạo Notebook:** Tạo Notebook mới trên Kaggle, chọn **GPU T4 x2** hoặc **GPU P100**.
3. **Thêm Dữ liệu:** Gắn (Add Data) dataset `animeinbet-code` và `atd12k-benchmark` (hoặc tập mẫu rút gọn).
4. **Upload Notebook:** Tải file `benchmark_AnimeInbet.ipynb` lên Kaggle và chọn **Run All Cells**.
5. **Xem Kết Quả:** Bảng tổng hợp Markdown sẽ được in ra màn hình và file `benchmark_results_AnimeInbet.csv` sẽ được lưu tự động tại `/kaggle/working/`.

👉 *Xem chi tiết đầy đủ tại: [benchmark_AnimeInbet_guide.md](benchmark_AnimeInbet_guide.md)*
