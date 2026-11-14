# Hướng Dẫn & Kế Hoạch Benchmark Các Model Frame Interpolation (TestModels)

Tài liệu này tổng hợp danh sách các mô hình, bảng chỉ số đánh giá chuẩn hóa từ [benchmark_metrics_analysis.md](file:///e:/CODE/PBL/PBL6/Test/benchmark_metrics_analysis.md), quy chuẩn cấu trúc dataset mẫu **ATD-12K**, và cung cấp **Prompt chuẩn** để người dùng sao chép và yêu cầu Agent tạo kịch bản/hướng dẫn benchmark chi tiết cho từng mô hình cụ thể trên môi trường Kaggle GPU.

---

## 1. Cấu Trúc Dataset Mẫu ATD-12K

Benchmark được thiết kế để chạy trước tiên trên dataset có cấu trúc theo chuẩn **ATD-12K** (Animation Triplet Dataset).

> [!NOTE]
> Thư mục `ATD-12K` hiện có trong thư mục `Test/` là **bản rút gọn (sample/mock dataset)** dùng để kiểm tra pipeline, test luồng đọc dữ liệu và debug code trước khi chạy toàn bộ trên Kaggle với dataset đầy đủ.

### 1.1 Cấu trúc thư mục mẫu

```text
ATD-12K/
├── test_2k_540p/                 # Tập test độ phân giải 540p (thường dùng nhất cho benchmark)
│   └── <clip_sequence_name>/     # Ví dụ: Japan_v2_3_172427_s2
│       ├── frame1.png            # Input frame 1 (t = 0.0)
│       ├── frame2.png            # Ground Truth target frame (t = 0.5)
│       └── frame3.png            # Input frame 2 (t = 1.0)
├── test_2k_original/             # Tập test độ phân giải gốc
│   └── <clip_sequence_name>/
│       ├── frame1.png
│       ├── frame2.png
│       └── frame3.png
├── test_2k_annotations/          # Annotation / metadata json (nếu cần dùng)
│   └── <clip_sequence_name>/
│       └── <clip_sequence_name>.json
├── test_2k_pre_calc_sgm_flows/   # Optical flow tính toán sẵn (Segment/Flow)
└── train_10k/                    # Tập huấn luyện mẫu (triplets tương tự)
```

### 1.2 Nguyên tắc vận hành Benchmark trên Triplet
1. **Input:** Cặp ảnh `frame1.png` ($t=0$) và `frame3.png` ($t=1$).
2. **Inference:** Model sinh ra frame trung gian $\hat{I}_{0.5}$ (tại $t=0.5$).
3. **Ground Truth (GT):** `frame2.png` ($t=0.5$).
4. **Evaluation:** So sánh $\hat{I}_{0.5}$ với `frame2.png` theo các chỉ số phù hợp với từng model.

*Lưu ý xử lý domain dữ liệu:*
- **Với model RGB (AnimeInterp, UPR-Net, uniVIP):** Đọc trực tiếp ảnh RGB từ dataset.
- **Với model Line Inbetweening (AnimeInbet, tps-inbetween):** Nếu benchmark trên domain line art thuần, cần chuyển đổi triplet sang dạng line art / edge map (hoặc sử dụng tập line art tương ứng), sau đó trích xuất point cloud / đường nét để tính Chamfer Distance, Hausdorff, Line IoU, Disconnection Rate.

---

## 2. Danh Sách Model & Bảng Tổng Hợp Chỉ Số Benchmark

Dưới đây là bảng đánh giá mức độ phù hợp của từng chỉ số đối với 5 model trong thư mục `TestModels/`, được trích xuất nguyên vẹn từ [benchmark_metrics_analysis.md](file:///e:/CODE/PBL/PBL6/Test/benchmark_metrics_analysis.md):

### 2.1 Bảng Tổng Hợp Toàn Bộ Chỉ Số

| Chỉ số | AnimeInbet | tps-inbetween | AnimeInterp | UPR-Net | uniVIP |
|---|:---:|:---:|:---:|:---:|:---:|
| **Chamfer Distance (CD)** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Weighted CD (WCD)** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **LPIPS** | ❌ | ⚠️ | ✅ | ✅ | ✅ |
| **EMD** | ⚠️ | ⚠️ | ❌ | ❌ | ❌ |
| **RRLD** | ⚠️ | ⚠️ | ❌ | ❌ | ❌ |
| **PSNR** | ❌ | ⚠️ | ✅ | ✅ | ✅ |
| **SSIM** | ❌ | ⚠️ | ✅ | ✅ | ✅ |
| **IE** | ❌ | ⚠️ | ⚠️ | ⚠️ | ⚠️ |
| **FVD** | ❌ | ❌ | ⚠️ | ✅ | ⚠️ |
| **Warping Error** | ❌ | ❌ | ⚠️ | ✅ | ⚠️ |
| **Hausdorff Distance** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Line IoU** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Disconnection Rate** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Arbitrary t Support** | ❌ | ✅ | ❌ | ✅ | ✅ |
| **Inference Speed** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **VRAM Peak** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Fine-tuning Speed** | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ |

> **Chú thích quy ước:**
> - ✅ **Phù hợp:** Bắt buộc/Khuyến nghị đo trực tiếp trong benchmark chính.
> - ⚠️ **Cần cân nhắc / Có điều kiện:** Chỉ đo khi có kịch bản mở rộng (ví dụ: tạo sequence video cho FVD, hoặc chạy test case chuyển đổi).
> - ❌ **Không phù hợp:** Không áp dụng (ví dụ: PSNR/SSIM cho output sparse skeleton của AnimeInbet, hoặc CD cho ảnh RGB).

---

## 3. Chi Tiết Toàn Bộ 17 Chỉ Số Cho Từng Model (Chuẩn Hóa Để So Sánh Chéo)

Để khi hoàn thành benchmark, tất cả mô hình đều có **đủ 17 cột chỉ số** trong bảng kết quả tổng hợp phục vụ so sánh chéo (các chỉ số không áp dụng sẽ được ghi nhận là `N/A` hoặc `None`), dưới đây là danh sách chi tiết 17 chỉ số cho từng model với đúng các đánh dấu phân loại:

### 3.1 Model: `AnimeInbet` (ICCV 2023 - Cartoon Line Inbetweening)
*Đặc điểm output:* Sparse skeleton (nét vẽ rời rạc ~34KB, không phải ảnh hoàn chỉnh).

| STT | Tên Chỉ Số | Trạng Thái | Hướng Dẫn Đo / Xử Lý |
|:---:|---|:---:|---|
| 1 | **Chamfer Distance (CD)** | ✅ | Đo trung bình khoảng cách nearest-neighbor giữa các điểm nét thưa |
| 2 | **Weighted CD (WCD)** | ✅ | Gán trọng số cao hơn cho các junction point / điểm nối |
| 3 | **LPIPS** | ❌ | Không áp dụng (`N/A`) — Deep feature pretrain không tối ưu cho sparse line art |
| 4 | **EMD (Earth Mover's Distance)** | ⚠️ | Đo khoảng cách phân phối Wasserstein (tính toán nặng $O(n^3)$) |
| 5 | **RRLD** | ⚠️ | Sai lệch tương đối trên restricted set nếu có định nghĩa hình học |
| 6 | **PSNR** | ❌ | Không áp dụng (`N/A`) — Output là sparse skeleton, PSNR vô nghĩa |
| 7 | **SSIM** | ❌ | Không áp dụng (`N/A`) — Cấu trúc pixel thưa, SSIM không tương thích |
| 8 | **IE (Interpolation Error)** | ❌ | Không áp dụng (`N/A`) — Không so sánh trực tiếp pixel-level |
| 9 | **FVD** | ❌ | Không áp dụng (`N/A`) — Domain nhị phân thưa không dùng I3D |
| 10 | **Warping Error** | ❌ | Không áp dụng (`N/A`) — Không phải sequence video RGB |
| 11 | **Hausdorff Distance** | ✅ | Đo khoảng cách sai lệch tối đa (worst-case) giữa các đường nét |
| 12 | **Line IoU** | ✅ | Tính IoU đường nét sau khi dilate kernel nhỏ ($d$ pixel) |
| 13 | **Disconnection Rate** | ✅ | Đo tỷ lệ đường nét bị đứt đoạn so với Ground Truth |
| 14 | **Arbitrary t Support** | ❌ | Không hỗ trợ (`N/A` hoặc cố định $t=0.5$) |
| 15 | **Inference Speed** | ✅ | Đo thời gian xử lý trung bình (ms/frame và FPS trên GPU) |
| 16 | **VRAM Peak** | ✅ | Đo dung lượng GPU tối đa tiêu thụ (`torch.cuda.max_memory_allocated()`) |
| 17 | **Fine-tuning Speed** | ⚠️ | Đo thời gian train/epoch nếu có kịch bản fine-tune |

---

### 3.2 Model: `tps-inbetween` (TPS-based Animation Inbetweening)
*Đặc điểm output:* Line art thuần (`No Color`), line art có bán màu (`Color`), hỗ trợ arbitrary $t$.

| STT | Tên Chỉ Số | Trạng Thái | Hướng Dẫn Đo / Xử Lý |
|:---:|---|:---:|---|
| 1 | **Chamfer Distance (CD)** | ✅ | Đo độ khớp point cloud đường nét |
| 2 | **Weighted CD (WCD)** | ✅ | Đo khoảng cách có trọng số trên các điểm cấu trúc |
| 3 | **LPIPS** | ⚠️ | Đo trên subfolder `Color` (bán màu) để tham khảo perceptual |
| 4 | **EMD** | ⚠️ | Đo chi phí vận chuyển phân phối điểm nét |
| 5 | **RRLD** | ⚠️ | Đo sai lệch tương đối hình học |
| 6 | **PSNR** | ⚠️ | Đo trên các ảnh subfolder có output pixel đầy đủ |
| 7 | **SSIM** | ⚠️ | Đo độ tương đồng cấu trúc trên subfolder có output đầy đủ |
| 8 | **IE (Interpolation Error)** | ⚠️ | Đo sai số pixel MAE/MSE trên ảnh đầy đủ |
| 9 | **FVD** | ❌ | Không áp dụng (`N/A`) |
| 10 | **Warping Error** | ❌ | Không áp dụng (`N/A`) |
| 11 | **Hausdorff Distance** | ✅ | Đo khoảng cách cực đại giữa các nét vẽ |
| 12 | **Line IoU** | ✅ | Tính tỷ lệ bao phủ nét vẽ (đặc biệt trên subfolder `No Color`) |
| 13 | **Disconnection Rate** | ✅ | Phát hiện đứt gãy đường nét sinh ra |
| 14 | **Arbitrary t Support** | ✅ | Hỗ trợ custom $t \in [0, 1]$ qua `parse_time_steps()` |
| 15 | **Inference Speed** | ✅ | Đo thời gian inference trung bình trên GPU |
| 16 | **VRAM Peak** | ✅ | Đo bộ nhớ GPU đỉnh |
| 17 | **Fine-tuning Speed** | ⚠️ | Đo thời gian huấn luyện bổ sung |

---

### 3.3 Model: `AnimeInterp` (CVPR 2021 - Animation Video Frame Interpolation)
*Đặc điểm output:* RGB anime đầy đủ màu sắc và line art có màu nhạt.

| STT | Tên Chỉ Số | Trạng Thái | Hướng Dẫn Đo / Xử Lý |
|:---:|---|:---:|---|
| 1 | **Chamfer Distance (CD)** | ❌ | Không áp dụng (`N/A`) — Model xử lý ảnh RGB dày đặc |
| 2 | **Weighted CD (WCD)** | ❌ | Không áp dụng (`N/A`) |
| 3 | **LPIPS** | ✅ | Đo perceptual similarity qua mạng VGG/AlexNet |
| 4 | **EMD** | ❌ | Không áp dụng (`N/A`) |
| 5 | **RRLD** | ❌ | Không áp dụng (`N/A`) |
| 6 | **PSNR** | ✅ | Đo tỷ lệ tín hiệu/nhiễu pixel RGB (dB) |
| 7 | **SSIM** | ✅ | Đo độ tương đồng cấu trúc, độ sáng và tương phản |
| 8 | **IE (Interpolation Error)** | ⚠️ | Đo MAE/MSE trực tiếp giữa pred và GT |
| 9 | **FVD** | ⚠️ | Đo khi tạo chuỗi nhiều frame liên tiếp (sequence generation) |
| 10 | **Warping Error** | ⚠️ | Đo flickering/temporal stability khi có sequence |
| 11 | **Hausdorff Distance** | ❌ | Không áp dụng (`N/A`) |
| 12 | **Line IoU** | ❌ | Không áp dụng (`N/A`) |
| 13 | **Disconnection Rate** | ❌ | Không áp dụng (`N/A`) |
| 14 | **Arbitrary t Support** | ❌ | Không hỗ trợ trực tiếp arbitrary $t$ (mặc định $t=0.5$) |
| 15 | **Inference Speed** | ✅ | Đo tốc độ nội suy (ms/frame, FPS) |
| 16 | **VRAM Peak** | ✅ | Đo dung lượng VRAM tiêu thụ |
| 17 | **Fine-tuning Speed** | ⚠️ | Đo thời gian huấn luyện |

---

### 3.4 Model: `UPR-Net` (CVPR 2023 - Multi-Scale Video Frame Interpolation)
*Đặc điểm output:* RGB anime/video chất lượng cao, có sẵn sequence video MP4.

| STT | Tên Chỉ Số | Trạng Thái | Hướng Dẫn Đo / Xử Lý |
|:---:|---|:---:|---|
| 1 | **Chamfer Distance (CD)** | ❌ | Không áp dụng (`N/A`) |
| 2 | **Weighted CD (WCD)** | ❌ | Không áp dụng (`N/A`) |
| 3 | **LPIPS** | ✅ | Đo sự khác biệt nhận thức sâu (deep perceptual loss) |
| 4 | **EMD** | ❌ | Không áp dụng (`N/A`) |
| 5 | **RRLD** | ❌ | Không áp dụng (`N/A`) |
| 6 | **PSNR** | ✅ | Chỉ số chuẩn tái tạo pixel RGB |
| 7 | **SSIM** | ✅ | Chỉ số cấu trúc hình ảnh |
| 8 | **IE (Interpolation Error)** | ⚠️ | Đo sai số nội suy pixel |
| 9 | **FVD** | ✅ | Áp dụng trực tiếp với sequence output / video MP4 |
| 10 | **Warping Error** | ✅ | Đo độ ổn định thời gian qua optical flow trên sequence |
| 11 | **Hausdorff Distance** | ❌ | Không áp dụng (`N/A`) |
| 12 | **Line IoU** | ❌ | Không áp dụng (`N/A`) |
| 13 | **Disconnection Rate** | ❌ | Không áp dụng (`N/A`) |
| 14 | **Arbitrary t Support** | ✅ | Hỗ trợ qua tham số `time_period` / arbitrary time step |
| 15 | **Inference Speed** | ✅ | Đo tốc độ xử lý GPU |
| 16 | **VRAM Peak** | ✅ | Đo dung lượng VRAM tiêu thụ |
| 17 | **Fine-tuning Speed** | ⚠️ | Đo thời gian fine-tuning |

---

### 3.5 Model: `uniVIP` (ICASSP 2025 - Arbitrary-Time Interpolation & Prediction)
*Đặc điểm output:* RGB anime/video, hỗ trợ thời gian liên tục $t \in [0, 1]$ và prediction $t > 1$.

| STT | Tên Chỉ Số | Trạng Thái | Hướng Dẫn Đo / Xử Lý |
|:---:|---|:---:|---|
| 1 | **Chamfer Distance (CD)** | ❌ | Không áp dụng (`N/A`) |
| 2 | **Weighted CD (WCD)** | ❌ | Không áp dụng (`N/A`) |
| 3 | **LPIPS** | ✅ | Đo perceptual similarity |
| 4 | **EMD** | ❌ | Không áp dụng (`N/A`) |
| 5 | **RRLD** | ❌ | Không áp dụng (`N/A`) |
| 6 | **PSNR** | ✅ | Đo chất lượng tái tạo RGB |
| 7 | **SSIM** | ✅ | Đo độ tương đồng cấu trúc RGB |
| 8 | **IE (Interpolation Error)** | ⚠️ | Đo sai số pixel |
| 9 | **FVD** | ⚠️ | Đo khi sinh sequence video mở rộng |
| 10 | **Warping Error** | ⚠️ | Đo độ ổn định thời gian trên chuỗi frame |
| 11 | **Hausdorff Distance** | ❌ | Không áp dụng (`N/A`) |
| 12 | **Line IoU** | ❌ | Không áp dụng (`N/A`) |
| 13 | **Disconnection Rate** | ❌ | Không áp dụng (`N/A`) |
| 14 | **Arbitrary t Support** | ✅ | Tính năng cốt lõi (hỗ trợ $t \in [0, 1]$ và cả $t > 1$) |
| 15 | **Inference Speed** | ✅ | Đo tốc độ xử lý GPU |
| 16 | **VRAM Peak** | ✅ | Đo đỉnh bộ nhớ VRAM |
| 17 | **Fine-tuning Speed** | ⚠️ | Đo thời gian huấn luyện |

---

## 4. PROMPT CHUẨN DÀNH CHO AGENT (Dùng Trên Kaggle)

> [!TIP]
> **Cách sử dụng:** Sao chép toàn bộ khối văn bản bên dưới, thay thế giá trị trong `<TÊN_MODEL>` (ví dụ: `UPR-Net`, `AnimeInterp`, `tps-inbetween`, `AnimeInbet`, hoặc `uniVIP`), sau đó gửi cho AI/Agent. Agent sẽ tự động tạo tài liệu hướng dẫn và file notebook `.ipynb` hoàn chỉnh được lưu trực tiếp vào thư mục `Benchmark/<TÊN_MODEL>/`.

````markdown
Bạn là chuyên gia về AI/Deep Learning trong lĩnh vực Video Frame Interpolation và Animation Inbetweening. 
Hãy tạo HƯỚNG DẪN CHI TIẾT, ĐƠN GIẢN VÀ FILE NOTEBOOK BENCHMARK HOÀN CHỈNH để tôi có thể chạy Benchmark mô hình <TÊN_MODEL> trên môi trường Kaggle Notebook (GPU T4 x2 hoặc P100).

### 1. BỐI CẢNH & TÀI NGUYÊN:
- Thư mục mô hình: Nằm trong `TestModels/<TÊN_MODEL>` (chứa mã nguồn inference/demo và pretrained weights).
- Dataset benchmark: Cấu trúc dạng triplet tương tự như ATD-12K (Animation Triplet Dataset):
  + Cấu trúc thư mục: `test_2k_540p/<clip_name>/frame1.png`, `frame2.png`, `frame3.png`.
  + Quy ước: `frame1.png` ($t=0$) và `frame3.png` ($t=1$) là 2 frame đầu vào; `frame2.png` ($t=0.5$) là Ground Truth.
- Môi trường chạy: Kaggle Notebook (Linux, Python 3.10+, PyTorch, CUDA GPU).
- **Vị trí lưu trữ đầu ra:** Toàn bộ tài liệu hướng dẫn và notebook benchmark cho mô hình này phải được lưu trong thư mục:
  `Benchmark/<TÊN_MODEL>/`
  ├── `README.md` (hoặc `benchmark_<TÊN_MODEL>_guide.md`): Hướng dẫn chi tiết từng bước.
  ├── `benchmark_<TÊN_MODEL>.ipynb`: File Jupyter Notebook hoàn chỉnh để upload lên Kaggle chạy trực tiếp.
  └── `benchmark_results_<TÊN_MODEL>.csv`: Nơi lưu trữ bảng kết quả chuẩn hóa sau khi chạy.

### 2. QUY CHUẨN 17 CHỈ SỐ BENCHMARK CHO <TÊN_MODEL>:
Để đảm bảo kết quả benchmark của tất cả các mô hình có thể ghép nối và so sánh chéo trực tiếp trên cùng một bảng dữ liệu chuẩn hóa gồm 17 chỉ số:
- Danh sách 17 chỉ số chuẩn của hệ thống:
  1. Chamfer Distance (CD)
  2. Weighted CD (WCD)
  3. LPIPS
  4. EMD
  5. RRLD
  6. PSNR
  7. SSIM
  8. IE (Interpolation Error)
  9. FVD
  10. Warping Error
  11. Hausdorff Distance
  12. Line IoU
  13. Disconnection Rate
  14. Arbitrary t Support
  15. Inference Speed (ms/frame & FPS)
  16. VRAM Peak (MB)
  17. Fine-tuning Speed
- **Nguyên tắc đo cho <TÊN_MODEL>:**
  + Chỉ số được đánh dấu ✅: Bắt buộc viết hàm đo tính toán thực tế trên tập dữ liệu.
  + Chỉ số được đánh dấu ⚠️: Triển khai đo nếu thỏa mãn điều kiện mở rộng (hoặc cấu hình tùy chọn).
  + Chỉ số được đánh dấu ❌: Ghi nhận giá trị là `N/A` hoặc `None` trong bảng kết quả.

### 3. NỘI DUNG HƯỚNG DẪN & NOTEBOOK CẦN CUNG CẤP:
Hãy tạo file notebook `.ipynb` và tài liệu markdown hướng dẫn với cấu trúc từng cell step-by-step rõ ràng (kèm code Python hoàn chỉnh có thể copy-paste chạy trực tiếp):

1. **Cell 1: Thiết lập môi trường & Cài đặt thư viện phụ thuộc:**
   - Cài đặt các package cần thiết (PyTorch phù hợp CUDA, lpips, opencv-python, scikit-image, scipy, timm, etc.).
   - Kiểm tra GPU khả dụng (`nvidia-smi`, `torch.cuda.get_device_name(0)`).

2. **Cell 2: Chuẩn bị Source Code & Pretrained Checkpoints:**
   - Hướng dẫn đưa code của `<TÊN_MODEL>` và file trọng số pretrained lên Kaggle (qua Kaggle Dataset hoặc Git/Drive/Wget).
   - Thiết lập `sys.path` và nạp model vào bộ nhớ GPU.

3. **Cell 3: Dataloader / Hàm đọc dữ liệu Triplet chuẩn ATD-12K:**
   - Viết hàm hoặc PyTorch Dataset để quét qua các folder triplet (`frame1`, `frame3` làm input; `frame2` làm GT).
   - Xử lý tiền xử lý / resize / chuyển đổi kiểu dữ liệu phù hợp với kiến trúc của `<TÊN_MODEL>`.

4. **Cell 4: Module tính toán các Metrics chuẩn:**
   - Viết code Python tính toán đầy đủ các metric ✅ và ⚠️ tương ứng của `<TÊN_MODEL>`.
   - Đo chính xác Inference Latency (loại bỏ GPU warmup) và `torch.cuda.max_memory_allocated()`.

5. **Cell 5: Pipeline Benchmark tự động & Xuất báo cáo 17 Chỉ Số:**
   - Chạy vòng lặp benchmark trên toàn bộ tập dữ liệu mẫu/đầy đủ.
   - Hiển thị thanh tiến trình (tqdm).
   - Tổng hợp điểm trung bình, min, max, std cho từng chỉ số.
   - Xuất kết quả chi tiết ra file `Benchmark/<TÊN_MODEL>/benchmark_results_<TÊN_MODEL>.csv` chứa đủ các cột tương ứng với 17 chỉ số (các chỉ số ❌ điền `N/A`) và in bảng tóm tắt định dạng Markdown/Pandas DataFrame.
   - Trực quan hóa (plot) một số frame kết quả dự đoán đối chiếu với Ground Truth.

6. **Lưu ý & Troubleshooting:**
   - Hướng dẫn xử lý các lỗi thường gặp (Out of Memory, lệch channel RGB/BGR, sai kích thước padding chia hết cho 16/32, lỗi phiên bản CUDA/PyTorch).

Hãy xuất cả nội dung hướng dẫn và tạo file `Benchmark/<TÊN_MODEL>/benchmark_<TÊN_MODEL>.ipynb` hoàn chỉnh trong workspace.
````
