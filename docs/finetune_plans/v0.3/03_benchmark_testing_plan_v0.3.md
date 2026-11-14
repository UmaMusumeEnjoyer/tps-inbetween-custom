# KẾ HOẠCH ĐÁNH GIÁ ĐỐI CHUẨN & SO SÁNH ĐỐI ĐẦU PHIÊN BẢN v0.3 (BENCHMARK & EVALUATION PLAN v0.3)

> **Mô hình Trọng tâm:** TPS-Inbetween (*Thin-Plate Spline-based Interpolation for Animation Line Inbetweening*)  
> **Phiên bản Đánh giá:** v0.3 (Zero-Ghosting & High-Stability Cloud Training)  
> **Trọng số Checkpoint:** `checkpoints/experiments/v0.3/model_tps_v0.3_best.pt`  
> **Kế thừa & Nâng cấp từ:** [03_benchmark_testing_plan_v0.2.md](file:///e:/CODE/PBL/PBL6/docs/finetune_plans/v0.2/03_benchmark_testing_plan_v0.2.md)  
> **Ngày phê duyệt:** 07/10/2026  

---

## 1. MỤC TIÊU VÀ BỐI CẢNH ĐÁNH GIÁ (EVALUATION CONTEXT)

Sau khi phiên bản v0.2 đã khắc phục triệt để lỗi sụp đổ chế độ (Mode Collapse - ảnh trắng 100% của v0.1), kiểm toán thực nghiệm cho thấy v0.2 vẫn còn nhược điểm lớn: **hiện tượng bóng ma chồng hình (ghosting) xuất hiện tương đương với TPS Coarse Warp** do phép cộng tuyến tính cứng `(1-t)*I0 + t*I1` và giới hạn UNet $\pm 0.15$.

Mục tiêu cốt lõi của kế hoạch benchmark v0.3 là:
1. **Kiểm chứng khả năng triệt tiêu bóng ma (Ghosting Elimination):** Chứng minh định lượng và định tính rằng số lượng `Ghost Pixels` giảm sâu $\ge 50\%$ so với cả Coarse TPS lẫn Finetuned v0.2, đặc biệt tại các vùng chuyển động cơ thể lớn (`clip_000018`, `clip_000046`).
2. **Bảo toàn $100\%$ tính nguyên vẹn của nét vẽ (Anti-Collapse Invariance):** Giữ vững tỷ lệ sụp đổ trắng ở mức $0.0\%$, Stroke Recall duy trì $\ge 85.0\%$.
3. **Cải thiện độ chính xác pixel (PSNR & SSIM Gain):** Tăng trưởng PSNR thực tế ít nhất $+1.0$ dB so với Coarse TPS Warp (khắc phục điểm nghẽn $+0.02$ dB của v0.2).
4. **Đối chuẩn trên 5 Suite Kiểm Thử:** Đánh giá trên toàn bộ tập Validation (508 triplets) và tập dữ liệu thực tế `visual_samples` (`TestRandom Bamboo`, `TestImage1`, `TestImage2`, `ChayGo`).

---

## 2. DANH MỤC CÁC ĐỐI TƯỢNG SO SÁNH ĐỐI ĐẦU (HEAD-TO-HEAD CANDIDATES)

| Ký hiệu | Mô hình Đánh giá | Trọng số / Thuật toán | Nguồn gốc | Vai trò & Mục tiêu Đối chuẩn |
|:---|:---|:---|:---:|:---|
| **`M0: TPS-Original`** | TPS-Inbetween Gốc | `checkpoints/pretrained_baselines/tps-original/model_latest.pt` | Mixamo 3D | Baseline gốc chưa finetune |
| **`M1: TPS-AlgoFix`** | TPS Sửa Thuật Toán | Trọng số gốc + $\lambda=0.2$, Border/Interior Anchors | Nhóm nghiên cứu | Đối chứng thuật toán hình học |
| **`M2: TPS-Finetuned-v0.1`** | TPS Fine-tuned v0.1 | `checkpoints/experiments/v0.1/model_tps_v0.1_best.pt` | Kaggle (Revoked) | Đối chứng sụp đổ chế độ (Mode Collapse) |
| **`M3: TPS-Finetuned-v0.2`** | TPS Fine-tuned v0.2 | `checkpoints/experiments/v0.2/model_tps_v0.2_best.pt` | Kaggle v0.2 | Đối chứng bảo toàn nét nhưng còn ghosting |
| **`M4: TPS-Finetuned-v0.3`** | **TPS Fine-tuned v0.3** | `checkpoints/experiments/v0.3/model_tps_v0.3_best.pt` | **Kaggle v0.3** | **Ứng viên sản phẩm chính thức (Target)** |
| **`B1: AnimeInbet`** | AnimeInbet (ICCV 2023) | Mạng vector hóa hình học nét vẽ hoạt hình | Baseline Cloned | Đối chuẩn SOTA chuyên Line Inbetweening |
| **`B2: SAIN`** | SAIN (ArXiv 2023) | Sketch-Aware Interpolation Network | Baseline Cloned | Đối chuẩn SOTA chuyên Sketch Inbetweening |
| **`B3: UPR-Net`** | UPR-Net (CVPR 2023) | Kim tự tháp hồi quy đa tầng | Baseline Cloned | SOTA Video Frame Interpolation |

---

## 3. HỆ THỐNG CÁC CHỈ SỐ KIỂM TOÁN CHÍNH THỨC (METRICS SYSTEM)

| Nhóm Chỉ số | Tên Chỉ số | Đơn vị / Khoảng | Ý nghĩa Kỹ thuật | Xu hướng | Ngưỡng Nghiệm Thu v0.3 |
|:---|:---|:---:|:---|:---:|:---:|
| **Khử Bóng Ma** | **`Ghost Pixels`** | Số lượng pixel | Số pixel xám mờ ($180 \le p \le 250$) nằm ngoài nét GT | **Càng thấp càng tốt** | **$\le 7,000$ px** (Giảm $\ge 50\%$ vs v0.2) |
| **Độ phủ nét vẽ** | **`Stroke Recall`** | $[0, 100\%]$ | Tỷ lệ nét vẽ Ground Truth được tái tạo trong bán kính 5px | **Càng cao càng tốt** | **$\ge 85.0\%$** |
| **Bảo toàn mực** | **`Density Ratio`** | $[0, \infty)$ | Tỷ lệ tổng số pixel nét vẽ dự đoán chia cho Ground Truth | **Gần 1.0 càng tốt** | **$0.95 \le \text{Ratio} \le 1.15$** |
| **Độ chính xác ảnh** | **`PSNR`** | dB | Tỷ số tín hiệu cực đại trên nhiễu | **Càng cao càng tốt** | **$\ge 22.0$ dB** ($+1.0$ dB vs Coarse) |
| **Độ tương đồng** | **`SSIM`** | $[0, 1]$ | Độ tương đồng cấu trúc đường nét | **Càng cao càng tốt** | **$\ge 0.915$** |
| **Khoảng cách nét**| **`Chamfer Dist`** | Khoảng cách | Khoảng cách trung bình giữa các đường biên nét vẽ | **Càng thấp càng tốt** | **$\le 0.0014$** |
| **Phòng chống sụp đổ**| **`Blank Rate`** | $\%$ | Tỷ lệ khung hình sinh ra bị trắng tinh | **Bắt buộc 0.0%** | **$0.0\%$ (Tuyệt đối không có)** |

---

## 4. QUY TRÌNH BENCHMARK TỰ ĐỘNG BẰNG SCRIPT

Script benchmark sẽ được triển khai tự động qua lệnh:
```bash
python continuous_benchmark/inference_scripts/run_tps_benchmark_v03.py \
    --model_path checkpoints/experiments/v0.3/model_tps_v0.3_best.pt \
    --data_root data/processed/anime_tps_finetune_dataset_ready \
    --visual_samples_dir data/test_sets/visual_samples \
    --output_dir continuous_benchmark/metrics/tps-inbetween-finetuned-v0.3 \
    --visual_dir continuous_benchmark/visual_audit/finetune_v0.3 \
    --audit_clips clip_000001 clip_000018 clip_000031 clip_000046 clip_000053 clip_000064 clip_000077 clip_000089 clip_000110
```

### Các Sản Phẩm Đầu Ra Kiểm Toán Cần Đạt:
1. `val_508_metrics.csv`: Bảng số liệu chi tiết toàn bộ 508 cặp khung hình validation.
2. `summary_metrics.json`: Tổng hợp so sánh đối đầu trực tiếp giữa Finetuned v0.3 vs Finetuned v0.2 vs TPS Coarse.
3. `side_by_side_comparison/`: 10 ảnh so sánh 4 cột trực quan chứng minh nét vẽ sạch bóng ma.
4. `difference_heatmaps/`: Bản đồ nhiệt chứng minh sai số tại các vùng cơ thể chuyển động đã thu hẹp đáng kể.
5. `zoomed_details/`: Cắt phóng to $400\%$ các chi tiết mắt, tóc, nếp gấp áo.
