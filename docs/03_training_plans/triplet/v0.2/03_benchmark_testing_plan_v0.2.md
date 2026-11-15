# KẾ HOẠCH ĐÁNH GIÁ ĐỐI CHUẨN & SO SÁNH ĐỐI ĐẦU PHIÊN BẢN v0.2 (BENCHMARK & EVALUATION PLAN v0.2)

> **Mô hình Trọng tâm:** TPS-Inbetween (*Thin-Plate Spline-based Interpolation for Animation Line Inbetweening*)  
> **Phiên bản Đánh giá:** v0.2 (Production-Grade Anti-Collapse Finetune)  
> **Trọng số Checkpoint:** `checkpoints/experiments/v0.2/model_tps_v0.2_best.pt`  
> **Kế thừa & Nâng cấp từ:** [03_benchmark_testing_plan_v0.1.md](file:///e:/CODE/PBL/PBL6/docs/finetune_plans/v0.1/03_benchmark_testing_plan_v0.1.md)  
> **Ngày phê duyệt:** 07/10/2026  

---

## 1. MỤC TIÊU VÀ BỐI CẢNH ĐÁNH GIÁ (EVALUATION CONTEXT)

Sau đợt hậu kiểm toán phiên bản v0.1 (phát hiện lỗi sụp đổ chế độ toàn bộ ảnh thành nền trắng tinh do 3 nguyên nhân kỹ thuật: lệch hệ tọa độ `backward_warp`, hàm mất mát EDT Loss đơn hướng, và UNet residual Sigmoid âm cực đại), phiên bản v0.2 đã được tái cấu trúc toàn diện:
1. **Dual-Coordinate Warp Layer:** Tách biệt tuyệt đối tọa độ chuẩn hóa $[0, 1] \to [-1, 1]$ cho Coarse TPS Flow và tọa độ Pixel thực cho Fine IFNet Flow.
2. **Khống chế tuyệt đối Bounded Tanh Residual:** $res = \tanh(x) \times 0.15$ khởi tạo trọng số 0, bảo đảm mô hình không bao giờ có thể triệt tiêu quá 15% nét vẽ gốc.
3. **Anti-Collapse Loss Engine:** Stroke Dice Loss ($\mathcal{L}_{Dice}$ 0.40) + Bidirectional EDT Loss ($\mathcal{L}_{EDT\_bi}$ 0.20) + Stroke Density Loss (0.10).
4. **Smart Checkpoint Manager v0.2:** Rào chắn cứng bắt buộc $\text{Stroke Recall} \ge 70\%$ mới xét lưu model.

Mục tiêu của kế hoạch benchmark v0.2 là:
- Chứng minh **triệt tiêu 100% hiện tượng sụp đổ ảnh trắng (Blank Collapse Rate = 0.0%)**.
- Đo lường định lượng và định tính sự cải thiện vượt bậc về độ phủ nét, độ sắc nét và khử bóng ma so với TPS Gốc và Finetuned v0.1.
- **Bổ sung bài kiểm tra đối chuẩn chuyên sâu trên tập mẫu thực tế `visual_samples`** (`TestImage1`, `TestImage2`, `TestRandom`, `ChayGo`).

---

## 2. DANH MỤC CÁC ĐỐI TƯỢNG SO SÁNH ĐỐI ĐẦU (HEAD-TO-HEAD CANDIDATES)

| Ký hiệu | Mô hình Đánh giá | Trọng số / Thuật toán | Nguồn gốc | Vai trò & Mục tiêu Đối chuẩn |
|:---|:---|:---|:---:|:---|
| **`M0: TPS-Original`** | TPS-Inbetween Gốc | `checkpoints/pretrained_baselines/tps-original/model_latest.pt` | Mixamo 3D | Đo lường domain gap ban đầu (bóng ma, méo nét) |
| **`M1: TPS-AlgoFix`** | TPS Sửa Thuật Toán | Trọng số gốc + $\lambda=0.2$, Border/Interior Anchors | Nhóm nghiên cứu | Đối chứng năng lực can thiệp thuần hình học |
| **`M2: TPS-Finetuned-v0.1`** | TPS Fine-tuned v0.1 | `checkpoints/experiments/v0.1/model_tps_v0.1_best.pt` | Kaggle (Revoked) | Đối chứng sụp đổ chế độ (Mode Collapse baseline) |
| **`M3: TPS-Finetuned-v0.2`** | **TPS Fine-tuned v0.2** | `checkpoints/experiments/v0.2/model_tps_v0.2_best.pt` | Kaggle v0.2 | **Mô hình trọng tâm nghiệm thu đề tài** |
| **`B1: AnimeInbet`** | AnimeInbet (ICCV 2023) | Mạng vector hóa hình học nét vẽ hoạt hình | Baseline Cloned | Đối chuẩn SOTA chuyên Line Inbetweening |
| **`B2: SAIN`** | SAIN (ArXiv 2023) | Sketch-Aware Interpolation Network | Baseline Cloned | Đối chuẩn SOTA chuyên Sketch Inbetweening |
| **`B3: UPR-Net`** | UPR-Net (CVPR 2023) | Kim tự tháp hồi quy đa tầng | Baseline Cloned | Đại diện SOTA Video Frame Interpolation công nghiệp |

---

## 3. HỆ THỐNG CÁC TẬP KIỂM THỬ ĐỐI CHUẨN (BENCHMARK TEST SUITES)

Kế hoạch v0.2 mở rộng hệ thống kiểm thử thành **5 tập dữ liệu độc lập**, trong đó bổ sung chuyên biệt tập `visual_samples`:

```
                               ┌─────────────────────────────────────────────────────────┐
                               │           HỆ THỐNG BỘ TEST ĐỐI CHUẨN TOÀN DIỆN v0.2      │
                               └─────────────────────────────────────────────────────────┘
                                                            │
         ┌───────────────────────┬──────────────────────────┼───────────────────────┬───────────────────────┐
         ▼                       ▼                          ▼                       ▼                       ▼
  [TẬP 1: IN-DOMAIN VAL]   [TẬP 2: VISUAL SAMPLES]    [TẬP 3: CONTINUOUS SCENE] [TẬP 4: HARD SAKUGA]  [TẬP 5: ATD-12K TEST]
  - 508 Triplets Anime     - TestImage1 (F1-F12)      - Cảnh ChayGo 8s 24fps    - Motion lớn 10-20px  - 759 Triplets Nhật
  - Đa dạng mốc thời gian   - TestImage2 (F1-F12)      - Đo nhấp nháy nét        - Đo sức chịu đựng    - Đối chuẩn học thuật
  - Đo độ hội tụ thực      - TestRandom (Bamboo)      - Kiểm tra chuyển động    - Outlier RANSAC
```

### 3.1. Tập 1 — In-Domain Validation Suite (`data/processed/anime_tps_finetune_dataset_ready/val/`)
- **Quy mô:** 508 Triplets anime được gán nhãn chính xác từ các cảnh Sakugabooru & ATD-12K.
- **Mục đích:** Đo đạc độ chính xác pixel, độ phủ nét và tính ổn định trên nhiều mốc $t \in [0.25, 0.75]$.

### 3.2. Tập 2 — Visual Samples Test Suite (`data/test_sets/visual_samples/`) ★ [MỤC BỔ SUNG MỚI]
Tập dữ liệu kiểm thử thực tế đa dạng nguồn vẽ anime, kiểm tra khả năng nội suy trong điều kiện thực chiến:
1. **`TestRandom/` (Bamboo Series):**
   - Bộ 3 khung hình nét vẽ tự do: `Bamboo1.png` ($I_0$), `Bamboo2.png` ($I_{GT}$ tại $t=0.5$), `Bamboo3.png` ($I_1$).
   - Độ phân giải: $720 \times 400$, định dạng line art không nền.
   - Thách thức: Nét vẽ mảnh, cành trúc đan xen phức tạp.
2. **`TestImage1/` (Anime Character Scene 1):**
   - Chuỗi 12 frames $F_1 \dots F_{12}$ trích xuất từ video $24$fps kèm thông số hiển thị `time.txt`.
   - Keyframe anchors: $F_1, F_2, F_{12}$.
   - In-between frames cần nội suy: $F_3 \dots F_{11}$ (với các mốc thời gian $t$ thực tế từ 0.01s đến 0.11s).
   - Thách thức: Biến dạng biểu cảm khuôn mặt nhân vật và tóc mái.
3. **`TestImage2/` (Anime Character Scene 2):**
   - Chuỗi 12 frames $F_1 \dots F_{12}$ kèm `time.txt`, độ phân giải $1280 \times 720$.
   - Kiểm tra các đoạn chuyển động trung gian giữa các cặp keyframe kế cận.
4. **`ChayGo/` (Chạy Gỗ Lineart Series):**
   - Chuỗi 257 frames với nhãn thời gian từ `0.0s.png` đến `49.4s.png`.
   - Kiểm thử đoạn 1: $0.0s \to 0.5s$ với các frame ground truth $0.1s, 0.2s, 0.3s$.
   - Kiểm thử đoạn 2: $0.5s \to 1.0s$ với các frame ground truth $0.6s, 0.7s, 0.8s$.

### 3.3. Tập 3 — Phân Cảnh Liên Tục ChayGo 8s (`data/lineart_experiments/ChayGoAnimeLinesDetail_NoBG/`)
- Ghép toàn bộ chuỗi frame nội suy thành video 24fps để kiểm tra hiện tượng flickering và độ mượt thị giác.

### 3.4. Tập 4 — Chuyển Động Khó Sakuga (`data/test_sets/test_sakuga_hard/`)
- Đánh giá khả năng thích ứng khi độ dịch chuyển vượt $15$px.

### 3.5. Tập 5 — Tập Chuẩn Quốc Tế ATD-12K Lineart (`data/test_sets/ATD-12K/test_2k_540p/`)
- 759 triplets anime đối chuẩn với các công bố khoa học quốc tế.

---

## 4. MA TRẬN CHỈ SỐ ĐO LƯỜNG ĐỊNH LƯỢNG (QUANTITATIVE METRICS)

| Nhóm Chỉ số | Tên Chỉ số | Đơn vị / Khoảng | Ý nghĩa Kỹ thuật | Xu hướng | Ngưỡng Nghiệm Thu v0.2 |
|:---|:---|:---:|:---|:---:|:---:|
| **Bảo toàn Nét vẽ** | **Blank Collapse Rate** | $\%$ | Tỷ lệ ảnh sụp đổ thành nền trắng 100% | **0.0%** | **BẮT BUỘC $= 0.0\%$** |
| | **Stroke Recall** | $\%$ | Tỷ lệ nét GT được dự đoán phủ đúng trong bán kính 5px | Càng cao càng tốt | **$\ge 80.0\%$** |
| | **Stroke Density Ratio** | $[0, 2]$ | Tỷ lệ tổng số pixel nét pred / nét GT | Tiệm cận 1.0 | **$0.80 \sim 1.25$** |
| **Độ trung thực Pixel** | **PSNR** | dB | Tỉ số tín hiệu cực đại trên nhiễu | Càng cao càng tốt | **$\ge 18.0$ dB** |
| **Cấu trúc Nét** | **SSIM** | $[0, 1]$ | Tương đồng cấu trúc và tương phản cục bộ | Càng cao càng tốt | **$\ge 0.85$** |
| **Khoảng cách Hình học**| **Chamfer Distance** | $[0, 1]$ | Khoảng cách trung bình giữa 2 đám mây nét (phạt 1.0 nếu mất nét) | Càng thấp càng tốt | **$\le 0.005$** |
| **Đo lường Bóng ma** | **Ghost Pixels** | Điểm ảnh | Số pixel xám mờ (180 ~ 250) trên nền trắng | Càng thấp càng tốt | **$< 350$ px** |
| **Độ Méo Nét** | **Line Tortuosity** | $\ge 1.0$ | Tỉ số độ dài đường cong nét so với đường thẳng | Tiệm cận 1.0 | **$< 1.25$** |
| **Tốc độ Xử lý** | **Inference Time** | ms | Thời gian nội suy 1 frame 540p | Càng nhanh càng tốt | **$< 1500$ ms (GPU)** |

---

## 5. QUY TRÌNH KIỂM TOÁN TRỰC QUAN ĐỊNH TÍNH (QUALITATIVE VISUAL AUDIT)

Để đảm bảo kết quả trung thực, hệ thống sinh tự động 4 nhóm ảnh kiểm toán trực quan:
1. **Side-by-Side Panel 4 Cột:**
   - Cột 1: Keyframe 0 ($t = 0.0$)
   - Cột 2: Coarse TPS Warp (Khung hình biến dạng hình học nền tảng)
   - Cột 3: Finetuned v0.2 (Khung hình nội suy hoàn thiện)
   - Cột 4: Ground Truth (Khung hình thực tế)
2. **Difference Heatmaps (Bản đồ nhiệt sai số):**
   - Ánh xạ $|I_{pred} - I_{GT}|$ qua `cv2.COLORMAP_JET`.
   - Màu xanh dương: Sai số $= 0$. Màu đỏ: Lệch nét hoặc bóng ma.
3. **Zoomed Detail Crops (Phóng to chi tiết 400%):**
   - Cắt phóng to $400\%$ tại các vùng nét phức tạp (tóc mái, mắt, bàn tay) kích thước $480 \times 480$.
4. **Visual Samples Custom Panels:**
   - Xuất riêng bảng đối sánh cho từng mẫu trong `visual_samples/` (`Bamboo`, `TestImage1`, `TestImage2`, `ChayGo`).

---

## 6. KỊCH BẢN THỰC THI KIỂM THỬ BẰNG MÃ NGUỒN

Quy trình benchmark v0.2 được thực thi tự động qua script `bench_triplet_02_v0_2_eval.py`:

```bash
# BƯỚC 1: Chạy Benchmark Toàn diện (Validation Split + Visual Samples)
python continuous_benchmark/inference_scripts/bench_triplet_02_v0_2_eval.py \
    --model_path checkpoints/experiments/v0.2/model_tps_v0.2_best.pt \
    --dataset_csv data/processed/anime_tps_finetune_dataset_ready/triplets.csv \
    --split val \
    --visual_samples_dir data/test_sets/visual_samples \
    --output_dir continuous_benchmark/metrics/tps-inbetween-finetuned-v0.2 \
    --visual_dir continuous_benchmark/visual_audit/finetune_v0.2

# BƯỚC 2: Cập nhật Báo cáo Tổng thể Hệ thống Benchmark
python continuous_benchmark/metrics/generate_overall_benchmark.py
```

---

## 7. TIÊU CHUẨN NGHIỆM THU CHÍNH THỨC v0.2

Checkpoint `model_tps_v0.2_best.pt` được xem là **ĐẠT CHUẨN NGHIỆM THU XUẤT SẮC** và được chuyển giao sang `checkpoints/production_candidates/tps_anime_candidate_v0.2.pt` khi thỏa mãn đồng thời:
- [x] **Blank Collapse Rate $= 0.0\%$** trên toàn bộ các tập test.
- [x] **Stroke Recall $\ge 80.0\%$** (bảo toàn trọn vẹn nét vẽ).
- [x] **PSNR $\ge 18.0$ dB** và **SSIM $\ge 0.85$**.
- [x] Khử bóng ma hiệu quả (Ghost Pixels giảm đáng kể so với Coarse Warp).
- [x] Hoạt động ổn định trên toàn bộ các ca thử nghiệm của `visual_samples`.
