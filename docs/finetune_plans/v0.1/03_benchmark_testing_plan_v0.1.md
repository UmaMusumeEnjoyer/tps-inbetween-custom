# KẾ HOẠCH ĐÁNH GIÁ ĐỐI CHUẨN & SO SÁNH ĐỐI ĐẦU PHIÊN BẢN v0.1 (BENCHMARK & EVALUATION PLAN v0.1)

> **Mục tiêu:** Quy định toàn bộ phương pháp luận, kịch bản thực thi, hệ thống chỉ số đo lường định lượng và tiêu chuẩn kiểm toán thị giác (visual audit) để đối đầu trực tiếp giữa **Mô hình TPS-Inbetween Gốc (Pretrained)** và **Mô hình TPS-Inbetween Tinh chỉnh v0.1 (Fine-tuned v0.1)** sau khi tải checkpoint về máy trạm cục bộ.

---

## 1. MÔ TẢ CÁC ĐỐI TƯỢNG SO SÁNH (HEAD-TO-HEAD CANDIDATES)

Bài kiểm tra đối chuẩn sẽ so sánh trực diện 3 cấu hình mô hình của họ TPS-Inbetween cùng với các mô hình baseline hàng đầu:

| Ký hiệu Mã | Tên Đầy Đủ | Trọng Số / Thuật Toán Sử Dụng | Nguồn gốc | Vai trò trong Nghiên cứu |
|:---|:---|:---|:---:|:---|
| **`M0: TPS-Original`** | TPS-Inbetween Gốc | `checkpoints/pretrained_baselines/tps-original/model_latest.pt` | Tác giả gốc (Mixamo 3D) | Baseline gốc ban đầu |
| **`M1: TPS-AlgoFix`** | TPS-Inbetween Sửa Thuật Toán | Trọng số gốc + Giải thuật $\lambda=0.2$, Border/Interior Anchors, Post Ghost Filter | Nhóm phát triển tự cải tiến | Đối chứng hiệu quả can thiệp thuật toán thuần túy |
| **`M2: TPS-Finetuned-v0.1`** | TPS-Inbetween Fine-tuned v0.1 | `checkpoints/experiments/v0.1/model_tps_v0.1_best.pt` | Huấn luyện trên Kaggle | **Mô hình trọng tâm cần chứng minh hiệu quả** |
| **`B1: AnimeInbet`** | AnimeInbet (ICCV 2023) | Mạng vector hóa hình học nét vẽ hoạt hình | Baseline Cloned | Đối chứng SOTA chuyên Line Inbetweening |
| **`B2: SAIN`** | SAIN (ArXiv 2023) | Sketch-Aware Interpolation Network | Baseline Cloned | Đối chứng SOTA chuyên Sketch Inbetweening |

---

## 2. CÁC TẬP KIỂM THỬ ĐỘC LẬP (BENCHMARK TEST SUITES)

Để loại trừ hoàn toàn nguy cơ quá khớp (overfitting) và kiểm tra khả năng tổng quát hóa, bài test được thực hiện trên 4 tập dữ liệu:

```
                  ┌─────────────────────────────────────────────────────────┐
                  │              HỆ THỐNG BỘ TEST ĐỐI CHUẨN                │
                  └─────────────────────────────────────────────────────────┘
                                               │
         ┌───────────────────────┬─────────────┴───────────┬───────────────────────┐
         ▼                       ▼                         ▼                       ▼
  [TẬP 1: IN-DOMAIN VAL]   [TẬP 2: CONTINUOUS SCENE]  [TẬP 3: HARD SAKUGA]  [TẬP 4: ATD-12K TEST 2K]
  - 508 triplets val       - Cảnh anime ChayGo 8s     - Biên độ dịch chuyển   - 759 triplets anime
  - Đo lường độ hội tụ     - Kiểm tra flickering 24fps   lớn (10 ~ 20px)       - Đối chuẩn học thuật
```

1. **Tập 1 — In-Domain Validation Suite (`data/processed/anime_tps_finetune_dataset_ready/val/`):**
   - Quy mô: **508 Triplets** đa dạng mốc thời gian $t \in [0.25, 0.75]$.
   - Mục đích: Đo lường mức độ tiếp thu domain nét vẽ anime thực tế so với tập huấn luyện.
2. **Tập 2 — Phân Cảnh Liên Tục ChayGo (`data/test_sets/visual_samples/` & `ChayGoAnimeLinesDetail_NoBG/`):**
   - Quy mô: Chuỗi keyframe liên tục dài 8 giây (tương đương 192 frames ở 24fps).
   - Mục đích: Đánh giá độ ổn định thời gian (temporal consistency), phát hiện hiện tượng nhấp nháy nét (flickering) khi xem dưới dạng video chuyển động thực tế.
3. **Tập 3 — Chuyển Động Phức Tạp Biên Độ Lớn (`data/test_sets/test_sakuga_hard/`):**
   - Quy mô: Các phân cảnh hành động tốc độ cao từ Sakugabooru có độ dịch chuyển $10 - 20$px.
   - Mục đích: Đánh giá sức chịu đựng của mô hình khi chuyển động vượt ra ngoài vùng tối ưu của TPS.
4. **Tập 4 — Tập Chuẩn Quốc Tế ATD-12K Lineart (`data/test_sets/ATD-12K/test_2k_540p/`):**
   - Quy mô: 759 triplets của phần anime Nhật Bản (`Japan_*` và `Hayao_*`) được trích xuất lineart.
   - Mục đích: Cung cấp kết quả đối chuẩn khách quan để so sánh trực tiếp với các bài báo khoa học.

---

## 3. MA TRẬN CHỈ SỐ ĐO LƯỜNG ĐỊNH LƯỢNG (QUANTITATIVE METRICS)

| Nhóm Chỉ số | Tên Chỉ số | Đơn vị / Khoảng | Ý nghĩa Kỹ thuật | Xu hướng mong muốn |
|:---|:---|:---:|:---|:---:|
| **Độ trung thực Pixel** | **PSNR** | dB ($0 \to \infty$) | Tỉ số tín hiệu cực đại trên nhiễu; đo độ khớp pixel giữa ảnh dự đoán và Ground Truth | **Càng cao càng tốt** (Kỳ vọng: $> 16.5$ dB) |
| **Cấu trúc Nét vẽ** | **SSIM** | $[0, 1]$ | Độ tương đồng cấu trúc đường nét và độ tương phản cục bộ | **Càng cao càng tốt** (Kỳ vọng: $> 0.82$) |
| **Cảm nhận Thị giác** | **LPIPS** (Alex/VGG) | $[0, 1]$ | Khoảng cách nhận thức thị giác của con người | **Càng thấp càng tốt** (Kỳ vọng: $< 0.15$) |
| **Đo lường Bóng ma** | **Ghost Pixels (180-250)** | Số lượng pixel | Đếm tổng số pixel xám mờ (cường độ 180 ~ 250) xuất hiện vô cớ trên vùng nền trắng | **Càng thấp càng tốt** (Kỳ vọng: $< 500$ px) |
| **Độ Méo Nét** | **Line Tortuosity** | $\ge 1.0$ | Tỉ số chiều dài đường cong nét vẽ thực tế so với khoảng cách đường thẳng; phát hiện nét bị nhăn/uốn lượn | **Càng tiệm cận 1.0 càng tốt** (Kỳ vọng: $< 1.25$) |
| **Khoảng cách Hình học** | **Chamfer Distance (CD)** | Điểm ảnh | Khoảng cách không gian Euclidean trung bình giữa các đám mây điểm của nét vẽ | **Càng thấp càng tốt** |
| **Độ Ổn định Thời gian** | **Temporal PSNR Variance** | dB | Độ lệch chuẩn PSNR giữa các mốc $t \in \{0.25, 0.33, 0.5, 0.67, 0.75\}$ | **Càng thấp càng tốt** (Kỳ vọng: $< 1.2$ dB) |

---

## 4. QUY TRÌNH KIỂM TOÁN TRỰC QUAN ĐỊNH TÍNH (QUALITATIVE VISUAL AUDIT)

Bên cạnh các chỉ số số học, thẩm mỹ hoạt họa đòi hỏi kiểm toán trực quan khắt khe:

### 4.1. Bảng So Sánh Ghép Song Song 5 Cột (5-Way Side-by-Side Panel)
Mỗi mẫu kiểm thử sẽ được render thành một ảnh tổng hợp gồm 5 cột:
```
┌──────────────┬──────────────┬──────────────┬──────────────────┬──────────────┐
│  Keyframe 0  │ TPS-Original │  TPS-AlgoFix │ TPS-Finetune-v0.1│ Ground Truth │
│   (t = 0.0)  │  (Mixamo 3D) │  (Thuật toán)│   (Mô hình mới)  │   (Thực tế)  │
└──────────────┴──────────────┴──────────────┴──────────────────┴──────────────┘
```

### 4.2. Bản Đồ Nhiệt Sai Số (Absolute Error Heatmap)
- Tính toán $|I_{\text{pred}} - I_{GT}|$ và ánh xạ qua dải màu `cv2.COLORMAP_JET`:
  - Màu xanh dương: Hoàn toàn trùng khớp sai số bằng 0.
  - Màu đỏ: Vùng nét bị lệch vị trí hoặc bóng ma xuất hiện đậm.
- Tiêu chí: Vùng màu đỏ trên ảnh của `TPS-Finetune-v0.1` phải thu hẹp tối thiểu 60% so với `TPS-Original`.

### 4.3. Phóng To Chi Tiết Mũi Tên Đỏ (Zoomed Crop on Critical Areas)
Cắt phóng to $400\%$ tại 3 vị trí nhạy cảm nhất của nhân vật:
1. **Sợi tóc mái thanh mảnh (Hair Contours):** Kiểm tra xem nét mảnh có bị biến mất hoặc đứt gãy không.
2. **Khuôn mặt & Viền mắt (Facial Features):** Kiểm tra xem có bị bóng ma mắt kép (double vision / ghosting) không.
3. **Bàn tay và ngón tay (Fingers / Hands):** Nơi có chuyển động xoay và độ dời thay đổi nhanh.

### 4.4. Đánh Giá Trải Nghiệm Video Chuyển Động (24fps Continuous Playback)
- Sử dụng script `pipelines/utils/frame_to_vid.py` để ghép toàn bộ chuỗi frame nội suy của phân cảnh ChayGo thành video 24fps.
- Sử dụng công cụ phát video chuyên nghiệp (hoặc trình duyệt) soi từng frame (frame-by-frame scrubbing) để phát hiện giật hình.

---

## 5. KỊCH BẢN THỰC THI KIỂM THỬ TỰ ĐỘNG BẰNG MÃ NGUỒN

Khi tải mô hình `model_tps_v0.1_best.pt` về đặt tại `checkpoints/experiments/v0.1/`, quy trình kiểm thử được kích hoạt tự động theo 4 bước:

```bash
# BƯỚC 1: Chạy nội suy trên 508 Triplets của tập Val
python continuous_benchmark/inference_scripts/bench_triplet_01_v0_1_eval.py \
    --model_path checkpoints/experiments/v0.1/model_tps_v0.1_best.pt \
    --dataset_csv data/processed/anime_tps_finetune_dataset_ready/triplets.csv \
    --output_dir continuous_benchmark/metrics/tps-inbetween-finetuned-v0.1/val_run/

# BƯỚC 2: Chạy kiểm toán trên cảnh mẫu ChayGo
python continuous_benchmark/inference_scripts/run_tps_fix.py \
    --checkpoint checkpoints/experiments/v0.1/model_tps_v0.1_best.pt \
    --output_name tps_finetuned_v0.1_chaygo \
    --dataset_dir data/lineart_experiments/ChayGoAnimeLinesDetail_NoBG

# BƯỚC 3: Tổng hợp báo cáo Markdown, CSV và Word tự động
python continuous_benchmark/metrics/generate_overall_benchmark.py

# BƯỚC 4: Tạo video so sánh song song đối chiếu
python demo_01_atd12k_orig_tps/generate_continuous_scene_demo.py
```

---

## 6. TIÊU CHUẨN THÔNG QUA & CHUYỂN GIAO SẢN PHẨM

Một checkpoint phiên bản v0.1 được xem là **Thành Công Xuất Sắc** và được quyền cập nhật thành ứng viên sản phẩm khi thỏa mãn:
1. **Vượt trội mô hình gốc:** PSNR cao hơn ít nhất $+4.5$ dB, SSIM cao hơn $+0.12$.
2. **Triệt tiêu bóng ma thực chất:** Số lượng Ghost Pixels giảm $> 70\%$ mà **không cần bật cờ lọc bóng ma hậu xử lý** (`--clean_ghosting`).
3. **Tốc độ giữ nguyên:** Thời gian suy luận trên 1 frame 540p không vượt quá $3.5$ giây trên GPU RTX 3050 hoặc T4.
4. Khi đạt đủ các tiêu chí trên, trọng số sẽ được sao chép vào `checkpoints/production_candidates/tps_anime_candidate_v0.1.pt` để tích hợp vào ứng dụng `product_app/`.
