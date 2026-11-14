# KẾ HOẠCH TOÀN DIỆN: CONTINUOUS BENCHMARK TIẾN HÓA MÔ HÌNH TPS (V0.1 ──► V3.0 RE2)

> **Mục tiêu:** Thiết lập quy trình kiểm thử tự động, chuẩn hóa và toàn diện để đánh giá định lượng sự tiến hóa của toàn bộ các thế hệ checkpoint TPS-Inbetween (từ phiên bản sơ khai v0.1 đến phiên bản hoàn thiện v3.0 re2) trên cùng một chuỗi chuyển động anime 540p chuẩn (`data/lineart_experiments/test`).

---

## 1. MỤC TIÊU & NGUYÊN TẮC THẨM ĐỊNH

1. **Tính công bằng & Nhất quán tuyệt đối (Fair & Deterministic Evaluation):**
   * Tất cả các thế hệ mô hình đều được kiểm thử trên cùng tập chuỗi 24 khung hình 540p ($960 \times 540$).
   * Sử dụng chung cấu hình chuẩn hóa: **7 Keyframes cố định** (F00, F03, F07, F11, F14, F19, F23) và **17 Inbetweens cần AI tái tạo**.
   * Cố định thuật toán đối sánh hình học: Trích xuất SIFT keypoints ($\le 80$ điểm), giải phương trình Robust TPS Thin-Plate Spline và tính toán TPS Optical Flow.
2. **Đo lường đa chiều (Multi-dimensional Metrics):**
   * **Cổ điển:** PSNR (dB), SSIM.
   * **Nét vẽ & Hoạt họa (Line Art & Cartoon):** Stroke Recall (%), Stroke Density Ratio (x), Skeleton Recall (%), Hair-Tip Recall (%).
   * **Độ sạch nền & Sai số:** Far Ghost Pixels, Dark Ghost Pixels, IE-MAE.
   * **Hình thái & Phân bố tọa độ:** Weighted Chamfer Distance (WCD), 1D-Sliced Earth Mover's Distance (S-EMD).
   * **Hiệu năng:** Latency (ms/frame), Throughput (FPS).
3. **Lưu trữ chuẩn hóa & Tập trung:**
   * Tất cả kết quả (CSV, JSON summary, Video so sánh MP4, Bảng Montage 24-frame, Khung hình chi tiết) được quy hoạch vào thư mục trung tâm:  
     `checkpoints/experiments/continuous_benchmarks/<version>/`

---

## 2. MA TRẬN ÁNH XẠ TOÀN BỘ CHECKPOINTS (V0.1 ──► V3.0 RE2)

Hệ thống đã rà soát và xác định chính xác đường dẫn của 15 thế hệ mô hình:

| STT | Thế hệ | Phiên bản (Version) | Đường dẫn Checkpoint Khảo sát | Lớp Kiến trúc Tương ứng | Trạng thái |
| :---: | :---: | :--- | :--- | :--- | :---: |
| **0** | Baseline | **Pretrained TPS** | `checkpoints/pretrained_baselines/tps-original/model_latest.pt` | `TPSInbetween` | ⚪ Chờ chạy |
| **1** | Gen 1 | **v0.1** | `checkpoints/experiments/v0.1/model_tps_v0.1_best.pt` | `TPSInbetweenFinetune` | ⚪ Chờ chạy |
| **2** | Gen 1 | **v0.2** | `checkpoints/experiments/v0.2/model_tps_v0.2_best.pt` | `TPSInbetweenFinetune_v2` | ⚪ Chờ chạy |
| **3** | Gen 1 | **v0.3** | `checkpoints/experiments/v0.3/model_tps_v0.3_best.pt` | `TPSInbetweenFinetune_v3` | ⚪ Chờ chạy |
| **4** | Gen 2 | **v1.0** | `checkpoints/experiments/v1.0/model_tps_v1.0_best.pt` | `TPSInbetweenFinetune_v10` | ⚪ Chờ chạy |
| **5** | Gen 2 | **v1.1** | `checkpoints/experiments/v1.1/model_tps_v1.1_best.pt` | `TPSInbetweenFinetune_v10` | ⚪ Chờ chạy |
| **6** | Gen 3 | **v2.0** | `checkpoints/experiments/v2.0/v2.0.2/tps_finetune_v2.0_artifacts/checkpoints_v1.1/model_tps_v2.0_best.pt` | `TPSInbetweenFinetune_v10` | ⚪ Chờ chạy |
| **7** | Gen 3 | **v2.1** | `checkpoints/experiments/v2.1/tps_finetune_v2.1_artifacts/checkpoints_v2.1/model_tps_v1.1_best.pt` | `TPSInbetweenFinetune_v10` | ⚪ Chờ chạy |
| **8** | Gen 3 | **v2.2** | `checkpoints/experiments/v2.2/tps_finetune_v2.2_artifacts/checkpoints_v2.2/model_tps_v2.2_best.pt` | `TPSInbetweenFinetune_v10` | ⚪ Chờ chạy |
| **9** | Gen 3 | **v2.3** | `checkpoints/experiments/v2.3/tps_finetune_v2.3_artifacts/checkpoints_v2.3/model_tps_v2.3_best.pt` | `TPSInbetweenFinetune_v10` | ⚪ Chờ chạy |
| **10** | Gen 4 | **v2.4** | `checkpoints/experiments/v2.4/checkpoints_v2.4/model_tps_v2.4_best.pt` | `TPSInbetweenFinetune_v10` | ⚪ Chờ chạy |
| **11** | Gen 4 | **v2.5** | `checkpoints/experiments/v2.5/checkpoints_v2.5/model_tps_v2.5_best.pt` | `TPSInbetweenFinetune_v10` | ⚪ Chờ chạy |
| **12** | Gen 4 | **v2.6** | `checkpoints/experiments/v2.6/checkpoints_v2.6/model_tps_v2.6_best.pt` | `TPSInbetweenFinetune_v10` | 🟢 **ĐÃ HOÀN TẤT** |
| **13** | Gen 5 | **v3.0.5** | `checkpoints/experiments/v3.0 (2.6)/v3.0.5 (25epoch)/checkpoints_v3.0/model_tps_v3.0_best_progress.pt` | `TPSInbetweenFinetune_v10` | ⚪ Chờ chạy |
| **14** | Gen 6 | **v3.0 re1** | `checkpoints/experiments/v3.0 re/v3.0 re1/checkpoints_v3.0_remastered/model_tps_v3.0_remastered_ep21.pt` | `TPSInbetweenFinetune_v10` | ⚪ Chờ chạy |
| **15** | Gen 6 | **v3.0 re2 (Soup)** | `checkpoints/experiments/v3.0 re/v3.0 re2/checkpoints_v3.0_remastered/model_tps_v3.0_remastered_soup.pt` | `TPSInbetweenFinetune_v10` | 🟢 **ĐÃ HOÀN TẤT** |
| **16** | Gen 6 | **v3.0 re2 (Ep12)** | `checkpoints/experiments/v3.0 re/v3.0 re2/checkpoints_v3.0_remastered/model_tps_v3.0_remastered_ep12.pt` | `TPSInbetweenFinetune_v10` | ⚪ Chờ chạy |

---

## 3. QUY HOẠCH CẤU TRÚC THƯ MỤC LƯU TRỮ

Thư mục lưu trữ tập trung: `E:\CODE\PBL\PBL6\checkpoints\experiments\continuous_benchmarks\`

```
continuous_benchmarks/
├── v0.1/
│   ├── benchmark_results.csv
│   ├── summary.json
│   ├── side_by_side.mp4
│   ├── montage_24frames.png
│   └── frames/ (frames_reconstructed & frames_side_by_side)
├── v0.2/ ...
├── v2.6/                     ◄── [ĐÃ CÓ ĐẦY ĐỦ]
├── v3.0_re2_soup/            ◄── [ĐÃ CÓ ĐẦY ĐỦ]
└── MASTER_SUMMARY/
    ├── evolution_timeline_master.csv       ◄── Bảng tổng hợp đối đầu toàn diện 16 phiên bản
    ├── evolution_metrics_comparison.png    ◄── Biểu đồ đường tiến hóa (SSIM, Recall, Density, Ghost)
    └── evolution_radar_chart.png          ◄── Biểu đồ radar năng lực đa chiều
```

---

## 4. THIẾT KẾ RUNNER ENGINE TỰ ĐỘNG (`run_batch_evolution_benchmarks.py`)

Xây dựng script tự động hóa batch runner đặt tại:  
`continuous_benchmark/inference_scripts/run_batch_evolution_benchmarks.py`

### Các tính năng cốt lõi:
1. **Dynamic Model Loader:**
   * Tự động nhận diện cấu trúc checkpoint qua tên phiên bản:
     * Dòng v0.1 $\rightarrow$ `TPSInbetweenFinetune`
     * Dòng v0.2 $\rightarrow$ `TPSInbetweenFinetune_v2`
     * Dòng v0.3 $\rightarrow$ `TPSInbetweenFinetune_v3`
     * Dòng v1.0 trở đi $\rightarrow$ `TPSInbetweenFinetune_v10` (xử lý tiền tố `module.`, `base_model.` bằng `strict=False`).
2. **Cơ chế cách ly lỗi (Fault-tolerant Execution):**
   * Mỗi phiên bản được bọc trong khối `try-except`. Nếu một checkpoint gặp vấn đề về cấu trúc trọng số, runner sẽ ghi nhận log lỗi và tiếp tục thực thi các phiên bản kế tiếp mà không làm sập toàn bộ chuỗi benchmark.
3. **Tự động đóng gói & Xuất Master Report:**
   * Sau khi duyệt qua tất cả các checkpoint, script tự động gom toàn bộ file `summary.json` của từng version để tạo ra bảng tổng hợp: `evolution_timeline_master.csv`.
   * Tự động vẽ đồ thị tiến trình tiến hóa (`evolution_metrics_comparison.png`) thể hiện rõ sự tăng trưởng của Stroke Recall, SSIM và sự sụt giảm của Ghost Pixels.

---

## 5. DỰ TOÁN THỜI GIAN & TÀI NGUYÊN THỰC THI

* **Tốc độ suy luận đã đo đạc thực tế trên RTX 3050 Laptop:**
  * Thời gian nội suy 17 frame inbetween: **~5.0 giây/model**.
  * Thời gian xuất Video MP4 và Montage: **~1.5 giây/model**.
  * Tổng thời gian cho mỗi model: **~6.5 giây**.
* **Tổng thời gian dự kiến cho 14 checkpoint còn lại:**
  $$T_{total} = 14 \text{ models} \times 6.5 \text{s} \approx 91 \text{ giây} \ (\approx 1.5 \text{ phút})$$
* **Kết luận:** Quy trình cực kỳ nhẹ và nhanh, hoàn toàn có thể chạy mượt mà ngay trên máy local mà không lo nóng máy hay tràn RAM.

---

## 6. LỘ TRÌNH THỰC THI 3 BƯỚC

1. **Bước 1 (Đã hoàn tất):**
   * Dọn dẹp sạch sẽ toàn bộ các file test v2.6 và v3.0 tại thư mục gốc `PBL6`.
   * Khởi tạo thư mục trung tâm `checkpoints/experiments/continuous_benchmarks/` và lưu trữ hoàn chỉnh kết quả của v2.6 và v3.0_re2.
2. **Bước 2:**
   * Biên soạn script `continuous_benchmark/inference_scripts/run_batch_evolution_benchmarks.py` tích hợp bộ nạp động và đo lường đầy đủ 10 chỉ số.
3. **Bước 3:**
   * Thực thi script batch runner để quét toàn bộ 14 checkpoint còn lại.
   * Kết xuất `evolution_timeline_master.csv` và biểu đồ trực quan hóa tiến hóa mô hình.
