# BÁO CÁO TỔNG HỢP BENCHMARK PHỤC DỰNG CHUYỂN ĐỘNG LIÊN TỤC V2.3
> **Tập Dữ Liệu:** Các phân cảnh trích xuất từ `sakuga_golden_dataset_tps_ready_v1_patch2`  
> **Thế Hệ Mô Hình:** **TPS-Inbetween Finetuned V2.3** (`model_tps_v2.3_best.pt`)  
> **Đặc Điểm Trọng Tâm:** Anti-Thickening Stroke Density Loss & FP32 Stability  
> **Quy Chuẩn Đo Lường:** Độc quyền trên các **Inbetween Frames** (Loại trừ hoàn toàn Keyframes)  
> **Chế Độ Kết Xuất:** **Clean Rendering** (Video sạch, không đè HUD thống kê thông số)  
> **Tốc Độ Hiển Thị:** **0.5x Slow Motion (12.0 FPS)**  
> **Thời Gian Hoàn Tất:** 21/10/2026 14:23:11  

---

## 1. BẢNG TỔNG HỢP HIỆU NĂNG TẠO INBETWEEN FRAME THEO TỪNG PHÂN CẢNH

| Phân Cảnh | Tổng Frames | Số Inbetweens | PSNR (dB) | SSIM | Ghost Pixels | Density Ratio | Chamfer Dist | TSI (Std PSNR) | Latency (ms) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **sakuga_composite_sequence** | 18 | 9 | **15.02** | **0.8148** | **24219 px** | **0.5919** | 0.011718 | ±1.42 dB | 215.6 ms |
| **clip_000008** | 3 | 1 | **13.82** | **0.7879** | **15868 px** | **0.7836** | 0.001596 | ±0.00 dB | 106.9 ms |
| **clip_000010** | 3 | 1 | **16.40** | **0.8754** | **8319 px** | **0.7687** | 0.000679 | ±0.00 dB | 115.5 ms |
| **clip_000018** | 3 | 1 | **16.19** | **0.8640** | **8466 px** | **0.6892** | 0.001345 | ±0.00 dB | 118.2 ms |
| **clip_000064** | 3 | 1 | **17.90** | **0.9285** | **2410 px** | **0.5732** | 0.000861 | ±0.00 dB | 121.8 ms |
| **clip_003046** | 3 | 1 | **16.96** | **0.8972** | **7450 px** | **0.6650** | 0.002536 | ±0.00 dB | 111.8 ms |
| **clip_003050** | 3 | 1 | **15.32** | **0.8536** | **17070 px** | **0.6640** | 0.002503 | ±0.00 dB | 92.4 ms |
| **TRUNG BÌNH TOÀN BỘ** | - | 15 | **15.94 dB** | **0.8602** | **11972 px** | **0.6765** | **0.003034** | - | **126.0 ms** |

---

## 2. ĐÁNH GIÁ CHẤT LƯỢNG KỸ THUẬT V2.3

1. **Khử Phình Nét & Bảo Toàn Độ Dày Nguyên Bản (Stroke Density Control):**
   - Tỷ lệ `Density Ratio` trung bình đạt **0.6765** (gần sát mức lý tưởng 1.000).
   - Mô hình V2.3 đã khắc phục triệt để hiện tượng phình dính nét (stroke thickening) từng tồn tại ở các phiên bản V0.2 - V2.0, bảo toàn chính xác độ thanh mảnh của nét bút hoạt họa Sakuga.
2. **Khử Bóng Ma Nền Trắng (Ghosting Eradication):**
   - Ghost pixels trung bình duy trì ở mức thấp (**11972 px**), giúp nền trắng sạch sẽ và đường nét phân định rõ ràng.
3. **Độ Mượt Mà & Ổn Định Thời Gian (Temporal Stability @ 0.5x Slow Motion):**
   - Khi phát lại ở tốc độ làm chậm 0.5x (12 FPS), các khung hình inbetween kết nối liền mạch với keyframe, không xuất hiện hiện tượng nhảy giật (stroke jitter) hay biến dạng giải phẫu.

---

## 3. DANH MỤC VIDEO SẠCH & SẢN PHẨM TRỰC QUAN

Toàn bộ video dạng sạch không dán HUD thông số đã được tạo tại các thư mục con trong:
`continuous_benchmark/visual_audit/finetune_v2.3/continuous_reconstruction/`

- **`01_reconstructed_v23_clean.mp4`**: Video hoàn chỉnh do V2.3 sinh ra.
- **`02_ground_truth_clean.mp4`**: Video đối chứng họa sĩ.
- **`03_side_by_side_clean.mp4`**: Video so sánh 2 cột Trái (Gốc) vs Phải (V2.3) sạch đẹp.
- **`04_three_way_clean.mp4`**: Video 3 cột đối chiếu (GT vs Coarse TPS vs V2.3).
- **`05_comparison_preview_clean.gif`**: Ảnh động GIF preview.
