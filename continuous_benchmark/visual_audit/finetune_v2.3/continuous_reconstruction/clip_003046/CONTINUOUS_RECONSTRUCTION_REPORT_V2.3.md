# BÁO CÁO PHỤC DỰNG CHUYỂN ĐỘNG LIÊN TỤC V2.3 (CLEAN INBETWEEN BENCHMARK)
> **Phân cảnh thử nghiệm:** `clip_003046`  
> **Bộ dữ liệu gốc:** `sakuga_golden_dataset_tps_ready_v1_patch2`  
> **Model Checkpoint:** `model_tps_v2.3_best.pt` (Anti-Thickening & FP32 Standard)  
> **Chế độ hiển thị:** **Clean Video (Không dán thanh HUD thống kê)**  
> **Tốc độ kết xuất:** **0.5x Slow Motion (12.0 FPS)**  
> **Tổng số khung hình:** **3 frames** (Keyframes: 2, Inbetweens: 1)  
> **Ngày thực hiện:** 21/10/2026 14:23:10  

---

## 1. CHỈ SỐ ĐÁNH GIÁ CHẤT LƯỢNG TẠO INBETWEEN FRAME (V2.3)
*(Lưu ý: Bảng chỉ số dưới đây chỉ tính toán trên các Inbetween frames sinh ra bởi mô hình, hoàn toàn không gộp Keyframe)*

| Chỉ Số Đánh Giá Inbetween | Kết Quả Đạt Được (V2.3) | Đơn Vị | Ý Nghĩa Kỹ Thuật |
|:---|:---:|:---:|:---|
| **PSNR Trung Bình** | **16.96 dB** | dB (Cao hơn là tốt) | Độ bám sát pixel so với nét vẽ của họa sĩ |
| **SSIM Trung Bình** | **0.8972** | [0, 1] (Cao hơn là tốt) | Độ tương đồng cấu trúc đường nét |
| **Lượng Ghost Pixels** | **7450 px** | Pixels (Thấp hơn là tốt) | Lượng nét bóng ma dư thừa trên nền trắng |
| **Tỷ Lệ Mật Độ Nét (Density)** | **0.6650** | Tỷ số (Lý tưởng ~ 1.0) | Khắc phục phình nét, giữ nét thanh mảnh |
| **Chamfer Distance (CD)** | **0.002536** | Distance (Thấp hơn là tốt) | Độ lệch hình học vector đường nét |
| **Độ Ổn Định Thời Gian (TSI)** | **±0.00 dB** | dB (Thấp hơn là tốt) | Tránh nhấp nháy chất lượng qua từng frame |
| **Thời Gian Suy Luận Trung Bình** | **111.8 ms** | ms / frame | Tốc độ nội suy thời gian thực trên GPU |

---

## 2. DANH MỤC TỆP XUẤT BẢN DẠNG CLEAN (KHÔNG HUD)

1. **`01_reconstructed_v23_clean.mp4`**: Chuỗi hoạt họa phục dựng hoàn chỉnh (12 FPS, 0.5x).
2. **`02_ground_truth_clean.mp4`**: Chuỗi hoạt họa gốc của họa sĩ Sakuga.
3. **`03_side_by_side_clean.mp4`**: Video so sánh 2 cột (Gốc Trái vs V2.3 Phải) hiển thị nét vẽ nguyên bản.
4. **`04_three_way_clean.mp4`**: Video 3 cột đối chiếu: Ground Truth vs Coarse TPS vs V2.3.
5. **`05_comparison_preview_clean.gif`**: Ảnh động GIF preview trực quan.
6. **`inbetween_metrics.csv`**: Bảng dữ liệu số đo chi tiết từng inbetween frame.
7. **`frames_reconstructed/`**: Thư mục lưu các khung hình PNG sạch do V2.3 sinh ra.
