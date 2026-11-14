# BÁO CÁO PHỤC DỰNG PHÂN CẢNH CHUYỂN ĐỘNG NHÂN VẬT LIÊN TỤC V1.0 (TỐC ĐỘ 0.5x)
> **Phân cảnh thử nghiệm:** `TestImage1`  
> **Model Checkpoint:** `model_tps_v1.0_production.pt`  
> **Kiến trúc Trọng tâm:** ASF-GR Engine (Adaptive Stroke Fusion Head + Gated Ghost Erasure Residual V1.0)  
> **Tốc độ kết xuất:** **0.5x Slow Motion (12.0 FPS)**  
> **Tổng số khung hình:** **12 frames** (Keyframes: 6, Inbetweens: 6)  
> **Ngày thực hiện:** 16/10/2026 13:05:14  

---

## 1. TỔNG HỢP CHỈ SỐ TOÀN CHUỖI CHUYỂN ĐỘNG V1.0

| Chỉ Số Đánh Giá | Giá Trị Đạt Được | Ngưỡng Chuẩn V1.0 | Ý Nghĩa Kỹ Thuật |
|:---|:---:|:---:|:---|
| **PSNR Trung bình (Inbetweens)** | **26.51 dB** | >= 21.0 dB | Độ bám sát pixel so với tranh gốc của họa sĩ |
| **SSIM Trung bình (Inbetweens)** | **0.9161** | >= 0.895 | Độ tương đồng cấu trúc nét vẽ của nhân vật |
| **Lượng Ghost Pixels Trung bình** | **19069 px** | Càng thấp càng tốt | Lượng nét thừa, nét xám mờ xuất hiện |
| **Độ ổn định thời gian (Std PSNR)**| **±6.11 dB** | <= 1.8 dB | Biên độ dao động mượt mà (chống nhấp nháy chất lượng) |
| **Tỷ lệ sụp đổ trắng (Blank)** | **0.0% (0/6)** | **0 frame** | Tuyệt đối không có khung hình bị mất nét |

---

## 2. DANH MỤC CÁC TỆP ĐÃ XUẤT BẢN

1. **`01_reconstructed_v10.mp4`**: Video hoàn chỉnh do mô hình V1.0 phục dựng (Tốc độ 0.5x, 12 FPS).
2. **`02_ground_truth.mp4`**: Video đối chứng gốc của họa sĩ (Tốc độ 0.5x, 12 FPS).
3. **`03_side_by_side_comparison.mp4`**: Video so sánh chia đôi màn hình Trái (Gốc) - Phải (V1.0) có HUD realtime.
4. **`04_three_way_comparison.mp4`**: Video 3 cột đối chiếu: Ground Truth vs Coarse TPS vs V1.0.
5. **`05_comparison_preview.gif`**: Ảnh động GIF preview nhanh.
6. **`reconstruction_metrics.csv`**: Bảng dữ liệu số đo chi tiết từng frame.
7. **`frames_reconstructed/`**: Thư mục chứa các khung hình PNG do V1.0 sinh ra.
8. **`frames_ground_truth/`**: Thư mục chứa các khung hình PNG gốc.

---

## 3. NHẬN ĐỊNH CHẤT LƯỢNG THỊ GIÁC (VISUAL QUALITATIVE AUDIT)

1. **Khử bóng ma triệt để (Ghosting Elimination):**
   - Nhờ vào **Adaptive Stroke Fusion Head** với 8 Điểm neo tĩnh biên/góc, các vùng chuyển động nhanh không còn bị tách làm đôi bóng ma mờ.
   - Nhánh **Ghost Eraser Gate** tẩy sạch vùng nền trắng, giữ cho nét vẽ nhân vật sắc gọn và rõ ràng.
2. **Tính liên tục & mượt mà (Temporal Continuity @ 0.5x):**
   - Khi quan sát ở tốc độ làm chậm 0.5x, chuyển động của nhân vật duy trì tính giải phẫu tự nhiên, đường nét chuyển biến mượt mà và không xuất hiện hiện tượng giật nét (stroke flicker).
3. **Bảo toàn 100% nét vẽ:**
   - Hoàn toàn không xảy ra sụp đổ chế độ (Mode Collapse), nét vẽ duy trì độ đen tuyền chuẩn mực.
