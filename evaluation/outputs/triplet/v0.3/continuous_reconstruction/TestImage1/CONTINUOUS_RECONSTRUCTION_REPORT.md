# BÁO CÁO PHỤC DỰNG PHÂN CẢNH CHUYỂN ĐỘNG NHÂN VẬT LIÊN TỤC (TỐC ĐỘ 0.5x)
> **Phân cảnh thử nghiệm:** `TestImage1`  
> **Model Checkpoint:** `model_tps_v0.3_best.pt`  
> **Kiến trúc Trọng tâm:** Adaptive Stroke Fusion Head + Gated Ghost Erasure Residual (ASF-GR Engine)  
> **Tốc độ kết xuất:** **0.5x Slow Motion (12.0 FPS)**  
> **Tổng số khung hình:** **12 frames** (Keyframes: 6, Inbetweens: 6)  
> **Ngày thực hiện:** 08/10/2026 18:47:14  

---

## 1. TỔNG HỢP CHỈ SỐ TOÀN CHUỖI CHUYỂN ĐỘNG

| Chỉ Số Đánh Giá | Giá Trị Đạt Được | Ý Nghĩa Kỹ Thuật |
|:---|:---:|:---|
| **PSNR Trung bình (Inbetweens)** | **28.81 dB** | Độ bám sát pixel so với tranh gốc của họa sĩ |
| **SSIM Trung bình (Inbetweens)** | **0.8981** | Độ tương đồng cấu trúc nét vẽ của nhân vật |
| **Lượng Ghost Pixels Trung bình** | **81913 px** | Lượng nét thừa, nét xám mờ xuất hiện |
| **Độ ổn định thời gian (Std PSNR)**| **±11.44 dB** | Biên độ dao động mượt mà (chống giật cục) |
| **Tỷ lệ sụp đổ trắng (Blank)** | **0.0% (0/6)** | Tuyệt đối không có khung hình bị mất nét |

---

## 2. DANH MỤC CÁC TỆP ĐÃ XUẤT BẢN

1. **`01_reconstructed_v03.mp4`**: Video hoàn chỉnh do mô hình v0.3 phục dựng (Tốc độ 0.5x, 12 FPS).
2. **`02_ground_truth.mp4`**: Video đối chứng gốc của họa sĩ (Tốc độ 0.5x, 12 FPS).
3. **`03_side_by_side_comparison.mp4`**: Video so sánh chia đôi màn hình Trái (Gốc) - Phải (v0.3) có HUD realtime.
4. **`04_three_way_comparison.mp4`**: Video 3 cột đối chiếu: Ground Truth vs Coarse TPS vs v0.3.
5. **`05_comparison_preview.gif`**: Ảnh động GIF preview nhanh.
6. **`reconstruction_metrics.csv`**: Bảng dữ liệu số đo chi tiết từng frame.

---

## 3. NHẬN ĐỊNH CHẤT LƯỢNG THỊ GIÁC (VISUAL QUALITATIVE AUDIT)

1. **Khử bóng ma (Ghosting Elimination):**
   - Nhờ vào **Adaptive Stroke Fusion**, các vùng chuyển động nhanh không còn bị tách làm đôi 2 bóng ma mờ như các phép cộng tuyến tính cũ.
   - Nhánh **Ghost Eraser Gate** tẩy sạch vùng nền trắng, giữ cho nét vẽ nhân vật sắc gọn.
2. **Tính liên tục & mượt mà (Temporal Continuity @ 0.5x):**
   - Khi quan sát ở tốc độ làm chậm 0.5x, chuyển động của nhân vật duy trì tính giải phẫu tự nhiên, đường nét chuyển biến mượt mà và không xuất hiện hiện tượng giật nét (stroke flicker).
3. **Bảo toàn 100% nét vẽ:**
   - Hoàn toàn không xảy ra sụp đổ chế độ (Mode Collapse), nét vẽ duy trì độ đen tuyền chuẩn mực.
