# BÁO CÁO ĐỐI CHỨNG: KẾT QUẢ NỘI SUY TPS-INBETWEEN TRÊN DATASET ATD-12K

> **Thử nghiệm:** Nạp 2 cặp Keyframe mẫu từ dataset mới tạo vào mô hình TPS-Inbetween gốc (chưa fine-tune) để tạo inbetween frame và đối chứng với Ground Truth thực tế.

---

## 1. Bảng Thông Số Đo Đạc & Chất Lượng Đối Chứng

| Chỉ số | Ví dụ 1 (Triplet #10) | Ví dụ 2 (Triplet #132) | Pipeline Cũ (ChayGo 0.5s) |
|:---|:---:|:---:|:---:|
| **Keyframe 0 ($I_0$)** | `000030_smooth.png` | `000237_smooth.png` | `0.0s.png` |
| **Ground Truth ($I_{GT}$)** | `000033_smooth.png` | `000238_smooth.png` | `0.1s.png` |
| **Keyframe 1 ($I_1$)** | `000036_smooth.png` | `000240_smooth.png` | `0.5s.png` |
| **Bước nhảy frame ($\Delta F$)** | **6 frames** ($0.250s$) | **3 frames** ($0.125s$) | ~12 frames ($0.500s$) |
| **Mốc thời gian $t$** | **$t = 0.500$** (Chính giữa) | **$t = 0.333$** (Lệch nhịp 1/3) | Gán cứng $0.2$ |
| **Độ dời trung bình (Mean Disp)** | **3.83 px** | **2.39 px** | $20.10$ px (Max $178.9$ px) |
| **Số điểm inliers sạch** | **185 điểm** | **354 điểm** | Bắt cặp sai nhiều |
| **PSNR so với Ground Truth** | **13.49 dB** | **16.80 dB** | $10.24 \sim 11.44$ dB |
| **SSIM so với Ground Truth** | **0.7848** | **0.9118** | $\sim 0.6880$ |
| **Chất lượng nét vẽ** | **Nét sạch, không xé rách, không ghosting** | **Nét mượt, bảo toàn cấu trúc** | Rách nét, bóng ma kép |

---

## 2. Danh Sách File Lưu Trữ Trong Từng Thư Mục

### Thư mục `Example_1_Triplet_10/`:
- `01_keyframe0_I0.png`: Keyframe bắt đầu ($I_0$).
- `02_ground_truth_IGT.png`: Frame Ground Truth thực tế của họa sĩ ($I_{GT}$).
- `03_predicted_inbetween_tps.png`: **Frame Inbetween do mô hình TPS-Inbetween sinh ra** ($I_{pred}$).
- `04_keyframe1_I1.png`: Keyframe kết thúc ($I_1$).
- `05_comparison_side_by_side.png`: **Ảnh ghép so sánh 2 hàng** trực quan giữa $I_{GT}$ và $I_{pred}$.
- `06_error_heatmap.png`: Bản đồ nhiệt biểu thị sai lệch tuyệt đối $|I_{pred} - I_{GT}|$.

### Thư mục `Example_2_Triplet_132/`:
- `01_keyframe0_I0.png`: Keyframe bắt đầu ($I_0$).
- `02_ground_truth_IGT.png`: Frame Ground Truth thực tế của họa sĩ ($I_{GT}$).
- `03_predicted_inbetween_tps.png`: **Frame Inbetween do mô hình TPS-Inbetween sinh ra** ($I_{pred}$).
- `04_keyframe1_I1.png`: Keyframe kết thúc ($I_1$).
- `05_comparison_side_by_side.png`: **Ảnh ghép so sánh 2 hàng** trực quan giữa $I_{GT}$ và $I_{pred}$.
- `06_error_heatmap.png`: Bản đồ nhiệt biểu thị sai lệch tuyệt đối $|I_{pred} - I_{GT}|$.

---

## 3. Thử Nghiệm Chuỗi Cắt Cảnh Liên Tục 7.92s (`Continuous_Scene_Demo_8s/`)

Đã nạp liên tục 64 keyframe liên tiếp trong phân cảnh hành động dài 7.92s (191 frames ở 24fps) để mô hình sinh ra 63 frame inbetween hoàn chỉnh:
- **`01_tps_interpolated.mp4`**: Video 24fps hoàn chỉnh do TPS-Inbetween tạo ra.
- **`02_ground_truth.mp4`**: Video 24fps thực tế của họa sĩ.
- **`03_side_by_side_comparison.mp4`**: Video đối chiếu song song Split-Screen kèm hiển thị PSNR/SSIM trực tiếp trên từng frame.
- **`04_comparison_preview.gif`**: Ảnh động preview nhanh.
- **`CONTINUOUS_DEMO_REPORT.md`**: Báo cáo chi tiết toàn bộ 63 inbetweens (PSNR trung bình: **16.73 dB**, SSIM trung bình: **0.9091**).


