# BÁO CÁO ĐỐI CHUẨN NGHIỆM THU TPS-INBETWEEN FINETUNED v0.1 (CẬP NHẬT KIỂM TOÁN)
> **Ngày cập nhật:** 07/10/2026 12:11:48  
> **Model Checkpoint:** `model_tps_v0.1_best.pt`  
> **Tập dữ liệu:** `triplets.csv` (Split: `val`, Quy mô: 15 Triplets)  
> **Kết luận chính thức:** **❌ KHÔNG ĐẠT NGHIỆM THU - PHÁT HIỆN SỤP ĐỔ CHẾ ĐỘ ẢNH TRẮNG (MODE COLLAPSE)**

---

## 1. PHÁT HIỆN KIỂM TOÁN CHÍNH (CRITICAL AUDIT FINDINGS)

Trong đợt nghiệm thu ban đầu, mô hình hiển thị chỉ số PSNR $\approx 15.11$ dB và Ghost Pixels $= 0$. Tuy nhiên, khi kiểm toán pixel thị giác trực tiếp (Visual Audit):
1. **Sụp đổ chế độ (Mode Collapse):** Toàn bộ 15 ảnh dự đoán của checkpoint v0.1 đều là **nền trắng tinh 100% (cường độ pixel = 255)**, số lượng pixel nét vẽ bằng 0.
2. **Nguyên nhân chỉ số bị đánh lừa (Phantom Metric Trap):**
   - Hơn 95% diện tích tranh hoạt hình là nền trắng, do đó dự đoán toàn bộ màu trắng vẫn đạt PSNR $\approx 15.1$ dB vì sai số chỉ nằm ở 5% diện tích nét vẽ.
   - Ghost pixels công thức `(pred >= 180 & pred <= 250 & gt > 250)` trả về 0 vì không có pixel nào $\le 250$.
   - Chamfer Distance trong code cũ trả về 0.0 khi ảnh rỗng nét (`p_mask` empty), tạo ra ảo tưởng về sự hoàn hảo.
3. **Nguyên nhân kỹ thuật trong quá trình train:**
   - **Lỗi Warp Layer:** `backward_warp` cộng trực tiếp pixel optical flow vào tọa độ $[-1, 1]$ khiến việc lấy mẫu vọt ra viền ngoài.
   - **Lỗi EDT Loss đơn hướng:** $\mathcal{L}_{EDT} = pred \times EDT_{target}^2$ chỉ phạt nét vẽ sai nhưng thưởng cho việc không vẽ gì ($pred = 0 \implies loss = 0$).
   - **Lỗi Residual UNet:** $res = \text{Sigmoid} \times 2 - 1$ bị UNet kéo về $-1.0$, kết hợp với `clamp(base - 1, 0, 1)` gọt sạch toàn bộ nét vẽ thành 0.

---

## 2. BẢNG SO SÁNH ĐỐI CHUẨN ĐỊNH LƯỢNG CHÍNH XÁC

| Chỉ Số Đánh Giá | TPS Coarse Warp (Chuẩn) | Finetuned v0.1 (Model Thực Tế) | Trạng Thái Kiểm Toán |
|:---|:---:|:---:|:---:|
| **Tỷ lệ ảnh sụp đổ trắng (Blank Collapse)** | **0.0%** | **100.0% (15/15)** | ❌ Sụp đổ hoàn toàn |
| **Độ phủ nét vẽ (Stroke Recall)** | **87.7%** | **0.0%** | ❌ Mất sạch nét vẽ |
| **Chamfer Distance (Có phạt mất nét)** | **0.002087** | **1.000000** | ❌ Phạt tối đa (1.0) |
| **PSNR Toàn Diện** | **21.35 dB** | **16.77 dB** | ⚠️ Bẫy nền trắng 95% |
| **SSIM** | **0.8818** | **0.8441** | ⚠️ Bị chi phối bởi nền |
| **Tốc độ suy luận trung bình** | - | **1332.0 ms** | Đạt yêu cầu thời gian thực |

---

## 3. PHÂN TÍCH THEO MỐC THỜI GIAN NỘI SUY $t$

```
     t  psnr_count  psnr_mean  psnr_std  ssim_mean  ssim_std  stroke_recall_mean  chamfer_distance_mean  is_blank_collapse_mean
0.2500           3  15.178200  0.839093   0.811133  0.055029                 0.0                    1.0                     1.0
0.3333           2  17.208850  7.258663   0.781500  0.226416                 0.0                    1.0                     1.0
0.4000           1  12.954900       NaN   0.689600       NaN                 0.0                    1.0                     1.0
0.5000           9  17.620911  2.634365   0.886222  0.066709                 0.0                    1.0                     1.0
```

---

## 4. BẰNG CHỨNG THỊ GIÁC (VISUAL AUDIT EVIDENCE)
Các hình ảnh kiểm toán đã được cập nhật nhãn cảnh báo đỏ rõ ràng:
- Bảng so sánh 4 cột: `continuous_benchmark/visual_audit/finetune_v0.1/side_by_side_comparison/`
- Bản đồ nhiệt sai số: `continuous_benchmark/visual_audit/finetune_v0.1/difference_heatmaps/`
- Ảnh phóng to chi tiết: `continuous_benchmark/visual_audit/finetune_v0.1/zoomed_details/`

**Nhận định đối chứng:**
- **Cột TPS Coarse Warp:** Giữ trọn vẹn toàn bộ nét vẽ nhân vật (tóc, viền mắt, trang phục), chứng minh giải thuật TPS và GlueStick keypoints hoạt động rất tốt.
- **Cột Finetuned v0.1:** Bị xóa trắng hoàn toàn do mạng UNet residual bị âm cực đại.

---

## 5. HÀNH ĐỘNG TIẾP THEO (NEXT STEPS CHO PHIÊN BẢN v0.2)
1. **Thu hồi đề cử sản phẩm v0.1:** Không chuyển giao file `model_tps_v0.1_best.pt` vào production.
2. **Triển khai Finetune v0.2 trên Kaggle:**
   - Sửa hàm `backward_warp` tách biệt chuẩn pixel flow và normalized TPS flow.
   - Đổi EDT Loss sang **Bidirectional Chamfer Loss + Dice Stroke Loss** để bắt buộc mô hình phải duy trì nét vẽ.
   - Đổi Residual thành $res = \tanh(x) \times 0.15$ khởi tạo trọng số 0 để không thể triệt tiêu nét warp gốc.
   - Thêm cơ chế Stroke Recall Guard $\ge 70\%$ trong `SmartCheckpointManager`.
