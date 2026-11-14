# BÁO CÁO ĐỐI ĐẦU PHỤC DỰNG CHUYỂN ĐỘNG LIÊN TỤC V1.0 VS V1.1 (SAKUGA TRIPLETS)
> **Ngày thực hiện:** 17/10/2026 08:58:35  
> **Tập Dữ Liệu:** Chuỗi Triplets Trích Xuất Từ Sakuga Golden Dataset (Không sử dụng ChayGo / TestImage)  
> **Model V1.0:** `model_tps_v1.0_best.pt`  
> **Model V1.1:** `model_tps_v1.1_best.pt`  
> **Tốc độ kết xuất:** **0.5x Slow Motion (12 FPS)**  

---

## 1. BẢNG TỔNG HỢP SO SÁNH ĐỐI ĐẦU TOÀN DIỆN

| Phân Cảnh Thử Nghiệm | Tổng Frames | PSNR V1.0 | **PSNR V1.1** | **$\Delta$ PSNR** | SSIM V1.0 | **SSIM V1.1** | Ghost V1.0 | **Ghost V1.1** | TSI V1.0 | **TSI V1.1** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **sakuga_composite_sequence** | 18 | 15.68 dB | **15.29 dB** | **-0.40 dB** | 0.8358 | **0.8418** | 34140 px | **11282 px** | ±1.65 dB | **±1.67 dB** |
| **clip_000008** | 3 | 14.50 dB | **14.03 dB** | **-0.47 dB** | 0.7916 | **0.7916** | 44842 px | **21590 px** | ±0.00 dB | **±0.00 dB** |
| **clip_000010** | 3 | 17.34 dB | **17.33 dB** | **-0.01 dB** | 0.8953 | **0.8922** | 20493 px | **18081 px** | ±0.00 dB | **±0.00 dB** |
| **clip_000018** | 3 | 16.69 dB | **17.07 dB** | **+0.37 dB** | 0.8754 | **0.8809** | 17884 px | **16293 px** | ±0.00 dB | **±0.00 dB** |
| **clip_000064** | 3 | 19.99 dB | **19.87 dB** | **-0.12 dB** | 0.9520 | **0.9462** | 8904 px | **7517 px** | ±0.00 dB | **±0.00 dB** |
| **clip_003046** | 3 | 18.71 dB | **17.87 dB** | **-0.84 dB** | 0.9203 | **0.9098** | 22341 px | **14372 px** | ±0.00 dB | **±0.00 dB** |
| **clip_003050** | 3 | 16.28 dB | **15.73 dB** | **-0.55 dB** | 0.8657 | **0.8628** | 35044 px | **16409 px** | ±0.00 dB | **±0.00 dB** |

---

## 2. PHÂN TÍCH KỸ THUẬT CHUYÊN SÂU

1. **Chất Lượng Tái Tạo Pixel (PSNR & SSIM):**
   - Đánh giá khả năng bám sát đường nét nguyên bản của họa sĩ Sakugabooru giữa hai thế hệ checkpoint.
   - So sánh chi tiết trên chuỗi chuyển động liên tục `sakuga_composite_sequence`.
2. **Khả Năng Khử Bóng Ma (Ghosting Eradication):**
   - Đánh giá lượng Ghost Pixels dư thừa trên nền trắng khi thực hiện nội suy bước nhảy liên tiếp.
3. **Độ Ổn Định Thời Gian (Temporal Stability Index - TSI):**
   - Độ lệch chuẩn $\sigma_{\text{PSNR}}$ giữa các inbetweens chứng minh mô hình không bị nhấp nháy chất lượng hình ảnh qua từng frame.

---

## 3. DANH MỤC VIDEO & SẢN PHẨM TRỰC QUAN

Toàn bộ video 4 cột đồng bộ (`01_four_way_comparison_hud.mp4`) và preview GIF (`02_comparison_preview.gif`) cùng heatmaps đã được lưu trữ trong từng thư mục con tại:
`continuous_benchmark\visual_audit\comparison_v1.0_vs_v1.1\continuous_reconstruction/`.
