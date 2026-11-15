# PHỤC DỰNG PHÂN CẢNH CHUYỂN ĐỘNG NHÂN VẬT LIÊN TỤC V1.0
> **Phiên bản:** TPS-Inbetween Fine-tuned V1.0 (Sakuga Golden Production Candidate)  
> **Cập nhật:** 16/10/2026 13:05:14  

## 1. TỔNG QUAN KẾT QUẢ CÁC PHÂN CẢNH

| Phân Cảnh | Tổng Khung Hình | PSNR Trung Bình | SSIM Trung Bình | Ghost Pixels Trung Bình | Độ Ổn Định (Std PSNR) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **ChayGo** | 72 frames | **18.90 dB** | **0.7049** | **0 px** | ±0.11 dB |
| **TestImage1** | 12 frames | **26.51 dB** | **0.9161** | **19069 px** | ±6.11 dB |

## 2. DANH MỤC THƯ MỤC CON

- [`ChayGo/`](ChayGo/): Phân cảnh chuyển động thân thể biên độ lớn (chạy nước rút, vạt áo bay, bước sải dài).
- [`TestImage1/`](TestImage1/): Phân cảnh diễn hoạt khuôn mặt và biểu cảm (quay đầu, chớp mắt, khẩu hình miệng, tóc bay).
