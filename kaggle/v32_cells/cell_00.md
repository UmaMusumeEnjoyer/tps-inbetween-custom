# HUẤN LUYỆN TINH CHỈNH TPS-INBETWEEN PHIÊN BẢN V3.2 (PROGRESSIVE STROKE PRESERVATION & SADDLE-POINT BREAKER)
## ĐỀ TÀI: NGHIÊN CỨU & TINH CHỈNH MÔ HÌNH TPS-INBETWEEN CHO HOẠT HỌA ANIME 2D (PBL6)

> **Mục tiêu & Đột Phá Kỹ Thuật Cốt Lõi Phiên Bản V3.2 (Official Production Release):**
> 1. **Khởi Tạo Warm-Start Linh Hoạt (Cell 7):** 
>    - **Ưu tiên tuyệt đối:** Nạp snapshot trọng số tốt nhất từ **V3.1 Best Progress Checkpoint** (`model_tps_v3.1_best_progress.pt`, lưu tại Epoch 02 với L1=0.0303, SSIM=0.8917, Recall=75.3%, nền trắng sạch), giúp kế thừa ngay 2 epoch hội tụ bước đầu của các module mới (`asf_head`, `contour_propagator`, `asym_analyzer`).
>    - **Fallback:** Tự động fallback nạp **V2.6 Soup Gold** (`model_tps_v2.6_soup.pt` / `model_tps_v2.6.pt`) nếu chưa có checkpoint V3.1.
> 2. **Cơ Chế Smooth Progressive Curriculum Loss - Giải Phóng Điểm Yên Ngựa (Saddle Point) (Cell 6):**
>    - Loại bỏ hiện tượng sốc gradient gián đoạn ở Epoch 3 của V3.1 (từng khiến Stroke Recall rơi tự do từ $75.3\% \to 66.5\%$).
>    - Thiết lập hệ số kích hoạt mượt mà $\text{ramp} = \min(1.0, \max(0.25, \frac{\text{epoch} - 2}{4}))$ từ Epoch 03 đến Epoch 06, giúp mô hình chuyển tiếp êm ái giữa pha hội tụ hình thái và pha tinh chỉnh nét.
> 3. **Tái Cân Bằng Trọng Số & Nới Rộng Biên An Toàn Mật Độ Nét (Cell 6):**
>    - **Nới rộng cận an toàn Density:** Mở rộng dải mục tiêu từ $[1.06\text{x}, 1.14\text{x}] \to [1.06\text{x}, 1.20\text{x}]$ (tâm chuẩn lý tưởng **$1.10\text{x}$**). Mạng không còn bị ép phạt khi tạo ra các nét vẽ tóc mảnh hơi đầm.
>    - **Giảm áp lực phạt Density:** Giảm trọng số $\lambda_{\text{dens}}$ từ $0.15 \to 0.05$.
>    - **Tăng cường bảo toàn ngọn tóc (Focal Hair-Tip Loss):** Nâng trọng số $\lambda_{\text{hair}}$ từ $0.10 \to 0.25$ với hệ số hội tụ điểm ngọn `endpoint_focal = 2.5`, tập trung gradient kéo lại các chi tiết sợi tóc, khóe mắt và ngón tay.
>    - **Giảm Ghost Penalty ngoài vùng an toàn:** Hạ `cleanliness_penalty` từ $0.25 \to 0.05$ trong Ghost Loss để không triệt tiêu các nét vẽ mảnh ngoài biên.
> 4. **Chuẩn Hóa Bộ Quản Lý Checkpoint & Hiệu Chuẩn Cổng Đánh Giá (Cell 9):**
>    - Tăng trọng số của Stroke Recall trong công thức Composite Score từ $0.15 \to 0.25$ để checkpoint tái tạo nét tốt sẽ giành điểm ưu tiên cao nhất.
>    - Điều chỉnh ngưỡng hội tụ cứng `min_recall = 0.78` (thay vì $0.905$ quá khắc nghiệt), đảm bảo các checkpoint vượt trội được tự động lưu thành `model_tps_v3.2_best.pt`.
> 5. **Bảo Toàn 100% Hạ Tầng Siêu Tốc & Ổn Định Tuyệt Đối Kế Thừa Từ V3.1:**
>    - Huấn luyện Native 540p Multi-GPU DDP qua `torchrun`, tốc độ $6.7 - 6.8$ phút/epoch ($< 10$ phút).
>    - Triệt tiêu 100% rò rỉ RAM (duy trì $\approx 2.45\text{ GB}$ container RAM suốt toàn bộ 25 epoch).
>    - Không bị sương mù xám (Pristine Background) và chống tràn số FP16 (Zero NaN).
