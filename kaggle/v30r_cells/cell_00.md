# HUẤN LUYỆN TINH CHỈNH TPS-INBETWEEN PHIÊN BẢN V3.0 REMASTERED (R2)
## ĐỀ TÀI: NGHIÊN CỨU & TINH CHỈNH MÔ HÌNH TPS-INBETWEEN CHO HOẠT HỌA ANIME 2D (PBL6)

> **Mục tiêu & Đột Phá Kỹ Thuật Cốt Lõi Phiên Bản V3.0-Remastered R2 (Production Release):**
> 1. **Khôi Phục Trọn Vẹn 168 Tensors & Cực Tiểu Phẳng V2.6 (Cell 5, 7):**
>    - Kế thừa 100% bồn cực tiểu phẳng (Flat Minima) của V2.6 Gold (`model_tps_v2.6_best.pt`, `Recall = 92.42%`, `Density = 1.10x`).
>    - Khôi phục mô-đun `AdaptiveStrokeFusionHead` (`self.asf_head`) và hệ số `detail_add * 0.25` ở `SynthesisUnet`, bảo đảm 168/168 tensors khớp tuyệt đối ngay từ Epoch 0.
> 2. **Kích Hoạt 100% Trường Chuyển Động Thô TPS Cache 0-Copy (Cell 2, 3, 4, 10):**
>    - Khắc phục triệt để lỗi thiếu trường TPS (`coarse_flow = 0`) ở bản R1 bằng cách gắn kết bộ đệm `matches/*.npz` đa luồng sang NVMe SSD (`/tmp/sakuga_ssd_v30r`) và ánh xạ `.npy` mmap 0-copy.
>    - Tích hợp chốt chặn bắt buộc **Epoch 0 Sanity Check** đảm bảo `TPS Valid >= 95%` và `Stroke Recall >= 91.5%` trước khi bước vào huấn luyện.
> 3. **Bộ Chỉ Số Định Lượng Mới Thay Thế SSIM (WCD + 1D-Sliced EMD + Topology) (Cell 8, 9):**
>    - Loại bỏ hoàn toàn SSIM khỏi hàm chấm điểm chọn Best Checkpoint (do SSIM bị thiên lệch bởi 95% diện tích nền trắng).
>    - Thẩm định bằng **Weighted Chamfer Distance (`WCD`)** và **1D-Sliced EMD** chuẩn bài báo gốc TPS (*Zhu et al., HIT 2024*), kết hợp **Topology Skeleton Recall** và **Hair-Tip Recall** đo trực tiếp độ liền mạch khung xương 1 px và ngọn tóc vuốt nhọn.
> 4. **Hệ Thống Loss Liên Tục Không Giật Cục (Continuous Loss Engine) (Cell 6):**
>    - Xóa bỏ nhánh rẽ gián đoạn ở Epoch 3; duy trì hàm mục tiêu liên tục từ Epoch 1:
>      $$\mathcal{L} = 1.0\,\mathcal{L}_{\text{Charb}} + 0.35\,\mathcal{L}_{\text{DT}} + 0.20\,\mathcal{L}_{\text{Lap}} + 0.30\,\mathcal{L}_{\text{Tversky-clDice}} + 0.25\,\mathcal{L}_{\text{FineTip}} + 0.08\,\mathcal{L}_{\text{STE-Dens}} + 0.05\,\mathcal{L}_{\text{FBFC-Ghost}}$$
>    - `Truncated Distance Transform Loss` ($\mathcal{L}_{\text{DT}}$) tạo mặt dốc dẫn hướng hình học liên tục trên không gian thưa; `FineStrokeAndTipLoss` bảo vệ triệt để ngọn tóc và nét mảnh không bị bào mòn.
> 5. **Soft-Knee Gamma RGBA Exporter (Triệt Tiêu 100% Nền Vàng, Giữ Rõ Ngọn Tóc) (Cell 5, 8, 10):**
>    - Điểm gãy nền siêu thấp $\tau_{\text{noise}} = 0.020$: mọi nhiễu quét giấy ngoài nhân vật có `Alpha = 0` tuyệt đối, `RGB = (0, 0, 0)`.
>    - Đường cong chuyển tiếp mềm `smoothstep` kết hợp $\gamma = 0.78$ giúp ngọn tóc mảnh ($p = 0.06$) đạt `Alpha = 31` (sắc nét tự nhiên) thay vì bị tới $98.4\%$ trong suốt như công thức cũ.
> 6. **Chiến Lược Differential LR & Hạ Tầng Multi-GPU DDP (Cell 1, 10):**
>    - `1.0e-5` cho backbone đã hội tụ (`encoder, IFNet`); `3.0e-5` cho nhánh tổng hợp nét (`asf_head, unet`).
>    - Cosine Annealing 16 Epochs (`~55-65 phút` trên 2x T4), tốc độ cực nhanh với bộ đệm NVMe SSD.
