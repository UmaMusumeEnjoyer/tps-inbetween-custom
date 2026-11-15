# KẾ HOẠCH HUẤN LUYỆN TINH CHỈNH PHIÊN BẢN v0.2 (FINETUNE PLAN v0.2)
> **Mô hình Trọng tâm:** TPS-Inbetween (*Thin-Plate Spline-based Interpolation for Animation Line Inbetweening*)  
> **Phiên bản:** v0.2 (Production-Grade Anti-Collapse Finetune)  
> **Nền tảng Huấn luyện:** Kaggle Cloud GPU (NVIDIA Tesla T4 x2 hoặc P100 16GB VRAM)  
> **Ngày khởi tạo:** 07/10/2026  
> **Kế thừa & Khắc phục từ:** [01_finetune_plan_v0.1.md](file:///e:/CODE/PBL/PBL6/docs/finetune_plans/v0.1/01_finetune_plan_v0.1.md)

---

## 1. HẬU KIỂM TOÁN v0.1 & NGUYÊN NHÂN SỤP ĐỔ CHẾ ĐỘ (POST-MORTEM ANALYSIS)

Trong đợt nghiệm thu checkpoint `model_tps_v0.1_best.pt`, hệ thống kiểm toán pixel thực tế phát hiện toàn bộ ảnh sinh ra đều là **nền trắng tinh 100% (cường độ pixel = 255)**, tỷ lệ Stroke Recall $= 0.0\%$.

### 1.1. Ba Lỗi Kỹ Thuật Chí Mạng trong Pipeline v0.1
1. **Lệch hệ tọa độ trong hàm `backward_warp` (Coordinate Mismatch):**
   - Luồng biến dạng thô TPS tính trong không gian chuẩn hóa $[0, 1]$, trong khi luồng quang học tinh chỉnh từ IFNet xuất ra theo đơn vị Pixel thực (khoảng $\pm 3.5$px).
   - Hàm `backward_warp` cũ cộng trực tiếp pixel displacement vào lưới $[-1, 1]$ mà không chia cho $(W/2, H/2)$, khiến tọa độ lấy mẫu vọt ra khỏi phạm vi ảnh $[2.5, 4.5]$ và bị kẹp vào viền trắng (border padding).
2. **Hàm mất mát EDT Loss đơn hướng (Unilateral Loss Shortcut):**
   - Công thức cũ: $\mathcal{L}_{EDT} = \text{mean}(pred \times EDT_{target}^2)$.
   - Hàm này chỉ phạt khi mô hình vẽ nét sai vị trí ($pred > 0$), nhưng cho điểm 0 tuyệt đối khi mô hình **không vẽ gì ($pred = 0$)**. Do nền trắng chiếm $> 95\%$ ảnh anime, mạng neural học được đường tắt (shortcut) xóa sạch nét vẽ để giảm loss về gần 0.
3. **Công thức Residual của Synthesis UNet bị âm cực đại:**
   - Công thức cũ: $res = \text{Sigmoid}(x) \times 2.0 - 1.0 \in [-1.0, 1.0]$.
   - Mạng UNet học cách đưa output về 0 để $res \to -1.0$. Khi đó, $\text{clamp}(base - 1.0, 0, 1) = 0.0$, biến UNet thành một chiếc "tẩy trắng" xóa sạch khung hình nội suy TPS.

### 1.2. Bẫy Chỉ Số (Phantom Metric Trap)
- PSNR đạt $15.11$ dB vì 95% diện tích nền trắng khớp hoàn hảo.
- Ghost Pixels $= 0$ vì không có pixel nào $\le 250$.
- Chamfer Distance cũ trả về $0.0$ khi không có nét vẽ nào.
$\implies$ Checkpoint v0.1 đã bị thu hồi hoàn toàn khỏi danh sách ứng viên sản phẩm.

---

## 2. KIẾN TRÚC MÔ HÌNH VÀ CƠ CHẾ NỘI SUY MỚI TRONG v0.2

### 2.1. Phân Tách Hai Lớp Warp Chuẩn Hóa Riêng Biệt
- **`warp_normalized_flow(img, flow)`**:
  - Dành riêng cho Coarse TPS Flow (tọa độ $[0, 1]$).
  - Công thức lưới: `grid = (grid_01 + flow) * 2.0 - 1.0` (ánh xạ chuẩn xác về $[-1, 1]$ của PyTorch `grid_sample`).
- **`warp_pixel_flow(img, flow)`**:
  - Dành riêng cho Fine Optical Flow từ IFNet (tọa độ pixel thực).
  - Chuẩn hóa: `flow_norm = [flow_x / ((W-1)/2), flow_y / ((H-1)/2)]`.
  - Công thức lưới: `grid = grid_neg1_to_1 + flow_norm`.

### 2.2. Tái Cấu Trúc Khối Synthesis UNet Residual (Khống Chế Biên Độ Tuyệt Đối)
Thay thế hoàn toàn cơ chế Sigmoid cũ bằng hàm **Bounded Tanh Residual** được khởi tạo trọng số 0:
```python
class SynthesisUnet_v2(nn.Module):
    def __init__(self, indim=9, outdim=1, c=16):
        super().__init__()
        # ... Downsampling và Upsampling giữ nguyên ...
        self.conv = nn.Conv2d(c, outdim, 3, 1, 1)
        # Khởi tạo trọng số lớp cuối bằng 0 tuyệt đối
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, ...):
        # ...
        # Giới hạn biên độ thay đổi tối đa trong khoảng [-0.15, +0.15]
        res = torch.tanh(self.conv(x)) * 0.15
        return res
```
- **Ý nghĩa sống còn:** Ngay cả khi UNet gặp bất kỳ ngoại lệ nào, nó **không bao giờ có thể triệt tiêu quá 15% cường độ nét vẽ** của khung hình biến dạng Coarse TPS gốc ($base \ge 0.8 \implies pred \ge 0.65$). Nét vẽ được bảo toàn 100%.

---

## 3. THIẾT KẾ HÀM MẤT MÁT CHỐNG SỤP ĐỔ CHẾ ĐỘ (ANTI-COLLAPSE LOSS DESIGN)

Tổng hàm mất mát trong v0.2:
$$\mathcal{L}_{total} = \mathcal{L}_{L1} + 0.15 \mathcal{L}_{VGG} + 0.40 \mathcal{L}_{Dice} + 0.20 \mathcal{L}_{EDT\_bi} + 0.10 \mathcal{L}_{Density}$$

### 3.1. Stroke Dice Loss ($\mathcal{L}_{Dice}$ - Trọng số 0.40)
Ngăn chặn triệt để hiện tượng mất nét vẽ:
$$\mathcal{L}_{Dice}(P, G) = 1.0 - \frac{2 \sum (P \cdot G) + \epsilon}{\sum P^2 + \sum G^2 + \epsilon}$$
- Nếu mô hình dự đoán $P = 0$ (trắng tinh): $\mathcal{L}_{Dice} = 1.0$ (phạt tối đa!).
- Mạng neural bắt buộc phải vẽ đầy đủ nét của nhân vật thì Dice Loss mới giảm.

### 3.2. Bidirectional EDT Loss ($\mathcal{L}_{EDT\_bi}$ - Trọng số 0.20)
$$\mathcal{L}_{EDT\_bi} = \text{mean}(P \times EDT_{G}^2) + \text{mean}(G \times EDT_{P}^2)$$
- Thành phần 1: Phạt nét vẽ sai vị trí.
- Thành phần 2: Phạt các nét vẽ trên Ground Truth bị mô hình bỏ quên!

### 3.3. Stroke Density Conservation Loss ($\mathcal{L}_{Density}$ - Trọng số 0.10)
$$\mathcal{L}_{Density} = \left| \text{mean}(P) - \text{mean}(G) \right|$$
- Bảo toàn tổng lượng mực vẽ của khung hình trung gian.

---

## 4. BỘ QUẢN LÝ CHECKPOINT THÔNG MINH v0.2 (ANTI-CHEAT CHECKPOINT MANAGER)

Hệ thống `SmartCheckpointManager_v2` bổ sung rào chắn kiểm toán nghiêm ngặt:
1. **Rào chắn cứng (Hard Gate):**
   - $\text{Stroke Recall} \ge 70.0\%$ (tỷ lệ nét GT được dự đoán phủ đúng trong bán kính 5px).
   - Nếu $\text{Stroke Recall} < 70.0\%$, checkpoint bị **LOẠI BỎ NGAY LẬP TỨC**, bất kể điểm PSNR cao đến mức nào.
2. **Điểm tổng hợp xếp hạng (Composite Quality Score):**
   $$\text{Score} = \text{PSNR} + 20 \times \text{SSIM} + 25 \times \text{StrokeRecall} - 0.0002 \times \text{GhostPixels}$$
3. **Cảnh báo trực quan Live Monitoring:**
   - Trong mỗi epoch, in chi tiết `Pred Stroke %` vs `GT Stroke %` để người dùng giám sát nét vẽ từng giây.

---

## 5. THÔNG SỐ HUẤN LUYỆN (HYPERPARAMETERS)

| Tham Số | Giá Trị v0.2 | Ghi Chú |
|:---|:---:|:---|
| **Batch Size per GPU** | 4 | Batch thực tế với Gradient Accumulation: $4 \times 2 = 8$ |
| **Gradient Accumulation** | 2 | Ổn định gradient trên GPU 16GB |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999$, weight_decay=1e-4) | Tối ưu hóa suy giảm trọng số |
| **Learning Rate** | $1.5 \times 10^{-4}$ | Warmup 2 epochs, Cosine Annealing về $1.0 \times 10^{-5}$ |
| **Total Epochs** | 25 Epochs | Đủ để hội tụ IFNet và Bounded UNet |
| **Keypoints Clamping** | Tối đa 80 điểm | Rút ngắn thời gian giải TPS, chống nhiễu gấp nếp |
| **AMP Precision** | FP16 Enabled (`torch.amp.autocast`) | Tiết kiệm 45% VRAM |

---

## 6. TIÊU CHUẨN NGHIỆM THU CHÍNH THỨC (ACCEPTANCE CRITERIA v0.2)

1. **Stroke Recall:** $\ge 80.0\%$ trên toàn bộ 508 val triplets (Tuyệt đối không có bất kỳ mẫu nào bị trắng).
2. **PSNR:** $\ge 18.0$ dB (Đo lường thật sự trên cấu trúc nét, không bị chi phối bởi nền).
3. **SSIM:** $\ge 0.85$.
4. **Chamfer Distance:** $\le 0.005$ (Không có mẫu nào bị phạt 1.0).
5. **Ghost Pixels:** Giảm tối thiểu $50\%$ so với Coarse TPS Warp mà vẫn giữ nguyên độ mảnh của nét vẽ.
