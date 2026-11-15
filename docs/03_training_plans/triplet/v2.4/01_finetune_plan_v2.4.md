# KẾ HOẠCH NÂNG CẤP & XÂY DỰNG NOTEBOOK FINETUNE TPS-INBETWEEN PHIÊN BẢN V2.4
## ĐỀ TÀI: NGHIÊN CỨU & TINH CHỈNH MÔ HÌNH TPS-INBETWEEN CHO HOẠT HỌA ANIME 2D (PBL6)

> **Tài liệu tham chiếu:**
> - Cấu trúc dự án: `PROJECT_STRUCTURE.md`
> - Notebook thực thi V2.3: `finetune-v2-3 (1).ipynb`
> - Nhật ký & Báo cáo Benchmark V2.3: `checkpoints/experiments/v2.3/training_history_v2.3.csv`, `checkpoints/experiments/v2.3/BENCHMARK_REPORT_V2.3.md`
> - Sản phẩm đích của kế hoạch: `kaggle/06_train_tps_inbetween_finetune_v2.4.ipynb`, `kaggle/finetune-v2-4.ipynb` và `kaggle/generate_v24_notebook.py`

---

## 1. CHẨN ĐOÁN KỸ THUẬT TỪ KẾT QUẢ THỰC THI & BENCHMARK V2.3

Qua phân tích mã nguồn và nhật ký chạy thực tế 16 epochs của `finetune-v2-3 (1).ipynb` cùng báo cáo `BENCHMARK_REPORT_V2.3.md`, có **4 nguyên nhân gốc rễ** dẫn đến các hạn chế hiện tại:

### 1.1. Tại sao phạt Density trong V2.3 quá nặng khiến mật độ nét tụt sâu (`0.6795` trên Benchmark, `0.729–0.799` trên Val)?
Trong **Cell 6** của V2.3, `StrokeDensityRegularizationLoss` được định nghĩa như sau:
```python
p_soft_mask = torch.sigmoid((p_ink - 0.15) * 25.0)
g_hard_mask = (g_ink > 0.15).float()
```
1. **Sai lệch toán học tại vùng nền trắng (Sigmoid Background Floor Bias):**
   - Với các pixel nền trắng hoàn toàn ($p_{\text{ink}} = 0.0$), hàm Sigmoid cho giá trị:
     $$\sigma((0.0 - 0.15) \times 25.0) = \sigma(-3.75) \approx 0.02298 > 0$$
   - Trên một khung hình crop $512 \times 512$ ($262,144\text{ px}$), riêng vùng nền trắng ($\approx 97\%$ diện tích ảnh) đã cộng dồn một lượng mực ảo cố định vào tử số:
     $$\text{Background Bias} \approx 255,000 \times 0.02298 \approx \mathbf{5,860\text{ px ảo}}$$
   - Trong khi đó, mẫu số `sum_g_mask` dùng ngưỡng cứng `(g_ink > 0.15).float()` nên vùng nền trắng bằng **$0.0\text{ px}$** (tổng diện tích nét thật chỉ khoảng $3,000 - 6,000\text{ px}$).
   - **Hệ quả:** Tỷ số `ratio_mask = sum_p_mask / sum_g_mask` bên trong hàm Loss luôn bị thổi phồng lên mức **$1.80 - 3.00$**, ngay cả khi `density_ratio` thực tế (đo bằng ngưỡng cứng ở Cell 8) đã tụt xuống **$0.73$**! Do đó, nhánh phạt phình nét `dev_high = F.relu(ratio - 1.02)` luôn kích hoạt ở mức rất lớn, còn nhánh bảo vệ thiếu nét `dev_low` không bao giờ được kích hoạt.
2. **Trọng số phạt chồng chéo quá lớn:**
   - Trong V2.3, `loss_density` vừa phạt bất đối xứng bậc hai (`2.0 * dev_high + 0.5 * dev_high**2`), vừa được nhân hệ số $\lambda_{\text{density}} = \mathbf{0.65}$ (tăng hơn gấp đôi so với $0.30$ của V2.2).
   - Đồng thời, `GhostStrokePenaltyLoss` ($\lambda_{\text{ghost}} = 0.25$) bổ sung thêm thành phần `near_halo` phạt trực tiếp mọi pixel $p_{\text{ink}} > 0.02$ nằm ngoài `target_f < 0.02`. Sự kết hợp này tạo ra áp lực xóa mực cực mạnh trên toàn khung hình.

### 1.2. Tại sao các nét vẽ quá mỏng bị biến mất dẫn đến mất chi tiết (`Stroke Recall` giảm từ `92.40%` xuống `79.92%`)?
1. **Đặc thù cường độ của nét mảnh sau phép biến dạng (Warping Attenuation):**
   - Các chi tiết mảnh (sợi tóc, lông mi, mắt, ngón tay, nếp gấp quần áo hoặc các nét chuyển động nhanh) có bề rộng chỉ $1 - 2\text{ px}$. Sau khi đi qua phép nội suy song tuyến tính (`F.grid_sample` trong `warp_normalized_flow` và `warp_pixel_flow`), cường độ mực của chúng tại khung hình trung gian bị pha loãng xuống dải $p_{\text{ink}} \in [0.12, 0.35]$.
2. **Hiệu ứng "Tẩy nhầm nét mảnh" tại `SynthesisUnet_v1`:**
   - Đầu ra của mô hình được tính bằng `pred_t = clamp(gate_erase * base_interp + detail_add, 0, 1)`.
   - Khi `loss_density` và `near_halo` phạt nặng nhưng không phân biệt giữa **"quầng mờ xung quanh nét dày"** và **"bản thân nét mảnh có cường độ thấp"**, cổng `gate_erase` học cách triệt tiêu toàn bộ các pixel có cường độ $p_{\text{ink}} < 0.30$, làm các nét mảnh đứt đoạn hoặc biến mất.
3. **Thiếu hàm mất mát chuyên biệt bảo vệ vùng nét mảnh:**
   - `StrokeDiceLoss` ($\lambda = 0.30$ trong V2.3, giảm từ $0.40$) tính tổng trên toàn ảnh nên bị áp đảo bởi các đường viền dày (chiếm $>80\%$ tổng số pixel mực). Việc mất đi các nét chi tiết $1\text{ px}$ hầu như không làm thay đổi giá trị Dice toàn cục nhưng lại gây mất chi tiết thị giác rõ rệt.

### 1.3. Tại sao không chọn được Best Checkpoint trong suốt quá trình train (`is_best = False` cả 16 Epochs)?
Quan sát dữ liệu thực tế từ `training_history_v2.3.csv`:
- Suốt 16 epochs, `density_ratio` nằm trong khoảng `[0.7290, 0.7995]` và `stroke_recall` nằm trong khoảng `[80.83%, 87.71%]` (Epoch 1 và Epoch 10 đạt tốt nhất với `Recall ~ 87.6%`, `Density ~ 0.799`, `Composite Score = 0.9454 - 0.9618`).
- Tuy nhiên, `SmartCheckpointManager_v23` (Cell 9) đặt **Hard Gates** cố định theo thông số phình nét cũ của V2.0:
  - Đòi hỏi `density_ratio` $\in [0.90, 1.12]$ và `recall >= 0.915` ($91.5\%$).
  - Nghiêm trọng hơn, ngay cả nhánh dự phòng `Fallback Best Progress` cũng đặt điều kiện `if not self.has_strict_best and not is_blank and recall >= self.min_recall:` (vẫn bắt buộc `recall >= 91.5%`).
- **Hệ quả:** Không có bất kỳ epoch nào vượt qua nổi cổng `recall >= 91.5%`, dẫn đến **cả `model_tps_v2.3_best.pt` lẫn `model_tps_v2.3_best_progress.pt` đều không được tạo ra**; khi ngắt ở Epoch 17, hệ thống chỉ còn lại `latest.pt` / `interrupted.pt`.

### 1.4. Phân tích Giới hạn RAM Kaggle & Thời gian mỗi Epoch trong V2.3
Từ `training_history_v2.3.csv`:
- **Thời gian mỗi Epoch:** Ổn định ở mức **~413.5 giây (~6.9 phút/epoch)** nhờ cơ chế nạp trước ảnh vào RAM (`img_buffers`) và bỏ tính Loss nặng trong pha Validation.
- **Hiện tượng rò rỉ tuyến tính Host RAM (`+0.847 GB / epoch`):**
  - Process RSS tăng đều đặn đúng `+0.847 GB` sau mỗi epoch: từ `4.52 GB` (Ep 1) $\to$ `12.16 GB` (Ep 10) $\to$ `17.24 GB` (Ep 16, Container cgroup lên tới `21.17 GB`). Nếu chạy đủ 25–30 epochs, Container RAM sẽ chạm ngưỡng `29–30 GB` của Kaggle và gây lỗi OOM Kill.
  - **Nguyên nhân:** (1) `self.img_buffers` đang lưu dưới dạng Python `list` chứa `20,202` đối tượng `bytes` riêng lẻ; mỗi đầu epoch khi `DataLoader` (`persistent_workers=False`) fork 2 worker mới và truy cập phần tử trong list, Python tăng biến đếm tham chiếu (`ob_refcnt`) trên từng object làm kích hoạt Copy-on-Write (CoW); (2) `nn.DataParallel` trong pha Train tạo/hủy 714 OS threads (`pthread`) mỗi epoch làm tích tụ bộ nhớ C++ thread-local của autograd/CUDA.

---

## 2. KIẾN TRÚC GIẢI PHÁP TRỌNG TÂM CHO PHIÊN BẢN V2.4

### 2.1. Chỉnh sửa Hàm Phạt Mật Độ: `SymmetricRampDensityLoss` (Cell 6)
Thay thế Sigmoid lệch nền bằng hàm **Piecewise Linear Ramp khả vi đối xứng** cho cả `pred` và `target`:
1. **Triệt tiêu 100% nhiễu nền trắng (Zero Background Floor):**
   $$M_{\text{ramp}}(x) = \text{clamp}\left(\frac{x - 0.08}{0.14}, 0.0, 1.0\right)$$
   - Mọi pixel nền trắng hoặc nhiễu cực nhỏ ($x \le 0.08$) có $M_{\text{ramp}}(x) = \mathbf{0.0}$ tuyệt đối (0 byte nhiễu nền).
   - Vùng chuyển tiếp $[0.08, 0.22]$ (tâm tại ngưỡng chuẩn $0.15$) có đạo hàm hằng số mượt mà $\frac{\partial M}{\partial x} = \frac{1}{0.14} \approx 7.14$, giúp truyền gradient ổn định mà không bị bão hòa như Sigmoid.
   - Áp dụng **cùng một hàm $M_{\text{ramp}}$** cho cả tử số (`pred`) và mẫu số (`target`) để tỷ số `ratio_mask` phản ánh chính xác $100\%$ tỷ lệ mật độ nét thực tế ở Cell 8.
2. **Điều chỉnh dải mục tiêu & Phạt bảo vệ 2 chiều (Two-Sided Protection):**
   - Đặt dải mục tiêu không phạt (Dead-zone) tại **`[target_min = 0.88, target_max = 1.04]`** (tương ứng với độ dày nét chuẩn của Ground Truth, mảnh hơn rõ rệt so với mức $1.3188$ của V2.0 nhưng đầy đặn hơn mức $0.6795$ của V2.3).
   - **Tăng hệ số phạt khi thiếu nét (`dev_low`) và giảm phạt khi dư nét (`dev_high`):**
     $$\mathcal{L}_{\text{density}} = \mathbf{1.8} \cdot \text{ReLU}(0.88 - r) + \mathbf{1.0} \cdot \text{ReLU}(r - 1.04) + 0.25 \cdot \text{ReLU}(r - 1.04)^2$$
   - Khi mật độ nét giảm xuống dưới $0.88$, hệ số $1.8 \cdot \text{dev\_low}$ lập tức đẩy mực tăng trở lại, ngăn chặn triệt để hiện tượng bào mòn nét.
3. **Giảm trọng số tổng hợp:** Hạ $\lambda_{\text{density}}$ từ `0.65` xuống **`0.20`**.

### 2.2. Khắc phục Mất Chi Tiết Nét Mảnh: `FineStrokePreservationLoss` & Tinh chỉnh Kiến trúc/Loss (Cell 5 & Cell 6)
Để giữ được độ mảnh của nét chính mà **không làm biến mất các nét chi tiết mỏng**, V2.4 áp dụng 3 cơ chế phối hợp (với **0ms overhead** nhờ tái sử dụng phép toán):

1. **Hàm mất mát bảo tồn nét mảnh chuyên biệt (`FineStrokePreservationLoss` - Cell 6):**
   - **Tách mặt nạ nét mảnh tự động trên GPU:** Trong quá trình tính `SoftClDiceLoss`, mô hình đã tính sẵn phép mở hình thái học `opened_gt = soft_open(target)` (lọc bỏ các nét mảnh $1-2\text{ px}$, chỉ giữ lại nét dày). Do đó, hiệu số:
     $$W_{\text{thin}} = \text{clamp}(\text{target} - \text{opened\_gt}, 0.0, 1.0)$$
     chính là **bản đồ các chi tiết nét mảnh** (tóc, mắt, ngón tay, đường viền mỏng) của Ground Truth!
   - **Tạo bản đồ trọng số bảo vệ chi tiết ($W_{\text{detail}}$):**
     $$W_{\text{detail}} = 1.0 + 4.0 \cdot W_{\text{thin}} + 2.0 \cdot \text{skel}_{\text{gt}}$$
   - **Phạt bất đối xứng chống mất nét (Asymmetric Under-Prediction Penalty):**
     Khi `pred < target` trên vùng có nét của Ground Truth ($\text{target} > 0.10$), phạt gấp **3.0 lần** theo trọng số $W_{\text{detail}}$:
     $$\mathcal{L}_{\text{fine}} = \frac{\sum \text{ReLU}(\text{target} - \text{pred}) \cdot W_{\text{detail}} \cdot \mathbb{I}(\text{target} > 0.10)}{\sum \mathbb{I}(\text{target} > 0.10) + 1.0}$$
   - Nhờ tái sử dụng trực tiếp `opened_gt` và `skel_gt` từ `SoftClDiceLoss`, hàm loss này **không tốn thêm bất kỳ phép `max_pool2d` nào**, giữ nguyên tốc độ train!

2. **Tăng độ nhạy khung xương (Centerline Sensitivity Boost) trong `SoftClDiceLoss`:**
   - Thay vì F1 cân bằng $(0.5 \text{ Precision} + 0.5 \text{ Sensitivity})$, V2.4 dùng **Tversky-clDice** ưu tiên Sensitivity ($\alpha = 0.35$ cho `tprec`, $\beta = 0.65$ cho `tsens`), buộc mạng không được làm đứt khung xương của các nét mảnh.

3. **Bảo vệ vùng biên nét mảnh trong `GhostStrokePenaltyLoss` & Khởi tạo Bias `SynthesisUnet_v1`:**
   - Trong `GhostStrokePenaltyLoss`: Thay điều kiện `outside_exact_gt = (target_f < 0.02)` bằng vùng ngoài của mặt nạ giãn nở nhẹ $3\times 3$ (`t_dilated_3x3 = F.max_pool2d((target_f > 0.12).float(), 3, 1, 1)`). Điều này tạo ra "hành lang an toàn $1\text{ px}$" bảo vệ các nét vẽ mỏng không bị hàm phạt Halo bào mòn.
   - Trong `SynthesisUnet_v1` (Cell 5): Điều chỉnh bias khởi tạo của `conv_erase` từ `2.0` ($\sigma(2.0) \approx 0.88$) lên `2.5` ($\sigma(2.5) \approx 0.924$) và `conv_add` từ `-3.0` lên `-2.2` ($\sigma(-2.2) \approx 0.10$) khi không có trọng số sẵn, đồng thời dùng Warm-start từ **V2.0.2 Best** (checkpoint giàu chi tiết nét mảnh nhất, `Recall = 92.4%`) để V2.4 làm mảnh nét một cách có kiểm soát thay vì kế thừa các nét đã bị mất của V2.3.

### 2.3. Bảng Đối Chiếu Cấu Hình Trọng Số Loss (V2.2 vs V2.3 vs V2.4)

| Thành phần Hàm Mất Mát | V2.2 ($\lambda$) | V2.3 ($\lambda$) | **V2.4 ($\lambda$ Đề xuất)** | Vai trò & Lý do điều chỉnh trong V2.4 |
|:---|:---:|:---:|:---:|:---|
| `L1 Reconstruction Loss` | `1.00` | `1.00` | **`1.00`** | Nền tảng tái tạo cường độ pixel toàn cục |
| `VGGPerceptualLoss` | `0.15` | `0.15` | **`0.15`** | Duy trì cấu trúc thị giác cấp cao |
| `StrokeDiceLoss` | `0.40` | `0.30` | **`0.40`** | Khôi phục lên `0.40` để bảo đảm độ liền mạch thân nét |
| `SoftClDiceLoss (β=0.65)` | `0.20` | `0.25` | **`0.35`** | Ưu tiên bảo toàn khung xương (`tsens`) chống đứt nét mỏng |
| **`FineStrokePreservationLoss`** | *(Không có)* | *(Không có)* | **`0.30` [MỚI]** | Phạt nặng khi làm biến mất nét mảnh ($1-2\text{ px}$) của GT |
| `StrokeDensityRegularization` | `0.30` | `0.65` *(Quá nặng)* | **`0.20` [GIẢM SÂU]** | Ramp đối xứng không nhiễu nền, dải `[0.88, 1.04]`, chống hụt nét |
| `GhostStrokePenaltyLoss` | `0.10` | `0.25` | **`0.18`** | Khử bóng ma xa & halo ngoài hành lang bảo vệ $3\times 3$ |
| `GPUMorphologicalDistance` | `0.20` | `0.15` | **`0.15`** | Kéo nét dự đoán bám sát quỹ đạo hình thái học GT |
| `DirectionalTangentCoherence` | `0.20` | `0.20` | **`0.20`** | Giữ mượt tiếp tuyến đường cong anime |

---

### 2.4. Thiết kế lại `SmartCheckpointManager_v24` (Cell 9): Đảm bảo 100% Chọn được Best Checkpoint

Khắc phục triệt để việc không lưu được Best Checkpoint ở V2.3 bằng **Cơ chế Chấm điểm & Chọn Checkpoint 3 Tầng (Multi-Tier Selection)**:

1. **Hiệu chỉnh Hard Gates phù hợp với đặc tính nét thanh mảnh (Tier 1 - Strict Best):**
   - `is_blank == False` (Bắt buộc 0% ảnh trắng).
   - `recall >= 0.865` (**86.5%** — ngưỡng thực tế chuẩn xác khi độ dày nét giảm về vùng $[0.88, 1.00]$ thay vì phình $+32\%$ như V2.0).
   - `0.82 <= density_ratio <= 1.08` (Mục tiêu lý tưởng `target_density = 0.94`).
2. **Công thức `Sakuga Composite Score V2.4` (`S_val` - Càng thấp càng tốt):**
   $$S_{\text{val}} = 0.30 \cdot \left(\frac{\text{Ghost}}{14322.38}\right) + 0.25 \cdot \left(\frac{\text{Chamfer}}{0.002241}\right) + 0.30 \cdot \left(\frac{\max(0, 1.0 - \text{Recall})}{1.0 - 0.923635}\right) + 0.15 \cdot |\text{Density} - 0.94|$$
   - Tăng trọng số `Recall` từ $0.20 \to \mathbf{0.30}$ để ưu tiên các checkpoint giữ được nhiều chi tiết nét mảnh nhất.
   - Đặt điểm neo mật độ lý tưởng tại **`0.94`** (điểm cân bằng vàng giữa độ thanh mảnh và độ đầy đủ chi tiết).
3. **Cơ chế `Penalized Score` cho Tầng Dự Phòng (`model_tps_v2.4_best_progress.pt` - Tier 2):**
   - Loại bỏ hoàn toàn điều kiện chặn cứng `recall >= min_recall` ở nhánh Fallback!
   - Tính điểm phạt mềm cho mọi epoch không bị Blank:
     $$S_{\text{penalized}} = S_{\text{val}} + 2.0 \cdot \text{ReLU}(0.865 - \text{Recall}) + 1.5 \cdot \text{ReLU}(0.82 - \text{Density}) + 1.5 \cdot \text{ReLU}(\text{Density} - 1.08)$$
   - Bất kỳ epoch nào có $S_{\text{penalized}}$ tốt hơn kỷ lục trước đó đều được lưu ngay lập tức vào `model_tps_v2.4_best_progress.pt` (và nếu chưa có Strict Best, tự động cập nhật bản sao sang `model_tps_v2.4_best.pt`).
   - **Đảm bảo:** Kể cả khi người dùng bấm dừng (Interrupt) ở bất kỳ epoch nào, luôn có sẵn `model_tps_v2.4_best.pt` là epoch có chất lượng tổng thể tốt nhất tính đến thời điểm đó!

---

### 2.5. Tối ưu Hóa Giới Hạn Host RAM Kaggle & Thời Gian Huấn Luyện Mỗi Epoch (Cell 4, 7, 10)

Để đảm bảo notebook chạy mượt 25 epochs trên Kaggle mà **không bị tràn RAM (giữ cố định < 8.5 GB)** và **thời gian mỗi epoch $\le 370\text{s}$**:

1. **Đóng gói Ảnh RAM dạng Mảng Phẳng Duy Nhất (`Flat Contiguous Byte Arena` - Cell 4):**
   - Nối toàn bộ các file PNG thành **1 mảng NumPy 1 chiều duy nhất** `self.img_arena = np.frombuffer(b''.join(raw_list), dtype=np.uint8)` kèm mảng chỉ mục `self.img_offsets = np.cumsum(...)` (`int64`).
   - Khi worker đọc ảnh: `buf = self.img_arena[start:end]` chỉ tạo view NumPy đọc-chỉ (read-only) trên vùng nhớ phẳng, **không đụng vào reference count của Python objects**, triệt tiêu 100% hiện tượng tăng RAM qua các lần fork worker!
2. **Khắc phục Rò rỉ C++ Thread-Local trong Vòng Lặp Huấn Luyện & Validation (Cell 7 & Cell 10):**
   - Đặt `VAL_BATCH_SIZE = 4` và sử dụng trực tiếp hàm `calculate_metrics_batch_gpu(pred, tgt)` (tính song song toàn batch trên GPU trong 1 lần gọi duy nhất) thay cho vòng lặp `for i in range(pred.shape[0]): calculate_metrics_v1(...)` ở Cell 10.
   - Giải phóng tường minh `torch.cuda.empty_cache()`, `gc.collect()` và `malloc_trim(0)` sau mỗi pha.
