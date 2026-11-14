# KẾ HOẠCH NÂNG CẤP HÀM LOSS & COMPOSITE SCORE CHO LẦN FINE-TUNE TIẾP THEO
## Tinh chỉnh `model_latest.pt` trên `final_dataset` — Trọng tâm: Chống Tô Đậm Nét Quá Mức (Anti-Over-Thickening)

> **Mô hình gốc (Warm-Start):** [`model_latest.pt`](file:///E:/CODE/PBL/PBL6/checkpoints/pretrained_baselines/tps-original/model_latest.pt) (`TPSInbetweenOriginal` — `10,068,668` active params, **100% Zero Architectural Modification**)
> **Dataset huấn luyện:** [`final_dataset`](file:///E:/CODE/PBL/PBL6/Dataset/sakuga_pipeline_sequence/final_dataset) (`1,485` Train / `190` Val / `198` Bench1 sequences, $N \in \{3,5,7\}$)
> **Mã nguồn đích sẽ chỉnh sửa:**
> - [`ddp_worker_tps_sequence_final.py`](file:///E:/CODE/PBL/PBL6/kaggle_finetune_sequence_final/ddp_worker_tps_sequence_final.py)
> - [`run_benchmark_suite.py`](file:///E:/CODE/PBL/PBL6/Dataset/sakuga_pipeline_sequence/final_dataset/sakuga_golden_benchmark_sequences_v2/run_benchmark_suite.py)
> - [`generate_kaggle_finetune_nb.py`](file:///E:/CODE/PBL/PBL6/kaggle_finetune_sequence_final/generate_kaggle_finetune_nb.py) $\rightarrow$ tái sinh [`train_and_benchmark_tps_sequence_final_kaggle.ipynb`](file:///E:/CODE/PBL/PBL6/kaggle_finetune_sequence_final/train_and_benchmark_tps_sequence_final_kaggle.ipynb)
>
> **Cơ sở chuyên gia từ `E:\CODE\PBL\PBL6\docs`:**
> - [`tps-inbetween.pdf`](file:///E:/CODE/PBL/PBL6/docs/DEEPLEARNING/tps-inbetween.pdf) *(Eq. 10–15: Symmetric DT Loss, Pixel Count Constraint $\mathcal{L}_{cnt}$, WCD Area Penalty $\mathcal{H}(\mathbf{b}_0, \mathbf{b}_1)$)*
> - [`ai_model_training_knowledge_notes.md`](file:///E:/CODE/PBL/PBL6/docs/ai_model_training_knowledge_notes.md) *(Mục 3.2 Validation FP32, Mục 3.6 Cô lập Test, Mục 6.1 Stroke Thickness Loss)*
> - [`tps_v3_training_optimization_guide.md`](file:///E:/CODE/PBL/PBL6/docs/tps_v3_training_optimization_guide.md) *(Mục 2.5 Tránh xung đột hàm loss hỗn hợp, Mục 3.2 Pure FP32 Val)*
> - [`01_finetune_plan_v2.4.md`](file:///E:/CODE/PBL/PBL6/docs/finetune_plans/v2.4/01_finetune_plan_v2.4.md), [`01_finetune_plan_v2.6.md`](file:///E:/CODE/PBL/PBL6/docs/finetune_plans/v2.6/01_finetune_plan_v2.6.md) & [`generate_v25_notebook.py`](file:///E:/CODE/PBL/PBL6/kaggle/generate_v25_notebook.py) *(Bài học V2.3 teo nét `0.58x` $\to$ V2.4 phình nét `1.458x` $\to$ V2.5 Exact-STE Density Calibration `1.0101x` + Bidirectional Stroke Width)*
> - [`02_retrain_plan_v3.0_remastered_r2.md`](file:///E:/CODE/PBL/PBL6/docs/finetune_plans/v3.0_remastered/02_retrain_plan_v3.0_remastered_r2.md) & [`01_finetune_plan_v3.0_540p.md`](file:///E:/CODE/PBL/PBL6/docs/finetune_plans/v3.0/01_finetune_plan_v3.0_540p.md) *(Continuous Loss không nhảy bậc, Background Floor Cleanliness, `SakugaWCDTopologyManager` chuẩn hóa theo Baseline Epoch 0 + Hard Gates)*

---

## 1. Chẩn đoán Nguyên nhân Gây Tô Đậm Nét Quá Mức trong Lần Chạy #1

Đối chiếu mã nguồn [`ddp_worker_tps_sequence_final.py`](file:///E:/CODE/PBL/PBL6/kaggle_finetune_sequence_final/ddp_worker_tps_sequence_final.py) với dữ liệu thực nghiệm tại [`training_history_finetune.csv`](file:///E:/CODE/PBL/PBL6/checkpoints/Re-train/Sequence/training_history_finetune.csv) và các tài liệu chuyên gia trong `docs/`, hiện tượng phình nét (`density_ratio = 1.1943` trên Val, `1.1671` trên Bench #1) và tăng `ghost` gấp $4\times$ bắt nguồn từ **5 lỗ hổng kỹ thuật cụ thể**:

1. **Hàm Loss thiếu hoàn toàn lực kìm hãm độ dày & diện tích mực:**
   - `AdaptiveLineartLoss` hiện tại chỉ gồm: `Charbonnier + 0.60*SymDT(12) + 0.25*Sobel + 0.25*Soft-clDice + 0.15*VGG16`.
   - Không có `StrokeDensityRegularizationLoss` ($\mathcal{L}_{cnt}$ trong [`tps-inbetween.pdf`](file:///E:/CODE/PBL/PBL6/docs/DEEPLEARNING/tps-inbetween.pdf) Eq. 11; Exact-STE trong V2.5), không có thành phần phạt tràn mực trên vành đai viền 1–2px (`BidirectionalStrokeWidthLoss`), và không có `GhostLoss` ngoài hành lang $5\times 5$.
   - Trên Val, đúng tại **Epoch 2** (khi `density_ratio = 1.0402`), `WCD_DoG` đạt tốt nhất trên cả 3 nhóm ($N=3, 5, 7$). Từ Epoch 3 trở đi, vì không có lực cản phình nét, `SymDT` + `Soft-clDice` kéo `density_ratio` phình lên `1.1943` tương tự hiện tượng V2.4 từng bị (`1.4580x` trong [`01_finetune_plan_v2.4.md`](file:///E:/CODE/PBL/PBL6/docs/finetune_plans/v2.4/01_finetune_plan_v2.4.md)).
2. **Lực kéo một chiều ra biên trong `loss_dt` và `SoftClDiceLoss`:**
   - Trong [`ddp_worker_tps_sequence_final.py`](file:///E:/CODE/PBL/PBL6/kaggle_finetune_sequence_final/ddp_worker_tps_sequence_final.py#L736-L738), `loss_dt = mean(pred * dt_g) + 0.15 * soft_inverse_dt_coverage(...)` dùng bản đồ khoảng cách tuyến tính 12 bước (`dt_g` chỉ bằng `1/13 ≈ 0.077` ở cách 1px ngoài nét GT) nên phạt rất nhẹ việc phình rộng thêm 1–2px, trong khi `SoftClDiceLoss` (`tsens = sum(t_skel * pred) / sum(t_skel)`) thưởng trực tiếp cho các pixel mực đậm phủ lên khung xương GT.
   - Trái với công thức gốc Eq. (10) của [`tps-inbetween.pdf`](file:///E:/CODE/PBL/PBL6/docs/DEEPLEARNING/tps-inbetween.pdf) ($\mathcal{L}_{dt} = \|\mathcal{D}(y_t) - \mathcal{D}(y_t^{gt})\|_1$) vốn so sánh bản đồ khoảng cách của *cả* dự đoán và GT nên phạt ngay nếu nét dự đoán nở to làm phình vùng khoảng cách 0.
3. **Lỗi đảo ngược hệ số phạt diện tích `h_weight` trong `compute_wcd_batch_gpu`:**
   - Tại [`ddp_worker_tps_sequence_final.py`](file:///E:/CODE/PBL/PBL6/kaggle_finetune_sequence_final/ddp_worker_tps_sequence_final.py#L781):
     `h_weight = 0.5 * (1.0 + minimum(area_p, area_g) / maximum(area_p, area_g))`
   - Khi `area_p` phình lên `1.20 * area_g`, tỷ số `min/max` giảm còn `0.833` $\implies$ `h_weight` giảm xuống `0.917` (nhỏ hơn `1.0` khi khớp diện tích!). Điều này **ngược hoàn toàn** với Eq. (15) của [`tps-inbetween.pdf`](file:///E:/CODE/PBL/PBL6/docs/DEEPLEARNING/tps-inbetween.pdf) và [`cell_08.py`](file:///E:/CODE/PBL/PBL6/kaggle/v30r_cells/cell_08.py#L56):
     $$\mathcal{H}(\mathbf{b}_0, \mathbf{b}_1) = \text{Sigmoid}\!\left(\frac{|\text{area}_p - \text{area}_g|}{\min(\text{area}_p, \text{area}_g)}\right) \in [0.5, 1.0)$$
     vốn phải **tăng** từ $0.5$ lên $1.0$ để **phạt** khi diện tích nét bị phình hoặc teo.
4. **Công thức `comp_score` lệch thang đo, dùng `wcd` raw và thiếu kiểm soát `Density` + `Ghost`:**
   - Công thức cũ (`0.45*wcd + 0.25*(semd*3.0) + 0.20*(1 - skel_recall) + 0.10*val_l1`) không chuẩn hóa theo Epoch-0 Baseline: `0.45 * wcd` ($\approx 1.03$) chiếm **~95%** tổng điểm trong khi `semd*3` ($\approx 0.04$) và `val_l1` ($\approx 0.035$) bị triệt tiêu.
   - Dùng `wcd` raw (`p > 0.15`) thay vì `wcd_dog` và đặt trọng số `0%` cho `density_ratio` và `ghost` khiến thuật toán loại Epoch 2 (`WCD_DoG = 2.6822`, `density = 1.0402`) và chọn Epoch 13 (`density = 1.1943`, `ghost = 6,054 px`, `WCD_DoG = 2.7186`).
5. **Validation chạy ở FP16 (`autocast`):**
   - [`run_validation_suite`](file:///E:/CODE/PBL/PBL6/kaggle_finetune_sequence_final/ddp_worker_tps_sequence_final.py#L1382) bật `autocast('cuda')` FP16, trái quy tắc vàng trong [`ai_model_training_knowledge_notes.md`](file:///E:/CODE/PBL/PBL6/docs/ai_model_training_knowledge_notes.md#L348) và [`01_finetune_plan_v3.0_540p.md`](file:///E:/CODE/PBL/PBL6/docs/finetune_plans/v3.0/01_finetune_plan_v3.0_540p.md#L102) yêu cầu chạy Validation thuần **FP32** để không méo ngưỡng `0.08`/`0.15` trên nền trắng và nét mảnh.

---

## 2. Thiết kế Chi tiết Sửa đổi Hàm Loss (`AdaptiveLineartLoss_v2`)

Giữ nguyên triết lý từ [`02_retrain_plan_v3.0_remastered_r2.md`](file:///E:/CODE/PBL/PBL6/docs/finetune_plans/v3.0_remastered/02_retrain_plan_v3.0_remastered_r2.md#L134): **100% liên tục và khả vi từ Epoch 1** (không chia nhánh `if epoch <= k` gây sốc Momentum của AdamW), đồng thời cân bằng lực co/nở theo chuẩn **V2.5 Exact-STE + Bidirectional Stroke Width** để tránh lặp lại cả hai thái cực: teo nét như V2.3 (`0.58x`) lẫn phình nét như V2.4 (`1.458x`) và Run #1 (`1.19x`).

### 2.1. Tổng hợp Công thức Mới cho Chế độ `LONG`

$$\mathcal{L}_{\text{total}} = 1.00\,\mathcal{L}_{\text{Charb}} + 0.35\,\mathcal{L}_{\text{SymDT-Map}} + 0.20\,\mathcal{L}_{\text{Sobel}} + 0.20\,\mathcal{L}_{\text{clDice-Bal}} + 0.15\,\mathcal{L}_{\text{Perc}} + \mathbf{0.50\,\mathcal{L}_{\text{STE-Dens}}} + \mathbf{0.25\,\mathcal{L}_{\text{BiStrokeWidth}}} + \mathbf{0.20\,\mathcal{L}_{\text{GhostClean}}}$$

Chi tiết 4 thay đổi trọng tâm trong hàm Loss:

1. **Bổ sung `ExactSTEDensityLoss` ($\lambda_{\text{dens}} = 0.50$, Neo tâm chuẩn `1.00`, dải `[0.96, 1.04]`):**
   - Kế thừa trực tiếp từ [`generate_v25_notebook.py`](file:///E:/CODE/PBL/PBL6/kaggle/generate_v25_notebook.py#L584-L645) (đã kiểm chứng thực tế đưa V2.5 về `1.0101x` từ nền `1.458x` của V2.4) kết hợp chuẩn Zero-Background-Floor của V2.4:
     - Đếm diện tích nhị phân chính xác tại ngưỡng đánh giá `0.15`: $m_{\text{hard}} = \mathbb{I}(x > 0.15)$.
     - Surrogate Ramp không nhiễu nền ($x \le 0.06 \implies 0.0$) và không bão hòa đạo hàm ở vùng mực đặc:
       $$m_{\text{soft}}(x) = \text{clamp}\!\left(\frac{x - 0.06}{0.18},\, 0.0,\, 1.0\right) + 0.25 \cdot \text{ReLU}(x - 0.24)$$
     - Straight-Through Estimator: $m_{\text{ste}} = m_{\text{hard}}.\text{detach}() + (m_{\text{soft}} - m_{\text{soft}}.\text{detach}())$.
     - Kết hợp $80\%$ tỷ lệ diện tích nhị phân STE + $20\%$ tỷ lệ tổng mực liên tục ($\text{ReLU}(p - 0.05)$):
       $$r = 0.80 \cdot \frac{\sum m_{\text{ste}}(P)}{\sum m_{\text{ste}}(G) + \epsilon} + 0.20 \cdot \frac{\sum \text{ReLU}(P - 0.05)}{\sum G + \epsilon}$$
     - **Hàm phạt siết phình nét bất đối xứng (Target `[0.96, 1.04]`, center `1.00`):**
       $$\text{dev}_{\text{low}} = \text{ReLU}(0.96 - r), \qquad \text{dev}_{\text{high}} = \text{ReLU}(r - 1.04)$$
       $$\mathcal{L}_{\text{STE-Dens}} = 1.8\,\text{dev}_{\text{low}} + 2.8\,\text{dev}_{\text{high}} + 1.5\,\text{dev}_{\text{high}}^2 + 0.06\,(r - 1.00)^2$$
     - *Lưu ý thiết kế theo V2.6 vs Sequence Run #1:* V2.6 chọn tâm `1.10` cho kiến trúc ASF-GR 168 tensors, nhưng trên kiến trúc gốc `model_latest.pt` ở Run #1, điểm hội tụ hình học tối ưu nhất (Epoch 2 trên cả $N=3,5,7$) xuất hiện tại `density = 1.0402` và trên Bench #2 mức `1.034` đã bắt đầu có dấu hiệu đậm hơn GT ở viền mặt/tóc. Do đó đặt dải mục tiêu `[0.96, 1.04]` (tâm `1.00`) giúp khóa chặt nét đúng độ mảnh của Ground Truth.

2. **Bổ sung `BidirectionalStrokeWidthLoss` ($\lambda_{\text{width}} = 0.25$):**
   - Kế thừa từ [`generate_v25_notebook.py`](file:///E:/CODE/PBL/PBL6/kaggle/generate_v25_notebook.py#L705-L738) + `HairTip ROI` từ V3.0-R2:
     - **Nhánh 1 — Bảo tồn lõi nét mảnh & ngọn tóc (`loss_under`, chỉ tính trong `gt_core = (gt > 0.15)`):**
       - $W_{\text{thin}} = \text{clamp}(G - \text{soft\_open}(G), 0, 1)$, kết hợp `skel_gt` và `endpoints_roi` (ngọn tóc):
         $W_{\text{core}} = 1.0 + 2.5\,W_{\text{thin}} + 1.5\,\text{skel}_{\text{gt}} + 2.0\,\text{endpoints}_{\text{roi}}$.
       - $\mathcal{L}_{\text{under}} = \frac{\sum \text{ReLU}(G - P) \cdot W_{\text{core}} \cdot \text{gt\_core}}{\sum \text{gt\_core} + 1.0}$.
     - **Nhánh 2 — Chặn phình viền 1–2px quanh nét thật (`loss_over_edge`, trực tiếp chống tô đậm nét):**
       - Tạo vành đai viền sát nét GT: $\text{edge\_ring} = \text{clamp}(\text{dilate}_{5\times 5}(\text{gt\_core}) - \text{gt\_core},\, 0,\, 1)$.
       - Phạt tràn mực ngay tại 1–2px sát mép nét (nơi mà `GhostLoss` ngoài $5\times 5$ không chạm tới được!):
         $$\mathcal{L}_{\text{over\_edge}} = \frac{\sum \big(\text{ReLU}(P - \max(G, 0.06)) + 2.0\,\text{ReLU}(P - 0.14)\big) \cdot \text{edge\_ring}}{\sum \text{edge\_ring} + 1.0}$$
     - Tổng hợp: $\mathcal{L}_{\text{BiStrokeWidth}} = \mathcal{L}_{\text{under}} + 1.50\,\mathcal{L}_{\text{over\_edge}}$ (ưu tiên chặn tràn viền $1.5\times$ để nét không phình sang 2 bên).

3. **Bổ sung `GhostAndBackgroundCleanlinessLoss` ($\lambda_{\text{ghost}} = 0.20$):**
   - Khắc phục trực tiếp hiện tượng `ghost` vọt từ `1,537 px` lên `6,196 px` và `dark_ghost` lên `2,501 px` trong Run #1:
     - Ngoài hành lang $5\times 5$ của GT (`outside_5x5 = 1.0 - dilate_5x5((gt > 0.08).float())`):
       - Phạt sương xám & rò nền (`Background Floor Cleanliness` từ [`01_finetune_plan_v3.0_540p.md`](file:///E:/CODE/PBL/PBL6/docs/finetune_plans/v3.0/01_finetune_plan_v3.0_540p.md#L82)): `bg_haze = ReLU(P - 0.015) * outside_5x5`.
       - Phạt nặng bóng mờ trung tính & đậm: `far_ghost = ReLU(P - 0.08) * outside_5x5` và `dark_ghost = ReLU(P - 0.35) * outside_5x5`.
     - Cận biên ngoài lõi thực (`outside_core = (gt < 0.08).float()`): `near_halo = ReLU(P - 0.08) * outside_core`.
     - Công thức: $\mathcal{L}_{\text{GhostClean}} = 1.0\,\overline{\text{bg\_haze}} + 1.5\,\overline{\text{far\_ghost}} + 3.0\,\overline{\text{dark\_ghost}} + 0.4\,\overline{\text{near\_halo}}$.

4. **Hiệu chỉnh `Symmetric DT` ($\lambda_{\text{dt}} = 0.35$) & `SoftClDiceLoss` cân bằng ($\lambda_{\text{cldice}} = 0.20$):**
   - Trong `loss_dt`: Hạ trọng số tổng từ `0.60` xuống `0.35` và chuyển từ kéo nở 1 chiều sang **hỗn hợp cân bằng đối xứng kiểu Eq. (10) bài báo gốc**:
     - `loss_dt_sym = F.l1_loss(approx_soft_dt_map(pred, steps=8), approx_soft_dt_map(gt, steps=8))` (trong đó `approx_soft_dt_map` truyền gradient trực tiếp qua `pred` dùng chuỗi `max_pool2d` trên tensor liên tục `pred`, tương tự `TruncatedDifferentiableDTLoss` trong [`cell_06.py`](file:///E:/CODE/PBL/PBL6/kaggle/v30r_cells/cell_06.py#L54-L78), kết hợp `0.5 * mean(pred * dt_g)` để vừa kéo nét lệch về GT vừa phạt khi `pred` nở rộng làm biến dạng bản đồ khoảng cách).
   - Trong `SoftClDiceLoss`: Hạ từ `0.25` xuống `0.20` và dùng **Tversky-clDice cân bằng (`alpha=0.50, beta=0.50`)** như V2.5 để `tprec` (chống khung xương phình/rác) ngang trọng lượng với `tsens` (chống đứt nét).

### Bảng đối chiếu Hàm Loss: Run #1 vs Kế hoạch Run #2

| Thành phần Loss | Run #1 (Hiện tại) | **Run #2 (Kế hoạch Anti-Thickening)** | Cơ sở & Vai trò |
| :--- | :---: | :---: | :--- |
| `Charbonnier` | `1.00` | **`1.00`** | Nền tảng tái tạo cường độ pixel |
| `DT Loss` | `0.60` (`p2g + 0.15*g2p`, 12 steps) | **`0.35`** (`SymDT Map L1 + 0.5*p2g`, 8 steps) | Giảm lực kéo nở biên, đối xứng Eq. (10) `tps-inbetween.pdf` |
| `Sobel Edge` | `0.25` | **`0.20`** | Căn chỉnh độ sắc biên, nhường chỗ cho `BiStrokeWidth` |
| `Soft-clDice` | `0.25` (F1) | **`0.20`** (`alpha=0.50, beta=0.50`) | Giữ liền nét nhưng không hy sinh precision |
| `Perceptual (VGG16)` | `0.15` | **`0.15`** | Giữ cấu trúc thị giác tổng thể |
| **`ExactSTEDensityLoss`** | ❌ `0.00` | **`0.50`** (`[0.96, 1.04]`, tâm `1.00`) | Khóa chặt tỷ lệ diện tích nét ngay tại ngưỡng `0.15` (V2.5) |
| **`BidirectionalStrokeWidthLoss`** | ❌ `0.00` | **`0.25`** (`1.0*under + 1.5*over_edge`) | Chặn tràn mực vùng viền 1–2px quanh nét GT (V2.5) |
| **`GhostAndBackgroundCleanliness`** | ❌ `0.00` | **`0.20`** | Triệt tiêu `ghost` $4\times$ và sương xám nền (V2.5 + V3.0) |

---

## 3. Thiết kế Chi tiết Sửa đổi Công thức `Composite Score` & Logic Chọn Checkpoint

### 3.1. Sửa lỗi Hàm Đo `WCD` & Chuyển Validation sang Pure FP32

1. **Khôi phục `H(b0, b1)` chuẩn Eq. (15) [`tps-inbetween.pdf`](file:///E:/CODE/PBL/PBL6/docs/DEEPLEARNING/tps-inbetween.pdf) trong `compute_wcd_batch_gpu`:**
   ```python
   # Sửa từ: 0.5 * (1.0 + min_area / max_area)  [SAI CHIỀU - thưởng khi phình nét]
   # Thành chuẩn bài báo Eq. (15) & cell_08.py:
   h_weight = torch.sigmoid(
       torch.abs(area_p - area_g) / torch.clamp(torch.minimum(area_p, area_g), min=1.0)
   )
   ```
   - Khi `area_p == area_g`: `h_weight = 0.50` (tối thiểu). Khi `area_p` phình lên `1.20 * area_g` hoặc teo xuống `0.80 * area_g`: `h_weight` tăng về `0.55–0.65`, **phạt trực tiếp** sai lệch độ dày nét.
   - *Lưu ý đối chiếu:* Trong `run_benchmark_suite.py` và `val_baseline_m0_m1.json`, sẽ log song song cả `wcd_dog` (chuẩn mới dùng `h_weight` Eq. 15 để làm thước đo chính xác) và `wcd_dog_legacy` (nếu cần so sánh lại với bảng số Run #1).
2. **Tắt AMP trong pha Validation ([`run_validation_suite`](file:///E:/CODE/PBL/PBL6/kaggle_finetune_sequence_final/ddp_worker_tps_sequence_final.py#L1382)):**
   - Chạy suy luận Val bằng `with torch.inference_mode():` thuần **FP32** (không bọc `torch.amp.autocast('cuda')`), đảm bảo ngưỡng đo `0.08`, `0.15`, `ghost` và `density_ratio` chính xác tuyệt đối, thời gian Val tăng không đáng kể (~0.2m $\to$ ~0.35m/epoch trên 420 frames).

### 3.2. Công thức `Composite Score` Mới (Chuẩn hóa Baseline Epoch 0 + Phạt Phình Nét)

Lấy mốc chuẩn `ref_*` tự động từ chính kết quả đo **`M1_Original` tại Epoch 0** trên tập Val (420 frames), để mọi thành phần trong `score` đều có thứ nguyên chuẩn hóa quanh `1.0`:

$$S_{\text{val}} = 0.30 \left(\frac{\text{WCD\_DoG}}{\text{WCD\_DoG}_{\text{M1}}}\right) + 0.10 \left(\frac{\text{S-EMD}}{\text{S-EMD}_{\text{M1}}}\right) + 0.20 \left(\frac{1 - \text{SkelRecall}}{1 - \text{SkelRecall}_{\text{M1}}}\right) + 0.10 \left(\frac{1 - \text{TipRecall}}{1 - \text{TipRecall}_{\text{M1}}}\right) + \mathbf{0.18 \left(\frac{|\text{Density} - 1.00|}{0.10}\right)} + \mathbf{0.12 \left(\frac{\text{Ghost}}{\max(1000.0,\, \text{Ghost}_{\text{M1}})}\right)}$$

**Giải thích tỷ trọng (Tổng = `1.00`):**
- **Nhóm Hình học không thiên vị độ đậm (`40%`):** `WCD_DoG` (`30%`) + `S-EMD` (`10%`) — loại bỏ `WCD` raw khỏi hàm chọn checkpoint để không bị đánh lừa bởi việc làm đậm nét.
- **Nhóm Bảo toàn Nét mảnh & Ngọn tóc (`30%`):** `SkelRecall` (`20%`) + `TipRecall` (`10%`).
- **Nhóm Chống Tô Đậm & Sạch Nền (`30%`):**
  - `|Density - 1.00| / 0.10` (`18%`): Nếu `density_ratio` lệch `1.19` như Epoch 13 Run #1, riêng thành phần này cộng thêm `0.18 * 1.9 = +0.342` vào `score` (rất nặng).
  - `Ghost / max(1000, Ghost_M1)` (`12%`): Nếu `ghost` tăng $4.3\times$ như Epoch 13 Run #1 (`6,054` vs `1,409`), thành phần này tăng từ `0.12` lên `+0.516`.

### 3.3. Cơ chế Chọn Checkpoint 2 Tầng (Vẫn giữ Policy đúng 2 file `best.pt` & `last.pt`)

Áp dụng bài học từ V2.3/V2.4/V2.5/V3.0-R2 (`TwoFileCheckpointManager` chỉ lưu 2 file):
1. **Tier 1 — Strict Hard Gates (Ưu tiên tuyệt đối cho `model_tps_finetuned_best.pt`):**
   Một epoch chỉ được xem xét ghi đè Strict Best khi thỏa mãn đồng thời 4 cổng:
   - `0.94 <= density_ratio <= 1.08` *(Cổng chống teo nét & chống tô đậm)*
   - `skel_recall >= max(0.835, M1_skel_recall + 0.015)` *(Cổng chống đứt nét)*
   - `wcd_dog <= M1_wcd_dog` *(Bắt buộc hình học không tệ hơn mô hình gốc)*
   - `ghost <= max(3200.0, 2.2 * M1_ghost)` *(Cổng chống tăng bóng mờ diện rộng)*
2. **Tier 2 — Penalized Score Fallback (Dự phòng nếu chưa có epoch nào qua cả 4 cổng):**
   $$S_{\text{pen}} = S_{\text{val}} + 2.5\,\text{ReLU}(0.94 - \text{Dens}) + 2.5\,\text{ReLU}(\text{Dens} - 1.08) + 2.0\,\text{ReLU}(\text{Skel}_{\min} - \text{Skel}) + 1.5\,\text{ReLU}\!\left(\frac{\text{WCD\_DoG} - \text{WCD\_DoG}_{\text{M1}}}{\text{WCD\_DoG}_{\text{M1}}}\right) + 0.3\,\text{ReLU}\!\left(\frac{\text{Ghost} - \text{Ghost}_{\max}}{\text{Ghost}_{\max}}\right)$$
   - Khi chưa từng đạt Tier 1 (`has_strict_best == False`), `best.pt` sẽ cập nhật theo `S_pen` nhỏ nhất để luôn đảm bảo có checkpoint tốt nhất khi dừng sớm. Ngay khi có epoch đầu tiên đạt đủ 4 cổng Tier 1, `has_strict_best` bật `True` và chỉ các epoch vượt Tier 1 + đạt $S_{\text{val}}$ nhỏ hơn mới được ghi đè `best.pt`.

---

## 4. Điều chỉnh Nhẹ Tối ưu Hóa & Giám sát Chuỗi $N \in \{3,5,7\}$ (Không đổi Kiến trúc)

1. **Learning Rate & Lịch học:**
   - Giữ `lr_backbone = 1.0e-5`, hạ nhẹ `lr_unet` từ `2.5e-5` xuống **`1.8e-5`** (với warmup ep 1–2: `0.35 -> 0.70 -> 1.0` tại ep 3, cosine decay xuống `0.10` tại ep 15) để giảm tốc độ phình trọng số nhánh `OriginalUnet` sau Epoch 2.
2. **Log chi tiết & Dashboard chống điểm mù:**
   - Trong `training_history_finetune.csv`: ghi thêm 7 thành phần loss (`loss_charb`, `loss_dt`, `loss_dens`, `loss_stroke_w`, `loss_ghost`, `loss_cldice`, `loss_perc`) + `density_ratio_N3/N5/N7`, `ghost_N3/N5/N7`, `is_strict_best`.
   - Trong Cell 9 Dashboard: tách riêng đường `WCD_DoG` theo `N=3`, `N=5`, `N=7` và thêm biểu đồ `Ghost Pixels` + vùng an toàn `Density [0.94, 1.08]` để quan sát ngay hiện tượng lệch giữa `N=3` và `N=5, 7`.
3. **Tuân thủ Kỷ luật Validation vs Test (`ai_model_training_knowledge_notes.md` Mục 3.6):**
   - Toàn bộ hiệu chỉnh trọng số loss / Hard Gates chỉ căn cứ trên **Val set (`190` seqs / `420` frames)** và kết quả Run #1.
   - Tùy chọn **Hậu kiểm Model Soup trên Val set** (Cell 5b/Cell 7): sau Epoch 15, kiểm tra thêm tổ hợp tuyến tính $\theta(\alpha) = (1 - \alpha)\theta_{\text{M1}} + \alpha \theta_{\text{M3\_Best}}$ với $\alpha \in \{0.5, 0.7, 0.85, 1.0\}$ trên **Val set** (không chạm Benchmark #1 & #2) xem có giúp cân bằng thêm hình học chuỗi dài $N=5, 7$ hay không.
   - **Benchmark #1 (`198` seqs / `149` scenes) & Benchmark #2 (`24` frames)** chỉ chạy **1 lần cuối cùng** sau khi huấn luyện hoàn tất để nghiệm thu công bằng.

---

## 5. Kế hoạch Triển khai & Kiểm thử Xác minh (Verification Plan)

### Bước 1: Chỉnh sửa Mã nguồn trong `kaggle_finetune_sequence_final/` và `run_benchmark_suite.py`
1. Cập nhật [`ddp_worker_tps_sequence_final.py`](file:///E:/CODE/PBL/PBL6/kaggle_finetune_sequence_final/ddp_worker_tps_sequence_final.py):
   - Thay `AdaptiveLineartLoss` bằng phiên bản v2 có `ExactSTEDensityLoss`, `BidirectionalStrokeWidthLoss`, `GhostAndBackgroundCleanlinessLoss`, `SymmetricDT` và `Tversky-clDice` cân bằng.
   - Sửa `compute_wcd_batch_gpu` theo Eq. (15) bài báo gốc (`sigmoid(|area_p - area_g| / min)`), giữ tương thích log `wcd_dog_legacy` nếu cần đối chiếu.
   - Tắt AMP trong `run_validation_suite` (chạy pure FP32), bổ sung breakdown `density_ratio` và `ghost` theo $N \in \{3,5,7\}$.
   - Cập nhật `TwoFileCheckpointManager` + công thức `comp_score` mới chuẩn hóa theo Epoch-0 `M1` với 2 tầng Tier 1 Hard Gates / Tier 2 Penalized Score.
2. Cập nhật [`run_benchmark_suite.py`](file:///E:/CODE/PBL/PBL6/Dataset/sakuga_pipeline_sequence/final_dataset/sakuga_golden_benchmark_sequences_v2/run_benchmark_suite.py):
   - Đồng bộ hàm `compute_wcd_batch_gpu` fallback khi chạy độc lập.
   - Bổ sung kiểm tra `density_ratio` và `ghost` trong phần tổng kết kiểm định (giữ nguyên Wilcoxon 149 scenes trên `wcd_dog`, `skel_recall`, `f1`, `psnr`, `ssim`).
3. Cập nhật [`generate_kaggle_finetune_nb.py`](file:///E:/CODE/PBL/PBL6/kaggle_finetune_sequence_final/generate_kaggle_finetune_nb.py):
   - Cập nhật mô tả Cell 0, tham số Cell 5 (`lr_unet=1.8e-5`), biểu đồ Cell 9, rồi chạy tái sinh [`train_and_benchmark_tps_sequence_final_kaggle.ipynb`](file:///E:/CODE/PBL/PBL6/kaggle_finetune_sequence_final/train_and_benchmark_tps_sequence_final_kaggle.ipynb).

### Bước 2: Kiểm thử Cục bộ Trước khi Chạy Kaggle (Local Sanity & Gradient Tests)
1. **Unit test hành vi chống phình nét (qua `.venv\Scripts\python.exe`):**
   - Tạo ảnh GT nét mảnh `1-2px` và 3 bản dự đoán giả lập: `(A) Khớp chuẩn`, `(B) Phình dày +1px quanh nét`, `(C) Teo nét / đứt đoạn`, `(D) Nhiễu nền mờ 0.04`.
   - Xác nhận định lượng: `Loss(A) < Loss(B)`, `Loss(A) < Loss(C)`, `Loss(A) < Loss(D)`, gradient của `ExactSTEDensityLoss` + `BidirectionalStrokeWidthLoss` trên vùng viền nở của `(B)` mang dấu **dương** (ép giảm độ đậm nét ở biên) và không sinh `NaN`/`Inf` dưới `autocast('cuda', fp16)`.
2. **Kiểm tra `comp_score` & `WCD` Eq. (15):**
   - Xác nhận khi phình nét `(B)`, `WCD_paper(B) > WCD_paper(A)` và `comp_score(B) > comp_score(A)` (loại bỏ triệt để hiện tượng thưởng điểm khi tô đậm).
3. **Dry-run DDP Worker (`--dry_run`):**
   - Chạy 2 batches trên `TPSInbetweenOriginal` từ `model_latest.pt`, xác nhận khớp `158/158` tensors (`10,068,668` params) và `|M1(x) - M3_init(x)|_max == 0.0`.

### Bước 3: Tiêu chí Nghiệm thu Sau khi Huấn luyện Lại trên Kaggle (Hard Acceptance Targets)

| Tiêu chí (So với `M1_Original` và Run #1 `M3_Best`) | `M1_Original` (Bench #1) | Run #1 `M3_Best` (Bench #1) | **Mục tiêu Run #2 (Bench #1)** | **Mục tiêu Run #2 (Bench #2 - 17 frames)** |
| :--- | :---: | :---: | :---: | :---: |
| **`Density Ratio`** | `0.8211` | `1.1671` *(Phình)* | **`0.96 – 1.06`** *(Tâm `~1.00`)* | **`0.94 – 1.04`** *(Hết bết đậm)* |
| **`WCD_DoG` ($\downarrow$)** | `2.7869` | `2.7917` | **`< 2.74`** *(Thắng `M1` có $p < 0.01$ cả mean lẫn scene)* | **`<= 5.20`** *(Không tụt mạnh so với `5.14` của `M1`)* |
| **`Skel Recall` ($\uparrow$)** | `81.37%` | `86.93%` | **`>= 85.5%`** *(Giữ lợi thế chống đứt nét)* | **`>= 55.0%`** *(Vượt `M1` `50.5%`)* |
| **`Ghost Pixels` ($\downarrow$)** | `1,537.3 px` | `6,195.6 px` | **`<= 2,500 px`** *(Giảm $\ge 60\%$ so với Run #1)* | **`<= 4,800 px`** |
| **`PWCR` ($\uparrow$)** | `93.25%` | `89.64%` | **`>= 92.0%`** | **`>= 90.5%`** |
| **`Stroke Precision` ($\uparrow$)** | `0.9312` | `0.9228` | **`>= 0.935`** | **`>= 0.775`** |
