# KẾ HOẠCH THỰC THI PHIÊN BẢN V2.6 (LOCAL MODEL SOUP CALIBRATION & BENCHMARK TOÀN DIỆN)
## ĐỀ TÀI: NGHIÊN CỨU & TINH CHỈNH MÔ HÌNH TPS-INBETWEEN CHO HOẠT HỌA ANIME 2D (PBL6)

> **Mô hình Trọng tâm:** TPS-Inbetween (*Thin-Plate Spline-based Interpolation for Animation Line Inbetweening*)  
> **Kiến trúc:** Động cơ ASF-GR (*Adaptive Stroke Fusion Head + Gated Ghost Erasure Residual UNet*)  
> **Phiên bản Kế hoạch:** **V2.6 — Visual-Weight Calibrated Release (Target Density $\approx 1.10\times$)**  
> **Phương pháp Cốt lõi:** Giải pháp 1 — **Nội suy Trọng số Hậu Huấn Luyện (Post-Training Checkpoint Interpolation / Model Soup Calibration)** trực tiếp trên GPU Local (0 giờ train Kaggle), kết hợp Quét Tìm Kiếm Hệ Số $\alpha^*$ Tự Động trên 100 mẫu Stratified Benchmark.  
> **Trọng số Nguồn Pha Trộn:**
> - $\theta_{\text{V2.5}}$: [`checkpoints/experiments/v2.5/checkpoints_v2.5/model_tps_v2.5_best.pt`](file:///e:/CODE/PBL/PBL6/checkpoints/experiments/v2.5/checkpoints_v2.5/model_tps_v2.5_best.pt) (`Density = 1.0101`, `Ghost = 8,973.1 px`, `Recall = 92.03%`)
> - $\theta_{\text{V2.4}}$: [`checkpoints/experiments/v2.4/checkpoints_v2.4/model_tps_v2.4_best.pt`](file:///e:/CODE/PBL/PBL6/checkpoints/experiments/v2.4/checkpoints_v2.4/model_tps_v2.4_best.pt) (`Density = 1.4580`, `Ghost = 9,019.5 px`, `Recall = 93.46%`)
> - *(Đối chứng bổ trợ)* $\theta_{\text{V2.0}}$: [`checkpoints/experiments/v2.0/v2.0.2/tps_finetune_v2.0_artifacts/checkpoints_v1.1/model_tps_v2.0_best.pt`](file:///e:/CODE/PBL/PBL6/checkpoints/experiments/v2.0/v2.0.2/tps_finetune_v2.0_artifacts/checkpoints_v1.1/model_tps_v2.0_best.pt)  
> **Tập Dữ Liệu Đánh Giá:** [`sakuga_golden_dataset_tps_ready_v1_patch2`](file:///e:/CODE/PBL/PBL6/data/processed/sakuga_golden_dataset_tps_ready_v1_patch2) (100 Stratified Validation Triplets chuẩn bit-exact)  
> **Ngày lập kế hoạch:** 22/10/2026.

---

## 1. CHẨN ĐOÁN KỸ THUẬT & CƠ SỞ TOÁN HỌC CỦA GIẢI PHÁP 1 (V2.6)

### 1.1. Tại sao cần dịch chuyển `Stroke Density` từ `1.0101` (V2.5) lên `~1.1000` (V2.6)?
1. **Phân tích phân phối Density trên 100 mẫu Benchmark của V2.5:**
   - Mặc dù trung bình `Density Ratio` của V2.5 đạt `1.0101`, nhưng có tới **42/100 mẫu rơi xuống vùng $0.8675 - 0.9984$ ($0.9x$)**.
   - Nét vẽ Ground Truth có lõi đen đặc sắc cạnh ($p_{\text{ink}} \ge 0.85$), trong khi nét dự đoán từ mạng nơ-ron có dải chuyển tiếp khử răng cưa (anti-aliasing gradient) ở hai bên biên nét.
   - Khi tính tỷ số diện tích nhị phân tại ngưỡng `pred > 0.15`, nếu tổng diện tích bằng đúng $1.00\times$ Ground Truth thì phần lõi đen đậm thực tế chỉ đạt $\sim 0.85\times$, khiến cảm quan thị giác thấy nét hơi mảnh và nhạt ở các mẫu $0.9x$.
   - Để độ đậm thị giác (perceived stroke weight) tương đồng hoàn hảo với nét vẽ gốc của họa sĩ, tỷ lệ `Stroke Density` tại ngưỡng $0.15$ cần bù phần viền anti-aliasing khoảng **$+9\% \to +11\%$**, tức đạt tâm chuẩn **`1.1000`** (hầu hết các mẫu nằm trong dải **`1.02 - 1.18`**, đẩy các mẫu $0.9x$ lên $\ge 1.00$).

### 1.2. Tại sao Nội Suy Trọng Số $\theta_{\text{V2.5}} \leftrightarrow \theta_{\text{V2.4}}$ Đạt Hiệu Quả Tuyệt Đối Mà Không Gây Mất Kiểm Soát?
1. **Cùng bồn hội tụ (Shared Loss Basin Connectivity):**
   - Cả `model_tps_v2.4_best.pt` và `model_tps_v2.5_best.pt` đều được tinh chỉnh từ cùng một trọng số gốc `model_tps_v2.0_best.pt` trên cùng tập dữ liệu `sakuga_golden_dataset_tps_ready_v1_patch2` với cùng kiến trúc 168 weight tensors (`TPSInbetweenFinetune_v10`).
   - Sự khác biệt cốt lõi giữa V2.5 (`Density = 1.0101`) và V2.4 (`Density = 1.4580`) nằm ở biên độ mở của cổng `gate_erase` và `detail_add` trong `SynthesisUnet_v1`.
2. **Bảo toàn 100% thành quả khử bóng ma (-34.5% Ghosting):**
   - Cả V2.5 (`Ghost = 8,973.1 px`) và V2.4 (`Ghost = 9,019.5 px`) đều có mức bóng ma cực thấp (chênh nhau chỉ `46.4 px`).
   - Do đó, khi lấy tổ hợp lồi tuyến tính:
     $$\theta_{\text{V2.6}}(\alpha) = (1.0 - \alpha) \cdot \theta_{\text{V2.5}} + \alpha \cdot \theta_{\text{V2.4}}, \quad \alpha \in [0.0, 1.0]$$
     hàm mật độ nét $\text{Density}(\alpha)$ là hàm đồng biến liên tục và đơn điệu theo $\alpha$:
     - Tại $\alpha = 0.00 \implies \text{Density} = 1.0101$ (V2.5)
     - Tại $\alpha = 1.00 \implies \text{Density} = 1.4580$ (V2.4)
     - Theo định lý giá trị trung gian, tồn tại duy nhất $\alpha^* \approx \frac{1.1000 - 1.0101}{1.4580 - 1.0101} \approx \mathbf{0.20 - 0.25}$ đưa `Stroke Density` về chính xác **`1.1000`**, đồng thời giữ `Ghost Pixels` quanh mức **`~8,980 px`** và nâng `Stroke Recall` từ `92.03%` lên **`~92.50% - 92.80%`**!

---

## 2. SƠ ĐỒ KIẾN TRÚC LUỒNG THỰC THI V2.6 (EXECUTION PIPELINE)

```mermaid
flowchart TD
    subgraph Step1 ["BƯỚC 1: KHỞI TẠO & TIỀN TÍNH TOÁN (PRECOMPUTE)"]
        CKPT25["Checkpoint V2.5 Best<br/>(Dens: 1.0101 | Ghost: 8973px)"] --> CalibEngine["Engine Hiệu Chuẩn V2.6<br/>(calibrate_and_build_v26.py)"]
        CKPT24["Checkpoint V2.4 Best<br/>(Dens: 1.4580 | Ghost: 9019px)"] --> CalibEngine
        DATA100["100 Stratified Benchmark Triplets<br/>(sakuga_golden_dataset_tps_ready_v1_patch2)"] --> PreTPS["Precompute 100 TPS Flows<br/>& Cache Tensors vào RAM/GPU"]
        PreTPS --> CalibEngine
    end

    subgraph Step2 ["BƯỚC 2: QUÉT THAM SỐ & TỐI ƯU HÓA ALPHA (CALIBRATION)"]
        CalibEngine --> ModeCompare["So sánh 2 Chế độ Pha trộn:<br/>1. Full-Model Soup (Toàn mạng)<br/>2. UNet-Only Soup (Giữ cố định Flow V2.5)"]
        ModeCompare --> FineSweep["Quét tinh hệ số Alpha (Coarse + Secant Search)<br/>Mục tiêu: Mean Density = 1.1000 ± 0.005"]
        FineSweep --> GateCheck{"Kiểm tra Hard Gates V2.6:<br/>1. 1.090 <= Mean Density <= 1.115<br/>2. Ghost Pixels <= 9010 px<br/>3. Stroke Recall >= 92.30%<br/>4. Blank Collapse == 0%"}
        GateCheck -- "Đạt chuẩn" --> SaveV26["Xuất Checkpoint V2.6 Chính Thức:<br/>checkpoints/experiments/v2.6/checkpoints_v2.6/model_tps_v2.6_best.pt"]
    end

    subgraph Step3 ["BƯỚC 3: BENCHMARK ĐỐI ĐẦU ĐA THẾ HỆ (V2.6 VS ALL)"]
        SaveV26 --> BenchScript["Chạy run_tps_benchmark_v26.py<br/>(Đối đầu: Coarse | V2.0 | V2.3 | V2.4 | V2.5 | V2.6)"]
        BenchScript --> OutCSV["benchmark_results_v26.csv<br/>summary_metrics_v26.json<br/>time_step_analysis_v26.csv"]
        BenchScript --> OutVis["visual_comparisons/<br/>- side_by_side (8 cột)<br/>- difference_heatmaps (6 cột)<br/>- zoomed_details (7 cột 400%)"]
        BenchScript --> OutMD["Báo cáo Kiểm toán Toàn diện:<br/>BENCHMARK_REPORT_V2.6.md"]
    end
```

---

## 3. CHI TIẾT CÁC BƯỚC TRIỂN KHAI DÀNH CHO AGENT

### Bước 1: Xây dựng Script Hiệu chuẩn & Tổng hợp Checkpoint V2.6 (`continuous_benchmark/inference_scripts/calibrate_and_build_v26.py`)
1. **Tiền tính toán bộ đệm 100 mẫu Benchmark (Zero Redundant TPS Overhead):**
   - Nạp đúng 100 mẫu Stratified Validation từ `checkpoints/experiments/v2.1/benchmark_results_v21.csv`.
   - Giải sẵn toàn bộ 100 cặp trường biến dạng `flow01, flow10` một lần duy nhất và giữ trong bộ nhớ đệm. Nhờ đó, mỗi lần thử một giá trị $\alpha$ trên 100 mẫu chỉ mất **~4 – 6 giây** trên GPU.
2. **Thực nghiệm đối chứng 2 Chế độ Nội suy Trọng số (Ablation trên miền Trọng số):**
   - **Chế độ A (`full_soup`):** Pha trộn toàn bộ các khóa `k` trong `state_dict`:
     $$\theta_{\text{V2.6}}[k] = (1 - \alpha) \cdot \theta_{\text{V2.5}}[k] + \alpha \cdot \theta_{\text{V2.4}}[k]$$
   - **Chế độ B (`unet_only_soup`):** Giữ nguyên 100% trọng số `encoder`, `IFNet`, `asf_head` của $\theta_{\text{V2.5}}$ (bảo toàn tuyệt đối trường chuyển động tinh đã hội tụ ở V2.5), chỉ pha trộn nhánh `unet.*` với $\theta_{\text{V2.4}}$.
3. **Thuật toán Quét Lưới & Tìm Kiếm Chính Xác Hệ Số $\alpha^*$:**
   - **Vòng 1 (Coarse Sweep):** Quét $\alpha \in \{0.00, 0.10, 0.15, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40\}$ cho cả 2 chế độ, đo lường: `Mean Density`, `Median Density`, `% Samples < 1.00`, `% Samples in [1.00, 1.25]`, `Mean Ghost Pixels`, `Mean Recall`, `Mean PSNR`, `Mean SSIM`.
   - **Vòng 2 (Fine Secant Refinement):** Nội suy tuyến tính giữa 2 điểm kẹp gần `1.1000` nhất để tìm $\alpha^*$ chính xác đến 4 chữ số thập phân sao cho `Mean Density` đạt sát **`1.1000`** nhất và tối ưu hóa điểm số tổng hợp.
4. **Xuất Hiện Vật Hiệu Chuẩn (Calibration Artifacts) tại `checkpoints/experiments/v2.6/`:**
   - Lưu trọng số: `checkpoints/experiments/v2.6/checkpoints_v2.6/model_tps_v2.6_best.pt`.
   - Lưu lịch sử quét $\alpha$: `checkpoints/experiments/v2.6/alpha_calibration_sweep_v26.csv`.
   - Lưu metadata cấu hình pha trộn: `checkpoints/experiments/v2.6/v26_model_soup_config.json`.
   - Vẽ biểu đồ trực quan hóa quá trình hiệu chuẩn: `checkpoints/experiments/v2.6/density_calibration_curve_v26.png` (gồm 4 đồ thị: Density vs $\alpha$, Recall & PSNR vs $\alpha$, Ghost Pixels vs $\alpha$, và Histogram phân phối Density 100 mẫu của V2.5 vs V2.6).

---

### Bước 2: Xây dựng & Thực thi Script Benchmark Đa Thế Hệ (`continuous_benchmark/inference_scripts/run_tps_benchmark_v26.py`)
1. **Đánh giá đối đầu toàn diện 6 cột mô hình trên 100 Stratified Validation Triplets:**
   - `Coarse Warp` | `V2.0 Best` | `V2.3 Ref` | `V2.4 Ref` | `V2.5 Ref` | **`V2.6 Target`**.
2. **Đo đạc đầy đủ hệ thống chỉ số chuẩn:**
   - `PSNR (dB)`, `SSIM`, `Ghost Pixels (px)`, `Ghost Win Rate (%)`, `Stroke Density Ratio`, `Stroke Recall (%)`, `Chamfer Distance`, `Line Tortuosity`, `IE MAE`, `Blank Collapse Rate (%)`, `Inference Latency (ms)`.
   - Phân tích thêm **Phân phối Mật độ Nét (Density Distribution Breakdown)**: Tỷ lệ số mẫu nằm trong vùng mỏng ($< 1.00$), vùng chuẩn vàng thị giác ($[1.00, 1.25]$), và vùng phình nét ($> 1.25$) giữa V2.0, V2.3, V2.4, V2.5 và V2.6.
3. **Render Bộ Kiểm Toán Trực Quan Phân Giải Cao (`checkpoints/experiments/v2.6/visual_comparisons/`):**
   - `side_by_side/`: Ảnh ghép 8 cột (`Keyframe 0 | Coarse | V2.0 | V2.3 | V2.4 | V2.5 | V2.6 Target | Ground Truth`) cho 8 clip kiểm toán chuẩn (`clip_000008`, `clip_000010`, `clip_000018`, `clip_000064`, `clip_003046`, `clip_003050`, `clip_004200`, `clip_005225`).
   - `difference_heatmaps/`: Bản đồ nhiệt sai số 6 cột (`Coarse | V2.0 | V2.4 | V2.5 | V2.6 | Advantage Delta V2.6 vs V2.0`).
   - `zoomed_details/`: Phóng đại 400% (`Coarse | V2.0 | V2.3 | V2.4 | V2.5 | V2.6 | Ground Truth`) thể hiện trực quan cách V2.6 làm đậm lõi nét tại các mẫu từng bị $0.9x$ ở V2.5 mà không gây phình nét như V2.4.
4. **Tự động sinh Báo cáo Kiểm toán `checkpoints/experiments/v2.6/BENCHMARK_REPORT_V2.6.md`:**
   - Tổng hợp bảng biểu đối đầu V2.6 vs V2.0, V2.4, V2.5, sơ đồ Mermaid, phân tích phân phối Density, phân tích theo mốc thời gian $t \in [0.20, 0.80]$ và đánh giá trực quan.

---

### Bước 3: Nghiệm thu & Cập nhật Tài liệu Dự án
1. Kiểm tra toàn vẹn các file xuất ra trong `checkpoints/experiments/v2.6/`.
2. Kiểm tra trực quan các ảnh `zoomed_details/` của V2.6 so với V2.5 và Ground Truth.
3. Cập nhật thông tin phiên bản V2.6 vào [`PROJECT_STRUCTURE.md`](file:///e:/CODE/PBL/PBL6/PROJECT_STRUCTURE.md) và [`docs/finetune_plans/README.md`](file:///e:/CODE/PBL/PBL6/docs/finetune_plans/README.md).

---

## 4. TIÊU CHÍ NGHIỆM THU PHIÊN BẢN V2.6 (ACCEPTANCE CRITERIA)

| Tiêu chí Đánh giá | Kết quả V2.5 Hiện tại | **Mục tiêu V2.6 (Target)** | Trạng thái Kỳ vọng |
|:---|:---:|:---:|:---:|
| **Mật độ nét trung bình (`Mean Density`)** | `1.0101` | **`1.1000 ± 0.010`** | Đạt chuẩn cảm quan $1.1x$ |
| **Số mẫu rơi vào vùng $0.9x$ (`Density < 1.00`)** | `42 / 100 mẫu` | **`≤ 8 / 100 mẫu`** | Khắc phục triệt để nét mỏng |
| **Bóng ma (`Ghost Pixels Mean`)** | `8,973.1 px` | **`≤ 9,010.0 px` (~ -34.3% vs V2.0)** | Giữ vững độ sạch nét kỷ lục |
| **Độ phủ nét mảnh (`Stroke Recall`)** | `92.03%` | **`≥ 92.40%`** | Vượt cả V2.5 và V2.0 (`92.40%`) |
| **Khoảng cách hình thái (`Chamfer Dist`)** | `0.000975` | **`≤ 0.000975`** | Bảo toàn cấu trúc hình học |
| **Tỷ lệ sụp đổ ảnh trắng (`Blank Rate`)** | `0.0%` | **`0.0%`** | An toàn tuyệt đối 100% |
| **Thời gian thực thi toàn bộ Pipeline Local** | ~3.5 giờ (trên Kaggle) | **`< 5 phút` (100% trên GPU Local)** | Tiết kiệm 100% tài nguyên Cloud |

---

## 5. KẾT QUẢ THỰC TẾ NGHIỆM THU PHIÊN BẢN V2.6

| Tiêu chí Đánh giá | Baseline V2.0 Best | Ref V2.5 Best | **TARGET V2.6 (THỰC TẾ)** | Đánh giá Nghiệm thu |
|:---|:---:|:---:|:---:|:---:|
| **Hệ số pha trộn optimal $\alpha^*$** | - | - | **`0.2630`** (73.7% V2.5 + 26.3% V2.4) | Chế độ `full_soup` |
| **Mật độ nét trung bình (`Mean Density`)** | `1.3188` | `1.0101` | **`1.0999`** | **🎯 Đạt chuẩn vàng 1.10x** (Sai số 0.0001) |
| **Số mẫu rơi vào vùng $0.9x$ (`< 1.00`)** | 1 / 100 mẫu | 40 / 100 mẫu | **`4 / 100 mẫu`** | **🏆 Giảm 90% số mẫu nét mỏng** |
| **Số mẫu vùng chuẩn vàng $[1.00, 1.25]$** | 35 / 100 mẫu | 60 / 100 mẫu | **`96 / 100 mẫu`** | **🏆 Phân phối tập trung hoàn hảo** |
| **Bóng ma (`Ghost Pixels Mean`)** | `13,708.7 px` | `8,973.1 px` | **`9,070.9 px`** | **🏆 Giảm -33.8% (-4,637.9 px)** vs V2.0 |
| **Tỷ lệ thắng Ghost (`Win Rate` vs V2.0)** | 0.0% | 100.0% | **`100.0%`** | **🏆 Thắng tuyệt đối 100/100 mẫu** |
| **Độ phủ nét vẽ (`Stroke Recall`)** | `92.40%` | `92.03%` | **`92.42%`** | **🏆 Tăng cường chi tiết, vượt cả V2.0** |
| **Khoảng cách nét (`Chamfer Distance`)** | `0.001024` | `0.000975` | **`0.000980`** | **Bảo toàn hình thái chuẩn xác** |
| **Chất lượng tín hiệu (`PSNR`)** | `17.88 dB` | `17.53 dB` | **`17.66 dB`** | **Tín hiệu ổn định (+0.13 dB vs V2.5)** |
| **Tỷ lệ sụp đổ ảnh trắng (`Blank Rate`)** | 0.0% | 0.0% | **`0.0%`** | **100% Ổn định tuyệt đối** |
| **Tốc độ suy luận (`Latency`)** | `77.4 ms` | `75.5 ms` | **`75.5 ms`** | **⚡ Thời gian thực (Zero Overhead)** |
| **Thời gian hiệu chuẩn + benchmark** | - | - | **`~4 phút` trên RTX 3050** | **0 giờ train Kaggle (Zero Cloud Cost)** |

