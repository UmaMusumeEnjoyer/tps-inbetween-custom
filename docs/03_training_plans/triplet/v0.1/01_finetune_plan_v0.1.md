# KẾ HOẠCH HUẤN LUYỆN TINH CHỈNH PHIÊN BẢN v0.1 (FINETUNE PLAN v0.1)
> **Mô hình Trọng tâm:** TPS-Inbetween (*Thin-Plate Spline-based Interpolation for Animation Line Inbetweening*)  
> **Phiên bản:** v0.1 (Experimental Baseline Finetune)  
> **Nền tảng Huấn luyện:** Kaggle Cloud GPU (NVIDIA Tesla T4 x2 hoặc NVIDIA Tesla P100 16GB VRAM)  
> **Ngày khởi tạo:** 06/10/2026

---

## 1. TỔNG QUAN & BỐI CẢNH BÀI TOÁN

### 1.1. Vấn đề của Mô hình Gốc (Pretrained Baseline)
Mô hình TPS-Inbetween gốc (`checkpoints/pretrained_baselines/tps-original/model_latest.pt`) được huấn luyện bởi tác giả trên tập dữ liệu **Mixamo 3D** (`ml240data`):
- **Ưu điểm:** Học tốt quy luật biến dạng hình học không gian 3D dạng vector mượt mà, nội suy được tại mốc thời gian $t \in (0, 1)$ tùy ý với độ dời chuyển động nhỏ ($\approx 3 - 5$px).
- **Hạn chế nghiêm trọng khi áp dụng cho Anime 2D (Domain Gap):**
  1. **Hiện tượng bóng ma (Ghosting Artifacts):** Xuất hiện vùng xám mờ (pixel xám cường độ 180 ~ 250) xung quanh nét vẽ do mạng UNet của mô hình gốc cố gắng pha trộn độ sáng (linear cross-fade) thay vì biến dạng hình học thực tế của nét vẽ 2D.
  2. **Méo nét & uốn lượn (Line Distortion / Wiggle):** Khi biên độ dịch chuyển của nhân vật tăng lên ($> 8$px), ma trận nội suy TPS bị gấp nếp (foldover), dẫn đến nét vẽ bị đứt đoạn hoặc méo mó.
  3. **Mờ nhạt các nét mảnh (Vanishing Hairlines):** Hàm mất mát $L_1$ truyền thống có xu hướng làm mờ (blurring) các chi tiết nét vẽ thanh mảnh của nhân vật anime.

### 1.2. Mục Tiêu Trọng Tâm của Phiên Bản v0.1
1. **Thích nghi miền dữ liệu (Domain Adaptation):** Huấn luyện mô hình làm quen với đặc trưng nét vẽ anime 2D thực tế (độ dày nét không đồng đều, nền trắng tuyệt đối 255, nét đen đậm 0, có khử răng cưa sub-pixel).
2. **Khắc phục Ghosting:** Ép mạng UNet học cách tổng hợp nét vẽ dứt khoát, loại bỏ tối thiểu 70% số pixel bóng mờ mờ nhạt mà không cần phụ thuộc vào ngưỡng hậu xử lý thô bạo (post-processing thresholding).
3. **Bảo toàn khả năng nội suy tùy ý $t \in (0, 1)$:** Đảm bảo mô hình giữ vững năng lực nội suy tại mọi thời điểm liên tục, không bị sụp đổ phân phối thời gian (time-step collapse) chỉ về mốc giữa $t=0.5$.
4. **Chứng minh tính khả thi trên Kaggle:** Chạy toàn bộ quá trình fine-tune trong môi trường Kaggle GPU hoàn toàn ổn định, không bị ngắt quãng do tràn RAM (CPU 30GB) hay tràn VRAM (GPU 16GB).

---

## 2. ĐẶC TẢ TẬP DỮ LIỆU HUẤN LUYỆN (DATASET SPECIFICATION)

Huấn luyện trực tiếp trên tập dữ liệu đã qua tiền xử lý, kiểm duyệt và đóng gói cục bộ:  
`data/processed/anime_tps_finetune_dataset_ready/`

### 2.1. Thống Kê Số Lượng & Phân Chia Tập
- **Tổng số Triplet:** 3,388 bộ ba khung hình $(I_0, I_{GT}, I_1)$ đạt chuẩn chất lượng cao.
- **Tập Huấn luyện (Train Split):** **2,880 triplets** (chiếm 85.0%).
- **Tập Thẩm định (Validation Split):** **508 triplets** (chiếm 15.0%, hoàn toàn độc lập, dùng để giám sát và chọn Best Checkpoint).

### 2.2. Cấu Trúc Thành Phần Dữ Liệu
```plaintext
anime_tps_finetune_dataset_ready/
├── train/                            # 2,880 thư mục clip (clip_000002, clip_000003, ...)
│   └── clip_XXXXXX/
│       ├── 000001.png                # Keyframe I_0 (thời điểm t = 0.0)
│       ├── 000002.png                # Ground Truth I_GT (thời điểm t)
│       └── 000003.png                # Keyframe I_1 (thời điểm t = 1.0)
├── val/                              # 508 thư mục clip đánh giá (clip_000001, clip_000015, ...)
│   └── clip_XXXXXX/
│       ├── 000001.png, 000002.png, 000003.png
├── matches/                          # Tọa độ điểm khớp GlueStick đã tính toán sẵn
│   └── clip_XXXXXX/
│       └── match_000001_000003.npy   # Mảng numpy (2, N, 2) tọa độ chuẩn hóa [0, 1]
└── triplets.csv                      # Metadata chi tiết từng mẫu (split, clip_id, t, disp, inliers, ...)
```

### 2.3. Đặc Trưng Dữ Liệu Quan Trọng cho Training
- **Độ phân giải:** $960 \times 540$ (chuẩn 540p chất lượng cao của Anime Triplet Dataset).
- **Phân bố nhãn thời gian $t$:**
  - $t = 0.5000$ (chiếm ~50%): Tương ứng nhịp chuyển động chuẩn on-twos.
  - $t = 0.3333$ & $t = 0.6667$ (chiếm ~30%): Tương ứng nhịp chuyển động on-threes.
  - $t = 0.2500$ & $t = 0.7500$ (chiếm ~20%): Tương ứng nhịp chuyển động on-fours.
  *Điều này cung cấp sự đa dạng cực lớn giúp mô hình học biểu diễn chuyển động thời gian liên tục.*
- **Độ dời chuyển động trung bình (Mean Displacement):** Dao động từ $0.15$px đến $14.8$px (trung vị $1.65$px), hoàn toàn nằm trong biên độ tối ưu của thuật toán biến dạng TPS.
- **Số điểm so khớp Inliers:** Trung bình $620$ cặp điểm/triplet (tối thiểu 89, tối đa 1,240 cặp điểm), đảm bảo mạng TPS giải ma trận tọa độ cực kỳ ổn định, không bị suy biến.

---

## 3. CHIẾN LƯỢC KIẾN TRÚC & ĐÓNG BĂNG MÔ HÌNH (FREEZING STRATEGY)

Mô hình TPS-Inbetween bao gồm 4 khối mạng chính. Để fine-tune hiệu quả trên tập dữ liệu anime mà không phá vỡ các đặc trưng đã học, chiến lược phân bổ như sau:

```
[KIẾN TRÚC TPS-INBETWEEN]
┌─────────────────────────────────────────────────────────────┐
│ 1. GlueStick Feature Matcher (SuperPoint + Wireframe + GNN) │
│    → KHÔNG NẠP VÀO RUNTIME KAGGLE                           │
│    (Tọa độ điểm khớp đã được đọc trực tiếp từ file .npy)    │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│ 2. TPS Coordinate Grid Solver (utils.py)                    │
│    → CHẠY TRONG torch.no_grad()                             │
│    (Nội suy giải tích ma trận Spline phi tham số)           │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│ 3. Contextnet (Feature Encoder)                             │
│    → ĐÓNG BĂNG (Freeze) hoặc LR siêu nhỏ (1e-6)             │
│    (Trích xuất đặc trưng đa mức coarse-to-fine)             │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│ 4. IFNet (Motion Refinement Network)                        │
│    → FINE-TUNE (Learning Rate = 2e-5)                       │
│    (Học tinh chỉnh trường dòng quang optical flow anime)     │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│ 5. Synthesis UNet (Alpha Blending & Residual Synthesis)     │
│    → FINE-TUNE (Learning Rate = 5e-5)                       │
│    (Học tổng hợp nét vẽ anime sắc sảo, triệt tiêu bóng ma)  │
└─────────────────────────────────────────────────────────────┘
```

> [!TIP]
> **Tối ưu VRAM then chốt:** Việc loại bỏ hoàn toàn module GlueStick khỏi runtime huấn luyện giúp **tiết kiệm ngay lập tức 1.5 GB VRAM** và loại bỏ 100% thời gian trích xuất đặc trưng trong vòng lặp forward pass!

---

## 4. HÀM MẤT MÁT TỔNG HỢP (COMPOSITE LOSS FUNCTION)

Huấn luyện ảnh nét vẽ đen trắng (Line Art) đòi hỏi hàm loss chuyên biệt vì nền trắng chiếm đa số (>90% diện tích ảnh), loss $L_1$ thông thường sẽ ưu tiên tái tạo nền trắng và bỏ qua chi tiết nét vẽ.

$$\mathcal{L}_{\text{total}} = \mathcal{L}_1 + \alpha \cdot \mathcal{L}_{\text{Perceptual}} + \beta \cdot \mathcal{L}_{\text{EDT}}$$

### 4.1. Thành phần Pixel-wise $L_1$ Loss
$$\mathcal{L}_1 = \frac{1}{|\Omega|} \sum_{p \in \Omega} |I_{\text{pred}}(p) - I_{GT}(p)|$$
Đảm bảo độ chính xác mức độ pixel và cường độ sáng tổng thể.

### 4.2. Thành phần Perceptual Loss ($\mathcal{L}_{\text{Perceptual}}$)
Sử dụng mạng VGG-16 đã tiền huấn luyện (đóng băng trọng số, đặt ở chế độ `eval()`), trích xuất đặc trưng tại tầng `relu2_2` và `relu3_3`:
$$\mathcal{L}_{\text{Perceptual}} = \sum_{l \in \{2, 3\}} \frac{1}{C_l H_l W_l} \|\phi_l(I_{\text{pred}}) - \phi_l(I_{GT})\|_2^2$$
Giúp bảo toàn hình thái đường cong và kết cấu đường nét ở mức độ ngữ nghĩa thị giác.

### 4.3. Thành phần Euclidean Distance Transform Loss ($\mathcal{L}_{\text{EDT}}$)
Đây là **thành phần cốt lõi** giải quyết triệt để rào cản domain gap của line art:
$$\mathcal{L}_{\text{EDT}} = \frac{1}{|\Omega|} \sum_{p \in \Omega} I_{\text{pred}}(p) \cdot D_{GT}(p)^2$$
- $D_{GT}(p)$ là bản đồ khoảng cách Euclidean từ pixel $p$ tới nét vẽ gần nhất của ảnh thực tế $I_{GT}$.
- Khi nét vẽ dự đoán $I_{\text{pred}}$ bị dịch chuyển dù chỉ $2 - 4$ pixel so với Ground Truth, $\mathcal{L}_{\text{EDT}}$ tạo ra một "giếng thế năng" hút các pixel nét về đúng vị trí chuẩn xác, ngăn chặn hiện tượng mờ nhòe do gradient vanishing của $L_1$.

### 4.4. Trọng số Khởi điểm Đề xuất cho v0.1
- $\alpha = 0.10$ (Perceptual Loss Weight)
- $\beta = 0.40$ (EDT Loss Weight)

---

## 5. THIẾT LẬP SIÊU THAM SỐ (HYPERPARAMETERS)

| Tham số | Giá trị v0.1 | Giải thích & Rationale |
|:---|:---:|:---|
| **Base Model Checkpoint** | `ckpt/model_latest.pt` | Trọng số gốc của tác giả TPS-Inbetween |
| **Optimizer** | `AdamW` | Khả năng điều chuẩn L2 Weight Decay ổn định |
| **Weight Decay** | $1 \times 10^{-4}$ | Chống hiện tượng quá khớp (overfitting) |
| **Learning Rate (IFNet)** | $2 \times 10^{-5}$ | Giữ luồng flow mượt, không làm hỏng coarse flow |
| **Learning Rate (UNet)** | $5 \times 10^{-5}$ | Đủ linh hoạt để học khử bóng ma và làm sắc nét |
| **LR Scheduler** | `CosineAnnealingLR` | $T_{\max} = 25$ epochs, $\eta_{\min} = 1 \times 10^{-6}$ |
| **Warmup** | 2 epochs | Tăng dần LR từ $10^{-6}$ để ổn định gradient ban đầu |
| **Physical Batch Size** | **4** | Phù hợp an toàn với 16GB VRAM GPU Kaggle |
| **Gradient Accumulation** | **2 bước** | Tạo ra **Effective Batch Size = 8** |
| **Độ phân giải Patch (Train)** | $512 \times 512$ (Random Crop) | Tối ưu VRAM cực hạn, tập trung vào vùng chứa nét vẽ |
| **Độ phân giải Val (Eval)** | $544 \times 960$ (Pad to 32) | Đánh giá trên toàn khung hình thực tế |
| **Mixed Precision** | **PyTorch AMP (FP16)** | Giảm 50% bộ nhớ activations, tăng 2x tốc độ tính |
| **Tổng số Epochs** | **25 epochs** | ~18,000 steps huấn luyện (đủ hội tụ tốt cho v0.1) |
| **Tần suất Validation** | Mỗi 1 epoch | Tính PSNR, SSIM, Ghost Pixels trên 508 val triplets |
| **Early Stopping** | Patience = 6 epochs | Dừng sớm nếu `val_PSNR` không cải thiện |

---

## 6. QUẢN LÝ TÀI NGUYÊN & HẠN MỨC KAGGLE (BUDGET ALLOCATION)

Kaggle cấp phát tài nguyên miễn phí với các ràng buộc:
- **Thời gian chạy tối đa mỗi phiên:** 12 giờ liên tục.
- **Hạn mức GPU hàng tuần:** 30 giờ GPU/tuần.
- **Bộ nhớ đĩa làm việc (`/kaggle/working`):** 19.5 GB.

### 6.1. Dự Toán Thời Gian & Ngân Sách GPU
- Tốc độ xử lý dự kiến: ~1.2 giây/batch (gồm 4 triplets, AMP FP16, Gradient Accumulation).
- 1 Epoch tập train ($2,880 \div 4 = 720$ batches): Mất khoảng **7 ~ 8 phút**.
- 1 Lượt Validation (508 triplets, no_grad): Mất khoảng **2.5 phút**.
- **Tổng thời gian 1 Epoch hoàn chỉnh:** $\approx 10$ phút.
- **Thời gian huấn luyện trọn gói 25 Epochs:** $\approx 250$ phút (**~4.1 giờ GPU**).
- **Đánh giá:** Chỉ tiêu thụ chưa đầy **15%** hạn mức GPU tuần của Kaggle, an toàn tuyệt đối!

### 6.2. Kiểm Soát Đĩa Cứng Kaggle
- Chỉ lưu trữ 3 checkpoints then chốt:
  1. `model_tps_v0.1_best.pt` (Checkpoint có `val_PSNR` cao nhất).
  2. `model_tps_v0.1_latest.pt` (Checkpoint của epoch gần nhất phục vụ resume).
  3. `model_tps_v0.1_epoch15.pt` (Mốc trung gian).
- Tổng dung lượng checkpoint trên đĩa Kaggle: $< 1.0$ GB (giới hạn 19.5 GB không bao giờ bị đe dọa).

---

## 7. TIÊU CHÍ NGHIỆM THU PHIÊN BẢN v0.1 (ACCEPTANCE CRITERIA)

Sau khi hoàn thành 25 epochs và tải checkpoint `model_tps_v0.1_best.pt` về máy trạm cục bộ, mô hình được nghiệm thu đạt chuẩn v0.1 khi thỏa mãn các điều kiện sau trên tập 508 validation triplets:

| Chỉ số Đánh giá | Mô hình Gốc (Mixamo 3D) | Mục tiêu Nghiệm thu v0.1 |
|:---|:---:|:---:|
| **PSNR (dB)** | $\approx 11.2 \pm 0.8$ dB | **$\ge 16.5$ dB** |
| **SSIM** | $\approx 0.67 \pm 0.05$ | **$\ge 0.82$** |
| **Số Pixel Bóng ma (180 ~ 250)** | $> 5,200$ pixels | **$< 800$ pixels** (Giảm $\ge 80\%$) |
| **Line Tortuosity (Độ méo nét)** | $\approx 1.48$ | **$< 1.25$** |
| **Khả năng nội suy tùy ý $t$** | Đồng đều mọi $t$ | Sai số giữa các mốc $t \in \{0.25, 0.33, 0.5, 0.67\}$ không chênh lệch quá $1.5$ dB |
