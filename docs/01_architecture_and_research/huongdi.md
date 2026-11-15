# Hướng Đi Fine-tune TPS-Inbetween cho Anime 2D
> **Mục tiêu tổng quát:** Tự xây dựng dataset từ anime 2D thực tế, fine-tune model TPS-Inbetween để model vừa giữ nguyên khả năng nội suy với mọi giá trị $t$ bất kỳ, vừa được chuyên biệt hóa cho phong cách nét vẽ anime 2D, tạo ra frame inbetween sạch nét — hạn chế tối đa chỉnh sửa thủ công của họa sĩ.

---

## I. Bức Tranh Toàn Cảnh

```
[Hiện tại]
  Model TPS-Inbetween (train trên ml240data - Mixamo 3D)
       │
       ├── Ưu điểm: Hoạt động tốt với ảnh có motion nhỏ (~4px), nét vector 3D
       └── Nhược điểm: Domain gap lớn với anime 2D thực, ghosting, méo nét khi motion lớn

[Mục tiêu sau fine-tune]
  Model TPS-Inbetween (anime-finetuned)
       │
       ├── Giữ nguyên: Khả năng nội suy với t bất kỳ trong (0, 1)
       ├── Cải thiện: Nét vẽ sạch, không ghosting, không méo với anime 2D
       └── Đạt được: Tạo frame inbetween chính xác, hạn chế clean-up thủ công
```

---

## II. Ba Giai Đoạn Thực Hiện

---

### GIAI ĐOẠN 1 — Xây Dựng Dataset Anime 2D

#### 1.1. Nguyên tắc lấy mẫu (Timing & Spacing)

Model TPS-Inbetween được thiết kế cho **displacement trung bình ~4px** giữa hai keyframe. Để fine-tune hiệu quả:

| Thông số | Giá trị đề xuất | Lý do |
|:---|:---:|:---|
| **Video nguồn** | Anime 24fps, nét rõ ràng | Chuẩn công nghiệp Nhật Bản |
| **Bước nhảy keyframe** ($\Delta k$) | **2 ~ 4 frames** | Displacement ≈ 4–10px, trong design envelope của model |
| **Tương ứng thời gian** ($\Delta t$) | **0.083s ~ 0.167s** ở 24fps | Tương đương "On 2s" đến "On 3s" trong anime |
| **Ground truth $I_{GT}$** | Frame giữa thực tế | $t = \frac{k_{GT} - k_0}{k_1 - k_0}$ |
| **Tối đa $\Delta k$ cho phép** | 6 frames | Cần RANSAC verify displacement < 20px |

> **Ví dụ cụ thể:** Lấy frame_0 làm $I_0$, frame_3 làm $I_1$, thì:
> - frame_1 là ground truth với $t = 1/3 \approx 0.333$
> - frame_2 là ground truth với $t = 2/3 \approx 0.667$

#### 1.2. Pipeline Trích Xuất Dữ liệu (từ anime màu → line art dataset)

```
BƯỚC 1: Thu thập video nguồn
─────────────────────────────
  Nguồn: Đoạn cắt anime 2D chất lượng cao (1080p, 24fps)
  Yêu cầu:
    - Cảnh nét rõ ràng, không blur do chuyển động nhanh (motion blur)
    - Nhân vật đủ lớn trong frame (chiếm >30% diện tích ảnh)
    - Không cảnh cắt cảnh đột ngột (scene cut) giữa đoạn
    - Thời lượng tối thiểu: 5s mỗi đoạn (= 120 frame ở 24fps)
    - Mục tiêu: 5,000+ triplet (I_0, I_GT, I_1)

BƯỚC 2: Trích xuất frame
─────────────────────────
  Công cụ: ffmpeg
  Lệnh:
    ffmpeg -i anime_clip.mp4 -vf "fps=24" frames/%05d.png
  Đầu ra: Chuỗi PNG 24fps đánh số thứ tự

BƯỚC 3: Tách nền nhân vật
──────────────────────────
  Công cụ: anime-segmentation (đã có trong TestDataset/)
  Đầu ra: Frame RGBA, nền trong suốt, giữ nguyên màu nhân vật
  Kiểm soát chất lượng:
    - Không "alpha bleeding" quanh rìa nét
    - Không cắt mất phần nét vẽ của nhân vật

BƯỚC 4: Trích xuất line art chi tiết
──────────────────────────────────────
  Công cụ: AniLines (đã có trong TestDataset/AniLines-Anime-Lineart-Extractor/)
  Cài đặt: Dùng mode "detail" (như ChayGoAnimeLinesDetail)
  Đầu ra: Line art grayscale (white background, black/gray lines)
  
  ❌ KHÔNG ĐƯỢC làm:
    - Binary hóa 1-bit (NoGay) → mất gradient → GlueStick sinh outlier
    - Dùng Anime2Sketch nếu nét bị quá mượt (mất chi tiết nhỏ)

BƯỚC 5: Chuẩn hóa và làm mượt nét
────────────────────────────────────
  Công cụ: smooth_lines.py (có trong TestDataset/ChayGoAnimeLinesDetail_NoBG/)
  Mục đích:
    - Giữ anti-aliasing sub-pixel dọc rìa nét
    - Chuẩn hóa nền về trắng tinh (255)
    - Giảm nhiễu nhỏ không phải nét vẽ
  Đầu ra: _smooth.png — định dạng chuẩn giống ml240data
  
  Kiểm tra chất lượng:
    - White% > 85% (nền trắng chiếm đa số)
    - Dark(<30)% > 1% (đủ nét vẽ)
    - Laplacian variance > 1000 (nét đủ sắc nét)

BƯỚC 6: Lọc và tạo triplet
────────────────────────────
  Với mỗi cặp (frame_k, frame_{k+Δk}):
    1. Chạy GlueStick để đếm số điểm khớp
    2. RANSAC lọc outlier → nếu còn ≥ 20 inlier sạch: giữ cặp này
    3. Đo displacement trung bình → nếu < 20px: hợp lệ cho training
    4. Tạo triplet cho TẤT CẢ frame trung gian:
         t = (k_GT - k_0) / (k_1 - k_0)
    5. Lưu file .npy chứa matched keypoints (đã qua RANSAC)
```

#### 1.3. Cấu trúc Dataset

```
E:\Dataset\AnimeLine_FineTune\
├── train\
│   ├── clip_001\         ← mỗi thư mục là 1 đoạn anime đã xử lý
│   │   ├── 00001_smooth.png
│   │   ├── 00002_smooth.png
│   │   └── ...
│   └── clip_002\
├── val\                  ← 10-15% dữ liệu dùng để đánh giá trong lúc train
├── triplets.csv          ← Danh sách tất cả triplet: (I_0_path, I_1_path, I_GT_path, t)
└── matches\
    ├── clip_001\
    │   ├── 00001_00003.npy    ← matched keypoints I_0=frame1, I_1=frame3
    │   └── ...
```

**File `triplets.csv` mẫu:**
```csv
i0_path,i1_path,igt_path,t
train/clip_001/00001_smooth.png,train/clip_001/00004_smooth.png,train/clip_001/00002_smooth.png,0.333
train/clip_001/00001_smooth.png,train/clip_001/00004_smooth.png,train/clip_001/00003_smooth.png,0.667
...
```

#### 1.4. Quy mô Dataset Mục tiêu

| Loại dữ liệu | Số lượng tối thiểu | Ghi chú |
|:---|:---:|:---|
| Đoạn anime nguồn | 30+ đoạn cắt | Đa dạng hành động, nhiều nhân vật |
| Frame line art sau xử lý | 5,000+ frame | Sau khi lọc chất lượng |
| Triplet (I_0, I_GT, I_1) hợp lệ | **5,000+ triplet** | Tương đương ml240data (>10k triplet) |
| Tỷ lệ train/val | 85% / 15% | |

---

### GIAI ĐOẠN 2 — Fine-tune Model

#### 2.1. Chiến lược Fine-tune (Không train từ đầu)

```
[Các thành phần model TPS-Inbetween]

  ┌─────────────────────────────────────────────────────┐
  │  GlueStick (SuperPoint + Wireframe + Matcher)        │  ← ĐÓNG BĂNG (freeze)
  │  → Giữ nguyên khả năng feature matching              │
  └─────────────────────────────────────────────────────┘
                          ↓
  ┌─────────────────────────────────────────────────────┐
  │  TPS Solver (utils.py)                               │  ← KHÔNG THAY ĐỔI CODE
  │  → Chỉ dùng λ=0.2, border anchors, interior anchors  │    (đã fix trong pipeline)
  └─────────────────────────────────────────────────────┘
                          ↓
  ┌─────────────────────────────────────────────────────┐
  │  IFNet (Motion Refinement Network)                   │  ← FINE-TUNE (lr nhỏ)
  │  → Học cách tinh chỉnh flow cho anime 2D             │
  └─────────────────────────────────────────────────────┘
                          ↓
  ┌─────────────────────────────────────────────────────┐
  │  UNet (Synthesis & Alpha Blending)                   │  ← FINE-TUNE (lr nhỏ)
  │  → Học cách tổng hợp nét vẽ anime, loại ghosting     │
  └─────────────────────────────────────────────────────┘
```

**Lý do đóng băng GlueStick:** GlueStick đã được train trên hàng triệu ảnh, khả năng feature matching tốt với cả line art. Nếu train lại sẽ mất khả năng này và cần GPU/thời gian rất lớn.

#### 2.2. Hàm Mất mát (Loss Function) Chuyên biệt cho Nét Vẽ

```python
# Loss tổng hợp = L1 + Perceptual + EDT
L_total = L1 + α * L_Perceptual + β * L_EDT

# L1: Pixel-wise loss cơ bản
L1 = mean(|I_pred - I_GT|)

# L_Perceptual (VGG): Đo lường sự tương đồng về cấu trúc tổng thể
L_Perceptual = ||VGG(I_pred) - VGG(I_GT)||^2

# L_EDT: Kéo nét vẽ về đúng vị trí - QUAN TRỌNG NHẤT cho line art
# D_GT(p) = khoảng cách Euclidean từ pixel p tới nét vẽ gần nhất trong I_GT
L_EDT = mean(I_pred(p) * D_GT(p)^2 for p in image)
```

**Tại sao cần EDT Loss:** L1 thuần túy bị "gradient vanishing" với line art — nếu nét dự đoán lệch 2px, L1 không biết cần kéo nét về hướng nào. EDT tạo ra "giếng thế năng" hút nét về đúng vị trí ground truth.

**Hệ số đề xuất khởi đầu:** α = 0.1, β = 0.5 (tune trong quá trình train).

#### 2.3. Cấu hình Huấn luyện

```python
# Hyperparameters khởi đầu
learning_rate_IFNet = 1e-5   # Nhỏ để không "quên" kiến thức cũ
learning_rate_UNet  = 1e-5
batch_size          = 4      # Tùy VRAM GPU (4 với 8GB, 8 với 16GB)
epochs              = 50     # ~20 epoch ổn định, 50 để hội tụ tốt
optimizer           = AdamW  # Weight decay 1e-4

# Learning rate schedule
scheduler = CosineAnnealingLR(T_max=50, eta_min=1e-7)

# Validation metric
monitor = PSNR   # Lưu checkpoint khi val_PSNR tốt nhất
```

#### 2.4. Quy trình Huấn luyện An toàn

```
TRƯỚC KHI TRAIN:
  1. Sao lưu checkpoint gốc: model_latest.pt → model_original_backup.pt
  2. Không thay đổi file utils.py, tpsinbet.py gốc
  3. Tạo script train riêng: train_finetune.py

TRONG QUÁ TRÌNH TRAIN:
  - Lưu checkpoint mỗi 5 epoch: model_ep5.pt, model_ep10.pt, ...
  - Log val_PSNR, val_SSIM, val_L1 mỗi epoch → đồ thị hội tụ
  - Dừng sớm (early stopping) nếu val_PSNR không tăng sau 10 epoch

KẾT QUẢ MONG ĐỢI:
  - PSNR trên val set: từ ~11dB (hiện tại) → 15-20dB (sau fine-tune)
  - Ghosting pixels (180~250): từ >5000px → <100px (không cần hậu xử lý)
  - Wiggle/distortion: giảm >80% so với baseline
```

---

### GIAI ĐOẠN 3 — Đánh Giá & Triển Khai

#### 3.1. Tiêu chí Đánh giá

Kiểm thử trên **bộ test riêng biệt** (không dùng trong train/val):

| Tiêu chí | Trước fine-tune | Mục tiêu sau fine-tune |
|:---|:---:|:---:|
| **PSNR** (so với GT lineart) | ~11 dB | ≥ 16 dB |
| **SSIM** | ~0.67 | ≥ 0.85 |
| **Ghost pixels** (180~250) | >5,000 px | < 200 px |
| **Line Tortuosity** (độ méo nét) | ~1.47 | < 1.20 |
| **Chỉnh sửa thủ công cần thiết** | Nhiều | Tối thiểu |
| **Thời gian inference** | ~3-5s/frame | ≤ 5s/frame (giữ nguyên) |

#### 3.2. Kiểm tra Khả năng Tổng quát (Generalization)

Mô hình sau fine-tune phải được kiểm tra trên:
1. **Cùng nhân vật, cảnh mới** → Đảm bảo không overfit vào một clip
2. **Nhân vật khác, phong cách khác** → Đảm bảo tổng quát
3. **Giá trị $t$ đa dạng** → $t \in \{0.1, 0.25, 0.33, 0.5, 0.67, 0.75, 0.9\}$
4. **Đoạn chuyển động nhanh** → Xác nhận không vỡ nét với displacement cao hơn

#### 3.3. Triển khai trong Pipeline Hiện tại

Sau khi có `model_anime_finetuned.pt`:
```bash
# Chạy inference với model mới
python run_tps_fix.py \
  --checkpoint path/to/model_anime_finetuned.pt \
  --output_name tps-finetuned \
  --lambd 0.2 \
  --use_border_anchors \
  --use_interior_anchors

# So sánh với model gốc
python run_tps_fix.py \
  --checkpoint ckpt/model_latest.pt \
  --output_name tps-original
```

---

## III. Lộ Trình & Thứ Tự Ưu Tiên

```
TUẦN 1-2: Chuẩn bị Pipeline Tạo Dataset
  ✦ Thu thập 10 đoạn cắt anime (đa dạng hành động)
  ✦ Viết script tự động: video → frame → lineart → smooth → triplet
  ✦ Chạy thử trên ChayGo để xác nhận pipeline đúng
  ✦ Kiểm tra chất lượng bằng tay trên 50 triplet đầu tiên

TUẦN 3-4: Xây dựng Dataset Đủ Quy Mô
  ✦ Xử lý đủ 30+ đoạn cắt anime
  ✦ Đạt 5,000+ triplet hợp lệ (displacement < 20px, ≥ 20 inlier)
  ✦ Chia train/val (85%/15%)
  ✦ Pre-compute tất cả file .npy matches

TUẦN 5-6: Cài đặt & Chạy Fine-tune
  ✦ Cài đặt EDT Loss, DataLoader tương thích
  ✦ Chạy train 50 epoch trên GPU
  ✦ Monitor val_PSNR, lưu checkpoint tốt nhất
  ✦ Điều chỉnh hyperparameter nếu chưa hội tụ

TUẦN 7: Đánh giá & So sánh
  ✦ Chạy bộ test toàn diện (PSNR, SSIM, Ghost, Tortuosity)
  ✦ Đánh giá bằng mắt thường trên nhiều clip mới
  ✦ So sánh trực tiếp: model gốc vs model fine-tuned
  ✦ Viết báo cáo kết quả
```

---

## IV. Các Rủi Ro & Cách Phòng Tránh

| Rủi ro | Mức độ | Cách phòng tránh |
|:---|:---:|:---|
| Overfitting (học thuộc dataset, không tổng quát) | Cao | Đa dạng nguồn anime, data augmentation (flip, rotate nhẹ) |
| Catastrophic forgetting (mất khả năng gốc) | Trung bình | Learning rate nhỏ (1e-5), đóng băng GlueStick |
| Dataset chất lượng thấp | Cao | Kiểm tra thủ công 10% mẫu trước khi train |
| VRAM không đủ | Cao | Giảm batch_size, dùng gradient checkpointing |
| Thời gian train quá dài | Trung bình | Dùng GPU cloud (Colab Pro/Kaggle) nếu cần |

---

## V. Tóm Tắt Mục Tiêu Cuối Cùng

> **Model TPS-Inbetween sau fine-tune phải đạt được:**
>
> 1. **Giữ nguyên API**: Nhận đầu vào $(I_0, I_1, t)$ với $t \in (0, 1)$ bất kỳ, không thay đổi interface.
> 2. **Chuyên biệt cho anime 2D**: PSNR ≥ 16dB trên test set anime thực tế (so với ~11dB hiện tại).
> 3. **Nét vẽ sạch**: Không ghosting, không wiggle/distortion — có thể dùng trực tiếp mà không cần hậu xử lý thủ công.
> 4. **Tổng quát hóa**: Hoạt động tốt với nhân vật, cảnh, phong cách anime khác nhau.
> 5. **Bảo toàn model gốc**: `model_latest.pt` không bị ghi đè, luôn có backup để so sánh.
