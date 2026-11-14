# KIẾN TRÚC & QUY CHUẨN DỮ LIỆU DỰ ÁN PBL6 (DATA ARCHITECTURE REFERENCE)

> **Dành cho Lập trình viên và AI Agent:**  
> Vì toàn bộ thư mục dữ liệu gốc `data/` (~22.2 GB) và `Dataset/` (~4.2 GB) chứa dung lượng rất lớn và đã được đưa vào `.gitignore`, thư mục `data_samples/` này cung cấp **mẫu dữ liệu tối giản (minimal representative sample)** phản ánh chính xác 100% cấu trúc, định dạng file, metadata và schema của toàn bộ pipeline dữ liệu trong dự án.
>
> Khi clone repo về, bất kỳ AI Agent hay developer nào cũng có thể đọc tài liệu này và kiểm tra các file trong `data_samples/` để hiểu kiến trúc dữ liệu mà không cần tải hàng chục GB dataset gốc.

---

## 1. TỔNG QUAN HỆ THỐNG DỮ LIỆU

Dự án nghiên cứu **TPS-Inbetween cho hoạt hình Anime 2D** (sinh khung hình trung gian inbetween giữa 2 keyframe nét vẽ). Pipeline dữ liệu gồm 2 thư mục chính:

1. **`data/` (Data Store):** Nơi lưu trữ video thô, dữ liệu sau xử lý (lineart, keypoints, triplets) và các tập test benchmark.
2. **`Dataset/` (Data Engineering Hub):** Chứa toàn bộ crawler, notebook Kaggle, pipeline xử lý tự động và script ghép nối dataset (V1, Patch 2, Patch 3, Sequence V2).

```plaintext
E:\CODE\PBL\PBL6\
├── data/                                 # [GITIGNORED] Dữ liệu thực tế (~22.2 GB)
│   ├── crawled/                          # Video MP4 anime tải từ Sakugabooru
│   ├── lineart_experiments/              # Thử nghiệm bóc tách nét (AniLines, Anime2Sketch)
│   ├── processed/                        # Dữ liệu chuẩn hóa đưa vào train/val
│   │   ├── sakuga_golden_dataset_tps_ready_v1_patch3/    # Golden Dataset Triplet V3.0
│   │   └── sakuga_golden_dataset_sequences_tps_v2/       # Sequence Dataset V2
│   └── test_sets/                        # Các bộ test chuẩn benchmark
│       └── sakuga_continuous_triplets/  # 6 clip benchmark đo mượt chuyển động
│
├── Dataset/                              # [GITIGNORED phần cache nặng] Scripts & Pipelines
│   ├── crawlers/                         # Script cào Sakugabooru theo batch
│   ├── sakuga_pipeline/                  # Pipeline phiên bản v1
│   ├── sakuga_pipeline v2/               # Pipeline phiên bản v2 (Kaggle Cloud)
│   └── sakuga_pipeline_sequence/         # Pipeline chuỗi khung hình V2 (Sequence)
│
└── data_samples/                         # [LƯU TRÊN GIT] Bản sao mẫu kiến trúc (< 1 MB)
    ├── DATA_ARCHITECTURE.md              # File tài liệu này
    ├── triplet_sample/                   # Mẫu kiến trúc bộ ba khung hình (Triplet)
    ├── sequence_sample/                  # Mẫu kiến trúc chuỗi đa khung hình (Sequence)
    └── test_set_sample/                  # Mẫu kiến trúc tập test benchmark
```

---

## 2. CHI TIẾT MẪU KIẾN TRÚC 1: TRIPLET DATASET (`data_samples/triplet_sample/`)

Đây là định dạng cốt lõi phục vụ huấn luyện mô hình TPS-Inbetween.

### 2.1. Cấu trúc thư mục
```plaintext
triplet_sample/
├── DATASET_INFO.json             # Metadata cấu hình & thống kê chuyển động
├── triplets.csv                  # Danh sách chỉ mục và độ đo chất lượng từng triplet
├── quality_audit_report.csv      # Báo cáo kiểm định phễu lọc chất lượng (Audit Report)
├── train/
│   └── clip_000007/              # ID của từng triplet
│       ├── 000001.png            # Keyframe bắt đầu (I_0) - thời điểm t=0
│       ├── 000002.png            # Ground-truth inbetween (I_gt) - thời điểm t
│       └── 000003.png            # Keyframe kết thúc (I_1) - thời điểm t=1
└── matches/
    └── clip_000007/
        └── match_000001_000003.npy  # Tọa độ điểm tương ứng GlueStick giữa I_0 và I_1
```

### 2.2. Đặc tả file hình ảnh (`000001.png`, `000002.png`, `000003.png`)
* **Độ phân giải:** 540p ($960 \times 540$) hoặc 270p ($480 \times 270$).
* **Không gian màu:** Grayscale hoặc RGB (nét vẽ đen trên nền trắng hoặc RGBA bóc nền).
* **Định dạng:** PNG lossless.

### 2.3. Đặc tả file tương ứng điểm (`match_000001_000003.npy`)
* **Mục đích:** Cung cấp các cặp điểm tương ứng (keypoint correspondences) phục vụ mạng Thin-Plate Spline (TPS) biến dạng hình học.
* **Shape:** `(2, N, 2)` với `dtype: float32`.
  * `data[0]`: Mảng kích thước $(N, 2)$ chứa tọa độ $(x, y)$ chuẩn hóa trong khoảng $[0.0, 1.0]$ trên Keyframe $I_0$.
  * `data[1]`: Mảng kích thước $(N, 2)$ chứa tọa độ $(x, y)$ chuẩn hóa trong khoảng $[0.0, 1.0]$ tương ứng trên Keyframe $I_1$.
  * $N$: Số lượng điểm match hợp lệ (inliers), thường từ $30$ đến $300$ điểm.

### 2.4. Bảng Schema `triplets.csv`
| Cột | Kiểu | Ý nghĩa |
| :--- | :--- | :--- |
| `split` | string | Phân tập: `train` hoặc `val`. |
| `clip_id` | string | Mã định danh thư mục clip (ví dụ `clip_000007`). |
| `t` | float | Thời gian chuẩn hóa của frame trung gian $t \in (0, 1)$ (thường là $0.5$ hoặc $0.6667$). |
| `delta_f` | int | Khoảng cách số frame gốc trong video anime giữa $I_0$ và $I_1$ (ví dụ: bước nhảy 2, 4, 6 frame). |
| `video_name` | string | Tên file video nguồn tải từ Sakugabooru. |
| `frame0` | string | Đường dẫn tương đối đến Keyframe 0. |
| `frame_gt` | string | Đường dẫn tương đối đến Ground-truth inbetween frame. |
| `frame1` | string | Đường dẫn tương đối đến Keyframe 1. |
| `match_file` | string | Đường dẫn tương đối đến file `.npy` tương ứng điểm GlueStick. |
| `mean_disp` | float | Độ dịch chuyển trung bình (mean displacement) của các điểm match (đo chuyển động). |
| `inliers` | int | Số lượng điểm match hợp lệ qua kiểm định hình học RANSAC. |
| `wsds` | float | Điểm nhất quán hướng chuyển động (Weighted Spatial Directional Smoothness). |
| `pwcr` | float | Tỷ lệ bảo toàn chiều rộng nét vẽ (Pixel Width Consistency Ratio). |
| `dccr` | float | Tỷ lệ nhất quán mật độ nét vẽ (Density Center Consistency Ratio). |
| `source_origin` | string | Phân đợt dữ liệu (`V1_Patch2 Legacy`, `Patch3 Occlusion`,...). |
| `patch_occ_score`| float | Điểm phát hiện che khuất cục bộ (Patch Occlusion Score). |

---

## 3. CHI TIẾT MẪU KIẾN TRÚC 2: SEQUENCE DATASET (`data_samples/sequence_sample/`)

Được dùng cho các thí nghiệm mở rộng chuỗi thời gian liên tục $N$ frame (Multi-frame Sequence Interpolation).

### 3.1. Cấu trúc thư mục
* `train/clip_XXXXXX/`: Chứa các frame tuần tự `000001.png`, `000002.png`, `000003.png`,...
* `matches/clip_XXXXXX/`: Chứa file `match_000001_000003.npy` giữa hai đầu mút của chuỗi.
* `sequences.csv`: Chứa thông tin về độ dài chuỗi (`seq_len`), scene ID và chỉ số quang sai mượt thời gian.

---

## 4. CHI TIẾT MẪU KIẾN TRÚC 3: TEST BENCHMARK (`data_samples/test_set_sample/`)

Tập test tiêu chuẩn được dùng để đánh giá mô hình sau mỗi lần huấn luyện (tính PSNR, SSIM, Chamfer Distance, Ghosting Ratio):
* `clip_000008/`: Các file `frame_00.png`, `frame_01.png`, `frame_02.png`.
* `extracted_triplets_manifest.csv`: Manifest ánh xạ clip test, bước nhảy `delta_f` và video nguồn.

---

## 5. CODE MẪU CHO AGENT: CÁCH ĐỌC VÀ LOAD DỮ LIỆU BẰNG PYTHON

Đoạn code sau mô phỏng chính xác cách một `Dataset` PyTorch trong dự án đọc các mẫu dữ liệu này:

```python
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class SakugaTripletDatasetSample(Dataset):
    def __init__(self, sample_root="data_samples/triplet_sample"):
        self.sample_root = sample_root
        self.df = pd.read_csv(os.path.join(sample_root, "triplets.csv"))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Đường dẫn file
        f0_path = os.path.join(self.sample_root, row['frame0'])
        fgt_path = os.path.join(self.sample_root, row['frame_gt'])
        f1_path = os.path.join(self.sample_root, row['frame1'])
        match_path = os.path.join(self.sample_root, row['match_file'])
        
        # Load ảnh
        img0 = Image.open(f0_path).convert('RGB')
        img_gt = Image.open(fgt_path).convert('RGB')
        img1 = Image.open(f1_path).convert('RGB')
        
        # Load GlueStick keypoint correspondences (Shape: 2, N, 2)
        matches = np.load(match_path)  # matches[0]: pts in frame0, matches[1]: pts in frame1
        
        return {
            'clip_id': row['clip_id'],
            't': torch.tensor(row['t'], dtype=torch.float32),
            'img0': img0,
            'img_gt': img_gt,
            'img1': img1,
            'matches': torch.from_numpy(matches)
        }

if __name__ == "__main__":
    ds = SakugaTripletDatasetSample()
    item = ds[0]
    print(f"Loaded clip: {item['clip_id']}, t={item['t'].item()}")
    print(f"Keypoints shape: {item['matches'].shape} (2, N, 2 normalized)")
```

---

## 6. HƯỚNG DẪN TÁI TẠO HOẶC TẢI TOÀN BỘ DATASET GỐC

Nếu bạn cần chạy huấn luyện thực tế với toàn bộ 22GB+ dataset:
1. **Dùng Kaggle Cloud:** Xem các notebook tự động hóa trong `Dataset/sakuga_pipeline_sequence/` (ví dụ `00_crawl_sequences_kaggle.ipynb`, `01_extract_sequences_candidates_kaggle.ipynb`, `02_process_sequences_tps_kaggle.ipynb`).
2. **Chạy Crawler nội bộ:** Sử dụng các file `.bat` trong `Dataset/crawlers/` (ví dụ `run_crawl_sakugabooru_batch3.bat`).
3. **Liên hệ tác giả:** Để nhận link tải trực tiếp các archive `.tar.gz` (V1 Patch 3 Golden Dataset) từ Google Drive hoặc OneDrive của nhóm nghiên cứu.
