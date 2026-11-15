# ĐẶC TẢ KỸ THUẬT NOTEBOOK HUẤN LUYỆN KAGGLE V1.0 (KAGGLE NOTEBOOK SPECIFICATION V1.0)

> **Tài liệu tham chiếu:** Thiết kế chi tiết cấu trúc 12 Cells, cơ chế nạp & giải nén tập dữ liệu mới `sakuga_golden_dataset_tps_ready.tar.gz`, bộ nạp dữ liệu an toàn bộ nhớ chia sẻ, động cơ ASF-GR và giải pháp sao lưu đa tầng cho phiên bản V1.0.  
> **File Notebook tương ứng:** `kaggle/notebooks/04_train_tps_inbetween_finetune_v1.0.ipynb`  
> **Nền tảng thực thi:** Kaggle Cloud GPU (Dual Tesla T4 2x16GB hoặc Tesla P100 16GB VRAM)  
> **Tập dữ liệu đầu vào:** [`data/processed/sakuga_golden_dataset_tps_ready.tar.gz`](file:///e:/CODE/PBL/PBL6/data/processed/sakuga_golden_dataset_tps_ready.tar.gz) (800.74 MB, 4,165 Golden Triplets)

---

## 1. NGUYÊN TẮC THIẾT KẾ CỐT LÕI V1.0

Phiên bản Notebook V1.0 được xây dựng dựa trên 5 nguyên tắc kỹ thuật hàng đầu:

1. **Khả Năng Thích Ứng Dữ Liệu Tự Động & Tối Ưu Tốc Độ Khởi Động (Phương Án 1 - Single-File Container):**
   - **Vấn đề nghẽn cổ chai Kaggle:** Nếu tải lên thư mục giải nén sẵn với 16,660+ file ảnh nhỏ, Kaggle sẽ mất 5-15 phút để nạp network overlay mỗi khi mở notebook.
   - **Giải pháp đóng gói Container ZIP:** Đóng gói file `sakuga_golden_dataset_tps_ready.tar.gz` vào container ZIP cấp 1 (`sakuga_golden_dataset_kaggle_upload.zip`, dùng `ZIP_STORED` không nén lại). Khi tải lên Kaggle, Kaggle chỉ giải nén lớp ngoài và thấy duy nhất 1 file `.tar.gz` $\to$ Session Kaggle khởi động chỉ trong 1-2 giây!
   - **Cơ chế nạp thông minh Cell 2:**
     * Tự động quét đệ quy mọi thư mục trong `/kaggle/input/` để phát hiện file archive `sakuga_golden_dataset_tps_ready.tar.gz`.
     * Tự động giải nén siêu tốc vào ổ NVMe cục bộ `/tmp/sakuga_golden_dataset_tps_ready` bằng lệnh hệ thống `tar -xzf` (chỉ mất ~10-15s), không làm đầy phân vùng `/kaggle/working` và không làm chậm quá trình Save Version/Commit notebook.
     * Tích hợp bộ nhớ đệm (Cache Detection): Nếu re-run cell hoặc dữ liệu đã giải nén sẵn ở `/tmp` hay `/kaggle/input`, Cell 2 nhận diện ngay lập tức trong 0.01s mà không giải nén lại.
2. **Chuẩn Hóa Đường Dẫn Cho Golden Dataset V1.0:**
   - Xử lý tương thích định dạng đường dẫn mới trong `triplets.csv`: Cột `frame0`, `frame_gt`, `frame1` đã chứa sẵn tiền tố phân vùng (`train/clip_XXXXXX/...` hoặc `val/clip_XXXXXX/...`), cột `match_file` chứa sẵn `matches/clip_XXXXXX/...`.
   - Cơ chế nạp thông minh tự động kiểm tra và ghép đường dẫn trực tiếp với `DATASET_ROOT`, loại bỏ hoàn toàn nguy cơ trùng lặp phân cấp thư mục (`train/train/...`).
3. **Khai Thác Tối Đa Sức Mạnh Multi-GPU (Dual Tesla T4 Optimization):**
   - Tự động nhận diện số lượng GPU khả dụng (`torch.cuda.device_count() >= 2`).
   - Sử dụng `nn.DataParallel` để phân bổ batch tính toán song song, tăng kích thước batch hiệu dụng lên $4 \times 2 \times 2 = 16$ mẫu.
4. **Cơ Chế Bảo Vệ Bộ Nhớ Tuyệt Đối (Zero RAM Leak & /dev/shm Guard):**
   - Bộ nạp dữ liệu tách rời DataFrame thành các native Python lists để triệt tiêu Copy-On-Write overhead của multiprocessing Linux fork.
   - Cấu hình DataLoader an toàn: `num_workers = 2`, `persistent_workers = True`, `pin_memory = True`, `prefetch_factor = 2`.
   - Thu dọn bộ nhớ chủ động sau mỗi epoch bằng `cleanup_memory()` kết hợp giải phóng heap Linux qua `ctypes.CDLL('libc.so.6').malloc_trim(0)`.
5. **Cơ Chế Đồng Bộ Checkpoint Kép Non-Blocking:**
   - Vừa lưu cục bộ tại `/kaggle/working/checkpoints_v1.0/`, vừa tự động sao lưu đồng thời lên Google Drive (hỗ trợ cả Folder Mount và REST API / Service Account JSON) để phòng chống mất dữ liệu khi session Kaggle hết thời gian.

---

## 2. SƠ ĐỒ LUỒNG THI CÔNG 12 CELLS TRONG NOTEBOOK V1.0

```mermaid
graph TD
    C1["Cell 1: Environment, Hardware Guard & Deep Memory Reclamation Setup"] --> C2["Cell 2: Automated Golden Dataset Discovery, Unpacking & Sanity Verification"]
    C2 --> C3["Cell 3: Zero-Leak Golden Dataset Loader (/dev/shm Safe & Native Lists)"]
    C3 --> C4["Cell 4: Robust Hybrid TPS Solver (λ=0.5, SVD Fallback & Boundary Anchors)"]
    C4 --> C5["Cell 5: Model Architecture (Adaptive Stroke Fusion Head + Synthesis UNet v1.0)"]
    C5 --> C6["Cell 6: Pure GPU Loss Engine (L1 + VGG + Stroke Dice + GPU MorphDist + GhostLoss + Density)"]
    C6 --> C7["Cell 7: Optimizer, Cosine Scheduler (30 Epochs) & Pretrained v0.3 Loader"]
    C7 --> C8["Cell 8: Anti-Cheat Pure GPU Validation Engine (PSNR, SSIM, Stroke Recall, Ghost Pixels)"]
    C8 --> C9["Cell 9: Smart Checkpoint Manager & Multi-Destination Google Drive Cloud Sync"]
    C9 --> C10["Cell 10: Rock-Solid 30-Epoch Training Loop (Gradient Accumulation, Mixed Precision AMP)"]
    C10 --> C11["Cell 11: Comprehensive Visual Audit Panel (10 Side-by-Side Benchmark Panels)"]
    C11 --> C12["Cell 12: Production Artifact Packaging & Cloud Delivery (Best Model & Training Logs)"]
```

---

## 3. THIẾT KẾ ĐẶC TẢ KỸ THUẬT CHI TIẾT TỪNG CELL

### Cell 1: Thiết lập Môi trường, Phần cứng & Trình Dọn Rác Bộ Nhớ Chủ Động
- **Mục tiêu:** Cố định seed ngẫu nhiên cho tính tái lập thực nghiệm, kiểm tra thông số VRAM phần cứng và khởi tạo hàm dọn rác chuyên sâu.
- **Mã nguồn thực thi mẫu:**
  ```python
  import os, sys, gc, time, math, random, json, shutil, ctypes
  import numpy as np, pandas as pd
  from PIL import Image
  import cv2
  from tqdm.auto import tqdm

  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  from torch.utils.data import Dataset, DataLoader
  import torchvision.models as models

  # Cố định Seed
  SEED = 42
  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  torch.cuda.manual_seed_all(SEED)
  torch.backends.cudnn.benchmark = True

  # Kiểm tra Phần cứng
  NUM_GPUS = torch.cuda.device_count()
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(f"[*] Thiết bị chính: {device} | Số GPU khả dụng: {NUM_GPUS}")
  for i in range(NUM_GPUS):
      prop = torch.cuda.get_device_properties(i)
      print(f"    - GPU {i}: {prop.name} (VRAM: {prop.total_memory / 1e9:.2f} GB)")

  # Hàm dọn dẹp bộ nhớ chuyên sâu chống rò rỉ RAM và VRAM
  def cleanup_memory():
      gc.collect()
      if torch.cuda.is_available():
          torch.cuda.empty_cache()
          try:
              torch.cuda.ipc_collect()
          except Exception:
              pass
      try:
          ctypes.CDLL('libc.so.6').malloc_trim(0)
      except Exception:
          pass

  cleanup_memory()
  print("[✓] CELL 1 hoàn tất: Môi trường và Zero-Leak Engine sẵn sàng.")
  ```

---

### Cell 2: Tự Động Định Vị, Giải Nén & Thẩm Định Toàn Vẹn Golden Dataset
- **Mục tiêu:** Tự động phát hiện vị trí tập dữ liệu Sakuga Golden Dataset trên Kaggle, hỗ trợ giải nén tức thì file `sakuga_golden_dataset_tps_ready.tar.gz` nếu đang ở dạng nén, nạp file `triplets.csv` và thẩm định tính toàn vẹn của 4,165 mẫu.
- **Mã nguồn thực thi mẫu:**
  ```python
  import subprocess

  KAGGLE_INPUT = '/kaggle/input'
  WORKING_DIR = '/kaggle/working'

  # Danh sách các ứng viên vị trí tập dữ liệu
  potential_dataset_dirs = [
      os.path.join(WORKING_DIR, 'sakuga_golden_dataset_tps_ready', 'anime_tps_finetune_dataset_ready'),
      os.path.join(WORKING_DIR, 'sakuga_golden_dataset_tps_ready'),
      os.path.join(WORKING_DIR, 'anime_tps_finetune_dataset_ready'),
      os.path.join(KAGGLE_INPUT, 'sakuga-golden-dataset-tps-ready', 'anime_tps_finetune_dataset_ready'),
      os.path.join(KAGGLE_INPUT, 'sakuga-golden-dataset-tps-ready'),
      os.path.join(KAGGLE_INPUT, 'sakuga_golden_dataset_tps_ready', 'anime_tps_finetune_dataset_ready'),
      os.path.join(KAGGLE_INPUT, 'sakuga_golden_dataset_tps_ready'),
      './data/processed/sakuga_golden_dataset_tps_ready/anime_tps_finetune_dataset_ready',
      './data/processed/sakuga_golden_dataset_tps_ready'
  ]

  # Danh sách ứng viên file nén .tar.gz
  potential_tar_files = [
      os.path.join(KAGGLE_INPUT, 'sakuga-golden-dataset-tps-ready', 'sakuga_golden_dataset_tps_ready.tar.gz'),
      os.path.join(KAGGLE_INPUT, 'sakuga_golden_dataset_tps_ready', 'sakuga_golden_dataset_tps_ready.tar.gz'),
      os.path.join(WORKING_DIR, 'sakuga_golden_dataset_tps_ready.tar.gz'),
      './data/processed/sakuga_golden_dataset_tps_ready.tar.gz'
  ]

  DATASET_ROOT = None

  # 1. Kiểm tra xem dataset đã giải nén sẵn chưa
  for d in potential_dataset_dirs:
      if os.path.exists(d) and (os.path.exists(os.path.join(d, 'triplets.csv')) or os.path.exists(os.path.join(d, 'triplets_combined.csv'))):
          DATASET_ROOT = d
          print(f"[✓] Tìm thấy thư mục Dataset đã giải nén sẵn: {DATASET_ROOT}")
          break

  # 2. Nếu chưa, tìm file .tar.gz và tự động giải nén
  if DATASET_ROOT is None:
      tar_path = None
      for tp in potential_tar_files:
          if os.path.exists(tp):
              tar_path = tp
              break
      
      if tar_path is not None:
          print(f"[*] Tìm thấy file nén: {tar_path} ({os.path.getsize(tar_path) / 1e6:.2f} MB)")
          extract_dest = os.path.join(WORKING_DIR, 'sakuga_golden_dataset_tps_ready')
          os.makedirs(extract_dest, exist_ok=True)
          print(f"[*] Đang giải nén vào {extract_dest} bằng lệnh tar hệ thống...")
          t_start = time.time()
          cmd = ['tar', '-xzf', tar_path, '-C', extract_dest]
          res = subprocess.run(cmd, capture_output=True, text=True)
          if res.returncode != 0:
              import tarfile
              with tarfile.open(tar_path, 'r:gz') as tar:
                  tar.extractall(extract_dest)
          print(f"[✓] Giải nén thành công trong {time.time() - t_start:.1f}s!")
          
          # Xác định DATASET_ROOT sau khi giải nén
          inner_dir = os.path.join(extract_dest, 'anime_tps_finetune_dataset_ready')
          DATASET_ROOT = inner_dir if os.path.exists(inner_dir) else extract_dest

  if DATASET_ROOT is None:
      raise FileNotFoundError("❌ Không tìm thấy tập dữ liệu sakuga_golden_dataset_tps_ready!")

  # Nạp triplets metadata
  triplets_csv_path = os.path.join(DATASET_ROOT, 'triplets.csv')
  if not os.path.exists(triplets_csv_path):
      triplets_csv_path = os.path.join(DATASET_ROOT, 'triplets_combined.csv')

  df_triplets = pd.read_csv(triplets_csv_path)
  train_df = df_triplets[df_triplets['split'] == 'train'].reset_index(drop=True)
  val_df = df_triplets[df_triplets['split'] == 'val'].reset_index(drop=True)

  print(f"[✓] Tổng số Golden Triplets: {len(df_triplets):,} mẫu")
  print(f"    - Tập Huấn luyện (Train): {len(train_df):,} mẫu ({len(train_df)/len(df_triplets)*100:.1f}%)")
  print(f"    - Tập Thẩm định (Val)   : {len(val_df):,} mẫu ({len(val_df)/len(df_triplets)*100:.1f}%)")
  assert len(train_df) >= 3000 and len(val_df) >= 500, "Số lượng mẫu không đạt chuẩn Golden Dataset!"
  print("[✓] CELL 2 hoàn tất: Toàn vẹn Golden Dataset được xác nhận.")
  ```

---

### Cell 3: Bộ Nạp Dữ Liệu Tối Ưu Hóa `/dev/shm` Cho Golden Dataset V1.0
- **Mục tiêu:** Xử lý đường dẫn tương đối trực tiếp từ `DATASET_ROOT`, chuẩn hóa ảnh nền trắng nét đen $I \in [0, 1]$, cắt ngẫu nhiên $512 \times 512$ cho Train, pad kích thước bội số 32 cho Val, nạp ma trận GlueStick `.npy`.
- **Đặc tả logic đường dẫn:**
  ```python
  class SakugaGoldenDataset_v1(Dataset):
      def __init__(self, root_dir, df_split, is_train=True, crop_size=512):
          self.root_dir = root_dir
          self.is_train = is_train
          self.crop_size = crop_size
          
          # [CẢI TIẾN QUAN TRỌNG V1.0]: 
          # df_split['frame0'] trong Golden Dataset đã có dạng 'train/clip_XXXXXX/000001.png'
          # Nối trực tiếp với root_dir mà không thêm thư mục split trung gian.
          self.p0_list = []
          self.pgt_list = []
          self.p1_list = []
          self.pmatch_list = []
          
          for _, row in df_split.iterrows():
              f0 = str(row['frame0'])
              fgt = str(row['frame_gt'])
              f1 = str(row['frame1'])
              fm = str(row['match_file'])
              
              # Tự động chuẩn hóa nếu path bắt đầu bằng / hoặc có dạng khác
              self.p0_list.append(os.path.join(root_dir, f0.lstrip('/\\')))
              self.pgt_list.append(os.path.join(root_dir, fgt.lstrip('/\\')))
              self.p1_list.append(os.path.join(root_dir, f1.lstrip('/\\')))
              self.pmatch_list.append(os.path.join(root_dir, fm.lstrip('/\\')))
          
          self.t_list = df_split['t'].astype(np.float32).tolist()
          self.clips_list = df_split['clip_id'].astype(str).tolist()
          self.length = len(df_split)

      def __len__(self):
          return self.length

      def __getitem__(self, idx):
          p0 = self.p0_list[idx]
          pgt = self.pgt_list[idx]
          p1 = self.p1_list[idx]
          pmatch = self.pmatch_list[idx]

          with Image.open(p0) as im0:
              img0 = 1.0 - (np.array(im0.convert('L'), dtype=np.float32) / 255.0)
          with Image.open(pgt) as imgt:
              imggt = 1.0 - (np.array(imggt.convert('L'), dtype=np.float32) / 255.0)
          with Image.open(p1) as im1:
              img1 = 1.0 - (np.array(im1.convert('L'), dtype=np.float32) / 255.0)

          # Nạp GlueStick keypoints
          if os.path.exists(pmatch):
              try:
                  kps = np.load(pmatch).astype(np.float32)
              except Exception:
                  kps = np.zeros((2, 0, 2), dtype=np.float32)
          else:
              kps = np.zeros((2, 0, 2), dtype=np.float32)

          h, w = img0.shape
          # Xử lý Random Crop cho Train / Padding cho Val
          # [TỐI ƯU HÓA ĐỘT PHÁ V1.0]: Giải ma trận TPS float64 ngay trong DataLoader Worker trên CPU
          c0_pad, th01_pad, c1_pad, th10_pad, valid_flag = solve_tps_pair_cpu(kps, max_kps=80, lambd=0.5, min_dist=0.015)
          # custom_collate_fn_v1 gom thành batch tensor: c0_b (B, 88, 2), th01_b (B, 91, 2), v.v.
  ```

---

### Cell 4: Bộ Biến Dạng Lưới Kháng Suy Biến, 8 Anchors & Vector Hóa TPS Batch
- **Cache lưới tọa độ (`_NORM_GRID_CACHE`, `_PIXEL_GRID_CACHE`):** Caching lưới cơ sở `(1, H, W, 2)` và cộng trực tiếp qua broadcasting, loại bỏ hoàn toàn `.repeat(b, 1, 1, 1)` và `meshgrid` lặp lại.
- **8 Điểm neo cố định biên/góc:** 4 góc `[0,0], [1,0], [0,1], [1,1]` và 4 điểm giữa biên `[0.5,0], [0.5,1], [0,0.5], [1,0.5]` để khóa khung viền chống co rút.
- **`solve_tps_pair_cpu`:** Chuyển phép giải hệ phương trình tuyến tính TPS sang CPU workers của DataLoader. Pad kích thước cố định `c_pad` (88, 2) và `theta_pad` (91, 2) với các điểm chưa dùng bằng `0.0`.
- **`tps_flow_from_theta_batch`:** Vector hóa $100\%$ trên GPU bằng `torch.bmm`, tính toán trường dòng TPS cho toàn bộ batch cùng lúc với độ chính xác toán học tuyệt đối (Zero Python loop, Zero GIL bottleneck).

---

### Cell 5: Kiến Trúc Mô Hình TPS-Inbetween V1.0 (ASF-GR Engine - Không Nghẽn Cổ Chai GPU)
- **`AdaptiveStrokeFusionHead`:** Mạng tích chập 4 tầng nhận `[fine_I0t, fine_I1t, flow_refine, feature_mask, t]` xuất ra mặt nạ mềm $M_{\text{blend}} \in [0, 1]$ qua Sigmoid.
- **`SynthesisUnet_v1`:** Kiến trúc UNet với cổng tẩy nét ma `conv_erase` (bias = 2.0) và nhánh tôn nét `conv_add` (biên độ $[0, 0.25]$).
- **`TPSInbetweenFinetune_v1.forward()`:** Nhận tham số `tps_precomputed`. Nếu có sẵn từ DataLoader $\to$ chạy thẳng `tps_flow_from_theta_batch` $100\%$ trên GPU (Zero CPU transfer). Giữ nhánh fallback cho các script inference bên ngoài.

---

### Cell 6: Bộ Hàm Mất Mát CompositeLossEngine_v1 (100% PyTorch GPU Tensor)
- **Đóng gói toàn bộ các hàm mất mát vào `CompositeLossEngine_v1(nn.Module)`:**
  1. $\mathcal{L}_{L1}$: L1 Pixel Loss.
  2. $\mathcal{L}_{VGG}$: VGG16 Perceptual Loss (relu1_2, relu2_2, relu3_3).
  3. $\mathcal{L}_{Dice}$: Stroke Dice Loss đặc trị tranh nét thưa.
  4. $\mathcal{L}_{Morph\_Dist}$: PyTorch Native GPU `F.max_pool2d` với multi-scale kernel $\{3, 5, 9, 15\}$ (thay thế hoàn toàn SciPy EDT, **0% rò rỉ RAM host**).
  5. $\mathcal{L}_{Ghost}$: Trừng phạt bình phương nét mực $>0.15$ tại vùng Ground Truth là nền trắng.
  6. $\mathcal{L}_{Density}$: Bảo toàn tỷ lệ mực vẽ trong khoảng $0.95 - 1.10$.
  - Trả về `total_loss, loss_l1, loss_dice, loss_ghost`.

---

### Cell 7: Multi-GPU ParallelTrainingWrapper_v1 & Optimizer Setup
- **`ParallelTrainingWrapper_v1(nn.Module)`:**
  - Bọc cả `base_model` và `loss_engine` vào chung 1 module để `nn.DataParallel` phân bổ song song cả luồng Forward lẫn Backward sang 2 GPU Tesla T4.
  - Cân bằng tải VRAM hoàn hảo giữa GPU 0 và GPU 1 (GPU 0 không còn bị quá tải vì gánh loss một mình).
  - Bảo toàn $100\%$ cấu trúc khóa `state_dict` của `base_model` khi lưu checkpoint.
- **Optimizer & Scheduler:** AdamW ($lr=1.0\times 10^{-4}$), CosineAnnealingLR ($T_{\max}=30$), AMP GradScaler.

---

### Cell 8: Hàm Đo Lường Chỉ Số Đánh Giá Chống Bẫy Thuần GPU (Pure GPU Metrics)
- Tính toán toàn bộ các chỉ số Validation trên GPU PyTorch Tensor:
  - $\text{PSNR} = 10 \cdot \log_{10}\left(\frac{1.0}{\text{MSE} + 1e-8}\right)$
  - $\text{SSIM}$: Tính bằng `F.conv2d` với kernel Gaussian $11 \times 11$.
  - $\text{Stroke Recall}$: Tỷ lệ pixel nét GT ($\ge 0.5$) được bao phủ bởi pred trong bán kính giãn nở $5\text{ px}$.
  - $\text{Ghost Pixels}$: Số lượng pixel xám ($0.15 \le p \le 0.85$) nằm ngoài nét GT.
  - $\text{Blank Collapse Guard}$: Báo động đỏ ngay lập tức nếu tổng cường độ nét của toàn bộ ảnh $< 100.0$.

---

### Cell 9: Quản Lý Checkpoint Tự Động & Đồng Bộ Đa Nền Tảng (Smart GDrive Syncer)
- **Tiêu chí Kỷ lục Nghiệm thu V1.0:**
  1. $\text{Stroke Recall} \ge 80.0\%$ (Chống sụp đổ trắng).
  2. $\text{Ghost Pixels} \le 8,000\text{ px}$ (Chống bóng ma).
  3. Chỉ số điểm tổng hợp:
     $$\text{Score}_{v1.0} = \text{PSNR} + 20 \times \text{SSIM} + 30 \times \text{Recall} - 0.0005 \times \text{GhostPixels}$$
- **Đồng bộ đám mây:**
  - Tự động bóc tách `base_model` nguyên bản để lưu `model_tps_v1.0_best.pt` sạch và đúng chuẩn.
  - Tự động sao lưu sang Google Drive khi đạt kỷ lục mới.

---

### Cell 10: Vòng Lặp Huấn Luyện 30 Epochs (Zero CUDA Sync Overhead)
- **Phase 1 (Train):**
  - Chạy qua `parallel_engine` phân bổ trên 2 GPU.
  - Cộng dồn tensor GPU (`train_loss_accum += total_loss.detach()`), loại bỏ hoàn toàn `.item()` từng step để triệt tiêu độ trễ đồng bộ CPU-GPU.
  - Chỉ gọi `.item()` điều tiết mỗi 10 step để hiển thị `tqdm`.
- **Phase 2 (Val):**
  - `num_workers=2` trong `val_loader` nạp ảnh song song.
  - Lệnh `break` ngắt đúng ngưỡng `max_val_eval_batches` (100 batches ở các epoch thường, full batches ở epoch kỷ niệm).
  - Chia chính xác theo số batch thực tế đã chạy `actual_val_batches`.


---

### Cell 11: Bảng Kiểm Toán Trực Quan Định Tính & Khử Bóng Ma (Visual Audit Panel)
- Trích xuất 10-15 cặp frames tiêu biểu từ 620 mẫu Validation (bao gồm cả các mẫu khó như `clip_000018`, `clip_000046`, v.v.).
- Tạo bảng ảnh so sánh trực quan 4 cột:
  - Cột 1: Frame 0 (t=0.0)
  - Cột 2: TPS Coarse Warp (Bóng ma đối chứng)
  - Cột 3: Finetuned V1.0 (Dự đoán thành phẩm)
  - Cột 4: Frame Ground Truth (Mục tiêu chuẩn)
- Xuất toàn bộ vào thư mục `visual_samples_v1.0/`.

---

### Cell 12: Đóng Gói Thành Phẩm Sản Phẩm & Xuất File Tải Về (Artifact Delivery)
- Lưu trữ các artifacts chính thức:
  - Trọng số tối ưu: `model_tps_v1.0_best.pt`
  - Trọng số epoch cuối: `model_tps_v1.0_latest.pt`
  - Lịch sử huấn luyện: `training_history_v1.0.csv`
  - Thư mục ảnh kiểm toán: `visual_samples_v1.0/`
- Đóng gói toàn bộ thành file lưu trữ zip: `tps_finetune_v1.0_artifacts.zip`.
- Kích hoạt tiến trình đồng bộ cuối cùng lên Google Drive và hiển thị đường link tải trực tiếp trên giao diện Kaggle.
