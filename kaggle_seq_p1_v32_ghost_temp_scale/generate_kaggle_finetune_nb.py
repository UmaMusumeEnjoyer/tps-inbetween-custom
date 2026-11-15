#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_kaggle_finetune_nb.py
--------------------------------------------------------------------------------
Trình tạo Kaggle Notebook Tự Thân (11 Cells: Cell 0 -> Cell 10):
    train_and_benchmark_tps_sequence_final_kaggle.ipynb

Thực thi theo Option 3 (LONG — Huấn luyện Chuỗi Toàn Quỹ Đạo 540p):
  - Khai thác 100% 2x GPU Tesla T4 (DDP torchrun + Length-Width Bucket Sampler).
  - 100% Zero Architectural Modification giữa M1 (model_latest.pt) và M3 (Finetuned).
  - Cell 6 đã comment `os._exit(0)` để Reset Kernel giải phóng RAM vẫn giữ Output.
  - Chính sách Checkpoint tinh gọn: Chỉ duy trì 2 file (`best.pt` và `last.pt`).
  - Benchmark kép không thiên vị (Dual-Benchmark):
      1) Tập cách ly `sakuga_golden_benchmark_sequences_v2` (198 clips / 412 frames)
      2) Tập thực tế liên tục `data/lineart_experiments/test` (24 frames 540p)
--------------------------------------------------------------------------------
"""

import os
import sys
import json
import base64

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
WORKER_PATH = os.path.join(THIS_DIR, "ddp_worker_tps_sequence_final.py")
BENCH_SUITE_PATH = os.path.join(
    REPO_ROOT, "Dataset", "sakuga_pipeline_sequence", "final_dataset",
    "sakuga_golden_benchmark_sequences_v2", "run_benchmark_suite.py"
)
GLUESTICK_NPZ_PATH = os.path.join(THIS_DIR, "lineart_test_gluestick_matches.npz")


def _read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _make_md(source_str):
    lines = source_str.strip("\n").splitlines(keepends=True)
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def _make_code(source_str):
    lines = source_str.strip("\n").splitlines(keepends=True)
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


def build_notebook():
    worker_code = _read_text(WORKER_PATH)
    bench_code = _read_text(BENCH_SUITE_PATH)
    gluestick_b64 = _read_b64(GLUESTICK_NPZ_PATH)

    cells = []

    # ==========================================================================
    # CELL 0: MARKDOWN - MỤC TIÊU KHOA HỌC, KIẾN TRÚC & OPTION 3 (LONG MODE)
    # ==========================================================================
    c0_md = r"""# 🔬 PHA 1 (V3.2): DIỆT BÓNG MỜ (`2A`), ỔN ĐỊNH THỜI GIAN (`2C`) & CO GIÃN HÌNH HỌC
## Option 3 (LONG — Huấn Luyện Chuỗi Đầy Đủ 540p) | `10,068,668` Active Params

---

### 🎯 1. Mục Tiêu Thực Nghiệm Pha 1 (Chưa Sửa Kiến Trúc Để Đo Độc Lập Tác Động Dữ Liệu + Loss)
* **Nền tảng khởi tạo:** Tự động warm-start từ `Sequence 2 Ep15` (`model_tps_finetuned_last.pt` EMA nếu gắn vào Input) hoặc từ `model_latest.pt`, đối chứng công bằng với **`M1_Original`** (`model_latest.pt`).
* **Các nâng cấp cốt lõi ở Pha 1:**
  1. **Vá lỗi pha loãng gradient `GhostAndBackgroundCleanlinessLoss`:** Chuẩn hóa theo diện tích nét GT thay vì `far_count` (~480k px) và nâng `w_ghost = 0.45`.
  2. **Thêm Temporal Consistency Loss (`L_temp`, `w_temp = 0.50`):** Phạt sai phân chuyển động và nhấp nháy nền giữa các frame kế tiếp trên chuỗi $N \in \{5, 7\}$ (`2C`).
  3. **Paired Spatial Augmentation + 120 Chuỗi Hình Học Tổng Hợp/Epoch:** Thêm H-Flip/V-Flip đồng bộ trên ảnh + TPS flow, trộn chuỗi hình tròn/elip dịch xa (`60–420 px`) & thu/phóng (`0.42×–2.20×`) với **Lagrangian Direct TPS**.
  4. **Triple-Benchmark:** Đo thêm `temp_diff_mae` / `temp_bg_flicker` trên Bench #2 và bổ sung **Benchmark #3 (Defense Geometry Suite)** kiểm tra trực tiếp 2 hình tròn co giãn & dịch xa!

---

### ⚙️ 2. So Sánh 3 Option Tốc Độ — Phiên Bản Này Kích Hoạt Sẵn **Option 3 (`LONG`)**

| Thông Số | ⚡ Option 1 (`SHORT`) | ⚖️ Option 2 (`MEDIUM`) | 🔬 **Option 3 (`LONG` — ĐANG BẬT)** |
| :--- | :---: | :---: | :---: |
| **Thời gian / Epoch (2x T4)** | ~0.6 – 0.9 phút | ~1.5 – 2.2 phút | **~4.5 – 5.8 phút** (20 Ep $\approx$ 95–115 phút) |
| **Độ phân giải Train** | Ink Crop `384x384` | Motion Crop `512x512` | **Full-Frame 540p (`544 x W_pad`)** |
| **Chiến lược lấy mẫu** | 1 frame ngẫu nhiên/seq | 3,169 mẫu đơn/epoch | **Chuỗi đa khung hình ($N \in \{3,5,7\}$) + 120 Synthetic Geometry seqs/epoch** |
| **Bộ Loss** | Charb + DT(8) + Sobel + Density + Width + Clean | + clDice + L_temp | **9-Term: Charb + Sym-DT(8) + Sobel + clDice + VGG16 + STE-Density + BiStrokeWidth + GhostClean(Fixed) + TemporalConsistency** |
| **Độ mịn lưới TPS** | `downscale = 4` | `downscale = 4` | **`downscale = 2` ($272 \times 480$)** |

---

### 📋 3. Chuẩn Bị 3 Dataset Trên Kaggle (Tab **+ Add Input**)
Bật **Accelerator: GPU T4 x2** và **Internet: ON**. Thêm 3 dataset sau:
1. **`dungbaoviec123/tps-baseline-model`** (Checkpoint gốc `model_latest.pt` đã upload tại `/kaggle/input/datasets/dungbaoviec123/tps-baseline-model`).
2. **`sakuga-golden-dataset-v2`** (Dataset ZIP gộp chung 2 thư mục `sakuga_golden_dataset_sequences_tps_v2` và `sakuga_golden_benchmark_sequences_v2`).
3. **`anime-lineart-continuous-test`** (Dataset upload riêng chứa 24 frames 540p `0.0s.png..2.8s.png` và `continuous_keyframes_spec.json`).

> 💡 **Lưu ý Vận Hành Trên Kaggle:**
> - Toàn bộ dữ liệu được tự động định vị và mount/symlink sang NVMe SSD `/tmp/tps_sequence_ssd/` (không tốn dung lượng ổ đĩa 20GB `/kaggle/working`).
> - **Chỉ lưu đúng 2 checkpoint** (`model_tps_finetuned_best.pt` và `model_tps_finetuned_last.pt`).
> - Nếu RAM gần ngưỡng `23.5 GB`, dùng **Cell 6** (bỏ `#`, Run Cell 6 $\to$ Comment lại `#` $\to$ **Run All**) để reset Kernel sạch sẽ mà không mất dữ liệu & tự động Auto-Resume từ epoch gần nhất!
"""
    cells.append(_make_md(c0_md))

    # ==========================================================================
    # CELL 1: CODE - MÔI TRƯỜNG, KIỂM TRA 2x GPU & NHÚNG WORKER TỰ THÂN
    # ==========================================================================
    c1_code = f'''# ==============================================================================
# 🛠️ [CELL 1]: KHỞI TẠO MÔI TRƯỜNG, KHÓA RAM C & NHÚNG MÃ NGUỒN ĐỘC LẬP
# ==============================================================================
import os
import sys
import gc
import base64
import torch
import psutil

# Khóa phân mảnh glibc & tắt đa luồng OpenCV gây ngốn RAM phụ trong DDP
os.environ["MALLOC_TRIM_THRESHOLD_"] = "65536"
os.environ["MALLOC_ARENA_MAX"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

try:
    import cv2
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
except ImportError:
    pass

torch.backends.cudnn.benchmark = True

IS_KAGGLE = os.path.exists("/kaggle")
WORKING_DIR = "/kaggle/working" if IS_KAGGLE else os.path.abspath("./kaggle_local_test_ws")
OUTPUT_CKPT_DIR = os.path.join(WORKING_DIR, "checkpoints_tps_finetune")
BENCH_OUT_DIR = os.path.join(WORKING_DIR, "benchmark_outputs")
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(OUTPUT_CKPT_DIR, exist_ok=True)
os.makedirs(BENCH_OUT_DIR, exist_ok=True)

if WORKING_DIR not in sys.path:
    sys.path.insert(0, WORKING_DIR)

gpu_count = torch.cuda.device_count()
print("=" * 84)
print(f"🖥️ KIỂM TRA PHẦN CỨNG: {{gpu_count}} GPU khả dụng (Kaggle={{IS_KAGGLE}})")
for i in range(gpu_count):
    p = torch.cuda.get_device_properties(i)
    print(f"   👉 GPU {{i}}: {{p.name}} | VRAM = {{p.total_memory / (1024**3):.2f}} GB")
mem = psutil.virtual_memory()
print(f"🧠 Host RAM: Tổng = {{mem.total / (1024**3):.1f}} GB | Đang dùng = {{mem.used / (1024**3):.1f}} GB | Trống = {{mem.available / (1024**3):.1f}} GB")
print(f"📁 Working Dir: {{WORKING_DIR}}")
print("=" * 84)

# Ghi các file thực thi độc lập vào WORKING_DIR
worker_script_path = os.path.join(WORKING_DIR, "ddp_worker_tps_sequence_final.py")
with open(worker_script_path, "w", encoding="utf-8") as f:
    f.write({repr(worker_code)})

bench_script_path = os.path.join(WORKING_DIR, "run_benchmark_suite.py")
with open(bench_script_path, "w", encoding="utf-8") as f:
    f.write({repr(bench_code)})

GLUESTICK_TEST_NPZ = os.path.join(WORKING_DIR, "lineart_test_gluestick_matches.npz")
with open(GLUESTICK_TEST_NPZ, "wb") as f:
    f.write(base64.b64decode({repr(gluestick_b64)}))

print(f"[✓] Đã ghi ddp_worker_tps_sequence_final.py ({{os.path.getsize(worker_script_path):,}} B)")
print(f"[✓] Đã ghi run_benchmark_suite.py ({{os.path.getsize(bench_script_path):,}} B)")
print(f"[✓] Đã giải mã GlueStick Cache cho lineart_experiments/test ({{os.path.getsize(GLUESTICK_TEST_NPZ):,}} B)")
sys.stdout.flush()
'''
    cells.append(_make_code(c1_code))

    # ==========================================================================
    # CELL 2: CODE - ĐỊNH VỊ INPUTS & STAGING SYMLINK/SSD SIÊU TỐC SANG /tmp
    # ==========================================================================
    c2_code = r'''# ==============================================================================
# ⚡ [CELL 2]: ĐỊNH VỊ TOÀN BỘ INPUTS & STAGING SANG NVMe SSD (/tmp)
# ==============================================================================
import os
import sys
import glob
import time
import shutil
import zipfile
import pandas as pd

SSD_ROOT = "/tmp/tps_sequence_ssd" if os.path.exists("/tmp") else os.path.join(WORKING_DIR, "_ssd_cache")
TRAIN_STAGE_DIR = os.path.join(SSD_ROOT, "train_set")
BENCH_STAGE_DIR = os.path.join(SSD_ROOT, "bench_set")
LINEART_TEST_STAGE = os.path.join(SSD_ROOT, "lineart_test")
EXTRACT_CACHE_DIR = os.path.join(SSD_ROOT, "extracted_archives")
for d in [SSD_ROOT, TRAIN_STAGE_DIR, BENCH_STAGE_DIR, LINEART_TEST_STAGE, EXTRACT_CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

SEARCH_ROOTS = [
    "/kaggle/input",
    "/kaggle/input/datasets",
    "/kaggle/input/tps-baseline-model",
    "/kaggle/input/datasets/dungbaoviec123/tps-baseline-model",
    SSD_ROOT,
] if os.path.exists("/kaggle/input") else [
    SSD_ROOT,
    "E:/CODE/PBL/PBL6/Dataset/sakuga_pipeline_sequence/final_dataset",
    "E:/CODE/PBL/PBL6/data/lineart_experiments/test",
    "E:/CODE/PBL/PBL6/checkpoints/pretrained_baselines/tps-original",
    os.path.abspath("."),
]

def find_file_in_roots(filename, roots=None):
    if roots is None:
        roots = SEARCH_ROOTS
    for root_base in roots:
        if not os.path.exists(root_base):
            continue
        hits = glob.glob(os.path.join(root_base, "**", filename), recursive=True)
        if hits:
            # Ưu tiên path ngắn nhất / trực diện nhất
            hits.sort(key=lambda p: len(p))
            return hits[0]
    return None

def safe_stage_dir_links(src_root, dst_root, subdirs):
    for sub in subdirs:
        src_p = os.path.join(src_root, sub)
        dst_p = os.path.join(dst_root, sub)
        if not os.path.exists(src_p):
            continue
        if os.path.abspath(src_p) == os.path.abspath(dst_p):
            continue
        if os.path.islink(dst_p) or (os.path.isdir(dst_p) and len(os.listdir(dst_p)) > 0):
            continue
        if os.path.isdir(dst_p):
            try:
                os.rmdir(dst_p)
            except Exception:
                pass
        try:
            os.symlink(src_p, dst_p, target_is_directory=True)
        except Exception:
            shutil.copytree(src_p, dst_p, dirs_exist_ok=True)

def safe_copy_file(src_p, dst_p):
    if os.path.abspath(src_p) != os.path.abspath(dst_p):
        shutil.copy2(src_p, dst_p)

print("=" * 84)
print("🔍 [CELL 2]: ĐANG TỰ ĐỘNG ĐỊNH VỊ CHECKPOINT GỐC & TẬP DỮ LIỆU INPUT...")
print("=" * 84)

# 0. QUÉT VÀ TỰ ĐỘNG GIẢI NÉN FILE ZIP (NẾU NGƯỜI DÙNG UPLOAD FILE ZIP THÔ)
# Hỗ trợ cả 2 kịch bản:
# - ZIP gộp cả 2 tập (sakuga_golden_dataset_sequences_tps_v2 + sakuga_golden_benchmark_sequences_v2)
# - Các ZIP lẻ riêng lẻ từng dataset
already_extracted_zips = set()
for rb in [r for r in SEARCH_ROOTS if r != SSD_ROOT and os.path.exists(r)]:
    # Tìm kiếm ở mức nông (độ sâu tối đa 2 cấp) để phản hồi tức thì
    candidate_zips = glob.glob(os.path.join(rb, "*.zip")) + glob.glob(os.path.join(rb, "*", "*.zip"))
    for zpath in candidate_zips:
        if zpath in already_extracted_zips:
            continue
        try:
            with zipfile.ZipFile(zpath, 'r') as zf:
                names = zf.namelist()
                has_train = any(n.endswith("train_sequences.csv") for n in names)
                has_bench = any(n.endswith("benchmark_sequences.csv") for n in names)
                has_spec = any(n.endswith("continuous_keyframes_spec.json") for n in names)

                if has_train or has_bench or has_spec:
                    z_stem = os.path.splitext(os.path.basename(zpath))[0]
                    dest_extract = os.path.join(EXTRACT_CACHE_DIR, z_stem)
                    if not os.path.exists(dest_extract) or len(os.listdir(dest_extract)) == 0:
                        os.makedirs(dest_extract, exist_ok=True)
                        print(f"📦 Phát hiện Archive ZIP: {os.path.basename(zpath)} (has_train={has_train}, has_bench={has_bench}, has_spec={has_spec})")
                        print(f"   -> Đang giải nén nhanh sang SSD {dest_extract}...", flush=True)
                        zf.extractall(dest_extract)
                        print(f"   [✓] Giải nén thành công {os.path.basename(zpath)}.")
                    already_extracted_zips.add(zpath)
        except Exception as e:
            print(f"   [!] Bỏ qua file zip lỗi {zpath}: {e}")

# 1. Định vị Checkpoint M1 gốc (model_latest.pt)
# Ưu tiên các đường dẫn xác định từ Kaggle Dataset dungbaoviec123/tps-baseline-model
direct_model_candidates = [
    "/kaggle/input/datasets/dungbaoviec123/tps-baseline-model/model_latest.pt",
    "/kaggle/input/datasets/dungbaoviec123/tps-baseline-model/tps-original/model_latest.pt",
    "/kaggle/input/tps-baseline-model/model_latest.pt",
    "/kaggle/input/tps-baseline-model/tps-original/model_latest.pt",
    os.path.abspath("checkpoints/pretrained_baselines/tps-original/model_latest.pt"),
    "E:/CODE/PBL/PBL6/checkpoints/pretrained_baselines/tps-original/model_latest.pt",
]
M1_CHECKPOINT_PATH = None
for cand in direct_model_candidates:
    if os.path.isfile(cand):
        M1_CHECKPOINT_PATH = cand
        break

if not M1_CHECKPOINT_PATH:
    M1_CHECKPOINT_PATH = find_file_in_roots("model_latest.pt")

if not M1_CHECKPOINT_PATH or not os.path.exists(M1_CHECKPOINT_PATH):
    avail_in = glob.glob("/kaggle/input/*") if os.path.exists("/kaggle/input") else []
    raise FileNotFoundError(
        "❌ Không tìm thấy `model_latest.pt`! Vui lòng thêm dataset `dungbaoviec123/tps-baseline-model` vào Notebook Input.\n"
        f"   Các thư mục Input hiện có: {avail_in}"
    )
print(f"[✓] 1. Baseline Original Checkpoint : {M1_CHECKPOINT_PATH} ({os.path.getsize(M1_CHECKPOINT_PATH)/(1024**2):.1f} MB)")

# 1b. Tự động tìm Warm-Start Checkpoint Pha 1 (tùy chọn: Sequence 2 Ep15 `model_tps_finetuned_last.pt`)
INIT_CHECKPOINT_PATH = find_file_in_roots("model_tps_finetuned_last.pt")
if not INIT_CHECKPOINT_PATH and os.path.isfile("E:/CODE/PBL/PBL6/checkpoints/Re-train/Sequence 2/checkpoints/model_tps_finetuned_last.pt"):
    INIT_CHECKPOINT_PATH = "E:/CODE/PBL/PBL6/checkpoints/Re-train/Sequence 2/checkpoints/model_tps_finetuned_last.pt"
if INIT_CHECKPOINT_PATH and os.path.isfile(INIT_CHECKPOINT_PATH):
    print(f"[✓] 1b. Warm-Start Checkpoint (EMA) : {INIT_CHECKPOINT_PATH} ({os.path.getsize(INIT_CHECKPOINT_PATH)/(1024**2):.1f} MB)")
else:
    INIT_CHECKPOINT_PATH = M1_CHECKPOINT_PATH
    print("[*] 1b. Không gắn checkpoint Seq2 Ep15 -> Khởi tạo trực tiếp từ model_latest.pt.")

# 2. Định vị Tập Train/Val Sequence (sakuga_golden_dataset_sequences_tps_v2)
train_csv_found = find_file_in_roots("train_sequences.csv")
if not train_csv_found:
    avail_in = glob.glob("/kaggle/input/*") if os.path.exists("/kaggle/input") else []
    raise FileNotFoundError(
        "❌ Không tìm thấy `train_sequences.csv`! Vui lòng gắn Dataset chứa `sakuga_golden_dataset_sequences_tps_v2`.\n"
        f"   Các thư mục Input hiện có: {avail_in}"
    )

TRAIN_SRC_ROOT = os.path.dirname(train_csv_found)
VAL_CSV_SRC = os.path.join(TRAIN_SRC_ROOT, "val_sequences.csv")
assert os.path.isfile(VAL_CSV_SRC), f"❌ Thiếu val_sequences.csv tại {TRAIN_SRC_ROOT}"

safe_stage_dir_links(TRAIN_SRC_ROOT, TRAIN_STAGE_DIR, ["train", "val", "matches"])
safe_copy_file(train_csv_found, os.path.join(TRAIN_STAGE_DIR, "train_sequences.csv"))
safe_copy_file(VAL_CSV_SRC, os.path.join(TRAIN_STAGE_DIR, "val_sequences.csv"))

TRAIN_CLIPS_DIR = os.path.join(TRAIN_STAGE_DIR, "train")
VAL_CLIPS_DIR = os.path.join(TRAIN_STAGE_DIR, "val")
TRAIN_MATCHES_DIR = os.path.join(TRAIN_STAGE_DIR, "matches")
df_tr_check = pd.read_csv(os.path.join(TRAIN_STAGE_DIR, "train_sequences.csv"))
df_va_check = pd.read_csv(os.path.join(TRAIN_STAGE_DIR, "val_sequences.csv"))
print(f"[✓] 2. Dataset Train/Val Sequence   : {TRAIN_SRC_ROOT}")
print(f"       -> Train: {len(df_tr_check):,} seqs ({df_tr_check['scene_id'].nunique():,} scenes) | Val: {len(df_va_check):,} seqs ({df_va_check['scene_id'].nunique():,} scenes)")

# 3. Định vị Tập Benchmark Cách Ly (sakuga_golden_benchmark_sequences_v2)
bench_csv_found = find_file_in_roots("benchmark_sequences.csv")
BENCH_READY = False
BENCH_ROOT_PATH = None
if bench_csv_found and os.path.exists(bench_csv_found):
    b_src = os.path.dirname(bench_csv_found)
    safe_stage_dir_links(b_src, BENCH_STAGE_DIR, ["clips", "matches"])
    safe_copy_file(bench_csv_found, os.path.join(BENCH_STAGE_DIR, "benchmark_sequences.csv"))
    BENCH_ROOT_PATH = BENCH_STAGE_DIR
    BENCH_READY = True
    df_bn_check = pd.read_csv(os.path.join(BENCH_STAGE_DIR, "benchmark_sequences.csv"))
    print(f"[✓] 3. Benchmark #1 (Isolated Set)  : {b_src} -> {len(df_bn_check):,} seqs / {df_bn_check['scene_id'].nunique():,} scenes")
else:
    print("[⚠️] 3. Benchmark #1 (sakuga_golden_benchmark_sequences_v2): CHƯA ĐƯỢC GẮN VÀO INPUT (Cell 7 sẽ bỏ qua nếu chưa gắn).")

# 4. Định vị Tập Thực Tế Liên Tục (Benchmark #2: anime-lineart-continuous-test)
direct_lineart_dirs = [
    "/kaggle/input/anime-lineart-continuous-test",
    "/kaggle/input/datasets/dungbaoviec123/anime-lineart-continuous-test",
    "/kaggle/input/lineart-experiments-test",
    "/kaggle/input/test",
]
spec_found = None
for ldir in direct_lineart_dirs:
    sp = os.path.join(ldir, "continuous_keyframes_spec.json")
    if os.path.isfile(sp):
        spec_found = sp
        break

if not spec_found:
    spec_found = find_file_in_roots("continuous_keyframes_spec.json")

LINEART_TEST_READY = False
LINEART_TEST_PATH = None
if spec_found and os.path.exists(spec_found):
    l_src = os.path.dirname(spec_found)
    for fn in os.listdir(l_src):
        src_f = os.path.join(l_src, fn)
        dst_f = os.path.join(LINEART_TEST_STAGE, fn)
        if os.path.isfile(src_f):
            safe_copy_file(src_f, dst_f)
    LINEART_TEST_PATH = LINEART_TEST_STAGE
    LINEART_TEST_READY = True
    png_count = len(glob.glob(os.path.join(LINEART_TEST_PATH, "*.png")))
    print(f"[✓] 4. Benchmark #2 (Lineart Test)  : {l_src} -> staged {png_count} png frames vào {LINEART_TEST_PATH}")
else:
    print("[⚠️] 4. Benchmark #2 (Lineart Test): CHƯA ĐƯỢC GẮN VÀO INPUT (Cell 8 sẽ bỏ qua nếu chưa gắn).")

print("=" * 84)
sys.stdout.flush()
'''
    cells.append(_make_code(c2_code))

    # ==========================================================================
    # CELL 3: CODE - TIỀN TÍNH TOÁN TPS CACHE CHUẨN BACKWARD WARP (LAMBD=0.01)
    # ==========================================================================
    c3_code = r'''# ==============================================================================
# 🚀 [CELL 3]: TIỀN TÍNH TOÁN TPS CACHE CHUẨN BACKWARD WARP (16 LUỒNG CPU + MMAP)
# ==============================================================================
import os
import sys
import glob
import time
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from ddp_worker_tps_sequence_final import solve_tps_backward_pair

TPS_CACHE_TRAINVAL = os.path.join(SSD_ROOT, "tps_cache_trainval_bw.npz")
TRAIN_CSV_IDX_PATH = os.path.join(SSD_ROOT, "train_sequences_indexed.csv")
VAL_CSV_IDX_PATH = os.path.join(SSD_ROOT, "val_sequences_indexed.csv")
MAX_PTS = 128
TPS_LAMBDA = 0.01

if os.path.exists(TPS_CACHE_TRAINVAL) and os.path.exists(TRAIN_CSV_IDX_PATH) and os.path.exists(VAL_CSV_IDX_PATH):
    print(f"[♻️] Phát hiện TPS Cache Backward-Warp đã tạo sẵn tại {TPS_CACHE_TRAINVAL} — Bỏ qua bước tính lại!")
else:
    t0 = time.time()
    df_tr = pd.read_csv(os.path.join(TRAIN_STAGE_DIR, "train_sequences.csv"))
    df_va = pd.read_csv(os.path.join(TRAIN_STAGE_DIR, "val_sequences.csv"))
    df_tr['_clip_root'] = TRAIN_CLIPS_DIR
    df_va['_clip_root'] = VAL_CLIPS_DIR
    df_all = pd.concat([df_tr, df_va], ignore_index=True)
    n_total = len(df_all)
    df_all['cache_idx'] = np.arange(n_total, dtype=np.int32)

    df_tr_out = df_all.iloc[:len(df_tr)].drop(columns=['_clip_root']).copy()
    df_va_out = df_all.iloc[len(df_tr):].drop(columns=['_clip_root']).copy()
    df_tr_out.to_csv(TRAIN_CSV_IDX_PATH, index=False)
    df_va_out.to_csv(VAL_CSV_IDX_PATH, index=False)

    c0_all = np.zeros((n_total, MAX_PTS, 2), dtype=np.float32)
    th01_all = np.zeros((n_total, MAX_PTS + 3, 2), dtype=np.float32)
    c1_all = np.zeros((n_total, MAX_PTS, 2), dtype=np.float32)
    th10_all = np.zeros((n_total, MAX_PTS + 3, 2), dtype=np.float32)
    valid_all = np.zeros((n_total,), dtype=np.float32)
    hw_all = np.zeros((n_total, 2), dtype=np.int32)

    def _task(item):
        idx, row = item
        cid = str(row['clip_id'])
        sl = int(row['seq_len'])
        c_root = row['_clip_root']
        img0_p = os.path.join(c_root, cid, "000001.png")
        with Image.open(img0_p) as im:
            w, h = im.size

        m_p = os.path.join(TRAIN_MATCHES_DIR, cid, f"match_000001_{sl:06d}.npy")
        if not os.path.exists(m_p):
            cands = sorted(glob.glob(os.path.join(TRAIN_MATCHES_DIR, cid, "*.npy")))
            m_p = cands[0] if cands else ""

        if m_p and os.path.exists(m_p):
            kps = np.load(m_p).astype(np.float32)
            c0, th01, c1, th10, vf = solve_tps_backward_pair(kps[0], kps[1], max_pts=MAX_PTS, lambd=TPS_LAMBDA)
        else:
            c0 = c1 = np.zeros((MAX_PTS, 2), dtype=np.float32)
            th01 = th10 = np.zeros((MAX_PTS + 3, 2), dtype=np.float32)
            vf = 0.0
        return idx, c0, th01, c1, th10, vf, h, w

    print(f"⚙️ Đang giải ma trận Backward-Warp TPS (MAX_PTS={MAX_PTS}, lambd={TPS_LAMBDA}) cho {n_total:,} clips bằng 16 luồng...", flush=True)
    with ThreadPoolExecutor(max_workers=min(16, (os.cpu_count() or 4) * 2)) as pool:
        for idx, c0, th01, c1, th10, vf, h, w in pool.map(_task, list(df_all.iterrows())):
            c0_all[idx], th01_all[idx] = c0, th01
            c1_all[idx], th10_all[idx] = c1, th10
            valid_all[idx] = vf
            hw_all[idx] = [h, w]

    np.savez(
        TPS_CACHE_TRAINVAL,
        c0=c0_all, th01=th01_all, c1=c1_all, th10=th10_all,
        valid=valid_all, orig_hw=hw_all
    )
    elapsed = time.time() - t0
    print(f"[✓] HOÀN TẤT TPS CACHE TRONG {elapsed:.1f}s -> {TPS_CACHE_TRAINVAL} ({os.path.getsize(TPS_CACHE_TRAINVAL)/(1024**2):.1f} MB)")
    print(f"    + Tỷ lệ nghiệm TPS hợp lệ: {int(valid_all.sum()):,}/{n_total:,} ({valid_all.mean()*100:.2f}%)")
    print(f"    + Giải kích thước khung hình W: [{hw_all[:,1].min()}, {hw_all[:,1].max()}] x H={hw_all[:,0].min()}")
sys.stdout.flush()
'''
    cells.append(_make_code(c3_code))

    # ==========================================================================
    # CELL 4: CODE - KIỂM TOÁN 100% ĐỒNG NHẤT KIẾN TRÚC M1 == M3 (0% BIAS)
    # ==========================================================================
    c4_code = r'''# ==============================================================================
# 🔬 [CELL 4]: KIỂM TOÁN TOÁN HỌC KHẲNG ĐỊNH 100% ĐỒNG NHẤT KIẾN TRÚC (M1 == M3)
# ==============================================================================
import gc
import torch
from ddp_worker_tps_sequence_final import TPSInbetweenOriginal, load_tps_weights

print("=" * 84)
print("🔍 KIỂM TOÁN TOÁN HỌC: CAM KẾT 0% SAI KHÁC KIẾN TRÚC GIỮA ORIGINAL (M1) VÀ FINETUNED (M3)")
print("=" * 84)

model_m1 = TPSInbetweenOriginal()
load_tps_weights(model_m1, M1_CHECKPOINT_PATH, device='cpu', prefer_ema=False, is_rank0=True)

model_m3 = TPSInbetweenOriginal()
model_m3.load_state_dict(model_m1.state_dict(), strict=True)

keys_m1 = set(model_m1.state_dict().keys())
keys_m3 = set(model_m3.state_dict().keys())
assert keys_m1 == keys_m3, "❌ LỖI: Phát hiện sai khác danh sách tensor!"

for k in keys_m1:
    assert model_m1.state_dict()[k].shape == model_m3.state_dict()[k].shape, f"❌ Lệch shape tại {k}"
    assert torch.equal(model_m1.state_dict()[k], model_m3.state_dict()[k]), f"❌ Lệch giá trị khởi tạo tại {k}"

# Kiểm định sai số đầu ra = 0.0 trên tensor giả lập
model_m1.eval(); model_m3.eval()
with torch.no_grad():
    dx0 = torch.rand(1, 1, 128, 128)
    dx1 = torch.rand(1, 1, 128, 128)
    dt = torch.tensor([[[[0.5]]]])
    dcf = torch.zeros(1, 2, 128, 128)
    out1 = model_m1(dx0, dx1, dt, dcf, dcf)[0]
    out3 = model_m3(dx0, dx1, dt, dcf, dcf)[0]
    max_diff = float((out1 - out3).abs().max().item())
assert max_diff == 0.0, f"❌ Sai khác đầu ra ban đầu != 0: {max_diff}"

p_enc = sum(p.numel() for p in model_m1.encoder.parameters())
p_ifn = sum(p.numel() for p in model_m1.IFNet.parameters())
p_une = sum(p.numel() for p in model_m1.unet.parameters())
p_tot = sum(p.numel() for p in model_m1.parameters())

print(f"[✓] Contextnet params   : {p_enc:,}")
print(f"[✓] IFNet (b0..b2)      : {p_ifn:,}")
print(f"[✓] OriginalUnet params : {p_une:,}")
print(f"[✓] TỔNG ACTIVE PARAMS  : {p_tot:,} (Khớp 100% trước khi huấn luyện)")
print(f"[✓] Sai số đầu ra ban đầu |M1(x) - M3_init(x)|_max = {max_diff:.1f}")
print("=" * 84)

del model_m1, model_m3, dx0, dx1, dcf, out1, out3
gc.collect()
sys.stdout.flush()
'''
    cells.append(_make_code(c4_code))

    # ==========================================================================
    # CELL 5: CODE - THỰC THI HUẤN LUYỆN DDP 2x GPU THEO OPTION 3 (LONG)
    # ==========================================================================
    c5_code = r'''# ==============================================================================
# 🚀 [CELL 5]: THỰC THI HUẤN LUYỆN DDP 2x GPU — OPTION 3 (LONG MODE ĐƯỢC DUYỆT)
# ==============================================================================
import os
import sys
import subprocess
import torch

# Có thể chuyển đổi giữa: "SHORT" | "MEDIUM" | "LONG"
TRAIN_MODE = "LONG"

NUM_GPUS = max(1, torch.cuda.device_count())
TOTAL_EPOCHS = 20
GRAD_ACCUM = 2          # Kết hợp với DDPLengthWidthBatchSampler (~16-20 frames / opt step trên 2 GPU)
MAX_CHUNK_FRAMES = 3    # Tối ưu cho 15GB VRAM của Tesla T4 (~4.2 GB đỉnh, không OOM cả khi N=7 & W=1480)
LR_BACKBONE = 1.0e-5
LR_UNET = 1.8e-5
EMA_DECAY = 0.995
SYNTH_COUNT = 120
W_GHOST = 0.45
W_TEMP = 0.50
RAM_LIMIT_GB = 23.5

cmd = [
    "torchrun",
    f"--nproc_per_node={NUM_GPUS}",
    "--master_port=29505",
    worker_script_path,
    "--train_csv", TRAIN_CSV_IDX_PATH,
    "--val_csv", VAL_CSV_IDX_PATH,
    "--train_clips_dir", TRAIN_CLIPS_DIR,
    "--val_clips_dir", VAL_CLIPS_DIR,
    "--tps_cache", TPS_CACHE_TRAINVAL,
    "--m1_checkpoint", M1_CHECKPOINT_PATH,
    "--init_checkpoint", INIT_CHECKPOINT_PATH,
    "--output_dir", OUTPUT_CKPT_DIR,
    "--train_mode", TRAIN_MODE,
    "--epochs", str(TOTAL_EPOCHS),
    "--grad_accum", str(GRAD_ACCUM),
    "--max_chunk_frames", str(MAX_CHUNK_FRAMES),
    "--lr_backbone", str(LR_BACKBONE),
    "--lr_unet", str(LR_UNET),
    "--ema_decay", str(EMA_DECAY),
    "--synth_count", str(SYNTH_COUNT),
    "--w_ghost", str(W_GHOST),
    "--w_temp", str(W_TEMP),
    "--num_workers", "2",
    "--ram_limit_gb", str(RAM_LIMIT_GB),
]

print("=" * 84)
print(f"🚀 CHẠY HUẤN LUYỆN DDP ({NUM_GPUS}x GPU) | OPTION 3 MODE={TRAIN_MODE} | {TOTAL_EPOCHS} EPOCHS")
print("📋 Chính sách lưu Checkpoint: CHỈ LƯU ĐÚNG 2 FILE (`best.pt` và `last.pt`)")
print("Lệnh:", " ".join(cmd))
print("=" * 84, flush=True)

sub_env = os.environ.copy()
sub_env["PYTHONUNBUFFERED"] = "1"
sub_env["OMP_NUM_THREADS"] = "1"
sub_env["MKL_NUM_THREADS"] = "1"

proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=sub_env)
for line in iter(proc.stdout.readline, ''):
    sys.stdout.write(line)
    sys.stdout.flush()
proc.stdout.close()
ret = proc.wait()

print("\n" + "=" * 84)
if ret == 0:
    print(f"🎉 HUẤN LUYỆN HOÀN TẤT {TOTAL_EPOCHS}/{TOTAL_EPOCHS} EPOCHS!")
elif ret == 42:
    print("⏸️ [WATCHDOG RAM TẠM DỪNG AN TOÀN - MÃ 42]: Đã lưu state vào `model_tps_finetuned_last.pt`.")
    print("👉 HƯỚNG DẪN: Bỏ dấu '#' ở 2 dòng trong Cell 6 -> Bấm Run riêng Cell 6 để giải phóng RAM,")
    print("   rồi gắn lại '#' và bấm Run All -> Cell 5 sẽ tự động Auto-Resume từ epoch kế tiếp!")
else:
    raise RuntimeError(f"❌ Tiến trình huấn luyện DDP kết thúc bất thường với exit code {ret}!")
print("=" * 84)
sys.stdout.flush()
'''
    cells.append(_make_code(c5_code))

    # ==========================================================================
    # CELL 6: CODE (ĐÃ COMMENT) - RESET KERNEL TRÁNH TRÀN RAM GIỮ OUTPUT
    # ==========================================================================
    c6_code = r'''# ==============================================================================
# 🔄 [CELL 6 - KHẨN CẤP]: RESET KERNEL TRÁNH TRÀN RAM NHƯNG VẪN GIỮ OUTPUT
# ==============================================================================
# Hướng dẫn sử dụng khi Host RAM tăng cao (hoặc khi Cell 5 báo mã dừng 42):
#   1. Bỏ dấu '#' ở 2 dòng code dưới đây (`import os` và `os._exit(0)`),
#      rồi chỉ bấm Run (chạy) RIÊNG Cell 6 này.
#   2. Đợi 3-5 giây cho Kernel Kaggle tự khởi động lại sạch RAM về ~1 GB
#      (Mọi file đã tạo trong /kaggle/working và /tmp được GIỮ NGUYÊN 100%).
#   3. Đặt lại dấu '#' vào 2 dòng dưới để vô hiệu hóa Cell 6,
#      sau đó bấm "Run All":
#      -> Cell 3 tự dùng lại TPS Cache, Cell 5 tự Auto-Resume từ `model_tps_finetuned_last.pt`!

# import os
# os._exit(0)
'''
    cells.append(_make_code(c6_code))

    # ==========================================================================
    # CELL 7: CODE - BENCHMARK #1: TẬP CHUỖI ĐỘC LẬP (198 CLIPS / 412 FRAMES)
    # ==========================================================================
    c7_code = r'''# ==============================================================================
# 🔬 [CELL 7]: BENCHMARK #1 — ĐỐI CHUẨN TẬP CÁCH LY (`sakuga_golden_benchmark_sequences_v2`)
# ==============================================================================
import os
import gc
import json
import torch
import pandas as pd
from IPython.display import display, Image as IPyImage
from ddp_worker_tps_sequence_final import TPSInbetweenOriginal, load_tps_weights
from run_benchmark_suite import run_benchmark_isolated_sequences

BENCH1_OUT = os.path.join(BENCH_OUT_DIR, "bench1_isolated_sequences")
best_ckpt_path = os.path.join(OUTPUT_CKPT_DIR, "model_tps_finetuned_best.pt")
last_ckpt_path = os.path.join(OUTPUT_CKPT_DIR, "model_tps_finetuned_last.pt")

if not BENCH_READY:
    print("[⚠️ BỎ QUA CELL 7]: Tập `sakuga_golden_benchmark_sequences_v2` chưa có trong Input.")
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    models_dict = {}

    m1 = TPSInbetweenOriginal().to(device)
    load_tps_weights(m1, M1_CHECKPOINT_PATH, device=device, prefer_ema=False)
    models_dict['M1_Original'] = m1.eval()

    if os.path.exists(best_ckpt_path):
        m3_b = TPSInbetweenOriginal().to(device)
        load_tps_weights(m3_b, best_ckpt_path, device=device, prefer_ema=True)
        models_dict['M3_Best'] = m3_b.eval()

    if os.path.exists(last_ckpt_path):
        m3_l = TPSInbetweenOriginal().to(device)
        load_tps_weights(m3_l, last_ckpt_path, device=device, prefer_ema=True)
        models_dict['M3_Last'] = m3_l.eval()

    df_sum1, stat_audit1 = run_benchmark_isolated_sequences(
        BENCH_ROOT_PATH, models_dict, device, BENCH1_OUT, downscale_factor=2, save_visuals=True
    )

    print("\n📋 BẢNG PHÂN TẦNG THEO ĐỘ DÀI CHUỖI N={3,5,7} & BIÊN ĐỘ CHUYỂN ĐỘNG:")
    df_strat = pd.read_csv(os.path.join(BENCH1_OUT, "benchmark_stratified_breakdown.csv"))
    display(df_strat.round(4))

    for img_name in ["TOP3_IMPROVED.png", "TOP3_FAILURE_OR_HARD.png"]:
        p = os.path.join(BENCH1_OUT, "audit_cases_top3", img_name)
        if os.path.exists(p):
            print(f"\n🖼️ Kiểm toán trực quan: {img_name}")
            display(IPyImage(filename=p))

    del models_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
'''
    cells.append(_make_code(c7_code))

    # ==========================================================================
    # CELL 8: CODE - BENCHMARK #2: TẬP THỰC TẾ LIÊN TỤC (`lineart_experiments/test`)
    # ==========================================================================
    c8_code = r'''# ==============================================================================
# 🎬 [CELL 8]: BENCHMARK #2 — TẠO FRAME THỰC TẾ LIÊN TỤC (`data/lineart_experiments/test`)
# ==============================================================================
import os
import gc
import torch
import pandas as pd
from IPython.display import display, Image as IPyImage
from ddp_worker_tps_sequence_final import TPSInbetweenOriginal, load_tps_weights
from run_benchmark_suite import run_benchmark_continuous_lineart_test, run_benchmark_defense_geometry_suite

BENCH2_OUT = os.path.join(BENCH_OUT_DIR, "bench2_continuous_lineart_test")
BENCH3_OUT = os.path.join(BENCH_OUT_DIR, "bench3_defense_geometry")
best_ckpt_path = os.path.join(OUTPUT_CKPT_DIR, "model_tps_finetuned_best.pt")
last_ckpt_path = os.path.join(OUTPUT_CKPT_DIR, "model_tps_finetuned_last.pt")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
models_dict = {}

m1 = TPSInbetweenOriginal().to(device)
load_tps_weights(m1, M1_CHECKPOINT_PATH, device=device, prefer_ema=False)
models_dict['M1_Original'] = m1.eval()

if os.path.exists(best_ckpt_path):
    m3_b = TPSInbetweenOriginal().to(device)
    load_tps_weights(m3_b, best_ckpt_path, device=device, prefer_ema=True)
    models_dict['M3_Best'] = m3_b.eval()

if os.path.exists(last_ckpt_path):
    m3_l = TPSInbetweenOriginal().to(device)
    load_tps_weights(m3_l, last_ckpt_path, device=device, prefer_ema=True)
    models_dict['M3_Last'] = m3_l.eval()

if not LINEART_TEST_READY:
    print("[⚠️ BỎ QUA BENCHMARK #2]: Tập `data/lineart_experiments/test` chưa có trong Input.")
else:
    df_sum2 = run_benchmark_continuous_lineart_test(
        LINEART_TEST_PATH, models_dict, device, BENCH2_OUT,
        gluestick_npz=GLUESTICK_TEST_NPZ, downscale_factor=2
    )
    if df_sum2 is not None:
        print("\n📋 TỔNG HỢP THEO ĐỘ DÀI CHẶNG (delta_f = 3, 4, 5):")
        display(df_sum2.round(4))

    montage_p = os.path.join(BENCH2_OUT, "montage_6segments_M1_vs_M3.png")
    if os.path.exists(montage_p):
        print("\n🖼️ Đối đầu trực quan 6 phân đoạn liên tiếp [Overlay | M1 Orig | M3 Fine | GT]:")
        display(IPyImage(filename=montage_p))

df_sum3 = run_benchmark_defense_geometry_suite(models_dict, device, BENCH3_OUT, downscale_factor=2)
if df_sum3 is not None:
    display(df_sum3.round(4))
    for gname in ["ShrinkingCircle_75to32_grid.png", "ShrinkingCircle_StaticGround_grid.png"]:
        gp = os.path.join(BENCH3_OUT, gname)
        if os.path.exists(gp):
            display(IPyImage(filename=gp))

del models_dict
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
'''
    cells.append(_make_code(c8_code))

    # ==========================================================================
    # CELL 9: CODE - DASHBOARD HỘI TỤ & BÁO CÁO ĐÁNH GIÁ KHÁCH QUAN TỔNG HỢP
    # ==========================================================================
    c9_code = r'''# ==============================================================================
# 📈 [CELL 9]: DASHBOARD HỘI TỤ TRAINING & NHẬN ĐỊNH KHÁCH QUAN KHOA HỌC
# ==============================================================================
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

hist_csv = os.path.join(WORKING_DIR, "training_history_finetune.csv")
val_base_json = os.path.join(WORKING_DIR, "val_baseline_m0_m1.json")

if os.path.exists(hist_csv):
    df_h = pd.read_csv(hist_csv)
    base_data = json.load(open(val_base_json, 'r', encoding='utf-8')) if os.path.exists(val_base_json) else {}
    m0_b = base_data.get('M0_Coarse', {})
    m1_b = base_data.get('M1_Original', {})

    ep = df_h['epoch'].values
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("TPS-Inbetween Option 3 (LONG): Fine-tuning Convergence vs Original Baseline (M1)", fontsize=15, fontweight='bold')

    axes[0, 0].plot(ep, df_h['train_loss'], 'b-o', label='Train 8-Term Loss')
    axes[0, 0].plot(ep, df_h['val_l1'], 'r--s', label='Val L1')
    axes[0, 0].set_title("Hội Tụ Loss & Val L1"); axes[0, 0].grid(True, alpha=0.3); axes[0, 0].legend()

    axes[0, 1].plot(ep, df_h['wcd'], 'purple', marker='o', label='M3 WCD (Eq.15)')
    axes[0, 1].plot(ep, df_h['wcd_dog'], 'darkcyan', marker='^', label='M3 WCD (DoG - Unbiased)')
    if 'wcd_dog' in m1_b:
        axes[0, 1].axhline(m1_b['wcd_dog'], color='red', linestyle='--', label=f"M1 Orig WCD_DoG ({m1_b['wcd_dog']:.4f})")
    axes[0, 1].set_title("Sai Số Hình Học WCD ↓"); axes[0, 1].grid(True, alpha=0.3); axes[0, 1].legend()

    axes[0, 2].plot(ep, df_h['skel_recall'] * 100, 'g-o', label='M3 Skeleton Recall (%)')
    axes[0, 2].plot(ep, df_h['tip_recall'] * 100, 'lime', linestyle='--', marker='d', label='M3 Hair-Tip Recall (%)')
    if 'skel_recall' in m1_b:
        axes[0, 2].axhline(m1_b['skel_recall'] * 100, color='red', linestyle='--', label=f"M1 Orig Skel ({m1_b['skel_recall']*100:.1f}%)")
    axes[0, 2].set_title("Bảo Toàn Khung Xương & Ngọn Tóc ↑"); axes[0, 2].grid(True, alpha=0.3); axes[0, 2].legend()

    axes[1, 0].plot(ep, df_h['psnr'], 'navy', marker='o', label='M3 PSNR (dB)')
    if 'psnr' in m1_b:
        axes[1, 0].axhline(m1_b['psnr'], color='red', linestyle='--', label=f"M1 Orig PSNR ({m1_b['psnr']:.2f}dB)")
    axes[1, 0].set_title("PSNR (dB) ↑"); axes[1, 0].grid(True, alpha=0.3); axes[1, 0].legend()

    axes[1, 1].plot(ep, df_h['density_ratio'], 'teal', marker='o', label='Stroke Density Ratio (1.0=Exact)')
    axes[1, 1].axhline(1.0, color='gray', linestyle=':')
    axes[1, 1].axhline(1.07, color='orange', linestyle='--', label='Hard Gate Max (1.07)')
    axes[1, 1].axhline(0.93, color='orange', linestyle='--', label='Hard Gate Min (0.93)')
    axes[1, 1].set_title("Tỷ Lệ Mật Độ Nét (Chống Tẩy / Dày Nét)"); axes[1, 1].grid(True, alpha=0.3); axes[1, 1].legend()

    axes[1, 2].plot(ep, df_h['ghost'], 'darkorange', marker='o', label='Val Ghost Pixels ↓')
    if 'ghost' in m1_b:
        axes[1, 2].axhline(m1_b['ghost'], color='red', linestyle='--', label=f"M1 Orig Ghost ({m1_b['ghost']:.0f})")
    axes[1, 2].set_title("Ghost Pixels Trên Tập Val ↓"); axes[1, 2].grid(True, alpha=0.3); axes[1, 2].legend()

    plt.tight_layout()
    dash_path = os.path.join(WORKING_DIR, "training_convergence_dashboard.png")
    plt.savefig(dash_path, dpi=150)
    plt.show()
    print(f"[✓] Đã lưu Dashboard hội tụ tại: {dash_path}")
else:
    print("[*] Chưa có training_history_finetune.csv.")

# Tóm tắt kết luận tự động từ dữ liệu Benchmark #1 (nếu có)
audit_json = os.path.join(BENCH_OUT_DIR, "bench1_isolated_sequences", "benchmark_statistical_audit.json")
if os.path.exists(audit_json):
    au = json.load(open(audit_json, 'r', encoding='utf-8'))
    verdict = au.get('hypothesis_verdict', 'N/A')
    h2h = au.get('head_to_head_M3_vs_M1', {})
    lines = [
        f"### ⚖️ TỔNG KẾT ĐỐI CHỨNG KHÁCH QUAN: `{verdict}`",
        "- **Cam kết kiến trúc:** `M1` và `M3` cùng chạy trên đúng `TPSInbetweenOriginal` (**10,068,668 active params**, **0% thay đổi kiến trúc**).",
        "- **Các điểm yếu cố hữu của kiến trúc gốc (vẫn tồn tại dù dùng dataset nào):** vùng che khuất 1-chiều (occlusion), một lưới TPS phẳng, và trường `IFNet` không nhận trực tiếp kênh thời gian `t`."
    ]
    for mc in ['wcd_dog', 'semd', 'skel_recall', 'f1', 'psnr']:
        if mc in h2h:
            st = h2h[mc]
            lines.append(f"- `{mc}`: M1 = `{st['m1_mean']:.4f}` $\\longrightarrow$ M3 = `{st['m3_mean']:.4f}` (WinRate = **{st['frame_win_rate_pct']:.1f}%**, Scene Wilcoxon $p$ = `{st['scene_wilcoxon_p']:.2e}`)")
    display(Markdown("\n".join(lines)))
'''
    cells.append(_make_code(c9_code))

    # ==========================================================================
    # CELL 10: CODE - ĐÓNG GÓI THÀNH PHẨM TINH GỌN VÀO 1 FILE ZIP
    # ==========================================================================
    c10_code = r'''# ==============================================================================
# 📦 [CELL 10]: ĐÓNG GÓI CHECKPOINT BEST + LAST & TOÀN BỘ KẾT QUẢ BENCHMARK (.ZIP)
# ==============================================================================
import os
import zipfile
import shutil
import torch

RELEASE_ZIP = os.path.join(WORKING_DIR, "tps_sequence_finetune_option3_release.zip")
STRIP_TEMP = os.path.join("/tmp" if os.path.exists("/tmp") else WORKING_DIR, "_strip_ckpt_tmp")
os.makedirs(STRIP_TEMP, exist_ok=True)

# Rút gọn optimizer_state khi đóng gói release để tải về nhanh (chỉ giữ trọng số chuẩn)
best_src = os.path.join(OUTPUT_CKPT_DIR, "model_tps_finetuned_best.pt")
last_src = os.path.join(OUTPUT_CKPT_DIR, "model_tps_finetuned_last.pt")

pack_items = []
for src_p, zip_name in [(best_src, "checkpoints/model_tps_finetuned_best.pt"),
                        (last_src, "checkpoints/model_tps_finetuned_last.pt")]:
    if os.path.exists(src_p):
        raw = torch.load(src_p, map_location='cpu', weights_only=False)
        # Chọn EMA weights làm model_state chuẩn cho file tải về gọn (~52MB/file)
        clean_sd = raw.get('model_state')
        if zip_name.endswith("last.pt") and raw.get('ema_state') is not None:
            es = raw['ema_state']
            clean_sd = es['shadow'] if isinstance(es, dict) and 'shadow' in es else clean_sd
        lite_obj = {
            'epoch': raw.get('epoch'),
            'best_score': raw.get('best_score'),
            'model_state': clean_sd,
            'history': raw.get('history', []),
        }
        tmp_out = os.path.join(STRIP_TEMP, os.path.basename(zip_name))
        torch.save(lite_obj, tmp_out)
        pack_items.append((tmp_out, zip_name))

# Đính kèm file log training & dashboard
for fname in ["training_history_finetune.csv", "val_baseline_m0_m1.json", "training_convergence_dashboard.png"]:
    fp = os.path.join(WORKING_DIR, fname)
    if os.path.exists(fp):
        pack_items.append((fp, fname))

# Đính kèm toàn bộ output Benchmark #1 & #2
if os.path.exists(BENCH_OUT_DIR):
    for root, _, files in os.walk(BENCH_OUT_DIR):
        for f in files:
            full_p = os.path.join(root, f)
            rel_p = os.path.relpath(full_p, WORKING_DIR)
            pack_items.append((full_p, rel_p))

with zipfile.ZipFile(RELEASE_ZIP, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for src_f, arc_name in pack_items:
        zf.write(src_f, arcname=arc_name)

shutil.rmtree(STRIP_TEMP, ignore_errors=True)

print("=" * 84)
print(f"📦 ĐÃ ĐÓNG GÓI THÀNH CÔNG {len(pack_items)} TỆP VÀO: {RELEASE_ZIP}")
print(f"   Dung lượng nén: {os.path.getsize(RELEASE_ZIP)/(1024**2):.2f} MB")
print(f"📁 Kiểm tra thư mục Checkpoint (chỉ còn đúng best & last): {os.listdir(OUTPUT_CKPT_DIR)}")
print("=" * 84)
'''
    cells.append(_make_code(c10_code))

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "gpuClass": "standard",
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "dataSources": [],
                "isGpuEnabled": True,
                "isInternetEnabled": True,
                "language": "python"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    out_nb_path = os.path.join(THIS_DIR, "train_and_benchmark_tps_sequence_final_kaggle.ipynb")
    with open(out_nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"[✓] Đã tạo Notebook Kaggle Option 3 (LONG): {out_nb_path}")
    print(f"    Số lượng Cells: {len(cells)} (Cell 0 -> Cell 10)")
    print(f"    Kích thước tệp: {os.path.getsize(out_nb_path)/(1024**2):.2f} MB")
    return out_nb_path


if __name__ == "__main__":
    build_notebook()
