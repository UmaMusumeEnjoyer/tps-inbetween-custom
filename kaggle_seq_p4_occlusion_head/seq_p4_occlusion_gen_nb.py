#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_kaggle_occlusion_nb.py
--------------------------------------------------------------------------------
Trình tạo Kaggle Notebook Tự Thân Pha 4 (Phase 4 Occlusion Head):
    train_tps_occlusion_kaggle.ipynb

Thực thi Thử nghiệm Cách ly:
  - Tên mã: Phase4_Occlusion / M4_Occ
  - Khởi tạo Warm-Start 100% từ Phase 3 Remake (`model_tps_finetuned_best.pt`).
  - Lớp ConvTranspose2d của IFNet.block2 xuất 6 kênh (4 flow + 2 occlusion masks).
  - Khởi tạo bias 2 kênh mặt nạ = +2.0 (Sigmoid ≈ 0.88), weights = 0.0.
  - Hàm loss OcclusionAwareLoss (Photometric có trọng số M + Coverage constraint + Edge-aware smoothness).
  - Tự động xuất 2 checkpoint độc lập:
      model_tps_occlusion_best.pt
      model_tps_occlusion_last.pt
  - Chạy Triple-Benchmark so sánh đối đầu trực tiếp:
      Pha 2 Blended vs Pha 3 Remake vs Pha 4 Occlusion Head.
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

MODEL_OCCLUSION_PATH = os.path.join(THIS_DIR, "seq_p4_occlusion_model.py")
WORKER_OCCLUSION_PATH = os.path.join(THIS_DIR, "seq_p4_occlusion_worker.py")
MULTILAYER_PATH = os.path.join(REPO_ROOT, "pipelines", "seq_p3_multi_layer_tps.py")
BUILD_CACHE_PATH = os.path.join(REPO_ROOT, "pipelines", "seq_p3_build_multilayer_cache.py")
WORKER_P2_PATH = os.path.join(REPO_ROOT, "kaggle_seq_p2_v33_ghost_purge", "seq_p2_v33_ghost_purge_worker.py")
WORKER_P3_PATH = os.path.join(REPO_ROOT, "kaggle_seq_p3_multilayer_tps", "seq_p3_v2_multilayer_tps_worker.py")
BENCH_SUITE_PATH = os.path.join(
    REPO_ROOT, "Dataset", "sakuga_pipeline_sequence", "final_dataset",
    "sakuga_golden_benchmark_sequences_v2", "run_benchmark_suite.py"
)
GLUESTICK_NPZ_PATH = os.path.join(REPO_ROOT, "kaggle_seq_p2_v33_ghost_purge", "lineart_test_gluestick_matches.npz")


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
    model_occ_code = _read_text(MODEL_OCCLUSION_PATH)
    worker_occ_code = _read_text(WORKER_OCCLUSION_PATH)
    ml_code = _read_text(MULTILAYER_PATH)
    build_cache_code = _read_text(BUILD_CACHE_PATH)
    worker_p2_code = _read_text(WORKER_P2_PATH)
    worker_p3_code = _read_text(WORKER_P3_PATH)
    bench_code = _read_text(BENCH_SUITE_PATH)
    gluestick_b64 = _read_b64(GLUESTICK_NPZ_PATH)

    cells = []

    c0_md = r"""# 🔬 THỬ NGHIỆM CÁCH LY PHA 4: OCCLUSION-AWARE HEAD (TPSInbetweenOcclusion)
## Warm-Start 100% từ Phase 3 Remake Best | 6 Epochs | 2x GPU Tesla T4 (DDP)
### Tên mã: `Phase4_Occlusion` / `M4_Occ`

---

### 🎯 1. Mục Tiêu & Cải Tiến Kỹ Thuật Độc Lập
1. **Kiến trúc Occlusion Head trong IFNet.block2:**
   - Lớp `lastconv` xuất 6 kênh: $[F_{0\to t}^{(2)}, F_{1\to t}^{(2)}, M_{0\to t}^{(1)}, M_{1\to t}^{(1)}]$.
   - Chuẩn hóa 2 mặt nạ qua hàm $\text{Sigmoid} \in [0.0, 1.0]$.
2. **Bộ Trộn Trọng số Chuẩn hóa (Normalized Visibility Blending):**
   $$W_0 = (1 - t) \cdot M_{0\to t}, \quad W_1 = t \cdot M_{1\to t}$$
   $$I_{\text{fused}} = \frac{W_0 \cdot \text{Warp}(I_0, F_{0\to t}) + W_1 \cdot \text{Warp}(I_1, F_{1\to t})}{W_0 + W_1 + \epsilon}$$
3. **Warm-Start mượt mà tuyệt đối (Zero Loss Shock):**
   - Kế thừa trọn vẹn 158 tensors từ checkpoint `model_tps_finetuned_best.pt` của Pha 3 Remake.
   - Khởi tạo bias 2 kênh mặt nạ che khuất $= +2.0$ ($\text{Sigmoid}(+2.0) \approx 0.8808$, coi ban đầu mọi vùng đều nhìn thấy được).
4. **Hàm Mất Mát Nhận Thức Che Khuất (OcclusionAwareLoss):**
   - $L_{\text{photo}} = M_{0\to t} \cdot \rho(I_{0\to t} - I_t^{\text{GT}}) + M_{1\to t} \cdot \rho(I_{1\to t} - I_t^{\text{GT}})$.
   - $L_{\text{cov}} = \text{mean}\left(\text{ReLU}(0.95 - (M_{0\to t} + M_{1\to t}))^2\right)$ (chống hiện tượng mạng lười triệt tiêu $M \to 0$).
   - $L_{\text{smooth}}$: Làm mượt mặt nạ có hướng dẫn biên nét ảnh GT.
"""
    cells.append(_make_md(c0_md))

    c1_code = f'''# ==============================================================================
# 🛠️ [CELL 1]: KHỞI TẠO MÔI TRƯỜNG & NHÚNG MÃ NGUỒN PHA 4 OCCLUSION
# ==============================================================================
import os
import sys
import gc
import base64
import torch
import psutil

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
WORKING_DIR = "/kaggle/working" if IS_KAGGLE else os.path.abspath("./kaggle_seq_01_local_test_workspace")
OUTPUT_CKPT_DIR = os.path.join(WORKING_DIR, "checkpoints_phase4_occlusion")
BENCH_OUT_DIR = os.path.join(WORKING_DIR, "bench_outputs_seq_p4_occlusion")
for _d in (WORKING_DIR, OUTPUT_CKPT_DIR, BENCH_OUT_DIR):
    os.makedirs(_d, exist_ok=True)

if WORKING_DIR not in sys.path:
    sys.path.insert(0, WORKING_DIR)

gpu_count = torch.cuda.device_count()
print("=" * 84)
print(f"🖥️ KIỂM TRA PHẦN CỨNG: {{gpu_count}} GPU khả dụng (Kaggle={{IS_KAGGLE}})")
for i in range(gpu_count):
    p = torch.cuda.get_device_properties(i)
    print(f"   👉 GPU {{i}}: {{p.name}} | VRAM = {{p.total_memory / (1024**3):.2f}} GB")
mem = psutil.virtual_memory()
print(f"🧠 Host RAM: Tổng = {{mem.total / (1024**3):.1f}} GB | Trống = {{mem.available / (1024**3):.1f}} GB")
print("=" * 84)

# Ghi các module cách ly của Phase 4
model_occ_path = os.path.join(WORKING_DIR, "seq_p4_occlusion_model.py")
with open(model_occ_path, "w", encoding="utf-8") as f:
    f.write({repr(model_occ_code)})

worker_occ_path = os.path.join(WORKING_DIR, "seq_p4_occlusion_worker.py")
with open(worker_occ_path, "w", encoding="utf-8") as f:
    f.write({repr(worker_occ_code)})

ml_path = os.path.join(WORKING_DIR, "seq_p3_multi_layer_tps.py")
with open(ml_path, "w", encoding="utf-8") as f:
    f.write({repr(ml_code)})

build_cache_path = os.path.join(WORKING_DIR, "seq_p3_build_multilayer_cache.py")
with open(build_cache_path, "w", encoding="utf-8") as f:
    f.write({repr(build_cache_code)})

worker_p2_path = os.path.join(WORKING_DIR, "seq_p2_v33_ghost_purge_worker.py")
with open(worker_p2_path, "w", encoding="utf-8") as f:
    f.write({repr(worker_p2_code)})

worker_p3_path = os.path.join(WORKING_DIR, "seq_p3_v2_multilayer_tps_worker.py")
with open(worker_p3_path, "w", encoding="utf-8") as f:
    f.write({repr(worker_p3_code)})

bench_script_path = os.path.join(WORKING_DIR, "run_benchmark_suite.py")
with open(bench_script_path, "w", encoding="utf-8") as f:
    f.write({repr(bench_code)})

GLUESTICK_TEST_NPZ = os.path.join(WORKING_DIR, "lineart_test_gluestick_matches.npz")
with open(GLUESTICK_TEST_NPZ, "wb") as f:
    f.write(base64.b64decode({repr(gluestick_b64)}))

print(f"[✓] Đã ghi seq_p4_occlusion_model.py ({{os.path.getsize(model_occ_path):,}} B)")
print(f"[✓] Đã ghi seq_p4_occlusion_worker.py ({{os.path.getsize(worker_occ_path):,}} B)")
print(f"[✓] Đã ghi seq_p3_multi_layer_tps.py ({{os.path.getsize(ml_path):,}} B)")
print(f"[✓] Đã ghi run_benchmark_suite.py ({{os.path.getsize(bench_script_path):,}} B)")
sys.stdout.flush()
'''
    cells.append(_make_code(c1_code))

    c2_code = r'''# ==============================================================================
# ⚡ [CELL 2]: ĐỊNH VỊ TOÀN BỘ INPUTS, WARM-START CHECKPOINT (PHA 3 REMAKE BEST)
# ==============================================================================
import os
import glob
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
    "E:/CODE/PBL/PBL6/checkpoints/Re-train/Sequence 2 pha 3 remake/checkpoints",
    "E:/CODE/PBL/PBL6/checkpoints/Re-train/Sequence 2 Blended/checkpoints",
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
            hits.sort(key=lambda p: len(p))
            return hits[0]
    return None

def safe_stage_dir_links(src_root, dst_root, subdirs):
    for sub in subdirs:
        src_p = os.path.join(src_root, sub)
        dst_p = os.path.join(dst_root, sub)
        if not os.path.exists(src_p) or os.path.abspath(src_p) == os.path.abspath(dst_p):
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
print("🔍 [CELL 2]: ĐANG TỰ ĐỘNG ĐỊNH VỊ CHECKPOINT & DATASET CHO PHA 4 OCCLUSION...")
print("=" * 84)

already_extracted_zips = set()
for rb in [r for r in SEARCH_ROOTS if r != SSD_ROOT and os.path.exists(r)]:
    candidate_zips = glob.glob(os.path.join(rb, "*.zip")) + glob.glob(os.path.join(rb, "*", "*.zip"))
    for zpath in candidate_zips:
        if zpath in already_extracted_zips:
            continue
        try:
            with zipfile.ZipFile(zpath, 'r') as zf:
                names = zf.namelist()
                if any(n.endswith(("train_sequences.csv", "benchmark_sequences.csv", "continuous_keyframes_spec.json")) for n in names):
                    z_stem = os.path.splitext(os.path.basename(zpath))[0]
                    dest_extract = os.path.join(EXTRACT_CACHE_DIR, z_stem)
                    if not os.path.exists(dest_extract) or len(os.listdir(dest_extract)) == 0:
                        os.makedirs(dest_extract, exist_ok=True)
                        print(f"📦 Giải nén {os.path.basename(zpath)} sang SSD...", flush=True)
                        zf.extractall(dest_extract)
                    already_extracted_zips.add(zpath)
        except Exception as e:
            print(f"   [!] Bỏ qua zip lỗi {zpath}: {e}")

# 1. Checkpoint Warm-Start: Ưu tiên model_tps_finetuned_best.pt (Phase 3 Remake Best)
P3_BEST_CHECKPOINT_PATH = find_file_in_roots("model_tps_finetuned_best.pt")
if not P3_BEST_CHECKPOINT_PATH:
    P3_BEST_CHECKPOINT_PATH = find_file_in_roots("model_tps_blended.pt")
if not P3_BEST_CHECKPOINT_PATH:
    P3_BEST_CHECKPOINT_PATH = find_file_in_roots("model_latest.pt")
assert P3_BEST_CHECKPOINT_PATH and os.path.exists(P3_BEST_CHECKPOINT_PATH), "❌ Thiếu checkpoint warm-start!"
print(f"[✓] 1. Warm-Start Checkpoint (Pha 3 Remake) : {P3_BEST_CHECKPOINT_PATH}")

BLENDED_CKPT_PATH = find_file_in_roots("model_tps_blended.pt")
M1_CHECKPOINT_PATH = find_file_in_roots("model_latest.pt")

# 2. Dataset Sequences
train_csv_found = find_file_in_roots("train_sequences.csv")
assert train_csv_found, "❌ Thiếu `train_sequences.csv`!"
TRAIN_SRC_ROOT = os.path.dirname(train_csv_found)
safe_stage_dir_links(TRAIN_SRC_ROOT, TRAIN_STAGE_DIR, ["train", "val", "matches"])
safe_copy_file(train_csv_found, os.path.join(TRAIN_STAGE_DIR, "train_sequences.csv"))
safe_copy_file(os.path.join(TRAIN_SRC_ROOT, "val_sequences.csv"), os.path.join(TRAIN_STAGE_DIR, "val_sequences.csv"))

PRECOMPUTED_ML_CACHE_SRC = find_file_in_roots("tps_cache_multilayer.npz")
TRAIN_CLIPS_DIR = os.path.join(TRAIN_STAGE_DIR, "train")
VAL_CLIPS_DIR = os.path.join(TRAIN_STAGE_DIR, "val")
print(f"[✓] 2. Dataset Train/Val Sequence          : {TRAIN_SRC_ROOT}")

# 3. Benchmark sets
bench_csv_found = find_file_in_roots("benchmark_sequences.csv")
BENCH_READY, BENCH_ROOT_PATH = False, None
if bench_csv_found and os.path.exists(bench_csv_found):
    b_src = os.path.dirname(bench_csv_found)
    safe_stage_dir_links(b_src, BENCH_STAGE_DIR, ["clips", "matches"])
    safe_copy_file(bench_csv_found, os.path.join(BENCH_STAGE_DIR, "benchmark_sequences.csv"))
    BENCH_ROOT_PATH, BENCH_READY = BENCH_STAGE_DIR, True
    print(f"[✓] 3. Benchmark #1 (Isolated 198 clips)   : {b_src}")

spec_found = find_file_in_roots("continuous_keyframes_spec.json")
LINEART_TEST_READY, LINEART_TEST_PATH = False, None
if spec_found and os.path.exists(spec_found):
    l_src = os.path.dirname(spec_found)
    for fn in os.listdir(l_src):
        src_f = os.path.join(l_src, fn)
        if os.path.isfile(src_f):
            safe_copy_file(src_f, os.path.join(LINEART_TEST_STAGE, fn))
    LINEART_TEST_PATH, LINEART_TEST_READY = LINEART_TEST_STAGE, True
    print(f"[✓] 4. Benchmark #2 (Lineart Test 24f)     : {l_src}")
print("=" * 84)
'''
    cells.append(_make_code(c2_code))

    c3_code = r'''# ==============================================================================
# 🚀 [CELL 3]: NẠP HOẶC TẠO BỘ CACHE ĐA LỚP `tps_cache_multilayer.npz`
# ==============================================================================
import os
import shutil
import pandas as pd
from seq_p3_build_multilayer_cache import build_multilayer_cache

TPS_CACHE_MULTILAYER = os.path.join(SSD_ROOT, "tps_cache_multilayer.npz")
TRAIN_CSV_IDX_PATH = os.path.join(TRAIN_STAGE_DIR, "train_sequences_indexed.csv")
VAL_CSV_IDX_PATH = os.path.join(TRAIN_STAGE_DIR, "val_sequences_indexed.csv")

if PRECOMPUTED_ML_CACHE_SRC and os.path.exists(PRECOMPUTED_ML_CACHE_SRC):
    shutil.copy2(PRECOMPUTED_ML_CACHE_SRC, TPS_CACHE_MULTILAYER)
    df_tr = pd.read_csv(os.path.join(TRAIN_STAGE_DIR, "train_sequences.csv"))
    df_va = pd.read_csv(os.path.join(TRAIN_STAGE_DIR, "val_sequences.csv"))
    df_tr['cache_idx'] = range(len(df_tr))
    df_va['cache_idx'] = range(len(df_tr), len(df_tr) + len(df_va))
    df_tr.to_csv(TRAIN_CSV_IDX_PATH, index=False)
    df_va.to_csv(VAL_CSV_IDX_PATH, index=False)
    print(f"[♻️] Đã nạp trực tiếp `tps_cache_multilayer.npz` ({os.path.getsize(TPS_CACHE_MULTILAYER)/(1024**2):.2f} MB)!")
elif os.path.exists(TPS_CACHE_MULTILAYER) and os.path.exists(TRAIN_CSV_IDX_PATH) and os.path.exists(VAL_CSV_IDX_PATH):
    print(f"[♻️] Phát hiện cache sẵn có tại {TPS_CACHE_MULTILAYER}!")
else:
    build_multilayer_cache(
        TRAIN_STAGE_DIR,
        out_npz_path=TPS_CACHE_MULTILAYER,
        max_pts=128, max_clusters=4, max_mask_pts=64, max_static_pts=96,
        lambd=0.01, eps_static_px=1.5
    )
'''
    cells.append(_make_code(c3_code))

    c4_code = r'''# ==============================================================================
# 🔬 [CELL 4]: KIỂM TOÁN KIẾN TRÚC MÔ HÌNH PHA 4 & WARM-START TỪ PHA 3 REMAKE
# ==============================================================================
import gc
import torch
from seq_p4_occlusion_model import TPSInbetweenOcclusion, load_tps_occlusion_weights

m4 = TPSInbetweenOcclusion()
load_tps_occlusion_weights(m4, P3_BEST_CHECKPOINT_PATH, device='cpu', prefer_ema=True, is_rank0=True)
p_tot = sum(p.numel() for p in m4.parameters())
print(f"[✓] TỔNG ACTIVE PARAMS (Phase 4 Occlusion) = {p_tot:,}")
print(f"[✓] IFNet.block2.lastconv weight shape      = {tuple(m4.IFNet.block2.lastconv.weight.shape)} (6 channels)")
print(f"[✓] IFNet.block2.lastconv bias shape        = {tuple(m4.IFNet.block2.lastconv.bias.shape)} (6 channels)")
print(f"[✓] Khởi tạo bias Occlusion Head            = {m4.IFNet.block2.lastconv.bias[4:].detach().tolist()} (Sigmoid ≈ 0.88)")

del m4
gc.collect()
'''
    cells.append(_make_code(c4_code))

    c5_code = r'''# ==============================================================================
# 🚀 [CELL 5]: CHẠY HUẤN LUYỆN PHA 4 OCCLUSION HEAD (6 EPOCHS | 2x GPU DDP)
# ==============================================================================
import os
import sys
import subprocess
import torch

NUM_GPUS = max(1, torch.cuda.device_count())
TOTAL_EPOCHS = 6
GRAD_ACCUM = 2
MAX_CHUNK_FRAMES = 3
LR_BACKBONE = 6.0e-6
LR_UNET = 1.8e-5
EMA_DECAY = 0.996
W_PHOTO = 0.25
W_COV = 0.50
W_SMOOTH = 0.10
W_GHOST = 0.32
W_TEMP = 0.55

cmd = [
    "torchrun",
    f"--nproc_per_node={NUM_GPUS}",
    "--master_port=29519",
    worker_occ_path,
    "--train_csv", TRAIN_CSV_IDX_PATH,
    "--val_csv", VAL_CSV_IDX_PATH,
    "--train_clips_dir", TRAIN_CLIPS_DIR,
    "--val_clips_dir", VAL_CLIPS_DIR,
    "--tps_cache", TPS_CACHE_MULTILAYER,
    "--init_checkpoint", P3_BEST_CHECKPOINT_PATH,
    "--output_dir", OUTPUT_CKPT_DIR,
    "--epochs", str(TOTAL_EPOCHS),
    "--grad_accum", str(GRAD_ACCUM),
    "--max_chunk_frames", str(MAX_CHUNK_FRAMES),
    "--lr_backbone", str(LR_BACKBONE),
    "--lr_unet", str(LR_UNET),
    "--ema_decay", str(EMA_DECAY),
    "--w_photo", str(W_PHOTO),
    "--w_cov", str(W_COV),
    "--w_smooth", str(W_SMOOTH),
    "--w_ghost", str(W_GHOST),
    "--w_temp", str(W_TEMP),
    "--num_workers", "2",
]

print("=" * 84)
print(f"🚀 BẮT ĐẦU HUẤN LUYỆN PHA 4 OCCLUSION ({NUM_GPUS}x GPU | {TOTAL_EPOCHS} EPOCHS)")
print(f"   Warm-Start: {P3_BEST_CHECKPOINT_PATH}")
print(f"   Outputs   : {OUTPUT_CKPT_DIR}")
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

if ret != 0:
    raise RuntimeError(f"❌ DDP Worker kết thúc bất thường với mã {ret}!")
else:
    print("🎉 HOÀN TẤT HUẤN LUYỆN PHA 4 OCCLUSION HEAD!")
'''
    cells.append(_make_code(c5_code))

    c6_code = r'''# ==============================================================================
# 🔄 [CELL 6 - KHẨN CẤP]: RESET KERNEL TRÁNH TRÀN RAM
# ==============================================================================
# import os
# os._exit(0)
'''
    cells.append(_make_code(c6_code))

    c7_code = r'''# ==============================================================================
# 🎬 [CELL 7]: BENCHMARK ĐỐI ĐẦU 4 PHIÊN BẢN (PHA 2 vs PHA 3 vs PHA 4 OCCLUSION)
# ==============================================================================
import os
import gc
import torch
import pandas as pd
from IPython.display import display, Image as IPyImage
from seq_p2_v33_ghost_purge_worker import TPSInbetweenOriginal, load_tps_weights
from seq_p4_occlusion_model import TPSInbetweenOcclusion, load_tps_occlusion_weights
from seq_dual_benchmark_suite import run_benchmark_continuous_lineart_test, run_benchmark_defense_geometry_suite

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
models_dict = {}

# 1. Pha 2 Blended
if BLENDED_CKPT_PATH and os.path.exists(BLENDED_CKPT_PATH):
    m_p2 = TPSInbetweenOriginal().to(device)
    load_tps_weights(m_p2, BLENDED_CKPT_PATH, device=device, prefer_ema=True)
    models_dict['Pha2_Blended'] = m_p2.eval()

# 2. Pha 3 Remake Raw
if P3_BEST_CHECKPOINT_PATH and os.path.exists(P3_BEST_CHECKPOINT_PATH):
    m_p3 = TPSInbetweenOriginal().to(device)
    load_tps_weights(m_p3, P3_BEST_CHECKPOINT_PATH, device=device, prefer_ema=True)
    models_dict['Pha3_Remake_Raw'] = m_p3.eval()

# 3. Pha 4 Occlusion Head
p4_best_ckpt = os.path.join(OUTPUT_CKPT_DIR, "model_tps_occlusion_best.pt")
p4_last_ckpt = os.path.join(OUTPUT_CKPT_DIR, "model_tps_occlusion_last.pt")

if os.path.exists(p4_best_ckpt):
    m_p4 = TPSInbetweenOcclusion().to(device)
    load_tps_occlusion_weights(m_p4, p4_best_ckpt, device=device, prefer_ema=True)
    models_dict['Pha4_Occlusion_Best'] = m_p4.eval()
elif os.path.exists(p4_last_ckpt):
    m_p4 = TPSInbetweenOcclusion().to(device)
    load_tps_occlusion_weights(m_p4, p4_last_ckpt, device=device, prefer_ema=True)
    models_dict['Pha4_Occlusion_Last'] = m_p4.eval()

print(f"📊 Đã nạp thành công {len(models_dict)} mô hình đối đầu: {list(models_dict.keys())}")

# Chạy Benchmark Case 1: Lineart Test 24 frames
BENCH2_OUT = os.path.join(BENCH_OUT_DIR, "bench_lineart_test_24f")
if LINEART_TEST_READY:
    df_sum2 = run_benchmark_continuous_lineart_test(
        LINEART_TEST_PATH, models_dict, device, BENCH2_OUT,
        gluestick_npz=GLUESTICK_TEST_NPZ, downscale_factor=2, use_multilayer=True
    )
    if df_sum2 is not None:
        print("📌 KẾT QUẢ CHUỖI 24 FRAMES THỰC TẾ:")
        display(df_sum2.round(4))

# Chạy Benchmark Case 2, 3, 4: Defense Geometry Suite
BENCH3_OUT = os.path.join(BENCH_OUT_DIR, "bench_defense_geometry")
df_sum3 = run_benchmark_defense_geometry_suite(models_dict, device, BENCH3_OUT, downscale_factor=2)
if df_sum3 is not None:
    print("📌 KẾT QUẢ BỘ DEFENSE GEOMETRY SUITE:")
    display(df_sum3.round(4))

del models_dict
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
'''
    cells.append(_make_code(c7_code))

    c8_code = r'''# ==============================================================================
# 📈 [CELL 8]: DASHBOARD HỘI TỤ TRAINING PHA 4 OCCLUSION
# ==============================================================================
import os
import pandas as pd
from IPython.display import display

hist_csv = os.path.join(OUTPUT_CKPT_DIR, "training_history_occlusion.csv")
if os.path.exists(hist_csv):
    df_h = pd.read_csv(hist_csv)
    print("📈 LỊCH SỬ HUẤN LUYỆN PHA 4 OCCLUSION HEAD:")
    display(df_h.round(4))
else:
    print("[*] Chưa tìm thấy training_history_occlusion.csv.")
'''
    cells.append(_make_code(c8_code))

    c9_code = r'''# ==============================================================================
# 📦 [CELL 9]: ĐÓNG GÓI CHECKPOINT & KẾT QUẢ BENCHMARK PHA 4 (.ZIP)
# ==============================================================================
import os
import zipfile
import shutil
import torch

RELEASE_ZIP = os.path.join(WORKING_DIR, "tps_sequence_phase4_occlusion_release.zip")
pack_items = []

for src_p, zip_name in [
    (os.path.join(OUTPUT_CKPT_DIR, "model_tps_occlusion_best.pt"), "checkpoints/model_tps_occlusion_best.pt"),
    (os.path.join(OUTPUT_CKPT_DIR, "model_tps_occlusion_last.pt"), "checkpoints/model_tps_occlusion_last.pt"),
    (os.path.join(OUTPUT_CKPT_DIR, "training_history_occlusion.csv"), "checkpoints/training_history_occlusion.csv"),
]:
    if os.path.exists(src_p):
        pack_items.append((src_p, zip_name))

if os.path.exists(BENCH_OUT_DIR):
    for root, _, files in os.walk(BENCH_OUT_DIR):
        for f in files:
            full_p = os.path.join(root, f)
            pack_items.append((full_p, os.path.relpath(full_p, WORKING_DIR)))

with zipfile.ZipFile(RELEASE_ZIP, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for src_f, arc_name in pack_items:
        zf.write(src_f, arcname=arc_name)

print(f"📦 ĐÃ ĐÓNG GÓI THÀNH CÔNG {len(pack_items)} TỆP VÀO: {RELEASE_ZIP} ({os.path.getsize(RELEASE_ZIP)/(1024**2):.2f} MB)")
'''
    cells.append(_make_code(c9_code))

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "gpuClass": "standard",
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True, "language": "python"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out_nb_path = os.path.join(THIS_DIR, "train_tps_occlusion_kaggle.ipynb")
    with open(out_nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"[✓] Đã tạo Kaggle Notebook Pha 4: {out_nb_path} ({os.path.getsize(out_nb_path)/(1024**2):.2f} MB)")
    return out_nb_path


if __name__ == "__main__":
    build_notebook()
