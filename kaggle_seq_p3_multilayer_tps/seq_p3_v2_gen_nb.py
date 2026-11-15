#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_kaggle_finetune_phase3_nb.py
--------------------------------------------------------------------------------
Trình tạo Kaggle Notebook Tự Thân Pha 3 (11 Cells: Cell 0 -> Cell 10):
    train_and_benchmark_tps_sequence_phase3_kaggle.ipynb

Thực thi Bước 4 của Pha 3:
  - Fine-tuning nhẹ 3 Epochs từ `model_tps_blended.pt` trên 2x GPU Tesla T4.
  - Dùng trọn vẹn Coarse TPS Đa lớp (`seq_p3_multi_layer_tps.py` + `tps_cache_multilayer.npz`).
  - Làm giàu dữ liệu tại chỗ: Static-Background Compositing (25%) + 180 chuỗi hình học 6 Mode.
  - Zero Architectural Modification (`TPSInbetweenOriginal`, 10,068,668 active params).
  - Tự động chạy Triple-Benchmark (Bench #1 198 clips, Bench #2 24 frames, Bench #3 Geometry).
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

MULTILAYER_PATH = os.path.join(REPO_ROOT, "pipelines", "seq_p3_multi_layer_tps.py")
BUILD_CACHE_PATH = os.path.join(REPO_ROOT, "pipelines", "seq_p3_build_multilayer_cache.py")
WORKER_P2_PATH = os.path.join(REPO_ROOT, "kaggle_seq_p2_v33_ghost_purge", "seq_p2_v33_ghost_purge_worker.py")
WORKER_P3_PATH = os.path.join(THIS_DIR, "seq_p3_v2_multilayer_tps_worker.py")
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
    ml_code = _read_text(MULTILAYER_PATH)
    build_cache_code = _read_text(BUILD_CACHE_PATH)
    worker_p2_code = _read_text(WORKER_P2_PATH)
    worker_p3_code = _read_text(WORKER_P3_PATH)
    bench_code = _read_text(BENCH_SUITE_PATH)
    gluestick_b64 = _read_b64(GLUESTICK_NPZ_PATH)

    cells = []

    c0_md = r"""# 🔬 PHA 3 v2 (BƯỚC 4 - CÂN BẰNG BẢO TOÀN NÉT): COARSE TPS ĐA LỚP & FINE-TUNING NHẸ
## Multi-Layer Clustered TPS + Phase3BalancedLineartLoss (4 Epochs) | `10,068,668` Active Params

---

### 🎯 1. Các Điều Chỉnh Cốt Lõi Của Pha 3 v2 (Khắc Phục Tụt `Skel Recall`, `Density`, `WCD_DoG`)
1. **Hàm Loss `Phase3BalancedLineartLoss` chống tẩy mảnh nét:**
   - Thay `GhostLoss` phạt bậc hai gắt sát biên bằng `Phase3ZoneCleanlinessLoss` (nới ngưỡng vành sát nét $0.06 \to 0.12$, chỉ phạt mạnh `Dark Ghost` $\ge 0.20$ ở nền xa, `w_ghost = 0.32`).
   - Nâng sàn `Phase3ExactDensityLoss` (`dead_zone_low = 0.99`, `w_dens = 0.70`), tăng `w_width = 0.35`, `w_cldice = 0.28`, `w_dt = 0.40`.
2. **Làm giàu dữ liệu đa lớp không loãng mật độ mực:**
   - Giảm `Static-Background Compositing` xuống nét mảnh `1.5-2.5 px` với tỷ lệ `p = 0.12` + `150` chuỗi hình học đa lớp (6 Mode).
3. **LR phẫu thuật & Cổng Gate đối đầu trực tiếp với `Pha 2 Blended`:**
   - `4 Epochs` với LR dịu (`lr_backbone = 6.0e-6`, `lr_unet = 1.8e-5`, `ema_decay = 0.996`).
   - Strict Non-Regression Gate: Không chấp nhận checkpoint nếu tụt `Skel Recall`, `Density` hay phình `WCD_DoG` so với `Pha 2 Blended` đầu vào.
"""
    cells.append(_make_md(c0_md))

    c1_code = f'''# ==============================================================================
# 🛠️ [CELL 1]: KHỞI TẠO MÔI TRƯỜNG & NHÚNG TOÀN BỘ MÃ NGUỒN PHA 3 ĐỘC LẬP
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
OUTPUT_CKPT_DIR = os.path.join(WORKING_DIR, "checkpoints_tps_finetune")
BENCH_OUT_DIR = os.path.join(WORKING_DIR, "benchmark_outputs")
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

print(f"[✓] Đã ghi seq_p3_multi_layer_tps.py ({{os.path.getsize(ml_path):,}} B)")
print(f"[✓] Đã ghi seq_p3_v2_multilayer_tps_worker.py ({{os.path.getsize(worker_p3_path):,}} B)")
print(f"[✓] Đã ghi run_benchmark_suite.py ({{os.path.getsize(bench_script_path):,}} B)")
sys.stdout.flush()
'''
    cells.append(_make_code(c1_code))

    c2_code = r'''# ==============================================================================
# ⚡ [CELL 2]: ĐỊNH VỊ TOÀN BỘ INPUTS, BLENDED CHECKPOINT & STAGING SANG SSD (/tmp)
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
    "E:/CODE/PBL/PBL6/checkpoints/pretrained_baselines/tps-original",
    "E:/CODE/PBL/PBL6/checkpoints/Re-train/Sequence 2 Blended/checkpoints",
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
print("🔍 [CELL 2]: ĐANG TỰ ĐỘNG ĐỊNH VỊ CHECKPOINT & DATASET CHO PHA 3...")
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

M1_CHECKPOINT_PATH = find_file_in_roots("model_latest.pt")
assert M1_CHECKPOINT_PATH and os.path.exists(M1_CHECKPOINT_PATH), "❌ Thiếu `model_latest.pt`!"
print(f"[✓] 1. Baseline M1 Checkpoint       : {M1_CHECKPOINT_PATH}")

# Ưu tiên Warm-Start từ model_tps_blended.pt (Pha 2 Blended)
INIT_CHECKPOINT_PATH = find_file_in_roots("model_tps_blended.pt")
if not INIT_CHECKPOINT_PATH:
    INIT_CHECKPOINT_PATH = find_file_in_roots("model_tps_finetuned_last.pt")
if not INIT_CHECKPOINT_PATH:
    INIT_CHECKPOINT_PATH = M1_CHECKPOINT_PATH
print(f"[✓] 1b. Warm-Start Checkpoint Pha 3 : {INIT_CHECKPOINT_PATH}")

train_csv_found = find_file_in_roots("train_sequences.csv")
assert train_csv_found, "❌ Thiếu `train_sequences.csv`!"
TRAIN_SRC_ROOT = os.path.dirname(train_csv_found)
safe_stage_dir_links(TRAIN_SRC_ROOT, TRAIN_STAGE_DIR, ["train", "val", "matches"])
safe_copy_file(train_csv_found, os.path.join(TRAIN_STAGE_DIR, "train_sequences.csv"))
safe_copy_file(os.path.join(TRAIN_SRC_ROOT, "val_sequences.csv"), os.path.join(TRAIN_STAGE_DIR, "val_sequences.csv"))

PRECOMPUTED_ML_CACHE_SRC = find_file_in_roots("tps_cache_multilayer.npz")
TRAIN_CLIPS_DIR = os.path.join(TRAIN_STAGE_DIR, "train")
VAL_CLIPS_DIR = os.path.join(TRAIN_STAGE_DIR, "val")
TRAIN_MATCHES_DIR = os.path.join(TRAIN_STAGE_DIR, "matches")
print(f"[✓] 2. Dataset Train/Val Sequence   : {TRAIN_SRC_ROOT}")

bench_csv_found = find_file_in_roots("benchmark_sequences.csv")
BENCH_READY, BENCH_ROOT_PATH = False, None
if bench_csv_found and os.path.exists(bench_csv_found):
    b_src = os.path.dirname(bench_csv_found)
    safe_stage_dir_links(b_src, BENCH_STAGE_DIR, ["clips", "matches"])
    safe_copy_file(bench_csv_found, os.path.join(BENCH_STAGE_DIR, "benchmark_sequences.csv"))
    BENCH_ROOT_PATH, BENCH_READY = BENCH_STAGE_DIR, True
    print(f"[✓] 3. Benchmark #1 (Isolated Set)  : {b_src}")

spec_found = find_file_in_roots("continuous_keyframes_spec.json")
LINEART_TEST_READY, LINEART_TEST_PATH = False, None
if spec_found and os.path.exists(spec_found):
    l_src = os.path.dirname(spec_found)
    for fn in os.listdir(l_src):
        src_f = os.path.join(l_src, fn)
        if os.path.isfile(src_f):
            safe_copy_file(src_f, os.path.join(LINEART_TEST_STAGE, fn))
    LINEART_TEST_PATH, LINEART_TEST_READY = LINEART_TEST_STAGE, True
    print(f"[✓] 4. Benchmark #2 (Lineart Test)  : {l_src}")
print("=" * 84)
'''
    cells.append(_make_code(c2_code))

    c3_code = r'''# ==============================================================================
# 🚀 [CELL 3]: TẠO HOẶC NẠP BỘ CACHE ĐA LỚP `tps_cache_multilayer.npz`
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
    print(f"[♻️] Đã nạp trực tiếp `tps_cache_multilayer.npz` có sẵn từ Input ({os.path.getsize(TPS_CACHE_MULTILAYER)/(1024**2):.2f} MB)!")
elif os.path.exists(TPS_CACHE_MULTILAYER) and os.path.exists(TRAIN_CSV_IDX_PATH) and os.path.exists(VAL_CSV_IDX_PATH):
    print(f"[♻️] Phát hiện `tps_cache_multilayer.npz` đã tạo sẵn tại {TPS_CACHE_MULTILAYER}!")
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
# 🔬 [CELL 4]: KIỂM TOÁN 100% ĐỒNG NHẤT KIẾN TRÚC (TPSInbetweenOriginal - 0% BIAS)
# ==============================================================================
import gc
import torch
from seq_p2_v33_ghost_purge_worker import TPSInbetweenOriginal, load_tps_weights

m1 = TPSInbetweenOriginal()
load_tps_weights(m1, M1_CHECKPOINT_PATH, device='cpu', prefer_ema=False, is_rank0=True)
p_tot = sum(p.numel() for p in m1.parameters())
print(f"[✓] TỔNG ACTIVE PARAMS = {p_tot:,} (100% Zero Architectural Modification)")
del m1
gc.collect()
'''
    cells.append(_make_code(c4_code))

    c5_code = r'''# ==============================================================================
# 🚀 [CELL 5]: CHẠY FINE-TUNING NHẸ PHA 3 (3 EPOCHS | 2x GPU DDP | MULTI-LAYER TPS)
# ==============================================================================
import os
import sys
import subprocess
import torch

NUM_GPUS = max(1, torch.cuda.device_count())
TOTAL_EPOCHS = 4
GRAD_ACCUM = 2
MAX_CHUNK_FRAMES = 3
LR_BACKBONE = 6.0e-6
LR_UNET = 1.8e-5
EMA_DECAY = 0.996
SYNTH_COUNT = 150
STATIC_AUG_PROB = 0.12
W_GHOST = 0.32
W_TEMP = 0.55
RAM_LIMIT_GB = 23.5

cmd = [
    "torchrun",
    f"--nproc_per_node={NUM_GPUS}",
    "--master_port=29508",
    worker_p3_path,
    "--train_csv", TRAIN_CSV_IDX_PATH,
    "--val_csv", VAL_CSV_IDX_PATH,
    "--train_clips_dir", TRAIN_CLIPS_DIR,
    "--val_clips_dir", VAL_CLIPS_DIR,
    "--tps_cache", TPS_CACHE_MULTILAYER,
    "--m1_checkpoint", M1_CHECKPOINT_PATH,
    "--init_checkpoint", INIT_CHECKPOINT_PATH,
    "--output_dir", OUTPUT_CKPT_DIR,
    "--epochs", str(TOTAL_EPOCHS),
    "--grad_accum", str(GRAD_ACCUM),
    "--max_chunk_frames", str(MAX_CHUNK_FRAMES),
    "--lr_backbone", str(LR_BACKBONE),
    "--lr_unet", str(LR_UNET),
    "--ema_decay", str(EMA_DECAY),
    "--synth_count", str(SYNTH_COUNT),
    "--static_aug_prob", str(STATIC_AUG_PROB),
    "--w_ghost", str(W_GHOST),
    "--w_temp", str(W_TEMP),
    "--num_workers", "2",
    "--ram_limit_gb", str(RAM_LIMIT_GB),
]

print("=" * 84)
print(f"🚀 BẮT ĐẦU FINE-TUNING PHA 3 ({NUM_GPUS}x GPU | {TOTAL_EPOCHS} EPOCHS | MULTI-LAYER TPS)")
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

if ret == 42:
    print("⏸️ [RAM WATCHDOG 42]: Bỏ '#' ở Cell 6, chạy Cell 6 để giải phóng RAM, rồi chạy lại.")
elif ret != 0:
    raise RuntimeError(f"❌ DDP Worker kết thúc bất thường với mã {ret}!")
else:
    print("🎉 HOÀN TẤT FINE-TUNING PHA 3!")
'''
    cells.append(_make_code(c5_code))

    c6_code = r'''# ==============================================================================
# 🔄 [CELL 6 - KHẨN CẤP]: RESET KERNEL TRÁNH TRÀN RAM NHƯNG VẪN GIỮ OUTPUT
# ==============================================================================
# import os
# os._exit(0)
'''
    cells.append(_make_code(c6_code))

    c7_code = r'''# ==============================================================================
# 🔬 [CELL 7]: BENCHMARK #1 — TẬP CÁCH LY (198 CLIPS / 412 FRAMES) VỚI MULTI-LAYER TPS
# ==============================================================================
import os
import gc
import torch
import pandas as pd
from IPython.display import display, Image as IPyImage
from seq_p2_v33_ghost_purge_worker import TPSInbetweenOriginal, load_tps_weights
from seq_dual_benchmark_suite import run_benchmark_isolated_sequences

BENCH1_OUT = os.path.join(BENCH_OUT_DIR, "bench1_isolated_sequences")
best_ckpt_path = os.path.join(OUTPUT_CKPT_DIR, "model_tps_finetuned_best.pt")
last_ckpt_path = os.path.join(OUTPUT_CKPT_DIR, "model_tps_finetuned_last.pt")

if not BENCH_READY:
    print("[⚠️ BỎ QUA CELL 7]: Tập Benchmark #1 chưa có trong Input.")
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    models_dict = {}
    m1 = TPSInbetweenOriginal().to(device)
    load_tps_weights(m1, M1_CHECKPOINT_PATH, device=device, prefer_ema=False)
    models_dict['M1_Original'] = m1.eval()
    if INIT_CHECKPOINT_PATH and os.path.exists(INIT_CHECKPOINT_PATH) and os.path.abspath(INIT_CHECKPOINT_PATH) != os.path.abspath(M1_CHECKPOINT_PATH):
        m_init = TPSInbetweenOriginal().to(device)
        load_tps_weights(m_init, INIT_CHECKPOINT_PATH, device=device, prefer_ema=True)
        models_dict['Pha2_Blended'] = m_init.eval()
    if os.path.exists(best_ckpt_path):
        m3_b = TPSInbetweenOriginal().to(device)
        load_tps_weights(m3_b, best_ckpt_path, device=device, prefer_ema=True)
        models_dict['M3_Best'] = m3_b.eval()
    if os.path.exists(last_ckpt_path):
        m3_l = TPSInbetweenOriginal().to(device)
        load_tps_weights(m3_l, last_ckpt_path, device=device, prefer_ema=True)
        models_dict['M3_Last'] = m3_l.eval()

    run_benchmark_isolated_sequences(
        BENCH_ROOT_PATH, models_dict, device, BENCH1_OUT, downscale_factor=2, save_visuals=True, use_multilayer=True
    )
    del models_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
'''
    cells.append(_make_code(c7_code))

    c8_code = r'''# ==============================================================================
# 🎬 [CELL 8]: BENCHMARK #2 (24 FRAMES) & BENCHMARK #3 (DEFENSE GEOMETRY SUITE)
# ==============================================================================
import os
import gc
import torch
from IPython.display import display, Image as IPyImage
from seq_p2_v33_ghost_purge_worker import TPSInbetweenOriginal, load_tps_weights
from seq_dual_benchmark_suite import run_benchmark_continuous_lineart_test, run_benchmark_defense_geometry_suite

BENCH2_OUT = os.path.join(BENCH_OUT_DIR, "bench2_continuous_lineart_test")
BENCH3_OUT = os.path.join(BENCH_OUT_DIR, "bench3_defense_geometry")
best_ckpt_path = os.path.join(OUTPUT_CKPT_DIR, "model_tps_finetuned_best.pt")
last_ckpt_path = os.path.join(OUTPUT_CKPT_DIR, "model_tps_finetuned_last.pt")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
models_dict = {}
m1 = TPSInbetweenOriginal().to(device)
load_tps_weights(m1, M1_CHECKPOINT_PATH, device=device, prefer_ema=False)
models_dict['M1_Original'] = m1.eval()
if INIT_CHECKPOINT_PATH and os.path.exists(INIT_CHECKPOINT_PATH) and os.path.abspath(INIT_CHECKPOINT_PATH) != os.path.abspath(M1_CHECKPOINT_PATH):
    m_init = TPSInbetweenOriginal().to(device)
    load_tps_weights(m_init, INIT_CHECKPOINT_PATH, device=device, prefer_ema=True)
    models_dict['Pha2_Blended'] = m_init.eval()
if os.path.exists(best_ckpt_path):
    m3_b = TPSInbetweenOriginal().to(device)
    load_tps_weights(m3_b, best_ckpt_path, device=device, prefer_ema=True)
    models_dict['M3_Best'] = m3_b.eval()
if os.path.exists(last_ckpt_path):
    m3_l = TPSInbetweenOriginal().to(device)
    load_tps_weights(m3_l, last_ckpt_path, device=device, prefer_ema=True)
    models_dict['M3_Last'] = m3_l.eval()

if LINEART_TEST_READY:
    df_sum2 = run_benchmark_continuous_lineart_test(
        LINEART_TEST_PATH, models_dict, device, BENCH2_OUT,
        gluestick_npz=GLUESTICK_TEST_NPZ, downscale_factor=2, use_multilayer=True
    )
    if df_sum2 is not None:
        display(df_sum2.round(4))

df_sum3 = run_benchmark_defense_geometry_suite(models_dict, device, BENCH3_OUT, downscale_factor=2)
if df_sum3 is not None:
    display(df_sum3.round(4))
    gp = os.path.join(BENCH3_OUT, "ShrinkingCircle_StaticGround_grid.png")
    if os.path.exists(gp):
        display(IPyImage(filename=gp))

del models_dict
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
'''
    cells.append(_make_code(c8_code))

    c9_code = r'''# ==============================================================================
# 📈 [CELL 9]: DASHBOARD HỘI TỤ TRAINING PHA 3
# ==============================================================================
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

hist_csv = os.path.join(WORKING_DIR, "training_history_finetune.csv")
if os.path.exists(hist_csv):
    df_h = pd.read_csv(hist_csv)
    display(df_h.round(4))
else:
    print("[*] Chưa có training_history_finetune.csv.")
'''
    cells.append(_make_code(c9_code))

    c10_code = r'''# ==============================================================================
# 📦 [CELL 10]: ĐÓNG GÓI CHECKPOINT PHA 3 & KẾT QUẢ BENCHMARK (.ZIP)
# ==============================================================================
import os
import zipfile
import shutil
import torch

RELEASE_ZIP = os.path.join(WORKING_DIR, "tps_sequence_phase3_release.zip")
STRIP_TEMP = os.path.join("/tmp" if os.path.exists("/tmp") else WORKING_DIR, "_strip_ckpt_tmp")
os.makedirs(STRIP_TEMP, exist_ok=True)

pack_items = []
for src_p, zip_name in [
    (os.path.join(OUTPUT_CKPT_DIR, "model_tps_finetuned_best.pt"), "checkpoints/model_tps_finetuned_best.pt"),
    (os.path.join(OUTPUT_CKPT_DIR, "model_tps_finetuned_last.pt"), "checkpoints/model_tps_finetuned_last.pt"),
]:
    if os.path.exists(src_p):
        raw = torch.load(src_p, map_location='cpu', weights_only=False)
        clean_sd = raw.get('model_state')
        if zip_name.endswith("last.pt") and raw.get('ema_state') is not None:
            es = raw['ema_state']
            clean_sd = es['shadow'] if isinstance(es, dict) and 'shadow' in es else clean_sd
        lite_obj = {'epoch': raw.get('epoch'), 'best_score': raw.get('best_score'), 'model_state': clean_sd, 'history': raw.get('history', [])}
        tmp_out = os.path.join(STRIP_TEMP, os.path.basename(zip_name))
        torch.save(lite_obj, tmp_out)
        pack_items.append((tmp_out, zip_name))

for fname in ["training_history_finetune.csv", "val_baseline_m0_m1.json"]:
    fp = os.path.join(WORKING_DIR, fname)
    if os.path.exists(fp):
        pack_items.append((fp, fname))

if os.path.exists(BENCH_OUT_DIR):
    for root, _, files in os.walk(BENCH_OUT_DIR):
        for f in files:
            full_p = os.path.join(root, f)
            pack_items.append((full_p, os.path.relpath(full_p, WORKING_DIR)))

with zipfile.ZipFile(RELEASE_ZIP, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for src_f, arc_name in pack_items:
        zf.write(src_f, arcname=arc_name)

shutil.rmtree(STRIP_TEMP, ignore_errors=True)
print(f"📦 ĐÃ ĐÓNG GÓI THÀNH CÔNG {len(pack_items)} TỆP VÀO: {RELEASE_ZIP} ({os.path.getsize(RELEASE_ZIP)/(1024**2):.2f} MB)")
'''
    cells.append(_make_code(c10_code))

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

    out_nb_path = os.path.join(THIS_DIR, "train_and_benchmark_tps_sequence_phase3_kaggle.ipynb")
    with open(out_nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"[✓] Đã tạo Notebook Kaggle Pha 3: {out_nb_path} ({os.path.getsize(out_nb_path)/(1024**2):.2f} MB)")
    return out_nb_path


if __name__ == "__main__":
    build_notebook()
