# [CELL 2]: Tự Động Định Vị, Giải Nén Siêu Tốc & Thẩm Định Toàn Vẹn Golden Dataset 540p (Patch 3 Ready)
import os
import sys
import gc
import time
import zipfile
import tarfile
import subprocess
import glob
import pandas as pd

KAGGLE_INPUT = '/kaggle/input' if os.path.exists('/kaggle/input') else './data'

TMP_DIR = os.path.join(WORKING_DIR, '_temp_dataset')
EXTRACT_DIR = os.path.join(TMP_DIR, 'sakuga_golden_dataset_tps_ready')
DATASET_EXTRACTED_TO_TEMP = False

DATASET_ROOT = None
SKIP_DATA_SUBDIRS = {'train', 'val', 'test', 'matches', 'frames', 'clips', 'crops', 'images', 'npy', 'wandb', '.git', '__pycache__', 'cache'}

search_locations = [KAGGLE_INPUT, EXTRACT_DIR, TMP_DIR, WORKING_DIR, './data/processed', '../data/processed', '.']

t_find_start = time.time()
print("=" * 80)
print("[CELL 2] 🔍 [Bước 1/3] QUÉT ĐỊNH VỊ SIÊU TỐC GOLDEN DATASET 540P (V1_PATCH3)...")
print("=" * 80)

# Ưu tiên kiểm tra trực tiếp các đường dẫn cố định chứa patch3
direct_dataset_candidates = [
    os.path.join(KAGGLE_INPUT, 'sakuga_golden_dataset_tps_ready_v1_patch3', 'anime_tps_finetune_dataset_ready'),
    os.path.join(KAGGLE_INPUT, 'sakuga_golden_dataset_tps_ready_v1_patch3'),
    os.path.join(KAGGLE_INPUT, 'sakuga-golden-dataset-tps-ready-v1-patch3', 'anime_tps_finetune_dataset_ready'),
    os.path.join(KAGGLE_INPUT, 'sakuga-golden-dataset-tps-ready-v1-patch3'),
    './data/processed/sakuga_golden_dataset_tps_ready_v1_patch3/anime_tps_finetune_dataset_ready',
    './data/processed/sakuga_golden_dataset_tps_ready_v1_patch3',
    EXTRACT_DIR,
    os.path.join(TMP_DIR, 'anime_tps_finetune_dataset_ready'),
    './data/processed/anime_tps_finetune_dataset_ready',
    './data/processed'
]

shallow_patterns = [
    os.path.join(KAGGLE_INPUT, 'datasets', '*', '*', 'anime_tps_finetune_dataset_ready'),
    os.path.join(KAGGLE_INPUT, '*', 'anime_tps_finetune_dataset_ready'),
    os.path.join(KAGGLE_INPUT, 'datasets', '*', '*'),
    os.path.join(KAGGLE_INPUT, '*')
]
for pat in shallow_patterns:
    for m_dir in glob.glob(pat, recursive=False):
        if m_dir not in direct_dataset_candidates:
            direct_dataset_candidates.append(m_dir)

for cand in direct_dataset_candidates:
    if os.path.exists(cand) and os.path.isdir(cand):
        if os.path.exists(os.path.join(cand, 'triplets.csv')) or os.path.exists(os.path.join(cand, 'triplets_combined.csv')):
            DATASET_ROOT = cand
            print(f"    ⚡ Định vị trực tiếp thành công tại: {DATASET_ROOT}")
            break

if DATASET_ROOT is None:
    print("    [*] Quét mở rộng tìm thư mục metadata...")
    for base_dir in search_locations:
        if not os.path.exists(base_dir):
            continue
        for root, dirs, files in os.walk(base_dir):
            dirs[:] = [d for d in dirs if d.lower() not in SKIP_DATA_SUBDIRS and not d.startswith('.') and not any(k in d.lower() for k in ['crop', 'clip', 'frame'])]
            if 'triplets.csv' in files or 'triplets_combined.csv' in files:
                if DATASET_ROOT is None or "patch3" in root.lower() or "v1_patch3" in root.lower():
                    DATASET_ROOT = root
                    if "patch3" in root.lower() or "v1_patch3" in root.lower():
                        break
        if DATASET_ROOT is not None and ("patch3" in DATASET_ROOT.lower() or "v1_patch3" in DATASET_ROOT.lower()):
            break

# Bước 2: Quét archive (.zip / .tar.gz) nếu chưa có thư mục giải nén
if DATASET_ROOT is None:
    print("[*] Quét tìm file nén (.zip / .tar.gz) chứa dataset...")
    archive_path = None
    archive_type = None
    valid_keywords = ['patch3', 'patch_3', 'v1_patch3', 'sakuga', 'golden', 'tps']

    for base_dir in search_locations:
        if not os.path.exists(base_dir):
            continue
        for root, dirs, files in os.walk(base_dir):
            dirs[:] = [d for d in dirs if d.lower() not in SKIP_DATA_SUBDIRS and not d.startswith('.') and not any(k in d.lower() for k in ['crop', 'clip', 'frame'])]
            for f in files:
                f_lower = f.lower()
                is_match = any(k in f_lower for k in valid_keywords)
                if is_match and (f_lower.endswith('.tar.gz') or f_lower.endswith('.tgz')):
                    archive_path = os.path.join(root, f)
                    archive_type = 'tar'
                    if "patch3" in f_lower: break
                elif is_match and f_lower.endswith('.zip'):
                    archive_path = os.path.join(root, f)
                    archive_type = 'zip'
                    if "patch3" in f_lower: break
            if archive_path is not None and "patch3" in archive_path.lower(): break
        if archive_path is not None and "patch3" in archive_path.lower(): break

    if archive_path is not None and os.path.exists(archive_path):
        print(f"[✓] Tìm thấy archive: {archive_path}. Tiến hành giải nén vào {EXTRACT_DIR}...")
        os.makedirs(EXTRACT_DIR, exist_ok=True)
        DATASET_EXTRACTED_TO_TEMP = True
        if archive_type == 'tar':
            subprocess.run(['tar', '-xf', archive_path, '-C', EXTRACT_DIR], check=True)
        else:
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(EXTRACT_DIR)
        DATASET_ROOT = EXTRACT_DIR

if DATASET_ROOT is None:
    raise FileNotFoundError("Không thể tìm thấy Sakuga Golden Dataset 540p! Vui lòng kiểm tra Kaggle Input.")

meta_csv = os.path.join(DATASET_ROOT, 'triplets.csv')
if not os.path.exists(meta_csv):
    meta_csv = os.path.join(DATASET_ROOT, 'triplets_combined.csv')

df_meta = pd.read_csv(meta_csv)
print(f"[✓] Đã nạp metadata: {meta_csv} ({len(df_meta):,} mẫu)")

if 'split' in df_meta.columns:
    train_df = df_meta[df_meta['split'] == 'train'].reset_index(drop=True)
    val_df = df_meta[df_meta['split'] == 'val'].reset_index(drop=True)
else:
    n_train = int(len(df_meta) * 0.85)
    train_df = df_meta.iloc[:n_train].reset_index(drop=True)
    val_df = df_meta.iloc[n_train:].reset_index(drop=True)

print(f"    - Mẫu Huấn Luyện (Train): {len(train_df):,}")
print(f"    - Mẫu Thẩm Định (Val)   : {len(val_df):,}")
drop_file_page_cache(meta_csv)
print("=" * 80)
sys.stdout.flush()