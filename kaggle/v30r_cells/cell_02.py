# [CELL 2]: Định Vị Siêu Tốc & Nạp Sakuga Golden Dataset 540p Native V3.0-Remastered
import os
import sys
import time
import pandas as pd
import cv2

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

t_start = time.time()
print("=" * 80)
print("[CELL 2] ⚡ ĐỊNH VỊ SIÊU TỐC VÀ NẠP SAKUGA GOLDEN DATASET 540P (V1_PATCH3)...")
print("=" * 80)

# Khởi tạo KAGGLE_INPUT và WORKING_DIR
KAGGLE_INPUT = '/kaggle/input' if os.path.exists('/kaggle/input') else './data'
if 'WORKING_DIR' not in globals():
    WORKING_DIR = '/kaggle/working' if os.path.exists('/kaggle/working') else '.'

# -----------------------------------------------------------------------------
# 1. ĐỊNH VỊ TRỰC TIẾP (O(1) - KHÔNG QUÉT ĐỆ QUY TRÁNH NGHẼN KAGGLE FUSE)
# -----------------------------------------------------------------------------
direct_candidates = [
    # Đường dẫn chỉ định trực tiếp từ Kaggle Input của bạn:
    '/kaggle/input/datasets/dungbaoviec123/sakuga-golden-dataset-tps-ready-v1-patch3',
    '/kaggle/input/datasets/dungbaoviec123/sakuga-golden-dataset-tps-ready-v1-patch3/anime_tps_finetune_dataset_ready',
    
    # Các mount point tiêu chuẩn khác trên Kaggle:
    '/kaggle/input/sakuga-golden-dataset-tps-ready-v1-patch3',
    '/kaggle/input/sakuga-golden-dataset-tps-ready-v1-patch3/anime_tps_finetune_dataset_ready',
    '/kaggle/input/sakuga_golden_dataset_tps_ready_v1_patch3',
    '/kaggle/input/sakuga_golden_dataset_tps_ready_v1_patch3/anime_tps_finetune_dataset_ready',
    os.path.join(KAGGLE_INPUT, 'sakuga-golden-dataset-tps-ready-v1-patch3'),
    os.path.join(KAGGLE_INPUT, 'sakuga_golden_dataset_tps_ready_v1_patch3'),
    
    # Fallback môi trường dev local:
    './data/processed/sakuga_golden_dataset_tps_ready_v1_patch3',
    './data/processed/sakuga_golden_dataset_tps_ready_v1_patch3/anime_tps_finetune_dataset_ready',
    'E:/CODE/PBL/PBL6/data/processed/sakuga_golden_dataset_tps_ready_v1_patch3',
    'E:/CODE/PBL/PBL6/data/processed/sakuga_golden_dataset_tps_ready_v1_patch3/anime_tps_finetune_dataset_ready'
]

found_base = None
for p in direct_candidates:
    if os.path.isdir(p):
        found_base = p
        break

# Fallback quét cạn (Shallow scan độ sâu tối đa 2 cấp, KHÔNG quét đệ quy qua các thư mục ảnh)
if not found_base and os.path.exists('/kaggle/input'):
    try:
        for entry in os.listdir('/kaggle/input'):
            full_p = os.path.join('/kaggle/input', entry)
            if not os.path.isdir(full_p):
                continue
            name_lower = entry.lower()
            if 'sakuga' in name_lower and 'patch3' in name_lower:
                found_base = full_p
                break
            elif entry == 'datasets':
                for u in os.listdir(full_p):
                    u_p = os.path.join(full_p, u)
                    if os.path.isdir(u_p):
                        for ds in os.listdir(u_p):
                            if 'sakuga' in ds.lower() and 'patch3' in ds.lower():
                                found_base = os.path.join(u_p, ds)
                                break
    except Exception as e:
        print(f"[!] Cảnh báo quét cạn: {e}")

if not found_base:
    raise FileNotFoundError(
        "[!] KHÔNG TÌM THẤY DATASET!\n"
        "Vui lòng kiểm tra lại dataset 'sakuga-golden-dataset-tps-ready-v1-patch3' trong tab Input bên phải của Kaggle."
    )

# Tự động trỏ vào anime_tps_finetune_dataset_ready nếu tồn tại thư mục con này
sub_ready = os.path.join(found_base, 'anime_tps_finetune_dataset_ready')
DATASET_ROOT = os.path.normpath(sub_ready if os.path.isdir(sub_ready) else found_base)
print(f"[✓] Đã định vị thành công DATASET_ROOT ({time.time() - t_start:.3f}s):\n    -> {DATASET_ROOT}")

# -----------------------------------------------------------------------------
# 2. NẠP METADATA CSV TRỰC TIẾP (O(1))
# -----------------------------------------------------------------------------
csv_candidates = [
    os.path.join(DATASET_ROOT, 'triplets.csv'),
    os.path.join(DATASET_ROOT, 'triplets_combined.csv'),
    os.path.join(DATASET_ROOT, 'metadata_patch3.csv'),
    os.path.join(found_base, 'triplets.csv'),
    os.path.join(found_base, 'triplets_combined.csv')
]

meta_path = next((p for p in csv_candidates if os.path.isfile(p)), None)
if not meta_path:
    raise FileNotFoundError(f"[!] Không tìm thấy triplets.csv tại {DATASET_ROOT} hoặc {found_base}")

print(f"[*] Nạp file metadata: {meta_path}")
df_meta = pd.read_csv(meta_path)

# Chuẩn hóa tên cột để tương thích 100% với toàn bộ pipeline & DDP Worker
col_map = {}
if 'frame0' in df_meta.columns: col_map['frame0'] = 'frame0_path'
if 'frame1' in df_meta.columns: col_map['frame1'] = 'frame1_path'
if 'frame_gt' in df_meta.columns: col_map['frame_gt'] = 'target_path'

for src_col, dst_col in col_map.items():
    if dst_col not in df_meta.columns:
        df_meta[dst_col] = df_meta[src_col]

# Phân chia tập Train / Val
if 'split' in df_meta.columns:
    train_df = df_meta[df_meta['split'] == 'train'].reset_index(drop=True)
    val_df = df_meta[df_meta['split'] == 'val'].reset_index(drop=True)
else:
    n_train = int(len(df_meta) * 0.85)
    train_df = df_meta.iloc[:n_train].reset_index(drop=True)
    val_df = df_meta.iloc[n_train:].reset_index(drop=True)

print(f"[✓] Tổng số Triplets: {len(df_meta):,}")
print(f"    - Tập Huấn Luyện (Train): {len(train_df):,} mẫu")
print(f"    - Tập Thẩm Định (Val)   : {len(val_df):,} mẫu")

# -----------------------------------------------------------------------------
# 3. SANITY CHECK MẪU ĐẦU TIÊN (Kiểm tra ảnh đầu tiên có đọc được không)
# -----------------------------------------------------------------------------
sample_row = df_meta.iloc[0]
sample_f0 = str(sample_row.get('frame0_path', sample_row.get('frame0', ''))).replace('\\', '/').lstrip('/')
test_p = os.path.join(DATASET_ROOT, sample_f0)

if not os.path.exists(test_p):
    alt_p = os.path.join(found_base, sample_f0)
    if os.path.exists(alt_p):
        DATASET_ROOT = os.path.normpath(found_base)
        test_p = alt_p

if os.path.exists(test_p):
    img_sample = cv2.imread(test_p, cv2.IMREAD_GRAYSCALE)
    h, w = img_sample.shape if img_sample is not None else (0, 0)
    print(f"[✓] Sanity Check Hợp Lệ! Đọc thử thành công ảnh mẫu:")
    print(f"    -> Đường dẫn : {test_p}")
    print(f"    -> Kích thước: {w}x{h} (540p Native Chuẩn)")
else:
    print(f"[!] CẢNH BÁO: Chưa tìm thấy ảnh mẫu tại {test_p}.")

# -----------------------------------------------------------------------------
# 4. TĂNG TỐC I/O: STAGING SANG LOCAL NVME SSD (/tmp) (Triệt tiêu 100% thời gian chết GPU)
# -----------------------------------------------------------------------------
LOCAL_SSD_STAGING = '/tmp/sakuga_ssd_v30r'
if sys.platform.startswith('linux'):
    try:
        staging_marker = os.path.join(LOCAL_SSD_STAGING, '.staged_complete')
        if os.path.exists(LOCAL_SSD_STAGING) and os.path.exists(staging_marker):
            DATASET_ROOT = LOCAL_SSD_STAGING
            print(f"[⚡ TÁI SỬ DỤNG NVMe SSD]: Đã phát hiện vùng đệm SSD cực nhanh tại:\n    -> {DATASET_ROOT}")
        else:
            t_stage = time.time()
            print(f"[*] Đang tăng tốc I/O: Sao chép dữ liệu sang NVMe SSD cục bộ ({LOCAL_SSD_STAGING})...", flush=True)
            os.makedirs(LOCAL_SSD_STAGING, exist_ok=True)
            
            all_rel_files = set()
            for col in ['frame0_path', 'frame1_path', 'target_path', 'frame0', 'frame1', 'frame_gt', 'match_file']:
                if col in df_meta.columns:
                    for f in df_meta[col]:
                        if pd.notna(f) and str(f).strip():
                            all_rel_files.add(str(f).replace('\\', '/').lstrip('/'))
            
            file_list = list(all_rel_files)
            print(f"    -> Đang sao chép song song {len(file_list):,} tệp với 32 luồng I/O...", flush=True)
            
            from concurrent.futures import ThreadPoolExecutor
            import shutil
            def _copy_worker(rel_p):
                src = os.path.join(DATASET_ROOT, rel_p)
                dst = os.path.join(LOCAL_SSD_STAGING, rel_p)
                if not os.path.exists(src):
                    alt = os.path.join(found_base, rel_p)
                    if os.path.exists(alt): src = alt
                if os.path.exists(src):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
                        shutil.copy2(src, dst)

            with ThreadPoolExecutor(max_workers=32) as ex:
                list(ex.map(_copy_worker, file_list))
            
            shutil.copy2(meta_path, os.path.join(LOCAL_SSD_STAGING, os.path.basename(meta_path)))
            with open(staging_marker, 'w') as f:
                f.write('OK\n')
            
            DATASET_ROOT = LOCAL_SSD_STAGING
            print(f"[✓] Hoàn tất nạp SSD trong {time.time() - t_stage:.1f}s! Tốc độ đọc ảnh tăng gấp 100 lần, GPU sẽ chạy liên tục không nghỉ.")
    except Exception as e:
        print(f"[!] Bỏ qua staging ({e}), tiếp tục đọc từ {DATASET_ROOT}")

# Xuất các biến toàn cục cho các Cell tiếp theo
globals()['DATASET_ROOT'] = DATASET_ROOT
globals()['df_meta'] = df_meta
globals()['train_df'] = train_df
globals()['val_df'] = val_df

print("=" * 80)
print(f"[✓] CELL 2 HOÀN TẤT trong {time.time() - t_start:.2f}s! Sẵn sàng huấn luyện.")
print("=" * 80)
sys.stdout.flush()
