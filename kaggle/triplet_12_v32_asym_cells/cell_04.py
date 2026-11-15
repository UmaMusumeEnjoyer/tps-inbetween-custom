# [CELL 4]: Lớp Dataset SakugaGoldenDataset_v32 (Full Precomputed TPS Cache 0-copy) & DataLoader 540p Siêu Tốc
import os
import sys
import gc
import random
import time
import shutil
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data.dataloader import default_collate

class TemporalReversalAugmentation_v32:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img0, imggt, img1, t_val,
                 c0_pad, th01_pad, c1_pad, th10_pad):
        if random.random() < self.p:
            img0_out, img1_out = img1, img0
            t_out = float(1.0 - t_val)
            gt_out = imggt
            c0_out, th01_out = c1_pad, th10_pad
            c1_out, th10_out = c0_pad, th01_pad
            return img0_out, gt_out, img1_out, t_out, c0_out, th01_out, c1_out, th10_out, True
        return img0, imggt, img1, float(t_val), c0_pad, th01_pad, c1_pad, th10_pad, False

def compute_v32_curriculum_weights(df):
    weights = np.ones(len(df), dtype=np.float32)
    t_vals = df["t"].values.astype(np.float32)
    disp_vals = df["mean_disp"].values.astype(np.float32) if "mean_disp" in df.columns else np.zeros(len(df), dtype=np.float32)
    occ_scores = df["patch_occ_score"].values.astype(np.float32) if "patch_occ_score" in df.columns else np.zeros(len(df), dtype=np.float32)
    ink_diffs = df["global_ink_diff"].values.astype(np.float32) if "global_ink_diff" in df.columns else np.zeros(len(df), dtype=np.float32)

    is_high_occ = (occ_scores >= 0.08)
    weights[is_high_occ] *= 1.40

    is_ink_diverge = (ink_diffs >= 0.15)
    weights[is_ink_diverge] *= 1.25

    is_boundary_t = ((t_vals >= 0.15) & (t_vals <= 0.30)) | ((t_vals >= 0.70) & (t_vals <= 0.85))
    is_large_disp = (disp_vals >= 7.0)
    weights[is_boundary_t] *= 1.20
    weights[is_large_disp] *= 1.25

    return weights

class SakugaGoldenDataset_v32(Dataset):
    def __init__(self, root_dir, df_split, is_train=True, cache_dir=None):
        self.root_dir = root_dir
        self.is_train = is_train
        self.split_name = 'train' if is_train else 'val'
        self.length = len(df_split)
        self.temporal_aug = TemporalReversalAugmentation_v32(p=0.5) if is_train else None

        # 1. Định vị đường dẫn gốc 1 lần duy nhất (< 0.005s)
        sample_f0 = str(df_split['frame0'].iloc[0]).replace('\\', '/').lstrip('/')
        actual_root = root_dir
        for cr in [root_dir, os.path.join(root_dir, 'anime_tps_finetune_dataset_ready'), os.path.join(root_dir, 'sakuga_golden_dataset_tps_ready_v1_patch3')]:
            if os.path.exists(os.path.join(cr, sample_f0)):
                actual_root = cr
                break

        sample_fm = str(df_split['match_file'].iloc[0]).replace('\\', '/').lstrip('/')
        actual_match_root = actual_root
        for cr in [actual_root, root_dir, os.path.join(root_dir, 'matches')]:
            if os.path.exists(os.path.join(cr, sample_fm)):
                actual_match_root = cr
                break
            if os.path.exists(os.path.join(cr, 'matches', sample_fm)):
                actual_match_root = os.path.join(cr, 'matches')
                break

        f0_list = [os.path.join(actual_root, str(f).replace('\\', '/').lstrip('/')) for f in df_split['frame0']]
        fgt_list = [os.path.join(actual_root, str(f).replace('\\', '/').lstrip('/')) for f in df_split['frame_gt']]
        f1_list = [os.path.join(actual_root, str(f).replace('\\', '/').lstrip('/')) for f in df_split['frame1']]
        fm_list = [os.path.join(actual_match_root, str(f).replace('\\', '/').lstrip('/')) for f in df_split['match_file']]

        self.p0_arr = np.array([f.encode('utf-8') for f in f0_list], dtype='S256')
        self.pgt_arr = np.array([f.encode('utf-8') for f in fgt_list], dtype='S256')
        self.p1_arr = np.array([f.encode('utf-8') for f in f1_list], dtype='S256')
        self.pmatch_arr = np.array([f.encode('utf-8') for f in fm_list], dtype='S256')
        self.t_arr = df_split['t'].to_numpy(dtype=np.float32, copy=True)
        self.clips_arr = np.array([str(c).encode('utf-8') for c in df_split['clip_id']], dtype='S64')

        # 2. Quản lý Cache Keypoints .npy uncompressed (mmap 0-copy)
        cd = cache_dir or CACHE_DIR
        os.makedirs(cd, exist_ok=True)
        kps_npy = os.path.join(cd, f'kps_data_{self.split_name}_v32.npy')
        cnt_npy = os.path.join(cd, f'kps_count_{self.split_name}_v32.npy')

        # Tái sử dụng cache v31 nếu có
        if not os.path.exists(kps_npy):
            old_kps = os.path.join(cd, f'kps_data_{self.split_name}_v31.npy')
            old_cnt = os.path.join(cd, f'kps_count_{self.split_name}_v31.npy')
            if os.path.exists(old_kps) and os.path.exists(old_cnt):
                try:
                    shutil.copyfile(old_kps, kps_npy)
                    shutil.copyfile(old_cnt, cnt_npy)
                except Exception:
                    pass

        def _fast_read_match_npy(path, max_cache=256):
            try:
                with open(path, 'rb') as f:
                    raw = f.read()
                data = np.frombuffer(raw[128:], dtype=np.float32)
                pts = data.reshape(-1, 2, 2)
                cnt = min(len(pts), max_cache)
                return pts[:cnt], cnt
            except Exception:
                try:
                    pts = np.load(path, allow_pickle=False)
                    cnt = min(len(pts), max_cache)
                    return pts[:cnt], cnt
                except Exception:
                    return None, 0

        if os.path.exists(kps_npy) and os.path.exists(cnt_npy):
            self.kps_data = np.load(kps_npy, mmap_mode='r')
            self.kps_count = np.load(cnt_npy, mmap_mode='r')
        else:
            MAX_KPS_CACHE = 256
            kps_buf = np.zeros((self.length, 2, MAX_KPS_CACHE, 2), dtype=np.float32)
            cnt_buf = np.zeros(self.length, dtype=np.int32)

            def _load_single_kps(i):
                m_path = self.pmatch_arr[i].decode('utf-8')
                pts, c = _fast_read_match_npy(m_path, max_cache=MAX_KPS_CACHE)
                if pts is not None and c > 0:
                    kps_buf[i, :, :c, :] = pts[:, :c, :2]
                    cnt_buf[i] = c

            with ThreadPoolExecutor(max_workers=32) as ex:
                list(ex.map(_load_single_kps, range(self.length)))

            np.save(kps_npy, kps_buf)
            np.save(cnt_npy, cnt_buf)
            self.kps_data = np.load(kps_npy, mmap_mode='r')
            self.kps_count = np.load(cnt_npy, mmap_mode='r')
            del kps_buf, cnt_buf
            gc.collect()

        # 3. [TỐI ƯU HÓA ĐỘT PHÁ V3.2]: Precompute TPS Đầy Đủ 100% Cho Cả Train và Val
        c0_npy = os.path.join(cd, f'{self.split_name}_c0_v32.npy')
        th01_npy = os.path.join(cd, f'{self.split_name}_th01_v32.npy')
        c1_npy = os.path.join(cd, f'{self.split_name}_c1_v32.npy')
        th10_npy = os.path.join(cd, f'{self.split_name}_th10_v32.npy')
        valid_npy = os.path.join(cd, f'{self.split_name}_valid_v32.npy')

        # Tái sử dụng v31 precomputed npy nếu có sẵn
        if not os.path.exists(c0_npy):
            for suf in ['c0', 'th01', 'c1', 'th10', 'valid']:
                src = os.path.join(cd, f'{self.split_name}_{suf}_v31.npy')
                dst = os.path.join(cd, f'{self.split_name}_{suf}_v32.npy')
                if os.path.exists(src) and not os.path.exists(dst):
                    try: shutil.copyfile(src, dst)
                    except Exception: pass

        all_tps_exist = all(os.path.exists(p) for p in [c0_npy, th01_npy, c1_npy, th10_npy, valid_npy])
        if all_tps_exist:
            self.tps_c0 = np.load(c0_npy, mmap_mode='r')
            self.tps_th01 = np.load(th01_npy, mmap_mode='r')
            self.tps_c1 = np.load(c1_npy, mmap_mode='r')
            self.tps_th10 = np.load(th10_npy, mmap_mode='r')
            self.tps_valid = np.load(valid_npy, mmap_mode='r')
            print(f"[✓] Nạp sẵn {self.length} mẫu TPS {self.split_name} mmap 0-copy (< 0.01s).")
        else:
            print(f"[*] Precomputing TPS cache đa luồng cho tập {self.split_name} ({self.length} mẫu)...", flush=True)
            c0_b, th01_b, c1_b, th10_b, valid_b = solve_tps_batch_all(
                self.kps_data, self.kps_count, w=960, h=540, pad_w=960, pad_h=544, max_kps=80, lambd=0.5
            )
            np.save(c0_npy, c0_b)
            np.save(th01_npy, th01_b)
            np.save(c1_npy, c1_b)
            np.save(th10_npy, th10_b)
            np.save(valid_npy, valid_b)
            self.tps_c0 = np.load(c0_npy, mmap_mode='r')
            self.tps_th01 = np.load(th01_npy, mmap_mode='r')
            self.tps_c1 = np.load(c1_npy, mmap_mode='r')
            self.tps_th10 = np.load(th10_npy, mmap_mode='r')
            self.tps_valid = np.load(valid_npy, mmap_mode='r')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        clip_name = int(idx) if self.is_train else self.clips_arr[idx].decode('utf-8')
        im0 = imread_gray(self.p0_arr[idx].decode('utf-8'))
        imgt = imread_gray(self.pgt_arr[idx].decode('utf-8'))
        im1 = imread_gray(self.p1_arr[idx].decode('utf-8'))

        if im0 is None: im0 = np.full((540, 960), 255, dtype=np.uint8)
        if imgt is None: imgt = np.full((540, 960), 255, dtype=np.uint8)
        if im1 is None: im1 = np.full((540, 960), 255, dtype=np.uint8)

        h, w = imgt.shape[:2]
        pad_h = ((h + 31) // 32) * 32
        pad_w = ((w + 31) // 32) * 32

        buf = np.full((3, pad_h, pad_w), 255, dtype=np.uint8)
        buf[0, :h, :w] = im0
        buf[1, :h, :w] = imgt
        buf[2, :h, :w] = im1
        im0_pad, imgt_pad, im1_pad = buf[0], buf[1], buf[2]

        # Nạp ma trận TPS 0-copy từ Precomputed Cache (0 ms computation time!)
        c0_pad = self.tps_c0[idx].copy()
        th01_pad = self.tps_th01[idx].copy()
        c1_pad = self.tps_c1[idx].copy()
        th10_pad = self.tps_th10[idx].copy()
        valid_flag = self.tps_valid[idx].copy()
        t_scalar = float(self.t_arr[idx])

        if self.is_train and self.temporal_aug is not None:
            im0_pad, imgt_pad, im1_pad, t_scalar, c0_pad, th01_pad, c1_pad, th10_pad, _ = self.temporal_aug(
                im0_pad, imgt_pad, im1_pad, t_scalar, c0_pad, th01_pad, c1_pad, th10_pad
            )

        return (
            torch.from_numpy(np.ascontiguousarray(im0_pad[None, :, :], dtype=np.uint8)),
            torch.from_numpy(np.ascontiguousarray(imgt_pad[None, :, :], dtype=np.uint8)),
            torch.from_numpy(np.ascontiguousarray(im1_pad[None, :, :], dtype=np.uint8)),
            torch.tensor(t_scalar, dtype=torch.float32),
            torch.from_numpy(c0_pad),
            torch.from_numpy(th01_pad),
            torch.from_numpy(c1_pad),
            torch.from_numpy(th10_pad),
            torch.from_numpy(valid_flag),
            clip_name,
            int(h),
            int(w)
        )

custom_collate_fn_v32 = default_collate

print("=" * 80)
print("[*] KHỞI TẠO DATASET VÀ DATALOADER 540P TOÀN PHẦN V3.2 (FULL PRECOMPUTED 0-COPY)...")
train_dataset = SakugaGoldenDataset_v32(DATASET_ROOT, train_df, is_train=True, cache_dir=CACHE_DIR)
val_dataset = SakugaGoldenDataset_v32(DATASET_ROOT, val_df, is_train=False, cache_dir=CACHE_DIR)

train_weights = compute_v32_curriculum_weights(train_df)
train_sampler = WeightedRandomSampler(
    weights=torch.as_tensor(train_weights, dtype=torch.double),
    num_samples=len(train_dataset),
    replacement=True
)

BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 1
VAL_BATCH_SIZE = 32

def _worker_init_fn(worker_id):
    torch.set_num_threads(1)
    cv2.setNumThreads(0)
    random.seed(42 + worker_id)
    np.random.seed(42 + worker_id)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    num_workers=2,
    pin_memory=False,
    drop_last=True,
    persistent_workers=False,
    worker_init_fn=_worker_init_fn,
    collate_fn=custom_collate_fn_v32
)

val_loader = DataLoader(
    val_dataset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=False,
    drop_last=False,
    persistent_workers=False,
    worker_init_fn=_worker_init_fn,
    collate_fn=custom_collate_fn_v32
)

print(f"[✓] DataLoaders V3.2 sẵn sàng: Train={len(train_loader)} batches | Val={len(val_loader)} batches")

# Lưu snapshot metadata bundle để các worker DDP nạp độc lập trong 0.05s
bundle_path = os.path.join(WORKING_DIR, 'dataset_bundle_v32.pt')
torch.save({
    'dataset_root': DATASET_ROOT,
    'cache_dir': CACHE_DIR,
    'train_df': train_df,
    'val_df': val_df,
    'train_weights': train_weights
}, bundle_path)
print(f"[✓] Đã lưu snapshot dataset bundle DDP: {bundle_path}")
print("=" * 80)
sys.stdout.flush()
