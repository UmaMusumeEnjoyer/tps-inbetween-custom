# [CELL 4]: Lớp Dataset SakugaGoldenDataset_v3 (.npy mmap 0-copy) & DataLoader Huấn Luyện 540p Toàn Phần
import os
import sys
import gc
import random
import time
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

class TemporalReversalAugmentation:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img0, imggt, img1, t_val, kps=None,
                 c0_pad=None, th01_pad=None, c1_pad=None, th10_pad=None):
        if random.random() < self.p:
            img0_out, img1_out = img1, img0
            t_out = float(1.0 - t_val)
            gt_out = imggt
            kps_out = np.stack([kps[1], kps[0]], axis=0) if (kps is not None and kps.shape[0] == 2) else kps
            c0_out, th01_out = c1_pad, th10_pad
            c1_out, th10_out = c0_pad, th01_pad
            return img0_out, gt_out, img1_out, t_out, kps_out, c0_out, th01_out, c1_out, th10_out, True
        return img0, imggt, img1, float(t_val), kps, c0_pad, th01_pad, c1_pad, th10_pad, False

def compute_v3_curriculum_weights(df):
    weights = np.ones(len(df), dtype=np.float32)
    t_vals = df["t"].values.astype(np.float32)
    disp_vals = df["mean_disp"].values.astype(np.float32) if "mean_disp" in df.columns else np.zeros(len(df), dtype=np.float32)
    occ_scores = df["patch_occ_score"].values.astype(np.float32) if "patch_occ_score" in df.columns else np.zeros(len(df), dtype=np.float32)
    ink_diffs = df["global_ink_diff"].values.astype(np.float32) if "global_ink_diff" in df.columns else np.zeros(len(df), dtype=np.float32)

    is_high_occ = (occ_scores >= 0.08)
    weights[is_high_occ] *= 1.45

    is_ink_diverge = (ink_diffs >= 0.15)
    weights[is_ink_diverge] *= 1.25

    is_boundary_t = ((t_vals >= 0.15) & (t_vals <= 0.30)) | ((t_vals >= 0.70) & (t_vals <= 0.85))
    is_large_disp = (disp_vals >= 7.0)
    weights[is_boundary_t] *= 1.20
    weights[is_large_disp] *= 1.25

    return weights

class SakugaGoldenDataset_v3(Dataset):
    def __init__(self, root_dir, df_split, is_train=True, crop_size=None, cache_dir=None):
        self.root_dir = root_dir
        self.is_train = is_train
        self.crop_size = crop_size
        self.split_name = 'train' if is_train else 'val'
        self.length = len(df_split)
        self.temporal_aug = TemporalReversalAugmentation(p=0.5) if is_train else None

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
        kps_npy = os.path.join(cd, f'kps_data_{self.split_name}_v3.npy')
        cnt_npy = os.path.join(cd, f'kps_count_{self.split_name}_v3.npy')

        if os.path.exists(kps_npy) and os.path.exists(cnt_npy):
            self.kps_data = np.load(kps_npy, mmap_mode='r')
            self.kps_count = np.load(cnt_npy, mmap_mode='r')
            print(f"[✓] Nạp có sẵn cache {self.split_name} ({self.length} mẫu .npy mmap 0-copy) < 0.05s.")
        else:
            print(f"[*] Preloading keypoints {self.split_name} ({self.length} mẫu) đa luồng an toàn...", flush=True)
            MAX_KPS_CACHE = 80
            kps_buf = np.zeros((self.length, 2, MAX_KPS_CACHE, 2), dtype=np.float32)
            cnt_buf = np.zeros(self.length, dtype=np.int32)

            def _load_single_kps(idx):
                pm = fm_list[idx]
                try:
                    kps = np.load(pm).astype(np.float32)
                    if kps.ndim == 3 and kps.shape[0] == 2 and kps.shape[2] == 2:
                        n = kps.shape[1]
                        if n > MAX_KPS_CACHE:
                            idx_sub = np.linspace(0, n - 1, MAX_KPS_CACHE, dtype=int)
                            kps_buf[idx] = kps[:, idx_sub, :]
                            cnt_buf[idx] = MAX_KPS_CACHE
                        elif n > 0:
                            kps_buf[idx, :, :n, :] = kps
                            cnt_buf[idx] = n
                except Exception:
                    pass

            with ThreadPoolExecutor(max_workers=4) as ex:
                list(tqdm(ex.map(_load_single_kps, range(self.length)), total=self.length, desc=f"   Loading KPS ({self.split_name})", leave=False))

            np.save(kps_npy, kps_buf)
            np.save(cnt_npy, cnt_buf)
            self.kps_data = np.load(kps_npy, mmap_mode='r')
            self.kps_count = np.load(cnt_npy, mmap_mode='r')
            del kps_buf, cnt_buf
            gc.collect()
            print(f"[✓] Đã tạo & mmap cache {self.split_name} ({self.length} mẫu) thành công.")

        # 3. Precompute TPS Cache cố định cho Validation (độ phân giải 540p -> 544 pad)
        if not self.is_train:
            val_c0_npy = os.path.join(cd, 'val_c0_v3.npy')
            val_th01_npy = os.path.join(cd, 'val_th01_v3.npy')
            val_c1_npy = os.path.join(cd, 'val_c1_v3.npy')
            val_th10_npy = os.path.join(cd, 'val_th10_v3.npy')
            val_valid_npy = os.path.join(cd, 'val_valid_v3.npy')

            all_val_exist = all(os.path.exists(p) for p in [val_c0_npy, val_th01_npy, val_c1_npy, val_th10_npy, val_valid_npy])
            if all_val_exist:
                self.val_c0 = np.load(val_c0_npy, mmap_mode='r')
                self.val_th01 = np.load(val_th01_npy, mmap_mode='r')
                self.val_c1 = np.load(val_c1_npy, mmap_mode='r')
                self.val_th10 = np.load(val_th10_npy, mmap_mode='r')
                self.val_valid = np.load(val_valid_npy, mmap_mode='r')
                print(f"[✓] Nạp có sẵn {self.length} mẫu TPS Validation mmap 0-copy (< 0.05s).")
            else:
                print(f"[*] Precomputing TPS cache cho Validation ({self.length} mẫu)...", flush=True)
                val_c0_buf = np.zeros((self.length, 88, 2), dtype=np.float32)
                val_th01_buf = np.zeros((self.length, 91, 2), dtype=np.float32)
                val_c1_buf = np.zeros((self.length, 88, 2), dtype=np.float32)
                val_th10_buf = np.zeros((self.length, 91, 2), dtype=np.float32)
                val_valid_buf = np.zeros((self.length, 1, 1, 1), dtype=np.float32)

                def _solve_single_val_tps(idx):
                    cnt = int(self.kps_count[idx])
                    kps = self.kps_data[idx, :, :cnt, :].copy() if cnt > 0 else np.zeros((2, 0, 2), dtype=np.float32)
                    w, h = 960, 540
                    pad_h = 544
                    pad_w = 960
                    if kps.shape[1] > 0:
                        kps_norm = kps.copy()
                        kps_norm[0, :, 0] = (kps_norm[0, :, 0] * w) / pad_w
                        kps_norm[0, :, 1] = (kps_norm[0, :, 1] * h) / pad_h
                        kps_norm[1, :, 0] = (kps_norm[1, :, 0] * w) / pad_w
                        kps_norm[1, :, 1] = (kps_norm[1, :, 1] * h) / pad_h
                    else:
                        kps_norm = kps
                    c0_p, th01_p, c1_p, th10_p, v = solve_tps_pair_cpu(kps_norm, max_kps=80, lambd=0.5, return_tensor=False)
                    val_c0_buf[idx] = c0_p
                    val_th01_buf[idx] = th01_p
                    val_c1_buf[idx] = c1_p
                    val_th10_buf[idx] = th10_p
                    val_valid_buf[idx, 0, 0, 0] = float(v)

                with ThreadPoolExecutor(max_workers=4) as ex:
                    list(tqdm(ex.map(_solve_single_val_tps, range(self.length)), total=self.length, desc="   Solving Val TPS", leave=False))

                np.save(val_c0_npy, val_c0_buf)
                np.save(val_th01_npy, val_th01_buf)
                np.save(val_c1_npy, val_c1_buf)
                np.save(val_th10_npy, val_th10_buf)
                np.save(val_valid_npy, val_valid_buf)
                self.val_c0 = np.load(val_c0_npy, mmap_mode='r')
                self.val_th01 = np.load(val_th01_npy, mmap_mode='r')
                self.val_c1 = np.load(val_c1_npy, mmap_mode='r')
                self.val_th10 = np.load(val_th10_npy, mmap_mode='r')
                self.val_valid = np.load(val_valid_npy, mmap_mode='r')
                del val_c0_buf, val_th01_buf, val_c1_buf, val_th10_buf, val_valid_buf
                gc.collect()
                print(f"[✓] Đã precompute và lưu {self.length} mẫu TPS Validation vào .npy mmap.")

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

        h, w = im0.shape
        if imgt.shape != (h, w): imgt = cv2.resize(imgt, (w, h), interpolation=cv2.INTER_NEAREST)
        if im1.shape != (h, w): im1 = cv2.resize(im1, (w, h), interpolation=cv2.INTER_NEAREST)

        # Đệm biên lên bội số của 32 (540p -> 544x960) bảo toàn 100% chi tiết toàn khung hình
        pad_h = ((h + 31) // 32) * 32
        pad_w = ((w + 31) // 32) * 32
        if pad_h != h or pad_w != w:
            im0_pad = np.pad(im0, ((0, pad_h - h), (0, pad_w - w)), mode='constant', constant_values=255)
            imgt_pad = np.pad(imgt, ((0, pad_h - h), (0, pad_w - w)), mode='constant', constant_values=255)
            im1_pad = np.pad(im1, ((0, pad_h - h), (0, pad_w - w)), mode='constant', constant_values=255)
        else:
            im0_pad, imgt_pad, im1_pad = im0, imgt, im1

        if self.is_train:
            # Huấn luyện toàn phần 540p (Không crop)
            cnt = int(self.kps_count[idx])
            kps = self.kps_data[idx, :, :cnt, :].copy() if cnt > 0 else np.zeros((2, 0, 2), dtype=np.float32)

            if kps.shape[1] > 0:
                kps_norm = kps.copy()
                kps_norm[0, :, 0] = (kps_norm[0, :, 0] * w) / pad_w
                kps_norm[0, :, 1] = (kps_norm[0, :, 1] * h) / pad_h
                kps_norm[1, :, 0] = (kps_norm[1, :, 0] * w) / pad_w
                kps_norm[1, :, 1] = (kps_norm[1, :, 1] * h) / pad_h
            else:
                kps_norm = kps

            c0_pad, th01_pad, c1_pad, th10_pad, valid_flag = solve_tps_pair_cpu(kps_norm, max_kps=80, lambd=0.5, return_tensor=False)
            valid_flag = np.full((1, 1, 1), float(valid_flag), dtype=np.float32)
            t_scalar = float(self.t_arr[idx])
            if self.temporal_aug is not None:
                im0_pad, imgt_pad, im1_pad, t_scalar, kps_norm, c0_pad, th01_pad, c1_pad, th10_pad, _ = self.temporal_aug(
                    im0_pad, imgt_pad, im1_pad, t_scalar, kps_norm, c0_pad, th01_pad, c1_pad, th10_pad
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
        else:
            # Thẩm định toàn phần 540p (Sử dụng precomputed cache)
            return (
                torch.from_numpy(np.ascontiguousarray(im0_pad[None, :, :], dtype=np.uint8)),
                torch.from_numpy(np.ascontiguousarray(imgt_pad[None, :, :], dtype=np.uint8)),
                torch.from_numpy(np.ascontiguousarray(im1_pad[None, :, :], dtype=np.uint8)),
                torch.tensor(float(self.t_arr[idx]), dtype=torch.float32),
                torch.from_numpy(self.val_c0[idx].copy()),
                torch.from_numpy(self.val_th01[idx].copy()),
                torch.from_numpy(self.val_c1[idx].copy()),
                torch.from_numpy(self.val_th10[idx].copy()),
                torch.from_numpy(self.val_valid[idx].copy()),
                clip_name,
                int(h),
                int(w)
            )

custom_collate_fn_v3 = default_collate

print("=" * 80)
print("[*] KHỞI TẠO DATASET VÀ DATALOADER 540P TOÀN PHẦN (FULL-FRAME 544x960, KHÔNG CROP, BATCH 8 + ACCUM 2)...")
train_dataset = SakugaGoldenDataset_v3(DATASET_ROOT, train_df, is_train=True, crop_size=None, cache_dir=CACHE_DIR)
val_dataset = SakugaGoldenDataset_v3(DATASET_ROOT, val_df, is_train=False, crop_size=None, cache_dir=CACHE_DIR)

train_weights = compute_v3_curriculum_weights(train_df)
train_sampler = WeightedRandomSampler(
    weights=torch.as_tensor(train_weights, dtype=torch.double),
    num_samples=len(train_dataset),
    replacement=True
)

BATCH_SIZE = 8   # 4 mẫu/GPU trên 2x T4 (VRAM an toàn < 9.2 GB ở 540p)
GRAD_ACCUM_STEPS = 2  # Tích lũy gradient 2 bước => Effective Batch Size = 16
VAL_BATCH_SIZE = 12  # 6 mẫu/GPU trên 2x T4 (Inference mode nhanh gấp đôi, VRAM < 3.2 GB)

def _worker_init_fn(worker_id):
    torch.set_num_threads(1)
    cv2.setNumThreads(0)
    random.seed(42 + worker_id)
    np.random.seed(42 + worker_id)

# [QUY TẮC SỐNG CÒN ZERO-LEAK]:
# 1. persistent_workers=False: Toàn bộ 4 worker bị huỷ và giải phóng 100% heap/RAM sau mỗi epoch!
# 2. pin_memory=False: Không dùng CUDA host caching allocator, triệt tiêu memory leak page-locked!
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    num_workers=4,
    pin_memory=False,
    drop_last=True,
    persistent_workers=False,
    worker_init_fn=_worker_init_fn,
    collate_fn=custom_collate_fn_v3
)

val_loader = DataLoader(
    val_dataset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=False,
    drop_last=False,
    persistent_workers=False,
    worker_init_fn=_worker_init_fn,
    collate_fn=custom_collate_fn_v3
)

print(f"[✓] DataLoaders sẵn sàng: Train={len(train_loader)} batches (Batch={BATCH_SIZE}, Accum={GRAD_ACCUM_STEPS}) | Val={len(val_loader)} batches")
print("=" * 80)
sys.stdout.flush()
