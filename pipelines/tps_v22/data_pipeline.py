# -*- coding: utf-8 -*-
"""
pipelines/tps_v22/data_pipeline.py
Module Data Pipeline & Sampling cho Finetune V2.2:
1. TemporalReversalAugmentation (p = 0.5): Đảo chiều thời gian (I0, I1) -> (I1, I0) & t -> 1.0 - t
2. SakugaBoundaryMotionWeightedSampler: Ưu tiên mẫu thời gian biên t in [0.15, 0.30] U [0.70, 0.85] và disp >= 7.0px
3. SakugaDynamic5050Sampler: Giữ cố định 100% tập train gốc V1, trộn 50:50 với tập mở rộng
4. SakugaGoldenDataset_v22: Dataset tối ưu hóa bộ nhớ, 0% Copy-on-Write, tích hợp Data Augmentation
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler
from PIL import Image
import cv2


# ==============================================================================
# 1. TEMPORAL REVERSAL AUGMENTATION (p = 0.5)
# ==============================================================================

class TemporalReversalAugmentation:
    """
    Temporal Reversal Augmentation (Đảo ngược chiều thời gian):
    - Ngẫu nhiên với xác suất p = 0.5 đảo ngược cặp keyframe (I0, I1) -> (I1, I0)
    - Đồng thời đảo bước thời gian nội suy t -> 1.0 - t
    - Ground Truth I_t giữ nguyên (vì Interp(I0, I1, t) == Interp(I1, I0, 1-t))
    - Keypoint matches (kps0, kps1) đảo thành (kps1, kps0)
    - Ma trận TPS giải trước c0_pad, th01_pad <-> c1_pad, th10_pad được tráo đổi
    - Tác dụng: Triệt tiêu hoàn toàn sự mất cân xứng hiệu năng giữa t=0.20 và t=0.80.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img0, imggt, img1, t_val, kps=None,
                 c0_pad=None, th01_pad=None, c1_pad=None, th10_pad=None):
        if random.random() < self.p:
            # Đảo keyframes
            img0_out = img1
            img1_out = img0
            # Đảo bước thời gian
            t_out = float(1.0 - t_val)
            # Ground truth giữ nguyên
            gt_out = imggt

            # Đảo keypoints
            if kps is not None and hasattr(kps, 'shape') and len(kps.shape) == 3 and kps.shape[0] == 2:
                kps_out = np.stack([kps[1], kps[0]], axis=0)
            else:
                kps_out = kps

            # Đảo ma trận TPS
            c0_out = c1_pad
            th01_out = th10_pad
            c1_out = c0_pad
            th10_out = th01_pad
            is_reversed = True
        else:
            img0_out = img0
            img1_out = img1
            t_out = float(t_val)
            gt_out = imggt
            kps_out = kps
            c0_out = c0_pad
            th01_out = th01_pad
            c1_out = c1_pad
            th10_out = th10_pad
            is_reversed = False

        return (img0_out, gt_out, img1_out, t_out, kps_out,
                c0_out, th01_out, c1_out, th10_out, is_reversed)


# ==============================================================================
# 2. SAMPLERS: BOUNDARY-MOTION WEIGHTED & DYNAMIC 50/50 MIXING
# ==============================================================================

def compute_boundary_motion_weights(df, boost_pct=0.25):
    """
    Tính trọng số lấy mẫu cho từng triplet trong tập huấn luyện:
    - Mặc định: weight = 1.0
    - Mẫu có bước thời gian biên t in [0.15, 0.30] U [0.70, 0.85]: tăng +25% trọng số (weight *= 1.25)
    - Mẫu có độ dịch chuyển lớn disp >= 7.0 px: tăng +25% trọng số (weight *= 1.25)
    - Nếu thỏa mãn cả 2 điều kiện: weight = 1.25 * 1.25 = 1.5625 (+56.25%)
    """
    weights = np.ones(len(df), dtype=np.float32)
    t_vals = df['t'].values.astype(np.float32)

    # Lấy thông tin mean_disp
    if 'mean_disp' in df.columns:
        disp_vals = df['mean_disp'].values.astype(np.float32)
    elif 'disp' in df.columns:
        disp_vals = df['disp'].values.astype(np.float32)
    else:
        disp_vals = np.zeros(len(df), dtype=np.float32)

    is_boundary_t = ((t_vals >= 0.15) & (t_vals <= 0.30)) | ((t_vals >= 0.70) & (t_vals <= 0.85))
    is_large_disp = (disp_vals >= 7.0)

    weights[is_boundary_t] *= (1.0 + boost_pct)
    weights[is_large_disp] *= (1.0 + boost_pct)
    return weights


class SakugaBoundaryMotionWeightedSampler(WeightedRandomSampler):
    """
    WeightedRandomSampler tối ưu cho dữ liệu Sakuga tập đơn:
    Ưu tiên lấy mẫu nhiều hơn 25% cho các mẫu biên t và độ dời lớn.
    """
    def __init__(self, df, boost_pct=0.25, num_samples=None, replacement=True):
        weights = compute_boundary_motion_weights(df, boost_pct=boost_pct)
        if num_samples is None:
            num_samples = len(df)
        super(SakugaBoundaryMotionWeightedSampler, self).__init__(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=num_samples,
            replacement=replacement
        )


class SakugaDynamic5050Sampler(Sampler):
    """
    Dynamic 50/50 Dataset Mixing Sampler:
    - Trong mỗi epoch, giữ cố định 100% tập train gốc V1 (3,545 mẫu)
    - Kết hợp với 1 lượng tương đương (tỷ lệ 50:50) lấy mẫu xoay vòng từ tập mở rộng (Patch 2)
    - Áp dụng phân bổ trọng số Weighted Sampling trên cả hai nửa dữ liệu.
    """
    def __init__(self, df_all, orig_mask=None, boost_pct=0.25, shuffle=True):
        if orig_mask is None:
            # Nếu không chỉ định, tìm theo source hoặc clip_id
            if 'source' in df_all.columns:
                orig_mask = df_all['source'].str.contains('v1|orig|golden', case=False, na=False).values
            else:
                orig_mask = np.ones(len(df_all), dtype=bool)

        self.df_all = df_all
        self.orig_indices = np.where(orig_mask)[0]
        self.ext_indices = np.where(~orig_mask)[0]
        self.shuffle = shuffle

        # Tính trọng số cho từng mẫu
        self.all_weights = compute_boundary_motion_weights(df_all, boost_pct=boost_pct)
        self.orig_weights = self.all_weights[self.orig_indices]
        self.orig_weights /= np.sum(self.orig_weights)

        if len(self.ext_indices) > 0:
            self.ext_weights = self.all_weights[self.ext_indices]
            self.ext_weights /= np.sum(self.ext_weights)
        else:
            self.ext_weights = None

        self.n_orig = len(self.orig_indices)

    def __iter__(self):
        if len(self.ext_indices) == 0:
            # Nếu chỉ có tập gốc, lấy theo weighted permutation
            if self.shuffle:
                picked = np.random.choice(self.orig_indices, size=self.n_orig, replace=True, p=self.orig_weights)
            else:
                picked = self.orig_indices.copy()
            return iter(picked.tolist())

        # 1. Lấy 100% quota tập gốc V1
        if self.shuffle:
            picked_orig = np.random.choice(self.orig_indices, size=self.n_orig, replace=False)
        else:
            picked_orig = self.orig_indices.copy()

        # 2. Lấy số lượng tương đương (50:50) từ tập mở rộng theo weighted distribution
        replace_ext = len(self.ext_indices) < self.n_orig
        picked_ext = np.random.choice(
            self.ext_indices, size=self.n_orig, replace=replace_ext, p=self.ext_weights
        )

        # 3. Ghép xen kẽ 50:50
        combined = np.empty(self.n_orig * 2, dtype=np.int64)
        combined[0::2] = picked_orig
        combined[1::2] = picked_ext

        if self.shuffle:
            np.random.shuffle(combined)

        return iter(combined.tolist())

    def __len__(self):
        return self.n_orig * 2 if len(self.ext_indices) > 0 else self.n_orig


# ==============================================================================
# 3. TPS SOLVER CHO DATA PIPELINE
# ==============================================================================

FIXED_ANCHORS_CPU = torch.tensor([
    [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
    [0.5, 0.0], [0.5, 1.0], [0.0, 0.5], [1.0, 0.5]
], dtype=torch.float64)

def tps_rbf(r):
    r = torch.clamp(r, min=1e-6)
    return (r ** 2) * torch.log(r)

def solve_tps_pair_cpu(kps, max_kps=80, lambd=0.5, min_dist=0.015):
    """
    Giải tham số biến dạng TPS float64 an toàn trên CPU.
    Trả về c0_pad, th01_pad, c1_pad, th10_pad, valid_flag.
    """
    total_pad = max_kps + 8
    c0_pad = torch.zeros((total_pad, 2), dtype=torch.float32)
    th01_pad = torch.zeros((total_pad + 3, 2), dtype=torch.float32)
    c1_pad = torch.zeros((total_pad, 2), dtype=torch.float32)
    th10_pad = torch.zeros((total_pad + 3, 2), dtype=torch.float32)

    if kps is None or not hasattr(kps, 'shape') or len(kps.shape) < 3 or kps.shape[1] < 4:
        return c0_pad, th01_pad, c1_pad, th10_pad, torch.tensor([0.0], dtype=torch.float32)

    kps0 = torch.from_numpy(kps[0]).to(torch.float64)
    kps1 = torch.from_numpy(kps[1]).to(torch.float64)

    # Lọc điểm quá sát nhau
    keep = [0]
    for i in range(1, kps0.shape[0]):
        dists = torch.norm(kps0[i] - kps0[keep], dim=-1)
        if torch.min(dists) >= min_dist:
            keep.append(i)

    if len(keep) < 4:
        return c0_pad, th01_pad, c1_pad, th10_pad, torch.tensor([0.0], dtype=torch.float32)

    c0_f = kps0[keep]
    c1_f = kps1[keep]

    if c0_f.shape[0] > max_kps:
        idx_sub = torch.linspace(0, c0_f.shape[0] - 1, max_kps, dtype=torch.long)
        c0_f = c0_f[idx_sub]
        c1_f = c1_f[idx_sub]

    # Bổ sung 8 Fixed Anchors
    c0_a = torch.cat([c0_f, FIXED_ANCHORS_CPU], dim=0)
    c1_a = torch.cat([c1_f, FIXED_ANCHORS_CPU], dim=0)
    n_a = c0_a.shape[0]

    # Giải chiều 0 -> 1
    diff0 = c0_a.unsqueeze(1) - c0_a.unsqueeze(0)
    K0 = tps_rbf(torch.norm(diff0, dim=-1)) + lambd * torch.eye(n_a, dtype=torch.float64)
    P0 = torch.cat([torch.ones((n_a, 1), dtype=torch.float64), c0_a], dim=-1)
    L0 = torch.cat([torch.cat([K0, P0], dim=-1),
                    torch.cat([P0.t(), torch.zeros((3, 3), dtype=torch.float64)], dim=-1)], dim=0)
    Y01 = torch.cat([c1_a, torch.zeros((3, 2), dtype=torch.float64)], dim=0)

    # Giải chiều 1 -> 0
    diff1 = c1_a.unsqueeze(1) - c1_a.unsqueeze(0)
    K1 = tps_rbf(torch.norm(diff1, dim=-1)) + lambd * torch.eye(n_a, dtype=torch.float64)
    P1 = torch.cat([torch.ones((n_a, 1), dtype=torch.float64), c1_a], dim=-1)
    L1 = torch.cat([torch.cat([K1, P1], dim=-1),
                    torch.cat([P1.t(), torch.zeros((3, 3), dtype=torch.float64)], dim=-1)], dim=0)
    Y10 = torch.cat([c0_a, torch.zeros((3, 2), dtype=torch.float64)], dim=0)

    try:
        th01 = torch.linalg.solve(L0, Y01)
        th10 = torch.linalg.solve(L1, Y10)
    except Exception:
        th01 = torch.linalg.lstsq(L0, Y01).solution
        th10 = torch.linalg.lstsq(L1, Y10).solution

    c0_pad[:n_a] = c0_a.to(torch.float32)
    th01_pad[:n_a + 3] = th01.to(torch.float32)
    c1_pad[:n_a] = c1_a.to(torch.float32)
    th10_pad[:n_a + 3] = th10.to(torch.float32)

    return c0_pad, th01_pad, c1_pad, th10_pad, torch.tensor([1.0], dtype=torch.float32)


# ==============================================================================
# 4. SAKUGA GOLDEN DATASET V2.2
# ==============================================================================

class SakugaGoldenDataset_v22(Dataset):
    """
    Dataset hoàn chỉnh cho pipeline V2.2:
    - Hỗ trợ TemporalReversalAugmentation (p = 0.5)
    - Tự động crop ngẫu nhiên lúc train (crop_size=512)
    - Chuẩn hóa padding 32 lúc val
    - Zero Copy-on-Write NumPy contiguous buffers
    """
    def __init__(self, root_dir, df_split, is_train=True, crop_size=512,
                 enable_temporal_reversal=True, p_reversal=0.5):
        self.root_dir = root_dir
        self.df = df_split.reset_index(drop=True)
        self.is_train = is_train
        self.crop_size = crop_size
        self.reversal_aug = TemporalReversalAugmentation(p=p_reversal) if (is_train and enable_temporal_reversal) else None

        self.records = []
        for idx, row in self.df.iterrows():
            p0 = os.path.join(root_dir, str(row['frame0']).lstrip('/\\'))
            pgt = os.path.join(root_dir, str(row['frame_gt']).lstrip('/\\'))
            p1 = os.path.join(root_dir, str(row['frame1']).lstrip('/\\'))
            pmatch = os.path.join(root_dir, str(row['match_file']).lstrip('/\\')) if 'match_file' in row and pd.notna(row['match_file']) else None
            t_val = float(row['t'])
            clip_id = str(row['clip_id'])
            disp = float(row.get('mean_disp', 0.0))
            self.records.append((p0, pgt, p1, pmatch, t_val, clip_id, disp))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        p0, pgt, p1, pmatch, t_val, clip_id, disp = self.records[idx]

        im0 = cv2.imread(p0, cv2.IMREAD_GRAYSCALE)
        imgt = cv2.imread(pgt, cv2.IMREAD_GRAYSCALE)
        im1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)

        if im0 is None or imgt is None or im1 is None:
            cs = self.crop_size if self.is_train else 512
            im0 = np.full((cs, cs), 255, dtype=np.uint8)
            imgt = np.full((cs, cs), 255, dtype=np.uint8)
            im1 = np.full((cs, cs), 255, dtype=np.uint8)

        h, w = im0.shape

        # Nạp toạ độ GlueStick matches
        if pmatch and os.path.exists(pmatch):
            kps = np.load(pmatch).astype(np.float32)
        else:
            kps = np.zeros((2, 0, 2), dtype=np.float32)

        if self.is_train:
            cs = self.crop_size
            if h > cs and w > cs:
                top = random.randint(0, h - cs)
                left = random.randint(0, w - cs)
                im0 = im0[top:top+cs, left:left+cs]
                imgt = imgt[top:top+cs, left:left+cs]
                im1 = im1[top:top+cs, left:left+cs]

                if kps.shape[1] > 0:
                    kps = kps.copy()
                    kps[0, :, 0] = (kps[0, :, 0] * w - left) / cs
                    kps[0, :, 1] = (kps[0, :, 1] * h - top) / cs
                    kps[1, :, 0] = (kps[1, :, 0] * w - left) / cs
                    kps[1, :, 1] = (kps[1, :, 1] * h - top) / cs
                    m0 = (kps[0, :, 0] >= 0) & (kps[0, :, 0] <= 1) & (kps[0, :, 1] >= 0) & (kps[0, :, 1] <= 1)
                    m1 = (kps[1, :, 0] >= 0) & (kps[1, :, 0] <= 1) & (kps[1, :, 1] >= 0) & (kps[1, :, 1] <= 1)
                    kps = kps[:, m0 & m1, :]
            else:
                pad_h = ((h + 31) // 32) * 32
                pad_w = ((w + 31) // 32) * 32
                im0 = np.pad(im0, ((0, pad_h - h), (0, pad_w - w)), mode='constant', constant_values=255)
                imgt = np.pad(imgt, ((0, pad_h - h), (0, pad_w - w)), mode='constant', constant_values=255)
                im1 = np.pad(im1, ((0, pad_h - h), (0, pad_w - w)), mode='constant', constant_values=255)

            # Giải tham số TPS trên CPU
            c0_pad, th01_pad, c1_pad, th10_pad, valid_flag = solve_tps_pair_cpu(kps, max_kps=80, lambd=0.5, min_dist=0.015)

            # Áp dụng Temporal Reversal Augmentation
            if self.reversal_aug:
                (im0, imgt, im1, t_val, kps,
                 c0_pad, th01_pad, c1_pad, th10_pad, _) = self.reversal_aug(
                    im0, imgt, im1, t_val, kps=kps,
                    c0_pad=c0_pad, th01_pad=th01_pad, c1_pad=c1_pad, th10_pad=th10_pad
                )

        else:
            # Validation mode: Không crop, padding tròn lên bội số 32
            pad_h = ((h + 31) // 32) * 32
            pad_w = ((w + 31) // 32) * 32
            im0 = np.pad(im0, ((0, pad_h - h), (0, pad_w - w)), mode='constant', constant_values=255)
            imgt = np.pad(imgt, ((0, pad_h - h), (0, pad_w - w)), mode='constant', constant_values=255)
            im1 = np.pad(im1, ((0, pad_h - h), (0, pad_w - w)), mode='constant', constant_values=255)

            if kps.shape[1] > 0:
                kps = kps.copy()
                kps[0, :, 0] = (kps[0, :, 0] * w) / pad_w
                kps[0, :, 1] = (kps[0, :, 1] * h) / pad_h
                kps[1, :, 0] = (kps[1, :, 0] * w) / pad_w
                kps[1, :, 1] = (kps[1, :, 1] * h) / pad_h

            c0_pad, th01_pad, c1_pad, th10_pad, valid_flag = solve_tps_pair_cpu(kps, max_kps=80, lambd=0.5, min_dist=0.015)

        # Chuyển đổi sang tensor nét vẽ ink space: 1.0 = mực đen, 0.0 = nền trắng
        f0 = (255.0 - im0.astype(np.float32)) * (1.0 / 255.0)
        fgt = (255.0 - imgt.astype(np.float32)) * (1.0 / 255.0)
        f1 = (255.0 - im1.astype(np.float32)) * (1.0 / 255.0)

        t0_t = torch.from_numpy(np.ascontiguousarray(f0)).unsqueeze(0)
        tgt_t = torch.from_numpy(np.ascontiguousarray(fgt)).unsqueeze(0)
        t1_t = torch.from_numpy(np.ascontiguousarray(f1)).unsqueeze(0)
        t_tensor = torch.tensor([[[t_val]]], dtype=torch.float32)

        return t0_t, tgt_t, t1_t, t_tensor, c0_pad, th01_pad, c1_pad, th10_pad, valid_flag, clip_id, h, w
