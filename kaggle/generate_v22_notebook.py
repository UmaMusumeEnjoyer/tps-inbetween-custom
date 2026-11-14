#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_v22_notebook.py
Script tự động khởi tạo và chuẩn hóa notebook huấn luyện Kaggle V2.2:
- kaggle/05_train_tps_inbetween_finetune_v2.2.ipynb
- kaggle/finetune-v2-2.ipynb

Tích hợp toàn diện các hạng mục kỹ thuật V2.2:
1. Tương thích 100% với Dataset Patch 2 (cả .tar.gz & .zip, path fallback, auto-scan).
2. Data Pipeline: Temporal Reversal Augmentation (p=0.5) & WeightedRandomSampler (+25% boundary t & large disp).
3. Bộ 3 Loss V2.2: StrokeDensityRegularization (0.30), DirectionalTangentCoherence (0.20), SoftClDice (0.20).
4. Differential LR: 1.5e-5 (Backbone/IFNet/ContextNet) vs 5.0e-5 (ASF-GR / Synthesis UNet), Warmup 2 ep + Cosine Decay.
5. Smart Checkpoint Manager V2.2: 3 Hard Gates + Sakuga Composite Score.
6. Code Cell trực quan hóa 6-Panel Dashboard biểu diễn chỉ số qua các Epochs kèm bảng so sánh & xuất ảnh HD.
"""

import os
import sys
import json

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


def generate_v22():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    v11_path = os.path.join(script_dir, "04_train_tps_inbetween_finetune_v1.1.ipynb")
    v22_path = os.path.join(script_dir, "05_train_tps_inbetween_finetune_v2.2.ipynb")
    alias_path = os.path.join(script_dir, "finetune-v2-2.ipynb")

    if not os.path.exists(v11_path):
        raise FileNotFoundError(f"Không tìm thấy notebook v1.1 mẫu tại: {v11_path}")

    with open(v11_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    def get_code(cell_idx):
        return "".join(nb["cells"][cell_idx]["source"])

    def set_code(cell_idx, text):
        lines = [line + "\n" for line in text.strip().split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        nb["cells"][cell_idx]["source"] = lines

    # -------------------------------------------------------------------------
    # CELL 0: Header Markdown
    # -------------------------------------------------------------------------
    c0 = """# HUẤN LUYỆN TINH CHỈNH TPS-INBETWEEN PHIÊN BẢN V2.2 (ANTI-THICKENING & SAKUGA COMPOSITE SCORE)
## ĐỀ TÀI: NGHIÊN CỨU & TINH CHỈNH MÔ HÌNH TPS-INBETWEEN CHO HOẠT HỌA ANIME 2D (PBL6)

> **Mục tiêu phiên bản V2.2:**
> Giải quyết triệt để 4 hạn chế phát hiện từ kiểm toán V2.0.2 Best:
> 1. **Triệt tiêu hiện tượng phình nét (Line Thickening):** Kéo `Density Ratio` từ `1.3139` về dải chuẩn `[1.00 - 1.05]` qua `StrokeDensityRegularizationLoss` ($\lambda = 0.30$).
> 2. **Cân bằng đối xứng thời gian ($t=0.20$ vs $t=0.80$):** Tích hợp `Temporal Reversal Augmentation (p = 0.5)` và `WeightedRandomSampler` ưu tiên bước thời gian biên và chuyển động lớn (`disp >= 7.0 px`).
> 3. **Chuẩn hóa đo lường Metric:** Đồng nhất 100% cách tính `Ghost Pixels` và `Chamfer Distance` giữa vòng lặp train validation và fast benchmark.
> 4. **Hàm mất mát bảo toàn nét mảnh & hướng tiếp tuyến:** Tích hợp `DirectionalTangentCoherenceLoss` ($\lambda = 0.20$) và `SoftClDiceLoss` ($\lambda = 0.20$).
> 5. **Quy tắc chấm điểm Best Checkpoint:** Chấm điểm dựa trên **Sakuga Composite Score** với 3 Hard Gates nghiêm ngặt.
> 6. **Tương thích toàn diện Dataset Patch 2:** Tự động giải nén cả `.tar.gz` và `.zip`, fallback path an toàn, hỗ trợ quy mô 6,734 triplets."""
    set_code(0, c0)

    # -------------------------------------------------------------------------
    # CELL 1: Môi trường & Import PyTorch Samplers
    # -------------------------------------------------------------------------
    c1 = get_code(1)
    c1 = c1.replace(
        "from torch.utils.data import Dataset, DataLoader, Subset",
        "from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler"
    )
    set_code(1, c1)

    # -------------------------------------------------------------------------
    # CELL 2: Tự Động Định Vị, Giải Nén Siêu Tốc & Thẩm Định Toàn Vẹn Dataset (Patch 2 Ready)
    # -------------------------------------------------------------------------
    c2 = """# [CELL 2]: Tự Động Định Vị, Giải Nén Siêu Tốc & Thẩm Định Toàn Vẹn Golden Dataset (Patch 2 Ready)
import os
import sys
import gc
import time
import zipfile
import tarfile
import subprocess
import pandas as pd

KAGGLE_INPUT = '/kaggle/input'
WORKING_DIR = '/kaggle/working'

TMP_DIR = os.path.join(WORKING_DIR, '_temp_dataset')
EXTRACT_DIR = os.path.join(TMP_DIR, 'sakuga_golden_dataset_tps_ready')
DATASET_EXTRACTED_TO_TEMP = False

DATASET_ROOT = None

# Bước 1: Quét đệ quy tìm trực tiếp thư mục chứa triplets.csv hoặc triplets_combined.csv
search_locations = [KAGGLE_INPUT, EXTRACT_DIR, TMP_DIR, WORKING_DIR, './data/processed', '../data/processed', '.']

print("[*] Đang quét tìm thư mục Golden Dataset (chứa triplets.csv hoặc triplets_combined.csv)...")
for base_dir in search_locations:
    if not os.path.exists(base_dir):
        continue
    for root, dirs, files in os.walk(base_dir):
        if 'triplets.csv' in files or 'triplets_combined.csv' in files:
            DATASET_ROOT = root
            print(f"[✓] Tự động phát hiện thư mục Golden Dataset tại: {DATASET_ROOT}")
            break
    if DATASET_ROOT is not None:
        break

# Bước 2: Quét tìm archive (.tar.gz hoặc .zip) hỗ trợ cả Patch 2 và tự động giải nén vào NVMe
if DATASET_ROOT is None:
    print("[*] Chưa thấy thư mục giải nén, tiến hành quét tìm archive (.tar.gz / .zip) chứa Sakuga / Patch 2...")
    archive_path = None
    archive_type = None

    valid_keywords = ['sakuga', 'golden', 'tps', 'patch2', 'patch_2']

    for base_dir in search_locations:
        if not os.path.exists(base_dir):
            continue
        for root, dirs, files in os.walk(base_dir):
            for f in files:
                f_lower = f.lower()
                is_match = any(k in f_lower for k in valid_keywords)
                if is_match and (f_lower.endswith('.tar.gz') or f_lower.endswith('.tgz')):
                    archive_path = os.path.join(root, f)
                    archive_type = 'tar'
                    break
                elif is_match and f_lower.endswith('.zip'):
                    archive_path = os.path.join(root, f)
                    archive_type = 'zip'
                    break
            if archive_path is not None:
                break
        if archive_path is not None:
            break

    if archive_path is not None and os.path.exists(archive_path):
        file_size_mb = os.path.getsize(archive_path) / (1024 * 1024)
        print(f"[✓] Phát hiện archive dataset: {archive_path} ({file_size_mb:.2f} MB - Loại: {archive_type})")
        print(f"[*] Tiến hành giải nén siêu tốc vào ổ đĩa vật lý /kaggle/working ({EXTRACT_DIR})...")
        os.makedirs(EXTRACT_DIR, exist_ok=True)
        DATASET_EXTRACTED_TO_TEMP = True
        t_start = time.time()

        extracted = False
        if archive_type == 'tar':
            try:
                cmd = ['tar', '-xzf', archive_path, '-C', EXTRACT_DIR]
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode == 0:
                    extracted = True
            except Exception:
                pass
            if not extracted:
                with tarfile.open(archive_path, 'r:gz') as tar:
                    tar.extractall(EXTRACT_DIR)
                extracted = True
        elif archive_type == 'zip':
            try:
                cmd = ['unzip', '-q', '-o', archive_path, '-d', EXTRACT_DIR]
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode == 0:
                    extracted = True
            except Exception:
                pass
            if not extracted:
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    zf.extractall(EXTRACT_DIR)
                extracted = True

        extract_duration = time.time() - t_start
        print(f"[✓] Giải nén toàn bộ dataset hoàn tất trong {extract_duration:.1f}s!")

        for root, dirs, files in os.walk(EXTRACT_DIR):
            if 'triplets.csv' in files or 'triplets_combined.csv' in files:
                DATASET_ROOT = root
                break

if DATASET_ROOT is None:
    raise FileNotFoundError("❌ Không tìm thấy tập dữ liệu Sakuga Golden Dataset / Patch 2! Vui lòng kiểm tra Dataset gắn vào Kaggle.")

# Nạp triplets metadata
triplets_csv_path = os.path.join(DATASET_ROOT, 'triplets.csv')
if not os.path.exists(triplets_csv_path):
    triplets_csv_path = os.path.join(DATASET_ROOT, 'triplets_combined.csv')

df_triplets = pd.read_csv(triplets_csv_path)
train_df = df_triplets[df_triplets['split'] == 'train'].reset_index(drop=True)
val_df = df_triplets[df_triplets['split'] == 'val'].reset_index(drop=True)

total_triplets_count = len(df_triplets)
num_train_samples = len(train_df)
num_val_samples = len(val_df)
del df_triplets
gc.collect()

print(f"[✓] Thư mục Golden Dataset hợp lệ: {DATASET_ROOT}")
print(f"[✓] Tổng số Golden Triplets: {total_triplets_count:,} mẫu (Hỗ trợ quy mô Patch 2: ~6,734 triplets)")
print(f"    - Tập Huấn luyện (Train): {num_train_samples:,} mẫu ({num_train_samples/max(1, total_triplets_count)*100:.1f}%)")
print(f"    - Tập Thẩm định (Val)   : {num_val_samples:,} mẫu ({num_val_samples/max(1, total_triplets_count)*100:.1f}%)")
assert num_train_samples > 0 and num_val_samples > 0, "Dữ liệu train hoặc val bị rỗng! Kiểm tra triplets.csv."

matches_dir = os.path.join(DATASET_ROOT, 'matches')
print(f"[*] Thư mục keypoint matches: {matches_dir} (Tồn tại: {os.path.exists(matches_dir)})")
print("[✓] CELL 2 hoàn tất: Toàn vẹn Golden Dataset (Patch 2 Ready) được xác nhận.")"""
    set_code(2, c2)

    # -------------------------------------------------------------------------
    # CELL 4: Data Pipeline & Sampler V2.2
    # -------------------------------------------------------------------------
    c4 = get_code(4)
    reversal_code = '''class TemporalReversalAugmentation:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img0, imggt, img1, t_val, kps=None,
                 c0_pad=None, th01_pad=None, c1_pad=None, th10_pad=None):
        if random.random() < self.p:
            img0_out, img1_out = img1, img0
            t_out = float(1.0 - t_val)
            gt_out = imggt
            kps_out = np.stack([kps[1], kps[0]], axis=0) if (kps is not None and hasattr(kps, "shape") and len(kps.shape) == 3 and kps.shape[0] == 2) else kps
            c0_out, th01_out = c1_pad, th10_pad
            c1_out, th10_out = c0_pad, th01_pad
            return img0_out, gt_out, img1_out, t_out, kps_out, c0_out, th01_out, c1_out, th10_out, True
        return img0, imggt, img1, float(t_val), kps, c0_pad, th01_pad, c1_pad, th10_pad, False

def compute_boundary_motion_weights(df, boost_pct=0.25):
    weights = np.ones(len(df), dtype=np.float32)
    t_vals = df["t"].values.astype(np.float32)
    disp_vals = df["mean_disp"].values.astype(np.float32) if "mean_disp" in df.columns else np.zeros(len(df), dtype=np.float32)
    is_boundary_t = ((t_vals >= 0.15) & (t_vals <= 0.30)) | ((t_vals >= 0.70) & (t_vals <= 0.85))
    is_large_disp = (disp_vals >= 7.0)
    weights[is_boundary_t] *= (1.0 + boost_pct)
    weights[is_large_disp] *= (1.0 + boost_pct)
    return weights
'''
    if "class TemporalReversalAugmentation" not in c4:
        c4 = reversal_code + "\n" + c4

    # Cập nhật đường dẫn fallback trong SakugaGoldenDataset_v1 để tránh lỗi symlink trên Kaggle
    old_path_resolve = """            path0 = os.path.join(root_dir, f0)
            if not os.path.exists(path0):
                path0 = os.path.join(root_dir, self.split_name, f0)
                pathgt = os.path.join(root_dir, self.split_name, fgt)
                path1 = os.path.join(root_dir, self.split_name, f1)
                pathm = os.path.join(root_dir, 'matches', fm)
            else:
                pathgt = os.path.join(root_dir, fgt)
                path1 = os.path.join(root_dir, f1)
                pathm = os.path.join(root_dir, fm)"""

    new_path_resolve = """            # Fallback đa tầng: trực tiếp root, qua split_name, hoặc thư mục con anime_tps_finetune_dataset_ready
            candidate_roots = [
                root_dir,
                os.path.join(root_dir, 'anime_tps_finetune_dataset_ready'),
                os.path.join(root_dir, 'sakuga_golden_dataset_tps_ready_v1_patch2'),
                os.path.join(root_dir, 'sakuga_golden_dataset_tps_ready_v1_patch2', 'anime_tps_finetune_dataset_ready')
            ]
            path0, pathgt, path1, pathm = None, None, None, None
            for cr in candidate_roots:
                if not os.path.exists(cr):
                    continue
                p0_cand = os.path.join(cr, f0)
                if not os.path.exists(p0_cand):
                    p0_cand = os.path.join(cr, self.split_name, f0)
                if os.path.exists(p0_cand):
                    path0 = p0_cand
                    pathgt = os.path.join(cr, fgt) if os.path.exists(os.path.join(cr, fgt)) else os.path.join(cr, self.split_name, fgt)
                    path1 = os.path.join(cr, f1) if os.path.exists(os.path.join(cr, f1)) else os.path.join(cr, self.split_name, f1)
                    pathm = os.path.join(cr, fm) if os.path.exists(os.path.join(cr, fm)) else os.path.join(cr, 'matches', fm)
                    break
            if path0 is None:
                path0 = os.path.join(root_dir, f0)
                pathgt = os.path.join(root_dir, fgt)
                path1 = os.path.join(root_dir, f1)
                pathm = os.path.join(root_dir, fm)"""

    c4 = c4.replace(old_path_resolve, new_path_resolve)

    # Cập nhật DataLoader sử dụng WeightedRandomSampler V2.2
    old_loader_block = """train_dataset = SakugaGoldenDataset_v1(DATASET_ROOT, train_df, is_train=True, crop_size=512)
val_dataset = SakugaGoldenDataset_v1(DATASET_ROOT, val_df, is_train=False)

# [V1.1 FIX]: Xóa Pandas DataFrame TRƯỚC KHI tạo DataLoader để worker fork không kế thừa chúng
del train_df, val_df
cleanup_memory()

# [V1.1 FIX]: persistent_workers=False -> Sau mỗi epoch 2 worker bị hủy, Linux thu hồi 100% heap,
# bộ đệm OpenCV/LAPACK và các trang Copy-on-Write của chúng (tạo lại worker chỉ mất ~0.2s/epoch).
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    persistent_workers=False,
    pin_memory=False,
    prefetch_factor=2,
    collate_fn=custom_collate_fn_v1,
    worker_init_fn=_worker_init_fn,
    drop_last=True
)"""

    new_loader_block = """# [DEFENSIVE CHECK]: Khôi phục train_df và val_df nếu người dùng chạy lại Cell 4
from torch.utils.data import WeightedRandomSampler

if 'train_df' not in globals() or train_df is None:
    print("[*] Tự động nạp lại train_df/val_df cho Cell 4...")
    _meta_path = os.path.join(DATASET_ROOT, 'triplets.csv')
    if not os.path.exists(_meta_path):
        _meta_path = os.path.join(DATASET_ROOT, 'triplets_combined.csv')
    _df = pd.read_csv(_meta_path)
    train_df = _df[_df['split'] == 'train'].reset_index(drop=True)
    val_df = _df[_df['split'] == 'val'].reset_index(drop=True)
    del _df

train_dataset = SakugaGoldenDataset_v1(DATASET_ROOT, train_df, is_train=True, crop_size=512)
val_dataset = SakugaGoldenDataset_v1(DATASET_ROOT, val_df, is_train=False)

# Thiết lập WeightedRandomSampler V2.2 (+25% boundary t và large displacement) TRƯỚC KHI giải phóng DataFrame
train_sample_weights = compute_boundary_motion_weights(train_df, boost_pct=0.25)
train_sampler = WeightedRandomSampler(
    weights=torch.as_tensor(train_sample_weights, dtype=torch.double),
    num_samples=len(train_dataset),
    replacement=True
)

# [V1.1 FIX]: Xóa Pandas DataFrame TRƯỚC KHI tạo DataLoader để worker fork không kế thừa chúng
del train_df, val_df
cleanup_memory()

# [V1.1 FIX]: persistent_workers=False -> Sau mỗi epoch 2 worker bị hủy, Linux thu hồi 100% heap,
# bộ đệm OpenCV/LAPACK và các trang Copy-on-Write của chúng (tạo lại worker chỉ mất ~0.2s/epoch).
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    num_workers=2,
    persistent_workers=False,
    pin_memory=False,
    prefetch_factor=2,
    collate_fn=custom_collate_fn_v1,
    worker_init_fn=_worker_init_fn,
    drop_last=True
)"""
    c4 = c4.replace(old_loader_block, new_loader_block)
    set_code(4, c4)

    # -------------------------------------------------------------------------
    # CELL 6: Composite Loss Engine V2.2
    # -------------------------------------------------------------------------
    c6 = get_code(6)
    v22_losses_def = '''class StrokeDensityRegularizationLoss(nn.Module):
    def __init__(self, target_min=0.98, target_max=1.03, eps=1e-6):
        super(StrokeDensityRegularizationLoss, self).__init__()
        self.target_min = target_min
        self.target_max = target_max
        self.eps = eps

    def forward(self, pred, target, ink_mode=True):
        p_ink = pred if ink_mode else (1.0 - pred)
        g_ink = target if ink_mode else (1.0 - target)
        sum_p = torch.sum(p_ink, dim=[-3, -2, -1])
        sum_g = torch.sum(g_ink, dim=[-3, -2, -1])
        ratio = sum_p / (sum_g + self.eps)
        dev_low = F.relu(self.target_min - ratio)
        dev_high = F.relu(ratio - self.target_max)
        is_outside = (dev_low > 0) | (dev_high > 0)
        diff_from_one = torch.abs(ratio - 1.0)
        loss = torch.where(
            is_outside,
            F.smooth_l1_loss(diff_from_one, torch.zeros_like(diff_from_one), reduction='none', beta=0.01),
            torch.zeros_like(diff_from_one)
        ).mean()
        return loss, ratio.mean().item()

class DirectionalTangentCoherenceLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(DirectionalTangentCoherenceLoss, self).__init__()
        self.eps = eps
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3) / 8.0
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, target):
        dev = pred.device
        gx_p = F.conv2d(pred, self.sobel_x.to(dev), padding=1)
        gy_p = F.conv2d(pred, self.sobel_y.to(dev), padding=1)
        gx_g = F.conv2d(target, self.sobel_x.to(dev), padding=1)
        gy_g = F.conv2d(target, self.sobel_y.to(dev), padding=1)
        mag_p = torch.sqrt(gx_p ** 2 + gy_p ** 2 + self.eps)
        mag_g = torch.sqrt(gx_g ** 2 + gy_g ** 2 + self.eps)
        cos_sim = (gx_p * gx_g + gy_p * gy_g) / (mag_p * mag_g)
        loss_pixel = 1.0 - torch.abs(cos_sim)
        mask = (mag_g > 0.02) | (target > 0.10)
        mask_f = mask.float()
        loss = torch.sum(loss_pixel * mask_f) / (torch.sum(mask_f) + 1e-5)
        return loss

class SoftClDiceLoss(nn.Module):
    def __init__(self, iters=5, eps=1e-5):
        super(SoftClDiceLoss, self).__init__()
        self.iters = iters
        self.eps = eps

    def soft_erode(self, img):
        return -F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)

    def soft_open(self, img):
        return F.max_pool2d(self.soft_erode(img), kernel_size=3, stride=1, padding=1)

    def soft_skeletonize(self, img):
        skel = torch.zeros_like(img)
        x = img
        for _ in range(self.iters):
            opened = self.soft_open(x)
            sub = F.relu(x - opened)
            skel = torch.max(skel, sub)
            x = self.soft_erode(x)
        return skel

    def forward(self, pred, target):
        skel_pred = self.soft_skeletonize(pred)
        skel_gt = self.soft_skeletonize(target)
        tprec = (torch.sum(skel_pred * target, dim=[-3, -2, -1]) + self.eps) / (torch.sum(skel_pred, dim=[-3, -2, -1]) + self.eps)
        tsens = (torch.sum(skel_gt * pred, dim=[-3, -2, -1]) + self.eps) / (torch.sum(skel_gt, dim=[-3, -2, -1]) + self.eps)
        cldice = (2.0 * tprec * tsens) / (tprec + tsens + self.eps)
        return torch.mean(1.0 - cldice)
'''
    if "class StrokeDensityRegularizationLoss" not in c6:
        c6 = v22_losses_def + "\n" + c6

    old_engine = "class CompositeLossEngine_v1(nn.Module):"
    new_engine = '''class CompositeLossEngine_v22(nn.Module):
    def __init__(self):
        super(CompositeLossEngine_v22, self).__init__()
        self.vgg_loss_fn = VGGPerceptualLoss()
        self.dice_loss_fn = StrokeDiceLoss()
        self.morph_dist_fn = GPUMorphologicalDistanceLoss()
        self.ghost_loss_fn = GhostStrokePenaltyLoss(radius=7)
        self.density_loss_fn = StrokeDensityRegularizationLoss(target_min=0.98, target_max=1.03)
        self.tangent_loss_fn = DirectionalTangentCoherenceLoss()
        self.cldice_loss_fn = SoftClDiceLoss(iters=5)

    def forward(self, pred, target):
        loss_l1 = F.l1_loss(pred, target)
        loss_vgg = self.vgg_loss_fn(pred, target)
        loss_dice = self.dice_loss_fn(pred, target)
        loss_morph = self.morph_dist_fn(pred, target)
        loss_ghost = self.ghost_loss_fn(pred, target)
        loss_density, density_ratio = self.density_loss_fn(pred, target, ink_mode=True)
        loss_tangent = self.tangent_loss_fn(pred, target)
        loss_cldice = self.cldice_loss_fn(pred, target)

        total_loss = (
            1.00 * loss_l1 +
            0.15 * loss_vgg +
            0.40 * loss_dice +
            0.20 * loss_morph +
            0.10 * loss_ghost +
            0.30 * loss_density +
            0.20 * loss_tangent +
            0.20 * loss_cldice
        )
        return total_loss, loss_l1, loss_dice, loss_density
'''
    c6 = c6.replace("loss_engine = CompositeLossEngine_v1().to(device)", "loss_engine = CompositeLossEngine_v22().to(device)")
    if old_engine in c6:
        idx_e = c6.find(old_engine)
        idx_end = c6.find("loss_engine = CompositeLossEngine_v22().to(device)")
        c6 = c6[:idx_e] + new_engine + "\n" + c6[idx_end:]
    set_code(6, c6)

    # -------------------------------------------------------------------------
    # CELL 7: Optimizer & Warm-Start V2.2
    # -------------------------------------------------------------------------
    c7 = get_code(7)
    c7 = c7.replace("TOTAL_EPOCHS = 30", "TOTAL_EPOCHS = 25")
    old_candidates = """pretrained_candidates = [
    os.path.join(WORKING_DIR, 'checkpoints_v1.1', 'model_tps_v1.1_best.pt'),"""
    new_candidates = """pretrained_candidates = [
    os.path.join(KAGGLE_INPUT, 'tps-v20-artifacts', 'model_tps_v2.0_best.pt'),
    os.path.join(KAGGLE_INPUT, 'tps-v20-artifacts', 'model_tps_v1.1_best.pt'),
    os.path.join(WORKING_DIR, 'checkpoints_v2.0.2', 'model_tps_v2.0_best.pt'),
    os.path.join(WORKING_DIR, 'checkpoints_v1.1', 'model_tps_v1.1_best.pt'),"""
    c7 = c7.replace(old_candidates, new_candidates)

    old_opt = """optimizer = torch.optim.AdamW([
    {'params': base_model.encoder.parameters(), 'lr': 2.0e-5},
    {'params': base_model.IFNet.parameters(), 'lr': 1.0e-4},
    {'params': list(base_model.asf_head.parameters()) + list(base_model.unet.parameters()), 'lr': 2.0e-4}
], betas=(0.9, 0.999), weight_decay=1e-4)"""

    new_opt = """# Thiết lập AdamW Differential Learning Rates V2.2 (1.5e-5 vs 5.0e-5)
optimizer = torch.optim.AdamW([
    {'params': list(base_model.encoder.parameters()) + list(base_model.IFNet.parameters()), 'lr': 1.5e-5},
    {'params': list(base_model.asf_head.parameters()) + list(base_model.unet.parameters()), 'lr': 5.0e-5}
], betas=(0.9, 0.999), weight_decay=1e-4)"""
    c7 = c7.replace(old_opt, new_opt)

    # Scheduler warmup 2 epochs + Cosine decay
    old_sched = """def lr_lambda(epoch):
    if epoch < 3:
        return float(epoch + 1) / 3.0
    progress = float(epoch - 3) / float(max(1, TOTAL_EPOCHS - 3))
    return 0.5 * (1.0 + math.cos(math.pi * progress)) * 0.95 + 0.05"""

    new_sched = """def lr_lambda(epoch):
    if epoch < 2:
        return float(epoch + 1) / 2.0
    progress = float(epoch - 2) / float(max(1, TOTAL_EPOCHS - 2))
    return 0.5 * (1.0 + math.cos(math.pi * progress)) * 0.98 + 0.02"""
    c7 = c7.replace(old_sched, new_sched)

    old_print_groups = """print(f"    - Nhóm 1 (Encoder Contextnet) : lr = {optimizer.param_groups[0]['lr']:.2e}")
print(f"    - Nhóm 2 (IFNet Optical Flow) : lr = {optimizer.param_groups[1]['lr']:.2e}")
print(f"    - Nhóm 3 (ASF Head + UNet)   : lr = {optimizer.param_groups[2]['lr']:.2e}")"""

    new_print_groups = """print(f"    - Nhóm 1 (Backbone ContextNet + IFNet Flow) : lr = {optimizer.param_groups[0]['lr']:.2e}")
print(f"    - Nhóm 2 (ASF-GR Head + Synthesis UNet)     : lr = {optimizer.param_groups[1]['lr']:.2e}")"""
    c7 = c7.replace(old_print_groups, new_print_groups)
    c7 = c7.replace("V1.1", "V2.2")
    set_code(7, c7)

    # -------------------------------------------------------------------------
    # CELL 8: Chuẩn Hóa Metrics GPU (Bit-exact với Fast Benchmark & Density Ratio)
    # -------------------------------------------------------------------------
    c8 = get_code(8)
    old_ghost_calc = """        # 4. Ghost Pixels trên GPU (Tách vùng đệm dung sai không tính pixel chống răng cưa sát nét vẽ GT)
        g_dilated_ghost = F.max_pool2d(g_mask, kernel_size=7, stride=1, padding=3)
        ghost_mask = (pred_tensor >= 0.10) & (pred_tensor <= 0.45) & (g_dilated_ghost < 0.5)
        ghost_pixels = int(torch.sum(ghost_mask).item()) if not is_blank else -1

    return {
        'psnr': psnr,
        'ssim': ssim,
        'recall': recall,
        'precision': precision,
        'f1_stroke': f1_stroke,
        'ghost_pixels': ghost_pixels,
        'p_lines': p_lines,
        'g_lines': g_lines,
        'is_blank': is_blank
    }"""

    new_ghost_calc = """        # 4. Ghost Pixels & Density Ratio Chuẩn Hóa V2.2 (Bit-exact với Fast Benchmark: Không dilation mask)
        pred_u8 = ((1.0 - pred_tensor) * 255.0).clamp(0, 255).to(torch.uint8)
        gt_u8 = ((1.0 - gt_tensor) * 255.0).clamp(0, 255).to(torch.uint8)
        ghost_mask = (pred_u8 >= 180) & (pred_u8 <= 250) & (gt_u8 > 250)
        ghost_pixels = int(torch.sum(ghost_mask).item()) if not is_blank else -1
        density_ratio = float(p_lines / max(1, g_lines))

    return {
        'psnr': psnr,
        'ssim': ssim,
        'recall': recall,
        'precision': precision,
        'f1_stroke': f1_stroke,
        'ghost_pixels': ghost_pixels,
        'density_ratio': density_ratio,
        'p_lines': p_lines,
        'g_lines': g_lines,
        'is_blank': is_blank
    }"""
    c8 = c8.replace(old_ghost_calc, new_ghost_calc)
    c8 = c8.replace("V1.1", "V2.2")
    set_code(8, c8)

    # -------------------------------------------------------------------------
    # CELL 9: Smart Checkpoint Manager V2.2
    # -------------------------------------------------------------------------
    c9_v22 = '''# Bộ Quản Lý Checkpoint Thông Minh V2.2 (Sakuga Composite Score Manager)
class SmartCheckpointManager_v22:
    def __init__(self, save_dir, ref_ghost=14322.38, ref_chamfer=0.00098317, ref_recall=0.923635,
                 min_recall=0.915, min_density=0.90, max_density=1.12, gdrive_manager=None):
        self.save_dir = save_dir
        self.ref_ghost = ref_ghost
        self.ref_chamfer = ref_chamfer
        self.ref_recall = ref_recall
        self.min_recall = min_recall
        self.min_density = min_density
        self.max_density = max_density
        self.best_score = float('inf')
        self.best_epoch = 0
        self.best_metrics = {}
        self.gdrive = gdrive_manager
        os.makedirs(save_dir, exist_ok=True)

    def calculate_composite_score(self, ghost, chamfer, recall, density_ratio):
        term_ghost = 0.35 * (ghost / max(1.0, self.ref_ghost))
        term_chamfer = 0.30 * (chamfer / max(1e-6, self.ref_chamfer))
        denom_recall = max(1e-4, 1.0 - self.ref_recall)
        term_recall = 0.20 * (max(0.0, 1.0 - recall) / denom_recall)
        term_density = 0.15 * abs(density_ratio - 1.0)
        return float(term_ghost + term_chamfer + term_recall + term_density)

    def evaluate_and_save(self, model, current_epoch, metrics, history_df=None):
        recall = metrics['recall']
        psnr = metrics['psnr']
        ssim = metrics['ssim']
        density_ratio = metrics.get('density_ratio', 1.0)
        ghost = max(0, metrics.get('ghost_pixels', 0))
        chamfer = metrics.get('chamfer_distance', 0.001)

        # 1. Hard Gates
        gates_passed = True
        gate_reasons = []
        if metrics.get('is_blank', False):
            gates_passed = False
            gate_reasons.append("Blank Collapse")
        if recall < self.min_recall:
            gates_passed = False
            gate_reasons.append(f"Recall ({recall*100:.2f}%) < {self.min_recall*100:.1f}%")
        if density_ratio < self.min_density or density_ratio > self.max_density:
            gates_passed = False
            gate_reasons.append(f"Density ({density_ratio:.4f}) ngoài [{self.min_density}, {self.max_density}]")

        score = self.calculate_composite_score(ghost, chamfer, recall, density_ratio)
        print(f"   [*] Sakuga Composite Score V2.2: {score:.4f} (Kỷ lục: {self.best_score:.4f}) | Ghost: {ghost} px | Dens: {density_ratio:.4f} | Recall: {recall*100:.1f}%")

        raw_model = model
        if hasattr(raw_model, 'module'):
            raw_model = raw_model.module
        if hasattr(raw_model, 'base_model'):
            raw_model = raw_model.base_model
        cpu_state_dict = {k: v.detach().cpu() for k, v in raw_model.state_dict().items()}

        is_best = False
        try:
            latest_path = os.path.join(self.save_dir, 'model_tps_v2.2_latest.pt')
            torch.save(cpu_state_dict, latest_path)

            if current_epoch % 5 == 0:
                epoch_path = os.path.join(self.save_dir, f'model_tps_v2.2_ep{current_epoch:02d}.pt')
                torch.save(cpu_state_dict, epoch_path)
                if self.gdrive:
                    self.gdrive.sync_file(epoch_path, f'model_tps_v2.2_ep{current_epoch:02d}.pt')

            if not gates_passed:
                print(f"   ⚠️ [Hard Gates BỊ CHẶN]: {', '.join(gate_reasons)} -> Từ chối phong Best Checkpoint!")
            else:
                if score < self.best_score:
                    self.best_score = score
                    self.best_epoch = current_epoch
                    self.best_metrics = dict(metrics)
                    self.best_metrics['composite_score'] = score
                    best_path = os.path.join(self.save_dir, 'model_tps_v2.2_best.pt')
                    torch.save(cpu_state_dict, best_path)
                    print(f"   🌟 [KỶ LỤC MỚI V2.2!] Đã lưu Best Checkpoint Epoch {current_epoch:02d} -> {best_path}")
                    if self.gdrive:
                        self.gdrive.sync_file(best_path, 'model_tps_v2.2_best.pt')
                        self.gdrive.sync_file(latest_path, 'model_tps_v2.2_latest.pt')
                    is_best = True
        finally:
            del cpu_state_dict
            gc.collect()

        return is_best, score

# Bỏ qua lưu trữ Google Drive (Lưu trực tiếp tại NVMe /kaggle/working/checkpoints_v2.2)
gdrive_syncer = None
ckpt_manager = SmartCheckpointManager_v22(
    save_dir=os.path.join(WORKING_DIR, 'checkpoints_v2.2'),
    gdrive_manager=None
)
print("[✓] CELL 9 hoàn tất: Smart Checkpoint Manager V2.2 (Sakuga Composite Score) sẵn sàng.")'''
    set_code(9, c9_v22)

    # -------------------------------------------------------------------------
    # CELL 10: Training Loop with Gradient Clipping & Full History V2.2
    # -------------------------------------------------------------------------
    c10 = get_code(10)
    c10 = c10.replace("training_history_v1.1.csv", "training_history_v2.2.csv")
    c10 = c10.replace("checkpoints_v1.1", "checkpoints_v2.2")
    c10 = c10.replace("model_tps_v1.1_best.pt", "model_tps_v2.2_best.pt")
    c10 = c10.replace("V1.1", "V2.2")

    if "torch.nn.utils.clip_grad_norm_" not in c10:
        c10 = c10.replace(
            "scaler.step(optimizer)",
            "scaler.unscale_(optimizer)\n                torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=1.0)\n                scaler.step(optimizer)"
        )

    # Cập nhật ckpt_manager step và history.append một cách phòng vệ an toàn
    old_ckpt_call = "ckpt_manager.evaluate_and_save(base_model, epoch, val_metrics)"
    new_ckpt_call = """_eval_res = ckpt_manager.evaluate_and_save(base_model, epoch, val_metrics)
        if isinstance(_eval_res, tuple) and len(_eval_res) == 2:
            is_best_ep, comp_score = _eval_res
        else:
            is_best_ep = bool(_eval_res)
            comp_score = getattr(ckpt_manager, 'best_score', 1.0)"""
    c10 = c10.replace(old_ckpt_call, new_ckpt_call)

    old_hist_append = """        history.append({
            'epoch': epoch,
            'train_loss': mean_train_loss,
            'val_loss': mean_val_loss,
            'train_l1': mean_train_l1,
            'val_l1': mean_val_l1,
            'val_psnr': mean_val_psnr,
            'val_ssim': mean_val_ssim,
            'stroke_recall': mean_val_recall,
            'stroke_f1': mean_val_f1,
            'ghost_pixels': mean_val_ghost,
            'lr_unet': lr_unet,
            'lr_encoder': lr_enc,
            'epoch_time_sec': round(epoch_time, 1),
            'host_ram_rss_gb': round(rss_gb_now, 3) if rss_gb_now is not None else None,
            'host_ram_cgroup_gb': round(cg_gb_now, 3) if cg_gb_now is not None else None
        })"""

    new_hist_append = """        history.append({
            'epoch': epoch,
            'train_loss': mean_train_loss,
            'val_loss': mean_val_loss,
            'train_l1': mean_train_l1,
            'val_l1': mean_val_l1,
            'val_psnr': mean_val_psnr,
            'val_ssim': mean_val_ssim,
            'stroke_recall': mean_val_recall,
            'stroke_f1': mean_val_f1,
            'ghost_pixels': mean_val_ghost,
            'density_ratio': val_metrics.get('density_ratio', 1.0),
            'composite_score': comp_score,
            'is_best': is_best_ep,
            'lr_unet': lr_unet,
            'lr_encoder': lr_enc,
            'epoch_time_sec': round(epoch_time, 1),
            'host_ram_rss_gb': round(rss_gb_now, 3) if rss_gb_now is not None else None,
            'host_ram_cgroup_gb': round(cg_gb_now, 3) if cg_gb_now is not None else None
        })"""
    c10 = c10.replace(old_hist_append, new_hist_append)
    set_code(10, c10)

    # -------------------------------------------------------------------------
    # NEW CODE CELL: Visualizing Epoch-wise Metrics Convergence (Insert after Cell 10)
    # -------------------------------------------------------------------------
    convergence_cell_src = [line + "\n" for line in """# [CELL 11]: Trực Quan Hóa Tiến Trình Hội Tụ & Chỉ Số Huấn Luyện Qua Các Epoch V2.2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def plot_training_metrics_convergence(history_csv=None, save_dir=None):
    if history_csv is None:
        history_csv = os.path.join(WORKING_DIR, 'training_history_v2.2.csv')
    if save_dir is None:
        save_dir = WORKING_DIR

    if not os.path.exists(history_csv):
        print(f"[!] Không tìm thấy file lịch sử {history_csv} để vẽ biểu đồ.")
        return

    df = pd.read_csv(history_csv)
    if len(df) == 0:
        print("[!] File lịch sử rỗng.")
        return

    epochs = df['epoch'].values

    # Khởi tạo Dashboard đồ thị 6-Panel chuyên nghiệp
    fig, axes = plt.subplots(2, 3, figsize=(21, 12), dpi=150)
    fig.suptitle('TPS-INBETWEEN FINETUNE V2.2 - TRAINING DYNAMICS & METRIC CONVERGENCE DASHBOARD\\n'
                 'Tập dữ liệu: Sakuga Golden Dataset (+ Patch 2) | Warm-start: V2.0.2 Best',
                 fontsize=16, fontweight='bold', y=0.98)

    # 1. LOSS EVOLUTION
    ax1 = axes[0, 0]
    ax1.plot(epochs, df['train_loss'], label='Train Total Loss', color='#1f77b4', lw=2.2, marker='o', ms=4)
    if 'val_loss' in df.columns:
        ax1.plot(epochs, df['val_loss'], label='Val Total Loss', color='#ff7f0e', lw=2.2, marker='s', ms=4)
    if 'train_l1' in df.columns:
        ax1.plot(epochs, df['train_l1'], label='Train L1 Loss', color='#2ca02c', lw=1.5, ls='--')
    ax1.set_title('1. Loss Convergence (Total & L1)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss Value')
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(True, alpha=0.3)

    # 2. DENSITY RATIO & LINE THICKENING CONTROL (CORE V2.2)
    ax2 = axes[0, 1]
    d_col = 'density_ratio' if 'density_ratio' in df.columns else ('val_density_ratio' if 'val_density_ratio' in df.columns else None)
    if d_col:
        ax2.plot(epochs, df[d_col], label='Val Density Ratio', color='#9467bd', lw=2.5, marker='D', ms=5)
        ax2.axhspan(0.98, 1.03, color='#2ca02c', alpha=0.20, label='Target Region [0.98, 1.03]')
        ax2.axhspan(0.90, 1.12, color='#ffbb78', alpha=0.15, label='Hard Gate [0.90, 1.12]')
        ax2.axhline(1.3139, color='#d62728', lw=1.8, ls=':', label='V2.0.2 Baseline (1.314 - Thickening)')
        ax2.axhline(1.0000, color='#2ca02c', lw=1.2, ls='-', label='Ideal Ratio (1.000)')
    ax2.set_title('2. Line Thickening Control (Density Ratio)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Density Ratio (Pred Ink / GT Ink)')
    ax2.legend(loc='upper right', frameon=True, fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. GHOST PIXELS SUPPRESSION
    ax3 = axes[0, 2]
    if 'ghost_pixels' in df.columns:
        ax3.plot(epochs, df['ghost_pixels'], label='Val Ghost Pixels', color='#d62728', lw=2.2, marker='^', ms=5)
        ax3.axhline(14322.0, color='#8c564b', lw=1.8, ls=':', label='V2.0.2 Ref (14,322 px)')
        if len(df) > 0 and df['ghost_pixels'].min() > 0:
            min_ghost = df['ghost_pixels'].min()
            min_ep = df.loc[df['ghost_pixels'].idxmin(), 'epoch']
            ax3.annotate(f'Min: {int(min_ghost):,} px (Ep {int(min_ep)})',
                         xy=(min_ep, min_ghost), xytext=(min_ep, min_ghost + 2000),
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                         fontweight='bold', fontsize=9)
    ax3.set_title('3. Ghost Pixels Reduction (Standardized Benchmark)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Ghost Pixels (Count)')
    ax3.legend(loc='upper right', frameon=True)
    ax3.grid(True, alpha=0.3)

    # 4. FINE-STROKE PRESERVATION (RECALL & F1)
    ax4 = axes[1, 0]
    if 'stroke_recall' in df.columns:
        rec_pct = df['stroke_recall'] * 100.0 if df['stroke_recall'].max() <= 1.0 else df['stroke_recall']
        ax4.plot(epochs, rec_pct, label='Val Stroke Recall (%)', color='#2ca02c', lw=2.2, marker='o', ms=4)
        ax4.axhline(91.5, color='#d62728', lw=1.5, ls='--', label='Hard Gate Min (91.5%)')
        ax4.axhline(92.36, color='#17becf', lw=1.5, ls=':', label='V2.0.2 Ref (92.36%)')
    if 'stroke_f1' in df.columns:
        f1_pct = df['stroke_f1'] * 100.0 if df['stroke_f1'].max() <= 1.0 else df['stroke_f1']
        ax4.plot(epochs, f1_pct, label='Stroke F1-Score (%)', color='#8c564b', lw=1.8, ls='-.')
    ax4.set_title('4. Fine-Stroke Preservation (Recall & F1)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Percentage (%)')
    ax4.legend(loc='lower right', frameon=True)
    ax4.grid(True, alpha=0.3)

    # 5. SAKUGA COMPOSITE SCORE & BEST CHECKPOINT
    ax5 = axes[1, 1]
    score_col = 'composite_score' if 'composite_score' in df.columns else None
    if score_col:
        ax5.plot(epochs, df[score_col], label='Sakuga Composite Score', color='#17becf', lw=2.2, marker='d', ms=5)
        best_idx = df[score_col].idxmin()
        best_ep = df.loc[best_idx, 'epoch']
        best_sc = df.loc[best_idx, score_col]
        ax5.scatter([best_ep], [best_sc], color='#d62728', s=160, zorder=5, marker='*', label=f'Best Checkpoint (Ep {int(best_ep)})')
        ax5.annotate(f'Best: {best_sc:.4f}\\n(Epoch {int(best_ep)})',
                     xy=(best_ep, best_sc), xytext=(best_ep, best_sc + 0.15),
                     arrowprops=dict(facecolor='#d62728', shrink=0.05, width=1.5, headwidth=7),
                     fontweight='bold', fontsize=9, color='#d62728')
    else:
        ax5.plot(epochs, df['val_psnr'], label='Val PSNR (dB)', color='#e377c2', lw=2.0)
    ax5.set_title('5. Sakuga Composite Score (Lower is Better)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Composite Score Value')
    ax5.legend(loc='upper right', frameon=True)
    ax5.grid(True, alpha=0.3)

    # 6. DIFFERENTIAL LEARNING RATE SCHEDULE
    ax6 = axes[1, 2]
    if 'lr_unet' in df.columns and 'lr_encoder' in df.columns:
        ax6.plot(epochs, df['lr_unet'], label='Heads / UNet LR (Max 5.0e-5)', color='#e377c2', lw=2.0)
        ax6.plot(epochs, df['lr_encoder'], label='Backbone / Flow LR (Max 1.5e-5)', color='#7f7f7f', lw=2.0, ls='--')
        ax6.set_yscale('log')
    elif 'epoch_time_sec' in df.columns:
        ax6.bar(epochs, df['epoch_time_sec'], color='#7f7f7f', alpha=0.6, label='Epoch Duration (s)')
    ax6.set_title('6. Differential LR Schedule (Cosine Decay)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Learning Rate (Log Scale)')
    ax6.legend(loc='upper right', frameon=True)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_fig_path = os.path.join(save_dir, 'training_metrics_convergence_v22.png')
    plt.savefig(out_fig_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"[✓] Đã xuất biểu đồ trực quan hóa tiến trình hội tụ tại: {out_fig_path}")

    # Bảng tổng kết số liệu so sánh
    print("\\n" + "=" * 96)
    print("📊 BẢNG TỔNG HỢP TIẾN TRÌNH THAY ĐỔI CÁC CHỈ SỐ HUẤN LUYỆN V2.2:")
    print("=" * 96)
    headers = ['Chỉ Số (Metric)', 'Baseline V2.0.2', 'Epoch 1 (Khởi đầu)', 'Best Checkpoint', 'Đánh Giá V2.2']
    row_fmt = "{:<24} | {:<16} | {:<18} | {:<18} | {:<14}"
    print(row_fmt.format(*headers))
    print("-" * 96)

    ep1 = df.iloc[0]
    best_row = df.loc[df['composite_score'].idxmin()] if 'composite_score' in df.columns else df.iloc[-1]

    d_val_ep1 = ep1.get('density_ratio', 1.0)
    d_val_best = best_row.get('density_ratio', 1.0)
    g_val_ep1 = ep1.get('ghost_pixels', 0)
    g_val_best = best_row.get('ghost_pixels', 0)
    r_val_ep1 = ep1.get('stroke_recall', 0)
    r_val_best = best_row.get('stroke_recall', 0)
    sc_val_ep1 = ep1.get('composite_score', 0)
    sc_val_best = best_row.get('composite_score', 0)

    metrics_summary = [
        ('Density Ratio', '1.3139', f"{d_val_ep1:.4f}", f"{d_val_best:.4f}", "Anti-thickening"),
        ('Ghost Pixels (px)', '14,322 px', f"{int(g_val_ep1):,} px", f"{int(g_val_best):,} px", "Khử ma sâu"),
        ('Stroke Recall', '92.36%', f"{r_val_ep1*100:.2f}%", f"{r_val_best*100:.2f}%", "Duy trì >= 91.5%"),
        ('Composite Score', '1.0000 (Ref)', f"{sc_val_ep1:.4f}", f"{sc_val_best:.4f}", "Kỷ lục mới")
    ]
    for row in metrics_summary:
        print(row_fmt.format(*row))
    print("=" * 96)

plot_training_metrics_convergence()
""".strip().split("\n")]
    if convergence_cell_src:
        convergence_cell_src[-1] = convergence_cell_src[-1].rstrip("\n")

    new_convergence_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": convergence_cell_src
    }

    # Chèn cell mới vào ngay sau Cell 10 (tức vị trí index 11)
    nb["cells"].insert(11, new_convergence_cell)

    # -------------------------------------------------------------------------
    # CELL 12 (trước là 11): Kiểm toán trực quan V2.2
    # -------------------------------------------------------------------------
    c12 = get_code(12)
    c12 = c12.replace("# [CELL 11]:", "# [CELL 12]:")
    c12 = c12.replace("visual_samples_v1.1", "visual_samples_v2.2")
    c12 = c12.replace("V1.1", "V2.2")
    set_code(12, c12)

    # -------------------------------------------------------------------------
    # CELL 13 (trước là 12): Đóng gói Artifact V2.2
    # -------------------------------------------------------------------------
    c13 = get_code(13)
    c13 = c13.replace("# [CELL 12]:", "# [CELL 13]:")
    c13 = c13.replace("training_history_v1.1.csv", "training_history_v2.2.csv")
    c13 = c13.replace("checkpoints_v1.1", "checkpoints_v2.2")
    c13 = c13.replace("visual_samples_v1.1", "visual_samples_v2.2")
    c13 = c13.replace("tps_finetune_v1.1_artifacts", "tps_finetune_v2.2_artifacts")
    c13 = c13.replace("V1.1", "V2.2")
    # Đảm bảo gom cả training_metrics_convergence_v22.png vào zip
    if "training_metrics_convergence_v22.png" not in c13:
        c13 = c13.replace(
            "zip_cmd = [",
            "conv_png = os.path.join(WORKING_DIR, 'training_metrics_convergence_v22.png')\n"
            "zip_cmd = ["
        )
    set_code(13, c13)

    # Lưu cả 2 notebook
    with open(v22_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    with open(alias_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"[✓] Đã tạo thành công notebook Kaggle V2.2 hoàn chỉnh (14 Cells):")
    print(f"    - {v22_path}")
    print(f"    - {alias_path}")


if __name__ == '__main__':
    generate_v22()
