#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_v11_notebook.py
Script tự động khởi tạo notebook huấn luyện Kaggle V1.1:
kaggle/triplet_05_v1.1_ghost_purge_fixed_kaggle.ipynb
và alias kaggle/finetune-v1-1.ipynb.
Nâng cấp toàn diện:
- Triệt tiêu 100% Copy-on-Write (CoW) trong Dataset & DataLoader
- Tránh giải nén vào tmpfs (/tmp), chuyển sang /kaggle/temp
- Mở khóa biên độ detail_add [0.0, 1.0] và unfreeze Contextnet encoder
- Tốc độ học phân tầng Differential LR (Encoder: 2e-5, IFNet: 1e-4, ASF/UNet: 2e-4)
- Bán kính an toàn Ghost Loss radius=7 và tăng trọng số Stroke Dice Loss lên 0.50
- Thêm Dilated Stroke F1 vào calculate_metrics_v1 và Composite Score mới
- Đồng nhất tập Validation val_loader_full cho toàn bộ 30 epochs
"""

import os
import sys
import json
import ast

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

def generate():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    v10_path = os.path.join(script_dir, "triplet_04_v1.0_golden_v1_kaggle.ipynb")
    v11_path = os.path.join(script_dir, "triplet_05_v1.1_ghost_purge_fixed_kaggle.ipynb")
    alias_path = os.path.join(script_dir, "finetune-v1-1.ipynb")

    if not os.path.exists(v10_path):
        raise FileNotFoundError(f"Không tìm thấy notebook v1.0 tại: {v10_path}")

    with open(v10_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    def get_code(cell_idx):
        return "".join(nb["cells"][cell_idx]["source"])

    def set_code(cell_idx, text):
        lines = [line + "\n" for line in text.strip().split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        nb["cells"][cell_idx]["source"] = lines

    # -------------------------------------------------------------------------
    # CELL 0: Markdown Header
    # -------------------------------------------------------------------------
    c0 = get_code(0)
    c0 = c0.replace("PHIÊN BẢN V1.0 (PRODUCTION RELEASE CANDIDATE V1.0)", "PHIÊN BẢN V1.1 (PRODUCTION RELEASE V1.1)")
    c0 = c0.replace("(Production Ready)", "(Production Ready V1.1)")
    c0 = c0.replace(
        "triệt tiêu hoàn toàn bóng ma chồng hình (Zero Ghosting Artifacts) và bảo toàn tính toàn vẹn 100% của nét vẽ hoạt hình.",
        "triệt tiêu hoàn toàn rò rỉ RAM trên Kaggle Linux Container (CoW-Proof, No tmpfs Extract), mở khóa biên độ vẽ bù nét khuyết `detail_add`, tinh chỉnh đa tầng Contextnet/IFNet/UNet (Differential Learning Rates) và hiệu chỉnh hệ thống chấm điểm toàn diện với Dilated Stroke F1."
    )
    c0 = c0.replace("ASF-GR Engine", "ASF-GR Engine V1.1")
    c0 = c0.replace("Ghost Penalty", "Calibrated Ghost Penalty")
    set_code(0, c0)

    # -------------------------------------------------------------------------
    # CELL 1: Environment & Active Garbage Collection
    # -------------------------------------------------------------------------
    c1 = get_code(1)
    c1 = c1.replace("[CELL 1]: Cấu hình Môi trường, Phần cứng & Trình Dọn Rác Bộ Nhớ Chủ Động", "[CELL 1]: Cấu hình Môi trường, Phần cứng & Trình Dọn Rác Bộ Nhớ Chủ Động (V1.1)")
    c1 = c1.replace("mp.set_sharing_strategy('file_system')", "mp.set_sharing_strategy('file_descriptor')")
    c1 = c1.replace("CELL 1 hoàn tất: Môi trường và hàm quản lý bộ nhớ sâu sẵn sàng.", "CELL 1 hoàn tất: Môi trường và hàm quản lý bộ nhớ sâu sẵn sàng (V1.1).")
    set_code(1, c1)

    # -------------------------------------------------------------------------
    # CELL 2: Locate & Extract Dataset (Tránh /tmp tmpfs)
    # -------------------------------------------------------------------------
    c2 = get_code(2)
    c2 = c2.replace("[CELL 2]: Tự Động Định Vị, Giải Nén Siêu Tốc & Thẩm Định Toàn Vẹn Golden Dataset V1.0", "[CELL 2]: Tự Động Định Vị, Giải Nén Siêu Tốc & Thẩm Định Toàn Vẹn Golden Dataset V1.1")
    old_tmp_block = """# [PHƯƠNG ÁN 1: ZIP-IN-ZIP CONTAINER & GIẢI NÉN NVMe SCRATCHPAD]
# Ưu tiên giải nén vào /tmp trên Kaggle (ổ NVMe cục bộ siêu tốc, không chiếm dung lượng /kaggle/working
# và không bị Kaggle mất thời gian nén lại 16,660 ảnh khi Save Version/Commit notebook).
TMP_DIR = '/tmp' if os.path.exists('/tmp') else os.path.join(WORKING_DIR, 'tmp')
EXTRACT_DIR = os.path.join(TMP_DIR, 'sakuga_golden_dataset_tps_ready')"""

    new_tmp_block = """# [PHƯƠNG ÁN TỐI ƯU V1.1: TRÁNH TMPFS RAM DISK /tmp]
# Ưu tiên sử dụng /kaggle/temp (nằm trên ổ đĩa ảo root của container, KHÔNG nằm trong /tmp tmpfs
# tránh mất 30GB RAM của Kaggle, đồng thời không nằm trong /kaggle/working tránh bị nén lại khi Save Version)
if os.path.exists('/kaggle'):
    TMP_DIR = '/kaggle/temp'
else:
    TMP_DIR = os.path.join(WORKING_DIR, 'temp')

os.makedirs(TMP_DIR, exist_ok=True)
EXTRACT_DIR = os.path.join(TMP_DIR, 'sakuga_golden_dataset_tps_ready')"""

    c2 = c2.replace(old_tmp_block, new_tmp_block)
    c2 = c2.replace(
        "search_locations = [KAGGLE_INPUT, EXTRACT_DIR, WORKING_DIR, './data/processed', '../data/processed', '.']",
        "search_locations = [KAGGLE_INPUT, EXTRACT_DIR, TMP_DIR, WORKING_DIR, './data/processed', '../data/processed', '.']"
    )
    c2 = c2.replace("CELL 2 hoàn tất: Toàn vẹn Golden Dataset được xác nhận.", "CELL 2 hoàn tất: Toàn vẹn Golden Dataset được xác nhận (Tránh tmpfs /tmp).")
    set_code(2, c2)

    # CELL 3: Giữ nguyên 100% kiến trúc Warp và Vectorized TPS Batch Solver

    # -------------------------------------------------------------------------
    # CELL 4: Zero CoW Dataset & Lean DataLoaders
    # -------------------------------------------------------------------------
    c4_v11 = """# [CELL 4]: Dataset & DataLoader An Toàn Bộ Nhớ Chia Sẻ & Giải TPS Nền Cho Golden Dataset V1.1
def _worker_init_fn(worker_id):
    configure_glibc_allocator()
    torch.set_num_threads(1)
    cv2.setNumThreads(0)
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)

class SakugaGoldenDataset_v1(Dataset):
    def __init__(self, root_dir, df_split, is_train=True, crop_size=512):
        self.root_dir = root_dir
        self.is_train = is_train
        self.crop_size = crop_size
        self.split_name = 'train' if is_train else 'val'
        self._access_count = 0
        self.length = len(df_split)

        # 1. Mã hóa toàn bộ đường dẫn ảnh và match_file thành mảng NumPy bytes cố định (dtype='S256')
        # Loại bỏ hoàn toàn mảng Python list / chuỗi Unicode động để triệt tiêu Copy-on-Write (CoW)
        p0_bytes = []
        pgt_bytes = []
        p1_bytes = []
        pmatch_bytes = []

        for _, row in df_split.iterrows():
            f0 = str(row['frame0']).lstrip('/\\\\')
            fgt = str(row['frame_gt']).lstrip('/\\\\')
            f1 = str(row['frame1']).lstrip('/\\\\')
            fm = str(row['match_file']).lstrip('/\\\\')

            path0 = os.path.join(root_dir, f0)
            if not os.path.exists(path0):
                path0 = os.path.join(root_dir, self.split_name, f0)
                pathgt = os.path.join(root_dir, self.split_name, fgt)
                path1 = os.path.join(root_dir, self.split_name, f1)
                pathm = os.path.join(root_dir, 'matches', fm)
            else:
                pathgt = os.path.join(root_dir, fgt)
                path1 = os.path.join(root_dir, f1)
                pathm = os.path.join(root_dir, fm)

            p0_bytes.append(path0.encode('utf-8'))
            pgt_bytes.append(pathgt.encode('utf-8'))
            p1_bytes.append(path1.encode('utf-8'))
            pmatch_bytes.append(pathm.encode('utf-8'))

        self.p0_arr = np.array(p0_bytes, dtype='S256')
        self.pgt_arr = np.array(pgt_bytes, dtype='S256')
        self.p1_arr = np.array(p1_bytes, dtype='S256')
        self.pmatch_arr = np.array(pmatch_bytes, dtype='S256')
        self.t_arr = df_split['t'].to_numpy(dtype=np.float32, copy=True)
        self.clips_arr = np.array([str(c).encode('utf-8') for c in df_split['clip_id']], dtype='S64')

        # 2. [TỐI ƯU HÓA CO-W]: Cấp phát 1 mảng NumPy 4D liền mạch duy nhất (length, 2, MAX_KPS_CACHE, 2)
        # và 1 mảng 1D kps_count để thay thế hoàn toàn Python list kps_cache
        MAX_KPS_CACHE = 256
        self.kps_data = np.zeros((self.length, 2, MAX_KPS_CACHE, 2), dtype=np.float32)
        self.kps_count = np.zeros(self.length, dtype=np.int32)

        print(f"[*] Preloading keypoints cache vào mảng NumPy 4D liền mạch ({self.split_name}: {self.length} mẫu)...")
        for idx in range(self.length):
            pmatch = self.pmatch_arr[idx].decode('utf-8')
            if os.path.exists(pmatch):
                try:
                    kps = np.load(pmatch).astype(np.float32)
                    if kps.ndim == 3 and kps.shape[0] == 2 and kps.shape[2] == 2:
                        n = kps.shape[1]
                        if n > MAX_KPS_CACHE:
                            idx_sub = np.linspace(0, n - 1, MAX_KPS_CACHE, dtype=int)
                            self.kps_data[idx] = kps[:, idx_sub, :]
                            self.kps_count[idx] = MAX_KPS_CACHE
                        elif n > 0:
                            self.kps_data[idx, :, :n, :] = kps
                            self.kps_count[idx] = n
                except Exception:
                    pass
        print(f"[✓] Đã nạp xong keypoints cache vào mảng NumPy liền mạch (CoW-Proof).")

        # 3. [TỐI ƯU HÓA CO-W]: Lưu TPS Cache Validation thành 5 mảng NumPy phẳng contiguous
        self.val_c0 = np.zeros((self.length, 88, 2), dtype=np.float32)
        self.val_th01 = np.zeros((self.length, 91, 2), dtype=np.float32)
        self.val_c1 = np.zeros((self.length, 88, 2), dtype=np.float32)
        self.val_th10 = np.zeros((self.length, 91, 2), dtype=np.float32)
        self.val_valid = np.zeros((self.length, 1, 1, 1), dtype=np.float32)

        if not self.is_train:
            print(f"[*] Precomputing TPS cache vào 5 mảng NumPy cho tập Validation ({self.length} mẫu)...")
            for idx in range(self.length):
                p0 = self.p0_arr[idx].decode('utf-8')
                cnt = int(self.kps_count[idx])
                kps = self.kps_data[idx, :, :cnt, :].copy() if cnt > 0 else np.zeros((2, 0, 2), dtype=np.float32)
                try:
                    with Image.open(p0) as img_hdr:
                        w, h = img_hdr.size
                except Exception:
                    w, h = 512, 512
                pad_h = ((h + 31) // 32) * 32
                pad_w = ((w + 31) // 32) * 32
                if kps is not None and kps.shape[1] > 0:
                    kps_norm = kps.copy()
                    kps_norm[0, :, 0] = (kps_norm[0, :, 0] * w) / pad_w
                    kps_norm[0, :, 1] = (kps_norm[0, :, 1] * h) / pad_h
                    kps_norm[1, :, 0] = (kps_norm[1, :, 0] * w) / pad_w
                    kps_norm[1, :, 1] = (kps_norm[1, :, 1] * h) / pad_h
                else:
                    kps_norm = kps
                c0_p, th01_p, c1_p, th10_p, v = solve_tps_pair_cpu(kps_norm, max_kps=80, lambd=0.5, min_dist=0.015)
                self.val_c0[idx] = c0_p.cpu().numpy() if isinstance(c0_p, torch.Tensor) else c0_p
                self.val_th01[idx] = th01_p.cpu().numpy() if isinstance(th01_p, torch.Tensor) else th01_p
                self.val_c1[idx] = c1_p.cpu().numpy() if isinstance(c1_p, torch.Tensor) else c1_p
                self.val_th10[idx] = th10_p.cpu().numpy() if isinstance(th10_p, torch.Tensor) else th10_p
                self.val_valid[idx, 0, 0, 0] = float(v.item() if isinstance(v, torch.Tensor) else v)
            print(f"[✓] Đã precompute xong {self.length} mẫu TPS Validation vào contiguous NumPy buffers.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self._access_count += 1
        if self._access_count % 250 == 0:
            gc.collect()
            try:
                ctypes.CDLL('libc.so.6').malloc_trim(0)
            except Exception:
                pass

        p0 = self.p0_arr[idx].decode('utf-8')
        pgt = self.pgt_arr[idx].decode('utf-8')
        p1 = self.p1_arr[idx].decode('utf-8')
        clip_name = self.clips_arr[idx].decode('utf-8')

        im0 = cv2.imread(p0, cv2.IMREAD_GRAYSCALE)
        imgt = cv2.imread(pgt, cv2.IMREAD_GRAYSCALE)
        im1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)

        if im0 is None or imgt is None or im1 is None:
            cs = self.crop_size if self.is_train else 512
            im0 = np.full((cs, cs), 255, dtype=np.uint8)
            imgt = np.full((cs, cs), 255, dtype=np.uint8)
            im1 = np.full((cs, cs), 255, dtype=np.uint8)

        h, w = im0.shape

        if self.is_train:
            cnt = int(self.kps_count[idx])
            kps = self.kps_data[idx, :, :cnt, :].copy() if cnt > 0 else np.zeros((2, 0, 2), dtype=np.float32)
            cs = self.crop_size
            if h > cs and w > cs:
                top = random.randint(0, h - cs)
                left = random.randint(0, w - cs)
                im0_crop = im0[top:top+cs, left:left+cs]
                imggt_crop = imgt[top:top+cs, left:left+cs]
                im1_crop = im1[top:top+cs, left:left+cs]

                if kps.shape[1] > 0:
                    kps[0, :, 0] = (kps[0, :, 0] * w - left) / cs
                    kps[0, :, 1] = (kps[0, :, 1] * h - top) / cs
                    kps[1, :, 0] = (kps[1, :, 0] * w - left) / cs
                    kps[1, :, 1] = (kps[1, :, 1] * h - top) / cs
                    mask0 = (kps[0, :, 0] >= 0) & (kps[0, :, 0] <= 1) & (kps[0, :, 1] >= 0) & (kps[0, :, 1] <= 1)
                    mask1 = (kps[1, :, 0] >= 0) & (kps[1, :, 0] <= 1) & (kps[1, :, 1] >= 0) & (kps[1, :, 1] <= 1)
                    kps = kps[:, mask0 & mask1, :]
            else:
                pad_h = ((h + 31) // 32) * 32
                pad_w = ((w + 31) // 32) * 32
                im0_crop = np.pad(im0, ((0, pad_h - h), (0, pad_w - w)), mode='constant', constant_values=255)
                imggt_crop = np.pad(imgt, ((0, pad_h - h), (0, pad_w - w)), mode='constant', constant_values=255)
                im1_crop = np.pad(im1, ((0, pad_h - h), (0, pad_w - w)), mode='constant', constant_values=255)

            img0 = (255 - im0_crop).astype(np.float32) * (1.0 / 255.0)
            imggt = (255 - imggt_crop).astype(np.float32) * (1.0 / 255.0)
            img1 = (255 - im1_crop).astype(np.float32) * (1.0 / 255.0)

            c0_pad, th01_pad, c1_pad, th10_pad, valid_flag = solve_tps_pair_cpu(kps, max_kps=80, lambd=0.5, min_dist=0.015)
            valid_flag = valid_flag.view(1, 1, 1)
        else:
            pad_h = ((h + 31) // 32) * 32
            pad_w = ((w + 31) // 32) * 32
            im0_pad = np.pad(im0, ((0, pad_h - h), (0, pad_w - w)), mode='constant', constant_values=255)
            imggt_pad = np.pad(imgt, ((0, pad_h - h), (0, pad_w - w)), mode='constant', constant_values=255)
            im1_pad = np.pad(im1, ((0, pad_h - h), (0, pad_w - w)), mode='constant', constant_values=255)

            img0 = (255 - im0_pad).astype(np.float32) * (1.0 / 255.0)
            imggt = (255 - imggt_pad).astype(np.float32) * (1.0 / 255.0)
            img1 = (255 - im1_pad).astype(np.float32) * (1.0 / 255.0)

            # Lấy trực tiếp từ contiguous NumPy arrays bằng .copy(), 0% CoW
            c0_pad = torch.from_numpy(self.val_c0[idx].copy())
            th01_pad = torch.from_numpy(self.val_th01[idx].copy())
            c1_pad = torch.from_numpy(self.val_c1[idx].copy())
            th10_pad = torch.from_numpy(self.val_th10[idx].copy())
            valid_flag = torch.from_numpy(self.val_valid[idx].copy())

        t0 = torch.from_numpy(np.ascontiguousarray(img0)).unsqueeze(0)
        tgt = torch.from_numpy(np.ascontiguousarray(imggt)).unsqueeze(0)
        t1 = torch.from_numpy(np.ascontiguousarray(img1)).unsqueeze(0)
        t_val = torch.tensor([[[float(self.t_arr[idx])]]], dtype=torch.float32)

        return t0, tgt, t1, t_val, c0_pad, th01_pad, c1_pad, th10_pad, valid_flag, clip_name

custom_collate_fn_v1 = default_collate

PER_GPU_BATCH = 8
BATCH_SIZE = PER_GPU_BATCH * max(1, NUM_GPUS)
VAL_BATCH_SIZE = max(1, NUM_GPUS)

train_dataset = SakugaGoldenDataset_v1(DATASET_ROOT, train_df, is_train=True, crop_size=512)
val_dataset = SakugaGoldenDataset_v1(DATASET_ROOT, val_df, is_train=False)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    persistent_workers=True,
    pin_memory=False,
    prefetch_factor=2,
    collate_fn=custom_collate_fn_v1,
    worker_init_fn=_worker_init_fn,
    drop_last=True
)

val_fast_subset = Subset(val_dataset, range(min(len(val_dataset), 100 * VAL_BATCH_SIZE)))

# Validation loaders: num_workers=0, persistent_workers=False để giải phóng 4 tiến trình ngầm & GB RAM IPC
val_loader_fast = DataLoader(
    val_fast_subset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
    collate_fn=custom_collate_fn_v1
)

val_loader_full = DataLoader(
    val_dataset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
    collate_fn=custom_collate_fn_v1
)

val_loader = val_loader_full

print(f"[✓] DataLoaders Golden Dataset V1.1 sẵn sàng: {len(train_loader)} batches (Train), {len(val_loader_fast)} batches (Val Fast), {len(val_loader_full)} batches (Val Full).")
print("[✓] CELL 4 hoàn tất: Dataset & DataLoaders triệt tiêu 100% Copy-on-Write (CoW).")"""
    set_code(4, c4_v11)

    # -------------------------------------------------------------------------
    # CELL 5: Architecture V1.1
    # -------------------------------------------------------------------------
    c5 = get_code(5)
    c5 = c5.replace("[CELL 5]: Kiến Trúc TPS-Inbetween V1.0", "[CELL 5]: Kiến Trúc TPS-Inbetween V1.1")
    c5 = c5.replace("Synthesis UNet v1.0 với Gated Ghost Erasure (GGER)", "Synthesis UNet v1.1 với Gated Ghost Erasure (GGER) & Unlocked Detail Add")
    c5 = c5.replace("nn.init.constant_(self.conv_add.bias, -2.0)", "nn.init.constant_(self.conv_add.bias, -3.0)")
    c5 = c5.replace("detail_add = torch.sigmoid(self.conv_add(x)) * 0.25", "detail_add = torch.sigmoid(self.conv_add(x))")
    c5 = c5.replace(
"""        # 3. Fine Motion Refinement
        with torch.no_grad():
            feats0 = self.encoder(x0, coarse_flow0t)
            feats1 = self.encoder(x1, coarse_flow1t)
        flow_refine, ifnet_mask = self.IFNet(coarse_I0t, coarse_I1t)""",
"""        # 3. Fine Motion Refinement (Cho phép gradient truyền qua Contextnet khi fine-tune V1.1)
        feats0 = self.encoder(x0, coarse_flow0t)
        feats1 = self.encoder(x1, coarse_flow1t)
        flow_refine, ifnet_mask = self.IFNet(coarse_I0t, coarse_I1t)"""
    )
    c5 = c5.replace("TPSInbetweenFinetune_v1 (ASF-GR Engine V1.0)", "TPSInbetweenFinetune_v1 (ASF-GR Engine V1.1 - Unlocked Detail Add)")
    set_code(5, c5)

    # -------------------------------------------------------------------------
    # CELL 6: Calibrated Loss Engine
    # -------------------------------------------------------------------------
    c6 = get_code(6)
    c6 = c6.replace("[CELL 6]: Bộ Hàm Mất Mát CompositeLossEngine_v1 (100% PyTorch GPU Tensor)", "[CELL 6]: Bộ Hàm Mất Mát CompositeLossEngine_v1 (100% PyTorch GPU Tensor V1.1)")
    old_ghost_class = """class GhostStrokePenaltyLoss(nn.Module):
    def __init__(self, radius=5):
        super(GhostStrokePenaltyLoss, self).__init__()
        self.radius = radius

    def forward(self, pred, target):
        pad = self.radius // 2
        t_dilated = F.max_pool2d((target > 0.15).float(), kernel_size=self.radius, stride=1, padding=pad)
        ghost_energy = F.relu(pred - 0.15) * (1.0 - t_dilated)
        return torch.mean(ghost_energy ** 2)"""

    new_ghost_class = """class GhostStrokePenaltyLoss(nn.Module):
    def __init__(self, radius=7):
        super(GhostStrokePenaltyLoss, self).__init__()
        # Tăng bán kính vùng an toàn từ radius=5 lên radius=7 (pad=3) để không phạt nét vẽ lệch nhẹ 1-3px
        self.radius = radius

    def forward(self, pred, target):
        pad = self.radius // 2
        t_dilated = F.max_pool2d((target > 0.15).float(), kernel_size=self.radius, stride=1, padding=pad)
        outside_gt = 1.0 - t_dilated
        # Phạt năng lượng bóng ma lơ lửng pred > 0.08 ngoài vùng nét vẽ GT
        ghost_energy = F.relu(pred - 0.08) * outside_gt
        return torch.mean(ghost_energy ** 2)"""

    c6 = c6.replace(old_ghost_class, new_ghost_class)
    c6 = c6.replace("self.ghost_loss_fn = GhostStrokePenaltyLoss()", "self.ghost_loss_fn = GhostStrokePenaltyLoss(radius=7)")
    old_total_loss = """        total_loss = (
            loss_l1 +
            0.15 * loss_vgg +
            0.35 * loss_dice +
            0.20 * loss_morph +
            0.15 * loss_ghost +
            0.10 * loss_dens
        )"""

    new_total_loss = """        # Cân bằng lại hệ số: Tăng loss_dice lên 0.50 để bảo toàn nét vẽ liền mạch
        total_loss = (
            1.00 * loss_l1 +
            0.15 * loss_vgg +
            0.50 * loss_dice +
            0.20 * loss_morph +
            0.10 * loss_ghost +
            0.05 * loss_dens
        )"""

    c6 = c6.replace(old_total_loss, new_total_loss)
    c6 = c6.replace("CompositeLossEngine_v1 100% GPU-Native V1.0", "CompositeLossEngine_v1 Hiệu Chỉnh V1.1 (100% GPU-Native)")
    set_code(6, c6)

    # -------------------------------------------------------------------------
    # CELL 7: Optimizer & Wrapper (Differential LR)
    # -------------------------------------------------------------------------
    c7 = get_code(7)
    c7 = c7.replace("[CELL 7]: Optimizer, Multi-GPU Parallel Training Wrapper & Nạp Checkpoint (Warm Start)", "[CELL 7]: Optimizer, Multi-GPU Parallel Training Wrapper & Nạp Checkpoint (Differential LR V1.1)")
    c7 = c7.replace("""for param in base_model.encoder.parameters():
    param.requires_grad = False""",
"""# Mở khóa toàn bộ tham số bao gồm cả Contextnet encoder để fine-tune đặc trưng Line Art
for param in base_model.parameters():
    param.requires_grad = True""")

    c7 = c7.replace(
"""pretrained_candidates = [
    os.path.join(WORKING_DIR, 'checkpoints_v1.0', 'model_tps_v1.0_best.pt'),""",
"""pretrained_candidates = [
    os.path.join(WORKING_DIR, 'checkpoints_v1.1', 'model_tps_v1.1_best.pt'),
    os.path.join(WORKING_DIR, 'checkpoints_v1.0', 'model_tps_v1.0_best.pt'),
    os.path.join(KAGGLE_INPUT, 'tps-v11-checkpoint', 'model_tps_v1.1_best.pt'),"""
    )

    old_opt_block = """optimizer = torch.optim.AdamW(
    trainable_params,
    lr=1.0e-4,
    betas=(0.9, 0.999),
    weight_decay=1e-4
)

TOTAL_EPOCHS = 30
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=TOTAL_EPOCHS,
    eta_min=1.0e-5
)"""

    new_opt_block = """# Thiết lập AdamW với Differential Learning Rates (Tốc độ học phân tầng)
optimizer = torch.optim.AdamW([
    {'params': base_model.encoder.parameters(), 'lr': 2.0e-5},
    {'params': base_model.IFNet.parameters(), 'lr': 1.0e-4},
    {'params': list(base_model.asf_head.parameters()) + list(base_model.unet.parameters()), 'lr': 2.0e-4}
], betas=(0.9, 0.999), weight_decay=1e-4)

TOTAL_EPOCHS = 30
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=TOTAL_EPOCHS,
    eta_min=1.0e-6
)"""

    c7 = c7.replace(old_opt_block, new_opt_block)
    c7 = c7.replace("Tổng số tham số huấn luyện V1.0", "Tổng số tham số huấn luyện V1.1")
    c7 = c7 + """\nprint(f"    - Nhóm 1 (Encoder Contextnet) : lr = {optimizer.param_groups[0]['lr']:.2e}")\nprint(f"    - Nhóm 2 (IFNet Optical Flow) : lr = {optimizer.param_groups[1]['lr']:.2e}")\nprint(f"    - Nhóm 3 (ASF Head + UNet)   : lr = {optimizer.param_groups[2]['lr']:.2e}")"""
    set_code(7, c7)

    # -------------------------------------------------------------------------
    # CELL 8: Anti-Cheat Metrics V1.1 (Dilated Stroke F1)
    # -------------------------------------------------------------------------
    c8 = get_code(8)
    c8 = c8.replace("[CELL 8]: Hàm Đo Lường Chỉ Số Thẩm Định Chống Bẫy V1.0 (100% Pure GPU Accelerated Metrics)", "[CELL 8]: Hàm Đo Lường Chỉ Số Thẩm Định Chống Bẫy V1.1 (100% Pure GPU Accelerated Metrics)")
    old_recall_ghost = """        # 3. Stroke Recall & Blank Check trên GPU
        p_mask = (pred_tensor > 0.15).float()
        g_mask = (gt_tensor > 0.15).float()
        p_lines = int(torch.sum(p_mask).item())
        g_lines = int(torch.sum(g_mask).item())

        if g_lines < 50:
            recall = 1.0
        else:
            p_dilated = F.max_pool2d(p_mask, kernel_size=7, stride=1, padding=3)
            hit = torch.sum((g_mask > 0) & (p_dilated > 0)).item()
            recall = float(hit / max(1, g_lines))

        is_blank = bool(p_lines < 100 and g_lines >= 500)

        # 4. Ghost Pixels trên GPU
        ghost_mask = (pred_tensor >= 0.10) & (pred_tensor <= 0.45) & (gt_tensor < 0.05)
        ghost_pixels = int(torch.sum(ghost_mask).item()) if not is_blank else -1

    return {
        'psnr': psnr,
        'ssim': ssim,
        'recall': recall,
        'ghost_pixels': ghost_pixels,
        'p_lines': p_lines,
        'g_lines': g_lines,
        'is_blank': is_blank
    }"""

    new_recall_ghost = """        # 3. Stroke Recall & F1-Stroke trên GPU
        p_mask = (pred_tensor > 0.15).float()
        g_mask = (gt_tensor > 0.15).float()
        p_lines = int(torch.sum(p_mask).item())
        g_lines = int(torch.sum(g_mask).item())

        if g_lines < 50:
            recall = 1.0
            precision = 1.0
            f1_stroke = 1.0
        else:
            p_dilated = F.max_pool2d(p_mask, kernel_size=7, stride=1, padding=3)
            hit = torch.sum((g_mask > 0) & (p_dilated > 0)).item()
            recall = float(hit / max(1, g_lines))

            # Dilated Stroke Precision & F1-Score: Đo độ sắc gọn và đầy đủ của nét vẽ
            g_dilated = F.max_pool2d(g_mask, kernel_size=7, stride=1, padding=3)
            prec_hit = torch.sum((p_mask > 0) & (g_dilated > 0)).item()
            precision = float(prec_hit / max(1, p_lines)) if p_lines > 0 else 0.0
            f1_stroke = float(2.0 * precision * recall / max(1e-6, precision + recall))

        is_blank = bool(p_lines < 100 and g_lines >= 500)

        # 4. Ghost Pixels trên GPU (Tách vùng đệm dung sai không tính pixel chống răng cưa sát nét vẽ GT)
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

    c8 = c8.replace(old_recall_ghost, new_recall_ghost)
    c8 = c8.replace("Bộ chỉ số Anti-Cheat Metrics V1.0 100% GPU sẵn sàng.", "Bộ chỉ số Anti-Cheat Metrics V1.1 (Kèm Dilated Stroke F1) 100% GPU sẵn sàng.")
    set_code(8, c8)

    # -------------------------------------------------------------------------
    # CELL 9: Smart Checkpoint Manager V1.1
    # -------------------------------------------------------------------------
    c9 = get_code(9)
    c9 = c9.replace("[CELL 9]: Bộ Quản Lý Checkpoint Thông Minh V1.0", "[CELL 9]: Bộ Quản Lý Checkpoint Thông Minh V1.1")
    c9 = c9.replace("/content/drive/MyDrive/TPS_v1.0_Checkpoints", "/content/drive/MyDrive/TPS_v1.1_Checkpoints")
    c9 = c9.replace("Bộ Quản Lý Checkpoint Thông Minh V1.0 (Production Candidate)", "Bộ Quản Lý Checkpoint Thông Minh V1.1 (Production Release)")

    old_score_eval = """        score = psnr + 20.0 * ssim + 30.0 * recall - 0.0005 * ghost
        print(f"   [*] Composite Score V1.0: {score:.2f} (Kỷ lục: {self.best_score:.2f}) | Ghost: {ghost} px | Recall: {recall*100:.1f}%")"""

    new_score_eval = """        f1_stroke = metrics.get('f1_stroke', recall)
        # Thang điểm đánh giá toàn diện V1.1: Phá vỡ trần giả tạo bằng Stroke F1
        score = (
            1.2 * psnr +
            20.0 * ssim +
            25.0 * recall +
            25.0 * f1_stroke -
            0.0005 * ghost
        )
        print(f"   [*] Composite Score V1.1: {score:.2f} (Kỷ lục: {self.best_score:.2f}) | Ghost: {ghost} px | Recall: {recall*100:.1f}% | F1: {f1_stroke*100:.1f}%")"""

    c9 = c9.replace(old_score_eval, new_score_eval)
    c9 = c9.replace("model_tps_v1.0_latest.pt", "model_tps_v1.1_latest.pt")
    c9 = c9.replace("model_tps_v1.0_ep{current_epoch:02d}.pt", "model_tps_v1.1_ep{current_epoch:02d}.pt")
    c9 = c9.replace("model_tps_v1.0_best.pt", "model_tps_v1.1_best.pt")
    c9 = c9.replace("KỶ LỤC MỚI V1.0!", "KỶ LỤC MỚI V1.1!")
    c9 = c9.replace("checkpoints_v1.0", "checkpoints_v1.1")
    c9 = c9.replace("Smart Checkpoint Manager V1.0 & GDrive Auto-Sync sẵn sàng.", "Smart Checkpoint Manager V1.1 & GDrive Auto-Sync sẵn sàng.")
    set_code(9, c9)

    # -------------------------------------------------------------------------
    # CELL 10: Training Loop V1.1
    # -------------------------------------------------------------------------
    c10 = get_code(10)
    c10 = c10.replace("[CELL 10]: Vòng Lặp Huấn Luyện & Thẩm Định Hoàn Chỉnh (30 Epochs V1.0 - Zero GPU Sync Overhead)", "[CELL 10]: Vòng Lặp Huấn Luyện & Thẩm Định Hoàn Chỉnh (30 Epochs V1.1 - Zero GPU Sync Overhead)")
    c10 = c10.replace("training_history_v1.0.csv", "training_history_v1.1.csv")
    c10 = c10.replace("BẮT ĐẦU HUẤN LUYỆN TINH CHỈNH TPS-INBETWEEN V1.0", "BẮT ĐẦU HUẤN LUYỆN TINH CHỈNH TPS-INBETWEEN V1.1")
    old_header_block = """print(f"   - Kiến trúc đột phá : Adaptive Stroke Fusion (ASF) + Gated Ghost Erasure (GGER V1.0)")
print(f"   - Phân tán Multi-GPU: Forward + Full Loss Engine chạy song song trên {max(1, NUM_GPUS)} GPU")
print(f"   - Tối ưu luồng tính : Vectorized TPS Batch (GPU BMM) + Không nghẽn .item() từng step")"""

    new_header_block = """print(f"   - Kiến trúc đột phá : Adaptive Stroke Fusion (ASF) + Unlocked GGER (V1.1)")
print(f"   - Tốc độ học đa tầng: Differential LR (Encoder: 2e-5, IFNet: 1e-4, ASF/UNet: 2e-4)")
print(f"   - Phân tán Multi-GPU: Forward + Full Loss Engine chạy song song trên {max(1, NUM_GPUS)} GPU")
print(f"   - Thẩm định chuẩn   : val_loader_full cho mọi epoch (Đồng nhất kỷ lục Best Score)")"""

    c10 = c10.replace(old_header_block, new_header_block)

    old_val_init = """        val_loss_accum = torch.tensor(0.0, device=device)
        val_l1_accum = torch.tensor(0.0, device=device)
        val_dice_accum = torch.tensor(0.0, device=device)
        val_psnr_list, val_ssim_list, val_recall_list, val_ghost_list = [], [], [], []

        active_val_loader = val_loader_full if (epoch % 5 == 0 or epoch == TOTAL_EPOCHS) else val_loader_fast"""

    new_val_init = """        val_loss_accum = torch.tensor(0.0, device=device)
        val_l1_accum = torch.tensor(0.0, device=device)
        val_dice_accum = torch.tensor(0.0, device=device)
        val_psnr_list, val_ssim_list, val_recall_list, val_f1_list, val_ghost_list = [], [], [], [], []

        # Đồng nhất sử dụng val_loader_full cho mọi epoch để điểm số công bằng và tuyệt đối chuẩn xác
        active_val_loader = val_loader_full"""

    c10 = c10.replace(old_val_init, new_val_init)

    old_val_metrics_sample = """                for i in range(pred.shape[0]):
                    m = calculate_metrics_v1(pred[i:i+1], tgt[i:i+1])
                    val_psnr_list.append(m['psnr'])
                    val_ssim_list.append(m['ssim'])
                    val_recall_list.append(m['recall'])
                    if m['ghost_pixels'] >= 0:
                        val_ghost_list.append(m['ghost_pixels'])"""

    new_val_metrics_sample = """                for i in range(pred.shape[0]):
                    m = calculate_metrics_v1(pred[i:i+1], tgt[i:i+1])
                    val_psnr_list.append(m['psnr'])
                    val_ssim_list.append(m['ssim'])
                    val_recall_list.append(m['recall'])
                    val_f1_list.append(m['f1_stroke'])
                    if m['ghost_pixels'] >= 0:
                        val_ghost_list.append(m['ghost_pixels'])"""

    c10 = c10.replace(old_val_metrics_sample, new_val_metrics_sample)

    old_val_postfix = """                if val_step % 10 == 0 or val_step == len(active_val_loader) - 1:
                    val_pbar.set_postfix({
                        'ValLoss': f"{v_loss.item():.4f}",
                        'Dice': f"{v_dice.item():.3f}",
                        'PSNR': f"{val_psnr_list[-1]:.1f}dB" if val_psnr_list else "..."
                    })"""

    new_val_postfix = """                if val_step % 10 == 0 or val_step == len(active_val_loader) - 1:
                    val_pbar.set_postfix({
                        'ValLoss': f"{v_loss.item():.4f}",
                        'Dice': f"{v_dice.item():.3f}",
                        'F1': f"{val_f1_list[-1]*100:.1f}%" if val_f1_list else "...",
                        'PSNR': f"{val_psnr_list[-1]:.1f}dB" if val_psnr_list else "..."
                    })"""

    c10 = c10.replace(old_val_postfix, new_val_postfix)

    old_mean_calc = """        mean_val_loss = float((val_loss_accum / num_val_batches).item())
        mean_val_l1 = float((val_l1_accum / num_val_batches).item())
        mean_val_dice = float((val_dice_accum / num_val_batches).item())
        mean_val_psnr = float(np.mean(val_psnr_list)) if val_psnr_list else 22.5
        mean_val_ssim = float(np.mean(val_ssim_list)) if val_ssim_list else 0.91
        mean_val_recall = float(np.mean(val_recall_list)) if val_recall_list else 0.88
        mean_val_ghost = float(np.mean(val_ghost_list)) if val_ghost_list else 4500.0

        epoch_time = time.time() - epoch_start_time

        print(f"=" * 85)
        print(f"📌 [EPOCH {epoch:02d}/{TOTAL_EPOCHS:02d} HOÀN TẤT] (Thời gian: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.2e})")
        print(f"   🏋️ TRAIN METRICS : Total Loss: {mean_train_loss:.4f} | L1: {mean_train_l1:.4f} | Dice: {mean_train_dice:.4f} | GhostLoss: {mean_train_ghost_l:.4f}")
        print(f"   🔍 VAL METRICS   : Total Loss: {mean_val_loss:.4f} | PSNR: {mean_val_psnr:.2f}dB | SSIM: {mean_val_ssim:.4f}")
        print(f"   🎯 GHOST & RECALL: Ghost Pixels: {mean_val_ghost:.0f} px | Stroke Recall: {mean_val_recall*100:.1f}%")
        print(f"=" * 85)

        val_metrics = {
            'val_loss': mean_val_loss,
            'train_loss': mean_train_loss,
            'psnr': mean_val_psnr,
            'ssim': mean_val_ssim,
            'recall': mean_val_recall,
            'ghost_pixels': mean_val_ghost
        }"""

    new_mean_calc = """        mean_val_loss = float((val_loss_accum / num_val_batches).item())
        mean_val_l1 = float((val_l1_accum / num_val_batches).item())
        mean_val_dice = float((val_dice_accum / num_val_batches).item())
        mean_val_psnr = float(np.mean(val_psnr_list)) if val_psnr_list else 22.5
        mean_val_ssim = float(np.mean(val_ssim_list)) if val_ssim_list else 0.91
        mean_val_recall = float(np.mean(val_recall_list)) if val_recall_list else 0.88
        mean_val_f1 = float(np.mean(val_f1_list)) if val_f1_list else 0.88
        mean_val_ghost = float(np.mean(val_ghost_list)) if val_ghost_list else 4500.0

        epoch_time = time.time() - epoch_start_time

        lr_enc = optimizer.param_groups[0]['lr']
        lr_unet = optimizer.param_groups[-1]['lr']

        print(f"=" * 85)
        print(f"📌 [EPOCH {epoch:02d}/{TOTAL_EPOCHS:02d} HOÀN TẤT] (Thời gian: {epoch_time:.1f}s | LR UNet/ASF: {lr_unet:.2e} | LR Enc: {lr_enc:.2e})")
        print(f"   🏋️ TRAIN METRICS : Total Loss: {mean_train_loss:.4f} | L1: {mean_train_l1:.4f} | Dice: {mean_train_dice:.4f} | GhostLoss: {mean_train_ghost_l:.4f}")
        print(f"   🔍 VAL METRICS   : Total Loss: {mean_val_loss:.4f} | PSNR: {mean_val_psnr:.2f}dB | SSIM: {mean_val_ssim:.4f}")
        print(f"   🎯 GHOST & STROKE: Ghost Pixels: {mean_val_ghost:.0f} px | Recall: {mean_val_recall*100:.1f}% | Stroke F1: {mean_val_f1*100:.1f}%")
        print(f"=" * 85)

        val_metrics = {
            'val_loss': mean_val_loss,
            'train_loss': mean_train_loss,
            'psnr': mean_val_psnr,
            'ssim': mean_val_ssim,
            'recall': mean_val_recall,
            'f1_stroke': mean_val_f1,
            'ghost_pixels': mean_val_ghost
        }"""

    c10 = c10.replace(old_mean_calc, new_mean_calc)

    old_hist_entry = """        history.append({
            'epoch': epoch,
            'train_loss': mean_train_loss,
            'val_loss': mean_val_loss,
            'train_l1': mean_train_l1,
            'val_l1': mean_val_l1,
            'val_psnr': mean_val_psnr,
            'val_ssim': mean_val_ssim,
            'stroke_recall': mean_val_recall,
            'ghost_pixels': mean_val_ghost,
            'lr': optimizer.param_groups[0]['lr'],
            'epoch_time_sec': round(epoch_time, 1)
        })"""

    new_hist_entry = """        history.append({
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
            'epoch_time_sec': round(epoch_time, 1)
        })"""

    c10 = c10.replace(old_hist_entry, new_hist_entry)
    c10 = c10.replace("training_history_v1.0.csv", "training_history_v1.1.csv")
    c10 = c10.replace("checkpoints_v1.0", "checkpoints_v1.1")
    c10 = c10.replace("model_tps_v1.0_interrupted.pt", "model_tps_v1.1_interrupted.pt")
    set_code(10, c10)

    # -------------------------------------------------------------------------
    # CELL 11: Visual Audit Samples
    # -------------------------------------------------------------------------
    c11 = get_code(11)
    c11 = c11.replace("[CELL 11]: Kiểm Toán Trực Quan Định Tính & Bằng Chứng Khử Bóng Ma V1.0", "[CELL 11]: Kiểm Toán Trực Quan Định Tính & Bằng Chứng Khử Bóng Ma V1.1")
    c11 = c11.replace("visual_samples_v1.0", "visual_samples_v1.1")
    c11 = c11.replace("khử bóng ma V1.0...", "khử bóng ma V1.1...")
    c11 = c11.replace("Finetuned V1.0 (Zero-Ghost)", "Finetuned V1.1 (Zero-Ghost)")
    c11 = c11.replace("_v10.png", "_v11.png")
    set_code(11, c11)

    # -------------------------------------------------------------------------
    # CELL 12: Packaging & Artifacts Export
    # -------------------------------------------------------------------------
    c12 = get_code(12)
    c12 = c12.replace("[CELL 12]: Đóng Gói Thành Phẩm Sản Phẩm V1.0", "[CELL 12]: Đóng Gói Thành Phẩm Sản Phẩm V1.1")
    c12 = c12.replace("training_history_v1.0.csv", "training_history_v1.1.csv")
    c12 = c12.replace("checkpoints_v1.0", "checkpoints_v1.1")
    c12 = c12.replace("model_tps_v1.0_best.pt", "model_tps_v1.1_best.pt")
    c12 = c12.replace("model_tps_v1.0_latest.pt", "model_tps_v1.1_latest.pt")
    c12 = c12.replace("tps_finetune_v1.0_artifacts", "tps_finetune_v1.1_artifacts")
    c12 = c12.replace("tps_finetune_v1.0_artifacts.zip", "tps_finetune_v1.1_artifacts.zip")
    c12 = c12.replace("V1.0 (PRODUCTION CANDIDATE)", "V1.1 (PRODUCTION RELEASE)")
    c12 = c12.replace("Best Checkpoint V1.0", "Best Checkpoint V1.1")
    c12 = c12.replace("lịch sử huấn luyện V1.0", "lịch sử huấn luyện V1.1")
    set_code(12, c12)

    # AST Syntax Verification
    code_cells = [c for c in nb['cells'] if c.get('cell_type') == 'code']
    for idx, c in enumerate(code_cells, start=1):
        src = "".join(c.get('source', []))
        ast.parse(src)

    with open(v11_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    with open(alias_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"[✓] Đã tạo thành công notebook V1.1 tại: {v11_path}")
    print(f"[✓] Đã tạo thành công notebook alias tại: {alias_path}")
    print(f"    - Tổng số cells: {len(nb['cells'])}")

if __name__ == "__main__":
    generate()
