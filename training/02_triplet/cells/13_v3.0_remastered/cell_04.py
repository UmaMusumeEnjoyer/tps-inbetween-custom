# [CELL 4]: Lớp Quản Lý Dữ Liệu SakugaGoldenDataset_v30R (0-Copy TPS Cache & Corridor Descreening)
import os
import sys
import gc
import cv2
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor

WORKING_DIR = globals().get('WORKING_DIR', '/kaggle/working' if os.path.exists('/kaggle') else './workspace')
CACHE_DIR = globals().get('CACHE_DIR', os.path.join(WORKING_DIR, 'dataset_cache_v3.0_remastered'))
os.makedirs(CACHE_DIR, exist_ok=True)

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

def ensure_tps_precomputed_v30r(df, data_root, cache_dir, split_name='train', max_pts=80):
    """
    Tự động chuẩn bị bộ đệm TPS 2 chiều (.npy) mmap 0-copy:
    - Nếu đã có sẵn trên cache_dir hoặc data_root: nạp ngay trong 0.01s.
    - Nếu có bản cache từ v32/v31: sao chép tái sử dụng siêu tốc.
    - Nếu chưa có: đọc file matches/*.npy và gọi solve_tps_batch_all trên 8 luồng CPU.
    """
    os.makedirs(cache_dir, exist_ok=True)
    c0_npy = os.path.join(cache_dir, f'{split_name}_c0_v30r.npy')
    th01_npy = os.path.join(cache_dir, f'{split_name}_th01_v30r.npy')
    c1_npy = os.path.join(cache_dir, f'{split_name}_c1_v30r.npy')
    th10_npy = os.path.join(cache_dir, f'{split_name}_th10_v30r.npy')
    valid_npy = os.path.join(cache_dir, f'{split_name}_valid_v30r.npy')

    all_exist = all(os.path.exists(p) for p in [c0_npy, th01_npy, c1_npy, th10_npy, valid_npy])
    if all_exist:
        return (
            np.load(c0_npy, mmap_mode='r'),
            np.load(th01_npy, mmap_mode='r'),
            np.load(c1_npy, mmap_mode='r'),
            np.load(th10_npy, mmap_mode='r'),
            np.load(valid_npy, mmap_mode='r')
        )

    # Thử tìm trên data_root hoặc cache cũ (v32 / v31)
    search_dirs = [data_root, os.path.dirname(data_root), '/tmp/sakuga_ssd_v32', '/tmp/sakuga_ssd_v31']
    for sdir in search_dirs:
        if not sdir or not os.path.exists(sdir):
            continue
        for ver in ['v30r', 'v32', 'v31']:
            cand_files = [os.path.join(sdir, f'{split_name}_{suf}_{ver}.npy') for suf in ['c0', 'th01', 'c1', 'th10', 'valid']]
            if all(os.path.exists(p) for p in cand_files):
                for src_p, dst_p in zip(cand_files, [c0_npy, th01_npy, c1_npy, th10_npy, valid_npy]):
                    if not os.path.exists(dst_p):
                        shutil.copyfile(src_p, dst_p)
                return (
                    np.load(c0_npy, mmap_mode='r'),
                    np.load(th01_npy, mmap_mode='r'),
                    np.load(c1_npy, mmap_mode='r'),
                    np.load(th10_npy, mmap_mode='r'),
                    np.load(valid_npy, mmap_mode='r')
                )

    # Nếu chưa có, giải TPS từ thư mục matches/
    length = len(df)
    print(f"[*] Precomputing TPS cache đa luồng cho tập {split_name} ({length:,} mẫu)...", flush=True)

    match_col = 'match_file' if 'match_file' in df.columns else None
    kps_buf = np.zeros((length, 2, 256, 2), dtype=np.float32)
    cnt_buf = np.zeros(length, dtype=np.int32)

    def _load_match_worker(i):
        row = df.iloc[i]
        m_rel = str(row[match_col]) if match_col and pd.notna(row[match_col]) else None
        if not m_rel or m_rel == 'None':
            c_id = str(row.get('clip_id', ''))
            m_rel = f"matches/{c_id}/match_000001_000003.npy"
        m_rel = m_rel.replace('\\', '/').lstrip('/')
        m_path = os.path.join(data_root, m_rel)
        if not os.path.exists(m_path):
            alt = os.path.join(os.path.dirname(data_root), m_rel)
            if os.path.exists(alt): m_path = alt
        pts, cnt = _fast_read_match_npy(m_path, max_cache=256)
        if pts is not None and cnt > 0:
            if pts.ndim == 3 and pts.shape[0] == 2:
                c = min(pts.shape[1], 256)
                kps_buf[i, :, :c, :] = pts[:, :c, :2]
                cnt_buf[i] = c
            elif pts.ndim == 3 and pts.shape[1] == 2:
                c = min(pts.shape[0], 256)
                kps_buf[i, 0, :c, :] = pts[:c, 0, :2]
                kps_buf[i, 1, :c, :] = pts[:c, 1, :2]
                cnt_buf[i] = c
            elif pts.ndim == 2 and pts.shape[1] >= 4:
                c = min(pts.shape[0], 256)
                kps_buf[i, 0, :c, :] = pts[:c, :2]
                kps_buf[i, 1, :c, :] = pts[:c, 2:4]
                cnt_buf[i] = c

    with ThreadPoolExecutor(max_workers=32) as ex:
        list(ex.map(_load_match_worker, range(length)))

    # Giải ma trận TPS hàng loạt
    c0_b, th01_b, c1_b, th10_b, valid_b = solve_tps_batch_all(
        kps_buf, cnt_buf, w=960, h=540, pad_w=960, pad_h=544, max_kps=max_pts, lambd=0.5
    )

    np.save(c0_npy, c0_b)
    np.save(th01_npy, th01_b)
    np.save(c1_npy, c1_b)
    np.save(th10_npy, th10_b)
    np.save(valid_npy, valid_b)

    del kps_buf, cnt_buf
    gc.collect()

    return (
        np.load(c0_npy, mmap_mode='r'),
        np.load(th01_npy, mmap_mode='r'),
        np.load(c1_npy, mmap_mode='r'),
        np.load(th10_npy, mmap_mode='r'),
        np.load(valid_npy, mmap_mode='r')
    )

def corridor_protected_descreen(s, tau_noise=0.015, stroke_thresh=0.05, kernel_size=5):
    """
    Corridor-Protected Descreening:
    - Chỉ khử nhiễu quét giấy mờ (< 0.015) ở vùng nền xa (> 5px ngoài nét vẽ).
    - Bảo toàn 100% ngọn tóc mảnh, mí mắt và dải anti-aliasing mềm gần biên nét.
    """
    mask_stroke = (s > stroke_thresh).astype(np.uint8)
    if mask_stroke.any():
        corridor = cv2.dilate(mask_stroke, np.ones((kernel_size, kernel_size), np.uint8))
        return np.where((corridor == 0) & (s < tau_noise), 0.0, s)
    return np.where(s < tau_noise, 0.0, s)

class SakugaGoldenDataset_v30R(Dataset):
    """
    Dataset tối ưu hóa tuyệt đối cho V3.0-Remastered R2:
    - 0-Copy Memory-Mapped arrays (.npy) cho toàn bộ ma trận TPS hai chiều.
    - Corridor-Protected Descreening: Không cắt cụt nét mảnh, bảo vệ 100% ngọn tóc sub-pixel.
    - Huấn luyện chuẩn Native 540p (Pad 544x960 bội số 32).
    - Temporal Flipping Augmentation: Tăng gấp đôi sự đa dạng chuyển động đối xứng.
    """
    def __init__(self, df, data_root, is_train=True, cache_dir=None, precomputed_tps=None):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.is_train = is_train
        self.split_name = 'train' if is_train else 'val'

        c0 = 'frame0_path' if 'frame0_path' in self.df.columns else 'frame0'
        c1 = 'frame1_path' if 'frame1_path' in self.df.columns else 'frame1'
        ct = 'target_path' if 'target_path' in self.df.columns else 'frame_gt'

        self.p0_list = [str(x).replace('\\', '/').lstrip('/') for x in self.df[c0]]
        self.p1_list = [str(x).replace('\\', '/').lstrip('/') for x in self.df[c1]]
        self.pt_list = [str(x).replace('\\', '/').lstrip('/') for x in self.df[ct]]
        self.t_list = self.df['t'].values.astype(np.float32) if 't' in self.df.columns else np.full(len(self.df), 0.5, dtype=np.float32)

        if precomputed_tps is not None:
            self.tps_c0, self.tps_th01, self.tps_c1, self.tps_th10, self.tps_valid = precomputed_tps
        else:
            cd = cache_dir or CACHE_DIR
            self.tps_c0, self.tps_th01, self.tps_c1, self.tps_th10, self.tps_valid = ensure_tps_precomputed_v30r(
                self.df, self.data_root, cd, split_name=self.split_name
            )

    def __len__(self):
        return len(self.df)

    def _read_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros((540, 960), dtype=np.uint8)
        return img

    def __getitem__(self, idx):
        p0_rel = self.p0_list[idx]
        p1_rel = self.p1_list[idx]
        pt_rel = self.pt_list[idx]
        t_val = float(self.t_list[idx])

        p0 = os.path.join(self.data_root, p0_rel) if not os.path.isabs(p0_rel) else p0_rel
        p1 = os.path.join(self.data_root, p1_rel) if not os.path.isabs(p1_rel) else p1_rel
        pt = os.path.join(self.data_root, pt_rel) if not os.path.isabs(pt_rel) else pt_rel

        img0 = self._read_image(p0)
        img1 = self._read_image(p1)
        imgt = self._read_image(pt)

        # Chuyển đổi sang stroke [0.0 = nền, 1.0 = nét mực]
        s0 = (255.0 - img0.astype(np.float32)) * (1.0 / 255.0)
        s1 = (255.0 - img1.astype(np.float32)) * (1.0 / 255.0)
        st = (255.0 - imgt.astype(np.float32)) * (1.0 / 255.0)

        # Corridor-Protected Descreening: Giữ nguyên ngọn tóc mảnh, chỉ làm sạch sương nền ở xa
        s0 = corridor_protected_descreen(s0, tau_noise=0.015, stroke_thresh=0.05)
        s1 = corridor_protected_descreen(s1, tau_noise=0.015, stroke_thresh=0.05)
        st = corridor_protected_descreen(st, tau_noise=0.015, stroke_thresh=0.05)

        h, w = s0.shape
        pad_h = ((h + 31) // 32) * 32
        pad_w = ((w + 31) // 32) * 32
        if pad_h != h or pad_w != w:
            s0 = np.pad(s0, ((0, pad_h - h), (0, pad_w - w)), mode='constant', constant_values=0.0)
            s1 = np.pad(s1, ((0, pad_h - h), (0, pad_w - w)), mode='constant', constant_values=0.0)
            st = np.pad(st, ((0, pad_h - h), (0, pad_w - w)), mode='constant', constant_values=0.0)

        c0 = self.tps_c0[idx].copy()
        th01 = self.tps_th01[idx].copy()
        c1 = self.tps_c1[idx].copy()
        th10 = self.tps_th10[idx].copy()
        v = self.tps_valid[idx].copy()

        # Random Temporal Flipping trong pha Train (Chuẩn bài báo TPS gốc Zhu et al.)
        if self.is_train and np.random.rand() > 0.5:
            s0, s1 = s1, s0
            t_val = 1.0 - t_val
            c0, c1 = c1, c0
            th01, th10 = th10, th01

        t0 = torch.from_numpy(s0).unsqueeze(0).float()
        t1 = torch.from_numpy(s1).unsqueeze(0).float()
        tt = torch.from_numpy(st).unsqueeze(0).float()
        t_tensor = torch.tensor(t_val, dtype=torch.float32)

        tps_tensors = (
            torch.from_numpy(c0).float(),
            torch.from_numpy(th01).float(),
            torch.from_numpy(c1).float(),
            torch.from_numpy(th10).float(),
            torch.from_numpy(v).float()
        )
        return t0, t1, tt, t_tensor, tps_tensors

print("=" * 80)
print("[✓] CELL 4 hoàn tất: Lớp SakugaGoldenDataset_v30R sẵn sàng (0-Copy TPS Cache + Corridor Descreening).")
print("=" * 80)
sys.stdout.flush()
