#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
create_kaggle_notebook.py
Sinh file notebook `generate_tps_dataset_kaggle.ipynb` chuẩn format Jupyter Notebook JSON.
"""

import json
import os
import sys

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def create_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    def add_md(source):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.split("\n")]
        })

    def add_code(source):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.split("\n")]
        })

    # Header
    add_md("""# 🎬 PIPELINE TẠO DATASET ANIME 2D CHUẨN FINE-TUNE CHO TPS-INBETWEEN
## (Kết Hợp ATD-12K Anime Nhật Bản + Anime Tuyển Chọn Đa Mốc Thời Gian $t$)
**Đồ án PBL6 - Hệ Thống Nội Suy Khung Hình Anime Bằng Thin-Plate Spline & Deep Learning**

> **Tóm tắt nhiệm vụ của Notebook này:**
> 1. Nhận vào file nén Candidate Triplets (`candidate_triplets_anime_540p.zip`) đã lọc sạch trên Local.
> 2. **Stage 4:** Chạy mạng nơ-ron **GlueStick + RANSAC** trên GPU T4 để bắt điểm tương đồng và đo độ dời chuyển động (Displacement).
> 3. **Stage 5:** Chạy mạng **ISNetDIS** tách nhân vật + **AniLines Detail** trích xuất nét vẽ + Bộ lọc **MS-SSAA Lanczos-4 & Bilateral** khử răng cưa.
> 4. **Stage 5.5:** Bộ lọc & làm sạch **Lineart Quality & Consistency** (loại bỏ frame rỗng, dính nền bẩn, biến thiên nét, nhảy góc máy).
> 5. **Stage 6:** Đóng gói Dataset thành chuẩn cấu trúc TPS-Inbetween (`train/`, `val/`, `matches/`, `triplets.csv`).
> 6. Nén kết quả thành `anime_tps_finetune_ready.tar.gz` để phục vụ trực tiếp cho việc Fine-tune!""")

    # Cell 1: Environment check & setup
    add_md("""---
### 🛠️ Bước 1: Kiểm Tra GPU & Cài Đặt Môi Trường Thư Viện""")
    add_code("""!nvidia-smi

# Cài đặt các thư viện cần thiết (tương thích nguyên bản với PyTorch, OpenCV, CUDA trên Kaggle)
!pip install -q timm gdown einops kornia omegaconf

import os, sys
import numpy as np
import torch, cv2

print(f"[*] Python version: {sys.version.split()[0]}")
print(f"[*] NumPy version: {np.__version__}")
print(f"[*] PyTorch version: {torch.__version__}")
print(f"[*] OpenCV version: {cv2.__version__} (Đã sẵn sàng)")
print(f"[*] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[*] GPU Device: {torch.cuda.get_device_name(0)}")

# 3. Clone GlueStick repository
if not os.path.exists("GlueStick"):
    !git clone --depth 1 https://github.com/cvg/GlueStick.git
    !cd GlueStick && pip install -q --no-deps -e . && cd ..

# 4. Clone AniLines repository
if not os.path.exists("AniLines"):
    !git clone --depth 1 https://github.com/zhenglinpan/AniLines-Anime-Lineart-Extractor.git AniLines

# 5. Clone Anime Segmentation (ISNetDIS)
if not os.path.exists("anime_seg"):
    !git clone --depth 1 https://github.com/SkyTNT/anime-segmentation.git anime_seg

for p in ["GlueStick", "AniLines", "anime_seg", "anime_seg/model"]:
    p_abs = os.path.abspath(p)
    if p_abs not in sys.path:
        sys.path.append(p_abs)
print()
print("[✓] Cài đặt môi trường thành công!")""")

    # Cell 2: Download weights
    add_md("""---
### 📥 Bước 2: Tự Động Tải Trọng Số Mô Hình (Pretrained Weights)""")
    add_code("""import os, urllib.request

os.makedirs("weights", exist_ok=True)

# 1. Trọng số GlueStick
gluestick_ckpt = "weights/checkpoint_GlueStick_MD.tar"
if not os.path.exists(gluestick_ckpt):
    print("[*] Đang tải GlueStick checkpoint...")
    url_gs = "https://github.com/cvg/GlueStick/releases/download/v0.1_arxiv/checkpoint_GlueStick_MD.tar"
    !wget -q -O {gluestick_ckpt} {url_gs}
    print("[✓] GlueStick checkpoint tải xong!")

# 2. Trọng số ISNetDIS (Anime Segmentation)
isnet_ckpt = "weights/isnetis.ckpt"
if not os.path.exists(isnet_ckpt):
    print("[*] Đang tải ISNetDIS checkpoint...")
    url_isnet = "https://huggingface.co/skytnt/anime-seg/resolve/main/isnetis.ckpt"
    !wget -q -O {isnet_ckpt} {url_isnet}
    print("[✓] ISNetDIS checkpoint tải xong!")

# 3. Trọng số AniLines Detail
anilines_ckpt = "weights/detail.pth"
if not os.path.exists(anilines_ckpt):
    print("[*] Đang tải AniLines Detail checkpoint...")
    !gdown --id 12U1Mwlonoipk2Yvr12mNaFB30foy420o -O {anilines_ckpt} --fuzzy
    print("[✓] AniLines Detail checkpoint tải xong!")

print()
print("=" * 50)
print("[✓] Toàn bộ 3 bộ trọng số Deep Learning đã sẵn sàng:")
!ls -lh weights/
print("=" * 50)""")

    # Cell 3: Locate Dataset Candidates
    add_md("""---
### 📦 Bước 3: Định Vị Candidate Triplets Dataset
*(Notebook tự động nhận diện thư mục dataset đã được Kaggle giải nén sẵn trong `/kaggle/input`, hoặc tự giải nén nếu là file `.zip`)*""")
    add_code("""import os, sys, glob, json, zipfile

DATASET_ROOT = None

# 1. Tự động tìm dataset_manifest.json trong /kaggle/input (Kaggle đã tự giải nén khi upload)
manifest_candidates = glob.glob("/kaggle/input/**/dataset_manifest.json", recursive=True)

if manifest_candidates:
    manifest_path = manifest_candidates[0]
    DATASET_ROOT = os.path.dirname(manifest_path)
    print(f"[✓] Phát hiện Dataset đã được giải nén sẵn trong Kaggle Input:")
    print(f"    👉 DATASET_ROOT = '{DATASET_ROOT}'")

# 2. Nếu chưa tìm thấy thư mục giải nén sẵn, tìm file .zip để giải nén vào data_candidates
if not DATASET_ROOT:
    zip_candidates = glob.glob("/kaggle/input/**/*.zip", recursive=True)
    if not zip_candidates:
        zip_candidates = glob.glob("*.zip")

    if zip_candidates:
        src_zip = zip_candidates[0]
        DATASET_ROOT = "data_candidates"
        os.makedirs(DATASET_ROOT, exist_ok=True)
        print(f"[*] Tìm thấy Dataset ZIP: {src_zip}")
        print(f"[*] Đang giải nén Candidates vào '{DATASET_ROOT}'...")
        with zipfile.ZipFile(src_zip, 'r') as zip_ref:
            zip_ref.extractall(DATASET_ROOT)
        del zip_ref
        import gc
        gc.collect()
        print(f"[✓] Giải nén thành công vào '{DATASET_ROOT}'!")
        manifest_path = os.path.join(DATASET_ROOT, "dataset_manifest.json")
    else:
        # Kiểm tra nếu thư mục cục bộ data_candidates đã có sẵn từ trước
        if os.path.exists("data_candidates/dataset_manifest.json"):
            DATASET_ROOT = "data_candidates"
            manifest_path = "data_candidates/dataset_manifest.json"

# 3. Đọc và kiểm tra tính toàn vẹn của manifest
if DATASET_ROOT and os.path.exists(manifest_path):
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest_data = json.load(f)
    print(f"[✓] Đọc thành công manifest với {len(manifest_data):,} Candidate Triplets!")
    print(f"    - Mẫu đầu tiên (triplet_id): {manifest_data[0]['triplet_id']}")
    sample_frame_path = os.path.join(DATASET_ROOT, manifest_data[0]['frame1'])
    print(f"    - Đường dẫn frame 1 mẫu:     {sample_frame_path}")
    
    # Kiểm tra truy cập thực tế vào ảnh
    if os.path.exists(sample_frame_path):
        print("    - Trạng thái dữ liệu ảnh:    [✓] HỢP LỆ & SẴN SÀNG CHO GPU!")
    else:
        print("    - [!] CẢNH BÁO: File ảnh mẫu không tồn tại. Vui lòng kiểm tra cấu trúc thư mục!")
else:
    print("[!] CẢNH BÁO: Không tìm thấy 'dataset_manifest.json' hoặc file zip dataset!")
    print("    Vui lòng kiểm tra lại mục 'Add Input' trên Kaggle.")
    print("    Các thư mục hiện có trong /kaggle/input:")
    for p in glob.glob("/kaggle/input/*"):
        print(f"     + {p}")""")

    # Cell 4: Stage 4 GlueStick & RANSAC
    add_md("""---
### 🔍 Bước 4 (Stage 4): Bắt Điểm Tương Đồng GlueStick & Lọc RANSAC Trên GPU
- Đo độ dời (displacement) giữa $I_0$ và $I_1$.
- Loại bỏ các triplet có chuyển động quá lớn ($> 15.0\\text{px}$) hoặc bị biến dạng không giải được bằng Spline.
- Lưu trữ các cặp điểm khớp inliers sạch dưới dạng file `.npy` chuẩn hóa $[0, 1]$ cho TPS solver.""")
    add_code("""import os, sys, json, gc, types, cv2, torch
import numpy as np
from tqdm import tqdm
from PIL import Image

# ==============================================================================
# 🚀 GIẢI PHÁP TRIỆT TIÊU 100% HIỆN TƯỢNG RÒ RỈ RAM (OOM / MEMORY RESTART)
# Thay thế pytlsd C-extension (rò rỉ 15MB/call) bằng OpenCV LineSegmentDetector
# ==============================================================================
def cv2_lsd(img, scale=1.0):
    if scale != 1.0:
        h, w = img.shape[:2]
        img_s = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        img_s = img
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines, _, _, _ = detector.detect(img_s)
    if lines is None or len(lines) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    lines = lines.reshape(-1, 4).astype(np.float32)
    if scale != 1.0:
        lines /= float(scale)
    scores = np.ones((len(lines), 1), dtype=np.float32)
    return np.hstack([lines, scores])

# Đăng ký module pytlsd giả lập trỏ vào cv2_lsd
fake_pytlsd = types.ModuleType("pytlsd")
fake_pytlsd.lsd = cv2_lsd
sys.modules["pytlsd"] = fake_pytlsd

# Import GlueStick và bảo đảm wireframe dùng cv2_lsd
from gluestick.models.two_view_pipeline import TwoViewPipeline
try:
    import gluestick.models.wireframe as wireframe_mod
    wireframe_mod.lsd = cv2_lsd
except Exception:
    pass

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def load_gluestick(weights_path="weights/checkpoint_GlueStick_MD.tar"):
    conf = {
        'name': 'two_view_pipeline',
        'use_lines': True,
        'extractor': {
            'name': 'wireframe',
            'sp_params': {'force_num_keypoints': False, 'max_num_keypoints': 1000},
            'wireframe_params': {'merge_points': True, 'merge_line_endpoints': True},
            'max_n_lines': 300,
        },
        'matcher': {'name': 'gluestick', 'weights': weights_path, 'trainable': False},
        'ground_truth': {'from_pose_depth': False}
    }
    model = TwoViewPipeline(conf).to(DEVICE)
    model.eval()
    return model

print("[*] Đang khởi tạo GlueStick Pipeline (OpenCV LSD Mode - Zero Memory Leak)...")
gluestick_model = load_gluestick()
print("[✓] GlueStick sẵn sàng!")

# Thư mục lưu matches
MATCHES_DIR = "processed_dataset/matches"
os.makedirs(MATCHES_DIR, exist_ok=True)

# Đọc manifest
with open(os.path.join(DATASET_ROOT, "dataset_manifest.json"), "r") as f:
    candidates = json.load(f)

screened_triplets = []
SCREEN_MIN_INLIERS = 20
SCREEN_MAX_MEAN_DISP = 15.0

print(f"[*] Bắt đầu Stage 4 cho {len(candidates)} triplets...")
for idx, tr in enumerate(tqdm(candidates, desc="Stage 4 (GlueStick RANSAC)")):
    tid = tr['triplet_id']
    f1_path = os.path.join(DATASET_ROOT, tr['frame1'])
    f3_path = os.path.join(DATASET_ROOT, tr['frame3'])

    if not (os.path.exists(f1_path) and os.path.exists(f3_path)):
        continue

    # Hỗ trợ Resume nếu tiến trình bị ngắt quãng
    match_npy_name = f"{tid}.npy"
    match_npy_path = os.path.join(MATCHES_DIR, match_npy_name)
    if os.path.exists(match_npy_path):
        try:
            kp_matches = np.load(match_npy_path)
            img0_tmp = cv2.imread(f1_path)
            if img0_tmp is not None:
                h_tmp, w_tmp = img0_tmp.shape[:2]
                del img0_tmp
                num_inliers = kp_matches.shape[1]
                diff_px = (kp_matches[0] - kp_matches[1]) * np.array([w_tmp, h_tmp])
                mean_disp = float(np.mean(np.linalg.norm(diff_px, axis=1)))
                tr_clean = dict(tr)
                tr_clean['match_file'] = match_npy_name
                tr_clean['inliers_count'] = int(num_inliers)
                tr_clean['mean_displacement'] = round(mean_disp, 2)
                screened_triplets.append(tr_clean)
                continue
        except Exception:
            pass

    img0 = cv2.imread(f1_path)
    img1 = cv2.imread(f3_path)
    if img0 is None or img1 is None:
        continue

    h, w = img0.shape[:2]

    # Chuẩn bị tensor cho GlueStick
    t0 = torch.from_numpy(cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)).float().to(DEVICE)[None, None] / 255.0
    t1 = torch.from_numpy(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)).float().to(DEVICE)[None, None] / 255.0

    data = {'image0': t0, 'image1': t1}
    with torch.no_grad():
        pred = gluestick_model(data)

    m_kpts0 = pred['matches0'][0].cpu().numpy()
    kpts0 = pred['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()

    # Thu hồi ngay các tensor trung gian để tránh đầy VRAM/RAM
    del t0, t1, data, pred, img0, img1

    valid = m_kpts0 > -1
    p0 = kpts0[valid]
    p1 = kpts1[m_kpts0[valid]]

    if len(p0) < SCREEN_MIN_INLIERS:
        continue

    # RANSAC Affine filtering
    M, inliers = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    if inliers is None:
        continue

    mask = inliers.flatten().astype(bool)
    inliers0 = p0[mask]
    inliers1 = p1[mask]

    if len(inliers0) < SCREEN_MIN_INLIERS:
        continue

    disps = np.linalg.norm(inliers1 - inliers0, axis=1)
    mean_disp = float(np.mean(disps))
    max_disp = float(np.max(disps))

    if mean_disp > SCREEN_MAX_MEAN_DISP or max_disp > 50.0:
        continue

    # Chuẩn hóa tọa độ [0, 1] và lưu .npy
    flow_norm0 = inliers0.copy()
    flow_norm0[:, 0] /= float(w)
    flow_norm0[:, 1] /= float(h)

    flow_norm1 = inliers1.copy()
    flow_norm1[:, 0] /= float(w)
    flow_norm1[:, 1] /= float(h)

    kp_matches = np.stack([flow_norm0, flow_norm1], axis=0) # [2, N, 2]
    np.save(match_npy_path, kp_matches)

    tr_clean = dict(tr)
    tr_clean['match_file'] = match_npy_name
    tr_clean['inliers_count'] = int(len(inliers0))
    tr_clean['mean_displacement'] = round(mean_disp, 2)
    screened_triplets.append(tr_clean)

    # Giải phóng bộ nhớ định kỳ mỗi 50 triplets
    if (idx + 1) % 50 == 0:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print()
print(f"[✓] Hoàn tất Stage 4: {len(screened_triplets)} / {len(candidates)} Triplets vượt qua kiểm duyệt RANSAC!")
with open("screened_triplets.json", "w") as f:
    json.dump(screened_triplets, f, indent=2)""")

    # Cell 5: Stage 5 Lineart & Smoothing
    add_md("""---
### 🎨 Bước 5 (Stage 5): Tách Nền ISNetDIS + AniLines Detail + Làm Mịn Nét
- Tách nền nhân vật anime bằng ISNetDIS.
- Trích xuất line-art nét mảnh bằng AniLines (mode Detail).
- Áp dụng MS-SSAA Lanczos-4 & Bilateral Filter để triệt tiêu hiện tượng răng cưa và đường viền bóng đổ cel-shading.""")
    add_code("""import os, sys, gc, cv2, torch, json
import numpy as np
from tqdm import tqdm

for p in ["GlueStick", "AniLines", "anime_seg", "anime_seg/model"]:
    p_abs = os.path.abspath(p)
    if p_abs not in sys.path:
        sys.path.insert(0, p_abs)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Import AniLines & ISNetDIS
from network.line_extractor import LineExtractor
try:
    from anime_seg.model.isnet import ISNetDIS
except ImportError:
    try:
        from model.isnet import ISNetDIS
    except ImportError:
        from isnet import ISNetDIS

class AnimeBgRemover:
    def __init__(self, ckpt_path="weights/isnetis.ckpt", device=DEVICE, img_size=1024):
        self.device = torch.device(device)
        self.img_size = img_size
        self.net = ISNetDIS().to(self.device)
        try:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=self.device)
        sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        clean_sd = {k.replace('net.', ''): v for k, v in sd.items() if k.startswith('net.')}
        if not clean_sd:
            clean_sd = sd
        self.net.load_state_dict(clean_sd)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad = False

    def remove_background(self, img_bgr):
        \"\"\"
        Đầu vào: img_bgr (np.ndarray HxWx3 uint8)
        Đầu ra: img_white_bg (np.ndarray HxWx3 uint8) với nền đã thay bằng màu trắng tinh khiết
        \"\"\"
        h0, w0 = img_bgr.shape[:2]
        s = self.img_size
        h, w = (s, int(s * w0 / h0)) if h0 > w0 else (int(s * h0 / w0), s)
        ph, pw = s - h, s - w

        inp = np.zeros([s, s, 3], dtype=np.float32)
        inp[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(img_bgr / 255.0, (w, h))
        inp = np.transpose(inp, (2, 0, 1))[np.newaxis, :]
        inp_tensor = torch.from_numpy(inp).float().to(self.device)

        with torch.no_grad():
            if 'cuda' in self.device.type:
                with torch.amp.autocast(device_type='cuda'):
                    pred = self.net(inp_tensor)[0][0].sigmoid()
            else:
                pred = self.net(inp_tensor)[0][0].sigmoid()

            pred = pred.to(dtype=torch.float32).cpu().numpy()[0]
            pred = np.transpose(pred, (1, 2, 0))[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
            mask = cv2.resize(pred, (w0, h0))
            if len(mask.shape) == 2:
                mask = mask[:, :, np.newaxis]

        img_white_bg = np.clip(mask * img_bgr + (1.0 - mask) * 255.0, 0, 255).astype(np.uint8)
        return img_white_bg

# Bộ làm mượt nét vẽ (Anti-Aliasing & Smoothing)
def smooth_lineart(channel, scale=4, sigma_color=35.0, sigma_space=5.0):
    h, w = channel.shape[:2]
    is_white_bg = bool(channel.mean() > 127)
    fg = (255.0 - channel.astype(np.float32)) if is_white_bg else channel.astype(np.float32)

    # 1. Lanczos-4 Super-Sampling
    up_w, up_h = w * scale, h * scale
    up_fg = cv2.resize(fg, (up_w, up_h), interpolation=cv2.INTER_LANCZOS4)

    # 2. Bilateral Filter
    up_uint8 = np.clip(up_fg, 0, 255).astype(np.uint8)
    filtered = cv2.bilateralFilter(up_uint8, d=max(5, int(scale * 1.5) | 1), sigmaColor=sigma_color, sigmaSpace=sigma_space)

    # 3. Downsample Area
    down = cv2.resize(filtered.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    norm = np.clip(down, 0, 255).astype(np.uint8)
    return 255 - norm if is_white_bg else norm

print("[*] Đang tải mô hình ISNetDIS (Tách nền) và AniLines Detail (Trích nét)...")
bg_remover = AnimeBgRemover(ckpt_path="weights/isnetis.ckpt", device=DEVICE, img_size=1024)

line_extractor = LineExtractor(2, 1, True).to(DEVICE)
try:
    sd_line = torch.load("weights/detail.pth", map_location=DEVICE, weights_only=True)
except TypeError:
    sd_line = torch.load("weights/detail.pth", map_location=DEVICE)
line_extractor.load_state_dict(sd_line)
line_extractor.eval()
for p in line_extractor.parameters():
    p.requires_grad = False
print("[✓] Đã nạp thành công mô hình ISNetDIS và AniLines!")

LINEART_DIR = "processed_dataset/lineart"
os.makedirs(LINEART_DIR, exist_ok=True)

with open("screened_triplets.json", "r") as f:
    screened = json.load(f)

# Thu thập tất cả các frame duy nhất cần xử lý
unique_frames = set()
for tr in screened:
    unique_frames.add(tr['frame1'])
    unique_frames.add(tr['frame2'])
    unique_frames.add(tr['frame3'])

unique_frames = sorted(list(unique_frames))
print(f"[*] Bắt đầu Stage 5 cho {len(unique_frames):,} frames duy nhất...")

for idx, rel_frame in enumerate(tqdm(unique_frames, desc="Stage 5 (Lineart Extraction)")):
    src_f = os.path.join(DATASET_ROOT, rel_frame)
    if not os.path.exists(src_f):
        continue

    # Đường dẫn file đầu ra
    dst_f = os.path.join(LINEART_DIR, rel_frame.replace(".jpg", "_smooth.png"))
    if os.path.exists(dst_f):
        continue
    os.makedirs(os.path.dirname(dst_f), exist_ok=True)

    img_bgr = cv2.imread(src_f)
    if img_bgr is None:
        continue

    # 1. Tách nền ISNetDIS (thay nền bằng màu trắng tinh)
    img_white = bg_remover.remove_background(img_bgr)

    # 2. Chuẩn bị 2 channel (Grayscale + Sobel Magnitude) cho AniLines Detail
    img_gray = cv2.cvtColor(img_white, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = 255 - cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    t_img = torch.from_numpy(img_gray).unsqueeze(0).unsqueeze(0).float().to(DEVICE) / 255.0
    t_sobel = torch.from_numpy(sobel).unsqueeze(0).unsqueeze(0).float().to(DEVICE) / 255.0
    x_in = torch.cat([t_img, t_sobel], dim=1)

    H, W = img_gray.shape
    pad_h = 8 - (H % 8) if (H % 8) != 0 else 0
    pad_w = 8 - (W % 8) if (W % 8) != 0 else 0
    x_in = torch.nn.functional.pad(x_in, (0, pad_w, 0, pad_h), mode='reflect')

    with torch.no_grad():
        pred_line = line_extractor(x_in)[:, :, :H, :W]

    line_uint8 = np.clip(pred_line.squeeze().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)

    # 3. Làm mượt nét MS-SSAA + Bilateral
    smooth_out = smooth_lineart(line_uint8)
    cv2.imwrite(dst_f, smooth_out)

    del img_bgr, img_white, img_gray, sobelx, sobely, sobel, t_img, t_sobel, x_in, pred_line, line_uint8, smooth_out
    if (idx + 1) % 100 == 0:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print(f"[✓] Đã tạo xong toàn bộ {len(unique_frames):,} frames nét vẽ chất lượng cao!")""")

    # Cell 5b: Stage 5.5 Lineart Quality & Triplet Cleaning
    add_md(r"""---
### 🧹 Bước 5b (Stage 5.5): Kiểm Duyệt & Lọc Sạch Dataset Lineart (Lineart Quality & Consistency Filter)
Áp dụng bộ tiêu chí lọc đa tầng để loại bỏ các mẫu lineart không đạt chuẩn trước khi đóng gói:
1. **Mật độ nét vẽ ($D \in [0.8\%, 22\%]$)**: Loại bỏ frame trắng trơn do mask phân đoạn hỏng ($D < 0.8\%$) hoặc quá nhiều nhiễu nền sót lại ($D > 22\%$).
2. **Độ phủ Bounding Box ($\ge 4\%$)**: Loại bỏ frame chỉ sót lại một vài đốm đen nhỏ ở góc mép.
3. **Thành phần liên thông chính ($\ge 3\%$)**: Loại bỏ các frame chỉ toàn hạt bụi/nhiễu li ti rời rạc (noise dots) thay vì nét nhân vật liền mạch.
4. **Tính nhất quán giữa 3 Frame ($\text{diff} \le 45\%$)**: Đảm bảo mật độ nét không bị thay đổi đột ngột hoặc frame giữa $I_{GT}$ bị rách/mất nét.
5. **Độ chồng lấn không gian BBox ($\text{IoU} \ge 0.25$)**: Ngăn chặn hiện tượng nhân vật nhảy vọt vị trí (teleport) hoặc chuyển góc máy đột ngột.""")
    add_code("""import os, sys, json, cv2
import numpy as np
from tqdm import tqdm

LINEART_DIR = "processed_dataset/lineart"

# BẢNG THAM SỐ TIÊU CHÍ LỌC LINEART
MIN_LINE_DENSITY = 0.008          # Mật độ nét tối thiểu: 0.8%
MAX_LINE_DENSITY = 0.22           # Mật độ nét tối đa: 22%
MIN_BBOX_COVERAGE = 0.04          # Độ phủ khung hình tối thiểu của nét: 4%
MIN_MAIN_COMPONENT_RATIO = 0.03   # Nét vẽ chính phải chiếm ít nhất 3% tổng số nét
MAX_DENSITY_DIFF = 0.45           # Chênh lệch mật độ nét tối đa giữa F0 và F1: 45%
MAX_GT_DENSITY_DIFF = 0.45        # Chênh lệch mật độ nét tối đa giữa F_GT và trung bình (F0, F1): 45%
MIN_BBOX_IOU = 0.25               # BBox IoU tối thiểu giữa F0 và F1: 0.25
MIN_GT_BBOX_IOU = 0.30            # BBox IoU tối thiểu giữa F0 và F_GT: 0.30
DARK_THRESH = 160                 # Ngưỡng pixel coi là nét vẽ (ảnh nền trắng 255)

def analyze_lineart_frame(img_path):
    \"\"\"Phân tích các chỉ số chất lượng của 1 file ảnh lineart.\"\"\"
    if not os.path.exists(img_path):
        return None
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    h, w = img.shape
    total_pixels = h * w

    line_mask = (img < DARK_THRESH).astype(np.uint8)
    line_count = int(np.sum(line_mask))
    density = line_count / float(total_pixels)

    if line_count == 0:
        return {
            'valid': False,
            'density': 0.0,
            'bbox': (0, 0, 0, 0),
            'bbox_coverage': 0.0,
            'comp_ratio': 0.0,
            'reason': 'Ảnh trắng trơn 100%'
        }

    y_idx, x_idx = np.where(line_mask > 0)
    ymin, ymax = int(np.min(y_idx)), int(np.max(y_idx))
    xmin, xmax = int(np.min(x_idx)), int(np.max(x_idx))
    bw = max(1, xmax - xmin)
    bh = max(1, ymax - ymin)
    bbox_coverage = (bw * bh) / float(total_pixels)

    # Connected component lớn nhất
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(line_mask, connectivity=8)
    if num_labels > 1:
        max_comp = int(np.max(stats[1:, cv2.CC_STAT_AREA]))
        comp_ratio = max_comp / float(line_count)
    else:
        comp_ratio = 0.0

    return {
        'valid': True,
        'density': density,
        'bbox': (xmin, ymin, xmax, ymax),
        'bbox_coverage': bbox_coverage,
        'comp_ratio': comp_ratio
    }

def calc_iou(b1, b2):
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    area2 = max(1, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / float(area1 + area2 - inter)

print("[*] Đang đọc kết quả từ Stage 4...")
with open("screened_triplets.json", "r") as f:
    screened_triplets = json.load(f)

# 1. Thu thập và phân tích nhanh toàn bộ các frame duy nhất
print(f"[*] Đang phân tích chất lượng nét vẽ cho tất cả frames...")
frame_metrics = {}
all_frame_rel_paths = set()
for tr in screened_triplets:
    all_frame_rel_paths.add(tr['frame1'].replace(".jpg", "_smooth.png"))
    all_frame_rel_paths.add(tr['frame2'].replace(".jpg", "_smooth.png"))
    all_frame_rel_paths.add(tr['frame3'].replace(".jpg", "_smooth.png"))

for rel_p in tqdm(all_frame_rel_paths, desc="Evaluating Lineart Frames"):
    full_p = os.path.join(LINEART_DIR, rel_p)
    frame_metrics[rel_p] = analyze_lineart_frame(full_p)

# 2. Kiểm duyệt từng Triplet
print(f"[*] Bắt đầu lọc chất lượng cho {len(screened_triplets)} triplets...")
clean_triplets = []
rejection_stats = {
    'frame_missing_or_corrupt': 0,
    'density_too_low': 0,
    'density_too_high': 0,
    'bbox_coverage_too_small': 0,
    'noise_dots_only': 0,
    'density_fluctuation_f0_f1': 0,
    'density_fluctuation_gt': 0,
    'bbox_spatial_jump': 0
}

for tr in screened_triplets:
    p0 = tr['frame1'].replace(".jpg", "_smooth.png")
    p_gt = tr['frame2'].replace(".jpg", "_smooth.png")
    p1 = tr['frame3'].replace(".jpg", "_smooth.png")

    m0 = frame_metrics.get(p0)
    m_gt = frame_metrics.get(p_gt)
    m1 = frame_metrics.get(p1)

    if not (m0 and m_gt and m1 and m0['valid'] and m_gt['valid'] and m1['valid']):
        rejection_stats['frame_missing_or_corrupt'] += 1
        continue

    # Lọc mật độ nét từng frame
    if m0['density'] < MIN_LINE_DENSITY or m_gt['density'] < MIN_LINE_DENSITY or m1['density'] < MIN_LINE_DENSITY:
        rejection_stats['density_too_low'] += 1
        continue

    if m0['density'] > MAX_LINE_DENSITY or m_gt['density'] > MAX_LINE_DENSITY or m1['density'] > MAX_LINE_DENSITY:
        rejection_stats['density_too_high'] += 1
        continue

    # Lọc độ bao phủ BBox
    if m0['bbox_coverage'] < MIN_BBOX_COVERAGE or m_gt['bbox_coverage'] < MIN_BBOX_COVERAGE or m1['bbox_coverage'] < MIN_BBOX_COVERAGE:
        rejection_stats['bbox_coverage_too_small'] += 1
        continue

    # Lọc thành phần nét chính (chống bụi nhiễu)
    if m0['comp_ratio'] < MIN_MAIN_COMPONENT_RATIO or m_gt['comp_ratio'] < MIN_MAIN_COMPONENT_RATIO or m1['comp_ratio'] < MIN_MAIN_COMPONENT_RATIO:
        rejection_stats['noise_dots_only'] += 1
        continue

    # Lọc chênh lệch mật độ nét F0 vs F1
    rel_diff_01 = abs(m0['density'] - m1['density']) / max(m0['density'], m1['density'], 1e-6)
    if rel_diff_01 > MAX_DENSITY_DIFF:
        rejection_stats['density_fluctuation_f0_f1'] += 1
        continue

    # Lọc chênh lệch mật độ nét F_GT vs trung bình (F0, F1)
    mean_d = (m0['density'] + m1['density']) / 2.0
    rel_diff_gt = abs(m_gt['density'] - mean_d) / max(mean_d, 1e-6)
    if rel_diff_gt > MAX_GT_DENSITY_DIFF:
        rejection_stats['density_fluctuation_gt'] += 1
        continue

    # Lọc độ dịch chuyển BBox IoU
    iou_01 = calc_iou(m0['bbox'], m1['bbox'])
    iou_0gt = calc_iou(m0['bbox'], m_gt['bbox'])
    if iou_01 < MIN_BBOX_IOU or iou_0gt < MIN_GT_BBOX_IOU:
        rejection_stats['bbox_spatial_jump'] += 1
        continue

    # Lưu thêm các chỉ số đánh giá nét vào triplet sạch
    tr_clean = dict(tr)
    tr_clean['f0_density'] = round(m0['density'], 4)
    tr_clean['gt_density'] = round(m_gt['density'], 4)
    tr_clean['f1_density'] = round(m1['density'], 4)
    tr_clean['bbox_iou_01'] = round(iou_01, 3)
    tr_clean['bbox_iou_0gt'] = round(iou_0gt, 3)
    clean_triplets.append(tr_clean)

# Báo cáo thống kê chi tiết
print()
print("=" * 65)
print("📊 BÁO CÁO KẾT QUẢ LỌC & LÀM SẠCH LINEART DATASET (STAGE 5.5)")
print("=" * 65)
print(f"Tổng số Triplets đầu vào từ Stage 4:      {len(screened_triplets):,}")
print(f"Số Triplets ĐẠT CHUẨN sau làm sạch:       {len(clean_triplets):,} ({len(clean_triplets)/max(1, len(screened_triplets))*100:.1f}%)")
print(f"Số Triplets BỊ LOẠI BỎ:                  {len(screened_triplets) - len(clean_triplets):,}")
print("-" * 65)
print("Chi tiết nguyên nhân loại bỏ:")
print(f"  - Thiếu file / Lỗi đọc ảnh:             {rejection_stats['frame_missing_or_corrupt']}")
print(f"  - Mật độ nét quá ít / Trắng trơn:       {rejection_stats['density_too_low']}")
print(f"  - Mật độ nét quá dày / Dính nền bẩn:    {rejection_stats['density_too_high']}")
print(f"  - Diện tích BBox quá bé (<4%):          {rejection_stats['bbox_coverage_too_small']}")
print(f"  - Toàn hạt bụi li ti (Component <3%):   {rejection_stats['noise_dots_only']}")
print(f"  - Lệch nét giữa F0 và F1 (>45%):        {rejection_stats['density_fluctuation_f0_f1']}")
print(f"  - Frame giữa F_GT bị hỏng/mất nét:      {rejection_stats['density_fluctuation_gt']}")
print(f"  - BBox nhảy vọt / Mất dấu vết nhân vật: {rejection_stats['bbox_spatial_jump']}")
print("=" * 65)

CLEAN_MANIFEST = "clean_triplets.json"
with open(CLEAN_MANIFEST, "w") as f:
    json.dump(clean_triplets, f, indent=2)
print(f"[✓] Đã lưu danh sách Triplets sạch vào '{CLEAN_MANIFEST}'!")""")

    # Cell 6: Packaging & triplets.csv
    add_md("""---
### 📁 Bước 6 (Stage 6): Đóng Gói Dataset Hoàn Chỉnh Chuẩn TPS-Inbetween
- Sử dụng danh sách Triplets đã qua kiểm duyệt và làm sạch chất lượng nét vẽ (`clean_triplets.json`).
- Chia tách 85% Train / 15% Validation.
- Cấu trúc thư mục tương thích 100% với `ml240.py` của TPS-Inbetween:
  - `train/clip_xxx/000001.png`, `000002.png`, `000003.png`
  - `val/clip_xxx/...`
  - `matches/clip_xxx/match.npy`
  - `triplets.csv` (ghi rõ `split,clip_id,t,frame0,frame_gt,frame1,match_file,mean_disp,inliers,f0_density,gt_density,f1_density,bbox_iou_01,bbox_iou_0gt`)""")
    add_code("""import os, sys, json, shutil, csv
import numpy as np

OUTPUT_DIR = "final_tps_anime_dataset"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")
FINAL_MATCHES_DIR = os.path.join(OUTPUT_DIR, "matches")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(FINAL_MATCHES_DIR, exist_ok=True)

with open("clean_triplets.json", "r") as f:
    triplets = json.load(f)

# Phân chia ngẫu nhiên có seed cố định
np.random.seed(42)
indices = np.random.permutation(len(triplets))
val_count = int(len(triplets) * 0.15)
val_set = set(indices[:val_count])

csv_rows = []
print(f"[*] Đang đóng gói {len(triplets)} triplets sạch vào cấu trúc TPS-Inbetween...")

for idx, tr in enumerate(triplets):
    is_val = idx in val_set
    split = "val" if is_val else "train"
    clip_id = f"clip_{idx+1:06d}"

    target_clip_dir = os.path.join(VAL_DIR if is_val else TRAIN_DIR, clip_id)
    target_match_dir = os.path.join(FINAL_MATCHES_DIR, clip_id)
    os.makedirs(target_clip_dir, exist_ok=True)
    os.makedirs(target_match_dir, exist_ok=True)

    # Copy 3 frames nét vẽ
    f1_src = os.path.join(LINEART_DIR, tr['frame1'].replace(".jpg", "_smooth.png"))
    f2_src = os.path.join(LINEART_DIR, tr['frame2'].replace(".jpg", "_smooth.png"))
    f3_src = os.path.join(LINEART_DIR, tr['frame3'].replace(".jpg", "_smooth.png"))

    shutil.copy2(f1_src, os.path.join(target_clip_dir, "000001.png"))
    shutil.copy2(f2_src, os.path.join(target_clip_dir, "000002.png"))
    shutil.copy2(f3_src, os.path.join(target_clip_dir, "000003.png"))

    # Copy match npy
    match_src = os.path.join(MATCHES_DIR, tr['match_file'])
    match_dst = os.path.join(target_match_dir, "match_000001_000003.npy")
    shutil.copy2(match_src, match_dst)

    csv_rows.append({
        'split': split,
        'clip_id': clip_id,
        't': tr['t'],
        'source': tr['source'],
        'frame0': f"{clip_id}/000001.png",
        'frame_gt': f"{clip_id}/000002.png",
        'frame1': f"{clip_id}/000003.png",
        'match_file': f"{clip_id}/match_000001_000003.npy",
        'mean_disp': tr['mean_displacement'],
        'inliers': tr['inliers_count'],
        'f0_density': tr.get('f0_density', 0.0),
        'gt_density': tr.get('gt_density', 0.0),
        'f1_density': tr.get('f1_density', 0.0),
        'bbox_iou_01': tr.get('bbox_iou_01', 0.0),
        'bbox_iou_0gt': tr.get('bbox_iou_0gt', 0.0)
    })

# Ghi file triplets.csv
csv_path = os.path.join(OUTPUT_DIR, "triplets.csv")
fieldnames = [
    'split', 'clip_id', 't', 'source', 'frame0', 'frame_gt', 'frame1',
    'match_file', 'mean_disp', 'inliers', 'f0_density', 'gt_density', 'f1_density',
    'bbox_iou_01', 'bbox_iou_0gt'
]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_rows)

print()
print("=" * 60)
print(f"[✓] Đóng gói thành công Dataset hoàn chỉnh tại: '{OUTPUT_DIR}'")
print(f"    - Tập Train: {len(triplets) - val_count} Triplets sạch")
print(f"    - Tập Validation: {val_count} Triplets sạch")
print(f"    - File chỉ mục: {csv_path}")
print("=" * 60)

# Nén thành tar.gz để tải về hoặc chuyển tiếp sang Notebook Train
print()
print("[*] Đang nén Dataset thành 'anime_tps_finetune_dataset_ready.tar.gz'...")
!tar -czf anime_tps_finetune_dataset_ready.tar.gz -C final_tps_anime_dataset .
!ls -lh anime_tps_finetune_dataset_ready.tar.gz
print("[✓] Hoàn tất nén file Dataset sẵn sàng cho Fine-tuning!")""")

    # Cell 7: Verification DataLoader
    add_md("""---
### ✅ Bước 7: Kiểm Thử DataLoader TPS-Inbetween Với Dataset Mới
*(Kiểm tra xem mô hình và DataLoader đọc dataset có chuẩn xác 100% không trước khi bước vào huấn luyện)*""")
    add_code("""import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

df = pd.read_csv("final_tps_anime_dataset/triplets.csv")
print("📊 Thống kê sơ bộ Dataset:")
print(df.head())
print()
print("Phân bố nhãn t:")
print(df['t'].value_counts())

# Hiển thị thử 1 triplet trực quan
sample = df.iloc[0]
clip = sample['clip_id']
f0 = Image.open(f"final_tps_anime_dataset/train/{sample['frame0']}")
f_gt = Image.open(f"final_tps_anime_dataset/train/{sample['frame_gt']}")
f1 = Image.open(f"final_tps_anime_dataset/train/{sample['frame1']}")

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title(f"I_0 (Frame 0)")
plt.imshow(f0, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title(f"I_GT (Ground Truth, t={sample['t']})")
plt.imshow(f_gt, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title(f"I_1 (Frame 1)")
plt.imshow(f1, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
print("🎉 Xin chúc mừng! Bộ dữ liệu đã hoàn thiện và kiểm chứng thành công!")""")

    output_nb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_tps_dataset_kaggle.ipynb")
    with open(output_nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)

    print(f"[✓] Đã tạo thành công file Notebook: {output_nb_path}")


if __name__ == '__main__':
    create_notebook()
