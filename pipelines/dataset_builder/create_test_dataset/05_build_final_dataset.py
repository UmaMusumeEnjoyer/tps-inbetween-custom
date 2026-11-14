#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_build_final_dataset.py (05_build_final_dataset.py)
------------------------------------------------------
Giai đoạn Deep Learning & Đóng gói Dataset hoàn chỉnh cho TPS-Inbetween trên máy tính cá nhân (Local PC).

Quy trình thực thi tuần tự:
1. [Stage 4] GlueStick Keypoint Matching & RANSAC Screening:
   - Đọc Candidate Triplets từ `candidate_triplets_anime_540p` (5,723 triplets vừa tạo ở bước trước).
   - Chạy mạng GlueStick đo độ dời chuyển động (displacement) giữa frame1 (I_0) và frame3 (I_1) trên GPU CUDA.
   - Áp dụng bộ lọc RANSAC Affine: loại bỏ cặp chuyển động quá lớn (> 15.0px) hoặc quá ít inliers (< 20).
   - Lưu trữ các cặp điểm khớp inliers sạch chuẩn hóa [0, 1] thành file `.npy`.
   - Xuất file `screened_triplets.json`.

2. [Stage 5] Line-Art Extraction & Cel-Shading Boundary Suppression:
   - Tách nền nhân vật anime bằng mô hình ISNetDIS (anime-segmentation).
   - Trích xuất nét vẽ hairline sắc nét bằng mô hình AniLines (Detail mode).
   - Làm mượt nét bằng bộ lọc MS-SSAA Lanczos-4 + Bilateral Filter (khử răng cưa và triệt tiêu viền bóng cel-shading).
   - Lưu trữ các frame nét vẽ mịn `_smooth.png`.

3. [Stage 5.5] Lineart Quality & Consistency Filter (Kiểm duyệt & Làm sạch Dataset):
   - Mật độ nét vẽ: D in [0.8%, 22%] (loại bỏ frame trắng trơn hoặc dính nền bẩn).
   - Độ phủ Bounding Box: >= 4% (loại bỏ frame chỉ còn vài đốm đen mép ảnh).
   - Thành phần nét chính: >= 3% (loại bỏ frame chỉ toàn hạt bụi/nhiễu li ti rời rạc).
   - Nhất quán mật độ nét: diff(F0, F1) <= 45%, diff(F_GT, trung bình) <= 45%.
   - Nhất quán không gian BBox: IoU(F0, F1) >= 0.25, IoU(F0, F_GT) >= 0.30 (chống nhảy góc máy / teleport).
   - Xuất file `clean_triplets.json` chứa các triplet đạt chuẩn kiểm duyệt.

4. [Stage 6] Export & Packaging Dataset Chuẩn TPS-Inbetween:
   - Phân chia 85% Train / 15% Validation (seed cố định).
   - Cấu trúc chuẩn ml240:
     train/clip_XXXXXX/000001.png, 000002.png, 000003.png
     val/clip_XXXXXX/000001.png, 000002.png, 000003.png
     matches/clip_XXXXXX/match_000001_000003.npy
     triplets.csv (split, clip_id, t, source, frame0, frame_gt, frame1, match_file, mean_disp, inliers, f0_density, gt_density, f1_density, bbox_iou_01, bbox_iou_0gt).
   - Nén kết quả thành `anime_tps_finetune_dataset_ready.tar.gz`.
"""

import os
import sys
import glob
import json
import time
import shutil
import random
import zipfile
import csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ==============================================================================
# CẤU HÌNH ĐƯỜNG DẪN CÁC MODULE & TRỌNG SỐ SẴN CÓ TRÊN MÁY TÍNH
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PBL6_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

# 1. Thêm site-packages của TestDataset vào sys.path để nạp omegaconf, timm, einops nếu cần
test_dataset_site = os.path.join(PBL6_DIR, "TestDataset", ".venv", "Lib", "site-packages")
if os.path.exists(test_dataset_site) and test_dataset_site not in sys.path:
    sys.path.insert(0, test_dataset_site)

# 2. Thêm đường dẫn tps-inbetween (chứa module gluestick)
TPS_ROOT = os.path.join(PBL6_DIR, "Test", "TestModels", "tps-inbetween")
if os.path.exists(TPS_ROOT) and TPS_ROOT not in sys.path:
    sys.path.insert(0, TPS_ROOT)

# 3. Thêm đường dẫn AniLines
ANILINES_DIR = os.path.join(PBL6_DIR, "TestDataset", "AniLines-Anime-Lineart-Extractor")
if os.path.exists(ANILINES_DIR) and ANILINES_DIR not in sys.path:
    sys.path.insert(0, ANILINES_DIR)

# 4. Thêm đường dẫn Smooth Lines
SMOOTH_DIR = os.path.join(PBL6_DIR, "TestDataset", "ChayGoAnimeLinesDetail_NoBG")
if os.path.exists(SMOOTH_DIR) and SMOOTH_DIR not in sys.path:
    sys.path.insert(0, SMOOTH_DIR)

# ĐƯỜNG DẪN TRỌNG SỐ MÔ HÌNH CỤC BỘ
GLUESTICK_WEIGHTS = os.path.join(TPS_ROOT, "model", "resources", "weights", "checkpoint_GlueStick_MD.tar")
ANIMESEG_CKPT = os.path.join(PBL6_DIR, "TestDataset", "anime-segmentation", "saved_models", "isnetis.ckpt")
ANILINES_WEIGHTS = os.path.join(ANILINES_DIR, "weights", "detail.pth")

# Nguồn dữ liệu Candidate Triplets (vừa được tạo ở bước trước)
CANDIDATE_DIR = os.path.join(BASE_DIR, "candidate_triplets_anime_540p")
CANDIDATE_ZIP = os.path.join(BASE_DIR, "candidate_triplets_anime_540p.zip")
CANDIDATE_MANIFEST = os.path.join(CANDIDATE_DIR, "dataset_manifest.json")

# Thư mục làm việc và kết quả đầu ra
WORK_FINAL_DIR = os.path.join(BASE_DIR, "work_final_stages")
MATCHES_DIR = os.path.join(WORK_FINAL_DIR, "matches")
LINEART_DIR = os.path.join(WORK_FINAL_DIR, "lineart")
SCREENED_MANIFEST = os.path.join(WORK_FINAL_DIR, "screened_triplets.json")

OUTPUT_DATASET_DIR = os.path.join(BASE_DIR, "final_tps_anime_dataset")
FINAL_ARCHIVE_PATH = os.path.join(BASE_DIR, "anime_tps_finetune_dataset_ready.tar.gz")

# THAM SỐ LỌC GLUESTIK & RANSAC
DEVICE = 'cuda:0'
SCREEN_MIN_INLIERS = 20          # Tối thiểu 20 inlier sạch
SCREEN_MAX_MEAN_DISP = 15.0      # Độ dời trung bình <= 15.0px (vùng an toàn toán học của TPS)
SCREEN_MAX_OUTLIER_DISP = 50.0   # Loại bỏ outlier lệch > 50px
SCREEN_RANSAC_THRESH = 5.0       # Ngưỡng RANSAC Affine 5.0px
VAL_RATIO = 0.15                 # 15% Validation, 85% Train

# THAM SỐ LỌC & LÀM SẠCH LINEART (STAGE 5.5)
MIN_LINE_DENSITY = 0.008          # Mật độ nét tối thiểu: 0.8%
MAX_LINE_DENSITY = 0.22           # Mật độ nét tối đa: 22%
MIN_BBOX_COVERAGE = 0.04          # Độ phủ khung hình tối thiểu của nét: 4%
MIN_MAIN_COMPONENT_RATIO = 0.03   # Nét vẽ chính phải chiếm ít nhất 3% tổng số nét
MAX_DENSITY_DIFF = 0.45           # Chênh lệch mật độ nét tối đa giữa F0 và F1: 45%
MAX_GT_DENSITY_DIFF = 0.45        # Chênh lệch mật độ nét tối đa giữa F_GT và trung bình (F0, F1): 45%
MIN_BBOX_IOU = 0.25               # BBox IoU tối thiểu giữa F0 và F1: 0.25
MIN_GT_BBOX_IOU = 0.30            # BBox IoU tối thiểu giữa F0 và F_GT: 0.30
DARK_THRESH = 160                 # Ngưỡng pixel coi là nét vẽ (ảnh nền trắng 255)
CLEAN_MANIFEST = os.path.join(WORK_FINAL_DIR, "clean_triplets.json")


def format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# ==============================================================================
# HÀM LÀM MƯỢT NÉT MS-SSAA LANCZOS-4 + BILATERAL
# ==============================================================================
def smooth_lineart_channel(channel, scale=4, sigma_color=35.0, sigma_space=5.0, smooth_iters=1):
    """
    Làm mượt nét vẽ bằng Multi-Scale Super-Sampling Anti-Aliasing (MS-SSAA)
    kết hợp Bilateral Filter bảo toàn đường biên và loại bỏ răng cưa.
    """
    import cv2
    import numpy as np

    h, w = channel.shape[:2]
    is_white_bg = bool(channel.mean() > 127)
    fg = (255.0 - channel.astype(np.float32)) if is_white_bg else channel.astype(np.float32)

    # 1. Super-Sampling phóng to bằng Lanczos-4
    up_w, up_h = w * scale, h * scale
    up_fg = cv2.resize(fg, (up_w, up_h), interpolation=cv2.INTER_LANCZOS4)

    # 2. Bilateral Filter trên canvas phóng đại
    up_uint8 = np.clip(up_fg, 0, 255).astype(np.uint8)
    filtered = up_uint8
    for i in range(smooth_iters):
        sc = sigma_color if i == 0 else sigma_color * 0.6
        ss = sigma_space if i == 0 else sigma_space * 0.7
        filtered = cv2.bilateralFilter(
            filtered,
            d=max(5, int(scale * 1.5) | 1),
            sigmaColor=sc,
            sigmaSpace=ss
        )

    # 3. Downsample Area
    down = cv2.resize(filtered.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    norm = np.clip(down, 0, 255).astype(np.uint8)
    return 255 - norm if is_white_bg else norm


# ==============================================================================
# GIAI ĐOẠN 4: GLUESTIK KEYPOINT MATCHING & RANSAC FILTER
# ==============================================================================
def run_stage4_gluestick(candidates, device=DEVICE):
    import cv2
    import torch
    import numpy as np
    from tqdm import tqdm
    from model.gluestick.models.two_view_pipeline import TwoViewPipeline

    print("\n" + "=" * 75)
    print("🔍 [STAGE 4/6]: SO KHỚP GLUESTIK & KIỂM DUYỆT RANSAC TRÊN GPU")
    print(f"   - Tổng số Candidate Triplets: {len(candidates):,} mẫu")
    print(f"   - Ngưỡng Inliers tối thiểu:   >= {SCREEN_MIN_INLIERS} điểm")
    print(f"   - Độ dời trung bình tối đa:   <= {SCREEN_MAX_MEAN_DISP} px")
    print("=" * 75)

    os.makedirs(MATCHES_DIR, exist_ok=True)

    import types, gc
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

    fake_pytlsd = types.ModuleType("pytlsd")
    fake_pytlsd.lsd = cv2_lsd
    sys.modules["pytlsd"] = fake_pytlsd

    try:
        import model.gluestick.models.wireframe as wireframe_mod
        wireframe_mod.lsd = cv2_lsd
    except Exception:
        pass

    # Khởi tạo GlueStick model
    conf = {
        'name': 'two_view_pipeline',
        'use_lines': True,
        'extractor': {
            'name': 'wireframe',
            'sp_params': {'force_num_keypoints': False, 'max_num_keypoints': 1000},
            'wireframe_params': {'merge_points': True, 'merge_line_endpoints': True},
            'max_n_lines': 300,
        },
        'matcher': {'name': 'gluestick', 'weights': GLUESTICK_WEIGHTS, 'trainable': False},
        'ground_truth': {'from_pose_depth': False}
    }
    print("[*] Đang khởi tạo GlueStick Pipeline trên GPU (OpenCV LSD Mode - Zero Leak)...")
    model = TwoViewPipeline(conf).to(device)
    model.eval()
    print("[✓] GlueStick đã sẵn sàng!")

    screened = []
    start_time = time.time()

    with tqdm(total=len(candidates), desc="Stage 4 (GlueStick RANSAC)",
              bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        for idx, tr in enumerate(candidates):
            tid = tr['triplet_id']
            f1_path = os.path.join(CANDIDATE_DIR, tr['frame1'])
            f3_path = os.path.join(CANDIDATE_DIR, tr['frame3'])

            if not (os.path.exists(f1_path) and os.path.exists(f3_path)):
                pbar.update(1)
                continue

            match_file_name = f"{tid}.npy"
            match_path = os.path.join(MATCHES_DIR, match_file_name)
            if os.path.exists(match_path):
                try:
                    kp_matches = np.load(match_path)
                    img0_tmp = cv2.imread(f1_path)
                    if img0_tmp is not None:
                        h_tmp, w_tmp = img0_tmp.shape[:2]
                        del img0_tmp
                        num_inliers = kp_matches.shape[1]
                        diff_px = (kp_matches[0] - kp_matches[1]) * np.array([w_tmp, h_tmp])
                        mean_disp = float(np.mean(np.linalg.norm(diff_px, axis=1)))
                        tr_clean = dict(tr)
                        tr_clean['match_file'] = match_file_name
                        tr_clean['inliers_count'] = int(num_inliers)
                        tr_clean['mean_displacement'] = round(mean_disp, 2)
                        screened.append(tr_clean)
                        pbar.update(1)
                        continue
                except Exception:
                    pass

            img0 = cv2.imread(f1_path)
            img1 = cv2.imread(f3_path)
            if img0 is None or img1 is None:
                pbar.update(1)
                continue

            h, w = img0.shape[:2]

            # Đưa frame vào tensor GPU
            t0 = torch.from_numpy(cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)).float().to(device)[None, None] / 255.0
            t1 = torch.from_numpy(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)).float().to(device)[None, None] / 255.0

            data = {'image0': t0, 'image1': t1}
            with torch.no_grad():
                pred = model(data)

            m_kpts0 = pred['matches0'][0].cpu().numpy()
            kpts0 = pred['keypoints0'][0].cpu().numpy()
            kpts1 = pred['keypoints1'][0].cpu().numpy()

            del t0, t1, data, pred, img0, img1

            valid = m_kpts0 > -1
            p0 = kpts0[valid]
            p1 = kpts1[m_kpts0[valid]]

            if len(p0) < SCREEN_MIN_INLIERS:
                pbar.update(1)
                continue

            # Lọc Affine RANSAC
            M, inliers = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=SCREEN_RANSAC_THRESH)
            if inliers is None:
                pbar.update(1)
                continue

            mask = inliers.flatten().astype(bool)
            inliers0 = p0[mask]
            inliers1 = p1[mask]

            if len(inliers0) < SCREEN_MIN_INLIERS:
                pbar.update(1)
                continue

            disps = np.linalg.norm(inliers1 - inliers0, axis=1)
            mean_disp = float(np.mean(disps))
            max_disp = float(np.max(disps))

            if mean_disp > SCREEN_MAX_MEAN_DISP or max_disp > SCREEN_MAX_OUTLIER_DISP:
                pbar.update(1)
                continue

            # Chuẩn hóa tọa độ [0, 1] và lưu file .npy
            flow_norm0 = inliers0.copy()
            flow_norm0[:, 0] /= float(w)
            flow_norm0[:, 1] /= float(h)

            flow_norm1 = inliers1.copy()
            flow_norm1[:, 0] /= float(w)
            flow_norm1[:, 1] /= float(h)

            kp_matches = np.stack([flow_norm0, flow_norm1], axis=0)  # Shape [2, N, 2]
            np.save(match_path, kp_matches)

            tr_clean = dict(tr)
            tr_clean['match_file'] = match_file_name
            tr_clean['inliers_count'] = int(len(inliers0))
            tr_clean['mean_displacement'] = round(mean_disp, 2)
            screened.append(tr_clean)

            if (idx + 1) % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Cập nhật postfix
            elapsed_sec = time.time() - start_time
            rate = len(screened) / elapsed_sec if elapsed_sec > 0 else 0
            pbar.set_postfix({
                'Đạt chuẩn': f"{len(screened)}/{idx+1} ({len(screened)/(idx+1)*100:.1f}%)",
                'Tốc độ': f"{rate:.1f} tr/s"
            })
            pbar.update(1)

    with open(SCREENED_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(screened, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 75)
    print(f"[✓] Hoàn tất Stage 4: {len(screened):,} / {len(candidates):,} Triplets vượt qua kiểm duyệt RANSAC!")
    print(f"    - Tỷ lệ đạt chuẩn: {len(screened) / len(candidates) * 100:.1f}%")
    print(f"    - Metadata lưu tại: {SCREENED_MANIFEST}")
    print("=" * 75)
    return screened


# ==============================================================================
# GIAI ĐOẠN 5: ISNetDIS + ANILINES DETAIL + MS-SSAA SMOOTHING
# ==============================================================================
def run_stage5_lineart(screened_triplets, device=DEVICE):
    import cv2
    import torch
    import numpy as np
    from tqdm import tqdm
    from PIL import Image
    from network.line_extractor import LineExtractor
    from network.anime_seg import AnimeBgRemover

    print("\n" + "=" * 75)
    print("🎨 [STAGE 5/6]: TRÍCH XUẤT NÉT VẼ ISNetDIS + ANILINES + MS-SSAA LÀM MƯỢT")
    print("=" * 75)

    os.makedirs(LINEART_DIR, exist_ok=True)

    # 1. Thu thập danh sách frame duy nhất cần xử lý (tránh xử lý trùng)
    unique_rel_frames = set()
    for tr in screened_triplets:
        unique_rel_frames.add(tr['frame1'])
        unique_rel_frames.add(tr['frame2'])
        unique_rel_frames.add(tr['frame3'])

    unique_rel_frames = sorted(list(unique_rel_frames))
    print(f"[*] Tổng số khung hình duy nhất cần trích xuất nét: {len(unique_rel_frames):,} frames.")

    # 2. Tải mô hình ISNetDIS và AniLines
    print("[*] Đang tải mô hình ISNetDIS (Tách nền anime)...")
    bg_remover = AnimeBgRemover(ckpt_path=ANIMESEG_CKPT, device=device, img_size=1024)

    print("[*] Đang tải mô hình AniLines Detail...")
    line_extractor = LineExtractor(2, 1, True).to(device)
    line_extractor.load_state_dict(torch.load(ANILINES_WEIGHTS, map_location=device, weights_only=True))
    line_extractor.eval()
    for p in line_extractor.parameters():
        p.requires_grad = False
    print("[✓] Các mô hình Line-art đã sẵn sàng!")

    start_time = time.time()
    processed_count = 0

    with tqdm(total=len(unique_rel_frames), desc="Stage 5 (Lineart Extraction)",
              bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} frames [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        for rel_frame in unique_rel_frames:
            src_f = os.path.join(CANDIDATE_DIR, rel_frame)
            dst_f = os.path.join(LINEART_DIR, rel_frame.replace(".jpg", "_smooth.png"))

            if not os.path.exists(src_f):
                pbar.update(1)
                continue

            # Nếu đã tồn tại thì bỏ qua (hỗ trợ resume)
            if os.path.exists(dst_f):
                processed_count += 1
                pbar.update(1)
                continue

            os.makedirs(os.path.dirname(dst_f), exist_ok=True)

            img_bgr = cv2.imread(src_f)
            if img_bgr is None:
                pbar.update(1)
                continue

            # 1. Tách nền bằng ISNetDIS (thay nền bằng màu trắng)
            img_white = bg_remover.remove_background(img_bgr)

            # 2. Trích xuất nét vẽ bằng AniLines
            img_gray = cv2.cvtColor(img_white, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel = cv2.magnitude(sobelx, sobely)
            sobel = 255 - cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

            t_img = torch.from_numpy(img_gray).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0
            t_sobel = torch.from_numpy(sobel).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0
            x_in = torch.cat([t_img, t_sobel], dim=1)

            H, W = img_gray.shape
            pad_h = 8 - (H % 8) if (H % 8) != 0 else 0
            pad_w = 8 - (W % 8) if (W % 8) != 0 else 0
            x_in = torch.nn.functional.pad(x_in, (0, pad_w, 0, pad_h), mode='reflect')

            with torch.no_grad():
                pred_line = line_extractor(x_in)[:, :, :H, :W]

            lineart_np = (pred_line.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

            # 3. Làm mượt nét MS-SSAA Lanczos-4 + Bilateral
            smooth_out = smooth_lineart_channel(lineart_np, scale=4, sigma_color=35.0, sigma_space=5.0)
            cv2.imwrite(dst_f, smooth_out)

            processed_count += 1
            pbar.update(1)

    print("\n" + "=" * 75)
    print(f"[✓] Hoàn tất Stage 5: Đã trích xuất xong toàn bộ {len(unique_rel_frames):,} frames nét vẽ mịn!")
    print(f"    - Thư mục lưu: {LINEART_DIR}")
    print("=" * 75)


# ==============================================================================
# GIAI ĐOẠN 5.5: KIỂM DUYỆT & LÀM SẠCH DATASET DỰA TRÊN LINEART QUALITY
# ==============================================================================
def analyze_lineart_frame(img_path):
    """Phân tích các chỉ số chất lượng của 1 file ảnh lineart."""
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

def calc_bbox_iou(b1, b2):
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    area2 = max(1, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / float(area1 + area2 - inter)

def run_stage5_5_clean_lineart(screened_triplets):
    print("\n" + "=" * 75)
    print("🧹 [STAGE 5.5/6]: KIỂM DUYỆT & LÀM SẠCH DATASET DỰA TRÊN CHẤT LƯỢNG LINEART")
    print(f"   - Mật độ nét hợp lệ:     [{MIN_LINE_DENSITY*100:.1f}%, {MAX_LINE_DENSITY*100:.1f}%]")
    print(f"   - Độ phủ BBox tối thiểu: {MIN_BBOX_COVERAGE*100:.1f}%")
    print(f"   - Biến thiên nét F0/F1:  <= {MAX_DENSITY_DIFF*100:.0f}%")
    print(f"   - Biến thiên nét F_GT:   <= {MAX_GT_DENSITY_DIFF*100:.0f}%")
    print(f"   - BBox IoU F0/F1:        >= {MIN_BBOX_IOU:.2f}")
    print(f"   - BBox IoU F0/F_GT:      >= {MIN_GT_BBOX_IOU:.2f}")
    print("=" * 75)

    # 1. Đánh giá chất lượng toàn bộ frames duy nhất
    unique_frame_rel = set()
    for tr in screened_triplets:
        unique_frame_rel.add(tr['frame1'].replace(".jpg", "_smooth.png"))
        unique_frame_rel.add(tr['frame2'].replace(".jpg", "_smooth.png"))
        unique_frame_rel.add(tr['frame3'].replace(".jpg", "_smooth.png"))

    print(f"[*] Đang phân tích chất lượng nét vẽ cho {len(unique_frame_rel):,} frames (Đa luồng CPU)...")
    frame_metrics = {}

    def _eval_frame(rel_p):
        full_p = os.path.join(LINEART_DIR, rel_p)
        return rel_p, analyze_lineart_frame(full_p)

    max_workers = min(16, max(1, os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for rel_p, m in tqdm(executor.map(_eval_frame, unique_frame_rel),
                             total=len(unique_frame_rel),
                             desc="Evaluating Lineart Frames"):
            frame_metrics[rel_p] = m

    # 2. Kiểm duyệt từng Triplet
    print(f"[*] Đang lọc chất lượng cho {len(screened_triplets):,} triplets...")
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

        if m0['density'] < MIN_LINE_DENSITY or m_gt['density'] < MIN_LINE_DENSITY or m1['density'] < MIN_LINE_DENSITY:
            rejection_stats['density_too_low'] += 1
            continue

        if m0['density'] > MAX_LINE_DENSITY or m_gt['density'] > MAX_LINE_DENSITY or m1['density'] > MAX_LINE_DENSITY:
            rejection_stats['density_too_high'] += 1
            continue

        if m0['bbox_coverage'] < MIN_BBOX_COVERAGE or m_gt['bbox_coverage'] < MIN_BBOX_COVERAGE or m1['bbox_coverage'] < MIN_BBOX_COVERAGE:
            rejection_stats['bbox_coverage_too_small'] += 1
            continue

        if m0['comp_ratio'] < MIN_MAIN_COMPONENT_RATIO or m_gt['comp_ratio'] < MIN_MAIN_COMPONENT_RATIO or m1['comp_ratio'] < MIN_MAIN_COMPONENT_RATIO:
            rejection_stats['noise_dots_only'] += 1
            continue

        rel_diff_01 = abs(m0['density'] - m1['density']) / max(m0['density'], m1['density'], 1e-6)
        if rel_diff_01 > MAX_DENSITY_DIFF:
            rejection_stats['density_fluctuation_f0_f1'] += 1
            continue

        mean_d = (m0['density'] + m1['density']) / 2.0
        rel_diff_gt = abs(m_gt['density'] - mean_d) / max(mean_d, 1e-6)
        if rel_diff_gt > MAX_GT_DENSITY_DIFF:
            rejection_stats['density_fluctuation_gt'] += 1
            continue

        iou_01 = calc_bbox_iou(m0['bbox'], m1['bbox'])
        iou_0gt = calc_bbox_iou(m0['bbox'], m_gt['bbox'])
        if iou_01 < MIN_BBOX_IOU or iou_0gt < MIN_GT_BBOX_IOU:
            rejection_stats['bbox_spatial_jump'] += 1
            continue

        tr_clean = dict(tr)
        tr_clean['f0_density'] = round(m0['density'], 4)
        tr_clean['gt_density'] = round(m_gt['density'], 4)
        tr_clean['f1_density'] = round(m1['density'], 4)
        tr_clean['bbox_iou_01'] = round(iou_01, 3)
        tr_clean['bbox_iou_0gt'] = round(iou_0gt, 3)
        clean_triplets.append(tr_clean)

    print("\n" + "=" * 75)
    print("📊 BÁO CÁO KẾT QUẢ LỌC & LÀM SẠCH LINEART DATASET (STAGE 5.5)")
    print("=" * 75)
    print(f"Tổng số Triplets đầu vào từ Stage 4:      {len(screened_triplets):,}")
    print(f"Số Triplets ĐẠT CHUẨN sau làm sạch:       {len(clean_triplets):,} ({len(clean_triplets)/max(1, len(screened_triplets))*100:.1f}%)")
    print(f"Số Triplets BỊ LOẠI BỎ:                  {len(screened_triplets) - len(clean_triplets):,}")
    print("-" * 75)
    print("Chi tiết nguyên nhân loại bỏ:")
    print(f"  - Thiếu file / Lỗi đọc ảnh:             {rejection_stats['frame_missing_or_corrupt']}")
    print(f"  - Mật độ nét quá ít / Trắng trơn:       {rejection_stats['density_too_low']}")
    print(f"  - Mật độ nét quá dày / Dính nền bẩn:    {rejection_stats['density_too_high']}")
    print(f"  - Diện tích BBox quá bé (<4%):          {rejection_stats['bbox_coverage_too_small']}")
    print(f"  - Toàn hạt bụi li ti (Component <3%):   {rejection_stats['noise_dots_only']}")
    print(f"  - Lệch nét giữa F0 và F1 (>45%):        {rejection_stats['density_fluctuation_f0_f1']}")
    print(f"  - Frame giữa F_GT bị hỏng/mất nét:      {rejection_stats['density_fluctuation_gt']}")
    print(f"  - BBox nhảy vọt / Mất dấu vết nhân vật: {rejection_stats['bbox_spatial_jump']}")
    print("=" * 75)

    with open(CLEAN_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(clean_triplets, f, indent=2)
    print(f"[✓] Đã lưu danh sách Triplets sạch vào '{CLEAN_MANIFEST}'!")

    return clean_triplets


# ==============================================================================
# GIAI ĐOẠN 6: PACKAGING & TRIPLETS.CSV CHUẨN TPS-INBETWEEN
# ==============================================================================
def run_stage6_export(clean_triplets, val_ratio=VAL_RATIO):
    print("\n" + "=" * 75)
    print("📦 [STAGE 6/6]: ĐÓNG GÓI DATASET HOÀN CHỈNH CHUẨN TPS-INBETWEEN (ML240)")
    print(f"   - Thư mục xuất: {OUTPUT_DATASET_DIR}")
    print(f"   - Tỷ lệ Train:  {(1.0 - val_ratio)*100:.0f}%")
    print(f"   - Tỷ lệ Val:    {val_ratio*100:.0f}%")
    print("=" * 75)

    train_dir = os.path.join(OUTPUT_DATASET_DIR, "train")
    val_dir = os.path.join(OUTPUT_DATASET_DIR, "val")
    final_matches_dir = os.path.join(OUTPUT_DATASET_DIR, "matches")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(final_matches_dir, exist_ok=True)

    # Phân chia ngẫu nhiên có kiểm soát
    random.seed(42)
    indices = list(range(len(clean_triplets)))
    random.shuffle(indices)
    val_count = int(len(clean_triplets) * val_ratio)
    val_set = set(indices[:val_count])

    csv_rows = []

    for idx, tr in enumerate(clean_triplets):
        is_val = idx in val_set
        split = "val" if is_val else "train"
        clip_id = f"clip_{idx+1:06d}"

        target_clip_dir = os.path.join(val_dir if is_val else train_dir, clip_id)
        target_match_dir = os.path.join(final_matches_dir, clip_id)
        os.makedirs(target_clip_dir, exist_ok=True)
        os.makedirs(target_match_dir, exist_ok=True)

        # 1. Sao chép 3 khung hình nét vẽ
        f1_src = os.path.join(LINEART_DIR, tr['frame1'].replace(".jpg", "_smooth.png"))
        f2_src = os.path.join(LINEART_DIR, tr['frame2'].replace(".jpg", "_smooth.png"))
        f3_src = os.path.join(LINEART_DIR, tr['frame3'].replace(".jpg", "_smooth.png"))

        shutil.copy2(f1_src, os.path.join(target_clip_dir, "000001.png"))
        shutil.copy2(f2_src, os.path.join(target_clip_dir, "000002.png"))
        shutil.copy2(f3_src, os.path.join(target_clip_dir, "000003.png"))

        # 2. Sao chép file matching keypoints
        match_src = os.path.join(MATCHES_DIR, tr['match_file'])
        match_dst = os.path.join(target_match_dir, "match_000001_000003.npy")
        shutil.copy2(match_src, match_dst)

        # 3. Ghi vào triplets.csv
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

    csv_path = os.path.join(OUTPUT_DATASET_DIR, "triplets.csv")
    fieldnames = [
        'split', 'clip_id', 't', 'source', 'frame0', 'frame_gt', 'frame1',
        'match_file', 'mean_disp', 'inliers', 'f0_density', 'gt_density', 'f1_density',
        'bbox_iou_01', 'bbox_iou_0gt'
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n" + "=" * 75)
    print(f"🎉 HOÀN THÀNH ĐÓNG GÓI DATASET CHUẨN TPS-INBETWEEN!")
    print(f"   - Thư mục Dataset: {OUTPUT_DATASET_DIR}")
    print(f"   - Tập Train:       {len(clean_triplets) - val_count:,} Triplets sạch")
    print(f"   - Tập Validation:  {val_count:,} Triplets sạch")
    print(f"   - File chỉ mục:    {csv_path}")
    print("=" * 75)


# ==============================================================================
# HÀM MAIN KIỂM TRA ĐIỀU KIỆN & CHẠY TOÀN BỘ PIPELINE
# ==============================================================================
def main():
    print("=" * 75)
    print("🚀 PIPELINE TẠO DATASET TPS-INBETWEEN HOÀN CHỈNH (STAGES 4 -> 6)")
    print("=" * 75)

    # 1. Kiểm tra file Candidate Triplets
    if not os.path.exists(CANDIDATE_MANIFEST):
        if os.path.exists(CANDIDATE_ZIP):
            print(f"[*] Chưa có thư mục '{CANDIDATE_DIR}', đang giải nén từ '{CANDIDATE_ZIP}'...")
            with zipfile.ZipFile(CANDIDATE_ZIP, 'r') as zf:
                zf.extractall(CANDIDATE_DIR)
            print("[✓] Đã giải nén xong Candidate Triplets!")
        else:
            raise FileNotFoundError(f"Không tìm thấy file: {CANDIDATE_ZIP}. Hãy chạy run_extract_and_mix.py trước!")

    with open(CANDIDATE_MANIFEST, "r", encoding="utf-8") as f:
        candidates = json.load(f)
    print(f"[✓] Đã nạp manifest với {len(candidates):,} Candidate Triplets.")

    # 2. Kiểm tra trọng số mô hình
    print("\n[*] Kiểm tra các bộ trọng số Deep Learning...")
    weights_check = [
        ("GlueStick MD", GLUESTICK_WEIGHTS),
        ("Anime Segmentation ISNetDIS", ANIMESEG_CKPT),
        ("AniLines Detail", ANILINES_WEIGHTS)
    ]
    for name, path in weights_check:
        if os.path.exists(path):
            print(f"   - [✓] {name:30s}: SẴN SÀNG ({os.path.getsize(path)/(1024**2):.1f} MB)")
        else:
            raise FileNotFoundError(f"Không tìm thấy file trọng số {name} tại: {path}")

    # 3. Stage 4: GlueStick Matching & RANSAC Screening
    if os.path.exists(SCREENED_MANIFEST) and '--force' not in sys.argv:
        print(f"\n[✓] Đã tìm thấy file kiểm duyệt có sẵn: '{SCREENED_MANIFEST}'. Đang tải lại...")
        with open(SCREENED_MANIFEST, "r", encoding="utf-8") as f:
            screened_triplets = json.load(f)
        print(f"    - Số lượng Triplets đạt chuẩn: {len(screened_triplets):,} mẫu.")
    else:
        screened_triplets = run_stage4_gluestick(candidates, device=DEVICE)

    # 4. Stage 5: Line-Art Extraction (ISNetDIS + AniLines + MS-SSAA)
    run_stage5_lineart(screened_triplets, device=DEVICE)

    # 5. Stage 5.5: Kiểm duyệt & Làm sạch Dataset Line-Art
    if os.path.exists(CLEAN_MANIFEST) and '--force' not in sys.argv:
        print(f"\n[✓] Đã tìm thấy file kiểm duyệt Lineart có sẵn: '{CLEAN_MANIFEST}'. Đang tải lại...")
        with open(CLEAN_MANIFEST, "r", encoding="utf-8") as f:
            clean_triplets = json.load(f)
        print(f"    - Số lượng Triplets sạch đạt chuẩn: {len(clean_triplets):,} mẫu.")
    else:
        clean_triplets = run_stage5_5_clean_lineart(screened_triplets)

    # 6. Stage 6: Export Dataset chuẩn TPS-Inbetween
    run_stage6_export(clean_triplets, val_ratio=VAL_RATIO)


if __name__ == '__main__':
    main()
