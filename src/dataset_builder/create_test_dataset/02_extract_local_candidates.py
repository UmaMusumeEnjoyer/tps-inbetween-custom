#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_extract_local_candidates.py
Khai thác Candidate Triplets từ các tập video trong AnimeGoc trên máy tính cá nhân (Local PC).

Quy trình:
1. Đọc các phân đoạn sạch (đã lọc bỏ OP/ED) từ `clean_segments_manifest.json`.
2. Đọc luồng video theo từng phân đoạn bằng OpenCV.
3. Bỏ qua cảnh đứng yên (Freeze / Talking head) và phát hiện cắt cảnh (Cut Scene).
4. Áp dụng Micro-Deduplication (Sobel Edge Difference + PSNR) trên CUDA để gom Unique Drawings.
5. Khai thác ứng viên Triplet kề cận đa mốc thời gian: t = 0.25, 0.33, 0.5, 0.67, 0.75.
6. Resize về chuẩn 540p (960x540) và lưu vào thư mục đệm.
"""

import os
import sys
import json
import glob
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Tham số cấu hình
TARGET_WIDTH = 960
TARGET_HEIGHT = 540
MIN_DELTA_F = 2      # Bước nhảy tối thiểu: 2 frame video (On-twos)
MAX_DELTA_F = 6      # Bước nhảy tối đa: 6 frame video (<= 0.25s)
PSNR_DEDUP_THRESH = 40.0
EDGE_DIFF_THRESH = 0.35
MAX_HOLD_FRAMES = 8
MAX_TRIPLETS_PER_EPISODE = 80   # Giới hạn số triplet mỗi tập để đảm bảo đa dạng


class CudaDedupWorker:
    def __init__(self, device=DEVICE):
        self.device = torch.device(device)
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]], dtype=torch.float32, device=self.device).view(1, 1, 3, 3) / 4.0
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                                [ 0.0,  0.0,  0.0],
                                [ 1.0,  2.0,  1.0]], dtype=torch.float32, device=self.device).view(1, 1, 3, 3) / 4.0
        self.sobel_x = sobel_x
        self.sobel_y = sobel_y

    def compute_edge(self, gray_tensor):
        pad = F.pad(gray_tensor, (1, 1, 1, 1), mode='replicate')
        gx = F.conv2d(pad, self.sobel_x)
        gy = F.conv2d(pad, self.sobel_y)
        return torch.sqrt(gx * gx + gy * gy + 1e-8)

    def is_duplicate(self, t1, edge1, t2):
        mse = torch.mean((t1 - t2) ** 2)
        psnr = 100.0 if mse.item() < 1e-10 else (10.0 * torch.log10(1.0 / mse)).item()
        edge2 = self.compute_edge(t2)
        delta_edge = (torch.mean(torch.abs(edge1 - edge2)) * 255.0).item()
        
        is_dup = (psnr >= PSNR_DEDUP_THRESH) and (delta_edge < EDGE_DIFF_THRESH)
        is_cut = (psnr < 14.0) or (delta_edge > 4.5)  # Chuyển cảnh đột ngột
        return is_dup, is_cut, edge2


def process_clean_segment(cap, start_sec, end_sec, fps, dedup_worker, max_triplets=40):
    """Xử lý 1 phân đoạn video sạch và trích xuất các candidate triplets."""
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    total_segment_frames = end_frame - start_frame
    if total_segment_frames < 10:
        return []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    unique_drawings = []  # Danh sách (frame_idx_in_video, img_540p_bgr)
    
    prev_tensor = None
    prev_edge = None
    hold_count = 0
    triplets = []

    for f_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Resize về 540p ngay từ đầu để tiết kiệm RAM & GPU
        frame_540p = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
        
        # Grayscale tensor trên GPU
        gray = cv2.cvtColor(frame_540p, cv2.COLOR_BGR2GRAY)
        tensor = torch.from_numpy(gray).float().to(dedup_worker.device).unsqueeze(0).unsqueeze(0) / 255.0

        if prev_tensor is None:
            prev_tensor = tensor
            prev_edge = dedup_worker.compute_edge(tensor)
            unique_drawings.append((f_idx, frame_540p))
            continue

        is_dup, is_cut, edge2 = dedup_worker.is_duplicate(prev_tensor, prev_edge, tensor)

        if is_cut:
            # Phát hiện cắt cảnh: khai thác triplet từ các drawing trước đó rồi reset
            triplets.extend(mine_triplets_from_drawings(unique_drawings))
            unique_drawings = [(f_idx, frame_540p)]
            prev_tensor = tensor
            prev_edge = edge2
            hold_count = 0
            if len(triplets) >= max_triplets:
                break
            continue

        if is_dup:
            hold_count += 1
            if hold_count > MAX_HOLD_FRAMES:
                # Nếu đứng hình quá dài, reset để tránh chuyển động trễ
                triplets.extend(mine_triplets_from_drawings(unique_drawings))
                unique_drawings = [(f_idx, frame_540p)]
                hold_count = 0
            continue
        else:
            # Là nét vẽ mới (drawing mới)
            unique_drawings.append((f_idx, frame_540p))
            prev_tensor = tensor
            prev_edge = edge2
            hold_count = 0

    if len(unique_drawings) >= 3:
        triplets.extend(mine_triplets_from_drawings(unique_drawings))

    return triplets[:max_triplets]


def mine_triplets_from_drawings(drawings):
    """Khai thác Candidate Triplets từ chuỗi drawings với step = 2."""
    triplets = []
    n = len(drawings)
    for k in range(n - 2):
        f0, img0 = drawings[k]
        f_gt, img_gt = drawings[k + 1]
        f1, img1 = drawings[k + 2]

        delta_f = f1 - f0
        if MIN_DELTA_F <= delta_f <= MAX_DELTA_F:
            t = round((f_gt - f0) / float(delta_f), 4)
            triplets.append({
                'f0': f0,
                'f_gt': f_gt,
                'f1': f1,
                'delta_f': delta_f,
                't': t,
                'img0': img0,
                'img_gt': img_gt,
                'img1': img1
            })
    return triplets


def extract_all_animegoc_candidates(manifest_path, output_dir):
    """Chạy trích xuất cho toàn bộ video trong AnimeGoc."""
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Không tìm thấy file manifest: {manifest_path}. Hãy chạy 01_op_ed_detector.py trước!")

    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    meta_file = os.path.join(output_dir, "animegoc_candidates_manifest.json")
    if os.path.exists(meta_file) and '--force' not in sys.argv:
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            if len(existing_data) > 500:
                print(f"[✓] Đã tìm thấy {len(existing_data)} triplets sẵn có trong '{output_dir}'. Bỏ qua trích xuất lại.")
                print("    (Nếu muốn chạy lại từ đầu, hãy xóa thư mục hoặc thêm cờ --force)")
                return len(existing_data)
        except Exception:
            pass

    dedup_worker = CudaDedupWorker(device=DEVICE)

    total_extracted = 0
    candidate_metadata = []

    print("=" * 70)
    print(f"🚀 [02_EXTRACT_LOCAL_CANDIDATES]: Khai thác Triplets từ {len(manifest)} video AnimeGoc")
    print(f"   - Thiết bị tăng tốc: {DEVICE}")
    print(f"   - Chuẩn kích thước: {TARGET_WIDTH}x{TARGET_HEIGHT} (540p)")
    print(f"   - Bước nhảy Delta F cho phép: {MIN_DELTA_F} đến {MAX_DELTA_F} frames")
    print("=" * 70)

    for vpath, vinfo in tqdm(manifest.items(), desc="Xử lý video Anime"):
        if not os.path.exists(vpath):
            continue

        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        clean_ranges = vinfo.get('clean_ranges', [])

        ep_triplets = []
        for s, e in clean_ranges:
            seg_triplets = process_clean_segment(cap, s, e, fps, dedup_worker, max_triplets=MAX_TRIPLETS_PER_EPISODE)
            ep_triplets.extend(seg_triplets)
            if len(ep_triplets) >= MAX_TRIPLETS_PER_EPISODE:
                break

        cap.release()

        # Lưu ảnh và ghi nhận triplet
        ep_name = os.path.splitext(vinfo['filename'])[0]
        for tr in ep_triplets[:MAX_TRIPLETS_PER_EPISODE]:
            total_extracted += 1
            folder_name = f"animegoc_{total_extracted:06d}"
            save_path = os.path.join(output_dir, folder_name)
            os.makedirs(save_path, exist_ok=True)

            cv2.imwrite(os.path.join(save_path, "frame1.jpg"), tr['img0'], [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(os.path.join(save_path, "frame2.jpg"), tr['img_gt'], [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(os.path.join(save_path, "frame3.jpg"), tr['img1'], [cv2.IMWRITE_JPEG_QUALITY, 95])

            candidate_metadata.append({
                'triplet_id': folder_name,
                'source': f"AnimeGoc_{ep_name}",
                't': tr['t'],
                'delta_frames': tr['delta_f']
            })

    meta_file = os.path.join(output_dir, "animegoc_candidates_manifest.json")
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(candidate_metadata, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"[✓] Đã khai thác thành công {total_extracted} Candidate Triplets từ AnimeGoc!")
    print(f"    - Metadata lưu tại: {meta_file}")
    print("=" * 70 + "\n")
    return total_extracted


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    m_path = os.path.join(base_dir, "clean_segments_manifest.json")
    out_cands = os.path.join(base_dir, "work", "candidates_animegoc")
    extract_all_animegoc_candidates(m_path, out_cands)
