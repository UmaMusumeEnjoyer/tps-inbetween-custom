#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_extract_and_mix.py
Script chạy trực tiếp trên Terminal để:
1. Trích xuất Candidate Triplets từ video Sakugabooru (sakugabooru_character_acting).
2. Lọc và chuẩn hóa Anime Nhật từ ATD-12K (E:\\Dataset\\datasets).
3. Hợp nhất, đánh số và đóng gói thành `candidate_triplets_anime_540p.zip`.
"""

import os
import sys
import glob
import json
import time
import shutil
import random
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Thiết lập Seed
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# ==============================================================================
# BẢNG THAM SỐ CẤU HÌNH ĐƯỜNG DẪN & CHỈ TIÊU
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))

SAKUGA_DIR = os.path.join(ROOT_DIR, "data", "sakugabooru_character_acting")
if not os.path.exists(SAKUGA_DIR):
    SAKUGA_DIR = os.path.join(ROOT_DIR, "sakugabooru_character_acting")
ATD12K_DIR = os.environ.get("ATD12K_DATASETS_DIR", r"E:\Dataset\datasets")

TARGET_WIDTH = 960            # Chuẩn 540p
TARGET_HEIGHT = 540
JPEG_QUALITY = 95

MAX_SAKUGA_TRIPLETS = 3500    # Mục tiêu Sakugabooru (đa mốc t)
MAX_TRIPLETS_PER_CLIP = 20    # Giới hạn tối đa số triplet mỗi clip
MAX_ATD12K_TRIPLETS = 4500    # Mục tiêu ATD-12K Anime Nhật (t = 0.5)

MIN_DELTA_F = 2               # On-twos (2 frames)
MAX_DELTA_F = 6               # <= 0.25 giây
PSNR_DEDUP_THRESH = 40.0
EDGE_DIFF_THRESH = 0.35
CUT_SCENE_PSNR = 14.0
CUT_SCENE_EDGE = 4.5
MAX_HOLD_FRAMES = 8


def format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# ==============================================================================
# CLASS LỌC DEDUP TRÊN CUDA & THUẬT TOÁN ĐA MỐC THỜI GIAN
# ==============================================================================
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
        is_cut = (psnr < CUT_SCENE_PSNR) or (delta_edge > CUT_SCENE_EDGE)
        return is_dup, is_cut, edge2


def mine_diverse_triplets(drawings):
    triplets = []
    n = len(drawings)
    if n < 3:
        return triplets

    for i in range(n - 2):
        f0, img0 = drawings[i]

        # 1. Step-2 (t = 0.5)
        f_gt = drawings[i + 1][0]
        img_gt = drawings[i + 1][1]
        f1 = drawings[i + 2][0]
        img1 = drawings[i + 2][1]
        delta_f = f1 - f0
        if MIN_DELTA_F <= delta_f <= MAX_DELTA_F:
            t = round((f_gt - f0) / float(delta_f), 4)
            triplets.append({
                'f0': f0, 'f_gt': f_gt, 'f1': f1,
                'delta_f': delta_f, 't': t,
                'img0': img0, 'img_gt': img_gt, 'img1': img1
            })

        # 2. Step-3 (t ~ 0.333 và t ~ 0.667)
        if i + 3 < n:
            f3 = drawings[i + 3][0]
            img3 = drawings[i + 3][1]
            delta_f3 = f3 - f0
            if MIN_DELTA_F <= delta_f3 <= MAX_DELTA_F:
                t_third_1 = round((drawings[i + 1][0] - f0) / float(delta_f3), 4)
                triplets.append({
                    'f0': f0, 'f_gt': drawings[i + 1][0], 'f1': f3,
                    'delta_f': delta_f3, 't': t_third_1,
                    'img0': img0, 'img_gt': drawings[i + 1][1], 'img1': img3
                })
                t_third_2 = round((drawings[i + 2][0] - f0) / float(delta_f3), 4)
                triplets.append({
                    'f0': f0, 'f_gt': drawings[i + 2][0], 'f1': f3,
                    'delta_f': delta_f3, 't': t_third_2,
                    'img0': img0, 'img_gt': drawings[i + 2][1], 'img1': img3
                })

        # 3. Step-4 (t ~ 0.25 và t ~ 0.75)
        if i + 4 < n:
            f4 = drawings[i + 4][0]
            img4 = drawings[i + 4][1]
            delta_f4 = f4 - f0
            if MIN_DELTA_F <= delta_f4 <= MAX_DELTA_F:
                t_quarter_1 = round((drawings[i + 1][0] - f0) / float(delta_f4), 4)
                triplets.append({
                    'f0': f0, 'f_gt': drawings[i + 1][0], 'f1': f4,
                    'delta_f': delta_f4, 't': t_quarter_1,
                    'img0': img0, 'img_gt': drawings[i + 1][1], 'img1': img4
                })
                t_quarter_3 = round((drawings[i + 3][0] - f0) / float(delta_f4), 4)
                triplets.append({
                    'f0': f0, 'f_gt': drawings[i + 3][0], 'f1': f4,
                    'delta_f': delta_f4, 't': t_quarter_3,
                    'img0': img0, 'img_gt': drawings[i + 3][1], 'img1': img4
                })

    return triplets


def extract_triplets_from_sakuga_clip(video_path, dedup_worker, max_triplets=MAX_TRIPLETS_PER_CLIP):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    unique_drawings = []
    prev_tensor = None
    prev_edge = None
    hold_count = 0
    all_triplets = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        h, w = frame.shape[:2]
        if w != TARGET_WIDTH or h != TARGET_HEIGHT:
            frame_540p = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
        else:
            frame_540p = frame

        gray = cv2.cvtColor(frame_540p, cv2.COLOR_BGR2GRAY)
        tensor = torch.from_numpy(gray).float().to(dedup_worker.device).unsqueeze(0).unsqueeze(0) / 255.0

        if prev_tensor is None:
            prev_tensor = tensor
            prev_edge = dedup_worker.compute_edge(tensor)
            unique_drawings.append((frame_idx, frame_540p))
            frame_idx += 1
            continue

        is_dup, is_cut, edge2 = dedup_worker.is_duplicate(prev_tensor, prev_edge, tensor)

        if is_cut:
            all_triplets.extend(mine_diverse_triplets(unique_drawings))
            unique_drawings = [(frame_idx, frame_540p)]
            prev_tensor = tensor
            prev_edge = edge2
            hold_count = 0
            if len(all_triplets) >= max_triplets * 2:
                break
            frame_idx += 1
            continue

        if is_dup:
            hold_count += 1
            if hold_count > MAX_HOLD_FRAMES:
                all_triplets.extend(mine_diverse_triplets(unique_drawings))
                unique_drawings = [(frame_idx, frame_540p)]
                hold_count = 0
        else:
            unique_drawings.append((frame_idx, frame_540p))
            prev_tensor = tensor
            prev_edge = edge2
            hold_count = 0

        frame_idx += 1

    cap.release()

    if len(unique_drawings) >= 3:
        all_triplets.extend(mine_diverse_triplets(unique_drawings))

    random.shuffle(all_triplets)
    return all_triplets[:max_triplets]


def process_single_atd_folder(item):
    folder_path, out_base_dir, out_index = item
    f1_src = os.path.join(folder_path, "frame1.jpg")
    f2_src = os.path.join(folder_path, "frame2.jpg")
    f3_src = os.path.join(folder_path, "frame3.jpg")

    if not (os.path.exists(f1_src) and os.path.exists(f2_src) and os.path.exists(f3_src)):
        return None

    img1 = cv2.imread(f1_src)
    img2 = cv2.imread(f2_src)
    img3 = cv2.imread(f3_src)
    if img1 is None or img2 is None or img3 is None:
        return None

    folder_id = f"atd12k_{out_index:06d}"
    target_dir = os.path.join(out_base_dir, folder_id)
    os.makedirs(target_dir, exist_ok=True)

    h, w = img1.shape[:2]
    if w != TARGET_WIDTH or h != TARGET_HEIGHT:
        img1 = cv2.resize(img1, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
        img3 = cv2.resize(img3, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

    cv2.imwrite(os.path.join(target_dir, "frame1.jpg"), img1, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    cv2.imwrite(os.path.join(target_dir, "frame2.jpg"), img2, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    cv2.imwrite(os.path.join(target_dir, "frame3.jpg"), img3, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    return {
        'source_type': 'atd12k_japan',
        'local_id': folder_id,
        'source_clip': os.path.basename(folder_path),
        't': 0.5,
        'delta_frames': 2,
        'folder_path': target_dir
    }


# ==============================================================================
# HÀM MAIN THỰC THI TOÀN BỘ PIPELINE
# ==============================================================================
def main():
    print("=" * 75)
    print("🚀 PIPELINE EXTRACT & TRỘN FRAME ANIME CHƯA QUA CHỈNH SỬA (LOCAL PC)")
    print(f"   - Thiết bị tính toán: {DEVICE}")
    if torch.cuda.is_available():
        print(f"   - GPU: {torch.cuda.get_device_name(0)}")
    print(f"   - Thư mục Sakugabooru: {SAKUGA_DIR}")
    print(f"   - Thư mục ATD-12K:     {ATD12K_DIR}")
    print("=" * 75)

    # 1. Quét video Sakugabooru
    print("\n[*] Đang quét danh sách video Sakugabooru...")
    sakuga_video_files = []
    if os.path.exists(SAKUGA_DIR):
        for root, dirs, files in os.walk(SAKUGA_DIR):
            for f in files:
                if f.lower().endswith(('.mp4', '.mkv', '.webm')):
                    sakuga_video_files.append(os.path.join(root, f))
    sakuga_video_files.sort()
    print(f"[✓] Tìm thấy {len(sakuga_video_files):,} video Sakugabooru.")

    if not sakuga_video_files:
        print(f"[!] Không tìm thấy video nào tại: {SAKUGA_DIR}")
        return

    # 2. Quét ATD-12K Japan & Hayao
    print("\n[*] Đang quét thư mục ATD-12K Anime Nhật (Japan_* và Hayao_*)...")
    atd12k_japan_folders = []
    atd12k_disney_count = 0

    atd_search_dirs = [ATD12K_DIR, os.path.join(ATD12K_DIR, "train_10k")]
    for s_dir in atd_search_dirs:
        if os.path.exists(s_dir):
            for root, dirs, files in os.walk(s_dir):
                for d in dirs:
                    if d.startswith("Japan_") or d.startswith("Hayao_"):
                        atd12k_japan_folders.append(os.path.join(root, d))
                    elif d.startswith("Disney_"):
                        atd12k_disney_count += 1
            if atd12k_japan_folders:
                break

    atd12k_japan_folders = sorted(list(set(atd12k_japan_folders)))
    print(f"[✓] Tìm thấy {len(atd12k_japan_folders):,} thư mục ATD-12K Anime Nhật đạt chuẩn (Japan + Hayao).")
    print(f"    (Đã lọc bỏ {atd12k_disney_count:,} thư mục Disney).")

    if not atd12k_japan_folders:
        print(f"[!] Không tìm thấy thư mục ATD-12K Anime nào tại: {ATD12K_DIR}")
        return

    # Thư mục làm việc tạm
    WORK_DIR = os.path.join(BASE_DIR, "work_candidates")
    SAKUGA_OUT_DIR = os.path.join(WORK_DIR, "sakuga")
    ATD_OUT_DIR = os.path.join(WORK_DIR, "atd12k")
    MERGED_DIR = os.path.join(BASE_DIR, "candidate_triplets_anime_540p")
    MERGED_TRIPLETS_DIR = os.path.join(MERGED_DIR, "triplets")
    ZIP_OUTPUT_PATH = os.path.join(BASE_DIR, "candidate_triplets_anime_540p.zip")

    os.makedirs(SAKUGA_OUT_DIR, exist_ok=True)
    os.makedirs(ATD_OUT_DIR, exist_ok=True)
    os.makedirs(MERGED_TRIPLETS_DIR, exist_ok=True)

    # 3. Trích xuất Sakugabooru
    print("\n" + "=" * 75)
    print(f"🎬 [BƯỚC 1/4] KHAI THÁC VIDEO SAKUGABOORU (Mục tiêu: {MAX_SAKUGA_TRIPLETS} triplets)")
    print("=" * 75)

    dedup_worker = CudaDedupWorker(device=DEVICE)
    sakuga_candidates_metadata = []
    total_sakuga_extracted = 0
    start_sakuga_time = time.time()

    with tqdm(total=len(sakuga_video_files), desc="Trích xuất Sakuga",
              bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} clips [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        for v_path in sakuga_video_files:
            clip_name = os.path.splitext(os.path.basename(v_path))[0]
            clip_triplets = extract_triplets_from_sakuga_clip(v_path, dedup_worker, max_triplets=MAX_TRIPLETS_PER_CLIP)

            for tr in clip_triplets:
                total_sakuga_extracted += 1
                folder_id = f"sakuga_{total_sakuga_extracted:06d}"
                target_dir = os.path.join(SAKUGA_OUT_DIR, folder_id)
                os.makedirs(target_dir, exist_ok=True)

                cv2.imwrite(os.path.join(target_dir, "frame1.jpg"), tr['img0'], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                cv2.imwrite(os.path.join(target_dir, "frame2.jpg"), tr['img_gt'], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                cv2.imwrite(os.path.join(target_dir, "frame3.jpg"), tr['img1'], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

                sakuga_candidates_metadata.append({
                    'source_type': 'sakugabooru',
                    'local_id': folder_id,
                    'source_clip': clip_name,
                    't': tr['t'],
                    'delta_frames': tr['delta_f'],
                    'folder_path': target_dir
                })

                if total_sakuga_extracted >= MAX_SAKUGA_TRIPLETS:
                    break

            elapsed_sec = time.time() - start_sakuga_time
            trip_rate = (total_sakuga_extracted / elapsed_sec) if elapsed_sec > 0 else 0
            rem_triplets = max(0, MAX_SAKUGA_TRIPLETS - total_sakuga_extracted)
            eta_sec = (rem_triplets / trip_rate) if trip_rate > 0 else 0

            pbar.set_postfix({
                'Triplets': f"{total_sakuga_extracted}/{MAX_SAKUGA_TRIPLETS}",
                'Tốc độ': f"{trip_rate:.1f} tr/s",
                'ETA': format_time(eta_sec)
            })
            pbar.update(1)

            if total_sakuga_extracted >= MAX_SAKUGA_TRIPLETS:
                print(f"\n[*] Đạt mục tiêu {MAX_SAKUGA_TRIPLETS} triplets. Dừng trích xuất Sakugabooru.")
                break

    print(f"[✓] Đã khai thác: {len(sakuga_candidates_metadata):,} triplets Sakugabooru.")

    # 4. Chuẩn hóa ATD-12K
    print("\n" + "=" * 75)
    selected_atd = atd12k_japan_folders[:MAX_ATD12K_TRIPLETS]
    print(f"🎌 [BƯỚC 2/4] CHUẨN HÓA ATD-12K ANIME NHẬT (Mục tiêu: {len(selected_atd)} triplets)")
    print("=" * 75)

    tasks = [(fpath, ATD_OUT_DIR, idx + 1) for idx, fpath in enumerate(selected_atd)]
    atd12k_candidates_metadata = []

    start_atd_time = time.time()
    with ThreadPoolExecutor(max_workers=8) as executor:
        pbar_atd = tqdm(executor.map(process_single_atd_folder, tasks), total=len(tasks),
                        desc="Chuẩn hóa ATD-12K",
                        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} triplets [{elapsed}<{remaining}, {rate_fmt}]')
        for res in pbar_atd:
            if res is not None:
                atd12k_candidates_metadata.append(res)
                pbar_atd.set_postfix({'Hợp lệ': len(atd12k_candidates_metadata)})

    print(f"[✓] Đã chuẩn hóa: {len(atd12k_candidates_metadata):,} triplets ATD-12K.")

    # 5. Hợp nhất & Đánh số
    print("\n" + "=" * 75)
    print(f"🔀 [BƯỚC 3/4] HỢP NHẤT VÀ ĐÁNH SỐ CHUẨN HÓA")
    print("=" * 75)

    all_pool = []
    all_pool.extend(sakuga_candidates_metadata)
    all_pool.extend(atd12k_candidates_metadata)

    random.seed(RANDOM_SEED)
    random.shuffle(all_pool)

    final_manifest = []
    global_id = 0

    with tqdm(total=len(all_pool), desc="Hợp nhất Triplets",
              bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} triplets [{elapsed}<{remaining}, {rate_fmt}]') as pbar_merge:
        for item in all_pool:
            global_id += 1
            new_triplet_id = f"triplet_{global_id:06d}"
            dst_folder = os.path.join(MERGED_TRIPLETS_DIR, new_triplet_id)
            os.makedirs(dst_folder, exist_ok=True)

            src_folder = item['folder_path']
            shutil.copy2(os.path.join(src_folder, "frame1.jpg"), os.path.join(dst_folder, "frame1.jpg"))
            shutil.copy2(os.path.join(src_folder, "frame2.jpg"), os.path.join(dst_folder, "frame2.jpg"))
            shutil.copy2(os.path.join(src_folder, "frame3.jpg"), os.path.join(dst_folder, "frame3.jpg"))

            final_manifest.append({
                'triplet_id': new_triplet_id,
                'source': item['source_clip'],
                'source_type': item['source_type'],
                't': item['t'],
                'delta_frames': item['delta_frames'],
                'frame1': f"triplets/{new_triplet_id}/frame1.jpg",
                'frame2': f"triplets/{new_triplet_id}/frame2.jpg",
                'frame3': f"triplets/{new_triplet_id}/frame3.jpg"
            })
            pbar_merge.update(1)

    manifest_path = os.path.join(MERGED_DIR, "dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(final_manifest, f, indent=2, ensure_ascii=False)

    print(f"[✓] Đã tạo dataset_manifest.json với {len(final_manifest):,} triplets.")

    # 6. Dọn dẹp và Nén ZIP
    print("\n" + "=" * 75)
    print(f"📦 [BƯỚC 4/4] NÉN TOÀN BỘ DATASET THÀNH FILE ZIP")
    print("=" * 75)

    print("[*] Đang dọn dẹp thư mục tạm work_candidates...")
    shutil.rmtree(WORK_DIR, ignore_errors=True)

    print("[*] Đang quét danh sách file cần nén...")
    all_files_to_zip = []
    for root, dirs, files in os.walk(MERGED_TRIPLETS_DIR):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, MERGED_DIR)
            all_files_to_zip.append((full_path, rel_path))

    total_files_to_compress = len(all_files_to_zip) + 1
    print(f"[*] Tổng số file cần nén: {total_files_to_compress:,} files...")
    print(f"[*] File đích: {ZIP_OUTPUT_PATH}")

    start_zip_time = time.time()
    with zipfile.ZipFile(ZIP_OUTPUT_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(manifest_path, arcname="dataset_manifest.json")
        with tqdm(total=len(all_files_to_zip), desc="Nén ZIP Dataset",
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} files [{elapsed}<{remaining}, {rate_fmt}]') as pbar_zip:
            for full_path, rel_path in all_files_to_zip:
                zipf.write(full_path, arcname=rel_path)
                pbar_zip.update(1)

    zip_duration = time.time() - start_zip_time
    zip_size_gb = os.path.getsize(ZIP_OUTPUT_PATH) / (1024 ** 3)

    print("\n" + "=" * 75)
    print(f"🎉 HOÀN TẤT ĐÓNG GÓI DATASET!")
    print(f"   - File nén:       {ZIP_OUTPUT_PATH}")
    print(f"   - Dung lượng nén: {zip_size_gb:.2f} GB")
    print(f"   - Thời gian nén:  {format_time(zip_duration)}")
    print(f"   - Tổng số mẫu:    {len(final_manifest):,} candidate triplets")
    print("=" * 75)

    # 7. Thống kê t và nguồn
    print("\n📈 BẢNG THỐNG KÊ TỔNG QUAN:")
    source_counts = Counter(x['source_type'] for x in final_manifest)
    print("   Phân bố theo nguồn dữ liệu:")
    for src, cnt in source_counts.items():
        print(f"      * {src:20s}: {cnt:,} triplets ({cnt/len(final_manifest)*100:.1f}%)")

    t_counts = Counter(round(x['t'], 3) for x in final_manifest)
    print("   Phân bố chi tiết theo mốc thời gian t:")
    for t_val in sorted(t_counts.keys()):
        print(f"      * t = {t_val:.3f}: {t_counts[t_val]:,} triplets ({t_counts[t_val]/len(final_manifest)*100:.1f}%)")

    print("\n" + "=" * 75)
    print(f"👉 File '{ZIP_OUTPUT_PATH}' đã sẵn sàng để upload lên Kaggle Dataset!")
    print("=" * 75)


if __name__ == '__main__':
    main()
