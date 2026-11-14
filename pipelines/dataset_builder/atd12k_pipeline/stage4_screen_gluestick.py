#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Giai đoạn 4: Bộ lọc dung sai chuyển động bằng GlueStick + RANSAC trên GPU CUDA.
- Đo độ dời (displacement) thực tế giữa I_0 và I_1.
- Loại bỏ các triplet có chuyển động quá lớn (> 15.0px) hoặc bị mismatch.
- Lưu trữ các cặp điểm khớp inliers sạch dưới dạng file .npy chuẩn hóa [0, 1].
"""

import os
import sys
import json
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from config import (
    CANDIDATE_TRIPLETS_FILE, SCREENED_TRIPLETS_FILE,
    MATCHES_CACHE_DIR, GLUESTICK_WEIGHTS, DEVICE,
    SCREEN_MIN_INLIERS, SCREEN_MAX_MEAN_DISP,
    SCREEN_MAX_OUTLIER_DISP, SCREEN_RANSAC_THRESH,
    MODELS_DIR
)

# Thêm đường dẫn tps-inbetween vào sys.path để import gluestick
TPS_ROOT = os.path.abspath(os.path.join(MODELS_DIR, 'tps-inbetween'))
if TPS_ROOT not in sys.path:
    sys.path.insert(0, TPS_ROOT)

from model.gluestick import batch_to_np
from model.gluestick.models.two_view_pipeline import TwoViewPipeline

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def norm_flow(flow, h, w):
    """Chuẩn hóa tọa độ điểm ảnh về dải [0, 1]."""
    flow_norm = flow.copy()
    flow_norm[:, 0] = flow_norm[:, 0] / float(w)
    flow_norm[:, 1] = flow_norm[:, 1] / float(h)
    return flow_norm


def load_gluestick_model(weights_path=GLUESTICK_WEIGHTS, device=DEVICE):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Không tìm thấy file trọng số GlueStick tại: {weights_path}")

    conf = {
        'name': 'two_view_pipeline',
        'use_lines': True,
        'extractor': {
            'name': 'wireframe',
            'sp_params': {
                'force_num_keypoints': False,
                'max_num_keypoints': 1000,
            },
            'wireframe_params': {
                'merge_points': True,
                'merge_line_endpoints': True,
            },
            'max_n_lines': 300,
        },
        'matcher': {
            'name': 'gluestick',
            'weights': weights_path,
            'trainable': False,
        },
        'ground_truth': {
            'from_pose_depth': False,
        }
    }

    print(f"[*] Khởi tạo GlueStick TwoViewPipeline trên thiết bị: {device}")
    model = TwoViewPipeline(conf).to(device).eval()
    return model


def screen_triplets(candidates_file=CANDIDATE_TRIPLETS_FILE,
                    output_file=SCREENED_TRIPLETS_FILE,
                    matches_dir=MATCHES_CACHE_DIR):
    print("=" * 70)
    print("🚀 [GIAI ĐOẠN 4]: Bộ lọc dung sai chuyển động GlueStick + RANSAC trên CUDA")
    print(f"   - Số inliers tối thiểu: {SCREEN_MIN_INLIERS}")
    print(f"   - Độ dời trung bình tối đa: {SCREEN_MAX_MEAN_DISP} px")
    print(f"   - Ngưỡng RANSAC Reprojection: {SCREEN_RANSAC_THRESH} px")
    print("=" * 70)

    if not os.path.exists(candidates_file):
        raise FileNotFoundError(f"Không tìm thấy file: {candidates_file}")

    with open(candidates_file, 'r', encoding='utf-8') as f:
        candidates = json.load(f)

    os.makedirs(matches_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    matching_model = load_gluestick_model(device=DEVICE)

    screened_triplets = []
    discarded_stats = {
        "too_few_inliers": 0,
        "mean_disp_too_high": 0,
        "extreme_outlier": 0
    }

    print(f"[*] Bắt đầu kiểm duyệt và tính toán matching cho {len(candidates)} triplets ứng viên...")

    for idx, tr in enumerate(tqdm(candidates, desc="GlueStick Screening (CUDA)")):
        save_match_path = os.path.join(matches_dir, f"{tr['triplet_id']}.npy")
        rejected_marker = os.path.join(matches_dir, f"{tr['triplet_id']}.rejected")

        p0 = tr["i0_path"]
        p1 = tr["i1_path"]

        img0_cv = cv2.imread(p0, cv2.IMREAD_GRAYSCALE)
        img1_cv = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)

        if img0_cv is None or img1_cv is None:
            continue

        h, w = img0_cv.shape[:2]

        # Kiểm tra nếu đã từ chối ở lượt trước
        if os.path.exists(rejected_marker):
            continue

        # Kiểm tra nếu đã xử lý thành công ở lượt trước
        if os.path.exists(save_match_path):
            kps_stack = np.load(save_match_path)
            num_inliers = kps_stack.shape[1]
            diff_px = (kps_stack[0] - kps_stack[1]) * np.array([w, h])
            disp_px = np.linalg.norm(diff_px, axis=1)
            mean_disp = float(np.mean(disp_px))
            max_disp = float(np.max(disp_px))
            accepted_triplet = dict(tr)
            accepted_triplet["inliers"] = num_inliers
            accepted_triplet["mean_disp"] = round(mean_disp, 3)
            accepted_triplet["max_disp"] = round(max_disp, 3)
            accepted_triplet["match_npy_path"] = save_match_path
            screened_triplets.append(accepted_triplet)
            continue

        # Giải phóng bộ nhớ GPU định kỳ
        if idx % 15 == 0:
            torch.cuda.empty_cache()

        try:
            t0 = torch.from_numpy(img0_cv).float().unsqueeze(0).unsqueeze(0).to(DEVICE) / 255.0
            t1 = torch.from_numpy(img1_cv).float().unsqueeze(0).unsqueeze(0).to(DEVICE) / 255.0

            with torch.no_grad():
                pred = matching_model({'image0': t0, 'image1': t1})
                pred = batch_to_np(pred)

            del t0, t1

            kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
            m0 = pred["matches0"]

            valid_matches = m0 != -1
            match_indices = m0[valid_matches]
            matched_kps0 = kp0[valid_matches]
            matched_kps1 = kp1[match_indices]

            raw_count = len(matched_kps0)

            # Lọc RANSAC Affine Partial 2D để loại bỏ triệt để điểm ngoại lai
            if raw_count >= 4:
                _, inliers = cv2.estimateAffinePartial2D(
                    matched_kps0, matched_kps1,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=SCREEN_RANSAC_THRESH
                )
                if inliers is not None:
                    inliers = inliers.ravel().astype(bool)
                    matched_kps0 = matched_kps0[inliers]
                    matched_kps1 = matched_kps1[inliers]

            num_inliers = len(matched_kps0)

            # 1. Kiểm tra số lượng inliers
            if num_inliers < SCREEN_MIN_INLIERS:
                discarded_stats["too_few_inliers"] += 1
                open(rejected_marker, 'w').close()
                continue

            # 2. Tính độ dời pixel
            displacements = np.linalg.norm(matched_kps0 - matched_kps1, axis=1)
            mean_disp = float(np.mean(displacements))
            max_disp = float(np.max(displacements))
            outliers_over_50 = int(np.sum(displacements > SCREEN_MAX_OUTLIER_DISP))

            # 3. Kiểm tra độ dời trung bình an toàn cho TPS
            if mean_disp > SCREEN_MAX_MEAN_DISP:
                discarded_stats["mean_disp_too_high"] += 1
                open(rejected_marker, 'w').close()
                continue

            # 4. Kiểm tra điểm dị biệt cực đoan
            if outliers_over_50 > 1:
                discarded_stats["extreme_outlier"] += 1
                open(rejected_marker, 'w').close()
                continue

            # ĐẠT TIÊU CHUẨN: Chuẩn hóa tọa độ và lưu file .npy
            n_kps0 = norm_flow(matched_kps0, h, w)
            n_kps1 = norm_flow(matched_kps1, h, w)
            kps_stack = np.stack((n_kps0, n_kps1), axis=0).astype(np.float32)

            np.save(save_match_path, kps_stack)

            accepted_triplet = dict(tr)
            accepted_triplet["inliers"] = num_inliers
            accepted_triplet["mean_disp"] = round(mean_disp, 3)
            accepted_triplet["max_disp"] = round(max_disp, 3)
            accepted_triplet["match_npy_path"] = save_match_path

            screened_triplets.append(accepted_triplet)

        except Exception as e:
            # Ghi nhận và bỏ qua nếu gặp lỗi bộ nhớ cục bộ ở frame cụ thể
            discarded_stats["extreme_outlier"] += 1
            torch.cuda.empty_cache()
            continue

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(screened_triplets, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("📊 KẾT QUẢ GLUESTICK & MOTION SCREENING:")
    print(f"   - Tổng số triplet ứng viên đưa vào kiểm tra: {len(candidates)}")
    print(f"   - Số triplet ĐẠT CHUẨN chấp nhận: {len(screened_triplets)}")
    print(f"   - Số triplet bị loại:")
    print(f"     + Quá ít điểm khớp inliers (< {SCREEN_MIN_INLIERS}): {discarded_stats['too_few_inliers']}")
    print(f"     + Độ dời quá lớn (> {SCREEN_MAX_MEAN_DISP}px): {discarded_stats['mean_disp_too_high']}")
    print(f"     + Chứa điểm ngoại lai dị biệt (> 50px): {discarded_stats['extreme_outlier']}")

    if len(screened_triplets) > 0:
        mean_disps = [t["mean_disp"] for t in screened_triplets]
        inliers_counts = [t["inliers"] for t in screened_triplets]
        print(f"   - Độ dời trung bình của tập đạt chuẩn: {sum(mean_disps)/len(mean_disps):.2f} px (Lý tưởng cho TPS!)")
        print(f"   - Số inliers trung bình: {sum(inliers_counts)/len(inliers_counts):.1f} điểm")
    print(f"   - Đã lưu danh sách triplet đạt chuẩn vào: {output_file}")
    print("=" * 70 + "\n")

    return screened_triplets


if __name__ == '__main__':
    screen_triplets()
