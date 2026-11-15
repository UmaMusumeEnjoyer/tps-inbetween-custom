#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Giai đoạn 3: Khai phá Triplet kề cận (Step = 2) & Tính nhãn thời gian t chuẩn xác.
- Khai thác bộ ba: I_0 = D_k, I_GT = D_{k+1}, I_1 = D_{k+2}
- Đảm bảo bước nhảy video nằm trong [MIN_VIDEO_GAP, MAX_VIDEO_GAP] (tương đương 2 - 6 frames ở 24fps)
- Tính nhãn thời gian t vật lý chính xác theo dòng thời gian animation.
"""

import os
import sys
import json
from config import (
    DRAWINGS_DIR, CANDIDATE_TRIPLETS_FILE,
    MIN_VIDEO_GAP, MAX_VIDEO_GAP, RAW_FRAMES_DIR
)

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def mine_triplets(drawings_json=os.path.join(DRAWINGS_DIR, "timeline_drawings.json"),
                  output_file=CANDIDATE_TRIPLETS_FILE):
    print("=" * 70)
    print("🚀 [GIAI ĐOẠN 3]: Khai phá ứng viên Triplet kề cận & Tính toán nhãn t chuẩn xác")
    print(f"   - Khoảng cách frame video cho phép: {MIN_VIDEO_GAP} <= Delta F <= {MAX_VIDEO_GAP}")
    print("=" * 70)

    if not os.path.exists(drawings_json):
        raise FileNotFoundError(f"Không tìm thấy file: {drawings_json}")

    with open(drawings_json, 'r', encoding='utf-8') as f:
        drawings = json.load(f)

    num_drawings = len(drawings)
    print(f"[*] Đọc thành công {num_drawings} bản vẽ độc lập.")

    candidate_triplets = []

    # Khai phá bộ ba với Step = 2 trong chuỗi bản vẽ độc lập
    for k in range(num_drawings - 2):
        d0 = drawings[k]
        d_gt = drawings[k + 1]
        d1 = drawings[k + 2]

        f0 = d0["frame_idx"]
        f_gt = d_gt["frame_idx"]
        f1 = d1["frame_idx"]

        delta_f = f1 - f0

        # Kiểm tra bước nhảy thời gian thực tế có nằm trong vùng thiết kế của TPS không
        if MIN_VIDEO_GAP <= delta_f <= MAX_VIDEO_GAP:
            # Tính nhãn thời gian t chuẩn xác theo vị trí vật lý
            t_val = (f_gt - f0) / float(delta_f)
            t_val = round(t_val, 4)

            triplet = {
                "triplet_id": len(candidate_triplets),
                "d0_id": d0["drawing_id"],
                "d_gt_id": d_gt["drawing_id"],
                "d1_id": d1["drawing_id"],
                "i0_file": d0["frame_file"],
                "igt_file": d_gt["frame_file"],
                "i1_file": d1["frame_file"],
                "i0_path": os.path.join(RAW_FRAMES_DIR, d0["frame_file"]),
                "igt_path": os.path.join(RAW_FRAMES_DIR, d_gt["frame_file"]),
                "i1_path": os.path.join(RAW_FRAMES_DIR, d1["frame_file"]),
                "f0": f0,
                "f_gt": f_gt,
                "f1": f1,
                "delta_frames": delta_f,
                "t": t_val
            }
            candidate_triplets.append(triplet)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(candidate_triplets, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("📊 KẾT QUẢ KHAI PHÁ TRIPLET:")
    print(f"   - Tổng số triplet ứng viên trích xuất được: {len(candidate_triplets)}")
    
    if len(candidate_triplets) > 0:
        t_values = [tr["t"] for tr in candidate_triplets]
        delta_f_values = [tr["delta_frames"] for tr in candidate_triplets]
        print(f"   - Phân bố bước nhảy Delta F: Min={min(delta_f_values)}, Max={max(delta_f_values)}, Mean={sum(delta_f_values)/len(delta_f_values):.2f} frames")
        print(f"   - Phân bố nhãn t: Min={min(t_values)}, Max={max(t_values)}, Mean={sum(t_values)/len(t_values):.4f}")
    print(f"   - Đã lưu danh sách ứng viên vào: {output_file}")
    print("=" * 70 + "\n")

    return candidate_triplets


if __name__ == '__main__':
    mine_triplets()
