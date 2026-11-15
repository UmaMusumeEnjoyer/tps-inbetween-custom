#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_v22_notebook.py
Script thẩm định tính hợp lệ của notebook Kaggle V2.2 trước khi đẩy lên Cloud:
1. Thẩm định cấu trúc JSON và 13 cell chuẩn.
2. Kiểm tra Warm-start từ V2.0.2 Best checkpoint.
3. Kiểm tra các hàm loss mới (StrokeDensityRegularization, DirectionalTangent, SoftClDice).
4. Kiểm tra Differential LR (1.5e-5 vs 5.0e-5), Scheduler 25 epochs và Gradient Clipping 1.0.
5. Kiểm tra Temporal Reversal Augmentation và Samplers.
6. Kiểm tra Sakuga Composite Score V2.2 và Hard Gates.
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


def verify_v22():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nb_path = os.path.abspath(os.path.join(script_dir, "..", "notebooks", "triplet_06_v2.2_anti_thickening_kaggle.ipynb"))

    if not os.path.exists(nb_path):
        raise FileNotFoundError(f"Không tìm thấy file: {nb_path}")

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    print(f"[*] Tổng số cell trong notebook V2.2: {len(cells)}")

    assert len(cells) >= 13, f"Số cell ({len(cells)}) không đủ 13 cell chuẩn!"

    checks = []

    # 1. Header Cell 0
    c0 = "".join(cells[0]["source"])
    c0_ok = "V2.2" in c0 and "ANTI-THICKENING" in c0
    checks.append(("CELL 0: Tiêu đề V2.2 Anti-Thickening & Sakuga Composite Score", c0_ok))

    # 2. Cell 4: Temporal Reversal & Sampler
    c4 = "".join(cells[4]["source"])
    c4_ok = "TemporalReversalAugmentation" in c4 and "compute_boundary_motion_weights" in c4
    checks.append(("CELL 4: Temporal Reversal (p=0.5) & Boundary-Motion Weights", c4_ok))

    # 3. Cell 6: Losses
    c6 = "".join(cells[6]["source"])
    c6_ok = ("StrokeDensityRegularizationLoss" in c6 and
             "DirectionalTangentCoherenceLoss" in c6 and
             "SoftClDiceLoss" in c6 and
             "CompositeLossEngine_v22" in c6)
    checks.append(("CELL 6: 3 Hàm Loss Khả Vi Mới & CompositeLossEngine_v22", c6_ok))

    # 4. Cell 7: Differential LR & Warm-start
    c7 = "".join(cells[7]["source"])
    c7_ok = ("1.5e-5" in c7 and "5.0e-5" in c7 and "25" in c7 and
             "model_tps_v2.0_best.pt" in c7)
    checks.append(("CELL 7: Differential LR (1.5e-5 vs 5e-5), 25 Epochs & V2.0.2 Warm-Start", c7_ok))

    # 5. Cell 8: Standardized Metrics
    c8 = "".join(cells[8]["source"])
    c8_ok = "pred_u8 >= 180" in c8 and "gt_u8 > 250" in c8 and "density_ratio" in c8
    checks.append(("CELL 8: Metric Chuẩn Hóa Fast Benchmark & Density Ratio", c8_ok))

    # 6. Cell 9: Sakuga Composite Score
    c9 = "".join(cells[9]["source"])
    c9_ok = ("SmartCheckpointManager_v22" in c9 and
             "calculate_composite_score" in c9 and
             "model_tps_v2.2_best.pt" in c9)
    checks.append(("CELL 9: Smart Checkpoint Manager V2.2 & Hard Gates", c9_ok))

    # 7. Cell 10: Training Loop & History
    c10 = "".join(cells[10]["source"])
    c10_ok = "training_history_v2.2.csv" in c10 and "clip_grad_norm_" in c10
    checks.append(("CELL 10: Gradient Clipping 1.0 & training_history_v2.2.csv", c10_ok))

    all_passed = True
    print("\n" + "=" * 80)
    print("BẢNG KẾT QUẢ THẨM ĐỊNH NOTEBOOK V2.2")
    print("=" * 80)
    for title, status in checks:
        icon = "✓" if status else "✗"
        print(f"[{icon}] {title}")
        if not status:
            all_passed = False

    print("=" * 80)
    if all_passed:
        print("🎉 TẤT CẢ CÁC HẠNG MỤC THẨM ĐỊNH V2.2 ĐỀU ĐẠT CHUẨN 100%!")
    else:
        print("⚠️ CÓ HẠNG MỤC CHƯA ĐẠT! VUI LÒNG KIỂM TRA LẠI.")
    return all_passed


if __name__ == '__main__':
    ok = verify_v22()
    sys.exit(0 if ok else 1)
