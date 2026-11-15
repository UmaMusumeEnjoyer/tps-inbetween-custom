#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_v23_notebook.py
Script thẩm định tính hợp lệ toàn diện của notebook Kaggle V2.3 trước khi đẩy lên Cloud:
1. Thẩm định cấu trúc JSON và 14 cell chuẩn (cho cả file chính và alias).
2. Kiểm tra Cell 6: Bọc FP32 & tắt autocast trong StrokeDensityRegularizationLoss và DirectionalTangentCoherenceLoss.
3. Kiểm tra Cell 8: Thêm xấp xỉ Chamfer Distance đa tỷ lệ GPU trong calculate_metrics_v1.
4. Kiểm tra Cell 9: SmartCheckpointManager V2.3 loại bỏ lưu epoch 5, 10, 15, 20..., chỉ lưu Best & Last Checkpoint.
5. Kiểm tra Cell 10: Thu thập val_density_list, val_chamfer_list, truyền đầy đủ vào val_metrics và history.
6. Kiểm tra Cell 11: 6-Panel Dashboard và bảng tổng hợp có Chamfer Distance.
7. Kiểm tra Cell 13: Đóng gói artifact V2.3 với Best và Final Checkpoint.
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


def verify_single_notebook(nb_path):
    print(f"\n[*] Đang thẩm định: {nb_path}")
    if not os.path.exists(nb_path):
        raise FileNotFoundError(f"Không tìm thấy file: {nb_path}")

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    print(f"[*] Tổng số cell trong notebook: {len(cells)}")
    assert len(cells) == 14, f"Số cell ({len(cells)}) không đúng 14 cell chuẩn!"

    checks = []

    # 1. Header Cell 0
    c0 = "".join(cells[0]["source"])
    c0_ok = "V2.3" in c0 and "NUMERICAL STABILITY" in c0
    checks.append(("CELL 0: Tiêu đề V2.3 Numerical Stability & GPU Chamfer", c0_ok))

    # 1.b Cell 1: cuDNN Stability Mode (Benchmark = False để ngăn SIGSEGV Multi-GPU DataParallel)
    c1 = "".join(cells[1]["source"])
    c1_ok = "torch.backends.cudnn.benchmark = False" in c1
    checks.append(("CELL 1: Ổn định cuDNN (benchmark=False chống crash SIGSEGV)", c1_ok))

    # 2. Cell 4: Temporal Reversal, Sampler & Fast Multithreaded Loading & RAM Buffering
    c4 = "".join(cells[4]["source"])
    c4_ok = (
        "TemporalReversalAugmentation" in c4
        and "self.temporal_aug = TemporalReversalAugmentation" in c4
        and "self.temporal_aug(" in c4
        and "t_val = np.full((1, 1, 1), t_scalar" in c4
        and "VAL_BATCH_SIZE = max(1, NUM_GPUS)" in c4
        and "persistent_workers=False" in c4
        and "pin_memory=False" in c4
        and "ThreadPoolExecutor" in c4
        and "self.img_buffers" in c4
        and "imdecode" in c4
        and "tqdm" in c4
    )
    checks.append(("CELL 4: Tối ưu I/O RAM Buffering, Khử SIGBUS/IPC & Augmentation", c4_ok))

    # 3. Cell 6: Float32 Protected Losses & Zero-Sync CPU-GPU
    c6 = "".join(cells[6]["source"])
    c6_ok = (
        "torch.amp.autocast('cuda', enabled=False)" in c6
        and "return loss, ratio_mask.mean().detach()" in c6
        and "ratio_mask.mean().item()" not in c6
        and "thresh=0.15" in c6
        and "temp=25.0" in c6
        and "valid_gt = sum_g_mask > 20.0" in c6
        and "ratio = 0.7 * ratio_mask + 0.3 * ratio_raw" in c6
        and "near_halo" in c6
        and "far_ghost" in c6
        and "0.65 * loss_density" in c6
        and "0.25 * loss_ghost" in c6
        and "CompositeLossEngine_v23" in c6
    )
    checks.append(("CELL 6: Threshold-Aware Density (0.15) & Anti-Halo Ghost Loss (Zero-Sync)", c6_ok))

    # 4. Cell 7: Differential LR & Warm-start V2.2
    c7 = "".join(cells[7]["source"])
    c7_ok = (
        "1.5e-5" in c7
        and "5.0e-5" in c7
        and "25" in c7
        and "model_tps_v2.2_best.pt" in c7
    )
    checks.append(("CELL 7: Differential LR (1.5e-5 vs 5e-5) & V2.2 Checkpoint Warm-Start", c7_ok))

    # 5. Cell 8: Chamfer Distance GPU Siêu Tốc & Vectorized Batched Metrics
    c8 = "".join(cells[8]["source"])
    c8_ok = (
        "chamfer_t = torch.mean(p_mask * (1.0 - g_dilated)) + torch.mean(g_mask * (1.0 - p_dilated))" in c8
        and "calculate_metrics_batch_gpu" in c8
        and "'chamfer_distance': chamfer_distance" in c8
        and "'density_ratio': density_ratio" in c8
    )
    checks.append(("CELL 8: GPU Fast Chamfer & Vectorized calculate_metrics_batch_gpu", c8_ok))

    # 6. Cell 9: Smart Checkpoint Manager V2.3 (Adaptive Hard Gate & Fallback Best Tracking)
    c9 = "".join(cells[9]["source"])
    c9_no_epoch_5 = "current_epoch % 5 == 0" not in c9
    c9_has_best_and_last = (
        "model_tps_v2.3_best.pt" in c9
        and "model_tps_v2.3_latest.pt" in c9
        and "model_tps_v2.3_last.pt" in c9
        and "ref_chamfer=0.002241" in c9
        and "model_tps_v2.3_best_progress.pt" in c9
        and "best_relaxed_score" in c9
        and "SmartCheckpointManager_v23" in c9
    )
    checks.append(("CELL 9: Chuẩn hóa ref_chamfer=0.002241 & Fallback Best Tracking", c9_no_epoch_5 and c9_has_best_and_last))

    # 7. Cell 10: Fast Validation Inference, Metrics & RAM Leak Prevention
    c10 = "".join(cells[10]["source"])
    c10_compute_loss_false = "compute_loss=False" in c10
    c10_l1_direct = "v_l1 = F.l1_loss(pred, tgt)" in c10
    c10_val_lists = "val_density_list, val_chamfer_list = [], []" in c10
    c10_val_append = (
        ("val_density_list.extend(bm['density_ratio'])" in c10 or "val_density_list.append(m['density_ratio'])" in c10)
        and ("val_chamfer_list.extend(bm['chamfer_distance'])" in c10 or "val_chamfer_list.append(m['chamfer_distance'])" in c10)
    )
    c10_val_dict = (
        "'density_ratio': mean_val_density" in c10
        and "'chamfer_distance': mean_val_chamfer" in c10
    )
    c10_ram_leak_fix = (
        ("val_step + 1) % 50 == 0" in c10 or "(val_step + 1) % 100 == 0" in c10)
        and "gc.collect()" in c10
        and "malloc_trim(0)" in c10
    )
    c10_history_csv = "training_history_v2.3.csv" in c10
    checks.append(("CELL 10: Fast Val Inference, Metric Tracking & Triệt tiêu Host RAM Leak",
                   c10_compute_loss_false and c10_l1_direct and c10_val_lists and c10_val_append and c10_val_dict and c10_ram_leak_fix and c10_history_csv))

    # 8. Cell 11: Dashboard & Summary Table
    c11 = "".join(cells[11]["source"])
    c11_ok = (
        "training_history_v2.3.csv" in c11
        and "training_metrics_convergence_v23.png" in c11
        and "Chamfer Distance" in c11
    )
    checks.append(("CELL 11: 6-Panel Dashboard & Bảng tổng hợp có Chamfer Distance", c11_ok))

    # 9. Cell 12: Visual Samples V2.3
    c12 = "".join(cells[12]["source"])
    c12_ok = "visual_samples_v2.3" in c12
    checks.append(("CELL 12: Thư mục kiểm toán trực quan visual_samples_v2.3", c12_ok))

    # 10. Cell 13: Packaging
    c13 = "".join(cells[13]["source"])
    c13_ok = (
        "model_tps_v2.3_best.pt" in c13
        and "model_tps_v2.3_last.pt" in c13
        and "tps_finetune_v2.3_artifacts.zip" in c13
        and "training_history_v2.3.csv" in c13
    )
    checks.append(("CELL 13: Đóng gói Artifacts V2.3 (Best & Final Checkpoint)", c13_ok))

    all_passed = True
    print("=" * 80)
    print(f"BẢNG KẾT QUẢ THẨM ĐỊNH NOTEBOOK: {os.path.basename(nb_path)}")
    print("=" * 80)
    for title, status in checks:
        icon = "✓" if status else "✗"
        print(f"[{icon}] {title}")
        if not status:
            all_passed = False

    print("=" * 80)
    if all_passed:
        print(f"🎉 TẤT CẢ 10 HẠNG MỤC THẨM ĐỊNH V2.3 CỦA {os.path.basename(nb_path)} ĐỀU ĐẠT CHUẨN 100%!")
    else:
        print(f"⚠️ CÓ HẠNG MỤC CHƯA ĐẠT TRONG {os.path.basename(nb_path)}! VUI LÒNG KIỂM TRA LẠI.")
    return all_passed


def verify_v23():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nb_main = os.path.join(script_dir, "triplet_07_v2.3_zero_sync_chamfer_kaggle.ipynb")
    nb_alias = os.path.join(script_dir, "triplet_07_v2.3_zero_sync_chamfer_kaggle.ipynb")

    ok_main = verify_single_notebook(nb_main)
    ok_alias = # nb_alias check removed

    return ok_main and ok_alias


if __name__ == '__main__':
    ok = verify_v23()
    sys.exit(0 if ok else 1)
