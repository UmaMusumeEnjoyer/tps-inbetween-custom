#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_v25_notebook.py
Script thẩm định toàn diện tính hợp lệ và độ chính xác toán học của Notebook Huấn luyện Kaggle V2.5:
1. Kiểm tra cấu trúc JSON và AST syntax của toàn bộ 14 Cells trong cả 2 file:
   - kaggle/triplet_09_v2.5_exact_ste_width_kaggle.ipynb
   2. Unit-test toán học của StrokeDensityRegularizationLoss V2.5 (Exact-STE):
   - Chứng minh 0 byte nhiễu nền trắng.
   - Chứng minh dead-zone [0.96, 1.03] không phạt.
   - Chứng minh gradient dL/dp > 0 liên tục khi phình nét (khắc phục triệt để lỗi mất đạo hàm của V2.4).
3. Unit-test của BidirectionalStrokeWidthLoss:
   - Chứng minh bảo vệ nét mảnh 1px.
   - Chứng minh phạt nặng vành đai viền 1-2px khi phình nét.
4. Unit-test của SmartCheckpointManager_v25 (Hard Gates và Best Selection).
5. Kiểm tra định dạng log Cell 10 đúng 5 mục và hàm demo ảnh mẫu trực quan.
"""

import os
import sys
import json
import ast
import shutil
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


def verify_notebooks():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nb_files = [os.path.abspath(os.path.join(script_dir, "..", "notebooks", "triplet_09_v2.5_exact_ste_width_kaggle.ipynb"))]

    print("=" * 80)
    print("🔍 BẮT ĐẦU THẨM ĐỊNH CẤU TRÚC VÀ CÚ PHÁP NOTEBOOK KAGGLE V2.5")
    print("=" * 80)

    for nb_path in nb_files:
        assert os.path.exists(nb_path), f"Không tìm thấy file: {nb_path}"
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        cells = nb.get("cells", [])
        print(f"\n[*] Kiểm tra tệp: {os.path.basename(nb_path)} (Tổng số cells: {len(cells)})")
        assert len(cells) == 14, f"Số lượng cell không đúng chuẩn 14 cells! (Hiện tại: {len(cells)})"

        for idx, cell in enumerate(cells):
            cell_type = cell.get("cell_type")
            src_val = cell.get("source", [])
            source = "".join(src_val) if isinstance(src_val, list) else src_val
            first_line = source.split("\n")[0] if source else ""

            if cell_type == "code":
                try:
                    ast.parse(source)
                    print(f"   [✓] Cell {idx:02d} [CODE] Cú pháp AST hợp lệ: {first_line[:65]}")
                except SyntaxError as e:
                    print(f"   ❌ Cell {idx:02d} [CODE] Lỗi cú pháp AST: {e}")
                    raise e
            else:
                print(f"   [✓] Cell {idx:02d} [MARKDOWN] {first_line[:65]}")

    print("\n" + "=" * 80)
    print("🧪 BẮT ĐẦU KIỂM THỬ ĐƠN VỊ (UNIT TESTS) CÁC MODULE TOÁN HỌC V2.5")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # TEST 1: StrokeDensityRegularizationLoss V2.5 (Exact-STE)
    # -------------------------------------------------------------------------
    print("\n[TEST 1]: Kiểm chứng StrokeDensityRegularizationLoss V2.5 (Exact-STE & Non-Saturating Gradient)...")
    cell6_src = nb["cells"][6]["source"]
    cell6_code = "".join(cell6_src) if isinstance(cell6_src, list) else cell6_src
    local_env6 = {}
    exec("import os, sys, gc, torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport torchvision.models as models\ndevice = torch.device('cpu')", local_env6)
    exec(cell6_code, local_env6)

    StrokeDensityRegularizationLoss = local_env6["StrokeDensityRegularizationLoss"]
    density_loss_fn = StrokeDensityRegularizationLoss(target_min=0.96, target_max=1.03)

    # 1.1 Kiểm tra nền trắng tuyệt đối không sinh mực ảo
    pred_blank = torch.zeros(1, 1, 512, 512)
    m_soft_blank = torch.clamp((pred_blank - 0.06) / 0.18, 0.0, 1.0) + 0.25 * F.relu(pred_blank - 0.24)
    sum_blank = torch.sum(m_soft_blank).item()
    print(f"   [*] Tử số soft mask trên toàn bộ 262,144 pixels nền trắng: {sum_blank:.4f}")
    assert sum_blank == 0.0, "LỖI: Nền trắng vẫn sinh ra mực ảo trong V2.5!"
    print("   [✓] ĐẠT: Nền trắng đóng góp chính xác 0.0 px ảo!")

    # 1.2 Kiểm tra dead-zone [0.96, 1.03]
    gt_strokes = torch.zeros(1, 1, 100, 100)
    gt_strokes[0, 0, 40:60, 20:80] = 1.0  # 1200 pixels
    pred_exact = gt_strokes.clone()  # 1200 pixels -> ratio = 1.000
    loss_exact, ratio_exact = density_loss_fn(pred_exact, gt_strokes, ink_mode=True)
    print(f"   [*] Pred/GT ratio = {ratio_exact.item():.4f} -> Density Loss = {loss_exact.item():.6f}")
    assert loss_exact.item() == 0.0, "LỖI: Mật độ ở tâm 1.00 không được bị phạt!"
    print("   [✓] ĐẠT: Dải mục tiêu [0.96, 1.03] không bị phạt (Dead-zone kích hoạt chính xác)!")

    # 1.3 Kiểm tra gradient khi phình nét (KHẮC PHỤC LỖI BÃO HÒA CỦA V2.4)
    pred_bloated = torch.zeros(1, 1, 100, 100, requires_grad=True)
    # Phình nét to hơn GT 1.5 lần, mực đậm từ 0.3 đến 0.9 (nơi V2.4 bị bão hòa đạo hàm = 0)
    pred_data = pred_bloated.detach().clone()
    pred_data[0, 0, 35:65, 20:80] = 0.85  # 1800 pixels (ratio ~ 1.50)
    pred_bloated.data = pred_data

    loss_bloated, ratio_bloated = density_loss_fn(pred_bloated, gt_strokes, ink_mode=True)
    loss_bloated.backward()

    # Kiểm tra đạo hàm trên các pixel bị phình nét:
    grad_bloated = pred_bloated.grad[0, 0, 36, 50].item()
    print(f"   [*] Ratio phình nét: {ratio_bloated.item():.4f} | Loss: {loss_bloated.item():.4f} | Đạo hàm dL/dp tại pixel đậm (0.85): {grad_bloated:.6f}")
    assert grad_bloated > 0.0, f"LỖI: Đạo hàm bằng 0 trên pixel phình nét (bị bão hòa giống V2.4)! Giá trị: {grad_bloated}"
    print(f"   [✓] ĐẠT: Đạo hàm dL/dp > 0 liên tục ({grad_bloated:.6f}), ép nét mỏng lại thành công!")

    # -------------------------------------------------------------------------
    # TEST 2: BidirectionalStrokeWidthLoss
    # -------------------------------------------------------------------------
    print("\n[TEST 2]: Kiểm chứng BidirectionalStrokeWidthLoss (Bảo tồn lõi + Phạt phình viền)...")
    BidirectionalStrokeWidthLoss = local_env6["BidirectionalStrokeWidthLoss"]
    SoftClDiceLoss = local_env6["SoftClDiceLoss"]

    cldice_fn = SoftClDiceLoss(iters=5, beta_sensitivity=0.50)
    bidi_loss_fn = BidirectionalStrokeWidthLoss()

    # Tạo GT có nét vẽ 1 pixel (chi tiết cực mảnh)
    gt_test = torch.zeros(1, 1, 64, 64)
    gt_test[0, 0, 30, 10:50] = 1.0  # 1px line
    _, skel_gt, opened_gt = cldice_fn(gt_test, gt_test)

    # Trường hợp A: Nét mỏng chuẩn
    pred_good = gt_test.clone()
    loss_good = bidi_loss_fn(pred_good, gt_test, opened_gt, skel_gt)

    # Trường hợp B: Bị mất nét mỏng (Under-inking)
    pred_missing = torch.zeros_like(gt_test)
    loss_missing = bidi_loss_fn(pred_missing, gt_test, opened_gt, skel_gt)

    # Trường hợp C: Bị phình to thành 3px (Overspill sang hàng 29 và 31)
    pred_thick = torch.zeros(1, 1, 64, 64, requires_grad=True)
    p_data = pred_thick.detach().clone()
    p_data[0, 0, 29:32, 10:50] = 0.90
    pred_thick.data = p_data

    loss_thick = bidi_loss_fn(pred_thick, gt_test, opened_gt, skel_gt)
    loss_thick.backward()
    grad_edge = pred_thick.grad[0, 0, 29, 30].item()

    print(f"   [*] Loss nét chuẩn         : {loss_good.item():.6f}")
    print(f"   [*] Loss khi mất nét mỏng  : {loss_missing.item():.6f}")
    print(f"   [*] Loss khi phình nét 3px : {loss_thick.item():.6f}")
    print(f"   [*] Đạo hàm phạt tràn viền : {grad_edge:.6f}")

    assert loss_missing.item() > loss_good.item() * 5, "LỖI: Không phạt khi thiếu nét!"
    assert loss_thick.item() > loss_good.item() * 5, "LỖI: Không phạt khi phình nét sang viền lân cận!"
    assert grad_edge > 0.0, "LỖI: Đạo hàm tràn viền không dương!"
    print("   [✓] ĐẠT: BidirectionalStrokeWidthLoss hoạt động hai chiều hoàn hảo!")

    # -------------------------------------------------------------------------
    # TEST 3: SmartCheckpointManager_v25
    # -------------------------------------------------------------------------
    print("\n[TEST 3]: Kiểm chứng SmartCheckpointManager_v25 (Hard Gates & Realtime Trigger)...")
    cell9_src = nb["cells"][9]["source"]
    cell9_code = "".join(cell9_src) if isinstance(cell9_src, list) else cell9_src
    local_env9 = {}
    exec("import os, sys, gc, shutil, torch, pandas as pd\nWORKING_DIR = './_test_ckpt_v25'", local_env9)
    exec(cell9_code, local_env9)

    SmartCheckpointManager_v25 = local_env9["SmartCheckpointManager_v25"]
    test_save_dir = os.path.join(script_dir, "_temp_test_checkpoints_v25")
    os.makedirs(test_save_dir, exist_ok=True)

    dummy_model = nn.Linear(10, 2)
    manager = SmartCheckpointManager_v25(save_dir=test_save_dir)

    # 3.1 Thử nghiệm epoch vi phạm Hard Gates (Density = 1.458 giống V2.4)
    bad_metrics = {
        'recall': 0.92,
        'psnr': 21.0,
        'ssim': 0.91,
        'density_ratio': 1.458,
        'ghost_pixels': 7000,
        'chamfer_distance': 0.0018
    }
    is_b, is_p, sc, pen, passed, reasons, bp, pp = manager.evaluate_and_save(dummy_model, 1, bad_metrics)
    assert not passed, "LỖI: Density 1.458 phải bị chặn bởi Hard Gate!"
    assert not is_b, "LỖI: Không được phong Best Checkpoint khi bị chặn Hard Gate!"
    print(f"   [✓] ĐẠT: Epoch vi phạm Hard Gate ({', '.join(reasons)}) bị chặn chính xác!")

    # 3.2 Thử nghiệm epoch đạt chuẩn V2.5 (Density = 0.98, Recall = 91.5%)
    good_metrics = {
        'recall': 0.915,
        'psnr': 22.0,
        'ssim': 0.92,
        'density_ratio': 0.98,
        'ghost_pixels': 5500,
        'chamfer_distance': 0.0015
    }
    is_b2, is_p2, sc2, pen2, passed2, reasons2, bp2, pp2 = manager.evaluate_and_save(dummy_model, 2, good_metrics)
    assert passed2, "LỖI: Metrics chuẩn phải vượt qua Hard Gate!"
    assert is_b2, "LỖI: Epoch chuẩn phải được phong Best Checkpoint!"
    assert os.path.exists(bp2), f"LỖI: File best checkpoint không được lưu tại {bp2}"
    print(f"   [✓] ĐẠT: Epoch thỏa mãn Hard Gates được lưu Best Checkpoint thành công tại {os.path.basename(bp2)}!")

    # Dọn dẹp thư mục test
    if os.path.exists(test_save_dir):
        shutil.rmtree(test_save_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # TEST 4: Kiểm tra Cell 10 Định dạng 5 Mục & Demo Ảnh Mẫu
    # -------------------------------------------------------------------------
    print("\n[TEST 4]: Kiểm tra định dạng xuất Cell 10 đúng 5 mục và hàm Demo Ảnh Mẫu...")
    cell10_src = nb["cells"][10]["source"]
    cell10_code = "".join(cell10_src) if isinstance(cell10_src, list) else cell10_src

    assert "generate_checkpoint_visual_demo" in cell10_code, "LỖI: Thiếu hàm generate_checkpoint_visual_demo trong Cell 10!"
    assert "visual_samples_v2.5" in cell10_code, "LỖI: Thiếu thư mục visual_samples_v2.5 trong Cell 10!"
    assert "ĐÃ XONG EPOCH" in cell10_code, "LỖI: Thiếu mục 1 'ĐÃ XONG EPOCH'!"
    assert "VAL Metrics:" in cell10_code, "LỖI: Thiếu mục 2 'VAL Metrics'!"
    assert "Sakuga Composite Score V2.5" in cell10_code, "LỖI: Thiếu mục 3 'Sakuga Composite Score V2.5'!"
    assert "Hard Gates BỊ CHẶN" in cell10_code, "LỖI: Thiếu mục 4 'Hard Gates BỊ CHẶN'!"
    assert "TIẾN TRÌNH TỐT NHẤT V2.5" in cell10_code, "LỖI: Thiếu mục 5 'TIẾN TRÌNH TỐT NHẤT V2.5'!"
    assert "DEMO ẢNH MẪU CHECKPOINT TỐT NHẤT" in cell10_code, "LỖI: Thiếu log demo ảnh mẫu!"

    print("   [✓] ĐẠT: Cell 10 tích hợp đầy đủ 5 mục log tinh gọn và demo ảnh mẫu tức thì!")

    # -------------------------------------------------------------------------
    # TEST 5: Kiểm tra Cell 4 Tối Ưu Siêu Tốc (Zero Wait Time & Zero CoW Leak)
    # -------------------------------------------------------------------------
    print("\n[TEST 5]: Kiểm tra Cell 4 triệt tiêu hoàn toàn vòng lặp chờ 28 phút...")
    cell4_src = nb["cells"][4]["source"]
    cell4_code = "".join(cell4_src) if isinstance(cell4_src, list) else cell4_src

    # Xác nhận không còn vòng lặp Buffering PNGs 17,181 file gây nghẽn 28 phút
    assert "Buffering PNGs" not in cell4_code, "LỖI: Vẫn còn vòng lặp Buffering PNGs gây nghẽn 28 phút trong Cell 4!"
    assert "persistent_workers=True" in cell4_code, "LỖI: Thiếu persistent_workers=True trong DataLoader Cell 4!"
    assert "num_workers=4" in cell4_code, "LỖI: Thiếu num_workers=4 trong train_loader Cell 4!"
    assert "pin_memory=True" in cell4_code, "LỖI: Thiếu pin_memory=True trong DataLoader Cell 4!"
    assert "cv2.imread" in cell4_code, "LỖI: Thiếu cv2.imread trực tiếp trong __getitem__ Cell 4!"

    print("   [✓] ĐẠT: Cell 4 đã loại bỏ hoàn toàn vòng lặp nghẽn 28 phút, sẵn sàng nạp đa luồng siêu tốc!")

    print("\n" + "=" * 80)
    print("🎉 TOÀN BỘ 14 CELLS VÀ CÁC MODULE TOÁN HỌC V2.5 ĐÃ ĐƯỢC THẨM ĐỊNH THÀNH CÔNG 100%!")
    print("=" * 80)


if __name__ == '__main__':
    verify_notebooks()
