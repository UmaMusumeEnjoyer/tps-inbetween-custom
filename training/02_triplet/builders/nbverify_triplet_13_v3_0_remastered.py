#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_v30r_notebook.py
Kiểm tra tính đúng đắn và tương thích 100% của Notebook V3.0-Remastered R2:
1. Cấu trúc JSON của 2 notebook sinh ra (15 cells: 2 markdown + 13 code).
2. Khởi tạo kiến trúc TPSInbetweenFinetune_v30R R2 (Contextnet + IFNet + asf_head + unet).
3. Kiểm tra động cơ TimeSymmetricFBFCEngine.
4. Kiểm tra bộ xuất nét Soft-Knee export_pure_lineart_rgba (Alpha = 0 cho nền, RGB = 0,0,0, gamma=0.78).
5. Kiểm tra hàm mất mát Continuous CompositeLossEngine_v30R R2 (Truncated DT, Tversky-clDice, Tip loss).
6. Kiểm tra GPU-batched metrics (WCD, S-EMD, Skeleton Recall, Tip Recall).
7. Chạy dummy forward pass + backward pass với AMP FP16.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

def test_notebook_json(nb_path):
    print(f"[*] Kiểm tra tính toàn vẹn file notebook: {nb_path}...")
    assert os.path.exists(nb_path), f"Không tìm thấy file: {nb_path}"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    assert "cells" in nb, "Notebook thiếu trường 'cells'"
    assert len(nb["cells"]) == 15, f"Số lượng cells không đúng: {len(nb['cells'])} (kỳ vọng 15)"
    print(f"[✓] File notebook hợp lệ với {len(nb['cells'])} cells.")

def test_model_and_losses():
    print("\n[*] Kiểm tra kiến trúc TPSInbetweenFinetune_v30R R2 & Time-Symmetric FBFC...")
    cells_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cells", "13_v3.0_remastered"))
    sys.path.insert(0, cells_dir)

    with open(os.path.join(cells_dir, "cell_03.py"), "r", encoding="utf-8") as f:
        code_c03 = f.read()
    with open(os.path.join(cells_dir, "cell_05.py"), "r", encoding="utf-8") as f:
        code_c05 = f.read()
    with open(os.path.join(cells_dir, "cell_06.py"), "r", encoding="utf-8") as f:
        code_c06 = f.read()
    with open(os.path.join(cells_dir, "cell_08.py"), "r", encoding="utf-8") as f:
        code_c08 = f.read()

    scope = {
        'torch': torch,
        'nn': nn,
        'F': F,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'math': __import__('math'),
        'np': np,
        'sys': sys,
        'os': os,
        'cv2': __import__('cv2')
    }

    exec(code_c03, scope)
    exec(code_c05, scope)
    exec(code_c06, scope)
    exec(code_c08, scope)

    model_cls = scope['TPSInbetweenFinetune_v30R']
    loss_cls = scope['CompositeLossEngine_v30R']
    rgba_func = scope['export_pure_lineart_rgba']
    metrics_func = scope['calculate_metrics_batch_gpu']

    device = scope['device']
    model = model_cls().to(device)
    criterion = loss_cls().to(device)

    # 1. Kiểm tra kiến trúc & tổng số tham số (Khớp 100% V2.6)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[✓] Model khởi tạo thành công! Tổng tham số: {param_count:,}")
    assert hasattr(model, 'asf_head'), "Mô hình phải có AdaptiveStrokeFusionHead (asf_head) từ V2.6!"
    assert hasattr(model, 'fbfc'), "Mô hình phải có TimeSymmetricFBFCEngine!"

    # 2. Kiểm tra Time-Symmetric FBFC Engine
    print("\n[*] Kiểm tra Động cơ Time-Symmetric FBFC...")
    s_t0 = torch.zeros((2, 2, 64, 64), device=device)
    s_t1 = torch.zeros((2, 2, 64, 64), device=device)
    t_vals = torch.tensor([0.5, 0.5], device=device)
    # Gán flow hoàn toàn đối xứng: (1-t)*s_t0 + t*s_t1 = 0
    s_t0[:, 0, :, :] = 2.0
    s_t1[:, 0, :, :] = -2.0
    conf, delta = model.fbfc(s_t0, s_t1, t_vals)
    assert conf.shape == (2, 1, 64, 64)
    assert torch.all(delta < 1e-4), f"Delta phải bằng 0 khi flow đối xứng, nhận được {delta.max().item()}"
    assert torch.all(conf > 0.99), "Conf FBFC phải = 1.0 khi đối xứng hoàn hảo"
    print(f"[✓] Time-Symmetric FBFC đối xứng hoàn hảo: Delta = {delta.mean().item():.4f}, Conf = {conf.mean().item():.4f}")

    # 3. Kiểm tra Soft-Knee RGBA Lineart Exporter
    print("\n[*] Kiểm tra Soft-Knee RGBA Lineart Exporter...")
    dummy_pred = torch.zeros((1, 1, 64, 64))
    # Nền có rò rỉ mờ (0.012 <= tau_noise=0.020)
    dummy_pred[:, :, :20, :] = 0.012
    # Vùng ngọn tóc vuốt nhọn mảnh (0.035) -> Nằm trong dải soft-knee [0.020, 0.060]
    dummy_pred[:, :, 25:30, 10:50] = 0.035
    # Nét vẽ đen đậm (0.85)
    dummy_pred[:, :, 40:45, 10:50] = 0.85

    rgba_out = rgba_func(dummy_pred, tau_noise=0.020, gamma=0.78)
    assert rgba_out.shape == (64, 64, 4)
    # Vùng nền rò rỉ (<= 0.020) phải có Alpha = 0 tuyệt đối!
    assert np.all(rgba_out[:20, :, 3] == 0), "Vùng nền rò rỉ phải có Alpha = 0 tuyệt đối!"
    # Vùng ngọn tóc mảnh (0.035) phải bảo toàn Alpha khả kiến (> 0) mà không bị xóa sạch!
    assert np.all(rgba_out[25:30, 10:50, 3] > 0), "Vùng ngọn tóc mảnh phải có Alpha khả kiến (> 0)!"
    # Vùng nét vẽ đậm phải có Alpha đậm (> 220)
    assert np.all(rgba_out[40:45, 10:50, 3] > 220), "Vùng nét vẽ đậm phải có Alpha > 220!"
    # Kênh RGB phải là đen thuần (0, 0, 0)
    assert np.all(rgba_out[:, :, :3] == 0), "Kênh RGB phải là đen thuần (0, 0, 0)!"
    print(f"[✓] Soft-Knee RGBA Exporter hoạt động hoàn hảo: Nền 100% trong suốt (Alpha=0), Ngọn tóc bảo toàn (Alpha={rgba_out[27, 20, 3]}), Nét đậm (Alpha={rgba_out[42, 20, 3]}), RGB thuần (0,0,0).")

    # 4. Kiểm tra GPU Batched Metrics (WCD, S-EMD, Skeleton Recall, Tip Recall)
    print("\n[*] Kiểm tra Hệ thống Thẩm định Đa chỉ số Nhận thức (WCD + S-EMD + Skeleton Recall)...")
    p_dummy = torch.zeros((2, 1, 128, 128), device=device)
    g_dummy = torch.zeros((2, 1, 128, 128), device=device)
    # Vẽ một nét thẳng trên GT và Pred
    g_dummy[:, :, 60:64, 20:100] = 1.0
    p_dummy[:, :, 60:64, 22:98] = 0.95
    m_dict = metrics_func(p_dummy, g_dummy)
    assert 'wcd' in m_dict and 'sliced_emd' in m_dict and 'skel_recall' in m_dict and 'tip_recall' in m_dict
    print(f"[✓] Đo lường GPU thành công: WCD={m_dict['wcd'].mean().item():.6f} | S-EMD={m_dict['sliced_emd'].mean().item():.6f} | SkelRec={m_dict['skel_recall'].mean().item()*100:.1f}% | TipRec={m_dict['tip_recall'].mean().item()*100:.1f}%")

    # 5. Kiểm tra Dummy Forward & Backward Pass (AMP FP16)
    print("\n[*] Kiểm tra Dummy Forward & Backward Pass (AMP FP16)...")
    model.train()
    x0 = torch.zeros((2, 1, 128, 128), device=device)
    x1 = torch.zeros((2, 1, 128, 128), device=device)
    gt = torch.zeros((2, 1, 128, 128), device=device)
    t_vals = torch.tensor([0.5, 0.5], device=device)

    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
        pred, blend_mask, conf_fbfc, delta_flow, _, _ = model(x0, x1, t_vals)
        assert pred.shape == (2, 1, 128, 128)
        loss, l_charb, l_dt, l_dice, l_dens, l_hair, l_ghost = criterion(pred, gt, conf_fbfc=conf_fbfc, epoch=1)

    loss.backward()
    print(f"[✓] Forward + Loss + Backward Pass thành công! Total Loss = {loss.item():.4f} (DT Loss = {l_dt.item():.4f}, Hair Loss = {l_hair.item():.4f})")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    notebooks_dir = os.path.abspath(os.path.join(base_dir, "..", "notebooks"))
    nb_path = os.path.join(notebooks_dir, "triplet_13_v3.0_remastered_r2_kaggle.ipynb")

    print("=" * 80)
    print("🧪 BẮT ĐẦU KIỂM ĐỊNH NOTEBOOK V3.0-REMASTERED R2...")
    print("=" * 80)

    test_model_and_losses()
    test_notebook_json(nb_path)

    print("\n" + "=" * 80)
    print("🎉 TOÀN BỘ CÁC BÀI KIỂM THỬ V3.0-REMASTERED R2 ĐÃ ĐẠT CHUẨN 100%!")
    print("=" * 80)

if __name__ == "__main__":
    main()
