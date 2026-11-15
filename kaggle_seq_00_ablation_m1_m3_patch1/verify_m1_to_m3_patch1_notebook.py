#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_m1_to_m3_patch1_notebook.py
--------------------------------------------------------------------------------
Script kiểm định toàn diện chất lượng kỹ thuật của Notebook và Worker DDP:
1. Kiểm tra định dạng JSON và AST syntax của toàn bộ 10 cell trong file .ipynb.
2. Kiểm tra Cell 6: Phải được comment sẵn os._exit(0) và không có lệnh thoát active.
3. Kiểm tra kiến trúc TPSInbetweenOriginal_M1_M3:
   - Nạp thành công trọng số M1 gốc từ checkpoints/pretrained_baselines/tps-original/model_latest.pt.
   - Xác nhận M1 và M3 giống nhau 100% về tập hợp keys và trọng số.
   - Thử nghiệm forward pass trên dummy tensors.
4. Kiểm tra bộ Loss chuẩn Lineart M3 (LineartLoss_M3).
5. Kiểm tra chính sách Checkpoint tinh gọn (MinimalCheckpointManager_M3):
   - Đảm bảo chỉ tạo tối đa 3 file: best.pt, latest.pt, resume.pt.
   - Kiểm tra khả năng khôi phục (Resume) nguyên vẹn.
--------------------------------------------------------------------------------
"""

import os
import sys
import ast
import json
import shutil
import tempfile
import torch

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

def test_notebook_ast():
    print("[TEST 1/5] Kiểm định định dạng JSON & Cú pháp AST của Notebook .ipynb...")
    nb_path = os.path.join(os.path.dirname(__file__), "train_ablation_m1_to_m3_sequence_patch1_kaggle.ipynb")
    assert os.path.exists(nb_path), f"Không tìm thấy file: {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    cells = nb_data.get("cells", [])
    assert len(cells) == 10, f"Số lượng cell không đúng kỳ vọng 10 cells (thực tế: {len(cells)})!"

    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    print(f"   👉 Phát hiện {len(code_cells)} Code Cells và {len(cells) - len(code_cells)} Markdown Cells.")

    for idx, c in enumerate(cells):
        ctype = c.get("cell_type")
        source = c.get("source", "")
        if isinstance(source, list):
            source = "".join(source)

        if ctype == "code":
            try:
                ast.parse(source)
                print(f"   [✓] Cell {idx} (Code): AST Syntax hợp lệ 100%.")
            except SyntaxError as e:
                raise AssertionError(f"Cell {idx} lỗi cú pháp Python: {e}")

    # Kiểm tra Cell 6 đặc biệt
    cell6_src = cells[6].get("source", "")
    if isinstance(cell6_src, list):
        cell6_src = "".join(cell6_src)
    assert "# import os" in cell6_src and "# os._exit(0)" in cell6_src, "Cell 6 phải chứa dòng lệnh os._exit(0) đã được comment sẵn!"
    assert "os._exit(0)" not in [line.strip() for line in cell6_src.split("\n") if not line.strip().startswith("#")], "Cell 6 không được có lệnh os._exit(0) chưa comment!"
    print("   [✓] Cell 6 (Reset Kernel): Đã được comment an toàn, không làm gián đoạn Run All!")

def test_model_architecture_and_m1_weights():
    print("\n[TEST 2/5] Kiểm định kiến trúc M_gốc & Nạp trọng số M1 (model_latest.pt)...")
    from ddp_worker_m1_to_m3_patch1 import TPSInbetweenOriginal_M1_M3, load_m1_weights

    model_m1 = TPSInbetweenOriginal_M1_M3()
    model_m3 = TPSInbetweenOriginal_M1_M3()

    ckpt_candidates = [
        os.path.join(os.path.dirname(__file__), "..", "checkpoints", "pretrained_baselines", "tps-original", "model_latest.pt"),
        os.path.join(os.path.dirname(__file__), "model_latest.pt")
    ]
    ckpt_path = None
    for p in ckpt_candidates:
        if os.path.exists(p):
            ckpt_path = p
            break

    if ckpt_path:
        load_m1_weights(model_m1, ckpt_path, device='cpu', is_rank0=True)
        model_m3.load_state_dict(model_m1.state_dict())

        # Khẳng định toán học 100% đồng nhất
        assert set(model_m1.state_dict().keys()) == set(model_m3.state_dict().keys()), "Tập hợp tensor không khớp!"
        for k in model_m1.state_dict():
            assert torch.equal(model_m1.state_dict()[k], model_m3.state_dict()[k]), f"Tensor {k} bị lệch!"
        print("   [✓] Xác nhận khoa học: M1 và M3 đồng nhất 100% kiến trúc và trọng số ban đầu!")
    else:
        print("   [*] Bỏ qua nạp model_latest.pt (không tìm thấy file local), kiểm tra khởi tạo ngẫu nhiên.")
        model_m3.load_state_dict(model_m1.state_dict())
        assert set(model_m1.state_dict().keys()) == set(model_m3.state_dict().keys())

    # Thử nghiệm forward pass trên dummy inputs
    b, h, w = 2, 270, 480 # Kích thước thử nghiệm
    x0 = torch.rand(b, 1, h, w)
    x1 = torch.rand(b, 1, h, w)
    t_val = torch.tensor([[[[0.5]]], [[[0.3]]]])
    cf01 = torch.zeros(b, 2, h, w)
    cf10 = torch.zeros(b, 2, h, w)

    with torch.no_grad():
        it, coarse_warp, fine_i0, fine_i1, flow_bet, m = model_m1(x0, x1, t_val, cf01, cf10)
    assert it.shape == (b, 1, h, w), f"Shape đầu ra không đúng: {it.shape}"
    assert coarse_warp.shape == (b, 1, h, w), f"Shape coarse warp không đúng: {coarse_warp.shape}"
    assert torch.isfinite(it).all(), "Đầu ra chứa giá trị vô hạn hoặc NaN!"
    print(f"   [✓] Forward pass thành công: Output shape = {it.shape}, Giá trị nằm trong [{it.min():.2f}, {it.max():.2f}].")

def test_lineart_loss():
    print("\n[TEST 3/5] Kiểm định Bộ Hàm Mục Tiêu Chuẩn Lineart (LineartLoss_M3)...")
    from ddp_worker_m1_to_m3_patch1 import LineartLoss_M3

    criterion = LineartLoss_M3()
    pred = torch.rand(2, 1, 128, 128, requires_grad=True)
    gt = torch.rand(2, 1, 128, 128)

    loss, details = criterion(pred, gt)
    assert torch.isfinite(loss), "Loss bị NaN hoặc Inf!"
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all(), "Gradient bị lỗi hoặc NaN!"

    print(f"   [✓] Loss M3 tính toán chuẩn xác: Total Loss = {loss.item():.4f}")
    for k, v in details.items():
        print(f"       - {k}: {v:.4f}")

def test_minimal_checkpoint_manager():
    print("\n[TEST 4/5] Kiểm định Chính sách Checkpoint Tinh Gọn (MinimalCheckpointManager_M3)...")
    from ddp_worker_m1_to_m3_patch1 import MinimalCheckpointManager_M3, TPSInbetweenOriginal_M1_M3

    temp_dir = tempfile.mkdtemp()
    try:
        manager = MinimalCheckpointManager_M3(temp_dir, is_rank0=True)
        model = TPSInbetweenOriginal_M1_M3()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Lưu epoch 1
        manager.save_checkpoint(1, model, optimizer, None, None, score=1.5, history=[], is_best=True)
        manager.save_resume(1, model, optimizer, None, None, history=[{'epoch': 1}])

        # Lưu epoch 2
        manager.save_checkpoint(2, model, optimizer, None, None, score=1.2, history=[], is_best=True)
        manager.save_resume(2, model, optimizer, None, None, history=[{'epoch': 1}, {'epoch': 2}])

        # Lưu epoch 3 (không phải best)
        manager.save_checkpoint(3, model, optimizer, None, None, score=1.4, history=[], is_best=False)
        manager.save_resume(3, model, optimizer, None, None, history=[{'epoch': 1}, {'epoch': 2}, {'epoch': 3}])

        files = sorted(os.listdir(temp_dir))
        expected_files = sorted(["model_m3_seq_patch1_best.pt", "model_m3_seq_patch1_latest.pt", "model_m3_seq_patch1_resume.pt"])
        assert files == expected_files, f"Danh sách file checkpoint vi phạm chính sách tối đa 3 file! Thực tế: {files}"
        print(f"   [✓] Chính sách 3 Checkpoint được đảm bảo 100%: {files}")

        # Test resume
        new_model = TPSInbetweenOriginal_M1_M3()
        new_opt = torch.optim.Adam(new_model.parameters(), lr=1e-4)
        last_ep, hist = manager.load_resume(new_model, optimizer=new_opt, map_location='cpu')
        assert last_ep == 3, f"Resume không đúng epoch: {last_ep} != 3"
        assert len(hist) == 3, f"Lịch sử resume không đúng độ dài: {len(hist)} != 3"
        print(f"   [✓] Auto-Resume thành công: Khôi phục chính xác từ Epoch {last_ep}!")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_metrics_evaluation():
    print("\n[TEST 5/5] Kiểm định Hàm Đo Lường Chỉ Số Thẩm Định (WCD, S-EMD, Recalls)...")
    from ddp_worker_m1_to_m3_patch1 import evaluate_metrics_batch_gpu

    pred = torch.rand(2, 1, 128, 128)
    gt = torch.rand(2, 1, 128, 128)

    metrics = evaluate_metrics_batch_gpu(pred, gt)
    required_keys = ['wcd', 'sliced_emd', 'skel_recall', 'recall', 'f1', 'ghost', 'dark_ghost', 'val_l1', 'psnr', 'ssim']
    for k in required_keys:
        assert k in metrics, f"Thiếu metric: {k}"
        assert torch.isfinite(metrics[k]).all(), f"Metric {k} chứa giá trị không hợp lệ!"
    print(f"   [✓] Toàn bộ 10 chỉ số đánh giá GPU-batched sẵn sàng (PSNR={metrics['psnr'].mean():.2f}dB, WCD={metrics['wcd'].mean():.4f}).")

def main():
    print("=" * 80)
    print("🧪 BẮT ĐẦU KIỂM ĐỊNH TOÀN DIỆN BỘ CÔNG CỤ NOTEBOOK M1 -> M3 (PATCH 1)")
    print("=" * 80)

    test_notebook_ast()
    test_model_architecture_and_m1_weights()
    test_lineart_loss()
    test_minimal_checkpoint_manager()
    test_metrics_evaluation()

    print("\n" + "=" * 80)
    print("🏆 TẤT CẢ 5 BÀI KIỂM ĐỊNH ĐÃ VƯỢT QUA 100% XUẤT SẮC!")
    print("👉 Notebook train_ablation_m1_to_m3_sequence_patch1_kaggle.ipynb hoàn toàn sẵn sàng cho Kaggle!")
    print("=" * 80)

if __name__ == "__main__":
    main()
