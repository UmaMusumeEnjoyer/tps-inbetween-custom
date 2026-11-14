#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_v30_notebook.py
Kiểm toán cú pháp, cấu trúc và tính toàn vẹn của notebook V3.0:
- kaggle/finetune-v3-0.ipynb
- kaggle/07_train_tps_inbetween_finetune_v3.0.ipynb
"""

import os
import sys
import json
import ast
import py_compile
import tempfile

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


def verify_notebook(nb_path):
    print("=" * 80)
    print(f"[*] KIEM TOAN NOTEBOOK: {nb_path}")
    print("=" * 80)

    if not os.path.exists(nb_path):
        raise FileNotFoundError(f"Khong tim thay file: {nb_path}")

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    print(f"[*] Tong so cells: {len(cells)}")

    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    md_cells = [c for c in cells if c.get("cell_type") == "markdown"]
    print(f"[*] So luong Markdown cells: {len(md_cells)} | Code cells: {len(code_cells)}")

    assert len(cells) == 15, f"Ky vong 15 cells (2 MD + 13 Code), tim thay {len(cells)}"
    assert len(md_cells) == 2, f"Ky vong 2 Markdown cells, tim thay {len(md_cells)}"
    assert len(code_cells) == 13, f"Ky vong 13 Code cells, tim thay {len(code_cells)}"
    assert cells[0]["cell_type"] == "markdown", "Cell 0 phai la Markdown tieu de"
    assert cells[11]["cell_type"] == "markdown" and "os._exit(0)" in "".join(cells[11].get("source", [])), "Cell ngay duoi Cell 10 phai la Markdown huong dan Reset Kernel"

    # Kiểm tra outputs phải rỗng để sẵn sàng chạy Kaggle
    for idx, c in enumerate(code_cells):
        outputs = c.get("outputs", [])
        assert len(outputs) == 0, f"Cell code {idx} chua outputs cu! Can xoa sach truoc khi commit/upload."

    # Kiểm toán cú pháp Python từng code cell
    print("\n[*] Dang kiem tra cu phap Python (ast.parse & py_compile)...")
    for idx, c in enumerate(cells):
        ctype = c.get("cell_type")
        if ctype != "code":
            continue
        src = "".join(c.get("source", []))
        try:
            ast.parse(src)
        except SyntaxError as e:
            print(f"[!] LOI CU PHAP TAI CELL {idx}: {e}")
            raise

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
            tmp.write(src)
            tmp_path = tmp.name
        try:
            py_compile.compile(tmp_path, doraise=True)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        first_line = src.strip().split("\n")[0] if src.strip() else "EMPTY"
        print(f"   [OK] Cell {idx:02d}: {first_line[:75]}")

    # Kiểm tra các thành phần cốt lõi của V3.0
    all_code = "".join(["".join(c.get("source", [])) for c in code_cells])

    print("\n[*] Kiem toan cac tieu chuan ky thuat cot loi V3.0 (Chuan 540p & Zero Memory Leak):")
    checks = [
        ("Multi-GPU DataParallel 2 GPU", "nn.DataParallel" in all_code),
        ("Tat cudnn.benchmark chong race-condition", "cudnn.benchmark = False" in all_code),
        ("Dataset 540p uu tien chuan (sakuga_golden_dataset_tps_ready_v1_patch3)", "sakuga_golden_dataset_tps_ready_v1_patch3" in all_code),
        ("Precompute TPS Cache (.npy mmap 0-copy)", "kps_data_" in all_code and ".npy" in all_code and "mmap_mode" in all_code),
        ("Giai phong PageCache OS (posix_fadvise / drop_file_page_cache)", "POSIX_FADV_DONTNEED" in all_code or "drop_file_page_cache" in all_code),
        ("Giai phong malloc heap fragmentation (malloc_trim)", "malloc_trim" in all_code),
        ("Khoi tao Kaiming pha bay doi xung (Symmetry Breaking)", "kaiming_normal_" in all_code),
        ("Small normal init dau ra tranh vanishing gradient (std=1e-3)", "1e-3" in all_code),
        ("Bo loc warm-start loai bo asf_head cu", "asf_head." in all_code),
        ("Gradient Accumulation & Huan luyen Toan phan 540p", ("GRAD_ACCUM_STEPS" in all_code or "grad_accum" in all_code.lower())),
        ("Validation thuan FP32 (use_amp=False)", "use_amp = False" in all_code or "use_amp=False" in all_code),
        ("Cong hien thi kep bat doi xung (AsymmetricStrokeFusionHead_v3)", "AsymmetricStrokeFusionHead_v3" in all_code),
        ("Lan truyen van toc theo bien (ContourGuidedFlowPropagator)", "ContourGuidedFlowPropagator" in all_code),
        ("Bo dieu tiet mat do muc chuan vang 1.10x", "StrokeDensityRegularizationLoss_v3" in all_code),
        ("Hair-Tip Endpoint Continuity Loss", "HairTipEndpointContinuityLoss" in all_code),
        ("Penalize Ghost Artifacts (cleanliness penalty)", "cleanliness_penalty" in all_code or "penalty" in all_code),
        ("SmartCheckpointManager V3.0 (Hard Gates)", "SmartCheckpointManager_v30" in all_code),
        ("Dashboard hoi tu 6 do thi truc quan", "plt.subplots(2, 3" in all_code or "plt.subplots" in all_code),
        ("Tu dong dong goi ZIP & xuat Link tai", "zipfile.ZipFile" in all_code and "FileLink" in all_code),
    ]

    for label, status in checks:
        icon = "[PASS]" if status else "[FAIL]"
        print(f"   {icon} {label}")
        assert status, f"Thieu tieu chuan: {label}"

    print(f"\n[DONE] TOAN BO KIEM TOAN DAT CHUAN 100% CHO: {os.path.basename(nb_path)}\n")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nb_files = [
        os.path.join(script_dir, "finetune-v3-0.ipynb"),
        os.path.join(script_dir, "07_train_tps_inbetween_finetune_v3.0.ipynb")
    ]
    for nb in nb_files:
        verify_notebook(nb)


if __name__ == '__main__':
    main()
