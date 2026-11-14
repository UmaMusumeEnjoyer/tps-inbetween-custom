#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_v31_notebook.py
Script tự động khởi tạo notebook huấn luyện Kaggle V3.1 hoàn toàn mới:
- kaggle/finetune-v3-1.ipynb
- kaggle/08_train_tps_inbetween_finetune_v3.1.ipynb

Cấu trúc chuẩn 14 Interactive Cells (1 Markdown + 13 Python Code Cells + 1 Reset Markdown):
- Cell 00 (Markdown): Tiêu đề, mục tiêu, thông số đột phá kỹ thuật V3.1.
- Cell 01 (Code): Hardware, PyTorch Setup, posix_fadvise, malloc_trim & Memory Purge.
- Cell 02 (Code): Golden Dataset 540p (V1_Patch3) Discovery & Verification.
- Cell 03 (Code): Thin-Plate Spline (TPS) Deformation, Anchors & Vectorized Batch Flow Solver.
- Cell 04 (Code): SakugaGoldenDataset_v31 (Full Precomputed TPS Cache 0-copy, Batch 8 + Accum 2).
- Cell 05 (Code): Architecture TPSInbetweenFinetune_v31 (SynthesisUnet V2.6 Gated Head, Positive Bias +2.0, Lightweight Propagator).
- Cell 06 (Code): CompositeLossEngine_v31 (Fast Laplacian Edge, clDice iters=2, Gold 1.10x, Safe Ghost 11x11).
- Cell 07 (Code): Warm-Start Loader from V2.6 Gold Baseline & ParallelTrainingWrapper_v31.
- Cell 08 (Code): Pure GPU Batched Validation Engine (unpad 544 to 540x960, Pure FP32).
- Cell 09 (Code): SmartCheckpointManager_v31 with Hard Convergence Gates & Candidate Model Tracker.
- Cell 10 (Code): Full Training Loop (25 Epochs, Training Time < 10 Min/Epoch, Watchdog RAM, Resume).
- Cell 10 Note (Markdown): Hướng dẫn reset kernel bằng os._exit(0) nếu cần.
- Cell 11 (Code): Multi-metric Convergence Dashboard (6 Subplots).
- Cell 12 (Code): Qualitative Visual Audit Display (Sample Images).
- Cell 13 (Code): Automated Model Soup Generation & Artifact Packaging FileLink.
"""

import os
import sys
import json
import ast

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


def to_notebook_lines(text: str):
    """Chia text thành danh sách dòng có ký tự kết thúc \\n theo chuẩn Jupyter Notebook."""
    raw_lines = text.split("\n")
    lines = [line + "\n" for line in raw_lines[:-1]]
    if raw_lines:
        lines.append(raw_lines[-1])
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def build_notebook():
    cells_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v31_cells")
    if not os.path.isdir(cells_dir):
        raise FileNotFoundError(f"Thư mục chứa cells không tồn tại: {cells_dir}")

    cells = []

    # 1. Cell 00: Markdown
    md_path = os.path.join(cells_dir, "cell_00.md")
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
    cells.append({
        "cell_type": "markdown",
        "id": "c00_header",
        "metadata": {},
        "source": to_notebook_lines(md_content)
    })

    # 2. Cells 01 -> 13: Python Code Cells
    for i in range(1, 14):
        code_path = os.path.join(cells_dir, f"cell_{i:02d}.py")
        if not os.path.exists(code_path):
            raise FileNotFoundError(f"Thiếu file code cell: {code_path}")
        with open(code_path, "r", encoding="utf-8") as f:
            code_content = f.read()

        # Tự động đóng gói ddp_worker_v31.py vào Cell 10 để notebook hoàn toàn tự hành trên Kaggle
        if i == 10:
            worker_path = os.path.join(cells_dir, "ddp_worker_v31.py")
            if os.path.exists(worker_path):
                with open(worker_path, "r", encoding="utf-8") as wf:
                    worker_raw = wf.read()
                worker_repr = repr(worker_raw)
                injection_block = (
                    f"# Tự động sinh file worker DDP độc lập để torchrun nproc_per_node={'{NUM_GPUS}'} thực thi\n"
                    f"DDP_WORKER_CODE = {worker_repr}\n"
                    f"with open(DDP_WORKER_PATH, 'w', encoding='utf-8') as f:\n"
                    f"    f.write(DDP_WORKER_CODE)\n"
                )
                code_content = code_content.replace("# __DDP_WORKER_INJECTION_POINT__", injection_block)

        # Kiểm tra cú pháp bằng AST parser
        try:
            ast.parse(code_content, filename=f"cell_{i:02d}.py")
        except SyntaxError as e:
            raise SyntaxError(f"Lỗi cú pháp tại cell_{i:02d}.py: {e}")

        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "id": f"c{i:02d}_code",
            "metadata": {
                "trusted": True
            },
            "outputs": [],
            "source": to_notebook_lines(code_content)
        })

        # Đặt Markdown hướng dẫn Reset Kernel ngay dưới Cell 10 (Training)
        if i == 10:
            md_reset_path = os.path.join(cells_dir, "cell_10_reset_note.md")
            if os.path.exists(md_reset_path):
                with open(md_reset_path, "r", encoding="utf-8") as f:
                    md_reset_content = f.read()
                cells.append({
                    "cell_type": "markdown",
                    "id": "c10_reset_note",
                    "metadata": {},
                    "source": to_notebook_lines(md_reset_content)
                })

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    return nb


def main():
    print("[*] Đang đọc các cell từ kaggle/v31_cells/...")
    nb = build_notebook()
    print(f"[OK] Đã xây dựng thành công notebook với {len(nb['cells'])} cells (2 Markdown, 13 Code).")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_files = [
        os.path.join(base_dir, "finetune-v3-1.ipynb"),
        os.path.join(base_dir, "08_train_tps_inbetween_finetune_v3.1.ipynb")
    ]

    for p in out_files:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        size_kb = os.path.getsize(p) / 1024
        print(f"[OK] Ghi thành công {os.path.basename(p)} ({size_kb:.1f} KB, {len(nb['cells'])} cells).")

    print("[DONE] Hoàn tất khởi tạo notebook v3.1!")


if __name__ == "__main__":
    main()
