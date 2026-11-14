#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_v30_notebook.py
Script tự động khởi tạo notebook huấn luyện Kaggle V3.0 hoàn toàn mới từ đầu:
- kaggle/finetune-v3-0.ipynb
- kaggle/07_train_tps_inbetween_finetune_v3.0.ipynb

Cấu trúc chuẩn 14 Interactive Cells (1 Markdown + 13 Direct Python Code Cells):
- Cell 00 (Markdown): Tiêu đề, tài liệu kỹ thuật, mục tiêu & thông số V3.0 Golden 540p.
- Cell 01 (Code): Hardware, PyTorch Setup, posix_fadvise, malloc_trim & Memory Purge.
- Cell 02 (Code): Golden Dataset 540p Discovery & Verification.
- Cell 03 (Code): Thin-Plate Spline (TPS) Deformation, Anchors & Topological Filtering.
- Cell 04 (Code): SakugaGoldenDataset_v3 (.npy uncompressed mmap zero-copy, drop_file_page_cache, 512 crop).
- Cell 05 (Code): Architecture TPSInbetweenFinetune_v3 (Kaiming Normal PReLU & std=1e-3 Output Layers).
- Cell 06 (Code): CompositeLossEngine_v30 (2-Stage Curriculum, Gold 1.10x, HairTipLoss, Cleanliness Penalty).
- Cell 07 (Code): Warm-Start Loader (filter out asf_head.*) & ParallelTrainingWrapper_v3.
- Cell 08 (Code): Pure GPU Batched Validation Engine (unpad 544 to 540x960, Pure FP32).
- Cell 09 (Code): SmartCheckpointManager_v30 with Hard Convergence Gates & Best Weights Tracker.
- Cell 10 (Code): Full Training Loop (25 Epochs, Batch 8 + Accum 2 = 16, 2-Stage Schedule, Float Loss).
- Cell 11 (Code): Multi-metric Convergence Dashboard (6 Subplots).
- Cell 12 (Code): Qualitative Visual Audit Display (Side-by-side Input / GT / Warped / Blended).
- Cell 13 (Code): Artifact Packaging & Direct Download FileLink.
"""

import os
import sys
import json
import ast
import uuid

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
    # Loại bỏ dòng trống thừa ở cuối nếu có
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def build_notebook():
    cells_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v30_cells")
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

        # Ô Markdown hướng dẫn Reset Kernel đặt ngay dưới Cell 10 (Training)
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
    print("[*] Đang đọc 14 cells từ kaggle/v30_cells/...")
    nb = build_notebook()
    print(f"[OK] Đã xây dựng thành công notebook với {len(nb['cells'])} cells (1 Markdown, 13 Code).")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_files = [
        os.path.join(base_dir, "finetune-v3-0.ipynb"),
        os.path.join(base_dir, "07_train_tps_inbetween_finetune_v3.0.ipynb")
    ]

    for p in out_files:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        size_kb = os.path.getsize(p) / 1024
        print(f"[OK] Ghi thành công {os.path.basename(p)} ({size_kb:.1f} KB, {len(nb['cells'])} cells).")

    print("[DONE] Hoàn tất khởi tạo notebook v3.0!")


if __name__ == "__main__":
    main()
