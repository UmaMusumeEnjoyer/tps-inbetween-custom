#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import ast
import sys

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
nb_path = os.path.join(SCRIPT_DIR, "triplet_04_v1.0_golden_v1_kaggle.ipynb")

print(f"[*] Đang thẩm định file notebook: {nb_path}")

try:
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print(f"[✓] Đọc JSON thành công! Tổng số cells: {len(nb['cells'])}")
except Exception as e:
    print(f"[❌] Lỗi đọc JSON: {e}")
    sys.exit(1)

code_cells = [c for c in nb['cells'] if c.get('cell_type') == 'code']
print(f"[*] Số lượng code cells: {len(code_cells)}")

errors = []
for idx, cell in enumerate(code_cells, start=1):
    source = "".join(cell.get('source', []))
    try:
        ast.parse(source)
        print(f"    - Code Cell {idx:02d}: Cú pháp AST hợp lệ [OK]")
    except SyntaxError as se:
        print(f"    - Code Cell {idx:02d}: LỖI CÚ PHÁP: {se}")
        errors.append((idx, str(se)))

if errors:
    print(f"\n[❌] Có {len(errors)} cell bị lỗi cú pháp!")
    sys.exit(1)
else:
    print("\n🎉 100% TẤT CẢ CODE CELLS CÓ CÚ PHÁP PYTHON HỢP LỆ TUYỆT ĐỐI!")
