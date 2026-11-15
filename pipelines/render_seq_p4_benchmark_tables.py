#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render_seq_p4_benchmark_tables.py
--------------------------------------------------------------------------------
Vẽ các bảng chỉ số benchmark đối đầu dưới dạng hình ảnh chất lượng cao (High DPI):
    1. Table 1: Case 1 - Chuỗi 24 frames thực tế (17 in-betweens)
    2. Table 2: Cases 2, 3, 4 - Defense Geometry Suite tại t=0.50
    3. Table 3: Toàn bộ Cases 2, 3, 4 phân rã theo t=0.25, 0.50, 0.75
    4. Master Dashboard: Ghép tổng thể tất cả các bảng vào một dashboard trực quan

Quy ước trực quan:
    - [↑]: Cao hơn là tốt hơn
    - [↓]: Thấp hơn là tốt hơn
    - [≈1.0]: Tiệm cận 1.000 là tốt hơn
    - Ô có chỉ số tốt nhất trên từng cột được tô màu xanh lá nhạt (#dcfce7) với chữ đậm (#15803d).
--------------------------------------------------------------------------------
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
BENCH_DIR = os.path.join(REPO_ROOT, "bench_outputs_seq_p4_occlusion")
OUT_IMG_DIR = os.path.join(BENCH_DIR, "table_images")
os.makedirs(OUT_IMG_DIR, exist_ok=True)

# Thiết lập font an toàn đa nền tảng
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Segoe UI', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def find_best_idx(values, direction):
    """
    Tìm index của giá trị tốt nhất theo hướng tối ưu:
    - direction == 'up': max
    - direction == 'down': min
    - direction == 'near_one': min(abs(v - 1.0))
    """
    arr = np.array(values, dtype=float)
    if direction == 'up':
        return int(np.nanargmax(arr))
    elif direction == 'down':
        return int(np.nanargmin(arr))
    elif direction == 'near_one':
        diff = np.abs(arr - 1.0)
        return int(np.nanargmin(diff))
    return -1


def draw_styled_table(
    title,
    subtitle,
    headers,
    col_widths,
    rows_data,
    col_directions,
    out_path,
    figsize=(14, 5),
    dpi=220,
    header_bg="#1e293b",
    header_fg="#ffffff",
    row_bg_even="#ffffff",
    row_bg_odd="#f8fafc",
    best_bg="#dcfce7",
    best_fg="#15803d",
    group_col_idx=None
):
    """
    Vẽ bảng chỉ số dạng ảnh đồ họa sắc nét với Matplotlib.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.axis('off')

    num_rows = len(rows_data)
    num_cols = len(headers)

    # Tính toán tọa độ Y
    # Header ở trên cùng, các rows trải đều xuống dưới
    top_y = 0.82
    row_height = (top_y - 0.12) / max(1, num_rows + 1)
    header_y = top_y

    # Vẽ Title & Subtitle
    fig.text(0.04, 0.94, title, fontsize=15, fontweight='bold', color="#0f172a", ha='left', va='top')
    fig.text(0.04, 0.88, subtitle, fontsize=10, color="#475569", ha='left', va='top')

    # Vẽ Legend chỉ dẫn ở góc phải trên
    legend_str = "[↑] Cao hơn là tốt hơn    [↓] Thấp hơn là tốt hơn    [≈1.0] Tiệm cận 1.0 là tốt hơn    |    ■ Ô nền xanh lá = Giá trị tốt nhất"
    fig.text(0.96, 0.88, legend_str, fontsize=8.5, color="#059669", fontweight='semibold', ha='right', va='top')

    # Chuẩn bị tọa độ X các cột
    total_w = sum(col_widths)
    norm_widths = [w / total_w * 0.92 for w in col_widths]
    x_starts = [0.04]
    for w in norm_widths[:-1]:
        x_starts.append(x_starts[-1] + w)

    # 1. Vẽ Header
    for j in range(num_cols):
        rect = patches.FancyBboxPatch(
            (x_starts[j], header_y - row_height),
            norm_widths[j],
            row_height,
            boxstyle="square,pad=0",
            ec="#cbd5e1",
            fc=header_bg,
            lw=1.0
        )
        ax.add_patch(rect)
        hdr_text = headers[j]
        # Canh lề trái cho cột tên phiên bản/nhóm, canh giữa cho số liệu
        ha = 'left' if j < (2 if group_col_idx is not None else 1) else 'center'
        tx = x_starts[j] + 0.015 if ha == 'left' else x_starts[j] + norm_widths[j] / 2.0
        ax.text(
            tx,
            header_y - row_height / 2.0,
            hdr_text,
            fontsize=9.5,
            fontweight='bold',
            color=header_fg,
            ha=ha,
            va='center'
        )

    # Xác định các ô tốt nhất theo từng cột số liệu
    best_cells = set()
    start_metric_col = 2 if group_col_idx is not None else 1

    if group_col_idx is not None:
        # Nhóm theo từng Case
        groups = {}
        for r_i, r in enumerate(rows_data):
            g_key = r[group_col_idx]
            groups.setdefault(g_key, []).append(r_i)

        for g_key, r_indices in groups.items():
            for c_j in range(start_metric_col, num_cols):
                dir_type = col_directions[c_j]
                if dir_type is None:
                    continue
                vals = [float(rows_data[ri][c_j].replace(',', '').replace('%', '')) for ri in r_indices]
                b_idx = find_best_idx(vals, dir_type)
                if b_idx >= 0:
                    best_cells.add((r_indices[b_idx], c_j))
    else:
        for c_j in range(start_metric_col, num_cols):
            dir_type = col_directions[c_j]
            if dir_type is None:
                continue
            vals = [float(rows_data[ri][c_j].replace(',', '').replace('%', '')) for ri in range(num_rows)]
            b_idx = find_best_idx(vals, dir_type)
            if b_idx >= 0:
                best_cells.add((b_idx, c_j))

    # 2. Vẽ Data Rows
    for i in range(num_rows):
        cur_y = header_y - (i + 2) * row_height
        is_even = (i % 2 == 0)
        row_def_bg = row_bg_even if is_even else row_bg_odd

        for j in range(num_cols):
            is_best = (i, j) in best_cells
            cell_bg = best_bg if is_best else row_def_bg
            cell_fg = best_fg if is_best else "#1e293b"
            cell_weight = 'bold' if is_best else 'normal'

            rect = patches.FancyBboxPatch(
                (x_starts[j], cur_y),
                norm_widths[j],
                row_height,
                boxstyle="square,pad=0",
                ec="#e2e8f0",
                fc=cell_bg,
                lw=0.8
            )
            ax.add_patch(rect)

            cell_val = str(rows_data[i][j])
            ha = 'left' if j < start_metric_col else 'center'
            tx = x_starts[j] + 0.015 if ha == 'left' else x_starts[j] + norm_widths[j] / 2.0
            ax.text(
                tx,
                cur_y + row_height / 2.0,
                cell_val,
                fontsize=9.2,
                fontweight=cell_weight,
                color=cell_fg,
                ha=ha,
                va='center'
            )

    # Footer note
    fig.text(0.04, 0.04, "Antigravity Sequence Inbetweening Benchmark Suite  |  Verified outputs at: bench_outputs_seq_p4_occlusion/", fontsize=8, color="#94a3b8", ha='left', va='bottom')

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  [✓] Đã xuất ảnh bảng chỉ số: {out_path}")


# ==============================================================================
# 1. VẼ BẢNG CASE 1: LINEART TEST 24 FRAMES
# ==============================================================================
def render_case1_table():
    csv_path = os.path.join(BENCH_DIR, "summary_case1_lineart.csv")
    df = pd.read_csv(csv_path)

    title = "CASE 1: BẢNG SO SÁNH CHUỖI LINEART THỰC TẾ 24 FRAMES"
    subtitle = "Giá trị trung bình trên toàn bộ 17 frames trung gian (data/lineart_experiments/test)"

    headers = [
        "Phiên Bản Mô Hình",
        "PSNR (dB)\n[↑]",
        "SSIM\n[↑]",
        "Skeleton Recall\n[↑]",
        "Tip Recall\n[↑]",
        "F1 Score\n[↑]",
        "WCD_DoG (px)\n[↓]",
        "Ghost Pixels\n[↓]",
        "IE_MAE\n[↓]",
        "Density Ratio\n[≈1.0]"
    ]
    col_widths = [3.2, 1.4, 1.3, 1.7, 1.5, 1.3, 1.6, 1.6, 1.4, 1.6]
    col_directions = [None, 'up', 'up', 'up', 'up', 'up', 'down', 'down', 'down', 'near_one']

    rows_data = []
    for _, row in df.iterrows():
        rows_data.append([
            str(row.iloc[0]),
            f"{float(row.iloc[1]):.4f}",
            f"{float(row.iloc[2]):.4f}",
            f"{float(row.iloc[3]) * 100.0:.2f}%" if float(row.iloc[3]) <= 1.0 else f"{float(row.iloc[3]):.2f}%",
            f"{float(row.iloc[4]) * 100.0:.2f}%" if float(row.iloc[4]) <= 1.0 else f"{float(row.iloc[4]):.2f}%",
            f"{float(row.iloc[5]):.4f}",
            f"{float(row.iloc[6]):.4f}",
            f"{float(row.iloc[7]):,.1f}",
            f"{float(row.iloc[8]):.4f}",
            f"{float(row.iloc[9]):.4f}"
        ])

    out_file = os.path.join(OUT_IMG_DIR, "table_case1_lineart_summary.png")
    draw_styled_table(
        title, subtitle, headers, col_widths, rows_data, col_directions,
        out_file, figsize=(15, 3.8), dpi=220
    )


# ==============================================================================
# 2. VẼ BẢNG CASES 2, 3, 4: DEFENSE GEOMETRY TẠI t=0.50
# ==============================================================================
def render_geometry_t05_table():
    csv_path = os.path.join(BENCH_DIR, "summary_cases2_3_4_geometry_t05.csv")
    df = pd.read_csv(csv_path)

    title = "CASES 2, 3, 4: DEFENSE GEOMETRY SUITE TẠI THỜI ĐIỂM t=0.50"
    subtitle = "So sánh đối đầu tại vị trí biến dạng và chuyển động 360px cực đại (Mid-point Deformation)"

    headers = [
        "Bài Test Hình Học (Case)",
        "Phiên Bản",
        "PSNR (dB)\n[↑]",
        "SSIM\n[↑]",
        "Exact Skel Rec\n[↑]",
        "Exact Density\n[≈1.0]",
        "WCD_DoG (px)\n[↓]",
        "Ghost Px\n[↓]",
        "IE_MAE\n[↓]"
    ]
    col_widths = [3.4, 2.8, 1.4, 1.3, 1.6, 1.6, 1.6, 1.4, 1.4]
    col_directions = [None, None, 'up', 'up', 'up', 'near_one', 'down', 'down', 'down']

    rows_data = []
    for _, row in df.iterrows():
        c_name = str(row.iloc[0]).replace("Case 2: ", "").replace("Case 3: ", "").replace("Case 4: ", "").replace(" (t=0.50)", "")
        v_name = str(row.iloc[1])
        rows_data.append([
            c_name,
            v_name,
            f"{float(row.iloc[2]):.4f}",
            f"{float(row.iloc[3]):.4f}",
            f"{float(row.iloc[4]):.2f}%",
            f"{float(row.iloc[5]):.4f}",
            f"{float(row.iloc[6]):.4f}",
            f"{float(row.iloc[7]):,.1f}",
            f"{float(row.iloc[8]):.4f}"
        ])

    out_file = os.path.join(OUT_IMG_DIR, "table_cases2_3_4_geometry_t05.png")
    draw_styled_table(
        title, subtitle, headers, col_widths, rows_data, col_directions,
        out_file, figsize=(15.5, 6.2), dpi=220, group_col_idx=0
    )


# ==============================================================================
# 3. VẼ BẢNG CHI TIẾT TẤT CẢ CÁC MỐC t=0.25, 0.50, 0.75 CHO TỪNG CASE
# ==============================================================================
def render_full_geometry_breakdown():
    cases_files = [
        ("Case 2: Deforming Circle", os.path.join(BENCH_DIR, "case2_deforming_circle", "case2_deforming_circle_metrics.csv")),
        ("Case 3: Deforming Hollow Cross", os.path.join(BENCH_DIR, "case3_deforming_cross", "case3_deforming_cross_metrics.csv")),
        ("Case 4: Deforming Solid Cross", os.path.join(BENCH_DIR, "case4_deforming_solid_cross", "case4_deforming_solid_cross_metrics.csv")),
    ]

    title = "DEFENSE GEOMETRY SUITE: PHÂN RÃ CHI TIẾT THEO CÁC MỐC THỜI GIAN t"
    subtitle = "Đánh giá chi tiết tại t=0.25 (gần I0), t=0.50 (giữa hành trình), t=0.75 (gần I1)"

    headers = [
        "Bài Test & Thời Điểm t",
        "Phiên Bản",
        "PSNR (dB)\n[↑]",
        "SSIM\n[↑]",
        "Exact Skel Rec\n[↑]",
        "Exact Density\n[≈1.0]",
        "WCD_DoG (px)\n[↓]",
        "Ghost Px\n[↓]",
        "IE_MAE\n[↓]"
    ]
    col_widths = [3.6, 2.8, 1.4, 1.3, 1.6, 1.6, 1.6, 1.4, 1.4]
    col_directions = [None, None, 'up', 'up', 'up', 'near_one', 'down', 'down', 'down']

    rows_data = []
    versions = [
        ("Pha 3 Remake (Newpipeline)", "Pha3_Remake_Sharp"),
        ("Pha 4 Occlusion Head (Raw)", "Pha4_Occlusion_Raw"),
        ("Pha 4 Occlusion Head + Sharp", "Pha4_Occlusion_Sharp"),
    ]

    for c_title, c_path in cases_files:
        df = pd.read_csv(c_path)
        for tv in [0.25, 0.50, 0.75]:
            sub = df[df["t_val"] == tv].iloc[0]
            for v_label, v_prefix in versions:
                group_id = f"{c_title} (t={tv:.2f})"
                rows_data.append([
                    group_id,
                    v_label,
                    f"{float(sub[f'{v_prefix}_psnr']):.4f}",
                    f"{float(sub[f'{v_prefix}_ssim']):.4f}",
                    f"{float(sub[f'{v_prefix}_exact_skel_recall']):.2f}%",
                    f"{float(sub[f'{v_prefix}_exact_density']):.4f}",
                    f"{float(sub[f'{v_prefix}_wcd_dog']):.4f}",
                    f"{float(sub[f'{v_prefix}_ghost']):,.1f}",
                    f"{float(sub[f'{v_prefix}_ie_mae']):.4f}"
                ])

    out_file = os.path.join(OUT_IMG_DIR, "table_geometry_all_t_breakdown.png")
    draw_styled_table(
        title, subtitle, headers, col_widths, rows_data, col_directions,
        out_file, figsize=(16.5, 14.5), dpi=220, group_col_idx=0
    )


# ==============================================================================
# 4. MASTER DASHBOARD: GHÉP TỔNG HỢP CASE 1 VÀ CASES 2,3,4 THÀNH 1 ẢNH DUY NHẤT
# ==============================================================================
def render_master_dashboard():
    img1_path = os.path.join(OUT_IMG_DIR, "table_case1_lineart_summary.png")
    img2_path = os.path.join(OUT_IMG_DIR, "table_cases2_3_4_geometry_t05.png")

    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        return

    import cv2
    im1 = cv2.imread(img1_path)
    im2 = cv2.imread(img2_path)

    w_target = max(im1.shape[1], im2.shape[1])
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]

    # Resize để 2 bảng khớp chiều rộng
    im1_scaled = cv2.resize(im1, (w_target, int(h1 * (w_target / w1))), interpolation=cv2.INTER_LANCZOS4)
    im2_scaled = cv2.resize(im2, (w_target, int(h2 * (w_target / w2))), interpolation=cv2.INTER_LANCZOS4)

    # Thêm separator bar ở giữa
    sep = np.full((32, w_target, 3), 245, dtype=np.uint8)
    cv2.putText(sep, "------------------------------------------------------------------------------------------------------------------------------------------", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    dashboard = np.vstack([im1_scaled, sep, im2_scaled])
    out_dash = os.path.join(OUT_IMG_DIR, "table_master_benchmark_dashboard.png")
    cv2.imwrite(out_dash, dashboard)
    print(f"  [✓] Đã xuất ảnh Master Dashboard: {out_dash}")


def main():
    print("=" * 80)
    print("🎨 TIẾN HÀNH XUẤT CÁC BẢNG CHỈ SỐ BENCHMARK THÀNH HÌNH ẢNH")
    print(f"   Thư mục lưu ảnh: {OUT_IMG_DIR}")
    print("=" * 80)

    render_case1_table()
    render_geometry_t05_table()
    render_full_geometry_breakdown()
    render_master_dashboard()

    print("\n[✓] HOÀN TẤT XUẤT TOÀN BỘ ẢNH BẢNG BIỂU SO SÁNH.")


if __name__ == "__main__":
    main()
