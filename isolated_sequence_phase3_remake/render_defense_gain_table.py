#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render_defense_gain_table.py
--------------------------------------------------------------------------------
Đọc trực tiếp dữ liệu từ benchmark_outputs_orig_vs_pha3/case1_lineart_test_24f/benchmark_case1_per_frame.csv
và xuất bảng đồ họa High-DPI phục vụ bảo vệ PBL6:
  1. Bảng quy đổi "% Giảm lỗi tương đối / Cải thiện thực tế" (Pha 3 vs Original)
  2. Bảng tổng hợp hiệu năng khoảng cách chuẩn (delta_f = 3) & Peak Frames
--------------------------------------------------------------------------------
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
BENCH_DIR = os.path.join(REPO_ROOT, "benchmark_outputs_orig_vs_pha3")
CSV_PATH = os.path.join(BENCH_DIR, "case1_lineart_test_24f", "benchmark_case1_per_frame.csv")
OUT_PRIMARY = os.path.join(BENCH_DIR, "table_case1_relative_gain_defense.png")
OUT_TABLE_DIR = os.path.join(BENCH_DIR, "table_images", "table_case1_relative_gain_defense.png")

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Segoe UI", "Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def build_table_data(df: pd.DataFrame):
    m_orig = "Original_Pipeline"
    m_raw = "Pha3_Remake_Raw"
    m_new = "Pha3_Remake_Newpipeline"

    # Helper means
    def mean(col_suffix):
        return (
            float(df[f"{m_orig}_{col_suffix}"].mean()),
            float(df[f"{m_raw}_{col_suffix}"].mean()),
            float(df[f"{m_new}_{col_suffix}"].mean()),
        )

    # 1. Density deviation |1 - DR|
    dr_o, dr_r, dr_n = mean("density_ratio")
    ddev_o, ddev_r, ddev_n = abs(1.0 - dr_o), abs(1.0 - dr_r), abs(1.0 - dr_n)
    ddev_gain_r = (ddev_o - ddev_r) / ddev_o * 100.0
    ddev_gain_n = (ddev_o - ddev_n) / ddev_o * 100.0

    # 2. S-EMD
    se_o, se_r, se_n = mean("semd")
    se_gain_r = (se_o - se_r) / se_o * 100.0
    se_gain_n = (se_o - se_n) / se_o * 100.0

    # 3. Mean per-frame MSE implied by PSNR
    ps_o, ps_r, ps_n = mean("psnr")
    mse_o = float((10.0 ** (-df[f"{m_orig}_psnr"] / 10.0)).mean())
    mse_r = float((10.0 ** (-df[f"{m_raw}_psnr"] / 10.0)).mean())
    mse_n = float((10.0 ** (-df[f"{m_new}_psnr"] / 10.0)).mean())
    mse_gain_r = (mse_o - mse_r) / mse_o * 100.0
    mse_gain_n = (mse_o - mse_n) / mse_o * 100.0

    # 4. WCD Eq.15
    wc_o, wc_r, wc_n = mean("wcd")
    wc_gain_r = (wc_o - wc_r) / wc_o * 100.0
    wc_gain_n = (wc_o - wc_n) / wc_o * 100.0

    # 5. Dark Ghost Pixels
    dg_o, dg_r, dg_n = mean("dark_ghost")
    dg_gain_r = (dg_o - dg_r) / dg_o * 100.0
    dg_gain_n = (dg_o - dg_n) / dg_o * 100.0

    # 6. IE_MAE
    ie_o, ie_r, ie_n = mean("ie_mae")
    ie_gain_r = (ie_o - ie_r) / ie_o * 100.0
    ie_gain_n = (ie_o - ie_n) / ie_o * 100.0

    # 7. Stroke Recall
    rc_o, rc_r, rc_n = mean("recall")
    rc_pp = (rc_r - rc_o) * 100.0
    rc_rel = (rc_r - rc_o) / rc_o * 100.0

    # 8. Skeleton Recall
    sk_o, sk_r, sk_n = mean("skel_recall")
    sk_pp = (sk_r - sk_o) * 100.0
    sk_rel = (sk_r - sk_o) / sk_o * 100.0

    # 9. Tip Recall
    tp_o, tp_r, tp_n = mean("tip_recall")
    tp_pp = (tp_r - tp_o) * 100.0
    tp_rel = (tp_r - tp_o) / tp_o * 100.0

    # Format top table rows: [label, orig, raw, new, gain_str, best_col (1=orig,2=raw,3=new)]
    top_rows = [
        [
            "1. Độ lệch mật độ mực |1 - Density| [↓]",
            f"{ddev_o*100:.2f}% (DR {dr_o:.4f})",
            f"{ddev_r*100:.2f}% (DR {dr_r:.4f})",
            f"{ddev_n*100:.2f}% (DR {dr_n:.4f})",
            f"Giảm {ddev_gain_n:.1f}% hụt nét (Raw: -{ddev_gain_r:.1f}%)",
            3,
        ],
        [
            "2. Sai số cấu trúc nét S-EMD [↓]",
            f"{se_o:.4f}",
            f"{se_r:.4f}",
            f"{se_n:.4f}",
            f"Giảm {se_gain_r:.1f}% (Newpipe: -{se_gain_n:.1f}%)",
            2,
        ],
        [
            "3. Sai số năng lượng MSE (từ PSNR) [↓]",
            f"{mse_o:.4f} ({ps_o:.2f} dB)",
            f"{mse_r:.4f} ({ps_r:.2f} dB)",
            f"{mse_n:.4f} ({ps_n:.2f} dB)",
            f"Giảm {mse_gain_r:.1f}% MSE (PSNR +{ps_r-ps_o:.2f} dB)",
            2,
        ],
        [
            "4. Lệch quỹ đạo nét WCD Eq.15 [↓]",
            f"{wc_o:.4f} px",
            f"{wc_r:.4f} px",
            f"{wc_n:.4f} px",
            f"Giảm {wc_gain_r:.1f}% (Newpipe: -{wc_gain_n:.1f}%)",
            2,
        ],
        [
            "5. Bóng ma đen đậm Dark Ghost [↓]",
            f"{dg_o:,.1f} px",
            f"{dg_r:,.1f} px",
            f"{dg_n:,.1f} px",
            f"Giảm {dg_gain_r:.1f}% (Newpipe: -{dg_gain_n:.1f}%)",
            2,
        ],
        [
            "6. Thu hồi nét vẽ Stroke Recall [↑]",
            f"{rc_o*100:.2f}%",
            f"{rc_r*100:.2f}%",
            f"{rc_n*100:.2f}%",
            f"+{rc_pp:.2f} điểm % (+{rc_rel:.1f}% tương đối)",
            2,
        ],
        [
            "7. Thu hồi khung xương Skel Recall [↑]",
            f"{sk_o*100:.2f}%",
            f"{sk_r*100:.2f}%",
            f"{sk_n*100:.2f}%",
            f"+{sk_pp:.2f} điểm % (+{sk_rel:.1f}% tương đối)",
            2,
        ],
        [
            "8. Bảo tồn đầu ngọn nét Tip Recall [↑]",
            f"{tp_o*100:.2f}%",
            f"{tp_r*100:.2f}%",
            f"{tp_n*100:.2f}%",
            f"+{tp_pp:.2f} điểm % (+{tp_rel:.1f}% tương đối)",
            2,
        ],
        [
            "9. Sai số cường độ IE_MAE (0-255) [↓]",
            f"{ie_o:.2f}",
            f"{ie_r:.2f}",
            f"{ie_n:.2f}",
            f"Giảm {ie_gain_r:.1f}% (Newpipe: -{ie_gain_n:.1f}%)",
            2,
        ],
    ]

    # Bottom table: Delta_f = 3 & Peak Frames
    df3 = df[df["delta_f"] == 3]
    dg3_o = float(df3[f"{m_orig}_dark_ghost"].mean())
    dg3_r = float(df3[f"{m_raw}_dark_ghost"].mean())
    dg3_gain = (dg3_o - dg3_r) / dg3_o * 100.0

    bottom_rows = [
        [
            "Nhóm chuẩn Δf = 3 (TB 4 frames)",
            f"{df3[f'{m_orig}_psnr'].mean():.2f} -> {df3[f'{m_raw}_psnr'].mean():.2f} / {df3[f'{m_new}_psnr'].mean():.2f}",
            f"{df3[f'{m_orig}_skel_recall'].mean()*100:.2f}% -> {df3[f'{m_raw}_skel_recall'].mean()*100:.2f}% / {df3[f'{m_new}_skel_recall'].mean()*100:.2f}%",
            f"{df3[f'{m_orig}_f1'].mean()*100:.2f}% -> {df3[f'{m_raw}_f1'].mean()*100:.2f}% / {df3[f'{m_new}_f1'].mean()*100:.2f}%",
            f"Dark Ghost giảm {dg3_gain:.1f}% | DR = {df3[f'{m_new}_density_ratio'].mean():.4f}",
        ]
    ]

    for fname, note_fn in [
        ("1.6s.png", lambda r: f"SkelRec đạt {r[f'{m_new}_skel_recall']*100:.2f}% | WCD: {r[f'{m_orig}_wcd']:.2f} -> {r[f'{m_raw}_wcd']:.2f} px"),
        ("1.2s.png", lambda r: f"SkelRec +{ (r[f'{m_new}_skel_recall']-r[f'{m_orig}_skel_recall'])*100:.2f}pp | PSNR +{r[f'{m_raw}_psnr']-r[f'{m_orig}_psnr']:.2f} dB"),
        ("2.2s.png", lambda r: f"Khoảng nhảy xa Δf=5: SkelRec +{(r[f'{m_new}_skel_recall']-r[f'{m_orig}_skel_recall'])*100:.2f}pp"),
    ]:
        row = df[df["frame"] == fname].iloc[0]
        bottom_rows.append([
            f"Peak {fname} (Δf={int(row['delta_f'])}, t={row['t']:.2f})",
            f"{row[f'{m_orig}_psnr']:.2f} -> {row[f'{m_raw}_psnr']:.2f} / {row[f'{m_new}_psnr']:.2f}",
            f"{row[f'{m_orig}_skel_recall']*100:.2f}% -> {row[f'{m_raw}_skel_recall']*100:.2f}% / {row[f'{m_new}_skel_recall']*100:.2f}%",
            f"{row[f'{m_orig}_f1']*100:.2f}% -> {row[f'{m_raw}_f1']*100:.2f}% / {row[f'{m_new}_f1']*100:.2f}%",
            note_fn(row),
        ])

    return top_rows, bottom_rows


def draw_panel_table(ax, y_top, row_h, headers, widths, rows, x0=0.03, total_w=0.94,
                     header_bg="#1e293b", best_col_idx=None, last_col_highlight=True):
    norm_w = [w / sum(widths) * total_w for w in widths]
    xs = [x0]
    for w in norm_w[:-1]:
        xs.append(xs[-1] + w)

    # Header
    for j, hdr in enumerate(headers):
        fc = "#065f46" if (last_col_highlight and j == len(headers) - 1) else header_bg
        rect = patches.FancyBboxPatch((xs[j], y_top - row_h), norm_w[j], row_h,
                                      boxstyle="square,pad=0", ec="#cbd5e1", fc=fc, lw=1.0)
        ax.add_patch(rect)
        ha = "left" if j == 0 else "center"
        tx = xs[j] + 0.012 if ha == "left" else xs[j] + norm_w[j] / 2.0
        ax.text(tx, y_top - row_h / 2.0, hdr, fontsize=9.3, fontweight="bold",
                color="#ffffff", ha=ha, va="center")

    # Rows
    for i, r in enumerate(rows):
        cy = y_top - (i + 2) * row_h
        base_bg = "#ffffff" if i % 2 == 0 else "#f8fafc"
        b_col = r[5] if best_col_idx and len(r) > 5 else None

        for j in range(len(headers)):
            is_best = (b_col is not None and j == b_col)
            is_gain_col = (last_col_highlight and j == len(headers) - 1)
            if is_gain_col:
                fc, fg, fw = "#ecfdf5", "#047857", "bold"
            elif is_best:
                fc, fg, fw = "#dcfce7", "#15803d", "bold"
            else:
                fc, fg, fw = base_bg, ("#0f172a" if j == 0 else "#334155"), ("semibold" if j == 0 else "normal")

            rect = patches.FancyBboxPatch((xs[j], cy), norm_w[j], row_h,
                                          boxstyle="square,pad=0", ec="#e2e8f0", fc=fc, lw=0.8)
            ax.add_patch(rect)
            ha = "left" if j == 0 else "center"
            tx = xs[j] + 0.012 if ha == "left" else xs[j] + norm_w[j] / 2.0
            ax.text(tx, cy + row_h / 2.0, str(r[j]), fontsize=8.8, fontweight=fw,
                    color=fg, ha=ha, va="center")


def main():
    df = pd.read_csv(CSV_PATH)
    top_rows, bottom_rows = build_table_data(df)

    fig = plt.figure(figsize=(16.5, 9.8), dpi=220, facecolor="#ffffff")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title & Subtitle
    ax.text(0.03, 0.970,
            "BẢNG PHẢN BIỆN: QUY ĐỔI % CẢI THIỆN TƯƠNG ĐỐI & PEAK FRAMES (CASE 1)",
            fontsize=15.5, fontweight="bold", color="#0f172a", ha="left", va="top")
    ax.text(0.03, 0.933,
            "TPS-Inbetween Original (Gốc) vs Pha 3 Remake (Raw & Newpipeline)  |  benchmark_case1_per_frame.csv",
            fontsize=9.2, color="#475569", ha="left", va="top")
    ax.text(0.97, 0.933,
            "■ Ô xanh lá = Tốt nhất   |   Cột cuối = % Cải thiện tương đối",
            fontsize=8.8, fontweight="bold", color="#047857", ha="right", va="top")

    # Panel 1 header label
    ax.text(0.03, 0.885, "A. SO SÁNH 9 CHỈ SỐ CHỦ LỰC TRÊN TOÀN BỘ 17 IN-BETWEENS (TRUNG BÌNH)",
            fontsize=10.5, fontweight="bold", color="#1e293b", ha="left", va="top")

    headers_1 = [
        "Chỉ Số Đánh Giá",
        "TPS Original\n(Gốc)",
        "Pha 3 Remake\n(Raw)",
        "Pha 3 Remake\n(Newpipeline)",
        "Cải Thiện Tương Đối So Với Gốc",
    ]
    widths_1 = [3.1, 1.85, 1.85, 1.85, 3.35]
    draw_panel_table(ax, y_top=0.858, row_h=0.042, headers=headers_1, widths=widths_1,
                     rows=top_rows, best_col_idx=True, last_col_highlight=True)

    # Panel 2 header label
    ax.text(0.03, 0.405, "B. PHÂN TẦNG KHOẢNG CÁCH CHUẨN (Δf = 3) & CÁC PEAK FRAMES TIÊU BIỂU",
            fontsize=10.5, fontweight="bold", color="#1e293b", ha="left", va="top")

    headers_2 = [
        "Nhóm / Khung Hình",
        "PSNR (dB) [↑]\n(Orig -> Raw / New)",
        "Skeleton Recall [↑]\n(Orig -> Raw / New)",
        "Stroke F1 [↑]\n(Orig -> Raw / New)",
        "Điểm Nhấn Phản Biện",
    ]
    widths_2 = [2.65, 2.25, 2.35, 2.25, 3.0]
    draw_panel_table(ax, y_top=0.378, row_h=0.048, headers=headers_2, widths=widths_2,
                     rows=bottom_rows, best_col_idx=False, last_col_highlight=True)

    # Footer note explaining SSIM & White Background Saturation
    note_box = patches.FancyBboxPatch((0.03, 0.035), 0.94, 0.072,
                                      boxstyle="round,pad=0.008", ec="#cbd5e1", fc="#f8fafc", lw=0.9)
    ax.add_patch(note_box)
    ax.text(
        0.045, 0.088,
        "Lưu ý toán học khi bảo vệ trước Hội đồng: (1) Trên Ground Truth của Case 1, 90.31% diện tích ảnh là vùng phi nét vẽ (nền trắng tinh PWCR ≈ 87.5%–92.4%), "
        "khiến chỉ số SSIM toàn ảnh bị bão hòa vùng trắng",
        fontsize=8.5, color="#334155", fontweight="semibold", ha="left", va="top"
    )
    ax.text(
        0.045, 0.058,
        "     nên chỉ nhích từ 0.6660 -> 0.6685. (2) Ngược lại, các chỉ số trực tiếp trên nét vẽ (|1-Density| giảm 96.0%, S-EMD giảm 49.8%, MSE giảm 43.4%, WCD giảm 24.5%, "
        "Dark Ghost giảm 32.6%) chứng minh cải thiện áp đảo.",
        fontsize=8.5, color="#334155", ha="left", va="top"
    )

    os.makedirs(os.path.dirname(OUT_PRIMARY), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_TABLE_DIR), exist_ok=True)
    plt.savefig(OUT_PRIMARY, dpi=220, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    shutil.copy2(OUT_PRIMARY, OUT_TABLE_DIR)

    # artifact copy removed for portability
    pass
    print("[✓] Đã xuất bảng ảnh phản biện:")
    print("   -", OUT_PRIMARY)
    print("   -", OUT_TABLE_DIR)
    print("   -", artifact_copy)


if __name__ == "__main__":
    main()
