#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script thống kê chỉ số benchmark model từ file CSV kết quả (benchmark_results_tps-inbetween.csv).
Định dạng kết quả theo mẫu Final Verification và Mean metrics.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Đảm bảo UTF-8 encoding trên Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def find_default_csv():
    """Tự động tìm kiếm file benchmark_results_tps-inbetween.csv."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "Result", "benchmark_results_tps-inbetween.csv"),
        os.path.join(os.path.dirname(__file__), "benchmark_results_tps-inbetween.csv"),
        os.path.join(os.getcwd(), "Result", "benchmark_results_tps-inbetween.csv"),
        os.path.join(os.getcwd(), "benchmark_results_tps-inbetween.csv"),
    ]
    for p in candidates:
        if os.path.exists(p) and os.path.getsize(p) > 200: # Tìm file có data thực tế
            return p
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


COLUMN_MAPPING = {
    'clip_name': 'clip_name',
    'CD': 'chamfer_distance',
    'WCD': 'weighted_cd',
    'LPIPS': 'lpips',
    'EMD': 'emd',
    'RRLD': 'rrld',
    'PSNR': 'psnr',
    'SSIM': 'ssim',
    'IE': 'ie_mae',
    'FVD': 'fvd',
    'Warping_Error': 'warping_error',
    'Hausdorff': 'hausdorff_dist',
    'Line_IoU': 'line_iou',
    'Disconnection_Rate': 'disconnection_rate',
    'Arbitrary_t': 'arbitrary_t_support',
    'Speed_ms': 'inference_ms',
    'FPS': 'fps',
    'VRAM_MB': 'vram_peak_mb',
    'Fine_tuning_Speed': 'finetune_speed'
}


def summarize_benchmark(csv_path: str, standardize: bool = False, export_summary: bool = True):
    if not os.path.exists(csv_path):
        print(f"❌ Không tìm thấy file CSV tại: {csv_path}")
        return

    # 1. Đọc CSV
    df = pd.read_csv(csv_path)

    if standardize:
        # Chuẩn hóa tên cột theo schema 21 cột
        df = df.rename(columns=COLUMN_MAPPING)
        if 'model' not in df.columns:
            df.insert(1, 'model', 'TPS-Inbetween')
        if 'resolution' not in df.columns:
            df.insert(2, 'resolution', '960x540')

    # 2. Hiển thị thông tin kiểm tra (Final Verification)
    print("🔍 Final Verification")
    print("=" * 60)
    print(f"CSV rows     : {len(df)}")
    print(f"CSV columns  : {len(df.columns)}")
    print("\nColumn list  :")

    for col in df.columns:
        if len(df) > 0:
            val = df[col].iloc[0]
            # Format NaN
            if pd.isna(val) or str(val).strip() in ['nan', 'NaN', 'N/A', 'ERROR']:
                sample_val = "nan"
            else:
                sample_val = str(val)
        else:
            sample_val = "N/A"
        print(f"   {col:<25} → {sample_val}")

    print("\n" + "=" * 60)
    print("📊 Mean metrics over all clips:")

    # Danh sách các cột không phải metric định lượng
    non_numeric_cols = {
        'clip_name', 'model', 'resolution', 'status', 'error_msg',
        'arbitrary_t', 'arbitrary_t_support', 'finetune_speed', 'fine_tuning_speed'
    }

    numeric_summary = {}

    for col in df.columns:
        if col.lower() in non_numeric_cols:
            continue
        # Chuyển đổi sang dạng số (bỏ qua chuỗi lỗi 'N/A', 'ERROR')
        s = pd.to_numeric(df[col].replace({'N/A': np.nan, 'ERROR': np.nan}), errors='coerce')
        if s.notna().sum() > 0:
            mean_val = s.mean()
            print(f"   {col:<25} : {mean_val:.4f}")
            numeric_summary[col] = {
                "Mean": mean_val,
                "Median": s.median(),
                "Min": s.min(),
                "Max": s.max(),
                "Std": s.std()
            }

    print("\n" + "=" * 60)

    # 3. Tùy chọn xuất file summary CSV
    if export_summary and numeric_summary:
        summary_df = pd.DataFrame.from_dict(numeric_summary, orient='index')
        output_dir = os.path.dirname(csv_path)
        summary_csv_path = os.path.join(output_dir, "benchmark_summary_tps-inbetween.csv")
        summary_df.to_csv(summary_csv_path, index=True, index_label="Metric")
        print(f"💾 Đã lưu bảng thống kê chi tiết tại: {summary_csv_path}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Thống kê kết quả Benchmark TPS-Inbetween")
    parser.add_argument(
        "--csv",
        type=str,
        default=find_default_csv(),
        help="Đường dẫn đến file CSV kết quả benchmark"
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Chuẩn hóa tên cột sang schema 21 cột (giống UPR-Net)"
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Không xuất file tóm tắt summary CSV"
    )
    args = parser.parse_args()

    summarize_benchmark(args.csv, standardize=args.standardize, export_summary=not args.no_export)


if __name__ == "__main__":
    main()
