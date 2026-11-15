#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script tạo và cập nhật file báo cáo tổng hợp Benchmark (overall_benchmark.md & overall_benchmark.csv)
cho 5 mô hình Frame Interpolation trong thư mục TestModels:
  1. AnimeInbet (Line Inbetweening - Vector/Graph)
  2. tps-inbetween (Line Inbetweening - Thin-Plate Spline)
  3. AnimeInterp (RGB Animation Video Interpolation)
  4. UPR-Net (RGB Pyramid Recurrent Video Interpolation - Arbitrary-t)
  5. uniVIP (RGB Unified Video Interpolation - Arbitrary-t)

Sử dụng hoàn toàn Python Standard Library (không cần cài đặt thư viện ngoài).
"""

import os
import sys
import csv
import argparse
from datetime import datetime

# Đảm bảo UTF-8 output cho console Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Dữ liệu 17 chỉ số benchmark chuẩn hóa
BENCHMARK_DATA = [
    {
        "metric": "Chamfer Distance (CD)",
        "desc": "Khoảng cách trung bình giữa các điểm nét vẽ (Thấp hơn là tốt)",
        "group": "Line Art & Vector Geometry",
        "AnimeInbet": "0.0431",
        "tps-inbetween": "0.0011",
        "tps-finetuned-v0.1": "0.0031",
        "tps-finetuned-v0.2": "0.0017",
        "tps-finetuned-v0.3": "0.0021",
        "tps-finetuned-v1.0": "0.0010",
        "AnimeInterp": "N/A",
        "UPR-Net": "N/A",
        "uniVIP": "N/A",
        "unit": "distance",
        "best_dir": "min"
    },
    {
        "metric": "Weighted CD (WCD)",
        "desc": "CD có trọng số ưu tiên các nét vẽ chính (Thấp hơn là tốt)",
        "group": "Line Art & Vector Geometry",
        "AnimeInbet": "0.0431",
        "tps-inbetween": "0.0015",
        "tps-finetuned-v0.1": "0.0033",
        "tps-finetuned-v0.2": "0.0020",
        "tps-finetuned-v0.3": "0.0022",
        "tps-finetuned-v1.0": "0.0012",
        "AnimeInterp": "N/A",
        "UPR-Net": "N/A",
        "uniVIP": "N/A",
        "unit": "distance",
        "best_dir": "min"
    },
    {
        "metric": "Hausdorff Distance",
        "desc": "Khoảng cách lớn nhất giữa các nét; đo độ lệch nhiễu (Thấp hơn là tốt)",
        "group": "Line Art & Vector Geometry",
        "AnimeInbet": "0.2993",
        "tps-inbetween": "0.1654",
        "tps-finetuned-v0.1": "0.1250",
        "tps-finetuned-v0.2": "0.1180",
        "tps-finetuned-v0.3": "0.1150",
        "tps-finetuned-v1.0": "0.1080",
        "AnimeInterp": "N/A",
        "UPR-Net": "N/A",
        "uniVIP": "N/A",
        "unit": "distance",
        "best_dir": "min"
    },
    {
        "metric": "Line IoU",
        "desc": "Tỷ lệ diện tích chồng phủ của nét vẽ (Cao hơn là tốt, max 1.0)",
        "group": "Line Art & Vector Geometry",
        "AnimeInbet": "0.2673",
        "tps-inbetween": "0.4125",
        "tps-finetuned-v0.1": "0.5842",
        "tps-finetuned-v0.2": "0.6420",
        "tps-finetuned-v0.3": "0.6850",
        "tps-finetuned-v1.0": "0.7120",
        "AnimeInterp": "N/A",
        "UPR-Net": "N/A",
        "uniVIP": "N/A",
        "unit": "ratio [0-1]",
        "best_dir": "max"
    },
    {
        "metric": "Disconnection Rate",
        "desc": "Tỷ lệ nét vẽ bị đứt gãy không mong muốn (Thấp hơn là tốt)",
        "group": "Line Art & Vector Geometry",
        "AnimeInbet": "0.8421",
        "tps-inbetween": "3.5440",
        "tps-finetuned-v0.1": "1.0000",
        "tps-finetuned-v0.2": "0.4520",
        "tps-finetuned-v0.3": "0.3850",
        "tps-finetuned-v1.0": "0.3200",
        "AnimeInterp": "N/A",
        "UPR-Net": "N/A",
        "uniVIP": "N/A",
        "unit": "rate",
        "best_dir": "min"
    },
    {
        "metric": "EMD",
        "desc": "Chi phí tối thiểu biến đổi phân bố điểm nét vẽ (Thấp hơn là tốt)",
        "group": "Line Art & Vector Geometry",
        "AnimeInbet": "N/A",
        "tps-inbetween": "0.0639",
        "tps-finetuned-v0.1": "0.0410",
        "tps-finetuned-v0.2": "0.0380",
        "tps-finetuned-v0.3": "0.0350",
        "tps-finetuned-v1.0": "0.0310",
        "AnimeInterp": "N/A",
        "UPR-Net": "N/A",
        "uniVIP": "N/A",
        "unit": "cost",
        "best_dir": "min"
    },
    {
        "metric": "RRLD",
        "desc": "Sai lệch sau khi chuyển nét vẽ vector sang dạng pixel (Thấp hơn là tốt)",
        "group": "Line Art & Vector Geometry",
        "AnimeInbet": "28098.7",
        "tps-inbetween": "0.3968",
        "tps-finetuned-v0.1": "0.2150",
        "tps-finetuned-v0.2": "0.1850",
        "tps-finetuned-v0.3": "0.1650",
        "tps-finetuned-v1.0": "0.1450",
        "AnimeInterp": "N/A",
        "UPR-Net": "N/A",
        "uniVIP": "N/A",
        "unit": "loss",
        "best_dir": "min"
    },
    {
        "metric": "PSNR (dB)",
        "desc": "Độ chính xác toán học ở mức pixel (Cao hơn là tốt)",
        "group": "RGB Image Quality",
        "AnimeInbet": "N/A",
        "tps-inbetween": "14.1716",
        "tps-finetuned-v0.1": "15.1124",
        "tps-finetuned-v0.2": "21.0455",
        "tps-finetuned-v0.3": "20.1901",
        "tps-finetuned-v1.0": "18.7451",
        "AnimeInterp": "29.6176",
        "UPR-Net": "29.3598",
        "uniVIP": "29.3603",
        "unit": "dB",
        "best_dir": "max"
    },
    {
        "metric": "SSIM",
        "desc": "Độ tương đồng cấu trúc, độ sáng và tương phản (Cao hơn là tốt, max 1.0)",
        "group": "RGB Image Quality",
        "AnimeInbet": "N/A",
        "tps-inbetween": "0.2235",
        "tps-finetuned-v0.1": "0.7910",
        "tps-finetuned-v0.2": "0.9080",
        "tps-finetuned-v0.3": "0.8934",
        "tps-finetuned-v1.0": "0.9127",
        "AnimeInterp": "0.9386",
        "UPR-Net": "0.9349",
        "uniVIP": "0.9367",
        "unit": "score [0-1]",
        "best_dir": "max"
    },
    {
        "metric": "LPIPS",
        "desc": "Độ sai lệch hình ảnh đánh giá theo thị giác AI (Thấp hơn là tốt)",
        "group": "Perceptual Quality",
        "AnimeInbet": "N/A",
        "tps-inbetween": "0.5473",
        "tps-finetuned-v0.1": "0.1420",
        "tps-finetuned-v0.2": "0.0840",
        "tps-finetuned-v0.3": "0.0780",
        "tps-finetuned-v1.0": "0.0710",
        "AnimeInterp": "0.0665",
        "UPR-Net": "0.0613",
        "uniVIP": "0.0651",
        "unit": "distance",
        "best_dir": "min"
    },
    {
        "metric": "IE (Interpolation Error)",
        "desc": "Lỗi sai lệch pixel giữa khung hình tạo ra và thực tế (Thấp hơn là tốt)",
        "group": "RGB Image Quality",
        "AnimeInbet": "N/A",
        "tps-inbetween": "40.8025",
        "tps-finetuned-v0.1": "12.7464",
        "tps-finetuned-v0.2": "4.8210",
        "tps-finetuned-v0.3": "4.1200",
        "tps-finetuned-v1.0": "3.8200",
        "AnimeInterp": "5.4093",
        "UPR-Net": "2.9200",
        "uniVIP": "0.0110",
        "unit": "MAE / error",
        "best_dir": "min"
    },
    {
        "metric": "FVD",
        "desc": "Độ chân thực và nhất quán chuyển động toàn chuỗi video (Thấp hơn là tốt)",
        "group": "Video Temporal Consistency",
        "AnimeInbet": "N/A",
        "tps-inbetween": "N/A",
        "tps-finetuned-v0.1": "N/A",
        "tps-finetuned-v0.2": "N/A",
        "tps-finetuned-v0.3": "N/A",
        "tps-finetuned-v1.0": "N/A",
        "AnimeInterp": "N/A",
        "UPR-Net": "82.5822",
        "uniVIP": "N/A",
        "unit": "Fréchet Distance",
        "best_dir": "min"
    },
    {
        "metric": "Warping Error",
        "desc": "Lỗi biến dạng pixel theo luồng quang học (Thấp hơn là tốt)",
        "group": "Video Temporal Consistency",
        "AnimeInbet": "N/A",
        "tps-inbetween": "N/A",
        "tps-finetuned-v0.1": "N/A",
        "tps-finetuned-v0.2": "N/A",
        "tps-finetuned-v0.3": "N/A",
        "tps-finetuned-v1.0": "N/A",
        "AnimeInterp": "N/A",
        "UPR-Net": "0.0198",
        "uniVIP": "N/A",
        "unit": "optical error",
        "best_dir": "min"
    },
    {
        "metric": "Arbitrary t Support",
        "desc": "Khả năng chèn khung hình tại vị trí mốc thời gian t bất kỳ",
        "group": "Functional Capability",
        "AnimeInbet": "❌ Không",
        "tps-inbetween": "✅ Có",
        "tps-finetuned-v0.1": "✅ Có",
        "tps-finetuned-v0.2": "✅ Có",
        "tps-finetuned-v0.3": "✅ Có",
        "tps-finetuned-v1.0": "✅ Có",
        "AnimeInterp": "❌ Không (Cố định t=0.5)",
        "UPR-Net": "✅ Có",
        "uniVIP": "✅ Có",
        "unit": "feature",
        "best_dir": "feature"
    },
    {
        "metric": "Inference Speed",
        "desc": "Thời gian xử lý 1 khung hình (ms) / Tốc độ khung hình (FPS)",
        "group": "Hardware & Efficiency",
        "AnimeInbet": "397.04 ms (2.64 FPS)",
        "tps-inbetween": "140.65 ms (7.11 FPS)",
        "tps-finetuned-v0.1": "1360.28 ms (CPU)",
        "tps-finetuned-v0.2": "1319.33 ms (CPU)",
        "tps-finetuned-v0.3": "52.30 ms (19.12 FPS)",
        "tps-finetuned-v1.0": "47.52 ms (21.04 FPS)",
        "AnimeInterp": "2128.60 ms (0.50 FPS)",
        "UPR-Net": "248.61 ms (4.02 FPS)",
        "uniVIP": "143.86 ms (7.00 FPS)",
        "unit": "ms / FPS",
        "best_dir": "speed"
    },
    {
        "metric": "VRAM Peak (MB)",
        "desc": "Dung lượng bộ nhớ GPU tối đa tiêu thụ",
        "group": "Hardware & Efficiency",
        "AnimeInbet": "691.85 MB",
        "tps-inbetween": "538.70 MB",
        "tps-finetuned-v0.1": "< 1000 MB",
        "tps-finetuned-v0.2": "< 1000 MB",
        "tps-finetuned-v0.3": "< 1000 MB",
        "tps-finetuned-v1.0": "< 1000 MB",
        "AnimeInterp": "2936.00 MB",
        "UPR-Net": "3340.00 MB",
        "uniVIP": "850.02 MB",
        "unit": "MB",
        "best_dir": "min"
    },
    {
        "metric": "Fine-tuning Speed",
        "desc": "Tốc độ tinh chỉnh mô hình",
        "group": "Training & Adaptation",
        "AnimeInbet": "N/A",
        "tps-inbetween": "N/A",
        "tps-finetuned-v0.1": "~4.1h (25ep/Kaggle)",
        "tps-finetuned-v0.2": "~2.2h (Dual-T4)",
        "tps-finetuned-v0.3": "~4.2h (20ep/Dual-T4)",
        "tps-finetuned-v1.0": "~1.8h (30ep/Dual-T4)",
        "AnimeInterp": "N/A",
        "UPR-Net": "N/A",
        "uniVIP": "N/A",
        "unit": "speed",
        "best_dir": "speed"
    }
]

MODELS = ["AnimeInbet", "tps-inbetween", "tps-finetuned-v0.1", "tps-finetuned-v0.2", "tps-finetuned-v0.3", "tps-finetuned-v1.0", "AnimeInterp", "UPR-Net", "uniVIP"]

def load_data_from_csv(csv_path: str):
    """Đọc dữ liệu từ file CSV nếu người dùng cung cấp file ngoài."""
    if not os.path.exists(csv_path):
        print(f"⚠️ Cảnh báo: Không tìm thấy file CSV tại {csv_path}. Sử dụng dữ liệu mặc định.")
        return BENCHMARK_DATA
    
    rows = []
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            metric_name = r.get("Chỉ số (Metric)", r.get("metric", ""))
            desc = r.get("Miêu tả ngắn & Hướng tối ưu", r.get("description", ""))
            row_dict = {
                "metric": metric_name,
                "desc": desc,
                "AnimeInbet": r.get("AnimeInbet", "N/A"),
                "tps-inbetween": r.get("tps-inbetween", "N/A"),
                "tps-finetuned-v0.1": r.get("tps-finetuned-v0.1", "N/A"),
                "tps-finetuned-v0.2": r.get("tps-finetuned-v0.2", "N/A"),
                "tps-finetuned-v0.3": r.get("tps-finetuned-v0.3", "N/A"),
                "tps-finetuned-v1.0": r.get("tps-finetuned-v1.0", "N/A"),
                "AnimeInterp": r.get("AnimeInterp", "N/A"),
                "UPR-Net": r.get("UPR-Net", "N/A"),
                "uniVIP": r.get("uniVIP", "N/A"),
            }
            # Match existing group / metadata if available
            matched = next((item for item in BENCHMARK_DATA if item["metric"].lower() in metric_name.lower() or metric_name.lower() in item["metric"].lower()), None)
            if matched:
                row_dict["group"] = matched.get("group", "Chỉ số tổng quát")
                row_dict["unit"] = matched.get("unit", "")
                row_dict["best_dir"] = matched.get("best_dir", "")
                if row_dict["tps-finetuned-v1.0"] == "N/A" and "tps-finetuned-v1.0" in matched:
                    row_dict["tps-finetuned-v1.0"] = matched["tps-finetuned-v1.0"]
            else:
                row_dict["group"] = "Chỉ số bổ sung"
                row_dict["unit"] = ""
                row_dict["best_dir"] = ""
            rows.append(row_dict)
    return rows if rows else BENCHMARK_DATA


def export_to_csv(data: list, output_csv_path: str):
    """Xuất dữ liệu benchmark ra định dạng CSV chuẩn."""
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    fieldnames = ["Chỉ số (Metric)", "Miêu tả ngắn & Hướng tối ưu", "AnimeInbet", "tps-inbetween", "tps-finetuned-v0.1", "tps-finetuned-v0.2", "tps-finetuned-v0.3", "tps-finetuned-v1.0", "AnimeInterp", "UPR-Net", "uniVIP"]
    with open(output_csv_path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for item in data:
            writer.writerow([
                item["metric"],
                item["desc"],
                item.get("AnimeInbet", "N/A"),
                item.get("tps-inbetween", "N/A"),
                item.get("tps-finetuned-v0.1", "N/A"),
                item.get("tps-finetuned-v0.2", "N/A"),
                item.get("tps-finetuned-v0.3", "N/A"),
                item.get("tps-finetuned-v1.0", "N/A"),
                item.get("AnimeInterp", "N/A"),
                item.get("UPR-Net", "N/A"),
                item.get("uniVIP", "N/A"),
            ])
    print(f"✅ Đã xuất file CSV tại: {output_csv_path}")


def generate_markdown(data: list, output_md_path: str):
    """Tạo tài liệu Markdown Overall Benchmark chuyên nghiệp với phân tích chi tiết."""
    os.makedirs(os.path.dirname(os.path.abspath(output_md_path)), exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Xây dựng bảng tổng hợp markdown
    table_lines = []
    table_lines.append("| Chỉ số (Metric) | Miêu tả ngắn & Hướng tối ưu | AnimeInbet | tps-inbetween (gốc) | tps-finetuned v0.1 | tps-finetuned v0.2 | tps-finetuned v0.3 | **tps-finetuned V1.0 (Production)** | AnimeInterp | UPR-Net | uniVIP |")
    table_lines.append("|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    
    for row in data:
        m = row["metric"]
        d = row["desc"]
        a_inbet = row.get("AnimeInbet", "N/A")
        tps = row.get("tps-inbetween", "N/A")
        tps_ft = row.get("tps-finetuned-v0.1", "N/A")
        tps_ft_v02 = row.get("tps-finetuned-v0.2", "N/A")
        tps_ft_v03 = row.get("tps-finetuned-v0.3", "N/A")
        tps_ft_v10 = row.get("tps-finetuned-v1.0", "N/A")
        a_interp = row.get("AnimeInterp", "N/A")
        upr = row.get("UPR-Net", "N/A")
        univip = row.get("uniVIP", "N/A")
        table_lines.append(f"| **{m}** | {d} | {a_inbet} | {tps} | {tps_ft} | {tps_ft_v02} | {tps_ft_v03} | **{tps_ft_v10}** | {a_interp} | {upr} | {univip} |")
    
    master_table_md = "\n".join(table_lines)

    doc_template = """# 📊 Báo Cáo Tổng Hợp Kết Quả Benchmark Toàn Diện (Overall Benchmark)

> **Hệ thống đánh giá:** PBL6 Animation Frame Interpolation Benchmark Suite  
> **Dataset tham chiếu:** ATD-12K (Animation Triplet Dataset - 540p resolution)  
> **Thời điểm xuất bản:** {timestamp}  
> **Danh sách 5 mô hình được đánh giá:**
> 1. [AnimeInbet](file:///e:/CODE/PBL/PBL6/Test/Benchmark/AnimeInbet/benchmark_AnimeInbet_guide.md) *(Deep Geometrized Cartoon Line Inbetweening - ICCV 2023)*
> 2. [tps-inbetween](file:///e:/CODE/PBL/PBL6/Test/Benchmark/tps-inbetween/benchmark_tps-inbetween_guide.md) *(Thin-Plate Spline Motion Representation - CVPR 2024)*
> 3. [AnimeInterp](file:///e:/CODE/PBL/PBL6/Test/Benchmark/AnimeInterp/benchmark_AnimeInterp_guide.md) *(Segment-Guided Optical Flow Anime VFI - CVPR 2021)*
> 4. [UPR-Net](file:///e:/CODE/PBL/PBL6/Test/Benchmark/UPR-Net/benchmark_UPR-Net_guide.md) *(Unified Pyramid Recurrent Network - TPAMI 2023)*
> 5. [uniVIP](file:///e:/CODE/PBL/PBL6/Test/Benchmark/uniVIP/benchmark_uniVIP_guide.md) *(Unified Video Interpolation with Dual-Path Alignment - CVPR 2024)*

---

## 📑 Mục Lục
1. [Bảng Tổng Hợp Toàn Bộ 17 Chỉ Số Benchmark](#1-bảng-tổng-hợp-toàn-bộ-17-chỉ-số-benchmark)
2. [Phân Tích Chi Tiết Theo Nhóm Domain Bài Toán](#2-phân-tích-chi-tiết-theo-nhóm-domain-bài-toán)
   - [2.1 Nhóm Cartoon Line Art Inbetweening (AnimeInbet vs tps-inbetween)](#21-nhóm-cartoon-line-art-inbetweening-animeinbet-vs-tps-inbetween)
   - [2.2 Nhóm RGB Video Frame Interpolation (AnimeInterp vs UPR-Net vs uniVIP)](#22-nhóm-rgb-video-frame-interpolation-animeinterp-vs-upr-net-vs-univip)
3. [Hiệu Năng & Tài Nguyên Phần Cứng (Speed & VRAM)](#3-hiệu-năng--tài-nguyên-phần-cứng-speed--vram)
4. [So Sánh Tính Năng Nội Suy Thời Gian Mịn (Arbitrary-t Support)](#4-so-sánh-tính-năng-nội-suy-thời-gian-mịn-arbitrary-t-support)
5. [Ma Trận Khuyến Nghị Triển Khai (Deployment Recommendations)](#5-ma-trận-khuyến-nghị-triển-khai-deployment-recommendations)
6. [Hướng Dẫn Tái Hiện Kết Quả (Reproduction Guide)](#6-hướng-dẫn-tái-hiện-kết-quả-reproduction-guide)

---

## 1. Bảng Tổng Hợp Toàn Bộ 17 Chỉ Số Benchmark

Dưới đây là bảng kết quả đo đạc chuẩn hóa trên cùng tập dữ liệu thực nghiệm:

{master_table_md}

> [!NOTE]
> - Các ô ghi **N/A** thể hiện chỉ số **không áp dụng** cho đặc thù của miền mô hình đó (ví dụ: các mô hình RGB không tính Chamfer Distance hình học vector, và các mô hình Line Art không tính PSNR/SSIM RGB).
> - Để tham khảo hướng dẫn chi tiết từng model, hãy truy cập các thư mục con tương ứng tại `Benchmark/<Model_Name>/`.

---

## 2. Phân Tích Chi Tiết Theo Nhóm Domain Bài Toán

```mermaid
graph TD
    A[TestModels Benchmark] --> B[Domain 1: Line Art Inbetweening]
    A --> C[Domain 2: RGB Video Frame Interpolation]
    
    B --> B1[AnimeInbet - Graph & Spectral]
    B --> B2[tps-inbetween - Thin-Plate Spline]
    
    C --> C1[AnimeInterp - Recurrent Optical Flow]
    C --> C2[UPR-Net - Pyramid Recurrent Arbitrary-t]
    C --> C3[uniVIP - Dual Alignment Arbitrary-t]
```

### 2.1 Nhóm Cartoon Line Art Inbetweening (AnimeInbet vs tps-inbetween)

Nhóm này xử lý tranh vẽ nét nhị phân/đơn sắc (Line Art / Sketch Inbetweening). Không áp dụng các metric pixel thông thường (PSNR/SSIM) vì không gian màu chủ yếu là nền trắng/nét đen, thay vào đó đo bằng độ chính xác hình học của các đường nét.

| Chỉ Số Đánh Giá | AnimeInbet | tps-inbetween | Mô Hình Vượt Trội | Nhận Xét & Phân Tích Kỹ Thuật |
|---|:---:|:---:|:---:|---|
| **Chamfer Distance (CD) ↓** | 0.0431 | **0.0011** | 🏆 **tps-inbetween** | tps-inbetween thấp hơn **39 lần**, cho thấy vị trí các điểm nét vẽ trung gian bám cực sát với ground-truth. |
| **Weighted CD (WCD) ↓** | 0.0431 | **0.0015** | 🏆 **tps-inbetween** | Khi gán trọng số ưu tiên cho các nét viền chủ đạo, tps-inbetween duy trì độ chính xác vượt trội (thấp hơn ~28 lần). |
| **Hausdorff Distance ↓** | 0.2993 | **0.1654** | 🏆 **tps-inbetween** | tps-inbetween giảm gần **45%** khoảng cách cực đại, hạn chế tối đa các điểm dị biệt (outliers) hoặc nét bay tự do. |
| **Line IoU ↑** | 0.2673 | **0.4125** | 🏆 **tps-inbetween** | Tỷ lệ chồng phủ diện tích nét vẽ của tps-inbetween đạt **41.25%** (so với 26.73% của AnimeInbet), nét vẽ dày dặn và khớp tốt hơn. |
| **Disconnection Rate ↓** | **0.8421** | 3.5440 | 🏆 **AnimeInbet** | AnimeInbet có tỷ lệ đứt nét thấp hơn **4.2 lần** nhờ cơ chế Spectral Graph Topology giữ liên kết topo chặt chẽ, tránh đứt nét khi chuyển động lớn. |
| **EMD (Earth Mover's) ↓** | N/A | **0.0639** | 🏆 **tps-inbetween** | Chi phí phân bố dịch chuyển điểm nét rất nhỏ (0.0639). |
| **RRLD (Raster-to-Vector) ↓** | 28098.7 | **0.3968** | 🏆 **tps-inbetween** | AnimeInbet gặp lỗi rasterization rất lớn ở dạng thưa (sparse skeleton), trong khi tps-inbetween tái tạo nét raster hoàn chỉnh (0.3968). |

> [!TIP]
> **Tóm Lược Nhóm Line Art:**
> - **tps-inbetween** là mô hình **toàn diện và sắc nét nhất** cho bài toán vẽ nét hoạt hình, kiểm soát tốt toàn bộ vị trí không gian (CD, WCD, Hausdorff, Line IoU, RRLD).
> - **AnimeInbet** có thế mạnh đặc thù về **tính bảo toàn topo đường nét** (Disconnection Rate thấp hơn rõ rệt), thích hợp làm khung xương liên kết (intermediate topological skeleton).

---

### 2.2 Nhóm RGB Video Frame Interpolation (AnimeInterp vs UPR-Net vs uniVIP)

Nhóm này xử lý các khung hình hoạt hình RGB hoàn chỉnh, có màu sắc, background và ánh sáng đổ bóng phức tạp.

| Chỉ Số Đánh Giá | AnimeInterp | UPR-Net | uniVIP | Mô Hình Vượt Trội | Nhận Xét & Phân Tích Kỹ Thuật |
|---|:---:|:---:|:---:|:---:|---|
| **PSNR (dB) ↑** | **29.6176** | 29.3598 | 29.3603 | 🏆 **AnimeInterp** | Cả 3 model đều đạt chất lượng tái tạo màu sắc rất cao (>29.3 dB). AnimeInterp nhỉnh hơn 0.25 dB nhờ module Segment-Guided Flow chuyên sâu cho Anime. |
| **SSIM [0-1] ↑** | **0.9386** | 0.9349 | 0.9367 | 🏆 **AnimeInterp** | Cả 3 model đều đạt độ tương đồng cấu trúc xuất sắc (>0.934). Mức chênh lệch giữa 3 mô hình là cực nhỏ (<0.4%). |
| **LPIPS ↓** | 0.0665 | **0.0613** | 0.0651 | 🏆 **UPR-Net** | UPR-Net cho chất lượng thị giác tự nhiên nhất theo cảm nhận mạng nơ-ron sâu (LPIPS thấp nhất 0.0613), giảm thiểu vệt mờ (blur artifacts). |
| **IE (Interpolation Error) ↓** | 5.4093 | 2.9200 | **0.0110** | 🏆 **uniVIP** | **uniVIP tạo đột phá** với sai số nội suy pixel cực thấp (**0.0110 MAE**), tốt hơn gấp ~265 lần so với UPR-Net và ~490 lần so với AnimeInterp. |
| **FVD (Video Fréchet) ↓** | N/A | **82.5822** | N/A | 🏆 **UPR-Net** | UPR-Net hỗ trợ xuất chuỗi video sequence chuẩn, đo được độ mượt mà tổng thể FVD đạt 82.58. |
| **Warping Error ↓** | N/A | **0.0198** | N/A | 🏆 **UPR-Net** | Sai số dịch chuyển theo luồng quang học cực thấp (0.0198), thể hiện độ ổn định chuyển động thời gian (temporal consistency). |

> [!IMPORTANT]
> **Tóm Lược Nhóm RGB:**
> - **uniVIP:** Là mô hình có **sai số nội suy pixel thấp nhất tuyệt đối (IE = 0.0110)** và duy trì chỉ số PSNR/SSIM ở mức đỉnh cao, đồng thời tốc độ xử lý nhanh nhất nhóm RGB.
> - **UPR-Net:** Dẫn đầu về **độ chân thực thị giác (LPIPS = 0.0613)**, hỗ trợ nội suy chuỗi video hoàn chỉnh với chỉ số FVD và Warping Error xuất sắc.
> - **AnimeInterp:** Đạt điểm PSNR và SSIM cao nhất, tuy nhiên thời gian inference rất lâu và không hỗ trợ Arbitrary-t.

---

## 3. Hiệu Năng & Tài Nguyên Phần Cứng (Speed & VRAM)

Biểu đồ so sánh tốc độ (FPS) và mức độ chiếm dụng VRAM của 5 mô hình:

```text
========================================================================================
MÔ HÌNH               TỐC ĐỘ (FPS) ↑                     PEAK VRAM (MB) ↓
========================================================================================
tps-inbetween         [████████████████████] 7.11 FPS    [███                 ] 538.70 MB
uniVIP                [███████████████████ ] 7.00 FPS    [█████               ] 850.02 MB
UPR-Net               [███████████         ] 4.02 FPS    [████████████████████] 3340.00 MB
AnimeInbet            [███████             ] 2.64 FPS    [████                ] 691.85 MB
AnimeInterp           [█                   ] 0.50 FPS    [█████████████████   ] 2936.00 MB
========================================================================================
```

### Bảng Chi Tiết Phần Cứng:

| Mô Hình | Domain | Thời Gian Xử Lý (ms/frame) | Tốc Độ Khung Hình (FPS) | Peak VRAM (MB) | Đánh Giá Khả Năng Triển Khai Thực Tế |
|---|:---:|:---:|:---:|:---:|---|
| **tps-inbetween** | Line Art | **140.65 ms** | **7.11 FPS** | **538.70 MB** | 🟢 **Siêu nhẹ & Nhanh nhất:** Chạy mượt trên GPU phổ thông, VRAM cực thấp. |
| **uniVIP** | RGB Video | **143.86 ms** | **7.00 FPS** | **850.02 MB** | 🟢 **Tối ưu nhất cho RGB:** Tốc độ 7 FPS, VRAM < 1GB, lý tưởng cho production. |
| **UPR-Net** | RGB Video | 248.61 ms | 4.02 FPS | 3340.00 MB | 🟡 **Cân bằng tốt:** Chất lượng video cao, cần GPU tầm trung (≥ 4GB VRAM). |
| **AnimeInbet** | Line Art | 397.04 ms | 2.64 FPS | 691.85 MB | 🟡 **Nhẹ về VRAM:** Tốc độ ở mức trung bình (2.64 FPS) do tính toán đồ thị phổ. |
| **AnimeInterp** | RGB Video | 2128.60 ms | 0.50 FPS | 2936.00 MB | 🔴 **Nặng & Chậm:** Tốn 2.1s/frame, không thích hợp cho realtime hoặc batch lớn. |

---

## 4. So Sánh Tính Năng Nội Suy Thời Gian Mịn (Arbitrary-t Support)

Khả năng chèn khung hình tại vị trí mốc thời gian $t \\in (0, 1)$ bất kỳ là yếu tố cốt lõi để tạo hiệu ứng Slow-Motion mượt mà, chuyển đổi linh hoạt tốc độ khung hình (vd: 12fps -> 24fps -> 60fps -> 120fps).

```mermaid
flowchart LR
    subgraph Fixed_t ["Cố định t = 0.5 (Chỉ 1 frame giữa)"]
        A1[AnimeInbet] --> B1["Output: t = 0.5 duy nhất"]
        A2[AnimeInterp] --> B2["Output: t = 0.5 duy nhất"]
    end

    subgraph Arbitrary_t ["Arbitrary-t (Hỗ trợ t tùy ý liên tục)"]
        C1[tps-inbetween] --> D1["Output: t = 0.1, 0.25, 0.5, 0.75, 0.9..."]
        C2[UPR-Net] --> D2["Output: t = 0.1, 0.25, 0.5, 0.75, 0.9..."]
        C3[uniVIP] --> D3["Output: t = 0.1, 0.25, 0.5, 0.75, 0.9..."]
    end
```

- **Nhóm Cố Định ($t = 0.5$):**
  - `AnimeInbet` và `AnimeInterp` chỉ sinh được frame tại $t = 0.5$. Nếu muốn tạo chuỗi nhiều frame, phải nội suy đệ quy (recursive interpolation) làm tích lũy sai số và nhân cấp số nhân thời gian xử lý.
- **Nhóm Hỗ Trợ Tùy Ý ($t \\in [0, 1]$):**
  - `tps-inbetween`, `UPR-Net`, `uniVIP` cho phép truyền trực tiếp mốc thời gian $t$, tính toán tức thì vị trí biến dạng của vật thể tại thời điểm mong muốn mà không gây méo hình.

---

## 5. Ma Trận Khuyến Nghị Triển Khai (Deployment Recommendations)

Dựa trên toàn bộ kết quả thực nghiệm, dưới đây là định hướng lựa chọn mô hình cho từng bài toán cụ thể:

| Kịch Bản Ứng Dụng | Mô Hình Khuyến Nghị Số 1 | Mô Hình Dự Phòng | Lý Do Lựa Chọn |
|---|---|---|---|
| **Nội suy Line Art / Tranh phác thảo 2D** | 🥇 **tps-inbetween** | 🥈 AnimeInbet | Độ chính xác hình học cao nhất (CD = 0.0011), hỗ trợ arbitrary-t, tốc độ 7.11 FPS, VRAM chỉ 538 MB. |
| **Bảo toàn cấu trúc Skeleton / Rigging nét** | 🥇 **AnimeInbet** | 🥈 tps-inbetween | Disconnection Rate thấp kỷ lục (0.8421), duy trì liên kết các nét tốt nhất khi chuyển động phức tạp. |
| **Sản xuất Anime RGB / Production Pipeline** | 🥇 **uniVIP** | 🥈 UPR-Net | IE cực thấp (0.0110), tốc độ 7.00 FPS, VRAM < 1GB, hỗ trợ chèn thời gian tùy ý. |
| **Render Video Slow-Motion Chất Lượng Cao** | 🥇 **UPR-Net** | 🥈 uniVIP | LPIPS tốt nhất (0.0613), tính nhất quán chuyển động toàn chuỗi video cao (FVD = 82.58, Warping Error = 0.0198). |
| **Phục dựng khung hình tĩnh cực nét (High PSNR)** | 🥇 **AnimeInterp** | 🥈 uniVIP | PSNR cao nhất (29.62 dB), chi tiết sắc nét cho từng frame tĩnh đơn lẻ. |

---

## 6. Hướng Dẫn Tái Hiện Kết Quả (Reproduction Guide)

Để chạy lại script tổng hợp hoặc cập nhật bảng số liệu tự động từ các file CSV kết quả:

```powershell
# Chạy script tạo lại báo cáo Markdown & CSV
python Benchmark/generate_overall_benchmark.py

# Hoặc tùy chỉnh đường dẫn xuất file
python Benchmark/generate_overall_benchmark.py --output Benchmark/overall_benchmark.md --export-csv Benchmark/overall_benchmark.csv
```

### Cấu trúc liên kết các tài liệu chi tiết:
- 📖 [Hướng dẫn Benchmark AnimeInbet](file:///e:/CODE/PBL/PBL6/Test/Benchmark/AnimeInbet/benchmark_AnimeInbet_guide.md)
- 📖 [Hướng dẫn Benchmark tps-inbetween](file:///e:/CODE/PBL/PBL6/Test/Benchmark/tps-inbetween/benchmark_tps-inbetween_guide.md)
- 📖 [Hướng dẫn Benchmark AnimeInterp](file:///e:/CODE/PBL/PBL6/Test/Benchmark/AnimeInterp/benchmark_AnimeInterp_guide.md)
- 📖 [Hướng dẫn Benchmark UPR-Net](file:///e:/CODE/PBL/PBL6/Test/Benchmark/UPR-Net/benchmark_UPR-Net_guide.md)
- 📖 [Hướng dẫn Benchmark uniVIP](file:///e:/CODE/PBL/PBL6/Test/Benchmark/uniVIP/benchmark_uniVIP_guide.md)
"""

    content = doc_template.format(timestamp=timestamp, master_table_md=master_table_md)

    with open(output_md_path, mode='w', encoding='utf-8') as f:
        f.write(content.strip() + "\n")
    print(f"✅ Đã tạo thành công file Markdown báo cáo tại: {output_md_path}")


def print_console_summary(data: list):
    """In tóm tắt bảng benchmark ra console."""
    print("\n" + "=" * 110)
    print("📊 BẢNG TỔNG HỢP KẾT QUẢ BENCHMARK CHUNG (OVERALL BENCHMARK)")
    print("=" * 110)
    print(f"{'Chỉ số (Metric)':<26} | {'AnimeInbet':<11} | {'tps (gốc)':<11} | {'tps-v0.1':<12} | {'AnimeInterp':<11} | {'UPR-Net':<10} | {'uniVIP':<8}")
    print("-" * 110)
    for row in data:
        m = row["metric"]
        a_inbet = row.get("AnimeInbet", "N/A")
        tps = row.get("tps-inbetween", "N/A")
        tps_ft = row.get("tps-finetuned-v0.1", "N/A")
        tps_ft_v02 = row.get("tps-finetuned-v0.2", "N/A")
        a_interp = row.get("AnimeInterp", "N/A")
        upr = row.get("UPR-Net", "N/A")
        univip = row.get("uniVIP", "N/A")
        print(f"{m:<26} | {a_inbet:<11} | {tps:<11} | {tps_ft:<12} | {a_interp:<11} | {upr:<10} | {univip:<8}")
    print("=" * 110 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Tạo file tổng hợp Overall Benchmark (.md, .docx và .csv)")
    parser.add_argument("--csv", type=str, default=None, help="Đường dẫn file CSV dữ liệu đầu vào (tùy chọn)")
    parser.add_argument("--output", type=str, default=None, help="Đường dẫn file Markdown đầu ra (mặc định: Benchmark/overall_benchmark.md)")
    parser.add_argument("--export-csv", type=str, default=None, help="Đường dẫn file CSV đầu ra (mặc định: Benchmark/overall_benchmark.csv)")
    parser.add_argument("--export-docx", type=str, default=None, help="Đường dẫn file Word đầu ra (mặc định: Benchmark/overall_benchmark.docx)")
    args = parser.parse_args()

    # Xác định đường dẫn mặc định trong thư mục metrics/baselines
    base_dir = os.path.dirname(os.path.abspath(__file__))
    benchmark_dir = os.path.abspath(os.path.join(base_dir, "..", "..", "metrics", "baselines"))

    output_md = args.output if args.output else os.path.join(benchmark_dir, "overall_benchmark.md")
    output_csv = args.export_csv if args.export_csv else os.path.join(benchmark_dir, "overall_benchmark.csv")
    output_docx = args.export_docx if args.export_docx else os.path.join(benchmark_dir, "overall_benchmark.docx")

    print(f"🚀 Bắt đầu quá trình tạo Overall Benchmark...")
    print(f"📂 Thư mục đích: {benchmark_dir}")

    # 1. Nạp dữ liệu
    if args.csv and os.path.exists(args.csv):
        print(f"📖 Đang đọc dữ liệu từ: {args.csv}")
        data = load_data_from_csv(args.csv)
    else:
        data = BENCHMARK_DATA

    # 2. In tóm tắt ra console
    print_console_summary(data)

    # 3. Xuất file Markdown
    generate_markdown(data, output_md)

    # 4. Xuất file CSV
    export_to_csv(data, output_csv)

    # 5. Xuất file Word (.docx) nếu có module docx
    docx_created = False
    try:
        from generate_overall_benchmark_docx import build_docx_document
        build_docx_document(output_docx)
        docx_created = True
    except ImportError:
        try:
            import importlib.util
            docx_script = os.path.join(benchmark_dir, "generate_overall_benchmark_docx.py")
            if os.path.exists(docx_script):
                spec = importlib.util.spec_from_file_location("generate_overall_benchmark_docx", docx_script)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                mod.build_docx_document(output_docx)
                docx_created = True
        except Exception as e:
            print(f"⚠️ Không thể tạo file .docx: {e}")
    except Exception as e:
        print(f"⚠️ Lỗi khi xuất file Word .docx: {e}")

    print(f"\n🎉 Hoàn thành! Toàn bộ file báo cáo đã sẵn sàng tại:")
    print(f"   📄 Markdown: {output_md}")
    print(f"   📊 CSV     : {output_csv}")
    if docx_created:
        print(f"   📝 Word Doc: {output_docx}\n")


if __name__ == "__main__":
    main()
