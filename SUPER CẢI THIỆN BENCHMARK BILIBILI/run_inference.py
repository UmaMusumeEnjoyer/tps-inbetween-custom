#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_inference.py
--------------------------------------------------------------------------------
Script chạy suy luận nhanh (Quickstart Runner) cho pipeline Phase 3 Remake
thuộc thư mục cách ly: SUPER CẢI THIỆN BENCHMARK BILIBILI.

Ví dụ sử dụng:
  # Chạy nội suy 2 ảnh đơn giản tại t = 0.5:
  python run_inference.py --img0 frame_0.png --img1 frame_1.png --t 0.5 --out output_t05.png

  # Chạy trên GPU với bộ lọc làm sắc nét:
  python run_inference.py --img0 frame_0.png --img1 frame_1.png --device cuda --mode balanced
--------------------------------------------------------------------------------
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from seq_p3_inference_pipeline import Phase3InferencePipeline, DEFAULT_PHASE3_CKPT


def parse_args():
    parser = argparse.ArgumentParser(description="Chạy nội suy khung hình Anime Lineart (Phase 3 Remake - Polygon-Aware TPS)")
    parser.add_argument("--img0", type=str, required=True, help="Đường dẫn ảnh keyframe đầu (I0)")
    parser.add_argument("--img1", type=str, required=True, help="Đường dẫn ảnh keyframe cuối (I1)")
    parser.add_argument("--out", type=str, default="interpolated_output.png", help="Đường dẫn file ảnh xuất ra")
    parser.add_argument("--t", type=float, default=0.5, help="Thời điểm nội suy t in (0, 1), mặc định 0.5")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_PHASE3_CKPT, help="Đường dẫn checkpoint (.pt)")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda', 'cpu', hoặc để None tự động)")
    parser.add_argument("--no_sharpener", action="store_true", help="Tắt bộ lọc sắc nét thích ứng tầng 3")
    parser.add_argument("--mode", type=str, default="balanced", choices=["balanced", "crisp", "subtle"], help="Chế độ bộ lọc sắc nét")
    parser.add_argument("--no_direct_t", action="store_true", help="Tắt tính toán trực tiếp luồng tại thời điểm t")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("🚀 KHỞI ĐỘNG PIPELINE PHASE 3 REMAKE (ISOLATED STANDALONE)")
    print(f" • Checkpoint: {args.ckpt}")
    print(f" • Input 0   : {args.img0}")
    print(f" • Input 1   : {args.img1}")
    print(f" • Mốc t     : {args.t}")
    print(f" • Output    : {args.out}")
    print("=" * 70)

    pipe = Phase3InferencePipeline(
        checkpoint_path=args.ckpt,
        device=args.device,
        use_sharpener=not args.no_sharpener,
        sharpener_mode=args.mode,
    )

    res = pipe.interpolate_pair(
        args.img0,
        args.img1,
        t=args.t,
        direct_t_flows=not args.no_direct_t,
    )

    out_arr = res["inbetween_u8"]
    Image.fromarray(out_arr).save(args.out)
    print(f"\n[✓] Hoàn tất nội suy! Ảnh đã được lưu tại: {args.out}")


if __name__ == "__main__":
    main()
