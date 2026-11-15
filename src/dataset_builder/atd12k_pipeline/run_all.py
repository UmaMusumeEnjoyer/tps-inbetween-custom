#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script Điều Phối Toàn Diện (Master Orchestrator):
Thực thi trọn gói từ Giai đoạn 1 đến Giai đoạn 6 và kiểm chứng thử nghiệm.
"""

import os
import sys
import time

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from stage1_extract_frames import extract_frames
from stage2_micro_dedup import run_micro_dedup
from stage3_mine_triplets import mine_triplets
from stage4_screen_gluestick import screen_triplets
from stage5_process_lineart import run_lineart_processing
from stage6_export_dataset import export_dataset
from validate_pipeline import run_validation


def main():
    total_start_time = time.time()

    print("\n" + "=" * 80)
    print("🚀 BẮT ĐẦU THỰC THI TOÀN BỘ PIPELINE ATD-12K CHO TPS-INBETWEEN")
    print("=" * 80 + "\n")

    # Giai đoạn 1
    t0 = time.time()
    num_raw = extract_frames()
    print(f"⏱️ Giai đoạn 1 hoàn thành trong: {time.time() - t0:.1f}s\n")

    # Giai đoạn 2
    t0 = time.time()
    drawings = run_micro_dedup()
    print(f"⏱️ Giai đoạn 2 hoàn thành trong: {time.time() - t0:.1f}s\n")

    # Giai đoạn 3
    t0 = time.time()
    candidates = mine_triplets()
    print(f"⏱️ Giai đoạn 3 hoàn thành trong: {time.time() - t0:.1f}s\n")

    # Giai đoạn 4
    t0 = time.time()
    screened = screen_triplets()
    print(f"⏱️ Giai đoạn 4 hoàn thành trong: {time.time() - t0:.1f}s\n")

    # Giai đoạn 5
    t0 = time.time()
    num_lines = run_lineart_processing()
    print(f"⏱️ Giai đoạn 5 hoàn thành trong: {time.time() - t0:.1f}s\n")

    # Giai đoạn 6
    t0 = time.time()
    num_exported = export_dataset()
    print(f"⏱️ Giai đoạn 6 hoàn thành trong: {time.time() - t0:.1f}s\n")

    # Kiểm chứng
    t0 = time.time()
    val_results = run_validation()
    print(f"⏱️ Kiểm chứng hoàn thành trong: {time.time() - t0:.1f}s\n")

    total_time = time.time() - total_start_time
    print("=" * 80)
    print(f"🏁 TỔNG THỜI GIAN THỰC THI PIPELINE: {total_time:.1f}s ({total_time/60.0:.2f} phút)")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
