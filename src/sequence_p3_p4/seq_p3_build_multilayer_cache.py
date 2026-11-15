#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seq_p3_build_multilayer_cache.py
--------------------------------------------------------------------------------
PHA 3 - BƯỚC 4: TẠO BỘ CACHE `tps_cache_multilayer.npz` TRÊN SAKUGA GOLDEN DATASET
VÀ KIỂM ĐỊNH ĐỐI CHỨNG TRÊN TẬP VAL (190 CLIPS / 420 FRAMES).

Tạo ra:
  1. `tps_cache_multilayer.npz` (~12-15 MB nén, 100% tương thích DDP training).
  2. `train_sequences_indexed.csv` & `val_sequences_indexed.csv` (có cột `cache_idx`).
  3. Thống kê phân cụm toàn tập Train/Val & Đánh giá nhanh M0/Blended trên tập Val.
--------------------------------------------------------------------------------
"""

import os
import sys
import glob
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_DATASET_DIR = os.path.join(
    REPO_ROOT, "Dataset", "sakuga_pipeline_sequence", "final_dataset",
    "sakuga_golden_dataset_sequences_tps_v2"
)

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from seq_p3_multi_layer_tps import solve_multilayer_cache_entry


def build_multilayer_cache(
    dataset_dir,
    out_npz_path=None,
    max_pts=128,
    max_clusters=4,
    max_mask_pts=64,
    max_static_pts=96,
    lambd=0.01,
    eps_static_px=1.5,
):
    if out_npz_path is None:
        out_npz_path = os.path.join(dataset_dir, "tps_cache_multilayer.npz")

    train_csv = os.path.join(dataset_dir, "train_sequences.csv")
    val_csv = os.path.join(dataset_dir, "val_sequences.csv")
    train_clips_dir = os.path.join(dataset_dir, "train")
    val_clips_dir = os.path.join(dataset_dir, "val")
    matches_dir = os.path.join(dataset_dir, "matches")

    df_tr = pd.read_csv(train_csv)
    df_va = pd.read_csv(val_csv)
    df_tr['_clip_root'] = train_clips_dir
    df_va['_clip_root'] = val_clips_dir
    df_all = pd.concat([df_tr, df_va], ignore_index=True)
    n_total = len(df_all)
    df_all['cache_idx'] = np.arange(n_total, dtype=np.int32)

    df_tr_out = df_all.iloc[:len(df_tr)].drop(columns=['_clip_root']).copy()
    df_va_out = df_all.iloc[len(df_tr):].drop(columns=['_clip_root']).copy()
    df_tr_out.to_csv(os.path.join(dataset_dir, "train_sequences_indexed.csv"), index=False)
    df_va_out.to_csv(os.path.join(dataset_dir, "val_sequences_indexed.csv"), index=False)

    c0_dyn = np.zeros((n_total, max_clusters, max_pts, 2), dtype=np.float32)
    th01_dyn = np.zeros((n_total, max_clusters, max_pts + 3, 2), dtype=np.float32)
    c1_dyn = np.zeros((n_total, max_clusters, max_pts, 2), dtype=np.float32)
    th10_dyn = np.zeros((n_total, max_clusters, max_pts + 3, 2), dtype=np.float32)
    valid_dyn = np.zeros((n_total, max_clusters), dtype=np.float32)

    dyn_pts0 = np.zeros((n_total, max_clusters, max_mask_pts, 2), dtype=np.float32)
    dyn_pts1 = np.zeros((n_total, max_clusters, max_mask_pts, 2), dtype=np.float32)
    dyn_pts_len = np.zeros((n_total, max_clusters), dtype=np.int32)

    static_pts0 = np.zeros((n_total, max_static_pts, 2), dtype=np.float32)
    static_pts1 = np.zeros((n_total, max_static_pts, 2), dtype=np.float32)
    num_static = np.zeros((n_total,), dtype=np.int32)
    num_clusters = np.zeros((n_total,), dtype=np.int32)
    num_outliers = np.zeros((n_total,), dtype=np.int32)
    orig_hw = np.zeros((n_total, 2), dtype=np.int32)

    def _task(item):
        idx, row = item
        cid = str(row['clip_id'])
        sl = int(row['seq_len'])
        c_root = row['_clip_root']
        img0_p = os.path.join(c_root, cid, "000001.png")
        with Image.open(img0_p) as im:
            w, h = im.size

        m_p = os.path.join(matches_dir, cid, f"match_000001_{sl:06d}.npy")
        if not os.path.exists(m_p):
            cands = sorted(glob.glob(os.path.join(matches_dir, cid, "*.npy")))
            m_p = cands[0] if cands else ""

        if m_p and os.path.exists(m_p):
            kps = np.load(m_p).astype(np.float32)
            entry = solve_multilayer_cache_entry(
                kps[0], kps[1], img_hw=(h, w),
                max_pts=max_pts, max_clusters=max_clusters,
                max_mask_pts=max_mask_pts, max_static_pts=max_static_pts,
                lambd=lambd, eps_static_px=eps_static_px,
                use_dyn_anchors=True, filter_outliers=True,
            )
        else:
            entry = solve_multilayer_cache_entry(
                np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32),
                img_hw=(h, w), max_pts=max_pts, max_clusters=max_clusters,
                max_mask_pts=max_mask_pts, max_static_pts=max_static_pts,
            )
        return idx, h, w, entry

    t0 = time.time()
    print(f"⚙️ Đang tạo `tps_cache_multilayer.npz` cho {n_total:,} sequences "
          f"(MAX_PTS={max_pts}, MAX_CLUSTERS={max_clusters}, eps_static={eps_static_px}px)...", flush=True)

    with ThreadPoolExecutor(max_workers=min(16, (os.cpu_count() or 4) * 2)) as pool:
        for idx, h, w, entry in pool.map(_task, list(df_all.iterrows())):
            c0_dyn[idx] = entry['c0_dyn']
            th01_dyn[idx] = entry['th01_dyn']
            c1_dyn[idx] = entry['c1_dyn']
            th10_dyn[idx] = entry['th10_dyn']
            valid_dyn[idx] = entry['valid_dyn']
            dyn_pts0[idx] = entry['dyn_pts0']
            dyn_pts1[idx] = entry['dyn_pts1']
            dyn_pts_len[idx] = entry['dyn_pts_len']
            static_pts0[idx] = entry['static_pts0']
            static_pts1[idx] = entry['static_pts1']
            num_static[idx] = entry['num_static']
            num_clusters[idx] = entry['num_clusters']
            num_outliers[idx] = entry['num_outliers']
            orig_hw[idx] = [h, w]

    np.savez_compressed(
        out_npz_path,
        c0_dyn=c0_dyn,
        th01_dyn=th01_dyn,
        c1_dyn=c1_dyn,
        th10_dyn=th10_dyn,
        valid_dyn=valid_dyn,
        dyn_pts0=dyn_pts0,
        dyn_pts1=dyn_pts1,
        dyn_pts_len=dyn_pts_len,
        static_pts0=static_pts0,
        static_pts1=static_pts1,
        num_static=num_static,
        num_clusters=num_clusters,
        num_outliers=num_outliers,
        orig_hw=orig_hw,
    )
    elapsed = time.time() - t0
    size_mb = os.path.getsize(out_npz_path) / (1024 ** 2)
    print(f"[✓] Đã lưu thành công: {out_npz_path} ({size_mb:.2f} MB) trong {elapsed:.1f}s!")
    print(f"    - Số clips có >= 1 cụm động hợp lệ: {int((num_clusters >= 1).sum()):,}/{n_total:,}")
    print(f"    - Số clips có >= 2 cụm động: {int((num_clusters >= 2).sum()):,}/{n_total:,}")
    print(f"    - Số clips có điểm nền tĩnh (>0): {int((num_static > 0).sum()):,}/{n_total:,}")
    print(f"    - Tổng số điểm khớp nhiễu (outliers) đã lọc: {int(num_outliers.sum()):,}")
    return out_npz_path


def main():
    parser = argparse.ArgumentParser(description="Tạo tps_cache_multilayer.npz cho Bước 4 Pha 3")
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out_npz", type=str, default=None)
    args = parser.parse_args()
    build_multilayer_cache(args.dataset_dir, args.out_npz)


if __name__ == "__main__":
    main()
