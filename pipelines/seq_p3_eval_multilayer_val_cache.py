#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kiểm chứng tps_cache_multilayer.npz trên toàn bộ 190 clips / 420 frames tập Val."""

import os
import sys
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATASET_DIR = os.path.join(
    REPO_ROOT, "Dataset", "sakuga_pipeline_sequence", "final_dataset",
    "sakuga_golden_dataset_sequences_tps_v2"
)
for p in [SCRIPT_DIR, os.path.join(REPO_ROOT, "kaggle_seq_p2_v33_ghost_purge")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from seq_p3_multi_layer_tps import reconstruct_multilayer_flow_batch_from_cache
from seq_p2_v33_ghost_purge_worker import (
    TPSInbetweenOriginal,
    load_tps_weights,
    solve_tps_backward_pair,
    tps_flow_from_theta_batch,
    evaluate_metrics_dual_regime,
)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    blended_ckpt = os.path.join(
        REPO_ROOT, "checkpoints", "Re-train", "Sequence 2 Blended", "checkpoints", "model_tps_blended.pt"
    )
    model = TPSInbetweenOriginal().to(device).eval()
    load_tps_weights(model, blended_ckpt, device=device, prefer_ema=True)

    df_va = pd.read_csv(os.path.join(DATASET_DIR, "val_sequences_indexed.csv"))
    cache = np.load(os.path.join(DATASET_DIR, "tps_cache_multilayer.npz"))
    val_dir = os.path.join(DATASET_DIR, "val")
    matches_dir = os.path.join(DATASET_DIR, "matches")

    rows = []
    for _, row in df_va.iterrows():
        cid = str(row["clip_id"])
        sl = int(row["seq_len"])
        cidx = int(row["cache_idx"])

        def _read(fk):
            with Image.open(os.path.join(val_dir, cid, f"{fk:06d}.png")) as im:
                return (255.0 - np.asarray(im.convert("L"), dtype=np.float32)) / 255.0

        im0 = _read(1)
        im1 = _read(sl)
        h, w = im0.shape
        hp, wp = ((h + 31) // 32) * 32, ((w + 31) // 32) * 32
        x0 = F.pad(torch.from_numpy(im0)[None, None], (0, wp - w, 0, hp - h)).to(device)
        x1 = F.pad(torch.from_numpy(im1)[None, None], (0, wp - w, 0, hp - h)).to(device)
        hw_t = torch.tensor([[h, w]], dtype=torch.int32, device=device)

        # Single-layer flow
        m_p = os.path.join(matches_dir, cid, f"match_000001_{sl:06d}.npy")
        if not os.path.exists(m_p):
            cands = sorted(glob.glob(os.path.join(matches_dir, cid, "*.npy")))
            m_p = cands[0] if cands else ""
        kps = np.load(m_p).astype(np.float32)
        c0_s, t01_s, c1_s, t10_s, vf_s = solve_tps_backward_pair(kps[0], kps[1], max_pts=128, lambd=0.01)
        cf01_s = tps_flow_from_theta_batch(torch.from_numpy(t01_s)[None].to(device), torch.from_numpy(c0_s)[None].to(device), torch.tensor([vf_s], device=device), hw_t, hp, wp, device, 2)
        cf10_s = tps_flow_from_theta_batch(torch.from_numpy(t10_s)[None].to(device), torch.from_numpy(c1_s)[None].to(device), torch.tensor([vf_s], device=device), hw_t, hp, wp, device, 2)

        # Multi-layer flow from cache
        cf01_m, cf10_m = reconstruct_multilayer_flow_batch_from_cache(
            torch.from_numpy(cache["c0_dyn"][cidx:cidx+1]).to(device),
            torch.from_numpy(cache["th01_dyn"][cidx:cidx+1]).to(device),
            torch.from_numpy(cache["c1_dyn"][cidx:cidx+1]).to(device),
            torch.from_numpy(cache["th10_dyn"][cidx:cidx+1]).to(device),
            torch.from_numpy(cache["valid_dyn"][cidx:cidx+1]).to(device),
            torch.from_numpy(cache["dyn_pts0"][cidx:cidx+1]).to(device),
            torch.from_numpy(cache["dyn_pts1"][cidx:cidx+1]).to(device),
            torch.from_numpy(cache["dyn_pts_len"][cidx:cidx+1]).to(device),
            torch.from_numpy(cache["static_pts0"][cidx:cidx+1]).to(device),
            torch.from_numpy(cache["static_pts1"][cidx:cidx+1]).to(device),
            torch.from_numpy(cache["num_static"][cidx:cidx+1]).to(device),
            torch.from_numpy(cache["num_clusters"][cidx:cidx+1]).to(device),
            hw_t, hp, wp, device, downscale_factor=2,
        )

        n_stat = int(cache["num_static"][cidx])
        n_clus = int(cache["num_clusters"][cidx])

        for k in range(1, sl - 1):
            fk = k + 1
            tv = float(k) / float(sl - 1)
            tt = torch.tensor([[[[tv]]]], dtype=torch.float32, device=device)
            gt_t = torch.from_numpy(_read(fk))[None, None].to(device)
            with torch.no_grad():
                pr_s, cw_s, *_ = model(x0, x1, tt, cf01_s, cf10_s)
                pr_m, cw_m, *_ = model(x0, x1, tt, cf01_m, cf10_m)
            m0_s = evaluate_metrics_dual_regime(cw_s[:, :, :h, :w], gt_t)
            m0_m = evaluate_metrics_dual_regime(cw_m[:, :, :h, :w], gt_t)
            nn_s = evaluate_metrics_dual_regime(pr_s[:, :, :h, :w], gt_t)
            nn_m = evaluate_metrics_dual_regime(pr_m[:, :, :h, :w], gt_t)
            rows.append({
                "clip_id": cid, "num_static": n_stat, "num_clusters": n_clus,
                "m0_s_psnr": float(m0_s["psnr"].item()), "m0_m_psnr": float(m0_m["psnr"].item()),
                "m0_s_ghost": float(m0_s["ghost"].item()), "m0_m_ghost": float(m0_m["ghost"].item()),
                "nn_s_psnr": float(nn_s["psnr"].item()), "nn_m_psnr": float(nn_m["psnr"].item()),
                "nn_s_ghost": float(nn_s["ghost"].item()), "nn_m_ghost": float(nn_m["ghost"].item()),
                "nn_s_wcd_dog": float(nn_s["wcd_dog"].item()), "nn_m_wcd_dog": float(nn_m["wcd_dog"].item()),
                "nn_s_skel": float(nn_s["skel_recall"].item()), "nn_m_skel": float(nn_m["skel_recall"].item()),
                "nn_s_dens": float(nn_s["density_ratio"].item()), "nn_m_dens": float(nn_m["density_ratio"].item()),
            })

    df = pd.DataFrame(rows)
    for tag, sub in [
        ("ALL_VAL_420_FRAMES", df),
        ("WITH_STATIC_>=10_PTS", df[df["num_static"] >= 10]),
        ("MULTI_CLUSTER_>=2", df[df["num_clusters"] >= 2]),
    ]:
        print(f"\n=== {tag} (N={len(sub)} frames) ===")
        print(sub.mean(numeric_only=True).round(4).to_string())


if __name__ == "__main__":
    main()
