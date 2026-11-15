#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_pha3_vs_pha2_blended.py
--------------------------------------------------------------------------------
BỘ BENCHMARK ĐỐI ĐẦU TRỰC TIẾP (ZERO-BIAS HEAD-TO-HEAD):
    Pha 2 Blended (`model_tps_blended.pt` - Checkpoint đầu vào)
        vs.
    Pha 3 Best (Ep 1 - `Sequence 2 pha 3/checkpoints/model_tps_finetuned_best.pt`)
    Pha 3 Last (Ep 3 - `Sequence 2 pha 3/checkpoints/model_tps_finetuned_last.pt`)

Chạy cùng 1 tầng Coarse TPS Đa lớp (`Multi-Layer Clustered TPS`) trên cả 3 bài kiểm định:
  1. Benchmark #1: Tập chuỗi cách ly `sakuga_golden_benchmark_sequences_v2` (198 clips / 412 frames / 149 scenes)
     + Kiểm định thống kê Wilcoxon Signed-Rank theo scene (Pha3_Best vs Pha2_Blended & Pha3_Last vs Pha2_Blended).
     + Xuất Top-3 cải thiện mạnh nhất & Top-3 giảm chất lượng mạnh nhất (5 cột trực quan).
  2. Benchmark #2: Chuỗi 24 frames 540p thực tế (`data/lineart_experiments/test` - 17 in-betweens)
     + Xuất bảng từng frame, bảng tổng hợp (bao gồm Temp Diff MAE & Temp BG Flicker),
       ảnh Side-by-Side 4 cột [Input | Pha 2 Blended | Pha 3 Best | GT], Error Heatmaps,
       Montage 6 phân đoạn và GIF/MP4 đối đầu.
  3. Benchmark #3: Defense Geometry Suite (Case 1..4)
     + Xuất bảng chi tiết & lưới đối chiếu từng Case [M0 | Pha 2 Blended | Pha 3 Best | Pha 3 Last | GT].
--------------------------------------------------------------------------------
"""

import os
import sys
import json
import time
import glob
import math
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

try:
    import cv2
    cv2.setNumThreads(0)
except ImportError:
    cv2 = None

try:
    import imageio
except ImportError:
    imageio = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
BENCH1_DIR = os.path.join(
    REPO_ROOT, "Dataset", "sakuga_pipeline_sequence", "final_dataset",
    "sakuga_golden_benchmark_sequences_v2"
)
LINEART_TEST_DIR = os.path.join(REPO_ROOT, "data", "lineart_experiments", "test")
GLUESTICK_NPZ = os.path.join(REPO_ROOT, "data_samples", "caches", "lineart_test_gluestick_matches.npz")

for _p in (SCRIPT_DIR, BENCH1_DIR, os.path.join(REPO_ROOT, "training", "03_sequence", "p2_v33_ghost_purge"), os.path.join(REPO_ROOT, "src", "sequence_p3_p4")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from seq_p3_multi_layer_tps import solve_multi_layer_tps
from seq_dual_benchmark_suite import (
    TPSInbetweenOriginal,
    load_tps_weights,
    evaluate_metrics_dual_regime,
    extract_hybrid_contour_matches,
    load_ink_image,
    ink_to_u8_gray,
    compute_wilcoxon_p,
)


def _render_5col_panel(im0_u8, im1_u8, m0_u8, base_u8, p3_u8, gt_u8, title_str, stats_base, stats_p3):
    h, w = gt_u8.shape
    overlay = (0.5 * im0_u8.astype(np.float32) + 0.5 * im1_u8.astype(np.float32)).astype(np.uint8)
    cols = [overlay, m0_u8, base_u8, p3_u8, gt_u8]
    titles = [
        ("Overlay I0/I1", (210, 210, 210)),
        ("M0 Multi-TPS", (180, 180, 180)),
        (f"Pha2 Blended ({stats_base})", (100, 175, 255)),
        (f"Pha3 Best ({stats_p3})", (90, 240, 120)),
        ("Ground Truth", (245, 245, 245)),
    ]
    head_h, foot_h = 34, 28
    canvas = np.full((head_h + h + foot_h, w * 5, 3), 26, dtype=np.uint8)
    for i, (arr, (lbl, clr)) in enumerate(zip(cols, titles)):
        bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        canvas[head_h:head_h + h, i * w:(i + 1) * w] = bgr
        cv2.putText(canvas, lbl, (i * w + 8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.50, clr, 2, cv2.LINE_AA)
        if i > 0:
            cv2.line(canvas, (i * w, 0), (i * w, head_h + h + foot_h), (70, 70, 70), 2)
    cv2.putText(canvas, title_str, (12, head_h + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def run_bench1_head_to_head(models_dict, device, out_dir):
    print("\n" + "=" * 90)
    print("🔬 [BENCHMARK #1]: 198 CLIPS / 412 FRAMES / 149 SCENES (PHA 2 BLENDED vs PHA 3)")
    print("=" * 90, flush=True)
    os.makedirs(out_dir, exist_ok=True)

    df_bench = pd.read_csv(os.path.join(BENCH1_DIR, "benchmark_sequences.csv"))
    clips_dir = os.path.join(BENCH1_DIR, "clips")
    matches_dir = os.path.join(BENCH1_DIR, "matches")

    per_sample_rows = []
    t0 = time.time()

    for row_idx, row in df_bench.iterrows():
        cid = str(row['clip_id'])
        sid = str(row['scene_id'])
        seq_len = int(row['seq_len'])
        disp = float(row.get('mean_disp', 0.0))

        im0 = load_ink_image(os.path.join(clips_dir, cid, "000001.png"))
        im1 = load_ink_image(os.path.join(clips_dir, cid, f"{seq_len:06d}.png"))
        h, w = im0.shape
        h_pad, w_pad = ((h + 31) // 32) * 32, ((w + 31) // 32) * 32
        x0 = F.pad(torch.from_numpy(im0)[None, None], (0, w_pad - w, 0, h_pad - h)).to(device)
        x1 = F.pad(torch.from_numpy(im1)[None, None], (0, w_pad - w, 0, h_pad - h)).to(device)

        m_cands = sorted(glob.glob(os.path.join(matches_dir, cid, "*.npy")))
        if m_cands:
            kps = np.load(m_cands[0]).astype(np.float32)
            cf01, cf10, _ = solve_multi_layer_tps(
                kps[0], kps[1], (h, w), h_pad, w_pad, device,
                direct_t_flows=False, max_pts=128, lambd=0.01,
                downscale_factor=2, eps_static_px=1.5,
                use_dyn_anchors=True, transition_width_px=22.0, falloff_radius_px=105.0
            )
        else:
            cf01 = torch.zeros((1, 2, h_pad, w_pad), dtype=torch.float32, device=device)
            cf10 = torch.zeros((1, 2, h_pad, w_pad), dtype=torch.float32, device=device)

        for k in range(1, seq_len - 1):
            fk = k + 1
            tv = float(k) / float(seq_len - 1)
            tt = torch.tensor([[[[tv]]]], dtype=torch.float32, device=device)
            gt_ink = torch.from_numpy(load_ink_image(os.path.join(clips_dir, cid, f"{fk:06d}.png")))[None, None].to(device)

            rec = {
                'clip_id': cid, 'scene_id': sid, 'seq_len': seq_len, 'frame_k': fk,
                't_val': round(tv, 4), 'mean_disp': disp, 'orig_w': w, 'orig_h': h
            }
            cw_done = False
            for m_name, mdl in models_dict.items():
                with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
                    pr_p, cw_p, *_ = mdl(x0.float(), x1.float(), tt, cf01.float(), cf10.float())
                if not cw_done:
                    mm0 = evaluate_metrics_dual_regime(cw_p[:, :, :h, :w], gt_ink)
                    for mk, mv in mm0.items():
                        rec[f"M0_{mk}"] = float(mv.item())
                    cw_done = True
                mm = evaluate_metrics_dual_regime(pr_p[:, :, :h, :w], gt_ink)
                for mk, mv in mm.items():
                    rec[f"{m_name}_{mk}"] = float(mv.item())
            per_sample_rows.append(rec)

        if (row_idx + 1) % 50 == 0 or (row_idx + 1) == len(df_bench):
            print(f"   ... {row_idx + 1}/{len(df_bench)} clips ({time.time() - t0:.1f}s)", flush=True)

    df_res = pd.DataFrame(per_sample_rows)
    df_res.to_csv(os.path.join(out_dir, "bench1_per_sample.csv"), index=False)

    all_tags = ['M0'] + list(models_dict.keys())
    metric_cols = ['psnr', 'ssim', 'ie_mae', 'pwcr', 'rgd', 'ghost', 'dark_ghost', 'wcd', 'wcd_dog',
                   'semd', 'skel_recall', 'tip_recall', 'recall', 'precision', 'f1', 'density_ratio']
    summary_rows = []
    for tag in all_tags:
        sr = {'model_tag': tag, 'frames': len(df_res), 'scenes': df_res['scene_id'].nunique()}
        for mc in metric_cols:
            sr[mc] = float(df_res[f"{tag}_{mc}"].mean())
        summary_rows.append(sr)
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(os.path.join(out_dir, "bench1_overall_summary.csv"), index=False)

    print("\n📊 KẾT QUẢ BENCHMARK #1 (412 FRAMES / 149 SCENES - MULTI-LAYER TPS - FP32):")
    print(df_sum[['model_tag', 'psnr', 'ssim', 'ie_mae', 'ghost', 'wcd', 'wcd_dog', 'semd', 'skel_recall', 'tip_recall', 'f1', 'density_ratio']].round(4).to_string(index=False))

    # Kiểm định thống kê Wilcoxon theo scene (Pha3_Best vs Pha2_Blended và Pha3_Last vs Pha2_Blended)
    scene_df = df_res.groupby('scene_id').mean(numeric_only=True)
    h2h_audit = {}
    metrics_spec = [
        ('psnr', True), ('ssim', True), ('ie_mae', False), ('ghost', False), ('dark_ghost', False),
        ('wcd', False), ('wcd_dog', False), ('semd', False),
        ('skel_recall', True), ('tip_recall', True), ('f1', True), ('density_ratio', None)
    ]
    for cmp_tag in ['Pha3_Best', 'Pha3_Last']:
        h2h_audit[f"{cmp_tag}_vs_Pha2_Blended"] = {}
        print(f"\n⚖️ ĐỐI ĐẦU THEO SCENE (149 SCENES): {cmp_tag} vs Pha2_Blended")
        for mc, better_higher in metrics_spec:
            c_a, c_b = f"Pha2_Blended_{mc}", f"{cmp_tag}_{mc}"
            v_a, v_b = df_res[c_a].values, df_res[c_b].values
            if better_higher is True:
                wr = float(np.mean(v_b > v_a)) * 100.0
            elif better_higher is False:
                wr = float(np.mean(v_b < v_a)) * 100.0
            else:
                wr = float(np.mean(np.abs(v_b - 1.0) < np.abs(v_a - 1.0))) * 100.0
            pval = compute_wilcoxon_p(scene_df[c_b].values, scene_df[c_a].values)
            delta = float(np.mean(v_b) - np.mean(v_a))
            h2h_audit[f"{cmp_tag}_vs_Pha2_Blended"][mc] = {
                'pha2_blended': float(np.mean(v_a)),
                'pha3': float(np.mean(v_b)),
                'delta': delta,
                'win_rate_pct': round(wr, 2),
                'wilcoxon_p': pval,
            }
            print(f"   - {mc:13s}: Blended={np.mean(v_a):.4f} -> {cmp_tag}={np.mean(v_b):.4f} "
                  f"(Delta={delta:+.4f} | WinRate={wr:5.1f}% | p={pval:.2e})")

    with open(os.path.join(out_dir, "bench1_h2h_vs_blended.json"), "w", encoding="utf-8") as f:
        json.dump(h2h_audit, f, indent=2, ensure_ascii=False)

    # Xuất Top-3 Cải thiện & Top-3 Giảm sút (theo delta_wcd_dog giữa Pha3_Best và Pha2_Blended)
    if cv2 is not None:
        df_res['delta_wcd_dog'] = df_res['Pha3_Best_wcd_dog'] - df_res['Pha2_Blended_wcd_dog']
        best3 = df_res.sort_values('delta_wcd_dog', ascending=True).head(3)
        worst3 = df_res.sort_values('delta_wcd_dog', ascending=False).head(3)
        vis_dir = os.path.join(out_dir, "audit_cases_top3")
        os.makedirs(vis_dir, exist_ok=True)
        for grp_lbl, sub_df in [("TOP3_PHA3_BETTER", best3), ("TOP3_PHA3_WORSE", worst3)]:
            panels = []
            for _, sr in sub_df.iterrows():
                cid, fk, sl = sr['clip_id'], int(sr['frame_k']), int(sr['seq_len'])
                im0_l = load_ink_image(os.path.join(clips_dir, cid, "000001.png"))
                im1_l = load_ink_image(os.path.join(clips_dir, cid, f"{sl:06d}.png"))
                gt_l = load_ink_image(os.path.join(clips_dir, cid, f"{fk:06d}.png"))
                hh, ww = im0_l.shape
                hp, wp = ((hh + 31) // 32) * 32, ((ww + 31) // 32) * 32
                x0p = F.pad(torch.from_numpy(im0_l)[None, None], (0, wp - ww, 0, hp - hh)).to(device)
                x1p = F.pad(torch.from_numpy(im1_l)[None, None], (0, wp - ww, 0, hp - hh)).to(device)
                mc = sorted(glob.glob(os.path.join(matches_dir, cid, "*.npy")))
                if mc:
                    kps = np.load(mc[0]).astype(np.float32)
                    cf01_, cf10_, _ = solve_multi_layer_tps(
                        kps[0], kps[1], (hh, ww), hp, wp, device,
                        direct_t_flows=False, max_pts=128, lambd=0.01,
                        downscale_factor=2, eps_static_px=1.5,
                        use_dyn_anchors=True, transition_width_px=22.0, falloff_radius_px=105.0
                    )
                else:
                    cf01_ = torch.zeros((1, 2, hp, wp), dtype=torch.float32, device=device)
                    cf10_ = torch.zeros((1, 2, hp, wp), dtype=torch.float32, device=device)
                tt = torch.tensor([[[[float(sr['t_val'])]]]], dtype=torch.float32, device=device)
                with torch.no_grad():
                    pr_b, cw_, *_ = models_dict['Pha2_Blended'](x0p, x1p, tt, cf01_, cf10_)
                    pr_3, *_ = models_dict['Pha3_Best'](x0p, x1p, tt, cf01_, cf10_)
                panels.append(_render_5col_panel(
                    ink_to_u8_gray(x0p[:, :, :hh, :ww]), ink_to_u8_gray(x1p[:, :, :hh, :ww]),
                    ink_to_u8_gray(cw_[:, :, :hh, :ww]), ink_to_u8_gray(pr_b[:, :, :hh, :ww]),
                    ink_to_u8_gray(pr_3[:, :, :hh, :ww]), (255.0 - gt_l * 255.0).clip(0, 255).astype(np.uint8),
                    f"[{grp_lbl}] {cid} (N={sl}, k={fk}, t={sr['t_val']:.2f}) | dWCD_DoG={sr['delta_wcd_dog']:+.4f}",
                    f"WCD={sr['Pha2_Blended_wcd_dog']:.3f}|Sk={sr['Pha2_Blended_skel_recall']*100:.0f}%|Gh={sr['Pha2_Blended_ghost']:.0f}",
                    f"WCD={sr['Pha3_Best_wcd_dog']:.3f}|Sk={sr['Pha3_Best_skel_recall']*100:.0f}%|Gh={sr['Pha3_Best_ghost']:.0f}"
                ))
            max_pw = max(p.shape[1] for p in panels)
            norm_panels = [np.pad(p, ((0, 0), (0, max_pw - p.shape[1]), (0, 0)), constant_values=26) for p in panels]
            cv2.imwrite(os.path.join(vis_dir, f"{grp_lbl}.png"), np.vstack(norm_panels))

    return df_sum, h2h_audit


def run_bench2_head_to_head(models_dict, device, out_dir):
    print("\n" + "=" * 90)
    print("🎬 [BENCHMARK #2]: 24 FRAMES 540P THỰC TẾ (PHA 2 BLENDED vs PHA 3 BEST & LAST)")
    print("=" * 90, flush=True)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(LINEART_TEST_DIR, "continuous_keyframes_spec.json"), "r", encoding="utf-8") as f:
        spec = json.load(f)
    frames_spec = spec["frames"]
    key_indices = sorted(spec["key_indices"])
    gt_imgs_u8 = [np.asarray(Image.open(os.path.join(LINEART_TEST_DIR, it["filename"])).convert("L"), dtype=np.uint8) for it in frames_spec]
    h, w = gt_imgs_u8[0].shape
    h_pad, w_pad = ((h + 31) // 32) * 32, ((w + 31) // 32) * 32
    gs_cache = dict(np.load(GLUESTICK_NPZ))

    seg_data = {}
    for i in range(len(key_indices) - 1):
        ka, kb = key_indices[i], key_indices[i + 1]
        kps = gs_cache[f"seg_{ka}_{kb}"].astype(np.float32)
        ima, imb = gt_imgs_u8[ka], gt_imgs_u8[kb]
        x0_p = F.pad(torch.from_numpy((255.0 - ima.astype(np.float32)) / 255.0)[None, None], (0, w_pad - w, 0, h_pad - h)).to(device)
        x1_p = F.pad(torch.from_numpy((255.0 - imb.astype(np.float32)) / 255.0)[None, None], (0, w_pad - w, 0, h_pad - h)).to(device)
        cf01, cf10, _ = solve_multi_layer_tps(
            kps[0], kps[1], (h, w), h_pad, w_pad, device,
            direct_t_flows=False, max_pts=128, lambd=0.01,
            downscale_factor=2, eps_static_px=1.5,
            use_dyn_anchors=True, transition_width_px=22.0, falloff_radius_px=105.0
        )
        seg_data[(ka, kb)] = {'x0': x0_p, 'x1': x1_p, 'cf01': cf01, 'cf10': cf10}

    tags = ['M0'] + list(models_dict.keys())
    reconstructed = {t: [img.copy() for img in gt_imgs_u8] for t in tags}
    per_frame_rows = []

    for item in frames_spec:
        f_idx = int(item["frame_idx"])
        if item["type"] != "INBETWEEN":
            continue
        ka, kb, tv = int(item["ref_key_a"]), int(item["ref_key_b"]), float(item["t_val"])
        seg = seg_data[(ka, kb)]
        gt_ink = torch.from_numpy((255.0 - gt_imgs_u8[f_idx].astype(np.float32)) / 255.0)[None, None].to(device)
        tt = torch.tensor([[[[tv]]]], dtype=torch.float32, device=device)

        row = {'frame_idx': f_idx, 'filename': item['filename'], 'segment': f"{ka}->{kb}", 'delta_f': kb - ka, 't_val': round(tv, 4)}
        cw_done = False
        for m_name, mdl in models_dict.items():
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
                pr_p, cw_p, *_ = mdl(seg['x0'].float(), seg['x1'].float(), tt, seg['cf01'].float(), seg['cf10'].float())
            if not cw_done:
                cw_crop = cw_p[:, :, :h, :w]
                reconstructed['M0'][f_idx] = ink_to_u8_gray(cw_crop)
                mm0 = evaluate_metrics_dual_regime(cw_crop, gt_ink)
                for mk, mv in mm0.items():
                    row[f"M0_{mk}"] = float(mv.item())
                cw_done = True
            pr_crop = pr_p[:, :, :h, :w]
            reconstructed[m_name][f_idx] = ink_to_u8_gray(pr_crop)
            mm = evaluate_metrics_dual_regime(pr_crop, gt_ink)
            for mk, mv in mm.items():
                row[f"{m_name}_{mk}"] = float(mv.item())
        per_frame_rows.append(row)

    for row in per_frame_rows:
        fi = int(row['frame_idx'])
        g_cur = (255.0 - gt_imgs_u8[fi].astype(np.float32)) / 255.0
        g_prev = (255.0 - gt_imgs_u8[fi - 1].astype(np.float32)) / 255.0
        both_bg = (g_cur < 0.04) & (g_prev < 0.04)
        dg = g_cur - g_prev
        for t in tags:
            p_cur = (255.0 - reconstructed[t][fi].astype(np.float32)) / 255.0
            p_prev = (255.0 - reconstructed[t][fi - 1].astype(np.float32)) / 255.0
            dp = p_cur - p_prev
            row[f"{t}_temp_diff_mae"] = float(np.mean(np.abs(dp - dg)) * 255.0)
            row[f"{t}_temp_bg_flicker"] = float(np.sum(np.abs(dp) * both_bg))

    df_frames = pd.DataFrame(per_frame_rows)
    df_frames.to_csv(os.path.join(out_dir, "bench2_per_frame_vs_blended.csv"), index=False)

    summary_rows = []
    for grp_name, sub_df in [("ALL_17_INBETWEENS", df_frames)] + [(f"delta_f={d}", df_frames[df_frames['delta_f'] == d]) for d in sorted(df_frames['delta_f'].unique())]:
        for t in tags:
            sr = {'group': grp_name, 'model_tag': t, 'frames': len(sub_df)}
            for mc in ['psnr', 'ssim', 'ie_mae', 'ghost', 'dark_ghost', 'temp_diff_mae', 'temp_bg_flicker',
                       'wcd', 'wcd_dog', 'semd', 'skel_recall', 'tip_recall', 'f1', 'density_ratio', 'pwcr', 'rgd']:
                sr[mc] = float(sub_df[f"{t}_{mc}"].mean())
            summary_rows.append(sr)
    df_sum2 = pd.DataFrame(summary_rows)
    df_sum2.to_csv(os.path.join(out_dir, "bench2_summary_vs_blended.csv"), index=False)

    print("\n📊 KẾT QUẢ BENCHMARK #2 (ALL 17 IN-BETWEEN FRAMES - MULTI-LAYER TPS):")
    cols_show = ['model_tag', 'psnr', 'ssim', 'ie_mae', 'ghost', 'temp_diff_mae', 'temp_bg_flicker', 'wcd_dog', 'skel_recall', 'f1', 'density_ratio']
    print(df_sum2[df_sum2['group'] == 'ALL_17_INBETWEENS'][cols_show].round(4).to_string(index=False))

    # Xuất ảnh Side-by-Side 4 cột [Input | Pha2 Blended | Pha3 Best | GT], Error Heatmaps, Montage & GIF/MP4
    if cv2 is not None:
        sbs_dir = os.path.join(out_dir, "frames_side_by_side")
        heat_dir = os.path.join(out_dir, "error_heatmaps")
        os.makedirs(sbs_dir, exist_ok=True)
        os.makedirs(heat_dir, exist_ok=True)
        gif_frames, montage_rows = [], []

        for item in frames_spec:
            idx = int(item['frame_idx'])
            is_k = (item['type'] == 'KEYFRAME')
            gt_u8 = gt_imgs_u8[idx]
            b_u8 = reconstructed['Pha2_Blended'][idx]
            p3_u8 = reconstructed['Pha3_Best'][idx]

            if is_k:
                ov_u8 = gt_u8
                sb, sp3 = "KEYFRAME", "KEYFRAME"
            else:
                ka, kb = int(item['ref_key_a']), int(item['ref_key_b'])
                ov_u8 = (0.5 * gt_imgs_u8[ka].astype(np.float32) + 0.5 * gt_imgs_u8[kb].astype(np.float32)).astype(np.uint8)
                fr = df_frames[df_frames['frame_idx'] == idx].iloc[0]
                sb = f"PSNR:{fr['Pha2_Blended_psnr']:.2f}|Sk:{fr['Pha2_Blended_skel_recall']*100:.0f}%|Gh:{fr['Pha2_Blended_ghost']:.0f}"
                sp3 = f"PSNR:{fr['Pha3_Best_psnr']:.2f}|Sk:{fr['Pha3_Best_skel_recall']*100:.0f}%|Gh:{fr['Pha3_Best_ghost']:.0f}"

                d_b = np.clip(np.abs(b_u8.astype(np.float32) - gt_u8.astype(np.float32)) * 2.5, 0, 255).astype(np.uint8)
                d_3 = np.clip(np.abs(p3_u8.astype(np.float32) - gt_u8.astype(np.float32)) * 2.5, 0, 255).astype(np.uint8)
                hm_pair = np.hstack([cv2.applyColorMap(d_b, cv2.COLORMAP_JET), cv2.applyColorMap(d_3, cv2.COLORMAP_JET)])
                hm_head = np.full((32, w * 2, 3), 24, dtype=np.uint8)
                cv2.putText(hm_head, f"PHA2 BLENDED ERR [MAE={fr['Pha2_Blended_ie_mae']:.2f}]", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 175, 255), 2, cv2.LINE_AA)
                cv2.putText(hm_head, f"PHA3 BEST ERR [MAE={fr['Pha3_Best_ie_mae']:.2f}]", (w + 12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 240, 120), 2, cv2.LINE_AA)
                cv2.imwrite(os.path.join(heat_dir, f"heatmap_{idx:02d}_{item['filename']}"), np.vstack([hm_head, hm_pair]))

            head_h = 34
            canvas = np.full((h + head_h, w * 4, 3), 24, dtype=np.uint8)
            cols = [
                (ov_u8, f"INPUT [{item['filename']}]", (210, 210, 210)),
                (b_u8, f"PHA2 BLENDED ({sb})", (100, 175, 255)),
                (p3_u8, f"PHA3 BEST ({sp3})", (80, 240, 120)),
                (gt_u8, "GROUND TRUTH", (245, 245, 245)),
            ]
            for ci, (arr, lbl, colr) in enumerate(cols):
                canvas[head_h:, ci * w:(ci + 1) * w] = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                cv2.putText(canvas, lbl, (ci * w + 10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.46, colr, 2, cv2.LINE_AA)
                if ci > 0:
                    cv2.line(canvas, (ci * w, 0), (ci * w, h + head_h), (70, 70, 70), 2)
            cv2.imwrite(os.path.join(sbs_dir, f"frame_{idx:02d}_{item['filename']}"), canvas)
            half_frame = cv2.resize(canvas, (w * 2, (h + head_h) // 2))
            gif_frames.append(half_frame[:, :, ::-1])
            if (not is_k) and (idx in {2, 5, 9, 12, 17, 21}):
                montage_rows.append(half_frame)

        if montage_rows:
            cv2.imwrite(os.path.join(out_dir, "montage_6segments_Blended_vs_Pha3.png"), np.vstack(montage_rows))
        if imageio is not None:
            imageio.mimsave(os.path.join(out_dir, "lineart_test_Blended_vs_Pha3.gif"), gif_frames, duration=0.14, loop=0)
        mp4_path = os.path.join(out_dir, "lineart_test_Blended_vs_Pha3.mp4")
        vw = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*'mp4v'), 8.0, (gif_frames[0].shape[1], gif_frames[0].shape[0]))
        for gf in gif_frames:
            vw.write(gf[:, :, ::-1])
        vw.release()
    return df_sum2


def run_bench3_head_to_head(models_dict, device, out_dir):
    print("\n" + "=" * 90)
    print("📐 [BENCHMARK #3]: DEFENSE GEOMETRY SUITE (PHA 2 BLENDED vs PHA 3 BEST & LAST)")
    print("=" * 90, flush=True)
    os.makedirs(out_dir, exist_ok=True)
    H, W = 384, 640
    t_list = [0.25, 0.50, 0.75]

    def _draw_ellipse_case(cx0, cy0, rx0, ry0, th0, cx1, cy1, rx1, ry1, th1, static_line=False):
        def _fn(t):
            im = np.zeros((H, W), dtype=np.float32)
            cx = int(round((1.0 - t) * cx0 + t * cx1))
            cy = int(round((1.0 - t) * cy0 + t * cy1))
            rx = int(round((1.0 - t) * rx0 + t * rx1))
            ry = int(round((1.0 - t) * ry0 + t * ry1))
            th = max(2, int(round((1.0 - t) * th0 + t * th1)))
            cv2.ellipse(im, (cx, cy), (rx, ry), 0, 0, 360, 1.0, th, lineType=cv2.LINE_AA)
            if static_line:
                cv2.line(im, (80, 330), (560, 330), 1.0, 4, lineType=cv2.LINE_AA)
            return im
        return _fn

    cases = [
        ("ShrinkingCircle_75to32", _draw_ellipse_case(120, 192, 75, 75, 5, 520, 192, 32, 32, 4, False)),
        ("ExpandingCircle_30to70", _draw_ellipse_case(120, 192, 30, 30, 4, 520, 192, 70, 70, 5, False)),
        ("ShrinkingCircle_StaticGround", _draw_ellipse_case(120, 180, 75, 75, 5, 520, 180, 32, 32, 4, True)),
        ("SquashOval90x40_to_Circle45", _draw_ellipse_case(140, 192, 90, 40, 4, 490, 192, 45, 45, 4, False)),
    ]

    tags = ['M0'] + list(models_dict.keys())
    rows = []

    for cname, gt_fn in cases:
        im0, im1 = gt_fn(0.0), gt_fn(1.0)
        k0, k1 = extract_hybrid_contour_matches(im0, im1, n_pts_per_comp=32)
        x0 = torch.from_numpy(im0)[None, None].to(device)
        x1 = torch.from_numpy(im1)[None, None].to(device)
        vis_rows = {t: [] for t in tags}
        gt_imgs = []

        for tv in t_list:
            gt_im = gt_fn(tv)
            gt_imgs.append(gt_im)
            gt_t = torch.from_numpy(gt_im)[None, None].to(device)
            cf0t, cf1t, _ = solve_multi_layer_tps(
                k0, k1, (H, W), H, W, device,
                t_val=tv, direct_t_flows=True, max_pts=128, lambd=0.01,
                downscale_factor=2, eps_static_px=1.5, use_dyn_anchors=False,
                transition_width_px=18.0, falloff_radius_px=160.0
            )
            tt = torch.tensor([[[[tv]]]], dtype=torch.float32, device=device)
            rec = {'case': cname, 't_val': tv}
            cw_done = False
            for m_name, mdl in models_dict.items():
                with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
                    pr, cw, *_ = mdl(x0.float(), x1.float(), tt, cf0t.float(), cf1t.float(), direct_t_flows=True)
                if not cw_done:
                    vis_rows['M0'].append(ink_to_u8_gray(cw))
                    mm0 = evaluate_metrics_dual_regime(cw, gt_t)
                    for mk, mv in mm0.items():
                        rec[f"M0_{mk}"] = float(mv.item())
                    cw_done = True
                vis_rows[m_name].append(ink_to_u8_gray(pr))
                mm = evaluate_metrics_dual_regime(pr, gt_t)
                for mk, mv in mm.items():
                    rec[f"{m_name}_{mk}"] = float(mv.item())
            rows.append(rec)

        panel_rows = []
        u0 = (255.0 - im0 * 255.0).clip(0, 255).astype(np.uint8)
        u1 = (255.0 - im1 * 255.0).clip(0, 255).astype(np.uint8)
        for tg in tags:
            strip = np.hstack([cv2.cvtColor(u, cv2.COLOR_GRAY2BGR) for u in ([u0] + vis_rows[tg] + [u1])])
            cv2.putText(strip, tg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 120, 220), 2, cv2.LINE_AA)
            panel_rows.append(strip)
        gt_strip = np.hstack([cv2.cvtColor(u, cv2.COLOR_GRAY2BGR) for u in ([u0] + [(255.0 - g * 255.0).clip(0, 255).astype(np.uint8) for g in gt_imgs] + [u1])])
        cv2.putText(gt_strip, "GroundTruth", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 150, 40), 2, cv2.LINE_AA)
        panel_rows.append(gt_strip)
        cv2.imwrite(os.path.join(out_dir, f"{cname}_Blended_vs_Pha3_grid.png"), np.vstack(panel_rows))

    df_cases = pd.DataFrame(rows)
    df_cases.to_csv(os.path.join(out_dir, "bench3_per_frame_vs_blended.csv"), index=False)

    sum_rows = []
    for tg in tags:
        r = {'model_tag': tg, 'frames': len(df_cases)}
        for mc in ['psnr', 'ssim', 'ie_mae', 'wcd_dog', 'f1', 'density_ratio', 'ghost']:
            r[mc] = float(df_cases[f"{tg}_{mc}"].mean())
        sum_rows.append(r)
    df_sum3 = pd.DataFrame(sum_rows)
    df_sum3.to_csv(os.path.join(out_dir, "bench3_summary_vs_blended.csv"), index=False)
    print("\n📐 KẾT QUẢ BENCHMARK #3 (12 FRAMES HÌNH HỌC - MULTI-LAYER TPS):")
    print(df_sum3.round(4).to_string(index=False))
    return df_sum3


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_root = os.path.join(
        REPO_ROOT, "checkpoints", "Re-train", "Sequence 2 pha 3", "benchmark_vs_pha2_blended"
    )
    os.makedirs(out_root, exist_ok=True)

    ckpts = {
        'Pha2_Blended': os.path.join(REPO_ROOT, "checkpoints", "Re-train", "Sequence 2 Blended", "checkpoints", "model_tps_blended.pt"),
        'Pha3_Best': os.path.join(REPO_ROOT, "checkpoints", "Re-train", "Sequence 2 pha 3", "checkpoints", "model_tps_finetuned_best.pt"),
        'Pha3_Last': os.path.join(REPO_ROOT, "checkpoints", "Re-train", "Sequence 2 pha 3", "checkpoints", "model_tps_finetuned_last.pt"),
    }

    models_dict = {}
    for tag, cp in ckpts.items():
        m = TPSInbetweenOriginal().to(device).eval()
        load_tps_weights(m, cp, device=device, prefer_ema=True)
        models_dict[tag] = m

    run_bench1_head_to_head(models_dict, device, os.path.join(out_root, "bench1_isolated_sequences"))
    run_bench2_head_to_head(models_dict, device, os.path.join(out_root, "bench2_continuous_lineart_test"))
    run_bench3_head_to_head(models_dict, device, os.path.join(out_root, "bench3_defense_geometry"))
    print(f"\n🎉 HOÀN TẤT TOÀN BỘ BENCHMARK ĐỐI ĐẦU TẠI: {out_root}")


if __name__ == "__main__":
    main()
