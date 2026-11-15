#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_seq_p3_validate_multilayer.py
--------------------------------------------------------------------------------
Kịch bản Kiểm định Thực nghiệm Pha 3 (Bước 1, Bước 2 & Bước 3 Offline Validation):
So sánh trực tiếp:
  - Single-Layer Global TPS (Baseline M0 & Blended Model)
  - Multi-Layer Clustered TPS (Pha 3 M0_MultiLayer & Blended + Multi-Layer TPS)
trên:
  1. Bộ kiểm định hình học Defense Geometry Suite (Case 1, Case 2, Case 3: Circle + Static Ground, Case 4)
  2. Chuỗi 24 frames 540p thực tế liên tục (`data/lineart_experiments/test` - 17 inbetween frames)
--------------------------------------------------------------------------------
"""

import os
import sys
import json
import time
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
BENCH_DIR = os.path.join(
    REPO_ROOT, "Dataset", "sakuga_pipeline_sequence", "final_dataset", "sakuga_golden_benchmark_sequences_v2"
)
for p in [SCRIPT_DIR, BENCH_DIR, os.path.join(REPO_ROOT, "kaggle_seq_p2_v33_ghost_purge")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from seq_p3_multi_layer_tps import (
    segment_motion_keypoints,
    solve_multi_layer_tps,
    verify_static_zero_hessian,
)
from seq_dual_benchmark_suite import (
    TPSInbetweenOriginal,
    load_tps_weights,
    solve_tps_direct,
    solve_tps_backward_pair,
    tps_flow_from_theta_batch,
    evaluate_metrics_dual_regime,
    extract_hybrid_contour_matches,
    ink_to_u8_gray,
)


def measure_ground_line_distortion(ink_pred, y_ground=330, x_start=80, x_end=560, band_half=15):
    """
    Đo độ phẳng tuyệt đối và sai số vị trí theo phương đứng (Y) của đường kẻ mặt đất tĩnh (y = 330).
    Trả về:
      - max_y_dev_px: Độ lệch tâm Y cực đại theo từng cột X so với y_ground (px).
      - mean_y_dev_px: Độ lệch tâm Y trung bình (px).
      - ground_psnr_db: PSNR riêng trên dải băng chứa đường kẻ mặt đất tĩnh.
    """
    band = ink_pred[y_ground - band_half:y_ground + band_half + 1, x_start:x_end + 1]
    ys = np.arange(y_ground - band_half, y_ground + band_half + 1, dtype=np.float32)[:, None]
    weights = np.where(band > 0.15, band, 0.0)
    col_sum = weights.sum(axis=0)
    valid_cols = col_sum > 0.1
    if not np.any(valid_cols):
        return 99.0, 99.0
    y_centroids = (weights[:, valid_cols] * ys).sum(axis=0) / col_sum[valid_cols]
    devs = np.abs(y_centroids - float(y_ground))
    return float(np.max(devs)), float(np.mean(devs))


def run_step1_step2_defense_geometry_validation(model, device, out_dir):
    print("\n" + "=" * 90)
    print("📐 [KIỂM ĐỊNH #1]: DEFENSE GEOMETRY SUITE (CASE 1..4 & CASE 3 CIRCLE + STATIC GROUND)")
    print("=" * 90)
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
        ("Case1_ExpandingCircle_30to70", _draw_ellipse_case(120, 192, 30, 30, 4, 520, 192, 70, 70, 5, False)),
        ("Case2_ShrinkingCircle_75to32", _draw_ellipse_case(120, 192, 75, 75, 5, 520, 192, 32, 32, 4, False)),
        ("Case3_ShrinkingCircle_StaticGround", _draw_ellipse_case(120, 180, 75, 75, 5, 520, 180, 32, 32, 4, True)),
        ("Case4_SquashOval90x40_to_Circle45", _draw_ellipse_case(140, 192, 90, 40, 4, 490, 192, 45, 45, 4, False)),
    ]

    hw_t = torch.tensor([[H, W]], dtype=torch.int32, device=device)
    rows = []

    for cname, gt_fn in cases:
        im0, im1 = gt_fn(0.0), gt_fn(1.0)
        k0, k1 = extract_hybrid_contour_matches(im0, im1, n_pts_per_comp=32)
        seg_info = segment_motion_keypoints(k0, k1, img_hw=(H, W), eps_static_px=1.5)
        print(f"\n🔹 {cname}: Tổng điểm={len(k0)} -> Tĩnh={seg_info['num_static']}, "
              f"Động={seg_info['num_dynamic']} ({seg_info['num_clusters']} cụm), Nhiễu={seg_info['num_outliers']}")

        x0 = torch.from_numpy(im0)[None, None].to(device)
        x1 = torch.from_numpy(im1)[None, None].to(device)

        vis_single_m0, vis_multi_m0 = [], []
        vis_single_nn, vis_multi_nn = [], []
        vis_masks, vis_gts = [], []

        for tv in t_list:
            gt_im = gt_fn(tv)
            vis_gts.append((255.0 - gt_im * 255.0).clip(0, 255).astype(np.uint8))
            gt_t = torch.from_numpy(gt_im)[None, None].to(device)
            tt = torch.tensor([[[[tv]]]], dtype=torch.float32, device=device)

            # 1. Single-Layer Global TPS
            kt = (1.0 - tv) * k0 + tv * k1
            c_t0, th_t0, v0 = solve_tps_direct(kt, k0, max_pts=128, lambd=0.01, use_anchors=False)
            c_t1, th_t1, v1 = solve_tps_direct(kt, k1, max_pts=128, lambd=0.01, use_anchors=False)
            cf0t_s = tps_flow_from_theta_batch(torch.from_numpy(th_t0)[None].to(device), torch.from_numpy(c_t0)[None].to(device), torch.tensor([v0], device=device), hw_t, H, W, device, 2)
            cf1t_s = tps_flow_from_theta_batch(torch.from_numpy(th_t1)[None].to(device), torch.from_numpy(c_t1)[None].to(device), torch.tensor([v1], device=device), hw_t, H, W, device, 2)
            with torch.no_grad():
                pr_s, cw_s, *_ = model(x0.float(), x1.float(), tt, cf0t_s.float(), cf1t_s.float(), direct_t_flows=True)

            # 2. Multi-Layer Clustered TPS (Pha 3)
            cf0t_m, cf1t_m, meta = solve_multi_layer_tps(
                k0, k1, (H, W), H, W, device,
                t_val=tv, direct_t_flows=True, max_pts=128, lambd=0.01,
                downscale_factor=2, eps_static_px=1.5, use_dyn_anchors=False,
                transition_width_px=18.0, falloff_radius_px=160.0
            )
            with torch.no_grad():
                pr_m, cw_m, *_ = model(x0.float(), x1.float(), tt, cf0t_m.float(), cf1t_m.float(), direct_t_flows=True)

            m0_s = evaluate_metrics_dual_regime(cw_s, gt_t)
            m0_m = evaluate_metrics_dual_regime(cw_m, gt_t)
            nn_s = evaluate_metrics_dual_regime(pr_s, gt_t)
            nn_m = evaluate_metrics_dual_regime(pr_m, gt_t)

            # Đo độ phẳng mặt đất trên Case 3
            cw_s_np = cw_s[0, 0].detach().cpu().numpy()
            cw_m_np = cw_m[0, 0].detach().cpu().numpy()
            pr_s_np = pr_s[0, 0].detach().cpu().numpy()
            pr_m_np = pr_m[0, 0].detach().cpu().numpy()

            # Đo Hessian vùng đường kẻ đất tĩnh (y in [312, 350], x in [70, 570])
            ground_zone = torch.zeros((1, 1, H, W), dtype=torch.bool, device=device)
            ground_zone[:, :, 312:350, 70:570] = True
            hess_single_ground = verify_static_zero_hessian(cf0t_s, ground_zone)
            hess_multi_ground = verify_static_zero_hessian(cf0t_m, ground_zone)

            if "StaticGround" in cname:
                g_max_s, g_mean_s = measure_ground_line_distortion(pr_s_np)
                g_max_m, g_mean_m = measure_ground_line_distortion(pr_m_np)
                # Đo sai số vùng trong lòng hình tròn của M0
                cx_t = int(round((1.0 - tv) * 120 + tv * 520))
                cy_t = 180
                r_in = max(5, int(round((1.0 - tv) * 75 + tv * 32)) - 8)
                yy, xx = np.ogrid[:H, :W]
                inner_circle = ((xx - cx_t) ** 2 + (yy - cy_t) ** 2) <= (r_in ** 2)
                inner_art_s = float(np.sum(cw_s_np[inner_circle] > 0.05))
                inner_art_m = float(np.sum(cw_m_np[inner_circle] > 0.05))
            else:
                g_max_s = g_mean_s = g_max_m = g_mean_m = 0.0
                inner_art_s = inner_art_m = 0.0

            rec = {
                'case': cname, 't_val': tv,
                'M0_Single_psnr': float(m0_s['psnr'].item()),
                'M0_Multi_psnr': float(m0_m['psnr'].item()),
                'M0_Single_ghost': float(m0_s['ghost'].item()),
                'M0_Multi_ghost': float(m0_m['ghost'].item()),
                'M0_Single_inner_artifact_px': inner_art_s,
                'M0_Multi_inner_artifact_px': inner_art_m,
                'Blended_Single_psnr': float(nn_s['psnr'].item()),
                'Blended_Multi_psnr': float(nn_m['psnr'].item()),
                'Blended_Single_ghost': float(nn_s['ghost'].item()),
                'Blended_Multi_ghost': float(nn_m['ghost'].item()),
                'Blended_Single_wcd_dog': float(nn_s['wcd_dog'].item()),
                'Blended_Multi_wcd_dog': float(nn_m['wcd_dog'].item()),
                'Blended_Single_f1': float(nn_s['f1'].item()),
                'Blended_Multi_f1': float(nn_m['f1'].item()),
                'Blended_Single_density': float(nn_s['density_ratio'].item()),
                'Blended_Multi_density': float(nn_m['density_ratio'].item()),
                'hessian_ground_single': hess_single_ground,
                'hessian_ground_multi': hess_multi_ground,
                'ground_y_dev_single_px': g_mean_s,
                'ground_y_dev_multi_px': g_mean_m,
            }
            rows.append(rec)

            print(f"   [t={tv:.2f}] M0 PSNR: {rec['M0_Single_psnr']:.2f} -> {rec['M0_Multi_psnr']:.2f} dB "
                  f"(InnerArtifact: {inner_art_s:.0f}px -> {inner_art_m:.0f}px) | "
                  f"Blended PSNR: {rec['Blended_Single_psnr']:.2f} -> {rec['Blended_Multi_psnr']:.2f} dB | "
                  f"Ghost: {rec['Blended_Single_ghost']:.0f} -> {rec['Blended_Multi_ghost']:.0f} px | "
                  f"Hessian(Static): {hess_single_ground:.5f} -> {hess_multi_ground:.6f}")

            vis_single_m0.append(ink_to_u8_gray(cw_s))
            vis_multi_m0.append(ink_to_u8_gray(cw_m))
            vis_single_nn.append(ink_to_u8_gray(pr_s))
            vis_multi_nn.append(ink_to_u8_gray(pr_m))
            mask_u8 = (meta['mask_total'][0, 0].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            vis_masks.append(mask_u8)

        # Xuất ảnh đối đầu trực quan cho từng Case
        if cv2 is not None:
            u0 = (255.0 - im0 * 255.0).clip(0, 255).astype(np.uint8)
            u1 = (255.0 - im1 * 255.0).clip(0, 255).astype(np.uint8)
            strips = []
            for label, seq_imgs, color in [
                ("M0 Single-Layer TPS", [u0] + vis_single_m0 + [u1], (40, 80, 220)),
                ("M0 Multi-Layer TPS (Phase 3)", [u0] + vis_multi_m0 + [u1], (20, 160, 40)),
                ("Blended + Single-Layer TPS", [u0] + vis_single_nn + [u1], (180, 90, 30)),
                ("Blended + Multi-Layer TPS (Phase 3)", [u0] + vis_multi_nn + [u1], (10, 140, 10)),
                ("Ground Truth", [u0] + vis_gts + [u1], (60, 60, 60)),
            ]:
                row_bgr = np.hstack([cv2.cvtColor(u, cv2.COLOR_GRAY2BGR) for u in seq_imgs])
                cv2.putText(row_bgr, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
                strips.append(row_bgr)
            cv2.imwrite(os.path.join(out_dir, f"{cname}_Phase3_Comparison.png"), np.vstack(strips))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "phase3_defense_geometry_comparison.csv"), index=False)
    return df


def run_step3_continuous_24frames_validation(model, device, out_dir):
    print("\n" + "=" * 90)
    print("🎬 [KIỂM ĐỊNH #2]: CHUỖI 24 FRAMES 540P THỰC TẾ (`data/lineart_experiments/test`)")
    print("=" * 90)
    os.makedirs(out_dir, exist_ok=True)

    test_dir = os.path.join(REPO_ROOT, "data", "lineart_experiments", "test")
    spec_path = os.path.join(test_dir, "continuous_keyframes_spec.json")
    gs_npz_path = os.path.join(REPO_ROOT, "kaggle_seq_p1_v32_ghost_temp_scale", "lineart_test_gluestick_matches.npz")

    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    frames_spec = spec["frames"]
    key_indices = sorted(spec["key_indices"])

    gt_imgs_u8 = [
        np.asarray(Image.open(os.path.join(test_dir, item["filename"])).convert("L"), dtype=np.uint8)
        for item in frames_spec
    ]
    h, w = gt_imgs_u8[0].shape
    h_pad, w_pad = ((h + 31) // 32) * 32, ((w + 31) // 32) * 32
    gs_cache = dict(np.load(gs_npz_path))
    hw_t = torch.tensor([[h, w]], dtype=torch.int32, device=device)

    # Tiền xử lý từng segment (ka -> kb)
    seg_inputs = {}
    print("\n📊 Thống kê Phân cụm Điểm chuyển động (Bước 1) trên 6 chặng Keyframes:")
    for i in range(len(key_indices) - 1):
        ka, kb = key_indices[i], key_indices[i + 1]
        kps = gs_cache[f"seg_{ka}_{kb}"].astype(np.float32)
        ima_u8, imb_u8 = gt_imgs_u8[ka], gt_imgs_u8[kb]
        x0_p = F.pad(torch.from_numpy((255.0 - ima_u8.astype(np.float32)) / 255.0)[None, None], (0, w_pad - w, 0, h_pad - h)).to(device)
        x1_p = F.pad(torch.from_numpy((255.0 - imb_u8.astype(np.float32)) / 255.0)[None, None], (0, w_pad - w, 0, h_pad - h)).to(device)

        # 1. Single-Layer Backward-Pair TPS (Baseline hiện tại)
        c0_np, t01_np, c1_np, t10_np, vf = solve_tps_backward_pair(kps[0], kps[1], max_pts=128, lambd=0.01)
        cf01_single = tps_flow_from_theta_batch(torch.from_numpy(t01_np)[None].to(device), torch.from_numpy(c0_np)[None].to(device), torch.tensor([vf], device=device), hw_t, h_pad, w_pad, device, 2)
        cf10_single = tps_flow_from_theta_batch(torch.from_numpy(t10_np)[None].to(device), torch.from_numpy(c1_np)[None].to(device), torch.tensor([vf], device=device), hw_t, h_pad, w_pad, device, 2)

        # 2. Multi-Layer Clustered Backward-Pair TPS (Pha 3: có dùng biên neo cho cụm động nhân vật + mặt nạ Voronoi + lọc nhiễu)
        cf01_multi, cf10_multi, meta_ml = solve_multi_layer_tps(
            kps[0], kps[1], (h, w), h_pad, w_pad, device,
            direct_t_flows=False, max_pts=128, lambd=0.01, downscale_factor=2,
            eps_static_px=1.5, use_dyn_anchors=True, transition_width_px=22.0,
            falloff_radius_px=105.0, filter_outliers=True
        )
        seg = meta_ml['segmentation']
        print(f"   - seg_{ka:02d}_{kb:02d}: Tổng={len(kps[0]):3d} | Tĩnh={seg['num_static']:2d} | "
              f"Động={seg['num_dynamic']:3d} ({seg['num_clusters']} cụm) | Outliers bị loại={seg['num_outliers']:2d} | "
              f"Static Hessian Max={meta_ml['static_hessian_max']:.6f}")

        seg_inputs[(ka, kb)] = {
            'x0': x0_p, 'x1': x1_p, 'kps': kps,
            'cf01_single': cf01_single, 'cf10_single': cf10_single,
            'cf01_multi': cf01_multi, 'cf10_multi': cf10_multi,
        }

    modes = ['M0_Single', 'M0_Multi', 'Blended_Single', 'Blended_Multi']
    reconstructed = {m: [img.copy() for img in gt_imgs_u8] for m in modes}
    per_frame_rows = []

    for item in frames_spec:
        f_idx = int(item['frame_idx'])
        if item['type'] != 'INBETWEEN':
            continue
        ka, kb, tv = int(item['ref_key_a']), int(item['ref_key_b']), float(item['t_val'])
        seg = seg_inputs[(ka, kb)]
        gt_ink = torch.from_numpy((255.0 - gt_imgs_u8[f_idx].astype(np.float32)) / 255.0)[None, None].to(device)
        tt = torch.tensor([[[[tv]]]], dtype=torch.float32, device=device)

        with torch.no_grad():
            pr_s, cw_s, *_ = model(seg['x0'].float(), seg['x1'].float(), tt.float(), seg['cf01_single'].float(), seg['cf10_single'].float())
            pr_m, cw_m, *_ = model(seg['x0'].float(), seg['x1'].float(), tt.float(), seg['cf01_multi'].float(), seg['cf10_multi'].float())

        outs = {
            'M0_Single': cw_s[:, :, :h, :w],
            'M0_Multi': cw_m[:, :, :h, :w],
            'Blended_Single': pr_s[:, :, :h, :w],
            'Blended_Multi': pr_m[:, :, :h, :w],
        }
        row = {'frame_idx': f_idx, 'segment': f"{ka}->{kb}", 't_val': tv}
        for m_name, pred_crop in outs.items():
            reconstructed[m_name][f_idx] = ink_to_u8_gray(pred_crop)
            mm = evaluate_metrics_dual_regime(pred_crop, gt_ink)
            for mk, mv in mm.items():
                row[f"{m_name}_{mk}"] = float(mv.item())
        per_frame_rows.append(row)

    # Tính Temporal Diff MAE & Temporal BG Flicker
    for row in per_frame_rows:
        fi = int(row['frame_idx'])
        g_cur = (255.0 - gt_imgs_u8[fi].astype(np.float32)) / 255.0
        g_prev = (255.0 - gt_imgs_u8[fi - 1].astype(np.float32)) / 255.0
        both_bg = (g_cur < 0.04) & (g_prev < 0.04)
        dg = g_cur - g_prev
        for m_name in modes:
            p_cur = (255.0 - reconstructed[m_name][fi].astype(np.float32)) / 255.0
            p_prev = (255.0 - reconstructed[m_name][fi - 1].astype(np.float32)) / 255.0
            dp = p_cur - p_prev
            row[f"{m_name}_temp_diff_mae"] = float(np.mean(np.abs(dp - dg)) * 255.0)
            row[f"{m_name}_temp_bg_flicker"] = float(np.sum(np.abs(dp) * both_bg))

    df_frames = pd.DataFrame(per_frame_rows)
    df_frames.to_csv(os.path.join(out_dir, "phase3_24frames_per_frame.csv"), index=False)

    sum_rows = []
    for m_name in modes:
        sr = {'mode': m_name}
        for mc in ['psnr', 'ssim', 'ie_mae', 'ghost', 'temp_diff_mae', 'temp_bg_flicker', 'wcd_dog', 'f1', 'skel_recall', 'density_ratio']:
            sr[mc] = float(df_frames[f"{m_name}_{mc}"].mean())
        sum_rows.append(sr)
    df_sum = pd.DataFrame(sum_rows)
    df_sum.to_csv(os.path.join(out_dir, "phase3_24frames_summary.csv"), index=False)

    print("\n📊 BẢNG TỔNG HỢP 17 IN-BETWEEN FRAMES (`data/lineart_experiments/test`):")
    print(df_sum.round(4).to_string(index=False))
    return df_sum


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    blended_ckpt = os.path.join(
        REPO_ROOT, "checkpoints", "Re-train", "Sequence 2 Blended", "checkpoints", "model_tps_blended.pt"
    )
    out_dir = os.path.join(REPO_ROOT, "checkpoints", "Re-train", "Phase3_MultiLayer_TPS_Validation")
    os.makedirs(out_dir, exist_ok=True)

    model = TPSInbetweenOriginal().to(device).eval()
    load_tps_weights(model, blended_ckpt, device=device, prefer_ema=True)

    run_step1_step2_defense_geometry_validation(model, device, out_dir)
    run_step3_continuous_24frames_validation(model, device, out_dir)


if __name__ == "__main__":
    main()
