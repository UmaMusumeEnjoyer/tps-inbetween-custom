#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast Benchmark & Đối Đầu Toàn Diện TPS-Inbetween V3.2:
- Model Baseline V2.6 Best (Model Soup Calibrated - Warm Start gốc)
- Model V3.0.5 Best Progress (Epoch 25 - V3.0 Final)
- Model V3.2 Best Progress (Epoch 03 / Kỷ lục Composite Score)
- Model V3.2 Soup (Model Soup kết hợp Best Progress & Latest)
- Model V3.2 Latest (Epoch 25)

Tập dữ liệu kiểm định chuẩn vàng: 20 Stratified Validation Samples từ sakuga_golden_dataset_tps_ready_v1_patch3
(Track A: 12 Core Audit Samples + Track B: 8 High-Occlusion Turnaround Samples)
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from bench_triplet_utils_and_v1_0_vs_orig import (
    conv,
    deconv,
    Conv2,
    Contextnet,
    IFNet,
    TPSInbetweenFinetune_v10,
    robust_tps_theta_from_points,
    tps_flow_from_theta,
    warp_normalized_flow,
    warp_pixel_flow,
    compute_psnr,
    compute_ssim,
    compute_stroke_metrics,
    compute_ghost_pixels,
    compute_ie_mae,
    compute_chamfer_distance
)
from bench_triplet_04_v1_0_eval import compute_line_tortuosity

# ==============================================================================
# 1. KIẾN TRÚC MÔ HÌNH V3.0 (V3.0.5)
# ==============================================================================

class AsymmetricFeatureAnalyzer(nn.Module):
    def __init__(self, in_channels=34):
        super(AsymmetricFeatureAnalyzer, self).__init__()
        self.net = nn.Sequential(
            conv(in_channels, 32, 3, 1, 1),
            conv(32, 16, 3, 1, 1),
            nn.Conv2d(16, 2, 3, 1, 1)
        )

    def forward(self, fine_I0t, fine_I1t, feat0_sub, feat1_sub):
        b, _, h, w = fine_I0t.shape
        f0_up = F.interpolate(feat0_sub, size=(h, w), mode='bilinear', align_corners=False)
        f1_up = F.interpolate(feat1_sub, size=(h, w), mode='bilinear', align_corners=False)
        inp = torch.cat([fine_I0t, fine_I1t, f0_up, f1_up], dim=1)
        occ_logits = self.net(inp)
        occ_confidence = torch.sigmoid(occ_logits)
        return occ_confidence


class ContourGuidedFlowPropagator_v30(nn.Module):
    def __init__(self):
        super(ContourGuidedFlowPropagator_v30, self).__init__()
        self.conv_in = conv(7, 32, 3, 1, 1)
        self.d1 = conv(32, 32, 3, 1, 1, dilation=1)
        self.d2 = conv(32, 32, 3, 1, 2, dilation=2)
        self.d4 = conv(32, 32, 3, 1, 4, dilation=4)
        self.conv_out = nn.Conv2d(32, 4, 3, 1, 1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, coarse_I0t, coarse_I1t, flow_refine):
        shared_contour = torch.minimum(coarse_I0t, coarse_I1t)
        inp = torch.cat([coarse_I0t, coarse_I1t, flow_refine, shared_contour], dim=1)
        x = self.conv_in(inp)
        x = self.d1(x) + self.d2(x) + self.d4(x)
        delta_flow = self.conv_out(x)
        return flow_refine + delta_flow, delta_flow


class AsymmetricStrokeFusionHead_v30(nn.Module):
    def __init__(self):
        super(AsymmetricStrokeFusionHead_v30, self).__init__()
        self.net = nn.Sequential(
            conv(10, 32, 3, 1, 1),
            conv(32, 32, 3, 1, 1),
            nn.Conv2d(32, 2, 3, 1, 1)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.gamma_occ = nn.Parameter(torch.full((1, 2, 1, 1), 0.35))

    def forward(self, fine_I0t, fine_I1t, flow, ifnet_mask, t_vals, occ_confidence):
        b, _, h, w = fine_I0t.shape
        hs, ws = h // 2, w // 2
        t_sub = t_vals.expand(b, 1, hs, ws)
        inp_full = torch.cat([fine_I0t, fine_I1t, flow, ifnet_mask, occ_confidence], dim=1)
        inp_sub = F.avg_pool2d(inp_full, 2, 2)
        inp = torch.cat([inp_sub[:, :7], t_sub, inp_sub[:, 7:]], dim=1)
        residual_logits_sub = self.net(inp)
        residual_logits = F.interpolate(residual_logits_sub, size=(h, w), mode='bilinear', align_corners=False)

        prior_0 = torch.logit(torch.clamp(1.0 - t_vals, 1e-4, 1.0 - 1e-4))
        prior_1 = torch.logit(torch.clamp(t_vals, 1e-4, 1.0 - 1e-4))
        prior_logits = torch.cat([prior_0, prior_1], dim=1)

        vis_logits = prior_logits + residual_logits + self.gamma_occ * occ_confidence
        visibility_gates = torch.sigmoid(vis_logits)
        v0 = visibility_gates[:, 0:1]
        v1 = visibility_gates[:, 1:2]
        return v0, v1


class SynthesisUnet_v30(nn.Module):
    def __init__(self, indim=9, c=16):
        super(SynthesisUnet_v30, self).__init__()
        self.down0 = Conv2(indim, 2*c)
        self.down1 = Conv2(4*c, 4*c)
        self.down2 = Conv2(8*c, 8*c)
        self.down3 = Conv2(16*c, 16*c)
        self.up0 = deconv(32*c, 8*c)
        self.up1 = deconv(16*c, 4*c)
        self.up2 = deconv(8*c, 2*c)
        self.up3 = deconv(4*c, c)

        self.conv_erase = nn.Conv2d(c, 1, 3, 1, 1)
        self.conv_add = nn.Conv2d(c, 1, 3, 1, 1)

    def forward(self, img0, img1, flow0, flow1, warped0, warped1, mask, c0, c1):
        s0 = self.down0(torch.cat((img0, img1, warped0, warped1, flow0, flow1, mask), 1))
        s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
        s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
        s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
        x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))
        x = self.up1(torch.cat((x, s2), 1))
        x = self.up2(torch.cat((x, s1), 1))
        x = self.up3(torch.cat((x, s0), 1))

        gate_erase = torch.sigmoid(self.conv_erase(x))
        detail_add = torch.sigmoid(self.conv_add(x))
        return gate_erase, detail_add


class TPSInbetweenFinetune_v30(nn.Module):
    def __init__(self):
        super(TPSInbetweenFinetune_v30, self).__init__()
        self.encoder = Contextnet(indim=1)
        self.IFNet = IFNet()
        self.asym_analyzer = AsymmetricFeatureAnalyzer(in_channels=34)
        self.contour_propagator = ContourGuidedFlowPropagator_v30()
        self.asf_head = AsymmetricStrokeFusionHead_v30()
        self.unet = SynthesisUnet_v30(indim=9)

    def forward(self, x0, x1, t_val, coarse_flow01, coarse_flow10):
        if not isinstance(t_val, torch.Tensor):
            t_val = torch.tensor(t_val, device=x0.device, dtype=x0.dtype)
        b, c, h, w = x0.shape
        t_vals = t_val.view(b, 1, 1, 1)

        coarse_flow0t = coarse_flow01 * (t_vals ** 2) - (1.0 - t_vals) * t_vals * coarse_flow10
        coarse_flow1t = coarse_flow10 * ((1.0 - t_vals) ** 2) - t_vals * (1.0 - t_vals) * coarse_flow01

        coarse_I0t = warp_normalized_flow(x0, coarse_flow0t)
        coarse_I1t = warp_normalized_flow(x1, coarse_flow1t)
        coarse_interp = (1.0 - t_vals) * coarse_I0t + t_vals * coarse_I1t

        feats0 = self.encoder(x0, coarse_flow0t)
        feats1 = self.encoder(x1, coarse_flow1t)
        flow_raw, ifnet_mask = self.IFNet(coarse_I0t, coarse_I1t)

        flow_guided, _ = self.contour_propagator(coarse_I0t, coarse_I1t, flow_raw)

        fine_I0t = warp_pixel_flow(coarse_I0t, flow_guided[:, :2])
        fine_I1t = warp_pixel_flow(coarse_I1t, flow_guided[:, 2:4])

        occ_conf = self.asym_analyzer(fine_I0t, fine_I1t, feats0[0], feats1[0])

        v0, v1 = self.asf_head(fine_I0t, fine_I1t, flow_guided, ifnet_mask, t_vals, occ_conf)
        base_interp = torch.clamp(v0 * fine_I0t + v1 * fine_I1t, 0.0, 1.0)

        gate_erase, detail_add = self.unet(
            x0, x1,
            coarse_flow0t, coarse_flow1t,
            fine_I0t, fine_I1t, ifnet_mask, feats0, feats1
        )
        pred_t = torch.clamp(gate_erase * base_interp + detail_add, 0.0, 1.0)
        return pred_t, coarse_interp, (v0, v1), occ_conf


# ==============================================================================
# 2. KIẾN TRÚC MÔ HÌNH V3.2
# ==============================================================================

class ContourGuidedFlowPropagator_v32(nn.Module):
    def __init__(self):
        super(ContourGuidedFlowPropagator_v32, self).__init__()
        self.conv_in = conv(7, 24, 3, 1, 1)
        self.dw_d1 = nn.Sequential(
            nn.Conv2d(24, 24, 3, 1, 1, dilation=1, groups=24, bias=False),
            nn.Conv2d(24, 24, 1, bias=True),
            nn.PReLU(24)
        )
        self.dw_d2 = nn.Sequential(
            nn.Conv2d(24, 24, 3, 1, 2, dilation=2, groups=24, bias=False),
            nn.Conv2d(24, 24, 1, bias=True),
            nn.PReLU(24)
        )
        self.conv_out = nn.Conv2d(24, 4, 3, 1, 1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, coarse_I0t, coarse_I1t, flow_refine):
        b, _, h, w = coarse_I0t.shape
        c0_s = F.avg_pool2d(coarse_I0t, 2, 2)
        c1_s = F.avg_pool2d(coarse_I1t, 2, 2)
        fl_s = F.avg_pool2d(flow_refine, 2, 2) * 0.5
        shared_contour = torch.minimum(c0_s, c1_s)
        inp = torch.cat([c0_s, c1_s, fl_s, shared_contour], dim=1)
        x = self.conv_in(inp)
        x = x + self.dw_d1(x) + self.dw_d2(x)
        delta_flow_sub = self.conv_out(x)
        delta_flow = F.interpolate(delta_flow_sub, size=(h, w), mode='bilinear', align_corners=False) * 2.0
        return flow_refine + delta_flow, delta_flow


class TPSInbetweenFinetune_v32(nn.Module):
    def __init__(self):
        super(TPSInbetweenFinetune_v32, self).__init__()
        self.encoder = Contextnet(indim=1)
        self.IFNet = IFNet()
        self.asym_analyzer = AsymmetricFeatureAnalyzer(in_channels=34)
        self.contour_propagator = ContourGuidedFlowPropagator_v32()
        self.asf_head = AsymmetricStrokeFusionHead_v30()
        self.unet = SynthesisUnet_v30(indim=9)

    def forward(self, x0, x1, t_val, coarse_flow01, coarse_flow10):
        if not isinstance(t_val, torch.Tensor):
            t_val = torch.tensor(t_val, device=x0.device, dtype=x0.dtype)
        b, c, h, w = x0.shape
        t_vals = t_val.view(b, 1, 1, 1)

        coarse_flow0t = coarse_flow01 * (t_vals ** 2) - (1.0 - t_vals) * t_vals * coarse_flow10
        coarse_flow1t = coarse_flow10 * ((1.0 - t_vals) ** 2) - t_vals * (1.0 - t_vals) * coarse_flow01

        coarse_I0t = warp_normalized_flow(x0, coarse_flow0t)
        coarse_I1t = warp_normalized_flow(x1, coarse_flow1t)
        coarse_interp = (1.0 - t_vals) * coarse_I0t + t_vals * coarse_I1t

        feats0 = self.encoder(x0, coarse_flow0t)
        feats1 = self.encoder(x1, coarse_flow1t)
        flow_raw, ifnet_mask = self.IFNet(coarse_I0t, coarse_I1t)

        flow_guided, _ = self.contour_propagator(coarse_I0t, coarse_I1t, flow_raw)

        fine_I0t = warp_pixel_flow(coarse_I0t, flow_guided[:, :2])
        fine_I1t = warp_pixel_flow(coarse_I1t, flow_guided[:, 2:4])

        occ_conf = self.asym_analyzer(fine_I0t, fine_I1t, feats0[0], feats1[0])

        v0, v1 = self.asf_head(fine_I0t, fine_I1t, flow_guided, ifnet_mask, t_vals, occ_conf)
        base_interp = torch.clamp(v0 * fine_I0t + v1 * fine_I1t, 0.0, 1.0)

        gate_erase, detail_add = self.unet(
            x0, x1,
            coarse_flow0t, coarse_flow1t,
            fine_I0t, fine_I1t, ifnet_mask, feats0, feats1
        )
        pred_t = torch.clamp(gate_erase * base_interp + detail_add, 0.0, 1.0)
        return pred_t, coarse_interp, (v0, v1), occ_conf


# ==============================================================================
# HÀM BỔ TRỢ ĐO LƯỜNG VÀ KIỂM ĐỊNH
# ==============================================================================

def filter_topological_consistency(kps0, kps1, max_dev=0.18, radius=0.25):
    n = len(kps0)
    if n <= 4:
        return np.ones(n, dtype=bool)
    diffs = kps1 - kps0
    dists = np.linalg.norm(diffs, axis=1)
    median_disp = np.median(dists)
    mad = np.median(np.abs(dists - median_disp)) + 1e-6
    mask = (dists >= median_disp - 3.0 * mad) & (dists <= median_disp + 3.0 * mad)
    return mask


def compute_dark_ghost_pixels(pred_u8, gt_u8, is_blank=False):
    if is_blank:
        return -1
    gt_bin = (gt_u8 < 220).astype(np.uint8)
    gt_dilated = cv2.dilate(gt_bin, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=1)
    dark_ghost_mask = (pred_u8 < 180) & (gt_dilated == 0)
    return int(np.sum(dark_ghost_mask))


def render_visual_audits_v32(output_dir, clip_id, t_val, k0_np, p_coarse_np,
                             p_v26_np, p_v305_np, p_v32prog_np, p_v32soup_np, gt_np,
                             m_v26, m_v305, m_v32prog, m_v32soup):
    h, w = k0_np.shape
    sbs_dir = os.path.join(output_dir, 'side_by_side')
    hm_dir = os.path.join(output_dir, 'difference_heatmaps')
    zm_dir = os.path.join(output_dir, 'zoomed_details')
    for d in [sbs_dir, hm_dir, zm_dir]:
        os.makedirs(d, exist_ok=True)

    k0_bgr = cv2.cvtColor(k0_np, cv2.COLOR_GRAY2BGR)
    coarse_bgr = cv2.cvtColor(p_coarse_np, cv2.COLOR_GRAY2BGR)
    v26_bgr = cv2.cvtColor(p_v26_np, cv2.COLOR_GRAY2BGR)
    v305_bgr = cv2.cvtColor(p_v305_np, cv2.COLOR_GRAY2BGR)
    v32prog_bgr = cv2.cvtColor(p_v32prog_np, cv2.COLOR_GRAY2BGR)
    v32soup_bgr = cv2.cvtColor(p_v32soup_np, cv2.COLOR_GRAY2BGR)
    gt_bgr = cv2.cvtColor(gt_np, cv2.COLOR_GRAY2BGR)

    def add_header_footer(img_panel, title, tag, stat_str, is_highlight=False, is_winner=False):
        p = img_panel.copy()
        cv2.rectangle(p, (0, 0), (w, 38), (30, 30, 30), -1)
        header_color = (0, 255, 120) if is_winner else ((0, 220, 255) if is_highlight else (255, 255, 255))
        cv2.putText(p, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.44, header_color, 2)
        cv2.rectangle(p, (0, h - 34), (w, h), (25, 25, 25), -1)
        cv2.putText(p, tag, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 2)
        cv2.putText(p, stat_str, (72, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (230, 230, 230), 1)
        return p

    p0 = add_header_footer(k0_bgr, "Keyframe 0 (t=0.0)", "[INPUT]", f"{w}x{h} Lineart")
    pc = add_header_footer(coarse_bgr, "TPS Coarse Warp", "[COARSE]", "Thin-Plate Spline")
    p26 = add_header_footer(v26_bgr, "Warm-Start V2.6 Best", "[V2.6]",
                            f"{m_v26['psnr']:.1f}dB | Rec:{m_v26['recall']*100:.1f}% | Dens:{m_v26['density']:.2f}")
    p305 = add_header_footer(v305_bgr, "Predecessor V3.0.5", "[V3.0.5]",
                             f"{m_v305['psnr']:.1f}dB | Rec:{m_v305['recall']*100:.1f}% | Dens:{m_v305['density']:.2f}")
    p32p = add_header_footer(v32prog_bgr, "Target V3.2 Progress", "[V3.2 Prog]",
                             f"{m_v32prog['psnr']:.1f}dB | Rec:{m_v32prog['recall']*100:.1f}% | Dens:{m_v32prog['density']:.2f}",
                             is_highlight=True)
    p32s = add_header_footer(v32soup_bgr, "Target V3.2 Soup", "[V3.2 Soup]",
                             f"{m_v32soup['psnr']:.1f}dB | Rec:{m_v32soup['recall']*100:.1f}% | Dens:{m_v32soup['density']:.2f}",
                             is_winner=True)
    pgt = add_header_footer(gt_bgr, f"Ground Truth (t={t_val:.2f})", "[TARGET]", "Sakuga Artist Inbetween")

    panels = [p0, pc, p26, p305, p32p, p32s, pgt]
    sbs_combined = np.hstack(panels)
    sbs_path = os.path.join(sbs_dir, f"{clip_id}_t{t_val:.2f}_comparison_7col.png")
    cv2.imwrite(sbs_path, sbs_combined)

    # Zoomed 400% detail crops
    cy, cx = h // 2, w // 2
    cs = 110
    y1, y2 = max(0, cy - cs), min(h, cy + cs)
    x1, x2 = max(0, cx - cs), min(w, cx + cs)

    crop_c = cv2.resize(p_coarse_np[y1:y2, x1:x2], (360, 360), interpolation=cv2.INTER_NEAREST)
    crop_26 = cv2.resize(p_v26_np[y1:y2, x1:x2], (360, 360), interpolation=cv2.INTER_NEAREST)
    crop_305 = cv2.resize(p_v305_np[y1:y2, x1:x2], (360, 360), interpolation=cv2.INTER_NEAREST)
    crop_32p = cv2.resize(p_v32prog_np[y1:y2, x1:x2], (360, 360), interpolation=cv2.INTER_NEAREST)
    crop_32s = cv2.resize(p_v32soup_np[y1:y2, x1:x2], (360, 360), interpolation=cv2.INTER_NEAREST)
    crop_gt = cv2.resize(gt_np[y1:y2, x1:x2], (360, 360), interpolation=cv2.INTER_NEAREST)

    zm_crops = [crop_c, crop_26, crop_305, crop_32p, crop_32s, crop_gt]
    zm_panel = np.hstack(zm_crops)
    zm_bgr = cv2.cvtColor(zm_panel, cv2.COLOR_GRAY2BGR)

    step_w = 360
    titles = ["Coarse TPS", "Warm-Start V2.6", "Predecessor V3.0.5", "V3.2 Progress", "V3.2 Soup", "Ground Truth"]
    for i, t in enumerate(titles):
        x_pos = i * step_w + 10
        cv2.putText(zm_bgr, t, (x_pos, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 2)

    zm_path = os.path.join(zm_dir, f"{clip_id}_t{t_val:.2f}_zoomed_400pct.png")
    cv2.imwrite(zm_path, zm_bgr)


def _load_compat(model, ckpt_path, device):
    sd = torch.load(ckpt_path, map_location=device)
    if 'state_dict' in sd:
        sd = sd['state_dict']
    sd = {k.replace('module.', '').replace('base_model.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)


# ==============================================================================
# 3. QUY TRÌNH BENCHMARK CHÍNH
# ==============================================================================

def run_fast_benchmark_v32(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print("=" * 80)
    print("🔬 KHỞI ĐỘNG BENCHMARK & ĐỐI ĐẦU TRỰC DIỆN TPS-INBETWEEN V3.2")
    print(f"[*] Thiết bị kiểm định: {device} | Độ phân giải: Native 540p Lineart")
    print("=" * 80)

    # 1. Nạp Model V2.6 Baseline (Warm Start)
    print(f"\n[1/5] Đang nạp Model V2.6 Baseline (Warm-Start) từ: {args.model_v26} ...")
    model_v26 = TPSInbetweenFinetune_v10().to(device)
    _load_compat(model_v26, args.model_v26, device)
    model_v26.eval()
    print("[✓] Nạp Model V2.6 Baseline thành công!")

    # 2. Nạp Model V3.0.5 Predecessor
    print(f"\n[2/5] Đang nạp Model V3.0.5 (Epoch 25 Final) từ: {args.model_v305} ...")
    model_v305 = TPSInbetweenFinetune_v30().to(device)
    _load_compat(model_v305, args.model_v305, device)
    model_v305.eval()
    print("[✓] Nạp Model V3.0.5 thành công!")

    # 3. Nạp Model V3.2 Best Progress
    print(f"\n[3/5] Đang nạp Model V3.2 Best Progress từ: {args.model_v32_prog} ...")
    model_v32_prog = TPSInbetweenFinetune_v32().to(device)
    _load_compat(model_v32_prog, args.model_v32_prog, device)
    model_v32_prog.eval()
    print("[✓] Nạp Model V3.2 Best Progress thành công!")

    # 4. Nạp Model V3.2 Soup
    print(f"\n[4/5] Đang nạp Model V3.2 Soup từ: {args.model_v32_soup} ...")
    model_v32_soup = TPSInbetweenFinetune_v32().to(device)
    _load_compat(model_v32_soup, args.model_v32_soup, device)
    model_v32_soup.eval()
    print("[✓] Nạp Model V3.2 Soup thành công!")

    # 5. Nạp Model V3.2 Latest (nếu có)
    has_latest = os.path.exists(args.model_v32_latest)
    if has_latest:
        print(f"\n[5/5] Đang nạp Model V3.2 Latest (Ep 25) từ: {args.model_v32_latest} ...")
        model_v32_latest = TPSInbetweenFinetune_v32().to(device)
        _load_compat(model_v32_latest, args.model_v32_latest, device)
        model_v32_latest.eval()
        print("[✓] Nạp Model V3.2 Latest thành công!")
    else:
        model_v32_latest = None

    # Chuẩn bị danh sách clip
    df_all = pd.read_csv(args.dataset_csv)
    track_a_clips = [
        'clip_000008', 'clip_000010', 'clip_000018', 'clip_000064',
        'clip_000179', 'clip_000400', 'clip_000672', 'clip_000761',
        'clip_003046', 'clip_003050', 'clip_004200', 'clip_005225'
    ]
    track_b_clips = [
        'clip_008183', 'clip_007606', 'clip_007604', 'clip_007635',
        'clip_007607', 'clip_007427', 'clip_006754', 'clip_006743'
    ]
    target_clips = track_a_clips + track_b_clips
    df_eval = df_all[df_all['clip_id'].isin(target_clips)].copy()
    df_eval['order'] = df_eval['clip_id'].map({cid: i for i, cid in enumerate(target_clips)})
    df_eval = df_eval.sort_values('order').drop(columns=['order']).reset_index(drop=True)
    df_eval['track'] = df_eval['clip_id'].apply(lambda c: 'Track B (Turnaround)' if c in track_b_clips else 'Track A (Standard)')
    print(f"\n[*] Đã chuẩn bị {len(df_eval)} mẫu kiểm toán đối chứng phân tầng (Track A: {len(track_a_clips)}, Track B: {len(track_b_clips)}).")

    os.makedirs(args.output_dir, exist_ok=True)
    records = []
    t_start_total = time.time()

    pbar = tqdm(df_eval.iterrows(), total=len(df_eval), desc="🚀 Đang Chạy Benchmark Đối Đầu V3.2", unit="sample")
    for idx, row in pbar:
        clip_id = str(row['clip_id'])
        t_val = float(row['t'])
        mean_disp = float(row.get('mean_disp', 0.0))

        p0_path = os.path.join(args.data_root, str(row['frame0']).replace('\\', '/').lstrip('/'))
        pgt_path = os.path.join(args.data_root, str(row['frame_gt']).replace('\\', '/').lstrip('/'))
        p1_path = os.path.join(args.data_root, str(row['frame1']).replace('\\', '/').lstrip('/'))
        pmatch = os.path.join(args.data_root, str(row['match_file']).replace('\\', '/').lstrip('/')) if 'match_file' in row and pd.notna(row['match_file']) else None

        im0_raw = cv2.imread(p0_path, cv2.IMREAD_GRAYSCALE)
        gt_np = cv2.imread(pgt_path, cv2.IMREAD_GRAYSCALE)
        im1_raw = cv2.imread(p1_path, cv2.IMREAD_GRAYSCALE)

        if im0_raw is None or gt_np is None or im1_raw is None:
            print(f"[!] Bỏ qua mẫu {clip_id} do thiếu ảnh: {p0_path}")
            continue

        f0 = 1.0 - (im0_raw.astype(np.float32) / 255.0)
        fgt = 1.0 - (gt_np.astype(np.float32) / 255.0)
        f1 = 1.0 - (im1_raw.astype(np.float32) / 255.0)

        h, w = f0.shape
        pad_h = ((h + 31) // 32) * 32
        pad_w = ((w + 31) // 32) * 32
        f0_pad = np.pad(f0, ((0, pad_h - h), (0, pad_w - w)), mode='constant')
        fgt_pad = np.pad(fgt, ((0, pad_h - h), (0, pad_w - w)), mode='constant')
        f1_pad = np.pad(f1, ((0, pad_h - h), (0, pad_w - w)), mode='constant')

        if pmatch and os.path.exists(pmatch):
            kps = np.load(pmatch).astype(np.float32)
            kps0 = kps[0].copy()
            kps1 = kps[1].copy()
            kps0[:, 0] = (kps0[:, 0] * w) / pad_w
            kps0[:, 1] = (kps0[:, 1] * h) / pad_h
            kps1[:, 0] = (kps1[:, 0] * w) / pad_w
            kps1[:, 1] = (kps1[:, 1] * h) / pad_h

            topo_idx = filter_topological_consistency(kps0, kps1, max_dev=0.18, radius=0.25)
            kps0 = kps0[topo_idx]
            kps1 = kps1[topo_idx]

            c0 = torch.from_numpy(kps0).to(device)
            c1 = torch.from_numpy(kps1).to(device)
            if c0.shape[0] > 80:
                idx_sub = torch.linspace(0, c0.shape[0] - 1, 80, dtype=torch.long, device=device)
                c0 = c0[idx_sub]
                c1 = c1[idx_sub]
            theta01, c0_u = robust_tps_theta_from_points(c0, c1, lambd=0.5, add_anchors=True)
            theta10, c1_u = robust_tps_theta_from_points(c1, c0, lambd=0.5, add_anchors=True)
            flow01 = tps_flow_from_theta(theta01, c0_u, pad_h, pad_w, device)
            flow10 = tps_flow_from_theta(theta10, c1_u, pad_h, pad_w, device)
        else:
            flow01 = torch.zeros((1, 2, pad_h, pad_w), device=device)
            flow10 = torch.zeros((1, 2, pad_h, pad_w), device=device)

        t0_dev = torch.from_numpy(f0_pad).unsqueeze(0).unsqueeze(0).to(device)
        t1_dev = torch.from_numpy(f1_pad).unsqueeze(0).unsqueeze(0).to(device)

        # 1. Suy luận V2.6
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0_time = time.perf_counter()
        with torch.no_grad():
            out_v26, coarse_interp = model_v26(t0_dev, t1_dev, t_val, flow01, flow10)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        lat_v26 = (time.perf_counter() - t0_time) * 1000.0
        p_v26_np = ((1.0 - out_v26[:, :, :h, :w]) * 255.0).clamp(0, 255).byte().cpu().numpy()[0, 0]
        p_coarse_np = ((1.0 - coarse_interp[:, :, :h, :w]) * 255.0).clamp(0, 255).byte().cpu().numpy()[0, 0]

        # 2. Suy luận V3.0.5
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0_time = time.perf_counter()
        with torch.no_grad():
            out_v305, _, _, _ = model_v305(t0_dev, t1_dev, t_val, flow01, flow10)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        lat_v305 = (time.perf_counter() - t0_time) * 1000.0
        p_v305_np = ((1.0 - out_v305[:, :, :h, :w]) * 255.0).clamp(0, 255).byte().cpu().numpy()[0, 0]

        # 3. Suy luận V3.2 Best Progress
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0_time = time.perf_counter()
        with torch.no_grad():
            out_v32prog, _, _, _ = model_v32_prog(t0_dev, t1_dev, t_val, flow01, flow10)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        lat_v32prog = (time.perf_counter() - t0_time) * 1000.0
        p_v32prog_np = ((1.0 - out_v32prog[:, :, :h, :w]) * 255.0).clamp(0, 255).byte().cpu().numpy()[0, 0]

        # 4. Suy luận V3.2 Soup
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0_time = time.perf_counter()
        with torch.no_grad():
            out_v32soup, _, _, _ = model_v32_soup(t0_dev, t1_dev, t_val, flow01, flow10)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        lat_v32soup = (time.perf_counter() - t0_time) * 1000.0
        p_v32soup_np = ((1.0 - out_v32soup[:, :, :h, :w]) * 255.0).clamp(0, 255).byte().cpu().numpy()[0, 0]

        # 5. Suy luận V3.2 Latest (nếu có)
        if has_latest:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0_time = time.perf_counter()
            with torch.no_grad():
                out_v32lat, _, _, _ = model_v32_latest(t0_dev, t1_dev, t_val, flow01, flow10)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            lat_v32lat = (time.perf_counter() - t0_time) * 1000.0
            p_v32lat_np = ((1.0 - out_v32lat[:, :, :h, :w]) * 255.0).clamp(0, 255).byte().cpu().numpy()[0, 0]
        else:
            lat_v32lat = 0.0
            p_v32lat_np = p_v32soup_np

        # Tính toán metric cho từng model
        def get_metrics(p_np, lat):
            ps = compute_psnr(p_np, gt_np)
            ss = compute_ssim(p_np, gt_np)
            ma = compute_ie_mae(p_np, gt_np)
            sm = compute_stroke_metrics(p_np, gt_np)
            gh = compute_ghost_pixels(p_np, gt_np, is_blank=sm['is_blank_collapse'])
            dgh = compute_dark_ghost_pixels(p_np, gt_np, is_blank=sm['is_blank_collapse'])
            ch = compute_chamfer_distance(p_np, gt_np)

            p_mask = (p_np < 200).astype(np.uint8)
            g_mask = (gt_np < 200).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            p_dilated = cv2.dilate(p_mask, kernel)
            g_dilated = cv2.dilate(g_mask, kernel)
            tp_recall = np.sum((g_mask > 0) & (p_dilated > 0))
            recall = float(tp_recall / max(1, np.sum(g_mask > 0)))
            tp_prec = np.sum((p_mask > 0) & (g_dilated > 0))
            prec = float(tp_prec / max(1, np.sum(p_mask > 0)))
            f1 = float(2 * prec * recall / max(1e-6, prec + recall))
            density = float(sm['stroke_density_ratio'])

            return {
                'psnr': float(ps),
                'ssim': float(ss),
                'mae': float(ma),
                'recall': float(sm['stroke_recall']),
                'precision': prec,
                'f1': f1,
                'density': density,
                'ghost': int(gh),
                'dark_ghost': int(dgh),
                'chamfer': float(ch),
                'latency': float(lat)
            }

        m_v26 = get_metrics(p_v26_np, lat_v26)
        m_v305 = get_metrics(p_v305_np, lat_v305)
        m_v32prog = get_metrics(p_v32prog_np, lat_v32prog)
        m_v32soup = get_metrics(p_v32soup_np, lat_v32soup)
        m_v32lat = get_metrics(p_v32lat_np, lat_v32lat)

        render_visual_audits_v32(
            args.output_dir, clip_id, t_val, im0_raw, p_coarse_np,
            p_v26_np, p_v305_np, p_v32prog_np, p_v32soup_np, gt_np,
            m_v26, m_v305, m_v32prog, m_v32soup
        )

        row_dict = {
            'clip_id': clip_id,
            'track': str(row.get('track', 'Track A (Standard)')),
            't_val': t_val,
            'disp': mean_disp,
            # V2.6
            'ssim_v26': m_v26['ssim'],
            'psnr_v26': m_v26['psnr'],
            'mae_v26': m_v26['mae'],
            'recall_v26': m_v26['recall'],
            'density_v26': m_v26['density'],
            'f1_v26': m_v26['f1'],
            'ghost_v26': m_v26['ghost'],
            'dark_ghost_v26': m_v26['dark_ghost'],
            'chamfer_v26': m_v26['chamfer'],
            'latency_v26': m_v26['latency'],
            # V3.0.5
            'ssim_v305': m_v305['ssim'],
            'psnr_v305': m_v305['psnr'],
            'mae_v305': m_v305['mae'],
            'recall_v305': m_v305['recall'],
            'density_v305': m_v305['density'],
            'f1_v305': m_v305['f1'],
            'ghost_v305': m_v305['ghost'],
            'dark_ghost_v305': m_v305['dark_ghost'],
            'chamfer_v305': m_v305['chamfer'],
            'latency_v305': m_v305['latency'],
            # V3.2 Best Progress
            'ssim_v32prog': m_v32prog['ssim'],
            'psnr_v32prog': m_v32prog['psnr'],
            'mae_v32prog': m_v32prog['mae'],
            'recall_v32prog': m_v32prog['recall'],
            'density_v32prog': m_v32prog['density'],
            'f1_v32prog': m_v32prog['f1'],
            'ghost_v32prog': m_v32prog['ghost'],
            'dark_ghost_v32prog': m_v32prog['dark_ghost'],
            'chamfer_v32prog': m_v32prog['chamfer'],
            'latency_v32prog': m_v32prog['latency'],
            # V3.2 Soup
            'ssim_v32soup': m_v32soup['ssim'],
            'psnr_v32soup': m_v32soup['psnr'],
            'mae_v32soup': m_v32soup['mae'],
            'recall_v32soup': m_v32soup['recall'],
            'density_v32soup': m_v32soup['density'],
            'f1_v32soup': m_v32soup['f1'],
            'ghost_v32soup': m_v32soup['ghost'],
            'dark_ghost_v32soup': m_v32soup['dark_ghost'],
            'chamfer_v32soup': m_v32soup['chamfer'],
            'latency_v32soup': m_v32soup['latency'],
            # V3.2 Latest
            'ssim_v32lat': m_v32lat['ssim'],
            'psnr_v32lat': m_v32lat['psnr'],
            'mae_v32lat': m_v32lat['mae'],
            'recall_v32lat': m_v32lat['recall'],
            'density_v32lat': m_v32lat['density'],
            'f1_v32lat': m_v32lat['f1'],
            'ghost_v32lat': m_v32lat['ghost'],
            'dark_ghost_v32lat': m_v32lat['dark_ghost'],
            'chamfer_v32lat': m_v32lat['chamfer'],
            'latency_v32lat': m_v32lat['latency'],
        }
        records.append(row_dict)

    df_results = pd.DataFrame(records)
    csv_path = os.path.join(args.output_dir, 'benchmark_results_v32.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"\n[✓] Đã xuất kết quả bảng số liệu chi tiết: {csv_path}")

    # Tổng kết bảng điểm trung bình
    def compute_summary(df_sub, tag):
        models = [('v26', 'V2.6 Baseline'), ('v305', 'V3.0.5 Predecessor'),
                  ('v32prog', 'V3.2 Best Progress'), ('v32soup', 'V3.2 Soup'),
                  ('v32lat', 'V3.2 Latest')]
        summary = {}
        for m_key, m_name in models:
            summary[m_name] = {
                'ssim': float(df_sub[f'ssim_{m_key}'].mean()),
                'psnr': float(df_sub[f'psnr_{m_key}'].mean()),
                'mae': float(df_sub[f'mae_{m_key}'].mean()),
                'recall': float(df_sub[f'recall_{m_key}'].mean()),
                'f1': float(df_sub[f'f1_{m_key}'].mean()),
                'density': float(df_sub[f'density_{m_key}'].mean()),
                'ghost': float(df_sub[f'ghost_{m_key}'].mean()),
                'dark_ghost': float(df_sub[f'dark_ghost_{m_key}'].mean()),
                'chamfer': float(df_sub[f'chamfer_{m_key}'].mean()),
                'latency_ms': float(df_sub[f'latency_{m_key}'].mean())
            }
        return summary

    summary_all = {
        'overall_20_clips': compute_summary(df_results, 'Overall 20 Clips'),
        'track_a_12_standard': compute_summary(df_results[df_results['track'].str.contains('Track A')], 'Track A (Standard)'),
        'track_b_8_turnaround': compute_summary(df_results[df_results['track'].str.contains('Track B')], 'Track B (Turnaround)'),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_time_seconds': float(time.time() - t_start_total)
    }

    json_path = os.path.join(args.output_dir, 'summary_metrics_v32.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_all, f, indent=2, ensure_ascii=False)
    print(f"[✓] Đã xuất tổng kết JSON: {json_path}")

    # In kết quả đối đầu trực diện ra terminal
    print("\n" + "=" * 80)
    print("🏆 BẢNG TỔNG KẾT ĐỐI ĐẦU CHUẨN VÀNG: V3.2 VS V3.0.5 VS V2.6 (OVERALL 20 CLIPS)")
    print("=" * 80)
    df_sum = pd.DataFrame(summary_all['overall_20_clips']).T
    print(df_sum[['ssim', 'psnr', 'mae', 'recall', 'density', 'f1', 'ghost', 'dark_ghost', 'latency_ms']].to_string())

    print("\n" + "=" * 80)
    print("🎯 BẢNG TRACK A: 12 CORE AUDIT SAMPLES (STANDARD TRANSLATION / MOTION)")
    print("=" * 80)
    df_track_a = pd.DataFrame(summary_all['track_a_12_standard']).T
    print(df_track_a[['ssim', 'psnr', 'mae', 'recall', 'density', 'f1', 'ghost', 'dark_ghost', 'latency_ms']].to_string())

    print("\n" + "=" * 80)
    print("🔄 BẢNG TRACK B: 8 HIGH-OCCLUSION TURNAROUND SAMPLES (XOAY NGƯỜI / BIẾN MẤT NÉT)")
    print("=" * 80)
    df_track_b = pd.DataFrame(summary_all['track_b_8_turnaround']).T
    print(df_track_b[['ssim', 'psnr', 'mae', 'recall', 'density', 'f1', 'ghost', 'dark_ghost', 'latency_ms']].to_string())
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fast Benchmark & Đối Đầu Toàn Diện TPS-Inbetween V3.2")
    parser.add_argument('--model_v26', type=str,
                        default='checkpoints/experiments/v2.6/checkpoints_v2.6/model_tps_v2.6_best.pt',
                        help='Đường dẫn checkpoint model Baseline V2.6 Best')
    parser.add_argument('--model_v305', type=str,
                        default='checkpoints/experiments/v3.0 (2.6)/v3.0.5 (25epoch)/checkpoints_v3.0/model_tps_v3.0_best_progress.pt',
                        help='Đường dẫn checkpoint model V3.0.5 Predecessor')
    parser.add_argument('--model_v32_prog', type=str,
                        default='checkpoints/experiments/v3.2/checkpoints_v3.2/model_tps_v3.2_best_progress.pt',
                        help='Đường dẫn checkpoint model V3.2 Best Progress')
    parser.add_argument('--model_v32_soup', type=str,
                        default='checkpoints/experiments/v3.2/checkpoints_v3.2/model_tps_v3.2_soup.pt',
                        help='Đường dẫn checkpoint model V3.2 Soup')
    parser.add_argument('--model_v32_latest', type=str,
                        default='checkpoints/experiments/v3.2/checkpoints_v3.2/model_tps_v3.2_latest.pt',
                        help='Đường dẫn checkpoint model V3.2 Latest')
    parser.add_argument('--data_root', type=str,
                        default='data/processed/sakuga_golden_dataset_tps_ready_v1_patch3',
                        help='Thư mục chứa ảnh validation')
    parser.add_argument('--dataset_csv', type=str,
                        default='data/processed/sakuga_golden_dataset_tps_ready_v1_patch3/triplets_combined.csv',
                        help='Đường dẫn metadata triplets CSV')
    parser.add_argument('--output_dir', type=str,
                        default='checkpoints/experiments/v3.2/benchmark',
                        help='Thư mục lưu trữ kết quả benchmark và hình ảnh đối chiếu')
    parser.add_argument('--cpu', action='store_true',
                        help='Chạy trên CPU')

    args = parser.parse_args()
    run_fast_benchmark_v32(args)
