#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ddp_worker_triplet_asym_v31.py
Dedicated DistributedDataParallel (DDP) Training & Validation Engine for V3.1.
Supports 2x Tesla T4 GPUs on Kaggle with NCCL backend, torchrun multi-processing,
unbuffered streaming logs (<3s initial log), DistributedWeightedSampler,
model.no_sync() gradient accumulation, and single-GPU fallback.
"""

import os
import sys
import time
import math
import glob
import json
import psutil
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Cấu hình môi trường phân tán & unbuffered stdout
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# =============================================================================
# 1. HÀM TIỆN ÍCH TPS & TOÁN TỬ KHÔNG GIAN (CELL 3)
# =============================================================================
_NORM_GRID_CACHE = {}
_PIXEL_GRID_CACHE = {}
_GRID_CACHE = {}

def warp_normalized_flow(img, flow):
    b, _, h, w = img.shape
    dev = flow.device
    key = (h, w, dev)
    base_grid = _NORM_GRID_CACHE.get(key)
    if base_grid is None:
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, h, device=dev, dtype=torch.float32),
            torch.linspace(0, 1, w, device=dev, dtype=torch.float32), indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).contiguous()
        _NORM_GRID_CACHE[key] = base_grid

    with torch.amp.autocast('cuda', enabled=False):
        flow_f32 = flow.float()
        grid = ((base_grid + flow_f32.permute(0, 2, 3, 1)) * 2.0 - 1.0).contiguous()
        out = F.grid_sample(img.float().contiguous(), grid, padding_mode='border', align_corners=True)
    return out.to(img.dtype)

def warp_pixel_flow(img, flow):
    b, _, h, w = img.shape
    dev = flow.device
    key = (h, w, dev)
    base_grid = _PIXEL_GRID_CACHE.get(key)
    if base_grid is None:
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=dev, dtype=torch.float32),
            torch.linspace(-1, 1, w, device=dev, dtype=torch.float32), indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).contiguous()
        _PIXEL_GRID_CACHE[key] = base_grid

    with torch.amp.autocast('cuda', enabled=False):
        flow_f32 = flow.float()
        flow_norm = torch.cat([
            flow_f32[:, 0:1, :, :] / max(1.0, (w - 1.0) / 2.0),
            flow_f32[:, 1:2, :, :] / max(1.0, (h - 1.0) / 2.0)
        ], dim=1)
        grid = (base_grid + flow_norm.permute(0, 2, 3, 1)).contiguous()
        out = F.grid_sample(img.float().contiguous(), grid, padding_mode='border', align_corners=True)
    return out.to(img.dtype)

def tps_rbf(r):
    r = torch.clamp(r, min=1e-6)
    return (r ** 2) * torch.log(r)

def get_grid_pts(h, w, device):
    key = (h, w, device)
    pts = _GRID_CACHE.get(key)
    if pts is None:
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, h, device=device, dtype=torch.float32),
            torch.linspace(0, 1, w, device=device, dtype=torch.float32), indexing='ij'
        )
        pts = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2).contiguous()
        _GRID_CACHE[key] = pts
    return pts

@torch.no_grad()
def tps_flow_from_theta_batch(theta_batch, c_src_batch, valid_mask, h, w, device, downscale_factor=4):
    with torch.amp.autocast('cuda', enabled=False):
        theta_f32 = theta_batch.float()
        c_src_f32 = c_src_batch.float()
        valid_f32 = valid_mask.float()
        b = theta_f32.shape[0]

        if downscale_factor > 1:
            h_sub = h // downscale_factor
            w_sub = w // downscale_factor
        else:
            h_sub, w_sub = h, w

        pts = get_grid_pts(h_sub, w_sub, device)
        total_pts = pts.shape[0]
        pts_exp = pts.unsqueeze(0).expand(b, -1, -1)

        dist = torch.cdist(pts_exp, c_src_f32, p=2.0)
        U = tps_rbf(dist)

        ones = torch.ones((b, total_pts, 1), device=device, dtype=pts.dtype)
        basis = torch.cat([U, ones, pts_exp], dim=-1)

        mapped = torch.bmm(basis, theta_f32)
        flow = (mapped - pts_exp).view(b, h_sub, w_sub, 2).permute(0, 3, 1, 2).contiguous()

        if downscale_factor > 1:
            flow = F.interpolate(flow, size=(h, w), mode='bilinear', align_corners=True)

        flow.mul_(valid_f32.view(b, 1, 1, 1))
        return flow

# =============================================================================
# 2. KIẾN TRÚC MÔ HÌNH V3.1 (CELL 5 - NATIVE 540P + WARM-START V2.6 COMPATIBLE)
# =============================================================================
def conv(in_p, out_p, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_p, out_p, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_p)
    )

def deconv(in_p, out_p, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_p, out_p, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.PReLU(out_p)
    )

class Conv2(nn.Module):
    def __init__(self, in_p, out_p, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_p, out_p, 3, stride, 1)
        self.conv2 = conv(out_p, out_p, 3, 1, 1)

    def forward(self, x):
        return self.conv2(self.conv1(x))

class Contextnet(nn.Module):
    def __init__(self, indim=1, c=16):
        super(Contextnet, self).__init__()
        self.conv1 = Conv2(indim, c)
        self.conv2 = Conv2(c, 2*c)
        self.conv3 = Conv2(2*c, 4*c)
        self.conv4 = Conv2(4*c, 8*c)

    def forward(self, x, flow):
        x = self.conv1(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode='bilinear', align_corners=False) * 0.5
        f1 = warp_normalized_flow(x, flow)
        x = self.conv2(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode='bilinear', align_corners=False) * 0.5
        f2 = warp_normalized_flow(x, flow)
        x = self.conv3(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode='bilinear', align_corners=False) * 0.5
        f3 = warp_normalized_flow(x, flow)
        x = self.conv4(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode='bilinear', align_corners=False) * 0.5
        f4 = warp_normalized_flow(x, flow)
        return [f1, f2, f3, f4]

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(*[conv(c, c) for _ in range(8)])
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    def forward(self, x, flow, scale):
        if scale != 1:
            x = F.interpolate(x, scale_factor=1./scale, mode='bilinear', align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1./scale, mode='bilinear', align_corners=False) * (1./scale)
            x = torch.cat((x, flow), 1)
        x = self.conv0(x.contiguous())
        x = self.convblock(x) + x
        with torch.amp.autocast('cuda', enabled=False):
            tmp = self.lastconv(x.float().contiguous())
            tmp = F.interpolate(tmp, scale_factor=scale * 2, mode='bilinear', align_corners=False)
            flow = tmp[:, :4] * scale * 2
            mask = tmp[:, 4:5]
        return flow, mask

class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(6, c=240)
        self.block1 = IFBlock(13+4, c=150)
        self.block2 = IFBlock(13+4, c=90)

    def forward(self, I0t, I1t, scale=[4, 2, 1]):
        img0 = I0t.repeat(1, 3, 1, 1)
        img1 = I1t.repeat(1, 3, 1, 1)
        warped_i0 = I0t
        warped_i1 = I1t
        flow = None
        mask = None
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow is not None:
                flow_d, mask_d = stu[i](torch.cat((img0, img1, warped_i0.repeat(1, 3, 1, 1), warped_i1.repeat(1, 3, 1, 1), mask), 1), flow, scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            if i < 2:
                warped_i0 = warp_pixel_flow(I0t, flow[:, :2])
                warped_i1 = warp_pixel_flow(I1t, flow[:, 2:4])
        return flow, torch.sigmoid(mask)

class AsymmetricFeatureAnalyzer(nn.Module):
    def __init__(self, in_channels=34):
        super(AsymmetricFeatureAnalyzer, self).__init__()
        self.net = nn.Sequential(
            conv(in_channels, 32, 3, 1, 1),
            conv(32, 16, 3, 1, 1),
            nn.Conv2d(16, 2, 3, 1, 1)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -1.5)

    def forward(self, fine_I0t, fine_I1t, feat0_sub, feat1_sub):
        b, _, h, w = fine_I0t.shape
        hs, ws = feat0_sub.shape[-2], feat0_sub.shape[-1]
        i0_sub = F.interpolate(fine_I0t, size=(hs, ws), mode='bilinear', align_corners=False)
        i1_sub = F.interpolate(fine_I1t, size=(hs, ws), mode='bilinear', align_corners=False)
        inp = torch.cat([i0_sub, i1_sub, feat0_sub, feat1_sub], dim=1)
        occ_logits_sub = self.net(inp)
        occ_logits = F.interpolate(occ_logits_sub, size=(h, w), mode='bilinear', align_corners=False)
        return torch.sigmoid(occ_logits)

class ContourGuidedFlowPropagator_v31(nn.Module):
    def __init__(self):
        super(ContourGuidedFlowPropagator_v31, self).__init__()
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

class AsymmetricStrokeFusionHead_v31(nn.Module):
    def __init__(self):
        super(AsymmetricStrokeFusionHead_v31, self).__init__()
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

class SynthesisUnet_v31(nn.Module):
    def __init__(self, indim=9, c=16):
        super(SynthesisUnet_v31, self).__init__()
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
        nn.init.zeros_(self.conv_erase.weight)
        nn.init.constant_(self.conv_erase.bias, 2.0)
        nn.init.zeros_(self.conv_add.weight)
        nn.init.constant_(self.conv_add.bias, -2.0)

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

class TPSInbetweenFinetune_v31(nn.Module):
    def __init__(self):
        super(TPSInbetweenFinetune_v31, self).__init__()
        self.encoder = Contextnet(indim=1)
        self.IFNet = IFNet()
        self.asym_analyzer = AsymmetricFeatureAnalyzer(in_channels=34)
        self.contour_propagator = ContourGuidedFlowPropagator_v31()
        self.asf_head = AsymmetricStrokeFusionHead_v31()
        self.unet = SynthesisUnet_v31(indim=9)
        self._init_v31_weights()

    def _init_v31_weights(self):
        for m in [self.asym_analyzer, self.contour_propagator, self.asf_head]:
            for mod in m.modules():
                if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(mod.weight, a=0.25, mode='fan_out', nonlinearity='leaky_relu')
                    if mod.bias is not None:
                        nn.init.zeros_(mod.bias)
        for head in [self.asf_head.net[-1], self.contour_propagator.conv_out]:
            if hasattr(head, 'weight') and head.weight is not None:
                nn.init.normal_(head.weight, mean=0.0, std=1e-3)
                if head.bias is not None:
                    nn.init.zeros_(head.bias)

    def forward(self, x0, x1, t_vals, tps_precomputed=None):
        if x0.dtype == torch.uint8:
            x0 = (255.0 - x0.float()) * (1.0 / 255.0)
        if x1.dtype == torch.uint8:
            x1 = (255.0 - x1.float()) * (1.0 / 255.0)

        b, c, h, w = x0.shape
        dev = x0.device
        t_vals = t_vals.view(b, 1, 1, 1).to(dev)

        if tps_precomputed is not None:
            c0_b, th01_b, c1_b, th10_b, valid_b = tps_precomputed
            coarse_flow01 = tps_flow_from_theta_batch(th01_b.to(dev), c0_b.to(dev), valid_b.to(dev), h, w, dev)
            coarse_flow10 = tps_flow_from_theta_batch(th10_b.to(dev), c1_b.to(dev), valid_b.to(dev), h, w, dev)
        else:
            coarse_flow01 = torch.zeros(b, 2, h, w, device=dev)
            coarse_flow10 = torch.zeros(b, 2, h, w, device=dev)

        coarse_flow0t = coarse_flow01 * (t_vals ** 2) - (1.0 - t_vals) * t_vals * coarse_flow10
        coarse_flow1t = coarse_flow10 * ((1.0 - t_vals) ** 2) - t_vals * (1.0 - t_vals) * coarse_flow01

        coarse_I0t = warp_normalized_flow(x0, coarse_flow0t)
        coarse_I1t = warp_normalized_flow(x1, coarse_flow1t)

        feats0 = self.encoder(x0, coarse_flow0t)
        feats1 = self.encoder(x1, coarse_flow1t)
        flow_raw, ifnet_mask = self.IFNet(coarse_I0t, coarse_I1t)

        flow_guided, delta_flow = self.contour_propagator(coarse_I0t, coarse_I1t, flow_raw)
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
        return pred_t, (v0, v1), occ_conf

# =============================================================================
# 3. HÀM MẤT MÁT TỔNG HỢP V3.1 (CELL 6)
# =============================================================================
def soft_erode(img):
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)

def soft_dilate(img):
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))

def soft_open(img):
    return soft_dilate(soft_erode(img))

def soft_skeletonize(img, iters=2):
    img = torch.clamp(img, 0.0, 1.0)
    skel = F.relu(img - soft_open(img))
    for _ in range(iters - 1):
        img = soft_erode(img)
        skel = skel + F.relu(img - soft_open(img))
    return torch.clamp(skel, 0.0, 1.0)

class FastLaplacianEdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        lap_kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('lap_kernel', lap_kernel)

    def forward(self, pred, target):
        p_f = pred.float()
        g_f = target.float()
        l_p1 = F.conv2d(p_f, self.lap_kernel, padding=1)
        l_g1 = F.conv2d(g_f, self.lap_kernel, padding=1)
        loss1 = F.l1_loss(l_p1, l_g1)

        p_sub = F.avg_pool2d(p_f, 2, 2)
        g_sub = F.avg_pool2d(g_f, 2, 2)
        l_p2 = F.conv2d(p_sub, self.lap_kernel, padding=1)
        l_g2 = F.conv2d(g_sub, self.lap_kernel, padding=1)
        loss2 = F.l1_loss(l_p2, l_g2)
        return loss1 + 0.5 * loss2

class SoftClDiceLoss_v31(nn.Module):
    def __init__(self, iters=2, smooth=1.0):
        super().__init__()
        self.iters = iters
        self.smooth = smooth

    def forward(self, pred, target, t_skel=None):
        p_skel = soft_skeletonize(pred, iters=self.iters)
        if t_skel is None:
            t_skel = soft_skeletonize(target, iters=self.iters)
        tprec = (torch.sum(p_skel * target) + self.smooth) / (torch.sum(p_skel) + self.smooth)
        tsens = (torch.sum(t_skel * pred) + self.smooth) / (torch.sum(t_skel) + self.smooth)
        cldice = 2.0 * (tprec * tsens) / (tprec + tsens + 1e-6)
        return 1.0 - cldice

class StrokeDensityRegularizationLoss_v31(nn.Module):
    def __init__(self, target_min=1.06, target_max=1.14, target_center=1.10, ramp_weight=0.25):
        super().__init__()
        self.target_min = target_min
        self.target_max = target_max
        self.target_center = target_center
        self.ramp_weight = ramp_weight

    def forward(self, pred_ink, gt_ink):
        p_ink = torch.clamp(pred_ink, 0.0, 1.0)
        g_ink = torch.clamp(gt_ink, 0.0, 1.0)
        m_soft_p = torch.clamp((p_ink - 0.06) / 0.18, 0.0, 1.0) + self.ramp_weight * F.relu(p_ink - 0.24)
        m_soft_g = torch.clamp((g_ink - 0.06) / 0.18, 0.0, 1.0) + self.ramp_weight * F.relu(g_ink - 0.24)
        m_hard_p = (p_ink > 0.15).float()
        m_hard_g = (g_ink > 0.15).float()
        m_ste_p = m_hard_p + (m_soft_p - m_soft_p.detach())
        m_ste_g = m_hard_g + (m_soft_g - m_soft_g.detach())
        sum_p = torch.sum(m_ste_p, dim=[-3, -2, -1])
        sum_g = torch.sum(m_ste_g, dim=[-3, -2, -1])
        ratio = sum_p / torch.clamp(sum_g, min=10.0)
        dev_low = torch.clamp(self.target_min - ratio, min=0.0, max=3.0)
        dev_high = torch.clamp(ratio - self.target_max, min=0.0, max=3.0)
        loss_low = 3.0 * (dev_low ** 2)
        loss_high = 2.0 * dev_high + 1.5 * (dev_high ** 2)
        center_pull = 0.08 * ((ratio - self.target_center) ** 2)
        return torch.mean(loss_low + loss_high + center_pull)

class HairTipEndpointContinuityLoss_v31(nn.Module):
    def __init__(self, endpoint_focal=1.5):
        super().__init__()
        self.endpoint_focal = endpoint_focal

    def forward(self, pred_ink, gt_ink, skel_gt=None):
        p_ink = torch.clamp(pred_ink, 0.0, 1.0)
        g_ink = torch.clamp(gt_ink, 0.0, 1.0)
        if skel_gt is None:
            skel_gt = soft_skeletonize(g_ink, iters=2)
        nbrs = F.avg_pool2d(skel_gt, kernel_size=3, stride=1, padding=1) * 9.0 - skel_gt
        endpoints = (nbrs <= 1.8) & (skel_gt > 0.20)
        hair_tip_roi = soft_dilate(endpoints.float())
        tip_under = F.relu(g_ink - p_ink) * hair_tip_roi
        return self.endpoint_focal * torch.mean(tip_under)

class OcclusionAwareGhostLoss_v31(nn.Module):
    def forward(self, pred, gt, v0, v1):
        pred_f = torch.clamp(pred, 0.0, 1.0)
        gt_f = torch.clamp(gt, 0.0, 1.0)
        gt_safe_mask = soft_dilate(soft_dilate((gt_f > 0.08).float()))
        outside_safe = 1.0 - gt_safe_mask
        far_ghost = F.relu(pred_f - 0.04) * outside_safe
        dark_ghost = F.relu(pred_f - 0.30) * outside_safe
        cleanliness_penalty = 0.25 * F.relu(pred_f - 0.01) * outside_safe
        gate_penalty = F.relu(v0 + v1 - 1.08)
        return 1.0 * torch.mean(far_ghost) + 2.5 * torch.mean(dark_ghost) + torch.mean(cleanliness_penalty) + 0.08 * torch.mean(gate_penalty)

class CompositeLossEngine_v31(nn.Module):
    def __init__(self):
        super().__init__()
        self.laplacian_loss = FastLaplacianEdgeLoss()
        self.cldice = SoftClDiceLoss_v31(iters=2)
        self.density_loss = StrokeDensityRegularizationLoss_v31(target_min=1.06, target_max=1.14, target_center=1.10)
        self.hair_tip_loss = HairTipEndpointContinuityLoss_v31(endpoint_focal=1.5)
        self.ghost_loss = OcclusionAwareGhostLoss_v31()

    def forward(self, pred, target, v0=None, v1=None, epoch=1):
        loss_l1 = F.l1_loss(pred, target)
        loss_lap = self.laplacian_loss(pred, target)

        with torch.no_grad():
            skel_gt = soft_skeletonize(target, iters=2)

        if epoch <= 2:
            # Epoch 1-2: Hội tụ thô ổn định trên L1 + Laplacian + clDice nhẹ
            # Theo Mục 2.5 & Mục 4 của Optimization Guide: không kích hoạt density/hair/ghost sớm
            loss_dice = self.cldice(pred, target, t_skel=skel_gt)
            total = 1.0 * loss_l1 + 0.25 * loss_lap + 0.05 * loss_dice
            loss_dens = torch.tensor(0.0, device=pred.device)
            loss_hair = torch.tensor(0.0, device=pred.device)
        else:
            # Epoch 3+: Bổ sung điều tiết mật độ nét chuẩn vàng 1.10x, bảo toàn ngọn tóc & anti-ghosting
            loss_dice = self.cldice(pred, target, t_skel=skel_gt)
            loss_dens = self.density_loss(pred, target)
            loss_hair = self.hair_tip_loss(pred, target, skel_gt=skel_gt)
            loss_ghost = self.ghost_loss(pred, target, v0, v1) if (v0 is not None and v1 is not None) else torch.tensor(0.0, device=pred.device)
            total = 1.0 * loss_l1 + 0.25 * loss_lap + 0.10 * loss_dice + 0.15 * loss_dens + 0.10 * loss_hair + 0.05 * loss_ghost

        return total, loss_l1, loss_dice, loss_dens, loss_hair

# =============================================================================
# 4. BỘ ĐO LƯỜNG CHỈ SỐ BATCHED PURE GPU METRICS (CELL 8)
# =============================================================================
def _create_gaussian_1d_kernel(kernel_size=11, sigma=1.5, device='cuda'):
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    g_x = g.view(1, 1, 1, kernel_size).repeat(5, 1, 1, 1)
    g_y = g.view(1, 1, kernel_size, 1).repeat(5, 1, 1, 1)
    return g_x, g_y

_GAUSS_X, _GAUSS_Y = None, None

def calculate_metrics_batch_gpu(pred_tensor, gt_tensor):
    global _GAUSS_X, _GAUSS_Y
    with torch.no_grad():
        p = torch.clamp(pred_tensor.float(), 0.0, 1.0)
        g = torch.clamp(gt_tensor.float(), 0.0, 1.0)
        b, c, h, w = p.shape
        dev = p.device

        mse = torch.mean((p - g) ** 2, dim=[-3, -2, -1])
        psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-8))

        if _GAUSS_X is None or _GAUSS_X.device != dev:
            _GAUSS_X, _GAUSS_Y = _create_gaussian_1d_kernel(11, 1.5, dev)
        gx = _GAUSS_X[:c] if _GAUSS_X.shape[0] >= c else _GAUSS_X[0:1].repeat(c, 1, 1, 1)
        gy = _GAUSS_Y[:c] if _GAUSS_Y.shape[0] >= c else _GAUSS_Y[0:1].repeat(c, 1, 1, 1)

        def _conv2d_separable(t):
            t_pad = F.pad(t, (5, 5, 5, 5), mode='reflect')
            return F.conv2d(F.conv2d(t_pad, gx, groups=c), gy, groups=c)

        mu_p, mu_g = _conv2d_separable(p), _conv2d_separable(g)
        mu_p_sq, mu_g_sq, mu_pg = mu_p ** 2, mu_g ** 2, mu_p * mu_g
        sigma_p_sq = _conv2d_separable(p * p) - mu_p_sq
        sigma_g_sq = _conv2d_separable(g * g) - mu_g_sq
        sigma_pg = _conv2d_separable(p * g) - mu_pg
        c1_s, c2_s = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2.0 * mu_pg + c1_s) * (2.0 * sigma_pg + c2_s)) / ((mu_p_sq + mu_g_sq + c1_s) * (sigma_p_sq + sigma_g_sq + c2_s))
        ssim = torch.mean(ssim_map, dim=[-3, -2, -1])

        p_bin = (p > 0.15).float()
        g_bin = (g > 0.15).float()
        tp = torch.sum(p_bin * g_bin, dim=[-3, -2, -1])
        fn = torch.sum((1.0 - p_bin) * g_bin, dim=[-3, -2, -1])
        fp = torch.sum(p_bin * (1.0 - g_bin), dim=[-3, -2, -1])
        recall = tp / torch.clamp(tp + fn, min=1e-5)
        f1 = (2.0 * tp) / torch.clamp(2.0 * tp + fp + fn, min=1e-5)

        sum_p = torch.sum(p_bin, dim=[-3, -2, -1])
        sum_g = torch.sum(g_bin, dim=[-3, -2, -1])
        density = sum_p / torch.clamp(sum_g, min=1.0)

        g_dilated = soft_dilate(g_bin)
        outside_core = 1.0 - g_dilated
        ghost_px = torch.sum((p > 0.10).float() * outside_core, dim=[-3, -2, -1])
        dark_ghost_px = torch.sum((p > 0.30).float() * outside_core, dim=[-3, -2, -1])
        p_sum = torch.clamp(sum_p, min=1.0)
        chamfer = torch.sum(torch.abs(p - g) * (1.0 - g), dim=[-3, -2, -1]) / (p_sum * max(1, h * w))

        return ssim, psnr, recall, f1, density, ghost_px, dark_ghost_px, chamfer

# =============================================================================
# 5. DISTRIBUTED WEIGHTED SAMPLER & DATASET (CELL 4)
# =============================================================================
def drop_file_page_cache(file_path):
    if sys.platform != 'win32':
        try:
            fd = os.open(file_path, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except Exception:
            pass

def imread_gray(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        drop_file_page_cache(path)
        return img
    except Exception:
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

class TemporalReversalAugmentation_v31:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img0, imggt, img1, t_val,
                 c0_pad, th01_pad, c1_pad, th10_pad):
        import random
        if random.random() < self.p:
            img0_out, img1_out = img1, img0
            t_out = float(1.0 - t_val)
            gt_out = imggt
            c0_out, th01_out = c1_pad, th10_pad
            c1_out, th10_out = c0_pad, th01_pad
            return img0_out, gt_out, img1_out, t_out, c0_out, th01_out, c1_out, th10_out, True
        return img0, imggt, img1, float(t_val), c0_pad, th01_pad, c1_pad, th10_pad, False

class DistributedWeightedSampler_v31(torch.utils.data.Sampler):
    def __init__(self, dataset, weights, num_samples=None, replacement=True, num_replicas=1, rank=0, seed=42):
        self.dataset = dataset
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples if num_samples is not None else len(dataset)
        self.replacement = replacement
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed
        self.num_samples_per_replica = int(math.ceil(self.num_samples / self.num_replicas))
        self.total_size = self.num_samples_per_replica * self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch * 1000)
        indices = torch.multinomial(self.weights, self.total_size, self.replacement, generator=g).tolist()
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples_per_replica

class SakugaGoldenDataset_v31(torch.utils.data.Dataset):
    def __init__(self, dataset_root, df_split, is_train=True, temporal_aug=None, cache_dir=None, max_kps=80, lambd=0.5):
        self.dataset_root = dataset_root
        self.df = df_split.reset_index(drop=True)
        self.is_train = is_train
        self.temporal_aug = temporal_aug
        self.max_kps = max_kps
        self.lambd = lambd
        self.length = len(self.df)
        self.split_name = 'train' if is_train else 'val'

        actual_root = dataset_root
        actual_match_root = dataset_root
        if 'sakuga_ssd_v31' in dataset_root or 'sakuga_golden_dataset_tps_ready_v1_patch3' in dataset_root:
            cand_img = os.path.join(dataset_root, 'anime_tps_finetune_dataset_ready')
            if os.path.exists(cand_img):
                actual_root = cand_img
            cand_match = os.path.join(dataset_root, 'anime_tps_finetune_dataset_ready_kps')
            if os.path.exists(cand_match):
                actual_match_root = cand_match

        f0_list = [os.path.join(actual_root, str(f).replace('\\', '/').lstrip('/')) for f in self.df['frame0']]
        fgt_list = [os.path.join(actual_root, str(f).replace('\\', '/').lstrip('/')) for f in self.df['frame_gt']]
        f1_list = [os.path.join(actual_root, str(f).replace('\\', '/').lstrip('/')) for f in self.df['frame1']]

        self.p0_arr = np.array([f.encode('utf-8') for f in f0_list], dtype='S256')
        self.pgt_arr = np.array([f.encode('utf-8') for f in fgt_list], dtype='S256')
        self.p1_arr = np.array([f.encode('utf-8') for f in f1_list], dtype='S256')
        self.t_arr = self.df['t'].to_numpy(dtype=np.float32, copy=True)
        self.clips_arr = np.array([str(c).encode('utf-8') for c in self.df['clip_id']], dtype='S64')

        cd = cache_dir or os.path.dirname(dataset_root)
        c0_npy = os.path.join(cd, f'{self.split_name}_c0_v31.npy')
        th01_npy = os.path.join(cd, f'{self.split_name}_th01_v31.npy')
        c1_npy = os.path.join(cd, f'{self.split_name}_c1_v31.npy')
        th10_npy = os.path.join(cd, f'{self.split_name}_th10_v31.npy')
        valid_npy = os.path.join(cd, f'{self.split_name}_valid_v31.npy')

        self.tps_c0 = np.load(c0_npy, mmap_mode='r')
        self.tps_th01 = np.load(th01_npy, mmap_mode='r')
        self.tps_c1 = np.load(c1_npy, mmap_mode='r')
        self.tps_th10 = np.load(th10_npy, mmap_mode='r')
        self.tps_valid = np.load(valid_npy, mmap_mode='r')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        clip_name = int(idx) if self.is_train else self.clips_arr[idx].decode('utf-8')
        im0 = imread_gray(self.p0_arr[idx].decode('utf-8'))
        imgt = imread_gray(self.pgt_arr[idx].decode('utf-8'))
        im1 = imread_gray(self.p1_arr[idx].decode('utf-8'))

        if im0 is None: im0 = np.full((540, 960), 255, dtype=np.uint8)
        if imgt is None: imgt = np.full((540, 960), 255, dtype=np.uint8)
        if im1 is None: im1 = np.full((540, 960), 255, dtype=np.uint8)

        h, w = im0.shape
        if imgt.shape != (h, w): imgt = cv2.resize(imgt, (w, h), interpolation=cv2.INTER_NEAREST)
        if im1.shape != (h, w): im1 = cv2.resize(im1, (w, h), interpolation=cv2.INTER_NEAREST)

        pad_h = ((h + 31) // 32) * 32
        pad_w = ((w + 31) // 32) * 32
        buf = np.full((3, pad_h, pad_w), 255, dtype=np.uint8)
        buf[0, :h, :w] = im0
        buf[1, :h, :w] = imgt
        buf[2, :h, :w] = im1
        im0_pad, imgt_pad, im1_pad = buf[0], buf[1], buf[2]

        c0_pad = self.tps_c0[idx].copy()
        th01_pad = self.tps_th01[idx].copy()
        c1_pad = self.tps_c1[idx].copy()
        th10_pad = self.tps_th10[idx].copy()
        valid_flag = self.tps_valid[idx].copy()
        t_scalar = float(self.t_arr[idx])

        if self.is_train and self.temporal_aug is not None:
            im0_pad, imgt_pad, im1_pad, t_scalar, c0_pad, th01_pad, c1_pad, th10_pad, _ = self.temporal_aug(
                im0_pad, imgt_pad, im1_pad, t_scalar, c0_pad, th01_pad, c1_pad, th10_pad
            )

        return (
            torch.from_numpy(np.ascontiguousarray(im0_pad[None, :, :], dtype=np.uint8)),
            torch.from_numpy(np.ascontiguousarray(imgt_pad[None, :, :], dtype=np.uint8)),
            torch.from_numpy(np.ascontiguousarray(im1_pad[None, :, :], dtype=np.uint8)),
            torch.tensor(t_scalar, dtype=torch.float32),
            torch.from_numpy(c0_pad),
            torch.from_numpy(th01_pad),
            torch.from_numpy(c1_pad),
            torch.from_numpy(th10_pad),
            torch.from_numpy(valid_flag),
            clip_name,
            int(h),
            int(w)
        )

custom_collate_fn_v31 = torch.utils.data.dataloader.default_collate

# =============================================================================
# 6. CHECKPOINT MANAGER & RESUME (CELL 9)
# =============================================================================
RESUME_CKPT_NAME = 'model_tps_v3.1_resume.pt'

class SmartCheckpointManager_v31:
    def __init__(self, save_dir, ref_ghost=9070.9, ref_chamfer=0.00230, ref_recall=0.930,
                 ref_ssim=0.900, ref_val_l1=0.0308, ref_dark_ghost=750.0,
                 min_recall=0.905, min_density=1.02, max_density=1.18, target_density=1.10):
        self.save_dir = save_dir
        self.ref_ghost = ref_ghost
        self.ref_chamfer = ref_chamfer
        self.ref_recall = ref_recall
        self.ref_ssim = ref_ssim
        self.ref_val_l1 = ref_val_l1
        self.ref_dark_ghost = ref_dark_ghost
        self.min_recall = min_recall
        self.min_density = min_density
        self.max_density = max_density
        self.target_density = target_density
        self.best_score = float('inf')
        self.best_progress_score = float('inf')
        self.best_epoch = 0
        self.resume_path = os.path.join(save_dir, RESUME_CKPT_NAME)
        self.top_candidates = []
        os.makedirs(save_dir, exist_ok=True)

    def calculate_score(self, m):
        density_dev = abs(m['density'] - self.target_density)
        dark_ghost_term = m['dark_ghost'] / max(1.0, self.ref_dark_ghost)
        score = (
            0.25 * (m['ghost'] / self.ref_ghost) +
            0.20 * (m['chamfer'] / self.ref_chamfer) +
            0.15 * ((1.0 - m['recall']) / (1.0 - self.ref_recall)) +
            0.15 * ((1.0 - m['ssim']) / (1.0 - self.ref_ssim)) +
            0.15 * (m['val_l1'] / self.ref_val_l1) +
            0.05 * (density_dev / 0.10) +
            0.05 * dark_ghost_term
        )
        return float(score)

    @staticmethod
    def _atomic_save(obj, path):
        tmp = path + '.tmp'
        torch.save(obj, tmp)
        os.replace(tmp, path)

    def save_resume(self, epoch, model, optimizer=None, scaler=None, history=None, completed=True):
        raw_model = model.module if hasattr(model, 'module') else (model.base_model if hasattr(model, 'base_model') else model)
        state = {
            'epoch': int(epoch) if completed else int(epoch) - 1,
            'model': {k: v.detach().cpu() for k, v in raw_model.state_dict().items()},
            'optimizer': optimizer.state_dict() if optimizer is not None else None,
            'scaler': scaler.state_dict() if scaler is not None else None,
            'best_score': self.best_score,
            'best_progress_score': self.best_progress_score,
            'best_epoch': self.best_epoch,
            'history': list(history) if history is not None else [],
        }
        self._atomic_save(state, self.resume_path)

    def load_resume(self, model, optimizer=None, scaler=None, map_location='cpu'):
        if not os.path.exists(self.resume_path):
            return 0, []
        state = torch.load(self.resume_path, map_location=map_location, weights_only=False)
        raw_model = model.module if hasattr(model, 'module') else (model.base_model if hasattr(model, 'base_model') else model)
        raw_model.load_state_dict(state['model'], strict=False)
        if optimizer is not None and state.get('optimizer') is not None:
            try: optimizer.load_state_dict(state['optimizer'])
            except Exception: pass
        if scaler is not None and state.get('scaler') is not None:
            try: scaler.load_state_dict(state['scaler'])
            except Exception: pass
        self.best_score = state.get('best_score', float('inf'))
        self.best_progress_score = state.get('best_progress_score', float('inf'))
        self.best_epoch = state.get('best_epoch', 0)
        ep = int(state.get('epoch', 0))
        hist = [h for h in state.get('history', []) if int(h.get('epoch', 0)) <= ep]
        return ep, hist

    def evaluate_and_save(self, epoch, metrics, model, optimizer=None):
        score = self.calculate_score(metrics)
        raw_model = model.module if hasattr(model, 'module') else (model.base_model if hasattr(model, 'base_model') else model)
        sd = {k: v.detach().cpu() for k, v in raw_model.state_dict().items()}

        latest_path = os.path.join(self.save_dir, 'model_tps_v3.1_latest.pt')
        self._atomic_save(sd, latest_path)

        density_ok = (self.min_density <= metrics['density'] <= self.max_density)
        recall_ok = (metrics['recall'] >= self.min_recall)
        gates_passed = (density_ok and recall_ok)

        is_best = False
        if gates_passed and (score < self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            is_best = True
            best_path = os.path.join(self.save_dir, 'model_tps_v3.1_best.pt')
            self._atomic_save(sd, best_path)

        is_progress = False
        if score < self.best_progress_score:
            self.best_progress_score = score
            is_progress = True
            prog_path = os.path.join(self.save_dir, 'model_tps_v3.1_best_progress.pt')
            self._atomic_save(sd, prog_path)

        if epoch >= 15:
            cand_path = os.path.join(self.save_dir, f'model_tps_v3.1_ep{epoch:02d}.pt')
            self._atomic_save(sd, cand_path)
            self.top_candidates.append((score, epoch, cand_path))
            self.top_candidates.sort(key=lambda x: x[0])
            if len(self.top_candidates) > 5:
                rem = self.top_candidates.pop(-1)
                if os.path.exists(rem[2]):
                    try: os.remove(rem[2])
                    except Exception: pass

        return score, gates_passed, is_best, is_progress

def generate_checkpoint_visual_demo_v31(model, current_epoch, val_ds, dev, working_dir, num_samples=3):
    demo_dir = os.path.join(working_dir, 'visual_samples_v3.1')
    os.makedirs(demo_dir, exist_ok=True)
    if len(val_ds) == 0: return []
    step_idx = max(1, len(val_ds) // (num_samples + 1))
    sample_indices = [min(len(val_ds) - 1, step_idx * (i + 1)) for i in range(num_samples)]
    raw_model = model.module if hasattr(model, 'module') else (model.base_model if hasattr(model, 'base_model') else model)
    was_training = raw_model.training
    raw_model.eval()

    saved_files = []
    with torch.inference_mode():
        for s_idx, idx in enumerate(sample_indices):
            batch_single = custom_collate_fn_v31([val_ds[idx]])
            t0 = batch_single[0].to(dev)
            tgt = batch_single[1].to(dev)
            t1 = batch_single[2].to(dev)
            t_val = batch_single[3].to(dev)
            c0_b = batch_single[4].to(dev)
            th01_b = batch_single[5].to(dev)
            c1_b = batch_single[6].to(dev)
            th10_b = batch_single[7].to(dev)
            valid_b = batch_single[8].to(dev)
            clip_name = str(batch_single[9][0])
            orig_h = int(batch_single[10][0]) if len(batch_single) > 10 else 540
            orig_w = int(batch_single[11][0]) if len(batch_single) > 11 else 960

            if t0.dtype == torch.uint8: t0 = ((255.0 - t0.float()) * (1.0 / 255.0)).contiguous()
            if tgt.dtype == torch.uint8: tgt = ((255.0 - tgt.float()) * (1.0 / 255.0)).contiguous()
            if t1.dtype == torch.uint8: t1 = ((255.0 - t1.float()) * (1.0 / 255.0)).contiguous()

            res = raw_model(t0, t1, t_val, tps_precomputed=(c0_b, th01_b, c1_b, th10_b, valid_b))
            pred_t, (v0, v1) = res[0], res[1]

            if orig_h > 0 and orig_w > 0 and (pred_t.shape[-2] != orig_h or pred_t.shape[-1] != orig_w):
                pred_t = pred_t[:, :, :orig_h, :orig_w]
                t0 = t0[:, :, :orig_h, :orig_w]
                tgt = tgt[:, :, :orig_h, :orig_w]
                v0 = v0[:, :, :orig_h, :orig_w]
                v1 = v1[:, :, :orig_h, :orig_w]

            img_0 = ((1.0 - t0) * 255.0).clamp(0, 255).byte().cpu().numpy()[0, 0]
            img_pred = ((1.0 - pred_t) * 255.0).clamp(0, 255).byte().cpu().numpy()[0, 0]
            img_gt = ((1.0 - tgt) * 255.0).clamp(0, 255).byte().cpu().numpy()[0, 0]
            v0_map = (v0.clamp(0, 1) * 255.0).byte().cpu().numpy()[0, 0]
            v1_map = (v1.clamp(0, 1) * 255.0).byte().cpu().numpy()[0, 0]
            vis_heatmap = np.stack([v0_map, np.zeros_like(v0_map), v1_map], axis=-1)

            p_bin = (pred_t > 0.15).float().sum().item()
            g_bin = (tgt > 0.15).float().sum().item()
            s_density = p_bin / max(1.0, g_bin)

            h_c, w_c = img_0.shape
            panel = np.hstack([
                cv2.cvtColor(img_0, cv2.COLOR_GRAY2BGR),
                vis_heatmap,
                cv2.cvtColor(img_pred, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(img_gt, cv2.COLOR_GRAY2BGR)
            ])
            cv2.putText(panel, f"Epoch {current_epoch:02d} | Clip: {clip_name} | Density: {s_density:.3f}x",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(panel, "I0 Input", (20, h_c - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2)
            cv2.putText(panel, "V0(B) / V1(R) Heatmap", (w_c + 20, h_c - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(panel, "TPS-Inbetween V3.1 Pred", (2 * w_c + 20, h_c - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 0), 2)
            cv2.putText(panel, "Ground Truth", (3 * w_c + 20, h_c - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2)

            out_fn = f"val_ep{current_epoch:02d}_sample{s_idx+1}_{clip_name}.png"
            out_p = os.path.join(demo_dir, out_fn)
            cv2.imwrite(out_p, panel)
            saved_files.append(out_p)

    if was_training:
        raw_model.train()
    return saved_files

# =============================================================================
# 7. CHƯƠNG TRÌNH CHÍNH (DDP ENGINE RUNNER)
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--working-dir', type=str, default='/kaggle/working')
    parser.add_argument('--ckpt-dir', type=str, default='')
    parser.add_argument('--stage-dir', type=str, default='/tmp/sakuga_ssd_v31')
    parser.add_argument('--total-epochs', type=int, default=25)
    parser.add_argument('--per-gpu-batch', type=int, default=4)
    parser.add_argument('--grad-accum-steps', type=int, default=2)
    parser.add_argument('--single-gpu', action='store_true', default=False)
    parser.add_argument('--pbar-pipe', action='store_true', default=False)
    args = parser.parse_args()

    working_dir = args.working_dir
    ckpt_dir = args.ckpt_dir if args.ckpt_dir else os.path.join(working_dir, 'checkpoints_v3.1')
    stage_dir = args.stage_dir
    total_epochs = args.total_epochs
    per_gpu_batch = args.per_gpu_batch
    grad_accum_steps = args.grad_accum_steps
    history_csv_path = os.path.join(working_dir, 'training_history_v3.1.csv')

    # Thiết lập phân tán
    is_ddp = ('RANK' in os.environ and 'WORLD_SIZE' in os.environ and not args.single_gpu)
    if is_ddp:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        device = torch.device(f'cuda:{local_rank}')
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    is_rank0 = (rank == 0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if is_rank0:
        print("=" * 80, flush=True)
        print(f"🚀 [V3.1 DDP ENGINE] Khởi tạo huấn luyện: {world_size} GPUs | Per-GPU Batch={per_gpu_batch} | Accum={grad_accum_steps} | Effective Batch={per_gpu_batch*world_size*grad_accum_steps}", flush=True)
        print("=" * 80, flush=True)

    # 1. Nạp metadata bundle & precomputed cache
    bundle_path = os.path.join(working_dir, 'dataset_bundle_v31.pt')
    if not os.path.exists(bundle_path):
        bundle_path = os.path.join(stage_dir, 'dataset_bundle_v31.pt')

    if os.path.exists(bundle_path):
        bundle = torch.load(bundle_path, weights_only=False)
        ds_root = bundle.get('dataset_root', stage_dir)
        cd_dir = bundle.get('cache_dir', stage_dir)
        train_df = bundle['train_df']
        val_df = bundle['val_df']
        train_weights = bundle['train_weights']
    else:
        raise FileNotFoundError(f"Không tìm thấy dataset_bundle_v31.pt tại {working_dir} hoặc {stage_dir}!")

    temporal_aug = TemporalReversalAugmentation_v31(p=0.5)
    train_ds = SakugaGoldenDataset_v31(ds_root, train_df, is_train=True, temporal_aug=temporal_aug, cache_dir=cd_dir)
    val_ds = SakugaGoldenDataset_v31(ds_root, val_df, is_train=False, temporal_aug=None, cache_dir=cd_dir)

    if is_ddp:
        train_sampler = DistributedWeightedSampler_v31(train_ds, train_weights, num_replicas=world_size, rank=rank)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = torch.utils.data.WeightedRandomSampler(train_weights, num_samples=len(train_ds), replacement=True)
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=per_gpu_batch,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn_v31,
        prefetch_factor=2
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=12,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        collate_fn=custom_collate_fn_v31,
        prefetch_factor=2
    )

    # 2. Xây dựng mô hình & Nạp trọng số khởi tạo
    base_model = TPSInbetweenFinetune_v31().to(device)
    init_ckpt = os.path.join(ckpt_dir, '_init_warmstart_v31.pt')
    if os.path.exists(init_ckpt):
        sd = torch.load(init_ckpt, map_location=device, weights_only=False)
        base_model.load_state_dict(sd, strict=False)
        if is_rank0:
            print(f"[✓] Nạp thành công snapshot trọng số khởi tạo: {init_ckpt}", flush=True)

    criterion = CompositeLossEngine_v31().to(device)

    # Phân nhóm tham số AdamW
    param_group_bb, param_group_un, param_group_v31 = [], [], []
    for name, param in base_model.named_parameters():
        if any(k in name for k in ['asym_analyzer', 'contour_propagator', 'asf_head']):
            param_group_v31.append(param)
        elif 'unet' in name:
            param_group_un.append(param)
        else:
            param_group_bb.append(param)

    optimizer = torch.optim.AdamW([
        {'params': param_group_bb, 'lr': 0.0, 'weight_decay': 1e-4},
        {'params': param_group_un, 'lr': 2.0e-5, 'weight_decay': 1e-4},
        {'params': param_group_v31, 'lr': 4.0e-5, 'weight_decay': 1e-4}
    ], betas=(0.9, 0.999), eps=1e-8)

    scaler = torch.amp.GradScaler('cuda', enabled=True)
    ckpt_manager = SmartCheckpointManager_v31(ckpt_dir)

    # Nạp resume nếu có
    start_epoch = 1
    history = []
    last_ep, hist = ckpt_manager.load_resume(base_model, optimizer, scaler, map_location=device)
    if last_ep > 0:
        start_epoch = last_ep + 1
        history = hist
        if is_rank0:
            print(f"[✓] RESUME V3.1: Tiếp tục từ Epoch {start_epoch:02d} (Kỷ lục điểm: {ckpt_manager.best_score:.4f})", flush=True)

    # Bọc mô hình DDP (find_unused_parameters=False loại bỏ hoàn toàn cảnh báo reducer & tăng tốc autograd)
    if is_ddp:
        ddp_model = DDP(base_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    else:
        ddp_model = base_model

    if is_rank0:
        print(f"🚀 BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN V3.1: EPOCH {start_epoch:02d} -> {total_epochs:02d}", flush=True)

    watchdog_paused = False
    train_start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        t_ep_start = time.time()
        t_train_start = time.time()

        if is_ddp and hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)

        # Cập nhật Learning Rate (Đóng băng backbone ở Ep 1 bằng LR=0.0 mà không sửa graph autograd DDP)
        if epoch == 1:
            lr_bb, lr_un, lr_v31 = 0.0, 1.8e-5, 3.5e-5
        elif epoch <= 3:
            lr_bb, lr_un, lr_v31 = 5.0e-6, 1.8e-5, 3.5e-5
        else:
            progress = float(epoch - 3) / float(max(1, total_epochs - 3))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr_bb = 1.0e-5 * max(0.1, cosine_decay)
            lr_un = 1.8e-5 * max(0.1, cosine_decay)
            lr_v31 = 3.5e-5 * max(0.1, cosine_decay)

        optimizer.param_groups[0]['lr'] = lr_bb
        optimizer.param_groups[1]['lr'] = lr_un
        optimizer.param_groups[2]['lr'] = lr_v31

        ddp_model.train()
        train_loss_accum = 0.0
        dens_accum = 0.0
        step_count = 0
        optimizer.zero_grad(set_to_none=True)

        num_batches = len(train_loader)
        t_batch_start = time.time()

        for step_idx, batch in enumerate(train_loader):
            t0 = batch[0].to(device, non_blocking=True)
            tgt = batch[1].to(device, non_blocking=True)
            t1 = batch[2].to(device, non_blocking=True)
            t_val = batch[3].to(device, non_blocking=True)
            c0_b = batch[4].to(device, non_blocking=True)
            th01_b = batch[5].to(device, non_blocking=True)
            c1_b = batch[6].to(device, non_blocking=True)
            th10_b = batch[7].to(device, non_blocking=True)
            valid_b = batch[8].to(device, non_blocking=True)

            if tgt.dtype == torch.uint8:
                tgt_f = ((255.0 - tgt.float()) * (1.0 / 255.0)).contiguous()
            else:
                tgt_f = tgt.float()

            is_accum = ((step_idx + 1) % grad_accum_steps != 0 and (step_idx + 1) != num_batches)
            sync_context = ddp_model.no_sync() if (is_ddp and is_accum) else torch.autograd.set_detect_anomaly(False)

            with sync_context:
                with torch.amp.autocast('cuda', enabled=True):
                    res = ddp_model(t0, t1, t_val, tps_precomputed=(c0_b, th01_b, c1_b, th10_b, valid_b))
                    pred = res[0]
                    v0, v1 = res[1]
                    total_l, l1_l, dice_l, dens_l, hair_l = criterion(pred, tgt_f, v0, v1, epoch=epoch)
                    loss = total_l.mean() / grad_accum_steps

                scaler.scale(loss).backward()

            loss_val = float(loss.detach().item()) * grad_accum_steps
            train_loss_accum += loss_val
            dens_accum += float(dens_l.mean().item())
            step_count += 1

            if not is_accum:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=1.0, norm_type=2.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if is_rank0:
                if args.pbar_pipe:
                    print(
                        f"__PBAR_TRAIN__:ep={epoch}|total_ep={total_epochs}|step={step_idx+1}|total={num_batches}|"
                        f"loss={loss_val:.4f}|dens={dens_accum/step_count:.3f}|lr={lr_v31:.1e}",
                        flush=True
                    )
                elif step_idx % 20 == 0 or step_idx == num_batches - 1:
                    elapsed = time.time() - t_batch_start
                    rate = (step_idx + 1) / max(0.1, elapsed)
                    pct = 100.0 * (step_idx + 1) / num_batches
                    rem_s = (num_batches - step_idx - 1) / max(0.01, rate)
                    sys.stdout.write(
                        f"\rTrain Ep {epoch:02d}/{total_epochs:02d} [LR={lr_v31:.1e}]: {pct:3.0f}% "
                        f"{step_idx+1}/{num_batches} [{elapsed/60:.1f}m<{rem_s/60:.1f}m, {1.0/max(1e-4, rate):.2f}s/it, "
                        f"loss={loss_val:.4f}, dens={dens_accum/step_count:.3f}]"
                    )
                    sys.stdout.flush()

            # Watchdog kiểm tra container RAM định kỳ
            if (step_idx + 1) % 100 == 0:
                mem_rss = psutil.Process().memory_info().rss / (1024 ** 3)
                if mem_rss >= 24.0:
                    if is_rank0:
                        if args.pbar_pipe:
                            print(f"__PBAR_TRAIN_DONE__:ep={epoch}", flush=True)
                        print(f"\n⚠️ [WATCHDOG]: RAM container = {mem_rss:.2f} GB >= 24 GB! Lưu resume và dừng tạm...", flush=True)
                        ckpt_manager.save_resume(epoch, base_model, optimizer, scaler, history, completed=False)
                    watchdog_paused = True
                    break

        if watchdog_paused:
            break

        if is_rank0:
            if args.pbar_pipe:
                print(f"__PBAR_TRAIN_DONE__:ep={epoch}", flush=True)
            else:
                print("", flush=True)

        train_time_m = (time.time() - t_train_start) / 60.0
        mean_train_loss = train_loss_accum / max(1, step_count)
        throughput = (len(train_ds) / world_size) / max(0.1, time.time() - t_train_start)

        # =====================================================================
        # THẨM ĐỊNH SONG SONG TRÊN CẢ 2 GPU
        # =====================================================================
        t_val_start = time.time()
        ddp_model.eval()

        val_stats_local = torch.zeros(10, dtype=torch.float64, device=device)
        with torch.inference_mode():
            for val_step, batch in enumerate(val_loader):
                t0 = batch[0].to(device, non_blocking=True)
                tgt = batch[1].to(device, non_blocking=True)
                t1 = batch[2].to(device, non_blocking=True)
                t_val = batch[3].to(device, non_blocking=True)
                c0_b = batch[4].to(device, non_blocking=True)
                th01_b = batch[5].to(device, non_blocking=True)
                c1_b = batch[6].to(device, non_blocking=True)
                th10_b = batch[7].to(device, non_blocking=True)
                valid_b = batch[8].to(device, non_blocking=True)
                orig_h = int(batch[10][0]) if len(batch) > 10 else 540
                orig_w = int(batch[11][0]) if len(batch) > 11 else 960

                with torch.amp.autocast('cuda', enabled=True):
                    res = ddp_model(t0, t1, t_val, tps_precomputed=(c0_b, th01_b, c1_b, th10_b, valid_b))
                    pred = res[0]

                if orig_h > 0 and orig_w > 0 and (pred.shape[-2] != orig_h or pred.shape[-1] != orig_w):
                    pred = pred[:, :, :orig_h, :orig_w]
                    tgt = tgt[:, :, :orig_h, :orig_w]

                if tgt.dtype == torch.uint8:
                    tgt = ((255.0 - tgt.float()) * (1.0 / 255.0)).contiguous()

                bs = pred.shape[0]
                vl1 = F.l1_loss(pred.float(), tgt.float())
                ssim, psnr, rec, f1, dens, gh, dgh, ch = calculate_metrics_batch_gpu(pred, tgt)

                val_stats_local[0] += float(vl1.item()) * bs
                val_stats_local[1] += float(ssim.sum().item())
                val_stats_local[2] += float(psnr.sum().item())
                val_stats_local[3] += float(rec.sum().item())
                val_stats_local[4] += float(f1.sum().item())
                val_stats_local[5] += float(dens.sum().item())
                val_stats_local[6] += float(gh.sum().item())
                val_stats_local[7] += float(dgh.sum().item())
                val_stats_local[8] += float(ch.sum().item())
                val_stats_local[9] += bs

                if is_rank0 and args.pbar_pipe:
                    print(f"__PBAR_VAL__:ep={epoch}|total_ep={total_epochs}|step={val_step+1}|total={len(val_loader)}", flush=True)

        if is_rank0 and args.pbar_pipe:
            print(f"__PBAR_VAL_DONE__:ep={epoch}", flush=True)

        if is_ddp:
            dist.all_reduce(val_stats_local, op=dist.ReduceOp.SUM)

        val_count = max(1.0, float(val_stats_local[9].item()))
        metrics_ep = {
            'val_l1': float(val_stats_local[0].item()) / val_count,
            'ssim': float(val_stats_local[1].item()) / val_count,
            'psnr': float(val_stats_local[2].item()) / val_count,
            'recall': float(val_stats_local[3].item()) / val_count,
            'f1': float(val_stats_local[4].item()) / val_count,
            'density': float(val_stats_local[5].item()) / val_count,
            'ghost': float(val_stats_local[6].item()) / val_count,
            'dark_ghost': float(val_stats_local[7].item()) / val_count,
            'chamfer': float(val_stats_local[8].item()) / val_count
        }

        val_time_m = (time.time() - t_val_start) / 60.0
        ep_time_m = (time.time() - t_ep_start) / 60.0

        if is_rank0:
            score, gates_ok, is_best, is_progress = ckpt_manager.evaluate_and_save(epoch, metrics_ep, base_model, optimizer)

            if epoch % 5 == 0 or is_best or is_progress:
                generate_checkpoint_visual_demo_v31(base_model, epoch, val_ds, device, working_dir, num_samples=3)

            mem_now = psutil.Process().memory_info().rss / (1024 ** 3)
            row = {
                'epoch': epoch,
                'train_loss': mean_train_loss,
                'val_l1': metrics_ep['val_l1'],
                'val_ssim': metrics_ep['ssim'],
                'val_psnr': metrics_ep['psnr'],
                'stroke_recall': metrics_ep['recall'],
                'stroke_f1': metrics_ep['f1'],
                'density_ratio': metrics_ep['density'],
                'ghost_pixels': metrics_ep['ghost'],
                'dark_ghost_pixels': metrics_ep['dark_ghost'],
                'composite_score': score,
                'host_ram_used_gb': mem_now,
                'train_time_minutes': train_time_m,
                'val_time_minutes': val_time_m,
                'time_minutes': ep_time_m
            }
            history.append(row)
            pd.DataFrame(history).to_csv(history_csv_path, index=False)
            ckpt_manager.save_resume(epoch, base_model, optimizer, scaler, history, completed=True)

            print(f"\n📌 HOÀN TẤT EPOCH {epoch:02d}/{total_epochs:02d}")
            print(f"⏱️ THỜI GIAN: Train = {train_time_m:.2f} phút (< 10 min) | Val = {val_time_m:.2f} phút | Tổng = {ep_time_m:.2f} phút | Tốc độ = {throughput:.1f} samples/s")
            print(f"🧠 RAM Container: {mem_now:.2f} GB")
            print(f"🔍 VAL V3.1: L1={metrics_ep['val_l1']:.4f} | SSIM={metrics_ep['ssim']:.4f} | PSNR={metrics_ep['psnr']:.2f}dB | Recall={metrics_ep['recall']*100:.1f}% | Density={metrics_ep['density']:.4f}x (Mục tiêu 1.10x)")
            print(f"🎯 Composite Score: {score:.4f} (Kỷ lục: {ckpt_manager.best_score:.4f})")
            if is_best:
                print(f"🌟 [KỶ LỤC MỚI V3.1]: Checkpoint Đỉnh Cao Đã Lưu -> model_tps_v3.1_best.pt")
            elif is_progress:
                print(f"📈 [TIẾN TRÌNH TỐT NHẤT]: Lưu checkpoint tiến trình -> model_tps_v3.1_best_progress.pt")
            sys.stdout.flush()

        if is_ddp:
            dist.barrier()

    if is_ddp:
        dist.destroy_process_group()

    if is_rank0:
        if watchdog_paused:
            print("\n⏸️ HUẤN LUYỆN TẠM DỪNG DO WATCHDOG RAM. Kernel có thể restart để train tiếp từ checkpoint resume.", flush=True)
            sys.exit(42)
        else:
            print(f"\n🎉 HOÀN TẤT HUẤN LUYỆN TOÀN BỘ {total_epochs} EPOCHS TRONG {(time.time() - train_start_time)/60:.1f} PHÚT!", flush=True)
            sys.exit(0)

if __name__ == '__main__':
    main()
