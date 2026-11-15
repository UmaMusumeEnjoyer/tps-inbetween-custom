#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ddp_worker_tps_occlusion.py
--------------------------------------------------------------------------------
ĐỘNG CƠ HUẤN LUYỆN DDP 2x GPU PHA 4 — OCCLUSION-AWARE HEAD (TPSInbetweenOcclusion)
Thử nghiệm Cách ly Độc lập: Phase 4 Occlusion Head (M4_Occ)
Giữ nguyên vẹn 100% Phase 3 Remake (Phase3_Remake / M3_Best).

Các cải tiến cốt lõi:
1. Warm-Start từ Phase 3 Remake Best (`model_tps_finetuned_best.pt`):
   Kế thừa 158/158 tensors. 2 kênh mặt nạ che khuất mới khởi tạo bias=+2.0 (Sigmoid ≈ 0.88).
2. Tối ưu hóa đồng thời 2 mục tiêu:
   - Lineart Reconstruction Loss (AdaptiveLineartLoss + Exact Density + Cleanliness)
   - Occlusion-Aware Loss:
     + L_photo (Photometric hiển thị có trọng số M0, M1)
     + L_cov (Coverage constraint: ReLU(0.95 - (M0 + M1))^2)
     + L_smooth (Edge-aware smoothness bám sát viền nét)
3. Checkpoint Manager độc lập:
   - model_tps_occlusion_best.pt
   - model_tps_occlusion_last.pt
--------------------------------------------------------------------------------
"""

import os
import sys
import gc
import time
import math
import json
import glob
import random
import argparse
import contextlib
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "65536")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

try:
    import cv2
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
except ImportError:
    cv2 = None

torch.backends.cudnn.benchmark = True

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

for _p in (
    SCRIPT_DIR,
    os.path.join(REPO_ROOT, "kaggle_experiment_occlusion"),
    os.path.join(REPO_ROOT, "kaggle_finetune_sequence_phase3"),
    os.path.join(REPO_ROOT, "kaggle_finetune_sequence_phase2"),
    os.path.join(REPO_ROOT, "pipelines"),
):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# Import model occlusion & loss
from model_occlusion import (
    TPSInbetweenOcclusion,
    load_tps_occlusion_weights,
    OcclusionAwareLoss,
    warp_normalized_flow,
    warp_pixel_flow,
)

# Import dataset & multi-layer pipeline components
from multi_layer_tps import (
    FIXED_TPS_ANCHORS,
    solve_multilayer_cache_entry,
    reconstruct_multilayer_flow_batch_from_cache,
)

from ddp_worker_tps_sequence_phase2 import (
    trim_memory,
    AdaptiveLineartLoss,
    BidirectionalStrokeWidthLoss,
    SoftClDiceLoss,
    VGGPerceptualOrLaplacianLoss,
    approx_gpu_distance_map,
    soft_dilate,
    soft_skeletonize,
    evaluate_metrics_dual_regime,
    DDPLengthWidthBatchSampler,
    AdaptiveEMA,
    _apply_paired_spatial_aug_gpu,
)

from ddp_worker_tps_sequence_phase3 import (
    Phase3ExactDensityLoss,
    Phase3ZoneCleanlinessLoss,
    SakugaMultiLayerDataset,
    pad_collate_fn_multilayer,
)


# ==============================================================================
# CHECKPOINT MANAGER ĐỘC LẬP CHO PHASE 4 OCCLUSION
# ==============================================================================
class OcclusionCheckpointManager:
    """
    Chính sách lưu trữ 2 file checkpoint cho Phase 4:
      1. model_tps_occlusion_best.pt : Trọng số EMA tốt nhất trên tập Val.
      2. model_tps_occlusion_last.pt : Trọng số Epoch gần nhất kèm trạng thái Resume đầy đủ.
    """
    def __init__(self, output_dir, is_rank0=True):
        self.output_dir = output_dir
        self.is_rank0 = is_rank0
        self.best_score = float('inf')
        self.best_epoch = 0
        os.makedirs(output_dir, exist_ok=True)
        self.best_path = os.path.join(output_dir, "model_tps_occlusion_best.pt")
        self.last_path = os.path.join(output_dir, "model_tps_occlusion_last.pt")

    def _atomic_save(self, obj, dst_path):
        tmp_path = dst_path + ".tmp"
        torch.save(obj, tmp_path)
        if os.path.exists(dst_path):
            os.remove(dst_path)
        os.rename(tmp_path, dst_path)

    def save_epoch(self, epoch, base_model, ema, optimizer, scaler, score, history, is_best):
        if not self.is_rank0:
            return

        if is_best:
            self.best_score = float(score)
            self.best_epoch = int(epoch)
            best_payload = {
                'epoch': int(epoch),
                'best_score': float(score),
                'model_state': {k: v.detach().cpu() for k, v in ema.shadow.items()},
                'raw_model_state': {k: v.detach().cpu() for k, v in base_model.state_dict().items()},
                'history': history,
                'tag': 'Phase4_Occlusion_Best',
            }
            self._atomic_save(best_payload, self.best_path)
            print(f"   💾 [BEST SAVED] {os.path.basename(self.best_path)} (Score={score:.4f} @ Ep {epoch:02d})", flush=True)

        last_payload = {
            'epoch': int(epoch),
            'best_epoch': self.best_epoch,
            'best_score': self.best_score,
            'model_state': {k: v.detach().cpu() for k, v in base_model.state_dict().items()},
            'ema_state': {'shadow': {k: v.detach().cpu() for k, v in ema.shadow.items()}},
            'optimizer_state': optimizer.state_dict(),
            'scaler_state': scaler.state_dict() if scaler is not None else None,
            'history': history,
            'tag': 'Phase4_Occlusion_Last',
        }
        self._atomic_save(last_payload, self.last_path)
        print(f"   💾 [LAST SAVED] {os.path.basename(self.last_path)} (Epoch {epoch:02d} complete)", flush=True)

    def try_resume(self, base_model, ema, optimizer, scaler, map_location='cpu'):
        if not os.path.exists(self.last_path):
            return 0, []
        try:
            ckpt = torch.load(self.last_path, map_location=map_location, weights_only=False)
            epoch = ckpt.get('epoch', 0)
            self.best_epoch = ckpt.get('best_epoch', epoch)
            self.best_score = ckpt.get('best_score', float('inf'))
            base_model.load_state_dict(ckpt['model_state'])
            if 'ema_state' in ckpt and hasattr(ema, 'shadow'):
                ema.shadow = {k: v.to(map_location) for k, v in ckpt['ema_state']['shadow'].items()}
            if optimizer is not None and 'optimizer_state' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state'])
            if scaler is not None and ckpt.get('scaler_state') is not None:
                scaler.load_state_dict(ckpt['scaler_state'])
            history = ckpt.get('history', [])
            return int(epoch), history
        except Exception as e:
            if self.is_rank0:
                print(f"   [!] Resume failed ({e}), starting fresh.", flush=True)
            return 0, []


# ==============================================================================
# HÀM VALIDATION CHO PHASE 4
# ==============================================================================
def validate_epoch_occlusion(val_loader, model, criterion, occ_loss_fn, device, is_rank0=True):
    model.eval()
    val_loss_sum = 0.0
    val_frames = 0
    all_metrics = []

    with torch.no_grad():
        for batch in val_loader:
            h_pad, w_pad = batch['h_pad'], batch['w_pad']
            orig_hw = batch['orig_hw']
            x0 = batch['x0'].to(device, non_blocking=True)
            x1 = batch['x1'].to(device, non_blocking=True)
            xk = batch['xk'].to(device, non_blocking=True)
            t_val = batch['t_val'].to(device, non_blocking=True).view(-1, 1, 1, 1)

            cf01, cf10 = reconstruct_multilayer_flow_batch_from_cache(
                batch['c0_dyn'].to(device, non_blocking=True),
                batch['th01_dyn'].to(device, non_blocking=True),
                batch['c1_dyn'].to(device, non_blocking=True),
                batch['th10_dyn'].to(device, non_blocking=True),
                batch['valid_dyn'].to(device, non_blocking=True),
                batch['dyn_pts0'].to(device, non_blocking=True),
                batch['dyn_pts1'].to(device, non_blocking=True),
                batch['dyn_pts_len'].to(device, non_blocking=True),
                batch['static_pts0'].to(device, non_blocking=True),
                batch['static_pts1'].to(device, non_blocking=True),
                batch['num_static'].to(device, non_blocking=True),
                batch['num_clusters'].to(device, non_blocking=True),
                orig_hw, h_pad, w_pad, device, downscale_factor=2,
                use_dyn_anchors=True, transition_width_px=22.0, falloff_radius_px=105.0
            )

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                pred, cw, f0, f1, fl, M0, M1 = model(x0, x1, t_val, cf01, cf10)
                l_main, d_main = criterion(pred, xk, valid_hw=orig_hw)
                l_occ, d_occ = occ_loss_fn(f0, f1, M0, M1, xk)
                l_total = l_main + l_occ

            b = x0.shape[0]
            val_loss_sum += float(l_total.item()) * b
            val_frames += b

            for i in range(b):
                hi, wi = int(orig_hw[i, 0].item()), int(orig_hw[i, 1].item())
                gt_i = xk[i:i + 1, :, :hi, :wi]
                pr_i = pred[i:i + 1, :, :hi, :wi]
                m = evaluate_metrics_dual_regime(pr_i, gt_i)
                m_batch = {k: float(m[k].item()) for k in m}
                m_batch['M0_mean'] = float(M0[i:i + 1, :, :hi, :wi].mean().item())
                m_batch['M1_mean'] = float(M1[i:i + 1, :, :hi, :wi].mean().item())
                m_batch['loss_occ_photo'] = d_occ['loss_occ_photo']
                m_batch['loss_occ_cov'] = d_occ['loss_occ_cov']
                all_metrics.append(m_batch)

    avg_loss = val_loss_sum / max(1, val_frames)
    summary = {'val_loss': avg_loss}
    if all_metrics:
        for k in all_metrics[0]:
            vals = [m[k] for m in all_metrics if k in m]
            summary[k] = float(np.mean(vals))
    return summary


# ==============================================================================
# HÀM MAIN HUẤN LUYỆN DDP CHO PHASE 4
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Phase 4 Occlusion Head DDP Training Worker")
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--train_clips_dir", type=str, required=True)
    parser.add_argument("--val_clips_dir", type=str, required=True)
    parser.add_argument("--tps_cache", type=str, required=True)
    parser.add_argument("--init_checkpoint", type=str, required=True, help="Path to Phase 3 Best checkpoint")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_phase4_occlusion")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--max_chunk_frames", type=int, default=3)
    parser.add_argument("--lr_backbone", type=float, default=6.0e-6)
    parser.add_argument("--lr_unet", type=float, default=1.8e-5)
    parser.add_argument("--ema_decay", type=float, default=0.996)
    parser.add_argument("--synth_count", type=int, default=150)
    parser.add_argument("--static_aug_prob", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--w_photo", type=float, default=0.25)
    parser.add_argument("--w_cov", type=float, default=0.50)
    parser.add_argument("--w_smooth", type=float, default=0.10)
    parser.add_argument("--w_ghost", type=float, default=0.32)
    parser.add_argument("--w_temp", type=float, default=0.55)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp = world_size > 1
    is_rank0 = (local_rank == 0)

    if is_ddp:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if is_rank0:
        print("=" * 84)
        print(f"🚀 [PHASE 4 OCCLUSION HEAD] KHỞI ĐỘNG DDP HUẤN LUYỆN ({world_size} GPUs)")
        print(f"   Warm-Start Input Checkpoint: {args.init_checkpoint}")
        print(f"   Output Checkpoint Dir      : {args.output_dir}")
        print(f"   Epochs: {args.epochs} | LR Backbone: {args.lr_backbone:.2e} | LR UNet: {args.lr_unet:.2e}")
        print(f"   Aux Loss Weights           : Photo={args.w_photo}, Cov={args.w_cov}, Smooth={args.w_smooth}")
        print("=" * 84, flush=True)

    # 1. Khởi tạo mô hình & nạp Warm-Start từ Phase 3 Remake
    base_model = TPSInbetweenOcclusion().to(device)
    load_tps_occlusion_weights(base_model, args.init_checkpoint, device=device, prefer_ema=True, is_rank0=is_rank0)

    ema = AdaptiveEMA(base_model, max_decay=args.ema_decay)
    ckpt_mgr = OcclusionCheckpointManager(args.output_dir, is_rank0=is_rank0)

    # 2. Khởi tạo Optimizer & Scaler
    backbone_params = list(base_model.encoder.parameters()) + list(base_model.IFNet.parameters())
    unet_params = list(base_model.unet.parameters())
    optimizer = torch.optim.AdamW([
        {'params': base_model.encoder.parameters(), 'lr': args.lr_backbone, 'weight_decay': 1e-4},
        {'params': base_model.IFNet.parameters(), 'lr': args.lr_backbone, 'weight_decay': 1e-4},
        {'params': unet_params, 'lr': args.lr_unet, 'weight_decay': 1e-4},
    ])
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # 3. Hàm mất mát
    lineart_loss = AdaptiveLineartLoss(train_mode="LONG", is_rank0=is_rank0, w_ghost=args.w_ghost, w_temp=args.w_temp).to(device)
    occ_loss_fn = OcclusionAwareLoss(
        w_photo=args.w_photo, w_cov=args.w_cov, w_smooth=args.w_smooth, cov_margin=0.95, smooth_alpha=10.0
    ).to(device)

    # 4. DataLoaders (kế thừa SakugaMultiLayerDataset & pad_collate_fn_multilayer)
    train_ds = SakugaMultiLayerDataset(
        args.train_csv, args.train_clips_dir, args.tps_cache,
        train_mode="LONG", is_train=True, seed=args.seed,
        synth_count=args.synth_count, static_aug_prob=args.static_aug_prob
    )
    val_ds = SakugaMultiLayerDataset(
        args.val_csv, args.val_clips_dir, args.tps_cache,
        train_mode="VAL", is_train=False, seed=args.seed,
        synth_count=0, static_aug_prob=0.0
    )

    train_sampler = DDPLengthWidthBatchSampler(
        train_ds.samples, world_size=world_size, rank=local_rank,
        batch_n3=4, batch_n5=2, batch_n7=1, seed=args.seed
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler, collate_fn=pad_collate_fn_multilayer,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0)
    )
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_ddp else None
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False, sampler=val_sampler,
        collate_fn=pad_collate_fn_multilayer, num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'), persistent_workers=(args.num_workers > 0), drop_last=False
    )

    # 5. Vòng lặp huấn luyện
    start_epoch = 1
    history = []
    last_ep, hist = ckpt_mgr.try_resume(base_model, ema, optimizer, scaler, map_location=device)
    if last_ep > 0:
        start_epoch = last_ep + 1
        history = hist
        if is_rank0:
            print(f"[♻️ RESUME] Tiếp tục huấn luyện từ Epoch {start_epoch:02d}...", flush=True)

    if is_ddp:
        ddp_model = DDP(base_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    else:
        ddp_model = base_model

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        if hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)

        # Lịch LR: Cosine decay nhẹ
        lr_scale = 0.5 * (1.0 + math.cos(math.pi * float(epoch - 1) / float(max(1, args.epochs))))
        lr_scale = max(0.20, lr_scale)
        optimizer.param_groups[0]['lr'] = args.lr_backbone * lr_scale
        optimizer.param_groups[1]['lr'] = args.lr_backbone * lr_scale
        optimizer.param_groups[2]['lr'] = args.lr_unet * lr_scale

        ddp_model.train()
        optimizer.zero_grad(set_to_none=True)
        acc_loss = 0.0
        acc_frames = 0
        num_batches = len(train_loader)
        grad_accum = max(1, args.grad_accum)

        for b_idx, batch in enumerate(train_loader):
            h_pad, w_pad = batch['h_pad'], batch['w_pad']
            orig_hw = batch['orig_hw'].to(device, non_blocking=True)
            x0 = batch['x0'].to(device, non_blocking=True)
            x1 = batch['x1'].to(device, non_blocking=True)
            xk_all = batch['xk_all'].to(device, non_blocking=True)
            t_vals = batch['t_vals'].to(device, non_blocking=True)
            is_direct = bool(batch.get('is_direct_t_flows', False))
            bsz, m_steps = xk_all.shape[0], xk_all.shape[1]

            if is_direct:
                cf01_steps, cf10_steps = [], []
                for m_i in range(m_steps):
                    cf0_i, cf1_i = reconstruct_multilayer_flow_batch_from_cache(
                        batch['c0_dyn_all'][:, m_i].to(device, non_blocking=True),
                        batch['th01_dyn_all'][:, m_i].to(device, non_blocking=True),
                        batch['c1_dyn_all'][:, m_i].to(device, non_blocking=True),
                        batch['th10_dyn_all'][:, m_i].to(device, non_blocking=True),
                        batch['valid_dyn_all'][:, m_i].to(device, non_blocking=True),
                        batch['dyn_pts0_all'][:, m_i].to(device, non_blocking=True),
                        batch['dyn_pts1_all'][:, m_i].to(device, non_blocking=True),
                        batch['dyn_pts_len_all'][:, m_i].to(device, non_blocking=True),
                        batch['static_pts0_all'][:, m_i].to(device, non_blocking=True),
                        batch['static_pts1_all'][:, m_i].to(device, non_blocking=True),
                        batch['num_static_all'][:, m_i].to(device, non_blocking=True),
                        batch['num_clusters_all'][:, m_i].to(device, non_blocking=True),
                        orig_hw, h_pad, w_pad, device, downscale_factor=2,
                        use_dyn_anchors=False, transition_width_px=18.0, falloff_radius_px=160.0
                    )
                    cf01_steps.append(cf0_i)
                    cf10_steps.append(cf1_i)
                cf01_f = torch.stack(cf01_steps, dim=1).reshape(bsz * m_steps, 2, h_pad, w_pad)
                cf10_f = torch.stack(cf10_steps, dim=1).reshape(bsz * m_steps, 2, h_pad, w_pad)
            else:
                cf01, cf10 = reconstruct_multilayer_flow_batch_from_cache(
                    batch['c0_dyn'].to(device, non_blocking=True),
                    batch['th01_dyn'].to(device, non_blocking=True),
                    batch['c1_dyn'].to(device, non_blocking=True),
                    batch['th10_dyn'].to(device, non_blocking=True),
                    batch['valid_dyn'].to(device, non_blocking=True),
                    batch['dyn_pts0'].to(device, non_blocking=True),
                    batch['dyn_pts1'].to(device, non_blocking=True),
                    batch['dyn_pts_len'].to(device, non_blocking=True),
                    batch['static_pts0'].to(device, non_blocking=True),
                    batch['static_pts1'].to(device, non_blocking=True),
                    batch['num_static'].to(device, non_blocking=True),
                    batch['num_clusters'].to(device, non_blocking=True),
                    orig_hw, h_pad, w_pad, device, downscale_factor=2,
                    use_dyn_anchors=True, transition_width_px=22.0, falloff_radius_px=105.0
                )
                _apply_paired_spatial_aug_gpu(x0, x1, xk_all, cf01, cf10, orig_hw, is_sequence=True)
                cf01_f = cf01.unsqueeze(1).expand(-1, m_steps, -1, -1, -1).reshape(bsz * m_steps, 2, h_pad, w_pad)
                cf10_f = cf10.unsqueeze(1).expand(-1, m_steps, -1, -1, -1).reshape(bsz * m_steps, 2, h_pad, w_pad)

            x0_f = x0.unsqueeze(1).expand(-1, m_steps, -1, -1, -1).reshape(bsz * m_steps, 1, h_pad, w_pad)
            x1_f = x1.unsqueeze(1).expand(-1, m_steps, -1, -1, -1).reshape(bsz * m_steps, 1, h_pad, w_pad)
            xk_f = xk_all.reshape(bsz * m_steps, 1, h_pad, w_pad)
            t_f = t_vals.reshape(bsz * m_steps, 1, 1, 1)
            hw_f = orig_hw.unsqueeze(1).expand(-1, m_steps, -1).reshape(bsz * m_steps, 2)

            total_f = x0_f.shape[0]
            do_opt = ((b_idx + 1) % grad_accum == 0) or ((b_idx + 1) == num_batches)
            chunk_sz = max(1, args.max_chunk_frames)
            batch_loss_sum = 0.0

            for c_start in range(0, total_f, chunk_sz):
                c_end = min(c_start + chunk_sz, total_f)
                need_sync = (do_opt and c_end == total_f)
                sync_ctx = (ddp_model.no_sync() if (is_ddp and not need_sync) else contextlib.nullcontext())

                with sync_ctx:
                    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        pred_c, cw_c, f0_c, f1_c, fl_c, M0_c, M1_c = ddp_model(
                            x0_f[c_start:c_end], x1_f[c_start:c_end],
                            t_f[c_start:c_end], cf01_f[c_start:c_end], cf10_f[c_start:c_end],
                            direct_t_flows=is_direct
                        )
                        gt_c = xk_f[c_start:c_end]
                        hw_c = hw_f[c_start:c_end]

                        l_main, d_main = lineart_loss(pred_c, gt_c, valid_hw=hw_c)
                        l_occ, d_occ = occ_loss_fn(f0_c, f1_c, M0_c, M1_c, gt_c)
                        l_step = l_main + l_occ
                        scaled_l = l_step * (float(c_end - c_start) / float(total_f * grad_accum))

                    scaler.scale(scaled_l).backward()
                    batch_loss_sum += float(l_step.detach().item()) * (c_end - c_start)

            acc_loss += batch_loss_sum
            acc_frames += total_f

            if do_opt:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(base_model)

            if args.dry_run and b_idx >= 1:
                break

        train_loss = acc_loss / max(1, acc_frames)
        t_train = (time.time() - t0) / 60.0

        # Đánh giá trên tập Validation (sử dụng trọng số EMA mượt mà)
        t_val0 = time.time()
        ema.apply_shadow(base_model)
        val_m = validate_epoch_occlusion(val_loader, base_model, lineart_loss, occ_loss_fn, device, is_rank0=is_rank0)
        ema.restore(base_model)
        t_val = (time.time() - t_val0) / 60.0

        # Composite score để chọn checkpoint tốt nhất
        comp_score = val_m.get('val_loss', 0.0) + 0.1 * val_m.get('loss_occ_cov', 0.0)
        is_best = comp_score < ckpt_mgr.best_score

        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_m.get('val_loss', 0.0),
            'val_psnr': val_m.get('psnr', 0.0),
            'val_ssim': val_m.get('ssim', 0.0),
            'val_skel_recall': val_m.get('skel_recall', 0.0),
            'val_tip_recall': val_m.get('tip_recall', 0.0),
            'val_M0_mean': val_m.get('M0_mean', 0.0),
            'val_M1_mean': val_m.get('M1_mean', 0.0),
            'comp_score': comp_score,
            'train_time_min': t_train,
            'val_time_min': t_val,
        }
        history.append(row)

        if is_rank0:
            hist_csv = os.path.join(args.output_dir, "training_history_occlusion.csv")
            pd.DataFrame(history).to_csv(hist_csv, index=False)
            ckpt_mgr.save_epoch(epoch, base_model, ema, optimizer, scaler, comp_score, history, is_best=is_best)
            print(
                f"📌 EP {epoch:02d}/{args.epochs:02d} [Phase4_Occ] | Train={t_train:.2f}m | Val={t_val:.2f}m | "
                f"Loss={train_loss:.4f} | ValLoss={row['val_loss']:.4f} | PSNR={row['val_psnr']:.2f}dB | "
                f"SkelRec={row['val_skel_recall']:.3f} | M0={row['val_M0_mean']:.3f} | Best={'YES' if is_best else 'NO'}",
                flush=True
            )

        if args.dry_run:
            break

    if is_rank0:
        print("=" * 84)
        print("🎉 [PHASE 4 OCCLUSION HEAD] HUẤN LUYỆN HOÀN TẤT THÀNH CÔNG!")
        print(f"   Best Checkpoint: {ckpt_mgr.best_path}")
        print(f"   Last Checkpoint: {ckpt_mgr.last_path}")
        print("=" * 84, flush=True)


if __name__ == "__main__":
    main()
