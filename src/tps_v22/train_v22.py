# -*- coding: utf-8 -*-
"""
src/tps_v22/train_v22.py
Script Huấn luyện Tinh chỉnh Toàn diện TPS-Inbetween V2.2 (25 Epochs):
Khởi động ấm (Warm-start) từ V2.0.2 Best Checkpoint.
Tích hợp đầy đủ 5 hạng mục cải tiến:
1. Loss V2.2: StrokeDensityRegularization, DirectionalTangentCoherence, SoftClDice.
2. Data Pipeline: Temporal Reversal Augmentation (p=0.5), Dynamic 50/50 Sampler / WeightedRandomSampler.
3. Differential LR: 1.5e-5 (Backbone/IFNet/Contextnet) vs 5.0e-5 (ASF-GR / Synthesis UNet).
4. Linear Warmup 2 epochs + CosineAnnealingLR về 1e-6, Gradient Clipping max_norm=1.0.
5. Smart Checkpoint Manager V2.2: Hard Gates + Sakuga Composite Score.
"""

import os
import sys
import time
import math
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

from src.tps_v22.losses import CompositeLossEngine_v22
from src.tps_v22.data_pipeline import (
    SakugaGoldenDataset_v22,
    SakugaBoundaryMotionWeightedSampler,
    SakugaDynamic5050Sampler
)
from src.tps_v22.smart_checkpoint_manager import SmartCheckpointManager_v22
from src.tps_v22.metrics import calculate_standardized_metrics
from src.tps_v22.optimized_engine import TPSInbetweenFinetune_v22, tps_flow_from_theta_fast


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện Tinh chỉnh TPS-Inbetween V2.2")
    parser.add_argument('--data_root', type=str,
                        default='data/processed/sakuga_golden_dataset_tps_ready_v1',
                        help='Thư mục chứa dữ liệu frames và matches')
    parser.add_argument('--dataset_csv', type=str,
                        default=None,
                        help='Đường dẫn file CSV metadata triplets')
    parser.add_argument('--warm_start', type=str,
                        default='checkpoints/experiments/v2.0/v2.0.2/tps_finetune_v2.0_artifacts/checkpoints_v1.1/model_tps_v2.0_best.pt',
                        help='Đường dẫn checkpoint khởi tạo V2.0.2 Best')
    parser.add_argument('--output_dir', type=str,
                        default='checkpoints/experiments/v2.2',
                        help='Thư mục lưu checkpoints và nhật ký huấn luyện V2.2')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Tổng số epoch huấn luyện (chuẩn 25 epochs)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Kích thước batch huấn luyện')
    parser.add_argument('--lr_backbone', type=float, default=1.5e-5,
                        help='Tốc độ học cho Backbone / IFNet / ContextNet')
    parser.add_argument('--lr_head', type=float, default=5.0e-5,
                        help='Tốc độ học cho Synthesis UNet & ASF Head')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Hệ số weight decay cho AdamW')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Ngưỡng gradient clipping')
    parser.add_argument('--p_reversal', type=float, default=0.5,
                        help='Xác suất lật chiều thời gian Temporal Reversal')
    parser.add_argument('--crop_size', type=int, default=512,
                        help='Kích thước crop ngẫu nhiên khi train')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Số luồng worker cho DataLoader')
    parser.add_argument('--eval_samples', type=int, default=100,
                        help='Số lượng mẫu validation dùng để đánh giá mỗi epoch')
    return parser.parse_args()


def get_lr_factor(epoch, total_epochs=25, warmup_epochs=2, min_lr=1e-6, target_lr=5e-5):
    """Tính tỷ số nhân learning rate theo Warmup + Cosine Annealing."""
    if epoch < warmup_epochs:
        alpha = (epoch + 1) / float(warmup_epochs)
        lr = min_lr + alpha * (target_lr - min_lr)
    else:
        progress = (epoch - warmup_epochs) / float(max(1, total_epochs - 1 - warmup_epochs))
        lr = min_lr + 0.5 * (target_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    return lr / target_lr


def train_v22(args):
    print("=" * 90)
    print("🔥 KHỞI ĐỘNG PIPELINE HUẤN LUYỆN TINH CHỈNH TPS-INBETWEEN V2.2 (25 EPOCHS)")
    print(f"   Warm-Start Checkpoint : {args.warm_start}")
    print(f"   Dataset Root          : {args.data_root}")
    print(f"   Output Directory      : {args.output_dir}")
    print(f"   Epochs                : {args.epochs}")
    print(f"   Batch Size            : {args.batch_size}")
    print(f"   Learning Rates        : Backbone={args.lr_backbone:.2e} | Heads={args.lr_head:.2e}")
    print("=" * 90)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Thiết bị huấn luyện: {device}")
    if device.type == 'cuda':
        print(f"    GPU: {torch.cuda.get_device_name(0)}")

    # Tự động phát hiện thư mục dataset có dữ liệu thực tế
    if not os.path.exists(os.path.join(args.data_root, 'train')):
        dataset_candidates = [
            'data/processed/sakuga_golden_dataset_tps_ready_v1_patch2',
            'data/processed/sakuga_golden_dataset_tps_ready'
        ]
        for dc in dataset_candidates:
            if os.path.exists(os.path.join(dc, 'train')):
                args.data_root = dc
                break

    # 1. Tìm file CSV dữ liệu
    if args.dataset_csv is None:
        candidates = [
            os.path.join(args.data_root, 'triplets_combined.csv'),
            os.path.join(args.data_root, 'triplets.csv'),
            'data/processed/sakuga_golden_dataset_tps_ready_v1_patch2/triplets_combined.csv',
            'data/processed/sakuga_golden_dataset_tps_ready/triplets.csv'
        ]
        csv_path = next((p for p in candidates if os.path.exists(p)), None)
        if csv_path is None:
            raise FileNotFoundError(f"Không tìm thấy file triplets CSV trong {args.data_root}")
    else:
        csv_path = args.dataset_csv

    print(f"[*] Nạp metadata từ: {csv_path}")
    df_all = pd.read_csv(csv_path)
    train_df = df_all[df_all['split'] == 'train'].reset_index(drop=True)
    val_df = df_all[df_all['split'] == 'val'].reset_index(drop=True)
    print(f"    Số lượng mẫu: Train={len(train_df)}, Val={len(val_df)}")

    # 2. Khởi tạo Dataset & DataLoader
    train_dataset = SakugaGoldenDataset_v22(
        root_dir=args.data_root,
        df_split=train_df,
        is_train=True,
        crop_size=args.crop_size,
        enable_temporal_reversal=True,
        p_reversal=args.p_reversal
    )

    val_dataset = SakugaGoldenDataset_v22(
        root_dir=args.data_root,
        df_split=val_df,
        is_train=False
    )

    # Khởi tạo Sampler: Kiểm tra xem có Patch mở rộng hay không
    if 'source' in train_df.columns and len(train_df['source'].unique()) > 1:
        print("[*] Áp dụng SakugaDynamic5050Sampler (Trộn 50:50 V1 và Patch mở rộng)")
        train_sampler = SakugaDynamic5050Sampler(train_df, boost_pct=0.25, shuffle=True)
    else:
        print("[*] Áp dụng SakugaBoundaryMotionWeightedSampler (Ưu tiên +25% mốc t biên & disp >= 7.0px)")
        train_sampler = SakugaBoundaryMotionWeightedSampler(train_df, boost_pct=0.25)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # 3. Khởi tạo Mô hình & Warm-Start từ V2.0.2 Best
    print("[*] Khởi tạo mô hình TPS-Inbetween V2.2 (ASF-GR Engine v1.0)...")
    model = TPSInbetweenFinetune_v22().to(device)

    # Kiểm tra đường dẫn warm-start
    ws_path = args.warm_start
    if not os.path.exists(ws_path):
        # Thử các đường dẫn thay thế
        alt_paths = [
            'checkpoints/experiments/v2.0/v2.0.2/tps_finetune_v2.0_artifacts/checkpoints_v1.1/model_tps_v2.0_best.pt',
            'checkpoints/experiments/v2.0/v2.0.2/tps_finetune_v2.0_artifacts/checkpoints_v1.1/model_tps_v1.1_best.pt',
            'checkpoints/experiments/v0.3/model_tps_v0.3_best.pt'
        ]
        ws_path = next((p for p in alt_paths if os.path.exists(p)), None)

    if ws_path and os.path.exists(ws_path):
        print(f"[*] Nạp trọng số Warm-Start từ: {ws_path} ...")
        state = torch.load(ws_path, map_location=device)
        if 'state_dict' in state:
            state = state['state_dict']
        msg = model.load_state_dict(state, strict=True)
        print(f"[✓] Warm-start thành công 100%! ({msg})")
    else:
        print("[!] CẢNH BÁO: Không tìm thấy checkpoint warm-start! Mô hình khởi tạo ngẫu nhiên.")

    # 4. Thiết lập Optimizer với Differential Learning Rates
    param_groups = [
        {
            'params': list(model.encoder.parameters()) + list(model.IFNet.parameters()),
            'lr': args.lr_backbone,
            'target_lr': args.lr_backbone,
            'name': 'backbone_flow'
        },
        {
            'params': list(model.asf_head.parameters()) + list(model.unet.parameters()),
            'lr': args.lr_head,
            'target_lr': args.lr_head,
            'name': 'heads_unet'
        }
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )

    # 5. Bộ Hàm Mất Mát V2.2
    loss_engine = CompositeLossEngine_v22(
        lambda_l1=1.00,
        lambda_vgg=0.15,
        lambda_dice=0.40,
        lambda_morph=0.20,
        lambda_ghost=0.10,
        lambda_density=0.30,
        lambda_tangent=0.20,
        lambda_cldice=0.20
    ).to(device)

    # 6. Smart Checkpoint Manager V2.2
    ckpt_manager = SmartCheckpointManager_v22(
        save_dir=args.output_dir,
        history_csv_path=os.path.join(args.output_dir, 'training_history_v2.2.csv')
    )

    # 7. VÒNG LẶP HUẤN LUYỆN CHÍNH
    TOTAL_EPOCHS = args.epochs
    print(f"\n🚀 BẮT ĐẦU HUẤN LUYỆN TRONG {TOTAL_EPOCHS} EPOCHS...")

    for epoch in range(1, TOTAL_EPOCHS + 1):
        epoch_start_time = time.time()
        model.train()

        # Cập nhật tốc độ học theo Warmup + Cosine Annealing
        for group in optimizer.param_groups:
            factor = get_lr_factor(
                epoch - 1,
                total_epochs=TOTAL_EPOCHS,
                warmup_epochs=2,
                min_lr=1e-6,
                target_lr=group['target_lr']
            )
            group['lr'] = group['target_lr'] * factor

        lr1 = optimizer.param_groups[0]['lr']
        lr2 = optimizer.param_groups[1]['lr']
        print(f"\n--- [EPOCH {epoch:02d}/{TOTAL_EPOCHS:02d}] LR_Backbone: {lr1:.2e} | LR_Heads: {lr2:.2e} ---")

        train_loss_accum = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Train Ep {epoch:02d}", unit="batch")
        for batch in pbar:
            t0, tgt, t1, t_val, c0_pad, th01_pad, c1_pad, th10_pad, valid_flag, clip_id, h, w = batch

            t0 = t0.to(device)
            tgt = tgt.to(device)
            t1 = t1.to(device)
            t_val = t_val.to(device)
            c0_pad = c0_pad.to(device)
            th01_pad = th01_pad.to(device)
            c1_pad = c1_pad.to(device)
            th10_pad = th10_pad.to(device)

            b_cur, _, h_cur, w_cur = t0.shape

            # Tính Coarse TPS Flow từ tham số đã giải
            flow01_list, flow10_list = [], []
            for b_i in range(b_cur):
                f01_i = tps_flow_from_theta_fast(th01_pad[b_i], c0_pad[b_i], h_cur, w_cur, device)
                f10_i = tps_flow_from_theta_fast(th10_pad[b_i], c1_pad[b_i], h_cur, w_cur, device)
                flow01_list.append(f01_i)
                flow10_list.append(f10_i)
            coarse_flow01 = torch.cat(flow01_list, dim=0)
            coarse_flow10 = torch.cat(flow10_list, dim=0)

            optimizer.zero_grad()

            # Forward
            pred_t = model(
                t0, t1, t_val,
                coarse_flow01=coarse_flow01,
                coarse_flow10=coarse_flow10
            )

            # Tính Loss V2.2
            total_loss, loss_dict = loss_engine(pred_t, tgt)

            total_loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            optimizer.step()

            train_loss_accum += total_loss.item()
            n_batches += 1

            pbar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'L1': f"{loss_dict['loss_l1'].item():.3f}",
                'Dice': f"{loss_dict['loss_dice'].item():.3f}",
                'Dens': f"{loss_dict['loss_density'].item():.3f}",
                'Ratio': f"{loss_dict['batch_density_ratio']:.3f}"
            })

        avg_train_loss = train_loss_accum / max(1, n_batches)

        # 8. VALIDATION & CHẤM ĐIỂM CHUẨN HÓA
        model.eval()
        val_psnr_list, val_ssim_list = [], []
        val_recall_list, val_density_list = [], []
        val_ghost_list, val_chamfer_list = [], []
        val_blank_count = 0
        val_loss_accum = 0.0
        val_count = 0

        eval_limit = min(len(val_loader), args.eval_samples)
        print(f"[*] Đang thẩm định trên {eval_limit} mẫu validation chuẩn hóa...")

        with torch.no_grad():
            for v_idx, v_batch in enumerate(val_loader):
                if v_idx >= eval_limit:
                    break
                t0, tgt, t1, t_val, c0_pad, th01_pad, c1_pad, th10_pad, valid_flag, clip_id, orig_h, orig_w = v_batch
                t0 = t0.to(device)
                tgt = tgt.to(device)
                t1 = t1.to(device)
                t_val = t_val.to(device)
                c0_pad = c0_pad.to(device)
                th01_pad = th01_pad.to(device)
                c1_pad = c1_pad.to(device)
                th10_pad = th10_pad.to(device)

                _, _, h_v, w_v = t0.shape
                f01 = tps_flow_from_theta_fast(th01_pad[0], c0_pad[0], h_v, w_v, device)
                f10 = tps_flow_from_theta_fast(th10_pad[0], c1_pad[0], h_v, w_v, device)

                pred_t = model(t0, t1, t_val, coarse_flow01=f01, coarse_flow10=f10)
                v_loss, _ = loss_engine(pred_t, tgt)
                val_loss_accum += v_loss.item()

                # Đo lường chuẩn hóa
                oh = int(orig_h.item()) if hasattr(orig_h, 'item') else int(orig_h)
                ow = int(orig_w.item()) if hasattr(orig_w, 'item') else int(orig_w)
                m = calculate_standardized_metrics(pred_t, tgt, orig_h=oh, orig_w=ow, compute_chamfer=True)

                val_psnr_list.append(m['psnr'])
                val_ssim_list.append(m['ssim'])
                val_recall_list.append(m['stroke_recall'])
                val_density_list.append(m['stroke_density_ratio'])
                if m['ghost_pixels'] >= 0:
                    val_ghost_list.append(m['ghost_pixels'])
                val_chamfer_list.append(m['chamfer_distance'])
                if m['is_blank_collapse']:
                    val_blank_count += 1
                val_count += 1

        avg_val_loss = val_loss_accum / max(1, val_count)
        val_metrics = {
            'psnr': float(np.mean(val_psnr_list)) if val_psnr_list else 0.0,
            'ssim': float(np.mean(val_ssim_list)) if val_ssim_list else 0.0,
            'stroke_recall': float(np.mean(val_recall_list)) if val_recall_list else 0.0,
            'stroke_density_ratio': float(np.mean(val_density_list)) if val_density_list else 1.0,
            'ghost_pixels': float(np.mean(val_ghost_list)) if val_ghost_list else 0.0,
            'chamfer_distance': float(np.mean(val_chamfer_list)) if val_chamfer_list else 0.0,
            'blank_rate': float(val_blank_count / max(1, val_count))
        }

        # 9. ĐÁNH GIÁ HARD GATES & LƯU CHECKPOINT V2.2
        is_best = ckpt_manager.evaluate_and_save(
            model=model,
            current_epoch=epoch,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            val_metrics=val_metrics
        )

        ep_duration = time.time() - epoch_start_time
        print(f"[*] Epoch {epoch:02d} hoàn thành trong {ep_duration:.1f}s | "
              f"Val PSNR: {val_metrics['psnr']:.2f}dB | SSIM: {val_metrics['ssim']:.4f} | "
              f"Recall: {val_metrics['stroke_recall']*100:.2f}% | Density: {val_metrics['stroke_density_ratio']:.4f} | "
              f"Ghost: {val_metrics['ghost_pixels']:.1f}px | Chamfer: {val_metrics['chamfer_distance']:.6f}")

    print("\n" + "=" * 90)
    print(f"🎉 HUẤN LUYỆN V2.2 HOÀN TẤT THÀNH CÔNG SAU {TOTAL_EPOCHS} EPOCHS!")
    print(f"   Best Epoch           : {ckpt_manager.best_epoch:02d}")
    print(f"   Best Composite Score : {ckpt_manager.best_score:.4f}")
    print(f"   Best Checkpoint Path : {os.path.join(args.output_dir, 'model_tps_v2.2_best.pt')}")
    print(f"   Training History CSV : {ckpt_manager.history_csv_path}")
    print("=" * 90)


if __name__ == '__main__':
    args = parse_args()
    train_v22(args)
