# [CELL 10]: Vòng Lặp Huấn Luyện & Thẩm Định Hoàn Chỉnh (25 Epochs V3.0 - Zero Leak, Watchdog RAM & Resume)
import os
import sys
import time
import math
import cv2
import numpy as np
import torch
from tqdm.auto import tqdm
TOTAL_EPOCHS = 25
BATCH_SIZE = 8   # 4 mẫu/GPU trên 2x T4
GRAD_ACCUM_STEPS = 2
WATCHDOG_CHECK_EVERY = 100  # kiểm tra RAM mỗi N step train
history = []
train_start_time = time.time()
history_csv_path = os.path.join(WORKING_DIR, 'training_history_v3.0.csv')

# Thiết lập bộ tối ưu hóa AdamW 3 nhóm tham số
param_group_backbone = []
param_group_unet = []
param_group_v3 = []

for name, param in base_model.named_parameters():
    if not param.requires_grad:
        continue
    if any(k in name for k in ['asym_analyzer', 'contour_propagator', 'flow_propagator', 'asf_head']):
        param_group_v3.append(param)
    elif 'unet' in name or 'fusion' in name:
        param_group_unet.append(param)
    else:
        param_group_backbone.append(param)

optimizer = torch.optim.AdamW([
    {'params': param_group_backbone, 'lr': 0.0, 'weight_decay': 1e-4},
    {'params': param_group_unet, 'lr': 1.2e-5, 'weight_decay': 1e-4},
    {'params': param_group_v3, 'lr': 2.5e-5, 'weight_decay': 1e-4}
], betas=(0.9, 0.999), eps=1e-8)

scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

# [RESUME]: Tự động train tiếp nếu có checkpoint resume (sau watchdog / restart kernel)
start_epoch = 1
_last_ep, _hist = ckpt_manager.load_resume(base_model, optimizer, scaler)
if _last_ep > 0:
    start_epoch = _last_ep + 1
    history = _hist
    print(f"[✓] RESUME: Đã nạp trạng thái sau Epoch {_last_ep:02d} -> tiếp tục từ Epoch {start_epoch:02d} "
          f"(best_score={ckpt_manager.best_score:.4f})")
del _hist

def generate_checkpoint_visual_demo_v3(model, current_epoch, val_ds, dev, working_dir, num_samples=3):
    demo_dir = os.path.join(working_dir, 'visual_samples_v3.0')
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
            batch_single = custom_collate_fn_v3([val_ds[idx]])
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
            del batch_single

            if t0.dtype == torch.uint8: t0 = ((255.0 - t0.float()) * (1.0 / 255.0)).contiguous()
            if tgt.dtype == torch.uint8: tgt = ((255.0 - tgt.float()) * (1.0 / 255.0)).contiguous()
            if t1.dtype == torch.uint8: t1 = ((255.0 - t1.float()) * (1.0 / 255.0)).contiguous()

            res = raw_model(t0, t1, t_val, tps_precomputed=(c0_b, th01_b, c1_b, th10_b, valid_b))
            pred_t, (v0, v1) = res[0], res[1]
            del res

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
            cv2.putText(panel, "Frame 0", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(panel, "V0(B)/V1(R) Gates", (w_c + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(panel, f"V3.0 Ep{current_epoch:02d} (D:{s_density:.2f})", (2*w_c + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 0), 2)
            cv2.putText(panel, "GT Target", (3*w_c + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            out_p = os.path.join(demo_dir, f"sample_ep{current_epoch:02d}_{s_idx+1:02d}_{clip_name}_v30.png")
            cv2.imwrite(out_p, panel)
            saved_files.append(out_p)
            del t0, tgt, t1, t_val, c0_b, th01_b, c1_b, th10_b, valid_b, pred_t, v0, v1, panel

    if was_training: raw_model.train()
    return saved_files

print("=" * 85)
print(f"🚀 BẮT ĐẦU HUẤN LUYỆN TINH CHỈNH TPS-INBETWEEN V3.0 ({TOTAL_EPOCHS} EPOCHS, bắt đầu từ Epoch {start_epoch:02d})")
print(f"   - Phân giải huấn luyện: 540p Toàn phần (Full-frame 544x960, Không crop) | Batch vật lý: {BATCH_SIZE} | Tích lũy: {GRAD_ACCUM_STEPS}")
print(f"   - Effective Batch Size : {BATCH_SIZE * GRAD_ACCUM_STEPS} | Workers: 4 (mmap 0-copy)")
print(f"   - RAM Container ban đầu: {format_host_ram()}")
print(f"   - Watchdog RAM: dừng êm & lưu resume khi Working Set > {RAM_SOFT_LIMIT_GB:.0f} GB")
print("=" * 85)
sys.stdout.flush()

watchdog_triggered = False

for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
    t_ep_start = time.time()

    # Lịch trình 2 giai đoạn:
    print(f"\n{'=' * 85}")
    if epoch <= 2:
        for p in param_group_backbone: p.requires_grad = False
        lr_enc = 0.0
        lr_unet = 1.2e-5
        lr_v3 = 2.5e-5 if epoch == 1 else 5.0e-5
        print(f"[*] [GIAI ĐOẠN 1: EPOCH {epoch:02d} WARMUP]: Đóng băng Backbone V2.6, khởi động êm các nhánh V3.")
    else:
        for p in param_group_backbone: p.requires_grad = True
        progress = max(0.0, min(1.0, float(epoch - 3) / max(1.0, float(TOTAL_EPOCHS - 3))))
        cos_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr_enc = 1.0e-6 + (1.0e-5 - 1.0e-6) * cos_decay
        lr_unet = 1.5e-6 + (3.0e-5 - 1.5e-6) * cos_decay
        lr_v3 = 2.0e-6 + (6.0e-5 - 2.0e-6) * cos_decay
        print(f"[*] [GIAI ĐOẠN 2: EPOCH {epoch:02d} COSINE DECAY]: Toàn mạng mở khóa, cos_decay = {cos_decay:.4f}")
    print(f"{'=' * 85}")
    sys.stdout.flush()

    optimizer.param_groups[0]['lr'] = lr_enc
    optimizer.param_groups[1]['lr'] = lr_unet
    optimizer.param_groups[2]['lr'] = lr_v3

    # Cập nhật epoch cho Loss Curriculum
    if hasattr(parallel_engine, 'current_epoch'):
        parallel_engine.current_epoch = epoch
    elif hasattr(parallel_engine, 'module'):
        parallel_engine.module.current_epoch = epoch

    # -------------------------------------------------------------------------
    # TRAIN PHASE
    # -------------------------------------------------------------------------
    parallel_engine.train()
    running_train_loss = torch.tensor(0.0, device=device)
    running_l1 = torch.tensor(0.0, device=device)
    running_dens = torch.tensor(0.0, device=device)
    optimizer.zero_grad(set_to_none=True)

    n_train_steps = len(train_loader)
    train_iter = iter(train_loader)
    train_pbar = tqdm(train_iter, total=n_train_steps, desc=f"🏋️ [Ep {epoch:02d}/{TOTAL_EPOCHS:02d}] Train", unit="batch", leave=False)
    steps_done = 0
    for step, batch in enumerate(train_pbar):
        t0 = batch[0].to(device, non_blocking=True)
        tgt = batch[1].to(device, non_blocking=True)
        t1 = batch[2].to(device, non_blocking=True)
        t_val = batch[3].to(device, non_blocking=True)
        c0_b = batch[4].to(device, non_blocking=True)
        th01_b = batch[5].to(device, non_blocking=True)
        c1_b = batch[6].to(device, non_blocking=True)
        th10_b = batch[7].to(device, non_blocking=True)
        valid_b = batch[8].to(device, non_blocking=True)
        del batch

        res_losses = parallel_engine(t0, tgt, t1, t_val, c0_b, th01_b, c1_b, th10_b, valid_b, compute_loss=True)
        loss_val = res_losses[0].mean()
        loss = loss_val / GRAD_ACCUM_STEPS
        scaler.scale(loss).backward()

        # Tích lũy trực tiếp trên GPU dạng Tensor - KHÔNG gọi .item() mỗi step để giải phóng hoàn toàn CUDA queue!
        running_train_loss += loss_val.detach()
        running_l1 += res_losses[1].mean().detach()
        running_dens += res_losses[3].mean().detach()

        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == n_train_steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % 20 == 0 or (step + 1) == n_train_steps:
            curr_loss = float(loss_val.detach().item())
            curr_dens = float(res_losses[3].mean().detach().item())
            train_pbar.set_postfix({'Loss': f"{curr_loss:.4f}", 'DensL': f"{curr_dens:.3f}"})

        del t0, tgt, t1, t_val, c0_b, th01_b, c1_b, th10_b, valid_b, res_losses, loss, loss_val
        steps_done = step + 1

        # [WATCHDOG RAM]: kiểm tra working set container định kỳ
        if (step + 1) % WATCHDOG_CHECK_EVERY == 0:
            ws_now = get_host_ram_gb()
            if ws_now > RAM_SOFT_LIMIT_GB:
                watchdog_triggered = True
                print(f"\n🛑 [WATCHDOG]: Working Set = {ws_now:.2f} GB > {RAM_SOFT_LIMIT_GB:.0f} GB tại Ep {epoch:02d} step {step+1}/{n_train_steps}!")
                break

    train_pbar.close()
    del train_pbar, train_iter  # giải phóng iterator -> shutdown toàn bộ worker
    cleanup_memory()

    if watchdog_triggered:
        optimizer.zero_grad(set_to_none=True)
        ckpt_manager.save_resume(epoch, base_model, optimizer, scaler, history, completed=False)
        print(f"💾 Đã lưu checkpoint resume -> {ckpt_manager.resume_path} (sẽ chạy lại Epoch {epoch:02d}).")
        print(f"👉 Hãy xem hướng dẫn chi tiết ở ô Markdown ngay bên dưới để Restart Kernel và tiếp tục huấn luyện với RAM sạch.")
        print(f"   RAM hiện tại: {format_host_ram()}")
        sys.stdout.flush()
        break

    mean_train_loss = float(running_train_loss.item()) / max(1, steps_done)
    mean_train_l1 = float(running_l1.item()) / max(1, steps_done)

    # -------------------------------------------------------------------------
    # VALIDATION PHASE (THUẦN FP32 - CHỐNG BẪY ĐẢO MÀNG SƯƠNG)
    # -------------------------------------------------------------------------
    parallel_engine.eval()
    if hasattr(parallel_engine, 'use_amp'):
        parallel_engine.use_amp = False
    elif hasattr(parallel_engine, 'module'):
        parallel_engine.module.use_amp = False

    val_l1_accum = torch.tensor(0.0, device=device)
    val_ssim_list = []
    val_psnr_list = []
    val_recall_list = []
    val_f1_list = []
    val_dens_list = []
    val_ghost_list = []
    val_dark_ghost_list = []
    val_chamfer_list = []

    val_iter = iter(val_loader)
    val_pbar = tqdm(val_iter, total=len(val_loader), desc=f"🔍 [Ep {epoch:02d}/{TOTAL_EPOCHS:02d}] Val  ", unit="batch", leave=False)
    with torch.inference_mode():
        for step, batch in enumerate(val_pbar):
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
            del batch

            pred = parallel_engine(t0, tgt, t1, t_val, c0_b, th01_b, c1_b, th10_b, valid_b, compute_loss=False)

            if orig_h > 0 and orig_w > 0 and (pred.shape[-2] != orig_h or pred.shape[-1] != orig_w):
                pred = pred[:, :, :orig_h, :orig_w]
                tgt = tgt[:, :, :orig_h, :orig_w]

            if tgt.dtype == torch.uint8:
                tgt = ((255.0 - tgt.float()) * (1.0 / 255.0)).contiguous()

            v_l1 = F.l1_loss(pred.float(), tgt.float())
            val_l1_accum += v_l1.detach()

            m_ssim, m_psnr, m_rec, m_f1, m_dens, m_gh, m_dgh, m_ch = calculate_metrics_batch_gpu(pred, tgt)
            val_ssim_list.extend(m_ssim.tolist())
            val_psnr_list.extend(m_psnr.tolist())
            val_recall_list.extend(m_rec.tolist())
            val_f1_list.extend(m_f1.tolist())
            val_dens_list.extend(m_dens.tolist())
            val_ghost_list.extend(m_gh.tolist())
            val_dark_ghost_list.extend(m_dgh.tolist())
            val_chamfer_list.extend(m_ch.tolist())
            del t0, tgt, t1, t_val, c0_b, th01_b, c1_b, th10_b, valid_b, pred, v_l1

    val_pbar.close()
    del val_pbar, val_iter
    if hasattr(parallel_engine, 'use_amp'): parallel_engine.use_amp = True
    elif hasattr(parallel_engine, 'module'): parallel_engine.module.use_amp = True

    metrics_ep = {
        'val_l1': float(val_l1_accum.item()) / max(1, len(val_loader)),
        'ssim': float(np.mean(val_ssim_list)),
        'psnr': float(np.mean(val_psnr_list)),
        'recall': float(np.mean(val_recall_list)),
        'f1': float(np.mean(val_f1_list)),
        'density': float(np.mean(val_dens_list)),
        'ghost': float(np.mean(val_ghost_list)),
        'dark_ghost': float(np.mean(val_dark_ghost_list)),
        'chamfer': float(np.mean(val_chamfer_list))
    }

    score, gates_ok, is_best, is_progress = ckpt_manager.evaluate_and_save(epoch, metrics_ep, base_model, optimizer)

    if epoch % 5 == 0 or is_best or is_progress:
        demo_files = generate_checkpoint_visual_demo_v3(base_model, epoch, val_dataset, device, WORKING_DIR, num_samples=3)

    # Dọn TRƯỚC rồi mới đo -> log phản ánh đúng phần RAM bị giữ lại
    cleanup_memory()
    ep_time_m = (time.time() - t_ep_start) / 60.0
    mem = get_memory_stats()

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
        'host_ram_used_gb': mem['working_set'],
        'ram_anon_gb': mem['anon'],
        'ram_file_gb': mem['file'],
        'ram_shmem_gb': mem['shmem'],
        'ram_devshm_gb': mem['devshm'],
        'ram_pss_gb': mem['pss'],
        'time_minutes': ep_time_m
    }
    history.append(row)
    pd.DataFrame(history).to_csv(history_csv_path, index=False)

    # Lưu resume sau mỗi epoch hoàn chỉnh
    ckpt_manager.save_resume(epoch, base_model, optimizer, scaler, history, completed=True)

    print(f"\n📌 ĐÃ XONG EPOCH {epoch:02d}/{TOTAL_EPOCHS:02d} ({ep_time_m:.1f} phút)")
    print(f"🧠 RAM: {format_host_ram(mem)}")
    if mem['working_set'] > RAM_WARN_GB:
        print(f"⚠️ [CẢNH BÁO RAM]: Working Set {mem['working_set']:.2f} GB > {RAM_WARN_GB:.0f} GB (watchdog dừng ở {RAM_SOFT_LIMIT_GB:.0f} GB)")
    print(f"🔍 VAL: L1={metrics_ep['val_l1']:.4f} | SSIM={metrics_ep['ssim']:.4f} | PSNR={metrics_ep['psnr']:.2f}dB | Recall={metrics_ep['recall']*100:.1f}% | Density={metrics_ep['density']:.4f}x (Mục tiêu 1.10x)")
    print(f"🎯 Sakuga Composite Score V3.0: {score:.4f} (Kỷ lục: {ckpt_manager.best_score:.4f})")
    if not gates_ok:
        print(f"⚠️ [Hard Gates BỊ CHẶN]: Density={metrics_ep['density']:.4f} (chuẩn [1.00, 1.18]), Recall={metrics_ep['recall']*100:.1f}% (chuẩn >= 90.5%)")
    if is_best:
        print(f"🌟 [KỶ LỤC MỚI V3.0]: Checkpoint Đỉnh Cao Đã Lưu -> model_tps_v3.0_best.pt")
    elif is_progress:
        print(f"📈 [TIẾN TRÌNH TỐT NHẤT]: Lưu checkpoint tiến trình -> model_tps_v3.0_best_progress.pt")

    sys.stdout.flush()

if watchdog_triggered:
    print(f"\n⏸️ HUẤN LUYỆN TẠM DỪNG DO WATCHDOG RAM. Xem hướng dẫn ở ô Markdown ngay bên dưới để Restart Kernel và tiếp tục.")
else:
    print(f"\n🎉 HOÀN TẤT HUẤN LUYỆN TOÀN BỘ {TOTAL_EPOCHS} EPOCHS TRONG {(time.time() - train_start_time)/60:.1f} PHÚT!")
sys.stdout.flush()
