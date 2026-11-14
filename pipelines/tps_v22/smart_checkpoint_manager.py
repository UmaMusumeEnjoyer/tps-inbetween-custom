# -*- coding: utf-8 -*-
"""
pipelines/tps_v22/smart_checkpoint_manager.py
Bộ Quản Lý Checkpoint Thông Minh V2.2 (Sakuga Composite Score Manager):
Triển khai quy tắc chấm điểm và lưu Best Checkpoint chuyên biệt cho V2.2:
1. Hard Gates (Điều kiện bắt buộc):
   - val_blank_rate == 0.0%
   - val_stroke_recall >= 0.915 (91.5%)
   - 0.90 <= val_density_ratio <= 1.12
2. Composite Score (S_val - Càng thấp càng tốt):
   score = 0.35 * (val_ghost / ref_ghost) +
           0.30 * (val_chamfer / ref_chamfer) +
           0.20 * ((1.0 - val_recall) / (1.0 - 0.9236)) +
           0.15 * abs(val_density_ratio - 1.0)
   (Chuẩn hóa theo baseline V2.0.2: ref_ghost=14,322.38 px, ref_chamfer=0.00098317, ref_recall=0.923635)
3. Lưu model_tps_v2.2_best.pt và cập nhật training_history_v2.2.csv.
"""

import os
import shutil
import pandas as pd
import torch


class SmartCheckpointManager_v22:
    def __init__(self,
                 save_dir,
                 history_csv_path=None,
                 ref_ghost=14322.38,
                 ref_chamfer=0.00098317,
                 ref_recall=0.923635,
                 min_recall=0.915,
                 min_density=0.90,
                 max_density=1.12,
                 gdrive_manager=None):
        self.save_dir = save_dir
        self.history_csv_path = history_csv_path or os.path.join(save_dir, 'training_history_v2.2.csv')
        self.ref_ghost = ref_ghost
        self.ref_chamfer = ref_chamfer
        self.ref_recall = ref_recall
        self.min_recall = min_recall
        self.min_density = min_density
        self.max_density = max_density
        self.gdrive = gdrive_manager

        self.best_score = float('inf') # Điểm phạt càng thấp càng tốt
        self.best_epoch = 0
        self.best_metrics = {}
        self.history_records = []

        os.makedirs(save_dir, exist_ok=True)

    def calculate_composite_score(self, ghost, chamfer, recall, density_ratio):
        """
        Tính Composite Score S_val chuẩn hóa theo baseline V2.0.2:
        score = 0.35*(ghost/ref_ghost) + 0.30*(chamfer/ref_chamfer) + 0.20*((1-recall)/(1-ref_recall)) + 0.15*|density_ratio - 1.0|
        """
        term_ghost = 0.35 * (ghost / max(1.0, self.ref_ghost))
        term_chamfer = 0.30 * (chamfer / max(1e-6, self.ref_chamfer))
        denom_recall = max(1e-4, 1.0 - self.ref_recall)
        term_recall = 0.20 * (max(0.0, 1.0 - recall) / denom_recall)
        term_density = 0.15 * abs(density_ratio - 1.0)

        score = term_ghost + term_chamfer + term_recall + term_density
        return float(score)

    def evaluate_and_save(self, model, current_epoch, train_loss, val_loss, val_metrics):
        """
        Thẩm định Hard Gates & Chấm điểm Composite Score cho checkpoint tại epoch hiện tại.
        """
        blank_rate = float(val_metrics.get('blank_rate', 0.0))
        recall = float(val_metrics.get('stroke_recall', 0.0))
        density_ratio = float(val_metrics.get('stroke_density_ratio', 1.0))
        ghost = float(val_metrics.get('ghost_pixels', 0.0))
        chamfer = float(val_metrics.get('chamfer_distance', 0.0))
        psnr = float(val_metrics.get('psnr', 0.0))
        ssim = float(val_metrics.get('ssim', 0.0))

        # 1. Kiểm tra Hard Gates
        gates_passed = True
        gate_reasons = []

        if blank_rate > 0.0:
            gates_passed = False
            gate_reasons.append(f"Blank Rate > 0.0% ({blank_rate*100:.1f}%)")

        if recall < self.min_recall:
            gates_passed = False
            gate_reasons.append(f"Recall ({recall*100:.2f}%) < {self.min_recall*100:.1f}%")

        if density_ratio < self.min_density or density_ratio > self.max_density:
            gates_passed = False
            gate_reasons.append(f"Density Ratio ({density_ratio:.4f}) ngoài dải [{self.min_density}, {self.max_density}]")

        # 2. Tính Composite Score
        composite_score = self.calculate_composite_score(ghost, chamfer, recall, density_ratio)

        # Trích xuất state_dict CPU sạch
        raw_model = model
        if hasattr(raw_model, 'module'):
            raw_model = raw_model.module
        if hasattr(raw_model, 'base_model'):
            raw_model = raw_model.base_model
        cpu_state_dict = {k: v.detach().cpu() for k, v in raw_model.state_dict().items()}

        is_best = False
        try:
            # Lưu checkpoint mới nhất
            latest_path = os.path.join(self.save_dir, 'model_tps_v2.2_latest.pt')
            torch.save(cpu_state_dict, latest_path)

            # Lưu checkpoint mỗi 5 epoch
            if current_epoch % 5 == 0:
                epoch_path = os.path.join(self.save_dir, f'model_tps_v2.2_ep{current_epoch:02d}.pt')
                torch.save(cpu_state_dict, epoch_path)
                if self.gdrive:
                    self.gdrive.sync_file(epoch_path, f'model_tps_v2.2_ep{current_epoch:02d}.pt')

            if not gates_passed:
                print(f"   ⚠️ [Hard Gates BỊ CHẶN] {', '.join(gate_reasons)} -> Từ chối phong Best Checkpoint!")
            else:
                print(f"   [✓] Hard Gates ĐẠT: Recall={recall*100:.2f}% | Density={density_ratio:.4f} | Blank={blank_rate*100:.1f}%")
                print(f"   [*] Sakuga Composite Score V2.2: {composite_score:.4f} (Kỷ lục hiện tại: {self.best_score:.4f})")

                if composite_score < self.best_score:
                    self.best_score = composite_score
                    self.best_epoch = current_epoch
                    self.best_metrics = dict(val_metrics)
                    self.best_metrics['composite_score'] = composite_score

                    best_path = os.path.join(self.save_dir, 'model_tps_v2.2_best.pt')
                    torch.save(cpu_state_dict, best_path)
                    print(f"   🌟 [KỶ LỤC MỚI V2.2!] Đã lưu Best Checkpoint Epoch {current_epoch:02d} -> {best_path}")

                    if self.gdrive:
                        self.gdrive.sync_file(best_path, 'model_tps_v2.2_best.pt')
                        self.gdrive.sync_file(latest_path, 'model_tps_v2.2_latest.pt')
                    is_best = True

            # Ghi nhận vào lịch sử
            record = {
                'epoch': current_epoch,
                'train_loss': round(float(train_loss), 4),
                'val_loss': round(float(val_loss), 4),
                'val_psnr': round(psnr, 2),
                'val_ssim': round(ssim, 4),
                'val_ghost': round(ghost, 1),
                'val_chamfer': round(chamfer, 6),
                'val_recall': round(recall, 4),
                'val_density_ratio': round(density_ratio, 4),
                'val_blank_rate': round(blank_rate, 4),
                'composite_score': round(composite_score, 4),
                'gates_passed': gates_passed,
                'is_best': is_best
            }
            self.history_records.append(record)

            # Cập nhật file CSV
            df_hist = pd.DataFrame(self.history_records)
            df_hist.to_csv(self.history_csv_path, index=False)

        finally:
            del cpu_state_dict

        return is_best
