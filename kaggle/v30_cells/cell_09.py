# [CELL 9]: Bộ Quản Lý Checkpoint Thông Minh V3.1 (Hard Gates Chuẩn Vàng 1.10x + Resume Checkpoint)
import os
import sys
import json
import torch

RESUME_CKPT_NAME = 'model_tps_v3.0_resume.pt'

class SmartCheckpointManager_v30:
    def __init__(self, save_dir, ref_ghost=9070.9, ref_chamfer=0.00230, ref_recall=0.930,
                 ref_ssim=0.900, ref_val_l1=0.0308, ref_dark_ghost=750.0,
                 min_recall=0.905, min_density=1.00, max_density=1.18, target_density=1.10):
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
        """Lưu trạng thái đầy đủ để train tiếp. completed=False nghĩa là epoch `epoch` bị ngắt giữa chừng
        (watchdog) -> khi resume sẽ chạy lại chính epoch đó."""
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
        del state

    def load_resume(self, model, optimizer=None, scaler=None, map_location='cpu'):
        """Trả về (last_completed_epoch, history) hoặc (0, []) nếu không có resume."""
        if not os.path.exists(self.resume_path):
            return 0, []
        state = torch.load(self.resume_path, map_location=map_location, weights_only=False)
        raw_model = model.module if hasattr(model, 'module') else (model.base_model if hasattr(model, 'base_model') else model)
        raw_model.load_state_dict(state['model'], strict=False)
        if optimizer is not None and state.get('optimizer') is not None:
            try:
                optimizer.load_state_dict(state['optimizer'])
            except Exception as e:
                print(f"[!] Không nạp được optimizer state ({e}), dùng optimizer mới.")
        if scaler is not None and state.get('scaler') is not None:
            try:
                scaler.load_state_dict(state['scaler'])
            except Exception:
                pass
        self.best_score = state.get('best_score', float('inf'))
        self.best_progress_score = state.get('best_progress_score', float('inf'))
        self.best_epoch = state.get('best_epoch', 0)
        ep = int(state.get('epoch', 0))
        hist = [h for h in state.get('history', []) if int(h.get('epoch', 0)) <= ep]
        del state
        return ep, hist

    def evaluate_and_save(self, epoch, metrics, model, optimizer=None):
        score = self.calculate_score(metrics)
        raw_model = model.module if hasattr(model, 'module') else (model.base_model if hasattr(model, 'base_model') else model)
        sd = {k: v.detach().cpu() for k, v in raw_model.state_dict().items()}

        # 1. Lưu checkpoint latest
        latest_path = os.path.join(self.save_dir, 'model_tps_v3.0_latest.pt')
        self._atomic_save(sd, latest_path)

        # 2. Kiểm tra Hard Gates
        density_ok = (self.min_density <= metrics['density'] <= self.max_density)
        recall_ok = (metrics['recall'] >= self.min_recall)
        gates_passed = (density_ok and recall_ok)

        # 3. Cập nhật Best Checkpoint khi thỏa mãn Hard Gates
        is_best = False
        if gates_passed and (score < self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            is_best = True
            best_path = os.path.join(self.save_dir, 'model_tps_v3.0_best.pt')
            self._atomic_save(sd, best_path)

        # 4. Cập nhật Best Progress
        is_progress = False
        if score < self.best_progress_score:
            self.best_progress_score = score
            is_progress = True
            prog_path = os.path.join(self.save_dir, 'model_tps_v3.0_best_progress.pt')
            self._atomic_save(sd, prog_path)

        del sd
        return score, gates_passed, is_best, is_progress

ckpt_manager = SmartCheckpointManager_v30(CKPT_DIR)
print(f"[✓] SmartCheckpointManager_v30 khởi tạo thành công tại: {CKPT_DIR}")
if os.path.exists(ckpt_manager.resume_path):
    print(f"[*] Phát hiện checkpoint resume: {ckpt_manager.resume_path} (Cell 10 sẽ tự train tiếp)")
sys.stdout.flush()
