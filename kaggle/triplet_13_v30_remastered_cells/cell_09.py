# [CELL 9]: Bộ Quản Lý Checkpoint Thông Minh V3.0-Remastered R2 (Perceptual & Topological Gates)
import os
import sys
import json
import torch

RESUME_CKPT_NAME = 'model_tps_v3.0_remastered_resume.pt'

class SmartCheckpointManager_v30R:
    def __init__(self, save_dir, 
                 ref_wcd=0.00120, ref_semd=0.00250, 
                 ref_skel_err=0.075, ref_stroke_err=0.075,
                 ref_ghost=6000.0, target_density=1.10,
                 min_recall=0.900, min_skel_recall=0.880,
                 min_density=1.05, max_density=1.18, max_wcd=0.00220):
        self.save_dir = save_dir
        self.ref_wcd = ref_wcd
        self.ref_semd = ref_semd
        self.ref_skel_err = ref_skel_err
        self.ref_stroke_err = ref_stroke_err
        self.ref_ghost = ref_ghost
        self.target_density = target_density

        # Hard Acceptance Gates
        self.min_recall = min_recall
        self.min_skel_recall = min_skel_recall
        self.min_density = min_density
        self.max_density = max_density
        self.max_wcd = max_wcd

        self.best_score = float('inf')
        self.best_progress_score = float('inf')
        self.best_epoch = 0
        self.resume_path = os.path.join(save_dir, RESUME_CKPT_NAME)
        self.top_candidates = []
        os.makedirs(save_dir, exist_ok=True)

    def calculate_score(self, m):
        density_dev = abs(m.get('density', 1.10) - self.target_density)
        semd = m.get('sliced_emd', m.get('semd', self.ref_semd))
        wcd = m.get('wcd', self.ref_wcd)
        skel_recall = m.get('skel_recall', 0.90)
        recall = m.get('recall', 0.90)
        ghost = m.get('ghost', 5000.0)

        # Công thức xếp hạng chuẩn nhận thức & hình thái học V3.0-R2:
        # SSIM trọng số 0 tuyệt đối để tránh bẫy nền trắng thưa
        score = (
            0.30 * (wcd / self.ref_wcd) +
            0.10 * (semd / self.ref_semd) +
            0.25 * ((1.0 - skel_recall) / max(1e-4, self.ref_skel_err)) +
            0.15 * ((1.0 - recall) / max(1e-4, self.ref_stroke_err)) +
            0.12 * (ghost / self.ref_ghost) +
            0.08 * (density_dev / 0.10)
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
            'top_candidates': self.top_candidates,
            'history': history or []
        }
        self._atomic_save(state, self.resume_path)

    def step(self, epoch, model, metrics, optimizer=None, scaler=None, history=None):
        score = self.calculate_score(metrics)
        raw_model = model.module if hasattr(model, 'module') else (model.base_model if hasattr(model, 'base_model') else model)
        sd = {k: v.detach().cpu() for k, v in raw_model.state_dict().items()}

        # 1. Lưu checkpoint latest
        latest_path = os.path.join(self.save_dir, 'model_tps_v3.0_remastered_latest.pt')
        self._atomic_save({'epoch': epoch, 'model': sd, 'metrics': metrics, 'score': score}, latest_path)

        # 2. Kiểm tra Hard Gates nghiệm thu
        density_ok = (self.min_density <= metrics.get('density', 0.0) <= self.max_density)
        recall_ok = (metrics.get('recall', 0.0) >= self.min_recall)
        skel_ok = (metrics.get('skel_recall', 0.0) >= self.min_skel_recall)
        wcd_ok = (metrics.get('wcd', float('inf')) <= self.max_wcd)
        gates_passed = density_ok and recall_ok and skel_ok and wcd_ok

        is_best = False
        if gates_passed and score < self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            is_best = True
            best_path = os.path.join(self.save_dir, 'model_tps_v3.0_remastered_best.pt')
            self._atomic_save({'epoch': epoch, 'model': sd, 'metrics': metrics, 'score': score}, best_path)
            print(f"🏆 [KỶ LỤC MỚI V3.0-R2] Epoch {epoch:02d}: Score={score:.4f} | WCD={metrics['wcd']:.5f} | SkelRec={metrics['skel_recall']*100:.2f}% | Recall={metrics['recall']*100:.2f}% | Dens={metrics['density']:.3f}x")

        # 3. Best Progress (theo dõi sự tiến bộ)
        if score < self.best_progress_score:
            self.best_progress_score = score
            prog_path = os.path.join(self.save_dir, 'model_tps_v3.0_remastered_best_progress.pt')
            self._atomic_save({'epoch': epoch, 'model': sd, 'metrics': metrics, 'score': score}, prog_path)

        # 4. Top candidates cho Model Soup
        cand_path = os.path.join(self.save_dir, f'model_tps_v3.0_remastered_ep{epoch:02d}.pt')
        self._atomic_save({'epoch': epoch, 'model': sd, 'metrics': metrics, 'score': score}, cand_path)
        self.top_candidates.append((score, epoch, cand_path))
        self.top_candidates.sort(key=lambda x: x[0])
        if len(self.top_candidates) > 5:
            _, _, old_path = self.top_candidates.pop(-1)
            if os.path.exists(old_path) and 'best' not in old_path and 'latest' not in old_path:
                try:
                    os.remove(old_path)
                except Exception:
                    pass

        self.save_resume(epoch, model, optimizer, scaler, history, completed=True)
        return score, is_best, gates_passed

ckpt_manager = SmartCheckpointManager_v30R(CKPT_DIR)
print("=" * 80)
print("[✓] CELL 9 hoàn tất: SmartCheckpointManager_v30R sẵn sàng (WCD + Topology Gates).")
print("=" * 80)
sys.stdout.flush()
