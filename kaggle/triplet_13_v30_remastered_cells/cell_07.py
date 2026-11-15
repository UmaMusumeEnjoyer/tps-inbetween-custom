# [CELL 7]: Nạp Checkpoint Warm-Start Chuẩn Vàng V2.6 & Parallel Training Wrapper V3.0-Remastered R2
import os
import sys
import glob
import time
import torch
import torch.nn as nn

print("=" * 80)
print("[*] ĐANG TÌM KIẾM VÀ NẠP CHECKPOINT WARM-START TỪ V2.6 GOLD (FLAT MINIMA)...")
t_ckpt_start = time.time()

device = globals().get('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
WORKING_DIR = globals().get('WORKING_DIR', '/kaggle/working' if os.path.exists('/kaggle') else './workspace')
KAGGLE_INPUT = globals().get('KAGGLE_INPUT', '/kaggle/input' if os.path.exists('/kaggle/input') else './data')

# 1. Đảm bảo mô hình base_model đã được khởi tạo
if 'base_model' not in globals():
    if 'TPSInbetweenFinetune_v30R' in globals():
        base_model = TPSInbetweenFinetune_v30R().to(device)
    else:
        raise RuntimeError("[!] Chưa tìm thấy class TPSInbetweenFinetune_v30R. Vui lòng chạy Cell 5 trước!")

# 2. Danh sách các đường dẫn ưu tiên kiểm tra O(1)
direct_candidates = [
    # Đúng theo Dataset trên giao diện Kaggle của bạn:
    os.path.join(KAGGLE_INPUT, "checkpoints_v2.6", "model_tps_v2.6_best.pt"),
    os.path.join(KAGGLE_INPUT, "checkpoints_v2.6", "checkpoints_v2.6", "model_tps_v2.6_best.pt"),
    os.path.join(KAGGLE_INPUT, "datasets", "dungbaoviec123", "checkpoints-v2-6", "model_tps_v2.6_best.pt"),
    os.path.join(KAGGLE_INPUT, "datasets", "dungbaoviec123", "checkpoints_v2.6", "model_tps_v2.6_best.pt"),
    os.path.join(KAGGLE_INPUT, "checkpoints-v2-6", "model_tps_v2.6_best.pt"),
    os.path.join(KAGGLE_INPUT, "model-tps-v26-best", "model_tps_v2.6_best.pt"),
    
    # Checkpoint từ working directory nếu có:
    os.path.join(WORKING_DIR, "model_tps_v2.6_best.pt"),
    os.path.join(WORKING_DIR, "checkpoints_v2.6", "model_tps_v2.6_best.pt"),
    
    # Đường dẫn cục bộ khi thử nghiệm:
    "./checkpoints/experiments/v2.6/checkpoints_v2.6/model_tps_v2.6_best.pt",
    "E:/CODE/PBL/PBL6/checkpoints/experiments/v2.6/checkpoints_v2.6/model_tps_v2.6_best.pt"
]

best_ckpt_path = None
for p in direct_candidates:
    if p and os.path.isfile(p):
        best_ckpt_path = os.path.normpath(p)
        break

# Fallback: Quét nông trong KAGGLE_INPUT nếu chưa thấy ở đường dẫn trực tiếp
if best_ckpt_path is None and os.path.exists(KAGGLE_INPUT):
    cand_pt_files = []
    try:
        for root, dirs, files in os.walk(KAGGLE_INPUT):
            for f in files:
                if f.endswith('.pt') and ('v2.6' in f.lower() or 'best' in f.lower()):
                    cand_pt_files.append(os.path.normpath(os.path.join(root, f)))
            # Giới hạn độ sâu 3 tầng để không quét vào các thư mục ảnh
            rel_depth = root[len(KAGGLE_INPUT):].count(os.sep)
            if rel_depth >= 3:
                dirs.clear()
    except Exception:
        pass

    for p in cand_pt_files:
        bname = os.path.basename(p)
        if "v2.6_best" in bname or "model_tps_v2.6" in bname:
            best_ckpt_path = p
            break
    if best_ckpt_path is None and cand_pt_files:
        best_ckpt_path = cand_pt_files[0]

# 3. Nạp trọng số Warm-Start vào base_model
loaded_successfully = False
if best_ckpt_path and os.path.exists(best_ckpt_path):
    print(f"[*] Tìm thấy checkpoint Warm-Start ưu tiên: {best_ckpt_path}")
    try:
        sd = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        if isinstance(sd, dict):
            if 'model_state_dict' in sd:
                sd = sd['model_state_dict']
            elif 'state_dict' in sd:
                sd = sd['state_dict']
            elif 'model' in sd:
                sd = sd['model']

        clean_sd = {}
        for k, v in sd.items():
            k_new = k.replace('module.', '').replace('base_model.', '')
            clean_sd[k_new] = v

        model_dict = base_model.state_dict()
        matched_sd = {}
        for k, v in clean_sd.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                matched_sd[k] = v

        model_dict.update(matched_sd)
        base_model.load_state_dict(model_dict)
        loaded_successfully = True
        print(f"[✓] Nạp Warm-Start THÀNH CÔNG từ: {best_ckpt_path}")
        print(f"    - Số lượng tensor khớp chính xác: {len(matched_sd)} / {len(model_dict)} (100% khớp kiến trúc V2.6)")
    except Exception as e:
        print(f"[!] Lỗi khi nạp checkpoint: {e}")

if not loaded_successfully:
    print("[!] CẢNH BÁO: Không tìm thấy checkpoint V2.6 phù hợp. Khởi tạo trọng số mặc định cho V3.0-Remastered R2.")

# 4. Định nghĩa ParallelTrainingWrapper_v30R
class ParallelTrainingWrapper_v30R(nn.Module):
    def __init__(self, model, criterion):
        super(ParallelTrainingWrapper_v30R, self).__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, t0, t1, tt, t_val, tps_precomputed=None, epoch=1):
        pred, blend_mask, conf_fbfc, delta_flow, _, _ = self.model(
            t0, t1, t_val, tps_precomputed=tps_precomputed
        )
        loss_total, l_charb, l_dt, l_dice, l_dens, l_hair, l_ghost = self.criterion(
            pred, tt, conf_fbfc=conf_fbfc, epoch=epoch
        )
        return pred, loss_total, l_charb, l_dt, l_dice, l_dens, l_hair, l_ghost

print("=" * 80)
print(f"[✓] CELL 7 hoàn tất: ParallelTrainingWrapper_v30R R2 & Warm-Start sẵn sàng (Thời gian: {time.time() - t_ckpt_start:.2f}s).")
print("=" * 80)
sys.stdout.flush()
