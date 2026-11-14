# [CELL 7]: Nạp Checkpoint Warm-Start (V3.1 Best Progress / V2.6 Gold) & Parallel Training Wrapper V3.2
import os
import sys
import glob
import time
import torch
import torch.nn as nn

print("=" * 80)
print("[*] ĐANG TÌM KIẾM VÀ NẠP CHECKPOINT WARM-START CHO V3.2...")
t_ckpt_start = time.time()

if 'KAGGLE_INPUT' not in globals():
    KAGGLE_INPUT = '/kaggle/input' if os.path.exists('/kaggle/input') else './data'

# In cấu trúc thư mục input để thẩm định trực quan
if os.path.exists(KAGGLE_INPUT):
    try:
        top_entries = os.listdir(KAGGLE_INPUT)
        print(f"[*] Các mục tại {KAGGLE_INPUT}: {top_entries}")
        if 'datasets' in top_entries:
            ds_path = os.path.join(KAGGLE_INPUT, 'datasets')
            try:
                ds_entries = os.listdir(ds_path)
                print(f"[*] Các mục trong {ds_path}: {ds_entries}")
                for entry in ds_entries[:5]:
                    sub_p = os.path.join(ds_path, entry)
                    if os.path.isdir(sub_p):
                        print(f"    - {entry}: {os.listdir(sub_p)}")
            except Exception as e:
                print(f"[*] Không thể listdir {ds_path}: {e}")
    except Exception as e:
        print(f"[*] Không thể listdir {KAGGLE_INPUT}: {e}")

# 1. Tìm kiếm nhanh bằng Glob mẫu đường dẫn Kaggle Datasets
discovered_pts = []
direct_patterns = [
    os.path.join(KAGGLE_INPUT, "*", "model_tps_v3.1_best_progress.pt"),
    os.path.join(KAGGLE_INPUT, "datasets", "*", "*", "model_tps_v3.1_best_progress.pt"),
    os.path.join(WORKING_DIR, "checkpoints_v3.1", "model_tps_v3.1_best_progress.pt"),
    os.path.join(KAGGLE_INPUT, "datasets", "*", "*", "model_tps_v2.6_best.pt"),
    os.path.join(KAGGLE_INPUT, "datasets", "*", "model_tps_v2.6_best.pt"),
    os.path.join(KAGGLE_INPUT, "*", "model_tps_v2.6_best.pt"),
    os.path.join(KAGGLE_INPUT, "datasets", "*", "*", "*.pt"),
    os.path.join(KAGGLE_INPUT, "datasets", "*", "*.pt"),
    os.path.join(KAGGLE_INPUT, "*", "*.pt"),
    os.path.join(WORKING_DIR, "checkpoints_v3.2", "*.pt"),
    os.path.join(WORKING_DIR, "checkpoints_v3.1", "*.pt"),
    os.path.join(WORKING_DIR, "checkpoints_v3.0", "*.pt"),
    "./checkpoints/experiments/v2.6/checkpoints_v2.6/model_tps_v2.6_best.pt",
    "E:/CODE/PBL/PBL6/checkpoints/experiments/v2.6/checkpoints_v2.6/model_tps_v2.6_best.pt"
]

for pat in direct_patterns:
    for f in glob.glob(pat):
        f_norm = os.path.normpath(f)
        if f_norm not in discovered_pts:
            discovered_pts.append(f_norm)

# 2. Quét sâu có tỉa nhánh (Pruned Walk)
def should_skip_dir(d_name):
    d_low = d_name.lower()
    if 'sakuga' in d_low:
        return True
    if d_low in {'frames', 'clips', 'matches', 'crops', 'images', 'npy', 'cache', '.git', '__pycache__'}:
        return True
    return False

search_roots = [KAGGLE_INPUT, WORKING_DIR, './checkpoints', 'E:/CODE/PBL/PBL6/checkpoints', '.']
for s_root in search_roots:
    if not os.path.exists(s_root):
        continue
    for root, dirs, files in os.walk(s_root):
        dirs[:] = [d for d in dirs if not should_skip_dir(d)]
        for f in files:
            if f.endswith('.pt') or f.endswith('.pth'):
                p_norm = os.path.normpath(os.path.join(root, f))
                if p_norm not in discovered_pts:
                    discovered_pts.append(p_norm)

# 3. Chấm điểm ưu tiên tự động cho các checkpoint tìm thấy (ưu tiên V3.1 Best Progress -> V2.6 Gold)
def score_ckpt(path):
    p = path.lower().replace('\\', '/')
    name = os.path.basename(p)
    score = 0
    if 'best_progress' in name and 'v3.1' in p:
        score += 2000
    elif 'v3.1' in p and 'best' in name:
        score += 1800
    elif 'model_tps_v2.6_best.pt' in name or 'model_tps_v2.6_soup.pt' in name:
        score += 1500
    elif 'v2.6' in p and 'best' in name:
        score += 1200
    elif 'v2.6' in p:
        score += 1000
    elif 'v2.5' in p and 'best' in name:
        score += 600
    elif 'v2.5' in p:
        score += 500
    elif 'best' in name:
        score += 400
    elif name.endswith('.pt'):
        score += 200
    return score

discovered_pts.sort(key=score_ckpt, reverse=True)

clean_sd = None
loaded_path = None

for cand in discovered_pts:
    if os.path.exists(cand) and os.path.isfile(cand):
        try:
            print(f"    [*] Đang đọc thử checkpoint: {cand} (Điểm ưu tiên: {score_ckpt(cand)})...")
            raw_sd = torch.load(cand, map_location="cpu", weights_only=False)
            if isinstance(raw_sd, dict) and "model_state_dict" in raw_sd:
                raw_sd = raw_sd["model_state_dict"]
            elif isinstance(raw_sd, dict) and "model" in raw_sd:
                raw_sd = raw_sd["model"]
            clean_sd = {k.replace("module.", "").replace("base_model.", ""): v for k, v in raw_sd.items()}
            loaded_path = cand
            print(f"    [✓] Đọc thành công checkpoint ({len(clean_sd)} tensors) từ: {cand}")
            break
        except Exception as e:
            print(f"    [!] Bỏ qua {cand}: {e}")

model_sd = base_model.state_dict()
matched_sd = {}
if clean_sd is not None:
    is_v3 = any(k in clean_sd for k in ['asf_head.net.0.0.weight', 'asf_head.net.2.weight', 'contour_propagator.conv_in.0.weight'])
    for k, v in clean_sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            # Nếu là V2.6 cũ: Bỏ qua asf_head.* để giữ khởi tạo chuyên dụng V3.2
            # Nếu là V3.1 (best_progress): Kế thừa 100% toàn bộ weights đã hội tụ qua Ep 2!
            if not is_v3 and k.startswith("asf_head."):
                continue
            matched_sd[k] = v
    base_model.load_state_dict(matched_sd, strict=False)
    print(f"[✓] Nạp hoàn tất {len(matched_sd)}/{len(model_sd)} tensors từ Warm-Start ({time.time() - t_ckpt_start:.2f}s).")
    print(f"    - Nguồn checkpoint: {loaded_path}")
    if is_v3:
        print("    - Kế thừa trọn vẹn cả Backbone và các module mới (asf_head, contour_propagator) từ V3.1 Best Progress.")
    else:
        print("    - Nhánh Backbone (Contextnet + IFNet + UNet) kế thừa từ V2.6 Gold.")
        print("    - Nhánh asf_head và contour_propagator được bảo toàn khởi tạo V3.2.")
else:
    print(f"⚠️ [CẢNH BÁO]: Không tìm thấy checkpoint cũ trong {time.time() - t_ckpt_start:.2f}s!")
    print("    - Tạm thời khởi chạy từ trọng số mặc định (Random Weights).")

# Lưu snapshot warm-start sạch để DDP workers nạp độc lập siêu nhanh mà không cần quét lại disk
os.makedirs(CKPT_DIR, exist_ok=True)
init_warmstart_path = os.path.join(CKPT_DIR, '_init_warmstart_v32.pt')
torch.save(base_model.state_dict(), init_warmstart_path)
print(f"[✓] Đã lưu snapshot khởi tạo DDP: {init_warmstart_path}")

class ParallelTrainingWrapper_v32(nn.Module):
    def __init__(self, model, criterion, use_amp=True):
        super(ParallelTrainingWrapper_v32, self).__init__()
        self.model = model
        self.criterion = criterion
        self.use_amp = use_amp
        self.current_epoch = 1

    def forward(self, img0, imggt, img1, t_val, c0, th01, c1, th10, valid, compute_loss=True):
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            res = self.model(img0, img1, t_val, tps_precomputed=(c0, th01, c1, th10, valid))
            pred = res[0]
            v0, v1 = res[1]
            if compute_loss:
                if imggt.dtype == torch.uint8:
                    imggt = ((255.0 - imggt.float()) * (1.0 / 255.0)).contiguous()
                total, l1, dice, dens, hair = self.criterion(pred, imggt, v0, v1, epoch=self.current_epoch)
                return total.unsqueeze(0), l1.unsqueeze(0), dice.unsqueeze(0), dens.unsqueeze(0), hair.unsqueeze(0)
            return pred, v0, v1

ParallelTrainingWrapper_v31 = ParallelTrainingWrapper_v32

print("=" * 80)
print("[✓] CELL 7 hoàn tất: Nạp Warm-Start & Đóng gói Parallel Training Wrapper V3.2 thành công.")
print("=" * 80)
sys.stdout.flush()
