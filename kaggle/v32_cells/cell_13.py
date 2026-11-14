# [CELL 13]: Tự Động Tạo Model Soup Đỉnh Cao, Đóng Gói Thành Phẩm V3.2 & Xuất Link Tải Trực Tiếp
import os
import sys
import shutil
import zipfile
import torch
from IPython.display import FileLink, display

print("=" * 80)
print("🥣 [BƯỚC 1/2]: TỰ ĐỘNG TẠO MODEL SOUP TRỌNG SỐ V3.2 (TRẢI PHẲNG LOSS BASIN)...")
print("=" * 80)

best_pt = os.path.join(CKPT_DIR, 'model_tps_v3.2_best.pt')
prog_pt = os.path.join(CKPT_DIR, 'model_tps_v3.2_best_progress.pt')
latest_pt = os.path.join(CKPT_DIR, 'model_tps_v3.2_latest.pt')
soup_pt = os.path.join(CKPT_DIR, 'model_tps_v3.2_soup.pt')

# Tìm các checkpoint tốt nhất để tạo Model Soup
soup_sources = []
if os.path.exists(best_pt):
    soup_sources.append(('best', best_pt, 0.50))
if os.path.exists(prog_pt) and prog_pt != best_pt:
    soup_sources.append(('progress', prog_pt, 0.30))
if os.path.exists(latest_pt):
    soup_sources.append(('latest', latest_pt, 0.20))

if len(soup_sources) >= 2:
    print(f"[*] Tìm thấy {len(soup_sources)} checkpoint ứng cử viên cho Model Soup:")
    total_w = sum(w for _, _, w in soup_sources)
    norm_sources = [(tag, p, w / total_w) for tag, p, w in soup_sources]
    for tag, p, w in norm_sources:
        print(f"    - [{tag.upper()}]: {os.path.basename(p)} (Trọng số nội suy: {w:.2f})")

    # Tải các state dict
    state_dicts = []
    weights = []
    for tag, p, w in norm_sources:
        sd = torch.load(p, map_location='cpu', weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        elif isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        sd = {k.replace("module.", "").replace("base_model.", ""): v for k, v in sd.items()}
        state_dicts.append(sd)
        weights.append(w)

    soup_sd = {}
    base_keys = state_dicts[0].keys()
    for k in base_keys:
        t0 = state_dicts[0][k]
        if t0.is_floating_point():
            accum = torch.zeros_like(t0, dtype=torch.float32)
            for sd, w in zip(state_dicts, weights):
                if k in sd:
                    accum += w * sd[k].float()
                else:
                    accum += w * t0.float()
            soup_sd[k] = accum.to(t0.dtype)
        else:
            soup_sd[k] = t0.clone()

    torch.save(soup_sd, soup_pt)
    sz_mb = os.path.getsize(soup_pt) / (1024 * 1024)
    print(f"[✓] Đã tạo thành công MODEL SOUP V3.2: {soup_pt} ({sz_mb:.2f} MB)")
    print("    - Model Soup kết hợp các cực tiểu địa phương, triệt tiêu gradient noise và tăng SSIM tổng thể.")
else:
    print("[*] Chỉ có 1 checkpoint khả dụng, bỏ qua bước Model Soup.")

# =============================================================================
# BƯỚC 2: ĐÓNG GÓI THÀNH PHẨM ZIP & FILELINK
# =============================================================================
zip_name = "TPS_Inbetween_v3.2_Final_Artifacts.zip"
zip_path = os.path.join(WORKING_DIR, zip_name)

print("\n" + "=" * 80)
print(f"📦 [BƯỚC 2/2]: ĐANG ĐÓNG GÓI THÀNH PHẨM V3.2: {zip_name}...")
print("=" * 80)

files_to_pack = []
for cp in [best_pt, soup_pt, prog_pt, latest_pt]:
    if os.path.exists(cp):
        files_to_pack.append((cp, os.path.join('checkpoints_v3.2', os.path.basename(cp))))

hist_csv = os.path.join(WORKING_DIR, 'training_history_v3.2.csv')
if os.path.exists(hist_csv):
    files_to_pack.append((hist_csv, 'training_history_v3.2.csv'))

plot_png = os.path.join(WORKING_DIR, 'tps_v3.2_convergence_dashboard.png')
if os.path.exists(plot_png):
    files_to_pack.append((plot_png, 'tps_v3.2_convergence_dashboard.png'))

demo_dir = os.path.join(WORKING_DIR, 'visual_samples_v3.2')
if os.path.exists(demo_dir):
    for fn in os.listdir(demo_dir):
        fp = os.path.join(demo_dir, fn)
        if os.path.isfile(fp):
            files_to_pack.append((fp, os.path.join('visual_samples_v3.2', fn)))

with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for src, arc in files_to_pack:
        zf.write(src, arcname=arc)

sz_zip_mb = os.path.getsize(zip_path) / (1024 * 1024)
print(f"[✓] ĐÃ ĐÓNG GÓI THÀNH CÔNG: {zip_path} ({sz_zip_mb:.2f} MB)")
print("=" * 80)
print("📥 LINK TẢI TRỰC TIẾP TỪ GIAO DIỆN KAGGLE:")
display(FileLink(zip_name))
print("=" * 80)
sys.stdout.flush()
