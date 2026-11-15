# [CELL 13]: Tự Động Tạo Model Soup Đỉnh Cao, Kiểm Toán Cảm Nhận & Đóng Gói Thành Phẩm V3.0-Remastered R2
import os
import sys
import glob
import shutil
import zipfile
import torch
import torch.nn.functional as F
import numpy as np
from IPython.display import FileLink, display

WORKING_DIR = globals().get('WORKING_DIR', '/kaggle/working' if os.path.exists('/kaggle') else './workspace')
CKPT_DIR = globals().get('CKPT_DIR', os.path.join(WORKING_DIR, 'checkpoints_v3.0_remastered'))

print("=" * 80)
print("🥣 [BƯỚC 1/3]: TỰ ĐỘNG TẠO MODEL SOUP TRỌNG SỐ V3.0-REMASTERED R2...")
print("=" * 80)

best_pt = os.path.join(CKPT_DIR, 'model_tps_v3.0_remastered_best.pt')
prog_pt = os.path.join(CKPT_DIR, 'model_tps_v3.0_remastered_best_progress.pt')
latest_pt = os.path.join(CKPT_DIR, 'model_tps_v3.0_remastered_latest.pt')
soup_pt = os.path.join(CKPT_DIR, 'model_tps_v3.0_remastered_soup.pt')

soup_sources = []
if os.path.exists(best_pt):
    soup_sources.append(('best', best_pt, 0.50))
if os.path.exists(prog_pt) and prog_pt != best_pt:
    soup_sources.append(('progress', prog_pt, 0.30))
if os.path.exists(latest_pt):
    soup_sources.append(('latest', latest_pt, 0.20))

# Quét thêm các checkpoint ep tốt nhất nếu có
cand_eps = sorted(glob.glob(os.path.join(CKPT_DIR, 'model_tps_v3.0_remastered_ep*.pt')))
if len(soup_sources) < 2 and len(cand_eps) >= 2:
    for i, ep_p in enumerate(cand_eps[-3:]):
        soup_sources.append((f'ep_cand_{i}', ep_p, 1.0 / len(cand_eps[-3:])))

if len(soup_sources) >= 2:
    print(f"[*] Tìm thấy {len(soup_sources)} checkpoint ứng cử viên cho Model Soup:")
    total_w = sum(w for _, _, w in soup_sources)
    norm_sources = [(tag, p, w / total_w) for tag, p, w in soup_sources]
    for tag, p, w in norm_sources:
        print(f"    - [{tag.upper()}]: {os.path.basename(p)} (Trọng số nội suy: {w:.2f})")

    state_dicts = []
    weights = []
    for tag, p, w in norm_sources:
        sd = torch.load(p, map_location='cpu', weights_only=False)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        elif isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
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

    torch.save({'model': soup_sd, 'metrics': {'soup': True}}, soup_pt)
    sz_mb = os.path.getsize(soup_pt) / (1024 * 1024)
    print(f"[✓] Đã tạo thành công MODEL SOUP V3.0-R2: {soup_pt} ({sz_mb:.2f} MB)")
else:
    print("[*] Chỉ có 1 checkpoint khả dụng, bỏ qua bước Model Soup.")

print("\n" + "=" * 80)
print("👁️ [BƯỚC 2/3]: KIỂM TOÁN CHỈ SỐ CẢM NHẬN & HÌNH THÁI HỌC CUỐI CÙNG...")
print("=" * 80)

hist_p = os.path.join(WORKING_DIR, 'training_history_v3.0_remastered.csv')
if os.path.exists(hist_p):
    df_h = pd.read_csv(hist_p)
    last_row = df_h.iloc[-1]
    best_row = df_h.loc[df_h['wcd'].idxmin()] if 'wcd' in df_h.columns else last_row

    print(f"[*] Thành tích Tốt Nhất Toàn Khóa Huấn Luyện (Epoch {int(best_row['epoch'])}):")
    print(f"    - Weighted Chamfer Distance (WCD): {best_row.get('wcd', 0.0):.6f} (Mục tiêu: <= 0.00120)")
    print(f"    - 1D-Sliced EMD:                   {best_row.get('sliced_emd', 0.0):.6f}")
    print(f"    - Skeleton Recall:                 {best_row.get('skel_recall', 0.0)*100:.2f}% (Mục tiêu: >= 88.0%)")
    print(f"    - Hair-Tip Recall:                 {best_row.get('tip_recall', 0.0)*100:.2f}% (Mục tiêu: >= 85.0%)")
    print(f"    - Stroke Recall:                   {best_row.get('stroke_recall', 0.0)*100:.2f}% (Mục tiêu: >= 92.0%)")
    print(f"    - Stroke Density Ratio:            {best_row.get('density_ratio', 0.0):.3f}x (Tâm chuẩn: 1.10x)")
    print(f"    - Nền Sạch Tuyệt Đối:              {best_row.get('bg_cleanliness', 0.0)*100:.2f}% (Mục tiêu: >= 99.5%)")

print("\n" + "=" * 80)
print("📦 [BƯỚC 3/3]: ĐÓNG GÓI THÀNH PHẨM V3.0-REMASTERED R2 THÀNH FILE .ZIP...")
print("=" * 80)

zip_output_path = os.path.join(WORKING_DIR, 'tps_inbetween_v3.0_remastered_release.zip')
with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(CKPT_DIR):
        for f in files:
            fp = os.path.join(root, f)
            arcname = os.path.relpath(fp, WORKING_DIR)
            zipf.write(fp, arcname)

    demo_dir = os.path.join(WORKING_DIR, 'visual_samples_v3.0_remastered')
    if os.path.exists(demo_dir):
        for root, _, files in os.walk(demo_dir):
            for f in files:
                fp = os.path.join(root, f)
                arcname = os.path.relpath(fp, WORKING_DIR)
                zipf.write(fp, arcname)

    if os.path.exists(hist_p):
        zipf.write(hist_p, os.path.basename(hist_p))

    dash_p = os.path.join(WORKING_DIR, 'tps_v3.0_remastered_convergence_dashboard.png')
    if os.path.exists(dash_p):
        zipf.write(dash_p, os.path.basename(dash_p))

zip_sz_mb = os.path.getsize(zip_output_path) / (1024 * 1024)
print(f"[✓] Đóng gói thành công: {zip_output_path} ({zip_sz_mb:.2f} MB)!")
print("[*] Bạn có thể tải trực tiếp file zip này qua link bên dưới:")
display(FileLink(os.path.relpath(zip_output_path, os.getcwd()) if not os.path.isabs(zip_output_path) else zip_output_path))
sys.stdout.flush()
