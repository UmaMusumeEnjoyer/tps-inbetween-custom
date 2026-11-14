# [CELL 13]: Đóng Gói Thành Phẩm Sản Phẩm V3.0 & Xuất Link Tải Trực Tiếp
import os
import sys
import zipfile
from IPython.display import FileLink
import os
import shutil
import zipfile
from IPython.display import FileLink, display

zip_name = "TPS_Inbetween_v3.0_Final_Artifacts.zip"
zip_path = os.path.join(WORKING_DIR, zip_name)

print("=" * 80)
print(f"📦 ĐANG ĐÓNG GÓI THÀNH PHẨM V3.0: {zip_name}...")
print("=" * 80)

files_to_pack = []
best_pt = os.path.join(CKPT_DIR, 'model_tps_v3.0_best.pt')
prog_pt = os.path.join(CKPT_DIR, 'model_tps_v3.0_best_progress.pt')
latest_pt = os.path.join(CKPT_DIR, 'model_tps_v3.0_latest.pt')

for cp in [best_pt, prog_pt, latest_pt]:
    if os.path.exists(cp):
        files_to_pack.append((cp, os.path.join('checkpoints_v3.0', os.path.basename(cp))))

hist_csv = os.path.join(WORKING_DIR, 'training_history_v3.0.csv')
if os.path.exists(hist_csv):
    files_to_pack.append((hist_csv, 'training_history_v3.0.csv'))

plot_png = os.path.join(WORKING_DIR, 'training_metrics_convergence_v30.png')
if os.path.exists(plot_png):
    files_to_pack.append((plot_png, 'training_metrics_convergence_v30.png'))

demo_dir = os.path.join(WORKING_DIR, 'visual_samples_v3.0')
if os.path.exists(demo_dir):
    for fn in os.listdir(demo_dir):
        fp = os.path.join(demo_dir, fn)
        if os.path.isfile(fp):
            files_to_pack.append((fp, os.path.join('visual_samples_v3.0', fn)))

with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for src, arc in files_to_pack:
        zf.write(src, arcname=arc)
        sz_mb = os.path.getsize(src) / (1024**2)
        print(f"   + Đã thêm: {arc} ({sz_mb:.2f} MB)")

zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
print("=" * 80)
print(f"🎉 ĐÓNG GÓI HOÀN TẤT THÀNH CÔNG: {zip_path} ({zip_size_mb:.2f} MB)")
print("=" * 80)
print("\n👉 NHẤN VÀO ĐƯỜNG LINK DƯỚI ĐÂY ĐỂ TẢI TRỰC TIẾP TOÀN BỘ THÀNH PHẨM V3.0:")
display(FileLink(zip_name))