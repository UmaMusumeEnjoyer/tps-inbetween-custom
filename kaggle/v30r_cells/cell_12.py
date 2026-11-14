# [CELL 12]: Kiểm Toán Trực Quan Định Tính & Bằng Chứng Nền Sạch RGBA V3.0-Remastered
import os
import sys
import glob
from IPython.display import display, Image

WORKING_DIR = globals().get('WORKING_DIR', '/kaggle/working' if os.path.exists('/kaggle') else './workspace')
demo_dir = os.path.join(WORKING_DIR, 'visual_samples_v3.0_remastered')
sample_pngs = sorted(glob.glob(os.path.join(demo_dir, '*.png')))

if not sample_pngs:
    print(f"[*] Chưa có ảnh mẫu trong {demo_dir}.")
else:
    print(f"[*] Tìm thấy {len(sample_pngs)} ảnh kiểm toán trực quan V3.0-Remastered:")
    for p in sample_pngs[-8:]:
        fn = os.path.basename(p)
        print(f"\n🖼️ Ảnh mẫu: {fn}")
        display(Image(filename=p, width=900))
sys.stdout.flush()
