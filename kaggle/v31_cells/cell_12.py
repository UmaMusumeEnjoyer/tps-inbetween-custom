# [CELL 12]: Kiểm Toán Trực Quan Định Tính & Bằng Chứng Nối Ngọn Tóc V3.1
import os
import sys
import glob
from IPython.display import display, Image

demo_dir = os.path.join(WORKING_DIR, 'visual_samples_v3.1')
sample_pngs = sorted(glob.glob(os.path.join(demo_dir, '*.png')))

if not sample_pngs:
    print(f"[*] Chưa có ảnh mẫu trong {demo_dir}.")
else:
    print(f"[*] Tìm thấy {len(sample_pngs)} ảnh kiểm toán trực quan V3.1:")
    for p in sample_pngs[-6:]:
        fn = os.path.basename(p)
        print(f"\n🖼️ Ảnh mẫu: {fn}")
        display(Image(filename=p, width=900))
