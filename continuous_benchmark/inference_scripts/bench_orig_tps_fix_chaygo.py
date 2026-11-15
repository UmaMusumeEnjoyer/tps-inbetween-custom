import os
import sys
import shutil
import argparse
import subprocess
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Hỗ trợ Unicode tiếng Việt trên Windows terminal
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
        sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass

# Đường dẫn gốc
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TEST_DIR = os.path.join(BASE_DIR, 'Test')
MODELS_DIR = os.path.join(TEST_DIR, 'TestModels')
FINAL_TEST_DIR = os.path.join(BASE_DIR, 'Final test')
PYTHON_EXE = os.path.join(TEST_DIR, '.venv', 'Scripts', 'python.exe')
DEFAULT_DATASET = os.path.join(BASE_DIR, 'TestDataset', 'ChayGoAnimeLinesDetail_NoBG')


def filter_inbetween_files(folder_path):
    all_files = os.listdir(folder_path)
    inbetween = []
    for f in all_files:
        if not f.lower().endswith('.png'):
            continue
        if f in ['1.000.png', '2.000.png', '0.000.png']:
            continue
        inbetween.append(f)
    return sorted(inbetween)


def run_tps_fix_pipeline(args):
    print("=======================================================")
    print("🚀 BẮT ĐẦU PIPELINE KIỂM THỬ TPS-INBETWEEN TỔNG HỢP CÁC FIX (`tps-fix`)")
    print("   [1] Điều chuẩn TPS: lambd = " + str(args.lambd) + " (Chống rung méo/uốn lượn)")
    print("   [2] Matcher: RANSAC Outlier Filter = " + str(args.use_ransac) + f" (thresh={args.ransac_thresh}px)")
    print("   [A] Điểm neo biên: Border Anchors = " + str(args.use_border_anchors))
    print("   [B] Điểm neo nội miền: Interior Anchors = " + str(args.use_interior_anchors) + f" (grid {args.grid_nx}x{args.grid_ny})")
    print("   [3] Dữ liệu: " + str(args.dataset_dir) + (" (Dùng keyframes mượt)" if args.use_smoothed else " (Keyframes gốc)"))
    print("   [4] Chuỗi mốc thời gian: " + str(args.times))
    print("   [5] Hậu xử lý: Khử bóng ma ghosting = " + str(args.clean_ghosting) + f" (thresh={args.ghost_thresh})")
    print("=======================================================\n")

    dataset_path = args.dataset_dir
    target_subdir = args.output_name
    times_list = [str(t) for t in args.times]

    # Kiểm tra keyframes
    k0_file = '0.0s_smooth.png' if args.use_smoothed else '0.0s.png'
    k1_file = '0.5s_smooth.png' if args.use_smoothed else '0.5s.png'
    k2_file = '1.0s_smooth.png' if args.use_smoothed else '1.0s.png'

    k0 = os.path.join(dataset_path, k0_file)
    k1 = os.path.join(dataset_path, k1_file)
    k2 = os.path.join(dataset_path, k2_file)

    for k in [k0, k1, k2]:
        if not os.path.exists(k):
            print(f"❌ Lỗi: Không tìm thấy keyframe '{k}'!")
            return False

    # Thư mục output: Final test/<target_subdir>
    output_dir = os.path.join(FINAL_TEST_DIR, target_subdir)
    os.makedirs(output_dir, exist_ok=True)

    tmp_seg1 = os.path.join(output_dir, '_tmp_seg1')
    tmp_seg2 = os.path.join(output_dir, '_tmp_seg2')
    os.makedirs(tmp_seg1, exist_ok=True)
    os.makedirs(tmp_seg2, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    # BƯỚC 1: Sao chép 3 Keyframes gốc vào thư mục kết quả
    print(f"[1/4] Sao chép 3 Keyframes gốc vào '{output_dir}'...")
    shutil.copy2(k0, os.path.join(output_dir, '0.0s.png'))
    shutil.copy2(k1, os.path.join(output_dir, '0.5s.png'))
    shutil.copy2(k2, os.path.join(output_dir, '1.0s.png'))
    print("      ✓ Đã lưu: 0.0s.png, 0.5s.png, 1.0s.png")

    model_cwd = os.path.join(MODELS_DIR, 'tps-inbetween')

    # BƯỚC 2: Chạy Đoạn 1 (0.0s -> 0.5s)
    print(f"\n[2/4] Chạy Đoạn 1: 0.0s.png -> 0.5s.png với t in {times_list}...")
    cmd1 = [
        PYTHON_EXE,
        '-m', 'demo.interp_custom_times_fix',
        '--frame0', k0,
        '--frame1', k1,
        '--times', *times_list,
        '--save_dir', tmp_seg1,
        '--prefix_1', '1',
        '--prefix_2', '2',
        '--digits', '3',
        '--lambd', str(args.lambd),
        '--ransac_thresh', str(args.ransac_thresh),
        '--ghost_thresh', str(args.ghost_thresh)
    ]
    if args.use_ransac:
        cmd1.append('--use_ransac')
    else:
        cmd1.append('--no_ransac')
    if args.clean_ghosting:
        cmd1.append('--clean_ghosting')
    else:
        cmd1.append('--no_clean_ghosting')
    if args.use_border_anchors:
        cmd1.append('--use_border_anchors')
    else:
        cmd1.append('--no_border_anchors')
    if args.use_interior_anchors:
        cmd1.append('--use_interior_anchors')
    else:
        cmd1.append('--no_interior_anchors')
    cmd1.extend(['--grid_nx', str(args.grid_nx), '--grid_ny', str(args.grid_ny)])

    print(f"      Lệnh thực thi: {' '.join(cmd1)}")
    res1 = subprocess.run(cmd1, cwd=model_cwd, env=env)
    if res1.returncode != 0:
        print(f"❌ Lỗi khi thực thi Đoạn 1 (Code: {res1.returncode})")
        return False

    seg1_files = filter_inbetween_files(tmp_seg1)
    seg1_mapping = [('0.1s.png', 0), ('0.2s.png', 1), ('0.3s.png', 2)]
    for target_name, idx in seg1_mapping:
        if idx < len(seg1_files):
            src_file = os.path.join(tmp_seg1, seg1_files[idx])
            dst_file = os.path.join(output_dir, target_name)
            shutil.copy2(src_file, dst_file)
            print(f"      ✓ Đã lưu: {target_name} (từ {seg1_files[idx]})")

    # BƯỚC 3: Chạy Đoạn 2 (0.5s -> 1.0s)
    print(f"\n[3/4] Chạy Đoạn 2: 0.5s.png -> 1.0s.png với t in {times_list}...")
    cmd2 = [
        PYTHON_EXE,
        '-m', 'demo.interp_custom_times_fix',
        '--frame0', k1,
        '--frame1', k2,
        '--times', *times_list,
        '--save_dir', tmp_seg2,
        '--prefix_1', '1',
        '--prefix_2', '2',
        '--digits', '3',
        '--lambd', str(args.lambd),
        '--ransac_thresh', str(args.ransac_thresh),
        '--ghost_thresh', str(args.ghost_thresh)
    ]
    if args.use_ransac:
        cmd2.append('--use_ransac')
    else:
        cmd2.append('--no_ransac')
    if args.clean_ghosting:
        cmd2.append('--clean_ghosting')
    else:
        cmd2.append('--no_clean_ghosting')
    if args.use_border_anchors:
        cmd2.append('--use_border_anchors')
    else:
        cmd2.append('--no_border_anchors')
    if args.use_interior_anchors:
        cmd2.append('--use_interior_anchors')
    else:
        cmd2.append('--no_interior_anchors')
    cmd2.extend(['--grid_nx', str(args.grid_nx), '--grid_ny', str(args.grid_ny)])

    print(f"      Lệnh thực thi: {' '.join(cmd2)}")
    res2 = subprocess.run(cmd2, cwd=model_cwd, env=env)
    if res2.returncode != 0:
        print(f"❌ Lỗi khi thực thi Đoạn 2 (Code: {res2.returncode})")
        return False

    seg2_files = filter_inbetween_files(tmp_seg2)
    seg2_mapping = [('0.6s.png', 0), ('0.7s.png', 1), ('0.8s.png', 2)]
    for target_name, idx in seg2_mapping:
        if idx < len(seg2_files):
            src_file = os.path.join(tmp_seg2, seg2_files[idx])
            dst_file = os.path.join(output_dir, target_name)
            shutil.copy2(src_file, dst_file)
            print(f"      ✓ Đã lưu: {target_name} (từ {seg2_files[idx]})")

    # BƯỚC 4: Dọn dẹp thư mục tạm và xác minh kết quả
    print(f"\n[4/4] Dọn dẹp thư mục tạm và xác minh toàn bộ 9 frames...")
    shutil.rmtree(tmp_seg1, ignore_errors=True)
    shutil.rmtree(tmp_seg2, ignore_errors=True)

    expected_frames = [
        '0.0s.png', '0.1s.png', '0.2s.png', '0.3s.png',
        '0.5s.png', '0.6s.png', '0.7s.png', '0.8s.png', '1.0s.png'
    ]

    all_ok = True
    print("\nDanh sách file kết quả trong:", output_dir)
    for ef in expected_frames:
        fpath = os.path.join(output_dir, ef)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            print(f"  [OK] {ef:10s} - {size:>8} bytes")
        else:
            print(f"  [MISSING] {ef:10s}")
            all_ok = False

    if not all_ok:
        print("\n❌ Thất bại: Một số frames bị thiếu.")
        return False

    print(f"\n🎉 THÀNH CÔNG: Model TPS-Inbetween (Fix Pipeline) đã hoàn thành đầy đủ 9 frames trong '{output_dir}'!")

    # BƯỚC 5: ĐO ĐẠC ĐỐI CHIẾU 3 BÊN (Benchmark Comparison)
    print("\n" + "=" * 90)
    print("📊 BẢNG SO SÁNH ĐỊNH LƯỢNG 3 BÊN: tps-nogay vs tps-nogay-newstep vs tps-fix")
    print("=" * 90)

    gt_nogay_dir = os.path.join(BASE_DIR, 'TestDataset', 'ChayGoAnimeLinesDetail_NoBG_NoGay')
    dir_nogay = os.path.join(FINAL_TEST_DIR, 'tps-nogay')
    dir_newstep = os.path.join(FINAL_TEST_DIR, 'tps-nogay-newstep')
    dir_fix = output_dir

    eval_frames = ['0.1s.png', '0.2s.png', '0.3s.png', '0.6s.png', '0.7s.png', '0.8s.png']
    print(f"{'Khung hình':<10} | {'tps-nogay':<14} | {'newstep':<14} | {'tps-fix (Mới)':<16} | {'Cải thiện':<12}")
    print("-" * 75)

    psnr_nogay, psnr_newstep, psnr_fix = [], [], []
    ssim_nogay, ssim_newstep, ssim_fix = [], [], []

    for ef in eval_frames:
        gt = cv2.imread(os.path.join(gt_nogay_dir, ef), cv2.IMREAD_GRAYSCALE)
        img_nogay = cv2.imread(os.path.join(dir_nogay, ef), cv2.IMREAD_GRAYSCALE) if os.path.exists(os.path.join(dir_nogay, ef)) else None
        img_newstep = cv2.imread(os.path.join(dir_newstep, ef), cv2.IMREAD_GRAYSCALE) if os.path.exists(os.path.join(dir_newstep, ef)) else None
        img_fix = cv2.imread(os.path.join(dir_fix, ef), cv2.IMREAD_GRAYSCALE)

        p1 = peak_signal_noise_ratio(gt, img_nogay) if img_nogay is not None else 0
        p2 = peak_signal_noise_ratio(gt, img_newstep) if img_newstep is not None else 0
        p3 = peak_signal_noise_ratio(gt, img_fix)
        s1 = structural_similarity(gt, img_nogay) if img_nogay is not None else 0
        s2 = structural_similarity(gt, img_newstep) if img_newstep is not None else 0
        s3 = structural_similarity(gt, img_fix)

        psnr_nogay.append(p1)
        psnr_newstep.append(p2)
        psnr_fix.append(p3)
        ssim_nogay.append(s1)
        ssim_newstep.append(s2)
        ssim_fix.append(s3)

        diff = p3 - max(p1, p2)
        diff_str = f"+{diff:.2f} dB" if diff >= 0 else f"{diff:.2f} dB"
        print(f"{ef:<10} | {p1:>6.2f}dB / {s1:.3f} | {p2:>6.2f}dB / {s2:.3f} | {p3:>6.2f}dB / {s3:.3f}  | {diff_str:>10}")

    print("-" * 75)
    avg_p1, avg_p2, avg_p3 = np.mean(psnr_nogay), np.mean(psnr_newstep), np.mean(psnr_fix)
    avg_s1, avg_s2, avg_s3 = np.mean(ssim_nogay), np.mean(ssim_newstep), np.mean(ssim_fix)
    avg_diff = avg_p3 - max(avg_p1, avg_p2)
    avg_diff_str = f"+{avg_diff:.2f} dB" if avg_diff >= 0 else f"{avg_diff:.2f} dB"
    print(f"{'TRUNG BÌNH':<10} | {avg_p1:>6.2f}dB / {avg_s1:.3f} | {avg_p2:>6.2f}dB / {avg_s2:.3f} | {avg_p3:>6.2f}dB / {avg_s3:.3f}  | {avg_diff_str:>10}")
    print("=" * 75 + "\n")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chạy kiểm thử TPS-Inbetween với toàn bộ 5 giải pháp can thiệp ngay")
    parser.add_argument('--dataset_dir', type=str, default=DEFAULT_DATASET,
                        help="Thư mục dataset (mặc định: TestDataset/ChayGoAnimeLinesDetail_NoBG)")
    parser.add_argument('--use_smoothed', action='store_true', default=True,
                        help="Dùng keyframes mượt nét (0.0s_smooth.png, 0.5s_smooth.png, 1.0s_smooth.png)")
    parser.add_argument('--output_name', type=str, default='tps-fix',
                        help="Thư mục kết quả trong Final test (mặc định: tps-fix)")
    parser.add_argument('--times', type=str, nargs='*', default=['0.1', '0.2', '0.3'],
                        help="Chuỗi mốc thời gian t (mặc định: 0.1 0.2 0.3)")
    parser.add_argument('--lambd', type=float, default=0.2,
                        help="Hệ số điều chuẩn TPS lambda (mặc định: 0.2)")
    parser.add_argument('--use_ransac', action='store_true', default=True,
                        help="Lọc ngoại lai RANSAC")
    parser.add_argument('--ransac_thresh', type=float, default=5.0,
                        help="Ngưỡng pixel RANSAC (mặc định: 5.0 px)")
    parser.add_argument('--clean_ghosting', action='store_true', default=True,
                        help="Hậu xử lý khử bóng ma alpha blend")
    parser.add_argument('--ghost_thresh', type=int, default=190,
                        help="Ngưỡng lọc bóng mờ (mặc định: 190)")
    parser.add_argument('--use_border_anchors', action='store_true', default=True,
                        help="Điểm neo viền biên (Giải pháp A)")
    parser.add_argument('--no_border_anchors', dest='use_border_anchors', action='store_false',
                        help="Tắt điểm neo viền biên")
    parser.add_argument('--use_interior_anchors', action='store_true', default=True,
                        help="Điểm neo nội miền vùng trống (Giải pháp B)")
    parser.add_argument('--no_interior_anchors', dest='use_interior_anchors', action='store_false',
                        help="Tắt điểm neo nội miền")
    parser.add_argument('--grid_nx', type=int, default=6, help="Số cột lưới nội miền")
    parser.add_argument('--grid_ny', type=int, default=4, help="Số hàng lưới nội miền")

    args = parser.parse_args()
    success = run_tps_fix_pipeline(args)
    sys.exit(0 if success else 1)
