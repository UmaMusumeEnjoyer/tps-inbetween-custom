import os
import sys
import shutil
import argparse
import subprocess

# Đảm bảo hiển thị Unicode tiếng Việt trên Windows terminal
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
        sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass

# Đường dẫn gốc
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
TEST_DIR = os.path.join(BASE_DIR, 'Test')
MODELS_DIR = os.path.join(BASE_DIR, 'third_party', 'baselines')
DATASET_DIR = os.path.join(BASE_DIR, 'TestDataset', 'ChayGoAnimeLinesDetail_NoBG')
FINAL_TEST_DIR = os.path.join(BASE_DIR, 'Final test')
PYTHON_EXE = os.path.join(TEST_DIR, '.venv', 'Scripts', 'python.exe')

SUPPORTED_MODELS = {
    'animeinbet': 'AnimeInbet',
    'animeinterp': 'AnimeInterp',
    'sain': 'SAIN',
    'tps-inbetween': 'tps-inbetween',
    'univip': 'uniVIP',
    'upr-net': 'UPR-Net'
}


def get_model_command(model_name, frame0, frame1, times_str_list, save_dir):
    """
    Trả về (cwd, command_list) để chạy script demo tương ứng của từng model.
    """
    model_cwd = os.path.join(MODELS_DIR, model_name)
    
    if model_name == 'AnimeInbet':
        cmd = [
            PYTHON_EXE,
            os.path.join('demo', 'interp_custom_times.py'),
            '--frame0', frame0,
            '--frame1', frame1,
            '--times', *times_str_list,
            '--save_dir', save_dir,
            '--prefix_1', '1',
            '--prefix_2', '2',
            '--digits', '3'
        ]
    elif model_name == 'AnimeInterp':
        cmd = [
            PYTHON_EXE,
            '-m', 'demo.interp_custom_times',
            '--frame0', frame0,
            '--frame1', frame1,
            '--times', *times_str_list,
            '--save_dir', save_dir,
            '--prefix_1', '1',
            '--prefix_2', '2',
            '--digits', '3'
        ]
    elif model_name == 'SAIN':
        cmd = [
            PYTHON_EXE,
            '-m', 'demo.interp_custom_times',
            '--frame0', frame0,
            '--frame1', frame1,
            '--times', *times_str_list,
            '--save_dir', save_dir,
            '--prefix_1', '1',
            '--prefix_2', '2',
            '--digits', '3'
        ]
    elif model_name == 'tps-inbetween':
        cmd = [
            PYTHON_EXE,
            '-m', 'demo.interp_custom_times',
            '--frame0', frame0,
            '--frame1', frame1,
            '--times', *times_str_list,
            '--save_dir', save_dir,
            '--prefix_1', '1',
            '--prefix_2', '2',
            '--digits', '3'
        ]
    elif model_name == 'uniVIP':
        ckpt = os.path.join('checkpoints', 'uniVIP-L.pkl')
        if not os.path.exists(os.path.join(model_cwd, ckpt)):
            ckpt = os.path.join('checkpoints', 'uniVIP-B.pkl')
            model_size = 'base'
        else:
            model_size = 'large'
        cmd = [
            PYTHON_EXE,
            '-m', 'demo.interp_custom_times',
            '--frame0', frame0,
            '--frame1', frame1,
            '--times', *times_str_list,
            '--save_dir', save_dir,
            '--model_size', model_size,
            '--model_file', ckpt,
            '--prefix_1', '1',
            '--prefix_2', '2',
            '--digits', '3'
        ]
    elif model_name == 'UPR-Net':
        ckpt = os.path.join('checkpoints', 'upr-llarge.pkl')
        if not os.path.exists(os.path.join(model_cwd, ckpt)):
            ckpt = os.path.join('checkpoints', 'upr-base.pkl')
            model_size = 'base'
        else:
            model_size = 'LARGE'
        cmd = [
            PYTHON_EXE,
            '-m', 'demo.interp_custom_times',
            '--frame0', frame0,
            '--frame1', frame1,
            '--times', *times_str_list,
            '--save_dir', save_dir,
            '--model_size', model_size,
            '--model_file', ckpt,
            '--prefix_1', '1',
            '--prefix_2', '2',
            '--digits', '3'
        ]
    else:
        raise ValueError(f"Model {model_name} không được hỗ trợ.")

    return model_cwd, cmd


def filter_inbetween_files(folder_path):
    """
    Lấy danh sách các file inbetween đã sinh (bỏ qua frame gốc 1.000.png, 2.000.png nếu có).
    Sắp xếp theo thứ tự thời gian t tăng dần.
    """
    all_files = os.listdir(folder_path)
    inbetween = []
    for f in all_files:
        if not f.lower().endswith('.png'):
            continue
        # Bỏ qua frame copy gốc
        if f in ['1.000.png', '2.000.png', '0.000.png']:
            continue
        inbetween.append(f)
    return sorted(inbetween)


def run_model_test(model_canonical_name, use_smoothed=False, dataset_dir=None, output_name=None, times=None):
    suffix_desc = " (VỚI KEYFRAMES ĐÃ LÀM MƯỢT - SMOOTHED)" if use_smoothed else ""
    print(f"\n=======================================================")
    print(f"🚀 BẮT ĐẦU CHẠY KIỂM THỬ MODEL: {model_canonical_name}{suffix_desc}")
    print(f"=======================================================")

    dataset_path = dataset_dir if dataset_dir else DATASET_DIR
    target_subdir = output_name if output_name else model_canonical_name
    times_list = [str(t) for t in times] if times and len(times) > 0 else ['0.2', '0.4', '0.6']

    # Kiểm tra keyframes
    k0_file = '0.0s_smooth.png' if use_smoothed else '0.0s.png'
    k1_file = '0.5s_smooth.png' if use_smoothed else '0.5s.png'
    k2_file = '1.0s_smooth.png' if use_smoothed else '1.0s.png'

    k0 = os.path.join(dataset_path, k0_file)
    k1 = os.path.join(dataset_path, k1_file)
    k2 = os.path.join(dataset_path, k2_file)

    for k in [k0, k1, k2]:
        if not os.path.exists(k):
            print(f"❌ Lỗi: Không tìm thấy keyframe '{k}'!")
            return False

    # Tạo thư mục output: Final test/<target_subdir>
    model_output_dir = os.path.join(FINAL_TEST_DIR, target_subdir)
    os.makedirs(model_output_dir, exist_ok=True)

    # Thư mục tạm thời cho 2 đoạn
    tmp_seg1 = os.path.join(model_output_dir, '_tmp_seg1')
    tmp_seg2 = os.path.join(model_output_dir, '_tmp_seg2')
    os.makedirs(tmp_seg1, exist_ok=True)
    os.makedirs(tmp_seg2, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    # -------------------------------------------------------------
    # BƯỚC 1: Sao chép 3 Keyframes gốc vào thư mục kết quả
    # -------------------------------------------------------------
    print(f"\n[1/4] Sao chép 3 Keyframes gốc vào '{model_output_dir}'...")
    shutil.copy2(k0, os.path.join(model_output_dir, '0.0s.png'))
    shutil.copy2(k1, os.path.join(model_output_dir, '0.5s.png'))
    shutil.copy2(k2, os.path.join(model_output_dir, '1.0s.png'))
    print("      ✓ Đã lưu: 0.0s.png, 0.5s.png, 1.0s.png")

    # -------------------------------------------------------------
    # BƯỚC 2: Chạy Đoạn 1 (0.0s.png -> 0.5s.png) với times_list
    # -------------------------------------------------------------
    print(f"\n[2/4] Chạy Đoạn 1: 0.0s.png -> 0.5s.png với t in {times_list}...")
    cwd_1, cmd_1 = get_model_command(model_canonical_name, k0, k1, times_list, tmp_seg1)
    print(f"      Lệnh thực thi: {' '.join(cmd_1)}")
    res1 = subprocess.run(cmd_1, cwd=cwd_1, env=env)
    if res1.returncode != 0:
        print(f"❌ Lỗi khi thực thi Đoạn 1 (Return code: {res1.returncode})")
        return False

    seg1_files = filter_inbetween_files(tmp_seg1)
    if len(seg1_files) != 3:
        print(f"⚠️ Cảnh báo: Đoạn 1 sinh ra {len(seg1_files)} files (kỳ vọng 3): {seg1_files}")
    
    seg1_mapping = [('0.1s.png', 0), ('0.2s.png', 1), ('0.3s.png', 2)]
    for target_name, idx in seg1_mapping:
        if idx < len(seg1_files):
            src_file = os.path.join(tmp_seg1, seg1_files[idx])
            dst_file = os.path.join(model_output_dir, target_name)
            shutil.copy2(src_file, dst_file)
            print(f"      ✓ Đã lưu: {target_name} (từ {seg1_files[idx]})")

    # -------------------------------------------------------------
    # BƯỚC 3: Chạy Đoạn 2 (0.5s.png -> 1.0s.png) với times_list
    # -------------------------------------------------------------
    print(f"\n[3/4] Chạy Đoạn 2: 0.5s.png -> 1.0s.png với t in {times_list}...")
    cwd_2, cmd_2 = get_model_command(model_canonical_name, k1, k2, times_list, tmp_seg2)
    print(f"      Lệnh thực thi: {' '.join(cmd_2)}")
    res2 = subprocess.run(cmd_2, cwd=cwd_2, env=env)
    if res2.returncode != 0:
        print(f"❌ Lỗi khi thực thi Đoạn 2 (Return code: {res2.returncode})")
        return False

    seg2_files = filter_inbetween_files(tmp_seg2)
    if len(seg2_files) != 3:
        print(f"⚠️ Cảnh báo: Đoạn 2 sinh ra {len(seg2_files)} files (kỳ vọng 3): {seg2_files}")

    seg2_mapping = [('0.6s.png', 0), ('0.7s.png', 1), ('0.8s.png', 2)]
    for target_name, idx in seg2_mapping:
        if idx < len(seg2_files):
            src_file = os.path.join(tmp_seg2, seg2_files[idx])
            dst_file = os.path.join(model_output_dir, target_name)
            shutil.copy2(src_file, dst_file)
            print(f"      ✓ Đã lưu: {target_name} (từ {seg2_files[idx]})")

    # -------------------------------------------------------------
    # BƯỚC 4: Dọn dẹp thư mục tạm và xác minh kết quả
    # -------------------------------------------------------------
    print(f"\n[4/4] Dọn dẹp thư mục tạm và xác minh toàn bộ 9 frames...")
    shutil.rmtree(tmp_seg1, ignore_errors=True)
    shutil.rmtree(tmp_seg2, ignore_errors=True)

    expected_frames = [
        '0.0s.png', '0.1s.png', '0.2s.png', '0.3s.png',
        '0.5s.png', '0.6s.png', '0.7s.png', '0.8s.png', '1.0s.png'
    ]

    all_ok = True
    print("\nDanh sách file kết quả trong:", model_output_dir)
    for ef in expected_frames:
        fpath = os.path.join(model_output_dir, ef)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            print(f"  [OK] {ef:10s} - {size:>8} bytes")
        else:
            print(f"  [MISSING] {ef:10s}")
            all_ok = False

    if all_ok:
        print(f"\n🎉 THÀNH CÔNG: Model '{model_canonical_name}' đã hoàn thành kiểm thử đầy đủ 9 frames!")
        return True
    else:
        print(f"\n❌ Thất bại: Một số frames bị thiếu trong '{model_canonical_name}'.")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chạy kiểm thử từng mô hình Video Frame Interpolation cho PBL6")
    parser.add_argument('--model', type=str, required=True,
                        help="Tên mô hình cần kiểm thử: AnimeInbet, AnimeInterp, tps-inbetween, uniVIP, UPR-Net")
    parser.add_argument('--use_smoothed', action='store_true',
                        help="Sử dụng các keyframe đã làm mượt (0.0s_smooth.png, 0.5s_smooth.png, 1.0s_smooth.png)")
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help="Đường dẫn đến thư mục dataset (mặc định: TestDataset/ChayGoAnimeLinesDetail_NoBG)")
    parser.add_argument('--output_name', type=str, default=None,
                        help="Tên thư mục con lưu kết quả trong Final test (mặc định: tên model)")
    parser.add_argument('--times', type=str, nargs='*', default=None,
                        help="Danh sách các mốc thời gian t nội suy (ví dụ: --times 0.1 0.2 0.3, mặc định: 0.2 0.4 0.6)")
    args = parser.parse_args()

    key = args.model.strip().lower()
    if key not in SUPPORTED_MODELS:
        print(f"Lỗi: Model '{args.model}' không nằm trong danh sách hỗ trợ.")
        print(f"Các model hợp lệ: {', '.join(SUPPORTED_MODELS.values())}")
        sys.exit(1)

    canonical_name = SUPPORTED_MODELS[key]
    success = run_model_test(
        canonical_name,
        use_smoothed=args.use_smoothed,
        dataset_dir=args.dataset_dir,
        output_name=args.output_name,
        times=args.times
    )
    sys.exit(0 if success else 1)
