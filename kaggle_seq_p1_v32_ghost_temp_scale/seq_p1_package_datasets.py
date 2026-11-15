"""
Tự động đóng gói 2 file ZIP chuẩn cấu trúc để tải lên Kaggle:
1. `sakuga_final_dataset_v2.zip` (~500 MB): Gộp chung cả 2 thư mục:
   - `sakuga_golden_dataset_sequences_tps_v2` (Train / Val Sequences)
   - `sakuga_golden_benchmark_sequences_v2` (Benchmark #1 Isolated Sequences)
2. `anime_lineart_continuous_test.zip` (~2.4 MB):
   - Thư mục `data/lineart_experiments/test` (Benchmark #2 Continuous Keyframes)
"""

import os
import sys
import time
import zipfile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "kaggle_upload_ready")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def zip_directory_contents(zip_path, folder_mapping):
    """
    folder_mapping: dict of {source_abs_dir: arcname_prefix}
    """
    print(f"📦 Đang tạo {os.path.basename(zip_path)}...")
    t0 = time.time()
    total_files = 0
    total_bytes = 0
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for src_dir, arc_prefix in folder_mapping.items():
            if not os.path.exists(src_dir):
                raise FileNotFoundError(f"Không tìm thấy thư mục: {src_dir}")
            for root, dirs, files in os.walk(src_dir):
                for f in files:
                    full_p = os.path.join(root, f)
                    rel_p = os.path.relpath(full_p, src_dir)
                    arc_name = os.path.join(arc_prefix, rel_p).replace("\\", "/") if arc_prefix else rel_p.replace("\\", "/")
                    zf.write(full_p, arc_name)
                    total_files += 1
                    total_bytes += os.path.getsize(full_p)
                    if total_files % 500 == 0:
                        print(f"   Đã nén {total_files:,} tệp ({total_bytes / (1024**2):.1f} MB)...", end="\r", flush=True)

    dt = time.time() - t0
    zip_size_mb = os.path.getsize(zip_path) / (1024**2)
    print(f"\n[✓] Hoàn tất: {os.path.basename(zip_path)}")
    print(f"    - Số tệp: {total_files:,} files | Dung lượng gốc: {total_bytes / (1024**2):.1f} MB | Dung lượng ZIP: {zip_size_mb:.1f} MB")
    print(f"    - Thời gian nén: {dt:.1f} giây")
    print(f"    - Đường dẫn: {zip_path}\n")

def main():
    print("=" * 84)
    print("🚀 ĐÓNG GÓI DATASET CHO KAGGLE FINE-TUNE & BENCHMARK TPS-INBETWEEN")
    print("=" * 84)
    
    # 1. Đóng gói gộp Train + Benchmark #1
    zip1_path = os.path.join(OUTPUT_DIR, "sakuga_final_dataset_v2.zip")
    mapping_1 = {
        os.path.join(PROJECT_ROOT, "Dataset", "sakuga_pipeline_sequence", "final_dataset", "sakuga_golden_dataset_sequences_tps_v2"): "sakuga_golden_dataset_sequences_tps_v2",
        os.path.join(PROJECT_ROOT, "Dataset", "sakuga_pipeline_sequence", "final_dataset", "sakuga_golden_benchmark_sequences_v2"): "sakuga_golden_benchmark_sequences_v2",
    }
    zip_directory_contents(zip1_path, mapping_1)

    # 2. Đóng gói riêng Benchmark #2
    zip2_path = os.path.join(OUTPUT_DIR, "anime_lineart_continuous_test.zip")
    mapping_2 = {
        os.path.join(PROJECT_ROOT, "data", "lineart_experiments", "test"): "",
    }
    zip_directory_contents(zip2_path, mapping_2)

    print("=" * 84)
    print("🎉 TẤT CẢ FILE ĐÃ SẴN SÀNG ĐỂ TẢI LÊN KAGGLE!")
    print(f"📁 Thư mục đầu ra: {OUTPUT_DIR}")
    print("=" * 84)
    print("\n👉 HƯỚNG DẪN TẢI LÊN KAGGLE:")
    print("1) TẬP GỘP (Train + Benchmark #1):")
    print(f"   - File tải lên: {zip1_path}")
    print("   - Title gợi ý : Sakuga Golden Dataset & Benchmark V2")
    print("   - Slug URL    : sakuga-golden-dataset-v2")
    print("2) TẬP BENCHMARK #2 (Test thực tế 24 frame 540p):")
    print(f"   - File tải lên: {zip2_path}")
    print("   - Title gợi ý : Anime Lineart Continuous Test 540p")
    print("   - Slug URL    : anime-lineart-continuous-test")
    print("3) CHECKPOINT GỐC ĐÃ CÓ TRÊN KAGGLE:")
    print("   - URL đã có   : /kaggle/input/datasets/dungbaoviec123/tps-baseline-model")
    print("   - Tệp model   : model_latest.pt")
    print("=" * 84)

if __name__ == "__main__":
    main()
