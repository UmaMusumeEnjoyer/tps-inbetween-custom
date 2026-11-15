# TPS-Inbetween Finetune & Unbiased Benchmark (Option 3: LONG Mode)

Thư mục này chứa bộ mã nguồn và Notebook Kaggle tự thân để kiểm chứng khách quan xem tập dữ liệu chuỗi **`sakuga_golden_dataset_sequences_tps_v2`** (`final_dataset`) có thực sự cải thiện mô hình gốc **`model_latest.pt`** (*Zhu et al., 2024*) hay không, trong điều kiện **giữ nguyên 100% kiến trúc mạng (`13,026,543` active params)**.

---

## Danh mục Tệp

1. [`train_and_benchmark_tps_sequence_final_kaggle.ipynb`](train_and_benchmark_tps_sequence_final_kaggle.ipynb)  
   Notebook Kaggle hoàn chỉnh gồm 11 Cells (`Cell 0 -> Cell 10`), cấu hình mặc định **`TRAIN_MODE = "LONG"` (Option 3)**:
   - Khai thác đồng thời 2x GPU Tesla T4 qua `torchrun` DDP kèm bộ chia lô cân bằng tải `DDPLengthWidthBatchSampler`.
   - Đọc dữ liệu qua NVMe SSD `/tmp/tps_sequence_ssd/` và bộ nhớ đệm TPS Mmap (không tốn 20GB đĩa `/kaggle/working`).
   - Tự động nhận diện cả 2 dạng: Dataset đã giải nén hoặc tệp `.zip` thô.
   - Ưu tiên tự động nhận diện Checkpoint gốc tại `/kaggle/input/datasets/dungbaoviec123/tps-baseline-model` hoặc `/kaggle/input/tps-baseline-model`.
   - Có **Cell 6 đã comment** (`# import os` / `# os._exit(0)`) để reset sạch Host RAM trên Kaggle mà vẫn giữ nguyên 100% file kết quả, tự động **Auto-Resume** từ `model_tps_finetuned_last.pt`.
   - Chính sách lưu Checkpoint tối giản: **Chỉ lưu 2 file** (`model_tps_finetuned_best.pt` và `model_tps_finetuned_last.pt`).
   - Tích hợp sẵn **Benchmark #1** (`sakuga_golden_benchmark_sequences_v2` — 198 clips / 412 frames) và **Benchmark #2** (`data/lineart_experiments/test` — 24 frames 540p).
2. [`package_datasets_for_kaggle.py`](package_datasets_for_kaggle.py)  
   Script tự động nén đúng cấu trúc 2 file ZIP chuẩn bị tải lên Kaggle:
   - `sakuga_final_dataset_v2.zip` (~500 MB): Gộp chung cả 2 thư mục Train/Val và Benchmark #1.
   - `anime_lineart_continuous_test.zip` (~2.4 MB): Thư mục Benchmark #2.
3. [`seq_p1_v32_ghost_temp_scale_worker.py`](seq_p1_v32_ghost_temp_scale_worker.py)  
   Worker DDP độc lập thực hiện `TPSInbetweenOriginal`, chuẩn Backward-Warp TPS (`lambd=0.01`, pad mult-32, giữ tỷ lệ gốc `orig_hw`), chế độ `LONG` mặc định với 5-term loss: Charbonnier + 12-step Symmetric DT + Sobel + Soft-clDice + VGG16/Laplacian Perceptual Loss, Cosine Warmup & Adaptive EMA.
4. [`generate_kaggle_finetune_nb.py`](generate_kaggle_finetune_nb.py)  
   Script tái sinh tự động Notebook từ `seq_p1_v32_ghost_temp_scale_worker.py`, `run_benchmark_suite.py` và `lineart_test_gluestick_matches.npz`.
5. [`lineart_test_gluestick_matches.npz`](lineart_test_gluestick_matches.npz)  
   6 cặp điểm neo **GlueStick MD** trích xuất chuẩn xác cho 6 phân đoạn keyframe của `data/lineart_experiments/test` (được nhúng sẵn bên trong Notebook để Benchmark #2 trên Kaggle chạy đồng nhất 100% mà không cần cài thêm GlueStick).
6. [`../Dataset/sakuga_pipeline_sequence/final_dataset/sakuga_golden_benchmark_sequences_v2/run_benchmark_suite.py`](../Dataset/sakuga_pipeline_sequence/final_dataset/sakuga_golden_benchmark_sequences_v2/run_benchmark_suite.py)  
   Script đối chuẩn độc lập (Dual-Regime: Raw & Threshold/DoG, phân tầng theo `N=3,5,7` & `mean_disp`, kiểm định Wilcoxon theo 149 scenes).

---

## Cấu Hình 3 Dataset Trên Kaggle

| STT | Dataset Kaggle | File Tải Lên | Title Gợi Ý Trên Kaggle | Slug URL Kaggle / Đường Dẫn Mount |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **TPS Baseline Model** *(Đã có)* | `model_latest.pt` | *(Có sẵn)* | `dungbaoviec123/tps-baseline-model`<br>`/kaggle/input/datasets/dungbaoviec123/tps-baseline-model` |
| **2** | **Tập Gộp Train + Bench #1** | `sakuga_final_dataset_v2.zip` (~500 MB) | `Sakuga Golden Dataset & Benchmark V2` | `sakuga-golden-dataset-v2`<br>`/kaggle/input/sakuga-golden-dataset-v2` |
| **3** | **Tập Benchmark #2** | `anime_lineart_continuous_test.zip` (~2.4 MB) | `Anime Lineart Continuous Test 540p` | `anime-lineart-continuous-test`<br>`/kaggle/input/anime-lineart-continuous-test` |

---

## Hướng Dẫn Vận Hành Trên Kaggle (2x GPU T4)

1. Mở Kaggle, tạo New Notebook hoặc Upload file [`train_and_benchmark_tps_sequence_final_kaggle.ipynb`](train_and_benchmark_tps_sequence_final_kaggle.ipynb).
2. Tại thanh thiết lập bên phải:
   - **Accelerator:** Chọn **GPU T4 x2**
   - **Internet:** Bật **ON** (để VGG16 tải trọng số pretrained cho Perceptual Loss trong Option 3)
   - **+ Add Input:** Thêm cả 3 dataset ở bảng trên.
3. Bấm **Run All**.  
   - Cell 1: Khởi tạo môi trường, khóa Host RAM C, nhúng worker.
   - Cell 2: Tự động định vị toàn bộ 3 dataset và staging siêu tốc sang NVMe SSD `/tmp`.
   - Cell 3: Tiền tính toán TPS backward warp cache (16 CPU threads).
   - Cell 4: Sanity check forward / backward gradient trên 2 GPU.
   - Cell 5: Huấn luyện DDP chuỗi đầy đủ 540p (15 epochs).
   - Cell 7: Benchmark #1 đối chuẩn tập cách ly (198 clips, 412 frames, Wilcoxon 149 scenes).
   - Cell 8: Benchmark #2 đối chuẩn chuỗi keyframe thực tế 540p (24 frames).
   - Cell 9 & 10: Tổng hợp bảng báo cáo khoa học, kiểm định giả thuyết $H_0/H_1$ và nén gói kết quả tải về.
