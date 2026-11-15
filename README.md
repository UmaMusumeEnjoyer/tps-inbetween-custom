# DỰ ÁN PBL6: NỘI SUY KHUNG HÌNH HOẠT HỌA ANIME 2D VỚI TPS-INBETWEEN

> Chi tiết toàn bộ kiến trúc cây thư mục chuẩn hóa (8+1 layout), chức năng từng module và vai trò của các thư mục clone từ GitHub đã được tổng hợp đầy đủ tại:  
> 👉 **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**

---

## TỔNG QUAN ĐỀ TÀI
- **Mục tiêu:** Tự động sinh khung hình trung gian (inbetweening) cho hoạt họa anime 2D từ cặp keyframe nét vẽ (line art), giải quyết bài toán limited animation (*"on-twos"*, *"on-threes"*), giảm tải công việc vẽ tay thủ công cho họa sĩ.
- **Mô hình trọng tâm:** **TPS-Inbetween** (Thin-Plate Spline-based Interpolation for Animation Line Inbetweening).
- **Mô hình đối chứng (Baselines):** AnimeInbet (ICCV 23), AnimeInterp (CVPR 21), SAIN (ArXiv 23), uniVIP (ICASSP 25), UPR-Net (CVPR 23).
- **Công cụ trích xuất & xử lý nét:** AniLines, Anime Segmentation (IS-Net), Real-CUGAN, Anime2Sketch, SketchKeras.

## CẤU TRÚC REPOSITORY (8+1 LAYOUT CHUẨN HÓA)
1. **`src/`**: Thư viện dùng chung (`src.repo_paths`, `src.tps_v22`, `src.sequence_p3_p4`, `src.dataset_builder`, `src.utils`).
2. **`training/`**: Mã nguồn và notebooks huấn luyện Kaggle phân tầng (`01_dataset_notebooks/`, `02_triplet/`, `03_sequence/`).
3. **`evaluation/`**: Hệ thống benchmark định lượng, metrics đối đầu và visual audit outputs (`scripts/`, `metrics/`, `outputs/`).
4. **`demos/`**: Các bộ demo độc lập (`01_atd12k_orig_tps/`, `02_seq_p3_sharp/`).
5. **`third_party/`**: Mã nguồn bên thứ ba clone từ GitHub (`baselines/`, `lineart_tools/`).
6. **`tools/`**: Công cụ hiệu chuẩn trọng số và scripts tiện ích nội bộ (`score_weight_calibrator/`, `auto_commit/`, `setup_env/`).
7. **`scripts/`**: Automation batch files và scripts vận hành (`run_tps_server.bat`, `run_auto_playwright_v26.bat`, ...).
8. **`data_samples/`**: Caches cố định và artifacts mẫu thử nghiệm nhỏ (`caches/`, `atd12k_build_artifacts/`).
- **`docs/`**: Tài liệu kỹ thuật, kế hoạch tái cấu trúc và phân tích thử nghiệm.

*Các thư mục dữ liệu lớn, checkpoints và ứng dụng local (`checkpoints/`, `data/`, `Dataset/`, `product_app/`, `scratch/`, `.venv/`) được lưu cục bộ và loại trừ khỏi Git tracking.*

Xem chi tiết hướng dẫn và phân tích tại [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) hoặc thư mục tài liệu [docs/](docs/).
