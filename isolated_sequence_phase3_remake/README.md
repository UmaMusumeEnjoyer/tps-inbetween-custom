# Gói Cách Ly & Độc Lập Hoàn Toàn: Sequence 2 Pha 3 Remake

> **Mục đích:** Gói độc lập hoàn toàn chứa toàn bộ mã nguồn, mô hình, trọng số loader, coarse TPS đa tầng, bộ lọc thích ứng và cache điểm neo của **Sequence 2 Pha 3 Remake (cả Pipeline Mới và Pipeline Cũ/Raw)**.
> Gói này không phụ thuộc vào bất kỳ thay đổi tên thư mục nào của các pha khác trong repo PBL6.

---

## 1. Cấu Trúc Gói Cách Ly
* `cache/lineart_test_gluestick_matches.npz`: Cache điểm neo GlueStick phục vụ benchmark nét vẽ.
* `ddp_worker_tps_sequence_phase2.py`: Định nghĩa kiến trúc UNet gốc `TPSInbetweenOriginal`, hàm nạp weights và thuật toán Backward TPS của **Pipeline Cũ / TPS Gốc**.
* `ddp_worker_tps_sequence_phase3.py`: Worker tinh chỉnh Pha 3.
* `multi_layer_tps.py`: Tầng 1 - Coarse Multi-Layer Clustered TPS (tách cụm tĩnh và tiền cảnh động).
* `adaptive_sharpener.py`: Tầng 3 - Bộ lọc tương phản cục bộ kéo dốc biên nét và khử ghost.
* `phase3_inference_pipeline.py`: Pipeline suy luận chính thức (hỗ trợ cả mode `--mode balanced` của Pipeline Mới và `--no_sharpener` của Pipeline Cũ/Raw).
* `benchmark_orig_vs_pha3.py`: Bộ benchmark đối đầu 4 Cases (Case 1 Lineart 24f, Case 2 Squash Circle, Case 3 Hollow Cross, Case 4 Solid Cross).
* `render_defense_gain_table.py`: Bộ sinh 2 bảng đồ họa High-DPI phản biện cho Case 1 phục vụ bảo vệ đồ án.
* `test_isolated_suite.py`: Bộ test tự động kiểm chứng 100% các thành phần trên môi trường `.venv`.

---

## 2. Lệnh Chạy Kiểm Thử Độc Lập
Chạy ngay từ thư mục gốc của repo hoặc bên trong thư mục này:

```powershell
# Chạy bộ test kiểm chứng tự động toàn diện:
.venv\Scripts\python.exe isolated_sequence_phase3_remake/test_isolated_suite.py

# Xuất frame nội suy đơn lẻ hoặc chuỗi:
.venv\Scripts\python.exe isolated_sequence_phase3_remake/phase3_inference_pipeline.py --help

# Xuất lại 2 bảng ảnh bảo vệ đồ án:
.venv\Scripts\python.exe isolated_sequence_phase3_remake/render_defense_gain_table.py
```
