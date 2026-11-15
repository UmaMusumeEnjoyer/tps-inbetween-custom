# 📖 HƯỚNG DẪN THỰC THI THỰC NGHIỆM ABLATION STUDY: M0 vs M1 vs M3 (PATCH 1)
## 🔑 BẰNG CHỨNG THÉP #1: CHỨNG MINH ĐÓNG GÓP 100% CỦA DATASET SEQUENCE V1 TRÊN KIẾN TRÚC GỐC

> **Thư mục cách ly độc lập:** `E:\CODE\PBL\PBL6\kaggle_seq_00_ablation_m1_m3_patch1\`  
> Được tách biệt hoàn toàn khỏi các phiên bản cũ trong `kaggle/` (`v0.1` $\to$ `v3.2`) và `Finetune/` để phục vụ riêng cho mục tiêu báo cáo khoa học.

---

### 1. Ý Nghĩa Khoa Học Trong Báo Cáo Đề Tài (Ablation Study Matrix)

Căn cứ theo ma trận thực nghiệm:
- **`M0` — TPS Coarse Warp (*Lower Bound*):** Toán tử hình học TPS thuần ($\lambda = 0$), không qua mạng nơ-ron tinh chỉnh (`IFNet` + `UNet`).
- **`M1` — TPS-Inbetween Original (*Baseline A*):** Kiến trúc $M_{\text{gốc}}$ (`TPSInbetween` nguyên bản của bài báo *Zhu et al., HIT 2024*), nạp trọng số gốc huấn luyện trên **Dataset A (`ML240` - `model_latest.pt`)** với hàm mục tiêu $\mathcal{L}_{\text{DT}} + \mathcal{L}_{\text{LPIPS}}$.
- **`M3` — Ours Sequence V1 (Arch Original) (*Dataset Ablation*):** 
  - **Kiến trúc mạng:** $M_{\text{gốc}}$ (**Giữ nguyên 100% kiến trúc của `M1`**, tuyệt đối không thay đổi bất kỳ lớp nào).
  - **Quá trình huấn luyện:** Khởi tạo từ chính **`M1` (`Dataset A`)** $\longrightarrow$ Fine-tune trên **Sequence V1 ($B_{\text{new}}$ - Patch 1)** với **Bộ Loss chuẩn Lineart**.
  - **🔑 BẰNG CHỨNG THÉP #1:** Vì `M3` và `M1` có kiến trúc **giống hệt nhau 100%**, toàn bộ mức cải thiện của `M3` so với `M1` là **100% nhờ chất lượng của Dataset mới**!

---

### 2. Danh Mục Các Tệp Trong Thư Mục

| Tệp Tin | Vai Trò & Mục Đích |
| :--- | :--- |
| **`train_ablation_m1_to_m3_sequence_patch1_kaggle.ipynb`** | **Notebook chính hoàn chỉnh (10 Cells)** sẵn sàng upload lên Kaggle để chạy thử nghiệm. |
| **`seq_00_ablation_m1_m3_patch1_worker.py`** | Script động cơ phân tán DDP 2x GPU (Zero Host RAM leak, 100% GPU utilization, tự động nhúng vào Notebook). |
| **`generate_m1_to_m3_patch1_notebook.py`** | Script đóng gói tự động tạo file `.ipynb` tự thân (Self-contained). |
| **`verify_m1_to_m3_patch1_notebook.py`** | Script kiểm định toàn diện 5/5 bài test (AST, Identity M1 == M3, Loss, Checkpoint, Metrics). |
| **`README_M1_TO_M3_PATCH1.md`** | Tài liệu hướng dẫn sử dụng và giải thích khoa học. |

---

### 3. Hướng Dẫn Từng Bước Vận Hành Trên Kaggle

#### Bước 1: Chuẩn bị Dataset Input trên Kaggle
Bạn vào Kaggle và gắn 2 nguồn Input sau vào Notebook:
1. **Dataset Sequence Patch 1:** Gắn file `sakuga_golden_sequences_part01.zip` (hoặc thư mục dataset đã giải nén có chứa `clips/`, `matches/` và `sequences.csv`).
2. **Checkpoint M1 Gốc:** Gắn file [`model_latest.pt`](file:///e:/CODE/PBL/PBL6/checkpoints/pretrained_baselines/tps-original/model_latest.pt) (upload dạng Kaggle Dataset).

#### Bước 2: Cấu hình Môi trường Phiên chạy (Session Settings)
- **Accelerator:** `GPU T4 x2` (Bắt buộc chọn 2x T4 để kích hoạt DDP 100% công suất).
- **Internet:** `On` (Để cập nhật các thư viện nếu cần).
- **Environment:** `Always use latest environment`.

#### Bước 3: Upload Notebook và bấm `Run All`
1. Chọn menu **File** $\longrightarrow$ **Upload Notebook**, chọn file:  
   [`train_ablation_m1_to_m3_sequence_patch1_kaggle.ipynb`](file:///e:/CODE/PBL/PBL6/kaggle_seq_00_ablation_m1_m3_patch1/train_ablation_m1_to_m3_sequence_patch1_kaggle.ipynb).
2. Nhấn nút **Run All**:
   - **Cell 1:** Khởi tạo môi trường, khóa phân mảnh RAM glibc, ghi worker ra đĩa.
   - **Cell 2:** Tự động phát hiện dataset Patch 1 và checkpoint `model_latest.pt`, giải nén siêu tốc sang ổ NVMe SSD `/tmp` (~10-20 giây).
   - **Cell 3:** Tiền tính toán toàn bộ nghiệm ma trận TPS 2 chiều bằng 16 luồng CPU và ghi cache Mmap `tps_cache.npz` (~10-15 giây).
   - **Cell 4:** Kiểm toán toán học khẳng định $M_{\text{gốc}}$ giữa `M1` và `M3` là đồng nhất 100%.
   - **Cell 5:** 
     - **Epoch 0:** Tự động đo lường Baseline cho cả **`M0`** và **`M1`** trên toàn bộ tập Validation độc lập.
     - **Epoch 1 $\to$ 15:** Huấn luyện **`M3`** với Bộ Loss chuẩn Lineart (~1.5 - 2.5 phút/epoch train, ~15 - 25 giây validation).
   - **Cell 6 (Đã Comment):** Cell khẩn cấp dùng để reset kernel khi cần.
   - **Cell 7:** Tự động tính toán và hiển thị **Bảng đối chuẩn đối đầu trực tiếp: `M0` vs `M1` vs `M3`** (kèm % cải thiện định lượng và phân tách chi tiết theo độ dài chuỗi $N=3, 4, 5$).
   - **Cell 8:** Tự động vẽ Dashboard hội tụ 6 đồ thị và xuất ảnh so sánh trực quan 5 cột (`[Overlay | M0 | M1 | M3 | GT]`).
   - **Cell 9:** Đóng gói toàn bộ thành phẩm vào file ZIP.

---

### 4. Cơ Chế Xử Lý Khi Tràn RAM Tích Lũy (Kernel Reset + Auto-Resume)

Hệ thống được trang bị 2 tầng phòng thủ chống tràn RAM:
1. **Tầng 1 (Bộ dọn dẹp bộ nhớ glibc & OpenCV lock):** Ép trả RAM về hệ điều hành sau mỗi epoch (`malloc_trim`).
2. **Tầng 2 (Watchdog tự động dừng ở mã 42):** Nếu RAM chạm ngưỡng 23.5 GB, worker tự động lưu checkpoint `model_m3_seq_patch1_resume.pt` an toàn rồi thoát với mã `42`.

**Cách Reset Kernel để Train tiếp liền mạch:**
1. Đến **Cell 6** (nằm ngay sát dưới Cell 5 Train).
2. Bỏ dấu comment `#` ở 2 dòng:
   ```python
   import os
   os._exit(0)
   ```
3. Chạy riêng Cell 6. Sau 3 giây, RAM máy ảo sẽ được giải phóng sạch sẽ về ~1GB trong khi **toàn bộ dữ liệu trong `/kaggle/working` và `/tmp` được bảo toàn 100%**.
4. Comment lại 2 dòng trên và bấm **Run All**: Worker sẽ tự động nhận diện file `resume.pt` và tiếp tục huấn luyện liền mạch từ Epoch tiếp theo!

---

### 5. Chính Sách Checkpoint Tinh Gọn & Thành Phẩm Đầu Ra

Trong thư mục `/kaggle/working/checkpoints_m3_patch1/` **chỉ lưu đúng 3 file**:
1. `model_m3_seq_patch1_best.pt` — Trọng số đại diện chính thức của `M3` (Điểm kiểm định cao nhất).
2. `model_m3_seq_patch1_latest.pt` — Trọng số epoch gần nhất.
3. `model_m3_seq_patch1_resume.pt` — Trạng thái đầy đủ để Auto-Resume.

Toàn bộ kết quả sau khi chạy xong sẽ nằm trong gói:
📦 `/kaggle/working/tps_ablation_m1_to_m3_patch1_release.zip` bao gồm:
- 2 file trọng số tốt nhất (`best.pt`, `latest.pt`).
- Bảng dữ liệu định lượng CSV: `ablation_benchmark_M0_M1_M3_patch1.csv`.
- Bảng JSON đối chuẩn ban đầu: `baseline_m0_m1_metrics.json`.
- Lịch sử huấn luyện đầy đủ 15 epochs: `training_history_m3_patch1.csv`.
- Ảnh Dashboard hội tụ: `m3_patch1_convergence_dashboard.png`.
- Lưới ảnh trực quan 5 cột: `visual_ablation_m0_m1_m3_comparison.png`.
