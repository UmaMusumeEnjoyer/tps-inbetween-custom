# HƯỚNG DẪN TẠO DATASET HYBRID (ATD-12K ANIME + SAKUGABOORU / ANIMEGOC) & CHẠY DEEP LEARNING TRÊN KAGGLE

Hệ thống hỗ trợ 2 quy trình thực thi linh hoạt:

### 🌟 QUY TRÌNH 1: TRÍCH XUẤT & TRỘN TRỰC TIẾP TRÊN KAGGLE (KHUYÊN DÙNG CHO SAKUGABOORU)
Dành cho trường hợp bạn upload video `sakugabooru_character_acting` lên Kaggle và tận dụng dataset `std12k` / `atd12k` có sẵn trên Kaggle:
1. **Notebook 0 (`00_extract_and_mix_sakugabooru_atd12k_kaggle.ipynb`):**
   - Đọc video `sakugabooru_character_acting` + lọc anime Nhật `Japan_*`, `Hayao_*` từ `std12k`.
   - Lọc frame trùng lặp (CUDA Micro-Dedup Sobel + PSNR) và khai thác Candidate Triplets đa mốc $t \in \{0.25, 0.333, 0.5, 0.667, 0.75\}$.
   - Trộn 2 nguồn thành raw frames chưa chỉnh sửa và đóng gói thành `candidate_triplets_anime_540p.zip` (~4 - 5 GB).
   - Tải file ZIP này về máy cá nhân, sau đó upload lên Kaggle Dataset để đưa vào Pipeline Deep Learning.
2. **Notebook 1 (`generate_tps_dataset_kaggle.ipynb`):**
   - Chạy Deep Learning nặng: GlueStick Keypoint Matching & RANSAC Screening + ISNetDIS tách nhân vật + AniLines Detail trích xuất nét vẽ.
   - Đóng gói thành `anime_tps_finetune_dataset_ready.tar.gz` phục vụ trực tiếp cho việc Fine-tune TPS-Inbetween.

---

### 💻 QUY TRÌNH 2: TIỀN XỬ LÝ CỤC BỘ TRÊN MÁY TÍNH CÁ NHÂN (LOCAL PC)
Dành cho trường hợp xử lý các tập phim truyền hình 24 phút trong `AnimeGoc` kèm tập ATD-12K cục bộ:
1. **Tầng 1 (Local PC - Không tốn giờ Kaggle):**
   - Tự động nhận diện & loại bỏ Opening (OP), Ending (ED), Preview, Eyecatch qua các script `01`, `02`, `03`, `04`.
   - Đóng gói thành `candidate_triplets_anime_540p.zip`.
2. **Tầng 2 (Kaggle Cloud - GPU Tesla T4):**
   - Chạy `generate_tps_dataset_kaggle.ipynb`.

---

## 1. GIẢI ĐÁP CHI TIẾT: CÓ THỂ NHẬN BIẾT & LOẠI BỎ OPENING/ENDING TỰ ĐỘNG KHÔNG?

### A. Kết quả kiểm chứng thực tế trên 62 tập phim trong thư mục `AnimeGoc`:
Chúng tôi đã chạy phân tích trực tiếp toàn bộ 62 tập phim MKV hiện có:
- **`Adachi and Shimamura` (12 tập):** **100% có Chapter metadata** phân tách rõ: `Intro`, `OP` (đúng 90.0s), `Episode`, `ED` (đúng 90.0s), `Outro`, `Preview`.
- **`Akebi's Sailor Uniform` (12 tập):** **100% có Chapter metadata**: `Prelude`, `OP` (đúng 90.0s), `Part A`, `Eyecatch`, `Part B`, `ED` (đúng 91.2s).
- **`There's No Freaking Way...` (12 tập):** **100% có Chapter metadata**: `Prologue`, `Opening` (đúng 90.0s), `Episode`, `Ending` (đúng 91.1s).
- **`The World Is Dancing` (13 tập):** **100% có Chapter metadata**: `Prologue`, `Opening` (đúng 89.9s), `Episode`, `Ending`, `Epilogue`.
- **`Watashi wo Tabetai` (13 tập):** 11/13 tập có chapter, chỉ 2 tập dùng luật thông minh.

👉 **KẾT QUẢ:** **60 / 62 tập phim (>96%) ĐÃ CÓ SẴN CHAPTER METADATA CHÍNH XÁC ĐẾN TỪNG MILI-GIÂY!**  
Script `01_op_ed_detector.py` tự động đọc các tag này và loại bỏ 100% OP/ED mà không cần suy đoán.

### B. Tại sao tuyệt đối KHÔNG NÊN cắt mù 5 phút đầu và 5 phút cuối?
1. **Lãng phí tới 42% dữ liệu quý giá:** 1 tập anime truyền hình dài 24 phút (1,440 giây). Cắt 5 phút đầu + 5 phút cuối = vứt bỏ 10 phút, chỉ còn lại 14 phút phim!
2. **Mất sạch các cảnh hành động mở đầu (Prologue / Cold Open):** Anime Nhật hầu như luôn có đoạn dẫn 1 - 2 phút trước khi nhạc OP vang lên (chứa các cảnh nhân vật đánh nhau, rượt đuổi hoặc đối thoại cao trào). Cắt 5 phút đầu sẽ làm mất toàn bộ các cảnh đắt giá này.
3. **Quy chuẩn công nghiệp Anime Nhật Bản:** Độ dài bài hát OP và ED gần như **100% cố định ở mức 89 - 90 giây (1 phút 30 giây)**. Khi không có chapter, chỉ cần cắt đúng cửa sổ 90 giây thay vì cắt cả 5 phút.

---

## 2. QUY TRÌNH THỰC HIỆN DƯỚI LOCAL (MÁY CỦA BẠN)

Mở terminal trong thư mục `Kaggle_Dataset_Maker`:

### Bước 1: Quét và Lập bản đồ phân đoạn sạch (Loại bỏ OP/ED)
```bash
python 01_op_ed_detector.py
```
- Quét qua 62 file video trong `AnimeGoc`.
- Xuất file `clean_segments_manifest.json` ghi lại chính xác các khoảng thời gian hợp lệ (giữ lại khoảng 20 - 22 phút phim chất lượng cao mỗi tập).

### Bước 2: Khai thác Candidate Triplets từ AnimeGoc
```bash
python 02_extract_local_candidates.py
```
- Sử dụng GPU RTX 3050 chạy Micro-Dedup Sobel + PSNR.
- Khai thác các triplet đa mốc thời gian ($t = 0.25, 0.33, 0.5, 0.67, 0.75$).
- Resize về 540p ($960 \times 540$) và lưu vào `work/candidates_animegoc/`.
- Thu hoạch: ~3,000 - 3,500 candidate triplets.

### Bước 3: Chuẩn hóa tập Anime Nhật của ATD-12K
```bash
python 03_pack_atd12k_japan.py
```
- Đọc thư mục `E:\Dataset\datasets\train_10k`.
- Lọc lấy đúng 4,543 triplet `Japan_*` và 338 triplet `Hayao_*` (loại bỏ 5,119 triplet Disney).
- Resize từ 1080p về 540p đa luồng.
- Lưu vào `work/candidates_atd12k/` với nhãn $t = 0.5$.

### Bước 4: Gom chung và Nén thành file ZIP cho Kaggle
```bash
python 04_merge_and_build_kaggle_input.py
```
- Hợp nhất hai nguồn, đánh số thứ tự từ `triplet_000001` đến hết.
- Sinh file `dataset_manifest.json`.
- Nén thành file `candidate_triplets_anime_540p.zip` (dung lượng khoảng **4.5 - 5.0 GB**).

---

## 3. QUY TRÌNH CHẠY DEEP LEARNING TRÊN KAGGLE

### Bước 3.1: Đưa Dataset lên Kaggle
1. Đăng nhập vào [Kaggle](https://www.kaggle.com/).
2. Chọn menu **Datasets** $\to$ **New Dataset**.
3. Đặt tên dataset: `anime-candidate-triplets-540p`.
4. Upload file `candidate_triplets_anime_540p.zip`. Bấm **Create**.

### Bước 3.2: Đưa Notebook lên Kaggle & Cấu hình phần cứng
1. Chọn menu **Code** $\to$ **New Notebook**.
2. Chọn **File** $\to$ **Upload Notebook** $\to$ Chọn file `generate_tps_dataset_kaggle.ipynb`.
3. Tại bảng điều khiển bên phải (**Notebook Options**):
   - **Accelerator:** Chọn **GPU T4 x2** (hoặc GPU P100).
   - **Internet:** Bật **On** (bắt buộc để tải weights mô hình).
4. Tại mục **Input:**
   - Bấm **+ Add Input**.
   - Tìm kiếm và chọn Dataset `anime-candidate-triplets-540p` vừa upload ở Bước 3.1.

### Bước 3.3: Chạy Pipeline Tạo Dataset (Run All)
1. Bấm nút **Run All**.
2. Notebook sẽ tự động thực hiện:
   - Tải weights cho GlueStick, AniLines Detail, ISNetDIS.
   - Giải nén candidate triplets.
   - Chạy Stage 4 (GlueStick RANSAC) đo chuyển động và lưu file `.npy`.
   - Chạy Stage 5 (AniLines + ISNetDIS + MS-SSAA) trích xuất line-art siêu mượt `_smooth.png`.
   - Chạy Stage 5.5 (Lineart Quality & Consistency Filter) loại bỏ triệt để các frame trắng trơn, rác nền hoặc mất nét.
   - Đóng gói chuẩn TPS-Inbetween (`train/`, `val/`, `matches/`, `triplets.csv`).
   - Nén kết quả thành `anime_tps_finetune_dataset_ready.tar.gz`.
3. **Thời gian chạy ước tính:** ~1.5 - 2.0 giờ GPU.
4. **Dung lượng kết quả:** File `anime_tps_finetune_dataset_ready.tar.gz` khoảng ~3.5 GB nằm trong output của Notebook, sẵn sàng nạp thẳng vào Notebook Fine-tune mô hình TPS-Inbetween!
