# PIPELINE XÂY DỰNG VÀ KIỂM CHỨNG DATASET ANIME CHO TPS-INBETWEEN
> **Dựa trên cơ chế trích xuất của ATD-12K (Anime Triplet Dataset) & Ràng buộc toán học của TPS-Inbetween**  
> **Mục tiêu:** Xây dựng quy trình tự động trích xuất các bộ ba triplet $(I_0, I_{GT}, I_1)$ từ video anime 2D thực tế, giải quyết triệt để vấn đề frame trùng lặp (Limited Animation / "On 2s", "On 3s") mà vẫn bảo toàn nhãn thời gian $t$ chính xác và nằm trong vùng dung sai chuyển động của TPS ($\le 15\text{px}$).

---

## I. BỐI CẢNH VÀ TẠI SAO CẦN CƠ CHẾ CỦA ATD-12K

### 1. Thách thức lớn nhất của Video Anime 2D
Khác với video người thật quay liên tục 24–60fps (mỗi frame đều có chuyển động):
* **Anime sử dụng hoạt họa giới hạn (Limited Animation):** Một video container 24fps thường chỉ chứa 8 đến 12 bản vẽ độc lập mỗi giây ("On 3s" giữ 3 frame video, "On 2s" giữ 2 frame video, hoặc hold pose tĩnh giữ hàng chục frame).
* **Không có bảng Timesheet (X-sheet) của Studio:** Khi khai thác anime thương mại, chúng ta không biết trước frame nào là Keyframe (Genga), frame nào là Inbetween (Douga), và frame nào chỉ là bản sao lặp lại (duplicate/held frame).

### 2. Sự khác biệt giữa Cách làm cũ (Thất bại) vs. Chuẩn ATD-12K

| Tiêu chí | Pipeline cũ (`scene_to_frame.py`) | Pipeline mới (Chuẩn ATD-12K + TPS Constraint) |
|:---|:---|:---|
| **Trích xuất video** | Bỏ frame dựa trên MAE thô ($>6.0$) $\rightarrow$ mất inbetween nhỏ | Trích xuất toàn bộ 24fps liên tục (`%06d.png`), giữ nguyên dòng thời gian |
| **Lọc trùng lặp (Dedup)** | Ngưỡng quá cao, xóa nhầm frame vẽ thật (mắt, miệng, tóc) | **Micro-Deduplication**: Ngưỡng siêu nhạy (PSNR $>42\text{dB}$, MAE $<0.4$), giữ lại toàn bộ cử động vi mô |
| **Ghi nhận Timeline** | Bỏ mất mốc thời gian, frame bị co cụm không đều | Lưu bảng ánh xạ: Mỗi bản vẽ $D_i$ tương ứng khoảng thời gian $[F_{\text{start}}, F_{\text{end}}]$ |
| **Xác định Triplet** | Tự gán Keyframe cách nhau $0.5\text{s}$ ($\Delta k \approx 12$ frames) | **Triplet Mining kề cận**: $I_0 = D_k$, $I_1 = D_{k+2}$, $I_{GT} = D_{k+1}$ (bước nhảy chỉ 1 bản vẽ) |
| **Nhãn thời gian $t$** | Gán cứng hoặc chia đều giả định | **Tính từ timeline thực tế:** $t = (F_{k+1} - F_k) / (F_{k+2} - F_k)$ chuẩn xác theo nhịp easing |
| **Dung sai chuyển động** | Không kiểm tra, độ dời lên tới $90 \sim 178\text{px}$ $\rightarrow$ rách nét, ghosting | **GlueStick Screening**: Bắt buộc Mean Disp $\le 15\text{px}$, Inliers $\ge 20$, Outliers $>50\text{px} \approx 0\%$ |
| **Line Art** | Nhị phân 1-bit (`NoGay`) làm gãy gradient | Grayscale chuẩn `ml240data` + Smooth sub-pixel + Nền trắng 255 |

---

## II. SƠ ĐỒ TOÀN CẢNH PIPELINE (ARCHITECTURE WORKFLOW)

```
[Video Anime 24fps Gốc]
         │
         ▼  (GIAI ĐOẠN 1: Pre-processing)
[Trích xuất chuỗi frame 24fps liên tục: 000001.png, 000002.png, ...]
         │
         ▼  (GIAI ĐOẠN 2: Micro-Deduplication theo chuẩn ATD-12K)
[Bảng danh sách Unique Drawings: D_0, D_1, D_2, ... kèm Timeline thực tế]
         │
         ▼  (GIAI ĐOẠN 3: Triplet Candidate Mining)
[Tập ứng viên Triplet kề cận: (I_0 = D_k, I_GT = D_{k+1}, I_1 = D_{k+2}) + nhãn t]
         │
         ▼  (GIAI ĐOẠN 4: TPS & GlueStick Motion Screening)
[Lọc cặp: Đo độ dời RANSAC, inliers >= 20, Mean Disp <= 15px]
         │
         ▼  (GIAI ĐOẠN 5: Line Art Pipeline Chuẩn ML240)
[Tách nền nhân vật → Trích xuất Line Art Detail → Smooth sub-pixel]
         │
         ▼  (GIAI ĐOẠN 6: Pre-compute Matches & Export)
[Bộ dữ liệu hoàn chỉnh: train/val clips + matches/*.npy + triplets.csv]
```

---

## III. CHI TIẾT 6 GIAI ĐOẠN THỰC THI

### GIAI ĐOẠN 1: Chuẩn Bị Video & Trích Xuất 24fps Hoàn Chỉnh

1. **Phát hiện chuyển cảnh (Scene Detection):**
   * Sử dụng thuật toán Content-Aware Detection (như `PySceneDetect`) để cắt video thành từng phân đoạn cảnh liên tục (shot).
   * **Nguyên tắc:** Loại bỏ hoàn toàn các khung hình chuyển cảnh (cut scene, fade in/out, wipe) để tránh ghép cặp hai cảnh không liên quan.
   * Thời lượng mỗi phân cảnh tối thiểu: $2.0\text{s}$ (tối thiểu 48 frame liên tục).

2. **Trích xuất chuỗi frame đầy đủ:**
   * Sử dụng lệnh `ffmpeg` chuẩn công nghiệp:
     ```bash
     ffmpeg -i clip_scene.mp4 -vf "fps=24" -vsync 0 frames/%06d.png
     ```
   * **Quy chuẩn đặt tên:** Bắt buộc dùng `%06d.png` (`000001.png`, `000002.png`,...). Không dùng số thực (`0.1s.png`) để tránh lỗi sắp xếp thứ tự từ điển (lexicographical sorting).

---

### GIAI ĐOẠN 2: Lọc Trùng Lặp Vi Mô (Micro-Deduplication - Chuẩn ATD-12K)

Mục đích: Phát hiện các khung hình bị lặp lại do vẽ "On 2s" hoặc "On 3s" mà **không làm mất các khung hình có chuyển động nhỏ** (chớp mắt, mở miệng, lọn tóc).

1. **Chỉ số đo lường kép (Dual-Metric Sensitivity):**
   So sánh frame liền kề $F_i$ và $F_{i+1}$:
   * **PSNR toàn khung hình:**
     $$\text{PSNR}(F_i, F_{i+1}) = 10 \cdot \log_{10}\left(\frac{255^2}{\text{MSE}}\right)$$
   * **Edge-Difference (Độ sai khác biên nét):** Áp dụng bộ lọc Sobel/Canny để tính sai khác riêng trên các đường nét nhân vật:
     $$\Delta_{\text{edge}} = \text{mean}(|\text{Sobel}(F_i) - \text{Sobel}(F_{i+1})|)$$

2. **Quy tắc quyết định trùng lặp:**
   * Một frame $F_{i+1}$ bị coi là **Trùng lặp (Duplicate)** nếu:
     $$\text{PSNR} > 42.0\text{ dB} \quad \text{VÀ} \quad \Delta_{\text{edge}} < 0.35$$
   * Nếu có chuyển động nhỏ (mắt chớp, miệng mở): PSNR có thể cao (~38–40dB) nhưng $\Delta_{\text{edge}} > 0.5$ $\longrightarrow$ **Được giữ lại làm bản vẽ độc lập (Unique Drawing)**.

3. **Cấu trúc dữ liệu đầu ra:**
   Mỗi video phân cảnh sinh ra danh sách bản vẽ:
   ```json
   [
     {"drawing_id": 0, "frame_file": "000001.png", "start_frame": 1, "duration": 2},
     {"drawing_id": 1, "frame_file": "000003.png", "start_frame": 3, "duration": 3},
     {"drawing_id": 2, "frame_file": "000006.png", "start_frame": 6, "duration": 2}
   ]
   ```

---

### GIAI ĐOẠN 3: Khai Phá Triplet (Triplet Mining) & Tính Nhãn Thời Gian $t$

Không lấy mẫu ngẫu nhiên khoảng cách xa. Chỉ khai phá các bộ ba dựa trên chuỗi Unique Drawings:

1. **Quy tắc chọn Triplet kề cận (Step = 2):**
   * Frame đầu $I_0 = D_k$
   * Frame cuối $I_1 = D_{k+2}$
   * Frame ground truth $I_{GT} = D_{k+1}$

2. **Tính toán giá trị $t$ chuẩn xác theo dòng thời gian vật lý:**
   Do mỗi bản vẽ được giữ trên màn hình với số frame khác nhau ($L_k$), giá trị $t$ không thể mặc định là $0.5$ mà phải được tính bằng:
   $$t = \frac{\text{start\_frame}(D_{k+1}) - \text{start\_frame}(D_k)}{\text{start\_frame}(D_{k+2}) - \text{start\_frame}(D_k)}$$
   * *Ví dụ:* $D_0$ bắt đầu tại frame 1, $D_1$ bắt đầu tại frame 3, $D_2$ bắt đầu tại frame 7:
     $$t = \frac{3 - 1}{7 - 1} = \frac{2}{6} \approx 0.333$$
   * Nhãn $t$ này phản ánh chính xác nhịp chuyển động (easing/spacing) của họa sĩ hoạt họa.

---

### GIAI ĐOẠN 4: Bộ Lọc Dung Sai Chuyển Động Cho TPS (Motion Screening)

Đây là bước quyết định dataset có dùng được cho mô hình `tps-inbetween` hay không. Mô hình TPS-Inbetween dựa trên phép biến dạng không gian liên tục, không thể xử lý độ dời quá lớn hoặc biến đổi tô-pô.

Với mỗi triplet ứng viên $(I_0, I_1, I_{GT})$:
1. **Chạy GlueStick trích xuất đặc trưng:**
   * Tìm tập điểm tương ứng $(p_0, p_1)$ giữa $I_0$ và $I_1$.
2. **Lọc hình học bằng RANSAC Affine:**
   * Loại bỏ các điểm ngoại lai (outliers) do bắt cặp sai chi tiết giống nhau.
3. **Tiêu chuẩn kiểm duyệt (Acceptance Criteria):**
   * **Số điểm inliers sạch:** $\ge 25$ điểm.
   * **Độ dời trung bình (Mean Displacement):**
     $$\bar{d} = \frac{1}{N} \sum_{i=1}^N \|p_{0,i} - p_{1,i}\| \le 15.0\text{ pixel}$$
   * **Độ dời cực đại an toàn:** Không có điểm nào lệch quá $50.0\text{ pixel}$ sau RANSAC.
   * Nếu không thỏa mãn các điều kiện trên (ví dụ: nhân vật quay người khuất mặt, đổi góc camera nhanh) $\longrightarrow$ **Bỏ qua triplet này**.

---

### GIAI ĐOẠN 5: Pipeline Line Art Chuẩn ML240

Để thu hẹp Domain Gap với tập train gốc `ml240data`:

1. **Tách nền (Background Removal):**
   * Sử dụng `anime-segmentation` để giữ lại chủ thể nhân vật, nền chuyển thành trong suốt.
2. **Trích xuất nét vẽ (Line Extraction):**
   * Sử dụng `AniLines` ở chế độ `detail`.
   * **Nghiêm cấm:** Không dùng nhị phân hóa cứng 1-bit (`NoGay`), không ép ngưỡng 0/255 làm mất gradient.
3. **Chuẩn hóa & Làm mượt (Sub-pixel Anti-Aliasing):**
   * Áp dụng `smooth_lines.py`:
     * Phóng đại Lanczos-4 sub-pixel.
     * Bilateral filter bảo toàn biên.
     * Đưa nền về trắng tinh ($255$) và nét đen tuyền ($0$) nhưng giữ dải xám mềm ở rìa nét ($1 \sim 254$).
     * Đầu ra: file `*_smooth.png`.

---

### GIAI ĐOẠN 6: Xuất Bản Dataset Hoàn Chỉnh Cho TPS-Inbetween

1. **Cấu trúc thư mục tương thích tuyệt đối với `ml240.py`:**
   ```text
   Dataset_Anime_TPS_Test/
   ├── train/
   │   ├── clip_001/
   │   │   ├── 000001_smooth.png
   │   │   ├── 000002_smooth.png
   │   │   └── ...
   │   └── clip_002/
   ├── val/
   │   └── clip_003/
   ├── matches/
   │   ├── clip_001/
   │   │   ├── 0.npy   ← Lưu stack (n_kps0, n_kps1) đã chuẩn hóa [0, 1]
   │   │   ├── 1.npy
   │   │   └── ...
   │   └── clip_002/
   └── triplets.csv
   ```

2. **Định dạng file `triplets.csv`:**
   ```csv
   i0_path,i1_path,igt_path,t,mean_disp,inlier_count
   train/clip_001/000001_smooth.png,train/clip_001/000003_smooth.png,train/clip_001/000002_smooth.png,0.500,4.21,86
   train/clip_001/000003_smooth.png,train/clip_001/000007_smooth.png,train/clip_001/000005_smooth.png,0.333,6.85,64
   ```

3. **Quy chuẩn file `.npy` (Pre-computed Matches):**
   * Khớp với định dạng mà `generate_matches.py` và `ml240.py` của `tps-inbetween` yêu cầu:
     * Mảng numpy 2 kênh: shape `(2, N, 2)`.
     * Tọa độ chuẩn hóa theo chiều rộng và chiều cao: $x \in [0, 1], y \in [0, 1]$.

---

## IV. QUY TRÌNH KIỂM CHỨNG THỬ NGHIỆM (VALIDATION PROTOCOL)

Để đánh giá xem phương pháp tạo dataset theo chuẩn ATD-12K này có hiệu quả hay không trước khi đưa vào huấn luyện mô hình, chúng ta tiến hành kiểm chứng qua 4 bài test độc lập:

### Bài Test 1: Kiểm chứng chất lượng Triplet thu được (Dataset Quality Metrics)
Đo đạc trên tập triplet được sinh ra từ video mẫu (`ChayGo.mp4`):
* **Tỷ lệ giữ lại hợp lệ (Yield Rate):** Đạt bao nhiêu triplet trên 100 frame video gốc.
* **Phân phối độ dời (Displacement Distribution):**
  * Tỷ lệ triplet có $\text{mean disp} \in [3\text{px}, 10\text{px}]$ phải chiếm $>75\%$.
  * Tỷ lệ điểm ngoại lai $>50\text{px}$ phải đạt $0\%$.
* **Độ phân bố của $t$:** Giá trị $t$ phải đa dạng trong khoảng $(0.2, 0.8)$, không bị dồn cục về một giá trị duy nhất.

### Bài Test 2: Kiểm chứng tính tương thích kỹ thuật với DataLoader TPS-Inbetween
Viết script test nạp dữ liệu bằng chính class `ML240` trong [Test/TestModels/tps-inbetween/datasets/ml240.py](file:///e:/CODE/PBL/PBL6/Test/TestModels/tps-inbetween/datasets/ml240.py):
* Kiểm tra `assert len(frame) + self.xN == len(self.all_frame_path[idx])` có vượt qua không.
* Kiểm tra tensor đầu ra: `img_input` shape `(B, 2, 1, 512, 512)`, `img_gt` shape `(B, 1, 1, 512, 512)`.
* Kiểm tra nạp file `matches/*.npy` không bị crash bộ nhớ hoặc sai kích thước.

### Bài Test 3: Kiểm chứng Inference trực tiếp trên mô hình TPS gốc (Zero-shot Benchmark)
Chạy mô hình `tps-inbetween` gốc (chưa fine-tune) trên các triplet mới sinh theo đúng cặp $I_0, I_1$ và mốc $t$ thực tế:
* **So sánh chất lượng ảnh sinh ra với Ground Truth $I_{GT}$:**
  * **PSNR kỳ vọng:** Đạt từ **$16\text{dB} \sim 22\text{dB}$** (thay vì chỉ $10 \sim 11\text{dB}$ như khi ép nhảy $0.5\text{s}$).
  * **SSIM kỳ vọng:** Đạt $> 0.80$.
  * **Ghosting:** Không còn hiện tượng 2 bóng nét vẽ cách nhau hàng chục pixel.
* **Ý nghĩa:** Nếu mô hình gốc đạt được PSNR $\ge 16\text{dB}$ trên các triplet này, điều đó chứng minh bộ triplet **đã nằm trọn vẹn trong Design Envelope của TPS**, sẵn sàng làm dữ liệu huấn luyện fine-tune chất lượng cao.

### Bài Test 4: Đánh giá bằng mắt trên 10 Triplet ngẫu nhiên
Hiển thị trực quan:
* Frame $I_0$, Frame $I_1$.
* Frame $I_{GT}$ (vẽ tay thực tế).
* Frame $I_{\text{pred}}$ (do TPS nội suy tại mốc $t$).
* Heatmap sai khác $|I_{\text{pred}} - I_{GT}|$.

---

## V. KẾ HOẠCH TRIỂN KHAI THỬ NGHIỆM TIẾP THEO

1. **Tạo script tạo dataset thử nghiệm:**
   * Viết script `build_atd_dataset_test.py` thực hiện tuần tự Giai đoạn 1 đến Giai đoạn 6 trên 1 đoạn video duy nhất (`ChayGo.mp4`).
2. **Chạy kiểm chứng chất lượng:**
   * Xuất báo cáo thống kê: số triplet tạo được, phân phối độ dời, đồ thị giá trị $t$.
3. **Thực thi Bài Test 3:**
   * Chạy inference thử trên 20 triplet đầu tiên để xác nhận PSNR tăng vọt từ $11\text{dB} \rightarrow >16\text{dB}$.
4. **Mở rộng toàn bộ:**
   * Sau khi kiểm chứng thành công trên `ChayGo.mp4`, áp dụng đồng loạt cho toàn bộ các video trong `AnimeGoc`.
