# Tổng Nhận định Toàn diện: Fine-tune TPS-Inbetween & Xây dựng Dataset

> Phân tích dựa trên: kết quả kiểm thử trong `Final test/`, phân tích trong `tps_inbetween_analysis_and_dataset_solution.md`, và đo đạc thực nghiệm trực tiếp trên dataset `ml240data` (train gốc) và chuỗi `ChayGo`.

---

## 1. Nhận xét về Sự Đúng đắn của Phương hướng Chuẩn bị Dataset để Fine-tune

**Kết luận: Phương hướng cơ bản là ĐÚNG, nhưng còn nhiều chi tiết quan trọng cần làm rõ trước khi triển khai.**

### Những điều đúng trong ý tưởng của bạn:

1. **Tải đoạn cắt anime về và trích xuất frame** → Đây là nguồn dữ liệu hợp lý. Dataset gốc `ml240data` (MixamoLine240) cũng có bản chất tương tự: là các chuỗi frame liên tục từ hoạt hình 3D được render thành line art. Fine-tune bằng anime 2D thực tế là hướng đi đúng để thu hẹp domain gap.

2. **Tách màu, giữ lại nét vẽ** → Hoàn toàn phù hợp với định dạng đầu vào của model. Qua đo đạc thực nghiệm, `ml240data` là ảnh **grayscale line art thuần túy** (97% white background, 1.4% dark lines, tất cả 3 channel RGB giống hệt nhau). Pipeline trích xuất line art từ anime màu là bước tiền xử lý bắt buộc và đúng hướng.

3. **Chỉ định random keyframe và ground truth inbetween** → Ý tưởng đúng về cơ chế tạo triplet $(I_0, I_1, I_t)$. Đây chính xác là định dạng mà model được thiết kế để học.

### Những điểm cần bổ sung/điều chỉnh quan trọng:

> [!IMPORTANT]
> **Điểm nguy hiểm nhất: Khoảng cách thời gian giữa $I_0$ và $I_1$ khi lấy mẫu ngẫu nhiên**

Chuỗi `ChayGo` trong `TestDataset/` được sample ở ~10fps (mỗi frame cách 0.1s). Nhưng các thử nghiệm hiện tại đang dùng khoảng cách 0.5s (tức 5 frame) → đây là nguyên nhân chính gây méo nét. Khi fine-tune:
- **Bước nhảy an toàn:** $\Delta k = 2 \sim 4$ frame (tức $\Delta t \approx 0.08s \sim 0.16s$ ở 24fps, hoặc $0.2s \sim 0.4s$ ở 10fps).
- **Tuyệt đối không lấy ngẫu nhiên $\Delta k$ quá lớn** (ví dụ $\Delta k = 5 \sim 12$ frame) nếu không có cơ chế phát hiện và loại bỏ cặp frame có displacement > 30px.

---

## 2. Nhận xét về Quy trình Xử lý Ảnh để Tạo Dataset

### Quy trình cần thiết (từ anime màu gốc → dataset fine-tune):

```
[1] Video anime gốc (có màu, ~24fps)
         ↓ Trích xuất frame (ffmpeg)
[2] Chuỗi frame PNG màu
         ↓ Tách nền nhân vật (anime-segmentation)
[3] Frame nhân vật đã tách nền (RGBA)
         ↓ Trích xuất line art (AniLines / Anime2Sketch / binarize_lines_nobg.py)
[4] Line art grayscale (white bg, black lines) ← ĐỊNH DẠNG CỦA ml240data
         ↓ Làm mượt nét (smooth_lines.py) - GIỮ anti-aliasing, KHÔNG binary hóa
[5] Line art smooth (như 0.0s_smooth.png trong ChayGoAnimeLinesDetail_NoBG)
         ↓ Chọn cặp keyframe (I_0, I_1) với Δk = 2~4 frames
[6] Tạo triplet: (I_0, I_GT_at_t, I_1) với t = k_gt / (k_1 - k_0)
```

### Các vấn đề kỹ thuật cụ thể của từng bước:

**Bước 3 (Tách nền):** Bạn đã có sẵn tool `anime-segmentation` trong `TestDataset/`. Cần đảm bảo mask đủ sắc nét, không bị "alpha bleeding" quanh rìa nét.

**Bước 4 (Trích xuất line art):** Bạn đã thử nghiệm nhiều phiên bản:
- `ChayGoAnimeLinesBasic` → Thiếu chi tiết
- `ChayGoAnimeLinesDetail` → Tốt hơn nhưng còn màu nền
- `ChayGoAnimeLinesDetail_NoBG` → Gần với định dạng ml240data nhất ✅
- `ChayGoAnimeLinesDetail_NoBG_NoGay` → **Sai hướng**: Binary 1-bit làm mất gradient → SuperPoint sinh góc giả → GlueStick match nhầm

> [!WARNING]
> **Không được dùng binary 1-bit (NoGay) cho cả quá trình fine-tune lẫn inference.** Đây là kết luận đã được chứng minh định lượng: `NoGay` sinh ra 19 điểm outlier >50px so với chỉ 9 điểm của `NoBG`, và có tới 4 điểm lệch >100px.

**Bước 6 (Tạo triplet):** Đây là bước tinh tế nhất:

| Sai (hiện tại) | Đúng (cần làm) |
|:---:|:---:|
| $I_0 =$ frame_0, $I_1 =$ frame_5, $I_{GT} =$ frame_2 | $I_0 =$ frame_0, $I_1 =$ frame_3, $I_{GT} =$ frame_1 hoặc frame_2 |
| $\Delta k = 5$ frames, displacement ~14px trung bình | $\Delta k = 2\sim3$ frames, displacement <8px |

**Lưu ý quan trọng về $t$:** Khi lấy $I_0 =$ frame_0, $I_1 =$ frame_3, $I_{GT} =$ frame_1: $t = 1/3 \approx 0.333$. Khi $I_{GT} =$ frame_2: $t = 2/3 \approx 0.667$. Giá trị $t$ phải tính toán chính xác theo vị trí tương đối thực tế, không phải gán cứng.

---

## 3. Nhận xét về Hiểu nhầm trong Vấn đề Tạo Dataset

> [!CAUTION]
> **Đây là điểm quan trọng nhất và bạn đang có một hiểu nhầm cơ bản cần phải làm rõ ngay.**

### Hiểu nhầm của bạn:

Bạn đang đánh giá chất lượng model TPS-Inbetween hiện tại bằng cách nhìn vào ảnh sinh ra trong `Final test/` và so sánh với ground truth trong `ChayGoAnimeLinesDetail_NoBG/` (hoặc `_NoGay`). Bạn lo ngại vì ảnh sinh ra **không giống ground truth**.

### Thực tế là gì:

**Ảnh trong `ChayGoAnimeLinesDetail_NoBG/` tại các mốc `0.1s.png`, `0.2s.png`, `0.3s.png` KHÔNG PHẢI là ground truth hợp lệ để đánh giá model hiện tại**, vì những lý do sau:

#### Lý do 1: Khoảng cách quá lớn giữa hai keyframe

Qua đo đạc thực nghiệm:
```
Keyframe 0.0s vs 0.5s: PSNR = 10.75 dB
Frame 0.0s vs 0.1s:    PSNR = 11.21 dB  ← hai ảnh gốc liên tiếp đã chỉ chênh 11.21 dB
Frame 0.1s vs 0.2s:    PSNR = 10.97 dB
```

Điều này có nghĩa là bản thân hai frame ground truth liên tiếp (0.0s và 0.1s) đã khác nhau đến mức PSNR chỉ ~11dB. Tức là **cả hai keyframe `0.0s` và `0.5s` đều cách `0.1s` khoảng 11dB** → Model sinh ra frame `0.1s` tốt nhất cũng chỉ đạt khoảng 11-12dB so với ground truth.

Kết quả thực tế của `tps-fix` đạt **11.44 dB PSNR / 0.688 SSIM** tại `0.1s.png` so với lineart thực tế → **Đây là kết quả hoàn toàn bình thường và ổn cho một model chưa được fine-tune**.

#### Lý do 2: Domain Gap giữa ml240data và ChayGo

| | ml240data (Train) | ChayGo (Test hiện tại) |
|:---|:---:|:---:|
| Nguồn gốc | 3D render (Mixamo) | 2D anime thực tế |
| FPS sampling | 30-60fps (Δt ≈ 0.03s) | 10fps (Δt = 0.1s) |
| Displacement trung bình | 3.84px | 13.9px (sau RANSAC) |
| Displacement cực đại | 19.59px | 178.89px |
| Khoảng cách I0→I1 trong test | 0.1s (10 frame) | 0.5s (5 sample frames) |

Model được thiết kế và train với displacement <20px, nhưng đang được test với displacement ~14px sau RANSAC, khoảng cách 0.5s → hoàn toàn **vượt quá design envelope của model**.

#### Lý do 3: PSNR 10-11 dB là kết quả bình thường, không phải lỗi

So sánh với ml240data frames liên tiếp (PSNR = 15-18dB giữa frame N và N+1), thì 10-11dB trên ChayGo phản ánh việc:
- Model đang nội suy qua khoảng cách lớn gấp 5-10 lần design envelope.
- Tuy nhiên **kết quả không phải rác hoàn toàn** (không phải 5-6dB), chứng tỏ các fix đã có tác dụng.

### Kết luận về hiểu nhầm:

> [!NOTE]
> **Bạn đang đúng khi đặt ra câu hỏi, nhưng đang nhìn vào sai tiêu chí.** Việc model hiện tại tạo ra ảnh không giống ground truth lineart tại `0.1s` khi được cho input `0.0s` và `0.5s` là **HOÀN TOÀN BÌNH THƯỜNG** vì:
> 1. Khoảng cách 0.5s là quá lớn so với thiết kế gốc của model.
> 2. Domain gap giữa Mixamo 3D và anime 2D là rất lớn.
> 3. Chưa fine-tune.
>
> Sau khi fine-tune đúng cách (bước nhảy 2-4 frame, pipeline line art đúng), model mới có khả năng sinh ra ảnh sát với ground truth lineart thực tế.

---

## 4. Tóm tắt Đánh giá & Đề xuất Hành động

| Vấn đề | Đánh giá | Khuyến nghị |
|:---|:---:|:---|
| Phương hướng dùng anime làm dataset | **✅ Đúng** | Tiếp tục, nhưng kiểm soát $\Delta k$ |
| Dùng NoGay (binary 1-bit) | **❌ Sai hoàn toàn** | Chỉ dùng NoBG với anti-aliasing |
| Khoảng cách keyframe 0.5s | **⚠️ Quá lớn cho model hiện tại** | Dùng $\Delta t \leq 0.2s$ (Δk 2~4 frame) |
| Kỳ vọng model hiện tại cho kết quả tốt trước fine-tune | **❌ Hiểu nhầm** | Kết quả 10-11dB là bình thường trước fine-tune |
| PSNR ~10-11dB trước fine-tune | **✅ Bình thường** | Sau fine-tune kỳ vọng đạt 15-20dB với Δt nhỏ |
| Pipeline Extract → Smooth → Train | **✅ Đúng hướng** | Chuẩn hóa thêm stroke width |

### Thứ tự ưu tiên khi bắt đầu fine-tune:
1. **Thu thập dữ liệu**: Lấy 5-10 đoạn cắt anime 24fps sạch nét, mỗi đoạn 5-30s.
2. **Trích xuất frame**: ffmpeg → PNG sequence (~120-720 frame/đoạn).
3. **Pipeline line art**: Anime-segmentation → AniLines detail → smooth_lines.py (KHÔNG binary hóa).
4. **Tạo triplet**: Chọn $I_0 =$ frame_k, $I_1 =$ frame_{k+2~4}, $I_{GT} =$ frame_{k+1} hoặc frame_{k+2} (tương ứng $t$).
5. **Pre-compute matches**: GlueStick + RANSAC → lưu `.npy`.
6. **Fine-tune**: Đóng băng GlueStick weights, train IFNet + UNet với $\mathcal{L}_1 + \mathcal{L}_{EDT} + \mathcal{L}_{VGG}$.
