# TỔNG HỢP KIẾN THỨC HUẤN LUYỆN MÔ HÌNH AI (AI MODEL TRAINING NOTES)
## Trích xuất từ các phiên thảo luận kỹ thuật của nhóm dự án PBL6

> **Tài liệu được tổng hợp và chuẩn hóa từ nội dung thảo luận kỹ thuật (3 file ghi âm cuộc họp)**  
> **Chủ đề trọng tâm:** Tiền xử lý & Tăng cường dữ liệu (Data Preprocessing & Augmentation), Kiến trúc mạng nơ-ron tích chập (CNN Architecture), Kiểm soát Overfitting, Mô hình toán tử Fourier (FNO) & Inductive Bias, Cơ chế Nội suy khung hình hoạt họa (Animation Inbetweening) và Thiết kế Hàm mất mát (Loss Function).

---

## MỤC LỤC
1. [Tổng quan Đề tài & Bản chất Bài toán](#1-tổng-quan-đề-tài--bản-chất-bài-toán)
2. [Tiền xử lý Dữ liệu & Biểu diễn Kênh Ảnh (Data Representation & Channels)](#2-tiền-xử-lý-dữ-liệu--biểu-diễn-kênh-ảnh-data-representation--channels)
3. [Chiến lược Tăng cường Dữ liệu theo Cặp (Paired Data Augmentation)](#3-chiến-lược-tăng-cường-dữ-liệu-theo-cặp-paired-data-augmentation)
4. [Kiến trúc Mạng Tích chập (CNN) & Kiểm soát Overfitting](#4-kiến-trúc-mạng-tích-chập-cnn--kiểm-soát-overfitting)
5. [Miền Tần số, Chuỗi Fourier & Thiên kiến Quy nạp (Inductive Bias)](#5-miền-tần-số-chuỗi-fourier--thiên-kiến-quy-nạp-inductive-bias)
6. [Thiết kế Hàm Mất mát (Loss Function) cho Nét vẽ Hoạt hình](#6-thiết-kế-hàm-mất-mát-loss-function-cho-nét-vẽ-hoạt-hình)
7. [Ma trận Khuyến nghị Thực thi cho Pipeline Huấn luyện PBL6](#7-ma-trận-khuyến-nghị-thực-thi-cho-pipeline-huấn-luyện-pbl6)

---

## 1. TỔNG QUAN ĐỀ TÀI & BẢN CHẤT BÀI TOÁN

Dựa trên phân tích kỹ thuật từ cuộc hội thoại, nhóm đã làm rõ bản chất cốt lõi của bài toán nhằm tránh nhầm lẫn về cấu trúc I/O (Input/Output) của mô hình:

```
[Frame t = 0 (I_0)] ──┐
                      ├──► [MÔ HÌNH NỘI SUY AI] ──► [Frame trung gian t = 0.5 (I_inbetween)]
[Frame t = 1 (I_1)] ──┘
```

* **Xác định đúng bài toán:**
  * **KHÔNG PHẢI:** Bài toán dự đoán từng nét vẽ tiếp theo (stroke-by-stroke generation).
  * **KHÔNG PHẢI:** Bài toán ngoại suy tương lai từ một khung hình duy nhất (Future Frame Extrapolation: $I_0 \rightarrow I_1$).
  * **CHÍNH XÁC LÀ:** Bài toán **Nội suy khung hình hoạt họa (Animation Frame Interpolation / Inbetweening)**.
* **Cơ chế Input / Output:**
  * **Đầu vào (Input):** Nhận đồng thời **2 khung hình mốc (Keyframes)**: Khung hình xuất phát $I_0$ (tại thời điểm $t_0$) và khung hình kết thúc $I_1$ (tại thời điểm $t_1$).
  * **Đầu ra (Output):** Dự đoán và sinh ra toàn bộ **hình ảnh hoàn chỉnh ở thời điểm giữa** ($I_t$, ví dụ tại $t = 0.5$).
* **Phân biệt Nội suy (Interpolation) và Ngoại suy (Extrapolation):**
  * **Nội suy (Interpolation):** Cho hai điểm dữ liệu đã biết $(x_0, x_1)$, ước lượng/dự đoán các giá trị nằm bên trong khoảng giữa hai điểm đó ($x_0 < x_t < x_1$). Trong thị giác máy tính, đây là nội suy chuyển động (motion inbetweening).
  * **Ngoại suy (Extrapolation):** Dự đoán các điểm nằm ngoài phạm vi quan sát ($x > x_1$ hoặc $x < x_0$), ví dụ dự đoán chuỗi hành động tương lai chưa diễn ra.

---

## 2. TIỀN XỬ LÝ DỮ LIỆU & BIỂU DIỄN KÊNH ẢNH (DATA REPRESENTATION & CHANNELS)

### 2.1. Lựa chọn Định dạng Ảnh: Nhị phân (Binary) vs. Mức xám (Grayscale) vs. Ảnh màu (RGB)
* **Ảnh nhị phân (Binary / Black-White - 1 bit/pixel):**
  * *Nhược điểm:* Đòi hỏi phải đặt một ngưỡng (threshold). Nếu đặt ngưỡng không chuẩn sẽ gây mất nét mảnh (thin lines) hoặc đứt gãy nét vẽ. Mất hoàn toàn hiệu ứng làm mượt cạnh (anti-aliasing).
* **Ảnh mức xám (Grayscale - $H \times W \times 1$):**
  * *Ưu điểm:* Là định dạng tối ưu cho line art anime. Giữ nguyên được sắc độ mềm của nét vẽ, biên độ gradient và tính liên tục của nét viền mà không cần chọn ngưỡng cứng.
* **Ảnh màu (RGB - $H \times W \times 3$):**
  * Nhiều bộ dataset hoạt hình hoặc kết quả trích xuất giữ nguyên định dạng 3 kênh dù cả 3 kênh đều mang giá trị như nhau (ảnh xám giả lập RGB).

### 2.2. Tác động của Số chiều Kênh (1 kênh vs 3 kênh) tới Huấn luyện
Khi xử lý ảnh nét vẽ đơn sắc nhưng để nguyên 3 kênh RGB:
1. **Chi phí tính toán tăng gấp 3 lần:** Ở tầng tích chập đầu tiên, kernel phải có kích thước $K \times K \times 3$ thay vì $K \times K \times 1$. Số lượng phép nhân chập (FLOPs) và lưu lượng bộ nhớ VRAM tăng tương ứng.
2. **Kích thước mô hình phình to:** Tăng số lượng tham số không cần thiết ở lớp trích xuất đặc trưng ban đầu.
3. **Giải pháp bắt buộc:** Trong bước tiền xử lý (preprocessing pipeline), phải chuẩn hóa và chuyển đổi tất cả ảnh về **1 kênh duy nhất ($H \times W \times 1$)** trước khi nạp vào DataLoader.

### 2.3. Cân nhắc Độ phân giải Không gian (Resolution Trade-offs)
* Các mốc kích thước được thảo luận: $224 \times 224$, $960 \times 540$ (chuẩn 540p), $1024 \times 1024$.
* **Đánh giá:**
  * Kích thước 540p ($960 \times 540$) là mức phân giải trung bình trong thị giác máy tính, đảm bảo giữ được độ mảnh của nét anime.
  * Tuy nhiên, nếu kích thước mô hình nhỏ và tập dữ liệu huấn luyện còn hạn chế (small model, small dataset), việc nạp ảnh độ phân giải lớn trực tiếp sẽ làm bùng nổ dung lượng VRAM và làm chậm tốc độ hội tụ. Cần crop theo patch hoặc resize về kích thước cân bằng (ví dụ $512 \times 512$ hoặc $256 \times 256$ trong giai đoạn tiền huấn luyện).

---

## 3. CHIẾN LƯỢC TĂNG CƯỜNG DỮ LIỆU THEO CẶP (PAIRED DATA AUGMENTATION)

Quá trình huấn luyện mô hình nội suy đòi hỏi kỹ thuật tăng cường dữ liệu đặc thù để không làm sai lệch bản chất chuyển động:

### 3.1. Các kỹ thuật tăng cường cơ bản
* **Random Rotation (Xoay góc nhỏ):** Xoay ngẫu nhiên một vài độ (ví dụ $\pm 3^\circ \sim 5^\circ$) để mô hình bất biến với góc nghiêng của nét vẽ.
* **Resize:** Thay đổi tỷ lệ kích thước để mô hình nhận diện nét vẽ ở các cự ly gần/xa.
* **Thêm nhiễu (Add Noise / Gaussian Noise):** Bổ sung nhiễu ngẫu nhiên mức thấp giúp tăng độ bền vững (robustness) của bộ trích xuất đặc trưng.

### 3.2. Quy tắc cốt lõi: Lật đồng bộ theo cặp (Joint / Paired Flip)
> [!IMPORTANT]
> **Nguyên tắc bất di bất dịch đối với bài toán Inbetweening:**
> * Nếu áp dụng phép lật (Horizontal Flip / Flip-side), **BẮT BUỘC phải lật đồng thời cả hai khung hình đầu vào ($I_0, I_1$) và khung hình nhãn ($I_{GT}$)**.
> * *Lý do:* Nếu chỉ lật một khung hình hoặc lật không đồng bộ, mối quan hệ vector chuyển động giữa $t=0$ và $t=1$ sẽ bị đảo lộn hoàn toàn, khiến mô hình học sai quy luật chuyển động vật lý.

```
ĐÚNG:   I_0 [Lật trái]  ──►  I_GT [Lật trái]  ◄──  I_1 [Lật trái]   (Chuyển động nhất quán)
SAI:    I_0 [Không lật] ──►  I_GT [Lật]       ◄──  I_1 [Không lật]  (Phá vỡ chuyển động)
```

* **Cơ chế xác suất (Probability-based Flip):** 
  * Không lật cố định trên toàn bộ dataset mà áp dụng ngẫu nhiên theo tỷ lệ xác suất (ví dụ: $p = 0.5$).
  * Khi một mẫu được chọn để lật, toàn bộ tuple $(I_0, I_t, I_1)$ cùng trải qua một ma trận biến đổi hình học giống hệt nhau.
* **Bất biến đối với hướng chữ:** Do miền dữ liệu là hình vẽ nhân vật hoạt họa (character animation) chứ không phải nhận diện văn bản (OCR), phép lật ngang không làm hỏng tính chất ngữ nghĩa của hình ảnh.

---

## 4. KIẾN TRÚC MẠNG TÍCH CHẬP (CNN) & KIỂM SOÁT OVERFITTING

### 4.1. Cơ chế Không gian (Spatial) và Kênh đặc trưng (Channels)
Mô phỏng quá trình truyền xuôi qua một khối tích chập tiêu chuẩn:
* **Đầu vào:** Giả sử ảnh có kích thước $1024 \times 1024 \times C_{in}$.
* **Qua lớp tích chập & Downsampling (Strided Conv / Pooling):**
  * Kích thước không gian giảm: $1024 \times 1024 \longrightarrow 512 \times 512$.
  * Chiều sâu kênh (Channel dimension / Feature maps) mở rộng: $C_{in} \longrightarrow 16 \longrightarrow 32 \longrightarrow 64$.

```
[Input: 1024 x 1024 x 3]
       │
       ▼ Conv + Downsample
[Feature Map: 512 x 512 x 16]   ──► Tầng nông: Cạnh dọc, cạnh ngang, viền đơn giản
       │
       ▼ Conv
[Feature Map: 512 x 512 x 32]   ──► Tầng giữa: Kết cấu cục bộ, góc, đường cong
       │
       ▼ Conv
[Feature Map: 512 x 512 x 64]   ──► Tầng sâu: Cấu trúc bộ phận trừu tượng
```

### 4.2. Ý nghĩa Biểu diễn của các Kênh (Abstract Features)
* Trong mạng tích chập, chiều sâu kênh không còn đại diện cho kênh màu sắc vật lý (RGB) mà đại diện cho **các bộ lọc đặc trưng trừu tượng (Feature Filters)**:
  * Kênh 1: Chuyên phát hiện các cạnh thẳng đứng (vertical edges).
  * Kênh 2: Chuyên phát hiện các cạnh nằm ngang (horizontal edges).
  * Các kênh tiếp theo: Phát hiện đường cong, góc nhọn, độ tương phản cục bộ, và các hình thái học phức tạp hơn.

### 4.3. Quan hệ giữa Dung lượng Kênh (Channel Capacity) và Overfitting
* **Bản chất của Overfitting ("Học vẹt"):**
  * Khi số lượng kênh được cấu hình quá lớn (ví dụ 64, 128, 256 ở các block sớm), số lượng tham số (weights) của mô hình tăng vọt.
  * Mô hình có xu hướng **ghi nhớ chi tiết vụn vặt và nhiễu** của tập dữ liệu huấn luyện thay vì khái quát hóa các quy luật biến dạng hình học.
* **Biện pháp cân bằng kiến trúc (Architecture Regularization):**
  * **Giữ nguyên độ phân giải không gian:** Đảm bảo đủ số pixel để biểu diễn nét vẽ sắc mảnh.
  * **Chủ động giảm số lượng kênh (Feature Channels):** Cắt giảm từ $64 \rightarrow 32$ hoặc $16$ ở các tầng đầu. Việc giới hạn số kênh đóng vai trò như một cơ chế "nghẽn cổ chai" (information bottleneck), ép mô hình chỉ được phép học các đặc trưng cốt lõi quan trọng nhất, hạn chế tối đa nguy cơ overfitting khi huấn luyện trên tập dữ liệu nhỏ.

---

## 5. MIỀN TẦN SỐ, CHUỖI FOURIER & THIÊN KIẾN QUY NẠP (INDUCTIVE BIAS)

### 5.1. Giới hạn của Mạng Nơ-ron Cổ điển (MLP, Transformer)
* Các kiến trúc như Đa tầng nơ-ron (MLP) hoặc Transformer học bằng cách xấp xỉ phân phối trong không gian đặc trưng rời rạc.
* **Khả năng:** Nội suy rất tốt bên trong không gian manifold của tập dữ liệu huấn luyện (in-distribution interpolation).
* **Điểm yếu:** Khả năng **ngoại suy (out-of-distribution extrapolation)** rất kém khi gặp các mẫu dữ liệu hoặc chuyển động nằm ngoài biên của dữ liệu huấn luyện.

### 5.2. Bản chất Miền Tần số trong Dữ liệu Hình ảnh
* Hình ảnh không chỉ là ma trận cường độ điểm ảnh rời rạc trong miền không gian (Spatial Domain), mà còn có thể biểu diễn tương đương trong **Miền tần số (Frequency Domain)** thông qua biến đổi Fourier (2D Fourier Transform):
  * **Tần số thấp (Low frequencies):** Đại diện cho các vùng biến thiên chậm, mảng màu phẳng, cấu trúc tổng thể và dáng hình nhân vật.
  * **Tần số cao (High frequencies):** Đại diện cho các cạnh viền đột ngột, nét vẽ mảnh, độ sắc nét và chi tiết kết cấu.
* **Tính chất chu kỳ (Periodicity):** Miền tần số có tính chất hàm sóng tuần hoàn, cho phép mô hình nắm bắt được các chu kỳ dao động và chuyển động tuần hoàn của hoạt họa.

### 5.3. Fourier Neural Operator (FNO) & Inductive Bias
* **Thiên kiến quy nạp (Inductive Bias):** Là tập hợp các giả định tiên nghiệm mà kiến trúc mô hình sử dụng để dự đoán đầu ra cho những dữ liệu chưa từng thấy trong quá trình huấn luyện.
* **Fast Fourier Neural Operator (FNO):**
  * Thay vì học ánh xạ giữa các vector rời rạc kích thước cố định như CNN/MLP, FNO học ánh xạ giữa các **không gian hàm liên tục (operator learning between function spaces)**.
  * Việc tích hợp chuỗi Fourier cung cấp cho mạng một **inductive bias mạnh mẽ về tính liên tục và tính tuần hoàn**.
  * Nhờ đó, mô hình cải thiện vượt bậc khả năng ngoại suy (extrapolation) và tổng quát hóa chuyển động nét vẽ trên các miền phân giải khác nhau mà không phụ thuộc cứng vào kích thước lưới điểm ảnh.

---

## 6. THIẾT KẾ HÀM MẤT MÁT (LOSS FUNCTION) CHO NÉT VẼ HOẠT HÌNH

Nhóm thảo luận nhấn mạnh: Các hàm mất mát tiêu chuẩn (như $L_1$, $L_2$, hay MSE thuần túy) không đủ để đảm bảo chất lượng cho ảnh nét vẽ hoạt hình vì chúng thường dẫn đến hiện tượng khung hình nội suy bị mờ nhòe (blurry/ghosting artifacts).

### 6.1. Thành phần Loss chuyên biệt cho Nét vẽ (Stroke-specific Loss)
Để tái tạo hoàn hảo nét vẽ trong hoạt hình 2D/anime, hàm loss cần bổ sung các thành phần giám sát cấu trúc đường nét:

1. **Stroke Thickness / Width Loss (Mất mát Độ dày Nét vẽ):**
   * Giám sát sự biến thiên bề rộng của nét bút.
   * Ngăn chặn hiện tượng nét vẽ bị nở to bất thường (stroke swelling) hoặc co rút, tiêu biến (stroke shrinkage) khi chuyển động giữa 2 keyframe.
2. **Line Continuity & Topology Loss (Mất mát Tính liên tục & Cấu trúc Nét):**
   * Đảm bảo nét viền không bị đứt đoạn hoặc gãy khúc ở các khung hình trung gian.
   * Thường được đo đạc thông qua gradient không gian ($\nabla I$) hoặc perceptual feature maps ở các tầng trích xuất cạnh.

### 6.2. Công thức Hàm Loss Tổng quát Đề xuất
$$\mathcal{L}_{total} = \lambda_{rec} \mathcal{L}_{rec}(I_{pred}, I_{gt}) + \lambda_{stroke} \mathcal{L}_{stroke}(I_{pred}, I_{gt}) + \lambda_{perc} \mathcal{L}_{perceptual}(I_{pred}, I_{gt})$$

* $\mathcal{L}_{rec}$: Reconstruction loss (Charbonnier Loss hoặc $L_1$ mượt) đo sai khác cường độ pixel.
* $\mathcal{L}_{stroke}$: Stroke thickness & edge loss đo sai khác về độ mảnh và gradient của nét vẽ.
* $\mathcal{L}_{perc}$: Perceptual loss (VGG / Deep feature) duy trì cấu trúc cảm nhận thị giác.

---

## 7. MA TRẬN KHUYẾN NGHỊ THỰC THI CHO PIPELINE HUẤN LUYỆN PBL6

Dưới đây là bảng tổng hợp đối chiếu từ các nội dung thảo luận kỹ thuật sang giải pháp lập trình cụ thể cho mã nguồn dự án:

| Hạng mục kỹ thuật | Vấn đề / Hiện tượng thảo luận | Quyết định Kỹ thuật & Triển khai vào Source Code |
| :--- | :--- | :--- |
| **Input Channels** | Ảnh nét vẽ nhưng để 3 kênh RGB làm tốn VRAM và tăng khối lượng tính toán gấp 3 lần. | Viết hàm tiền xử lý `transforms.Grayscale(num_output_channels=1)` trong DataLoader; cấu hình lớp Conv đầu vào `in_channels=1` (hoặc ghép kênh 2 ảnh: `in_channels=2`). |
| **Binarization** | Ảnh đen trắng nhị phân cần ngưỡng cứng, dễ mất nét mảnh. | Giữ nguyên ảnh **Grayscale chuẩn hóa [0, 1]** có anti-aliasing; không dùng `cv2.threshold` nhị phân cứng 0/255. |
| **Data Augmentation** | Lật ảnh (Flip) không đồng bộ giữa 2 khung hình phá vỡ logic chuyển động. | Xây dựng pipeline `PairedTransform`: khi gieo xúc xắc ngẫu nhiên ($p < 0.5$), áp dụng hàm `TF.hflip()` đồng thời cho cả $I_0$, $I_1$ và $I_{GT}$. |
| **Augmentation khác** | Nét vẽ cần đa dạng hóa tư thế và chống nhiễu. | Thêm `RandomRotation(degrees=(-3, 3))` đồng bộ theo cặp, thêm Gaussian noise biên độ nhỏ. |
| **CNN Channels** | Đặt số kênh quá lớn ($64, 128$) dễ dẫn đến học vẹt (overfitting) với tập dữ liệu nhỏ. | Giảm số lượng base channels của feature extractor (ví dụ: `base_channels = 16` hoặc `32`); áp dụng Dropout hoặc Weight Decay. |
| **Mô hình hóa Miền tần số** | Mạng nơ-ron rời rạc ngoại suy kém khi dự đoán chuyển động phức tạp. | Nghiên cứu tích hợp khối Spectral Convolution / FNO layer vào bottleneck để tăng Inductive Bias miền tần số. |
| **Cấu trúc I/O Mô hình** | Nhầm lẫn giữa sinh nét tiếp theo và nội suy khung hình. | Khẳng định kiến trúc là **Two-frame Inbetweening**: Tensor đầu vào có dạng $[B, 2, C, H, W]$ (hoặc nối kênh $[B, 2 \times C, H, W]$), Tensor đầu ra là $[B, C, H, W]$. |
| **Loss Function** | $L_1/L_2$ thuần túy gây nhòe nét ở frame trung gian. | Kết hợp loss nét vẽ (`Stroke Loss` dựa trên Sobel/Laplacian gradient) cùng với `Charbonnier Loss` để giữ nét sắc cạnh. |

---
*Tài liệu được lưu trữ tại `docs/ai_model_training_knowledge_notes.md` phục vụ công tác tra cứu kiến trúc, tối ưu code huấn luyện và viết báo cáo đề tài PBL6.*
