# CHIẾN LƯỢC XÂY DỰNG DATASET ANIME 2D & PHÂN TÍCH KỸ THUẬT FINE-TUNE TPS-INBETWEEN
## (Mô Hình Hybrid: Tiền Xử Lý Cục Bộ + Deep Learning & Huấn Luyện Trên Kaggle)

> **Tài liệu tham chiếu:** Phân tích khả thi kỹ thuật, kiến trúc dữ liệu và giải pháp tối ưu tài nguyên khi:
> 1. **Xử lý video thô (12-16 giờ) trên máy tính cá nhân (Local PC)** để tạo các Candidate Triplets sạch.
> 2. **Gom chung với tập con Anime của ATD-12K** (`Japan_*` và `Hayao_*`).
> 3. **Đưa lên Kaggle** để thực hiện các công đoạn Deep Learning nặng (GlueStick RANSAC, AniLines Lineart, Segmentation) và tiến hành **Fine-tune mô hình TPS-Inbetween**.

---

## MỤC LỤC
1. [Phân Tích Phương Án Phân Chia: Xử Lý Video Cục Bộ & Đưa Lên Kaggle](#1-phân-tích-phương-án-phân-chia-xử-lý-video-cục-bộ--đưa-lên-kaggle)
   - 1.1. Tại sao chuyển khâu xử lý video về Local là quyết định đúng đắn 100%?
   - 1.2. Giải quyết triệt để rào cản đĩa cứng 19.5 GB của Kaggle
   - 1.3. Tiết kiệm tối đa ngân sách 30 giờ GPU/tuần trên Kaggle
   - 1.4. Thiết kế Pipeline 2 tầng (Two-Tier Architecture)
2. [Cấu Trúc Dataset Kết Hợp: ATD-12K Anime + Video Đã Tuyển Chọn](#2-cấu-trúc-dataset-kết-hợp-atd-12k-anime--video-đã-tuyển-chọn)
   - 2.1. Nguồn ATD-12K Anime có sẵn (4,881 Triplet train + 759 Triplet test)
   - 2.2. Nguồn 3-4 bộ Anime xử lý cục bộ (12-16 tiếng video)
   - 2.3. Bảng ma trận tỷ lệ ghép nối và phân bổ vai trò
3. [Phân Tích Yêu Cầu Của Một Dataset Chuẩn Fine-tune Line-Art Animation](#3-phân-tích-yêu-cầu-của-một-dataset-chuẩn-fine-tune-line-art-animation)
   - 3.1. Phân bố bước nhảy thời gian ngẫu nhiên ($t$-distribution)
   - 3.2. Đa dạng hóa độ dày nét (Stroke Thickness Augmentation)
   - 3.3. Khử ranh giới bóng đổ nhân vật (Shadow Boundary Suppression)
   - 3.4. Kiểm soát biến đổi tô pô & che khuất (Occlusion Handling)
   - 3.5. Độ phủ không gian của điểm so khớp (Spatial Uniformity of Inliers)
4. [Đánh Giá Kỹ Thuật: Tính Đúng Đắn Khi Dùng Dataset Này Cho TPS-Inbetween](#4-đánh-giá-kỹ-thuật-tính-đúng-đắn-khi-dùng-dataset-này-cho-tps-inbetween)
   - 4.1. Sự tương thích về mặt kiến trúc lý thuyết
   - 4.2. Khắc phục khoảng cách miền (Domain Gap) của mô hình gốc Mixamo 3D
5. [Các Rủi Ro Kỹ Thuật Khi Fine-tune & Giải Pháp Phòng Chống](#5-các-rủi-ro-kỹ-thuật-khi-fine-tune--giải-pháp-phòng-chống)
   - 5.1. Rủi ro 1: Lệch pha phân phối $t$ (Time-step Collapse)
   - 5.2. Rủi ro 2: TPS Warping Foldover & Kỳ dị ma trận giải tích (Singular Matrix)
   - 5.3. Rủi ro 3: Hiện tượng Gradient Vanishing & Mờ nét do L1 Loss thuần túy
   - 5.4. Rủi ro 4: Nhiễu đường viền do trích xuất tự động (Spurious Edge Artifacts)
   - 5.5. Rủi ro 5: Quản lý hạn mức thời gian Kaggle (Bảng phân bổ ngân sách 30 giờ)
6. [Kế Hoạch Hành Động Chi Tiết 2 Pha (Local $\to$ Kaggle)](#6-kế-hoạch-hành-động-chi-tiết-2-pha-local--kaggle)
7. [Phân Tích Chuyên Sâu: Điểm Mạnh & Yếu Của Ý Kiến "Tự Tạo 100% Dataset Từ 25 Giờ Video Anime Thực Tế"](#7-phân-tích-chuyên-sâu-điểm-mạnh--yếu-của-ý-kiến-tự-tạo-100-dataset-từ-25-giờ-video-anime-thực-tế)
   - 7.1. Bối cảnh & Bản chất của ý kiến
   - 7.2. Phân tích điểm mạnh (Ưu điểm vượt trội)
   - 7.3. Phân tích điểm yếu (Thách thức kỹ thuật & Gánh nặng tài nguyên)
   - 7.4. Bảng so sánh đối đầu trực diện: Tự làm 100% vs Phương án Hybrid
   - 7.5. Đánh giá tổng kết & Lời khuyên chiến lược cho đồ án

---

## 1. PHÂN TÍCH PHƯƠNG ÁN PHÂN CHIA: XỬ LÝ VIDEO CỤC BỘ & ĐƯA LÊN KAGGLE

### 1.1. Tại sao chuyển khâu xử lý video về Local là quyết định đúng đắn 100%?

Trong phương án ban đầu (đưa video raw 16 tiếng lên Kaggle), hệ thống gặp phải **nghịch lý kép**: vừa nghẽn đĩa cứng vừa lãng phí GPU. Việc chuyển toàn bộ các công đoạn xử lý video thô về **máy tính cá nhân (Local PC)** mang lại 3 ưu thế vượt trội:

| Tiêu chí so sánh | Đưa Video Raw lên Kaggle | Tiền Xử Lý Video Cục Bộ Rồi Mới Đưa Lên |
|:---|:---|:---|
| **Giới hạn thời gian chạy** | Bị chặn cứng ở 12h/phiên và 30h/tuần | **Chạy không giới hạn** (chạy nền, treo máy qua đêm thoải mái) |
| **Tài nguyên sử dụng** | Tiêu tốn giờ GPU đắt đỏ của Kaggle vào việc giải mã MP4 | Dùng CPU đa nhân và GPU nội bộ (RTX 3050 Laptop) hoàn toàn miễn phí |
| **Dung lượng upload** | Phải upload hàng chục GB file video MP4 thô | **Chỉ upload file zip chứa Candidate Triplets đã lọc sạch (~4 - 6 GB)** |
| **Dung lượng đĩa đệm Kaggle** | Sập đĩa tạm 19.5 GB ngay khi bung 1.38 triệu frames | **Chỉ tốn ~6 GB**, hoàn toàn giải phóng đĩa `/kaggle/working` |
| **Ngân sách GPU Kaggle còn lại** | Chỉ còn ~20 giờ cho việc huấn luyện mô hình | **Dành trọn vẹn 25 - 26 giờ GPU chuyên cho việc Fine-tune!** |

---

### 1.2. Giải quyết triệt để rào cản đĩa cứng 19.5 GB của Kaggle

#### a. Phép tính khối lượng dữ liệu sau khi lọc cục bộ:
- 16 giờ video raw ở 24fps có **1,382,400 frames thô**.
- Khi xử lý trên máy tính cục bộ:
  1. Dùng `PySceneDetect` cắt cảnh $\rightarrow$ Tự động loại bỏ **75% cảnh tĩnh** (đối thoại mấp máy môi, phong cảnh đứng yên, credit).
  2. Áp dụng thuật toán **Micro-Dedup (Sobel Edge Diff + PSNR)** $\rightarrow$ Bỏ qua các frame trùng nét vẽ do nhịp giữ hình on-twos/on-threes.
  3. Áp dụng **Step-based Mining** ($\Delta F = 2 \sim 6$) $\rightarrow$ Thu hoạch được khoảng **3,000 – 4,000 Candidate Triplets** hành động chất lượng cao.
- **Tổng số ảnh sau khi lọc:**
  $$\text{Số lượng} \approx 3,500 \text{ triplets} \times 3 \text{ frames} = 10,500 \text{ ảnh màu RGB}$$
- Khi resize ảnh về chuẩn 540p ($960 \times 540$) và lưu dạng JPG chất lượng 95:
  $$\text{Dung lượng upload nguồn bổ sung} \approx 10,500 \times 180 \text{ KB} \approx \mathbf{1.89\text{ GB}!}$$
- Kết hợp với phần Anime của ATD-12K (`Japan_*` + `Hayao_*` gồm 4,881 triplets $\approx 14,643$ ảnh 540p):
  $$\text{Dung lượng upload ATD-12K Anime} \approx 14,643 \times 180 \text{ KB} \approx \mathbf{2.63\text{ GB}!}$$
- **TỔNG DUNG LƯỢNG UPLOAD LÊN KAGGLE:** **Chỉ khoảng 4.5 – 5.5 GB**!  
  *(Đóng gói thành 1 file ZIP duy nhất đưa vào `Kaggle Input Dataset`, hoàn toàn không chiếm dụng ổ đĩa làm việc `/kaggle/working`)*.

---

### 1.3. Tiết kiệm tối đa ngân sách 30 giờ GPU/tuần trên Kaggle

Khi đưa ảnh màu Candidate Triplets sạch lên Kaggle, tài nguyên GPU máy chủ (2x Tesla T4 16GB VRAM) chỉ tập trung vào đúng 2 công việc đòi hỏi sức mạnh mạng nơ-ron:
1. **GlueStick Keypoint Matching & RANSAC Screening (Stage 4):**
   - Chạy trên 2 GPU song song (`torch.nn.DataParallel`).
   - Tốc độ xử lý: ~8 triplet/giây $\implies$ 8,000 triplets mất khoảng **15 - 20 phút**!
2. **Line-art Extraction & Segmentation (Stage 5):**
   - Chạy mạng tách nền ISNetDIS và AniLines Detail (batch size = 6 trên 16GB VRAM).
   - Tốc độ xử lý: ~12 frame/giây $\implies$ 24,000 frame mất khoảng **30 - 40 phút**!
- **Tổng thời gian tạo dataset hoàn chỉnh trên Kaggle:** **Chỉ mất chưa đầy 1.5 – 2.0 giờ GPU!**
- **Hiệu quả:** Bạn còn nguyên vẹn **hơn 28 giờ GPU Kaggle** chỉ để dành cho việc Fine-tune mô hình (thoải mái chạy 50 - 70 epoch, chia nhỏ nhiều lần lưu checkpoint an toàn).

---

### 1.4. Thiết kế Pipeline 2 tầng (Two-Tier Architecture)

```mermaid
flowchart TD
    subgraph TẦNG 1: XỬ LÝ CỤC BỘ (LOCAL PC - KHÔNG GIỚI HẠN THỜI GIAN)
        A1[12-16 Giờ Video Anime Raw MP4] -->|PySceneDetect| A2[Phát hiện cắt cảnh & Bỏ cảnh tĩnh]
        A2 -->|Micro-Dedup Sobel+PSNR| A3[Lọc frame đứng hình]
        A3 -->|Step-based Triplet Mining| A4[Khai thác Triplet đa mốc t: 0.25, 0.33, 0.5, 0.67, 0.75]
        A4 --> A5[Tập Candidate Triplets Mở Rộng: ~3,500 triplets]
        
        B1[Ổ cứng E:\Dataset\datasets] -->|Lọc tiền tố Japan_* và Hayao_*| B2[Tập ATD-12K Anime: 4,881 triplets]
        
        A5 --> C[Gom chung & Đóng gói thành raw_candidate_triplets.zip ~5GB]
        B2 --> C
    end

    subgraph TẦNG 2: DEEP LEARNING & TRAINING TRÊN KAGGLE (GPU T4 x2)
        C -->|Upload lên Kaggle Input Dataset| D[Kaggle Dataset: anime_candidates_540p]
        
        D -->|Notebook 1: Chạy ~1.5 - 2.0 giờ GPU| E[Pipeline Stage 4 & 5 trên GPU T4]
        subgraph Pipeline Deep Learning
            E1[GlueStick Matcher + RANSAC Filter] -->|Sinh file .npy| E2[Lưới tọa độ điểm neo]
            E3[ISNetDIS Tách Nền + AniLines Detail] -->|Sinh ảnh lineart| E4[Ảnh nét mịn _smooth.png]
        end
        E --> E1
        E --> E3
        
        E2 --> F[Dataset Hoàn Thiện: 6,000+ Triplets chuẩn TPS]
        E4 --> F
        
        F -->|Notebook 2: Chạy ~20 - 25 giờ GPU| G[Fine-tune TPS-Inbetween IFNet + UNet]
        G --> H[Checkpoint: model_anime_finetuned.pt]
    end
```

---

## 2. CẤU TRÚC DATASET KẾT HỢP: ATD-12K ANIME + VIDEO ĐÃ TUYỂN CHỌN

### 2.1. Nguồn ATD-12K Anime có sẵn (4,881 Triplet train + 759 Triplet test)

- **Vị trí hiện tại:** Nằm trong thư mục `E:\Dataset\datasets\train_10k`.
- **Thành phần:**
  - `Japan_*`: 4,543 triplets từ các series anime truyền hình kinh điển.
  - `Hayao_*`: 338 triplets từ các tuyệt phẩm chiếu rạp của Studio Ghibli.
  - *(Đã loại bỏ hoàn toàn 5,119 triplets `Disney_*` không phù hợp)*.
- **Đặc điểm:** Tác giả ATD-12K đã cắt sẵn thành từng triplet 3 frame (`frame1.jpg`, `frame2.jpg`, `frame3.jpg`) với $t=0.5$.
- **Vai trò:** Cung cấp **sự đa dạng phong cách vô địch** (hàng trăm nhân vật, nét vẽ, góc quay khác nhau), bảo đảm mô hình sau khi train có thể tổng quát hóa (generalize) cho bất kỳ bộ anime nào.

---

### 2.2. Nguồn 3-4 bộ Anime xử lý cục bộ (12-16 tiếng video)

- **Tại sao cần bổ sung nguồn này khi đã có ATD-12K?**
  1. **Bổ sung các mốc thời gian tùy ý ($t \ne 0.5$):**  
     ATD-12K 100% là $t=0.5$. Nguồn video bổ sung này được đào sâu theo nhịp step-3 ($\Delta F = 3$) và step-4 ($\Delta F = 4$) để tạo ra các ground truth tại:
     $$t = \frac{1}{3} \approx 0.333, \quad t = \frac{2}{3} \approx 0.667, \quad t = \frac{1}{4} = 0.250, \quad t = \frac{3}{4} = 0.750$$
  2. **Cung cấp chuỗi hành động dài liên tục:** Giúp mô hình học được chuyển động mượt mà qua nhiều nhịp kế tiếp nhau (như demo 7.92s vừa kiểm chứng thành công).
  3. **Cập nhật phong cách anime thế hệ mới:** Bổ sung các kỹ thuật đổ bóng và chi tiết kỹ thuật số của các studio đương đại (như Ufotable, Mappa).

---

### 2.3. Bảng ma trận tỷ lệ ghép nối và phân bổ vai trò

| Thành phần Dataset | Số lượng Triplet mục tiêu | Tỷ lệ trong Dataset | Vai trò kỹ thuật then chốt |
|:---|:---:|:---:|:---|
| **ATD-12K Anime (Đã qua RANSAC)** | **3,500 triplets** | **58%** | Đa dạng hóa hàng trăm phong cách vẽ, chống học vẹt (overfitting) |
| **Anime Mở Rộng ($t = 0.5$)** | **1,000 triplets** | **17%** | Cập nhật phong cách anime hiện đại, chuyển động hành động nhanh |
| **Anime Mở Rộng ($t \ne 0.5$)** | **1,500 triplets** | **25%** | **Khắc phục điểm yếu chí mạng của ATD-12K:** Cung cấp mốc $t \in [0.2, 0.8]$ |
| **TỔNG CỘNG** | **6,000 triplets** | **100%** | **Quy mô vàng đạt chuẩn công nghiệp (tương đương Mixamo 3D gốc)** |

---

## 3. PHÂN TÍCH YÊU CẦU CỦA MỘT DATASET CHUẨN FINE-TUNE LINE-ART ANIMATION

Một dataset đạt chuẩn phục vụ bài toán nội suy nét vẽ 2D **KHÔNG ĐƯỢC CHỈ LÀ TẬP HỢP CÁC ẢNH ĐEN TRẮNG ĐƠN THUẦN**. Để mô hình hội tụ tốt và không bị sụp đổ (collapse), dataset bắt buộc phải đáp ứng 5 tiêu chuẩn bổ sung sau:

```
                              ┌──────────────────────────────────────────┐
                              │  5 TRỤ CỘT CỦA DATASET NÉT VẼ CHUẨN      │
                              └────────────────────┬─────────────────────┘
                                                   │
         ┌───────────────────┬─────────────────────┼─────────────────────┬───────────────────┐
         ▼                   ▼                     ▼                     ▼                   ▼
   1. Cân Bằng t       2. Biến Thiên Nét     3. Triệt Tiêu Bóng    4. Lọc Tô Pô        5. Phân Bố Đều
   (Uniform t-dist)    (Stroke Augment)      (No Shading Edge)     (RANSAC Inliers)    (Spatial Grid)
```

### 3.1. Phân bố bước nhảy thời gian ngẫu nhiên (Uniform $t$-Distribution)
- Trong bài toán tổng quát, người dùng có thể yêu cầu tăng tốc độ khung hình từ 8fps lên 24fps (cần $t=0.33, 0.67$) hoặc lên 60fps (cần $t=0.125, 0.25, 0.375...$).
- Dataset phải đảm bảo hàm mật độ xác suất của $t$ trải đều trên đoạn $[0.2, 0.8]$ thay vì tập trung một đỉnh nhọn duy nhất tại $0.5$.

### 3.2. Đa dạng hóa độ dày nét (Stroke Thickness Augmentation)
- Nét vẽ anime thực tế có độ dày từ 1px (râu, tóc, mi mắt) đến 4-5px (viền áo, cơ thể).
- Nếu chỉ train trên nét trích xuất cố định, mô hình sẽ bị lỗi đứt nét khi gặp nét quá mảnh, hoặc bị nhòe khi gặp nét dày.
- **Kỹ thuật cần bổ sung:** Trong DataLoader, áp dụng toán tử hình thái học ngẫu nhiên:
  - 15% xác suất áp dụng `cv2.dilate` (tăng độ dày nét thêm 1px).
  - 15% xác suất áp dụng `cv2.erode` (làm mảnh nét về dạng hairline).

### 3.3. Khử ranh giới bóng đổ nhân vật (Shadow Boundary Line Suppression)
- Vùng chuyển giao giữa mảng sáng và mảng tối trên khuôn mặt/quần áo (cel-shading border) bản chất **không có nét vẽ của họa sĩ**, nhưng các bộ trích nét thông thường sẽ vẽ ra một đường nét giả.
- Khi nhân vật xoay đầu, đường bóng đổ di chuyển với quy luật khác hẳn chuyển động cơ thể, khiến GlueStick bị đánh lừa và sinh ra flow biến dạng.
- **Tiêu chuẩn:** Phải kết hợp giữa `AniLines Detail` và `Bilateral Filtering` để làm phẳng các mảng chuyển màu trước khi trích nét.

### 3.4. Kiểm soát biến đổi tô pô & che khuất (Occlusion Handling)
- Hoạt họa 2D thường xuyên có hiện tượng: chớp mắt (đường mi mắt biến mất hoàn toàn), quay mặt (tai khuất ra sau đầu), vung tay (cánh tay che khuất thân mình).
- Phép biến dạng Thin-Plate Spline (TPS) giả định quan hệ ánh xạ 1-1 giữa các điểm. Nếu dataset có quá nhiều mẫu bị biến dạng tô pô đột ngột, ma trận TPS sẽ bị xoắn nát.
- **Tiêu chuẩn:** Tỷ lệ inlier sau RANSAC phải đạt tối thiểu $\ge 25$ điểm và tỷ lệ outlier/inlier không vượt quá 30%.

### 3.5. Độ phủ không gian của điểm so khớp (Spatial Uniformity of Inliers)
- Tránh hiện tượng 300 điểm match dồn hết vào đôi mắt và mái tóc dày đặc, còn toàn bộ phần tay chân di chuyển thì không có điểm match nào.
- **Tiêu chuẩn:** Sử dụng thuật toán **Grid-based Keypoint Retention**: Chia khung hình thành lưới $4 \times 4$ ô, mỗi ô bắt buộc phải có ít nhất 2-3 điểm match đóng vai trò neo giữ (anchors).

---

## 4. ĐÁNH GIÁ KỸ THUẬT: TÍNH ĐÚNG ĐẮN KHI DÙNG DATASET NÀY CHO TPS-INBETWEEN

### 4.1. Sự tương thích về mặt kiến trúc lý thuyết

Thiết kế dataset này **HOÀN TOÀN ĐÚNG ĐẮN VỀ MẶT KỸ THUẬT** đối với kiến trúc của TPS-Inbetween:

```
[Khung hình I_0, I_1] 
         │
         ├──► [GlueStick Matcher] ──► Tìm điểm tương đồng P_0, P_1 (Đã được lọc sạch bởi RANSAC)
         │                                       │
         │                                       ▼
         │                            [Thin-Plate Spline Solver]
         │                            Giải hệ phương trình năng lượng cực tiểu
         │                            Xây dựng lưới biến dạng liên tục F_t
         │                                       │
         ▼                                       ▼
[Warped Images I_{0->t}, I_{1->t}] ◄─────────────┘
         │
         ├──► [IFNet - Motion Refinement] ──► Tinh chỉnh flow sub-pixel dọc theo nét vẽ
         │
         └──► [UNet - Inpainting & Blending] ──► Tổng hợp nét, khử đường viền thừa (ghosting)
```

1. **Phù hợp với cơ chế TPS Solver:**  
   Bằng cách sàng lọc bước nhảy chuyển động $\Delta F$ để giữ độ dời trung bình trong khoảng $\mathbf{2.5 \sim 6.0\text{ px}}$, ta đưa dữ liệu vào đúng **"miền giải tích vàng"** của hàm Spline bậc hai $U(r) = r^2 \ln(r)$, loại bỏ hoàn toàn hiện tượng ma trận kỳ dị.
2. **Khắc phục triệt để khoảng cách miền (Domain Gap):**  
   Mô hình gốc được tác giả train trên dataset `MixamoLine240` (nhân vật hoạt hình 3D kết xuất dạng vector đơn giản). Khi áp dụng vào anime 2D Nhật Bản, các quy luật vẽ tay phi vật lý khiến mô hình gốc bị lúng túng. Việc nạp 6,000 triplet anime 2D chuẩn sẽ dạy cho `IFNet` và `UNet` đúng ngôn ngữ hình thể của anime.

---

## 5. CÁC RỦI RO KỸ THUẬT KHI FINE-TUNE & GIẢI PHÁP PHÒNG CHỐNG

*(Bỏ qua các rủi ro huấn luyện thông thường, tập trung vào rủi ro kỹ thuật mô hình và giới hạn Kaggle)*

### 5.1. Rủi ro 1: Lệch pha phân phối $t$ (Time-step Collapse)
- **Cơ chế rủi ro:** Nếu dataset chứa quá nhiều mẫu có $t = 0.5$ (do lấy nguyên bản từ ATD-12K), mạng nơ-ron sẽ tự tối ưu hóa bằng cách "bỏ qua" vector thời gian $t$ ở đầu vào, suy biến thành một mô hình nội suy chia đôi thông thường. Khi đưa vào sử dụng thực tế với $t=0.33$, mô hình vẫn sẽ sinh ra ảnh ở $t=0.5$.
- **Giải pháp kỹ thuật:**
  1. Giới hạn tỷ trọng ATD-12K không quá 60% tổng dataset.
  2. Bổ sung kỹ thuật **Temporal Inversion Augmentation** trong vòng lặp huấn luyện:
     $$\text{Nếu triplet là } (I_0, I_1, I_{GT}, t) \implies \text{Đổi thành } (I_1, I_0, I_{GT}, 1 - t)$$

---

### 5.2. Rủi ro 2: TPS Warping Foldover & Kỳ dị ma trận giải tích (Singular Matrix)
- **Cơ chế rủi ro:** Trong quá trình biến dạng TPS, ma trận kernel $K_{ij} = U(||P_i - P_j||)$ cần được nghịch đảo:
  $$L = \begin{bmatrix} K & P \\ P^T & 0 \end{bmatrix}, \quad W = L^{-1} Y$$
  Nếu có 2 điểm keypoint quá gần nhau ($||P_i - P_j|| < 0.5\text{ px}$) hoặc có các điểm nằm thẳng hàng tuyệt đối, định thức ma trận $\det(L) \to 0$, dẫn đến lỗi sụp số dấu phẩy động (NaN Loss) làm hỏng checkpoint.
- **Giải pháp kỹ thuật:**
  - Trong Stage 4 của Pipeline, cài đặt bộ lọc khoảng cách tối thiểu: Bất kỳ cặp điểm nào có khoảng cách $< 3\text{ px}$ sẽ bị gộp thành một điểm duy nhất.
  - Sử dụng tham số điều chuẩn $\lambda = 0.2$ (Tikhonov regularization) trong TPS solver để đảm bảo ma trận luôn khả nghịch đối xứng dương.

---

### 5.3. Rủi ro 3: Hiện tượng Gradient Vanishing & Mờ nét do L1 Loss thuần túy
- **Cơ chế rủi ro:** Line-art có đặc điểm 95% diện tích là màu trắng, nét vẽ đen chỉ chiếm 5%. Nếu chỉ dùng hàm mất mát pixel L1, mô hình vẽ nét bị lệch chỉ 1 pixel sẽ không có gradient dẫn hướng chỉ ra nét cần dịch chuyển sang trái hay sang phải. Mô hình sẽ chọn giải pháp an toàn là làm mờ nhạt nét vẽ thành màu xám để giảm phạt L1, sinh ra hiện tượng ghosting.
- **Giải pháp kỹ thuật:**
  - Bắt buộc phải tích hợp **EDT Loss (Euclidean Distance Transform Loss)** hoặc **Chamfer Distance Loss**:
    $$L_{EDT} = \frac{1}{|\Omega|} \sum_{p \in \Omega} (1 - I_{pred}(p)) \cdot D_{GT}^2(p)$$
    Trong đó $D_{GT}(p)$ là bản đồ khoảng cách Euclid tới nét vẽ chuẩn gần nhất, tạo ra một "phễu trọng lực" hút nét vẽ về đúng vị trí ground truth.

---

### 5.4. Rủi ro 4: Nhiễu đường viền do trích xuất tự động (Spurious Edge Artifacts)
- **Cơ chế rủi ro:** Khi anime-segmentation cắt nhân vật khỏi nền cảnh, vùng biên cắt thường để lại một đường viền sắc nhọn giả tạo (alpha boundary). Nếu AniLines biến đường viền này thành nét vẽ, mô hình sẽ học cách vẽ lại cả đường viền cắt nền.
- **Giải pháp kỹ thuật:**
  - Áp dụng kỹ thuật **Mask Erosion**: Co nhẹ mặt nạ phân đoạn vào trong 1-2 pixel trước khi trích xuất nét.
  - Sau khi trích xuất, chạy thuật toán lọc thành phần liên thông (Connected Components) để loại bỏ toàn bộ các đốm nét đơn lẻ có diện tích nhỏ hơn 10 pixel.

---

### 5.5. Rủi ro 5: Quản lý hạn mức thời gian Kaggle (Bảng phân bổ ngân sách 30 giờ)

Nhờ đã chuyển khâu giải mã video và lọc frame về máy cục bộ, bảng phân bổ ngân sách 30 giờ GPU/tuần của Kaggle trở nên **cực kỳ dư dả và an toàn**:

| Giai đoạn trên Kaggle | Thời gian GPU dự toán | Nhiệm vụ chi tiết | Tình trạng an toàn |
|:---|:---:|:---|:---:|
| **Stage 4 + Stage 5 (Deep Learning Pipeline)** | **2.0 giờ** | Chạy GlueStick RANSAC & AniLines Detail trên 2x GPU T4 cho ~8,000 triplets | Siêu an toàn (chạy 1 mạch) |
| **Fine-tune Lượt 1 (Epoch 1 - 30)** | **12.0 giờ** | Train IFNet + UNet với AdamW, lưu checkpoint mỗi 5 epoch | Đầy đủ 1 phiên chuẩn |
| **Fine-tune Lượt 2 (Epoch 31 - 60)** | **12.0 giờ** | Tiếp tục từ checkpoint epoch 30, giảm lr với CosineAnnealing | Đầy đủ 1 phiên chuẩn |
| **Đánh giá Test & Export Checkpoint** | **2.0 giờ** | Chạy inference kiểm thử trên tập test và kết xuất video | Dự phòng an toàn |
| **TỔNG CỘNG** | **28.0 / 30.0 giờ** | **Dư 2.0 giờ dự phòng** | **100% Khả thi trong 1 tuần** |

---

## 6. KẾ HOẠCH HÀNH ĐỘNG CHI TIẾT 2 PHA (LOCAL $\to$ KAGGLE)

```
PHA 1: TIỀN XỬ LÝ CỤC BỘ TRÊN MÁY TÍNH CÁ NHÂN (LOCAL PC)
───────────────────────────────────────────────────────────────────────
 Bước 1.1: Quét và sao chép 4,881 thư mục Anime từ E:\Dataset\datasets\train_10k (Japan_* và Hayao_*).
 Bước 1.2: Chạy script PySceneDetect + Micro-Dedup trên 3-4 bộ Anime raw (12-16h) để khai thác
           ~3,500 Candidate Triplets sạch (đa dạng mốc t = 0.25, 0.33, 0.5, 0.67, 0.75).
 Bước 1.3: Resize ảnh màu về chuẩn 540p và đóng gói thành 1 file duy nhất:
           candidate_triplets_anime_540p.zip (~5 GB).
 Bước 1.4: Upload file zip lên tài khoản Kaggle dưới dạng "New Dataset".

PHA 2: DEEP LEARNING PIPELINE & FINE-TUNE TRÊN KAGGLE (GPU T4 x2)
───────────────────────────────────────────────────────────────────────
 Bước 2.1: [Notebook 1 - Tạo Dataset Line-art Chuẩn] (Chạy ~2.0 giờ GPU)
           - Mount Dataset `candidate_triplets_anime_540p`.
           - Chạy Stage 4: GlueStick + RANSAC Filter $\to$ lưu file `.npy`.
           - Chạy Stage 5: ISNetDIS + AniLines Detail + Bilateral Smooth $\to$ lưu file `_smooth.png`.
           - Xuất kết quả ra `anime_finetune_dataset_ready.tar.gz`.

 Bước 2.2: [Notebook 2 - Huấn Luyện Fine-tune TPS-Inbetween] (Chạy ~24.0 giờ GPU)
           - Đóng băng nhánh GlueStick, chỉ mở gradient cho IFNet và UNet.
           - Huấn luyện 60 epoch với hàm mất mát tổng hợp: $L_{total} = L_1 + 0.1 L_{VGG} + 0.5 L_{EDT}$.
           - Áp dụng Temporal Inversion Augmentation để cân bằng phân phối $t$.
           - Lưu checkpoint tốt nhất: `model_anime_latest.pt`.

 Bước 2.3: [Kiểm thử & Đánh giá Cục Bộ]
           - Tải `model_anime_latest.pt` về máy tính.
           - Đưa vào thư mục `Demo_ATD12K_TPS_Inbetween` chạy thử trên chuỗi cắt cảnh 7.92s để đối chứng chất lượng.
```

---

## 7. PHÂN TÍCH CHUYÊN SÂU: ĐIỂM MẠNH & YẾU CỦA Ý KIẾN "TỰ TẠO 100% DATASET TỪ 25 GIỜ VIDEO ANIME THỰC TẾ"

### 7.1. Bối cảnh & Bản chất của ý kiến

Ý kiến đề xuất: **Không sử dụng bất kỳ dữ liệu nào của ATD-12K**. Thay vào đó, tự thu thập khoảng **25 giờ video Anime thực tế** (tương đương 50 – 60 tập anime truyền hình hoặc 12 – 15 phim hoạt họa chiếu rạp), sau đó áp dụng toàn bộ pipeline (SceneDetect $\to$ Micro-Dedup $\to$ Triplet Mining $\to$ GlueStick RANSAC $\to$ AniLines Lineart) để tự tạo ra một bộ Dataset 100% "made by user" với quy mô tương đương với cách nhóm tác giả quốc tế đã tạo ra tập dữ liệu ATD-12K ban đầu.

---

### 7.2. Phân Tích Điểm Mạnh (Ưu điểm vượt trội)

Nếu triển khai thành công, phương án này mang lại 5 lợi ích kỹ thuật rất lớn:

1. **Kiểm soát 100% về phân phối mốc thời gian ($t$-distribution):**
   - Đây là **ưu điểm kỹ thuật lớn nhất**. Trong ATD-12K, 100% triplet bị cố định tại mốc giữa $t=0.5$.
   - Khi tự khai thác từ 25 giờ video, ta hoàn toàn chủ động thiết kế mật độ xác suất đều cho $t$:
     $$t \in \{0.20, 0.25, 0.333, 0.40, 0.50, 0.60, 0.667, 0.75, 0.80\}$$
     Giúp mô hình TPS-Inbetween phát huy trọn vẹn sức mạnh "Arbitrary Time-step Interpolation" mà không bị thiên vị (bias) vào $t=0.5$.

2. **Tính thuần khiết miền dữ liệu & Sự hiện đại của nét vẽ (Domain Purity & Modernity):**
   - ATD-12K được thu thập từ nhiều năm trước, chứa nhiều anime cổ điển thập niên 80-90 hoặc bản scan có nhiễu hạt (grain) và mờ nét (blur).
   - Bạn có quyền chọn lọc 100% các tác phẩm đỉnh cao của các studio đương đại danh tiếng (như **Ufotable, MAPPA, Kyoto Animation, Wit Studio, CloverWorks**). Nét vẽ kỹ thuật số sắc sảo, độ phân giải 1080p chuẩn, không nhiễu nén, giúp mô hình học được nét vẽ hiện đại cực kỳ sạch.

3. **Đồng bộ hóa tuyệt đối quy chuẩn Pipeline (100% Pipeline Consistency):**
   - Toàn bộ dữ liệu đều đi qua cùng một tham số trích xuất nét (`AniLines Detail` + `Bilateral Smooth`) và cùng một ngưỡng lọc GlueStick RANSAC ($\text{disp} \le 8\text{px}$, inliers $\ge 25$).
   - Không bị hiện tượng "lệch phân phối" (domain shift) giữa hai nguồn dữ liệu khác nhau.

4. **Bảo tồn tính liên tục của chuỗi hành động (Continuous Temporal Coherence):**
   - Thay vì chỉ có các cặp 3-frame rời rạc, ta có thể lưu trữ toàn bộ các phân cảnh hành động liên tục từ 5s đến 10s (như phân cảnh 7.92s đã thực nghiệm).
   - Điều này tạo ra một bộ **Validation & Benchmark Video Set** độc nhất vô nhị để kiểm chứng độ mượt mà chuyển động mà ATD-12K gốc không thể cung cấp được.

5. **Giá trị đóng góp học thuật & Tính độc lập cao (Academic Novelty):**
   - Bộ Dataset mới 100% này có thể được đóng gói và công bố như một sản phẩm nghiên cứu độc lập (contributions) của đồ án tốt nghiệp/nghiên cứu khoa học, không bị phụ thuộc vào dataset có sẵn của bên thứ ba.

---

### 7.3. Phân Tích Điểm Yếu (Thách thức kỹ thuật & Gánh nặng tài nguyên)

Đi đôi với những điểm mạnh trên là 5 rào cản thực tế rất khốc liệt:

1. **Gánh nặng tính toán tiền xử lý cục bộ khổng lồ (Enormous Local Compute Overhead):**
   - **Phép tính khối lượng:**
     $$\text{Tổng số frames thô} = 25 \text{ giờ} \times 3,600 \times 24 = \mathbf{2,160,000\text{ frames}}$$
   - **Thời gian xử lý trên máy tính cá nhân:**
     - `PySceneDetect` quét và cắt cảnh cho 2.16 triệu frame: Mất khoảng **20 – 30 giờ chạy CPU liên tục**.
     - `Micro-Dedup` (Sobel diff + PSNR) trên GPU RTX 3050 Laptop: Mất khoảng **12 – 16 giờ GPU**.
     - **Tổng cộng:** Máy tính cá nhân phải hoạt động 100% công suất liên tục từ **1.5 đến 2 ngày đêm**, gây nóng máy, hao mòn phần cứng laptop và tiềm ẩn nguy cơ gián đoạn do mất điện/treo máy.

2. **Chi phí lưu trữ đĩa cứng & Băng thông mạng (Storage & Bandwidth):**
   - 25 giờ video 1080p chất lượng cao (bitrate $\ge 6\text{ Mbps}$) cần khoảng **70 – 100 GB** dung lượng tải về.
   - Vùng đệm ổ cứng SSD để bung frame và xử lý lọc dedup cần tối thiểu **150 – 200 GB dung lượng trống**.

3. **Rủi ro "Tái phát minh lại bánh xe" (Reinventing the Wheel):**
   - Bộ dữ liệu ATD-12K đã là một công trình khoa học được thẩm định (peer-reviewed) tại hội nghị thị giác máy tính quốc tế, tiêu tốn hàng tháng trời của cả một nhóm nghiên cứu lớn để thu thập, phân loại và kiểm duyệt.
   - Trong ổ cứng `E:\Dataset\datasets` của bạn **đã có sẵn 4,881 triplets Anime** đạt chuẩn (đã lọc bỏ Disney). Việc loại bỏ hoàn toàn 4,881 triplet này để làm lại từ số 0 là một sự đánh đổi chi phí cơ hội rất lớn về mặt thời gian.

4. **Nhiễu tự động hóa tích lũy do thiếu khâu duyệt bằng mắt người (Uncurated Visual Noise):**
   - Các tác giả ATD-12K có khâu lọc thủ công bằng mắt người (human screening) để loại bỏ các trường hợp đặc biệt.
   - Khi tự động hóa 100% trên 25 giờ video tải về từ Internet, rất dễ bị lọt vào dataset các tạp chất:
     - Dòng chữ phụ đề (hardsub) hoặc logo kênh truyền hình.
     - Chữ chạy credit ở đoạn mở đầu (Opening) và kết thúc (Ending).
     - Hiệu ứng lóe sáng toàn màn hình (flash/lens flare), khói bụi, hiệu ứng kỹ xảo 3D CGI khiến GlueStick bắt điểm sai.

5. **Gây áp lực lên hạn mức 30 giờ GPU/tuần của Kaggle:**
   - Nếu tự khai thác 25 giờ video, ta sẽ thu hoạch được khoảng **10,000 – 12,000 Candidate Triplets** (~36,000 ảnh).
   - Khi đưa 36,000 ảnh này lên Kaggle để chạy Stage 4 (GlueStick RANSAC) và Stage 5 (AniLines + Segmentation):
     - Thời gian GPU Kaggle tiêu tốn: **~5.0 – 6.0 giờ GPU**.
     - Thời gian còn lại cho việc Fine-tune mô hình trong tuần bị co hẹp xuống còn **~23 – 24 giờ**, làm giảm số epoch hoặc giảm độ sâu của quá trình huấn luyện.

---

### 7.4. Bảng So Sánh Đối Đầu Trực Diện

| Tiêu chí đánh giá | Phương án A: Tự làm 100% từ 25h Video | Phương án B: Chỉ dùng 100% ATD-12K Anime | Phương án C (HYBRID): ATD-12K + 12-16h Video |
|:---|:---:|:---:|:---:|
| **Quy mô Triplets đạt chuẩn** | 8,000 ~ 10,000 triplets | ~3,500 triplets | **~6,000 triplets (Chuẩn vàng)** |
| **Đa dạng hóa mốc thời gian $t$** | **Tuyệt đối ($t \in [0.2, 0.8]$)** | Kém (100% chỉ có $t=0.5$) | **Rất tốt (Cân bằng $t=0.5$ và $t \ne 0.5$)** |
| **Đa dạng hóa phong cách vẽ** | Rất tốt (nếu chọn nhiều bộ) | **Cực kỳ đa dạng (hàng trăm bộ)** | **Hoàn hảo (Kết hợp cổ điển + Hiện đại)** |
| **Thời gian tiền xử lý Local** | **Rất nặng (36 – 48 giờ chạy máy)** | 0 giờ (Có sẵn) | **Vừa phải (10 – 15 giờ chạy máy)** |
| **Dung lượng ổ cứng yêu cầu** | 200+ GB SSD | 15 GB | ~80 GB SSD |
| **Tiêu tốn GPU Kaggle tạo data** | ~5.0 - 6.0 giờ GPU | ~1.0 giờ GPU | **~2.0 giờ GPU** |
| **Giờ GPU Kaggle còn lại để Train**| ~23.0 giờ | ~28.0 giờ | **~26.0 giờ (Rất dư dả)** |
| **Rủi ro tạp chất (Sub, Credit, CGI)**| Cao (Cần lọc kỹ) | Rất thấp (Đã kiểm duyệt) | **Thấp (Đã qua lọc RANSAC)** |
| **Mức độ khả thi đồ án (1 người làm)**| ⚠️ Khá rủi ro về tiến độ | Rất dễ nhưng chất lượng $t$ hạn chế | ⭐ **TỐI ƯU NHẤT VỀ HIỆU QUẢ / CÔNG SỨC** |

---

### 7.5. Đánh Giá Tổng Kết & Lời Khuyên Chiến Lược Cho Đồ Án

1. **Nhận định:** Ý kiến *"Tự tạo 100% từ 25 giờ video"* là một ý tưởng **rất dũng cảm và có tính học thuật cao**, nhưng **không tối ưu về mặt kinh tế thời gian (Cost-Benefit Ratio)** đối với một cá nhân hoặc nhóm đồ án có thời hạn nộp bài cố định. Bạn sẽ phải mất ít nhất 3-4 ngày chỉ để giải quyết các lỗi vụn vặt của việc xử lý 2.16 triệu frame video thô trước khi bước vào khâu huấn luyện mô hình.
2. **Khuyến nghị chiến lược:**
   - **Giai đoạn 1 (Nên chọn Phương án Hybrid):** Tận dụng **4,881 triplets ATD-12K Anime có sẵn** (tiết kiệm ngay 15-20 giờ làm việc) + Xử lý thêm **3-4 bộ Anime hiện đại (12-16 tiếng)** để bổ sung các mốc $t \ne 0.5$. Phương án này mang lại **95% chất lượng của Phương án A nhưng chỉ tốn 40% công sức và thời gian**.
   - **Giai đoạn 2 (Mở rộng nếu còn thời gian):** Sau khi đã train thành công mô hình với phương án Hybrid và kiểm chứng kết quả tốt, nếu vẫn còn thời gian trước hạn nộp đồ án, bạn hoàn toàn có thể tải thêm video để mở rộng thành bộ dataset 100% tự tạo để lấy điểm cộng sáng tạo trong báo cáo.

---

> **KẾT LUẬN:** Phương án chia tách **"Tiền xử lý thô tại Local + Chạy Deep Learning & Train trên Kaggle"** kết hợp mô hình **Hybrid (ATD-12K Anime + Video Tuyển Chọn)** là kiến trúc tối ưu nhất về mọi mặt: loại bỏ hoàn toàn rủi ro sập ổ cứng 19.5 GB, không tốn dung lượng upload dư thừa, và bảo toàn 93% thời lượng GPU của Kaggle cho công tác huấn luyện cốt lõi.

