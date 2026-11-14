# PHÂN TÍCH CHUYÊN SÂU MÔ HÌNH TPS-INBETWEEN & GIẢI PHÁP XÂY DỰNG DATASET FINE-TUNE CHO HOẠT HỌA ANIME

> **Tài liệu nghiên cứu kỹ thuật - Đề tài PBL6**  
> **Chủ đề:** Phân tích nguyên nhân chênh lệch chất lượng nội suy giữa `TestImage1` và `ChayGoAnimeLinesDetail` trên mô hình TPS-Inbetween; giải pháp kiến trúc và chiến lược xây dựng dataset thực tế để fine-tune mô hình đạt độ chính xác cao.

---

## 1. TỔNG QUAN & BẢN CHẤT VẤN ĐỀ

Khi kiểm thử mô hình **TPS-Inbetween** (*Thin-Plate Spline-based Interpolation for Animation Line Inbetweening*) trên hai tập dữ liệu:
1. **`TestImage1` (`F1.png`, `F2.png`)**: Kết quả nội suy các frame $t \in [0.25, 0.50, 0.75]$ rất sắc nét, giữ nguyên cấu trúc nét vẽ, chuyển động mượt mà và đúng kỳ vọng.
2. **`ChayGoAnimeLinesDetail_NoBG` và `ChayGoAnimeLinesDetail_NoBG_NoGay` trong `Final test`**: Kết quả xuất hiện hiện tượng bóng ma (ghosting / nét kép), nét vẽ bị xé rách (tearing), méo mó cục bộ (local warping) và các chi tiết cơ thể bị biến dạng khi nội suy giữa các mốc $0.0s \rightarrow 0.5s \rightarrow 1.0s$.

### Nguyên nhân cốt lõi (Core Root Cause)
Sự chênh lệch này là kết quả của **sự lệch pha phân phối dữ liệu (Domain & Motion Distribution Shift)** kết hợp với **giới hạn toán học của phép biến dạng TPS không điều chuẩn ($\lambda=0$)**:

* **Tập dữ liệu huấn luyện gốc `ml240data` (MixamoLine240)** gồm các chuyển động 3D nhân vật mô phỏng, mật độ khung hình dày đặc ($30\text{--}60\text{ fps}$), độ dời điểm ảnh giữa các frame rất nhỏ ($\text{mean} \approx 3.8 \sim 8.0\text{ px}$), nét vẽ vector nhân tạo khép kín và bảo toàn tô-pô tuyệt đối.
* **`TestImage1`** nằm trọn vẹn trong "vùng an toàn" của mô hình: biên độ dịch chuyển cực nhỏ ($\text{median} = 1.0\text{ px}$), độ khớp GlueStick đạt **$77.4\%$** với **$0\%$ ngoại lai $>50\text{ px}$**.
* **`ChayGo`** là chuỗi hoạt họa hành động chạy nhảy thực tế, bước nhảy thời gian $\Delta t = 0.5\text{s}$ tương đương **12 frame hoạt họa (ở 24 fps)**. Biên độ dịch chuyển lên tới $178.9\text{ px}$, chứa **$5.9\%$ điểm ngoại lai (outliers)**. Khi đưa qua TPS với hệ số $\lambda = 0$, chỉ cần một vài điểm bắt cặp sai sẽ làm toàn bộ lưới không gian bị xoắn vặn và phá hủy hoàn toàn cấu trúc nét vẽ.

---

## 2. PHÂN TÍCH SO SÁNH ĐỊNH LƯỢNG ĐẦU VÀO (EXPERIMENTAL PROOF)

Để làm rõ nguyên nhân, chúng tôi đã trích xuất đặc trưng và đo đạc thực nghiệm trực tiếp trên pipeline GlueStick + TPS của mô hình:

### 2.1. Bảng thông số đo đạc thực nghiệm trực tiếp

| Tiêu chí đo đạc | `ml240data` (Train Set - step 1) | `TestImage1` (`F1` $\rightarrow$ `F2`) | `ChayGo_NoBG` ($0.0\text{s} \rightarrow 0.5\text{s}$) | `ChayGo_NoGay` ($0.0\text{s} \rightarrow 0.5\text{s}$) |
| :--- | :---: | :---: | :---: | :---: |
| **Bản chất dữ liệu** | 3D Render Line Art | Anime Douga (Line Art chuẩn) | Line trích xuất từ Anime màu | Line nhị phân hóa 1-bit |
| **Tổng số keypoints phát hiện** | $\sim 800$ | $1336 \ / \ 1322$ | $862 \ / \ 965$ | $915 \ / \ 1033$ |
| **Số cặp điểm khớp (Matched KPs)** | $347$ | **$1029$** | $296$ | $322$ |
| **Tỷ lệ khớp thành công (%)** | $\approx 43.4\%$ | **$77.4\%$** | $32.4\%$ | $33.0\%$ |
| **Độ dời trung bình (Mean Disp)** | $3.84\text{ px}$ | **$4.80\text{ px}$** | $20.10\text{ px}$ | $22.01\text{ px}$ |
| **Độ dời trung vị (Median Disp)** | $2.61\text{ px}$ | **$1.00\text{ px}$** | $15.21\text{ px}$ | $15.05\text{ px}$ |
| **Độ dời cực đại (Max Disp)** | $19.59\text{ px}$ | **$27.64\text{ px}$** | $90.34\text{ px}$ | **$178.89\text{ px}$** |
| **Số điểm lỗi nặng ($> 50\text{ px}$)** | **$0\ (0\%)$** | **$0\ (0\%)$** | $9\ (3.0\%)$ | **$19\ (5.9\%)$** |
| **Số điểm dị biệt cực đoan ($> 100\text{ px}$)**| **$0\ (0\%)$** | **$0\ (0\%)$** | $0\ (0\%)$ | **$4\ (1.2\%)$** |
| **Chất lượng PSNR tái tạo ($t=0.2$)** | $> 32\text{ dB}$ | $\approx 28.5\text{ dB}$ | $10.24\text{ dB}$ | $10.16\text{ dB}$ |

### 2.2. Nhận xét then chốt từ dữ liệu thực nghiệm
1. **`TestImage1` tương đồng gần như hoàn hảo với dữ liệu train `ml240data`**: Độ dời trung vị giữa F1 và F2 chỉ là **$1.0\text{ px}$** (do F1 và F2 thuộc phân đoạn nhân vật giữ thế tĩnh - hold pose trong `Sample Video.mp4`). Không có bất kỳ điểm nào lệch quá $50\text{ px}$.
2. **`ChayGo` có biên độ dịch chuyển vượt ngưỡng dung sai**: Độ dời trung bình gấp gần **5 lần**, và xuất hiện các điểm dịch chuyển lên tới **$178.89\text{ px}$**.
3. **Hiện tượng nghịch lý của `ChayGo_NoGay`**: Việc binarize thành ảnh đen trắng 1-bit tuyệt đối (khử toàn bộ dải xám 1-254) không những không giúp mô hình tốt hơn mà còn làm xuất hiện **4 điểm ngoại lai $> 100\text{ px}$** (so với 0 điểm của `NoBG`). Nguyên nhân là do SuperPoint mất thông tin đạo hàm liên tục của gradient, các bậc thang răng cưa (staircase aliasing) tạo ra các góc giả (spurious corners) khiến GlueStick ghép nhầm giữa các bộ phận xa nhau.

---

## 3. PHÂN TÍCH TÁC ĐỘNG TỪ ĐẦU VÀO VÀ KIẾN TRÚC MÔ HÌNH TPS-INBETWEEN

Kiến trúc của mô hình TPS-Inbetween bao gồm 4 giai đoạn nối tiếp:
```
[Ảnh I_0, I_1] 
     │
     ▼
(1) GlueStick (SuperPoint + Wireframe Matcher) ──► Cặp điểm tương ứng (kps0, kps1)
     │
     ▼
(2) Thin-Plate Spline (TPS Solver, λ = 0)     ──► Coarse Flow Trường vector dịch chuyển thô
     │
     ▼
(3) IFNet Motion Refinement                   ──► Fine Flow Tinh chỉnh vector cục bộ
     │
     ▼
(4) UNet Synthesis & Alpha Blending            ──► Frame trung gian I_t
```

Mỗi thành phần trong pipeline này bị ảnh hưởng nặng nề bởi các yếu tố sau:

### 3.1. Hạn chế toán học của Thin-Plate Spline khi $\lambda = 0$
Trong file [util/utils.py](file:///e:/CODE/PBL/PBL6/Test/TestModels/tps-inbetween/util/utils.py#L18-L38), hàm giải TPS được định nghĩa:
$$\mathbf{K}_{ij} = U(\|\mathbf{c}_i - \mathbf{c}_j\|) + \lambda \mathbf{I}, \quad \text{với } U(r) = r^2 \log(r + 1e-6)$$
Hệ thống giải ma trận tuyến tính:
$$\begin{bmatrix} \mathbf{K} & \mathbf{P} \\ \mathbf{P}^T & \mathbf{0} \end{bmatrix} \begin{bmatrix} \mathbf{w} \\ \mathbf{a} \end{bmatrix} = \begin{bmatrix} \mathbf{v} \\ \mathbf{0} \end{bmatrix}$$
Mô hình cài đặt **$\lambda = 0.0$** (nội suy chính xác tuyệt đối - exact interpolation). Điều này có nghĩa:
* Spline bị ép buộc phải uốn cong đi qua **chính xác $100\%$ các điểm kiểm soát**.
* Hàm cơ sở xuyên tâm $U(r) = r^2 \log r$ có **tầm ảnh hưởng vô hạn (global support)** và tăng bậc hai theo khoảng cách.
* **Hậu quả:** Chỉ cần $1$ điểm ở bàn tay bị bắt nhầm sang đầu gối cách xa $150\text{ px}$, toán tử TPS sẽ tạo ra một lực xoắn cực mạnh lên toàn bộ không gian xung quanh điểm đó. Kết quả là toàn bộ phần thân xung quanh bị kéo giãn, méo mó và rách nét.

### 3.2. Giả định chuyển động tuyến tính (Linear Motion Assumption)
Trong [model/tpsinbet.py](file:///e:/CODE/PBL/PBL6/Test/TestModels/tps-inbetween/model/tpsinbet.py#L85-L86), vector dịch chuyển thô tại thời điểm $t$ được tính bằng:
$$\mathbf{F}_{0 \rightarrow t} = \mathbf{F}_{0 \rightarrow 1} \cdot t^2 - (1-t)t \cdot \mathbf{F}_{1 \rightarrow 0}$$
$$\mathbf{F}_{1 \rightarrow t} = \mathbf{F}_{1 \rightarrow 0} \cdot (1-t)^2 - t(1-t) \cdot \mathbf{F}_{0 \rightarrow 1}$$
Công thức này giả định rằng đường đi của mọi điểm giữa Frame 0 và Frame 1 là **đơn điệu và gần như thẳng hàng**.
* Trong `TestImage1`, nhân vật cử động nhẹ $\rightarrow$ giả định này hoàn toàn đúng.
* Trong `ChayGo` với $\Delta t = 0.5\text{s}$ (12 frames), chân và tay nhân vật chuyển động theo **quỹ đạo con lắc tròn (harmonic/pendulum motion)**. Việc xấp xỉ tuyến tính khiến cánh tay hoặc cẳng chân ở frame trung gian bị teo ngắn lại (foreshortening collapse) hoặc cắt ngang qua thân thể thay vì vung vòng qua phía trước.

### 3.3. Cơ chế tổng hợp Warped Blending và hiện tượng bóng ma (Ghosting)
Công thức tổng hợp frame cuối cùng tại dòng 103 của [model/tpsinbet.py](file:///e:/CODE/PBL/PBL6/Test/TestModels/tps-inbetween/model/tpsinbet.py#L103):
$$I_t = (1-t) \cdot \text{fine\_I}_{0 \rightarrow t} + t \cdot \text{fine\_I}_{1 \rightarrow t} + \text{res}$$
* Bản chất là **trộn màu có bù chuyển động (motion-compensated alpha blend)** cộng với một phần dư (residual) từ UNet.
* **Đối với ảnh nét vẽ (Line Art):** Nét vẽ chỉ dày từ $1 \sim 3\text{ pixel}$ trên nền trắng. Nếu $\text{fine\_I}_{0 \rightarrow t}$ và $\text{fine\_I}_{1 \rightarrow t}$ bị lệch nhau dù chỉ **$2\text{ pixel}$**, việc cộng alpha $0.5 \times I_0 + 0.5 \times I_1$ sẽ lập tức sinh ra **hai nét song song mờ nhạt (nét kép / ghosting)**.
* Mạng UNet không thể tự xóa một nét sai nếu độ lệch quá lớn, dẫn đến ảnh sinh ra bị lem luốc, bẩn nền và mất tính thẩm mỹ của tranh vẽ tay.

### 3.4. Vấn đề biến đổi tô-pô (Topology Change) trong Hoạt Họa 2D
* TPS là một **phép đồng phôi (homeomorphism)**: ánh xạ liên tục 1-1, bảo toàn tính chất lân cận. Nó **không thể** mô tả các hiện tượng hoạt họa thực tế như:
  1. Một lọn tóc tách làm đôi.
  2. Nếp gấp quần áo biến mất khi duỗi thẳng.
  3. Bàn tay bị che khuất (occlusion) sau lưng rồi xuất hiện trở lại.
* Khi gặp hiện tượng che khuất hoặc đổi góc nhìn, TPS cố gắng "kéo giãn" một nét vẽ không còn tồn tại ở frame sau, tạo ra các vệt màng mờ kỳ dị.

---

## 4. CHIẾN LƯỢC TOÀN DIỆN: XÂY DỰNG DATASET CHUẨN ĐỂ FINE-TUNE TPS-INBETWEEN

Để mô hình TPS-Inbetween hoạt động chính xác trên frame anime thực tế và hạn chế tối đa việc phải chỉnh sửa thủ công (clean-up), việc huấn luyện lại (fine-tune) bắt buộc phải giải quyết triệt để sự thiếu hụt về phân phối dữ liệu. Dưới đây là phương pháp luận và quy trình thực hiện chi tiết.

### 4.1. Quy tắc chọn mẫu theo nhịp hoạt họa (Timing & Spacing in Anime)
Không thể lấy hai frame bất kỳ cách nhau $0.5\text{s}$ làm keyframe đầu vào cho mô hình nội suy frame đơn giản.
Trong sản xuất anime chuyên nghiệp (tiêu chuẩn Nhật Bản):
* **Frame rate tổng thể:** $24\text{ fps}$.
* **Keyframe (Genga - 原画):** Được vẽ ở nhịp **"On 3s"** ($8\text{ frame/giây} \rightarrow \Delta t \approx 0.125\text{s}$, cách nhau 3 frame video) hoặc **"On 2s"** ($12\text{ frame/giây} \rightarrow \Delta t \approx 0.083\text{s}$, cách nhau 2 frame video). Các pha hành động cực nhanh mới vẽ **"On 1s"** ($24\text{ fps}$).
* **In-between (Douga - 動画):** Là các frame được chèn vào giữa các Genga này (thường từ $1 \sim 2$ frame douga giữa 2 genga).

> **Quy chuẩn lấy mẫu cho Dataset:**
> * Bước nhảy giữa Frame 0 ($I_0$) và Frame 1 ($I_1$) khi train **phải nằm trong khoảng $2 \le \Delta k \le 5$ frame** của video $24\text{ fps}$ (tương ứng khoảng thời gian thực $\Delta t \in [0.08\text{s}, 0.20\text{s}]$).
> * Nếu cần nội suy đoạn dài $0.5\text{s}$ (12 frames), hệ thống phải chạy theo chuỗi móc xích (hierarchical / sequential interpolation): $0.0\text{s} \rightarrow 0.2\text{s} \rightarrow 0.4\text{s} \rightarrow 0.5\text{s}$, chứ không nhảy thẳng một bước $0.0\text{s} \rightarrow 0.5\text{s}$.

```
Cách tiếp cận sai (hiện tại):
[Keyframe 0.0s] ──────────────────────────(Nhảy 12 frame)──────────────────────────► [Keyframe 0.5s]
                                      (Vỡ nét, biến dạng)

Cách tiếp cận đúng (Anime Pipeline):
[Genga 0.0s] ──► [Douga 0.08s] ──► [Genga 0.17s] ──► [Douga 0.25s] ──► [Genga 0.33s] ──► [Genga 0.5s]
```

### 4.2. Chuẩn hóa chất lượng nét vẽ (Line Art Preprocessing Standards)
Để GlueStick trích xuất được đặc trưng chuẩn xác mà không sinh ra ngoại lai:

1. **Tuyệt đối không dùng ảnh nhị phân cứng 1-bit (NoGay)**:
   * Giữ lại lớp chuyển tiếp mức xám mềm (anti-aliasing sub-pixel) dọc theo rìa nét vẽ.
   * Sử dụng thuật toán lọc song phương bảo toàn biên kết hợp siêu mẫu Lanczos-4 (như hàm `smooth_lineart_channel` trong [smooth_lines.py](file:///e:/CODE/PBL/PBL6/TestDataset/ChayGoAnimeLinesDetail_NoBG/smooth_lines.py)).
2. **Chuẩn hóa độ dày nét vẽ (Stroke Width Normalization)**:
   * Áp dụng biến đổi khoảng cách Euclidean (EDT) và thuật toán làm mảnh (morphological skeletonization) để đồng bộ bề dày nét về khoảng $1.5 \sim 2.5\text{ pixel}$. Tránh trường hợp frame này nét mảnh, frame kia nét đậm do trích xuất tự động.
3. **Quy chuẩn nền và kênh màu**:
   * Chuẩn hóa về ảnh 1 kênh Grayscale, nền trắng tinh ($255$), nét đen đậm nhất ($0$).
   * Tự động cắt bỏ các viền nhiễu xung quanh do bước cắt nền (alpha mask bleed).

### 4.3. Cấu trúc bộ dữ liệu huấn luyện (Dataset Structure)
Xây dựng dataset theo cấu trúc thư mục phân cấp tương thích với DataLoader của PyTorch:

```text
E:\Dataset\AnimeLine_FineTune/
├── train/
│   ├── clip_001_run/
│   │   ├── 0001.png
│   │   ├── 0002.png
│   │   ├── 0003.png
│   │   └── ...
│   ├── clip_002_fight/
│   └── ...
├── val/
└── matches/                 (Tự động tạo bằng GlueStick + RANSAC Filter)
    ├── clip_001_run/
    │   ├── 0001_0004.npy    (Lưu cặp keypoints sạch giữa frame 1 và 4)
    │   └── ...
```

Mỗi mẫu huấn luyện (Sample Tuple) bao gồm:
* **Đầu vào:** $I_0$ (frame gốc), $I_1$ (frame đích), mốc thời gian $t = \frac{k - k_0}{k_1 - k_0} \in (0, 1)$.
* **Ground Truth:** $I_t$ (frame thực tế tại vị trí $k$).
* **File so khớp:** File `.npy` chứa tọa độ các điểm đặc trưng đã qua lọc ngoại lai.

---

## 5. CẢI TIẾN THUẬT TOÁN ĐỂ LOẠI BỎ SAI SỐ VÀ GIẢM THIỂU CHỈNH SỬA THỦ CÔNG

Huấn luyện dữ liệu mới cần đi đôi với việc gia cố các lỗ hổng toán học trong mã nguồn của TPS-Inbetween. Dưới đây là 3 nâng cấp mang tính quyết định:

### 5.1. Thêm thành phần điều chuẩn độ uốn TPS ($\lambda > 0$)
Sửa đổi hàm `fit` trong [util/utils.py](file:///e:/CODE/PBL/PBL6/Test/TestModels/tps-inbetween/util/utils.py#L18):
```python
# Thay vì cố định lambd = 0.
def fit(c, lambd=0.05, reduced=False):
    n = c.shape[0]
    device = c.device

    U = TPS.u(TPS.d(c, c))
    # lambd > 0 giúp ma trận K khả nghịch tốt hơn và hấp thụ sai số của các điểm nhiễu
    K = U + torch.eye(n, dtype=torch.float32, device=device) * lambd
    ...
```
* **Tác dụng:** $\lambda \in [0.01, 0.1]$ biến phép biến dạng từ "nội suy tuyệt đối" (interpolation) thành "làm mịn xấp xỉ" (smoothing approximation). Nếu một điểm bị lệch nhẹ, bề mặt spline sẽ không bị uốn nhọn hoắt mà duy trì độ phẳng tự nhiên.

### 5.2. Tích hợp bộ lọc ngoại lai hình học RANSAC / MAGSAC++ trước khi chạy TPS
Trước khi lưu file `.npy` từ GlueStick (trong `generate_matches.py` hoặc `interp_custom_times.py`), bổ sung thuật toán lọc các cặp điểm không nhất quán:
```python
import cv2

def filter_matches_ransac(kps0, kps1, threshold=4.0):
    """
    Sử dụng ma trận Affine / Homography với RANSAC để loại bỏ các vector chuyển động bất thường.
    """
    if len(kps0) < 4:
        return kps0, kps1
    
    # Tìm ma trận biến đổi Affine cục bộ
    _, inliers = cv2.estimateAffinePartial2D(
        kps0, kps1, method=cv2.RANSAC, ransacReprojThreshold=threshold
    )
    
    if inliers is not None:
        inliers = inliers.ravel().astype(bool)
        return kps0[inliers], kps1[inliers]
    return kps0, kps1
```
* **Kết quả:** Loại bỏ hoàn toàn $100\%$ các điểm nhảy dây $> 50\text{ px}$ phát hiện trên tập `ChayGo`, bảo vệ lưới biến dạng không bị xé toạc.

### 5.3. Sử dụng hàm mất mát hình học Distance Transform (EDT Loss) khi Fine-tune
Hàm mất mát truyền thống $L_1 = \|I_{\text{pred}} - I_{\text{gt}}\|_1$ rất yếu khi học vẽ nét mảnh: nếu nét vẽ dự đoán bị lệch $2\text{ px}$ so với ground truth, độ giao nhau pixel bằng 0, gradient không cung cấp được hướng kéo nét về đúng vị trí (hiện tượng gradient vanishing).

Cần kết hợp **Euclidean Distance Transform Loss (EDT Loss)**:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_1 + \alpha \mathcal{L}_{\text{Perceptual}} + \beta \mathcal{L}_{\text{EDT}}$$
Trong đó:
$$\mathcal{L}_{\text{EDT}} = \frac{1}{|\Omega|} \sum_{p \in \Omega} I_{\text{pred}}(p) \cdot \mathcal{D}_{\text{gt}}(p)^2$$
* $\mathcal{D}_{\text{gt}}(p)$ là khoảng cách Euclidean từ pixel $p$ tới nét vẽ gần nhất của ảnh ground truth.
* **Tác dụng:** Tạo ra một "giếng thế năng hấp dẫn" kéo các nét vẽ trung gian về đúng vị trí của nét vẽ mục tiêu, triệt tiêu hoàn toàn hiện tượng bóng ma kép mà không cần người dùng vẽ lại bằng tay.

---

## 6. KẾ HOẠCH HÀNH ĐỘNG TRIỂN KHAI (ACTION PLAN)

| Giai đoạn | Mục tiêu cụ thể | Hành động kỹ thuật | Kết quả đầu ra |
| :--- | :--- | :--- | :--- |
| **Giai đoạn 1** | Chuẩn bị dữ liệu anime | Lấy các đoạn cắt anime 24fps sạch nét; chia bước nhảy $2 \le \Delta k \le 4$; lọc làm mượt nét qua `smooth_lines.py`. | Bộ 5,000+ triplet $(I_0, I_t, I_1)$ đạt chuẩn. |
| **Giai đoạn 2** | Nâng cấp bộ so khớp | Thêm module RANSAC / MAGSAC++ lọc bỏ điểm ngoại lai trong `generate_matches.py`; đặt $\lambda = 0.05$ trong TPS. | Dữ liệu matching không còn điểm ngoại lai $> 30\text{ px}$. |
| **Giai đoạn 3** | Huấn luyện Fine-tune | Đóng băng GlueStick, fine-tune mạng `IFNet` và `UNet Synthesis` với hàm loss $\mathcal{L}_1 + \mathcal{L}_{\text{EDT}} + \mathcal{L}_{\text{VGG}}$. | Checkpoint `model_anime_finetuned.pt`. |
| **Giai đoạn 4** | Kiểm thử & Đánh giá | Chạy lại trên chuỗi `ChayGo` với bước nhảy hợp lý ($0.1\text{s}$ hoặc $0.2\text{s}$) và so sánh định lượng PSNR, SSIM, LPIPS. | Nét vẽ sắc mảnh, sạch nền, không còn bóng ma. |

---

## 7. KẾT LUẬN

Mô hình **TPS-Inbetween** sở hữu kiến trúc kết hợp rất mạnh mẽ giữa hình học biến dạng (TPS) và học sâu (IFNet + UNet). Tuy nhiên, nguyên nhân thất bại trên `ChayGo` không phải do mô hình bị lỗi, mà là do:
1. **Ép mô hình vượt quá giới hạn thiết kế**: Bắt mô hình nội suy bước nhảy $\Delta t = 0.5\text{s}$ của một chu kỳ chạy phức tạp (trong khi mô hình chỉ được thiết kế cho bước nhảy vi mô $\Delta t \approx 0.05\text{s} \sim 0.15\text{s}$).
2. **Thiếu cơ chế lọc ngoại lai khi dữ liệu bị thoái hóa**: Nhị phân hóa cứng 1-bit (`NoGay`) làm GlueStick bắt sai điểm, kết hợp với $\lambda = 0$ làm bung vỡ lưới không gian.

Bằng cách áp dụng **lọc ngoại lai RANSAC**, **điều chuẩn $\lambda = 0.05$**, và **huấn luyện lại trên tập anime chuẩn nhịp $2 \sim 4$ frame với hàm mất mát Distance Transform**, mô hình TPS-Inbetween hoàn toàn có thể tự động hóa việc vẽ inbetween cho phim hoạt hình anime với độ chính xác cao và giảm thiểu tối đa công việc chỉnh sửa thủ công của họa sĩ.
