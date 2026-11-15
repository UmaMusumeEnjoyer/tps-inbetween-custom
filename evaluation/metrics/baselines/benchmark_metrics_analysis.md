# Phân Tích Chỉ Số Benchmark cho TestModels
> **Cập nhật lần 2 (2026-08-30):** Đã kiểm tra trực tiếp toàn bộ output trong thư mục `demo/output` của cả 5 model. Phân tích được hiệu chỉnh dựa trên đặc điểm output quan sát được thực tế.

---

## Phát Hiện Quan Trọng Từ Demo Output

Sau khi kiểm tra trực tiếp tất cả output ảnh, có một số điểm **khác biệt đáng kể** so với phân loại ban đầu:

### Cấu trúc output quan sát được

| Model | Root output | Subfolder: Color | Subfolder: No Color | Subfolder: Rotate |
|---|---|---|---|---|
| **AnimeInbet** | 1 frame line art (rất thưa, ~34KB) | ❌ | ❌ | ❌ |
| **tps-inbetween** | 1 frame grayscale (~609KB) | ✅ Line art có màu nhẹ | ✅ Pure line art | ✅ Rotate test |
| **AnimeInterp** | 1 frame RGB anime (~657KB) | ✅ RGB đầy đủ màu | ✅ Line art có màu nhạt | ✅ Rotate test |
| **UPR-Net** | 1 frame RGB anime (~657KB) | ✅ RGB đầy đủ màu (`*C.png`) | ✅ Grayscale tốt | ✅ Rotate test |
| **uniVIP** | 1 frame RGB anime (~657KB) | ✅ RGB đầy đủ màu | ✅ Grayscale đẹp hơn | ✅ Rotate test |

### Đặc điểm output thực tế (đã quan sát)

**AnimeInbet** — Output là ảnh line art **cực kỳ thưa** (~34KB vs input ~654KB). Các nét vẽ rời rạc trên nền trắng, không có cấu trúc hoàn chỉnh của nhân vật — đây là đặc trưng của mô hình xử lý theo phương pháp vector/graph fusion (output là intermediate skeleton frame, chưa phải ảnh hoàn chỉnh). **Metric pixel-level như PSNR/SSIM gần như vô dụng với loại output này.**

**tps-inbetween** — Sinh ra **3 loại output**:
- `No Color`: Pure line art sạch, chất lượng cao, cấu trúc đầy đủ
- `Color`: Line art có màu nhẹ (bán màu)  
- `Rotate`: Test case xoay 90° — input-output là ảnh bamboo forest RGB grayscale

**AnimeInterp** — Sinh ra **3 loại output**:
- `No Color`: Line art có màu nhạt (sketch-colored) — input từ anime frame có sketch
- `Color`: RGB đầy đủ màu của ảnh anime chất lượng cao
- `Rotate`: Test case xoay, input/output là ảnh bamboo forest RGB grayscale nặng artifact

**UPR-Net** — Sinh ra **3 loại output** + **sequence output** (23 frames + `interpolated_video.mp4`):
- `Color Still`: RGB đầy đủ màu, chất lượng tốt nhất trong nhóm
- `No Color Still`: Grayscale chất lượng cao
- `Rotate`: Test case xoay
- `test_sequence_output/`: **Duy nhất có video MP4 output** — rất quan trọng cho FVD và Warping Error

**uniVIP** — Tương tự UPR-Net về chất lượng output, nhưng không có sẵn video sequence trong demo output hiện tại.

### Điểm sửa đổi quan trọng so với phân tích cũ

> [!WARNING]
> **Phát hiện 1:** AnimeInbet **không** phải model video interpolation RGB. Output của nó là **sparse line art skeleton** — rất thưa, không hoàn chỉnh như một frame thông thường. PSNR/SSIM trên output này là vô nghĩa tuyệt đối.

> [!WARNING]
> **Phát hiện 2:** tps-inbetween và AnimeInterp đều được test với **cùng input** (bao gồm cả input là ảnh anime RGB có màu). Điều này nghĩa là AnimeInterp **cũng xử lý được line art** (output `No Color` subfolder), tức boundary giữa 2 nhóm mờ hơn so với tưởng ban đầu.

> [!NOTE]
> **Phát hiện 3:** UPR-Net là **model duy nhất** có output video sequence hoàn chỉnh (`interpolated_video.mp4`), làm cho FVD và Warping Error **khả thi trực tiếp** chỉ với UPR-Net mà không cần xử lý thêm.

> [!NOTE]
> **Phát hiện 4:** Tất cả model (trừ AnimeInbet) đều có test case **Rotate** — đây là stress test cho large motion. Điều này mở ra khả năng so sánh temporal stability trên cùng test case.

---

## 1. Tổng quan các Model

Có **2 nhóm bài toán** chính trong `TestModels/`, tuy nhiên ranh giới giữa 2 nhóm **mờ hơn** so với lý thuyết ban đầu:

| Model | Paper | Bài toán | Output thực tế (quan sát demo) |
|---|---|---|---|
| **AnimeInbet** | ICCV 2023 | Cartoon Line Inbetweening (graph-based) | **Sparse skeleton** — nét vẽ rời rạc, ~34KB, không hoàn chỉnh |
| **tps-inbetween** | — | Animation Line Inbetweening (TPS-based) | Line art thuần hoặc có bán màu, hỗ trợ arbitrary t |
| **AnimeInterp** | CVPR 2021 | Animation Video Frame Interpolation | RGB anime **và** line art có màu nhạt (cùng pipeline) |
| **UPR-Net** | CVPR 2023 | General Video Frame Interpolation | RGB đầy đủ, **có sẵn video MP4 sequence** |
| **uniVIP** | ICASSP 2025 | Arbitrary-time Video Interpolation & Prediction | RGB + arbitrary t (kể cả t > 1) |

> **Lưu ý cập nhật:** AnimeInbet có output **đặc biệt khác** — sparse skeleton chứ không phải ảnh line art hoàn chỉnh như tps-inbetween. AnimeInterp và tps-inbetween đều xử lý được cả ảnh line art lẫn ảnh màu. UPR-Net là model duy nhất có sẵn video sequence output.

---

## 2. Định Nghĩa và Mức Độ Phù Hợp

### 2.1 Chamfer Distance (CD)
**Định nghĩa:** Đo trung bình khoảng cách nearest-neighbor giữa 2 point cloud:
$$CD(P,Q) = \frac{1}{|P|}\sum_{p \in P}\min_{q \in Q}\|p-q\| + \frac{1}{|Q|}\sum_{q \in Q}\min_{p \in P}\|q-p\|$$
Trong line art: mỗi pixel đường nét là một điểm trong point cloud.

| AnimeInbet | tps-inbetween | AnimeInterp | UPR-Net | uniVIP |
|:---:|:---:|:---:|:---:|:---:|
| ✅ | ✅ | ❌ | ❌ | ❌ |

**Lý do:** CD là metric chính cho line art (AnimeInbet đã implement sẵn `compute_cd.py`). Không có ý nghĩa với ảnh RGB vì ảnh RGB không phải point cloud thưa.

> **Lưu ý sau quan sát:** AnimeInbet output cực kỳ thưa (~34KB/frame). CD rất phù hợp trong trường hợp này vì chính xác là đo khoảng cách giữa 2 tập điểm thưa. tps-inbetween `No Color` subfolder cũng rất phù hợp.

---

### 2.2 Weighted Chamfer Distance (WCD)
**Định nghĩa:** Biến thể của CD, gán trọng số $w_i$ cho từng điểm dựa trên độ quan trọng cấu trúc (ví dụ: junction point có trọng số cao hơn):
$$WCD(P,Q) = \frac{1}{\sum w_p}\sum_{p \in P} w_p \cdot \min_{q \in Q}\|p-q\| + \frac{1}{\sum w_q}\sum_{q \in Q} w_q \cdot \min_{p \in P}\|q-p\|$$

| AnimeInbet | tps-inbetween | AnimeInterp | UPR-Net | uniVIP |
|:---:|:---:|:---:|:---:|:---:|
| ✅ | ✅ | ❌ | ❌ | ❌ |

**Lý do:** Tốt hơn CD thuần túy vì nhấn mạnh các điểm giao nhau quan trọng trong line art. Chưa được implement sẵn trong bất kỳ model nào, cần bổ sung.

---

### 2.3 LPIPS (Learned Perceptual Image Patch Similarity)
**Định nghĩa:** Đo sự khác biệt perceptual giữa 2 ảnh bằng cách so sánh deep feature map (VGG/AlexNet) được pretrain:
$$LPIPS(x, x_0) = \sum_l \frac{1}{H_l W_l}\sum_{h,w}\|w_l \odot (\hat{y}^l_{hw} - \hat{y}^l_{0,hw})\|^2_2$$

| AnimeInbet | tps-inbetween | AnimeInterp | UPR-Net | uniVIP |
|:---:|:---:|:---:|:---:|:---:|
| ⚠️ | ⚠️ | ✅ | ✅ | ✅ |

**Lý do:** Chuẩn mực cho video interpolation RGB. Với line art, network pretrain không tối ưu cho ảnh nhị phân nên kém chính xác hơn, nhưng vẫn có thể dùng để bổ sung góc nhìn perceptual.

---

### 2.4 Earth Mover's Distance (EMD)
**Định nghĩa:** Còn gọi là Wasserstein Distance. Đo chi phí tối thiểu để "vận chuyển" phân phối xác suất $P$ thành $Q$:
$$EMD(P,Q) = \min_{\gamma \in \Pi(P,Q)} \mathbb{E}_{(x,y)\sim\gamma}[\|x - y\|]$$

| AnimeInbet | tps-inbetween | AnimeInterp | UPR-Net | uniVIP |
|:---:|:---:|:---:|:---:|:---:|
| ⚠️ | ⚠️ | ❌ | ❌ | ❌ |

**Lý do:** Có thể dùng cho line art để đo sự khớp point cloud tương tự CD, nhưng độ phức tạp tính toán rất cao (O(n³)) và ít precedent trong literature. Thực tiễn kém, chỉ nên dùng nếu có lý do rõ ràng.

---

### 2.5 RRLD (Restricted Relative Linear Discrepancy)
**Định nghĩa:** Đo sự sai lệch tương đối của output so với ground truth trên một tập hạn chế (restricted set), thường được dùng trong các bài toán morphing/interpolation có ràng buộc hình học.

| AnimeInbet | tps-inbetween | AnimeInterp | UPR-Net | uniVIP |
|:---:|:---:|:---:|:---:|:---:|
| ⚠️ | ⚠️ | ❌ | ❌ | ❌ |

**Lý do:** Metric ít phổ biến, không có implementation chuẩn. Có thể phù hợp với line art nếu được định nghĩa rõ ràng trong context của bài toán, nhưng cần implement từ đầu.

---

### 2.6 PSNR (Peak Signal-to-Noise Ratio)
**Định nghĩa:** Đo tỉ lệ tín hiệu-nhiễu tính theo thang logarithm, dựa trên MSE:
$$PSNR = 10 \cdot \log_{10}\left(\frac{MAX_I^2}{MSE}\right)$$
Đơn vị: dB. Giá trị càng cao càng tốt.

| AnimeInbet | tps-inbetween | AnimeInterp | UPR-Net | uniVIP |
|:---:|:---:|:---:|:---:|:---:|
| ❌ | ⚠️ | ✅ | ✅ | ✅ |

**Lý do:** Chỉ số cơ bản của video interpolation, UPR-Net và uniVIP đều report trên Vimeo90K/UCF101/SNU-FILM. Với tps-inbetween, PSNR có thể dùng được trên các subfolder có output đầy đủ.

> **Lưu ý sau quan sát:** AnimeInbet output cực kỳ thưa (~34KB). Ảnh sparse skeleton không phải pixel-dense image nên PSNR/SSIM trên output này hoàn toàn **vô nghĩa** — đổi thành ❌.

---

### 2.7 SSIM (Structural Similarity Index)
**Định nghĩa:** Đo sự tương đồng về luminance, contrast và structure giữa 2 ảnh:
$$SSIM(x,y) = \frac{(2\mu_x\mu_y+c_1)(2\sigma_{xy}+c_2)}{(\mu_x^2+\mu_y^2+c_1)(\sigma_x^2+\sigma_y^2+c_2)}$$
Giá trị trong $[-1, 1]$. Giá trị càng cao càng tốt.

| AnimeInbet | tps-inbetween | AnimeInterp | UPR-Net | uniVIP |
|:---:|:---:|:---:|:---:|:---:|
| ❌ | ⚠️ | ✅ | ✅ | ✅ |

**Lý do:** Cùng bộ tiêu chuẩn với PSNR cho video interpolation. SSIM quan tâm đến cấu trúc nên với tps-inbetween (`No Color`) cũng có giá trị nhất định. Với AnimeInbet, vì output là sparse skeleton nên SSIM hoàn toàn không áp dụng được — đổi thành ❌.

---

### 2.8 IE (Interpolation Error)
**Định nghĩa:** Mean Absolute Error (hoặc MSE) giữa frame được interpolate và ground truth tại vị trí pixel:
$$IE = \frac{1}{HW}\sum_{i,j}|I_{pred}(i,j) - I_{gt}(i,j)|$$

| AnimeInbet | tps-inbetween | AnimeInterp | UPR-Net | uniVIP |
|:---:|:---:|:---:|:---:|:---:|
| ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ |

**Lý do:** Khái niệm đơn giản, áp dụng được cho tất cả nhưng không phổ biến trong literature của các model này. Cần thống nhất định nghĩa chính xác trước khi dùng (MAE hay MSE?).

---

### 2.9 FVD (Fréchet Video Distance)
**Định nghĩa:** Tương tự FID nhưng cho video. Dùng I3D (Inflated 3D ConvNet) pretrain trên Kinetics để extract feature, rồi tính Fréchet Distance giữa phân phối video sinh và video thật:
$$FVD = \|\mu_r - \mu_g\|^2 + Tr(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})$$

| AnimeInbet | tps-inbetween | AnimeInterp | UPR-Net | uniVIP |
|:---:|:---:|:---:|:---:|:---:|
| ❌ | ❌ | ⚠️ | ✅ | ⚠️ |

**Lý do:** Metric sequence-level, cần nhiều frame liên tiếp và I3D pretrain trên video tự nhiên. Vô nghĩa với line art vì I3D không hiểu domain nhị phân.

> **Lưu ý sau quan sát:** **UPR-Net là model duy nhất** có sẵn `test_sequence_output/` với 23 frame + `interpolated_video.mp4` — FVD áp dụng được ngay. Với uniVIP và AnimeInterp, cần generate sequence thêm mới đo được FVD (đổi sang ⚠️).

---

### 2.10 Warping Error / Temporal Stability
**Định nghĩa:** Warp frame $I_t$ sang $I_{t+1}$ bằng optical flow $F_{t \to t+1}$, rồi đo residual với frame thật $I_{t+1}$:
$$WE = \frac{1}{T}\sum_t \|I_{t+1} - \mathcal{W}(I_t, F_{t \to t+1})\|_1 \cdot M_t$$
Trong đó $M_t$ là occlusion mask.

| AnimeInbet | tps-inbetween | AnimeInterp | UPR-Net | uniVIP |
|:---:|:---:|:---:|:---:|:---:|
| ❌ | ❌ | ⚠️ | ✅ | ⚠️ |

**Lý do:** Đo flickering/temporal artifacts giữa các frame liên tiếp — artifact đặc trưng của video interpolation. Không áp dụng được cho line art vì output không phải video RGB liên tục.

> **Lưu ý sau quan sát:** UPR-Net có sẵn sequence output nên Warping Error áp dụng được ngay. AnimeInterp và uniVIP cần generate sequence trước (đổi sang ⚠️). Tất cả 4 model (trừ AnimeInbet) đều có test case **Rotate** — có thể so sánh temporal stability trên large motion test này.

---

### 2.11 Hausdorff Distance
**Định nghĩa:** Đo "worst-case" khoảng cách nearest-neighbor giữa 2 point set — tức khoảng cách tối đa:
$$H(P,Q) = \max\left(\max_{p \in P}\min_{q \in Q}\|p-q\|, \max_{q \in Q}\min_{p \in P}\|q-p\|\right)$$

| AnimeInbet | tps-inbetween | AnimeInterp | UPR-Net | uniVIP |
|:---:|:---:|:---:|:---:|:---:|
| ✅ | ✅ | ❌ | ❌ | ❌ |

**Lý do:** Bổ sung tốt cho CD (CD đo trung bình, Hausdorff đo worst-case). Phát hiện các đường nét bị "lạc" hoàn toàn — điều CD có thể bỏ qua. Chỉ có ý nghĩa với line art dạng point cloud.

---

### 2.12 Line IoU (Intersection over Union)
**Định nghĩa:** Tính IoU trên đường nét sau khi dilate một kernel nhỏ (tolerance $d$ pixels) để có sự linh hoạt về vị trí:
$$LineIoU = \frac{|dilate(P_d) \cap dilate(Q_d)|}{|dilate(P_d) \cup dilate(Q_d)|}$$

| AnimeInbet | tps-inbetween | AnimeInterp | UPR-Net | uniVIP |
|:---:|:---:|:---:|:---:|:---:|
| ✅ | ✅ | ❌ | ❌ | ❌ |

**Lý do:** Trực quan, dễ hiểu: "bao nhiêu % đường nét khớp với ground truth". Bổ sung cho CD/Hausdorff một góc nhìn về coverage. Hoàn toàn domain-specific cho line art.

---

### 2.13 Disconnection Rate
**Định nghĩa:** Tỷ lệ đường nét bị đứt/rời rạc trong output so với ground truth. Phát hiện bằng cách so sánh số lượng connected component hoặc endpoint:
$$DR = \frac{\text{số đoạn đứt trong } I_{pred}}{\text{số đường nét trong } I_{gt}}$$

| AnimeInbet | tps-inbetween | AnimeInterp | UPR-Net | uniVIP |
|:---:|:---:|:---:|:---:|:---:|
| ✅ | ✅ | ❌ | ❌ | ❌ |

**Lý do:** Đo artifact đặc trưng nhất của line inbetweening: đường nét sinh ra bị đứt giữa chừng. Metric domain-specific, không có ý nghĩa với video RGB.

> **Lưu ý sau quan sát:** AnimeInbet output đã rất thưa (sparse skeleton), nên Disconnection Rate có thể phản ánh ngay trong output — metric này thực chất đo được chất lượng của quá trình graph fusion. tps-inbetween `No Color` subfolder là test case lý tưởng.

---

### 2.14 Arbitrary t Support
**Định nghĩa:** Khả năng model sinh frame tại bất kỳ thời điểm $t$ nào, không chỉ $t=0.5$. Với uniVIP, $t$ có thể vượt ra ngoài $[0,1]$ (để prediction).

| AnimeInbet | tps-inbetween | AnimeInterp | UPR-Net | uniVIP |
|:---:|:---:|:---:|:---:|:---:|
| ❌ | ✅ (custom t) | ❌ | ✅ (`time_period`) | ✅ (thiết kế chính, t > 1) |

**Lý do:** Feature capability test, không phải accuracy metric. Quan trọng khi đánh giá tính linh hoạt của model trong ứng dụng thực tế (slow-motion, frame rate conversion).

> **Lưu ý sau quan sát:** Demo script của tps-inbetween dùng `parse_time_steps()` hỗ trợ custom t thực sự. AnimeInbet cũng có hàm tương tự nhưng chỉ chấp nhận t ∈ [0,1] và output là graph interpolation cố định, không phải continuous synthesis như các model còn lại.

## 3. Bảng Tổng Hợp

| Chỉ số | AnimeInbet | tps-inbetween | AnimeInterp | UPR-Net | uniVIP |
|---|:---:|:---:|:---:|:---:|:---:|
| Chamfer Distance (CD) | ✅ | ✅ | ❌ | ❌ | ❌ |
| Weighted CD (WCD) | ✅ | ✅ | ❌ | ❌ | ❌ |
| LPIPS | ❌ | ⚠️ | ✅ | ✅ | ✅ |
| EMD | ⚠️ | ⚠️ | ❌ | ❌ | ❌ |
| RRLD | ⚠️ | ⚠️ | ❌ | ❌ | ❌ |
| PSNR | ❌ | ⚠️ | ✅ | ✅ | ✅ |
| SSIM | ❌ | ⚠️ | ✅ | ✅ | ✅ |
| IE | ❌ | ⚠️ | ⚠️ | ⚠️ | ⚠️ |
| FVD | ❌ | ❌ | ⚠️ | ✅ | ⚠️ |
| Warping Error | ❌ | ❌ | ⚠️ | ✅ | ⚠️ |
| Hausdorff Distance | ✅ | ✅ | ❌ | ❌ | ❌ |
| Line IoU | ✅ | ✅ | ❌ | ❌ | ❌ |
| Disconnection Rate | ✅ | ✅ | ❌ | ❌ | ❌ |
| Arbitrary t Support | ❌ | ✅ | ❌ | ✅ | ✅ |
| Inference Speed | ✅ | ✅ | ✅ | ✅ | ✅ |
| VRAM Peak | ✅ | ✅ | ✅ | ✅ | ✅ |
| Fine-tuning Speed | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ |

> **Chú thích:** ✅ Phù hợp &nbsp;|&nbsp; ⚠️ Cần cân nhắc / có điều kiện &nbsp;|&nbsp; ❌ Không phù hợp
>
> **Thay đổi so với phiên bản cũ (trước quan sát output):** AnimeInbet đổi PSNR/SSIM/LPIPS/IE từ ⚠️ → ❌ do output là sparse skeleton. FVD/Warping Error của AnimeInterp và uniVIP đổi từ ✅ → ⚠️ (cần generate sequence trước). UPR-Net giữ ✅ cho FVD/Warping Error vì đã có sẵn video sequence output.

---

## 4. Tập Chỉ Số Khuyến Nghị

### Nhóm Line Inbetweening (AnimeInbet, tps-inbetween)
| Ưu tiên | Chỉ số | Lý do |
|---|---|---|
| **Core** | CD + Hausdorff Distance | Đo sự khớp hình học. Đặc biệt phù hợp AnimeInbet sparse skeleton |
| **Core** | Disconnection Rate | Phát hiện đường nét bị đứt, artifact đặc trưng nhất |
| **Core (tps-inbetween)** | Line IoU | Chỉ dùng với tps-inbetween `No Color` — output hoàn chỉnh hơn AnimeInbet |
| **Không nên dùng** | PSNR / SSIM (cho AnimeInbet) | Vô nghĩa với sparse skeleton |
| **Efficiency** | Inference Speed + VRAM Peak | Khả năng deploy thực tế |

### Nhóm Video Interpolation (AnimeInterp, UPR-Net, uniVIP)
| Ưu tiên | Chỉ số | Lý do |
|---|---|---|
| **Core** | PSNR + SSIM + LPIPS | Bộ ba tiêu chuẩn trong toàn bộ literature |
| **Core (UPR-Net ngay)** | Warping Error + FVD | UPR-Net áp dụng trực tiếp (có sẵn MP4). AnimeInterp/uniVIP cần generate sequence thêm |
| **Shared test** | Rotate case (4 model) | Cả 4 model (trừ AnimeInbet) có cùng large motion test — so sánh trực tiếp |
| **Feature** | Arbitrary t Support | Thứ tự: uniVIP (t>1) > UPR-Net > tps-inbetween > AnimeInterp ≈ AnimeInbet |
| **Efficiency** | Inference Speed + VRAM Peak | Khả năng deploy thực tế |
