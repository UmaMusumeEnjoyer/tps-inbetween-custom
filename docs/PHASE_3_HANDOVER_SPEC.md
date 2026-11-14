# TÀI LIỆU KẾ THỪA DỰ ÁN & KẾ HOẠCH TRIỂN KHAI PHA 3
## ĐỀ TÀI: NỘI SUY KHUNG HÌNH HOẠT HỌA ANIME LINEART VỚI COARSE TPS ĐA LỚP
*Tài liệu dành cho phiên làm việc mới (Handover Document for New Session)*  
*Thời gian cập nhật: 12/11/2026*

---

## 1. TỔNG QUAN HỆ THỐNG & KIẾN TRÚC MÔ HÌNH
* **Mô hình cốt lõi:** `TPSInbetweenOriginal` (~10.07M tham số).
  * **Đặc tả:** Mô hình nhận vào cặp ảnh lineart ($I_0, I_1$), tham số thời gian $t \in [0, 1]$, và hai trường dòng chảy TPS định hướng thô ($F_{0 \to t}, F_{1 \to t}$) sinh bởi thuật toán Thin Plate Spline (TPS) từ các điểm đặc trưng (GlueStick / Contour Points).
  * **Cấu trúc:** Gồm `IFNet` (Multi-scale Feature Warping Pyramid) + `OriginalUnet` (Decoder tinh chỉnh viền nét).
* **Môi trường cục bộ (Local PC):**
  * GPU: NVIDIA GeForce RTX 3050 Laptop GPU (CUDA).
  * Thư mục dự án: `E:\CODE\PBL\PBL6`
  * Python Venv: `E:\CODE\PBL\PBL6\.venv\Scripts\python.exe`
  * Dữ liệu benchmark chuẩn:
    * Chuỗi thực tế: `E:\CODE\PBL\PBL6\data\lineart_experiments\test` (24 frames 540p liên tục).
    * Bộ test hình học: `E:\CODE\PBL\PBL6\continuous_benchmark\test_circle_geometry` (Case 1, Case 2, Case 3).

---

## 2. TÓM TẮT MỤC TIÊU CÁC PHA VỪA QUA

| Giai đoạn | Mục tiêu chính | Trọng tâm kỹ thuật | Kết quả đạt được |
| :--- | :--- | :--- | :--- |
| **Pha 1** | Tối ưu hóa hàm Loss & Độ dày nét vẽ ($1B$) | `ExactSTEDensityLoss`, `GhostLoss` có gate động, `Paired Spatial Augmentation`, `TemporalLoss` trên cửa sổ 5-7 frames. | Nét vẽ đậm đà, giữ trọn ngọn tóc mảnh ($\text{SkelRec} = 86.1\%$, $\text{Density} = 0.987$). Tuy nhiên vẫn còn hiện tượng nhấp nháy nền ở chuỗi dài. |
| **Pha 2** | Triệt tiêu rung giật video ($2C$) & bóng mờ ($2A$) | Tăng cường `TemporalConsistencyLoss` trên các patch liên tục, tăng trọng số phạt bóng mờ nền `w_ghost = 1.10`. | Nền phẳng lì, khử nhấp nháy cực mạnh ($\text{Flicker}$ giảm tới $69\%$, $\text{Ghost}$ giảm $27\%$). Đổi lại: nét vẽ bị co mỏng nhẹ ($\text{Density} = 0.914$, $\text{SkelRec} = 82.7\%$). |
| **Pha 2 Blended** *(Model Soup)* | Dung hòa trọn vẹn điểm mạnh của Pha 1 & Pha 2 | Hòa trộn trọng số tuyến tính: $\mathbf{W}_{\text{blend}} = 0.55 \cdot \mathbf{W}_{\text{P1}} + 0.45 \cdot \mathbf{W}_{\text{P2}}$ trên toàn bộ 158 tensors. | **Cân bằng tối ưu:** Vừa giữ được nét đậm đà chuẩn mực của P1 ($\text{Density} \approx 1.02 - 1.10$, $\text{PSNR} > 27\text{dB}$), vừa giảm rung giật nền và sương mờ của P2. |
| **Pha 3** *(Mục tiêu sắp tới)* | **Tầng Coarse TPS Đa lớp (`M0`) + Lọc Curriculum** | Phân cụm chuyển động (Motion Segmentation) tách Keypoints Tĩnh và Động $\to$ Giải TPS 2 lớp độc lập. | **Xóa bỏ triệt để hiện tượng kéo lê méo nền tĩnh** khi nhân vật lướt qua; triệt tiêu hoàn toàn sương mờ giao thoa ranh giới (Occlusion Boundary). |

---

## 3. CÁC CHECKPOINT HIỆN CÓ & SỐ LIỆU ĐỊNH LƯỢNG MẤU CHỐT

### 3.1. Vị trí lưu trữ Checkpoints
1. **Pha 1 Last Checkpoint:**
   `E:\CODE\PBL\PBL6\checkpoints\Re-train\Sequence 2 pha1\checkpoints\model_tps_finetuned_last.pt`
2. **Pha 2 Last Checkpoint:**
   `E:\CODE\PBL\PBL6\checkpoints\Re-train\Sequence 2 pha2\checkpoints\model_tps_finetuned_last.pt`
3. **Blended Model Checkpoint (Mô hình tốt nhất hiện nay):**
   `E:\CODE\PBL\PBL6\checkpoints\Re-train\Sequence 2 Blended\checkpoints\model_tps_blended.pt`

### 3.2. Bảng so sánh định lượng đối đầu trên 17 in-between frames thực tế (24 frames 540p)
| Chỉ số | M1 Baseline | Pha 1 Last | Pha 2 Last | 🌟 Blended Model (0.55 P1 + 0.45 P2) |
| :--- | :---: | :---: | :---: | :---: |
| **PSNR** | 15.17 dB | 15.33 dB | **15.68 dB** | **15.53 dB** |
| **IE_MAE** (Sai số màu) | 16.59 | 15.76 | **13.93** | **14.99** |
| **Ghost Pixels** (px/frame) | 26,104 px | 36,805 px | **22,604 px** | **32,087 px** (giảm 4,718 px vs P1) |
| **Temp Diff MAE** | 22.38 | 18.69 | **17.41** | **18.24** |
| **Temp BG Flicker** | 10,210 | 4,272 | **2,246** | **3,611** (rất êm) |
| **Bảo toàn hình học (Circle $F_1$)** | 1.0000 | 1.0000 | 1.0000 | **1.0000 (Sai số tâm $\le 1.4\text{px}$)** |

---

## 4. VẤN ĐỀ CỐT LÕI CẦN GIẢI QUYẾT Ở PHA 3

### Hạn chế vật lý của Single-layer Global TPS
* **Bằng chứng thực nghiệm (Case 3 - Hình tròn co giãn trên nền mặt đất tĩnh):**
  * Khi hình tròn co nhỏ từ $75\text{px} \to 32\text{px}$ và dịch chuyển ngang $400\text{px}$, đường kẻ đất ở dưới hoàn toàn đứng yên.
  * Vì thuật toán `solve_tps_backward_pair` hiện tại giải một hệ phương trình TPS toàn cục duy nhất, năng lượng uốn (bending energy) của các điểm di động trên hình tròn lan tỏa sang cả các điểm đứng yên trên đường kẻ đất.
  * Hậu quả: Đường kẻ đất tĩnh bị **uốn cong nhẹ lên trên**, và tạo ra một vệt sương mờ xám ở vùng giao thoa giữa chuyển động và tĩnh tại.
* **Mạng neural không thể tự sửa nếu flow thô ban đầu đã sai:** Mạng học sâu cố warp ảnh theo flow sai, dẫn đến việc phải trade-off giữa xóa mờ và giữ nét.

---

## 5. KẾ HOẠCH CHI TIẾT THỰC HIỆN PHA 3 (COARSE TPS ĐA LỚP)

### Bước 1: Thiết kế Module Phân Cụm Điểm Chuyển Động (Motion Segmentation)
* **Ý tưởng:** Với tập điểm đối sánh $P_0 \leftrightarrow P_1$ thu được từ GlueStick/Contour:
  1. Tính vector vận tốc tương đối: $\vec{v}_i = P_{1,i} - P_{0,i}$.
  2. Phân lập cụm tĩnh: Các điểm có $\|\vec{v}_i\| < \epsilon_{\text{static}}$ (ví dụ $\le 1.5\text{ px}$) được gán vào **Cụm Nền (Static Layer)**.
  3. Phân cụm động: Sử dụng thuật toán phân cụm không gian - vận tốc (Spatial-Velocity DBSCAN hoặc RANSAC Homography) để nhóm các điểm thuộc cùng một vật thể chuyển động vào **Cụm Tiền Cảnh (Foreground Layer)**.

### Bước 2: Thiết kế Thuật toán Multi-Layer TPS Flow Solver
Thay vì sinh 1 trường flow duy nhất, hàm `solve_multi_layer_tps` sẽ:
1. Giải $TPS_{\text{static}}$ với cụm điểm tĩnh $\implies$ Sinh dòng chảy tĩnh $F_{\text{static}}$ (gần như là identity/zero flow).
2. Giải $TPS_{\text{dynamic}}$ với cụm điểm động $\implies$ Sinh dòng chảy động $F_{\text{dynamic}}$.
3. Tạo mặt nạ mềm (Soft Blending Mask $M$) dựa trên khoảng cách Hausdorff/Voronoi tới các điểm động:
   $$F_{\text{fused}} = M \odot F_{\text{dynamic}} + (1 - M) \odot F_{\text{static}}$$
4. Đảm bảo tại vùng nền tĩnh, dòng chảy có đạo hàm bậc 2 bằng 0 (phẳng tuyệt đối).

### Bước 3: Đánh giá Offline (Inference-time Validation)
* **Thực thi trên Local PC:** Sử dụng ngay mô hình `model_tps_blended.pt` mà **không cần train lại ngay lập tức**.
* Nạp $F_{\text{fused}}$ vào mô hình để chạy lại:
  * **Case 3 (Circle + Ground):** Kiểm tra đường thẳng tĩnh có phẳng lì $100\%$ hay không.
  * **Chuỗi 24 frames thực tế:** Đo đạc lại `Ghost Pixels` và `Temp BG Flicker`.

### Bước 4: Fine-tuning Nhẹ trên Kaggle (Nếu cần nâng cao)
* Nếu kết quả bước 3 chứng minh flow 2 lớp giúp ảnh sắc nét hơn, tạo bộ cache `tps_cache_multilayer.npz` trên tập Sakuga Golden Dataset và fine-tune 2-3 epoch trên Kaggle để mạng neural thích nghi hoàn toàn với đặc tính flow mới.

---

## 6. HƯỚNG DẪN KẾ THỪA CHO CỬA SỔ CHAT MỚI (HANDOVER PROMPT)

Khi bắt đầu một cửa sổ chat mới, người dùng chỉ cần gửi prompt sau:

> *"Tôi đang tiếp tục đề tài Anime Lineart Inbetweening PBL6. Hãy đọc file `E:\CODE\PBL\PBL6\docs\PHASE_3_HANDOVER_SPEC.md` để nắm toàn bộ bối cảnh kiến trúc `TPSInbetweenOriginal`, thành tựu của Pha 1, Pha 2, Pha 2 Blended, và tiến hành triển khai ngay **Bước 1 & Bước 2 của Pha 3: Xây dựng Module Coarse TPS Đa lớp (Multi-Layer Clustered TPS)** theo kế hoạch."*
