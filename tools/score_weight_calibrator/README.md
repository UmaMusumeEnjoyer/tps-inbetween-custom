# HƯỚNG DẪN SỬ DỤNG: SCORE WEIGHT CALIBRATOR

> **Ứng dụng Web Hiệu Chỉnh Trọng Số Hàm Score cho Mô Hình TPS-Inbetween**  
> Khắc phục hiện tượng bão hòa điểm số tại ngưỡng $60 - 62$ trong các phiên bản v0.1 đến v1.0.

---

## 1. Mục Đích & Bối Cảnh Kỹ Thuật

Trong quá trình huấn luyện mô hình TPS-Inbetween (từ v0.1 đến v1.0), hàm Score đánh giá checkpoint tối ưu được định nghĩa:
$$\text{Score}_{v1.0} = 1.0 \times \text{PSNR} + 20.0 \times \text{SSIM} + 30.0 \times \text{Recall} - 0.0005 \times \text{GhostPixels}$$

Do đặc tính của nét vẽ hoạt họa Anime 2D:
* $\text{SSIM}$ và $\text{Stroke Recall}$ nhanh chóng tiệm cận mức bão hòa ($>0.91$).
* $\text{PSNR}$ nét thưa trên nền trắng hiếm khi vượt quá $25\text{ dB}$.
* Dẫn đến điểm tổng hợp tự nhiên bị khóa quanh mức trung bình **55.98** và hầu như không bao giờ vượt qua mốc **61 - 62** điểm cho dù tiếp tục tăng epoch.

Công cụ **Score Weight Calibrator** sử dụng phương pháp **Human-in-the-Loop** để người dùng đánh giá trực quan 100 bộ triplet thực tế từ Sakuga Golden Dataset, sau đó thuật toán tự động khớp bộ trọng số mới thỏa mãn cảm quan của họa sĩ và trải đều trên thang điểm chuẩn $0 - 100$.

---

## 2. Cách Khởi Chạy Ứng Dụng

Mở Terminal và chạy lệnh:
```bash
python tools/score_weight_calibrator/run_calibrator.py
```
Ứng dụng sẽ tự động kích hoạt local HTTP server tại cổng `8080` (hoặc cổng trống tiếp theo) và mở trang web trên trình duyệt mặc định:
```text
http://127.0.0.1:8080/index.html
```

---

## 3. Các Tính Năng Chính Trên Giao Diện

1. **Khung Chiếu Trực Quan (Interactive Viewer):**
   * Hiển thị sắc nét **Frame thứ 2** (`000002.png` - Inbetween Ground Truth).
   * Tab chuyển đổi nhanh giữa: `Frame 1 (I₀)`, `Frame 2 (GT)`, `Frame 3 (I₁)`.
   * **Chế độ Flipbook (Phím Space):** Chạy vòng lặp chuyển động 3 khung hình ở tốc độ 10-12fps để cảm nhận độ mượt và độ giật của nét vẽ như khi xem phim hoạt hình thực tế.
2. **Hệ Thống 4 Lựa Chọn Đánh Giá (Phím 1, 2, 3, 4):**
   * `[1]` ❌ **Không** (25 điểm): Hình hỏng, biến dạng nét, rác nhiều.
   * `[2]` ⚠️ **Tạm chấp nhận** (50 điểm): Nét còn đứt gãy, ghosting nhẹ.
   * `[3]` 🆗 **Bình thường** (75 điểm): Đạt chuẩn chuyển động anime trung bình.
   * `[4]` ⭐ **Tốt** (100 điểm): Nét sạch, biến dạng hoàn hảo, chuẩn Golden.
3. **Bảng Chỉ Số Động (Live Metrics Panel):**
   * Hiển thị tức thời PSNR, SSIM, Recall, Ghost Pixels, Độ dời (Disp) và điểm Score v1.0 cũ.
4. **Bộ Điều Hướng 100 Triplets:**
   * Lưới 100 ô nhảy nhanh (Jump Grid) hiển thị mã màu tương ứng với lựa chọn đã đánh giá.
   * Tự động lưu tiến độ vào `localStorage` của trình duyệt, không lo mất dữ liệu khi tải lại trang.
5. **Nút "⚡ Demo Nạp Mẫu":**
   * Cho phép tự động điền mẫu đánh giá giả lập theo phân phối chất lượng để bạn có thể xem trước ngay bảng kết quả tính toán trọng số mà không cần bấm thủ công 100 lần.

---

## 4. Thuật Toán Tối Ưu Hóa Trọng Số

Ứng dụng áp dụng giải thuật **Constrained Least Squares Regression (Projected Gradient Descent - PGD)**:
$$\min_{w, b} \frac{1}{N} \sum_{i=1}^{100} \left( \hat{Y}_i - Y_i \right)^2 + \lambda \|w\|_2^2$$
với các ràng buộc vật lý tự nhiên:
* $w_{\text{psnr}} \ge 0, \quad w_{\text{ssim}} \ge 0, \quad w_{\text{recall}} \ge 0$ (Chỉ số chất lượng tăng $\to$ Điểm tăng).
* $w_{\text{ghost}} \ge 0$ (Được trừ đi trong công thức để trừng phạt nét ma: $- w_{\text{ghost}} \times \text{Ghost}$).
* Hệ số tự do $b$ giúp neo thang điểm chính xác vào khoảng $25 - 95$ điểm.

---

## 5. Cập Nhật Vào Mã Nguồn Huấn Luyện

Sau khi đánh giá hoàn tất, ứng dụng sẽ cung cấp sẵn đoạn mã Python để bạn thay thế vào **Cell 9** của notebook [`kaggle/04_train_tps_inbetween_finetune_v1.0.ipynb`](file:///e:/CODE/PBL/PBL6/kaggle/04_train_tps_inbetween_finetune_v1.0.ipynb):

```python
score = (
    w_psnr * psnr +
    w_ssim * ssim +
    w_recall * recall -
    w_ghost * ghost +
    bias
)
```
Mô hình sẽ không còn bị kẹt ở điểm số 60 - 62, mà phân hóa rõ rệt các checkpoint đạt chuẩn cao ($>80 - 90$ điểm).
