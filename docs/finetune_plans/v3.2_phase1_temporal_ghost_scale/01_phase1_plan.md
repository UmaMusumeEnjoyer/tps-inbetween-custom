# KẾ HOẠCH HUẤN LUYỆN PHA 1 (V3.2 — PHASE 1): DIỆT BÓNG MỜ (`2A`), ỔN ĐỊNH CHUỖI THỜI GIAN (`2C`) & CO GIÃN HÌNH HỌC

> **Nền tảng khởi tạo (Warm-Start):** [`model_tps_finetuned_last.pt`](file:///E:/CODE/PBL/PBL6/checkpoints/Re-train/Sequence%202/checkpoints/model_tps_finetuned_last.pt) (Epoch 15 EMA của `Sequence 2`) hoặc [`model_latest.pt`](file:///E:/CODE/PBL/PBL6/checkpoints/pretrained_baselines/tps-original/model_latest.pt) (`TPSInbetweenOriginal` — `10,068,668` active params, giữ nguyên kiến trúc gốc ở Pha 1 để đo tách bạch tác động của Loss, Augmentation & Dữ liệu trước khi sang Pha 2).
> **Mô hình đối chuẩn Baseline:** [`model_latest.pt`](file:///E:/CODE/PBL/PBL6/checkpoints/pretrained_baselines/tps-original/model_latest.pt) (`M1_Original`).
> **Cơ sở thực nghiệm trực tiếp:**
> - Kết quả [`Sequence 2`](file:///E:/CODE/PBL/PBL6/checkpoints/Re-train/Sequence%202): khóa chuẩn `Density Ratio = 0.9866 / 0.9688` và thắng `M1` trên `WCD_DoG`, nhưng còn `Ghost = 8,190 px` (Bench #1) / `20,796 px` (Bench #2) và `SkelRec = 57.80%` trên Bench #2.
> - Kết quả kiểm thử đối chứng 2 hình tròn (`Case 1–3`): `Hybrid Contour + Lagrangian Direct TPS` đưa `WCD_DoG` từ `12.085 -> 0.767` (`F1 = 0.999`), nhưng lộ rõ `Seq2 Ep15` bị tăng độ dày nét (`Dens = 1.274–1.360`) khi gặp phóng/thu $> 2\times$.

---

## 1. Các Lỗi Kỹ Thuật Cần Vá Ngay Ở Pha 1

1. **Vá lỗi pha loãng gradient trong `GhostAndBackgroundCleanlinessLoss`:**
   - Trong phiên bản `Sequence 2`, `loss_far` chia tổng mức phạt trên nền trắng cho `far_count` ($\approx 480,000\text{ px}$ ở 540p), làm tín hiệu phạt bị thu nhỏ $\sim 500,000\times$ ($\approx 0.0004$ trong `total_loss`), khiến `Ghost` tăng từ `2,736 -> 7,583 px` suốt 15 epochs.
   - **Giải pháp:** Chuẩn hóa `loss_far` theo diện tích nét Ground Truth `clamp(sum(gt > 0.15), min=500.0)` và nâng `w_ghost` từ `0.20` lên `0.45`.
2. **Sửa `gate_ghost` và `Composite Score` tránh khóa checkpoint ở Epoch 1:**
   - `Sequence 2` khóa `gate_ghost <= 3,200 px` khiến `model_tps_finetuned_best.pt` dừng cập nhật sau Epoch 1.
   - **Giải pháp:** Đặt `ghost_limit = max(6500.0, min(0.45 * m0_ghost, base_ghost * 4.8))`, tăng trọng số `term_ghost` trong `s_val` từ `0.12 -> 0.18` (trọng số `term_wcd = 0.28`, `term_skel = 0.20`, `term_dens = 0.18`, `term_semd = 0.08`, `term_tip = 0.08`), và dùng phạt tỷ lệ liên tục ở Tier 2 fallback.
3. **Bổ sung Ràng buộc Nhất quán Thời gian ($\mathcal{L}_{\text{temp}}$) cho bài toán `2C` (Rung/nhấp nháy video):**
   - Trong Mode `LONG` trên chuỗi $N \in \{5, 7\}$, bổ sung sai phân liên tiếp giữa hai frame kề nhau $(k, k+1)$ cùng chuỗi:
     $$\mathcal{L}_{\text{temp}} = \text{Charbonnier}\big((\hat{I}_{k+1} - \hat{I}_k) - (I^{\text{GT}}_{k+1} - I^{\text{GT}}_k)\big) + 0.5 \cdot \frac{\sum |\hat{I}_{k+1} - \hat{I}_k| \cdot \mathbb{I}(I^{\text{GT}}_k < 0.04 \land I^{\text{GT}}_{k+1} < 0.04)}{\max(500, \bar{A}_{\text{gt}})}$$
     với trọng số `w_temp = 0.50`, giúp quỹ đạo nét biến thiên mượt và triệt tiêu nhấp nháy nền giữa các frame kế tiếp mà **không tốn thêm VRAM** (tính trong chunk và qua ranh giới chunk bằng frame biên detach).
4. **Bổ sung Paired Spatial Augmentation + 8% Synthetic Geometric Primitives (Lagrangian Direct TPS):**
   - Trên tập thật `final_dataset`: thêm **Paired H-Flip ($p=0.50$)** và **Paired V-Flip ($p=0.25$)** đồng bộ cả ảnh $(I_0, I_t, I_1)$ lẫn vector dòng chảy `(cf01, cf10)` ngay trên GPU bên trong vùng `orig_hw`.
   - Trộn thêm **`120` chuỗi hình học tổng hợp mỗi epoch** ($N \in \{3, 5, 7\}$, `540×960`, gồm hình tròn/elip dịch xa `60–420 px` và co giãn `0.40× – 2.25×`, hình tròn + đường nền tĩnh, cung cong xoay, cặp 2 hình tròn độc lập) dùng **Lagrangian Direct TPS (`direct_t_flows=True`)** để dạy mạng:
     - Cân bằng lại độ dày nét khi co giãn kích thước lớn (`Dens -> 1.00`),
     - Khử bóng mờ lệch mép khi vật thể bay ngang qua vật thể tĩnh.

---

## 2. Nâng cấp Bộ Benchmark Đối Chứng (Dual -> Triple Benchmark)

Cập nhật [`run_benchmark_suite.py`](file:///E:/CODE/PBL/PBL6/Dataset/sakuga_pipeline_sequence/final_dataset/sakuga_golden_benchmark_sequences_v2/run_benchmark_suite.py):
1. **Benchmark #1:** Tập chuỗi cách ly 198 clips / 412 frames (giữ nguyên chuẩn công bằng).
2. **Benchmark #2:** Chuỗi anime liên tục `data/lineart_experiments/test` (chạy thuần **FP32**, bổ sung cột đo **Temporal Consistency**: `temp_diff_mae` và `temp_bg_flicker` trên 17 chặng chuyển tiếp).
3. **Benchmark #3 (`Defense Geometry Suite`):** Bộ test đối chứng 4 kịch bản hình học (Tròn thu nhỏ `75->32`, Tròn phóng to `30->70`, Tròn thu nhỏ + nền tĩnh, Elip biến dạng $\to$ Tròn) dùng **Hybrid Contour Resampling + Lagrangian Direct TPS**, xuất CSV + lưới đối chiếu trực quan.
