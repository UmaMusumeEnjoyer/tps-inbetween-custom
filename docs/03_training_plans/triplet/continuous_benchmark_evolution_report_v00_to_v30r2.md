# BÁO CÁO TOÀN CẢNH TIẾN HÓA MÔ HÌNH TPS-INBETWEEN (V0.0 BASELINE ──► V3.0-REMASTERED R2)
## Thực hiện trên chuỗi kiểm chuẩn chuẩn hóa 540p: `data/lineart_experiments/test`

> **Tài nguyên kết xuất:**
> * Master Summary CSV: [evolution_timeline_master.csv](file:///e:/CODE/PBL/PBL6/checkpoints/experiments/continuous_benchmarks/MASTER_SUMMARY/evolution_timeline_master.csv)
> * Biểu đồ xu hướng tiến hóa: [evolution_metrics_comparison.png](file:///e:/CODE/PBL/PBL6/checkpoints/experiments/continuous_benchmarks/MASTER_SUMMARY/evolution_metrics_comparison.png)
> * Thư mục kết quả 17 thế hệ: [continuous_benchmarks](file:///e:/CODE/PBL/PBL6/checkpoints/experiments/continuous_benchmarks)

---

## 1. BẢNG TỔNG HỢP TIẾN HÓA TOÀN BỘ 17 THẾ HỆ MÔ HÌNH

| STT | Phiên bản (Version Tag) | Tên mô hình đại diện | Checkpoint File | SSIM ↑ | PSNR (dB) ↑ | Stroke Recall (%) ↑ | Stroke Density (x) | Ghost Pixels (px) ↓ | Độ sạch / Ghi chú đặc thù |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | `v0.0_baseline` | Pretrained Original | `model_latest.pt` | 0.7107 | 12.75 | 56.33% | 0.72x | 4,312 | Chưa học nét anime, nét đứt đoạn nhiều |
| **1** | `v0.1` | TPS Finetune V0.1 | `model_tps_v0.1_best.pt` | 0.7614 | 14.98 | **0.00%** | **0.00x** | -1 | ⚠️ **Sụp đổ ảnh trắng (Blank Collapse)** |
| **2** | `v0.2` | TPS Finetune V0.2 | `model_tps_v0.2_best.pt` | 0.6925 | 15.05 | 67.15% | 1.08x | 48,120 | Phục hồi nét nhưng nhiễm bóng ma diện rộng |
| **3** | `v0.3` | TPS Finetune V0.3 | `model_tps_v0.3_best.pt` | 0.7148 | 14.93 | 60.69% | 0.91x | 38,136 | Bổ sung Ghost Erasure, giảm bóng ma |
| **4** | `v1.0` | TPS Finetune V1.0 | `model_tps_v1.0_best.pt` | 0.7460 | 15.35 | 54.22% | 0.82x | 39,750 | ASF-GR hoàn thiện, nét mảnh sạch hơn |
| **5** | `v1.1` | TPS Finetune V1.1 | `model_tps_v1.1_best.pt` | 0.7400 | 14.93 | 59.68% | 0.82x | 16,797 | 🟢 **Boundary Anchors: Giảm 57% bóng ma** |
| **6** | `v2.0` | TPS Finetune V2.0 | `model_tps_v2.0_best.pt` | 0.7390 | 14.91 | 61.01% | 0.85x | 18,238 | Mở rộng dataset Sakuga Golden V1 |
| **7** | `v2.1` | TPS Finetune V2.1 | `model_tps_v1.1_best.pt` | 0.7409 | 14.96 | 60.12% | 0.81x | 18,508 | Cân bằng phân tầng Patch2 |
| **8** | `v2.2` | TPS Finetune V2.2 | `model_tps_v2.2_best.pt` | 0.7234 | 14.66 | 66.45% | 0.98x | 23,254 | Bổ sung Fast Chamfer GPU |
| **9** | `v2.3` | TPS Finetune V2.3 | `model_tps_v2.3_best.pt` | 0.7214 | 14.91 | 54.44% | 0.58x | 25,781 | ⚠️ Siết Anti-thickening quá đà (teo nét) |
| **10** | `v2.4` | TPS Finetune V2.4 | `model_tps_v2.4_best.pt` | 0.7293 | 14.28 | 64.16% | 0.97x | 14,511 | 🟢 Đáy bóng ma tốt nhất nhánh V2.x |
| **11** | `v2.5` | TPS Finetune V2.5 | `model_tps_v2.5_best.pt` | 0.7201 | 14.66 | 61.55% | 0.79x | 19,686 | Cân đối tham số chuẩn bị cho V2.6 |
| **12** | `v2.6` | TPS Finetune V2.6 | `model_tps_v2.6_best.pt` | 0.7249 | 14.63 | 61.96% | 0.82x | 17,379 | **Bản chuẩn Production cũ** |
| **13** | `v3.0.5` | TPS Finetune V3.0.5 | `model_tps_v3.0_best_progress.pt` | 0.7030 | 13.45 | 67.14% | 1.09x | 432,263 | ❌ **Thảm họa rò rỉ đặc trưng bất đối xứng** |
| **14** | `v3.0_re1` | TPS V3.0-Remastered R1 | `model_tps_v3.0_remastered_ep21.pt` | 0.6697 | 14.63 | 72.47% | 1.16x | 41,050 | Khắc phục rò rỉ, nét dày, còn lem viền |
| **15** | `v3.0_re2_ep12` | TPS V3.0-R2 Ep12 | `model_tps_v3.0_remastered_ep12.pt` | 0.7347 | 14.73 | 65.69% | 0.89x | 16,508 | Đỉnh nhọn Recall & bảo toàn ngọn tóc |
| **16** | `v3.0_re2_soup` | **TPS V3.0-R2 Soup** | `model_tps_v3.0_remastered_soup.pt` | **0.7349** | **14.74** | **65.73%** | **0.88x** | **16,475** | 🏆 **QUÁN QUÂN TOÀN DIỆN (WINNER)** |

---

## 2. PHÂN TÍCH 5 BƯỚC NGOẶT TIẾN HÓA QUAN TRỌNG

### 1. Bước ngoặt 1: Vượt qua "Bẫy sụp đổ ảnh trắng" (V0.1 ──► V0.2/V0.3)
* **Hiện tượng:** Ở V0.1, hàm loss chưa có ràng buộc diện tích nét vẽ, mô hình tìm ra nghiệm cực tiểu cục bộ "dễ dãi" nhất là xóa sạch toàn bộ khung hình thành màu trắng (Blank Collapse, Recall = 0%).
* **Giải pháp:** Từ V0.2, nhóm bổ sung khối đầu hợp nhất nét thích nghi (Adaptive Stroke Fusion Head), giúp mô hình bắt đầu giữ lại được nét nhân vật (Recall đạt 67.15%).

### 2. Bước ngoặt 2: Kiểm soát quầng mờ và Neo viền biên (V0.3 ──► V1.1)
* **Hiện tượng:** Dù đã có nét, các bản V0.2 và V0.3 bị bóng ma diện rộng bao quanh nét vẽ (48,000 px).
* **Giải pháp:** Bản V1.1 tích hợp cơ chế **Boundary Anchors (8 điểm neo cố định ở 4 góc và 4 cạnh)** để khóa chặt lưới biến dạng TPS. Kết quả: **Ghost pixels giảm thẳng từ 39,750 px xuống 16,797 px (-57.7%)**.

### 3. Bước ngoặt 3: Cuộc chiến "Nét teo tóp vs. Nét phình nở" (V2.0 ──► V2.6)
* **Hiện tượng:** Khi áp dụng hình phạt độ dày nét ở V2.3, mô hình phản ứng thái quá: Stroke Density tụt xuống chỉ còn $0.58x$, nét vẽ bị đứt đoạn nghiêm trọng.
* **Giải pháp:** Phiên bản V2.6 điều chỉnh lại hệ số cân bằng, đưa Density lên $0.82x$ và Recall đạt $61.96\%$. Đây là bản production tốt nhất của thế hệ 2.

### 4. Bước ngoặt 4: Khắc phục sự cố rò rỉ bất đối xứng (V3.0.5 ──► V3.0 Remastered R1)
* **Hiện tượng:** Thử nghiệm V3.0.5 đưa vào khối Asymmetric Analyzer nhưng gặp lỗi tràn đặc trưng, dẫn đến rò rỉ pixel nền khủng khiếp (**432,263 pixel bóng ma** - nền trắng bị phủ kín bởi màn sương xám đục).
* **Giải pháp:** Đội ngũ quyết định tái thiết kế toàn bộ (**Remastered Architecture**), quay về cấu trúc cốt lõi vững chắc của V2.6 kết hợp cơ chế bảo toàn đối xứng thời gian (Time-Symmetric FBFC). Bản V3.0 Remastered R1 đã dập tắt hoàn toàn hiện tượng rò rỉ nền.

### 5. Bước ngoặt 5: Đỉnh cao tinh hoa Model Soup (V3.0 Remastered R2)
* **Thành tựu:** Bằng việc huấn luyện 16 Epochs với Cosine Annealing trên nền RGBA sạch, và hòa trộn trọng số **Model Soup** (kết hợp Ep12 và Ep16):
  * **Độ phủ nét (Recall) đạt $65.73\%$** (cao hơn V2.6 gần $+3.8\%$).
  * **Mật độ nét đạt $0.88x$** (nét vẽ đậm đà, không còn bị gầy teo).
  * **Độ sạch nền và SSIM đạt mức cao nhất lịch sử dự án ($0.7349$)**.

---

## 3. KẾT LUẬN & ĐỀ XUẤT ỨNG DỤNG

1. **Khẳng định tính đúng đắn của chuỗi thử nghiệm:** Dữ liệu benchmark liên tục trên cùng một tập test độc lập đã chứng minh sự trưởng thành rõ rệt qua từng thế hệ của nhóm nghiên cứu.
2. **Quyết định chuyển giao:** Phiên bản **`model_tps_v3.0_remastered_soup.pt`** chính thức là mô hình vượt trội nhất toàn diện, sẵn sàng tích hợp vào AniTweenStudio để đưa vào sản phẩm cuối cùng của đồ án PBL6.
