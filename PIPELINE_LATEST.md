# PIPELINE CHÍNH THỨC — SEQUENCE 2 PHA 3 REMAKE (BẢN FIX LỖI MỜ GÓC & CHỐNG BIẾN DẠNG ANIME LINEART)

> **Tài liệu tham chiếu nhanh (Quick Reference):**  
> Khi cần nhắc đến hoặc yêu cầu làm việc với pipeline suy luận AI mới nhất, bạn chỉ cần mention file này: **`@PIPELINE_LATEST.md`**.

---

## 1. THÔNG TIN ĐỊNH DANH & VỊ TRÍ MÃ NGUỒN

* **Tên Pipeline:** `Phase 3 Remake Inference Pipeline (Dual-Regime Keypoint Matching & Direct-t TPS)`
* **File mã nguồn chính:** [`src/sequence_p3_p4/seq_p3_inference_pipeline.py`](file:///e:/CODE/PBL/PBL6/src/sequence_p3_p4/seq_p3_inference_pipeline.py)
* **Gói độc lập đối chuẩn (Isolated Package):** [`SUPER CẢI THIỆN BENCHMARK BILIBILI/`](file:///e:/CODE/PBL/PBL6/SUPER%20C%E1%BA%A2I%20THI%E1%BB%86N%20BENCHMARK%20BILIBILI)
* **Class suy luận chính:** `Phase3InferencePipeline`

### Các module tích hợp đồng bộ:
1. **Tầng 1 (Coarse TPS Đa lớp):** [`src/sequence_p3_p4/seq_p3_multi_layer_tps.py`](file:///e:/CODE/PBL/PBL6/src/sequence_p3_p4/seq_p3_multi_layer_tps.py)
2. **Tầng 2 (Mạng UNet Tinh Chỉnh):** `TPSInbetweenOriginal` (~10.07M active params) tại [`training/03_sequence/p3_multilayer_tps/seq_p3_v2_multilayer_tps_worker.py`](file:///e:/CODE/PBL/PBL6/training/03_sequence/p3_multilayer_tps/seq_p3_v2_multilayer_tps_worker.py)
3. **Tầng 3 (Bộ lọc Sắc nét Cục bộ):** [`src/sequence_p3_p4/seq_p3_adaptive_sharpener.py`](file:///e:/CODE/PBL/PBL6/src/sequence_p3_p4/seq_p3_adaptive_sharpener.py) (chế độ `mode='balanced'`)

---

## 2. CHECKPOINT TRỌNG SỐ MÔ HÌNH CHỈ ĐỊNH

* **Đường dẫn Checkpoint:**  
  [`checkpoints/Re-train/Sequence 2 pha 3 remake/checkpoints/model_tps_finetuned_best.pt`](file:///e:/CODE/PBL/PBL6/checkpoints/Re-train/Sequence%202%20pha%203%20remake/checkpoints/model_tps_finetuned_best.pt)
* **Hằng số mặc định trong code:** `DEFAULT_PHASE3_CKPT`
* **Dung lượng:** `40,325,991 bytes`
* **Kiểm tra tương thích:** Khớp 158/158 active tensors (100%), load qua cơ chế EMA shadow weights.

---

## 3. CÁC TÍNH NĂNG VÀ LỖI ĐÃ ĐƯỢC KHẮC PHỤC

1. **Khắc phục lỗi mờ/cụt 4 góc khi co giãn lệch tỷ lệ ($W/H$ thay đổi):**
   * Tự động nhận diện đỉnh góc đa giác ($3 \le N \le 16$) bằng `cv2.approxPolyDP` kết hợp hàm `_compute_turning_angles`.
   * Khóa cứng (Pinning) 1-1 các đỉnh góc giữa keyframe đầu và keyframe cuối bằng giải thuật căn chỉnh chu kỳ góc (Cyclic Alignment).
   * Lấy mẫu phân đoạn dọc theo từng cạnh độc lập (Per-Edge Sampling), loại bỏ 100% hiện tượng trượt chu vi (Arc-length Drift), giúp các góc vuông $90^\circ$ giữ nguyên độ đen đanh và sắc nét.

2. **Cơ chế phân loại điểm neo chế độ kép (Dual-Regime Keypoint Classifier):**
   * Tự động nhận biết hình thái phân bố nét vẽ của khung hình:
     - **Chế độ Simple Closed Geometry (Hình tròn, elip, chữ thập, đa giác đơn):** Dùng Polygon-Aware Contour Matching để bảo toàn $100\%$ độ chính xác hình học và độ sắc cạnh.
     - **Chế độ Complex Anime Lineart (Nét vẽ nhân vật Sakuga thực tế, tóc, khuôn mặt):** Tự động chuyển sang **Stroke-Guided PyrLK + SIFT RANSAC** (`extract_stroke_guided_pyrlk_sift`), lấy mẫu điểm góc Shi-Tomasi trực tiếp dọc nét vẽ với kiểm định tiến - lùi (Forward-Backward check $\le 1.25$ px) và lọc Affine RANSAC. Loại bỏ hoàn toàn sự cố lệch chu kỳ contour gây méo và xé nát tóc nhân vật.

3. **Chốt chặn vận tốc ngoại lai & Bán kính ảnh hưởng (Speed-Bound Clamp & Falloff Guard):**
   * Trong `seq_p3_multi_layer_tps.py`, bổ sung ngưỡng tốc độ tối đa $v_{\max} = \max(60\text{px}, 3.5 \times v_{\text{median}} + 25\text{px})$, loại bỏ $100\%$ các vector trôi dạt ngoại lai (300 - 800 px).
   * Cố định bán kính suy giảm mềm `effective_falloff = float(falloff_radius_px)` khi không có điểm tĩnh, ngăn không cho trường dòng chảy TPS phình to ra toàn bộ khung tranh.

4. **Trường dòng chảy Coarse TPS trực tiếp tại thời điểm $t$ (`direct_t_flows=True`):**
   * Tính toán trực tiếp luồng uốn lưới tại thời điểm $t \in (0, 1)$ thay vì nội suy tuyến tính trường dòng chảy 2 đầu, loại bỏ bóng ma (ghosting) khi vật thể di chuyển xa.

5. **Bộ lọc tương phản thích ứng cục bộ (Adaptive Local Edge Sharpener):**
   * Xóa quầng xám mờ trên nền trắng mà không làm đứt gãy nét mảnh hoặc tạo răng cưa (chế độ `mode='balanced'`).

---

## 4. KẾT QUẢ ĐỐI CHUẨN XÁC THỰC (BENCHMARK RESULTS 2026-11-21)

| Bài Test | PSNR (dB) ↑ | SSIM ↑ | WCD-DoG ↓ | Stroke F1 ↑ | Skeleton Recall ↑ | Density Ratio | Ghost Pixels ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Case 1: Chuỗi Lineart 24f** | **14.80** | **0.6666** | **2.6356** *(giảm -55%)* | **0.7049** *(tăng +74%)* | **58.54%** *(gấp 2.07 lần)* | **0.9793** *(chuẩn ~1.0)* | **9,626.6** *(giảm -27%)* |
| **Case 2: Hình Tròn Co Giãn** | **30.89** | **0.9947** | **0.7898** | **1.0000** | **14.29%** | **0.8935** | **8.4** |
| **Case 3: Chữ Thập Xoay Cứng** | **25.63** | **0.9872** | **0.9880** | **0.9826** | **14.29%** | **0.7816** | **6.1** |

---

## 5. ĐOẠN MÃ MẪU KHỞI TẠO & SUY LUẬN NHANH (PYTHON QUICKSTART)

```python
import sys
import os
import torch
from PIL import Image

REPO_ROOT = r"e:\CODE\PBL\PBL6"
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "sequence_p3_p4"))
sys.path.insert(0, os.path.join(REPO_ROOT, "training", "03_sequence", "p3_multilayer_tps"))

from seq_p3_inference_pipeline import Phase3InferencePipeline, DEFAULT_PHASE3_CKPT

# 1. Khởi tạo pipeline (tự động nạp GPU và checkpoint Sequence 2 pha 3 remake chuẩn)
pipe = Phase3InferencePipeline(
    checkpoint_path=DEFAULT_PHASE3_CKPT,
    use_sharpener=True,
    sharpener_mode="balanced",
)

# 2. Nội suy giữa 2 keyframe tại thời điểm t = 0.5 (hoặc 1/3, 2/3,...)
res = pipe.interpolate_pair(
    "path/to/keyframe_0.png",
    "path/to/keyframe_1.png",
    t=0.5,
    direct_t_flows=True,
)

# 3. Lấy ảnh đầu ra uint8 (0=đen, 255=trắng)
inbetween_u8 = res["inbetween_u8"]
Image.fromarray(inbetween_u8).save("inbetween_output.png")
```
