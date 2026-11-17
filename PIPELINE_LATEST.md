# PIPELINE CHÍNH THỨC — SEQUENCE 2 PHA 3 REMAKE (BẢN ĐÃ FIX LỖI MỜ GÓC)

> **Tài liệu tham chiếu nhanh (Quick Reference):**  
> Khi cần nhắc đến hoặc yêu cầu làm việc với pipeline suy luận AI mới nhất, bạn chỉ cần mention file này: **`@PIPELINE_LATEST.md`**.

---

## 1. THÔNG TIN ĐỊNH DANH & VỊ TRÍ MÃ NGUỒN

* **Tên Pipeline:** `Phase 3 Remake Inference Pipeline (Polygon-Aware & Direct-t TPS)`
* **File mã nguồn chính:** [`src/sequence_p3_p4/seq_p3_inference_pipeline.py`](file:///e:/CODE/PBL/PBL6/src/sequence_p3_p4/seq_p3_inference_pipeline.py)
* **Class suy luận chính:** `Phase3InferencePipeline`
* **Commit sửa lỗi:** `ab56352` (*fix(pipeline): resolve corner blurring via polygon-aware contour matching*)

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
2. **Trường dòng chảy Coarse TPS trực tiếp tại thời điểm $t$ (`direct_t_flows=True`):**
   * Tính toán trực tiếp luồng uốn lưới tại thời điểm $t \in (0, 1)$ thay vì nội suy tuyến tính trường dòng chảy 2 đầu, loại bỏ bóng ma (ghosting) khi vật thể di chuyển xa.
3. **Bộ lọc tương phản thích ứng cục bộ (Adaptive Local Edge Sharpener):**
   * Xóa quầng xám mờ trên nền trắng mà không làm đứt gãy nét mảnh hoặc tạo răng cưa.

---

## 4. ĐOẠN MÃ MẪU KHỞI TẠO & SUY LUẬN NHANH (PYTHON QUICKSTART)

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
