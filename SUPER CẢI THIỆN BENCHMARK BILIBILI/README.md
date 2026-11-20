# 🚀 THƯ MỤC CÁCH LY: SUPER CẢI THIỆN BENCHMARK BILIBILI
## Trọn Bộ Pipeline Suy Luận & Checkpoint Tinh Chỉnh Tối Ưu (Phase 3 Remake)

> **Mục đích của thư mục này:**  
> Đây là thư mục độc lập hoàn toàn (**isolated standalone package**), được trích xuất và đóng gói đầy đủ từ pipeline chuẩn tại `PIPELINE_LATEST.md` và checkpoint tối ưu nhất `model_tps_finetuned_best.pt`.  
> Thư mục này giúp bạn có thể chạy suy luận, kiểm thử đối chứng hoặc thực hiện benchmark toàn diện trên các tập dữ liệu Anime 2D phức tạp (như Bilibili animation clips, Sakuga sequence dataset) mà **không bị phụ thuộc** vào toàn bộ cấu trúc mã nguồn huấn luyện của dự án.

---

## 1. THÀNH PHẦN FILE BÊN TRONG THƯ MỤC

| Tên File | Vai Trò & Chức Năng | Dung Lượng |
| :--- | :--- | :--- |
| **`model_tps_finetuned_best.pt`** | Checkpoint trọng số mô hình tốt nhất (Epoch tối ưu, nạp khớp 158/158 tensors qua EMA shadow weights). | **~38.46 MB** (40,325,991 bytes) |
| **`seq_p3_inference_pipeline.py`** | Class điều phối suy luận chính `Phase3InferencePipeline`, tích hợp trọn vẹn 3 tầng xử lý. Tự động ưu tiên nạp checkpoint nội bộ thư mục. | ~23 KB |
| **`seq_p3_multi_layer_tps.py`** | Tầng 1: Động cơ Coarse TPS Đa Lớp Polygon-Aware, Direct-$t$ flows, khử trượt chu vi góc ($90^\circ$). | ~37 KB |
| **`model_architecture.py`** | Tầng 2: Định nghĩa kiến trúc mạng UNet nguyên bản `TPSInbetweenOriginal` (~10.07M active params, cam kết 0% sửa kiến trúc) và hàm `load_tps_weights`. | ~11 KB |
| **`seq_p3_v2_multilayer_tps_worker.py`** | Proxy tương thích ngược để nạp kiến trúc mô hình một cách liền mạch. | ~0.3 KB |
| **`seq_p3_adaptive_sharpener.py`** | Tầng 3: Bộ lọc tương phản thích ứng cục bộ (`mode='balanced'`), xóa quầng mờ halo mà không làm đứt gãy nét vẽ. | ~14 KB |
| **`run_inference.py`** | Script dòng lệnh (CLI Runner) cho phép chạy nội suy giữa 2 khung hình một cách nhanh chóng. | ~2.5 KB |
| **`README.md`** | Tài liệu giải thích chi tiết, cơ chế hoạt động và hướng dẫn sử dụng thư mục này. | ~5 KB |

---

## 2. NGUYÊN LÝ HOẠT ĐỘNG: KIẾN TRÚC 3 TẦNG ĐỒNG BỘ

Pipeline trong thư mục này giải quyết triệt để 3 bài toán lớn nhất của việc nội suy hoạt hình Anime 2D:

### Tầng 1: Coarse TPS Đa Lớp Polygon-Aware (`seq_p3_multi_layer_tps.py`)
- **Khắc phục biến dạng góc:** Sử dụng thuật toán nhận diện đỉnh đa giác (`approxPolyDP`) kết hợp căn chỉnh chu kỳ góc (`Cyclic Alignment`) và lấy mẫu phân đoạn độc lập trên từng cạnh (`Per-Edge Sampling`). Nhờ đó, các góc nhọn, góc vuông trên nhân vật hoặc khung cảnh giữ nguyên $90^\circ$ mà không bị co cụm mờ mép khi tỷ lệ khung hình thay đổi.
- **Direct-$t$ Flow Computation:** Tính toán trực tiếp trường dòng chảy uốn lưới tại thời điểm $t \in (0, 1)$ thay vì nội suy tuyến tính trường dòng chảy 2 đầu, loại bỏ hiện tượng bóng ma (ghosting) khi nhân vật di chuyển nhanh hoặc góc quay lớn.

### Tầng 2: Mạng UNet Tinh Chỉnh (`model_architecture.py` & Checkpoint `.pt`)
- **Kiến trúc nguyên bản:** Kế thừa 100% mạng `TPSInbetweenOriginal` của Zhu et al. (HIT 2024) với 10,068,668 active parameters, đảm bảo tính công bằng khoa học tuyệt đối khi đối chuẩn.
- **Trọng số tinh chỉnh:** Checkpoint `model_tps_finetuned_best.pt` được huấn luyện với bộ 8 hàm mất mát chống dày nét (8-Term Anti-Thickening Loss), cân bằng tỷ lệ nét về xấp xỉ $1.00$ và nâng cao độ liền mạch khung hình.

### Tầng 3: Bộ lọc Làm Nét Thích Ứng Cục Bộ (`seq_p3_adaptive_sharpener.py`)
- **Khử Halo & Quầng Mờ:** Áp dụng thuật toán kéo dốc sườn chuyển tiếp mức xám thích ứng cục bộ (`mode='balanced'`).
- Tự động phân tách ranh giới giữa nét mực (ink) và nền trắng, triệt tiêu 100% hiện tượng lem xám mà không gây gãy nét mảnh hoặc tạo gai răng cưa.

---

## 3. HƯỚNG DẪN SỬ DỤNG NHANH (QUICKSTART)

### Cách 1: Sử dụng dòng lệnh (CLI Runner)

Bạn có thể chạy trực tiếp file `run_inference.py` ngay trong thư mục này:

```bash
# Di chuyển vào thư mục
cd "SUPER CẢI THIỆN BENCHMARK BILIBILI"

# Chạy nội suy tại mốc t = 0.5
python run_inference.py --img0 path/to/frame_0.png --img1 path/to/frame_1.png --t 0.5 --out output_t05.png

# Chạy trên GPU với chế độ làm nét crisp
python run_inference.py --img0 frame_0.png --img1 frame_1.png --device cuda --mode crisp --out result.png
```

### Cách 2: Gọi trong mã Python

```python
import os
import sys
from PIL import Image

# Import trực tiếp pipeline từ thư mục hiện tại
from seq_p3_inference_pipeline import Phase3InferencePipeline, DEFAULT_PHASE3_CKPT

# 1. Khởi tạo pipeline (tự động phát hiện GPU/CPU và nạp checkpoint model_tps_finetuned_best.pt nội bộ)
pipe = Phase3InferencePipeline(
    checkpoint_path=DEFAULT_PHASE3_CKPT,
    use_sharpener=True,
    sharpener_mode="balanced",  # 'balanced', 'crisp', hoặc 'subtle'
)

# 2. Nội suy giữa 2 keyframe
result = pipe.interpolate_pair(
    img0="path/to/frame_0.png",
    img1="path/to/frame_1.png",
    t=0.5,
    direct_t_flows=True,
)

# 3. Trích xuất và lưu ảnh kết quả
inbetween_u8 = result["inbetween_u8"]
Image.fromarray(inbetween_u8).save("interpolated_frame.png")
print("Đã xuất frame nội suy thành công!")
```

---

## 4. TƯƠNG THÍCH VÀ ĐÓNG GÓI GITHUB

- Checkpoint `model_tps_finetuned_best.pt` có dung lượng **38.46 MB**, nằm an toàn dưới ngưỡng giới hạn file đơn của GitHub (**100 MB**) và dưới cả ngưỡng cảnh báo (**50 MB**).
- Thư mục được thiết kế hoàn toàn khép kín để sẵn sàng phục vụ các bài kiểm tra đối chứng (Benchmark Suite) với Bilibili Anime Lineart, Sakuga Golden Dataset và Lagrangian TPS Suite.
