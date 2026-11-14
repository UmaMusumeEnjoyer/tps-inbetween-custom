# Hướng Dẫn Sử Dụng Demo AnimeInbet (Arbitrary Time-step Line Inbetweening)

Tài liệu này hướng dẫn chi tiết cách chạy thử nghiệm mô hình **AnimeInbet** (*Deep Geometrized Cartoon Line Inbetweening* - ICCV 2023), đặc biệt là tính năng nội suy các khung hình hoạt họa vẽ nét (cartoon line drawings) tại các mốc thời gian $t$ bất kỳ $t \in (0.0, 1.0)$ xuất phát từ Frame 1 sang Frame 2.

---

## 1. Chuẩn Bị Môi Trường & Checkpoint

### Kích hoạt môi trường ảo
Mở PowerShell hoặc Terminal và di chuyển đến thư mục gốc của `AnimeInbet`:
```powershell
cd e:\CODE\PBL\PBL6\Test\TestModels\AnimeInbet
```
Kích hoạt môi trường ảo `.venv`:
```powershell
..\..\.venv\Scripts\Activate.ps1
```

### File trọng số mô hình (Checkpoint):
- **Trọng số toàn bộ Pipeline**: `./experiments/inbetweener_full/ckpt/epoch_20.pt` (hoặc chỉ định tùy ý qua tham số `--checkpoint`).
- *Lưu ý*: Nếu chưa tải file checkpoint, script vẫn có thể khởi tạo mô hình với cấu hình mặc định để kiểm thử kiến trúc đồ thị và quy trình nội suy.

---

## 2. Hướng Dẫn Sử Dụng Script Demo `interp_custom_times.py`

Script `demo/interp_custom_times.py` được thiết kế linh hoạt, hỗ trợ:
- **Tự động trích xuất Vector Graph**: Tự động nhận diện nét vẽ, trích xuất tọa độ đỉnh (keypoints) và đồ thị liên kết topo từ ảnh raster (nếu không có file nhãn `.json`).
- **Hỗ trợ tệp nhãn JSON**: Tự động tìm hoặc nạp file `.json` theo định dạng MixamoLine240 nếu có sẵn.
- **Đầu ra quy chuẩn**: Tất cả ảnh xuất ra được lưu trực tiếp vào thư mục `--save_dir` (mặc định `./demo/output`).
- **Quy tắc đặt tên file**:
  - Frame 1 gốc: `{prefix_1}.000.png` (Ví dụ: `1.000.png`).
  - Frame 2 gốc: `{prefix_2}.000.png` (Ví dụ: `2.000.png`).
  - Khung hình in-between: `{prefix_1}.{xxx}.png` với `{xxx}` là giá trị $t$ được làm tròn theo số chữ số thập phân `--digits` (Ví dụ: $t=0.25 \rightarrow$ `1.250.png`, $t=0.067 \rightarrow$ `1.067.png`).
- **Tự động xử lý kích thước ảnh**: Kiểm tra 2 ảnh cùng kích thước, tự động padding lên bội số của 32 và crop lại chính xác kích thước ban đầu trước khi lưu.
- **Không tạo file ảnh rác**: Không lưu các file phụ như bản đồ dòng quang học (optical flow) hay ảnh trộn (overlay).
- **Tự động nhận diện phần cứng**: Tự động dùng GPU CUDA nếu khả dụng, fallback về CPU nếu không có GPU hoặc khi truyền cờ `--cpu`.

---

## 3. Các Trường Hợp Sử Dụng Phổ Biến (CLI Examples)

### Cách A: Truyền danh sách mốc thời gian $t$ tùy ý
Truyền danh sách các số thực trong khoảng $(0.0, 1.0)$ qua tham số `--times` (cách nhau bởi khoảng trắng hoặc dấu phẩy):

```powershell
python demo/interp_custom_times.py --frame0 ../../TestImage1/F1.png --frame1 ../../TestImage1/F2.png --times 0.25 0.5 0.75 --save_dir ./demo/output
```

*Ví dụ với các mốc thời gian lẻ hoặc cách nhau bằng dấu phẩy:*
```powershell
python demo/interp_custom_times.py --frame0 ../../TestImage1/F1.png --frame1 ../../TestImage1/F2.png --times 0.125,0.375,0.625,0.875 --save_dir ./demo/output
```

---

### Cách B: Tự động chia đều $N$ khung hình
Nếu muốn hệ thống tự động tính toán các mốc thời gian $t$ chia đều trong khoảng $(0, 1)$, sử dụng cờ `--num_frames N`:

```powershell
# Tự động chia đều sinh 3 khung hình tại t = [0.25, 0.50, 0.75]
python demo/interp_custom_times.py --frame0 ../../TestImage1/F1.png --frame1 ../../TestImage1/F2.png --num_frames 3 --save_dir ./demo/output
```

---

### Cách C: Tùy biến số thứ tự Frame (Nội suy giữa cặp Frame bất kỳ)
Khi nội suy giữa cặp frame khác (ví dụ: giữa **Frame 3** và **Frame 4**):

```powershell
python demo/interp_custom_times.py --frame0 ../../TestImage1/F3.png --frame1 ../../TestImage1/F4.png --times 0.5 --prefix_1 3 --prefix_2 4 --save_dir ./demo/output
```
*Kết quả sinh ra sẽ gồm:*
- `3.000.png` (Frame 3 gốc)
- `3.500.png` (Khung hình nội suy tại $t=0.5$)
- `4.000.png` (Frame 4 gốc)

---

### Cách D: Chỉ định Checkpoint và kiến trúc mô hình
Nếu bạn đã tải file trọng số `epoch_20.pt`:

```powershell
python demo/interp_custom_times.py --frame0 ../../TestImage1/F1.png --frame1 ../../TestImage1/F2.png --checkpoint ./experiments/inbetweener_full/ckpt/epoch_20.pt --times 0.25 0.5 0.75 --save_dir ./demo/output
```

---

### Cách E: Tùy chỉnh độ dày nét vẽ (Line Width) & Số lượng điểm đỉnh tối đa (Max Keypoints)
Đối với các bản vẽ nét mảnh hoặc độ phân giải cao:

```powershell
python demo/interp_custom_times.py --frame0 ../../TestImage1/F1.png --frame1 ../../TestImage1/F2.png --num_frames 3 --line_width 2 --max_kpts 2500 --save_dir ./demo/output
```

---

### Cách F: Bắt buộc chạy trên CPU
Nếu muốn kiểm thử trên CPU hoặc máy không có card đồ họa NVIDIA:

```powershell
python demo/interp_custom_times.py --frame0 ../../TestImage1/F1.png --frame1 ../../TestImage1/F2.png --num_frames 2 --cpu --save_dir ./demo/output
```

---

## 4. Bảng Chi Tiết Toàn Bộ Tham Số CLI

| Tham số | Bắt buộc | Mặc định | Ý nghĩa & Mô tả chi tiết |
|---|---|---|---|
| `--frame0` | **Có** | - | Đường dẫn đến file ảnh Frame 1 ($I_0$). |
| `--frame1` | **Có** | - | Đường dẫn đến file ảnh Frame 2 ($I_1$). |
| `--times` | Không | `None` | Danh sách mốc thời gian $t \in (0.0, 1.0)$ (ví dụ: `--times 0.2 0.35 0.5 0.8`). |
| `--num_frames` | Không | `None` | Số lượng khung hình chia đều tự động nếu không dùng `--times` (ví dụ: `--num_frames 3`). |
| `--save_dir` | Không | `./demo/output` | Thư mục lưu các file ảnh kết quả đầu ra. |
| `--prefix_1` | Không | `1` | Tiền tố đánh số cho Frame 1 (ví dụ: `1` $\rightarrow$ `1.000.png`, `1.xxx.png`). |
| `--prefix_2` | Không | `2` | Tiền tố đánh số cho Frame 2 (ví dụ: `2` $\rightarrow$ `2.000.png`). |
| `--digits` | Không | `3` | Số chữ số phần thập phân cho tên file (mặc định `3` $\rightarrow$ `1.500.png`, `1.250.png`). |
| `--model_size` | Không | `InbetweenerTM` | Kiến trúc mô hình (`InbetweenerTM` hoặc `InbetweenerM2`). |
| `--checkpoint` | Không | `None` | Đường dẫn file checkpoint trọng số (.pt). |
| `--config` | Không | `configs/cr_inbetweener_full.yaml` | Đường dẫn file cấu hình YAML của mô hình. |
| `--label0` | Không | `None` | File nhãn đồ thị `.json` cho Frame 1 (tùy chọn, tự động trích xuất nếu không có). |
| `--label1` | Không | `None` | File nhãn đồ thị `.json` cho Frame 2 (tùy chọn, tự động trích xuất nếu không có). |
| `--line_width` | Không | `2` | Độ dày đường nét (pixel) khi kết xuất in-between frames. |
| `--max_kpts` | Không | `2000` | Giới hạn số đỉnh đồ thị tối đa để tối ưu hóa bộ nhớ GPU/CPU. |
| `--cpu` | Không | `False` | Bắt buộc chạy trên CPU (mặc định tự động nhận diện CUDA). |

---

## 5. Quy Chuẩn Cấu Trúc File Đầu Ra

Khi chạy với lệnh:
```powershell
python demo/interp_custom_times.py --frame0 F1.png --frame1 F2.png --times 0.25 0.5 0.75 --prefix_1 1 --prefix_2 2 --digits 3 --save_dir ./demo/output
```

Thư mục đầu ra `./demo/output` sẽ chứa:
```text
demo/output/
├── 1.000.png    # Frame 1 gốc
├── 1.250.png    # Khung hình in-between tại t = 0.25
├── 1.500.png    # Khung hình in-between tại t = 0.50
├── 1.750.png    # Khung hình in-between tại t = 0.75
└── 2.000.png    # Frame 2 gốc
```
