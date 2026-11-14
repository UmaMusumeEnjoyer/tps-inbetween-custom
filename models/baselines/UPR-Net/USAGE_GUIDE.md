# Hướng Dẫn Sử Dụng Demo UPR-Net (Arbitrary Time-step Interpolation)

Tài liệu này hướng dẫn chi tiết cách chạy thử nghiệm mô hình **UPR-Net** (*Unified Pyramid Recurrent Network for Video Frame Interpolation*), đặc biệt là tính năng nội suy khung hình tại các mốc thời gian $t$ bất kỳ $t \in [0.0, 1.0]$.

---

## 1. Chuẩn Bị Môi Trường & Checkpoints

### Kích hoạt môi trường
Mở PowerShell/Terminal và chuyển đến thư mục gốc của UPR-Net:
```powershell
cd e:\CODE\PBL\PBL6\Test\TestModels\UPR-Net
```
Kích hoạt môi trường ảo (ví dụ `.venv`):
```powershell
..\..\.venv\Scripts\Activate.ps1
```

### Các file trọng số (Checkpoints) có sẵn trong `./checkpoints`:
- `upr-base.pkl`: Bản tiêu chuẩn (nhẹ ~1.7M params, tốc độ cao nhất).
- `upr-large.pkl`: Bản mở rộng, chi tiết tốt hơn.
- `upr-llarge.pkl`: Bản lớn nhất (chất lượng nội suy cao nhất).

---

## 2. Hướng Dẫn Sử Dụng Script Demo Tùy Biến `interp_custom_times.py`

Script `demo/interp_custom_times.py` cho phép truyền danh sách mốc thời gian $t$ bất kỳ và xuất kết quả theo quy chuẩn:
- **Đầu ra**: Lưu vào thư mục chỉ định (mặc định `./demo/output`).
- **Frame 1**: Tự động copy và lưu lại dưới tên `1.000.png`.
- **Frame 2**: Tự động copy và lưu lại dưới tên `2.000.png`.
- **In-between Frames**: Tự động sinh và lưu theo tên `1.xxx.png` (trong đó `xxx` tương ứng với giá trị $t$).

---

### Các Trường Hợp Sử Dụng Phổ Biến

#### Cách A: Truyền danh sách mốc thời gian $t$ tùy ý
Truyền danh sách các số thực trong khoảng $(0, 1)$ qua cờ `--times` (cách nhau bởi khoảng trắng hoặc dấu phẩy):

```powershell
python -m demo.interp_custom_times --frame0 ../../TestImage1/F1.png --frame1 ../../TestImage1/F2.png --times 0.25 0.5 0.75 --save_dir ./demo/output
```

*Ví dụ với các mốc thời gian lẻ:*
```powershell
python -m demo.interp_custom_times --frame0 ../../TestImage1/F1.png --frame1 ../../TestImage1/F2.png --times 0.12 0.35 0.68 0.85 --save_dir ./demo/output
```

#### Cách B: Tự động chia đều $N$ khung hình
Nếu không muốn nhập từng số $t$, sử dụng `--num_frames N` để mô hình tự động chia đều khoảng cách:

```powershell
# Tự động sinh 3 khung hình tại t = [0.25, 0.50, 0.75]
python -m demo.interp_custom_times --frame0 ../../TestImage1/F1.png --frame1 ../../TestImage1/F2.png --num_frames 3 --save_dir ./demo/output
```

#### Cách C: Sử dụng mô hình chất lượng cao hơn (`large` hoặc `LARGE`)
```powershell
python -m demo.interp_custom_times --frame0 ../../TestImage1/F1.png --frame1 ../../TestImage1/F2.png --times 0.2 0.4 0.6 0.8 --model_size LARGE --model_file ./checkpoints/upr-llarge.pkl --save_dir ./demo/output
```

#### Cách D: Tùy biến số thứ tự Frame (Nội suy giữa Frame bất kỳ)
Ví dụ bạn đang nội suy giữa **Frame 3** (`F3.png`) và **Frame 4** (`F4.png`):
```powershell
python -m demo.interp_custom_times --frame0 ../../TestImage1/F3.png --frame1 ../../TestImage1/F4.png --times 0.5 --prefix_1 3 --prefix_2 4 --save_dir ./demo/output
```
*Kết quả sinh ra sẽ gồm: `3.000.png`, `3.500.png`, `4.000.png`.*

---

## 3. Bảng Chi Tiết Tham Số của `interp_custom_times.py`

| Tham số | Bắt buộc | Mặc định | Ý nghĩa & Ví dụ |
|---|---|---|---|
| `--frame0` | **Có** | - | Đường dẫn file ảnh Frame 1 (ví dụ: `../../TestImage1/F1.png`) |
| `--frame1` | **Có** | - | Đường dẫn file ảnh Frame 2 (ví dụ: `../../TestImage1/F2.png`) |
| `--times` | Không | `None` | Danh sách mốc thời gian $t \in (0, 1)$ (ví dụ: `--times 0.25 0.5 0.75`) |
| `--num_frames` | Không | `None` | Số frame nội suy chia đều (ví dụ: `--num_frames 3`) |
| `--save_dir` | Không | `./demo/output` | Thư mục lưu các ảnh kết quả |
| `--prefix_1` | Không | `1` | Tiền tố đánh số cho Frame 1 (ví dụ: `1` -> `1.000.png`) |
| `--prefix_2` | Không | `2` | Tiền tố đánh số cho Frame 2 (ví dụ: `2` -> `2.000.png`) |
| `--digits` | Không | `3` | Số chữ số thập phân cho tên file (mặc định: `3` -> `1.500.png`) |
| `--model_size` | Không | `base` | Kích thước mô hình: `base`, `large`, hoặc `LARGE` |
| `--model_file` | Không | `./checkpoints/upr-base.pkl` | Đường dẫn file checkpoint tương ứng |
| `--pyr_level` | Không | Tự tính | Số tầng kim tự tháp (để trống sẽ tự động tính theo kích thước ảnh) |

---

## 4. Cấu Trúc File Kết Quả Đầu Ra

Ví dụ khi chạy lệnh với `--times 0.25 0.5 0.75`:
```text
demo/output/
├── 1.000.png   (Frame 1 gốc)
├── 1.250.png   (Frame trung gian nội suy tại t = 0.25)
├── 1.500.png   (Frame trung gian nội suy tại t = 0.50)
├── 1.750.png   (Frame trung gian nội suy tại t = 0.75)
└── 2.000.png   (Frame 2 gốc)
```

---

## 5. Các Script Demo Mặc Định Khác

### Nội suy 1 ảnh trung gian (`interp_imgs.py`)
```powershell
python -m demo.interp_imgs --frame0 demo/images/beanbags0.png --frame1 demo/images/beanbags1.png --time_period 0.5 --save_dir demo/output
```

### Nội suy toàn bộ thư mục chuỗi ảnh và xuất Video MP4 (`interp_sequence.py`)
```powershell
python -m demo.interp_sequence --input_dir ../../TestImage1 --save_dir demo/test_sequence_output --interp_factor 2 --create_video --fps 24
```
