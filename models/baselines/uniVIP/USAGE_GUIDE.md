# Hướng Dẫn Sử Dụng Demo uniVIP (Arbitrary Time-step Interpolation)

Tài liệu này hướng dẫn chi tiết cách chạy thử nghiệm mô hình **uniVIP** (*A Unified Video Frame Interpolation and Prediction Framework*), đặc biệt là tính năng nội suy khung hình tại các mốc thời gian $t$ bất kỳ $t \in (0.0, 1.0)$.

---

## 1. Chuẩn Bị Môi Trường & Checkpoints

### Kích hoạt môi trường
Mở PowerShell/Terminal và chuyển đến thư mục gốc của `uniVIP`:
```powershell
cd e:\CODE\PBL\PBL6\Test\TestModels\uniVIP
```
Kích hoạt môi trường ảo (ví dụ `.venv`):
```powershell
..\..\.venv\Scripts\Activate.ps1
```

### Các file trọng số (Checkpoints) có sẵn trong `./checkpoints`:
- `uniVIP-B.pkl`: Bản tiêu chuẩn (Base model - nhẹ, tốc độ xử lý nhanh).
- `uniVIP-L.pkl`: Bản lớn (Large model - chất lượng tái tạo chi tiết sắc nét và chính xác hơn).

---

## 2. Hướng Dẫn Sử Dụng Script Demo Tùy Biến `interp_custom_times.py`

Script `demo/interp_custom_times.py` cho phép truyền danh sách mốc thời gian $t$ bất kỳ và xuất kết quả theo quy chuẩn:
- **Đầu ra**: Lưu trực tiếp vào thư mục chỉ định qua `--save_dir` (mặc định `./demo/output`).
- **Frame 1**: Tự động copy và lưu lại dưới tên `{prefix_1}.000.png` (Ví dụ: `1.000.png`).
- **Frame 2**: Tự động copy và lưu lại dưới tên `{prefix_2}.000.png` (Ví dụ: `2.000.png`).
- **In-between Frames**: Tự động sinh và lưu theo tên `{prefix_1}.{xxx}.png` (trong đó `{xxx}` tương ứng với giá trị $t$ theo số chữ số `--digits`, ví dụ $t=0.25 \rightarrow$ `1.250.png`).
- **Không sinh file ảnh phụ**: Chỉ xuất ảnh frame gốc và in-between frames (loại bỏ overlay hay flow map).

---

### Các Trường Hợp Sử Dụng Phổ Biến

#### Cách A: Truyền danh sách mốc thời gian $t$ tùy ý
Truyền danh sách các số thực trong khoảng $(0.0, 1.0)$ qua cờ `--times` (cách nhau bởi khoảng trắng hoặc dấu phẩy):

```powershell
python -m demo.interp_custom_times --frame0 ../../TestImage1/F1.png --frame1 ../../TestImage1/F2.png --times 0.25 0.5 0.75 --save_dir ./demo/output
```

*Ví dụ với các mốc thời gian thập phân lẻ hoặc dùng dấu phẩy:*
```powershell
python -m demo.interp_custom_times --frame0 ../../TestImage1/F1.png --frame1 ../../TestImage1/F2.png --times 0.125,0.375,0.625,0.875 --save_dir ./demo/output
```

#### Cách B: Tự động chia đều $N$ khung hình
Nếu không muốn nhập thủ công từng giá trị $t$, bạn dùng `--num_frames N` để mô hình tự động tính các mốc $t$ chia đều:

```powershell
# Tự động sinh 3 khung hình tại t = [0.25, 0.50, 0.75]
python -m demo.interp_custom_times --frame0 ../../TestImage1/F1.png --frame1 ../../TestImage1/F2.png --num_frames 3 --save_dir ./demo/output
```

#### Cách C: Sử dụng mô hình chất lượng cao hơn (`uniVIP-L`)
Để sử dụng bản Large (`uniVIP-L`), thêm cờ `--model_size large` (script sẽ tự động liên kết đến checkpoint `checkpoints/uniVIP-L.pkl`):

```powershell
python -m demo.interp_custom_times --frame0 ../../TestImage1/F1.png --frame1 ../../TestImage1/F2.png --times 0.2 0.4 0.6 0.8 --model_size large --save_dir ./demo/output
```

#### Cách D: Tùy biến số thứ tự Frame (Nội suy giữa cặp Frame bất kỳ)
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
| `--prefix_1` | Không | `1` | Tiền tố đánh số cho Frame 1 (ví dụ: `1` $\rightarrow$ `1.000.png`) |
| `--prefix_2` | Không | `2` | Tiền tố đánh số cho Frame 2 (ví dụ: `2` $\rightarrow$ `2.000.png`) |
| `--digits` | Không | `3` | Số chữ số thập phân cho tên file (mặc định: `3` $\rightarrow$ `1.500.png`) |
| `--model_size` | Không | `base` | Kích thước mô hình: `base` / `B` hoặc `large` / `L` |
| `--model_file` / `--checkpoint` | Không | `./checkpoints/uniVIP-B.pkl` | Đường dẫn file checkpoint tương ứng |
| `--fixed_pyr_level` | Không | Tự tính | Cố định số tầng kim tự tháp (để trống sẽ tự tính theo độ phân giải) |

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

## 5. Các Script Demo Mặc Định Khác của uniVIP

### Nội suy 1 ảnh trung gian gốc (`synthesize_frame.py`)
```powershell
python -m demo.synthesize_frame --frame0 demo/images/toytrain0.jpg --frame1 demo/images/toytrain1.jpg --time_period 0.5 --save_root ./demo/output
```

### Sinh video Slow-motion (`gen_slow_motion.py`)
```powershell
python -m demo.gen_slow_motion --frame0 demo/images/toytrain0.jpg --frame1 demo/images/toytrain1.jpg --multiple 8 --save_root ./demo/output
```
