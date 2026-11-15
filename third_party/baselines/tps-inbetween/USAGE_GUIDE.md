# Hướng Dẫn Sử Dụng Demo TPS-Inbetween (Arbitrary Time-step Interpolation)

Tài liệu này hướng dẫn chi tiết cách chạy thử nghiệm mô hình **TPS-Inbetween** (*Thin-Plate Spline-based Interpolation for Animation Line Inbetweening*), đặc biệt là tính năng nội suy các khung hình hoạt họa trung gian tại các mốc thời gian $t$ bất kỳ $t \in (0.0, 1.0)$.

---

## 1. Chuẩn Bị Môi Trường & Checkpoints

### Kích hoạt môi trường
Mở PowerShell/Terminal và chuyển đến thư mục gốc của `tps-inbetween`:
```powershell
cd e:\CODE\PBL\PBL6\Test\TestModels\tps-inbetween
```
Kích hoạt môi trường ảo (ví dụ `.venv`):
```powershell
..\..\.venv\Scripts\Activate.ps1
```

### Các file trọng số (Checkpoints) cần có:
- **TPS-Inbetween Model**: `./ckpt/model_latest.pt` (Mô hình chính tổng hợp khung hình).
- **GlueStick Matcher**: `./model/resources/weights/checkpoint_GlueStick_MD.tar` (Mô hình trích xuất và so khớp đặc trưng điểm/đoạn thẳng).

---

## 2. Hướng Dẫn Sử Dụng Script Demo Tùy Biến `interp_custom_times.py`

Script `demo/interp_custom_times.py` cho phép truyền danh sách mốc thời gian $t$ bất kỳ và xuất kết quả theo quy chuẩn:
- **Đầu ra**: Lưu trực tiếp vào thư mục chỉ định qua `--save_dir` (mặc định `./demo/output`).
- **Frame 1**: Tự động copy và lưu lại dưới tên `{prefix_1}.000.png` (Ví dụ: `1.000.png`).
- **Frame 2**: Tự động copy và lưu lại dưới tên `{prefix_2}.000.png` (Ví dụ: `2.000.png`).
- **In-between Frames**: Tự động sinh và lưu theo tên `{prefix_1}.{xxx}.png` (trong đó `{xxx}` tương ứng với giá trị $t$ theo số chữ số `--digits`, ví dụ $t=0.25 \rightarrow$ `1.250.png`, $t=0.067 \rightarrow$ `1.067.png`).
- **Tự động xử lý kích thước ảnh**: Tự động padding kích thước ảnh chia hết cho 32 và crop lại chính xác theo kích thước gốc trước khi lưu.
- **Tối ưu bộ nhớ**: Áp dụng cơ chế tính toán TPS chia khối (chunking), không lo tràn bộ nhớ GPU (CUDA Out-of-Memory) khi xử lý ảnh độ phân giải cao.
- **Không sinh file ảnh phụ**: Tự động dọn dẹp các tệp tạm thời, chỉ lưu ảnh frame gốc và in-between frames (loại bỏ matching map hay flow map trong thư mục output).

---

### Các Trường Hợp Sử Dụng Phổ Biến

#### Cách A: Truyền danh sách mốc thời gian $t$ tùy ý
Truyền danh sách các số thực trong khoảng $(0.0, 1.0)$ qua cờ `--times` (cách nhau bởi khoảng trắng hoặc dấu phẩy):

```powershell
python -m demo.interp_custom_times --frame0 assets/input1_0.png --frame1 assets/input1_1.png --times 0.25 0.5 0.75 --save_dir ./demo/output
```

*Ví dụ với các mốc thời gian thập phân lẻ hoặc dùng dấu phẩy:*
```powershell
python -m demo.interp_custom_times --frame0 assets/input1_0.png --frame1 assets/input1_1.png --times 0.125,0.375,0.625,0.875 --save_dir ./demo/output
```

#### Cách B: Tự động chia đều $N$ khung hình
Nếu không muốn nhập thủ công từng giá trị $t$, bạn dùng `--num_frames N` để mô hình tự động tính các mốc $t$ chia đều:

```powershell
# Tự động sinh 3 khung hình tại t = [0.25, 0.50, 0.75]
python -m demo.interp_custom_times --frame0 assets/input1_0.png --frame1 assets/input1_1.png --num_frames 3 --save_dir ./demo/output
```

#### Cách C: Chạy với dữ liệu ảnh trong thư mục Test (`TestImage1`, `TestImage2`)
```powershell
python -m demo.interp_custom_times --frame0 ../../TestImage1/F1.png --frame1 ../../TestImage1/F2.png --times 0.25 0.5 0.75 --save_dir ./demo/output
```

#### Cách D: Tùy biến số thứ tự Frame (Nội suy giữa cặp Frame bất kỳ)
Ví dụ bạn đang nội suy giữa **Frame 3** (`F3.png`) và **Frame 4** (`F4.png`):
```powershell
python -m demo.interp_custom_times --frame0 ../../TestImage1/F3.png --frame1 ../../TestImage1/F4.png --times 0.5 --prefix_1 3 --prefix_2 4 --save_dir ./demo/output
```
*Kết quả sinh ra sẽ gồm: `3.000.png`, `3.500.png`, `4.000.png`.*

#### Cách E: Bắt buộc chạy trên CPU
Nếu không có GPU hoặc muốn chạy kiểm thử trên CPU:
```powershell
python -m demo.interp_custom_times --frame0 assets/input1_0.png --frame1 assets/input1_1.png --num_frames 2 --cpu --save_dir ./demo/output
```

---

## 3. Bảng Chi Tiết Tham Số của `interp_custom_times.py`

| Tham số | Bắt buộc | Mặc định | Ý nghĩa & Ví dụ |
|---|---|---|---|
| `--frame0` | **Có** | - | Đường dẫn file ảnh Frame 1 (ví dụ: `assets/input1_0.png` hoặc `../../TestImage1/F1.png`) |
| `--frame1` | **Có** | - | Đường dẫn file ảnh Frame 2 (ví dụ: `assets/input1_1.png` hoặc `../../TestImage1/F2.png`) |
| `--times` | Không | `None` | Danh sách mốc thời gian $t \in (0, 1)$ (ví dụ: `--times 0.25 0.5 0.75`) |
| `--num_frames` | Không | `None` | Số frame nội suy chia đều (ví dụ: `--num_frames 3`) |
| `--save_dir` | Không | `./demo/output` | Thư mục lưu các ảnh kết quả |
| `--prefix_1` | Không | `1` | Tiền tố đánh số cho Frame 1 (ví dụ: `1` $\rightarrow$ `1.000.png`, `1.xxx.png`) |
| `--prefix_2` | Không | `2` | Tiền tố đánh số cho Frame 2 (ví dụ: `2` $\rightarrow$ `2.000.png`) |
| `--digits` | Không | `3` | Số chữ số thập phân cho tên file (mặc định: `3` $\rightarrow$ `1.500.png`, `1.250.png`) |
| `--model_size` | Không | `default` | Kích thước mô hình / biến thể |
| `--checkpoint` / `--model_path` | Không | `./ckpt/model_latest.pt` | Đường dẫn file checkpoint mô hình TPS-Inbetween |
| `--matcher_weights` | Không | `./model/resources/weights/checkpoint_GlueStick_MD.tar` | Đường dẫn file trọng số GlueStick Matcher |
| `--cpu` | Không | `False` | Bắt buộc chạy trên CPU (mặc định tự động nhận diện GPU `cuda`) |

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

## 5. Script Demo Gốc của TPS-Inbetween (`demo.py`)

Demo mặc định của tác giả để sinh chuỗi ảnh đều và file GIF:
```powershell
python demo.py --image1 assets/input1_0.png --image2 assets/input1_1.png --xN 30 --save_path ./output
```
