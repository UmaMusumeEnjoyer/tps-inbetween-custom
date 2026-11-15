# Hướng Dẫn Sử Dụng Demo AnimeInterp (Arbitrary Time-step Interpolation)

Tài liệu này hướng dẫn chi tiết cách chạy thử nghiệm mô hình **AnimeInterp** (*Deep Animation Video Interpolation in the Wild - CVPR 2021*), đặc biệt là tính năng nội suy các khung hình hoạt họa trung gian tại các mốc thời gian $t$ bất kỳ $t \in (0.0, 1.0)$.

---

## 1. Chuẩn Bị Môi Trường & Checkpoints

### Kích hoạt môi trường
Mở PowerShell/Terminal và chuyển đến thư mục gốc của `AnimeInterp`:
```powershell
cd e:\CODE\PBL\PBL6\Test\TestModels\AnimeInterp
```
Kích hoạt môi trường ảo (ví dụ `.venv`):
```powershell
..\..\.venv\Scripts\Activate.ps1
```

### Tải file trọng số Checkpoint:
Tải file checkpoint được huấn luyện sẵn của mô hình AnimeInterp:
- **Tên file**: `anime_interp_full.ckpt`
- **Vị trí lưu trữ**: `./checkpoints/anime_interp_full.ckpt`
- **Link tải chính thức**:
  - [Dropbox Download Link](https://www.dropbox.com/s/oc8juclx1775qib/anime_interp_full.ckpt?dl=0)
  - [OneDrive Download Link](https://entuedu-my.sharepoint.com/:u:/g/personal/siyao002_e_ntu_edu_sg/EY3SG0-IajxKj9HMPz__zOMBvyJdrA-SlwpyHYFkDsQtng?e=q7nGlu)

*(Lưu ý: Nếu chưa tải checkpoint, script demo vẫn có thể khởi chạy kiểm thử luồng hoạt động với trọng số khởi tạo).*

---

## 2. Hướng Dẫn Sử Dụng Script Demo Tùy Biến `interp_custom_times.py`

Script `demo/interp_custom_times.py` cho phép truyền danh sách mốc thời gian $t$ bất kỳ và xuất kết quả theo quy chuẩn:
- **Đầu ra**: Lưu trực tiếp vào thư mục chỉ định qua `--save_dir` (mặc định `./demo/output`).
- **Frame 1**: Tự động copy và lưu lại dưới tên `{prefix_1}.000.png` (Ví dụ: `1.000.png`).
- **Frame 2**: Tự động copy và lưu lại dưới tên `{prefix_2}.000.png` (Ví dụ: `2.000.png`).
- **In-between Frames**: Tự động sinh và lưu theo tên `{prefix_1}.{xxx}.png` (trong đó `{xxx}` tương ứng với giá trị $t$ theo số chữ số `--digits`, ví dụ $t=0.25 \rightarrow$ `1.250.png`, $t=0.32 \rightarrow$ `1.320.png`, $t=0.067 \rightarrow$ `1.067.png`).
- **Tự động xử lý kích thước ảnh**: Tự động padding kích thước ảnh chia hết cho 32 (phù hợp với mạng trích xuất RFR và GridNet) và crop lại chính xác theo kích thước gốc trước khi lưu.
- **Tăng tốc xử lý**: Tự động ưu tiên nhân CUDA CuPy SoftSplat khi có GPU và CuPy, hoặc fallback an toàn sang pure PyTorch ForwardWarp.
- **Không sinh file ảnh phụ**: Chỉ xuất ảnh frame gốc và in-between frames (loại bỏ overlay hay flow map phụ).

---

### Các Trường Hợp Sử Dụng Phổ Biến

#### Cách A: Truyền danh sách mốc thời gian $t$ tùy ý
Truyền danh sách các số thực trong khoảng $(0.0, 1.0)$ qua cờ `--times` (cách nhau bởi khoảng trắng hoặc dấu phẩy):

```powershell
python -m demo.interp_custom_times --frame0 ../../TestImage1/F1.png --frame1 ../../TestImage1/F2.png --times 0.2 0.25 0.32 0.5 --save_dir ./demo/output
```

*Ví dụ với các mốc thời gian dùng dấu phẩy:*
```powershell
python -m demo.interp_custom_times --frame0 ../../TestImage1/F1.png --frame1 ../../TestImage1/F2.png --times 0.125,0.375,0.625,0.875 --save_dir ./demo/output
```

#### Cách B: Tự động chia đều $N$ khung hình
Nếu không muốn nhập thủ công từng giá trị $t$, bạn dùng `--num_frames N` để mô hình tự động tính các mốc $t$ chia đều:

```powershell
# Tự động sinh 3 khung hình tại t = [0.25, 0.50, 0.75]
python -m demo.interp_custom_times --frame0 ../../TestImage1/F1.png --frame1 ../../TestImage1/F2.png --num_frames 3 --save_dir ./demo/output
```

#### Cách C: Chạy với file checkpoint tùy chỉnh
Nếu lưu file checkpoint ở vị trí khác, bạn có thể truyền qua `--checkpoint`:

```powershell
python -m demo.interp_custom_times --frame0 ../../TestImage1/F1.png --frame1 ../../TestImage1/F2.png --times 0.25 0.5 0.75 --checkpoint ./checkpoints/anime_interp_full.ckpt --save_dir ./demo/output
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
python -m demo.interp_custom_times --frame0 ../../TestImage1/F1.png --frame1 ../../TestImage1/F2.png --num_frames 2 --cpu --save_dir ./demo/output
```

#### Cách F: Sử dụng chế độ PyTorch ForwardWarp thuần (Tắt CuPy)
Nếu không muốn sử dụng CuPy SoftSplat:
```powershell
python -m demo.interp_custom_times --frame0 ../../TestImage1/F1.png --frame1 ../../TestImage1/F2.png --times 0.5 --no_cupy --save_dir ./demo/output
```

---

## 3. Bảng Chi Tiết Tham Số của `interp_custom_times.py`

| Tham số | Bắt buộc | Mặc định | Ý nghĩa & Ví dụ |
|---|---|---|---|
| `--frame0` | **Có** | - | Đường dẫn file ảnh Frame 1 (ví dụ: `../../TestImage1/F1.png`) |
| `--frame1` | **Có** | - | Đường dẫn file ảnh Frame 2 (ví dụ: `../../TestImage1/F2.png`) |
| `--times` | Không | `None` | Danh sách mốc thời gian $t \in (0, 1)$ (ví dụ: `--times 0.2 0.25 0.32 0.5`) |
| `--num_frames` | Không | `None` | Số frame nội suy chia đều (ví dụ: `--num_frames 3`) |
| `--save_dir` | Không | `./demo/output` | Thư mục lưu các ảnh kết quả |
| `--prefix_1` | Không | `1` | Tiền tố đánh số cho Frame 1 (ví dụ: `1` $\rightarrow$ `1.000.png`, `1.xxx.png`) |
| `--prefix_2` | Không | `2` | Tiền tố đánh số cho Frame 2 (ví dụ: `2` $\rightarrow$ `2.000.png`) |
| `--digits` | Không | `3` | Số chữ số thập phân cho tên file (mặc định: `3` $\rightarrow$ `1.500.png`, `1.250.png`) |
| `--model_size` | Không | `anime_interp` | Kiểu mô hình AnimeInterp |
| `--checkpoint` / `--model_file` | Không | `./checkpoints/anime_interp_full.ckpt` | Đường dẫn file checkpoint mô hình AnimeInterp |
| `--no_cupy` | Không | `False` | Vô hiệu hóa CuPy SoftSplat, chuyển sang PyTorch thuần |
| `--cpu` | Không | `False` | Bắt buộc chạy trên CPU (mặc định tự động nhận diện GPU `cuda`) |

---

## 4. Cấu Trúc File Kết Quả Đầu Ra

Ví dụ khi chạy lệnh với `--times 0.2 0.25 0.32 0.5`:
```text
demo/output/
├── 1.000.png   (Frame 1 gốc)
├── 1.200.png   (Frame trung gian nội suy tại t = 0.20)
├── 1.250.png   (Frame trung gian nội suy tại t = 0.25)
├── 1.320.png   (Frame trung gian nội suy tại t = 0.32)
├── 1.500.png   (Frame trung gian nội suy tại t = 0.50)
└── 2.000.png   (Frame 2 gốc)
```

---

## 5. Kiến Trúc & Nguyên Lý Hoạt Động của AnimeInterp

AnimeInterp được thiết kế chuyên biệt cho việc nội suy video hoạt hình 2D (Anime/Cartoon) với 3 khối cốt lõi:
1. **Segment-Guided Matching (SGM)**: Trích xuất và so khớp các vùng màu phẳng và đường viền nét vẽ để xử lý đặc thù thiếu texture của hoạt hình.
2. **Recurrent Flow Refinement (RFR)**: Kiến trúc luồng quang học lặp lấy cảm hứng từ RAFT để xử lý các chuyển động lớn và phi tuyến tính.
3. **Forward Warping & GridNet Synthesis**: Dịch chuyển đặc trưng đa tỷ lệ và khung hình vào thời điểm $t$ thông qua Forward Warping (SoftSplat), sau đó đưa qua GridNet để tổng hợp khung hình in-between cuối cùng với độ chi tiết cao và không bị nhòe đường biên nét vẽ.
