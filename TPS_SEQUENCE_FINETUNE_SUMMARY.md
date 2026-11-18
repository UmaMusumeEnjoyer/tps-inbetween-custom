# TỔNG HỢP FINETUNE: TỪ TPS GỐC ĐẾN PIPELINE MỚI NHẤT
(Bản tối ưu hiển thị trên điện thoại - không dùng dấu sao định dạng)

--------------------------------------------------

1. CỐT LÕI: CÓ ĐỔI CẤU TRÚC HAY THAM SỐ KHÔNG?

- Kiến trúc mạng (TPSInbetweenOriginal):
  + Giữ nguyên 100% (0% thay đổi cấu trúc).
  + Giữ nguyên mọi lớp tích chập, số kênh, hàm kích hoạt và công thức hòa trộn ảnh đầu ra.
- Số tham số hoạt động (Active Parameters):
  + Giữ nguyên chính xác 10,068,668 tham số (158 tensors).
- Lược bỏ các nhánh thừa (Dead Keys) từ checkpoint gốc:
  + File gốc model_latest.pt chứa 252 tensors (13.95M tham số) do dính code thừa từ RIFE không dùng đến (block_tea, contextnet, unet bên trong IFNet).
  + Quá trình Sequence đã lọc bỏ 94 tensors thừa (3.88M tham số) này, chỉ giữ đúng 158 tensors hoạt động và lưu kèm trọng số làm mượt EMA.

--------------------------------------------------

2. PHẦN NÀO ĐÃ ĐƯỢC FINETUNE?

Toàn bộ 158/158 tensors (10.07M tham số) đều được mở khóa huấn luyện (requires_grad = True), chia làm 2 nhóm tốc độ học:

- Nhóm Backbone (Contextnet + IFNet) - Tốc độ học thấp:
  + Contextnet (Encoder): 24 tensors (293,712 tham số). Tinh chỉnh nhẹ để bắt đặc trưng nét vẽ mảnh hoạt họa.
  + IFNet (Block 0, 1, 2): 96 tensors (6,819,675 tham số). Tinh chỉnh luồng chuyển động tinh và mặt nạ pha trộn giữa 2 ảnh.
- Nhóm UNet tinh chỉnh nét - Tốc độ học cao hơn:
  + OriginalUnet: 38 tensors (2,955,281 tham số).
  + Thay đổi tham số mạnh nhất (độ lệch lớn gấp 3 lần Backbone).
  + Đảm nhận: xóa bóng ma (Ghost), chống nở dày nét (giữ tỷ lệ nét chuẩn ~1.0) và giữ liền mạch sợi tóc mảnh.

--------------------------------------------------

3. NHỮNG LỚP NÀO GIỮ NGUYÊN / ĐÓNG BĂNG?

- Trong mạng chính (Chế độ train chuẩn):
  + Không đóng băng lớp nào. Toàn bộ 158 tensors đều được cập nhật giá trị mới.
- Các mô-đun phụ trợ bên ngoài (Đóng băng 100%):
  + GlueStick + SuperPoint: Giữ nguyên trọng số pretrained, chỉ dùng trích xuất điểm nối giữa 2 khung hình.
  + VGG16 Perceptual Loss: Khóa cứng toàn bộ tham số, chỉ dùng để đo sai số thị giác khi huấn luyện.

--------------------------------------------------

4. TÓM TẮT TIẾN HÓA QUA CÁC PHA

- M1 gốc (model_latest.pt): Dễ đứt nét mảnh, bị bóng mờ nền khi nhân vật di chuyển xa.
- Sequence V1 & Run 1: Sửa quy ước uốn lưới Backward-Warp, bổ sung đệm tỷ lệ khung hình chuẩn 540p.
- Sequence 2 Pha 1: Bổ sung bộ Loss 8 thành phần, khóa chặt độ dày nét chuẩn (~1.0) và giữ ngọn tóc mảnh.
- Sequence 2 Pha 2: Tăng phạt bóng mờ, giảm 69% hiện tượng nhấp nháy nền.
- Sequence 2 Blended (Model Soup): Hòa trộn trọng số tuyến tính (55% Pha 1 + 45% Pha 2) để vừa đậm nét vừa sạch nền.
- Pha 3 Remake (Bản hiện tại trong PIPELINE_LATEST.md): Khởi tạo từ Blended, tinh chỉnh nhẹ thích nghi với trường dòng chảy đa lớp (Multi-layer TPS).

--------------------------------------------------

5. THAY ĐỔI Ở TẦNG PIPELINE (NGOÀI MẠNG)

Mạng neural giữ nguyên nhưng quy trình xử lý 3 tầng của pipeline được nâng cấp toàn diện:

- Tầng 1 - Coarse TPS Đa lớp:
  + Tách riêng lớp nền tĩnh (vận tốc <= 1.5 px) và các cụm chuyển động động, triệt tiêu lỗi méo nền tĩnh.
  + Tự động khóa các góc đa giác (Polygon-Aware) để góc vuông không bị cụt/mờ khi co giãn khung hình.
  + Hỗ trợ luồng trực tiếp tại thời điểm t (direct_t_flows = True).
- Tầng 2 - Mạng UNet:
  + Chạy suy luận chuẩn xác ở chế độ FP32 với bộ trọng số EMA tốt nhất.
- Tầng 3 - Bộ lọc nét cục bộ (Adaptive Sharpener):
  + Bộ lọc tương phản thông minh dựa trên bản đồ khoảng cách nét, kéo đen đanh lõi nét và xóa sạch 63% quầng xám mờ trên nền trắng.
