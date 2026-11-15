# Tỉ lệ phóng to (2, 3 hoặc 4)
scale = 2

# Chọn trọng số không khử nhiễu để tập trung khôi phục nét
model_path2 = "weights_v3/up2x-latest-no-denoise.pth"

# Bật FP16 để tối ưu tốc độ và bộ nhớ cho RTX 3050
half = True

# Chia khối xử lý để tránh tràn 4GB VRAM (2 hoặc 3 là rất an toàn)
tile = 2

# Cache mode: 1 (tiết kiệm VRAM, chất lượng giữ nguyên)
cache_mode = 1

# Giảm nhẹ alpha để làm nổi bật và sắc nét lại các đường nét mờ
alpha = 0.85

mode = "image"
device = "cuda:0"
input_dir = "input_dir1"
output_dir = "output_dir"
