# 🔬 THỬ NGHIỆM CÁCH LY PHA 4: OCCLUSION-AWARE HEAD (TPSInbetweenOcclusion)

## 1. Không Gian Cách Ly Tuyệt Đối (100% Isolation Namespace)
| Thành phần | Sequence 2 Pha 3 Remake (Giữ Nguyên 100%) | Thử nghiệm Mới: Phase 4 Occlusion Head (Cách ly) |
| :--- | :--- | :--- |
| **Tên mã (Codename)** | `Phase3_Remake` / `M3_Best` | `Phase4_Occlusion` / `M4_Occ` |
| **Thư mục Mã nguồn Kaggle** | `kaggle_seq_p3_multilayer_tps/` | `kaggle_seq_p4_occlusion_head/` |
| **Thư mục Checkpoint** | `checkpoints/Re-train/Sequence 2 pha 3 remake/` | `checkpoints/Re-train/Sequence 4 Occlusion Head/` |
| **File Checkpoint Chính** | `model_tps_finetuned_best.pt` | `model_tps_occlusion_best.pt` |
| **Thư mục Kết quả Benchmark** | `verified_benchmark_sharpener/` | `bench_outputs_seq_p4_occlusion/` |
| **Script Suy luận / Test** | `pipelines/seq_p3_inference_pipeline.py` | `pipelines/seq_p4_occlusion_pipeline.py` |
| **Script Benchmark Đối Đầu** | `pipelines/run_verified_benchmark_suite_p3_sharp.py` | `pipelines/benchmark_pha4_vs_pha3.py` |

---

## 2. Kiến Trúc Mô Hình (`TPSInbetweenOcclusion`)
File: `kaggle_seq_p4_occlusion_head/seq_p4_occlusion_model.py`
- **Occlusion Head trong IFNet.block2**: Lớp `ConvTranspose2d` cuối cùng xuất 6 kênh thay vì 5:
  $$\text{Output} = \left[ F_{0\to t}^{(2)}, F_{1\to t}^{(2)}, M_{0\to t}^{(1)}, M_{1\to t}^{(1)} \right]$$
- **Hai mặt nạ che khuất chuẩn hóa qua Sigmoid**: $M_{0\to t}, M_{1\to t} \in [0.0, 1.0]$.
- **Bộ Trộn Trọng số Chuẩn hóa (Normalized Visibility Blending)**:
  $$W_0 = (1 - t) \cdot M_{0\to t}, \quad W_1 = t \cdot M_{1\to t}$$
  $$I_{\text{fused}} = \frac{W_0 \cdot \text{Warp}(I_0, F_{0\to t}) + W_1 \cdot \text{Warp}(I_1, F_{1\to t})}{W_0 + W_1 + \epsilon}$$
  $$I_t = \text{clamp}\left(I_{\text{fused}} + \text{residual}, 0.0, 1.0\right)$$
- **Khởi tạo Warm-Start (Zero Loss Shock)**:
  - Nạp 158/158 tensors từ `model_tps_finetuned_best.pt` (Phase 3 Remake).
  - Khởi tạo bias cho 2 kênh mặt nạ mới bằng $+2.0$ ($\text{Sigmoid}(+2.0) \approx 0.8808$, ban đầu coi mọi vùng đều nhìn thấy được), weight $= 0.0$.
  - Đảm bảo tại step 0, tỷ lệ trộn chính xác bằng $(1-t) \cdot I_0 + t \cdot I_1$, không gây sốc mất mát.

---

## 3. Hàm Mất Mát Nhận Thức Che Khuất (`OcclusionAwareLoss`)
Hàm Loss phụ trợ gồm 3 thành phần:
1. **Photometric Loss có trọng số hiển thị ($L_{\text{photo}}$)**:
   $$L_{\text{photo}} = M_{0\to t} \cdot \rho(I_{0\to t} - I_t^{\text{GT}}) + M_{1\to t} \cdot \rho(I_{1\to t} - I_t^{\text{GT}})$$
   với $\rho(x) = \sqrt{x^2 + \epsilon^2}$ (Charbonnier Loss).
2. **Ràng buộc Nhất quán Bao phủ ($L_{\text{cov}}$ - Coverage Constraint)**:
   $$L_{\text{cov}} = \text{mean}\left(\text{ReLU}(0.95 - (M_{0\to t} + M_{1\to t}))^2\right)$$
   Phạt nặng nếu mạng "lười biếng" triệt tiêu cả 2 mặt nạ về 0 để né phạt lỗi tái tạo.
3. **Làm mượt Mặt nạ theo biên nét ($L_{\text{smooth}}$ - Edge-aware Smoothness)**:
   $$L_{\text{smooth}} = \sum_{c \in \{0, 1\}} \left( |\partial_x M_c| \cdot e^{-\alpha |\partial_x I^{\text{GT}}|} + |\partial_y M_c| \cdot e^{-\alpha |\partial_y I^{\text{GT}}|} \right)$$
   Giữ cho biên của mặt nạ $M$ phẳng mịn trên nền và bám sắc nét theo đường viền nét vẽ GT.

---

## 4. Danh Sách Tệp Đã Tạo Trong Thử Nghiệm Cách Ly

1. `kaggle_seq_p4_occlusion_head/seq_p4_occlusion_model.py`:
   - Class `TPSInbetweenOcclusion`
   - Class `IFNetOcclusion`
   - Class `OcclusionAwareLoss`
   - Hàm `load_tps_occlusion_weights`
2. `kaggle_seq_p4_occlusion_head/seq_p4_occlusion_worker.py`:
   - Động cơ huấn luyện 2x GPU DDP độc lập.
   - Quản lý checkpoint tự động `OcclusionCheckpointManager` (xuất đúng 2 file `model_tps_occlusion_best.pt` và `model_tps_occlusion_last.pt`).
3. `kaggle_seq_p4_occlusion_head/generate_kaggle_occlusion_nb.py`:
   - Script tự động biên soạn notebook Kaggle tự thân hoàn chỉnh.
4. `kaggle_seq_p4_occlusion_head/train_tps_occlusion_kaggle.ipynb`:
   - Notebook hoàn chỉnh 10 cells, sẵn sàng tải lên Kaggle và bấm Run ngay lập tức.
5. `pipelines/seq_p4_occlusion_pipeline.py`:
   - Pipeline suy luận và xuất frame kèm trích xuất trực quan mặt nạ $M_0, M_1$ và Blend Weight.
6. `pipelines/benchmark_pha4_vs_pha3.py`:
   - Benchmark đối đầu 4 phiên bản trên cùng 1 biểu đồ: Pha 2 Blended vs Pha 3 Raw vs Pha 3 + Sharpener vs Pha 4 Occlusion.
7. `bench_outputs_seq_p4_occlusion/`:
   - Thư mục lưu kết quả đối đầu (ảnh 6 cột, GIF hoạt họa, bảng CSV).
