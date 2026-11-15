# HUẤN LUYỆN TINH CHỈNH TPS-INBETWEEN PHIÊN BẢN V3.0 (ASYMMETRIC STROKE & OCCLUSION-AWARE GENERATION - 540P SAFE MEMORY)
## ĐỀ TÀI: NGHIÊN CỨU & TINH CHỈNH MÔ HÌNH TPS-INBETWEEN CHO HOẠT HỌA ANIME 2D (PBL6)

> **Mục tiêu & Đột Phá Kỹ Thuật Cốt Lõi Phiên Bản V3.0 (Official Production 14-Cell Interactive Release):**
> 1. **Kiến Trúc Hiển Thị Kép Bất Đối Xứng (Cell 5):** Phá bỏ hoàn toàn ràng buộc pha trộn tuyến tính $V_0 + V_1 = 1.0$. Trang bị `AsymmetricStrokeFusionHead_v3` xuất ra 2 cổng hiển thị độc lập $(V_0, V_1) \in [0, 1]^2$ kết hợp `AsymmetricFeatureAnalyzer` và `ContourGuidedFlowPropagator` để tự động bảo toàn 100% độ đậm nét cho các chi tiết đơn nguồn (mắt, miệng, tai, nếp áo) khi nhân vật quay đầu ($A \to B$ và $B \to A$).
> 2. **Khởi Tạo Kaiming Phá Bẫy Đối Xứng & 2-Stage Curriculum (Cell 5 & 6):** Khởi tạo Kaiming Normal ($a=0.25$) cho các lớp PReLU Conv và Small Normal ($\sigma=10^{-3}$) cho các lớp xuất cổng/dòng, phá bỏ triệt để bẫy 0-gradient ban đầu. Lịch trình 2 giai đoạn: Epoch 1–2 đóng băng backbone + warmup êm các nhánh V3 mới; Epoch 3–25 mở toàn mạng kết hợp Cosine Decay.
> 3. **Bảo Toàn Đỉnh Ngọn Tóc & Chuẩn Hóa Mật Độ Nét Chuẩn Vàng 1.10x (Cell 6):**
>    - `HairTipEndpointContinuityLoss` ($\lambda = 0.30$) với cơ chế dò tìm điểm đầu mút trên khung xương nét vẽ (`skel_gt`), kích hoạt trọng số focal $4.0\times$ để triệt tiêu hoàn toàn đứt nét ở đỉnh các lọn tóc mảnh.
>    - `StrokeDensityRegularizationLoss_v3` với dải Dead-zone chuẩn $[1.04, 1.14]$ (tâm chuẩn lý tưởng **$1.10\times$**), kết hợp Exact-STE tại ngưỡng $0.15$ và nhánh Leaky Gradient.
>    - `OcclusionAwareGhostLoss` tích hợp Background Cleanliness Penalty ($0.35 \cdot \text{ReLU}(p_{\text{f}} - 0.008)$) triệt tiêu màng sương và bóng ma xa.
> 4. **An Toàn Tuyệt Đối VRAM & Host RAM Trên Tập Gốc 540p (Cell 1, 3, 4, 10):**
>    - Huấn luyện trên tập gốc 540p (`sakuga_golden_dataset_tps_ready_v1_patch3`, 8,554 triplets, $540 \times 960$) với Random Crop $512 \times 512$.
>    - `BATCH_SIZE = 8` (4 mẫu/GPU trên 2x T4), `GRAD_ACCUM_STEPS = 2` $\implies$ Effective Batch Size = 16, VRAM $< 9.2\text{ GB}$/GPU (tránh OOM trên GPU 0 khi tính VGG + clDice).
>    - Tiền tính toán cache TPS dạng `.npy` uncompressed kết hợp `np.load(..., mmap_mode='r')` (0 heap RAM, zero Copy-on-Write page duplication giữa DataLoader workers).
>    - Tự động giải phóng Linux PageCache qua `posix_fadvise(POSIX_FADV_DONTNEED)` khi đọc ảnh và dọn dẹp RAM vật lý (`cleanup_memory()`, `malloc_trim(0)`) sau mỗi epoch.
> 5. **Validation Thuần FP32 (Cell 7, 8, 10):** Tắt hoàn toàn autocast FP16 khi thẩm định (`use_amp=False`) với `torch.inference_mode()`, unpad từ 544 về đúng độ phân giải gốc $540 \times 960$, đo lường $100\%$ chỉ số trên GPU không làm sai lệch mật độ nét.
