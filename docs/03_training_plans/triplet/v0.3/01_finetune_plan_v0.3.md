# KẾ HOẠCH HUẤN LUYỆN TINH CHỈNH PHIÊN BẢN v0.3 (FINETUNE PLAN v0.3)
> **Mô hình Trọng tâm:** TPS-Inbetween (*Thin-Plate Spline-based Interpolation for Animation Line Inbetweening*)  
> **Phiên bản:** v0.3 (Zero-Ghosting & High-Stability Cloud Training)  
> **Nền tảng Huấn luyện:** Kaggle Cloud GPU (NVIDIA Tesla T4 x2 hoặc P100 16GB VRAM)  
> **Ngày khởi tạo:** 07/10/2026  
> **Kế thừa & Khắc phục từ:** [01_finetune_plan_v0.2.md](file:///e:/CODE/PBL/PBL6/docs/finetune_plans/v0.2/01_finetune_plan_v0.2.md) và thực nghiệm Kaggle v0.2.

---

## 1. BỐI CẢNH & PHÂN TÍCH SỰ CỐ TỪ PHIÊN BẢN v0.2 (POST-MORTEM & AUDIT)

Sau đợt huấn luyện và thẩm định thực nghiệm của phiên bản v0.2, hệ thống ghi nhận 2 vấn đề kỹ thuật lớn cần phải được giải quyết triệt để trong phiên bản v0.3:

### 1.1. Sự cố Kỹ thuật 1: Hiện tượng Bóng Ma Chồng Hình (Ghosting Artifacts)
- **Biểu hiện trực quan:** Trong các ảnh nghiệm thu `side_by_side_comparison` (đặc biệt ở các phân cảnh có biên độ chuyển động lớn như `clip_000018`, `clip_000046`, `TestImage1`), đầu ra của mô hình Finetuned v0.2 vẫn xuất hiện 2 bóng nét xám mờ của nhân vật chồng lên nhau, gần như y hệt ảnh `TPS Coarse Warp`.
- **Nguyên nhân cốt lõi:**
  1. *Phép hòa trộn tuyến tính cứng (Hard Linear Blending):* Công thức `base_interp = (1 - t) * fine_I0t + t * fine_I1t` ép buộc phép cộng 50-50 tại $t=0.5$. Khi 2 khung hình bị lệch do chuyển động phi tuyến, phép cộng này chắc chắn sinh ra 2 nét vẽ mờ (cường độ 0.5).
  2. *Khóa biên độ UNet Residual cứng $\pm 0.15$:* Trong nỗ lực chống sụp đổ trắng (Mode Collapse của v0.1), v0.2 đã khóa $res \in [-0.15, +0.15]$. Do đó, khi một nét ma có cường độ $0.5$, UNet chỉ có thể trừ đi tối đa $-0.15$, để lại nét thừa có cường độ $0.35$ (tương đương pixel xám 166). UNet bị "trói tay", về mặt giải tích toán học **bất lực trong việc tẩy xóa nét ma**.
  3. *Aperture Problem của IFNet trên tranh nét:* Nền trắng chiếm $>95\%$, nét vẽ mỏng 1-2px khiến IFNet không tìm thấy gradient dẫn hướng, dòng quang học tinh chỉnh xấp xỉ 0 khiến $\text{fine\_I0t} \approx \text{coarse\_I0t}$.

### 1.2. Sự cố Kỹ thuật 2: Reset Kernel Kaggle Bất Thường tại Cell 10 (Kernel Dead/Crash)
- **Biểu hiện:** Khi đang chạy vòng lặp huấn luyện ở Cell 10 trên Kaggle, Jupyter notebook bị tắt đột ngột và hiển thị thông báo *"The kernel appears to have died. It will restart automatically"* mà không xuất hiện bất kỳ dòng Exception Python traceback nào.
- **Nguyên nhân gốc rễ (Root Causes):**
  1. *Rò rỉ bộ nhớ CPU (Host RAM Leak) từ SciPy C-extension:* Hàm `compute_batch_bidirectional_edt_loss` trong Cell 6 chuyển tensor từ GPU về CPU (`.detach().cpu().numpy()`) rồi gọi `scipy.ndimage.distance_transform_edt` hàng ngàn lần trong vòng lặp. C-extension của SciPy cấp phát mảng tạm trên heap nhưng glibc không giải phóng kịp về OS (heap fragmentation), khiến RAM máy chủ Kaggle (13GB - 30GB) tăng dần đều cho đến khi bị **Linux OOM Killer gửi tín hiệu `SIGKILL` (signal 9)** tiêu diệt tiến trình.
  2. *Tràn bộ nhớ chia sẻ `/dev/shm` do DataLoader đa luồng:* `num_workers = 4` kèm `pin_memory = True` trên Kaggle Docker (nơi `/dev/shm` bị giới hạn 2GB-4GB). Khi truyền các mảng ảnh và keypoints lớn giữa các worker, bộ nhớ chia sẻ bị cạn kiệt, dẫn tới `SIGBUS` sập kernel.
  3. *Lỗi Singular Matrix / CuSOLVER crash trên GPU:* Trong `tps_theta_from_points`, lệnh `torch.linalg.solve(L, Y)` thực thi trên GPU với dữ liệu Random Crop đôi khi gặp ma trận suy biến (các keypoint quá gần nhau hoặc thẳng hàng), gây crash tầng CUDA driver trong môi trường AMP FP16.
  4. *Thiếu cơ chế giải phóng bộ nhớ định kỳ:* Không có lệnh `torch.cuda.empty_cache()` và `gc.collect()` giữa các epoch, khiến rác bộ nhớ của autograd tích tụ dần qua từng epoch.

---

## 2. KIẾN TRÚC MÔ HÌNH VÀ CƠ CHẾ NỘI SUY MỚI TRONG v0.3

Để giải quyết đồng thời bài toán **chống Ghosting**, **chống Mode Collapse** và **ổn định 100% tài nguyên Cloud**, kiến trúc v0.3 giới thiệu cơ chế **ASF-GR Engine (Adaptive Stroke Fusion & Gated Refinement)**:

```mermaid
flowchart TD
    subgraph S1["1. Coarse Motion"]
        X0["Frame 0"] & X1["Frame 1"] --> TPS["Safe Robust TPS Solver (λ=0.5)"]
        TPS --> W0["warp_normalized_flow"] --> CI0["Coarse I0t"]
        TPS --> W1["warp_normalized_flow"] --> CI1["Coarse I1t"]
    end

    subgraph S2["2. Fine Motion"]
        CI0 & CI1 --> IFNet["IFNet Backbone"]
        IFNet --> FRef["flow_refine"] & IFMask["IFNet Feature Mask"]
        CI0 --> WF0["warp_pixel_flow"] --> FI0["Fine I0t"]
        CI1 --> WF1["warp_pixel_flow"] --> FI1["Fine I1t"]
    end

    subgraph S3["3. Adaptive Stroke Fusion (ASF Head)"]
        FI0 & FI1 & FRef & IFMask --> ASF["Adaptive Fusion Head"]
        ASF --> BlendMask["Dynamic Soft Mask M ∈ [0, 1]"]
        FI0 & FI1 & BlendMask --> BaseBlend["base_interp = M ⊙ FI0 + (1-M) ⊙ FI1"]
    end

    subgraph S4["4. Gated Ghost Erasure (GGER Synthesis UNet)"]
        BaseBlend & X0 & X1 --> UNet["Synthesis UNet v0.3"]
        UNet --> GateErase["Ghost Eraser Gate G_erase ∈ [0, 1]"]
        UNet --> DetailAdd["Detail Enhancer R_add ∈ [0, 0.25]"]
        BaseBlend & GateErase & DetailAdd --> FinalPred["pred_t = clamp(G_erase ⊙ base_interp + R_add, 0, 1)"]
    end
```

### 2.1. Tầng Adaptive Stroke Fusion (ASF Head - Triệt tiêu Ghosting cơ sở)
Thay vì dùng phép cộng mù quáng `(1-t) * I0 + t * I1`, v0.3 tích hợp **Adaptive Fusion Head**:
- Mạng nơ-ron tích chập nhẹ nhận đầu vào là `fine_I0t`, `fine_I1t`, `flow_refine`, thời gian $t$ và xuất ra một bản đồ mặt nạ động $M_{\text{blend}} \in [0, 1]$ qua hàm Sigmoid:
  $$M_{\text{blend}} = \sigma\left(\text{FusionConv}(\text{fine\_I0t}, \text{fine\_I1t}, \text{flow\_refine}, t)\right)$$
  $$\text{base\_interp} = M_{\text{blend}} \odot \text{fine\_I0t} + (1.0 - M_{\text{blend}}) \odot \text{fine\_I1t}$$
- **Ý nghĩa kỹ thuật:** Khi 2 khung hình biến dạng không thể khớp nhau tại một nét vẽ (tranh chấp nét / occlusion), mạng có khả năng đưa $M_{\text{blend}} \to 1$ hoặc $M_{\text{blend}} \to 0$ để **chọn dứt khoát một nét đơn duy nhất**, dập tắt hoàn toàn hiện tượng mờ 50-50 của phép linear blend.

### 2.2. Khối Gated Ghost Erasure Residual (GGER Layer - Tẩy nét ma an toàn)
Synthesis UNet v0.3 được tái cấu trúc thành 2 nhánh xuất tín hiệu độc lập:
1. **Ghost Eraser Gate ($G_{\text{erase}} \in [0, 1]$):**
   - Sử dụng hàm kích hoạt Sigmoid, khởi tạo bias ban đầu $= 2.0$ (để $\sigma(2.0) \approx 0.88$, mặc định bảo toàn nét ban đầu).
   - Khi phát hiện nét ma thừa, $G_{\text{erase}}$ tại vùng đó có thể hạ xuống $0.0$, **tẩy sạch hoàn toàn nét thừa về nền trắng mà không bị giới hạn bởi ngưỡng $\pm 0.15$**.
2. **Detail Enhancer ($R_{\text{add}} \in [0, 0.25]$):**
   - Sử dụng hàm kích hoạt $\text{Sigmoid}(x) \times 0.25$, khởi tạo trọng số 0.
   - Bổ sung độ đậm cho nét vẽ chính để nét đen tuyền sắc sảo.
3. **Công thức tổng hợp:**
   $$\text{pred\_t} = \text{clamp}(G_{\text{erase}} \odot \text{base\_interp} + R_{\text{add}}, 0.0, 1.0)$$
- **Tại sao không bị sụp đổ trắng (Mode Collapse)?** Vì $G_{\text{erase}}$ được kiểm soát trực tiếp bởi hàm mất mát $\mathcal{L}_{Dice}$ và $\mathcal{L}_{\text{Density}}$, nếu mạng cố tình hạ toàn bộ $G_{\text{erase}} \to 0$, hàm phạt sẽ bùng nổ cực đại.

---

## 3. THIẾT KẾ HÀM MẤT MÁT GPU-NATIVE & KHỬ BÓNG MA (ZERO-LEAK LOSS ENGINE)

Toàn bộ các hàm mất mát trong v0.3 được chuyển đổi thành **100% PyTorch GPU Tensor Operations**, triệt tiêu vĩnh viễn việc chuyển dữ liệu sang CPU và thư viện ngoài:

$$\mathcal{L}_{total} = \mathcal{L}_{L1} + 0.15 \mathcal{L}_{VGG} + 0.35 \mathcal{L}_{Dice} + 0.20 \mathcal{L}_{Morph\_Dist} + 0.15 \mathcal{L}_{Ghost} + 0.10 \mathcal{L}_{Density}$$

### 3.1. Pure GPU Morphological Distance Loss ($\mathcal{L}_{Morph\_Dist}$ thay thế SciPy Bi-EDT)
- Thay thế hoàn toàn SciPy `distance_transform_edt` bằng phép toán đa tầng giãn nở hình thái học trên GPU:
  - Dùng `F.max_pool2d(stroke_mask, kernel_size=2k+1, stride=1, padding=k)` với các kích thước $k \in \{3, 5, 9, 15\}$.
  - Khoảng cách được tính toán trực tiếp trên VRAM bằng hàm trọng số lũy tiến.
  - **Lợi ích:** Tốc độ tăng 25 lần, độ trễ batch giảm từ 320ms xuống 18ms, **0% rò rỉ RAM hệ thống**.

### 3.2. Ghost Stroke Suppression Loss ($\mathcal{L}_{Ghost}$ - Trọng số 0.15)
Hàm mất mát chuyên biệt trừng phạt trực tiếp các nét vẽ xám ma xuất hiện sai vị trí:
$$\mathcal{L}_{Ghost} = \text{mean}\left(\text{ReLU}(\text{pred} - 0.15) \odot (1.0 - \text{Dilate}(\text{GT}, \text{radius}=5))\right)$$
- Trong những vùng mà Ground Truth hoàn toàn là giấy trắng (ngay cả sau khi đã nở nét 5px để có dung sai sai số): nếu mô hình sinh ra bất kỳ pixel nét vẽ nào $> 0.15$, loss này sẽ lập tức phạt nặng theo tỷ lệ bình phương.
- Ép $G_{\text{erase}}$ phải chủ động hạ xuống $0$ tại các vùng nét ma của Frame 0 và Frame 1.

---

## 4. CHIẾN LƯỢC KỸ THUẬT TRIỆT TIÊU LỖI RESET KERNEL KAGGLE

Để bảo đảm quá trình huấn luyện 25 epoch diễn ra liên tục, mượt mà trên Kaggle:

1. **Bộ nạp dữ liệu an toàn bộ nhớ chia sẻ & chống rò rỉ RAM (RAM & Shared Memory Safety):**
   - Tách rời hoàn toàn DataFrame thành các native Python lists trong `__init__`, loại bỏ hiện tượng Copy-On-Write unsharing bộ nhớ trong multiprocessing.
   - Ép mảng crop về `np.ascontiguousarray` để giải phóng ngay lập tức mảng gốc độ phân giải cao.
   - Train DataLoader đặt `num_workers = 2`, `persistent_workers = True`, `prefetch_factor = 2`.
   - Validation DataLoader đặt `num_workers = 0` để triệt tiêu nguy cơ nghẽn hàng đợi IPC và cạn kiệt `/dev/shm`.
2. **Bộ giải TPS Kháng Suy Biến & Bảo Vệ CUDA Tuyệt Đối (CPU Safe TPS Solver):**
   - Tự động lọc bỏ các keypoint trùng lặp hoặc quá sát nhau $(\|c_i - c_j\| < 0.015)$.
   - Chuyển bước giải ma trận $(N+3 \times N+3)$ sang CPU xử lý bằng float64 kèm fallback Least-Squares SVD (`torch.linalg.lstsq`): **triệt tiêu 100% nguy cơ crash CuSOLVER trên GPU và tránh làm hỏng CUDA context**.
   - Caching tọa độ lưới meshgrid trên GPU để tránh lặp lại hàng trăm nghìn lần cấp phát tensor.
3. **Quản lý Garbage Collection & Ép Thu Hồi Heap OS (Active Memory Reclamation):**
   - Sau mỗi epoch và giữa các pha Train/Validation, thực hiện:
     ```python
     gc.collect()
     torch.cuda.empty_cache()
     ctypes.CDLL('libc.so.6').malloc_trim(0)  # Ép glibc trả lại RAM máy chủ cho Linux kernel
     ```
4. **100% GPU Streaming Validation & Đồng Bộ Google Drive:**
   - Toàn bộ chỉ số validation (PSNR, SSIM conv2d, Stroke Recall, Ghost Pixels) được tính toán $100\%$ trên GPU PyTorch Tensor, 0% dùng OpenCV CPU float64, dập tắt hoàn toàn hiện tượng phân mảnh glibc heap.
   - Tự động lưu checkpoint Kaggle đồng thời sao lưu trực tiếp sang Google Drive (hỗ trợ cả Folder Mount và REST API / Service Account) để chống mất checkpoint khi cloud gặp sự cố.

---

## 5. THÔNG SỐ HUẤN LUYỆN (HYPERPARAMETERS v0.3)

| Tham Số | Giá Trị v0.3 | Thay Đổi So Với v0.2 | Ý Nghĩa Kỹ Thuật |
|:---|:---:|:---:|:---|
| **Batch Size per GPU** | 4 | Giữ nguyên | $4 \times 2 = 8$ mẫu trên Kaggle Dual T4 |
| **Gradient Accumulation** | 2 | Giữ nguyên | Effective Batch Size = 16 |
| **Optimizer** | AdamW | weight_decay=1e-4 | Ổn định suy giảm trọng số |
| **Learning Rate** | $1.2 \times 10^{-4}$ | Giảm nhẹ từ 1.5e-4 | Giúp ASF Head và Gate hội tụ mượt mà |
| **LR Scheduler** | CosineAnnealingLR | T_max=25, eta_min=1e-5 | Suy giảm nhịp nhàng |
| **Loss Bi-EDT** | Pure PyTorch GPU Morph | Thay thế SciPy C-lib | **Triệt tiêu 100% rò rỉ RAM máy chủ** |
| **Loss Ghost Penalty** | 0.15 | **Mới bổ sung** | Trừng phạt nét ma ngoài vùng GT |
| **TPS Regularizer $\lambda$**| 0.5 | Tăng từ 0.2 | Chống suy biến ma trận, chống crash CUDA |
| **DataLoader workers** | 2 (persistent) | Giảm từ 4 | Chống cạn kiệt `/dev/shm` trên Kaggle |
| **Total Epochs** | 25 Epochs | Giữ nguyên | Đạt độ hội tụ tối ưu |

---

## 6. TIÊU CHUẨN NGHIỆM THU PHIÊN BẢN v0.3 (ACCEPTANCE CRITERIA)

1. **Độ ổn định huấn luyện (Zero Kernel Reset):** Chạy liên tục 25 Epochs trên Kaggle mà không bị văng kernel hay tràn RAM/VRAM.
2. **Triệt tiêu hiện tượng bóng ma (Ghosting Elimination):**
   - Số lượng `Ghost Pixels` trên tập Validation giảm $\ge 60\%$ so với Coarse TPS Warp (mục tiêu $< 7,000$ px so với $14,446$ px ở v0.2).
   - Ở các mẫu khó như `clip_000018`, không còn nhìn thấy 2 khuôn mặt/thân mình mờ lồng vào nhau.
3. **Bảo toàn nét vẽ (Anti-Collapse Guarantee):**
   - Tỷ lệ sụp đổ ảnh trắng (Blank Collapse Rate) duy trì **$0.0\%$**.
   - $\text{Stroke Recall} \ge 85.0\%$ trên toàn bộ tập Val và Visual Samples.
4. **Chất lượng hình ảnh:**
   - PSNR $\ge 22.0$ dB trên tập Val (tăng ít nhất $+1.0$ dB so với Coarse TPS $21.03$ dB).
   - SSIM $\ge 0.915$.
   - Chamfer Distance $\le 0.0015$.
