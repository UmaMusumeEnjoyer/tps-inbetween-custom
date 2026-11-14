# DỰ ÁN PBL6: NỘI SUY KHUNG HÌNH HOẠT HỌA ANIME 2D VỚI TPS-INBETWEEN

> Chi tiết toàn bộ kiến trúc cây thư mục, chức năng từng module và vai trò của các thư mục clone từ GitHub đã được tổng hợp đầy đủ tại:  
> 👉 **[PROJECT_STRUCTURE.md](file:///e:/CODE/PBL/PBL6/PROJECT_STRUCTURE.md)**

---

## TỔNG QUAN ĐỀ TÀI
- **Mục tiêu:** Tự động sinh khung hình trung gian (inbetweening) cho hoạt họa anime 2D từ cặp keyframe nét vẽ (line art), giải quyết bài toán limited animation (*"on-twos"*, *"on-threes"*), giảm tải công việc vẽ tay thủ công cho họa sĩ.
- **Mô hình trọng tâm:** **TPS-Inbetween** (Thin-Plate Spline-based Interpolation for Animation Line Inbetweening).
- **Mô hình đối chứng (Baselines):** AnimeInbet (ICCV 23), AnimeInterp (CVPR 21), SAIN (ArXiv 23), uniVIP (ICASSP 25), UPR-Net (CVPR 23).
- **Công cụ trích xuất & xử lý nét:** AniLines, Anime Segmentation (IS-Net), Real-CUGAN, Anime2Sketch, SketchKeras.

Xem chi tiết hướng dẫn và phân tích tại [PROJECT_STRUCTURE.md](file:///e:/CODE/PBL/PBL6/PROJECT_STRUCTURE.md) hoặc thư mục tài liệu [docs/](file:///e:/CODE/PBL/PBL6/docs/).
