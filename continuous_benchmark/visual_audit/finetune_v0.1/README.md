# KIỂM TOÁN TRỰC QUAN ĐỐI ĐẦU FINETUNE v0.1 (VISUAL AUDIT v0.1)

> **Mục đích:** Lưu trữ các hình ảnh render kiểm toán chất lượng đối đầu trực quan giữa **TPS-Inbetween Gốc (Pretrained)** và **TPS-Inbetween Fine-tuned v0.1**, bao gồm ảnh so sánh đa cột, bản đồ nhiệt sai số (difference heatmaps) và video 24fps đối chiếu.

---

## 1. CÁC TÀI NGUYÊN KIỂM TOÁN TRỰC QUAN TẠI ĐÂY

1. **`side_by_side_comparison/`**:
   - Ảnh so sánh 5 cột song song: Keyframe 0 | TPS Gốc | TPS Sửa Thuật Toán | TPS Fine-tuned v0.1 | Ground Truth.
2. **`difference_heatmaps/`**:
   - Bản đồ nhiệt thể hiện mức độ lệch nét và vùng bóng ma xám.
3. **`zoomed_details/`**:
   - Ảnh phóng to 400% các vùng chi tiết tóc mái, mắt nhân vật và viền trang phục.
4. **`continuous_video_clips/`**:
   - Video đối chiếu chuyển động liên tục 24fps (cảnh ChayGo 8s).
