# KẾT QUẢ ĐỐI CHUẨN ĐỊNH LƯỢNG TPS-INBETWEEN FINETUNED v0.1

> **Mục đích:** Lưu trữ các báo cáo số liệu, file CSV, JSON và Markdown đánh giá định lượng năng lực nội suy của mô hình **TPS-Inbetween Fine-tuned v0.1** sau khi tải về từ Kaggle.

---

## 1. CÁC TỆP ĐẦU RA DỰ KIẾN TẠI ĐÂY

Khi chạy kịch bản đối chuẩn `docs/finetune_plans/v0.1/03_benchmark_testing_plan_v0.1.md`, các tệp sau sẽ được tự động xuất vào thư mục này:

1. **`val_508_metrics.csv`**: Bảng điểm chi tiết (PSNR, SSIM, Ghost Pixels, Tortuosity) cho từng clip trong 508 triplets thẩm định.
2. **`summary_metrics.json`**: Giá trị trung bình và phương sai các chỉ số.
3. **`time_step_analysis.csv`**: Bảng đánh giá theo từng mốc thời gian $t \in \{0.25, 0.33, 0.5, 0.67, 0.75\}$ để chứng minh khả năng nội suy liên tục.
4. **`tps_finetuned_v0.1_benchmark_report.md`**: Báo cáo phân tích so sánh chi tiết so với mô hình gốc.

---

## 2. LỆNH CHẠY KIỂM THỬ ĐỐI CHUẨN ĐẦU RA

```bash
python continuous_benchmark/inference_scripts/run_final_test.py \
    --model_name tps-inbetween-finetuned-v0.1 \
    --checkpoint checkpoints/experiments/v0.1/model_tps_v0.1_best.pt \
    --output_dir continuous_benchmark/metrics/tps-inbetween-finetuned-v0.1
```
