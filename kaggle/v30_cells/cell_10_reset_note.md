### 🔄 HƯỚNG DẪN RESTART KERNEL & TIẾP TỤC HUẤN LUYỆN (KHI CHẠM NGƯỠNG RAM / WATCHDOG 28 GB)

> [!CAUTION]
> **TUYỆT ĐỐI KHÔNG BẤM NÚT "FACTORY RESET" (Biểu tượng xoay tròn ↺ ở góc trên bên phải thanh công cụ Kaggle):**  
> Nút Factory Reset sẽ xóa sạch máy ảo và làm mất toàn bộ thư mục `/kaggle/working` cùng các file checkpoint đã lưu!

Khi **Cell 10** chạm ngưỡng RAM và hiển thị thông báo `[WATCHDOG]: Working Set > 28 GB`, hãy làm theo 2 bước cực kỳ đơn giản sau:

---

#### Bước 1: Tạo Cell Code mới để Reset RAM về 1 GB sạch sẽ
Nhấn nút **`+ Code`** (ngay bên dưới Cell này), copy đoạn mã dưới đây, dán vào và bấm **Run (Play)**:

```python
import os
os._exit(0)
```

> **Cơ chế:** Lệnh `os._exit(0)` sẽ yêu cầu tiến trình Python kernel cũ thoát ngay lập tức. Hệ điều hành Linux sẽ thu hồi toàn bộ 28 GB RAM về mức sạch ban đầu (~1 GB). Sau 3 giây, Kaggle sẽ tự động khởi động một kernel mới. **Toàn bộ ổ đĩa `/kaggle/working` và các checkpoint được bảo toàn nguyên vẹn 100%!**

---

#### Bước 2: Nhấn "Run All" để tiếp tục huấn luyện
Sau khi kernel đã khởi động lại (thanh RAM Kaggle UI trở về màu xanh ~1.5 GB):
1. Nhấn nút **Run All** trên thanh công cụ notebook (hoặc chạy tuần tự từ Cell 1 đến Cell 10).
2. Các Cell 1 đến Cell 9 sẽ lướt qua trong 10–15 giây.
3. Đến **Cell 10**, hệ thống sẽ tự động phát hiện `model_tps_v3.0_resume.pt` và in log:
   ```text
   [✓] RESUME: Đã nạp trạng thái sau Epoch XX -> tiếp tục từ Epoch YY...
   ```
4. Quá trình huấn luyện sẽ tiếp tục mượt mà cho đến khi hoàn tất 25 epochs!
