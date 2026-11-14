### 🔄 HƯỚNG DẪN RESTART KERNEL & TIẾP TỤC HUẤN LUYỆN (KHI CẦN RESET RAM / CHẠM WATCHDOG 28 GB)

> [!CAUTION]
> **TUYỆT ĐỐI KHÔNG BẤM NÚT "FACTORY RESET" (Biểu tượng xoay tròn ↺ ở góc trên bên phải thanh công cụ Kaggle):**  
> Nút Factory Reset sẽ xóa sạch toàn bộ máy ảo và làm mất toàn bộ thư mục `/kaggle/working` cùng các file checkpoint đã lưu!

Khi **Cell 10** hiển thị thông báo `[WATCHDOG TRIGGERED]: RAM container >= 28 GB` hoặc khi bạn muốn chủ động làm mới RAM máy ảo:

---

#### Bước 1: Tạo Cell Code mới để Reset RAM về 1 GB sạch sẽ
Nhấn nút **`+ Code`** (ngay bên dưới Cell này), copy đoạn mã dưới đây, dán vào và bấm **Run (Play)**:

```python
import os
os._exit(0)
```

> **Cơ chế:** Lệnh `os._exit(0)` sẽ yêu cầu tiến trình Python kernel cũ thoát ngay lập tức. Hệ điều hành Linux sẽ thu hồi toàn bộ RAM về mức sạch ban đầu (~1.2 GB). Sau 3 giây, Kaggle sẽ tự động khởi động một kernel mới. **Toàn bộ ổ đĩa `/kaggle/working` và các checkpoint được bảo toàn nguyên vẹn 100%!**

---

#### Bước 2: Nhấn "Run All" để tiếp tục huấn luyện
Sau khi kernel đã khởi động lại (thanh RAM Kaggle UI trở về màu xanh ~1.5 GB):
1. Nhấn nút **Run All** trên thanh công cụ notebook (hoặc chạy tuần tự từ Cell 1 đến Cell 10).
2. Các Cell 1 đến Cell 9 sẽ lướt qua trong 10 – 15 giây.
3. Đến **Cell 10**, hệ thống sẽ tự động phát hiện `model_tps_v3.2_resume.pt` và in log:
   ```text
   [✓] RESUME V3.2: Đã nạp trạng thái sau Epoch XX -> tiếp tục từ Epoch YY...
   ```
4. Quá trình huấn luyện sẽ tiếp tục mượt mà cho đến khi hoàn tất 25 epochs!
