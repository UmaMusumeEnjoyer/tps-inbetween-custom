### 🔄 HƯỚNG DẪN RESTART KERNEL & TIẾP TỤC HUẤN LUYỆN V3.0-REMASTERED (RESUME SYSTEM)

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
Sau khi kernel đã khởi động lại:
1. Nhấn nút **Run All** trên thanh công cụ notebook (hoặc chạy tuần tự từ Cell 1 đến Cell 10).
2. Đến **Cell 10**, hệ thống sẽ tự động phát hiện `model_tps_v3.0_remastered_resume.pt` và in log:
   ```text
   [✓] RESUME V3.0-R: Đã nạp trạng thái sau Epoch XX -> tiếp tục từ Epoch YY...
   ```
3. Quá trình huấn luyện sẽ tiếp tục mượt mà cho đến khi hoàn tất 25 epochs!
