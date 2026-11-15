"""
Module định vị thư mục gốc (REPO_ROOT) chuẩn cho toàn bộ repo.
Hỗ trợ tìm kiếm theo marker .git hoặc cặp thư mục ('src', 'training').
"""

import os


def find_repo_root(start_path: str | None = None) -> str:
    """
    Tìm kiếm thư mục gốc repo bằng cách duyệt ngược từ start_path lên trên tối đa 8 cấp.
    Marker nhận diện:
    1. Có thư mục .git
    2. Hoặc có đồng thời cả 2 thư mục 'src' và 'training'
    """
    cur = os.path.abspath(start_path or __file__)
    if os.path.isfile(cur):
        cur = os.path.dirname(cur)
    for _ in range(8):
        if os.path.isdir(os.path.join(cur, ".git")) or (
            os.path.isdir(os.path.join(cur, "src"))
            and os.path.isdir(os.path.join(cur, "training"))
        ):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    raise RuntimeError(f"Không tìm thấy thư mục gốc repo từ: {start_path or __file__}")
