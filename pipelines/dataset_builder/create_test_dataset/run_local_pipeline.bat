@echo off
chcp 65001 >nul
echo ======================================================================
echo 🚀 HỆ THỐNG TỰ ĐỘNG CHUẨN BỊ DATASET HYBRID CHO KAGGLE
echo ======================================================================

cd /d "%~dp0"

echo [*] Kích hoạt môi trường ảo Python .venv...
call .venv\Scripts\activate.bat

echo.
echo [1/3] Đang chạy Bước 2: Khai thác Candidate Triplets từ AnimeGoc...
python 02_extract_local_candidates.py
if %ERRORLEVEL% NEQ 0 (
    echo [!] Lỗi tại Bước 2!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [2/3] Đang chạy Bước 3: Lọc & Chuẩn hóa tập Anime Nhật từ ATD-12K...
python 03_pack_atd12k_japan.py
if %ERRORLEVEL% NEQ 0 (
    echo [!] Lỗi tại Bước 3!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [3/3] Đang chạy Bước 4: Hợp nhất & Đóng gói candidate_triplets_anime_540p.zip...
python 04_merge_and_build_kaggle_input.py
if %ERRORLEVEL% NEQ 0 (
    echo [!] Lỗi tại Bước 4!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ======================================================================
echo 🎉 HOÀN THÀNH TOÀN BỘ CÔNG ĐOẠN LOCAL!
echo File 'candidate_triplets_anime_540p.zip' đã sẵn sàng để upload lên Kaggle.
echo ======================================================================
pause
