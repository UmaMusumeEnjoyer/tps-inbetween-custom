@echo off
chcp 65001 >nul
echo ===================================================
echo [TPS-Inbetween] Dang kiem tra thay doi & dong bo...
echo ===================================================
git status -s
echo.
set /p commit_msg="Nhap noi dung commit (Nhan Enter de dung mac dinh 'Update code'): "
if "%commit_msg%"=="" set commit_msg=Update code and experiments

echo.
echo Dang them file va tao commit...
git add .
git commit -m "%commit_msg%"
echo.
echo Dang day len GitHub...
git push origin main

echo.
echo ===================================================
echo [HOAN TAT] Da dong bo toan bo thay doi len GitHub!
echo ===================================================
pause
