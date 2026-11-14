@echo off
chcp 65001 > nul
title AniTweenStudio - Playwright Auto Interpolation (TPS V2.6)
cd /d "%~dp0"

echo =====================================================================
echo   AniTweenStudio Playwright E2E Auto Interpolation Runner (TPS V2.6)
echo =====================================================================
echo.

:: 1. Kiem tra va khoi dong Backend TPS Server tren port 8000 neu chua chay
netstat -ano | findstr :8000 | findstr LISTENING > nul
if %errorlevel% neq 0 (
    echo [*] Dang khoi dong AI TPS Server tai http://127.0.0.1:8000...
    start "AniTweenStudio TPS Server" cmd /c "run_tps_server.bat"
    echo [*] Cho 5 giay de Server load model v2.6 vao GPU...
    timeout /t 5 /nobreak > nul
) else (
    echo [v] AI TPS Server dang hoat dong tren port 8000.
)

:: 2. Chay kich ban Playwright tu dong hoa
echo.
echo [*] Dang khoi chay Playwright Automation Script...
cd product_app\AniTweenStudio
call npm run auto:benchmark

echo.
echo =====================================================================
echo   Kich ban Playwright da chay thanh cong!
echo   Anh chup ket qua luu tai: auto_benchmark_playwright_result.png
echo =====================================================================
pause
