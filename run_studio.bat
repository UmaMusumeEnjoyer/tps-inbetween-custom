@echo off
chcp 65001 > nul
title AniTweenStudio - 2D Desktop Studio
cd /d "%~dp0"

echo =====================================================================
echo   AniTweenStudio Desktop Application Launcher
echo =====================================================================
echo.

:: 1. Kiem tra AI Server tren cong 8000
netstat -ano | findstr :8000 | findstr LISTENING > nul
if %errorlevel% neq 0 (
    echo [*] AI Backend Server chua chay. Dang tu dong khoi dong tai http://127.0.0.1:8000...
    start "AniTweenStudio TPS Server" cmd /c "scripts\run_tps_server.bat"
    echo [*] Cho 4 giay de AI Server khoi dong...
    timeout /t 4 /nobreak > nul
) else (
    echo [v] AI Backend Server dang san sang tren http://127.0.0.1:8000.
)

:: 2. Khoi dong giao dien AniTweenStudio
echo.
echo [*] Dang khoi dong giao dien AniTweenStudio...
cd product_app\AniTweenStudio
call npm run dev
