@echo off
chcp 65001 > nul
title AniTweenStudio - TPS V2.6 HTTP Server
cd /d "%~dp0\.."

set PYTHON_EXE=.venv\Scripts\python.exe
if not exist "%PYTHON_EXE%" (
    set PYTHON_EXE=python
)

echo =====================================================================
echo   AniTweenStudio TPS-Inbetween V2.6 Local HTTP Server
echo   Khoi dong Backend suy luan AI tren GPU/CPU tai http://127.0.0.1:8000
echo =====================================================================

"%PYTHON_EXE%" product_app\AniTweenStudio\scripts\tps_server.py --port 8000

pause
