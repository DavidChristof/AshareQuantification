@echo off
setlocal
cd /d "%~dp0"
title A-share Quant Launcher

REM ========== config (edit if needed) ==========
set "PYTHON=D:\Python\Python3_12\python.exe"
set "PORT=8001"
set "LOG_FILE=logs\api.log"
REM =============================================

if not exist "logs" mkdir "logs"

echo ============================================
echo    A-share Quant Monitor - Launcher
echo ============================================
echo.

REM 1. if port is listening, try to verify the API
powershell -NoProfile -Command "try{$c=New-Object Net.Sockets.TcpClient;$c.Connect('127.0.0.1',%PORT%);$c.Close();exit 0}catch{exit 1}"
if "%errorlevel%"=="0" (
    powershell -NoProfile -Command "try{$r=Invoke-WebRequest -Uri 'http://127.0.0.1:%PORT%/api/stocks' -TimeoutSec 5 -UseBasicParsing;if($r.StatusCode -eq 200){exit 0}}catch{};exit 1"
    if "%errorlevel%"=="0" (
        echo [OK] Service already running on port %PORT%.
        goto ready
    ) else (
        echo [WARN] Port %PORT% is in use but API not responding.
        echo        Another process may occupy it. Will try to start anyway.
        echo.
    )
)

REM 2. python exists?
if not exist "%PYTHON%" (
    echo [ERROR] Python not found: %PYTHON%
    echo          Edit the PYTHON variable at the top of this file.
    echo.
    pause
    exit /b 1
)

REM 3. start API (minimized window), log to file (real-time via -u)
echo [START] Launching API on port %PORT% ...
start "A-share-Quant-API (port %PORT%)" /min "%PYTHON%" -u -m uvicorn api.main:app --host 127.0.0.1 --port %PORT% > "%LOG_FILE%" 2>&1

REM 4. wait until ready (progress dots, ~2s each, max 6 min)
echo [WAIT] Warming up model ^& data, please wait ...
set /a N=0
:wait
powershell -NoProfile -Command "try{$r=Invoke-WebRequest -Uri 'http://127.0.0.1:%PORT%/api/stocks' -TimeoutSec 2 -UseBasicParsing;if($r.StatusCode -eq 200){exit 0}}catch{};exit 1"
if "%errorlevel%"=="0" goto ready
set /a N+=1
if %N% geq 180 goto timeout
<nul set /p="."
ping -n 3 127.0.0.1 >nul
goto wait

:ready
echo.
echo [DONE] Service ready. Opening dashboard ...
start "" "%~dp0frontend\index.html"
echo.
echo   Dashboard opened.  Data: http://127.0.0.1:%PORT%
echo   API window: minimized "A-share-Quant-API (port %PORT%)".
echo       Close that window to stop the service.
echo   Log file: %LOG_FILE%
echo   You may close this launcher window now.
echo.
pause
exit /b 0

:timeout
echo.
echo [ERROR] Startup timeout after ~6 minutes.
echo          Check the API window or log file: %LOG_FILE%
echo          Common causes: port occupied, network blocked, missing deps.
echo.
pause
exit /b 1


