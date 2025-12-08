@echo off
setlocal enabledelayedexpansion

echo Restarting Conjecture EndPoint App...

REM Configuration
set "HOST=127.0.0.1"
set "PORT=8001"
set "HEALTH_URL=http://%HOST%:%PORT%/health"
set "MAX_RETRIES=30"
set "RETRY_DELAY=2"

REM Kill existing endpoint processes
echo Terminating existing endpoint processes...
for /f "tokens=2" %%i in ('tasklist /fi "imagename eq python.exe" /fo csv ^| find "endpoint_app.py"') do (
    echo Killing process %%i
    taskkill /pid %%i /f >nul 2>&1
)

REM Give processes time to terminate
timeout /t 2 /nobreak >nul

REM Start endpoint app in background (non-blocking)
echo Starting endpoint app in background...
start /B python src/endpoint_app.py --host %HOST% --port %PORT%

REM Give service a moment to start
echo Giving service time to initialize...
timeout /t 3 /nobreak >nul

REM Health check polling
echo Performing health check...
set "retry_count=0"
set "health_ok=false"

:health_check_loop
if %retry_count% geq %MAX_RETRIES% (
    echo ❌ Health check failed after %MAX_RETRIES% attempts
    echo ❌ EndPoint App may not have started correctly
    exit /b 1
)

echo Attempt !retry_count! of %MAX_RETRIES%: Checking %HEALTH_URL%

REM Use curl to check health (available on Windows 10+)
curl -s -o nul -w "%%{http_code}" %HEALTH_URL% >temp_http_code.txt 2>&1
set /p http_code=<temp_http_code.txt
del temp_http_code.txt

if "%http_code%"=="200" (
    echo ✅ Health check passed (HTTP 200)
    set "health_ok=true"
    goto health_check_success
) else (
    echo ⚠️ Health check failed with HTTP code: %http_code%
    set /a retry_count+=1
    timeout /t %RETRY_DELAY% /nobreak >nul
    goto health_check_loop
)

:health_check_success
if "%health_ok%"=="true" (
    echo ✅ EndPoint App started successfully at http://%HOST%:%PORT%
    echo 📚 Documentation available at http://%HOST%:%PORT%/docs
    echo ✅ Done! Service is running and ready in background.
    exit /b 0
) else (
    echo ❌ EndPoint App failed to start properly
    exit /b 1
)