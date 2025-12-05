@echo off
if "%1"=="start" goto start
if "%1"=="stop" goto stop
if "%1"=="status" goto status
if "%1"=="restart" goto restart

echo Usage: %0 [start^|stop^|status^|restart]
echo   start   - Start LLM Local Router Service
echo   stop    - Stop LLM Local Router Service  
echo   status  - Check service status
echo   restart - Restart LLM Local Router Service
goto end

:start
echo Starting LLM Local Router Service...
start /B python llm_local_router_service.py
timeout /t 2 >nul
echo Service started on http://localhost:5677
goto end

:stop
echo Stopping LLM Local Router Service...
taskkill /F /IM python.exe >nul 2>&1
echo Service stopped
goto end

:status
echo Checking LLM Local Router Service status...
curl -s http://localhost:5677/health >nul 2>&1
if %errorlevel%==0 (
    echo Service is RUNNING on http://localhost:5677
) else (
    echo Service is STOPPED
)
goto end

:restart
echo Restarting LLM Local Router Service...
call %0 stop
timeout /t 2 >nul
call %0 start
goto end

:end