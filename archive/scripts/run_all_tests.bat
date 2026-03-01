@echo off
:: Conjecture Test Runner
:: Runs all provider tests and model comparisons

echo ===============================================
echo Conjecture Test Suite
echo ===============================================
echo.

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    exit /b 1
)

:: Set project root
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..

:: Change to project root
cd /d "%PROJECT_ROOT%"

:: Install dependencies if needed
echo Checking dependencies...
pip show aiohttp >nul 2>&1
if errorlevel 1 (
    echo Installing aiohttp...
    pip install aiohttp
)

pip show datasets >nul 2>&1
if errorlevel 1 (
    echo Installing datasets...
    pip install datasets
)

:: Step 1: Start Conjecture Provider
echo.
echo [1/3] Starting Conjecture Provider...
start "Conjecture Provider" python "%SCRIPT_DIR%\start_conjecture_provider.py"
echo Waiting for provider to initialize...
timeout /t 10 /nobreak >nul

:: Step 2: Test Provider Functionality
echo.
echo [2/3] Testing Conjecture Provider...
python "%SCRIPT_DIR%\test_conjecture_provider.py"
if errorlevel 1 (
    echo WARNING: Conjecture provider tests failed
    echo Continuing with model comparison tests...
)

:: Step 3: Run 4-Model Comparison
echo.
echo [3/3] Running 4-Model Comparison...
python "%SCRIPT_DIR%\run_4model_comparison.py"

echo.
echo ===============================================
echo Test Suite Complete
echo ===============================================
echo.
echo Results saved to:
echo   - research\results\4model_comparison_results_*.json
echo   - research\results\4model_comparison_summary_*.json
echo   - research\results\4model_comparison_report_*.md
echo.
echo Note: The Conjecture Provider is still running.
echo Close its window or press Ctrl+C in that window to stop it.
echo.
pause
