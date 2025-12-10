@echo off
echo Running Conjecture Tests with Static Analysis Integration...
echo.

REM Set PYTHONPATH to include project root
set PYTHONPATH=.

REM Suppress Pydantic warnings from DeepEval and other libraries
set PYTHONWARNINGS=ignore::UserWarning
set PYTHONWARNINGS=%PYTHONWARNINGS%,ignore::DeprecationWarning
set PYTHONWARNINGS=%PYTHONWARNINGS%,ignore::PendingDeprecationWarning

REM Check for help flag
if "%1"=="--help" goto :help
    echo.
    echo Conjecture Test Runner with Static Analysis Integration
    echo.
    echo Usage:
    echo   run_tests.bat [options]
    echo.
    echo Options:
    echo   --help                    Show this help message
    echo   --collect-only            Show test collection without running
    echo   --static                  Run static analysis tests only
    echo   --static-all              Run all static analysis tools directly
    echo   --ruff                    Run ruff linting only
    echo   --mypy                    Run mypy type checking only
    echo   --vulture                 Run vulture dead code detection only
    echo   --bandit                  Run bandit security analysis only
    echo   --unit                    Run unit tests only
    echo   --integration             Run integration tests only
    echo   --performance             Run performance tests only
    echo   --coverage                Run tests with coverage report
    echo   --parallel                Run tests in parallel
    echo   --all                     Run all tests including static analysis
    echo   --quick                   Run quick tests (unit + critical only)
    echo   --full                    Run full test suite with all analysis
    echo   --phased                  Run 3-phase testing system (30s timeouts)
    echo.
    echo Examples:
    echo   run_tests.bat --static    Run static analysis tests
    echo   run_tests.bat --ruff      Run ruff linting
    echo   run_tests.bat --coverage  Run tests with coverage
    echo   run_tests.bat --full      Run complete test suite
    echo   run_tests.bat --phased    Run 3-phase testing system
    echo.
    goto :end
)

REM Check for collect-only flag
if "%1"=="--collect-only" goto :collect_only
    echo Collecting tests...
    python -m pytest --collect-only
    goto :end
)

REM Check for static analysis flags
if "%1"=="--static" goto :static
    echo Running Static Analysis Tests...
    python -m pytest tests/ -v -m static_analysis
    echo.
    echo Static Analysis tests completed.
    goto :end
)

if "%1"=="--static-all" (
    echo Running All Static Analysis Tools...
    echo.
    echo Running ruff...
    ruff check . --format=concise
    echo.
    echo Running mypy...
    mypy src/
    echo.
    echo Running vulture...
    vulture src/ tests/ --min-confidence 80
    echo.
    echo Running bandit...
    bandit -r src/ -f text
    echo.
    echo All static analysis tools completed.
    goto :end
)

if "%1"=="--ruff" (
    echo Running ruff linting...
    ruff check . --format=concise
    echo.
    echo Running ruff format check...
    ruff format --check .
    echo.
    echo ruff analysis completed.
    goto :end
)

if "%1"=="--mypy" (
    echo Running mypy type checking...
    mypy src/ --show-error-codes
    echo.
    echo mypy analysis completed.
    goto :end
)

if "%1"=="--vulture" (
    echo Running vulture dead code detection...
    vulture src/ tests/ --min-confidence 80
    echo.
    echo vulture analysis completed.
    goto :end
)

if "%1"=="--bandit" (
    echo Running bandit security analysis...
    bandit -r src/ -f text
    echo.
    echo bandit analysis completed.
    goto :end
)

REM Check for test category flags
if "%1"=="--unit" (
    echo Running Unit Tests only...
    python -m pytest tests/ -v -m unit -m "not static_analysis"
    goto :end
)

if "%1"=="--integration" (
    echo Running Integration Tests only...
    python -m pytest tests/ -v -m integration
    goto :end
)

if "%1"=="--performance" (
    echo Running Performance Tests only...
    python -m pytest tests/ -v -m performance
    goto :end
)

REM Check for coverage flag
if "%1"=="--coverage" (
    echo Running Tests with Coverage Report...
    python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
    echo.
    echo Coverage report generated in htmlcov/
    goto :end
)

REM Check for parallel flag
if "%1"=="--parallel" (
    echo Running Tests in Parallel...
    python -m pytest tests/ -v -n auto
    goto :end
)

REM Check for quick tests flag
if "%1"=="--quick" (
    echo Running Quick Tests (Unit + Critical only)...
    python -m pytest tests/ -v -m "unit or critical" -m "not static_analysis" --maxfail=3
    goto :end
)

REM Check for full test suite flag
if "%1"=="--full" (
    echo Running Full Test Suite with Static Analysis...
    echo.
    echo Step 1: Running Static Analysis...
    python -m pytest tests/ -v -m static_analysis
    echo.
    echo Step 2: Running Regular Tests...
    python -m pytest tests/ -v -m "not static_analysis" --cov=src --cov-report=html --cov-report=term-missing
    echo.
    echo Full test suite completed.
    goto :end
)

REM Check for all tests flag
if "%1"=="--all" (
    echo Running All Tests including Static Analysis...
    python -m pytest tests/ -v
    echo.
    echo All tests completed.
    goto :end
)

REM Check for phased testing flag
if "%1"=="--phased" (
    echo Running 3-Phase Testing System...
    call run_phased_tests.bat
    goto :end
)

REM Run regular tests by default
echo Running Regular Tests...
python -m pytest tests/ -v -m "not static_analysis"
goto :end

:help

echo.
echo Test run completed.
echo.
echo Usage:
echo   run_tests.bat --help       Show all available options
echo   run_tests.bat              - Run regular tests only
echo   run_tests.bat --static     - Run static analysis tests only
echo   run_tests.bat --all        - Run all tests including static analysis
echo   run_tests.bat --full       - Run full test suite with static analysis and coverage
echo   run_tests.bat --phased     - Run 3-phase testing system with dynamic timeouts

:end