@echo off
echo 3-Phase Testing System with Dynamic Timeout Adjustment
echo.

REM Set PYTHONPATH to include project root
set PYTHONPATH=.

REM Suppress Pydantic warnings from DeepEval and other libraries
set PYTHONWARNINGS=ignore::UserWarning
set PYTHONWARNINGS=%PYTHONWARNINGS%,ignore::DeprecationWarning
set PYTHONWARNINGS=%PYTHONWARNINGS%,ignore::PendingDeprecationWarning

REM Check for help flag
if "%1"=="--help" goto :help
if "%1"=="-h" goto :help

REM Check for list phases flag
if "%1"=="--list-phases" goto :list_phases

REM Check for specific phase flag
if "%1"=="--phase" goto :run_phase

REM Check for config flag
if "%1"=="--config" goto :run_with_config

REM Run all phases by default
echo Running all 3 phases sequentially...
python tests/phased_testing_system.py
goto :end

:help
echo.
echo 3-Phase Testing System
echo.
echo Usage:
echo   run_phased_tests.bat [options]
echo.
echo Options:
echo   --help                    Show this help message
echo   --list-phases             List all phases and their configurations
echo   --phase PHASE_NAME        Run specific phase only
echo                             Valid phases: unit_e2e, static_analysis, benchmarks
echo   --config CONFIG_FILE      Use custom configuration file
echo.
echo Phase Descriptions:
echo   unit_e2e        - Unit and End-to-End Tests (30s timeout)
echo   static_analysis - Static Code Analysis (30s timeout)
echo   benchmarks     - Performance and Benchmarks (30s timeout)
echo.
echo Dynamic Timeout Adjustment:
echo   - If a phase passes 100%% of tests, timeout increases by 20%% for future runs
echo   - If a phase fails, current timeout is maintained
echo   - Configuration is saved to .phased_testing_config.json
echo.
echo Examples:
echo   run_phased_tests.bat                    - Run all phases
echo   run_phased_tests.bat --phase unit_e2e   - Run only unit/e2e tests
echo   run_phased_tests.bat --list-phases      - Show phase configurations
echo.
goto :end

:list_phases
python tests/phased_testing_system.py --list-phases
goto :end

:run_phase
if "%2"=="" (
    echo Error: Phase name required after --phase
    echo Usage: run_phased_tests.bat --phase PHASE_NAME
    echo Valid phases: unit_e2e, static_analysis, benchmarks
    goto :end
)
echo Running specific phase: %2
python tests/phased_testing_system.py --phase %2
goto :end

:run_with_config
if "%2"=="" (
    echo Error: Config file path required after --config
    echo Usage: run_phased_tests.bat --config CONFIG_FILE
    goto :end
)
echo Running with custom config: %2
python tests/phased_testing_system.py --config %2
goto :end

:end
echo.
echo Testing completed.