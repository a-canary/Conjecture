@echo off
echo Running Conjecture Tests with PYTHONPATH fix...
echo.

REM Set PYTHONPATH to include project root
set PYTHONPATH=.

REM Run pytest with the fixed path
python -m pytest tests/ -v

echo.
echo Test run completed.