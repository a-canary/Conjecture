@echo off
REM Conjecture CLI - Easy Execution Script
REM This script allows you to run conjecture from any directory

setlocal
set PYTHONPATH=d:\projects\Conjecture\src
cd /d "d:\projects\Conjecture"
python conjecture %*
endlocal