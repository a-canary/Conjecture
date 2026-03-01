@echo off
REM Comprehensive coverage reporting script for Conjecture project (Windows)

echo 🔍 Conjecture Coverage Analysis
echo ==============================
echo.

REM Set PYTHONPATH to include project root
set PYTHONPATH=.

REM Create coverage directory if it doesn't exist
if not exist coverage_reports mkdir coverage_reports
if not exist htmlcov mkdir htmlcov

REM Clean up previous coverage data
echo 🧹 Cleaning up previous coverage data...
if exist .coverage del .coverage
if exist coverage.xml del coverage.xml
if exist coverage.json del coverage.json

REM Run coverage analysis
echo 📊 Running coverage analysis...
python -m pytest tests/ --cov=src --cov-config=.coveragerc --cov-report=term-missing --cov-report=html:htmlcov --cov-report=xml:coverage.xml --cov-report=json:coverage.json -v

REM Generate additional reports
echo 📈 Generating additional coverage reports...
coverage html -d htmlcov
coverage xml -o coverage.xml
coverage json -o coverage.json

REM Copy reports to coverage_reports directory with timestamp
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set TIMESTAMP=%datetime:~0,8%_%datetime:~8,6%
copy coverage.xml "coverage_reports\coverage_%TIMESTAMP%.xml" >nul
copy coverage.json "coverage_reports\coverage_%TIMESTAMP%.json" >nul

REM Extract key metrics using Python
echo 📋 Coverage Summary:
python -c "
import json
import sys
import os

try:
    with open('coverage.json', 'r') as f:
        data = json.load(f)
    
    totals = data['totals']
    print(f'  Lines: {totals[\"percent_covered\"]:.1f}% ({totals[\"covered_lines\"]}/{totals[\"num_statements\"]})')
    
    if totals.get('num_branches', 0) > 0:
        print(f'  Branches: {totals[\"covered_branches\"]}/{totals[\"num_branches\"]} ({totals[\"percent_covered_branches\"]:.1f}%)')
    else:
        print('  Branches: N/A')
    
    print(f'  Missing Lines: {totals[\"missing_lines\"]}')
    
    # Check if we're meeting our goals
    line_coverage = totals['percent_covered']
    if line_coverage >= 80:
        print('  ✅ Goal achieved: 80%%+ line coverage!')
    elif line_coverage >= 60:
        print('  🟡 Progress made: 60%%+ line coverage')
    elif line_coverage >= 40:
        print('  🟠 Getting started: 40%%+ line coverage')
    else:
        print('  🔴 Need improvement: <40%% line coverage')
        
except Exception as e:
    print(f'  ❌ Error reading coverage data: {e}')
    sys.exit(1)
"

echo.
echo 📁 Reports generated:
echo   - HTML report: htmlcov\index.html
echo   - XML report: coverage.xml
echo   - JSON report: coverage.json
echo   - Timestamped reports: coverage_reports\

echo.
echo 🌐 Open HTML report with: start htmlcov\index.html
echo ✨ Coverage analysis complete!