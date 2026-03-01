#!/bin/bash
# Comprehensive coverage reporting script for Conjecture project

set -e  # Exit on any error

echo "🔍 Conjecture Coverage Analysis"
echo "=============================="
echo

# Set PYTHONPATH to include project root
export PYTHONPATH=.

# Create coverage directory if it doesn't exist
mkdir -p coverage_reports
mkdir -p htmlcov

# Clean up previous coverage data
echo "🧹 Cleaning up previous coverage data..."
coverage erase
rm -f coverage.xml coverage.json .coverage

# Run coverage analysis
echo "📊 Running coverage analysis..."
python -m pytest tests/ --cov=src --cov-config=.coveragerc --cov-report=term-missing --cov-report=html:htmlcov --cov-report=xml:coverage.xml --cov-report=json:coverage.json -v

# Generate additional reports
echo "📈 Generating additional coverage reports..."
coverage html -d htmlcov
coverage xml -o coverage.xml
coverage json -o coverage.json

# Copy reports to coverage_reports directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
cp coverage.xml "coverage_reports/coverage_${TIMESTAMP}.xml"
cp coverage.json "coverage_reports/coverage_${TIMESTAMP}.json"

# Extract key metrics
echo "📋 Coverage Summary:"
python -c "
import json
import sys

try:
    with open('coverage.json', 'r') as f:
        data = json.load(f)
    
    totals = data['totals']
    print(f'  Lines: {totals[\"percent_covered\"]:.1f}% ({totals[\"covered_lines\"]}/{totals[\"num_statements\"]})')
    print(f'  Branches: {totals[\"covered_branches\"}/{totals[\"num_branches\"]} ({totals[\"percent_covered_branches\"]:.1f}%)' if totals.get('num_branches', 0) > 0 else '  Branches: N/A')
    print(f'  Missing Lines: {totals[\"missing_lines\"]}')
    
    # Check if we're meeting our goals
    line_coverage = totals['percent_covered']
    if line_coverage >= 80:
        print('  ✅ Goal achieved: 80%+ line coverage!')
    elif line_coverage >= 60:
        print('  🟡 Progress made: 60%+ line coverage')
    elif line_coverage >= 40:
        print('  🟠 Getting started: 40%+ line coverage')
    else:
        print('  🔴 Need improvement: <40% line coverage')
        
except Exception as e:
    print(f'  ❌ Error reading coverage data: {e}')
    sys.exit(1)
"

echo
echo "📁 Reports generated:"
echo "  - HTML report: htmlcov/index.html"
echo "  - XML report: coverage.xml"
echo "  - JSON report: coverage.json"
echo "  - Timestamped reports: coverage_reports/"

echo
echo "🌐 Open HTML report with: open htmlcov/index.html (macOS) or start htmlcov/index.html (Windows)"
echo "✨ Coverage analysis complete!"