#!/bin/bash
# Setup script for coverage infrastructure

set -e

echo "🔧 Setting up Coverage Infrastructure"
echo "===================================="

# Make scripts executable
echo "📝 Making scripts executable..."
chmod +x scripts/run_coverage.sh
chmod +x scripts/compare_coverage.py
chmod +x scripts/coverage_baseline.py

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p coverage_reports
mkdir -p htmlcov

# Install coverage dependencies if not already installed
echo "📦 Checking dependencies..."
python -c "import pytest_cov" 2>/dev/null || pip install pytest-cov coverage
python -c "import coverage" 2>/dev/null || pip install coverage

echo "✅ Coverage infrastructure setup complete!"
echo ""
echo "🚀 Quick start:"
echo "  ./scripts/run_coverage.sh                    # Run coverage analysis"
echo "  python scripts/coverage_baseline.py --set-baseline  # Establish baseline"
echo "  python scripts/compare_coverage.py            # Compare coverage"
echo ""
echo "📖 Documentation: docs/COVERAGE_WORKFLOW.md"