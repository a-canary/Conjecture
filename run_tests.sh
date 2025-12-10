#!/bin/bash
echo "Running Conjecture Tests with Static Analysis Integration..."
echo

# Set PYTHONPATH to include project root
export PYTHONPATH=.

# Check for help flag
if [ "$1" == "--help" ]; then
    echo
    echo "Conjecture Test Runner with Static Analysis Integration"
    echo
    echo "Usage:"
    echo "  ./run_tests.sh [options]"
    echo
    echo "Options:"
    echo "  --help                    Show this help message"
    echo "  --collect-only            Show test collection without running"
    echo "  --static                  Run static analysis tests only"
    echo "  --static-all              Run all static analysis tools directly"
    echo "  --ruff                    Run ruff linting only"
    echo "  --mypy                    Run mypy type checking only"
    echo "  --vulture                 Run vulture dead code detection only"
    echo "  --bandit                  Run bandit security analysis only"
    echo "  --unit                    Run unit tests only"
    echo "  --integration             Run integration tests only"
    echo "  --performance             Run performance tests only"
    echo "  --coverage                Run tests with coverage report"
    echo "  --parallel                Run tests in parallel"
    echo "  --all                     Run all tests including static analysis"
    echo "  --quick                   Run quick tests (unit + critical only)"
    echo "  --full                    Run full test suite with all analysis"
    echo "  --phased                  Run 3-phase testing system (30s timeouts)"
    echo
    echo "Examples:"
    echo "  ./run_tests.sh --static    Run static analysis tests"
    echo "  ./run_tests.sh --ruff      Run ruff linting"
    echo "  ./run_tests.sh --coverage  Run tests with coverage"
    echo "  ./run_tests.sh --full      Run complete test suite"
    echo "  ./run_tests.sh --phased    Run 3-phase testing system"
    echo
    exit 0
fi

# Check for collect-only flag
if [ "$1" == "--collect-only" ]; then
    echo "Collecting tests..."
    python -m pytest --collect-only
    exit 0
fi

# Check for static analysis flags
if [ "$1" == "--static" ]; then
    echo "Running Static Analysis Tests..."
    python -m pytest tests/ -v -m static_analysis
    echo
    echo "Static Analysis tests completed."
    exit 0
fi

if [ "$1" == "--static-all" ]; then
    echo "Running All Static Analysis Tools..."
    echo
    echo "Running ruff..."
    ruff check . --format=concise
    echo
    echo "Running ruff format check..."
    ruff format --check .
    echo
    echo "Running mypy..."
    mypy src/
    echo
    echo "Running vulture..."
    vulture src/ tests/ --min-confidence 80
    echo
    echo "Running bandit..."
    bandit -r src/ -f text
    echo
    echo "All static analysis tools completed."
    exit 0
fi

if [ "$1" == "--ruff" ]; then
    echo "Running ruff linting..."
    ruff check . --format=concise
    echo
    echo "Running ruff format check..."
    ruff format --check .
    echo
    echo "ruff analysis completed."
    exit 0
fi

if [ "$1" == "--mypy" ]; then
    echo "Running mypy type checking..."
    mypy src/ --show-error-codes
    echo
    echo "mypy analysis completed."
    exit 0
fi

if [ "$1" == "--vulture" ]; then
    echo "Running vulture dead code detection..."
    vulture src/ tests/ --min-confidence 80
    echo
    echo "vulture analysis completed."
    exit 0
fi

if [ "$1" == "--bandit" ]; then
    echo "Running bandit security analysis..."
    bandit -r src/ -f text
    echo
    echo "bandit analysis completed."
    exit 0
fi

# Check for test category flags
if [ "$1" == "--unit" ]; then
    echo "Running Unit Tests only..."
    python -m pytest tests/ -v -m unit -m "not static_analysis"
    exit 0
fi

if [ "$1" == "--integration" ]; then
    echo "Running Integration Tests only..."
    python -m pytest tests/ -v -m integration
    exit 0
fi

if [ "$1" == "--performance" ]; then
    echo "Running Performance Tests only..."
    python -m pytest tests/ -v -m performance
    exit 0
fi

# Check for coverage flag
if [ "$1" == "--coverage" ]; then
    echo "Running Tests with Coverage Report..."
    python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
    echo
    echo "Coverage report generated in htmlcov/"
    exit 0
fi

# Check for parallel flag
if [ "$1" == "--parallel" ]; then
    echo "Running Tests in Parallel..."
    python -m pytest tests/ -v -n auto
    exit 0
fi

# Check for quick tests flag
if [ "$1" == "--quick" ]; then
    echo "Running Quick Tests (Unit + Critical only)..."
    python -m pytest tests/ -v -m "unit or critical" -m "not static_analysis" --maxfail=3
    exit 0
fi

# Check for full test suite flag
if [ "$1" == "--full" ]; then
    echo "Running Full Test Suite with Static Analysis..."
    echo
    echo "Step 1: Running Static Analysis..."
    python -m pytest tests/ -v -m static_analysis
    echo
    echo "Step 2: Running Regular Tests..."
    python -m pytest tests/ -v -m "not static_analysis" --cov=src --cov-report=html --cov-report=term-missing
    echo
    echo "Full test suite completed."
    exit 0
fi

# Check for all tests flag
if [ "$1" == "--all" ]; then
    echo "Running All Tests including Static Analysis..."
    python -m pytest tests/ -v
    echo
    echo "All tests completed."
    exit 0
fi

# Check for phased testing flag
if [ "$1" == "--phased" ]; then
    echo "Running 3-Phase Testing System..."
    ./run_phased_tests.sh
    exit_code=$?
    echo
    echo "Testing completed."
    exit $exit_code
fi

# Run regular tests by default
echo "Running Regular Tests..."
python -m pytest tests/ -v -m "not static_analysis"

echo
echo "Test run completed."
echo
echo "Usage:"
echo "  ./run_tests.sh --help       Show all available options"
echo "  ./run_tests.sh              - Run regular tests only"
echo "  ./run_tests.sh --static     - Run static analysis tests only"
echo "  ./run_tests.sh --all        - Run all tests including static analysis"
echo "  ./run_tests.sh --full       - Run full test suite with static analysis and coverage"
echo "  ./run_tests.sh --phased     - Run 3-phase testing system with dynamic timeouts"