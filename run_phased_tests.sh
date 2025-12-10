#!/bin/bash
echo "3-Phase Testing System with Dynamic Timeout Adjustment"
echo

# Set PYTHONPATH to include project root
export PYTHONPATH=.

# Check for help flag
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo
    echo "3-Phase Testing System"
    echo
    echo "Usage:"
    echo "  ./run_phased_tests.sh [options]"
    echo
    echo "Options:"
    echo "  --help                    Show this help message"
    echo "  --list-phases             List all phases and their configurations"
    echo "  --phase PHASE_NAME        Run specific phase only"
    echo "                             Valid phases: unit_e2e, static_analysis, benchmarks"
    echo "  --config CONFIG_FILE      Use custom configuration file"
    echo
    echo "Phase Descriptions:"
    echo "  unit_e2e        - Unit and End-to-End Tests (30s timeout)"
    echo "  static_analysis - Static Code Analysis (30s timeout)"
    echo "  benchmarks     - Performance and Benchmarks (30s timeout)"
    echo
    echo "Dynamic Timeout Adjustment:"
    echo "  - If a phase passes 100% of tests, timeout increases by 20% for future runs"
    echo "  - If a phase fails, current timeout is maintained"
    echo "  - Configuration is saved to .phased_testing_config.json"
    echo
    echo "Examples:"
    echo "  ./run_phased_tests.sh                    - Run all phases"
    echo "  ./run_phased_tests.sh --phase unit_e2e   - Run only unit/e2e tests"
    echo "  ./run_phased_tests.sh --list-phases      - Show phase configurations"
    echo
    exit 0
fi

# Check for list phases flag
if [ "$1" == "--list-phases" ]; then
    python tests/phased_testing_system.py --list-phases
    exit $?
fi

# Check for specific phase flag
if [ "$1" == "--phase" ]; then
    if [ -z "$2" ]; then
        echo "Error: Phase name required after --phase"
        echo "Usage: ./run_phased_tests.sh --phase PHASE_NAME"
        echo "Valid phases: unit_e2e, static_analysis, benchmarks"
        exit 1
    fi
    echo "Running specific phase: $2"
    python tests/phased_testing_system.py --phase "$2"
    exit $?
fi

# Check for config flag
if [ "$1" == "--config" ]; then
    if [ -z "$2" ]; then
        echo "Error: Config file path required after --config"
        echo "Usage: ./run_phased_tests.sh --config CONFIG_FILE"
        exit 1
    fi
    echo "Running with custom config: $2"
    python tests/phased_testing_system.py --config "$2"
    exit $?
fi

# Run all phases by default
echo "Running all 3 phases sequentially..."
python tests/phased_testing_system.py
exit_code=$?

echo
echo "Testing completed."
exit $exit_code