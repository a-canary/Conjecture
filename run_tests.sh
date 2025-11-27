#!/bin/bash
echo "Running Conjecture Tests with PYTHONPATH fix..."
echo

# Set PYTHONPATH to include project root
export PYTHONPATH=.

# Run pytest with the fixed path
python -m pytest tests/ -v

echo
echo "Test run completed."