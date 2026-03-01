#!/usr/bin/env python
import os
import sys
import subprocess

# Set UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Run the evaluator with option 1
result = subprocess.run(
    [sys.executable, 'benchmarks/benchmarking/run_bash_only_evaluator.py'],
    input='1\n',
    text=True,
    encoding='utf-8',
    errors='replace'
)
sys.exit(result.returncode)
