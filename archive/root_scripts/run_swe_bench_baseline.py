#!/usr/bin/env python
"""
Run SWE-Bench Bash-Only Evaluator with baseline evaluation.
Tests the improvements: fuzzy parsing, error feedback, and failure analysis.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from benchmarks.benchmarking.swe_bench_bash_only_evaluator import main


async def run_baseline():
    """Run baseline evaluation with 10 tasks and verbose logging"""
    print("\n" + "=" * 70)
    print("SC-FEAT-001: SWE-Bench Bash-Only Improvements - BASELINE EVALUATION")
    print("=" * 70)
    print("\nImprovements Implemented:")
    print("  ✓ Task 1: Failure Analysis with verbose logging")
    print("  ✓ Task 2: Fuzzy Command Extraction (case-insensitive, heredoc support)")
    print("  ✓ Task 3: Error Feedback Loop (PREVIOUS_ERROR section in prompts)")
    print("\n" + "=" * 70)

    # Run with verbose mode and 10 tasks
    await main(verbose=True, batch_size=10)


if __name__ == "__main__":
    asyncio.run(run_baseline())
