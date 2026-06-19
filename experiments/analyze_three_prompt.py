#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Analyze three-prompt benchmark results

Compares three-prompt architecture against direct baseline:
- Accuracy improvement
- Iteration patterns
- Confidence distributions
- Token efficiency
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime


def load_latest_result(pattern: str) -> Dict:
    """Load most recent result file matching pattern."""
    results_dir = Path("experiments/results")
    files = sorted(results_dir.glob(pattern), reverse=True)

    if not files:
        raise FileNotFoundError(f"No results found matching {pattern}")

    latest = files[0]
    print(f"Loading: {latest.name}")
    with open(latest) as f:
        return json.load(f)


def analyze_accuracy(data: Dict) -> None:
    """Analyze accuracy metrics."""
    print("\n" + "="*70)
    print("ACCURACY ANALYSIS")
    print("="*70)

    direct = data["direct"]
    three = data["three_prompt"]

    print(f"\nDirect Baseline:")
    print(f"  Correct: {direct['correct']}/{direct['total']}")
    print(f"  Accuracy: {direct['accuracy']}%")
    print(f"  Extraction failures: {direct['extraction_failures']}")

    print(f"\nThree-Prompt:")
    print(f"  Correct: {three['correct']}/{three['total']}")
    print(f"  Accuracy: {three['accuracy']}%")
    print(f"  Extraction failures: {three['extraction_failures']}")
    print(f"  Avg iterations: {three['avg_iterations']}")

    improvement = three["accuracy"] - direct["accuracy"]
    print(f"\nImprovement: {improvement:+.1f}pp")

    if improvement >= 5:
        status = "✅ SIGNIFICANT"
    elif improvement >= 2:
        status = "✓ PROMISING"
    elif improvement >= -2:
        status = "≈ NEUTRAL"
    else:
        status = "❌ REGRESSION"

    print(f"Status: {status}")


def analyze_efficiency(data: Dict) -> None:
    """Analyze token and time efficiency."""
    print("\n" + "="*70)
    print("EFFICIENCY ANALYSIS")
    print("="*70)

    direct = data["direct"]
    three = data["three_prompt"]

    direct_tokens_per = direct["total_tokens"] / direct["total"]
    three_tokens_per = three["total_tokens"] / three["total"]
    token_ratio = three_tokens_per / direct_tokens_per if direct_tokens_per > 0 else 0

    print(f"\nTokens per problem:")
    print(f"  Direct: {direct_tokens_per:,.0f}")
    print(f"  Three-Prompt: {three_tokens_per:,.0f}")
    print(f"  Ratio: {token_ratio:.1f}x")

    print(f"\nTime per problem:")
    print(f"  Direct: {direct['avg_time']:.2f}s")
    print(f"  Three-Prompt: {three['avg_time']:.2f}s")
    print(f"  Ratio: {three['avg_time']/direct['avg_time']:.1f}x")

    # Efficiency score: accuracy gain per token multiplier
    improvement = three["accuracy"] - direct["accuracy"]
    if token_ratio > 1:
        efficiency = improvement / (token_ratio - 1)
        print(f"\nEfficiency (accuracy gain per token multiplier):")
        print(f"  {efficiency:.1f}pp per 1x tokens")

    # Value assessment
    print(f"\nValue Assessment:")
    if improvement >= 5 and token_ratio <= 3:
        print("  ✅ HIGH VALUE: Significant improvement with acceptable cost")
    elif improvement >= 2 and token_ratio <= 4:
        print("  ✓ GOOD VALUE: Modest improvement with reasonable cost")
    elif improvement >= 0:
        print("  ≈ LOW VALUE: Marginal improvement for increased cost")
    else:
        print("  ❌ NEGATIVE VALUE: Regression despite increased cost")


def analyze_iterations(data: Dict) -> None:
    """Analyze iteration patterns."""
    print("\n" + "="*70)
    print("ITERATION ANALYSIS")
    print("="*70)

    three = data["three_prompt"]
    avg_iters = three["avg_iterations"]
    max_iters = data.get("max_iterations", 4)

    print(f"\nIterations:")
    print(f"  Average: {avg_iters:.2f}")
    print(f"  Maximum allowed: {max_iters}")
    print(f"  Utilization: {100*avg_iters/max_iters:.1f}%")

    if avg_iters < max_iters * 0.5:
        print(f"  Pattern: EARLY STOPPING (efficient)")
    elif avg_iters < max_iters * 0.8:
        print(f"  Pattern: MODERATE (balanced)")
    else:
        print(f"  Pattern: FULL EXPLORATION (may need tuning)")


def summarize_findings(data: Dict) -> None:
    """Summarize key findings."""
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    improvement = data["three_prompt"]["accuracy"] - data["direct"]["accuracy"]
    avg_iters = data["three_prompt"]["avg_iterations"]
    max_iters = data.get("max_iterations", 4)

    print(f"\nThree-Prompt vs Direct Baseline:")
    print(f"  Accuracy: {improvement:+.1f}pp")
    print(f"  Avg Iterations: {avg_iters:.1f}/{max_iters}")
    print(f"  Confidence Threshold: {data.get('confidence_threshold', 0.7)}")

    # Architecture assessment
    print(f"\nArchitecture Assessment:")

    if improvement >= 5:
        print("  ✅ VALIDATED: Significant improvement on math reasoning")
        print("  → Recommend: Scale to other reasoning benchmarks (BBH)")
    elif improvement >= 2:
        print("  ✓ PROMISING: Modest improvement, worth further testing")
        print("  → Recommend: Test on harder tasks (BBH)")
    elif improvement >= 0:
        print("  ≈ NEUTRAL: No clear benefit")
        print("  → Recommend: Test on harder tasks or adjust parameters")
    else:
        print("  ❌ PROBLEMATIC: Regression on math reasoning")
        print("  → Investigate: Why is performance worse?")

    # Self-regulation assessment
    if avg_iters < max_iters * 0.8:
        print(f"\n  ✅ SELF-REGULATION WORKING: Stopping before max iterations")
    else:
        print(f"\n  ⚠️ SELF-REGULATION WEAK: Hitting max iterations frequently")

    # Next steps
    print(f"\nNext Steps:")
    if improvement >= 2:
        print("  1. Run on BBH (hard reasoning benchmark)")
        print("  2. Run on MMLU (recall benchmark - expect neutral)")
        print("  3. Tune confidence threshold if needed")
        print("  4. Multi-model validation")
    elif improvement >= 0:
        print("  1. Tune confidence threshold (try 0.6, 0.8)")
        print("  2. Test on harder benchmarks (BBH)")
        print("  3. Analyze failure cases")
    else:
        print("  1. Debug: Why regression on math?")
        print("  2. Check prompt quality")
        print("  3. Compare claim quality vs direct")


def main():
    """Main analysis."""
    print("\n" + "="*70)
    print("THREE-PROMPT ARCHITECTURE ANALYSIS")
    print("="*70)

    # Load latest result
    data = load_latest_result("gsm8k_three_prompt_*.json")

    print(f"\nBenchmark: {data['benchmark']}")
    print(f"Model: {data['model']}")
    print(f"Problems: {data['n_problems']}")
    print(f"Timestamp: {data['timestamp']}")

    # Run analyses
    analyze_accuracy(data)
    analyze_efficiency(data)
    analyze_iterations(data)
    summarize_findings(data)

    print("\n" + "="*70)
    print(f"Analysis complete: {datetime.now().isoformat()}")
    print("="*70)


if __name__ == "__main__":
    main()
