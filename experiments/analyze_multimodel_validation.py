#!/usr/bin/env python3
"""
Multi-Model Three-Prompt Architecture Validation Analysis

Analyzes three-prompt architecture performance across multiple model sizes:
- Small (8B): Llama-3.1-8B
- Medium (32B): DeepSeek-R1-Qwen-32B
- Large (70B): Llama-3.1-70B
- Extra Large (670B): DeepSeek-V3 (baseline)

Calculates p-values for statistical significance testing.
"""

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    model: str
    model_size: str
    benchmark: str
    method: str
    accuracy: float
    n_problems: int


@dataclass
class Comparison:
    """Comparison between two methods."""
    model: str
    model_size: str
    benchmark: str
    direct_acc: float
    three_prompt_acc: float
    delta: float
    p_value: float
    n: int
    significant: bool

    def __str__(self):
        sig_marker = "✅ SIG" if self.significant else "≈ EQUIV"
        return (f"{self.model_size:12} | {self.benchmark:6} | "
                f"Direct: {self.direct_acc:5.1%} | "
                f"Three-Prompt: {self.three_prompt_acc:5.1%} | "
                f"Δ {self.delta:+6.1%} | p={self.p_value:.3f} | {sig_marker}")


def calculate_p_value(acc1: float, acc2: float, n: int) -> float:
    """Calculate p-value for difference between two accuracies.

    Uses two-proportion z-test with standard error calculation.
    """
    if n == 0:
        return 1.0

    diff = acc2 - acc1

    # Standard error for each proportion
    se1 = math.sqrt(acc1 * (1 - acc1) / n) if 0 < acc1 < 1 else 0
    se2 = math.sqrt(acc2 * (1 - acc2) / n) if 0 < acc2 < 1 else 0

    # Combined standard error
    se_diff = math.sqrt(se1**2 + se2**2)

    if se_diff == 0:
        return 1.0

    # Z-score and p-value
    z_score = diff / se_diff
    p_value = 2 * stats.norm.cdf(-abs(z_score))

    return p_value


def parse_result_file(filepath: Path) -> List[BenchmarkResult]:
    """Parse a three-prompt benchmark result file."""
    results = []

    try:
        with open(filepath) as f:
            data = json.load(f)

        # Skip non-dict results (old format)
        if not isinstance(data, dict):
            return results

        # Extract model info
        model = data.get("model", "unknown")
        n_problems = data.get("n_problems", 0)
        benchmark = data.get("benchmark", "unknown")

        # Map model to size category
        model_lower = model.lower()
        if "8b" in model_lower or "7b" in model_lower:
            model_size = "Small (8B)"
        elif "32b" in model_lower:
            model_size = "Medium (32B)"
        elif "70b" in model_lower:
            model_size = "Large (70B)"
        elif "deepseek" in model_lower and "v3" in model_lower:
            model_size = "XL (670B)"
        else:
            model_size = "Unknown"

        # Extract direct results
        if "direct" in data:
            direct_acc = data["direct"].get("accuracy", 0.0) / 100.0  # Convert percentage to decimal
            results.append(BenchmarkResult(
                model=model,
                model_size=model_size,
                benchmark=benchmark,
                method="direct",
                accuracy=direct_acc,
                n_problems=n_problems
            ))

        # Extract three-prompt results
        if "three_prompt" in data:
            tp_acc = data["three_prompt"].get("accuracy", 0.0) / 100.0  # Convert percentage to decimal
            results.append(BenchmarkResult(
                model=model,
                model_size=model_size,
                benchmark=benchmark,
                method="three_prompt",
                accuracy=tp_acc,
                n_problems=n_problems
            ))

    except Exception as e:
        print(f"Error parsing {filepath}: {e}", file=sys.stderr)

    return results


def load_all_results(results_dir: Path) -> List[BenchmarkResult]:
    """Load all three-prompt benchmark results."""
    all_results = []

    # Find all three-prompt result files
    pattern = "*three_prompt*.json"
    for filepath in sorted(results_dir.glob(pattern)):
        results = parse_result_file(filepath)
        all_results.extend(results)

    return all_results


def group_by_model_benchmark(results: List[BenchmarkResult]) -> Dict[Tuple[str, str], List[BenchmarkResult]]:
    """Group results by (model_size, benchmark)."""
    groups = {}

    for result in results:
        key = (result.model_size, result.benchmark)
        if key not in groups:
            groups[key] = []
        groups[key].append(result)

    return groups


def create_comparisons(groups: Dict[Tuple[str, str], List[BenchmarkResult]]) -> List[Comparison]:
    """Create comparisons between direct and three-prompt methods."""
    comparisons = []

    for (model_size, benchmark), results in groups.items():
        # Find direct and three-prompt results
        direct = next((r for r in results if r.method == "direct"), None)
        three_prompt = next((r for r in results if r.method == "three_prompt"), None)

        if not direct or not three_prompt:
            continue

        # Calculate comparison
        delta = three_prompt.accuracy - direct.accuracy
        p_value = calculate_p_value(direct.accuracy, three_prompt.accuracy, direct.n_problems)
        significant = p_value < 0.05

        comparisons.append(Comparison(
            model=direct.model,
            model_size=model_size,
            benchmark=benchmark,
            direct_acc=direct.accuracy,
            three_prompt_acc=three_prompt.accuracy,
            delta=delta,
            p_value=p_value,
            n=direct.n_problems,
            significant=significant
        ))

    return comparisons


def print_summary(comparisons: List[Comparison]):
    """Print summary table of all comparisons."""
    print("\n" + "="*90)
    print("MULTI-MODEL THREE-PROMPT ARCHITECTURE VALIDATION")
    print("="*90)
    print()

    # Group by model size
    size_order = ["Small (8B)", "Medium (32B)", "Large (70B)", "XL (670B)"]

    for size in size_order:
        size_comps = [c for c in comparisons if c.model_size == size]
        if not size_comps:
            continue

        print(f"\n{size}:")
        print("-" * 90)

        for comp in sorted(size_comps, key=lambda x: x.benchmark):
            print(comp)

    print("\n" + "="*90)
    print("\nKEY FINDINGS:")
    print("-" * 90)

    # Analyze patterns
    bbh_comps = [c for c in comparisons if "bbh" in c.benchmark.lower()]
    gsm8k_comps = [c for c in comparisons if "gsm8k" in c.benchmark.lower()]

    if bbh_comps:
        print(f"\nBBH (Hard Reasoning) - {len(bbh_comps)} models tested:")
        sig_improvements = [c for c in bbh_comps if c.significant and c.delta > 0]
        print(f"  - Significant improvements: {len(sig_improvements)}/{len(bbh_comps)}")
        if sig_improvements:
            for c in sig_improvements:
                print(f"    • {c.model_size}: {c.delta:+.1%} (p={c.p_value:.3f})")

    if gsm8k_comps:
        print(f"\nGSM8K (Saturated Math) - {len(gsm8k_comps)} models tested:")
        sig_changes = [c for c in gsm8k_comps if c.significant]
        print(f"  - Significant changes: {len(sig_changes)}/{len(gsm8k_comps)}")
        if sig_changes:
            for c in sig_changes:
                direction = "improvement" if c.delta > 0 else "regression"
                print(f"    • {c.model_size}: {c.delta:+.1%} (p={c.p_value:.3f}) - {direction}")
        else:
            print(f"  - All models statistically equivalent (no significant regressions)")

    print("\n" + "="*90)


def main():
    """Main analysis."""
    results_dir = Path("experiments/results")

    print(f"Loading results from {results_dir}...")
    results = load_all_results(results_dir)

    if not results:
        print("ERROR: No results found!", file=sys.stderr)
        return 1

    print(f"Loaded {len(results)} result entries")

    # Group and compare
    groups = group_by_model_benchmark(results)
    print(f"Found {len(groups)} model-benchmark combinations")

    comparisons = create_comparisons(groups)
    print(f"Created {len(comparisons)} comparisons")

    # Print summary
    print_summary(comparisons)

    return 0


if __name__ == "__main__":
    sys.exit(main())
