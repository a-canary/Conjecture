#!/usr/bin/env python3
"""
Analyze Benchmark Results and Update CSV

Reads all benchmark result JSON files, updates CSV tracker,
generates analysis for model-size optimization patterns.
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict


def load_all_results() -> List[Dict[str, Any]]:
    """Load all benchmark JSON result files."""
    results_dir = Path("experiments/results")
    all_results = []

    for json_file in results_dir.glob("*.json"):
        try:
            data = json.loads(json_file.read_text())
            data['source_file'] = json_file.name
            all_results.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return all_results


def update_csv(results: List[Dict[str, Any]]):
    """Update CSV with new benchmark results."""
    csv_file = Path("experiments/results/benchmark_results.csv")

    # Read existing entries
    existing = set()
    if csv_file.exists():
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['benchmark'], row['model'], row['method'], row['timestamp'])
                existing.add(key)

    # Add new entries
    new_entries = []
    for result in results:
        benchmark = result.get('benchmark', 'unknown')
        model = result.get('model', 'unknown')
        timestamp = result.get('timestamp', '')

        # Handle different result formats
        if 'direct' in result and 'decomposition' in result:
            # Standard format (direct vs decomposition)
            for method_name, method_data in [('direct', result['direct']), ('decomposition', result['decomposition'])]:
                key = (benchmark, model, method_name, timestamp)
                if key not in existing:
                    new_entries.append({
                        'benchmark': benchmark,
                        'dataset': benchmark,
                        'model': model,
                        'method': method_name,
                        'n_problems': method_data.get('total', 0),
                        'correct': method_data.get('correct', 0),
                        'accuracy': method_data.get('accuracy', 0),
                        'avg_time_sec': method_data.get('avg_time', 0),
                        'total_tokens': method_data.get('total_tokens', 0),
                        'extraction_failures': method_data.get('extraction_failures', 0),
                        'timestamp': timestamp,
                        'notes': f"{benchmark} test"
                    })

        elif 'methods' in result:
            # MMLU alternatives format
            for method_data in result['methods']:
                method_name = method_data.get('method', 'unknown')
                key = (benchmark, model, method_name, timestamp)
                if key not in existing:
                    new_entries.append({
                        'benchmark': benchmark,
                        'dataset': result.get('benchmark', benchmark),
                        'model': model,
                        'method': method_name,
                        'n_problems': method_data.get('total', 0),
                        'correct': method_data.get('correct', 0),
                        'accuracy': method_data.get('accuracy', 0),
                        'avg_time_sec': method_data.get('avg_time', 0),
                        'total_tokens': method_data.get('total_tokens', 0),
                        'extraction_failures': method_data.get('extraction_failures', 0),
                        'timestamp': timestamp,
                        'notes': f"Alternative method: {method_name}"
                    })

    # Append new entries to CSV
    if new_entries:
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'benchmark', 'dataset', 'model', 'method', 'n_problems', 'correct',
                'accuracy', 'avg_time_sec', 'total_tokens', 'extraction_failures',
                'timestamp', 'notes'
            ])
            for entry in new_entries:
                writer.writerow(entry)

        print(f"✅ Added {len(new_entries)} new entries to CSV")
    else:
        print("ℹ️  No new entries to add")


def analyze_patterns(results: List[Dict[str, Any]]):
    """Analyze patterns across benchmarks."""
    print("\n" + "="*70)
    print("BENCHMARK ANALYSIS")
    print("="*70)

    # Group by benchmark type
    by_benchmark = defaultdict(list)
    for result in results:
        benchmark = result.get('benchmark', 'unknown')

        if 'direct' in result and 'decomposition' in result:
            direct_acc = result['direct'].get('accuracy', 0)
            decomp_acc = result['decomposition'].get('accuracy', 0)
            improvement = decomp_acc - direct_acc

            by_benchmark[benchmark].append({
                'direct': direct_acc,
                'decomposition': decomp_acc,
                'improvement': improvement,
                'model': result.get('model', 'unknown')
            })

    # Print summary
    print(f"\n{'Benchmark':<20} {'Direct':>10} {'Decomp':>10} {'Δ':>8} {'Conclusion'}")
    print("-"*70)

    for benchmark, data in sorted(by_benchmark.items()):
        if data:
            avg_direct = sum(d['direct'] for d in data) / len(data)
            avg_decomp = sum(d['decomposition'] for d in data) / len(data)
            avg_improvement = sum(d['improvement'] for d in data) / len(data)

            if avg_improvement > 5:
                conclusion = "✅ Helps"
            elif avg_improvement > 0:
                conclusion = "⚠️  Modest"
            elif avg_improvement > -5:
                conclusion = "⚠️  Neutral"
            else:
                conclusion = "❌ Hurts"

            print(f"{benchmark:<20} {avg_direct:>9.1f}% {avg_decomp:>9.1f}% {avg_improvement:>+7.1f}pp {conclusion}")

    # Task type categorization
    print("\n" + "="*70)
    print("TASK TYPE ANALYSIS")
    print("="*70)

    reasoning_tasks = ['GSM8K', 'ARC-Challenge', 'BBH', 'Synthetic']
    recall_tasks = ['MMLU', 'TruthfulQA']
    commonsense_tasks = ['HellaSwag']

    def avg_improvement(task_list):
        improvements = []
        for benchmark in task_list:
            if benchmark in by_benchmark:
                improvements.extend([d['improvement'] for d in by_benchmark[benchmark]])
        return sum(improvements) / len(improvements) if improvements else 0

    reasoning_avg = avg_improvement(reasoning_tasks)
    recall_avg = avg_improvement(recall_tasks)
    commonsense_avg = avg_improvement(commonsense_tasks)

    print(f"\nReasoning tasks:    {reasoning_avg:+.1f}pp average")
    print(f"Recall tasks:       {recall_avg:+.1f}pp average")
    print(f"Commonsense tasks:  {commonsense_avg:+.1f}pp average")

    # Recommendation
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    if reasoning_avg > 5 and recall_avg < -5:
        print("\n✅ Validated: Task-type routing required")
        print("   - Use decomposition for reasoning tasks")
        print("   - Use direct/alternative for recall tasks")
    elif reasoning_avg > 0:
        print("\n⚠️  Partially validated: Decomposition helps reasoning")
        print(f"   - But only modestly ({reasoning_avg:+.1f}pp)")
    else:
        print("\n❌ Not validated: No clear benefit")


def main():
    """Main analysis function."""
    print("Loading benchmark results...")
    results = load_all_results()
    print(f"Loaded {len(results)} result files")

    print("\nUpdating CSV...")
    update_csv(results)

    print("\nAnalyzing patterns...")
    analyze_patterns(results)

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
