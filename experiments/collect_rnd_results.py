#!/usr/bin/env python3
"""
Collect and summarize all R&D experiment results.
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime


def parse_results_file(filepath: str) -> dict:
    """Parse experiment output file for results."""
    if not os.path.exists(filepath):
        return {"status": "file_not_found"}

    content = Path(filepath).read_text()
    if not content.strip():
        return {"status": "empty"}

    result = {"status": "completed", "raw": content[-2000:]}  # Last 2000 chars

    # Try to extract accuracy
    acc_match = re.search(r'accuracy[:\s]+(\d+\.?\d*)%', content, re.I)
    if acc_match:
        result["accuracy"] = float(acc_match.group(1))

    # Try to extract learning delta
    learn_match = re.search(r'learning[:\s]+([+-]?\d+\.?\d*)\s*pp', content, re.I)
    if learn_match:
        result["learning_delta"] = float(learn_match.group(1))

    # Check for confirmation
    if "✓ CONFIRMED" in content:
        result["hypothesis_confirmed"] = True
    elif "✗ NOT CONFIRMED" in content:
        result["hypothesis_confirmed"] = False

    return result


def collect_all_results():
    """Collect all R&D experiment results."""

    experiments = {
        "window_size": "/tmp/window_results.txt",
        "semantic_filter": "/tmp/semantic_filter_results.txt",
        "confidence_gating": "/tmp/confidence_gating_results.txt",
        "position_primacy": "/tmp/position_primacy_results.txt",
        "optimized_accumulation": "/tmp/optimized_accum_results.txt",
        "model_comparison": "/tmp/model_comparison_results.txt"
    }

    results = {
        "timestamp": datetime.now().isoformat(),
        "experiments": {}
    }

    for name, filepath in experiments.items():
        results["experiments"][name] = parse_results_file(filepath)

    return results


def print_summary(results: dict):
    """Print formatted summary."""
    print("=" * 70)
    print("R&D EXPERIMENT RESULTS SUMMARY")
    print(f"Collected: {results['timestamp']}")
    print("=" * 70)

    for name, data in results["experiments"].items():
        print(f"\n{name.upper()}:")
        print("-" * 40)

        if data["status"] == "file_not_found":
            print("  Status: File not found")
        elif data["status"] == "empty":
            print("  Status: Running (no output yet)")
        else:
            if "accuracy" in data:
                print(f"  Accuracy: {data['accuracy']}%")
            if "learning_delta" in data:
                print(f"  Learning: {data['learning_delta']:+.1f}pp")
            if "hypothesis_confirmed" in data:
                status = "CONFIRMED ✓" if data["hypothesis_confirmed"] else "NOT CONFIRMED ✗"
                print(f"  Hypothesis: {status}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    results = collect_all_results()
    print_summary(results)

    # Save to file
    output_path = "/workspace/data/rnd_results_summary.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
