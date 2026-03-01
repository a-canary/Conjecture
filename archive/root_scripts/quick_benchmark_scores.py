#!/usr/bin/env python3
"""
Quick Benchmark Score Reporter
Extracts and displays key metrics from cycle results without complex processing
"""

import json
import os
from pathlib import Path


def main():
    cycle_dir = Path("benchmarks/benchmarking/cycle_results")

    print("LATEST BENCHMARK SCORES")
    print("=" * 60)

    cycle_files = sorted(cycle_dir.glob("cycle_*_results.json"))

    scores = []

    for cycle_file in cycle_files:
        try:
            with open(cycle_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            cycle_num = data.get("cycle", "Unknown")
            title = data.get("title", "Unknown")[:50]
            overall_score = data.get("overall_score", 0)
            success = data.get("success", False)

            scores.append(
                {
                    "cycle": cycle_num,
                    "title": title,
                    "score": overall_score,
                    "success": success,
                }
            )

        except Exception as e:
            print(f"Error reading {cycle_file.name}: {str(e)[:30]}...")

    # Sort by cycle number if possible
    try:
        scores.sort(
            key=lambda x: int(x["cycle"])
            if isinstance(x["cycle"], (int, str)) and str(x["cycle"]).isdigit()
            else 999
        )
    except:
        pass

    print(f"\nRECENT CYCLE RESULTS")
    print("-" * 40)
    print(f"{'Cycle':<8} {'Score':<8} {'Status':<8} {'Title'}")
    print("-" * 40)

    for score_data in scores[-10:]:  # Last 10 cycles
        cycle_str = str(score_data["cycle"])
        status = "PASS" if score_data["success"] else "FAIL"
        score_str = (
            f"{score_data['score']:.1f}%"
            if isinstance(score_data["score"], (int, float))
            else f"{score_data['score']}"
        )

        print(f"{cycle_str:<8} {score_str:<8} {status:<8} {score_data['title']}")

    # Calculate success rate and average
    successful = len([s for s in scores if s["success"]])
    total = len(scores)
    success_rate = (successful / total * 100) if total > 0 else 0

    numeric_scores = [
        s["score"] for s in scores if isinstance(s["score"], (int, float))
    ]
    avg_score = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0
    best_score = max(numeric_scores) if numeric_scores else 0

    print(f"\nSUMMARY METRICS")
    print("-" * 40)
    print(f"Total Cycles: {total}")
    print(f"Successful: {successful}/{total} ({success_rate:.1f}%)")
    print(f"Average Score: {avg_score:.1f}%")
    print(f"Best Score: {best_score:.1f}%")

    if numeric_scores:
        # Find most recent successful cycle
        recent_successful = None
        for score_data in reversed(scores):
            if (
                score_data["success"]
                and isinstance(score_data["score"], (int, float))
                and score_data["score"] > 0
            ):
                recent_successful = score_data
                break

        if recent_successful:
            print(
                f"Best Recent: Cycle {recent_successful['cycle']} - {recent_successful['score']:.1f}% - {recent_successful['title'][:30]}"
            )


if __name__ == "__main__":
    main()
