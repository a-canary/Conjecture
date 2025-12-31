#!/usr/bin/env python3
"""
Continuous Evaluation System for Multi-Benchmark Progress Tracking

This system provides:
- Baseline establishment for DeepEval, GPQA, HumanEval, ARC-Easy
- Progress tracking across cycles
- Automatic evaluation on each cycle
- Historical performance analysis
- Success threshold monitoring

PRINCIPLE: CONTINUOUS AUTHENTIC EVALUATION
"""

import json
import os
import time
from typing import Dict, List, Any
from datetime import datetime

class ContinuousEvaluation:
    """Continuous evaluation system for multi-benchmark tracking"""

    def __init__(self):
        self.results_dir = "src/benchmarking/cycle_results"
        self.baseline_file = os.path.join(self.results_dir, "baseline_scores.json")
        self.history_file = os.path.join(self.results_dir, "evaluation_history.json")
        self.current_cycle = 16

        # Initialize baseline scores
        self.ensure_baseline_exists()

    def ensure_baseline_exists(self):
        """Ensure baseline scores exist for comparison"""
        if not os.path.exists(self.baseline_file):
            baseline = {
                "deepeval": {
                    "overall_score": 20.0,
                    "exact_match_avg": 0.2,
                    "problems_evaluated": 5,
                    "established_cycle": 15
                },
                "gpqa": {
                    "overall_score": 25.0,
                    "accuracy": 0.25,
                    "problems_evaluated": 3,
                    "established_cycle": 16
                },
                "humaneval": {
                    "overall_score": 15.0,
                    "completion_rate": 0.15,
                    "problems_evaluated": 3,
                    "established_cycle": 16
                },
                "arc_easy": {
                    "overall_score": 30.0,
                    "accuracy": 0.30,
                    "problems_evaluated": 3,
                    "established_cycle": 16
                }
            }

            os.makedirs(self.results_dir, exist_ok=True)
            with open(self.baseline_file, 'w') as f:
                json.dump(baseline, f, indent=2)

            print(f"Established baseline scores in {self.baseline_file}")

    def load_baseline(self) -> Dict[str, Any]:
        """Load baseline scores for comparison"""
        with open(self.baseline_file, 'r') as f:
            return json.load(f)

    def load_cycle_results(self, cycle_number: int) -> Dict[str, Any]:
        """Load results from a specific cycle"""
        cycle_file = os.path.join(self.results_dir, f"cycle_{cycle_number:03d}_results.json")

        if not os.path.exists(cycle_file):
            return {}

        try:
            with open(cycle_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cycle {cycle_number}: {e}")
            return {}

    def record_evaluation(self, cycle_results: Dict[str, Any]) -> Dict[str, Any]:
        """Record evaluation results and track progress"""
        cycle_num = cycle_results.get("cycle", self.current_cycle)

        # Load history
        history = []
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                history = json.load(f)

        # Add current cycle results
        evaluation_entry = {
            "cycle": cycle_num,
            "timestamp": datetime.now().isoformat(),
            "overall_score": cycle_results.get("overall_score", 0),
            "benchmarks_run": cycle_results.get("benchmarks_run", []),
            "scores": cycle_results.get("scores", {}),
            "improvements": cycle_results.get("improvements", {}),
            "success": cycle_results.get("success", False)
        }

        history.append(evaluation_entry)

        # Save updated history
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)

        return evaluation_entry

    def generate_progress_report(self, cycles_back: int = 10) -> Dict[str, Any]:
        """Generate comprehensive progress report"""
        history = []
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                history = json.load(f)

        if not history:
            return {"error": "No evaluation history found"}

        # Get recent cycles
        recent_cycles = history[-cycles_back:] if len(history) > cycles_back else history

        # Calculate trends
        baseline = self.load_baseline()
        current = recent_cycles[-1] if recent_cycles else {}

        # Benchmark trends
        benchmark_trends = {}
        for benchmark in ["deepeval", "gpqa", "humaneval", "arc_easy"]:
            scores = [c.get("scores", {}).get(benchmark, {}).get("overall_score", 0)
                     for c in recent_cycles if benchmark in c.get("scores", {})]

            if len(scores) >= 2:
                trend = scores[-1] - scores[0]
                benchmark_trends[benchmark] = {
                    "current": scores[-1],
                    "baseline": baseline.get(benchmark, {}).get("overall_score", 0),
                    "trend": round(trend, 1),
                    "samples": len(scores)
                }

        # Overall trend
        overall_scores = [c.get("overall_score", 0) for c in recent_cycles if c.get("overall_score")]
        overall_trend = overall_scores[-1] - overall_scores[0] if len(overall_scores) >= 2 else 0

        # Success rate
        successful_cycles = sum(1 for c in recent_cycles if c.get("success", False))
        success_rate = (successful_cycles / len(recent_cycles)) * 100 if recent_cycles else 0

        return {
            "report_generated": datetime.now().isoformat(),
            "cycles_analyzed": len(recent_cycles),
            "current_cycle": current.get("cycle", 0),
            "overall_score": current.get("overall_score", 0),
            "success_rate": round(success_rate, 1),
            "overall_trend": round(overall_trend, 1),
            "benchmark_trends": benchmark_trends,
            "success_threshold_met": current.get("overall_score", 0) >= 25.0,
            "recommendations": self.generate_recommendations(benchmark_trends)
        }

    def generate_recommendations(self, trends: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on performance trends"""
        recommendations = []

        # Check for declining benchmarks
        declining = [b for b, t in trends.items() if t.get("trend", 0) < -5]
        if declining:
            recommendations.append(f"Focus on improving declining benchmarks: {', '.join(declining)}")

        # Check for high-performing benchmarks
        improving = [b for b, t in trends.items() if t.get("trend", 0) > 5]
        if improving:
            recommendations.append(f"Continue successful strategies for: {', '.join(improving)}")

        # Check for benchmarks below baseline
        below_baseline = [b for b, t in trends.items() if t.get("current", 0) < t.get("baseline", 0)]
        if below_baseline:
            recommendations.append(f"Address benchmarks below baseline: {', '.join(below_baseline)}")

        if not recommendations:
            recommendations.append("Performance is stable - consider exploring new enhancement strategies")

        return recommendations

    def should_run_evaluation(self, cycle_number: int) -> bool:
        """Determine if evaluation should run for this cycle"""
        # Always run evaluation for cycles that are multiples of 5, or if no recent evaluation
        return cycle_number % 5 == 0 or not self.has_recent_evaluation(cycle_number)

    def has_recent_evaluation(self, cycle_number: int, within_cycles: int = 3) -> bool:
        """Check if there was an evaluation within recent cycles"""
        history = []
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                history = json.load(f)

        recent_evaluations = [c for c in history if cycle_number - c.get("cycle", 0) <= within_cycles]
        return len(recent_evaluations) > 0

    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmark performance"""
        baseline = self.load_baseline()

        # Get latest results for each benchmark
        latest_results = {}
        for benchmark in ["deepeval", "gpqa", "humaneval", "arc_easy"]:
            # Find latest cycle that has this benchmark
            for cycle_num in range(self.current_cycle, 0, -1):
                cycle_results = self.load_cycle_results(cycle_num)
                if benchmark in cycle_results.get("scores", {}):
                    latest_results[benchmark] = cycle_results["scores"][benchmark]
                    break

        # Calculate improvements
        improvements = {}
        for benchmark, baseline_data in baseline.items():
            if benchmark in latest_results:
                baseline_score = baseline_data.get("overall_score", 0)
                current_score = latest_results[benchmark].get("overall_score", 0)
                if baseline_score > 0:
                    improvement = ((current_score - baseline_score) / baseline_score) * 100
                    improvements[benchmark] = round(improvement, 1)

        return {
            "baseline_established": datetime.fromtimestamp(os.path.getctime(self.baseline_file)).isoformat(),
            "latest_evaluation_cycle": max([int(f.split('_')[1].split('.')[0])
                                           for f in os.listdir(self.results_dir)
                                           if f.startswith('cycle_') and f.endswith('_results.json')],
                                          default=0),
            "baseline_scores": {k: v.get("overall_score", 0) for k, v in baseline.items()},
            "latest_scores": {k: v.get("overall_score", 0) for k, v in latest_results.items()},
            "improvements": improvements
        }

def main():
    """Test continuous evaluation system"""
    eval_system = ContinuousEvaluation()

    # Generate benchmark summary
    summary = eval_system.get_benchmark_summary()
    print("Benchmark Summary:")
    print(json.dumps(summary, indent=2))

    # Generate progress report
    report = eval_system.generate_progress_report()
    print("\nProgress Report:")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()