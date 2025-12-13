#!/usr/bin/env python3
"""
Comprehensive Benchmark Score Reporter

Gathers and reports all benchmark scores from the codebase validation.
Includes historical cycle results, current evaluation metrics, and baseline comparisons.

PRINCIPLE: COMPLETE TRANSPARENCY OF BENCHMARK PERFORMANCE
"""

import json
import os
import glob
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

class BenchmarkScoreReporter:
    """Comprehensive benchmark score aggregation and reporting"""

    def __init__(self):
        self.benchmark_dir = Path("src/benchmarking")
        self.cycle_results_dir = self.benchmark_dir / "cycle_results"
        self.evaluation_results_dir = self.benchmark_dir / "evaluation_results"
        self.current_results = {}
        self.historical_results = []
        self.baseline_scores = {}

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate complete benchmark score report"""
        print("COMPREHENSIVE BENCHMARK SCORE REPORT")
        print("Aggregating all benchmark results from codebase validation")
        print("=" * 70)

        # Load baseline scores
        self.load_baseline_scores()

        # Load historical cycle results
        self.load_historical_cycle_results()

        # Load recent evaluation results
        self.load_recent_evaluation_results()

        # Load demonstration results
        self.load_demonstration_results()

        # Generate comprehensive analysis
        report = self.create_comprehensive_analysis()

        # Display results
        self.display_benchmark_scores(report)

        # Save report
        self.save_comprehensive_report(report)

        return report

    def load_baseline_scores(self):
        """Load established baseline benchmark scores"""
        baseline_file = self.cycle_results_dir / "baseline_scores.json"

        if baseline_file.exists():
            with open(baseline_file, 'r') as f:
                self.baseline_scores = json.load(f)
            print(f"[Loaded baseline scores from {baseline_file}]")
        else:
            # Create default baseline
            self.baseline_scores = {
                "deepeval": 40.0,
                "gpqa": 35.0,
                "humaneval": 45.0,
                "arc_easy": 50.0,
                "overall": 42.5,
                "established": datetime.now().isoformat(),
                "methodology": "Enhanced Local Evaluation"
            }
            print("[Using default baseline scores]")

    def load_historical_cycle_results(self):
        """Load all historical cycle result files"""
        cycle_files = list(self.cycle_results_dir.glob("cycle_*_results.json"))

        print(f"\n[HISTORICAL CYCLE RESULTS - {len(cycle_files)} files found]")

        for cycle_file in sorted(cycle_files):
            try:
                with open(cycle_file, 'r') as f:
                    cycle_data = json.load(f)

                # Extract key metrics
                cycle_summary = {
                    "cycle": cycle_data.get("cycle", "Unknown"),
                    "title": cycle_data.get("title", "Unknown"),
                    "timestamp": cycle_data.get("timestamp", cycle_file.stat().st_mtime),
                    "overall_score": cycle_data.get("overall_score", 0),
                    "success": cycle_data.get("success", False),
                    "methodology": cycle_data.get("methodology", "Unknown"),
                    "file_path": str(cycle_file)
                }

                # Add detailed scores if available
                if "scores" in cycle_data:
                    cycle_summary["detailed_scores"] = cycle_data["scores"]

                self.historical_results.append(cycle_summary)

            except Exception as e:
                print(f"[Error loading {cycle_file}: {e}]")

        print(f"[Loaded {len(self.historical_results)} historical cycle results]")

    def load_recent_evaluation_results(self):
        """Load recent evaluation framework results"""
        eval_files = []

        # Check evaluation_results directory
        if self.evaluation_results_dir.exists():
            eval_files.extend(self.evaluation_results_dir.glob("evaluation_*.json"))

        # Check for recent evaluation files in benchmarking directory
        eval_files.extend(self.benchmark_dir.glob("*evaluation*.json"))
        eval_files.extend(self.benchmark_dir.glob("*results*.json"))

        print(f"\n[RECENT EVALUATION RESULTS - {len(eval_files)} files found]")

        for eval_file in eval_files:
            try:
                with open(eval_file, 'r') as f:
                    eval_data = json.load(f)

                # Store recent evaluation
                self.current_results[eval_file.name] = {
                    "file": eval_file.name,
                    "timestamp": eval_data.get("evaluation_timestamp", eval_file.stat().st_mtime),
                    "overall_score": eval_data.get("overall_score", 0),
                    "methodology": eval_data.get("evaluation_method", "Unknown"),
                    "benchmarks_run": eval_data.get("benchmarks_run", []),
                    "scores": eval_data.get("scores", {}),
                    "performance_metrics": eval_data.get("performance_metrics", {})
                }

            except Exception as e:
                print(f"[Error loading {eval_file}: {e}]")

        print(f"[Loaded {len(self.current_results)} recent evaluation results]")

    def load_demonstration_results(self):
        """Load demonstration system results"""
        demo_files = [
            self.benchmark_dir / "detailed_evaluation_demo_results.json",
            self.benchmark_dir / "improved_claim_system_results.json"
        ]

        for demo_file in demo_files:
            if demo_file.exists():
                try:
                    with open(demo_file, 'r') as f:
                        demo_data = json.load(f)

                    self.current_results[f"demo_{demo_file.name}"] = {
                        "file": f"demo_{demo_file.name}",
                        "type": "demonstration",
                        "timestamp": demo_data.get("evaluation_timestamp", demo_file.stat().st_mtime),
                        "performance_metrics": demo_data.get("performance_metrics", {}),
                        "improvements": demo_data.get("improvements", {})
                    }

                except Exception as e:
                    print(f"⚠ Error loading demo file {demo_file}: {e}")

    def create_comprehensive_analysis(self) -> Dict[str, Any]:
        """Create comprehensive analysis of all benchmark data"""
        analysis = {
            "report_timestamp": datetime.now().isoformat(),
            "baseline_scores": self.baseline_scores,
            "historical_cycles": {
                "total_cycles": len(self.historical_results),
                "successful_cycles": len([c for c in self.historical_results if c.get("success", False)]),
                "success_rate": len([c for c in self.historical_results if c.get("success", False)]) / len(self.historical_results) * 100 if self.historical_results else 0,
                "cycles": self.historical_results
            },
            "current_evaluations": self.current_results,
            "benchmark_trends": self.calculate_benchmark_trends(),
            "performance_summary": self.create_performance_summary(),
            "recommendations": self.generate_recommendations()
        }

        return analysis

    def calculate_benchmark_trends(self) -> Dict[str, Any]:
        """Calculate trends across benchmark categories"""
        trends = {}

        # Collect scores by benchmark type
        benchmark_scores = {
            "deepeval": [],
            "gpqa": [],
            "humaneval": [],
            "arc_easy": [],
            "overall": []
        }

        # Extract scores from historical cycles
        for cycle in self.historical_results:
            if "detailed_scores" in cycle:
                scores = cycle["detailed_scores"]
                for benchmark, score_data in scores.items():
                    if isinstance(score_data, dict) and "overall_score" in score_data:
                        benchmark_scores[benchmark].append(score_data["overall_score"])
                    elif isinstance(score_data, (int, float)):
                        benchmark_scores[benchmark].append(score_data)

        # Add overall scores
        for cycle in self.historical_results:
            if cycle["overall_score"]:
                benchmark_scores["overall"].append(cycle["overall_score"])

        # Calculate trends
        for benchmark, scores in benchmark_scores.items():
            if scores:
                baseline_val = self.baseline_scores.get(benchmark, 0)
                latest_score = scores[-1] if scores else 0
                avg_score = sum(scores) / len(scores)
                best_score = max(scores) if scores else 0
                improvement = ((latest_score - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0

                trends[benchmark] = {
                    "baseline": baseline_val,
                    "latest": latest_score,
                    "average": round(avg_score, 2),
                    "best": best_score,
                    "improvement_percent": round(improvement, 1),
                    "data_points": len(scores),
                    "trend_direction": "up" if len(scores) > 1 and scores[-1] > scores[0] else "down" if len(scores) > 1 else "stable"
                }

        return trends

    def create_performance_summary(self) -> Dict[str, Any]:
        """Create overall performance summary"""
        summary = {
            "total_evaluations": len(self.historical_results) + len(self.current_results),
            "success_rate": 0,
            "average_improvement": 0,
            "top_performing_benchmarks": [],
            "benchmark_coverage": set(),
            "methodologies_used": set()
        }

        # Calculate success rate
        total_evaluations = len(self.historical_results) + len(self.current_results)
        successful_evaluations = len([c for c in self.historical_results if c.get("success", False)])
        summary["success_rate"] = round((successful_evaluations / total_evaluations * 100), 1) if total_evaluations > 0 else 0

        # Collect benchmark coverage
        for eval_result in self.current_results.values():
            if "benchmarks_run" in eval_result:
                summary["benchmark_coverage"].update(eval_result["benchmarks_run"])
            if "scores" in eval_result:
                summary["benchmark_coverage"].update(eval_result["scores"].keys())

        # Collect methodologies
        for cycle in self.historical_results:
            if cycle.get("methodology"):
                summary["methodologies_used"].add(cycle["methodology"])

        summary["benchmark_coverage"] = list(summary["benchmark_coverage"])
        summary["methodologies_used"] = list(summary["methodologies_used"])

        return summary

    def generate_recommendations(self) -> List[str]:
        """Generate data-driven recommendations based on benchmark results"""
        recommendations = []

        trends = self.calculate_benchmark_trends()

        # Analyze trends for recommendations
        declining_benchmarks = [b for b, t in trends.items() if t.get("trend_direction") == "down"]
        improving_benchmarks = [b for b, t in trends.items() if t.get("trend_direction") == "up"]
        high_improvement = [b for b, t in trends.items() if t.get("improvement_percent", 0) > 20]

        if declining_benchmarks:
            recommendations.append(f"Focus improvement efforts on declining benchmarks: {', '.join(declining_benchmarks)}")

        if improving_benchmarks:
            recommendations.append(f"Continue successful strategies for improving benchmarks: {', '.join(improving_benchmarks)}")

        if high_improvement:
            recommendations.append(f"Scale successful approaches that achieved >20% improvement: {', '.join(high_improvement)}")

        # Check coverage
        summary = self.create_performance_summary()
        if len(summary["benchmark_coverage"]) < 4:
            recommendations.append("Expand benchmark coverage to include more evaluation domains")

        # Check success rate
        if summary["success_rate"] < 60:
            recommendations.append("Improve validation methodology to increase success rate above 60%")

        if not recommendations:
            recommendations.append("Current performance is stable - explore new optimization techniques")

        return recommendations

    def display_benchmark_scores(self, report: Dict[str, Any]):
        """Display comprehensive benchmark scores"""
        print(f"\n{'='*70}")
        print("COMPREHENSIVE BENCHMARK SCORE REPORT")
        print(f"{'='*70}")

        # Baseline vs Current
        print(f"\nBASELINE VS CURRENT PERFORMANCE")
        print("-" * 50)
        baseline = report["baseline_scores"]
        trends = report["benchmark_trends"]

        for benchmark, baseline_score in baseline.items():
            if benchmark in trends:
                trend = trends[benchmark]
                direction = "^" if trend["trend_direction"] == "up" else "v" if trend["trend_direction"] == "down" else "-"
                print(f"{benchmark.upper():<12} {baseline_score:>6.1f} → {trend['latest']:>6.1f} ({trend['improvement_percent']:>+6.1f}%) {direction}")

        # Historical Cycle Summary
        history = report["historical_cycles"]
        print(f"\nHISTORICAL CYCLE SUMMARY")
        print("-" * 50)
        print(f"Total Cycles: {history['total_cycles']}")
        print(f"Successful Cycles: {history['successful_cycles']}")
        print(f"Success Rate: {history['success_rate']:.1f}%")

        if history["cycles"]:
            # Show last 5 cycles
            recent_cycles = history["cycles"][-5:]
            print(f"\nRecent Cycles:")
            for cycle in recent_cycles:
                status = "PASS" if cycle["success"] else "FAIL"
                print(f"  {status} Cycle {cycle['cycle']:>3}: {cycle['overall_score']:>6.1f}% - {cycle['title'][:40]}")

        # Current Evaluations
        current = report["current_evaluations"]
        if current:
            print(f"\nCURRENT EVALUATION RESULTS")
            print("-" * 50)
            for name, result in current.items():
                print(f"{name[:30]:<30} {result.get('overall_score', 0):>6.1f}% - {result.get('methodology', 'Unknown')[:25]}")

        # Performance Summary
        summary = report["performance_summary"]
        print(f"\nPERFORMANCE SUMMARY")
        print("-" * 50)
        print(f"Total Evaluations: {summary['total_evaluations']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Benchmark Coverage: {len(summary['benchmark_coverage'])} benchmarks")
        print(f"Methodologies: {len(summary['methodologies_used'])} approaches")

        # Recommendations
        recommendations = report["recommendations"]
        print(f"\nRECOMMENDATIONS")
        print("-" * 50)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")

        print(f"\n{'='*70}")
        print(f"Report generated: {report['report_timestamp'][:19]}")
        print(f"{'='*70}")

    def save_comprehensive_report(self, report: Dict[str, Any]):
        """Save comprehensive report to file"""
        report_file = self.benchmark_dir / "comprehensive_benchmark_report.json"

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n[Comprehensive report saved to: {report_file}]")

def main():
    """Generate comprehensive benchmark score report"""
    reporter = BenchmarkScoreReporter()
    report = reporter.generate_comprehensive_report()
    return report

if __name__ == "__main__":
    main()