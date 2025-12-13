#!/usr/bin/env python3
"""
Simple Benchmark Score Reporter

Reports key benchmark scores from code validation without complex processing.
Focuses on delivering the core metrics and recent results.

PRINCIPLE: CLEAR BENCHMARK SCORE REPORTING
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any

class SimpleBenchmarkReporter:
    """Simple, robust benchmark score reporting"""

    def __init__(self):
        self.benchmark_dir = "src/benchmarking"

    def generate_report(self) -> Dict[str, Any]:
        """Generate simple benchmark score report"""
        print("BENCHMARK SCORE REPORT")
        print("Code Validation Results Summary")
        print("=" * 50)

        # Collect benchmark results
        cycle_results = self.load_cycle_results()
        evaluation_results = self.load_evaluation_results()
        demo_results = self.load_demo_results()

        # Create report
        report = {
            "timestamp": datetime.now().isoformat(),
            "cycle_results": cycle_results,
            "evaluation_results": evaluation_results,
            "demo_results": demo_results,
            "summary": self.create_summary(cycle_results, evaluation_results, demo_results)
        }

        # Display results
        self.display_report(report)

        # Save report
        self.save_report(report)

        return report

    def load_cycle_results(self) -> List[Dict[str, Any]]:
        """Load cycle results safely"""
        cycle_results = []
        cycle_dir = os.path.join(self.benchmark_dir, "cycle_results")

        if not os.path.exists(cycle_dir):
            print("[No cycle results directory found]")
            return cycle_results

        cycle_files = [f for f in os.listdir(cycle_dir) if f.startswith("cycle_") and f.endswith("_results.json")]

        print(f"[Found {len(cycle_files)} cycle result files]")

        for cycle_file in sorted(cycle_files):
            file_path = os.path.join(cycle_dir, cycle_file)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                cycle_summary = {
                    "file": cycle_file,
                    "cycle": data.get("cycle", "Unknown"),
                    "title": data.get("title", "Unknown"),
                    "overall_score": data.get("overall_score", 0),
                    "success": data.get("success", False),
                    "methodology": data.get("methodology", "Unknown")
                }

                cycle_results.append(cycle_summary)

            except Exception as e:
                print(f"[Error loading {cycle_file}: {str(e)[:50]}...]")

        print(f"[Successfully loaded {len(cycle_results)} cycle results]")
        return cycle_results

    def load_evaluation_results(self) -> List[Dict[str, Any]]:
        """Load evaluation results safely"""
        eval_results = []

        # Look for evaluation files
        eval_patterns = [
            "evaluation_*.json",
            "*evaluation*.json",
            "improved_claim_system_results.json",
            "detailed_evaluation_demo_results.json"
        ]

        for pattern in eval_patterns:
            files = []
            try:
                import glob
                files = glob.glob(os.path.join(self.benchmark_dir, pattern))
            except:
                continue

            for file_path in files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                    eval_summary = {
                        "file": os.path.basename(file_path),
                        "overall_score": data.get("overall_score", 0),
                        "method": data.get("evaluation_method", data.get("methodology", "Unknown")),
                        "success": data.get("success", True),
                        "benchmarks": data.get("benchmarks_run", []),
                        "performance_metrics": data.get("performance_metrics", {})
                    }

                    eval_results.append(eval_summary)

                except Exception as e:
                    print(f"[Error loading {os.path.basename(file_path)}: {str(e)[:30]}...]")

        print(f"[Successfully loaded {len(eval_results)} evaluation results]")
        return eval_results

    def load_demo_results(self) -> List[Dict[str, Any]]:
        """Load demonstration results"""
        demo_results = []

        # Look for our recent demo files
        demo_files = [
            "improved_claim_system_results.json",
            "detailed_evaluation_demo_results.json"
        ]

        for demo_file in demo_files:
            file_path = os.path.join(self.benchmark_dir, demo_file)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                    demo_summary = {
                        "file": demo_file,
                        "type": "demonstration",
                        "performance_metrics": data.get("performance_metrics", {}),
                        "improvement": data.get("performance_metrics", {}).get("improvement", 0)
                    }

                    demo_results.append(demo_summary)

                except Exception as e:
                    print(f"[Error loading demo {demo_file}: {str(e)[:30]}...]")

        return demo_results

    def create_summary(self, cycles: List, evaluations: List, demos: List) -> Dict[str, Any]:
        """Create summary statistics"""
        summary = {
            "total_cycles": len(cycles),
            "successful_cycles": len([c for c in cycles if c.get("success", False)]),
            "average_cycle_score": sum(c.get("overall_score", 0) for c in cycles) / len(cycles) if cycles else 0,
            "best_cycle_score": max([c.get("overall_score", 0) for c in cycles]) if cycles else 0,
            "total_evaluations": len(evaluations),
            "average_evaluation_score": sum(e.get("overall_score", 0) for e in evaluations) / len(evaluations) if evaluations else 0,
            "demo_improvements": [d.get("improvement", 0) for d in demos],
            "benchmark_methods": list(set([c.get("methodology", "Unknown") for c in cycles] + [e.get("method", "Unknown") for e in evaluations]))
        }

        return summary

    def display_report(self, report: Dict[str, Any]):
        """Display benchmark report"""
        print(f"\nCYCLE RESULTS SUMMARY")
        print("-" * 40)

        cycles = report["cycle_results"]
        if cycles:
            print(f"Total Cycles: {len(cycles)}")
            successful = len([c for c in cycles if c.get("success", False)])
            print(f"Successful: {successful}/{len(cycles)} ({successful/len(cycles)*100:.1f}%)")

            # Show recent cycles
            recent_cycles = sorted(cycles, key=lambda x: x.get("cycle", 0))[-5:]
            print(f"\nRecent Cycles:")
            for cycle in recent_cycles:
                status = "PASS" if cycle.get("success", False) else "FAIL"
                print(f"  {status} Cycle {cycle.get('cycle', '?'):>3}: {cycle.get('overall_score', 0):>6.1f}% - {cycle.get('title', 'Unknown')[:40]}")

            # Best cycle
            best_cycle = max(cycles, key=lambda x: x.get("overall_score", 0))
            print(f"\nBest Cycle: Cycle {best_cycle.get('cycle', '?')} - {best_cycle.get('overall_score', 0):.1f}%")
        else:
            print("[No cycle results available]")

        print(f"\nEVALUATION RESULTS SUMMARY")
        print("-" * 40)

        evaluations = report["evaluation_results"]
        if evaluations:
            print(f"Total Evaluations: {len(evaluations)}")
            avg_score = sum(e.get("overall_score", 0) for e in evaluations) / len(evaluations)
            print(f"Average Score: {avg_score:.1f}%")

            print(f"\nEvaluations:")
            for eval_result in evaluations:
                score = eval_result.get("overall_score", 0)
                method = eval_result.get("method", "Unknown")[:25]
                print(f"  {score:>6.1f}% - {method}")
        else:
            print("[No evaluation results available]")

        print(f"\nDEMONSTRATION RESULTS")
        print("-" * 40)

        demos = report["demo_results"]
        if demos:
            for demo in demos:
                improvement = demo.get("improvement", 0)
                metrics = demo.get("performance_metrics", {})
                direct_acc = metrics.get("direct_accuracy", 0)
                conj_acc = metrics.get("conjecture_accuracy", 0)
                print(f"  {demo['file'][:30]:<30}")
                print(f"    Improvement: {improvement:+.1f}%")
                if direct_acc and conj_acc:
                    print(f"    Direct: {direct_acc:.1f}% → Conjecture: {conj_acc:.1f}%")
        else:
            print("[No demonstration results available]")

        # Overall summary
        summary = report["summary"]
        print(f"\nOVERALL SUMMARY")
        print("-" * 40)
        print(f"Total Validation Runs: {summary['total_cycles'] + summary['total_evaluations']}")
        print(f"Average Cycle Score: {summary['average_cycle_score']:.1f}%")
        print(f"Best Cycle Score: {summary['best_cycle_score']:.1f}%")
        print(f"Average Evaluation Score: {summary['average_evaluation_score']:.1f}%")

        if summary["demo_improvements"]:
            avg_improvement = sum(summary["demo_improvements"]) / len(summary["demo_improvements"])
            print(f"Average Demo Improvement: {avg_improvement:+.1f}%")

        methods = [m for m in summary["benchmark_methods"] if m != "Unknown"]
        if methods:
            print(f"Methods Used: {', '.join(methods[:3])}")

    def save_report(self, report: Dict[str, Any]):
        """Save report to file"""
        report_file = os.path.join(self.benchmark_dir, "simple_benchmark_report.json")

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n[Report saved to: {report_file}]")

def main():
    """Generate simple benchmark report"""
    reporter = SimpleBenchmarkReporter()
    return reporter.generate_report()

if __name__ == "__main__":
    main()