#!/usr/bin/env python3
"""
Automated Cycle Runner with Multi-Benchmark Evaluation

Integrates continuous evaluation into each development cycle:
- Automatically runs benchmarks when appropriate
- Tracks progress across cycles
- Maintains evaluation history
- Generates progress reports
- Ensures authentic evaluation on each cycle

PRINCIPLE: SYSTEMATIC CONTINUOUS IMPROVEMENT
"""

import asyncio
import json
import os
import sys
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.benchmarking.continuous_evaluation import ContinuousEvaluation
from src.benchmarking.cycle16_multi_benchmark_framework import MultiBenchmarkFramework

class AutomatedCycleRunner:
    """Automated cycle execution with integrated benchmark evaluation"""

    def __init__(self):
        self.continuous_eval = ContinuousEvaluation()
        self.multi_benchmark = MultiBenchmarkFramework()

    async def run_cycle_with_evaluation(self, cycle_number: int, cycle_focus: str = "") -> Dict[str, Any]:
        """Run a development cycle with integrated evaluation"""
        print(f"{'='*60}")
        print(f"AUTOMATED CYCLE {cycle_number:03d}")
        if cycle_focus:
            print(f"Focus: {cycle_focus}")
        print(f"{'='*60}")

        cycle_start = time.time()

        # Step 1: Check if evaluation should run
        should_run_eval = self.continuous_eval.should_run_evaluation(cycle_number)
        eval_results = {}

        if should_run_eval:
            print(f"\n📊 Running multi-benchmark evaluation for Cycle {cycle_number}")
            try:
                eval_results = await self.multi_benchmark.run_all_benchmarks()
                eval_results["cycle"] = cycle_number

                # Record in continuous evaluation system
                self.continuous_eval.record_evaluation(eval_results)

                print(f"✅ Evaluation completed - Overall Score: {eval_results.get('overall_score', 0):.1f}")

            except Exception as e:
                print(f"❌ Evaluation failed: {e}")
                eval_results = {"error": str(e), "cycle": cycle_number}
        else:
            print(f"\n⏭️  Skipping evaluation for Cycle {cycle_number} (recent evaluation exists)")

        # Step 2: Generate progress analysis
        progress_analysis = self.continuous_eval.generate_progress_report()

        # Step 3: Get benchmark summary
        benchmark_summary = self.continuous_eval.get_benchmark_summary()

        # Step 4: Compile cycle results
        cycle_results = {
            "cycle": cycle_number,
            "focus": cycle_focus,
            "execution_time_seconds": round(time.time() - cycle_start, 2),
            "evaluation_run": should_run_eval,
            "evaluation_results": eval_results,
            "progress_analysis": progress_analysis,
            "benchmark_summary": benchmark_summary,
            "recommendations": self.generate_cycle_recommendations(progress_analysis, eval_results),
            "next_steps": self.generate_next_steps(cycle_number, eval_results)
        }

        # Step 5: Save cycle results
        self.save_cycle_results(cycle_results)

        # Step 6: Display summary
        self.display_cycle_summary(cycle_results)

        return cycle_results

    def generate_cycle_recommendations(self, progress_analysis: Dict[str, Any], eval_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for the next cycle"""
        recommendations = []

        if progress_analysis.get("success_threshold_met", False):
            recommendations.append("✅ Success threshold met - continue current strategy")
        else:
            recommendations.append("🎯 Success threshold not met - consider new approaches")

        # Add recommendations from progress analysis
        progress_recs = progress_analysis.get("recommendations", [])
        recommendations.extend(progress_recs)

        # Add evaluation-specific recommendations
        if eval_results and "overall_score" in eval_results:
            score = eval_results["overall_score"]
            if score >= 30:
                recommendations.append("🚀 High performance achieved - aim for 35%+ next cycle")
            elif score >= 25:
                recommendations.append("📈 Good performance - focus on consistency")
            else:
                recommendations.append("⚠️  Performance needs improvement - review enhancement strategies")

        return recommendations

    def generate_next_steps(self, cycle_number: int, eval_results: Dict[str, Any]) -> List[str]:
        """Generate next steps for the following cycle"""
        next_steps = []

        # Determine when next evaluation should run
        next_eval_cycle = cycle_number + (5 - (cycle_number % 5))
        next_steps.append(f"Schedule next full evaluation for Cycle {next_eval_cycle}")

        # Add steps based on current performance
        if eval_results and "improvements" in eval_results:
            improvements = eval_results["improvements"]
            improving_benchmarks = [k for k, v in improvements.items() if v > 5]
            declining_benchmarks = [k for k, v in improvements.items() if v < -5]

            if improving_benchmarks:
                next_steps.append(f"Continue successful strategies for: {', '.join(improving_benchmarks)}")
            if declining_benchmarks:
                next_steps.append(f"Address performance issues in: {', '.join(declining_benchmarks)}")

        # Add standard development tasks
        next_steps.append("Review and optimize prompt engineering techniques")
        next_steps.append("Consider new enhancement strategies based on results")
        next_steps.append("Update baseline scores if significant improvements achieved")

        return next_steps

    def save_cycle_results(self, cycle_results: Dict[str, Any]):
        """Save cycle results to file"""
        results_dir = "src/benchmarking/cycle_results"
        os.makedirs(results_dir, exist_ok=True)

        cycle_file = os.path.join(results_dir, f"cycle_{cycle_results['cycle']:03d}_automated_results.json")

        with open(cycle_file, 'w') as f:
            json.dump(cycle_results, f, indent=2)

        print(f"\n💾 Cycle results saved to: {cycle_file}")

    def display_cycle_summary(self, cycle_results: Dict[str, Any]):
        """Display formatted cycle summary"""
        print(f"\n{'='*60}")
        print(f"CYCLE {cycle_results['cycle']:03d} SUMMARY")
        print(f"{'='*60}")

        print(f"Focus: {cycle_results.get('focus', 'General improvement')}")
        print(f"Execution Time: {cycle_results.get('execution_time_seconds', 0):.1f}s")
        print(f"Evaluation Run: {'Yes' if cycle_results.get('evaluation_run') else 'No'}")

        if cycle_results.get('evaluation_results') and 'overall_score' in cycle_results['evaluation_results']:
            eval_score = cycle_results['evaluation_results']['overall_score']
            print(f"Evaluation Score: {eval_score:.1f}%")

            improvements = cycle_results['evaluation_results'].get('improvements', {})
            if improvements:
                print("Improvements vs Baseline:")
                for benchmark, improvement in improvements.items():
                    print(f"  {benchmark}: {improvement:+.1f}%")

        progress = cycle_results.get('progress_analysis', {})
        if not progress.get('error'):
            print(f"Success Rate: {progress.get('success_rate', 0):.1f}%")
            print(f"Success Threshold Met: {'Yes' if progress.get('success_threshold_met') else 'No'}")

        print("\nRecommendations:")
        for rec in cycle_results.get('recommendations', [])[:3]:  # Show top 3
            print(f"  • {rec}")

        print(f"\n{'='*60}")

    async def run_continuous_cycles(self, start_cycle: int = 17, max_cycles: int = 5) -> List[Dict[str, Any]]:
        """Run multiple cycles continuously"""
        print(f"🔄 Starting continuous cycle execution from Cycle {start_cycle}")
        print(f"📋 Planning to run {max_cycles} cycles")
        print(f"📊 Full evaluations will run on cycles divisible by 5")
        print(f"⏰ Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        cycle_results = []

        for i in range(max_cycles):
            cycle_num = start_cycle + i
            cycle_focus = self.generate_cycle_focus(cycle_num)

            try:
                result = await self.run_cycle_with_evaluation(cycle_num, cycle_focus)
                cycle_results.append(result)

                # Brief pause between cycles
                if i < max_cycles - 1:
                    print(f"\n⏳ Preparing for Cycle {cycle_num + 1}...")
                    await asyncio.sleep(2)

            except KeyboardInterrupt:
                print(f"\n⏹️  Cycle execution interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Cycle {cycle_num} failed: {e}")
                continue

        # Generate overall summary
        self.generate_multi_cycle_summary(cycle_results)

        return cycle_results

    def generate_cycle_focus(self, cycle_number: int) -> str:
        """Generate a focus area for the given cycle"""
        focus_areas = [
            "Prompt optimization and refinement",
            "Mathematical reasoning enhancement",
            "Logical reasoning improvements",
            "Code generation and analysis",
            "Multi-step problem decomposition",
            "Context integration optimization",
            "Self-verification techniques",
            "Performance optimization",
            "Error handling and recovery",
            "Advanced reasoning strategies"
        ]

        return focus_areas[cycle_number % len(focus_areas)]

    def generate_multi_cycle_summary(self, cycle_results: List[Dict[str, Any]]):
        """Generate summary of multiple cycles"""
        if not cycle_results:
            return

        print(f"\n{'='*80}")
        print(f"MULTI-CYCLE EXECUTION SUMMARY")
        print(f"{'='*80}")

        successful_cycles = sum(1 for c in cycle_results if c.get('evaluation_results', {}).get('success', False))
        evaluations_run = sum(1 for c in cycle_results if c.get('evaluation_run'))

        print(f"Cycles Completed: {len(cycle_results)}")
        print(f"Evaluations Run: {evaluations_run}")
        print(f"Successful Cycles: {successful_cycles}")

        # Get scores from evaluated cycles
        scores = [c.get('evaluation_results', {}).get('overall_score', 0)
                 for c in cycle_results if c.get('evaluation_results') and 'overall_score' in c.get('evaluation_results', {})]

        if scores:
            print(f"Score Range: {min(scores):.1f} - {max(scores):.1f}")
            print(f"Average Score: {sum(scores) / len(scores):.1f}")

        print(f"\nExecution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")

async def main():
    """Main execution"""
    runner = AutomatedCycleRunner()

    # Run continuous cycles
    await runner.run_continuous_cycles(start_cycle=17, max_cycles=3)

if __name__ == "__main__":
    asyncio.run(main())