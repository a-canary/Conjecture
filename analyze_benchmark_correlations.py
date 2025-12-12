#!/usr/bin/env python3
"""
Benchmark Correlation Analysis

Analyzes correlations between benchmark scores and the number of claims
created or evaluated in systematic improvement cycles.

Hypothesis: There may be a correlation between improvement scores and
the threshold of 10+ claims created or evaluated.
"""

import json
import os
import glob
from typing import Dict, List, Tuple, Any
from pathlib import Path
import statistics
from collections import defaultdict

class BenchmarkAnalyzer:
    """Analyzes benchmark cycle results for correlations"""

    def __init__(self, results_dir: str = "src/benchmarking/cycle_results"):
        self.results_dir = results_dir
        self.cycles_data = {}
        self.analysis_results = {}

    def load_all_cycles(self) -> Dict[str, Any]:
        """Load all cycle result files"""
        print("Loading cycle results...")

        cycle_files = glob.glob(os.path.join(self.results_dir, "cycle_*_results.json"))

        for file_path in cycle_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Extract cycle number from filename
                filename = os.path.basename(file_path)
                cycle_num = filename.split('_')[1]

                self.cycles_data[cycle_num] = data
                print(f"  Loaded {filename}")

            except Exception as e:
                print(f"  Error loading {file_path}: {e}")

        print(f"Loaded {len(self.cycles_data)} cycle results")
        return self.cycles_data

    def extract_claims_data(self, cycle_data: Dict[str, Any]) -> Dict[str, int]:
        """Extract claims created and evaluated from cycle data"""
        claims_info = {
            "claims_created": 0,
            "claims_evaluated": 0,
            "problems_tested": 0,
            "test_cases": 0
        }

        # Extract from test_results
        test_results = cycle_data.get("test_results", {})

        # Count problems/tested
        if "total_problems" in test_results:
            claims_info["problems_tested"] = test_results["total_problems"]

        # Count test cases from reasoning_results
        reasoning_results = test_results.get("reasoning_results", [])
        claims_info["test_cases"] = len(reasoning_results)
        claims_info["claims_evaluated"] = len(reasoning_results)

        # Count from decomposition_results
        decomp_results = test_results.get("decomposition_results", [])
        if decomp_results:
            claims_info["test_cases"] = max(claims_info["test_cases"], len(decomp_results))
            claims_info["claims_evaluated"] = max(claims_info["claims_evaluated"], len(decomp_results))

        # Count from baseline_results and enhanced_results
        baseline_results = test_results.get("baseline_results", [])
        enhanced_results = test_results.get("enhanced_results", [])

        if baseline_results:
            claims_info["claims_evaluated"] += len(baseline_results)
        if enhanced_results:
            claims_info["claims_evaluated"] += len(enhanced_results)

        # Additional extraction from cycle-specific data
        # Some cycles might have additional claim creation data
        if "cycle_info" in cycle_data:
            cycle_info = cycle_data["cycle_info"]
            if "problems_tested" in cycle_info:
                claims_info["problems_tested"] = cycle_info["problems_tested"]

        # Look for any claims_created field
        if "claims_created" in cycle_data:
            claims_info["claims_created"] = cycle_data["claims_created"]

        return claims_info

    def extract_scores(self, cycle_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract improvement scores from cycle data"""
        scores = {}

        # Primary scores
        if "estimated_improvement" in cycle_data:
            scores["estimated_improvement"] = cycle_data["estimated_improvement"]

        if "measured_improvement" in cycle_data:
            scores["measured_improvement"] = cycle_data["measured_improvement"]

        if "actual_improvement" in cycle_data:
            scores["actual_improvement"] = cycle_data["actual_improvement"]

        # Success rates from test_results
        test_results = cycle_data.get("test_results", {})

        if "overall_success_rate" in test_results:
            scores["overall_success_rate"] = test_results["overall_success_rate"]

        if "enhanced_accuracy" in test_results:
            scores["enhanced_accuracy"] = test_results["enhanced_accuracy"]

        if "baseline_accuracy" in test_results:
            scores["baseline_accuracy"] = test_results["baseline_accuracy"]

        # Additional metrics
        success_fields = [
            "classification_accuracy", "strategy_relevance_rate",
            "feature_detection_rate", "strategy_accuracy",
            "indicator_detection_rate", "complexity_accuracy"
        ]

        for field in success_fields:
            if field in test_results:
                scores[field] = test_results[field]

        return scores

    def analyze_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between scores and claim counts"""
        print("\nAnalyzing correlations...")

        correlation_data = []

        for cycle_num, cycle_data in self.cycles_data.items():
            try:
                claims_info = self.extract_claims_data(cycle_data)
                scores = self.extract_scores(cycle_data)

                # Use the primary improvement score for correlation analysis
                improvement_score = None
                if "measured_improvement" in scores:
                    improvement_score = scores["measured_improvement"]
                elif "actual_improvement" in scores:
                    improvement_score = scores["actual_improvement"]
                elif "estimated_improvement" in scores:
                    improvement_score = scores["estimated_improvement"]
                elif "overall_success_rate" in scores:
                    improvement_score = scores["overall_success_rate"]

                if improvement_score is not None:
                    correlation_data.append({
                        "cycle": f"cycle_{cycle_num}",
                        "cycle_num": int(cycle_num),
                        "improvement_score": improvement_score,
                        "claims_created": claims_info["claims_created"],
                        "claims_evaluated": claims_info["claims_evaluated"],
                        "problems_tested": claims_info["problems_tested"],
                        "test_cases": claims_info["test_cases"],
                        "enhancement_type": cycle_data.get("enhancement_type", "Unknown")
                    })

            except Exception as e:
                print(f"  Error processing cycle {cycle_num}: {e}")

        print(f"Processed {len(correlation_data)} cycles for correlation analysis")

        # Calculate correlations
        correlations = self._calculate_correlations(correlation_data)

        # Test 10+ claims hypothesis
        threshold_analysis = self._test_claims_threshold(correlation_data)

        self.analysis_results = {
            "correlation_data": correlation_data,
            "correlations": correlations,
            "threshold_analysis": threshold_analysis
        }

        return self.analysis_results

    def _calculate_correlations(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate correlation coefficients"""
        correlations = {}

        # Extract data series
        scores = [item["improvement_score"] for item in data]
        claims_evaluated = [item["claims_evaluated"] for item in data]
        problems_tested = [item["problems_tested"] for item in data]
        test_cases = [item["test_cases"] for item in data]

        # Calculate correlations
        if len(scores) > 1:
            correlations["score_vs_claims_evaluated"] = self._correlation_coefficient(scores, claims_evaluated)
            correlations["score_vs_problems_tested"] = self._correlation_coefficient(scores, problems_tested)
            correlations["score_vs_test_cases"] = self._correlation_coefficient(scores, test_cases)

        return correlations

    def _correlation_coefficient(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(a * b for a, b in zip(x, y))
        sum_x2 = sum(a * a for a in x)
        sum_y2 = sum(b * b for b in y)

        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))**0.5

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _test_claims_threshold(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test hypothesis about 10+ claims threshold"""
        threshold_analysis = {
            "hypothesis": "There is a correlation between improvement scores and 10+ claims evaluated",
            "threshold_10_plus": {},
            "threshold_5_plus": {},
            "all_data": {}
        }

        # Test different thresholds
        for threshold in [10, 5]:
            group_name = f"threshold_{threshold}_plus" if threshold == 10 else "threshold_5_plus"

            above_threshold = [item for item in data if item["claims_evaluated"] >= threshold]
            below_threshold = [item for item in data if item["claims_evaluated"] < threshold]

            if above_threshold:
                above_scores = [item["improvement_score"] for item in above_threshold]
                above_avg = statistics.mean(above_scores)
                above_median = statistics.median(above_scores)
            else:
                above_avg = above_median = 0.0

            if below_threshold:
                below_scores = [item["improvement_score"] for item in below_threshold]
                below_avg = statistics.mean(below_scores)
                below_median = statistics.median(below_scores)
            else:
                below_avg = below_median = 0.0

            threshold_analysis[group_name] = {
                f"above_{threshold}_claims": {
                    "count": len(above_threshold),
                    "avg_score": above_avg,
                    "median_score": above_median,
                    "scores": above_scores
                },
                f"below_{threshold}_claims": {
                    "count": len(below_threshold),
                    "avg_score": below_avg,
                    "median_score": below_median,
                    "scores": below_scores
                },
                "difference": {
                    "avg_diff": above_avg - below_avg,
                    "median_diff": above_median - below_median
                }
            }

        # Overall statistics
        all_scores = [item["improvement_score"] for item in data]
        all_evaluated = [item["claims_evaluated"] for item in data]

        threshold_analysis["all_data"] = {
            "total_cycles": len(data),
            "avg_score": statistics.mean(all_scores),
            "median_score": statistics.median(all_scores),
            "avg_claims_evaluated": statistics.mean(all_evaluated),
            "median_claims_evaluated": statistics.median(all_evaluated)
        }

        return threshold_analysis

    def generate_report(self) -> str:
        """Generate a comprehensive analysis report"""
        if not self.analysis_results:
            return "No analysis results available. Run analyze_correlations() first."

        report = []
        report.append("# Benchmark Correlation Analysis Report")
        report.append("=" * 50)
        report.append("")

        # Data summary
        data = self.analysis_results["correlation_data"]
        report.append(f"## Data Summary")
        report.append(f"- Total cycles analyzed: {len(data)}")
        report.append(f"- Cycle range: {min(item['cycle_num'] for item in data)} - {max(item['cycle_num'] for item in data)}")
        report.append("")

        # Correlation results
        correlations = self.analysis_results["correlations"]
        report.append("## Correlation Analysis")
        report.append("Correlation coefficients between improvement scores and claim counts:")
        report.append("")

        for metric, corr in correlations.items():
            strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.3 else "weak"
            direction = "positive" if corr > 0 else "negative"
            report.append(f"- {metric}: {corr:.3f} ({strength} {direction})")
        report.append("")

        # Threshold analysis
        threshold = self.analysis_results["threshold_analysis"]
        report.append("## Claims Threshold Analysis")
        report.append("")

        # 10+ claims threshold
        above_10 = threshold["threshold_10_plus"]["above_10_claims"]
        below_10 = threshold["threshold_10_plus"]["below_10_claims"]

        report.append("### Hypothesis: 10+ Claims Evaluated")
        report.append(f"- Cycles with 10+ claims evaluated: {above_10['count']}")
        report.append(f"- Cycles with <10 claims evaluated: {below_10['count']}")
        report.append(f"- Average score (10+ claims): {above_10['avg_score']:.1f}%")
        report.append(f"- Average score (<10 claims): {below_10['avg_score']:.1f}%")
        report.append(f"- Score difference: {threshold['threshold_10_plus']['difference']['avg_diff']:.1f}%")
        report.append("")

        # 5+ claims threshold (additional insight)
        above_5 = threshold["threshold_5_plus"]["above_5_claims"]
        below_5 = threshold["threshold_5_plus"]["below_5_claims"]

        report.append("### Additional Analysis: 5+ Claims Evaluated")
        report.append(f"- Cycles with 5+ claims evaluated: {above_5['count']}")
        report.append(f"- Cycles with <5 claims evaluated: {below_5['count']}")
        report.append(f"- Average score (5+ claims): {above_5['avg_score']:.1f}%")
        report.append(f"- Average score (<5 claims): {below_5['avg_score']:.1f}%")
        report.append("")

        # Overall statistics
        all_data = threshold["all_data"]
        report.append("## Overall Statistics")
        report.append(f"- Average improvement score: {all_data['avg_score']:.1f}%")
        report.append(f"- Median improvement score: {all_data['median_score']:.1f}%")
        report.append(f"- Average claims evaluated: {all_data['avg_claims_evaluated']:.1f}")
        report.append(f"- Median claims evaluated: {all_data['median_claims_evaluated']:.1f}")
        report.append("")

        # Individual cycle data
        report.append("## Individual Cycle Data")
        report.append("| Cycle | Claims Evaluated | Improvement Score | Enhancement Type |")
        report.append("|-------|----------------|------------------|------------------|")

        for item in sorted(data, key=lambda x: x["cycle_num"]):
            report.append(f"| {item['cycle']} | {item['claims_evaluated']} | {item['improvement_score']:.1f}% | {item['enhancement_type']} |")
        report.append("")

        # Conclusions
        report.append("## Conclusions")

        if threshold["threshold_10_plus"]["difference"]["avg_diff"] > 0:
            report.append("[SUPPORTED] **HYPOTHESIS SUPPORTED**: Cycles with 10+ claims evaluated show higher average improvement scores.")
        else:
            report.append("[NOT SUPPORTED] **HYPOTHESIS NOT SUPPORTED**: No clear correlation between 10+ claims evaluated and improvement scores.")

        if correlations.get("score_vs_claims_evaluated", 0) > 0.3:
            report.append("[POSITIVE] **POSITIVE CORRELATION**: Found moderate positive correlation between claims evaluated and improvement scores.")
        elif correlations.get("score_vs_claims_evaluated", 0) < -0.3:
            report.append("[NEGATIVE] **NEGATIVE CORRELATION**: Found negative correlation between claims evaluated and improvement scores.")
        else:
            report.append("[WEAK] **WEAK CORRELATION**: Found weak or no correlation between claims evaluated and improvement scores.")

        return "\n".join(report)

    def run_analysis(self) -> str:
        """Run complete analysis and return report"""
        print("Starting benchmark correlation analysis...")

        # Load data
        self.load_all_cycles()

        # Analyze correlations
        self.analyze_correlations()

        # Generate report
        report = self.generate_report()

        # Save report
        report_path = "benchmark_correlation_report.md"
        with open(report_path, 'w') as f:
            f.write(report)

        print(f"\nAnalysis complete! Report saved to: {report_path}")
        print("\n" + "="*50)
        print(report)

        return report

def main():
    """Main analysis function"""
    analyzer = BenchmarkAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()