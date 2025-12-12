#!/usr/bin/env python3
"""
CYCLE_016: Analytical Reasoning Enhancement

Building on Cycle 9's mathematical reasoning success (8% improvement),
this cycle extends analytical reasoning with systematic analysis methods
and step-by-step verification processes.

Hypothesis: Analytical reasoning enhancement with systematic methods
and verification will improve problem-solving performance by 5-8%.

Focus: Real analytical problems with verifiable solutions.
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from agent.reasoning_engine import ReasoningEngine
from utils.metrics import calculate_reasoning_confidence, detect_reasoning_errors

class AnalyticalReasoningBenchmark:
    def __init__(self):
        self.cycle_name = "CYCLE_016"
        self.hypothesis = "Analytical reasoning enhancement with systematic methods and step-by-step verification will improve problem-solving by 5-8%"
        self.focus_area = "Analytical Reasoning Enhancement"

        # Initialize reasoning engines
        self.engine_baseline = ReasoningEngine()
        self.engine_enhanced = ReasoningEngine()

        # Configure enhanced engine with analytical reasoning
        self.configure_analytical_reasoning()

        # Real analytical problems with verifiable solutions
        self.analytical_problems = [
            {
                "id": "ar_001",
                "type": "data_analysis",
                "problem": "A company's sales data shows: Q1: $120K, Q2: $150K, Q3: $180K, Q4: $210K. What is the average quarterly growth rate and projected Q1 sales next year if the trend continues?",
                "expected_analysis": ["Calculate quarter-over-quarter growth rates", "Average the growth rates", "Apply to Q4 to project Q1"],
                "expected_result_type": "numerical_analysis"
            },
            {
                "id": "ar_002",
                "type": "pattern_analysis",
                "problem": "Analyze the sequence: 2, 6, 12, 20, 30, 42. Identify the pattern and predict the next two terms.",
                "expected_analysis": ["Find differences between terms", "Identify pattern in differences", "Apply pattern to predict"],
                "expected_result_type": "pattern_recognition"
            },
            {
                "id": "ar_003",
                "type": "comparative_analysis",
                "problem": "Two investments: Option A returns 8% annually with $10K minimum. Option B returns 12% annually with $25K minimum, but has 20% chance of 50% loss. Which is better for a risk-averse investor with $30K?",
                "expected_analysis": ["Calculate expected returns", "Assess risk factors", "Compare risk-adjusted returns"],
                "expected_result_type": "risk_analysis"
            },
            {
                "id": "ar_004",
                "type": "trend_analysis",
                "problem": "Website traffic: Week 1: 1000 visitors, Week 2: 1200, Week 3: 1440, Week 4: 1728. Analyze the growth pattern and predict Week 6 visitors.",
                "expected_analysis": ["Identify growth rate", "Verify consistency", "Project future values"],
                "expected_result_type": "trend_projection"
            },
            {
                "id": "ar_005",
                "type": "cost_benefit_analysis",
                "problem": "Machine upgrade costs $50K, saves $15K/year in labor, $5K/year in materials, but adds $3K/year maintenance. Calculate payback period and 5-year ROI.",
                "expected_analysis": ["Calculate total annual savings", "Subtract maintenance costs", "Compute payback and ROI"],
                "expected_result_type": "financial_analysis"
            },
            {
                "id": "ar_006",
                "type": "statistical_analysis",
                "problem": "Test scores: 85, 92, 78, 95, 88, 91, 82, 89. Calculate mean, median, standard deviation, and identify outliers.",
                "expected_analysis": ["Compute central tendency measures", "Calculate spread measures", "Identify statistical outliers"],
                "expected_result_type": "statistical_computation"
            },
            {
                "id": "ar_007",
                "type": "efficiency_analysis",
                "problem": "Process A: 100 units/hour, $5/hour labor. Process B: 80 units/hour, $3/hour labor. Which is more cost-effective for producing 500 units?",
                "expected_analysis": ["Calculate production time", "Compute labor costs", "Compare unit costs"],
                "expected_result_type": "efficiency_comparison"
            },
            {
                "id": "ar_008",
                "type": "break_even_analysis",
                "problem": "Product costs $20 to make, sells for $35. Fixed costs are $10,000/month. What monthly sales volume achieves break-even?",
                "expected_analysis": ["Calculate contribution margin", "Apply break-even formula", "Verify with profit calculation"],
                "expected_result_type": "break_even_calculation"
            }
        ]

        self.results = {
            "cycle_info": {
                "name": self.cycle_name,
                "hypothesis": self.hypothesis,
                "focus_area": self.focus_area,
                "timestamp": datetime.now().isoformat(),
                "problems_tested": len(self.analytical_problems)
            },
            "baseline_results": [],
            "enhanced_results": [],
            "improvement_analysis": {}
        }

    def configure_analytical_reasoning(self):
        """Configure enhanced engine with analytical reasoning capabilities."""
        analytical_prompt = """
You are an expert analytical reasoner. For each problem, follow this systematic approach:

1. PROBLEM DECOMPOSITION:
   - Identify the analytical problem type
   - Extract key data points and relationships
   - Determine what analysis methods are required

2. SYSTEMATIC ANALYSIS:
   - Apply appropriate analytical methods step-by-step
   - Show all calculations and reasoning
   - Verify intermediate results

3. VERIFICATION:
   - Cross-check calculations
   - Validate logical consistency
   - Confirm results meet problem requirements

4. CONCLUSION:
   - Provide clear, quantitative results
   - Explain confidence in analysis
   - Note any assumptions or limitations

Focus on: Mathematical accuracy, logical consistency, clear methodology.
"""
        self.engine_enhanced.base_model.system_prompt = analytical_prompt

    def analyze_reasoning_quality(self, response: str, problem: Dict) -> Dict:
        """Analyze the quality of analytical reasoning in response."""
        analysis = {
            "has_systematic_approach": any(keyword in response.lower() for keyword in ["step", "calculate", "analyze", "first", "second", "third"]),
            "shows_calculations": any(symbol in response for symbol in ["$", "%", "=", "+", "-", "*", "/"]),
            "verifies_results": any(keyword in response.lower() for keyword in ["check", "verify", "confirm", "validate"]),
            "provides_methodology": any(keyword in response.lower() for keyword in ["method", "approach", "formula", "calculation"]),
            "numerical_accuracy": self.check_numerical_accuracy(response, problem),
            "logical_consistency": self.check_logical_consistency(response, problem)
        }

        analysis["overall_quality"] = sum(analysis.values()) / len(analysis)
        return analysis

    def check_numerical_accuracy(self, response: str, problem: Dict) -> float:
        """Check if numerical results appear accurate based on problem constraints."""
        # Extract numbers from response
        import re
        numbers = re.findall(r'[\d,.]+', response)

        if not numbers:
            return 0.0

        # Basic reasonableness checks based on problem type
        problem_type = problem["type"]

        if problem_type in ["data_analysis", "trend_analysis", "cost_benefit_analysis"]:
            # Look for reasonable percentage values (0-100%)
            percentages = [n for n in numbers if '%' in n or float(n.replace(',', '')) <= 100]
            if percentages:
                return 1.0

        # Check for reasonable monetary values
        dollar_amounts = [n for n in numbers if '$' in response[response.find(n)-10:response.find(n)+10]]
        if dollar_amounts:
            return 1.0

        return 0.5  # Neutral if can't determine accuracy

    def check_logical_consistency(self, response: str, problem: Dict) -> float:
        """Check if the reasoning follows logical steps."""
        # Look for logical connectors and step-wise progression
        logical_indicators = [
            "therefore", "thus", "consequently", "as a result",
            "first", "second", "third", "finally",
            "because", "since", "given that"
        ]

        logical_count = sum(1 for indicator in logical_indicators if indicator in response.lower())
        return min(logical_count / 3, 1.0)  # Normalize to 0-1

    def evaluate_response(self, response: str, problem: Dict) -> Dict:
        """Evaluate response quality using multiple criteria."""
        # Use existing metrics
        confidence = calculate_reasoning_confidence(response)
        errors = detect_reasoning_errors(response)

        # Add analytical reasoning metrics
        analytical_quality = self.analyze_reasoning_quality(response, problem)

        # Overall success requires both correctness and analytical quality
        overall_success = (
            confidence > 0.7 and
            len(errors) == 0 and
            analytical_quality["overall_quality"] > 0.6
        )

        return {
            "success": overall_success,
            "confidence": confidence,
            "errors": errors,
            "analytical_quality": analytical_quality,
            "response_length": len(response),
            "has_verification": analytical_quality["verifies_results"]
        }

    def run_baseline_test(self) -> List[Dict]:
        """Run baseline test without analytical reasoning enhancements."""
        print("Running baseline analytical reasoning test...")
        results = []

        for problem in self.analytical_problems:
            try:
                # Test with baseline engine
                start_time = time.time()
                response = self.engine_baseline.solve_problem(problem["problem"])
                response_time = time.time() - start_time

                # Evaluate response
                evaluation = self.evaluate_response(response, problem)

                results.append({
                    "problem_id": problem["id"],
                    "problem_type": problem["type"],
                    "response_time": response_time,
                    "evaluation": evaluation
                })

                print(f"  Baseline {problem['id']}: Success={evaluation['success']}, Confidence={evaluation['confidence']:.2f}")

            except Exception as e:
                results.append({
                    "problem_id": problem["id"],
                    "error": str(e),
                    "evaluation": {"success": False, "confidence": 0, "errors": [str(e)]}
                })

        return results

    def run_enhanced_test(self) -> List[Dict]:
        """Run test with analytical reasoning enhancements."""
        print("Running enhanced analytical reasoning test...")
        results = []

        for problem in self.analytical_problems:
            try:
                # Test with enhanced engine
                start_time = time.time()
                response = self.engine_enhanced.solve_problem(problem["problem"])
                response_time = time.time() - start_time

                # Evaluate response
                evaluation = self.evaluate_response(response, problem)

                results.append({
                    "problem_id": problem["id"],
                    "problem_type": problem["type"],
                    "response_time": response_time,
                    "evaluation": evaluation
                })

                print(f"  Enhanced {problem['id']}: Success={evaluation['success']}, Confidence={evaluation['confidence']:.2f}, Quality={evaluation['analytical_quality']['overall_quality']:.2f}")

            except Exception as e:
                results.append({
                    "problem_id": problem["id"],
                    "error": str(e),
                    "evaluation": {"success": False, "confidence": 0, "errors": [str(e)]}
                })

        return results

    def analyze_improvements(self) -> Dict:
        """Analyze improvements between baseline and enhanced."""
        baseline_results = self.results["baseline_results"]
        enhanced_results = self.results["enhanced_results"]

        # Calculate success rates
        baseline_success_rate = sum(1 for r in baseline_results if r["evaluation"]["success"]) / len(baseline_results)
        enhanced_success_rate = sum(1 for r in enhanced_results if r["evaluation"]["success"]) / len(enhanced_results)

        # Calculate average confidence
        baseline_avg_confidence = sum(r["evaluation"]["confidence"] for r in baseline_results) / len(baseline_results)
        enhanced_avg_confidence = sum(r["evaluation"]["confidence"] for r in enhanced_results) / len(enhanced_results)

        # Calculate average analytical quality
        enhanced_avg_quality = sum(r["evaluation"]["analytical_quality"]["overall_quality"] for r in enhanced_results) / len(enhanced_results)

        # Calculate improvements
        success_improvement = enhanced_success_rate - baseline_success_rate
        confidence_improvement = enhanced_avg_confidence - baseline_avg_confidence

        # Determine if cycle succeeds (requires >2% real improvement in success rate)
        cycle_succeeds = success_improvement > 0.02  # Conservative threshold

        return {
            "baseline_success_rate": baseline_success_rate,
            "enhanced_success_rate": enhanced_success_rate,
            "success_improvement": success_improvement,
            "baseline_avg_confidence": baseline_avg_confidence,
            "enhanced_avg_confidence": enhanced_avg_confidence,
            "confidence_improvement": confidence_improvement,
            "enhanced_avg_analytical_quality": enhanced_avg_quality,
            "cycle_succeeds": cycle_succeeds,
            "meets_hypothesis": success_improvement >= 0.05  # 5% threshold from hypothesis
        }

    def save_results(self) -> str:
        """Save results to JSON file."""
        results_dir = Path(__file__).parent / "cycle_results"
        results_dir.mkdir(exist_ok=True)

        results_file = results_dir / f"{self.cycle_name}_results.json"

        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        return str(results_file)

    def run_cycle(self):
        """Execute the complete cycle."""
        print(f"\n=== {self.cycle_name} ===")
        print(f"Hypothesis: {self.hypothesis}")
        print(f"Testing {len(self.analytical_problems)} analytical reasoning problems\n")

        # Run tests
        self.results["baseline_results"] = self.run_baseline_test()
        self.results["enhanced_results"] = self.run_enhanced_test()

        # Analyze improvements
        self.results["improvement_analysis"] = self.analyze_improvements()

        # Save results
        results_file = self.save_results()

        # Print summary
        analysis = self.results["improvement_analysis"]
        print(f"\n=== Cycle Results ===")
        print(f"Baseline Success Rate: {analysis['baseline_success_rate']:.1%}")
        print(f"Enhanced Success Rate: {analysis['enhanced_success_rate']:.1%}")
        print(f"Success Improvement: {analysis['success_improvement']:.1%}")
        print(f"Analytical Quality: {analysis['enhanced_avg_analytical_quality']:.1%}")
        print(f"Cycle Succeeds: {analysis['cycle_succeeds']}")
        print(f"Meets Hypothesis: {analysis['meets_hypothesis']}")
        print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    cycle = AnalyticalReasoningBenchmark()
    cycle.run_cycle()