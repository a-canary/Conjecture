#!/usr/bin/env python3
"""
CYCLE_018: Complex Problem Decomposition

Extending Cycle 12's problem decomposition success, this cycle handles
multi-variable, multi-constraint problems with advanced structural
decomposition techniques.

Hypothesis: Enhanced complex problem decomposition with systematic
breakdown will improve multi-constraint problem-solving by 7-12%.

Focus: Complex problems with multiple variables and constraints.
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

class ComplexDecompositionBenchmark:
    def __init__(self):
        self.cycle_name = "CYCLE_018"
        self.hypothesis = "Enhanced complex problem decomposition will improve multi-constraint problem-solving by 7-12%"
        self.focus_area = "Complex Problem Decomposition"

        # Initialize reasoning engines
        self.engine_baseline = ReasoningEngine()
        self.engine_enhanced = ReasoningEngine()

        # Configure enhanced engine with complex decomposition
        self.configure_complex_decomposition()

        # Real complex problems with multiple variables and constraints
        self.complex_problems = [
            {
                "id": "cd_001",
                "type": "resource_allocation",
                "problem": "A factory produces 3 products with limited resources: 1000 labor hours, 500kg material, $50,000 budget. Product A: 2hrs, 1kg, $20, profit $15. Product B: 3hrs, 2kg, $35, profit $25. Product C: 4hrs, 1kg, $40, profit $30. Maximize profit.",
                "variables": ["product_A_quantity", "product_B_quantity", "product_C_quantity"],
                "constraints": ["labor_hours", "material_kg", "budget"],
                "objective": "maximize_profit"
            },
            {
                "id": "cd_002",
                "type": "scheduling",
                "problem": "Schedule 5 tasks with durations: T1=3hrs, T2=2hrs, T3=4hrs, T4=1hr, T5=3hrs. Constraints: T1 before T3, T2 after T4, max 4 tasks simultaneously, 8-hour workday. Minimize total completion time.",
                "variables": ["task_start_times", "resource_assignments"],
                "constraints": ["precedence", "resource_limit", "time_window"],
                "objective": "minimize_makespan"
            },
            {
                "id": "cd_003",
                "type": "portfolio_optimization",
                "problem": "Invest $100,000 across 4 assets with expected returns: A=8%, B=12%, C=15%, D=6%. Risk levels: A=low, B=medium, C=high, D=very low. Constraints: max 30% in any asset, min 10% in low-risk, total risk ≤ medium. Maximize expected return.",
                "variables": ["asset_allocations"],
                "constraints": ["budget", "diversification", "risk_limit"],
                "objective": "maximize_return"
            },
            {
                "id": "cd_004",
                "type": "supply_chain",
                "problem": "Company needs to supply 3 stores from 2 warehouses. Store demands: S1=100, S2=150, S3=200 units. Warehouse supplies: W1=250, W2=200 units. Shipping costs: W1→S1=$5, W1→S2=$7, W1→S3=$9, W2→S1=$6, W2→S2=$4, W2→S3=$8. Minimize total shipping cost.",
                "variables": ["shipment_quantities"],
                "constraints": ["supply", "demand", "non_negative"],
                "objective": "minimize_cost"
            },
            {
                "id": "cd_005",
                "type": "project_planning",
                "problem": "Construction project with 6 activities and dependencies: A(5)→C(3), B(4)→D(6), C(2)→E(4), D(3)→F(5). Each activity needs workers: A=3, B=2, C=4, D=3, E=2, F=4. Max 8 workers available. Find minimum project duration.",
                "variables": ["activity_start_times"],
                "constraints": ["dependencies", "resource_limit"],
                "objective": "minimize_duration"
            },
            {
                "id": "cd_006",
                "type": "production_planning",
                "problem": "Bakery produces 3 products using ovens and mixers. Available: 10 oven-hours/day, 15 mixer-hours/day. Product requirements: Bread(1 oven, 0.5 mixer, $2 profit), Cakes(2 oven, 1 mixer, $5 profit), Pastries(0.5 oven, 1.5 mixer, $3 profit). Maximize daily profit.",
                "variables": ["production_quantities"],
                "constraints": ["oven_capacity", "mixer_capacity"],
                "objective": "maximize_profit"
            },
            {
                "id": "cd_007",
                "type": "facility_location",
                "problem": "Locate 2 warehouses to serve 4 cities. City coordinates: A(0,0), B(10,0), C(0,10), D(10,10). Demand: A=100, B=150, C=120, D=180. Transportation cost = $0.50 × distance × units. Minimize total transportation cost.",
                "variables": ["warehouse_locations", "city_assignments"],
                "constraints": ["facility_count", "coverage"],
                "objective": "minimize_cost"
            },
            {
                "id": "cd_008",
                "type": "investment_planning",
                "problem": "Retirement planning over 30 years. Annual income: $60,000, expenses: $40,000, inflation: 3%, expected return: 7%. Goal: $1.5M retirement fund. Calculate required annual savings rate and verify feasibility.",
                "variables": ["savings_rate", "investment_allocation"],
                "constraints": ["income_limit", "retirement_goal", "inflation_impact"],
                "objective": "meet_goal"
            }
        ]

        self.results = {
            "cycle_info": {
                "name": self.cycle_name,
                "hypothesis": self.hypothesis,
                "focus_area": self.focus_area,
                "timestamp": datetime.now().isoformat(),
                "problems_tested": len(self.complex_problems)
            },
            "baseline_results": [],
            "enhanced_results": [],
            "improvement_analysis": {}
        }

    def configure_complex_decomposition(self):
        """Configure enhanced engine with complex decomposition capabilities."""
        decomposition_prompt = """
You are an expert at decomposing complex problems with multiple variables and constraints.

For each complex problem, follow this systematic decomposition approach:

1. PROBLEM STRUCTURE ANALYSIS:
   - Identify all variables and their types
   - List all constraints explicitly
   - Clarify the objective function
   - Identify interdependencies

2. DECOMPOSITION STRATEGY:
   - Break problem into manageable sub-problems
   - Identify primary vs. secondary constraints
   - Group related variables
   - Plan solution approach

3. STEP-BY-STEP SOLUTION:
   Sub-problem 1: [Name]
   - Variables involved
   - Constraints to consider
   - Solution method

   Sub-problem 2: [Name]
   - Variables involved
   - Constraints to consider
   - Solution method

   Continue until all sub-problems solved

4. INTEGRATION AND VERIFICATION:
   - Combine sub-problem solutions
   - Verify all constraints are satisfied
   - Check optimal solution quality
   - Validate against original problem

5. SENSITIVITY ANALYSIS:
   - Identify critical constraints
   - Test solution robustness
   - Consider alternative scenarios

Focus on: Systematic breakdown, constraint satisfaction, optimal integration.
"""
        self.engine_enhanced.base_model.system_prompt = decomposition_prompt

    def analyze_decomposition_quality(self, response: str, problem: Dict) -> Dict:
        """Analyze the quality of problem decomposition."""
        decomposition_indicators = [
            "sub-problem", "decompose", "break down", "separate",
            "first step", "second step", "third step",
            "constraint", "variable", "objective"
        ]

        structure_indicators = [
            "variables:", "constraints:", "objective:",
            "sub-problem 1", "sub-problem 2", "integration",
            "verify", "check", "satisfy"
        ]

        analysis = {
            "identifies_variables": "variables" in response.lower() or any(var in response.lower() for var in ["variable", "parameter", "unknown"]),
            "lists_constraints": "constraint" in response.lower() or any(word in response.lower() for word in ["limit", "restriction", "requirement"]),
            "clarifies_objective": "objective" in response.lower() or any(word in response.lower() for word in ["goal", "target", "optimize", "maximize", "minimize"]),
            "uses_decomposition": any(indicator in response.lower() for indicator in decomposition_indicators),
            "structured_approach": any(indicator in response.lower() for indicator in structure_indicators),
            "integrates_solution": any(word in response.lower() for word in ["combine", "integrate", "synthesize", "put together"]),
            "verifies_constraints": any(word in response.lower() for word in ["verify", "check", "satisfy", "meet"]),
            "handles_interdependencies": any(word in response.lower() for word in ["depend", "relate", "affect", "impact"])
        }

        # Check for problem-specific elements
        expected_variables = problem.get("variables", [])
        expected_constraints = problem.get("constraints", [])

        var_coverage = sum(1 for var in expected_variables if var.replace("_", " ") in response.lower()) / len(expected_variables) if expected_variables else 0
        constraint_coverage = sum(1 for constraint in expected_constraints if constraint.replace("_", " ") in response.lower()) / len(expected_constraints) if expected_constraints else 0

        analysis["variable_coverage"] = var_coverage
        analysis["constraint_coverage"] = constraint_coverage
        analysis["overall_decomposition_quality"] = sum(analysis.values()) / len(analysis)

        return analysis

    def evaluate_response(self, response: str, problem: Dict) -> Dict:
        """Evaluate response quality with emphasis on decomposition."""
        # Use existing metrics
        confidence = calculate_reasoning_confidence(response)
        errors = detect_reasoning_errors(response)

        # Add decomposition quality metrics
        decomposition_quality = self.analyze_decomposition_quality(response, problem)

        # Success requires correctness AND proper decomposition
        overall_success = (
            confidence > 0.7 and
            len(errors) == 0 and
            decomposition_quality["overall_decomposition_quality"] > 0.6
        )

        return {
            "success": overall_success,
            "confidence": confidence,
            "errors": errors,
            "decomposition_quality": decomposition_quality,
            "response_length": len(response),
            "complexity_handling": decomposition_quality["structured_approach"]
        }

    def run_baseline_test(self) -> List[Dict]:
        """Run baseline test without decomposition enhancements."""
        print("Running baseline complex decomposition test...")
        results = []

        for problem in self.complex_problems:
            try:
                start_time = time.time()
                response = self.engine_baseline.solve_problem(problem["problem"])
                response_time = time.time() - start_time

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
        """Run test with decomposition enhancements."""
        print("Running enhanced complex decomposition test...")
        results = []

        for problem in self.complex_problems:
            try:
                start_time = time.time()
                response = self.engine_enhanced.solve_problem(problem["problem"])
                response_time = time.time() - start_time

                evaluation = self.evaluate_response(response, problem)

                results.append({
                    "problem_id": problem["id"],
                    "problem_type": problem["type"],
                    "response_time": response_time,
                    "evaluation": evaluation
                })

                print(f"  Enhanced {problem['id']}: Success={evaluation['success']}, Decomposition={evaluation['decomposition_quality']['overall_decomposition_quality']:.2f}")

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

        # Calculate average decomposition quality
        enhanced_avg_decomposition = sum(r["evaluation"]["decomposition_quality"]["overall_decomposition_quality"] for r in enhanced_results) / len(enhanced_results)

        # Calculate complexity handling improvement
        baseline_complexity = sum(r["evaluation"]["decomposition_quality"]["structured_approach"] for r in baseline_results) / len(baseline_results)
        enhanced_complexity = sum(r["evaluation"]["decomposition_quality"]["structured_approach"] for r in enhanced_results) / len(enhanced_results)

        # Calculate improvements
        success_improvement = enhanced_success_rate - baseline_success_rate
        complexity_improvement = enhanced_complexity - baseline_complexity

        # Determine if cycle succeeds (requires >2% improvement in success rate)
        cycle_succeeds = success_improvement > 0.02

        return {
            "baseline_success_rate": baseline_success_rate,
            "enhanced_success_rate": enhanced_success_rate,
            "success_improvement": success_improvement,
            "baseline_complexity_handling": baseline_complexity,
            "enhanced_complexity_handling": enhanced_complexity,
            "complexity_improvement": complexity_improvement,
            "enhanced_avg_decomposition_quality": enhanced_avg_decomposition,
            "cycle_succeeds": cycle_succeeds,
            "meets_hypothesis": success_improvement >= 0.07  # 7% threshold from hypothesis
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
        print(f"Testing {len(self.complex_problems)} complex decomposition problems\n")

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
        print(f"Decomposition Quality: {analysis['enhanced_avg_decomposition_quality']:.1%}")
        print(f"Complexity Handling Improvement: {analysis['complexity_improvement']:.1%}")
        print(f"Cycle Succeeds: {analysis['cycle_succeeds']}")
        print(f"Meets Hypothesis: {analysis['meets_hypothesis']}")
        print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    cycle = ComplexDecompositionBenchmark()
    cycle.run_cycle()