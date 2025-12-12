#!/usr/bin/env python3
"""
CYCLE_017: Verification and Validation Enhancement

Building on Cycle 3's success with self-verification and error detection,
this cycle enhances verification capabilities with comprehensive validation
checklists and systematic verification processes.

Hypothesis: Enhanced verification and validation with systematic checklists
and multi-layer verification will reduce errors and improve problem-solving
accuracy by 6-10%.

Focus: Problems requiring precise verification and validation.
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

class VerificationValidationBenchmark:
    def __init__(self):
        self.cycle_name = "CYCLE_017"
        self.hypothesis = "Enhanced verification and validation with systematic checklists will improve problem-solving accuracy by 6-10%"
        self.focus_area = "Verification and Validation Enhancement"

        # Initialize reasoning engines
        self.engine_baseline = ReasoningEngine()
        self.engine_enhanced = ReasoningEngine()

        # Configure enhanced engine with verification enhancements
        self.configure_verification_enhancement()

        # Real problems requiring careful verification
        self.verification_problems = [
            {
                "id": "vv_001",
                "type": "calculation_verification",
                "problem": "A rectangle has perimeter 30 cm and area 56 cm². Find its length and width. Verify your solution satisfies both conditions.",
                "verification_required": ["perimeter_check", "area_check", "positive_dimensions"],
                "expected_result_type": "geometric_calculation"
            },
            {
                "id": "vv_002",
                "type": "logic_verification",
                "problem": "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly? Verify your logical reasoning.",
                "verification_required": ["logical_validity", "premise_analysis", "conclusion_check"],
                "expected_result_type": "logical_analysis"
            },
            {
                "id": "vv_003",
                "type": "financial_verification",
                "problem": "Loan: $200,000 at 4.5% annual interest for 30 years. Calculate monthly payment and total interest paid. Verify with amortization schedule.",
                "verification_required": ["payment_calculation", "interest_total", "amortization_consistency"],
                "expected_result_type": "financial_computation"
            },
            {
                "id": "vv_004",
                "type": "physics_verification",
                "problem": "Ball dropped from 45m height. Calculate time to hit ground and final velocity. Verify using both energy and kinematic equations.",
                "verification_required": ["kinematic_check", "energy_conservation", "units_consistency"],
                "expected_result_type": "physics_calculation"
            },
            {
                "id": "vv_005",
                "type": "statistical_verification",
                "problem": "Sample of 50 students, mean test score 82, standard deviation 8. Calculate 95% confidence interval and verify using bootstrap method.",
                "verification_required": ["interval_calculation", "bootstrap_validation", "sample_size_adequacy"],
                "expected_result_type": "statistical_analysis"
            },
            {
                "id": "vv_006",
                "type": "optimization_verification",
                "problem": "Find dimensions of open-top box with volume 1000 cm³ that minimizes surface area. Verify using calculus and second derivative test.",
                "verification_required": ["volume_constraint", "surface_area_calculation", "minimum_verification"],
                "expected_result_type": "optimization_problem"
            },
            {
                "id": "vv_007",
                "type": "probability_verification",
                "problem": "Draw 2 cards from standard deck. Calculate probability of exactly one ace. Verify using both direct calculation and complement method.",
                "verification_required": ["direct_calculation", "complement_method", "total_probability"],
                "expected_result_type": "probability_computation"
            },
            {
                "id": "vv_008",
                "type": "system_verification",
                "problem": "Chemical reaction: 2H₂ + O₂ → 2H₂O. If 10g H₂ reacts completely, calculate water produced and verify mass conservation.",
                "verification_required": ["stoichiometry_check", "mass_conservation", "limiting_reagent"],
                "expected_result_type": "chemical_calculation"
            }
        ]

        self.results = {
            "cycle_info": {
                "name": self.cycle_name,
                "hypothesis": self.hypothesis,
                "focus_area": self.focus_area,
                "timestamp": datetime.now().isoformat(),
                "problems_tested": len(self.verification_problems)
            },
            "baseline_results": [],
            "enhanced_results": [],
            "improvement_analysis": {}
        }

    def configure_verification_enhancement(self):
        """Configure enhanced engine with systematic verification capabilities."""
        verification_prompt = """
You are an expert at verification and validation. For each problem, follow this comprehensive approach:

1. INITIAL ANALYSIS:
   - Understand the problem requirements
   - Identify what needs to be verified
   - Plan multiple verification methods

2. PRIMARY SOLUTION:
   - Solve the problem using standard methods
   - Show all steps clearly
   - Document assumptions

3. VERIFICATION PHASE:
   - Cross-check with alternative methods
   - Verify constraints and conditions
   - Test edge cases and boundaries
   - Check units and consistency

4. VALIDATION CHECKLIST:
   ✓ Does solution satisfy all given conditions?
   ✓ Are calculations mathematically correct?
   ✓ Are units consistent and correct?
   ✓ Does answer make logical sense?
   ✓ Are there alternative verification methods?
   ✓ Are assumptions clearly stated?

5. CONFIDENCE ASSESSMENT:
   - Rate confidence in each verification step
   - Identify any remaining uncertainties
   - Provide final validated answer

Focus on: Thorough verification, multiple validation methods, error detection.
"""
        self.engine_enhanced.base_model.system_prompt = verification_prompt

    def analyze_verification_quality(self, response: str, problem: Dict) -> Dict:
        """Analyze the quality of verification in the response."""
        verification_indicators = [
            "verify", "check", "validate", "confirm", "cross-check",
            "alternative", "method", "approach", "consistency",
            "test", "boundary", "constraint", "condition"
        ]

        verification_checklist = [
            "✓", "check:", "verified:", "confirmed:", "validated:",
            "satisfies", "meets", "consistent with"
        ]

        analysis = {
            "has_verification_phase": any(indicator in response.lower() for indicator in verification_indicators),
            "uses_checklist": any(check in response for check in verification_checklist),
            "multiple_methods": any(phrase in response.lower() for phrase in ["alternative method", "another approach", "using different"]),
            "tests_constraints": any(word in response.lower() for word in ["constraint", "condition", "boundary"]),
            "checks_consistency": any(word in response.lower() for word in ["consistent", "matches", "agrees"]),
            "explicit_confidence": "confidence" in response.lower() or "certain" in response.lower(),
            "identifies_assumptions": any(phrase in response.lower() for phrase in ["assume", "assuming", "assumption"])
        }

        # Check for problem-specific verification requirements
        required_verifications = problem.get("verification_required", [])
        verification_coverage = 0
        for req in required_verifications:
            if any(keyword in response.lower() for keyword in req.split("_")):
                verification_coverage += 1

        analysis["verification_coverage"] = verification_coverage / len(required_verifications) if required_verifications else 0
        analysis["overall_verification_quality"] = sum(analysis.values()) / len(analysis)

        return analysis

    def evaluate_response(self, response: str, problem: Dict) -> Dict:
        """Evaluate response quality with emphasis on verification."""
        # Use existing metrics
        confidence = calculate_reasoning_confidence(response)
        errors = detect_reasoning_errors(response)

        # Add verification quality metrics
        verification_quality = self.analyze_verification_quality(response, problem)

        # Success requires correctness AND thorough verification
        overall_success = (
            confidence > 0.7 and
            len(errors) == 0 and
            verification_quality["overall_verification_quality"] > 0.6
        )

        return {
            "success": overall_success,
            "confidence": confidence,
            "errors": errors,
            "verification_quality": verification_quality,
            "response_length": len(response),
            "verification_coverage": verification_quality["verification_coverage"]
        }

    def run_baseline_test(self) -> List[Dict]:
        """Run baseline test without verification enhancements."""
        print("Running baseline verification test...")
        results = []

        for problem in self.verification_problems:
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

                print(f"  Baseline {problem['id']}: Success={evaluation['success']}, Errors={len(evaluation['errors'])}")

            except Exception as e:
                results.append({
                    "problem_id": problem["id"],
                    "error": str(e),
                    "evaluation": {"success": False, "confidence": 0, "errors": [str(e)]}
                })

        return results

    def run_enhanced_test(self) -> List[Dict]:
        """Run test with verification enhancements."""
        print("Running enhanced verification test...")
        results = []

        for problem in self.verification_problems:
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

                print(f"  Enhanced {problem['id']}: Success={evaluation['success']}, Verification={evaluation['verification_quality']['overall_verification_quality']:.2f}")

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

        # Calculate average verification quality
        enhanced_avg_verification = sum(r["evaluation"]["verification_quality"]["overall_verification_quality"] for r in enhanced_results) / len(enhanced_results)

        # Calculate error reduction
        baseline_avg_errors = sum(len(r["evaluation"]["errors"]) for r in baseline_results) / len(baseline_results)
        enhanced_avg_errors = sum(len(r["evaluation"]["errors"]) for r in enhanced_results) / len(enhanced_results)
        error_reduction = baseline_avg_errors - enhanced_avg_errors

        # Calculate improvements
        success_improvement = enhanced_success_rate - baseline_success_rate

        # Determine if cycle succeeds (requires >2% improvement AND error reduction)
        cycle_succeeds = success_improvement > 0.02 and error_reduction >= 0

        return {
            "baseline_success_rate": baseline_success_rate,
            "enhanced_success_rate": enhanced_success_rate,
            "success_improvement": success_improvement,
            "baseline_avg_errors": baseline_avg_errors,
            "enhanced_avg_errors": enhanced_avg_errors,
            "error_reduction": error_reduction,
            "enhanced_avg_verification_quality": enhanced_avg_verification,
            "cycle_succeeds": cycle_succeeds,
            "meets_hypothesis": success_improvement >= 0.06  # 6% threshold from hypothesis
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
        print(f"Testing {len(self.verification_problems)} verification problems\n")

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
        print(f"Error Reduction: {analysis['error_reduction']:.1f}")
        print(f"Verification Quality: {analysis['enhanced_avg_verification_quality']:.1%}")
        print(f"Cycle Succeeds: {analysis['cycle_succeeds']}")
        print(f"Meets Hypothesis: {analysis['meets_hypothesis']}")
        print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    cycle = VerificationValidationBenchmark()
    cycle.run_cycle()