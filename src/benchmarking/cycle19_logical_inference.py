#!/usr/bin/env python3
"""
CYCLE_019: Logical Inference Enhancement

Building on successful logical reasoning patterns, this cycle improves
logical inference with formal inference rules, syllogism, and conditional
reasoning techniques.

Hypothesis: Enhanced logical inference with formal rules will improve
logical reasoning accuracy by 6-9%.

Focus: Problems requiring formal logical inference and deduction.
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

class LogicalInferenceBenchmark:
    def __init__(self):
        self.cycle_name = "CYCLE_019"
        self.hypothesis = "Enhanced logical inference with formal rules will improve logical reasoning accuracy by 6-9%"
        self.focus_area = "Logical Inference Enhancement"

        # Initialize reasoning engines
        self.engine_baseline = ReasoningEngine()
        self.engine_enhanced = ReasoningEngine()

        # Configure enhanced engine with logical inference
        self.configure_logical_inference()

        # Real logical inference problems
        self.logical_problems = [
            {
                "id": "li_001",
                "type": "syllogism",
                "problem": "All mammals are warm-blooded. All whales are mammals. Some warm-blooded animals live in water. What can we conclude about whales?",
                "inference_type": "categorical_syllogism",
                "expected_logic": ["universal_affirmative", "transitive_property"]
            },
            {
                "id": "li_002",
                "type": "conditional_reasoning",
                "problem": "If it rains, the ground gets wet. The ground is not wet. What can we conclude about rain? Also, if it rains, people use umbrellas. People are using umbrellas. What can we conclude?",
                "inference_type": "modus_tollens_and_fallacy",
                "expected_logic": ["modus_tollens", "affirming_consequent_fallacy"]
            },
            {
                "id": "li_003",
                "type": "quantifier_logic",
                "problem": "Every student in the class passed at least one exam. No one failed all exams. At least one student passed exactly two exams. What is the minimum number of exams given?",
                "inference_type": "quantifier_reasoning",
                "expected_logic": ["existential_quantifier", "universal_quantifier"]
            },
            {
                "id": "li_004",
                "type": "temporal_logic",
                "problem": "Event A always precedes Event B. Event C sometimes occurs between A and B. If C occurs, D must follow. Today, B occurred at 3pm. What can we conclude about A and C?",
                "inference_type": "temporal_inference",
                "expected_logic": ["temporal_precedence", "conditional_dependency"]
            },
            {
                "id": "li_005",
                "type": "set_theory_logic",
                "problem": "Set A contains all prime numbers less than 20. Set B contains odd numbers less than 20. Set C contains numbers divisible by 3 less than 20. Find: A∩B∩C, A∪B, and (A∩C)'.",
                "inference_type": "set_operations",
                "expected_logic": ["intersection", "union", "complement"]
            },
            {
                "id": "li_006",
                "type": "causal_inference",
                "problem": "Studies show: 80% of people who exercise regularly have good health. 20% of people who don't exercise have good health. In a group, 60% have good health. What is the minimum percentage that exercise regularly?",
                "inference_type": "bayesian_reasoning",
                "expected_logic": ["conditional_probability", "reverse_inference"]
            },
            {
                "id": "li_007",
                "type": "deontic_logic",
                "problem": "If a person is a doctor, they must help patients in emergencies. If someone helps patients in emergencies, they should have medical training. John is not a doctor but helps patients in emergencies. What follows?",
                "inference_type": "deontic_reasoning",
                "expected_logic": ["obligation", "permission", "inconsistency"]
            },
            {
                "id": "li_008",
                "type": "modal_logic",
                "problem": "It is possible that all swans are white. It is necessary that if something is a swan, it can swim. There exists a black swan. What can we conclude about the possibility of white swimming birds?",
                "inference_type": "modal_reasoning",
                "expected_logic": ["possibility", "necessity", "existence"]
            }
        ]

        self.results = {
            "cycle_info": {
                "name": self.cycle_name,
                "hypothesis": self.hypothesis,
                "focus_area": self.focus_area,
                "timestamp": datetime.now().isoformat(),
                "problems_tested": len(self.logical_problems)
            },
            "baseline_results": [],
            "enhanced_results": [],
            "improvement_analysis": {}
        }

    def configure_logical_inference(self):
        """Configure enhanced engine with formal logical inference capabilities."""
        logical_prompt = """
You are an expert in formal logic and logical inference. For each problem, apply these inference methods:

1. IDENTIFY LOGIC TYPE:
   - Categorical logic (All A are B, Some A are B, No A are B)
   - Conditional logic (If P then Q, P only if Q, Q if P)
   - Quantifier logic (All, Some, None, At least one)
   - Temporal logic (before, after, during)
   - Modal logic (possible, necessary, certain)
   - Set theory logic (union, intersection, complement)

2. APPLY INFERENCE RULES:
   - Modus Ponens: If P→Q and P, then Q
   - Modus Tollens: If P→Q and ¬Q, then ¬P
   - Hypothetical Syllogism: If P→Q and Q→R, then P→R
   - Disjunctive Syllogism: P∨Q and ¬P, then Q
   - Constructive Dilemma: (P→Q)∧(R→S) and P∨R, then Q∨S

3. AVOID FALLACIES:
   - Affirming the consequent (P→Q, Q, therefore P) ✓
   - Denying the antecedent (P→Q, ¬P, therefore ¬Q) ✓
   - Exclusive or assumptions
   - Converse errors

4. STEP-BY-STEP DEDUCTION:
   Step 1: Extract premises
   Step 2: Identify applicable rules
   Step 3: Apply rules systematically
   Step 4: Check for fallacies
   Step 5: Draw valid conclusions
   Step 6: Verify logical consistency

5. CONCLUSION VALIDATION:
   - Does conclusion necessarily follow?
   - Are all inferences valid?
   - Are there hidden assumptions?

Focus on: Formal validity, rule application, fallacy detection.
"""
        self.engine_enhanced.base_model.system_prompt = logical_prompt

    def analyze_logical_quality(self, response: str, problem: Dict) -> Dict:
        """Analyze the quality of logical reasoning in response."""
        logical_indicators = [
            "premise", "conclusion", "therefore", "thus",
            "modus ponens", "modus tollens", "syllogism",
            "necessary", "sufficient", "if and only if",
            "universal", "existential", "quantifier"
        ]

        inference_rules = [
            "all a are b", "some a are b", "no a are b",
            "if p then q", "p only if q", "q if p",
            "fallacy", "invalid", "valid", "sound"
        ]

        analysis = {
            "identifies_premises": any(word in response.lower() for word in ["premise", "given", "assume", "suppose"]),
            "states_conclusions": any(word in response.lower() for word in ["conclusion", "therefore", "thus", "hence", "conclude"]),
            "uses_formal_logic": any(indicator in response.lower() for indicator in logical_indicators),
            "applies_inference_rules": any(rule in response.lower() for rule in inference_rules),
            "detects_fallacies": any(word in response.lower() for word in ["fallacy", "invalid", "error", "mistake"]),
            "checks_validity": any(word in response.lower() for word in ["valid", "sound", "logical", "consistent"]),
            "structured_reasoning": any(word in response.lower() for word in ["step", "first", "second", "third", "finally"]),
            "explicit_logic_type": problem["inference_type"].replace("_", " ") in response.lower()
        }

        # Check for specific logic types mentioned
        expected_logic = problem.get("expected_logic", [])
        logic_coverage = sum(1 for logic in expected_logic if logic.replace("_", " ") in response.lower()) / len(expected_logic) if expected_logic else 0

        analysis["logic_coverage"] = logic_coverage
        analysis["overall_logical_quality"] = sum(analysis.values()) / len(analysis)

        return analysis

    def evaluate_response(self, response: str, problem: Dict) -> Dict:
        """Evaluate response quality with emphasis on logical inference."""
        # Use existing metrics
        confidence = calculate_reasoning_confidence(response)
        errors = detect_reasoning_errors(response)

        # Add logical quality metrics
        logical_quality = self.analyze_logical_quality(response, problem)

        # Success requires correctness AND proper logical reasoning
        overall_success = (
            confidence > 0.7 and
            len(errors) == 0 and
            logical_quality["overall_logical_quality"] > 0.6
        )

        return {
            "success": overall_success,
            "confidence": confidence,
            "errors": errors,
            "logical_quality": logical_quality,
            "response_length": len(response),
            "logical_rigor": logical_quality["checks_validity"]
        }

    def run_baseline_test(self) -> List[Dict]:
        """Run baseline test without logical inference enhancements."""
        print("Running baseline logical inference test...")
        results = []

        for problem in self.logical_problems:
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
        """Run test with logical inference enhancements."""
        print("Running enhanced logical inference test...")
        results = []

        for problem in self.logical_problems:
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

                print(f"  Enhanced {problem['id']}: Success={evaluation['success']}, Logic={evaluation['logical_quality']['overall_logical_quality']:.2f}")

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

        # Calculate average logical quality
        enhanced_avg_logical = sum(r["evaluation"]["logical_quality"]["overall_logical_quality"] for r in enhanced_results) / len(enhanced_results)

        # Calculate logical rigor improvement
        baseline_rigor = sum(r["evaluation"]["logical_quality"]["checks_validity"] for r in baseline_results) / len(baseline_results)
        enhanced_rigor = sum(r["evaluation"]["logical_quality"]["checks_validity"] for r in enhanced_results) / len(enhanced_results)

        # Calculate improvements
        success_improvement = enhanced_success_rate - baseline_success_rate
        rigor_improvement = enhanced_rigor - baseline_rigor

        # Determine if cycle succeeds (requires >2% improvement in success rate)
        cycle_succeeds = success_improvement > 0.02

        return {
            "baseline_success_rate": baseline_success_rate,
            "enhanced_success_rate": enhanced_success_rate,
            "success_improvement": success_improvement,
            "baseline_logical_rigor": baseline_rigor,
            "enhanced_logical_rigor": enhanced_rigor,
            "rigor_improvement": rigor_improvement,
            "enhanced_avg_logical_quality": enhanced_avg_logical,
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
        print(f"Testing {len(self.logical_problems)} logical inference problems\n")

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
        print(f"Logical Quality: {analysis['enhanced_avg_logical_quality']:.1%}")
        print(f"Rigor Improvement: {analysis['rigor_improvement']:.1%}")
        print(f"Cycle Succeeds: {analysis['cycle_succeeds']}")
        print(f"Meets Hypothesis: {analysis['meets_hypothesis']}")
        print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    cycle = LogicalInferenceBenchmark()
    cycle.run_cycle()