#!/usr/bin/env python3
"""
CYCLE_020: Strategic Planning Enhancement

Building on multi-step reasoning success, this cycle adds a planning
phase before execution with "plan-do-review" pattern for improved
strategic problem-solving.

Hypothesis: Strategic planning enhancement with plan-do-review pattern
will improve multi-step problem-solving by 8-12%.

Focus: Complex problems requiring strategic planning and execution.
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

class StrategicPlanningBenchmark:
    def __init__(self):
        self.cycle_name = "CYCLE_020"
        self.hypothesis = "Strategic planning enhancement with plan-do-review pattern will improve multi-step problem-solving by 8-12%"
        self.focus_area = "Strategic Planning Enhancement"

        # Initialize reasoning engines
        self.engine_baseline = ReasoningEngine()
        self.engine_enhanced = ReasoningEngine()

        # Configure enhanced engine with strategic planning
        self.configure_strategic_planning()

        # Real strategic planning problems
        self.strategic_problems = [
            {
                "id": "sp_001",
                "type": "business_strategy",
                "problem": "A retail company faces 20% online competition growth, declining foot traffic by 15%, and rising costs. Develop a 3-year strategic plan to maintain profitability and market position.",
                "planning_complexity": "high",
                "time_horizon": "3_years",
                "constraints": ["budget", "resources", "market_trends"]
            },
            {
                "id": "sp_002",
                "type": "project_management",
                "problem": "Launch a mobile app in 6 months with $50K budget. Must include user authentication, data synchronization, and analytics. Team of 5 developers with varying skills. Create strategic project plan.",
                "planning_complexity": "medium",
                "time_horizon": "6_months",
                "constraints": ["timeline", "budget", "team_skills"]
            },
            {
                "id": "sp_003",
                "type": "career_planning",
                "problem": "Software engineer wants transition to AI/ML field in 2 years. Current skills: Python, web development. Budget for education: $5K/year. Time available: 10 hours/week. Create strategic skill development plan.",
                "planning_complexity": "medium",
                "time_horizon": "2_years",
                "constraints": ["time", "budget", "skill_gaps"]
            },
            {
                "id": "sp_004",
                "type": "investment_strategy",
                "problem": "30-year-old with $20K savings, $50K annual income, wants $1M by age 60. Risk tolerance: medium. Create strategic investment plan considering market cycles, inflation, and life events.",
                "planning_complexity": "high",
                "time_horizon": "30_years",
                "constraints": ["risk_tolerance", "income", "time_horizon"]
            },
            {
                "id": "sp_005",
                "type": "product_launch",
                "problem": "Launch eco-friendly cleaning product line. Initial budget: $100K. Target market: environmentally conscious consumers aged 25-45. Competitors: 3 established brands. Create go-to-market strategy.",
                "planning_complexity": "high",
                "time_horizon": "18_months",
                "constraints": ["budget", "competition", "target_market"]
            },
            {
                "id": "sp_006",
                "type": "organizational_change",
                "problem": "Traditional company needs digital transformation. 500 employees, resistant to change. Budget: $2M over 2 years. Must maintain operations during transition. Create change management strategy.",
                "planning_complexity": "very_high",
                "time_horizon": "2_years",
                "constraints": ["culture", "operations", "budget"]
            },
            {
                "id": "sp_007",
                "type": "educational_planning",
                "problem": "High school student wants admission to top engineering program. Current GPA: 3.7, SAT: 1350. Extracurriculars: limited. 2 years until application. Create comprehensive preparation strategy.",
                "planning_complexity": "medium",
                "time_horizon": "2_years",
                "constraints": ["academic_requirements", "extracurriculars", "time"]
            },
            {
                "id": "sp_008",
                "type": "startup_strategy",
                "problem": "Launch SaaS productivity tool for remote teams. Initial funding: $250K. Target: 1000 paying customers in year 1. Competition: 5 established players. Create growth strategy.",
                "planning_complexity": "high",
                "time_horizon": "1_year",
                "constraints": ["funding", "competition", "growth_targets"]
            }
        ]

        self.results = {
            "cycle_info": {
                "name": self.cycle_name,
                "hypothesis": self.hypothesis,
                "focus_area": self.focus_area,
                "timestamp": datetime.now().isoformat(),
                "problems_tested": len(self.strategic_problems)
            },
            "baseline_results": [],
            "enhanced_results": [],
            "improvement_analysis": {}
        }

    def configure_strategic_planning(self):
        """Configure enhanced engine with strategic planning capabilities."""
        planning_prompt = """
You are an expert strategic planner. Use the PLAN-DO-REVIEW framework for complex problems:

PLAN PHASE:
1. Situation Analysis
   - Current state assessment
   - Goal clarification and specification
   - Resource and constraint identification
   - SWOT analysis (Strengths, Weaknesses, Opportunities, Threats)

2. Strategy Development
   - Define clear objectives and milestones
   - Identify multiple strategic options
   - Evaluate options against criteria
   - Select optimal strategy with rationale

3. Action Planning
   - Break down into specific, actionable steps
   - Assign timelines and dependencies
   - Identify required resources
   - Create monitoring and evaluation criteria

DO PHASE:
4. Execution Strategy
   - Detailed implementation steps
   - Risk mitigation strategies
   - Resource allocation plan
   - Communication and coordination plan

5. Progress Tracking
   - Key performance indicators
   - Milestone checkpoints
   - Adaptation mechanisms
   - Success metrics

REVIEW PHASE:
6. Evaluation Framework
   - Success criteria and measures
   - Regular review schedule
   - Learning and adaptation loops
   - Contingency planning

7. Strategic Insights
   - Lessons learned integration
   - Continuous improvement plan
   - Long-term sustainability
   - Scalability considerations

Focus on: Comprehensive planning, realistic execution, measurable outcomes.
"""
        self.engine_enhanced.base_model.system_prompt = planning_prompt

    def analyze_planning_quality(self, response: str, problem: Dict) -> Dict:
        """Analyze the quality of strategic planning in response."""
        plan_indicators = [
            "plan", "strategy", "objective", "goal",
            "milestone", "timeline", "phase", "step"
        ]

        execution_indicators = [
            "implement", "execute", "action", "resource",
            "budget", "team", "schedule", "deliverable"
        ]

        review_indicators = [
            "review", "evaluate", "measure", "track",
            "monitor", "assess", "feedback", "improve"
        }

        analysis = {
            "has_plan_phase": any(word in response.lower() for word in plan_indicators),
            "has_execution_phase": any(word in response.lower() for word in execution_indicators),
            "has_review_phase": any(word in response.lower() for word in review_indicators),
            "identifies_constraints": any(word in response.lower() for word in ["constraint", "limitation", "challenge", "obstacle"]),
            "sets_milestones": any(word in response.lower() for word in ["milestone", "checkpoint", "deadline", "target"]),
            "allocates_resources": any(word in response.lower() for word in ["budget", "resource", "team", "allocation"]),
            "defines_metrics": any(word in response.lower() for word in ["metric", "kpi", "measure", "indicator"]),
            "considers_risks": any(word in response.lower() for word in ["risk", "contingency", "backup", "fallback"]),
            "time_horizon_match": problem["time_horizon"].replace("_", " ") in response.lower(),
            "structured_approach": any(word in response.lower() for word in ["phase 1", "phase 2", "step 1", "step 2", "first", "second"])
        }

        # Check for constraint coverage
        expected_constraints = problem.get("constraints", [])
        constraint_coverage = sum(1 for constraint in expected_constraints if constraint.replace("_", " ") in response.lower()) / len(expected_constraints) if expected_constraints else 0

        analysis["constraint_coverage"] = constraint_coverage
        analysis["overall_planning_quality"] = sum(analysis.values()) / len(analysis)

        return analysis

    def evaluate_response(self, response: str, problem: Dict) -> Dict:
        """Evaluate response quality with emphasis on strategic planning."""
        # Use existing metrics
        confidence = calculate_reasoning_confidence(response)
        errors = detect_reasoning_errors(response)

        # Add planning quality metrics
        planning_quality = self.analyze_planning_quality(response, problem)

        # Success requires correctness AND comprehensive planning
        overall_success = (
            confidence > 0.7 and
            len(errors) == 0 and
            planning_quality["overall_planning_quality"] > 0.6
        )

        return {
            "success": overall_success,
            "confidence": confidence,
            "errors": errors,
            "planning_quality": planning_quality,
            "response_length": len(response),
            "strategic_completeness": planning_quality["has_plan_phase"] and planning_quality["has_execution_phase"] and planning_quality["has_review_phase"]
        }

    def run_baseline_test(self) -> List[Dict]:
        """Run baseline test without strategic planning enhancements."""
        print("Running baseline strategic planning test...")
        results = []

        for problem in self.strategic_problems:
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
        """Run test with strategic planning enhancements."""
        print("Running enhanced strategic planning test...")
        results = []

        for problem in self.strategic_problems:
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

                print(f"  Enhanced {problem['id']}: Success={evaluation['success']}, Planning={evaluation['planning_quality']['overall_planning_quality']:.2f}")

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

        # Calculate average planning quality
        enhanced_avg_planning = sum(r["evaluation"]["planning_quality"]["overall_planning_quality"] for r in enhanced_results) / len(enhanced_results)

        # Calculate strategic completeness improvement
        baseline_completeness = sum(r["evaluation"]["planning_quality"]["has_plan_phase"] and
                                   r["evaluation"]["planning_quality"]["has_execution_phase"] for r in baseline_results) / len(baseline_results)
        enhanced_completeness = sum(r["evaluation"]["strategic_completeness"] for r in enhanced_results) / len(enhanced_results)

        # Calculate improvements
        success_improvement = enhanced_success_rate - baseline_success_rate
        completeness_improvement = enhanced_completeness - baseline_completeness

        # Determine if cycle succeeds (requires >2% improvement in success rate)
        cycle_succeeds = success_improvement > 0.02

        return {
            "baseline_success_rate": baseline_success_rate,
            "enhanced_success_rate": enhanced_success_rate,
            "success_improvement": success_improvement,
            "baseline_strategic_completeness": baseline_completeness,
            "enhanced_strategic_completeness": enhanced_completeness,
            "completeness_improvement": completeness_improvement,
            "enhanced_avg_planning_quality": enhanced_avg_planning,
            "cycle_succeeds": cycle_succeeds,
            "meets_hypothesis": success_improvement >= 0.08  # 8% threshold from hypothesis
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
        print(f"Testing {len(self.strategic_problems)} strategic planning problems\n")

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
        print(f"Planning Quality: {analysis['enhanced_avg_planning_quality']:.1%}")
        print(f"Strategic Completeness: {analysis['enhanced_strategic_completeness']:.1%}")
        print(f"Cycle Succeeds: {analysis['cycle_succeeds']}")
        print(f"Meets Hypothesis: {analysis['meets_hypothesis']}")
        print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    cycle = StrategicPlanningBenchmark()
    cycle.run_cycle()