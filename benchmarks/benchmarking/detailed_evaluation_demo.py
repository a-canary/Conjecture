#!/usr/bin/env python3
"""
Detailed Evaluation Process Demonstration

Shows complete evaluation workflow including:
- Actual test problems with expected answers
- Claims creation from LLM responses (simulated structure)
- Claims evaluation with scoring and confidence
- Detailed comparison tables
- Complete event logs showing chronological order

PRINCIPLE: TRANSPARENT EVALUATION PROCESS VISIBILITY
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

@dataclass
class Claim:
    id: str
    content: str
    confidence: float
    tags: List[str]
    state: str
    context: str = None
    metadata: Dict[str, Any] = None
    created_at: str = None

class DetailedEvaluationDemo:
    """Demonstration of complete evaluation process"""

    def __init__(self):
        self.claims_created = []
        self.evaluation_log = []
        self.results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "test_problems": [],
            "direct_responses": {},
            "conjecture_responses": {},
            "claims_created": [],
            "claims_evaluated": [],
            "evaluation_log": [],
            "comparison_table": [],
            "performance_metrics": {}
        }

    def run_evaluation_demo(self) -> Dict[str, Any]:
        """Run complete evaluation demonstration"""
        print("DETAILED EVALUATION PROCESS DEMONSTRATION")
        print("Complete Workflow: Problems -> Responses -> Claims -> Evaluation -> Results")
        print("=" * 80)

        # Test problems with real examples
        test_problems = self.load_test_problems()
        self.results["test_problems"] = test_problems

        print(f"\n[EVALUATING {len(test_problems)} TEST PROBLEMS]")
        print("=" * 50)

        for i, problem in enumerate(test_problems, 1):
            print(f"\n--- PROBLEM {i}/{len(test_problems)} ---")
            print(f"ID: {problem['id']}")
            print(f"Question: {problem['question']}")
            print(f"Expected: {problem['expected']}")
            print(f"Domain: {problem['domain']}")

            # Simulate direct LLM response
            direct_response = self.get_sample_direct_response(problem)
            self.results["direct_responses"][problem['id']] = direct_response
            print(f"\n[Direct Response]:")
            print(f"  {direct_response}")

            # Simulate Conjecture-enhanced response
            conjecture_response = self.get_sample_conjecture_response(problem)
            self.results["conjecture_responses"][problem['id']] = conjecture_response
            print(f"\n[Conjecture Response]:")
            print(f"  {conjecture_response}")

            # Create and evaluate claims
            print(f"\n[Creating and evaluating claims...]")
            claim_results = self.create_and_evaluate_claims(problem, direct_response, conjecture_response)

            self.results["claims_created"].extend(claim_results["created"])
            self.results["claims_evaluated"].extend(claim_results["evaluated"])
            self.results["evaluation_log"].extend(claim_results["log"])

            # Add to comparison table
            comparison = {
                "problem_id": problem['id'],
                "question": problem['question'],
                "expected": problem['expected'],
                "direct_response": direct_response,
                "conjecture_response": conjecture_response,
                "direct_correct": self.evaluate_correctness(problem['expected'], direct_response),
                "conjecture_correct": self.evaluate_correctness(problem['expected'], conjecture_response),
                "claims_created": len(claim_results["created"]),
                "claims_evaluated": len(claim_results["evaluated"])
            }
            self.results["comparison_table"].append(comparison)

            print(f"Problem {i} completed - {len(claim_results['created'])} claims created")

        # Generate metrics and display summary
        self.results["performance_metrics"] = self.calculate_performance_metrics(self.results)
        self.display_comprehensive_summary(self.results)

        return self.results

    def load_test_problems(self) -> List[Dict[str, Any]]:
        """Load real test problems across multiple domains"""
        return [
            {
                "id": "math_001",
                "question": "What is 15% of 240?",
                "expected": "36",
                "context": "Calculate percentage: 15% × 240 = 36",
                "domain": "mathematical"
            },
            {
                "id": "math_002",
                "question": "A train travels 300 miles in 4 hours. What is its average speed?",
                "expected": "75 mph",
                "context": "Speed = Distance / Time = 300/4 = 75",
                "domain": "mathematical"
            },
            {
                "id": "logic_001",
                "question": "All cats are animals. Some animals are pets. Can we conclude some cats are pets?",
                "expected": "No",
                "context": "Logical syllogism - the pets might be different species",
                "domain": "logical"
            },
            {
                "id": "coding_001",
                "question": "What is the time complexity of binary search?",
                "expected": "O(log n)",
                "context": "Binary search halves the search space each iteration",
                "domain": "coding"
            },
            {
                "id": "coding_002",
                "question": "Write a function to calculate factorial of a number",
                "expected": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
                "context": "Recursive implementation of factorial function",
                "domain": "coding"
            }
        ]

    def get_sample_direct_response(self, problem: Dict[str, Any]) -> str:
        """Sample direct LLM response (simulated for demonstration)"""
        responses = {
            "math_001": "To calculate 15% of 240, I multiply 240 by 0.15 which gives me 36.",
            "math_002": "The average speed is distance divided by time, so 300 divided by 4 equals 75 mph.",
            "logic_001": "Yes, since some cats are pets.",
            "coding_001": "The time complexity is O(n) because it searches through half the array.",
            "coding_002": "def factorial(n):\n    if n <= 1:\n        return 1\n    else:\n        return n * factorial(n-1)"
        }
        return responses.get(problem['id'], "Direct response for " + problem['question'])

    def get_sample_conjecture_response(self, problem: Dict[str, Any]) -> str:
        """Sample Conjecture-enhanced response (simulated for demonstration)"""
        responses = {
            "math_001": "To find 15% of 240: Step 1: Convert 15% to decimal: 0.15. Step 2: Multiply: 240 × 0.15 = 36. Verification: 36 ÷ 240 = 0.15 = 15%. The answer is 36.",
            "math_002": "Average speed calculation: Speed = Distance ÷ Time = 300 miles ÷ 4 hours = 75 mph. This assumes constant speed throughout the journey.",
            "logic_001": "No. While all cats are animals and some animals are pets, we cannot conclude that cats are among the pets. The animals that are pets might be dogs, birds, or other species.",
            "coding_001": "Binary search has O(log n) time complexity because it eliminates half of the remaining search space with each comparison, making it very efficient for sorted arrays.",
            "coding_002": "Here is a clean recursive factorial implementation:\n\ndef factorial(n):\n    \"\"\"Calculate factorial recursively\"\"\"\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n\nBase case verified, recursion will terminate correctly."
        }
        return responses.get(problem['id'], "Conjecture-enhanced response for " + problem['question'])

    def create_and_evaluate_claims(self, problem: Dict[str, Any], direct_response: str, conjecture_response: str) -> Dict[str, Any]:
        """Create and evaluate claims from both responses"""
        created_claims = []
        evaluated_claims = []
        log_entries = []

        # Create claims from direct response
        direct_claims = self.extract_claims_from_response(
            problem['id'], direct_response, "direct"
        )
        created_claims.extend(direct_claims)
        log_entries.append({
            "timestamp": datetime.now().isoformat(),
            "action": "claims_created",
            "source": "direct_response",
            "problem_id": problem['id'],
            "claim_count": len(direct_claims)
        })

        # Create claims from conjecture response
        conjecture_claims = self.extract_claims_from_response(
            problem['id'], conjecture_response, "conjecture"
        )
        created_claims.extend(conjecture_claims)
        log_entries.append({
            "timestamp": datetime.now().isoformat(),
            "action": "claims_created",
            "source": "conjecture_response",
            "problem_id": problem['id'],
            "claim_count": len(conjecture_claims)
        })

        # Evaluate all claims
        for claim in created_claims:
            evaluation = self.evaluate_claim(claim, problem)
            evaluated_claims.append(evaluation)
            log_entries.append({
                "timestamp": datetime.now().isoformat(),
                "action": "claim_evaluated",
                "claim_id": claim.id,
                "evaluation_score": evaluation.get("score", 0),
                "confidence": evaluation.get("confidence", 0)
            })

        return {
            "created": created_claims,
            "evaluated": evaluated_claims,
            "log": log_entries
        }

    def extract_claims_from_response(self, problem_id: str, response: str, source: str) -> List[Claim]:
        """Extract individual claims from LLM response"""
        claims = []

        # Split response into statements
        statements = self.parse_response(response)

        for i, statement in enumerate(statements):
            if len(statement.strip()) > 15:  # Only substantial statements
                claim = Claim(
                    id=f"{problem_id}_{source}_claim_{i+1:03d}",
                    content=statement.strip(),
                    confidence=0.8 if source == "conjecture" else 0.7,  # Conjecture responses typically more confident
                    tags=[source, problem_id.split('_')[0]],  # Source and domain tags
                    state="EXPLORE",
                    context=f"Problem: {problem_id}, Source: {source}",
                    metadata={
                        "source": source,
                        "problem_id": problem_id,
                        "response_snippet": statement[:100]
                    },
                    created_at=datetime.now().isoformat()
                )
                claims.append(claim)

        return claims

    def parse_response(self, response: str) -> List[str]:
        """Parse response into individual statements"""
        import re
        statements = re.split(r'[.!?]+', response)

        # Clean and filter statements
        cleaned_statements = []
        for stmt in statements:
            stmt = stmt.strip()
            if stmt and len(stmt) > 10:
                cleaned_statements.append(stmt)

        return cleaned_statements

    def evaluate_claim(self, claim: Claim, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate claim correctness and relevance"""
        expected = problem.get('expected', '').lower()
        claim_content = claim.content.lower()

        # Simple keyword matching for demonstration
        if any(word in claim_content for word in expected.lower().split()):
            score = 0.9
            confidence = 0.8
        elif claim.content and len(claim.content) > 30:
            score = 0.6
            confidence = 0.6
        else:
            score = 0.3
            confidence = 0.4

        return {
            "claim_id": claim.id,
            "score": score,
            "confidence": confidence,
            "relevance": 0.8 if problem['id'].split('_')[0] in claim.tags else 0.5,
            "evaluated_at": datetime.now().isoformat(),
            "evaluation_details": f"Evaluation for claim from {claim.metadata.get('source', 'unknown')}"
        }

    def evaluate_correctness(self, expected: str, actual: str) -> bool:
        """Evaluate response correctness"""
        expected_lower = expected.lower()
        actual_lower = actual.lower()

        # Check if expected answer is contained in actual response
        return expected_lower in actual_lower or any(word in actual_lower for word in expected_lower.split())

    def calculate_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance metrics"""
        comparison_table = results.get("comparison_table", [])

        total_problems = len(comparison_table)
        direct_correct = sum(1 for comp in comparison_table if comp["direct_correct"])
        conjecture_correct = sum(1 for comp in comparison_table if comp["conjecture_correct"])
        total_claims_created = sum(comp["claims_created"] for comp in comparison_table)
        total_claims_evaluated = sum(comp["claims_evaluated"] for comp in comparison_table)

        return {
            "total_problems": total_problems,
            "direct_accuracy": (direct_correct / total_problems * 100) if total_problems > 0 else 0,
            "conjecture_accuracy": (conjecture_correct / total_problems * 100) if total_problems > 0 else 0,
            "improvement": ((conjecture_correct - direct_correct) / direct_correct * 100) if direct_correct > 0 else 0,
            "total_claims_created": total_claims_created,
            "total_claims_evaluated": total_claims_evaluated,
            "claims_per_problem": total_claims_created / total_problems if total_problems > 0 else 0
        }

    def display_comprehensive_summary(self, results: Dict[str, Any]):
        """Display detailed evaluation summary"""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE EVALUATION SUMMARY")
        print(f"{'='*80}")

        # Performance metrics
        metrics = results.get("performance_metrics", {})
        print(f"\nPERFORMANCE METRICS")
        print(f"Total Problems: {metrics.get('total_problems', 0)}")
        print(f"Direct Accuracy: {metrics.get('direct_accuracy', 0):.1f}%")
        print(f"Conjecture Accuracy: {metrics.get('conjecture_accuracy', 0):.1f}%")
        print(f"Improvement: {metrics.get('improvement', 0):.1f}%")
        print(f"Total Claims Created: {metrics.get('total_claims_created', 0)}")
        print(f"Total Claims Evaluated: {metrics.get('total_claims_evaluated', 0)}")
        print(f"Average Claims per Problem: {metrics.get('claims_per_problem', 0):.1f}")

        # Detailed comparison table
        print(f"\nDETAILED COMPARISON TABLE")
        print("-" * 120)
        print(f"{'ID':<12} {'Question':<30} {'Expected':<15} {'Direct_C':<10} {'Conj_C':<10} {'Claims':<8}")
        print("-" * 120)

        for comp in results.get("comparison_table", []):
            direct_status = "PASS" if comp["direct_correct"] else "FAIL"
            conj_status = "PASS" if comp["conjecture_correct"] else "FAIL"

            print(f"{comp['problem_id']:<12} {comp['question'][:28]:<30} {comp['expected']:<15} "
                  f"{direct_status:<10} {conj_status:<10} {comp['claims_created']:<8}")

        # Claims analysis table
        print(f"\nCLAIMS ANALYSIS TABLE")
        print("-" * 100)
        print(f"{'Claim ID':<20} {'Content':<50} {'Score':<8} {'Confidence':<12} {'Source':<10}")
        print("-" * 100)

        for claim in results.get("claims_created", [])[:15]:  # Show first 15 claims
            score = next((eval_data.get("score", 0) for eval_data in results.get("claims_evaluated", [])
                         if eval_data.get("claim_id") == claim.id), 0)
            confidence = next((eval_data.get("confidence", 0) for eval_data in results.get("claims_evaluated", [])
                             if eval_data.get("claim_id") == claim.id), 0)
            source = claim.metadata.get("source", "unknown")[:8] if claim.metadata else "unknown"

            print(f"{claim.id:<20} {claim.content[:47]:<50} {score:<8.2f} {confidence:<12.2f} {source:<10}")

        if len(results.get("claims_created", [])) > 15:
            print(f"... and {len(results.get('claims_created', [])) - 15} more claims")

        # Evaluation event log (chronological)
        print(f"\nEVALUATION EVENT LOG (Chronological)")
        print("-" * 80)
        for log_entry in results.get("evaluation_log", []):
            timestamp = log_entry.get("timestamp", "")[:19]
            action = log_entry.get("action", "")
            details = f"{log_entry.get('source', '')} - {log_entry.get('problem_id', '')}"
            if 'claim_count' in log_entry:
                details += f" ({log_entry['claim_count']} claims)"
            elif 'claim_id' in log_entry:
                details += f" ({log_entry['claim_id'][:12]}...)"

            print(f"{timestamp:<20} {action:<20} {details}")

        print(f"\n{'='*80}")
        print("EVALUATION COMPLETE")
        print(f"Timestamp: {results.get('evaluation_timestamp', 'Unknown')}")
        print(f"{'='*80}")

def main():
    """Run detailed evaluation demonstration"""
    demo = DetailedEvaluationDemo()
    results = demo.run_evaluation_demo()

    # Save results to JSON file
    results_file = "src/benchmarking/detailed_evaluation_demo_results.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    # Convert claims to dictionaries for JSON serialization
    json_results = results.copy()
    json_results["claims_created"] = [asdict(claim) for claim in results["claims_created"]]

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    print(f"\n[Demonstration results saved to: {results_file}]")
    return results

if __name__ == "__main__":
    main()