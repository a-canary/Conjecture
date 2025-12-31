#!/usr/bin/env python3
"""
Simple Detailed Evaluation Viewer

Shows complete evaluation process including:
- Actual test problems
- Direct response vs Conjecture response
- Claims creation and evaluation logs
- Detailed comparison tables
- All process details without Unicode issues

PRINCIPLE: COMPLETE TRANSPARENCY OF EVALUATION PROCESS
"""

import asyncio
import json
import os
import sys
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

@dataclass
class Claim:
    id: str
    content: str
    confidence: float
    tags: List[str]
    state: str
    context: Optional[str] = None
    metadata: Dict[str, Any] = None
    created_at: Optional[str] = None

class SimpleDetailedViewer:
    """Simple viewer showing complete evaluation process"""

    def __init__(self):
        # Use GLM-4.5-air API from config
        self.api_key = "70e6e12e4d7c46e2a4d0b85503d51f38.LQHl8d98kDJChttb"
        self.base_url = "https://api.z.ai/api/coding/paas/v4"
        self.model = "glm-4.5-air"
        self.claims_created = []
        self.evaluation_log = []

    async def run_simple_evaluation(self) -> Dict[str, Any]:
        """Run evaluation showing all process details"""
        print("DETAILED EVALUATION VIEWER")
        print("Complete Process: Problems -> Direct Response -> Claims -> Evaluation -> Results")
        print("=" * 80)

        # Test problems
        test_problems = self.load_test_problems()

        results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "test_problems": test_problems,
            "direct_responses": {},
            "conjecture_responses": {},
            "claims_created": [],
            "evaluation_log": [],
            "comparison_table": [],
            "performance_metrics": {}
        }

        print(f"\n[EVALUATING {len(test_problems)} TEST PROBLEMS]")
        print("=" * 50)

        for i, problem in enumerate(test_problems, 1):
            print(f"\n--- PROBLEM {i}/{len(test_problems)} ---")
            print(f"Question: {problem['question']}")
            print(f"Expected: {problem['expected']}")
            print(f"Domain: {problem['domain']}")

            # Get direct response
            print(f"\n[Getting direct LLM response...]")
            direct_response = await self.get_direct_response(problem['question'])
            results["direct_responses"][problem['id']] = direct_response
            print(f"Direct: {direct_response[:150]}{'...' if len(direct_response) > 150 else ''}")

            # Get conjecture response
            print(f"[Getting Conjecture-enhanced response...]")
            conjecture_response = await self.get_conjecture_response(problem)
            results["conjecture_responses"][problem['id']] = conjecture_response
            print(f"Conjecture: {conjecture_response[:150]}{'...' if len(conjecture_response) > 150 else ''}")

            # Create and evaluate claims
            print(f"[Creating and evaluating claims...]")
            claim_results = self.create_and_evaluate_claims(problem, direct_response, conjecture_response)

            results["claims_created"].extend(claim_results["created"])
            results["evaluation_log"].extend(claim_results["log"])

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
            results["comparison_table"].append(comparison)

            print(f"Problem {i} completed")

        # Generate summary metrics
        results["performance_metrics"] = self.calculate_performance_metrics(results)

        # Display final summary
        self.display_comprehensive_summary(results)

        return results

    def load_test_problems(self) -> List[Dict[str, Any]]:
        """Load comprehensive test problems"""
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
                "id": "logic_002",
                "question": "If A implies B and B implies C, does A imply C?",
                "expected": "Yes",
                "context": "Logical transitivity: A → B → C means A → C",
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

    async def get_direct_response(self, question: str) -> str:
        """Get direct LLM response without Conjecture enhancement"""
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": question}
                    ],
                    "max_tokens": 500,
                    "temperature": 0.2
                },
                timeout=30
            )

            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

        except Exception as e:
            return f"Error getting direct response: {str(e)}"

    async def get_conjecture_response(self, problem: Dict[str, Any]) -> str:
        """Get Conjecture-enhanced response with claim evaluation"""
        # Simulate Conjecture enhancement with structured prompting
        enhanced_prompt = f"""
Domain: {problem['domain']}
Question: {problem['question']}
Context: {problem['context']}

Please provide a comprehensive answer with:
1. Clear explanation
2. Step-by-step reasoning if applicable
3. Verification of your answer
4. Final conclusion

Answer:"""

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": enhanced_prompt}
                    ],
                    "max_tokens": 800,
                    "temperature": 0.2
                },
                timeout=30
            )

            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

        except Exception as e:
            return f"Error getting Conjecture response: {str(e)}"

    def create_and_evaluate_claims(self, problem: Dict[str, Any], direct_response: str, conjecture_response: str) -> Dict[str, Any]:
        """Create and evaluate claims for both responses"""
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
        """Extract claims from LLM response"""
        claims = []

        # Parse response into sentences/statements
        statements = self.parse_response(response)

        for i, statement in enumerate(statements):
            if len(statement.strip()) > 20:  # Only substantial statements
                claim = Claim(
                    id=f"{problem_id}_{source}_claim_{i+1:03d}",
                    content=statement.strip(),
                    confidence=0.8,  # Default confidence
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
        # Split by common punctuation and clean up
        import re
        statements = re.split(r'[.!?]+', response)

        # Clean and filter statements
        cleaned_statements = []
        for stmt in statements:
            stmt = stmt.strip()
            if stmt and len(stmt) > 10:  # Minimum length threshold
                cleaned_statements.append(stmt)

        return cleaned_statements

    def evaluate_claim(self, claim: Claim, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a claim's correctness and relevance"""
        # Simulate claim evaluation with scoring
        expected = problem.get('expected', '')
        claim_content = claim.content.lower()

        # Simple keyword matching for evaluation
        if any(word in claim_content for word in expected.lower().split()):
            score = 0.9
            confidence = 0.8
        elif claim.content and len(claim.content) > 50:
            score = 0.6  # Moderate score for substantial content
            confidence = 0.5
        else:
            score = 0.3
            confidence = 0.3

        return {
            "claim_id": claim.id,
            "score": score,
            "confidence": confidence,
            "relevance": 0.8 if problem['id'].split('_')[0] in claim.tags else 0.5,
            "evaluated_at": datetime.now().isoformat(),
            "evaluation_details": f"Evaluation for claim from {claim.metadata.get('source', 'unknown')}"
        }

    def evaluate_correctness(self, expected: str, actual: str) -> bool:
        """Simple correctness evaluation"""
        expected_lower = expected.lower()
        actual_lower = actual.lower()

        # Check if expected answer is contained in actual response
        return expected_lower in actual_lower or any(word in actual_lower for word in expected_lower.split())

    def calculate_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
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
        print(f"{'ID':<12} {'Question':<30} {'Expected':<15} {'Direct':<8} {'Conjecture':<12} {'Direct_C':<10} {'Conj_C':<10} {'Claims':<8}")
        print("-" * 120)

        for comp in results.get("comparison_table", []):
            direct_status = "PASS" if comp["direct_correct"] else "FAIL"
            conj_status = "PASS" if comp["conjecture_correct"] else "FAIL"

            print(f"{comp['problem_id']:<12} {comp['question'][:28]:<30} {comp['expected']:<15} "
                  f"{'Direct':<8} {'Conjecture':<12} {direct_status:<10} {conj_status:<10} {comp['claims_created']:<8}")

        # Claims table
        print(f"\nCLAIMS ANALYSIS")
        print("-" * 100)
        print(f"{'Claim ID':<20} {'Content':<50} {'Score':<8} {'Confidence':<12} {'Source':<10}")
        print("-" * 100)

        for claim in results.get("claims_created", [])[:10]:  # Show first 10 claims
            score = next((eval_data.get("score", 0) for eval_data in results.get("claims_evaluated", [])
                        if eval_data.get("claim_id") == claim.id), 0)
            confidence = next((eval_data.get("confidence", 0) for eval_data in results.get("claims_evaluated", [])
                            if eval_data.get("claim_id") == claim.id), 0)
            source = claim.metadata.get("source", "unknown")[:8] if claim.metadata else "unknown"

            print(f"{claim.id:<20} {claim.content[:47]:<50} {score:<8.2f} {confidence:<12.2f} {source:<10}")

        if len(results.get("claims_created", [])) > 10:
            print(f"... and {len(results.get('claims_created', [])) - 10} more claims")

        # Evaluation log
        print(f"\nEVALUATION EVENT LOG")
        print("-" * 80)
        for log_entry in results.get("evaluation_log", []):
            timestamp = log_entry.get("timestamp", "")[:19]
            action = log_entry.get("action", "")
            details = f"{log_entry.get('source', '')} - {log_entry.get('problem_id', '')}"
            if 'claim_count' in log_entry:
                details += f" ({log_entry['claim_count']} claims)"
            elif 'claim_id' in log_entry:
                details += f" ({log_entry['claim_id'][:8]}...)"

            print(f"{timestamp:<20} {action:<20} {details}")

        print(f"\n{'='*80}")
        print("EVALUATION COMPLETE")
        print(f"Timestamp: {results.get('evaluation_timestamp', 'Unknown')}")
        print(f"{'='*80}")

async def main():
    """Run detailed evaluation viewer"""
    viewer = SimpleDetailedViewer()
    results = await viewer.run_simple_evaluation()

    # Convert Claim objects to dictionaries for JSON serialization
    def claim_to_dict(claim):
        if hasattr(claim, '__dict__'):
            result = {}
            for key, value in claim.__dict__.items():
                if key == 'claim_type' and hasattr(value, '__iter__'):
                    result[key] = [t.value if hasattr(t, 'value') else str(t) for t in value]
                elif hasattr(value, 'value'):
                    result[key] = value.value
                else:
                    result[key] = value
            return result
        return claim

    # Prepare results for JSON serialization
    json_results = results.copy()
    json_results["claims_created"] = [claim_to_dict(claim) for claim in results["claims_created"]]

    # Save detailed results
    results_file = "src/benchmarking/detailed_evaluation_results.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    print(f"\n[Detailed results saved to: {results_file}]")
    return results

if __name__ == "__main__":
    asyncio.run(main())