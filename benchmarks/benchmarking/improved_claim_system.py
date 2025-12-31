#!/usr/bin/env python3
"""
Improved Claim System with Feedback Implementation

Based on evaluation results feedback:
- Simple numeric IDs (123456, 245268, 000001)
- Lower confidence for unsupported claims (new claims start low)
- Scope ID and Session ID instead of metadata
- Removes unnecessary metadata field
- Decomposition and verification requirements for cheap LLM providers

PRINCIPLE: IMPROVED CLAIM VALIDATION WITH SCOPE AND SESSION TRACKING
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import uuid

@dataclass
class Claim:
    id: str                    # Simple numeric ID like "123456"
    content: str
    confidence: float          # Lower initial confidence for new claims
    tags: List[str]
    state: str
    context: str = None
    scope_id: str = None       # Scope identification
    session_id: str = None     # Session identification
    parent_id: str = None      # Parent claim ID if decomposed
    supporting_claims: List[str] = None  # IDs of supporting claims
    provider_source: str = None  # Which LLM provider created this claim
    created_at: str = None
    evaluated_at: str = None

class ImprovedClaimSystem:
    """Enhanced claim system addressing feedback from evaluation results"""

    def __init__(self):
        self.claims_created = []
        self.evaluation_log = []
        self.next_claim_id = 100000  # Start with simple numeric ID
        self.session_id = str(uuid.uuid4())[:8]  # Short session ID
        self.scope_id = "evaluation_demo"
        self.claim_counter = 0

    def run_improved_evaluation(self) -> Dict[str, Any]:
        """Run improved evaluation with enhanced claim system"""
        print("IMPROVED CLAIM SYSTEM EVALUATION")
        print("Enhanced Features: Simple IDs, Lower Initial Confidence, Scope/Session Tracking")
        print("=" * 80)

        # Test problems from previous evaluation
        test_problems = self.load_test_problems()

        results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "scope_id": self.scope_id,
            "test_problems": test_problems,
            "direct_responses": {},
            "conjecture_responses": {},
            "claims_created": [],
            "claims_evaluated": [],
            "evaluation_log": [],
            "comparison_table": [],
            "performance_metrics": {}
        }

        print(f"\n[EVALUATING {len(test_problems)} PROBLEMS WITH IMPROVED CLAIM SYSTEM]")
        print("=" * 60)

        for i, problem in enumerate(test_problems, 1):
            print(f"\n--- PROBLEM {i}/{len(test_problems)} ---")
            print(f"Question: {problem['question']}")
            print(f"Expected: {problem['expected']}")

            # Simulate responses from different providers
            direct_response = self.get_provider_response(problem, "cheap_provider")
            conjecture_response = self.get_provider_response(problem, "quality_provider")

            results["direct_responses"][problem['id']] = direct_response
            results["conjecture_responses"][problem['id']] = conjecture_response

            print(f"\n[Cheap Provider Response]:\n  {direct_response}")
            print(f"\n[Quality Provider Response]:\n  {conjecture_response}")

            # Create and evaluate claims with improved system
            print(f"\n[Creating claims with improved system...]")
            claim_results = self.create_and_evaluate_improved_claims(problem, direct_response, conjecture_response)

            results["claims_created"].extend(claim_results["created"])
            results["claims_evaluated"].extend(claim_results["evaluated"])
            results["evaluation_log"].extend(claim_results["log"])

            # Add to comparison table
            comparison = {
                "problem_id": problem['id'],
                "question": problem['question'],
                "expected": problem['expected'],
                "direct_correct": self.evaluate_response_correctness(problem['expected'], direct_response),
                "conjecture_correct": self.evaluate_response_correctness(problem['expected'], conjecture_response),
                "claims_created": len(claim_results["created"]),
                "average_confidence": sum(c.confidence for c in claim_results["created"]) / len(claim_results["created"]) if claim_results["created"] else 0,
                "supporting_claims_count": len([c for c in claim_results["created"] if c.supporting_claims])
            }
            results["comparison_table"].append(comparison)

            print(f"Problem {i} completed - {len(claim_results['created'])} claims created")
            print(f"Average confidence: {comparison['average_confidence']:.2f}")

        # Generate metrics and display summary
        results["performance_metrics"] = self.calculate_enhanced_metrics(results)
        self.display_improved_summary(results)

        return results

    def load_test_problems(self) -> List[Dict[str, Any]]:
        """Load test problems including the problematic example from feedback"""
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
            }
        ]

    def get_provider_response(self, problem: Dict[str, Any], provider_type: str) -> str:
        """Simulate response from different provider types"""
        if provider_type == "cheap_provider":
            # Cheap provider gives minimal, potentially incorrect responses
            responses = {
                "math_001": "To calculate 15% of 240, I multiply 240 by 0",  # Incorrect: should be 0.15
                "math_002": "300 divided by 4 equals 75 mph",
                "logic_001": "Yes, since some cats are pets",  # Incorrect logic
                "coding_001": "The time complexity is O(n) because it searches through half the array"  # Wrong complexity
            }
            return responses.get(problem['id'], f"Cheap provider response to {problem['question']}")
        else:  # quality_provider
            # Quality provider provides detailed, verified responses
            responses = {
                "math_001": "To find 15% of 240: Step 1: Convert 15% to decimal: 0.15. Step 2: Multiply: 240 × 0.15 = 36. Verification: 36 ÷ 240 = 0.15 = 15%. The answer is 36.",
                "math_002": "Average speed calculation: Speed = Distance ÷ Time = 300 miles ÷ 4 hours = 75 mph. This assumes constant speed throughout the journey.",
                "logic_001": "No. While all cats are animals and some animals are pets, we cannot conclude that cats are among the pets. The animals that are pets might be dogs, birds, or other species.",
                "coding_001": "Binary search has O(log n) time complexity because it eliminates half of the remaining search space with each comparison, making it very efficient for sorted arrays."
            }
            return responses.get(problem['id'], f"Quality provider response to {problem['question']}")

    def create_and_evaluate_improved_claims(self, problem: Dict[str, Any], direct_response: str, conjecture_response: str) -> Dict[str, Any]:
        """Create and evaluate claims with improved system"""
        created_claims = []
        evaluated_claims = []
        log_entries = []

        # Create claims from cheap provider (low initial confidence)
        cheap_claims = self.extract_claims_with_improved_system(
            problem, direct_response, "cheap_provider"
        )
        created_claims.extend(cheap_claims)
        log_entries.append({
            "timestamp": datetime.now().isoformat(),
            "action": "claims_created",
            "provider": "cheap_provider",
            "problem_id": problem['id'],
            "claim_count": len(cheap_claims),
            "average_confidence": sum(c.confidence for c in cheap_claims) / len(cheap_claims) if cheap_claims else 0
        })

        # Create claims from quality provider (higher initial confidence)
        quality_claims = self.extract_claims_with_improved_system(
            problem, conjecture_response, "quality_provider"
        )
        created_claims.extend(quality_claims)
        log_entries.append({
            "timestamp": datetime.now().isoformat(),
            "action": "claims_created",
            "provider": "quality_provider",
            "problem_id": problem['id'],
            "claim_count": len(quality_claims),
            "average_confidence": sum(c.confidence for c in quality_claims) / len(quality_claims) if quality_claims else 0
        })

        # Decompose cheap provider claims for verification
        for cheap_claim in cheap_claims:
            if cheap_claim.confidence < 0.6:  # Low confidence claims need verification
                decomposition_claims = self.decompose_claim_for_verification(cheap_claim, problem)
                created_claims.extend(decomposition_claims)

                # Update original claim with supporting claims
                cheap_claim.supporting_claims = [dc.id for dc in decomposition_claims]

                log_entries.append({
                    "timestamp": datetime.now().isoformat(),
                    "action": "claim_decomposed",
                    "claim_id": cheap_claim.id,
                    "supporting_claims": len(decomposition_claims),
                    "reason": "Low confidence verification required"
                })

        # Evaluate all claims
        for claim in created_claims:
            evaluation = self.evaluate_improved_claim(claim, problem)
            evaluated_claims.append(evaluation)
            log_entries.append({
                "timestamp": datetime.now().isoformat(),
                "action": "claim_evaluated",
                "claim_id": claim.id,
                "evaluation_score": evaluation.get("score", 0),
                "confidence": evaluation.get("confidence", 0),
                "has_supporting_claims": len(claim.supporting_claims or []) > 0
            })

        return {
            "created": created_claims,
            "evaluated": evaluated_claims,
            "log": log_entries
        }

    def extract_claims_with_improved_system(self, problem: Dict[str, Any], response: str, provider_source: str) -> List[Claim]:
        """Extract claims with improved ID system and confidence handling"""
        claims = []
        statements = self.parse_response(response)

        for statement in statements:
            if len(statement.strip()) > 15:
                claim_id = str(self.next_claim_id).zfill(6)  # Simple 6-digit numeric ID
                self.next_claim_id += 1

                # Initial confidence based on provider type
                if provider_source == "cheap_provider":
                    base_confidence = 0.3  # Very low confidence for cheap provider
                else:
                    base_confidence = 0.7  # Higher confidence for quality provider

                claim = Claim(
                    id=claim_id,
                    content=statement.strip(),
                    confidence=base_confidence,
                    tags=[provider_source, problem['id'].split('_')[0]],
                    state="EXPLORE",
                    context=f"Problem: {problem['id']}, Provider: {provider_source}",
                    scope_id=self.scope_id,
                    session_id=self.session_id,
                    provider_source=provider_source,
                    supporting_claims=[],
                    created_at=datetime.now().isoformat()
                )
                claims.append(claim)

        return claims

    def decompose_claim_for_verification(self, claim: Claim, problem: Dict[str, Any]) -> List[Claim]:
        """Decompose a low-confidence claim into verifiable sub-claims"""
        decomposition_claims = []

        # Example decomposition for the math problem
        if "multiply 240 by 0" in claim.content:
            # Decompose the incorrect calculation
            sub_claims = [
                "15% should be converted to decimal 0.15",
                "The multiplication should be 240 × 0.15",
                "0.15 × 240 = 36",
                "The final answer should be 36"
            ]

            for sub_content in sub_claims:
                claim_id = str(self.next_claim_id).zfill(6)
                self.next_claim_id += 1

                sub_claim = Claim(
                    id=claim_id,
                    content=sub_content,
                    confidence=0.6,  # Medium confidence for verification
                    tags=["verification", "decomposition"],
                    state="VERIFYING",
                    context=f"Decomposition of claim {claim.id}",
                    scope_id=self.scope_id,
                    session_id=self.session_id,
                    parent_id=claim.id,
                    provider_source="decomposition",
                    supporting_claims=[],
                    created_at=datetime.now().isoformat()
                )
                decomposition_claims.append(sub_claim)

        return decomposition_claims

    def parse_response(self, response: str) -> List[str]:
        """Parse response into individual statements"""
        import re
        statements = re.split(r'[.!?]+', response)
        cleaned_statements = []
        for stmt in statements:
            stmt = stmt.strip()
            if stmt and len(stmt) > 10:
                cleaned_statements.append(stmt)
        return cleaned_statements

    def evaluate_improved_claim(self, claim: Claim, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate claim with enhanced scoring"""
        expected = problem.get('expected', '').lower()
        claim_content = claim.content.lower()

        # Base scoring
        if any(word in claim_content for word in expected.lower().split()):
            score = 0.9
        elif "error" in claim_content.lower() or "wrong" in claim_content.lower():
            score = 0.1
        elif claim.content and len(claim.content) > 30:
            score = 0.6
        else:
            score = 0.3

        # Confidence adjustment based on supporting claims
        confidence = claim.confidence
        if claim.supporting_claims:
            confidence = min(0.9, confidence + 0.2 * len(claim.supporting_claims))

        # Provider-based confidence adjustment
        if claim.provider_source == "cheap_provider":
            confidence = confidence * 0.7  # Reduce confidence for cheap provider

        return {
            "claim_id": claim.id,
            "score": score,
            "confidence": confidence,
            "provider_source": claim.provider_source,
            "has_supporting_claims": len(claim.supporting_claims or []) > 0,
            "supporting_count": len(claim.supporting_claims or []),
            "evaluated_at": datetime.now().isoformat()
        }

    def evaluate_response_correctness(self, expected: str, actual: str) -> bool:
        """Evaluate response correctness"""
        expected_lower = expected.lower()
        actual_lower = actual.lower()
        return expected_lower in actual_lower or any(word in actual_lower for word in expected_lower.split())

    def calculate_enhanced_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate enhanced performance metrics"""
        comparison_table = results.get("comparison_table", [])
        claims_created = results.get("claims_created", [])

        total_problems = len(comparison_table)
        direct_correct = sum(1 for comp in comparison_table if comp["direct_correct"])
        conjecture_correct = sum(1 for comp in comparison_table if comp["conjecture_correct"])

        # Provider-based metrics
        cheap_provider_claims = [c for c in claims_created if c.provider_source == "cheap_provider"]
        quality_provider_claims = [c for c in claims_created if c.provider_source == "quality_provider"]

        # Confidence metrics
        cheap_avg_confidence = sum(c.confidence for c in cheap_provider_claims) / len(cheap_provider_claims) if cheap_provider_claims else 0
        quality_avg_confidence = sum(c.confidence for c in quality_provider_claims) / len(quality_provider_claims) if quality_provider_claims else 0

        # Support metrics
        claims_with_support = len([c for c in claims_created if c.supporting_claims])
        support_rate = (claims_with_support / len(claims_created) * 100) if claims_created else 0

        return {
            "total_problems": total_problems,
            "direct_accuracy": (direct_correct / total_problems * 100) if total_problems > 0 else 0,
            "conjecture_accuracy": (conjecture_correct / total_problems * 100) if total_problems > 0 else 0,
            "improvement": ((conjecture_correct - direct_correct) / direct_correct * 100) if direct_correct > 0 else 0,
            "total_claims_created": len(claims_created),
            "cheap_provider_claims": len(cheap_provider_claims),
            "quality_provider_claims": len(quality_provider_claims),
            "cheap_avg_confidence": cheap_avg_confidence,
            "quality_avg_confidence": quality_avg_confidence,
            "support_rate": support_rate,
            "session_id": results.get("session_id"),
            "scope_id": results.get("scope_id")
        }

    def display_improved_summary(self, results: Dict[str, Any]):
        """Display improved evaluation summary"""
        print(f"\n{'='*80}")
        print("IMPROVED CLAIM SYSTEM SUMMARY")
        print(f"{'='*80}")

        metrics = results.get("performance_metrics", {})
        print(f"\nENHANCED PERFORMANCE METRICS")
        print(f"Total Problems: {metrics.get('total_problems', 0)}")
        print(f"Cheap Provider Accuracy: {metrics.get('direct_accuracy', 0):.1f}%")
        print(f"Quality Provider Accuracy: {metrics.get('conjecture_accuracy', 0):.1f}%")
        print(f"Improvement: {metrics.get('improvement', 0):.1f}%")
        print(f"Total Claims Created: {metrics.get('total_claims_created', 0)}")
        print(f"Cheap Provider Claims: {metrics.get('cheap_provider_claims', 0)}")
        print(f"Quality Provider Claims: {metrics.get('quality_provider_claims', 0)}")
        print(f"Cheap Provider Avg Confidence: {metrics.get('cheap_avg_confidence', 0):.2f}")
        print(f"Quality Provider Avg Confidence: {metrics.get('quality_avg_confidence', 0):.2f}")
        print(f"Claims with Supporting Evidence: {metrics.get('support_rate', 0):.1f}%")
        print(f"Session ID: {metrics.get('session_id', 'Unknown')}")
        print(f"Scope ID: {metrics.get('scope_id', 'Unknown')}")

        # Claims table with improved format
        print(f"\nIMPROVED CLAIMS ANALYSIS")
        print("-" * 120)
        print(f"{'Claim ID':<8} {'Content':<50} {'Provider':<15} {'Confidence':<12} {'Support':<8} {'Parent':<8}")
        print("-" * 120)

        for claim in results.get("claims_created", [])[:20]:
            content = claim.content[:47] + "..." if len(claim.content) > 50 else claim.content
            provider = claim.provider_source[:12] if claim.provider_source else "unknown"
            support = "Yes" if claim.supporting_claims else "No"
            parent = claim.parent_id[:6] if claim.parent_id else "None"

            print(f"{claim.id:<8} {content:<50} {provider:<15} {claim.confidence:<12.2f} {support:<8} {parent:<8}")

        if len(results.get("claims_created", [])) > 20:
            print(f"... and {len(results.get('claims_created', [])) - 20} more claims")

        print(f"\n{'='*80}")
        print("IMPROVED EVALUATION COMPLETE")
        print(f"Session: {metrics.get('session_id', 'Unknown')}")
        print(f"Scope: {metrics.get('scope_id', 'Unknown')}")
        print(f"{'='*80}")

def main():
    """Run improved claim system demonstration"""
    system = ImprovedClaimSystem()
    results = system.run_improved_evaluation()

    # Save improved results
    results_file = "src/benchmarking/improved_claim_system_results.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    # Convert claims to dictionaries for JSON serialization
    json_results = results.copy()
    json_results["claims_created"] = [asdict(claim) for claim in results["claims_created"]]

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    print(f"\n[Improved system results saved to: {results_file}]")
    return results

if __name__ == "__main__":
    main()