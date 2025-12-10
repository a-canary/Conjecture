"""
LLM Evaluation Framework for Conjecture
Implements 50-point rubric for systematic comparison of LLM integrations
"""

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ...core.basic_models import BasicClaim, ClaimState, ClaimType

@dataclass
class LLMEvaluationResult:
    """Single LLM evaluation metric result"""

    metric_name: str
    score: float
    max_score: float
    details: str
    passed_threshold: bool

@dataclass
class LLMProcessingResult:
    """Result from LLM processing operation"""

    success: bool
    processed_claims: List["BasicClaim"]
    errors: List[str]
    processing_time: float
    tokens_used: int
    model_used: str

@dataclass
class LLMEvaluation:
    """Complete evaluation for an LLM implementation"""

    llm_name: str
    total_score: float
    max_score: float
    percentage: float
    results: List[LLMEvaluationResult]
    performance_summary: Dict[str, float]
    recommendation: str
    cost_estimate: Optional[float] = None

class LLMInterface(ABC):
    """Abstract interface for all LLM implementations to test"""

    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """Initialize LLM with configuration"""
        pass

    @abstractmethod
    def process_exploration(
        self, context_claims: List[BasicClaim], query: str, max_new_claims: int = 5
    ) -> "LLMProcessingResult":
        """Process exploration request to generate new claims"""
        pass

    @abstractmethod
    def validate_claim(
        self, claim: BasicClaim, context_claims: List[BasicClaim]
    ) -> "LLMProcessingResult":
        """Validate and potentially update an existing claim"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        pass

    @abstractmethod
    def test_connectivity(self) -> bool:
        """Test API connectivity"""
        pass

    @abstractmethod
    def close(self):
        """Cleanup resources"""
        pass

class LLMEvaluator:
    """Systematic evaluator for LLM integrations"""

    def __init__(self):
        self.test_context = self._generate_test_context()
        self.test_queries = self._generate_test_queries()
        self.test_claims = self._generate_test_claims()

    def _generate_test_context(self) -> List[BasicClaim]:
        """Generate diverse test context covering different domains"""
        return [
            BasicClaim(
                id="ctx_1",
                content="Quantum entanglement enables instantaneous correlations between particles",
                confidence=0.95,
                type=[ClaimType.CONCEPT],
                tags=["quantum-physics", "science"],
                state=ClaimState.VALIDATED,
                created=datetime.now(),
            ),
            BasicClaim(
                id="ctx_2",
                content="Machine learning algorithms identify patterns in large datasets",
                confidence=0.90,
                type=[ClaimType.CONCEPT],
                tags=["ai", "computer-science"],
                state=ClaimState.VALIDATED,
                created=datetime.now(),
            ),
            BasicClaim(
                id="ctx_3",
                content="Photosynthesis in plants converts light energy into chemical energy",
                confidence=0.98,
                type=[ClaimType.CONCEPT],
                tags=["biology", "botany"],
                state=ClaimState.VALIDATED,
                created=datetime.now(),
            ),
            BasicClaim(
                id="ctx_4",
                content="Neural networks learn through backpropagation gradient descent",
                confidence=0.87,
                type=[ClaimType.CONCEPT],
                tags=["ai", "deep-learning"],
                state=ClaimState.VALIDATED,
                created=datetime.now(),
            ),
            BasicClaim(
                id="ctx_5",
                content="Shakespeare wrote Hamlet around 1600 as a tragedy",
                confidence=0.96,
                type=[ClaimType.REFERENCE],
                tags=["literature", "shakespeare"],
                state=ClaimState.VALIDATED,
                created=datetime.now(),
            ),
        ]

    def _generate_test_queries(self) -> List[str]:
        """Generate test queries for exploration"""
        return [
            "How does quantum entanglement relate to information theory?",
            "What are the limitations of current machine learning approaches?",
            "How do biological systems optimize energy conversion?",
            "What are the practical applications of deep learning?",
            "How does Shakespeare's work influence modern literature?",
        ]

    def _generate_test_claims(self) -> List[BasicClaim]:
        """Generate test claims for validation"""
        return [
            BasicClaim(
                id="test_claim_high",
                content="Quantum computing will revolutionize cryptography within 5 years",
                confidence=0.90,
                type=[ClaimType.THESIS],
                tags=["quantum-computing", "cryptography"],
                state=ClaimState.EXPLORE,
                created=datetime.now(),
            ),
            BasicClaim(
                id="test_claim_low_conf",
                content="All AI systems will achieve human-level intelligence by 2030",
                confidence=0.30,
                type=[ClaimType.THESIS],
                tags=["agi", "future-technology"],
                state=ClaimState.EXPLORE,
                created=datetime.now(),
            ),
            BasicClaim(
                id="test_claim_concept",
                content="Deep learning uses multiple layers to extract hierarchical features",
                confidence=0.95,
                type=[ClaimType.CONCEPT],
                tags=["ai", "neural-networks"],
                state=ClaimState.EXPLORE,
                created=datetime.now(),
            ),
        ]

    def evaluate_llm(self, llm_impl: LLMInterface, llm_name: str) -> LLMEvaluation:
        """Run complete evaluation on an LLM implementation"""
        print(f"\n=== Evaluating {llm_name} ===")
        results = []
        performance_summary = {}

        try:
            # Rubric Criterion 1: API Connection & Configuration (10 points)
            result1 = self._test_api_connection(llm_impl)
            results.append(result1)

            # Rubric Criterion 2: Exploration Processing (15 points)
            result2, perf_data = self._test_exploration_processing(llm_impl)
            results.append(result2)
            performance_summary.update(perf_data)

            # Rubric Criterion 3: Claim Validation (10 points)
            result3, val_perf_data = self._test_claim_validation(llm_impl)
            results.append(result3)
            performance_summary.update(val_perf_data)

            # Rubric Criterion 4: Output Quality & Format (10 points)
            result4 = self._test_output_quality(llm_impl)
            results.append(result4)

            # Rubric Criterion 5: Performance & Reliability (5 points)
            result5, reliability_data = self._test_performance_reliability(llm_impl)
            results.append(result5)
            performance_summary.update(reliability_data)

        except Exception as e:
            print(f"Evaluation failed for {llm_name}: {e}")
            results.append(
                LLMEvaluationResult(
                    "Overall Evaluation",
                    0,
                    50,
                    f"Critical error during evaluation: {e}",
                    False,
                )
            )

        # Calculate total score
        total_score = sum(r.score for r in results)
        max_score = sum(r.max_score for r in results)
        percentage = (total_score / max_score * 100) if max_score > 0 else 0

        # Generate recommendation
        recommendation = self._generate_recommendation(
            percentage, results, performance_summary
        )

        # Estimate cost for 1000 operations
        cost_estimate = self._estimate_cost_1000_ops(performance_summary)

        return LLMEvaluation(
            llm_name=llm_name,
            total_score=total_score,
            max_score=max_score,
            percentage=percentage,
            results=results,
            performance_summary=performance_summary,
            recommendation=recommendation,
            cost_estimate=cost_estimate,
        )

    def _test_api_connection(self, llm_impl: LLMInterface) -> LLMEvaluationResult:
        """Test LLM API connection and configuration"""
        score = 10
        issues = []

        try:
            # Test basic connectivity
            connected = llm_impl.test_connectivity()
            if not connected:
                score -= 5
                issues.append("API connectivity test failed")

            # Test stats retrieval
            stats = llm_impl.get_stats()
            if not stats:
                score -= 3
                issues.append("Cannot retrieve API statistics")

            # Test configuration validation
            if not hasattr(llm_impl, "model_name"):
                score -= 2
                issues.append("Model configuration not accessible")

        except Exception as e:
            score = 0
            issues.append(f"Connection error: {e}")

        return LLMEvaluationResult(
            "API Connection & Configuration",
            score,
            10,
            "Passed" if score >= 8 else f"Issues: {', '.join(issues)}",
            score >= 8,
        )

    def _test_exploration_processing(
        self, llm_impl: LLMInterface
    ) -> Tuple[LLMEvaluationResult, Dict[str, float]]:
        """Test exploration processing capabilities"""
        score = 15
        issues = []
        perf_data = {}

        try:
            total_claims = 0
            total_time = 0
            total_tokens = 0

            # Test with multiple queries
            for i, query in enumerate(self.test_queries[:3]):  # Test 3 queries
                start_time = time.time()

                result = llm_impl.process_exploration(
                    context_claims=self.test_context[:3],  # Use subset for speed
                    query=query,
                    max_new_claims=3,
                )

                processing_time = time.time() - start_time

                if not result.success:
                    score -= 3
                    issues.append(f"Query {i + 1} processing failed: {result.errors}")
                    continue

                # Check claim generation
                if not result.processed_claims:
                    score -= 2
                    issues.append(f"Query {i + 1} generated no claims")
                else:
                    total_claims += len(result.processed_claims)

                    # Validate claim quality
                    for claim in result.processed_claims:
                        if not claim.content or len(claim.content.strip()) < 10:
                            score -= 1
                            issues.append(f"Query {i + 1}: poor claim content quality")

                        if not (0.0 <= claim.confidence <= 1.0):
                            score -= 1
                            issues.append(f"Query {i + 1}: invalid confidence score")

                total_time += processing_time
                total_tokens += result.tokens_used

            # Calculate performance metrics
            if total_claims > 0:
                perf_data["avg_claim_generation_time"] = total_time / total_claims
                perf_data["avg_tokens_per_claim"] = total_tokens / total_claims
                perf_data["total_processing_time"] = total_time
                perf_data["total_tokens_used"] = total_tokens

                # Performance scoring
                avg_time = perf_data["avg_claim_generation_time"]
                if avg_time > 5.0:  # >5s per claim
                    score -= 3
                    issues.append("Slow claim generation")
                elif avg_time > 2.0:  # >2s per claim
                    score -= 1
                    issues.append("Moderate generation speed")

        except Exception as e:
            score = 0
            issues.append(f"Exploration processing error: {e}")

        return LLMEvaluationResult(
            "Exploration Processing",
            score,
            15,
            "Passed" if score >= 12 else f"Issues: {', '.join(issues)}",
            score >= 12,
        ), perf_data

    def _test_claim_validation(
        self, llm_impl: LLMInterface
    ) -> Tuple[LLMEvaluationResult, Dict[str, float]]:
        """Test claim validation capabilities"""
        score = 10
        issues = []
        val_perf_data = {}

        try:
            total_validations = 0
            total_time = 0
            total_tokens = 0

            # Test validation of different types of claims
            for claim in self.test_claims:
                start_time = time.time()

                result = llm_impl.validate_claim(claim, self.test_context)

                processing_time = time.time() - start_time

                if not result.success:
                    score -= 2
                    issues.append(f"Validation failed for {claim.id}")
                    continue

                # Check validation result
                if not result.processed_claims:
                    score -= 1
                    issues.append(f"No validation result for {claim.id}")

                total_validations += 1
                total_time += processing_time
                total_tokens += result.tokens_used

            # Calculate performance metrics
            if total_validations > 0:
                val_perf_data["avg_validation_time"] = total_time / total_validations
                val_perf_data["avg_tokens_per_validation"] = (
                    total_tokens / total_validations
                )
                val_perf_data["total_validation_time"] = total_time

                # Performance scoring
                avg_time = val_perf_data["avg_validation_time"]
                if avg_time > 3.0:  # >3s per validation
                    score -= 2
                    issues.append("Slow validation processing")

        except Exception as e:
            score = 0
            issues.append(f"Claim validation error: {e}")

        return LLMEvaluationResult(
            "Claim Validation",
            score,
            10,
            "Passed" if score >= 8 else f"Issues: {', '.join(issues)}",
            score >= 8,
        ), val_perf_data

    def _test_output_quality(self, llm_impl: LLMInterface) -> LLMEvaluationResult:
        """Test output quality and format compliance"""
        score = 10
        issues = []

        try:
            # Test exploration output format
            result = llm_impl.process_exploration(
                context_claims=self.test_context[:2],
                query="Test query for format validation",
                max_new_claims=2,
            )

            if result.success and result.processed_claims:
                for claim in result.processed_claims:
                    # Check claim structure
                    if not hasattr(claim, "id") or not claim.id:
                        score -= 2
                        issues.append("Missing claim ID")

                    if not hasattr(claim, "content") or not claim.content.strip():
                        score -= 2
                        issues.append("Missing or empty claim content")

                    if not hasattr(claim, "confidence") or not isinstance(
                        claim.confidence, float
                    ):
                        score -= 2
                        issues.append("Invalid confidence value")

                    if not hasattr(claim, "type") or not claim.type:
                        score -= 2
                        issues.append("Missing claim type")

                    if not hasattr(claim, "state"):
                        score -= 2
                        issues.append("Missing claim state")
            else:
                score -= 5
                issues.append("Failed to generate output for quality testing")

        except Exception as e:
            score = 0
            issues.append(f"Output quality test error: {e}")

        return LLMEvaluationResult(
            "Output Quality & Format",
            score,
            10,
            "Passed" if score >= 8 else f"Issues: {', '.join(issues)}",
            score >= 8,
        )

    def _test_performance_reliability(
        self, llm_impl: LLMInterface
    ) -> Tuple[LLMEvaluationResult, Dict[str, float]]:
        """Test performance consistency and reliability"""
        score = 5
        issues = []
        reliability_data = {}

        try:
            # Test multiple rapid requests
            response_times = []
            successful_requests = 0

            for i in range(5):  # 5 rapid requests
                start_time = time.time()

                result = llm_impl.process_exploration(
                    context_claims=self.test_context[:1],  # Minimal context for speed
                    query=f"Reliability test query {i + 1}",
                    max_new_claims=1,
                )

                response_time = time.time() - start_time
                response_times.append(response_time)

                if result.success:
                    successful_requests += 1

            # Calculate reliability metrics
            reliability = successful_requests / 5.0
            reliability_data["reliability_rate"] = reliability
            reliability_data["avg_response_time"] = sum(response_times) / len(
                response_times
            )
            reliability_data["max_response_time"] = max(response_times)

            if reliability < 0.8:  # <80% success rate
                score -= 2
                issues.append(f"Low reliability: {reliability:.1%}")

            if reliability_data["max_response_time"] > 10.0:  # >10s
                score -= 2
                issues.append(f"High response time variability")

            # Check stats consistency
            stats = llm_impl.get_stats()
            if stats:
                reliability_data["api_stats_available"] = True
            else:
                score -= 1
                issues.append("API statistics not consistently available")

        except Exception as e:
            score = 0
            issues.append(f"Performance reliability test error: {e}")

        return LLMEvaluationResult(
            "Performance & Reliability",
            score,
            5,
            "Passed" if score >= 4 else f"Issues: {', '.join(issues)}",
            score >= 4,
        ), reliability_data

    def _generate_recommendation(
        self,
        percentage: float,
        results: List[LLMEvaluationResult],
        perf_data: Dict[str, float],
    ) -> str:
        """Generate recommendation based on evaluation results"""
        if percentage >= 90:
            return "HIGHLY RECOMMENDED - Excellent performance across all criteria"
        elif percentage >= 80:
            return "RECOMMENDED - Good performance with minor limitations"
        elif percentage >= 70:
            return "CONDITIONAL - Suitable for development, may need optimization"
        elif percentage >= 60:
            return "NOT RECOMMENDED FOR PRODUCTION - Significant limitations"
        else:
            return "NOT SUITABLE - Fails critical requirements"

    def _estimate_cost_1000_ops(self, perf_data: Dict[str, float]) -> Optional[float]:
        """Estimate cost for 1000 operations based on usage patterns"""
        try:
            # Calculate tokens per operation
            avg_tokens = perf_data.get("avg_tokens_per_claim", 0) + perf_data.get(
                "avg_tokens_per_validation", 0
            )

            # Estimate: 1000 operations (600 exploration + 400 validation)
            total_tokens = 1000 * avg_tokens

            # Rough pricing estimates (varies by provider)
            # Gemini: ~$0.00025 per 1K tokens (input) + $0.0005 per 1K tokens (output)
            # Using average of ~$0.0004 per 1K tokens
            estimated_cost = (total_tokens / 1000) * 0.0004

            return estimated_cost
        except:
            return None

    def print_evaluation_report(self, evaluation: LLMEvaluation):
        """Print detailed evaluation report"""
        print(f"\n{'=' * 60}")
        print(f"LLM EVALUATION REPORT: {evaluation.llm_name}")
        print(f"{'=' * 60}")
        print(
            f"Total Score: {evaluation.total_score}/{evaluation.max_score} ({evaluation.percentage:.1f}%)"
        )
        print(f"Recommendation: {evaluation.recommendation}")

        if evaluation.cost_estimate:
            print(f"Estimated Cost (1000 ops): ${evaluation.cost_estimate:.4f}")

        print(f"\nDETAILED RESULTS:")
        print("-" * 60)
        for result in evaluation.results:
            status = "✅ PASS" if result.passed_threshold else "❌ FAIL"
            print(
                f"{result.metric_name:25} | {result.score:.1f}/{result.max_score} | {status}"
            )
            if result.score < result.max_score:
                print(f"{'':27} | {result.details}")

        if evaluation.performance_summary:
            print(f"\nPERFORMANCE SUMMARY:")
            print("-" * 60)
            for metric, value in evaluation.performance_summary.items():
                if "time" in metric:
                    unit = "s" if value > 1 else "ms"
                    display_value = value if value > 1 else value * 1000
                    print(f"{metric:25} | {display_value:.3f} {unit}")
                elif "rate" in metric:
                    print(f"{metric:25} | {value:.1%}")
                else:
                    print(f"{metric:25} | {value}")

        print(f"\n{'=' * 60}\n")

def compare_llm_evaluations(
    evaluations: List[LLMEvaluation],
) -> Dict[str, LLMEvaluation]:
    """Compare multiple LLM evaluations and rank them"""
    rankings = {}

    # Sort by percentage score
    sorted_evaluations = sorted(evaluations, key=lambda x: x.percentage, reverse=True)

    print(f"\n{'=' * 80}")
    print(f"LLM COMPARISON RANKING")
    print(f"{'=' * 80}")

    for i, evaluation in enumerate(sorted_evaluations, 1):
        cost_str = (
            f"${evaluation.cost_estimate:.4f}" if evaluation.cost_estimate else "N/A"
        )
        print(
            f"{i}. {evaluation.llm_name:20} | {evaluation.percentage:5.1f}% | {evaluation.total_score}/{evaluation.max_score} | Cost/1k: {cost_str}"
        )
        print(f"{'':3} {evaluation.recommendation}")

    return {eval.llm_name: eval for eval in sorted_evaluations}
