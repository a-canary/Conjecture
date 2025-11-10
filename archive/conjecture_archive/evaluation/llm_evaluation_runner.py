"""
LLM Evaluation Runner for Conjecture
Tests all available LLM API implementations and generates comparison report
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_llm_dependency(name):
    """Check if an LLM dependency is available"""
    try:
        if name == "google-generativeai":
            import google.generativeai

            return True, google.generativeai.__version__
        elif name == "openai":
            import openai

            return True, openai.__version__
        elif name == "anthropic":
            import anthropic

            return True, anthropic.__version__
        else:
            return False, None
    except ImportError:
        return False, None


def create_mock_llm_interface():
    """Create mock LLM interface for testing without API calls"""
    import random
    import time
    from datetime import datetime

    from src.core.basic_models import BasicClaim, ClaimState, ClaimType
    from src.processing.llm.llm_evaluation_framework import (
        LLMInterface,
        LLMProcessingResult,
    )

    class MockLLMInterface(LLMInterface):
        """Mock LLM implementation for testing without API dependencies"""

        def __init__(self, config):
            self.config = config
            self.model_name = config.get("model_name", "mock-llm")
            self.stats = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_tokens": 0,
                "total_processing_time": 0.0,
            }

        def process_exploration(self, context_claims, query, max_new_claims=5):
            """Mock exploration processing"""
            start_time = time.time()
            self.stats["total_requests"] += 1

            try:
                # Generate mock claims based on query
                generated_claims = []

                # Mock claim templates based on common patterns
                claim_templates = [
                    (
                        "concept",
                        0.85,
                        f"Based on '{query}', this represents a conceptual advancement",
                    ),
                    ("example", 0.75, f"Example of {query} in practical application"),
                    ("skill", 0.80, f"Skill related to implementing {query}"),
                    ("reference", 0.90, f"Reference literature discussing {query}"),
                    ("thesis", 0.70, f"Thesis: {query} enables new capabilities"),
                ]

                for i in range(min(max_new_claims, len(claim_templates))):
                    claim_type, confidence, content = claim_templates[i]

                    claim = BasicClaim(
                        id=f"mock_gen_{int(time.time() * 1000)}_{i}",
                        content=content,
                        confidence=confidence + random.uniform(-0.1, 0.1),
                        type=[ClaimType(claim_type)],
                        tags=["mock-generated", "test"],
                        state=ClaimState.EXPLORE,
                        created=datetime.now(),
                    )
                    generated_claims.append(claim)

                processing_time = time.time() - start_time
                tokens_used = len(query) + sum(len(c.content) for c in generated_claims)

                self.stats["successful_requests"] += 1
                self.stats["total_tokens"] += tokens_used
                self.stats["total_processing_time"] += processing_time

                return LLMProcessingResult(
                    success=True,
                    processed_claims=generated_claims,
                    errors=[],
                    processing_time=processing_time,
                    tokens_used=tokens_used,
                    model_used=self.model_name,
                )

            except Exception as e:
                self.stats["failed_requests"] += 1
                processing_time = time.time() - start_time

                return LLMProcessingResult(
                    success=False,
                    processed_claims=[],
                    errors=[f"Mock processing failed: {e}"],
                    processing_time=processing_time,
                    tokens_used=0,
                    model_used=self.model_name,
                )

        def validate_claim(self, claim, context_claims):
            """Mock claim validation"""
            start_time = time.time()
            self.stats["total_requests"] += 1

            try:
                # Mock validation logic
                if claim.confidence > 0.8:
                    # High confidence claims get validated
                    validated_claims = [claim]
                    tokens_used = len(claim.content) + 50
                elif claim.confidence < 0.4:
                    # Low confidence claims get adjusted
                    adjusted_claim = BasicClaim(
                        id=claim.id,
                        content=claim.content,
                        confidence=0.5,  # Adjusted confidence
                        type=claim.type,
                        tags=claim.tags,
                        state=ClaimState.EXPLORE,
                        created=claim.created,
                    )
                    validated_claims = [adjusted_claim]
                    tokens_used = len(claim.content) + 75
                else:
                    # Medium confidence claims are accepted as-is
                    validated_claims = [claim]
                    tokens_used = len(claim.content) + 25

                processing_time = time.time() - start_time

                self.stats["successful_requests"] += 1
                self.stats["total_tokens"] += tokens_used
                self.stats["total_processing_time"] += processing_time

                return LLMProcessingResult(
                    success=True,
                    processed_claims=validated_claims,
                    errors=[],
                    processing_time=processing_time,
                    tokens_used=tokens_used,
                    model_used=self.model_name,
                )

            except Exception as e:
                self.stats["failed_requests"] += 1
                processing_time = time.time() - start_time

                return LLMProcessingResult(
                    success=False,
                    processed_claims=[],
                    errors=[f"Mock validation failed: {e}"],
                    processing_time=processing_time,
                    tokens_used=0,
                    model_used=self.model_name,
                )

        def get_stats(self):
            """Get processing statistics"""
            success_rate = 0.0
            if self.stats["total_requests"] > 0:
                success_rate = (
                    self.stats["successful_requests"] / self.stats["total_requests"]
                )

            avg_processing_time = 0.0
            if self.stats["successful_requests"] > 0:
                avg_processing_time = (
                    self.stats["total_processing_time"]
                    / self.stats["successful_requests"]
                )

            return {
                **self.stats,
                "success_rate": success_rate,
                "average_processing_time": avg_processing_time,
                "model_name": self.model_name,
                "api_available": True,
            }

        def test_connectivity(self):
            """Mock connectivity test"""
            return True  # Mock always available

        def close(self):
            """Mock cleanup"""
            pass

    return MockLLMInterface


def test_gemini_integration():
    """Test Gemini API integration if available"""
    gemini_available, version = check_llm_dependency("google-generativeai")

    if not gemini_available:
        return False, {}, "Google Generative AI library not installed"

    print(f"Testing Gemini API {version}...")

    try:
        # Check for API key from environment variables
        api_keys = {
            "gemini": os.getenv("GEMINI_API_KEY"),
        }

        if api_keys["gemini"]:
            from src.processing.llm.gemini_integration import GeminiProcessor
            from src.processing.llm.llm_evaluation_framework import LLMEvaluator

            # Initialize Gemini processor
            processor = GeminiProcessor(api_keys["gemini"], "gemini-1.5-flash")

            # Test connectivity
            if not processor.test_connectivity():
                return False, {}, "Gemini API connectivity test failed"

            # Run evaluation
            evaluator = LLMEvaluator()
            evaluation = evaluator.evaluate_llm(processor, "Gemini API")

            processor.close()

            # Return structured results
            return (
                True,
                {
                    "score": evaluation.percentage,
                    "total_score": evaluation.total_score,
                    "max_score": evaluation.max_score,
                    "processing_stats": evaluation.performance_summary,
                    "cost_estimate": evaluation.cost_estimate,
                },
                "Gemini API working correctly",
            )

        else:
            return False, {}, "Gemini API key not configured"

    except Exception as e:
        return False, {}, f"Gemini API error: {e}"


def test_mock_llm():
    """Test Mock LLM implementation"""
    print("Testing Mock LLM Implementation...")

    try:
        from src.processing.llm.llm_evaluation_framework import LLMEvaluator

        # Create mock interface
        mock_interface = create_mock_llm_interface()({"model_name": "mock-llm-v1"})

        # Run evaluation
        evaluator = LLMEvaluator()
        evaluation = evaluator.evaluate_llm(mock_interface, "Mock LLM")

        # Return structured results
        return (
            True,
            {
                "score": evaluation.percentage,
                "total_score": evaluation.total_score,
                "max_score": evaluation.max_score,
                "processing_stats": evaluation.performance_summary,
                "cost_estimate": 0.0,  # Free
            },
            "Mock LLM working correctly",
        )

    except Exception as e:
        return False, {}, f"Mock LLM error: {e}"


def calculate_llm_score(success, stats, criteria):
    """Calculate score based on LLM test results"""
    if not success:
        return 0

    score = 0
    score_float = stats.get("score", 0)

    # Convert percentage to 50-point scale
    score = (score_float / 100.0) * 50

    return round(score, 1)


def run_llm_evaluation():
    """Run complete evaluation of all available LLMs"""
    print("=" * 80)
    print("Conjecture LLM API EVALUATION")
    print("=" * 80)
    print("Success Criteria: ≥45/50 points, <2s average processing time")
    print("=" * 80)

    # Check dependencies first
    print("\n📦 DEPENDENCY CHECK:")
    print("-" * 40)

    deps = ["google-generativeai", "openai", "anthropic"]
    for dep in deps:
        available, version = check_llm_dependency(dep)
        status = "✅" if available else "❌"
        version_str = f" ({version})" if version else ""
        print(f"{status} {dep}{version_str}")

    # Test implementations
    print("\n🧪 TESTING LLM IMPLEMENTATIONS:")
    print("-" * 40)

    results = []

    # Test 1: Mock LLM (always available)
    mock_success, mock_stats, mock_msg = test_mock_llm()
    mock_score = calculate_llm_score(mock_success, mock_stats, 50)
    results.append(
        {
            "name": "Mock LLM",
            "success": mock_success,
            "score": mock_score,
            "stats": mock_stats,
            "message": mock_msg,
        }
    )

    # Test 2: Gemini API
    gemini_success, gemini_stats, gemini_msg = test_gemini_integration()
    gemini_score = calculate_llm_score(gemini_success, gemini_stats, 50)
    results.append(
        {
            "name": "Gemini API",
            "success": gemini_success,
            "score": gemini_score,
            "stats": gemini_stats,
            "message": gemini_msg,
        }
    )

    # Results summary
    print("\n📊 RESULTS SUMMARY:")
    print("-" * 40)

    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"{result['name']:<15} | {result['score']:>4.1f}/50 | {status}")
        if not result["success"]:
            print(f"{'':17} | {result['message']}")
        else:
            cost = result["stats"].get("cost_estimate", 0)
            cost_str = f"${cost:.4f}" if cost > 0 else "FREE"
            print(f"{'':17} | Cost/1k: {cost_str}")

    # Find best option
    successful = [r for r in results if r["success"]]

    if successful:
        best = max(successful, key=lambda x: x["score"])
        print(f"\n🏆 BEST OPTION: {best['name']}")
        print(f"   Score: {best['score']:.1f}/50")

        if best["score"] >= 45:
            print("   Status: ✅ MEETS SUCCESS CRITERIA")
        elif best["score"] >= 35:
            print("   Status: ⚠️  CLOSE TO CRITERIA")
        else:
            print("   Status: ❌ BELOW CRITERIA")

    # Detailed performance analysis
    print(f"\n💡 PERFORMANCE ANALYSIS:")

    if gemini_success:
        print("✅ Gemini API is available and working")
        print("   Recommended for production use with real AI capabilities")
    else:
        print("❌ Gemini API not available")
        print("   Install with: pip install google-generativeai")
        print("   Configure API key for full functionality")

    print("✅ Mock LLM always available")
    print("   Recommended for development and testing")

    # Processing performance comparison
    working_results = [r for r in results if r["success"]]
    if len(working_results) > 1:
        print(f"\n⚡ PERFORMANCE COMPARISON:")
        print("-" * 40)
        for result in working_results:
            stats = result["stats"]
            proc_time = stats.get("processing_stats", {}).get(
                "avg_claim_generation_time", 0
            )
            time_str = f"{proc_time:.2f}s" if proc_time > 0 else "N/A"

            cost = stats.get("cost_estimate", 0)
            cost_str = f"${cost:.4f}" if cost > 0 else "FREE"

            print(f"{result['name']:<15} | Time: {time_str:>7} | Cost/1k: {cost_str}")

    # Next steps recommendation
    best_available = max(
        [r for r in results if r["success"]], key=lambda x: x["score"], default=None
    )

    if best_available:
        print(f"\n🚀 NEXT STEPS:")
        print(f"1. Use {best_available['name']} for Conjecture development")

        if "Mock" in best_available["name"]:
            print("2. Install Gemini API for production capabilities")
            print("3. Configure API key for real AI processing")
        else:
            print("2. Test with real Conjecture claim processing workflows")
            print("3. Optimize prompts and processing parameters")

        print("4. Move to Phase 3: Terminal User Interface Development")

    # Cost analysis
    if any(r["stats"].get("cost_estimate", 0) > 0 for r in results if r["success"]):
        print(f"\n💰 COST ANALYSIS:")
        print("-" * 40)
        for result in results:
            if result["success"]:
                cost = result["stats"].get("cost_estimate", 0)
                if cost > 0:
                    monthly_cost = cost * 100  # Estimate 100k operations/month
                    print(
                        f"{result['name']:<15} | ${cost:.4f}/1k ops | ~${monthly_cost:.2f}/month"
                    )
                else:
                    print(f"{result['name']:<15} | FREE")

    print("\n" + "=" * 80)
    print("LLM EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_llm_evaluation()
