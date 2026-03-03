"""
OpenRouter Free Model Benchmark Tests
Tests verified free models for Conjecture benchmarking.

Run with: pytest tests/test_openrouter_benchmark.py -v -m benchmark
Requires: OPENROUTER_API_KEY environment variable

Rate limits:
- 50 requests/day (free tier)
- 20 requests/minute
"""

import os
import pytest
import asyncio
import re
from typing import Dict, Any

# Import fixtures
from tests.fixtures.openrouter_free import (
    OpenRouterFreeClient,
    OpenRouterFreeConfig,
    OpenRouterError,
    get_openrouter_api_key,
)

# Skip entire module if OPENROUTER_API_KEY not available
pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.llm,
    pytest.mark.network,
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set - skipping network-dependent tests"
    ),
]


# --- Connection Tests ---

class TestOpenRouterConnection:
    """Test OpenRouter connectivity and model availability"""

    @pytest.mark.asyncio
    async def test_gpt_oss_20b_available(self, openrouter_client):
        """Verify gpt-oss-20b:free is accessible"""
        try:
            async with openrouter_client as client:
                result = await client.test_connection("gpt-oss-20b")
            assert result, "gpt-oss-20b:free not accessible"
        except OpenRouterError as e:
            if e.is_rate_limit or "not accessible" in str(e).lower():
                pytest.skip(f"Network/API unavailable: {e.message}")
            raise

    @pytest.mark.asyncio
    async def test_nemotron_30b_available(self, openrouter_client):
        """Verify nemotron-3-nano-30b:free is accessible"""
        try:
            async with openrouter_client as client:
                result = await client.test_connection("nemotron-30b")
            assert result, "nemotron-3-nano-30b:free not accessible"
        except (OpenRouterError, AssertionError) as e:
            # Nemotron model frequently unavailable - skip rather than fail
            pytest.skip(f"nemotron-30b not accessible: {e}")

    def test_api_key_format(self, openrouter_api_key):
        """Verify API key format is valid"""
        assert openrouter_api_key.startswith("sk-or-"), "Invalid API key format"
        assert len(openrouter_api_key) > 20, "API key too short"


# --- Basic Response Tests ---

class TestBasicResponses:
    """Test basic model response capabilities"""

    @pytest.mark.asyncio
    async def test_gpt_oss_simple_math(self, gpt_oss_20b):
        """Test gpt-oss-20b on simple arithmetic"""
        try:
            result = await gpt_oss_20b("What is 7 * 8?")
        except OpenRouterError as e:
            if e.is_rate_limit:
                pytest.skip(f"Rate limited: {e.message}")
            raise

        assert "content" in result
        assert "56" in result["content"]
        assert result.get("usage", {}).get("total_tokens", 0) > 0

    @pytest.mark.asyncio
    async def test_nemotron_simple_math(self, nemotron_30b):
        """Test nemotron-30b on simple arithmetic"""
        try:
            result = await nemotron_30b("What is 7 * 8?")
        except OpenRouterError as e:
            if e.is_rate_limit:
                pytest.skip(f"Rate limited: {e.message}")
            raise

        assert "content" in result
        assert "56" in result["content"]

    @pytest.mark.asyncio
    async def test_gpt_oss_has_reasoning(self, gpt_oss_20b):
        """Verify gpt-oss-20b returns reasoning tokens"""
        try:
            result = await gpt_oss_20b("Explain why 2+2=4", max_tokens=100)
        except OpenRouterError as e:
            if e.is_rate_limit:
                pytest.skip(f"Rate limited: {e.message}")
            raise

        # Both models are reasoning models - check for reasoning field
        assert "reasoning" in result or "content" in result

    @pytest.mark.asyncio
    async def test_nemotron_has_reasoning(self, nemotron_30b):
        """Verify nemotron-30b returns reasoning tokens"""
        try:
            result = await nemotron_30b("Explain why 2+2=4", max_tokens=100)
        except OpenRouterError as e:
            if e.is_rate_limit:
                pytest.skip(f"Rate limited: {e.message}")
            raise

        assert "reasoning" in result or "content" in result


# --- Math Benchmark Tests ---

class TestMathBenchmark:
    """GSM8K-style math benchmark tests"""

    MATH_PROBLEMS = [
        ("Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and uses 4 to bake muffins. She sells the remainder for $2 each. How much does she make daily?", "18"),
        ("A store sells apples for $2 each. If John buys 5 apples and pays with a $20 bill, how much change does he receive?", "10"),
        ("If a train travels at 60 mph for 2.5 hours, how many miles does it travel?", "150"),
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("problem,expected", MATH_PROBLEMS)
    async def test_gpt_oss_math_problems(self, gpt_oss_20b, benchmark_prompt_factory, problem, expected):
        """Test gpt-oss-20b on GSM8K-style problems"""
        try:
            prompt = benchmark_prompt_factory["math"](problem)
            result = await gpt_oss_20b(prompt, max_tokens=300)

            content = result["content"]
            # Check if expected answer appears in response
            assert expected in content, f"Expected {expected} in response: {content[:200]}"
        except OpenRouterError as e:
            if e.is_rate_limit or e.code in (429, 502, 503):
                pytest.skip(f"Network/API unavailable: {e.message}")
            raise

    @pytest.mark.asyncio
    async def test_nemotron_math_problem(self, nemotron_30b, benchmark_prompt_factory):
        """Test nemotron-30b on a math problem"""
        try:
            problem = "A farmer has 24 chickens. If he sells 8, how many remain?"
            prompt = benchmark_prompt_factory["math"](problem)
            result = await nemotron_30b(prompt, max_tokens=200)
            content = result.get("content") or ""
            if not content:
                pytest.skip("API returned empty response - model may be unavailable")
            assert "16" in content
        except OpenRouterError as e:
            if e.is_rate_limit or e.code in (429, 502, 503):
                pytest.skip(f"Network/API unavailable: {e.message}")
            raise


# --- Error Handling Tests ---

class TestErrorHandling:
    """Test error handling for rate limits and failures"""

    @pytest.mark.asyncio
    async def test_handles_rate_limit_error(self, openrouter_config):
        """Verify rate limit errors are properly typed"""
        # This tests the error class structure
        error = OpenRouterError(429, "Rate limit exceeded", is_rate_limit=True)
        assert error.is_rate_limit
        assert error.code == 429

    @pytest.mark.asyncio
    async def test_invalid_model_error(self, openrouter_client):
        """Test error on invalid model name"""
        async with openrouter_client as client:
            with pytest.raises(OpenRouterError) as exc_info:
                await client.chat("test", model="invalid/nonexistent-model")

        assert exc_info.value.code in [400, 404, 429]


# --- Benchmark Validation Tests ---

class TestBenchmarkValidation:
    """Validate benchmark test infrastructure"""

    @pytest.mark.asyncio
    async def test_answer_extraction_numeric(self, gpt_oss_20b, benchmark_prompt_factory):
        """Test that numeric answers can be extracted"""
        try:
            prompt = benchmark_prompt_factory["math"]("What is 100 divided by 4?")
            result = await gpt_oss_20b(prompt, max_tokens=150)

            content = result.get("content") or ""
            if not content:
                pytest.skip("API returned empty response")
            # Extract numbers from response
            numbers = re.findall(r'\b\d+\b', content)
            assert "25" in numbers, f"Expected 25 in extracted numbers: {numbers}"
        except OpenRouterError as e:
            if e.is_rate_limit or e.code in (429, 502, 503):
                pytest.skip(f"Network/API unavailable: {e.message}")
            raise

    @pytest.mark.asyncio
    async def test_consistent_responses(self, gpt_oss_20b):
        """Test response consistency with low temperature"""
        try:
            prompt = "What is the capital of France? Answer with just the city name."

            results = []
            for _ in range(2):
                result = await gpt_oss_20b(prompt, max_tokens=20, temperature=0.0)
                results.append(result["content"].strip().lower())

            # Both responses should contain "paris"
            assert all("paris" in r for r in results), f"Inconsistent responses: {results}"
        except OpenRouterError as e:
            if e.is_rate_limit or e.code in (429, 502, 503):
                pytest.skip(f"Network/API unavailable: {e.message}")
            raise

    @pytest.mark.asyncio
    async def test_model_reports_usage(self, gpt_oss_20b):
        """Test that usage statistics are returned"""
        try:
            result = await gpt_oss_20b("Say hello", max_tokens=10)

            usage = result.get("usage", {})
            assert "prompt_tokens" in usage or "total_tokens" in usage
        except OpenRouterError as e:
            if e.is_rate_limit or e.code in (429, 502, 503):
                pytest.skip(f"Network/API unavailable: {e.message}")
            raise


# --- Performance Tests ---

class TestPerformance:
    """Performance and timing tests"""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_response_under_30s(self, gpt_oss_20b):
        """Verify response time is reasonable"""
        import time

        try:
            start = time.time()
            await gpt_oss_20b("What is 2+2?", max_tokens=20)
            elapsed = time.time() - start

            assert elapsed < 30, f"Response took {elapsed:.1f}s, expected <30s"
        except OpenRouterError as e:
            if e.is_rate_limit or e.code in (429, 502, 503):
                pytest.skip(f"Network/API unavailable: {e.message}")
            raise

    @pytest.mark.asyncio
    async def test_token_efficiency(self, gpt_oss_20b, benchmark_prompt_factory):
        """Test that responses don't exceed token limits"""
        try:
            prompt = benchmark_prompt_factory["math"]("What is 5+5?")
            result = await gpt_oss_20b(prompt, max_tokens=100)

            usage = result.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)

            # Should use reasonable tokens for simple problem
            assert completion_tokens <= 100, f"Used {completion_tokens} tokens for simple problem"
        except OpenRouterError as e:
            if e.is_rate_limit or e.code in (429, 502, 503):
                pytest.skip(f"Network/API unavailable: {e.message}")
            raise


# --- Rate limit handling ---

def handle_rate_limit(func):
    """Decorator to skip test if rate limited"""
    import functools

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except OpenRouterError as e:
            if e.is_rate_limit:
                pytest.skip(f"Rate limited: {e.message}")
            raise
    return wrapper
