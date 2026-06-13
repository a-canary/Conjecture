# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Tests for LLM Client (Endpoint Layer)

Tests:
- CircuitBreaker state machine (O-0005)
- LLMClient initialization and configuration
- Prompt building utilities
"""

import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from src.endpoint.llm_client import (
    CircuitBreaker,
    LLMClient,
    DEFAULT_MODEL,
    TOOL_CAPABLE_MODEL,
    build_claim_context,
    build_enhanced_prompt,
)


# ---------------------------------------------------------------------------
# CircuitBreaker Tests (O-0005)
# ---------------------------------------------------------------------------

class TestCircuitBreakerStates:
    """Test circuit breaker state transitions."""

    def test_initial_state_is_closed(self):
        """Test circuit breaker starts in CLOSED state."""
        cb = CircuitBreaker()
        assert cb.state == "CLOSED"
        assert cb._failures == 0

    def test_can_execute_when_closed(self):
        """Test requests allowed when CLOSED."""
        cb = CircuitBreaker()
        assert cb.can_execute() is True

    def test_failure_increments_count(self):
        """Test recording failure increments counter."""
        cb = CircuitBreaker()
        cb.record_failure()
        assert cb._failures == 1
        assert cb.state == "CLOSED"  # Still closed after 1 failure

    def test_success_resets_failures(self):
        """Test success resets failure count."""
        cb = CircuitBreaker()
        cb.record_failure()
        cb.record_failure()
        assert cb._failures == 2

        cb.record_success()
        assert cb._failures == 0
        assert cb.state == "CLOSED"


class TestCircuitBreakerThreshold:
    """Test circuit breaker threshold behavior."""

    def test_opens_at_threshold(self):
        """Test circuit opens after reaching failure threshold."""
        cb = CircuitBreaker(failure_threshold=3)

        cb.record_failure()  # 1
        cb.record_failure()  # 2
        assert cb.state == "CLOSED"

        cb.record_failure()  # 3 - threshold reached
        assert cb.state == "OPEN"

    def test_default_threshold_is_5(self):
        """Test default threshold is 5 failures."""
        cb = CircuitBreaker()
        assert cb.failure_threshold == 5

        for _ in range(4):
            cb.record_failure()
        assert cb.state == "CLOSED"

        cb.record_failure()  # 5th failure
        assert cb.state == "OPEN"

    def test_cannot_execute_when_open(self):
        """Test requests blocked when OPEN."""
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()

        assert cb.state == "OPEN"
        assert cb.can_execute() is False


class TestCircuitBreakerRecovery:
    """Test circuit breaker recovery behavior."""

    def test_transitions_to_half_open_after_timeout(self):
        """Test circuit transitions to HALF_OPEN after recovery timeout."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        cb.record_failure()

        assert cb.state == "OPEN"
        assert cb.can_execute() is False

        # Wait for recovery timeout
        time.sleep(0.15)

        assert cb.can_execute() is True
        assert cb.state == "HALF_OPEN"

    def test_can_execute_when_half_open(self):
        """Test requests allowed in HALF_OPEN state."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()

        time.sleep(0.02)
        assert cb.can_execute() is True  # Transitions to HALF_OPEN
        assert cb.state == "HALF_OPEN"
        assert cb.can_execute() is True  # Still allowed in HALF_OPEN

    def test_success_in_half_open_closes_circuit(self):
        """Test success in HALF_OPEN state closes circuit."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()

        time.sleep(0.02)
        cb.can_execute()  # Trigger HALF_OPEN
        assert cb.state == "HALF_OPEN"

        cb.record_success()
        assert cb.state == "CLOSED"
        assert cb._failures == 0

    def test_failure_in_half_open_reopens_circuit(self):
        """Test failure in HALF_OPEN reopens circuit."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()

        time.sleep(0.02)
        cb.can_execute()  # Trigger HALF_OPEN
        assert cb.state == "HALF_OPEN"

        cb.record_failure()
        assert cb.state == "OPEN"


class TestCircuitBreakerConfiguration:
    """Test circuit breaker configuration options."""

    def test_custom_failure_threshold(self):
        """Test custom failure threshold."""
        cb = CircuitBreaker(failure_threshold=10)
        assert cb.failure_threshold == 10

    def test_custom_recovery_timeout(self):
        """Test custom recovery timeout."""
        cb = CircuitBreaker(recovery_timeout=120.0)
        assert cb.recovery_timeout == 120.0

    def test_state_property(self):
        """Test state property accessor."""
        cb = CircuitBreaker()
        assert cb.state == "CLOSED"

        cb._state = "OPEN"
        assert cb.state == "OPEN"


# ---------------------------------------------------------------------------
# LLMClient Tests
# ---------------------------------------------------------------------------

class TestLLMClientInit:
    """Test LLMClient initialization."""

    def test_default_model(self):
        """Test default model is set."""
        client = LLMClient()
        assert client.model == DEFAULT_MODEL

    def test_custom_model(self):
        """Test custom model setting."""
        client = LLMClient(model="custom-model")
        assert client.model == "custom-model"

    def test_default_base_url(self):
        """Test default base URL."""
        client = LLMClient()
        assert "chutes.ai" in client.base_url

    def test_custom_base_url(self):
        """Test custom base URL."""
        client = LLMClient(base_url="https://custom.api/v1")
        assert client.base_url == "https://custom.api/v1"

    def test_api_key_from_env(self):
        """Test API key from environment."""
        with patch.dict("os.environ", {"CHUTES_API_KEY": "test-key-123"}):
            client = LLMClient()
            assert client.api_key == "test-key-123"

    def test_explicit_api_key(self):
        """Test explicit API key overrides env."""
        client = LLMClient(api_key="explicit-key")
        assert client.api_key == "explicit-key"

    def test_custom_circuit_breaker(self):
        """Test custom circuit breaker injection."""
        cb = CircuitBreaker(failure_threshold=10)
        client = LLMClient(circuit_breaker=cb)
        assert client._circuit_breaker.failure_threshold == 10


class TestLLMClientGetClient:
    """Test _get_client method."""

    def test_raises_without_api_key(self):
        """Test raises ValueError without API key."""
        with patch.dict("os.environ", {}, clear=True):
            client = LLMClient(api_key=None)
            # Remove any existing API key
            client.api_key = None

            with pytest.raises(ValueError, match="No API key"):
                client._get_client()

    def test_creates_client_with_api_key(self):
        """Test creates AsyncOpenAI client when API key present."""
        client = LLMClient(api_key="test-key")
        openai_client = client._get_client()

        assert openai_client is not None
        assert client._client is openai_client  # Cached

    def test_client_is_cached(self):
        """Test client is cached after first creation."""
        client = LLMClient(api_key="test-key")
        client1 = client._get_client()
        client2 = client._get_client()

        assert client1 is client2


# ---------------------------------------------------------------------------
# Prompt Building Utilities
# ---------------------------------------------------------------------------

class TestBuildClaimContext:
    """Test build_claim_context utility."""

    def test_empty_claims(self):
        """Test empty claims returns empty string."""
        result = build_claim_context([])
        assert result == ""

    def test_single_claim(self):
        """Test formatting single claim."""
        claims = [
            {"content": "The sky is blue", "confidence": 0.9}
        ]
        result = build_claim_context(claims)

        assert "Relevant knowledge" in result
        assert "The sky is blue" in result
        assert "90%" in result

    def test_multiple_claims(self):
        """Test formatting multiple claims."""
        claims = [
            {"content": "First claim", "confidence": 0.8},
            {"content": "Second claim", "confidence": 0.7},
        ]
        result = build_claim_context(claims)

        assert "1." in result
        assert "2." in result
        assert "First claim" in result
        assert "Second claim" in result

    def test_default_confidence(self):
        """Test default confidence when not provided."""
        claims = [{"content": "No confidence"}]
        result = build_claim_context(claims)

        assert "50%" in result  # Default 0.5


class TestBuildEnhancedPrompt:
    """Test build_enhanced_prompt utility."""

    def test_with_context(self):
        """Test prompt with claim context."""
        context = "Some context"
        query = "What is 2+2?"

        result = build_enhanced_prompt(query, context)

        assert context in result
        assert query in result
        assert "Think step-by-step" in result
        assert "Based on the above knowledge" in result

    def test_without_context(self):
        """Test prompt without claim context."""
        query = "What is 2+2?"
        result = build_enhanced_prompt(query, "")

        assert query in result
        assert "Think step-by-step" in result
        assert "Based on the above knowledge" not in result

    def test_verification_instruction(self):
        """Test prompt includes verification instruction."""
        result = build_enhanced_prompt("Test", "Context")
        assert "verify your answer" in result.lower()


# ---------------------------------------------------------------------------
# Model Constants
# ---------------------------------------------------------------------------

class TestModelConstants:
    """Test model constant definitions."""

    def test_default_model_defined(self):
        """Test DEFAULT_MODEL is defined."""
        assert DEFAULT_MODEL is not None
        assert len(DEFAULT_MODEL) > 0

    def test_tool_capable_model_defined(self):
        """Test TOOL_CAPABLE_MODEL is defined."""
        assert TOOL_CAPABLE_MODEL is not None
        assert len(TOOL_CAPABLE_MODEL) > 0

    def test_models_are_different(self):
        """Test default and tool-capable models are different."""
        assert DEFAULT_MODEL != TOOL_CAPABLE_MODEL
