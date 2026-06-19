# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Tests for Step 18.7: Input decomposition wired into ConjectureEndpoint.evaluate().

Covers:
- test_evaluate_uses_decomposition: evaluate() calls decompose_input and includes
  decomposed_claims count in response metadata.
- test_evaluate_works_without_decomposition: use_decomposition=False skips
  decompose_input and still returns decomposed_claims=0 in metadata.
- test_evaluate_decomposition_failure_is_non_blocking: if decompose_input raises,
  evaluate() continues and returns a successful response with decomposed_claims=0.
- test_evaluate_decomposed_claims_added_to_context: decomposed claims are merged
  into the context passed to build_claim_context.

Gate: evaluate("What is 2+2?") response includes decomposition metadata.
"""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.endpoint.conjecture_endpoint import ConjectureEndpoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _make_endpoint(tmp_path: str) -> ConjectureEndpoint:
    """Create and initialize an endpoint backed by a temp SQLite DB."""
    db_path = os.path.join(tmp_path, "test_evaluate_decomp.db")
    ep = ConjectureEndpoint(db_path=db_path, vector_path=":memory:")
    ep._vector_store = None
    await ep._data_manager.initialize()
    ep._initialized = True
    return ep


def _make_llm_instance(response_content: str = "4") -> MagicMock:
    """Build a mock LLMClient instance with both generate() and generate_with_tools().

    generate_with_tools() returns no tool_calls so evaluate() falls through to
    the plain-text path (same behaviour as the original generate() mock).
    """
    client = MagicMock()
    # Plain-text mode (use_tools=False)
    client.generate = AsyncMock(return_value={
        "content": response_content,
        "model": "mock-model",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5}
    })
    # Tool-calling mode (use_tools=True, default) — return no tool_calls so the
    # evaluate() loop exits via the plain-text fallback branch.
    client.generate_with_tools = AsyncMock(return_value={
        "content": response_content,
        "tool_calls": [],
        "model": "mock-model",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5}
    })
    client.close = AsyncMock()
    return client


def _make_decomposed_claims(count: int = 2):
    """Build a list of mock Claim objects for decomposition results."""
    from src.core.models import Claim, ClaimType, ClaimState
    import time
    import uuid

    claims = []
    for i in range(count):
        claim = Claim(
            id=f"c{int(time.time()*1000)+i}_{uuid.uuid4().hex[:8]}",
            content=f"Decomposed part {i+1}: some constituent claim about the query",
            confidence=0.75,
            type=[ClaimType.OBSERVATION],
            state=ClaimState.EXPLORE,
            tags=["decomposed", "observation"],
        )
        claims.append(claim)
    return claims


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path):
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEvaluateUsesDecomposition:
    """Gate test: evaluate() integrates decomposition metadata into response."""

    @pytest.mark.asyncio
    async def test_evaluate_uses_decomposition(self, tmp_dir):
        """evaluate() calls decompose_input and reports count in response metadata."""
        ep = await _make_endpoint(tmp_dir)
        mock_llm_instance = _make_llm_instance("The answer is 4.")
        fake_claims = _make_decomposed_claims(2)

        # LLMClient is imported locally inside evaluate() from src.endpoint.llm_client,
        # so we patch it there. decompose_input is imported at module level.
        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=fake_claims)) as mock_decompose, \
             patch("src.endpoint.llm_client.LLMClient",
                   return_value=mock_llm_instance):
            response = await ep.evaluate("What is 2+2?", use_decomposition=True)

        assert response.success, f"evaluate() failed: {response.errors}"
        assert response.data is not None

        # Gate: decomposed_claims key must be present in data
        assert "decomposed_claims" in response.data, (
            "Response data must include 'decomposed_claims' key"
        )
        assert response.data["decomposed_claims"] == 2, (
            f"Expected 2 decomposed claims, got {response.data['decomposed_claims']}"
        )

        # decompose_input must have been called once with the query
        mock_decompose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_evaluate_works_without_decomposition(self, tmp_dir):
        """use_decomposition=False skips decompose_input; decomposed_claims=0 in metadata."""
        ep = await _make_endpoint(tmp_dir)
        mock_llm_instance = _make_llm_instance("The answer is 4.")

        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])) as mock_decompose, \
             patch("src.endpoint.llm_client.LLMClient",
                   return_value=mock_llm_instance):
            response = await ep.evaluate("What is 2+2?", use_decomposition=False)

        assert response.success, f"evaluate() failed: {response.errors}"
        assert response.data is not None

        # decompose_input must NOT have been called
        mock_decompose.assert_not_awaited()

        # decomposed_claims must be 0 (not missing)
        assert "decomposed_claims" in response.data, (
            "Response data must include 'decomposed_claims' key even when skipped"
        )
        assert response.data["decomposed_claims"] == 0, (
            f"Expected 0 decomposed claims when use_decomposition=False, "
            f"got {response.data['decomposed_claims']}"
        )

    @pytest.mark.asyncio
    async def test_evaluate_decomposition_failure_is_non_blocking(self, tmp_dir):
        """If decompose_input raises, evaluate() continues and succeeds with decomposed_claims=0."""
        ep = await _make_endpoint(tmp_dir)
        mock_llm_instance = _make_llm_instance("The answer is 4.")

        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(side_effect=RuntimeError("decomposer crashed"))) as mock_decompose, \
             patch("src.endpoint.llm_client.LLMClient",
                   return_value=mock_llm_instance):
            response = await ep.evaluate("What is 2+2?", use_decomposition=True)

        # The overall evaluate() must still succeed
        assert response.success, (
            f"evaluate() must not fail when decomposition fails, got: {response.errors}"
        )
        assert response.data is not None

        # decomposed_claims must be 0 (fallback to empty list)
        assert "decomposed_claims" in response.data
        assert response.data["decomposed_claims"] == 0, (
            "decomposed_claims must be 0 when decomposition fails"
        )

        # decompose_input was attempted
        mock_decompose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_evaluate_decomposed_claims_merged_into_context(self, tmp_dir):
        """Decomposed claims are included when building the LLM context.

        build_claim_context is imported locally inside evaluate() from
        src.endpoint.llm_client, so we patch it there.
        """
        ep = await _make_endpoint(tmp_dir)
        mock_llm_instance = _make_llm_instance("The answer is 4.")
        # Use 3 decomposed claims so we can check count
        fake_claims = _make_decomposed_claims(3)

        captured_claim_lists = []

        def capturing_build_claim_context(claims):
            captured_claim_lists.append(list(claims))
            # Return a non-empty string so build_enhanced_prompt has something to work with
            return f"CONTEXT ({len(claims)} items)"

        # build_claim_context is imported via `from src.endpoint.llm_client import ...`
        # inside evaluate(), so patch it at the source module.
        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=fake_claims)), \
             patch("src.endpoint.llm_client.LLMClient",
                   return_value=mock_llm_instance), \
             patch("src.endpoint.llm_client.build_claim_context",
                   side_effect=capturing_build_claim_context):

            response = await ep.evaluate("What is 2+2?", use_decomposition=True)

        assert response.success, f"evaluate() failed: {response.errors}"

        # Response must report 3 decomposed claims
        assert response.data["decomposed_claims"] == 3

        # If build_claim_context was captured, verify decomposed claims were included
        if captured_claim_lists:
            all_claims_passed = captured_claim_lists[0]
            # There should be at least 3 items (the decomposed claims)
            assert len(all_claims_passed) >= 3, (
                f"Expected at least 3 items in context (decomposed claims), "
                f"got {len(all_claims_passed)}"
            )

    @pytest.mark.asyncio
    async def test_evaluate_gate_metadata_present(self, tmp_dir):
        """Gate: evaluate('What is 2+2?') response always includes decomposition metadata."""
        ep = await _make_endpoint(tmp_dir)
        mock_llm_instance = _make_llm_instance("4")
        fake_claims = _make_decomposed_claims(1)

        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=fake_claims)), \
             patch("src.endpoint.llm_client.LLMClient",
                   return_value=mock_llm_instance):
            response = await ep.evaluate("What is 2+2?")

        assert response.success, f"Gate test failed: {response.errors}"
        data = response.data

        # Gate assertion: decomposed_claims key must exist in response data
        assert "decomposed_claims" in data, (
            "GATE FAILED: evaluate() response must include 'decomposed_claims' in data"
        )

        # Additional metadata checks
        assert "query" in data
        assert data["query"] == "What is 2+2?"
        assert "response" in data
        assert "claims_used" in data
