"""
Integration test for Step 20.4 and 20.6: ReasoningLoop wired into evaluate().

Gate: evaluate(query, use_reasoning_loop=True) returns an APIResponse whose
      data includes a 'reasoning_result' dict with the reasoning trace.

Tests:
  - test_evaluate_with_reasoning_loop_returns_reasoning_result
      Calls evaluate("What is 2+2?", use_reasoning_loop=True) with a mocked
      LLM that immediately responds.  Verifies the response includes a
      reasoning_result dict with halted_reason, tool_calls, and iterations.

  - test_evaluate_reasoning_loop_two_iterations
      LLM explores for one iteration (create_claim) then halts
      (respond_to_user).  Verifies iterations==2 and claims_created is
      non-empty.

  - test_evaluate_reasoning_loop_false_skips_loop
      use_reasoning_loop=False (default) does NOT produce reasoning_result
      key in response data.

  - test_evaluate_reasoning_loop_includes_response_text
      The top-level response key in data matches the text returned by
      respond_to_user.
"""

from __future__ import annotations

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.endpoint.conjecture_endpoint import ConjectureEndpoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _make_endpoint(tmp_path: str) -> ConjectureEndpoint:
    """Create and initialize an endpoint backed by a temp SQLite DB."""
    db_path = os.path.join(tmp_path, "test_reasoning.db")
    ep = ConjectureEndpoint(db_path=db_path, vector_path=":memory:")
    ep._vector_store = None
    await ep._data_manager.initialize()
    ep._initialized = True
    return ep


def _mock_llm_with_responses(responses: list) -> MagicMock:
    """Build a mock LLMClient that returns pre-canned generate_with_tools responses.

    Args:
        responses: List of response dicts, each returned in order from
            generate_with_tools.  Exhausting the list raises StopAsyncIteration.

    Returns:
        MagicMock with generate_with_tools, generate, and close mocked.
    """
    client = MagicMock()
    client.generate_with_tools = AsyncMock(side_effect=list(responses))
    client.generate = AsyncMock(return_value={
        "content": "plain text",
        "model": "mock-model",
        "usage": {},
    })
    client.close = AsyncMock()
    return client


def _respond_payload(text: str, supporting: list | None = None) -> dict:
    """Canned LLM response that calls respond_to_user."""
    args: dict = {"response": text}
    if supporting:
        args["supporting_claims"] = supporting
    return {
        "tool_calls": [{"name": "respond_to_user", "arguments": args}],
        "content": "",
        "model": "mock-model",
        "usage": {},
    }


def _create_claim_payload(content: str, confidence: float = 0.7) -> dict:
    """Canned LLM response that calls create_claim."""
    return {
        "tool_calls": [
            {
                "name": "create_claim",
                "arguments": {
                    "content": content,
                    "type": "observation",
                    "confidence": confidence,
                },
            }
        ],
        "content": "",
        "model": "mock-model",
        "usage": {},
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir(tmp_path):
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestEvaluateWithReasoningLoop:
    """Gate tests: evaluate(use_reasoning_loop=True) returns ReasoningResult."""

    @pytest.mark.asyncio
    async def test_evaluate_with_reasoning_loop_returns_reasoning_result(
        self, tmp_dir
    ):
        """Gate test: response data includes 'reasoning_result' with required keys.

        This is the primary gate for Steps 20.4 and 20.6.
        """
        ep = await _make_endpoint(tmp_dir)

        mock_llm = _mock_llm_with_responses(
            [_respond_payload("2 + 2 equals 4.")]
        )

        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=mock_llm):
            response = await ep.evaluate(
                "What is 2+2?",
                use_reasoning_loop=True,
                use_decomposition=False,
                max_tool_iterations=5,
            )

        assert response.success, (
            "evaluate() with use_reasoning_loop=True failed: {}".format(response.errors)
        )
        data = response.data
        assert data is not None, "response.data must not be None"

        # Gate: reasoning_result must be present
        assert "reasoning_result" in data, (
            "GATE FAILED: 'reasoning_result' key missing from response data. "
            "Got keys: {}".format(list(data.keys()))
        )

        rr = data["reasoning_result"]
        assert isinstance(rr, dict), "reasoning_result must be a dict"

        # Required keys in reasoning_result
        for key in ("halted_reason", "tool_calls", "iterations", "claims_created",
                    "supporting_claims"):
            assert key in rr, (
                "reasoning_result missing key '{}'. Got: {}".format(key, list(rr.keys()))
            )

        # halted_reason should be 'respond_to_user' (LLM called it immediately)
        assert rr["halted_reason"] == "respond_to_user", (
            "Expected halted_reason='respond_to_user', got '{}'".format(
                rr["halted_reason"]
            )
        )

        # Iterations should be 1 (immediate halt)
        assert rr["iterations"] == 1, (
            "Expected iterations=1, got {}".format(rr["iterations"])
        )

    @pytest.mark.asyncio
    async def test_evaluate_reasoning_loop_two_iterations(self, tmp_dir):
        """LLM explores in iteration 1 then responds in iteration 2.

        Verifies that the reasoning trace records both iterations.
        """
        ep = await _make_endpoint(tmp_dir)

        mock_llm = _mock_llm_with_responses(
            [
                _create_claim_payload("2 + 2 is an arithmetic operation"),
                _respond_payload("The answer is 4."),
            ]
        )

        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=mock_llm):
            response = await ep.evaluate(
                "What is 2+2?",
                use_reasoning_loop=True,
                use_decomposition=False,
                max_tool_iterations=5,
            )

        assert response.success, (
            "evaluate() failed: {}".format(response.errors)
        )
        rr = response.data["reasoning_result"]

        # Should have taken 2 iterations
        assert rr["iterations"] == 2, (
            "Expected 2 iterations, got {}".format(rr["iterations"])
        )

        # One claim was created in iteration 1
        assert len(rr["claims_created"]) == 1, (
            "Expected 1 claim created, got {}".format(len(rr["claims_created"]))
        )

        # halted_reason is respond_to_user (LLM chose to halt on iteration 2)
        assert rr["halted_reason"] == "respond_to_user"

        # tool_calls trace should contain create_claim and respond_to_user
        tool_names = [tc["tool"] for tc in rr["tool_calls"]]
        assert "create_claim" in tool_names, "Expected create_claim in trace"
        assert "respond_to_user" in tool_names, "Expected respond_to_user in trace"

    @pytest.mark.asyncio
    async def test_evaluate_reasoning_loop_includes_response_text(self, tmp_dir):
        """The top-level 'response' key matches the text from respond_to_user."""
        ep = await _make_endpoint(tmp_dir)

        expected_response = "The answer is definitely 4."
        mock_llm = _mock_llm_with_responses(
            [_respond_payload(expected_response)]
        )

        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=mock_llm):
            response = await ep.evaluate(
                "What is 2+2?",
                use_reasoning_loop=True,
                use_decomposition=False,
            )

        assert response.success, "evaluate() failed: {}".format(response.errors)
        data = response.data

        # Top-level response text must match respond_to_user payload
        assert data["response"] == expected_response, (
            "Expected '{}', got '{}'".format(expected_response, data["response"])
        )

    @pytest.mark.asyncio
    async def test_evaluate_reasoning_loop_false_skips_loop(self, tmp_dir):
        """use_reasoning_loop=False (default) produces no 'reasoning_result' key."""
        ep = await _make_endpoint(tmp_dir)

        mock_llm = _mock_llm_with_responses(
            [
                {
                    "tool_calls": [
                        {
                            "name": "respond_to_user",
                            "arguments": {
                                "response": "4",
                                "supporting_claims": [],
                            },
                        }
                    ],
                    "content": "",
                    "model": "mock-model",
                    "usage": {},
                }
            ]
        )

        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=mock_llm):
            response = await ep.evaluate(
                "What is 2+2?",
                use_reasoning_loop=False,
                use_decomposition=False,
                use_tools=True,
            )

        assert response.success, "evaluate() failed: {}".format(response.errors)
        data = response.data

        # reasoning_result must NOT be present when use_reasoning_loop=False
        assert "reasoning_result" not in data, (
            "reasoning_result must NOT be present when use_reasoning_loop=False. "
            "Got keys: {}".format(list(data.keys()))
        )

    @pytest.mark.asyncio
    async def test_evaluate_reasoning_loop_supporting_claims_propagated(
        self, tmp_dir
    ):
        """Claim IDs cited in respond_to_user appear in reasoning_result.supporting_claims."""
        ep = await _make_endpoint(tmp_dir)

        cited = ["c0000001", "c0000002"]
        mock_llm = _mock_llm_with_responses(
            [_respond_payload("Confirmed.", supporting=cited)]
        )

        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=mock_llm):
            response = await ep.evaluate(
                "Is water wet?",
                use_reasoning_loop=True,
                use_decomposition=False,
            )

        assert response.success
        rr = response.data["reasoning_result"]

        for claim_id in cited:
            assert claim_id in rr["supporting_claims"], (
                "Expected {} in supporting_claims, got {}".format(
                    claim_id, rr["supporting_claims"]
                )
            )
