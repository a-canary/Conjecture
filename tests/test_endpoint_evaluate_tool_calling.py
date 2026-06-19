# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Tests for Steps 19.3-19.5: evaluate() uses tool-calling LLM mode.

Gate: evaluate() can process tool calls and return structured results.

Covers:
- 19.3: evaluate() calls generate_with_tools() passing CLAIM_TOOLS
- 19.4: tool calls parsed from response; each call routed to executor
- 19.5: respond_to_user halts the loop; created_claim_ids + supporting_claims returned
- Plain-text fallback when LLM returns no tool calls
- Multiple tool calls in one iteration (create_claim + respond_to_user)
- Unknown tool name is handled without crashing
"""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.endpoint.conjecture_endpoint import ConjectureEndpoint
from src.process.claim_tools import CLAIM_TOOLS, RETRIEVAL_TOOLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _make_endpoint(tmp_path: str) -> ConjectureEndpoint:
    """Create and initialize an endpoint backed by a temp SQLite DB."""
    db_path = os.path.join(tmp_path, "test_tool_calling.db")
    ep = ConjectureEndpoint(db_path=db_path, vector_path=":memory:")
    ep._vector_store = None
    await ep._data_manager.initialize()
    ep._initialized = True
    return ep


def _llm_tool_response(tool_calls: list, content: str = "") -> dict:
    """Build a fake LLM generate_with_tools() return value."""
    return {
        "content": content,
        "tool_calls": tool_calls,
        "model": "mock-model",
        "usage": {"prompt_tokens": 20, "completion_tokens": 10},
    }


def _llm_plain_response(content: str = "The answer is 42.") -> dict:
    """Build a fake LLM response with no tool calls (plain-text fallback)."""
    return {
        "content": content,
        "tool_calls": [],
        "model": "mock-model",
        "usage": {"prompt_tokens": 15, "completion_tokens": 8},
    }


def _make_llm_instance(responses: list) -> MagicMock:
    """Build a mock LLMClient instance that returns the given responses in order."""
    instance = MagicMock()
    instance.generate_with_tools = AsyncMock(side_effect=responses)
    instance.generate = AsyncMock(return_value={
        "content": "plain fallback",
        "model": "mock-model",
        "usage": {},
    })
    instance.close = AsyncMock()
    return instance


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir(tmp_path):
    return str(tmp_path)


# ---------------------------------------------------------------------------
# 19.3 — evaluate() uses generate_with_tools with CLAIM_TOOLS
# ---------------------------------------------------------------------------


class TestEvaluateCallsGenerateWithTools:
    """19.3: evaluate() passes CLAIM_TOOLS to the LLM client."""

    @pytest.mark.asyncio
    async def test_evaluate_calls_generate_with_tools_by_default(self, tmp_dir):
        """evaluate() uses generate_with_tools when use_tools=True (the default)."""
        ep = await _make_endpoint(tmp_dir)

        instance = _make_llm_instance([_llm_plain_response("ok")])

        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=instance):
            response = await ep.evaluate("What is the sky color?", use_tools=True)

        assert response.success, f"evaluate() failed: {response.errors}"
        instance.generate_with_tools.assert_awaited_once()
        instance.generate.assert_not_awaited()

        # Verify CLAIM_TOOLS + RETRIEVAL_TOOLS were passed to generate_with_tools
        # (A-0015 adds RETRIEVAL_TOOLS so the LLM can request knowledge retrieval)
        call_kwargs = instance.generate_with_tools.call_args
        tools_arg = call_kwargs[1].get("tools")
        assert tools_arg is not None, "tools argument must be passed to generate_with_tools"
        expected_tools = CLAIM_TOOLS + RETRIEVAL_TOOLS
        assert tools_arg == expected_tools, (
            "generate_with_tools must be called with CLAIM_TOOLS + RETRIEVAL_TOOLS"
        )

    @pytest.mark.asyncio
    async def test_evaluate_use_tools_false_calls_generate(self, tmp_dir):
        """use_tools=False falls back to plain generate() without tools."""
        ep = await _make_endpoint(tmp_dir)

        instance = MagicMock()
        instance.generate = AsyncMock(return_value={
            "content": "plain answer",
            "model": "mock-model",
            "usage": {}
        })
        instance.generate_with_tools = AsyncMock()
        instance.close = AsyncMock()

        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=instance):
            response = await ep.evaluate("What is the sky color?", use_tools=False)

        assert response.success, f"evaluate() failed: {response.errors}"
        instance.generate.assert_awaited_once()
        instance.generate_with_tools.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tool_mode_response_includes_required_keys(self, tmp_dir):
        """Tool-mode response data has tool_calls, created_claim_ids, supporting_claims."""
        ep = await _make_endpoint(tmp_dir)

        instance = _make_llm_instance([_llm_plain_response("answer")])

        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=instance):
            response = await ep.evaluate("test query", use_tools=True)

        assert response.success
        data = response.data
        assert "tool_calls" in data, "tool_calls key must be in response data"
        assert "created_claim_ids" in data, "created_claim_ids key must be in response data"
        assert "supporting_claims" in data, "supporting_claims key must be in response data"
        assert "decomposed_claims" in data, "decomposed_claims key must be in response data"


# ---------------------------------------------------------------------------
# 19.4 — Parse tool calls from LLM response
# ---------------------------------------------------------------------------


class TestEvaluateParseToolCalls:
    """19.4: tool calls are parsed and routed to ClaimToolExecutor."""

    @pytest.mark.asyncio
    async def test_single_create_claim_tool_call(self, tmp_dir):
        """LLM returning a create_claim tool call results in a claim being created."""
        ep = await _make_endpoint(tmp_dir)

        # Iteration 1: create_claim; Iteration 2: respond_to_user
        instance = _make_llm_instance([
            _llm_tool_response([
                {
                    "id": "tc1",
                    "name": "create_claim",
                    "arguments": {
                        "content": "The sky appears blue due to Rayleigh scattering",
                        "type": "observation",
                        "confidence": 0.95,
                    },
                }
            ]),
            _llm_tool_response([
                {
                    "id": "tc2",
                    "name": "respond_to_user",
                    "arguments": {
                        "response": "The sky is blue due to Rayleigh scattering.",
                        "supporting_claims": [],
                    },
                }
            ]),
        ])

        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=instance):
            response = await ep.evaluate("Why is the sky blue?", use_tools=True)

        assert response.success, f"evaluate() failed: {response.errors}"
        data = response.data

        # At least one claim was created
        assert len(data["created_claim_ids"]) >= 1, (
            f"Expected at least 1 created claim, got {data['created_claim_ids']}"
        )

        # Tool calls log should record the create_claim call
        tools_called = [tc["name"] for tc in data["tool_calls"]]
        assert "create_claim" in tools_called

    @pytest.mark.asyncio
    async def test_failed_tool_call_is_logged_not_raised(self, tmp_dir):
        """A tool call with bad arguments logs failure but does not crash evaluate()."""
        ep = await _make_endpoint(tmp_dir)

        instance = _make_llm_instance([
            _llm_tool_response([
                {
                    "id": "tc_bad",
                    "name": "update_confidence",
                    "arguments": {
                        "claim_id": "c_does_not_exist",
                        "new_confidence": 0.9,
                        "reason": "testing error path",
                    },
                }
            ]),
            _llm_plain_response("fallback answer"),
        ])

        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=instance):
            response = await ep.evaluate("Test query for error handling", use_tools=True)

        assert response.success, (
            f"evaluate() must succeed even when a tool call fails: {response.errors}"
        )

        # The failed tool call should be in the log with success=False
        failed_calls = [tc for tc in response.data["tool_calls"] if not tc["success"]]
        assert len(failed_calls) >= 1, "Failed tool call must appear in tool_calls log"
        assert failed_calls[0]["error"] is not None


# ---------------------------------------------------------------------------
# 19.5 — respond_to_user halts loop; created_claim_ids and supporting_claims returned
# ---------------------------------------------------------------------------


class TestEvaluateRespondToUserHalt:
    """19.5: respond_to_user halts the tool loop and surfaces the response."""

    @pytest.mark.asyncio
    async def test_respond_to_user_halts_loop(self, tmp_dir):
        """Once respond_to_user is called, no further LLM calls are made."""
        ep = await _make_endpoint(tmp_dir)

        instance = _make_llm_instance([
            _llm_tool_response([
                {
                    "id": "tc_resp",
                    "name": "respond_to_user",
                    "arguments": {
                        "response": "The final answer is here.",
                        "supporting_claims": [],
                    },
                }
            ]),
            # This second response must NOT be consumed
            _llm_plain_response("should not be seen"),
        ])

        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=instance):
            response = await ep.evaluate(
                "Stop after first respond_to_user?", use_tools=True
            )

        assert response.success, f"evaluate() failed: {response.errors}"

        # Only one LLM call was made (the loop halted after respond_to_user)
        assert instance.generate_with_tools.await_count == 1, (
            f"Expected exactly 1 LLM call after respond_to_user, "
            f"got {instance.generate_with_tools.await_count}"
        )

        # Response text comes from respond_to_user
        assert response.data["response"] == "The final answer is here."

    @pytest.mark.asyncio
    async def test_respond_to_user_response_text_surfaced(self, tmp_dir):
        """The response field in data comes from respond_to_user payload."""
        ep = await _make_endpoint(tmp_dir)

        expected_response = "Based on my analysis, the capital is Paris."
        instance = _make_llm_instance([
            _llm_tool_response([
                {
                    "id": "tc",
                    "name": "respond_to_user",
                    "arguments": {
                        "response": expected_response,
                        "supporting_claims": ["c0000001", "c0000002"],
                    },
                }
            ]),
        ])

        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=instance):
            response = await ep.evaluate(
                "What is the capital of France?", use_tools=True
            )

        assert response.success
        assert response.data["response"] == expected_response, (
            f"Expected '{expected_response}', got '{response.data['response']}'"
        )
        # supporting_claims from respond_to_user are included in response data
        assert set(response.data["supporting_claims"]) == {"c0000001", "c0000002"}

    @pytest.mark.asyncio
    async def test_created_claim_ids_tracked_across_iterations(self, tmp_dir):
        """All claim IDs created across multiple tool iterations are aggregated."""
        ep = await _make_endpoint(tmp_dir)

        instance = _make_llm_instance([
            # Iteration 1: two create_claim calls
            _llm_tool_response([
                {
                    "id": "tc1a",
                    "name": "create_claim",
                    "arguments": {
                        "content": "First reasoning step: the question is about colors",
                        "type": "observation",
                        "confidence": 0.8,
                    },
                },
                {
                    "id": "tc1b",
                    "name": "create_claim",
                    "arguments": {
                        "content": "Second reasoning step: blue is a color of the sky",
                        "type": "assertion",
                        "confidence": 0.9,
                    },
                },
            ]),
            # Iteration 2: respond_to_user halts
            _llm_tool_response([
                {
                    "id": "tc2",
                    "name": "respond_to_user",
                    "arguments": {
                        "response": "Colors include blue, red, and green.",
                        "supporting_claims": [],
                    },
                }
            ]),
        ])

        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=instance):
            response = await ep.evaluate("What are some colors?", use_tools=True)

        assert response.success
        data = response.data

        # Two claims were created across two iterations
        assert len(data["created_claim_ids"]) == 2, (
            f"Expected 2 created claim IDs, got {data['created_claim_ids']}"
        )

    @pytest.mark.asyncio
    async def test_tool_calls_after_respond_to_user_not_executed(self, tmp_dir):
        """Tool calls appearing after respond_to_user in the same batch are skipped."""
        ep = await _make_endpoint(tmp_dir)

        instance = _make_llm_instance([
            _llm_tool_response([
                # respond_to_user comes first — should halt
                {
                    "id": "tc_halt",
                    "name": "respond_to_user",
                    "arguments": {
                        "response": "Done.",
                        "supporting_claims": [],
                    },
                },
                # This create_claim must NOT be executed
                {
                    "id": "tc_skip",
                    "name": "create_claim",
                    "arguments": {
                        "content": "This claim should not be created after respond_to_user",
                        "type": "observation",
                        "confidence": 0.5,
                    },
                },
            ]),
        ])

        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=instance):
            response = await ep.evaluate(
                "Halt after first respond_to_user in batch?", use_tools=True
            )

        assert response.success
        data = response.data
        assert data["response"] == "Done."
        # respond_to_user with no supporting_claims contributes no IDs
        assert data["created_claim_ids"] == [], (
            f"create_claim after respond_to_user must be skipped, "
            f"got {data['created_claim_ids']}"
        )


# ---------------------------------------------------------------------------
# Plain-text fallback
# ---------------------------------------------------------------------------


class TestEvaluatePlainTextFallback:
    """When LLM returns no tool calls, evaluate() uses plain-text content."""

    @pytest.mark.asyncio
    async def test_plain_text_fallback_when_no_tool_calls(self, tmp_dir):
        """If generate_with_tools returns empty tool_calls, content is used."""
        ep = await _make_endpoint(tmp_dir)

        expected_content = "The answer is 42, no tools needed."
        instance = _make_llm_instance([_llm_plain_response(expected_content)])

        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=instance):
            response = await ep.evaluate("What is the answer?", use_tools=True)

        assert response.success
        assert response.data["response"] == expected_content
        # In plain-text fallback, no claims were created
        assert response.data["created_claim_ids"] == []
        assert response.data["supporting_claims"] == []

    @pytest.mark.asyncio
    async def test_plain_text_response_still_includes_metadata(self, tmp_dir):
        """Plain-text fallback response still includes all required metadata keys."""
        ep = await _make_endpoint(tmp_dir)

        instance = _make_llm_instance([_llm_plain_response("some answer")])

        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=instance):
            response = await ep.evaluate("Test metadata presence", use_tools=True)

        assert response.success
        data = response.data
        for key in ("query", "response", "claims_used", "decomposed_claims",
                    "tool_calls", "created_claim_ids", "supporting_claims", "model"):
            assert key in data, f"Missing required key in response data: '{key}'"


# ---------------------------------------------------------------------------
# Gate test
# ---------------------------------------------------------------------------


class TestEvaluateToolCallingGate:
    """Gate: evaluate() can process tool calls and return structured results."""

    @pytest.mark.asyncio
    async def test_gate_tool_calling_end_to_end(self, tmp_dir):
        """Gate test: full tool-calling round trip with create_claim + respond_to_user."""
        ep = await _make_endpoint(tmp_dir)

        instance = _make_llm_instance([
            _llm_tool_response([
                {
                    "id": "tc_create",
                    "name": "create_claim",
                    "arguments": {
                        "content": "Gate test: water has the chemical formula H2O",
                        "type": "assertion",
                        "confidence": 0.99,
                    },
                },
                {
                    "id": "tc_respond",
                    "name": "respond_to_user",
                    "arguments": {
                        "response": "Water is H2O.",
                        "supporting_claims": [],
                    },
                },
            ]),
        ])

        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=instance):
            response = await ep.evaluate(
                "What is the chemical formula of water?", use_tools=True
            )

        # Gate assertions
        assert response.success, f"GATE FAILED: evaluate() returned error: {response.errors}"
        assert response.data is not None

        data = response.data
        assert data["response"] == "Water is H2O.", (
            f"GATE FAILED: unexpected response '{data['response']}'"
        )
        assert len(data["created_claim_ids"]) == 1, (
            f"GATE FAILED: expected 1 created claim ID, got {data['created_claim_ids']}"
        )
        assert "tool_calls" in data and len(data["tool_calls"]) >= 1, (
            "GATE FAILED: tool_calls must be present and non-empty"
        )
        assert "supporting_claims" in data, (
            "GATE FAILED: supporting_claims key must be present in response data"
        )
        assert "created_claim_ids" in data, (
            "GATE FAILED: created_claim_ids key must be present in response data"
        )
