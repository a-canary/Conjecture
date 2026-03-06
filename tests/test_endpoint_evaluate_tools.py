"""
Tests for Steps 19.3-19.5: Tool-calling mode in ConjectureEndpoint.evaluate().

Covers:
- test_evaluate_with_tools_creates_claims: evaluate(use_tools=True) executes
  create_claim tool calls and records them in tool_calls metadata.
- test_evaluate_tool_loop_terminates_on_respond: respond_to_user tool terminates
  the loop and its response text becomes the final response.
- test_evaluate_max_iterations_limit: loop terminates after max_tool_iterations
  even if respond_to_user is never called.
- test_evaluate_tools_false_skips_tool_calling: use_tools=False calls generate()
  not generate_with_tools() and returns no tool_calls key.
- test_evaluate_tools_no_tool_calls_in_response: if LLM returns no tool_calls,
  content is used as final response (backward compatible).
- test_evaluate_tools_response_includes_tool_calls_key: gate — tool_calls present
  in response.data when use_tools=True.

Gate: evaluate() with tools returns tool_calls in response metadata.
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
    db_path = os.path.join(tmp_path, "test_evaluate_tools.db")
    ep = ConjectureEndpoint(db_path=db_path, vector_path=":memory:")
    ep._vector_store = None
    await ep._data_manager.initialize()
    ep._initialized = True
    return ep


def _make_llm_instance_with_tools(tool_calls_sequence=None, content="") -> MagicMock:
    """Build a mock LLMClient that returns tool calls from generate_with_tools().

    Args:
        tool_calls_sequence: List of tool_calls lists to return on successive calls.
            Each element is a list of tool call dicts for one iteration.
            If None, returns a single empty tool_calls response.
        content: Text content to include in each LLM response.

    Returns:
        Mock LLMClient instance with generate_with_tools and generate mocked.
    """
    if tool_calls_sequence is None:
        tool_calls_sequence = [[]]

    responses = [
        {
            "content": content,
            "tool_calls": tcs,
            "model": "mock-model",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        for tcs in tool_calls_sequence
    ]

    client = MagicMock()
    client.generate_with_tools = AsyncMock(side_effect=responses)
    client.generate = AsyncMock(return_value={
        "content": content or "plain text response",
        "model": "mock-model",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    })
    client.close = AsyncMock()
    return client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir(tmp_path):
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEvaluateWithTools:
    """Gate tests for tool-calling mode in evaluate()."""

    @pytest.mark.asyncio
    async def test_evaluate_with_tools_creates_claims(self, tmp_dir):
        """evaluate(use_tools=True) executes create_claim and records it in tool_calls."""
        ep = await _make_endpoint(tmp_dir)

        # Sequence: first iteration returns a create_claim tool call,
        # second iteration returns a respond_to_user call.
        tool_calls_seq = [
            [
                {
                    "id": "call_001",
                    "name": "create_claim",
                    "arguments": {
                        "content": "2 plus 2 equals 4 by arithmetic",
                        "type": "assertion",
                        "confidence": 0.95,
                    },
                }
            ],
            [
                {
                    "id": "call_002",
                    "name": "respond_to_user",
                    "arguments": {
                        "response": "The answer is 4.",
                        "supporting_claims": [],
                    },
                }
            ],
        ]
        mock_llm = _make_llm_instance_with_tools(tool_calls_seq)

        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=mock_llm):
            response = await ep.evaluate(
                "What is 2+2?",
                use_tools=True,
                use_decomposition=False,
                max_tool_iterations=5,
            )

        assert response.success, f"evaluate() failed: {response.errors}"
        data = response.data
        assert data is not None

        # Gate: tool_calls must be in data
        assert "tool_calls" in data, "tool_calls key must be present when use_tools=True"

        # At least the create_claim and respond_to_user tool calls must be present
        tool_names = [tc["name"] for tc in data["tool_calls"]]
        assert "create_claim" in tool_names, (
            f"Expected create_claim in tool_calls, got: {tool_names}"
        )

    @pytest.mark.asyncio
    async def test_evaluate_tool_loop_terminates_on_respond(self, tmp_dir):
        """respond_to_user tool call terminates the loop and its text becomes the response."""
        ep = await _make_endpoint(tmp_dir)

        expected_response = "The answer to 2+2 is 4."
        tool_calls_seq = [
            [
                {
                    "id": "call_resp",
                    "name": "respond_to_user",
                    "arguments": {
                        "response": expected_response,
                        "supporting_claims": ["c0000001"],
                    },
                }
            ]
        ]
        mock_llm = _make_llm_instance_with_tools(tool_calls_seq)

        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=mock_llm):
            response = await ep.evaluate(
                "What is 2+2?",
                use_tools=True,
                use_decomposition=False,
                max_tool_iterations=5,
            )

        assert response.success, f"evaluate() failed: {response.errors}"
        data = response.data

        # The response text must come from respond_to_user
        assert data["response"] == expected_response, (
            f"Expected '{expected_response}', got '{data['response']}'"
        )

        # Only one iteration needed
        tool_names = [tc["name"] for tc in data["tool_calls"]]
        assert "respond_to_user" in tool_names, "respond_to_user must be in tool_calls"

        # generate_with_tools called exactly once (loop terminated immediately)
        assert mock_llm.generate_with_tools.await_count == 1, (
            f"Expected 1 LLM call, got {mock_llm.generate_with_tools.await_count}"
        )

    @pytest.mark.asyncio
    async def test_evaluate_max_iterations_limit(self, tmp_dir):
        """Loop terminates after max_tool_iterations even if respond_to_user never called."""
        ep = await _make_endpoint(tmp_dir)

        max_iters = 3
        # Each iteration: LLM returns a create_claim tool call (no respond_to_user)
        def _make_iter_calls(idx):
            return [
                {
                    "id": f"call_{idx:03d}",
                    "name": "create_claim",
                    "arguments": {
                        "content": f"Reasoning step {idx}: some observation about the problem",
                        "type": "observation",
                        "confidence": 0.7,
                    },
                }
            ]
        tool_calls_seq = [_make_iter_calls(i) for i in range(1, max_iters + 1)]
        mock_llm = _make_llm_instance_with_tools(tool_calls_seq, content="fallback text")

        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=mock_llm):
            response = await ep.evaluate(
                "What is 2+2?",
                use_tools=True,
                use_decomposition=False,
                max_tool_iterations=max_iters,
            )

        assert response.success, f"evaluate() must succeed even at max iterations: {response.errors}"
        data = response.data

        # generate_with_tools must have been called at most max_iters times
        assert mock_llm.generate_with_tools.await_count <= max_iters, (
            f"Expected at most {max_iters} LLM calls, "
            f"got {mock_llm.generate_with_tools.await_count}"
        )

        # tool_calls must be present
        assert "tool_calls" in data, "tool_calls key must be present when use_tools=True"

        # All logged tool calls must be create_claim (no respond_to_user)
        tool_names = {tc["name"] for tc in data["tool_calls"]}
        assert "respond_to_user" not in tool_names, (
            "respond_to_user should not appear when loop ends at max_iterations"
        )

    @pytest.mark.asyncio
    async def test_evaluate_tools_false_skips_tool_calling(self, tmp_dir):
        """use_tools=False uses plain generate() and returns no tool_calls key."""
        ep = await _make_endpoint(tmp_dir)

        mock_llm = _make_llm_instance_with_tools(content="plain answer")

        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=mock_llm):
            response = await ep.evaluate(
                "What is 2+2?",
                use_tools=False,
                use_decomposition=False,
            )

        assert response.success, f"evaluate() failed: {response.errors}"
        data = response.data

        # generate() used, NOT generate_with_tools
        mock_llm.generate.assert_awaited_once()
        mock_llm.generate_with_tools.assert_not_awaited()

        # No tool_calls key in non-tool mode
        assert "tool_calls" not in data, (
            "tool_calls must NOT be present when use_tools=False"
        )

    @pytest.mark.asyncio
    async def test_evaluate_tools_no_tool_calls_in_response(self, tmp_dir):
        """If LLM returns no tool calls, content text is used as final response."""
        ep = await _make_endpoint(tmp_dir)

        expected_text = "Direct answer: 4"
        # One iteration, no tool_calls (empty list), content is the response
        tool_calls_seq = [[]]  # empty tool calls
        mock_llm = _make_llm_instance_with_tools(tool_calls_seq, content=expected_text)

        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=mock_llm):
            response = await ep.evaluate(
                "What is 2+2?",
                use_tools=True,
                use_decomposition=False,
            )

        assert response.success, f"evaluate() failed: {response.errors}"
        data = response.data

        # Content should be used as final response
        assert data["response"] == expected_text, (
            f"Expected '{expected_text}', got '{data['response']}'"
        )

        # tool_calls key present but empty
        assert "tool_calls" in data
        assert data["tool_calls"] == [], (
            f"Expected empty tool_calls list, got: {data['tool_calls']}"
        )

    @pytest.mark.asyncio
    async def test_evaluate_tools_response_includes_tool_calls_key(self, tmp_dir):
        """Gate: evaluate() with use_tools=True always includes tool_calls in response data."""
        ep = await _make_endpoint(tmp_dir)

        tool_calls_seq = [
            [
                {
                    "id": "call_gate",
                    "name": "respond_to_user",
                    "arguments": {"response": "4", "supporting_claims": []},
                }
            ]
        ]
        mock_llm = _make_llm_instance_with_tools(tool_calls_seq)

        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=mock_llm):
            response = await ep.evaluate(
                "What is 2+2?",
                use_tools=True,
                use_decomposition=False,
            )

        assert response.success, f"GATE FAILED: {response.errors}"
        data = response.data

        # Gate assertion: tool_calls must exist in response data
        assert "tool_calls" in data, (
            "GATE FAILED: evaluate() with use_tools=True must include 'tool_calls' in data"
        )
        assert isinstance(data["tool_calls"], list), (
            "tool_calls must be a list"
        )

        # Each tool call record must have required keys
        for tc in data["tool_calls"]:
            assert "name" in tc, f"tool call missing 'name' key: {tc}"
            assert "arguments" in tc, f"tool call missing 'arguments' key: {tc}"
            assert "success" in tc, f"tool call missing 'success' key: {tc}"
            assert "iteration" in tc, f"tool call missing 'iteration' key: {tc}"

    @pytest.mark.asyncio
    async def test_evaluate_tool_calls_logged_with_iteration_number(self, tmp_dir):
        """Each tool call record includes the iteration number for traceability."""
        ep = await _make_endpoint(tmp_dir)

        tool_calls_seq = [
            # Iteration 1: create a claim
            [
                {
                    "id": "call_iter1",
                    "name": "create_claim",
                    "arguments": {
                        "content": "Step one of my reasoning process for the math problem",
                        "type": "observation",
                        "confidence": 0.8,
                    },
                }
            ],
            # Iteration 2: respond
            [
                {
                    "id": "call_iter2",
                    "name": "respond_to_user",
                    "arguments": {"response": "4", "supporting_claims": []},
                }
            ],
        ]
        mock_llm = _make_llm_instance_with_tools(tool_calls_seq)

        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=mock_llm):
            response = await ep.evaluate(
                "What is 2+2?",
                use_tools=True,
                use_decomposition=False,
            )

        assert response.success, f"evaluate() failed: {response.errors}"
        data = response.data
        tool_calls = data["tool_calls"]

        assert len(tool_calls) == 2, f"Expected 2 tool calls, got {len(tool_calls)}"

        # Iteration numbers should be 1 and 2 respectively
        iterations = [tc["iteration"] for tc in tool_calls]
        assert iterations[0] == 1, f"First tool call should be iteration 1, got {iterations[0]}"
        assert iterations[1] == 2, f"Second tool call should be iteration 2, got {iterations[1]}"
