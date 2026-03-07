"""
Tests for Phase 2: Suspend/Resume in ConjectureEndpoint (A-0015).

Gate requirements:
  - Gate 1: evaluate() with stub LLM emitting retrieve_knowledge returns
    status="paused" with a pause_id.
  - Gate 2: resume_evaluation(pause_id, ["evidence"]) returns
    status="complete" with non-empty response.

Covers:
  - test_evaluate_pauses_on_retrieve_knowledge: evaluate() returns paused
    when LLM calls retrieve_knowledge.
  - test_evaluate_pause_id_stored_in_paused_states: paused state is stored
    internally so resume can find it.
  - test_resume_evaluation_returns_complete: resume_evaluation() returns
    status="complete" with final response.
  - test_evaluate_no_retrieve_knowledge_unchanged: existing direct path
    (no retrieve_knowledge) is fully backward compatible.
  - test_evaluate_paused_response_structure: paused response contains
    all required fields.
  - test_resume_evaluation_unknown_pause_id: unknown pause_id returns error.
  - test_resume_evaluation_response_structure: complete response contains
    required fields.
  - test_paused_state_removed_after_resume: state is cleaned up on resume.
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
    db_path = os.path.join(tmp_path, "test_suspend_resume.db")
    ep = ConjectureEndpoint(db_path=db_path, vector_path=":memory:")
    ep._vector_store = None
    await ep._data_manager.initialize()
    ep._initialized = True
    return ep


def _make_llm_with_retrieve_knowledge(
    retrieval_query: str = "What is the capital of France?",
    tool_hint: str = None,
) -> MagicMock:
    """Stub LLM that emits a retrieve_knowledge call on the first invocation."""
    rk_tc = {
        "id": "call_rk_001",
        "name": "retrieve_knowledge",
        "arguments": {
            "query": retrieval_query,
        },
    }
    if tool_hint:
        rk_tc["arguments"]["tool_hint"] = tool_hint

    first_response = {
        "content": "",
        "tool_calls": [rk_tc],
        "model": "mock-model",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    client = MagicMock()
    client.generate_with_tools = AsyncMock(return_value=first_response)
    client.generate = AsyncMock(return_value={
        "content": "direct answer",
        "model": "mock-model",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    })
    client.close = AsyncMock()
    return client


def _make_llm_with_respond_to_user(response_text: str = "Paris is the capital.") -> MagicMock:
    """Stub LLM that emits a respond_to_user call on the first invocation."""
    respond_tc = {
        "id": "call_resp_001",
        "name": "respond_to_user",
        "arguments": {
            "response": response_text,
            "supporting_claims": [],
        },
    }

    response = {
        "content": "",
        "tool_calls": [respond_tc],
        "model": "mock-model",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    client = MagicMock()
    client.generate_with_tools = AsyncMock(return_value=response)
    client.generate = AsyncMock(return_value={
        "content": response_text,
        "model": "mock-model",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    })
    client.close = AsyncMock()
    return client


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir(tmp_path):
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Gate tests
# ---------------------------------------------------------------------------


class TestSuspendResume:
    """Phase 2 gate tests for suspend/resume protocol (A-0015)."""

    @pytest.mark.asyncio
    async def test_evaluate_pauses_on_retrieve_knowledge(self, tmp_dir):
        """GATE 1: evaluate() with stub LLM emitting retrieve_knowledge returns
        status='paused' with a pause_id."""
        ep = await _make_endpoint(tmp_dir)
        mock_llm = _make_llm_with_retrieve_knowledge(
            retrieval_query="capital of France"
        )

        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=mock_llm):
            response = await ep.evaluate(
                "What is the capital of France?",
                use_tools=True,
                use_decomposition=False,
                max_tool_iterations=5,
            )

        assert response.success, f"evaluate() must succeed on pause: {response.errors}"
        data = response.data
        assert data is not None

        # Gate assertion: status must be "paused"
        assert data.get("status") == "paused", (
            f"GATE 1 FAILED: expected status='paused', got {data.get('status')!r}"
        )

        # Gate assertion: pause_id must be present and non-empty
        pause_id = data.get("pause_id")
        assert pause_id, (
            f"GATE 1 FAILED: pause_id must be present and non-empty, got {pause_id!r}"
        )

    @pytest.mark.asyncio
    async def test_resume_evaluation_returns_complete(self, tmp_dir):
        """GATE 2: resume_evaluation(pause_id, ['evidence']) returns
        status='complete' with non-empty response."""
        ep = await _make_endpoint(tmp_dir)

        # Step A: pause with retrieve_knowledge
        pause_llm = _make_llm_with_retrieve_knowledge(
            retrieval_query="capital of France"
        )

        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=pause_llm):
            pause_response = await ep.evaluate(
                "What is the capital of France?",
                use_tools=True,
                use_decomposition=False,
                max_tool_iterations=5,
            )

        assert pause_response.success, f"pause failed: {pause_response.errors}"
        pause_id = pause_response.data["pause_id"]
        assert pause_id, "pause_id must be present"

        # Step B: resume with evidence, LLM now responds with final answer
        resume_llm = _make_llm_with_respond_to_user("Paris is the capital of France.")

        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=resume_llm):
            resume_response = await ep.resume_evaluation(
                pause_id=pause_id,
                retrieval_results=["The capital of France is Paris."],
                max_tool_iterations=5,
            )

        assert resume_response.success, (
            f"GATE 2 FAILED: resume_evaluation() returned error: {resume_response.errors}"
        )
        data = resume_response.data
        assert data is not None

        # Gate assertion: status must be "complete"
        assert data.get("status") == "complete", (
            f"GATE 2 FAILED: expected status='complete', got {data.get('status')!r}"
        )

        # Gate assertion: response must be non-empty
        response_text = data.get("response", "")
        assert response_text, (
            f"GATE 2 FAILED: response must be non-empty, got {response_text!r}"
        )

    # -----------------------------------------------------------------------
    # Additional correctness tests
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_evaluate_no_retrieve_knowledge_unchanged(self, tmp_dir):
        """Existing direct path (no retrieve_knowledge) is fully backward compatible.

        When the LLM calls respond_to_user without any retrieve_knowledge,
        the response must have status='complete' and include 'response' and
        'tool_calls' keys, matching the existing contract.
        """
        ep = await _make_endpoint(tmp_dir)

        expected_response = "The answer is 4."
        tool_calls_seq = [
            [
                {
                    "id": "call_001",
                    "name": "respond_to_user",
                    "arguments": {
                        "response": expected_response,
                        "supporting_claims": [],
                    },
                }
            ]
        ]

        responses = [
            {
                "content": "",
                "tool_calls": tcs,
                "model": "mock-model",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
            for tcs in tool_calls_seq
        ]

        mock_llm = MagicMock()
        mock_llm.generate_with_tools = AsyncMock(side_effect=responses)
        mock_llm.generate = AsyncMock(return_value={
            "content": "direct answer",
            "model": "mock-model",
            "usage": {},
        })
        mock_llm.close = AsyncMock()

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

        # Must have status="complete" (not "paused")
        assert data.get("status") == "complete", (
            f"Non-retrieval path must return status='complete', got {data.get('status')!r}"
        )

        # Must have the original required keys
        assert "tool_calls" in data, "tool_calls key must be present"
        assert "response" in data, "response key must be present"
        assert data["response"] == expected_response, (
            f"Expected '{expected_response}', got '{data['response']}'"
        )

        # _paused_states must be empty — no state was stored
        assert len(ep._paused_states) == 0, (
            "No paused state should be stored when retrieve_knowledge was not called"
        )

    @pytest.mark.asyncio
    async def test_evaluate_paused_response_structure(self, tmp_dir):
        """Paused response contains all required fields."""
        ep = await _make_endpoint(tmp_dir)
        mock_llm = _make_llm_with_retrieve_knowledge(
            retrieval_query="who wrote Hamlet",
            tool_hint="web_search",
        )

        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=mock_llm):
            response = await ep.evaluate(
                "Who wrote Hamlet?",
                use_tools=True,
                use_decomposition=False,
            )

        assert response.success
        data = response.data

        required_fields = ["status", "pause_id", "retrieval_request", "query"]
        for field in required_fields:
            assert field in data, (
                f"Paused response must include '{field}' field; got keys: {list(data.keys())}"
            )

        # retrieval_request sub-fields
        rr = data["retrieval_request"]
        assert "query" in rr, "retrieval_request must have 'query' field"
        assert rr["query"] == "who wrote Hamlet", (
            f"retrieval_request.query should match LLM's query arg, got {rr['query']!r}"
        )

    @pytest.mark.asyncio
    async def test_evaluate_pause_id_stored_in_paused_states(self, tmp_dir):
        """Paused state is stored internally keyed by pause_id."""
        ep = await _make_endpoint(tmp_dir)
        mock_llm = _make_llm_with_retrieve_knowledge("some query")

        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=mock_llm):
            response = await ep.evaluate(
                "Tell me something.",
                use_tools=True,
                use_decomposition=False,
            )

        pause_id = response.data["pause_id"]
        assert pause_id in ep._paused_states, (
            f"pause_id {pause_id!r} must be present in ep._paused_states"
        )

        state = ep._paused_states[pause_id]
        assert state.pending_retrieval.query == "some query", (
            f"Stored state's retrieval query should be 'some query', "
            f"got {state.pending_retrieval.query!r}"
        )

    @pytest.mark.asyncio
    async def test_resume_evaluation_unknown_pause_id(self, tmp_dir):
        """resume_evaluation with unknown pause_id returns an error response."""
        ep = await _make_endpoint(tmp_dir)

        response = await ep.resume_evaluation(
            pause_id="non-existent-pause-id",
            retrieval_results=["some evidence"],
        )

        assert not response.success, (
            "resume_evaluation with unknown pause_id must return success=False"
        )
        assert "PAUSE_ID_NOT_FOUND" in response.errors, (
            f"Expected PAUSE_ID_NOT_FOUND in errors, got: {response.errors}"
        )

    @pytest.mark.asyncio
    async def test_resume_evaluation_response_structure(self, tmp_dir):
        """Complete response from resume_evaluation contains required fields."""
        ep = await _make_endpoint(tmp_dir)

        # Pause
        pause_llm = _make_llm_with_retrieve_knowledge("some fact")
        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=pause_llm):
            pause_response = await ep.evaluate(
                "Tell me a fact.",
                use_tools=True,
                use_decomposition=False,
            )

        pause_id = pause_response.data["pause_id"]

        # Resume
        resume_llm = _make_llm_with_respond_to_user("Here is the fact.")
        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=resume_llm):
            resume_response = await ep.resume_evaluation(
                pause_id=pause_id,
                retrieval_results=["The fact you wanted is X."],
            )

        assert resume_response.success, f"resume failed: {resume_response.errors}"
        data = resume_response.data

        required_fields = ["status", "response", "tool_calls", "created_claim_ids"]
        for field in required_fields:
            assert field in data, (
                f"Resume response must include '{field}'; got keys: {list(data.keys())}"
            )

    @pytest.mark.asyncio
    async def test_paused_state_removed_after_resume(self, tmp_dir):
        """Paused state is cleaned up from _paused_states after resume_evaluation."""
        ep = await _make_endpoint(tmp_dir)

        # Pause
        pause_llm = _make_llm_with_retrieve_knowledge("cleanup test query")
        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=pause_llm):
            pause_response = await ep.evaluate(
                "Clean me up.",
                use_tools=True,
                use_decomposition=False,
            )

        pause_id = pause_response.data["pause_id"]
        assert pause_id in ep._paused_states, "State should be stored before resume"

        # Resume
        resume_llm = _make_llm_with_respond_to_user("Cleaned up.")
        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=resume_llm):
            await ep.resume_evaluation(
                pause_id=pause_id,
                retrieval_results=["Evidence."],
            )

        assert pause_id not in ep._paused_states, (
            "Paused state must be removed from _paused_states after resume"
        )
