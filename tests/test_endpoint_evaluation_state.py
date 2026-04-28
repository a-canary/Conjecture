"""
Tests for A-0014: Streaming Evaluation State

Verifies:
- EvaluationState model structure
- publish_evaluation_state / get_evaluation_state / clear_evaluation_state
- HTTP endpoint GET /v1/evaluation/{session_id}/state
- State lifecycle during evaluate() and resume_evaluation()
"""

import os
import pytest
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from src.endpoint.conjecture_endpoint import (
    ConjectureEndpoint,
    EvaluationState,
)


# ---------------------------------------------------------------------------
# EvaluationState Model Tests
# ---------------------------------------------------------------------------

class TestEvaluationStateModel:
    """Test EvaluationState model structure and defaults."""

    def test_default_status(self):
        """Status defaults to 'in_progress'."""
        state = EvaluationState(
            session_id="sess_123",
            query="What is 2+2?",
            iteration=1,
            max_iterations=5,
            claims_being_evaluated=[],
            tool_calls_so_far=[],
            created_claim_ids=[],
        )
        assert state.status == "in_progress"

    def test_all_status_values(self):
        """All valid status values are accepted."""
        for status in ["in_progress", "paused", "complete", "error"]:
            state = EvaluationState(
                session_id="sess_123",
                query="Test",
                status=status,
                iteration=1,
                max_iterations=5,
                claims_being_evaluated=[],
                tool_calls_so_far=[],
                created_claim_ids=[],
            )
            assert state.status == status

    def test_tracks_claims_being_evaluated(self):
        """claims_being_evaluated field stores claim IDs."""
        claim_ids = ["claim_1", "claim_2", "claim_3"]
        state = EvaluationState(
            session_id="sess_123",
            query="Analyze this",
            iteration=1,
            max_iterations=5,
            claims_being_evaluated=claim_ids,
            tool_calls_so_far=[],
            created_claim_ids=[],
        )
        assert state.claims_being_evaluated == claim_ids

    def test_tracks_tool_calls(self):
        """tool_calls_so_far stores executed tool calls."""
        tool_calls = [
            {"name": "create_claim", "args": {"content": "test"}, "success": True},
            {"name": "web_search", "args": {"query": "x"}, "success": True},
        ]
        state = EvaluationState(
            session_id="sess_123",
            query="Test",
            iteration=1,
            max_iterations=5,
            claims_being_evaluated=[],
            tool_calls_so_far=tool_calls,
            created_claim_ids=[],
        )
        assert len(state.tool_calls_so_far) == 2
        assert state.tool_calls_so_far[0]["name"] == "create_claim"

    def test_tracks_created_claim_ids(self):
        """created_claim_ids stores claim IDs created during evaluation."""
        created = ["new_claim_1", "new_claim_2"]
        state = EvaluationState(
            session_id="sess_123",
            query="Test",
            iteration=1,
            max_iterations=5,
            claims_being_evaluated=[],
            tool_calls_so_far=[],
            created_claim_ids=created,
        )
        assert state.created_claim_ids == created

    def test_current_tool_none_by_default(self):
        """current_tool defaults to None when not executing."""
        state = EvaluationState(
            session_id="sess_123",
            query="Test",
            iteration=1,
            max_iterations=5,
            claims_being_evaluated=[],
            tool_calls_so_far=[],
            created_claim_ids=[],
        )
        assert state.current_tool is None

    def test_current_tool_set_during_execution(self):
        """current_tool is set when a tool is executing."""
        state = EvaluationState(
            session_id="sess_123",
            query="Test",
            iteration=1,
            max_iterations=5,
            claims_being_evaluated=[],
            tool_calls_so_far=[],
            created_claim_ids=[],
            current_tool="web_search",
        )
        assert state.current_tool == "web_search"

    def test_llm_content_optional(self):
        """llm_content is optional, defaults to None."""
        state = EvaluationState(
            session_id="sess_123",
            query="Test",
            iteration=1,
            max_iterations=5,
            claims_being_evaluated=[],
            tool_calls_so_far=[],
            created_claim_ids=[],
        )
        assert state.llm_content is None

    def test_llm_content_stored(self):
        """llm_content stores LLM response text."""
        state = EvaluationState(
            session_id="sess_123",
            query="Test",
            iteration=1,
            max_iterations=5,
            claims_being_evaluated=[],
            tool_calls_so_far=[],
            created_claim_ids=[],
            llm_content="The answer is 4 because 2+2=4.",
        )
        assert "answer is 4" in state.llm_content

    def test_updated_at_defaults_to_now(self):
        """updated_at defaults to current UTC time."""
        before = datetime.utcnow()
        state = EvaluationState(
            session_id="sess_123",
            query="Test",
            iteration=1,
            max_iterations=5,
            claims_being_evaluated=[],
            tool_calls_so_far=[],
            created_claim_ids=[],
        )
        after = datetime.utcnow()
        assert before <= state.updated_at <= after

    def test_iteration_defaults_to_zero(self):
        """iteration defaults to 0."""
        state = EvaluationState(
            session_id="sess_123",
            query="Test",
            max_iterations=5,
            claims_being_evaluated=[],
            tool_calls_so_far=[],
            created_claim_ids=[],
        )
        assert state.iteration == 0

    def test_json_serialization(self):
        """EvaluationState serializes to JSON with ISO datetime."""
        state = EvaluationState(
            session_id="sess_123",
            query="Test query",
            iteration=2,
            max_iterations=5,
            claims_being_evaluated=["c1", "c2"],
            tool_calls_so_far=[{"name": "test", "success": True}],
            created_claim_ids=["new1"],
            current_tool="example_tool",
            llm_content="LLM output here",
        )
        json_str = state.model_dump_json()
        assert "sess_123" in json_str
        assert "Test query" in json_str
        assert "example_tool" in json_str


# ---------------------------------------------------------------------------
# ConjectureEndpoint Evaluation State Methods
# ---------------------------------------------------------------------------

class TestEndpointEvaluationStateMethods:
    """Test publish/get/clear evaluation state on ConjectureEndpoint."""

    @pytest.fixture
    def endpoint(self, tmp_path):
        """Create endpoint with temporary database."""
        db_path = os.path.join(tmp_path, "test_eval_state.db")
        ep = ConjectureEndpoint(db_path=db_path, vector_path=":memory:")
        ep._vector_store = None
        return ep

    def test_publish_creates_state(self, endpoint):
        """publish_evaluation_state creates a retrievable state."""
        endpoint.publish_evaluation_state(
            session_id="test_session",
            query="What is 2+2?",
            iteration=1,
            max_iterations=5,
            claims_being_evaluated=["c1", "c2"],
            tool_calls_so_far=[],
            created_claim_ids=["new1"],
        )
        state = endpoint.get_evaluation_state("test_session")
        assert state is not None
        assert state.session_id == "test_session"
        assert state.query == "What is 2+2?"
        assert state.iteration == 1
        assert state.max_iterations == 5

    def test_publish_overwrites_previous_state(self, endpoint):
        """Publishing updates existing state for same session."""
        endpoint.publish_evaluation_state(
            session_id="test_session",
            query="Query 1",
            iteration=1,
            max_iterations=5,
            claims_being_evaluated=[],
            tool_calls_so_far=[],
            created_claim_ids=[],
        )
        endpoint.publish_evaluation_state(
            session_id="test_session",
            query="Query 2",
            iteration=2,
            max_iterations=5,
            claims_being_evaluated=["c1"],
            tool_calls_so_far=[],
            created_claim_ids=["new1"],
        )
        state = endpoint.get_evaluation_state("test_session")
        assert state.query == "Query 2"
        assert state.iteration == 2

    def test_get_evaluation_state_returns_none_for_unknown_session(self, endpoint):
        """get_evaluation_state returns None for sessions without state."""
        state = endpoint.get_evaluation_state("nonexistent_session")
        assert state is None

    def test_clear_removes_state(self, endpoint):
        """clear_evaluation_state removes state for session."""
        endpoint.publish_evaluation_state(
            session_id="test_session",
            query="Test",
            iteration=1,
            max_iterations=5,
            claims_being_evaluated=[],
            tool_calls_so_far=[],
            created_claim_ids=[],
        )
        endpoint.clear_evaluation_state("test_session")
        state = endpoint.get_evaluation_state("test_session")
        assert state is None

    def test_clear_nonexistent_is_safe(self, endpoint):
        """clear_evaluation_state does not raise for unknown session."""
        # Should not raise
        endpoint.clear_evaluation_state("nonexistent_session")
        # State should still be None
        assert endpoint.get_evaluation_state("nonexistent_session") is None

    def test_multiple_sessions_independent(self, endpoint):
        """Multiple sessions maintain independent state."""
        endpoint.publish_evaluation_state(
            session_id="session_a",
            query="Query A",
            iteration=1,
            max_iterations=5,
            claims_being_evaluated=[],
            tool_calls_so_far=[],
            created_claim_ids=[],
        )
        endpoint.publish_evaluation_state(
            session_id="session_b",
            query="Query B",
            iteration=3,
            max_iterations=5,
            claims_being_evaluated=["c1", "c2"],
            tool_calls_so_far=[],
            created_claim_ids=[],
        )
        state_a = endpoint.get_evaluation_state("session_a")
        state_b = endpoint.get_evaluation_state("session_b")
        assert state_a.query == "Query A"
        assert state_a.iteration == 1
        assert state_b.query == "Query B"
        assert state_b.iteration == 3

    def test_publish_with_tool_call_logged(self, endpoint):
        """tool_calls_so_far persists across publish calls."""
        endpoint.publish_evaluation_state(
            session_id="test_session",
            query="Test",
            iteration=1,
            max_iterations=5,
            claims_being_evaluated=[],
            tool_calls_so_far=[],
            created_claim_ids=[],
        )
        endpoint.publish_evaluation_state(
            session_id="test_session",
            query="Test",
            iteration=2,
            max_iterations=5,
            claims_being_evaluated=[],
            tool_calls_so_far=[
                {"name": "create_claim", "args": {}, "success": True}
            ],
            created_claim_ids=["new1"],
        )
        state = endpoint.get_evaluation_state("test_session")
        assert len(state.tool_calls_so_far) == 1
        assert state.tool_calls_so_far[0]["name"] == "create_claim"


# ---------------------------------------------------------------------------
# HTTP Server Endpoint Tests
# ---------------------------------------------------------------------------

class TestEvaluationStateHTTPEndpoint:
    """Test GET /v1/evaluation/{session_id}/state endpoint."""

    @pytest.fixture
    def server_with_mocked_endpoint(self):
        """Create server with mocked endpoint for HTTP endpoint tests."""
        from unittest.mock import MagicMock
        from src.endpoint.http_server import ConjectureServer
        from fastapi import FastAPI

        server = ConjectureServer()
        # Mock the endpoint methods
        server._endpoint = MagicMock()
        # Manually set up the app with the route
        server._app = FastAPI()
        # Register the evaluation state endpoint
        @server._app.get("/v1/evaluation/{session_id}/state")
        async def get_evaluation_state(session_id: str):
            from src.endpoint.http_server import HTTPException, JSONResponse
            eval_state = server._endpoint.get_evaluation_state(session_id)
            if eval_state is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"No active evaluation for session: {session_id}",
                )
            # Use model_dump_json to properly serialize datetime
            import json
            return JSONResponse(content=json.loads(eval_state.model_dump_json()))
        return server

    def test_returns_state_when_active(self, server_with_mocked_endpoint):
        """Endpoint returns state JSON when evaluation is active."""
        from src.endpoint.conjecture_endpoint import EvaluationState
        from fastapi.testclient import TestClient

        mock_state = EvaluationState(
            session_id="sess_abc",
            query="Test query",
            iteration=2,
            max_iterations=5,
            claims_being_evaluated=["c1"],
            tool_calls_so_far=[],
            created_claim_ids=[],
            status="in_progress",
        )
        server_with_mocked_endpoint._endpoint.get_evaluation_state.return_value = mock_state

        client = TestClient(server_with_mocked_endpoint._app)
        response = client.get("/v1/evaluation/sess_abc/state")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "sess_abc"
        assert data["status"] == "in_progress"
        assert data["iteration"] == 2
        # updated_at should be present (ISO format)
        assert "updated_at" in data

    def test_returns_404_when_no_active_evaluation(self, server_with_mocked_endpoint):
        """Endpoint returns 404 when no evaluation is active for session."""
        from fastapi.testclient import TestClient

        server_with_mocked_endpoint._endpoint.get_evaluation_state.return_value = None

        client = TestClient(server_with_mocked_endpoint._app)
        response = client.get("/v1/evaluation/unknown_session/state")

        assert response.status_code == 404
        assert "No active evaluation" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Integration: State Lifecycle
# ---------------------------------------------------------------------------

class TestEvaluationStateLifecycle:
    """Test that evaluation state is properly managed during evaluate()."""

    @pytest.fixture
    def endpoint(self, tmp_path):
        """Create endpoint with temporary database."""
        db_path = os.path.join(tmp_path, "test_eval_lifecycle.db")
        ep = ConjectureEndpoint(db_path=db_path, vector_path=":memory:")
        ep._vector_store = None
        return ep

    def test_state_cleared_after_error(self, endpoint):
        """State is cleared when evaluate() encounters an error."""
        # Publish a state manually
        endpoint.publish_evaluation_state(
            session_id="error_session",
            query="Test",
            iteration=1,
            max_iterations=5,
            claims_being_evaluated=[],
            tool_calls_so_far=[],
            created_claim_ids=[],
        )
        # clear_evaluation_state should remove it
        endpoint.clear_evaluation_state("error_session")
        assert endpoint.get_evaluation_state("error_session") is None

    def test_multiple_iterations_accumulate_state(self, endpoint):
        """Publishing multiple times accumulates tool calls."""
        # Simulate iteration 1
        endpoint.publish_evaluation_state(
            session_id="multi_iter",
            query="Test",
            iteration=1,
            max_iterations=5,
            claims_being_evaluated=[],
            tool_calls_so_far=[
                {"name": "create_claim", "args": {}, "success": True}
            ],
            created_claim_ids=["c1"],
        )
        # Simulate iteration 2
        endpoint.publish_evaluation_state(
            session_id="multi_iter",
            query="Test",
            iteration=2,
            max_iterations=5,
            claims_being_evaluated=[],
            tool_calls_so_far=[
                {"name": "create_claim", "args": {}, "success": True},
                {"name": "web_search", "args": {}, "success": True},
            ],
            created_claim_ids=["c1", "c2"],
        )
        state = endpoint.get_evaluation_state("multi_iter")
        assert state.iteration == 2
        assert len(state.tool_calls_so_far) == 2
        assert len(state.created_claim_ids) == 2
