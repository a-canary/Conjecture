# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Tests for HTTP Server (Endpoint Layer)

Tests OpenAI-compatible API implementation:
- Request/Response models
- ConjectureServer initialization
- Route handlers (mocked)
- Phase 3: resume route, pause header, pause state inspection route
- Phase 3: end-to-end integration smoke test (in-process FastAPI test client)
"""

import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.endpoint.http_server import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionChoice,
    ChatCompletionUsage,
    ChatCompletionResponse,
    ConjectureServer,
    ResumeRequest,
    FASTAPI_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Model Tests
# ---------------------------------------------------------------------------

class TestChatMessage:
    """Test ChatMessage model."""

    def test_create_user_message(self):
        """Test creating a user message."""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_create_assistant_message(self):
        """Test creating an assistant message."""
        msg = ChatMessage(role="assistant", content="Hi there!")
        assert msg.role == "assistant"
        assert msg.content == "Hi there!"

    def test_create_system_message(self):
        """Test creating a system message."""
        msg = ChatMessage(role="system", content="You are helpful.")
        assert msg.role == "system"


class TestChatCompletionRequest:
    """Test ChatCompletionRequest model."""

    def test_minimal_request(self):
        """Test creating a minimal request."""
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hello")]
        )
        assert request.model == "conjecture"
        assert len(request.messages) == 1
        assert request.temperature == 0.7
        assert request.max_tokens == 1024
        assert request.stream is False

    def test_custom_temperature(self):
        """Test custom temperature setting."""
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Test")],
            temperature=0.3
        )
        assert request.temperature == 0.3

    def test_temperature_bounds(self):
        """Test temperature is within bounds."""
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Test")],
            temperature=2.0
        )
        assert request.temperature == 2.0

    def test_custom_max_tokens(self):
        """Test custom max_tokens setting."""
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Test")],
            max_tokens=256
        )
        assert request.max_tokens == 256

    def test_conversation_history(self):
        """Test multi-turn conversation."""
        request = ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content="You are helpful"),
                ChatMessage(role="user", content="Hi"),
                ChatMessage(role="assistant", content="Hello!"),
                ChatMessage(role="user", content="What's 2+2?"),
            ]
        )
        assert len(request.messages) == 4
        assert request.messages[-1].content == "What's 2+2?"


class TestChatCompletionChoice:
    """Test ChatCompletionChoice model."""

    def test_default_values(self):
        """Test default choice values."""
        choice = ChatCompletionChoice(
            message=ChatMessage(role="assistant", content="Response")
        )
        assert choice.index == 0
        assert choice.finish_reason == "stop"

    def test_custom_index(self):
        """Test custom index."""
        choice = ChatCompletionChoice(
            index=1,
            message=ChatMessage(role="assistant", content="Second"),
            finish_reason="length"
        )
        assert choice.index == 1
        assert choice.finish_reason == "length"


class TestChatCompletionUsage:
    """Test ChatCompletionUsage model."""

    def test_default_values(self):
        """Test default usage values."""
        usage = ChatCompletionUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_custom_values(self):
        """Test custom usage values."""
        usage = ChatCompletionUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150


class TestChatCompletionResponse:
    """Test ChatCompletionResponse model."""

    def test_default_generation(self):
        """Test response ID and timestamp generation."""
        response = ChatCompletionResponse(
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content="Hello")
                )
            ]
        )
        assert response.id.startswith("chatcmpl-")
        assert response.object == "chat.completion"
        assert response.model == "conjecture"
        assert response.created > 0

    def test_with_usage(self):
        """Test response with usage stats."""
        response = ChatCompletionResponse(
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content="Test")
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15
            )
        )
        assert response.usage.total_tokens == 15


# ---------------------------------------------------------------------------
# ConjectureServer Tests
# ---------------------------------------------------------------------------

class TestConjectureServer:
    """Test ConjectureServer initialization and configuration."""

    def test_default_config(self):
        """Test default server configuration."""
        server = ConjectureServer()
        assert server.host == "0.0.0.0"
        assert server.port == 8000
        assert server._endpoint is None
        assert server._app is None

    def test_custom_config(self):
        """Test custom server configuration."""
        server = ConjectureServer(host="127.0.0.1", port=9000)
        assert server.host == "127.0.0.1"
        assert server.port == 9000

    @pytest.mark.asyncio
    @pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
    async def test_initialize_creates_app(self):
        """Test that initialize creates FastAPI app."""
        server = ConjectureServer()

        # Mock ConjectureEndpoint at the import location
        with patch("src.endpoint.conjecture_endpoint.ConjectureEndpoint") as mock_endpoint:
            mock_instance = AsyncMock()
            mock_instance.get_current_session.return_value = MagicMock(id="test-session")
            mock_endpoint.return_value = mock_instance

            # Also patch the import inside initialize
            with patch.dict("sys.modules", {
                "src.endpoint.conjecture_endpoint": MagicMock(ConjectureEndpoint=mock_endpoint)
            }):
                await server.initialize()

                assert server._app is not None
                assert server._endpoint is not None

    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test clean shutdown."""
        server = ConjectureServer()
        mock_endpoint = AsyncMock()
        server._endpoint = mock_endpoint

        await server.shutdown()

        mock_endpoint.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_no_endpoint(self):
        """Test shutdown when no endpoint initialized."""
        server = ConjectureServer()
        # Should not raise
        await server.shutdown()


# ---------------------------------------------------------------------------
# Route Handler Tests (standalone, no async fixtures)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestRouteHandlers:
    """Test route handlers with a simple FastAPI app."""

    def test_route_setup_creates_endpoints(self):
        """Test that _setup_routes creates expected endpoints."""
        from fastapi import FastAPI

        server = ConjectureServer()
        # Create minimal mock endpoint
        mock_endpoint = MagicMock()
        mock_endpoint.get_current_session.return_value = MagicMock(id="test")
        mock_endpoint.claim_count.return_value = 0

        server._endpoint = mock_endpoint
        server._app = FastAPI()
        server._setup_routes()

        # Check routes were added
        routes = [r.path for r in server._app.routes]
        assert "/" in routes
        assert "/v1/models" in routes
        assert "/v1/chat/completions" in routes
        assert "/health" in routes
        assert "/v1/claims" in routes


# ---------------------------------------------------------------------------
# FASTAPI_AVAILABLE flag test
# ---------------------------------------------------------------------------

class TestFastAPIAvailable:
    """Test FASTAPI_AVAILABLE detection."""

    def test_fastapi_available_is_bool(self):
        """Test FASTAPI_AVAILABLE is a boolean."""
        assert isinstance(FASTAPI_AVAILABLE, bool)

    @pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
    def test_can_import_fastapi(self):
        """Test FastAPI can be imported when available."""
        from fastapi import FastAPI
        assert FastAPI is not None


# ---------------------------------------------------------------------------
# Phase 3: ResumeRequest model tests
# ---------------------------------------------------------------------------

class TestResumeRequest:
    """Test the ResumeRequest Pydantic model introduced in Phase 3."""

    def test_basic_construction(self):
        """ResumeRequest can be built with pause_id and results."""
        req = ResumeRequest(pause_id="abc-123", results=["Paris is the capital."])
        assert req.pause_id == "abc-123"
        assert req.results == ["Paris is the capital."]

    def test_empty_results_list(self):
        """ResumeRequest accepts an empty results list."""
        req = ResumeRequest(pause_id="xyz", results=[])
        assert req.results == []

    def test_multiple_results(self):
        """ResumeRequest stores all result strings."""
        req = ResumeRequest(
            pause_id="p1",
            results=["Result one.", "Result two.", "Result three."],
        )
        assert len(req.results) == 3


# ---------------------------------------------------------------------------
# Phase 3: Route registration tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestPhase3RouteRegistration:
    """Phase 3 routes are registered by _setup_routes."""

    def test_new_routes_registered(self):
        """POST /resume and GET /pause/{pause_id} routes are registered."""
        from fastapi import FastAPI

        server = ConjectureServer()
        mock_endpoint = MagicMock()
        mock_endpoint.get_current_session.return_value = MagicMock(id="test")
        mock_endpoint.claim_count.return_value = 0

        server._endpoint = mock_endpoint
        server._app = FastAPI()
        server._setup_routes()

        paths = [r.path for r in server._app.routes]
        assert "/v1/chat/completions/resume" in paths, (
            f"Expected /v1/chat/completions/resume in routes, got: {paths}"
        )
        assert "/v1/pause/{pause_id}" in paths, (
            f"Expected /v1/pause/{{pause_id}} in routes, got: {paths}"
        )


# ---------------------------------------------------------------------------
# Phase 3: Integration smoke test (in-process, using FastAPI TestClient)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestPhase3Integration:
    """End-to-end smoke test using FastAPI's TestClient.

    The ConjectureEndpoint is replaced by a mock so no real LLM or database
    is required.  This validates the full HTTP layer:
      1. POST /v1/chat/completions → 200 with X-Conjecture-Pause-ID header
      2. GET  /v1/pause/{pause_id} → 200 with PausedReasoningState JSON
      3. POST /v1/chat/completions/resume → 200 with final response
    """

    def _build_server_with_mock_endpoint(self, mock_endpoint) -> "ConjectureServer":
        """Create a ConjectureServer with a pre-wired mock endpoint and FastAPI app."""
        from fastapi import FastAPI

        server = ConjectureServer()
        server._endpoint = mock_endpoint
        server._app = FastAPI()
        server._setup_routes()
        return server

    def _make_paused_evaluate_result(self, pause_id: str):
        """Build an APIResponse-like MagicMock that returns status='paused'."""
        from src.endpoint.conjecture_endpoint import APIResponse

        return APIResponse(
            success=True,
            message="Evaluation paused — awaiting retrieval results",
            data={
                "status": "paused",
                "pause_id": pause_id,
                "retrieval_request": {
                    "query": "What is the capital of France?",
                    "tool_hint": None,
                    "claim_ids": [],
                },
                "query": "What is the capital of France?",
                "tool_calls_so_far": [],
                "created_claim_ids": [],
                "claims_used": 0,
            },
        )

    def _make_complete_resume_result(self, response_text: str):
        """Build an APIResponse-like MagicMock that returns status='complete'."""
        from src.endpoint.conjecture_endpoint import APIResponse

        return APIResponse(
            success=True,
            message="Evaluation complete (resumed)",
            data={
                "status": "complete",
                "response": response_text,
                "tool_calls": [],
                "created_claim_ids": [],
                "supporting_claims": [],
                "evidence_claims_count": 1,
                "model": "mock-model",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    def _make_paused_state(self, pause_id: str):
        """Build a PausedReasoningState to store in the endpoint's _paused_states."""
        from src.process.claim_tools import PausedReasoningState, RetrievalRequest

        return PausedReasoningState(
            session_id="session-smoke-test",
            iteration=0,
            messages=[],
            pending_retrieval=RetrievalRequest(
                query="What is the capital of France?",
                tool_hint=None,
                claim_ids=[],
            ),
            created_claim_ids=[],
        )

    def test_pause_header_present_when_paused(self):
        """POST /v1/chat/completions returns 200 with X-Conjecture-Pause-ID when paused."""
        from fastapi.testclient import TestClient

        pause_id = "smoke-pause-001"
        paused_response = self._make_paused_evaluate_result(pause_id)

        mock_endpoint = MagicMock()
        mock_endpoint.get_current_session.return_value = MagicMock(id="test-session")
        mock_endpoint.evaluate = AsyncMock(return_value=paused_response)

        server = self._build_server_with_mock_endpoint(mock_endpoint)
        client = TestClient(server._app)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "What is the capital of France?"}]
            },
        )

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        assert "X-Conjecture-Pause-ID" in resp.headers, (
            f"Expected X-Conjecture-Pause-ID header, got: {dict(resp.headers)}"
        )
        assert resp.headers["X-Conjecture-Pause-ID"] == pause_id, (
            f"Header value mismatch: {resp.headers['X-Conjecture-Pause-ID']!r} != {pause_id!r}"
        )

    def test_pause_response_body_is_well_formed(self):
        """Paused /v1/chat/completions body is OpenAI-compatible with JSON content."""
        from fastapi.testclient import TestClient

        pause_id = "smoke-pause-002"
        paused_response = self._make_paused_evaluate_result(pause_id)

        mock_endpoint = MagicMock()
        mock_endpoint.get_current_session.return_value = MagicMock(id="test-session")
        mock_endpoint.evaluate = AsyncMock(return_value=paused_response)

        server = self._build_server_with_mock_endpoint(mock_endpoint)
        client = TestClient(server._app)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Capital question"}]
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert "choices" in body, f"Expected 'choices' in body: {body}"
        assert len(body["choices"]) == 1

        # Content should be parseable JSON describing the paused state
        content_str = body["choices"][0]["message"]["content"]
        content = json.loads(content_str)
        assert content["status"] == "paused"
        assert content["pause_id"] == pause_id
        assert "retrieval_request" in content

    def test_get_pause_state_returns_json(self):
        """GET /v1/pause/{pause_id} returns the PausedReasoningState as JSON."""
        from fastapi.testclient import TestClient

        pause_id = "smoke-pause-003"
        paused_state = self._make_paused_state(pause_id)

        mock_endpoint = MagicMock()
        mock_endpoint.get_current_session.return_value = MagicMock(id="test-session")
        mock_endpoint._paused_states = {pause_id: paused_state}

        server = self._build_server_with_mock_endpoint(mock_endpoint)
        client = TestClient(server._app)

        resp = client.get(f"/v1/pause/{pause_id}")

        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.text}"
        )
        body = resp.json()
        assert body["session_id"] == "session-smoke-test"
        assert "pending_retrieval" in body
        assert body["pending_retrieval"]["query"] == "What is the capital of France?"

    def test_get_pause_state_404_for_unknown(self):
        """GET /v1/pause/{pause_id} returns 404 for unknown pause_id."""
        from fastapi.testclient import TestClient

        mock_endpoint = MagicMock()
        mock_endpoint.get_current_session.return_value = MagicMock(id="test-session")
        mock_endpoint._paused_states = {}

        server = self._build_server_with_mock_endpoint(mock_endpoint)
        client = TestClient(server._app)

        resp = client.get("/v1/pause/does-not-exist")
        assert resp.status_code == 404

    def test_resume_route_returns_final_response(self):
        """POST /v1/chat/completions/resume returns 200 with final response text."""
        from fastapi.testclient import TestClient

        pause_id = "smoke-pause-004"
        final_response_text = "Paris is the capital of France."
        complete_result = self._make_complete_resume_result(final_response_text)

        mock_endpoint = MagicMock()
        mock_endpoint.get_current_session.return_value = MagicMock(id="test-session")
        mock_endpoint.resume_evaluation = AsyncMock(return_value=complete_result)

        server = self._build_server_with_mock_endpoint(mock_endpoint)
        client = TestClient(server._app)

        resp = client.post(
            "/v1/chat/completions/resume",
            json={
                "pause_id": pause_id,
                "results": ["The capital of France is Paris."],
            },
        )

        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.text}"
        )
        body = resp.json()
        assert "choices" in body
        content = body["choices"][0]["message"]["content"]
        assert content == final_response_text, (
            f"Expected final response text, got: {content!r}"
        )

    def test_resume_route_404_for_unknown_pause_id(self):
        """POST /v1/chat/completions/resume returns 404 for unknown pause_id."""
        from src.endpoint.conjecture_endpoint import APIResponse
        from fastapi.testclient import TestClient

        error_result = APIResponse(
            success=False,
            message="No paused session found for pause_id: bad-id",
            errors=["PAUSE_ID_NOT_FOUND"],
        )

        mock_endpoint = MagicMock()
        mock_endpoint.get_current_session.return_value = MagicMock(id="test-session")
        mock_endpoint.resume_evaluation = AsyncMock(return_value=error_result)

        server = self._build_server_with_mock_endpoint(mock_endpoint)
        client = TestClient(server._app)

        resp = client.post(
            "/v1/chat/completions/resume",
            json={"pause_id": "bad-id", "results": []},
        )

        assert resp.status_code == 404, (
            f"Expected 404, got {resp.status_code}: {resp.text}"
        )

    def test_end_to_end_pause_then_resume(self, tmp_path):
        """Integration smoke test: POST → pause → GET state → POST resume → complete.

        This is the Phase 3 gate test. Uses real ConjectureEndpoint (not mocked)
        with mocked LLM so no external services are needed.
        """
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from src.endpoint.conjecture_endpoint import ConjectureEndpoint

        # ------------------------------------------------------------------
        # Build a real endpoint backed by a temp SQLite DB
        # ------------------------------------------------------------------
        db_path = str(tmp_path / "smoke_test.db")

        async def _init_endpoint():
            ep = ConjectureEndpoint(db_path=db_path, vector_path=":memory:")
            ep._vector_store = None
            await ep._data_manager.initialize()
            ep._initialized = True
            ep.start_session(metadata={"type": "smoke_test"})
            return ep

        import asyncio
        loop = asyncio.new_event_loop()
        real_endpoint = loop.run_until_complete(_init_endpoint())

        # ------------------------------------------------------------------
        # Build the server with real endpoint
        # ------------------------------------------------------------------
        server = ConjectureServer()
        server._endpoint = real_endpoint
        server._app = FastAPI()
        server._setup_routes()
        client = TestClient(server._app)

        # ------------------------------------------------------------------
        # Step 1: POST /v1/chat/completions with an LLM stub that pauses
        # ------------------------------------------------------------------
        pause_id = "e2e-smoke-pause"

        rk_response = {
            "content": "",
            "tool_calls": [
                {
                    "id": "call_rk",
                    "name": "retrieve_knowledge",
                    "arguments": {"query": "capital of France"},
                }
            ],
            "model": "mock-model",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        mock_pause_llm = MagicMock()
        mock_pause_llm.generate_with_tools = AsyncMock(return_value=rk_response)
        mock_pause_llm.generate = AsyncMock(return_value={"content": "", "model": "mock", "usage": {}})
        mock_pause_llm.close = AsyncMock()

        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=mock_pause_llm):
            resp1 = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Capital of France?"}]},
            )

        assert resp1.status_code == 200, f"Step 1 failed: {resp1.text}"
        assert "X-Conjecture-Pause-ID" in resp1.headers, (
            "Step 1: X-Conjecture-Pause-ID header missing"
        )

        returned_pause_id = resp1.headers["X-Conjecture-Pause-ID"]
        assert returned_pause_id, "Step 1: pause_id in header must be non-empty"

        # ------------------------------------------------------------------
        # Step 2: GET /v1/pause/{pause_id} to inspect the paused state
        # ------------------------------------------------------------------
        resp2 = client.get(f"/v1/pause/{returned_pause_id}")
        assert resp2.status_code == 200, f"Step 2 failed: {resp2.text}"
        state_body = resp2.json()
        assert "pending_retrieval" in state_body, (
            f"Step 2: pending_retrieval missing from state: {state_body}"
        )
        assert state_body["pending_retrieval"]["query"] == "capital of France", (
            f"Step 2: unexpected query in state: {state_body['pending_retrieval']['query']!r}"
        )

        # ------------------------------------------------------------------
        # Step 3: POST /v1/chat/completions/resume with retrieval results
        # ------------------------------------------------------------------
        final_answer = "Paris is the capital of France."

        resume_response = {
            "content": "",
            "tool_calls": [
                {
                    "id": "call_resp",
                    "name": "respond_to_user",
                    "arguments": {
                        "response": final_answer,
                        "supporting_claims": [],
                    },
                }
            ],
            "model": "mock-model",
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }

        mock_resume_llm = MagicMock()
        mock_resume_llm.generate_with_tools = AsyncMock(return_value=resume_response)
        mock_resume_llm.generate = AsyncMock(return_value={"content": "", "model": "mock", "usage": {}})
        mock_resume_llm.close = AsyncMock()

        with patch("src.endpoint.conjecture_endpoint.decompose_input",
                   new=AsyncMock(return_value=[])), \
             patch("src.endpoint.llm_client.LLMClient", return_value=mock_resume_llm):
            resp3 = client.post(
                "/v1/chat/completions/resume",
                json={
                    "pause_id": returned_pause_id,
                    "results": ["The capital of France is Paris."],
                },
            )

        assert resp3.status_code == 200, f"Step 3 failed: {resp3.text}"
        body3 = resp3.json()
        assert "choices" in body3, f"Step 3: no choices in response: {body3}"
        assert body3["choices"][0]["message"]["content"] == final_answer, (
            f"Step 3: unexpected content: {body3['choices'][0]['message']['content']!r}"
        )

        # Paused state should be consumed
        resp4 = client.get(f"/v1/pause/{returned_pause_id}")
        assert resp4.status_code == 404, (
            "Step 4: paused state should be removed after resume, got {resp4.status_code}"
        )

        loop.close()


# ---------------------------------------------------------------------------
# Async/sync wrapper pattern tests (DeprecationWarning fix)
# ----------------------------------------------------------------------------

@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestAsyncSyncWrapperNoDeprecationWarning:
    """Verify tree/trace/graph endpoints don't call asyncio.get_event_loop() (deprecated in Python 3.10+).

    The tree, trace, and graph endpoints use a sync wrapper pattern:
      async def get_claim_by_id(cid): ...
      def get_claim_sync(cid): asyncio.get_event_loop().run_until_complete(...)

    In Python 3.10+ asyncio.get_event_loop() raises DeprecationWarning when there
    is no current event loop, and RuntimeError in Python 3.12+.
    The fix is to use asyncio.new_event_loop() in a with-statement instead.
    """

    def _server_with_mock(self):
        """Return a server with routes set up and a mock endpoint."""
        from fastapi import FastAPI

        server = ConjectureServer()
        # Mock endpoint returning a minimal claim with one sub
        mock_claim = MagicMock()
        mock_claim.id = "claim-1"
        mock_claim.content = "Root claim"
        mock_claim.confidence = 0.9
        mock_claim.state = MagicMock(value="confirmed")
        mock_claim.type = []
        mock_claim.subs = []
        mock_claim.supers = []

        mock_endpoint = MagicMock()
        mock_endpoint.get_claim = AsyncMock(return_value=MagicMock(
            success=True,
            data={
                "id": "c00000001",
                "content": "Root claim",
                "confidence": 0.9,
                "state": "Validated",
                "type": [],
                "tags": [],
                "subs": [],
                "supers": [],
            }
        ))
        mock_endpoint.get_current_session.return_value = MagicMock(id="test")
        mock_endpoint.claim_count.return_value = 1

        server._endpoint = mock_endpoint
        server._app = FastAPI()
        server._setup_routes()
        return server

    def test_tree_endpoint_no_get_event_loop_call(self):
        """GET /v1/claims/{id}/tree must not call asyncio.get_event_loop()."""
        from fastapi.testclient import TestClient
        import warnings

        server = self._server_with_mock()
        client = TestClient(server._app, raise_server_exceptions=True)

        with patch("asyncio.get_event_loop") as mock_get_loop:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                response = client.get("/v1/claims/claim-1/tree")

            # Verify the endpoint worked (200)
            assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

            # Verify asyncio.get_event_loop was NOT called (the fix)
            assert mock_get_loop.call_count == 0, (
                f"asyncio.get_event_loop() was called {mock_get_loop.call_count} time(s) — "
                f"should use new_event_loop() pattern instead"
            )

            # Verify no DeprecationWarning about get_event_loop in our code
            for warning in w:
                assert "get_event_loop" not in str(warning.message), (
                    f"DeprecationWarning about get_event_loop: {warning.message}"
                )

    def test_trace_endpoint_no_get_event_loop_call(self):
        """GET /v1/claims/{id}/trace must not call asyncio.get_event_loop()."""
        from fastapi.testclient import TestClient
        import warnings

        server = self._server_with_mock()
        client = TestClient(server._app, raise_server_exceptions=True)

        with patch("asyncio.get_event_loop") as mock_get_loop:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                response = client.get("/v1/claims/claim-1/trace")

            assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
            assert mock_get_loop.call_count == 0, (
                f"asyncio.get_event_loop() was called {mock_get_loop.call_count} time(s)"
            )
            for warning in w:
                assert "get_event_loop" not in str(warning.message)

    def test_graph_endpoint_no_get_event_loop_call(self):
        """GET /v1/claims/{id}/graph must not call asyncio.get_event_loop()."""
        from fastapi.testclient import TestClient
        import warnings

        server = self._server_with_mock()
        client = TestClient(server._app, raise_server_exceptions=True)

        with patch("asyncio.get_event_loop") as mock_get_loop:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                response = client.get("/v1/claims/claim-1/graph")

            assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
            assert mock_get_loop.call_count == 0, (
                f"asyncio.get_event_loop() was called {mock_get_loop.call_count} time(s)"
            )
            for warning in w:
                assert "get_event_loop" not in str(warning.message)
