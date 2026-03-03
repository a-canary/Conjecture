"""
Tests for HTTP Server (Endpoint Layer)

Tests OpenAI-compatible API implementation:
- Request/Response models
- ConjectureServer initialization
- Route handlers (mocked)
"""

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
