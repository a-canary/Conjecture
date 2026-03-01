"""
HTTP Server for Conjecture LLM Endpoint

Per M-0007: Host Conjecture as LLM endpoint (localhost or VPS).
Provides OpenAI-compatible API that enhances queries with claim context.

Usage:
    python -m src.endpoint.http_server
    # or
    conjecture serve --port 8000
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# FastAPI/uvicorn imports (optional - graceful fallback)
try:
    from fastapi import FastAPI, Header, HTTPException, Request
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None


# ========== OpenAI-Compatible Request/Response Models ==========

class ChatMessage(BaseModel):
    """OpenAI-compatible chat message."""
    role: str = Field(..., description="Message role: system, user, assistant")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = Field(default="conjecture", description="Model name (ignored, uses configured LLM)")
    messages: List[ChatMessage] = Field(..., description="Conversation messages")
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=1024, ge=1, le=4096)
    stream: bool = Field(default=False, description="Streaming not yet supported")


class ChatCompletionChoice(BaseModel):
    """OpenAI-compatible choice in response."""
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionUsage(BaseModel):
    """Token usage stats."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(datetime.utcnow().timestamp()))
    model: str = "conjecture"
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage = Field(default_factory=ChatCompletionUsage)


# ========== Server Implementation ==========

class ConjectureServer:
    """HTTP server that wraps ConjectureEndpoint as OpenAI-compatible API."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.host = host
        self.port = port
        self._endpoint = None
        self._app = None

    async def initialize(self):
        """Initialize the endpoint and FastAPI app."""
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")

        from src.endpoint.conjecture_endpoint import ConjectureEndpoint

        # Initialize endpoint
        self._endpoint = ConjectureEndpoint(db_path="data/conjecture.db")
        await self._endpoint.initialize()
        self._endpoint.start_session(metadata={"type": "http_server"})

        # Create FastAPI app
        self._app = FastAPI(
            title="Conjecture LLM Endpoint",
            description="OpenAI-compatible API with claim-enhanced reasoning",
            version="1.0.0"
        )
        self._setup_routes()

        logger.info(f"Conjecture server initialized on {self.host}:{self.port}")

    def _setup_routes(self):
        """Configure API routes."""
        app = self._app

        @app.get("/")
        async def root():
            return {"service": "conjecture", "version": "1.0.0", "status": "ok"}

        @app.get("/v1/models")
        async def list_models():
            """List available models (OpenAI-compatible)."""
            return {
                "object": "list",
                "data": [
                    {
                        "id": "conjecture",
                        "object": "model",
                        "created": int(datetime.utcnow().timestamp()),
                        "owned_by": "conjecture"
                    }
                ]
            }

        @app.post("/v1/chat/completions")
        async def chat_completions(
            request: ChatCompletionRequest,
            x_session_id: Optional[str] = Header(None, alias="X-Session-ID")
        ):
            """OpenAI-compatible chat completions with claim enhancement."""
            try:
                # Extract user message (last user message in conversation)
                user_messages = [m for m in request.messages if m.role == "user"]
                if not user_messages:
                    raise HTTPException(status_code=400, detail="No user message provided")

                query = user_messages[-1].content

                # Use session from header if provided
                if x_session_id and x_session_id != self._endpoint.get_current_session().id:
                    self._endpoint.start_session(session_id=x_session_id)

                # Call evaluate with claim context
                result = await self._endpoint.evaluate(
                    query=query,
                    max_claims=10,
                    min_confidence=0.5,
                    include_reasoning=True
                )

                if not result.success:
                    raise HTTPException(status_code=500, detail=result.message)

                # Build response
                response_content = result.data.get("response", "")
                claims_used = result.data.get("claims_used", 0)

                response = ChatCompletionResponse(
                    model=result.data.get("model", "conjecture"),
                    choices=[
                        ChatCompletionChoice(
                            message=ChatMessage(role="assistant", content=response_content)
                        )
                    ],
                    usage=ChatCompletionUsage(
                        prompt_tokens=result.data.get("usage", {}).get("prompt_tokens", 0),
                        completion_tokens=result.data.get("usage", {}).get("completion_tokens", 0),
                        total_tokens=result.data.get("usage", {}).get("total_tokens", 0)
                    )
                )

                # Add custom headers for claim info
                headers = {
                    "X-Conjecture-Claims-Used": str(claims_used),
                    "X-Conjecture-Session": self._endpoint.get_current_session().id
                }

                return JSONResponse(
                    content=response.model_dump(),
                    headers=headers
                )

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Chat completion error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/health")
        async def health():
            """Health check endpoint."""
            session = self._endpoint.get_current_session()
            return {
                "status": "healthy",
                "session_id": session.id if session else None,
                "claims_count": self._endpoint.claim_count()
            }

        @app.post("/v1/claims")
        async def create_claim(content: str, confidence: float = 0.5, tags: List[str] = None):
            """Create a new claim (Conjecture-specific endpoint)."""
            result = await self._endpoint.create_claim(
                content=content,
                confidence=confidence,
                tags=tags or []
            )
            if not result.success:
                raise HTTPException(status_code=400, detail=result.message)
            return result.data

    async def run(self):
        """Start the server."""
        if not self._app:
            await self.initialize()

        config = uvicorn.Config(
            self._app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def shutdown(self):
        """Clean shutdown."""
        if self._endpoint:
            await self._endpoint.close()


def main():
    """Entry point for running the server directly."""
    import argparse

    parser = argparse.ArgumentParser(description="Conjecture LLM Endpoint Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen (default: 8000)")
    args = parser.parse_args()

    if not FASTAPI_AVAILABLE:
        print("ERROR: FastAPI not installed. Run: pip install fastapi uvicorn")
        return

    server = ConjectureServer(host=args.host, port=args.port)
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
