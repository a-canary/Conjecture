# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
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
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.utils.id_utils import generate_id

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
    id: str = Field(default_factory=lambda: generate_id("chatcmpl-"))
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(datetime.now(timezone.utc).timestamp()))
    model: str = "conjecture"
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage = Field(default_factory=ChatCompletionUsage)


# ========== Phase 3: Resume Request Model ==========

class ResumeRequest(BaseModel):
    """Request body for POST /v1/chat/completions/resume."""
    pause_id: str = Field(..., description="The pause_id returned by a paused completion")
    results: List[str] = Field(
        ...,
        description="List of retrieval result strings to incorporate as evidence claims",
    )


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
                        "created": int(datetime.now(timezone.utc).timestamp()),
                        "owned_by": "conjecture"
                    }
                ]
            }

        @app.post("/v1/chat/completions")
        async def chat_completions(
            request: ChatCompletionRequest,
            x_session_id: Optional[str] = Header(None, alias="X-Session-ID")
        ):
            """OpenAI-compatible chat completions with claim enhancement.

            When the underlying evaluation is paused (awaiting retrieval),
            the response includes the custom header ``X-Conjecture-Pause-ID``
            and the assistant message content carries a JSON-encoded description
            of the retrieval request so that OpenAI-compatible clients receive a
            well-formed response body.
            """
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

                data = result.data or {}
                status = data.get("status", "complete")

                # ------------------------------------------------------------------
                # Phase 3 Step 3.2: Handle paused status
                # ------------------------------------------------------------------
                if status == "paused":
                    pause_id = data.get("pause_id", "")
                    retrieval_request = data.get("retrieval_request", {})

                    # Build a well-formed assistant message content with retrieval info
                    paused_content = json.dumps({
                        "status": "paused",
                        "pause_id": pause_id,
                        "retrieval_request": retrieval_request,
                    })

                    response = ChatCompletionResponse(
                        model="conjecture",
                        choices=[
                            ChatCompletionChoice(
                                message=ChatMessage(
                                    role="assistant",
                                    content=paused_content,
                                ),
                                finish_reason="stop",
                            )
                        ],
                        usage=ChatCompletionUsage(
                            prompt_tokens=0,
                            completion_tokens=0,
                            total_tokens=0,
                        ),
                    )

                    headers = {
                        "X-Conjecture-Claims-Used": str(data.get("claims_used", 0)),
                        "X-Conjecture-Session": self._endpoint.get_current_session().id,
                        "X-Conjecture-Pause-ID": pause_id,
                    }

                    return JSONResponse(
                        content=response.model_dump(),
                        headers=headers,
                    )

                # ------------------------------------------------------------------
                # Normal (complete) path
                # ------------------------------------------------------------------
                response_content = data.get("response", "")
                claims_used = data.get("claims_used", 0)

                response = ChatCompletionResponse(
                    model=data.get("model", "conjecture"),
                    choices=[
                        ChatCompletionChoice(
                            message=ChatMessage(role="assistant", content=response_content)
                        )
                    ],
                    usage=ChatCompletionUsage(
                        prompt_tokens=data.get("usage", {}).get("prompt_tokens", 0),
                        completion_tokens=data.get("usage", {}).get("completion_tokens", 0),
                        total_tokens=data.get("usage", {}).get("total_tokens", 0)
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

        # ------------------------------------------------------------------
        # Phase 3 Step 3.1: POST /v1/chat/completions/resume
        # ------------------------------------------------------------------

        @app.post("/v1/chat/completions/resume")
        async def resume_completions(request: ResumeRequest):
            """Resume a paused chat completion after the caller has performed retrieval.

            Accepts a JSON body with:
            - pause_id: The ID returned by a paused /v1/chat/completions response
            - results: List of retrieval result strings to inject as evidence

            Returns an OpenAI-compatible response identical in shape to
            /v1/chat/completions.
            """
            try:
                result = await self._endpoint.resume_evaluation(
                    pause_id=request.pause_id,
                    retrieval_results=request.results,
                )

                if not result.success:
                    raise HTTPException(status_code=404, detail=result.message)

                data = result.data or {}
                status = data.get("status", "complete")

                # Handle another pause (chained retrieval)
                if status == "paused":
                    pause_id = data.get("pause_id", "")
                    retrieval_request = data.get("retrieval_request", {})

                    paused_content = json.dumps({
                        "status": "paused",
                        "pause_id": pause_id,
                        "retrieval_request": retrieval_request,
                    })

                    response = ChatCompletionResponse(
                        model="conjecture",
                        choices=[
                            ChatCompletionChoice(
                                message=ChatMessage(
                                    role="assistant",
                                    content=paused_content,
                                ),
                                finish_reason="stop",
                            )
                        ],
                        usage=ChatCompletionUsage(),
                    )

                    headers = {
                        "X-Conjecture-Pause-ID": pause_id,
                        "X-Conjecture-Session": self._endpoint.get_current_session().id,
                    }

                    return JSONResponse(
                        content=response.model_dump(),
                        headers=headers,
                    )

                # Complete path
                response_content = data.get("response", "")

                response = ChatCompletionResponse(
                    model=data.get("model", "conjecture"),
                    choices=[
                        ChatCompletionChoice(
                            message=ChatMessage(role="assistant", content=response_content)
                        )
                    ],
                    usage=ChatCompletionUsage(
                        prompt_tokens=data.get("usage", {}).get("prompt_tokens", 0),
                        completion_tokens=data.get("usage", {}).get("completion_tokens", 0),
                        total_tokens=data.get("usage", {}).get("total_tokens", 0),
                    ),
                )

                headers = {
                    "X-Conjecture-Session": self._endpoint.get_current_session().id,
                }

                return JSONResponse(
                    content=response.model_dump(),
                    headers=headers,
                )

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Resume completion error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # ------------------------------------------------------------------
        # Phase 3 Step 3.3: GET /v1/pause/{pause_id}
        # ------------------------------------------------------------------

        @app.get("/v1/pause/{pause_id}")
        async def get_paused_state(pause_id: str):
            """Return the PausedReasoningState for the given pause_id.

            Enables callers to inspect a pending paused session before deciding
            how to fulfil the retrieval request. Returns 404 if the pause_id is
            not found (already resumed, expired, or never existed).
            """
            paused_state = self._endpoint._paused_states.get(pause_id)
            if paused_state is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"No paused session found for pause_id: {pause_id}",
                )
            return JSONResponse(content=paused_state.model_dump())

        # ------------------------------------------------------------------
        # A-0014: GET /v1/evaluation/{session_id}/state — live reasoning polling
        # ------------------------------------------------------------------

        @app.get("/v1/evaluation/{session_id}/state")
        async def get_evaluation_state(session_id: str):
            """Return the current EvaluationState for a session (A-0014).

            Enables callers (CLI, TUI, MCP) to poll for live reasoning breakdown
            during an in-progress evaluation. Returns 404 if no evaluation is
            active for the given session.

            Response fields:
            - status: in_progress | paused | complete | error
            - iteration: current iteration (1-indexed)
            - max_iterations: limit
            - claims_being_evaluated: claim IDs relevant to this evaluation
            - tool_calls_so_far: tools executed with name, success, claim_ids
            - created_claim_ids: claims created during this evaluation
            - current_tool: tool name currently executing (None if between tools)
            - llm_content: LLM text content from current iteration
            - updated_at: ISO timestamp of last state update
            """
            eval_state = self._endpoint.get_evaluation_state(session_id)
            if eval_state is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"No active evaluation for session: {session_id}",
                )
            return JSONResponse(content=eval_state.model_dump())

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

        @app.get("/v1/claims/{claim_id}/tree")
        async def get_claim_tree(
            claim_id: str,
            depth: int = 3,
            min_confidence: float = 0.0
        ):
            """Get claim support tree (UX-0007).
            
            Returns a nested tree of claims showing the support structure.
            Query params:
                depth: max depth to traverse (default: 3, max: 10)
                min_confidence: minimum confidence threshold (0.0-1.0)
            """
            from src.utils.visualization import build_claim_tree
            
            # Cap depth at 10
            depth = min(depth, 10)
            
            # Get the root claim
            claim_result = await self._endpoint.get_claim(claim_id)
            if not claim_result.success:
                raise HTTPException(status_code=404, detail=claim_result.message)
            claim_data = claim_result.data
            
            # Build tree recursively
            async def get_claim_by_id(cid):
                result = await self._endpoint.get_claim(cid)
                if result.success:
                    return result.data
                return None
            
            # Sync wrapper for tree building
            def get_claim_sync(cid):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(get_claim_by_id(cid))
                finally:
                    loop.close()
            
            # Convert dict to Claim-like object
            from src.data.models import Claim
            claim = Claim(**claim_data) if isinstance(claim_data, dict) else claim_data
            
            tree = build_claim_tree(
                claim, get_claim_sync,
                max_depth=depth,
                min_confidence=min_confidence
            )
            
            return tree.to_dict()

        @app.get("/v1/claims/{claim_id}/trace")
        async def get_claim_trace(claim_id: str):
            """Get claim trace from root (UX-0007).
            
            Returns the chain of claims from root to the specified claim.
            """
            from src.utils.visualization import build_claim_trace
            
            # Get the target claim
            claim_result = await self._endpoint.get_claim(claim_id)
            if not claim_result.success:
                raise HTTPException(status_code=404, detail=claim_result.message)
            claim_data = claim_result.data
            
            # Sync wrapper for trace building
            async def get_claim_by_id(cid):
                result = await self._endpoint.get_claim(cid)
                if result.success:
                    return result.data
                return None
            
            def get_claim_sync(cid):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(get_claim_by_id(cid))
                finally:
                    loop.close()
            
            # Convert dict to Claim-like object
            from src.data.models import Claim
            claim = Claim(**claim_data) if isinstance(claim_data, dict) else claim_data
            
            trace = build_claim_trace(claim, get_claim_sync)
            
            return trace.to_dict()

        @app.get("/v1/claims/{claim_id}/graph")
        async def get_claim_graph(
            claim_id: str,
            depth: int = 2
        ):
            """Get claim graph as adjacency list (UX-0007).
            
            Returns D3.js/vis.js compatible graph format.
            Query params:
                depth: max depth to traverse (default: 2)
            """
            from src.utils.visualization import build_claim_graph
            
            # Get the root claim
            claim_result = await self._endpoint.get_claim(claim_id)
            if not claim_result.success:
                raise HTTPException(status_code=404, detail=claim_result.message)
            claim_data = claim_result.data
            
            # Sync wrapper for graph building
            async def get_claim_by_id(cid):
                result = await self._endpoint.get_claim(cid)
                if result.success:
                    return result.data
                return None
            
            def get_claim_sync(cid):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(get_claim_by_id(cid))
                finally:
                    loop.close()
            
            # Convert dict to Claim-like object
            from src.data.models import Claim
            claim = Claim(**claim_data) if isinstance(claim_data, dict) else claim_data
            
            graph = build_claim_graph(
                [claim.id], get_claim_sync,
                max_depth=depth
            )
            
            return graph.to_dict()

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
