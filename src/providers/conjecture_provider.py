#!/usr/bin/env python3
"""
Conjecture Local LLM Provider Server
Provides TellUser and AskUser tools on port 5678
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config.config import Config
from src.conjecture import Conjecture

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Conjecture Local Provider",
    description="Local LLM Provider with TellUser and AskUser tools",
    version="1.0.0",
)

# Global Conjecture instance
conjecture_instance = None


class ChatRequest(BaseModel):
    model: str
    messages: list
    max_tokens: Optional[int] = 2000
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list
    usage: Dict[str, int]


class TellUserRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = {}


class AskUserRequest(BaseModel):
    question: str
    context: Optional[Dict[str, Any]] = {}
    options: Optional[list] = None


@app.on_event("startup")
async def startup_event():
    """Initialize Conjecture on startup"""
    global conjecture_instance
    try:
        conjecture_instance = Conjecture()
        await conjecture_instance.start_services()
        logger.info("Conjecture provider initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Conjecture: {e}")
        # Continue without Conjecture - will provide mock responses


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global conjecture_instance
    if conjecture_instance:
        try:
            await conjecture_instance.stop_services()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "conjecture_initialized": conjecture_instance is not None,
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """Main chat completions endpoint"""
    try:
        # Extract the last user message
        user_message = ""
        for message in reversed(request.messages):
            if message.get("role") == "user":
                user_message = message.get("content", "")
                break

        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")

        # Process with Conjecture if available
        if conjecture_instance:
            try:
                # Use a simple response since process_task doesn't exist
                # In a full implementation, this would process through Conjecture
                response_content = (
                    f"Conjecture provider received: {user_message[:200]}..."
                )
            except Exception as e:
                logger.error(f"Conjecture processing failed: {e}")
                response_content = f"Conjecture processing error: {str(e)}"
        else:
            # Fallback response
            response_content = f"Conjecture provider received: {user_message[:200]}..."

        # Create OpenAI-compatible response
        response = ChatResponse(
            id=f"chatcmpl-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            created=int(datetime.now().timestamp()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_content},
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": count_tokens(user_message),
                "completion_tokens": count_tokens(response_content),
                "total_tokens": count_tokens(user_message) + count_tokens(response_content),
            },
        )

        return response

    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/tell_user")
async def tell_user(request: TellUserRequest):
    """TellUser tool - provides information to user"""
    try:
        # Process with Conjecture if available
        if conjecture_instance:
            try:
                # Use a simple response since process_task doesn't exist
                # In a full implementation, this would process through Conjecture
                response = f"Conjecture received message: {request.message}"
            except Exception as e:
                logger.error(f"Conjecture processing failed: {e}")
                response = f"Message delivered: {request.message}"
        else:
            response = f"Message delivered: {request.message}"

        return {
            "success": True,
            "message": request.message,
            "response": response,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"TellUser error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/ask_user")
async def ask_user(request: AskUserRequest):
    """AskUser tool - asks questions to user"""
    try:
        # Process with Conjecture if available
        if conjecture_instance:
            try:
                # Use a simple response since process_task doesn't exist
                # In a full implementation, this would process through Conjecture
                response = f"Conjecture received question: {request.question}"
            except Exception as e:
                logger.error(f"Conjecture processing failed: {e}")
                response = f"Question for user: {request.question}"
        else:
            response = f"Question for user: {request.question}"

        return {
            "success": True,
            "question": request.question,
            "response": response,
            "options": request.options,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"AskUser error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [
            {
                "id": "conjecture-local",
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "conjecture",
            }
        ],
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Conjecture Local LLM Provider",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "tell_user": "/tools/tell_user",
            "ask_user": "/tools/ask_user",
            "models": "/models",
            "health": "/health",
        },
    }


def count_tokens(text: str) -> int:
    """Simple token counter"""
    return len(text.split())


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Conjecture Local LLM Provider")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5678, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    print(f"🚀 Starting Conjecture Local Provider on http://{args.host}:{args.port}")
    print("📋 Available endpoints:")
    print("   POST /v1/chat/completions - Main chat endpoint")
    print("   POST /tools/tell_user - TellUser tool")
    print("   POST /tools/ask_user - AskUser tool")
    print("   GET /models - List models")
    print("   GET /health - Health check")

    uvicorn.run(
        "conjecture_provider:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
```

Now let's test the updated examine_responses.py script:
<tool_call>terminal
<arg_key>command</arg_key>
<arg_value>python scripts/examine_responses.py</arg_value>
<arg_key>cd</arg_key>
<arg_value>D:\projects\Conjecture</arg_value>
</tool_call>
