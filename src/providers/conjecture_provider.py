#!/usr/bin/env python3
"""
Conjecture Local LLM Provider Server
Provides TellUser and AskUser tools on port 5679
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Conjecture Local Provider",
    description="Local LLM Provider with TellUser and AskUser tools",
    version="1.0.0",
)

class ChatRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    model: str
    messages: list
    max_tokens: int = 42000
    temperature: float = 0.7
    stream: bool = False

class TellUserRequest(BaseModel):
    message: str

class AskUserRequest(BaseModel):
    question: str

class ModelInfo(BaseModel):
    name: str
    description: str
    capabilities: list

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    """Main chat completion endpoint"""
    try:
        # Simple test response for testing
        response_text = f"Processed request with {len(request.messages)} messages for model {request.model}"
        
        return {
            "id": f"chatcmpl-{datetime.now().timestamp()}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }

    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/tell_user")
async def tell_user(request: TellUserRequest):
    """TellUser tool - provides information to user"""
    try:
        return {
            "success": True,
            "message": request.message,
            "response": f"Message delivered: {request.message}",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"TellUser error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/ask_user")
async def ask_user(request: AskUserRequest):
    """AskUser tool - asks questions to user"""
    try:
        return {
            "success": True,
            "question": request.question,
            "response": f"Question for user: {request.question}",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"AskUser error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List available models"""
    try:
        models = [
            ModelInfo(
                name="conjecture-llm",
                description="Conjecture LLM integration",
                capabilities=["reasoning", "tool_use"]
            )
        ]
        
        return {"data": [model.__dict__ for model in models]}

    except Exception as e:
        logger.error(f"List models error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conjecture Local Provider")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5680, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Set port in environment for other scripts to use
    os.environ["CONJECTURE_PROVIDER_PORT"] = str(args.port)
    
    print("Starting Conjecture Local Provider...")
    print("   Available endpoints:")
    print("   POST /v1/chat/completions - Main chat endpoint")
    print("   POST /tools/tell_user - TellUser tool")
    print("   POST /tools/ask_user - AskUser tool")
    print("   GET /models - List models")
    print("   GET /health - Health check")
    print(f"\nStarting server on http://127.0.0.1:{args.port}")
    print("Press Ctrl+C to stop server")
    
    # Run provider
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )