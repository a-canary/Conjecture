# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Endpoint-layer Pydantic models.

Extracted from conjecture_endpoint.py to keep the endpoint class focused on
behavior. Three public models live here:

  - EvaluationState (A-0014): streaming evaluation state, polled by callers
    while a tool-calling or reasoning-loop evaluation is in progress.
  - Session (M-0007): scoped claim storage for a single conversation or run.
  - APIResponse (A-0007): standardized response wrapper returned by every
    endpoint method.

Re-exported from conjecture_endpoint for backward compatibility — existing
callers that do `from src.endpoint.conjecture_endpoint import APIResponse`
keep working unchanged.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class EvaluationState(BaseModel):
    """A-0014: Streaming Evaluation State.

    Tracks the real-time state of an in-progress evaluation.
    Exposed via polling endpoint so callers (CLI, TUI, MCP) can display
    live reasoning breakdown for active prompts (UX-0006).
    """
    session_id: str
    query: str
    status: str = Field(
        default="in_progress",
        description="One of: in_progress | paused | complete | error"
    )
    iteration: int = Field(default=0, description="Current iteration number (1-indexed)")
    max_iterations: int = Field(default=5, description="Maximum iterations allowed")
    claims_being_evaluated: List[str] = Field(
        default_factory=list,
        description="Claim IDs currently relevant to evaluation"
    )
    tool_calls_so_far: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Tool calls executed so far with name, args, success"
    )
    created_claim_ids: List[str] = Field(
        default_factory=list,
        description="Claim IDs created during this evaluation"
    )
    current_tool: Optional[str] = Field(
        default=None,
        description="Tool name currently executing (None if between tools)"
    )
    llm_content: Optional[str] = Field(
        default=None,
        description="LLM text content from current iteration"
    )
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(ser_json_tz='aware')


class Session(BaseModel):
    """Session for scoped claim storage per M-0007.

    Sessions persist claims locally during a conversation or benchmark run.
    Claims can be elevated to project/team scope by the LLM.
    """
    id: str = Field(default_factory=lambda: f"s{uuid.uuid4().hex[:8]}")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    claim_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(ser_json_tz='aware')


class APIResponse(BaseModel):
    """Standardized API response wrapper per A-0007.

    All endpoint methods return this wrapper with:
      - success: Whether the operation succeeded
      - data: The payload (claim, list of claims, evaluation result, etc.)
      - message: Human-readable status message
      - errors: List of error details if any
      - timestamp: When the response was generated
    """
    success: bool = Field(..., description="Whether the operation succeeded")
    data: Optional[Any] = Field(default=None, description="Response payload")
    message: str = Field(default="", description="Human-readable status message")
    errors: List[str] = Field(default_factory=list, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")

    model_config = ConfigDict(ser_json_tz='aware')


__all__ = ["EvaluationState", "Session", "APIResponse"]