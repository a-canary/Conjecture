# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Claim Tool Schema and Executor — A-0010

Per A-0010: "The LLM is given tools to CRUD claims, respond to user, and invoke
other skills. Responses aren't raw text — they're structured claim operations.
This makes all LLM reasoning traceable through the claim graph."

This module defines:
  - CLAIM_TOOLS: OpenAI function-calling format tool definitions
  - ToolResult: Result dataclass for tool execution
  - ClaimToolExecutor: Executes claim tools against a ClaimRepository
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.core.models import Claim, ClaimType, ClaimState, ClaimScope
from src.data.repositories import ClaimRepository

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool Definitions — OpenAI function-calling format
# ---------------------------------------------------------------------------

CLAIM_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "create_claim",
            "description": (
                "Create a new claim to record a fact, question, or reasoning step. "
                "Every reasoning step the LLM takes should be captured as a claim "
                "so that all reasoning is traceable through the claim graph."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The claim content — a clear, specific statement",
                    },
                    "type": {
                        "type": "string",
                        "enum": [
                            "goal",
                            "assertion",
                            "observation",
                            "assumption",
                            "reference",
                            "impression",
                            "conjecture",
                            "concept",
                            "example",
                        ],
                        "description": "Epistemological category of this claim",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Confidence in this claim from 0.0 (uncertain) to 1.0 (certain)",
                    },
                    "super_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "IDs of existing claims that this new claim provides "
                            "evidence FOR (i.e. this claim supports those claims)"
                        ),
                    },
                },
                "required": ["content", "type", "confidence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_confidence",
            "description": (
                "Update the confidence of an existing claim when new evidence "
                "or reasoning changes how certain we are about it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "claim_id": {
                        "type": "string",
                        "description": "ID of the claim to update",
                    },
                    "new_confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "New confidence value (0.0-1.0)",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Human-readable explanation of why the confidence changed",
                    },
                },
                "required": ["claim_id", "new_confidence", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "respond_to_user",
            "description": (
                "Provide the final response to the user's question or request. "
                "This terminates the tool-calling loop and delivers the answer. "
                "Always cite supporting claims that back up the response."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "The response text to deliver to the user",
                    },
                    "supporting_claims": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of claims in the graph that support this response",
                    },
                },
                "required": ["response"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explore_further",
            "description": (
                "Signal that you want to explore the problem further before responding. "
                "Use this when you're not yet confident in an answer and need to create "
                "more claims to investigate. This is the A-0012 'explore' choice. "
                "The alternative is respond_to_user (halt)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": "What aspect of the problem to explore next",
                    },
                    "confidence_gap": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Current confidence in having a good answer (0.0-1.0)",
                    },
                },
                "required": ["focus"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Retrieval Tool Definitions — A-0015 Delegated Tool Calling
# ---------------------------------------------------------------------------

RETRIEVAL_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_knowledge",
            "description": (
                "Request external knowledge retrieval from the calling system. "
                "When the LLM needs evidence it cannot derive from the current claim "
                "context, it calls this tool to pause reasoning and ask the caller to "
                "perform retrieval (e.g. database search, web search, file read). "
                "The endpoint suspends its reasoning loop and returns a "
                "PausedReasoningState to the caller, who resumes by providing results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural-language description of what information is needed. "
                            "Should be specific enough for the caller to locate relevant evidence."
                        ),
                    },
                    "tool_hint": {
                        "type": "string",
                        "description": (
                            "Optional hint naming the preferred retrieval tool "
                            "(e.g. 'claim_search', 'web_search', 'file_read'). "
                            "The caller may ignore this if the tool is unavailable."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Pydantic models for delegated retrieval (A-0015)
# ---------------------------------------------------------------------------


class RetrievalRequest(BaseModel):
    """Describes a retrieval request emitted by the endpoint when it calls
    ``retrieve_knowledge``.

    The caller receives this inside a ``ToolResult`` with ``paused=True``
    and is responsible for executing the retrieval, then resuming the
    reasoning session with the results.

    Attributes:
        query: Natural-language description of the information needed.
        tool_hint: Optional preferred retrieval tool name.
        claim_ids: IDs of claims created during the reasoning session so far,
            so the caller can pass them back when resuming.
    """

    query: str = Field(..., description="What information is needed")
    tool_hint: Optional[str] = Field(
        default=None,
        description="Preferred retrieval tool name (may be ignored by caller)",
    )
    claim_ids: List[str] = Field(
        default_factory=list,
        description="Claim IDs created so far in this reasoning session",
    )


class PausedReasoningState(BaseModel):
    """Captures the full state of a paused reasoning session so the caller
    can resume it after performing the requested retrieval.

    Attributes:
        session_id: Unique identifier for this reasoning session.
        iteration: Which tool-call iteration the session paused at (0-based).
        messages: The full message history up to the pause point.
        pending_retrieval: The ``RetrievalRequest`` the endpoint is waiting on.
        created_claim_ids: All claim IDs created during this session so far.
    """

    session_id: str = Field(..., description="Unique session identifier")
    iteration: int = Field(
        default=0,
        description="Tool-call iteration index when the session paused (0-based)",
    )
    messages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Full message history at the pause point",
    )
    pending_retrieval: RetrievalRequest = Field(
        ..., description="The retrieval request the caller must fulfil"
    )
    created_claim_ids: List[str] = Field(
        default_factory=list,
        description="All claim IDs created during this session so far",
    )


# ---------------------------------------------------------------------------
# ToolResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    """Result of executing a claim tool.

    Attributes:
        success: Whether the tool executed without error.
        result: The payload returned by the tool (e.g. a Claim, a message str,
                a dict, or a RetrievalRequest depending on the tool).
        claim_ids: IDs of claims created or modified during this execution.
        error: Error message if success is False.
        paused: True when the endpoint has suspended its reasoning loop and is
            waiting for the caller to perform retrieval (A-0015).  When True,
            ``result`` contains a ``RetrievalRequest``.
    """

    success: bool
    result: Any
    claim_ids: List[str] = field(default_factory=list)
    error: Optional[str] = None
    paused: bool = False


# ---------------------------------------------------------------------------
# Claim ID generation
# ---------------------------------------------------------------------------

_COUNTER: int = 0


def _next_claim_id() -> str:
    """Generate a sequential claim ID in the format c#######."""
    global _COUNTER
    _COUNTER += 1
    return f"c{_COUNTER:07d}"


def _reset_counter() -> None:
    """Reset the ID counter (for testing only)."""
    global _COUNTER
    _COUNTER = 0


# ---------------------------------------------------------------------------
# ClaimToolExecutor
# ---------------------------------------------------------------------------


class ClaimToolExecutor:
    """Executes claim tools against a ClaimRepository.

    Per A-0010, the LLM communicates exclusively via structured tool calls.
    This executor bridges LLM tool invocations to repository operations so
    that every reasoning step is persisted as a claim in the claim graph.

    Usage:
        repo = ClaimRepository()
        await repo.initialize()
        executor = ClaimToolExecutor(repo)

        result = await executor.execute_tool("create_claim", {
            "content": "The sky is blue",
            "type": "observation",
            "confidence": 0.95,
        })
        assert result.success
        assert len(result.claim_ids) == 1
    """

    def __init__(self, repository: ClaimRepository) -> None:
        self._repo = repository

    async def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> ToolResult:
        """Route a tool call to the appropriate handler.

        Args:
            tool_name: One of 'create_claim', 'update_confidence',
                'respond_to_user', 'explore_further', 'retrieve_knowledge'.
            args: Tool arguments matching the schema defined in CLAIM_TOOLS
                or RETRIEVAL_TOOLS.

        Returns:
            ToolResult with success flag, result payload, and affected claim IDs.
        """
        dispatch = {
            "create_claim": self._create_claim,
            "update_confidence": self._update_confidence,
            "respond_to_user": self._respond_to_user,
            "explore_further": self._explore_further,
            "retrieve_knowledge": self._retrieve_knowledge,
        }

        handler = dispatch.get(tool_name)
        if handler is None:
            return ToolResult(
                success=False,
                result=None,
                error=f"Unknown tool: '{tool_name}'. "
                      f"Valid tools: {list(dispatch.keys())}",
            )

        try:
            return await handler(args)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Tool '%s' raised an exception: %s", tool_name, exc)
            return ToolResult(
                success=False,
                result=None,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Private handlers
    # ------------------------------------------------------------------

    async def _create_claim(self, args: Dict[str, Any]) -> ToolResult:
        """Handle create_claim tool call.

        Creates a new Claim and persists it via the repository.  If
        super_ids are provided, support relationships are registered so
        that the new claim provides evidence FOR the named super claims.

        Required args:
            content (str): Claim text (min 5 chars).
            type (str): ClaimType value string.
            confidence (float): 0.0-1.0.

        Optional args:
            super_ids (list[str]): Claims this new claim supports.
        """
        content: str = args.get("content", "")
        type_str: str = args.get("type", "observation")
        confidence: float = float(args.get("confidence", 0.5))
        super_ids: List[str] = list(args.get("super_ids") or [])

        # Validate and map type string to ClaimType enum.
        # The tool schema constrains the set; still guard against bad values.
        try:
            claim_type = ClaimType(type_str)
        except ValueError:
            # Fall back to a sensible default rather than hard-failing.
            logger.warning(
                "Unknown claim type '%s', defaulting to 'observation'", type_str
            )
            claim_type = ClaimType.OBSERVATION

        claim_id = _next_claim_id()
        claim = Claim(
            id=claim_id,
            content=content,
            confidence=confidence,
            type=[claim_type],
            state=ClaimState.EXPLORE,
            scope=ClaimScope.USER_WORKSPACE,
            supers=super_ids,
        )

        created_claim = await self._repo.create(claim)

        # Wire up the reverse relationship on super claims (bidirectional).
        for super_id in super_ids:
            super_claim = await self._repo.get_by_id(super_id)
            if super_claim is not None:
                if claim_id not in super_claim.subs:
                    super_claim.subs.append(claim_id)
                    await self._repo.update(super_claim)
            else:
                logger.warning(
                    "super_id '%s' not found in repository; skipping reverse link",
                    super_id,
                )

        logger.debug(
            "create_claim: id=%s type=%s confidence=%.2f supers=%s",
            claim_id,
            claim_type.value,
            confidence,
            super_ids,
        )

        return ToolResult(
            success=True,
            result=created_claim,
            claim_ids=[claim_id],
        )

    async def _update_confidence(self, args: Dict[str, Any]) -> ToolResult:
        """Handle update_confidence tool call.

        Required args:
            claim_id (str): ID of the claim to update.
            new_confidence (float): 0.0-1.0.
            reason (str): Why the confidence changed.
        """
        claim_id: str = args.get("claim_id", "")
        new_confidence: float = float(args.get("new_confidence", 0.5))
        reason: str = args.get("reason", "")

        if not (0.0 <= new_confidence <= 1.0):
            return ToolResult(
                success=False,
                result=None,
                error=f"new_confidence {new_confidence} is outside valid range [0.0, 1.0]",
            )

        claim = await self._repo.get_by_id(claim_id)
        if claim is None:
            return ToolResult(
                success=False,
                result=None,
                error=f"Claim not found: '{claim_id}'",
            )

        old_confidence = claim.confidence
        claim.confidence = new_confidence
        claim.updated = datetime.now(timezone.utc)
        claim.mark_dirty(reason="confidence_change")

        updated_claim = await self._repo.update(claim)

        logger.debug(
            "update_confidence: id=%s %.2f -> %.2f reason='%s'",
            claim_id,
            old_confidence,
            new_confidence,
            reason,
        )

        return ToolResult(
            success=True,
            result={
                "claim": updated_claim,
                "old_confidence": old_confidence,
                "new_confidence": new_confidence,
                "reason": reason,
            },
            claim_ids=[claim_id],
        )

    async def _respond_to_user(self, args: Dict[str, Any]) -> ToolResult:
        """Handle respond_to_user tool call.

        This terminates the LLM tool-calling loop and delivers the final
        answer.  The response is returned as-is; no claim is persisted
        (the caller is responsible for deciding how to surface it).

        Required args:
            response (str): The response text for the user.

        Optional args:
            supporting_claims (list[str]): Claim IDs backing this response.
        """
        response: str = args.get("response", "")
        supporting_claims: List[str] = list(args.get("supporting_claims") or [])

        logger.debug(
            "respond_to_user: len=%d supporting_claims=%s",
            len(response),
            supporting_claims,
        )

        return ToolResult(
            success=True,
            result={
                "response": response,
                "supporting_claims": supporting_claims,
            },
            claim_ids=supporting_claims,
        )

    async def _explore_further(self, args: Dict[str, Any]) -> ToolResult:
        """Handle explore_further tool call (A-0012).

        This signals that the LLM wants to continue exploring rather than
        responding. The tool loop should continue after this call.

        Required args:
            focus (str): What aspect of the problem to explore.

        Optional args:
            confidence_gap (float): Current confidence in having a good answer.
        """
        focus: str = args.get("focus", "")
        confidence_gap: float = float(args.get("confidence_gap", 0.5))

        logger.debug(
            "explore_further: focus='%s' confidence_gap=%.2f",
            focus,
            confidence_gap,
        )

        return ToolResult(
            success=True,
            result={
                "action": "explore",
                "focus": focus,
                "confidence_gap": confidence_gap,
                "message": f"Exploring: {focus}",
            },
            claim_ids=[],
        )

    async def _retrieve_knowledge(self, args: Dict[str, Any]) -> ToolResult:
        """Handle retrieve_knowledge tool call (A-0015).

        This signals that the endpoint needs external evidence it cannot
        derive from the current claim context.  Execution is suspended and
        the caller receives a ``RetrievalRequest`` so it can perform the
        retrieval and resume the session.

        Required args:
            query (str): Natural-language description of what is needed.

        Optional args:
            tool_hint (str): Preferred retrieval tool name.
        """
        query: str = args.get("query", "")
        tool_hint: Optional[str] = args.get("tool_hint")

        retrieval_request = RetrievalRequest(
            query=query,
            tool_hint=tool_hint,
            claim_ids=[],  # caller can populate from session state
        )

        logger.debug(
            "retrieve_knowledge: query='%s' tool_hint=%s — pausing session",
            query,
            tool_hint,
        )

        return ToolResult(
            success=True,
            result=retrieval_request,
            claim_ids=[],
            paused=True,
        )
