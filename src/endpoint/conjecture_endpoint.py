# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
ConjectureEndpoint - Single Public API Entry Point

Per A-0003: ConjectureEndpoint as Public API
Per A-0007: Standardized API Response Wrapper
Per M-0007: Host Conjecture as LLM Endpoint

This is the Endpoint layer in the 4-layer architecture (A-0001):
  Presentation (CLI) -> Endpoint (this) -> Process (intelligence) -> Data (storage)

All external consumers (CLI, TUI, Web, MCP) should use this endpoint.
Sessions store claims in local DB; LLM can elevate claims to project/team scope.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Union
from pydantic import BaseModel, ConfigDict, Field

from src.data.data_manager import DataManager
from src.data.models import Claim, ClaimType, ClaimState
from src.core.models import DirtyReason
from src.process.input_decomposer import decompose_input
from src.utils.id_utils import generate_id

logger = logging.getLogger(__name__)


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
    id: str = Field(default_factory=lambda: generate_id("s"))
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


class ConjectureEndpoint:
    """Single public API entry point for Conjecture.

    Per A-0003, this class provides three core methods:
      - create_claim(): Create a new claim with content and confidence
      - get_claim(): Retrieve a claim by its ID
      - evaluate(): Evaluate claims using LLM reasoning (Process layer)

    All methods return APIResponse wrappers for consistent handling.

    Usage:
        endpoint = ConjectureEndpoint()
        await endpoint.initialize()

        # Create a claim
        response = await endpoint.create_claim(
            content="Python is a programming language",
            confidence=0.95,
            tags=["programming", "python"]
        )

        # Get a claim
        response = await endpoint.get_claim("c12345678")

        # Evaluate with reasoning
        response = await endpoint.evaluate(
            query="What is Python?",
            max_claims=10
        )
    """

    def __init__(self, db_path: str = "data/conjecture.db", vector_path: str = None):
        """Initialize the endpoint.

        Args:
            db_path: Path to SQLite database for claim storage
            vector_path: Path for FAISS vector index (default: alongside db)
        """
        self._data_manager = DataManager(db_path)
        self._vector_store = None
        self._vector_path = vector_path or db_path.replace('.db', '_vectors.faiss')
        self._initialized = False
        self._sessions: Dict[str, Session] = {}
        self._current_session: Optional[Session] = None
        # A-0015: stores suspended reasoning sessions keyed by pause_id
        from src.process.claim_tools import PausedReasoningState
        self._paused_states: Dict[str, "PausedReasoningState"] = {}
        # A-0014: stores live evaluation state keyed by session_id
        self._evaluation_states: Dict[str, "EvaluationState"] = {}

    async def initialize(self) -> APIResponse:
        """Initialize the endpoint and underlying data layer.

        Returns:
            APIResponse indicating success or failure
        """
        try:
            await self._data_manager.initialize()

            # Initialize vector store for semantic search
            try:
                from src.data.vector_store import VectorStore
                self._vector_store = VectorStore(index_path=self._vector_path)
                self._vector_store.initialize()
            except ImportError:
                logger.warning("Vector store not available, semantic search disabled")
                self._vector_store = None

            self._initialized = True
            return APIResponse(
                success=True,
                message="ConjectureEndpoint initialized successfully",
                data={"db_path": self._data_manager.db_path}
            )
        except Exception as e:
            logger.error(f"Failed to initialize endpoint: {e}")
            return APIResponse(
                success=False,
                message="Failed to initialize endpoint",
                errors=[str(e)]
            )

    async def close(self) -> APIResponse:
        """Close the endpoint and release resources.

        Returns:
            APIResponse indicating success or failure
        """
        try:
            await self._data_manager.close()
            self._initialized = False
            return APIResponse(
                success=True,
                message="ConjectureEndpoint closed successfully"
            )
        except Exception as e:
            return APIResponse(
                success=False,
                message="Failed to close endpoint",
                errors=[str(e)]
            )

    async def _ensure_initialized(self) -> Optional[APIResponse]:
        """Ensure endpoint is initialized. Returns error response if not.

        Returns:
            None if initialized, APIResponse error if not
        """
        if not self._initialized:
            return APIResponse(
                success=False,
                message="Endpoint not initialized. Call initialize() first.",
                errors=["NOT_INITIALIZED"]
            )
        return None

    # ========== Session Management (M-0007) ==========

    def start_session(self, session_id: Optional[str] = None, metadata: Optional[Dict] = None) -> APIResponse:
        """Start a new session for claim storage.

        Per M-0007, sessions scope claims locally. Claims persist across
        the session and can be elevated to project/team scope.

        Args:
            session_id: Optional custom session ID (auto-generated if not provided)
            metadata: Optional metadata to attach to session

        Returns:
            APIResponse with session data
        """
        session = Session(
            id=session_id or generate_id("s"),
            metadata=metadata or {}
        )
        self._sessions[session.id] = session
        self._current_session = session

        logger.info(f"Started session: {session.id}")
        return APIResponse(
            success=True,
            message=f"Session started: {session.id}",
            data=session.model_dump()
        )

    def get_session(self, session_id: Optional[str] = None) -> APIResponse:
        """Get session info.

        Args:
            session_id: Session ID to retrieve (uses current session if not provided)

        Returns:
            APIResponse with session data or error
        """
        if session_id:
            session = self._sessions.get(session_id)
        else:
            session = self._current_session

        if not session:
            return APIResponse(
                success=False,
                message="No active session",
                errors=["NO_SESSION"]
            )

        return APIResponse(
            success=True,
            message=f"Session: {session.id}",
            data={
                **session.model_dump(),
                "claim_count": len(session.claim_ids)
            }
        )

    def get_current_session(self) -> Optional[Session]:
        """Get current session object (internal use)."""
        return self._current_session

    def claim_count(self, session_id: Optional[str] = None) -> int:
        """Get count of claims in session.

        Args:
            session_id: Session to count (uses current if not provided)

        Returns:
            Number of claims in session
        """
        if session_id:
            session = self._sessions.get(session_id)
        else:
            session = self._current_session

        return len(session.claim_ids) if session else 0

    # ========== A-0014: Streaming Evaluation State ==========

    def publish_evaluation_state(
        self,
        session_id: str,
        query: str,
        iteration: int,
        max_iterations: int,
        claims_being_evaluated: List[str],
        tool_calls_so_far: List[Dict[str, Any]],
        created_claim_ids: List[str],
        status: str = "in_progress",
        current_tool: Optional[str] = None,
        llm_content: Optional[str] = None,
    ) -> None:
        """A-0014: Publish current evaluation state for polling.

        Called inside the tool-calling loop after each iteration so callers
        (CLI, TUI, MCP) can poll GET /v1/evaluation/{session_id}/state and
        display live reasoning breakdown (UX-0006).

        State is ephemeral — cleared when evaluation completes or errors.
        """
        self._evaluation_states[session_id] = EvaluationState(
            session_id=session_id,
            query=query,
            status=status,
            iteration=iteration,
            max_iterations=max_iterations,
            claims_being_evaluated=claims_being_evaluated,
            tool_calls_so_far=tool_calls_so_far,
            created_claim_ids=created_claim_ids,
            current_tool=current_tool,
            llm_content=llm_content,
            updated_at=datetime.utcnow(),
        )
        logger.debug(
            f"A-0014: Published eval state for {session_id}: "
            f"iter {iteration}/{max_iterations}, status={status}"
        )

    def get_evaluation_state(self, session_id: str) -> Optional[EvaluationState]:
        """A-0014: Retrieve current evaluation state for a session.

        Args:
            session_id: The session to get evaluation state for.

        Returns:
            EvaluationState if evaluation is in progress, else None.
        """
        return self._evaluation_states.get(session_id)

    def clear_evaluation_state(self, session_id: str) -> None:
        """A-0014: Clear evaluation state when evaluation ends (complete or error).

        Called by the evaluate() method when it returns.
        """
        self._evaluation_states.pop(session_id, None)

    # ========== Dirty Flag Cascade ==========

    async def _cascade_dirty_to_supers(
        self, super_ids: List[str], reason: DirtyReason = DirtyReason.SUPPORTING_CLAIM_CHANGED
    ) -> Set[str]:
        """Mark each super claim dirty because a sub changed or was created.

        Per A-0011: cascade is unidirectional toward root (only supers are marked,
        never subs). Called after a claim is created or updated with supers.

        Args:
            super_ids: Claim IDs that this claim provides evidence FOR.
            reason: The DirtyReason to record on each super.

        Returns:
            Set of claim IDs that were successfully marked dirty.
        """
        marked: Set[str] = set()
        now = datetime.now(timezone.utc)
        dirty_updates = {
            "is_dirty": True,
            "dirty_reason": reason.value,
            "dirty_timestamp": now.isoformat(),
            "dirty_priority": 8,  # medium-high priority for cascaded dirtying
        }
        for super_id in super_ids:
            try:
                success = await self._data_manager.update_claim(super_id, dirty_updates)
                if success:
                    marked.add(super_id)
                    logger.debug(f"Cascaded dirty flag to super claim {super_id}")
                else:
                    logger.warning(
                        f"Could not cascade dirty flag to {super_id}: claim not found"
                    )
            except Exception as e:
                logger.warning(f"Failed to cascade dirty flag to {super_id}: {e}")
        return marked

    # ========== Core Claim Operations ==========

    async def create_claim(
        self,
        content: str,
        confidence: float = 0.5,
        tags: Optional[List[str]] = None,
        claim_type: Optional[List[ClaimType]] = None,
        state: ClaimState = ClaimState.EXPLORE,
        supers: Optional[List[str]] = None,
        subs: Optional[List[str]] = None,
        **kwargs
    ) -> APIResponse:
        """Create a new claim.

        Per A-0003, this is one of the three core endpoint methods.

        Args:
            content: Claim content (10-5000 characters)
            confidence: Confidence score (0.0-1.0, default 0.5)
            tags: Optional list of tags for categorization
            claim_type: Optional list of ClaimType values
            state: Claim state (default: EXPLORE)
            supers: Optional list of claim IDs this supports
            subs: Optional list of claim IDs that support this
            **kwargs: Additional claim fields

        Returns:
            APIResponse with claim data or error details
        """
        error_response = await self._ensure_initialized()
        if error_response:
            return error_response

        try:
            # Validate content
            if not content or len(content.strip()) < 10:
                return APIResponse(
                    success=False,
                    message="Claim content must be at least 10 characters",
                    errors=["CONTENT_TOO_SHORT"]
                )

            if len(content) > 5000:
                return APIResponse(
                    success=False,
                    message="Claim content must be at most 5000 characters",
                    errors=["CONTENT_TOO_LONG"]
                )

            # Validate confidence
            if not 0.0 <= confidence <= 1.0:
                return APIResponse(
                    success=False,
                    message="Confidence must be between 0.0 and 1.0",
                    errors=["INVALID_CONFIDENCE"]
                )

            # Create the claim via data layer
            claim_id = await self._data_manager.create_claim(
                content=content,
                confidence=confidence,
                tags=tags or [],
                claim_type=claim_type,
                state=state,
                supers=supers or [],
                subs=subs or [],
                **kwargs
            )

            # Add embedding for semantic search
            if self._vector_store:
                try:
                    self._vector_store.add(claim_id, content)
                    self._vector_store.save()
                except Exception as e:
                    logger.warning(f"Failed to add embedding for {claim_id}: {e}")

            # Track claim in current session (M-0007)
            if self._current_session:
                self._current_session.claim_ids.append(claim_id)

            # A-0011: Cascade dirty flags to supers (unidirectional toward root).
            # When a new sub-claim is created that provides evidence FOR existing
            # super claims, those super claims must be re-evaluated.
            marked_dirty_ids: Set[str] = set()
            if supers:
                marked_dirty_ids = await self._cascade_dirty_to_supers(
                    supers, reason=DirtyReason.SUPPORTING_CLAIM_CHANGED
                )

            # Retrieve the created claim for full response
            claim = await self._data_manager.get_claim(claim_id)

            return APIResponse(
                success=True,
                message=f"Claim created: {claim_id}",
                data={
                    "id": claim_id,
                    "content": content,
                    "confidence": confidence,
                    "tags": tags or [],
                    "state": state.value if hasattr(state, 'value') else str(state),
                    "session_id": self._current_session.id if self._current_session else None,
                    "claim": claim.model_dump() if claim else None,
                    "supers_marked_dirty": sorted(marked_dirty_ids),
                }
            )

        except Exception as e:
            logger.error(f"Failed to create claim: {e}")
            return APIResponse(
                success=False,
                message="Failed to create claim",
                errors=[str(e)]
            )

    async def update_claim(
        self,
        claim_id: str,
        updates: Dict[str, Any],
    ) -> APIResponse:
        """Update an existing claim and cascade dirty flags to its supers.

        Per A-0011, when a claim changes (content, confidence, or relationships),
        all super claims are marked dirty so they can be re-evaluated.

        Args:
            claim_id: ID of the claim to update.
            updates: Dictionary of fields to update.

        Returns:
            APIResponse indicating success, with supers_marked_dirty in data.
        """
        error_response = await self._ensure_initialized()
        if error_response:
            return error_response

        try:
            # Fetch the claim before update so we can read its current supers
            original = await self._data_manager.get_claim(claim_id)
            if original is None:
                return APIResponse(
                    success=False,
                    message=f"Claim not found: {claim_id}",
                    errors=["CLAIM_NOT_FOUND"],
                )

            success = await self._data_manager.update_claim(claim_id, updates)
            if not success:
                return APIResponse(
                    success=False,
                    message=f"Update failed for claim: {claim_id}",
                    errors=["UPDATE_FAILED"],
                )

            # A-0011: Cascade dirty flags to supers when content or confidence changed.
            # Also cascade when supers list itself is being changed.
            content_changed = "content" in updates and updates["content"] != original.content
            confidence_changed = (
                "confidence" in updates
                and abs(updates["confidence"] - original.confidence) >= 0.01
            )
            supers_in_update = updates.get("supers", None)

            # Determine effective supers: use updated list if provided, else original
            effective_supers: List[str] = (
                supers_in_update if supers_in_update is not None else list(original.supers)
            )

            marked_dirty_ids: Set[str] = set()
            if effective_supers and (content_changed or confidence_changed or supers_in_update is not None):
                marked_dirty_ids = await self._cascade_dirty_to_supers(
                    effective_supers, reason=DirtyReason.SUPPORTING_CLAIM_CHANGED
                )

            updated_claim = await self._data_manager.get_claim(claim_id)
            return APIResponse(
                success=True,
                message=f"Claim updated: {claim_id}",
                data={
                    "id": claim_id,
                    "claim": updated_claim.model_dump() if updated_claim else None,
                    "supers_marked_dirty": sorted(marked_dirty_ids),
                },
            )

        except Exception as e:
            logger.error(f"Failed to update claim {claim_id}: {e}")
            return APIResponse(
                success=False,
                message=f"Failed to update claim: {claim_id}",
                errors=[str(e)],
            )

    async def get_claim(self, claim_id: str) -> APIResponse:
        """Retrieve a claim by its ID.

        Per A-0003, this is one of the three core endpoint methods.

        Args:
            claim_id: Unique claim identifier (format: c########)

        Returns:
            APIResponse with claim data or error if not found
        """
        error_response = await self._ensure_initialized()
        if error_response:
            return error_response

        try:
            claim = await self._data_manager.get_claim(claim_id)

            if claim is None:
                return APIResponse(
                    success=False,
                    message=f"Claim not found: {claim_id}",
                    errors=["CLAIM_NOT_FOUND"]
                )

            return APIResponse(
                success=True,
                message=f"Claim retrieved: {claim_id}",
                data=claim.model_dump() if hasattr(claim, 'model_dump') else claim
            )

        except Exception as e:
            logger.error(f"Failed to get claim {claim_id}: {e}")
            return APIResponse(
                success=False,
                message=f"Failed to retrieve claim: {claim_id}",
                errors=[str(e)]
            )

    async def evaluate(
        self,
        query: str,
        max_claims: int = 10,
        min_confidence: float = 0.5,
        include_reasoning: bool = True,
        use_decomposition: bool = True,
        use_tools: bool = True,
        max_tool_iterations: int = 5,
        use_reasoning_loop: bool = False,
        route: Optional["QueryType"] = None,
    ) -> APIResponse:
        """Evaluate claims using LLM reasoning.

        Per A-0003, this is one of the three core endpoint methods.
        Per A-0009, input is decomposed into constituent claims before reasoning.
        Per A-0010, the LLM operates via structured claim tools (CLAIM_TOOLS) so
        all reasoning is traceable through the claim graph.  Tool-calling is the
        default mode; set use_tools=False to fall back to plain-text generation.

        The LLM may call any combination of:
          - create_claim: Record a reasoning step as a claim.
          - update_confidence: Revise confidence on an existing claim.
          - respond_to_user: Deliver the final answer (halts the loop).

        When use_reasoning_loop=True the A-0012 ReasoningLoop is used instead of
        the tool-calling loop.  The LLM decides for itself when it has sufficient
        confidence to halt; the result includes the full reasoning trace.

        O-0009 Task-Type Routing: classify_query() maps queries to one of three
        prompt strategies — REASONING (three-prompt, 70B+), RECALL (cot_lite,
        lightweight), MATH (specialized).  Set route= to override auto-detection.

        Args:
            query: Natural language query to evaluate
            max_claims: Maximum claims to include in context (default 10)
            min_confidence: Minimum confidence for claim inclusion (default 0.5)
            include_reasoning: Whether to include reasoning steps (default True)
            use_decomposition: Whether to decompose input before evaluation (default True)
            use_tools: Whether to use A-0010 tool-calling mode (default True)
            max_tool_iterations: Max iterations for tool loop when use_tools=True (default 5)
            use_reasoning_loop: When True, use ReasoningLoop (A-0012) instead of the
                built-in tool-calling loop.  Returns a ReasoningResult in data (default False)
            route: Override auto-classification. REASONING=three-prompt, RECALL=cot_lite,
                MATH=specialized. Default None = auto-detect via classify_query().

        Returns:
            APIResponse with evaluation result and reasoning chain.
            Response data always includes 'decomposed_claims' key (count of decomposed
            claims, 0 if decomposition was skipped or failed).
            In tool mode, data also includes 'tool_calls', 'created_claim_ids',
            and 'supporting_claims'.
            When use_reasoning_loop=True, data includes 'reasoning_result' with the
            full ReasoningResult object.
        """
        error_response = await self._ensure_initialized()
        if error_response:
            return error_response

        # O-0009: classify query for prompt strategy routing
        if route is None:
            from src.agent.task_router import classify_query, QueryType as RT
            query_type = classify_query(query)
        else:
            query_type = route

        logger.info(f"O-0009 routing: query_type={query_type.value}")

        # O-0009: RECALL queries use lightweight prompt (cot_lite) — skip decomposition
        # REASONING and MATH use full three-prompt with decomposition
        if query_type.value == "recall":
            use_decomposition = False

        try:
            from src.endpoint.llm_client import (
                LLMClient, build_claim_context, build_enhanced_prompt,
                DEFAULT_MODEL, TOOL_CAPABLE_MODEL
            )

            # Use tool-capable model when tools are enabled
            model = TOOL_CAPABLE_MODEL if use_tools else DEFAULT_MODEL
            llm = LLMClient(model=model)

            # Step 1: A-0009 Input Decomposition (optional, non-blocking)
            # Decompose the query into constituent claims (questions, assertions, etc.)
            # If decomposition fails for any reason, we continue with the existing flow.
            decomposed_claims = []
            if use_decomposition:
                try:
                    decomposed_claims = await decompose_input(query, llm_client=llm)
                    logger.info(f"Decomposed input into {len(decomposed_claims)} claims")
                except Exception as decomp_err:
                    logger.warning(
                        f"Input decomposition failed (continuing without it): {decomp_err}"
                    )
                    decomposed_claims = []

            # Step 2: Search for relevant existing claims
            # First try query-based search, fall back to listing all claims
            claims_response = await self.search_claims(
                query=query,
                min_confidence=min_confidence,
                limit=max_claims
            )

            claims = []
            if claims_response.success and claims_response.data:
                claims = claims_response.data.get("claims", [])

            # Fallback: if no claims found via search, get all claims above confidence
            if not claims:
                all_claims_response = await self.search_claims(
                    min_confidence=min_confidence,
                    limit=max_claims
                )
                if all_claims_response.success and all_claims_response.data:
                    claims = all_claims_response.data.get("claims", [])

            # Step 3: Merge decomposed claims into context
            # Decomposed claims represent the query's constituent parts and enrich
            # the context passed to the LLM, alongside any retrieved stored claims.
            all_context_claims = list(claims)
            if decomposed_claims:
                # Convert core Claim objects to dicts compatible with build_claim_context
                for dc in decomposed_claims:
                    all_context_claims.append(
                        dc.model_dump() if hasattr(dc, "model_dump") else dc
                    )

            # Step 3b: A-0012 ReasoningLoop path (optional)
            # When use_reasoning_loop=True, delegate to the ReasoningLoop which
            # implements the LLM-directed halt-or-explore decision loop.
            if use_reasoning_loop:
                try:
                    from src.process.reasoning_loop import ReasoningLoop
                    from src.data.repositories import ClaimRepository

                    repo = ClaimRepository()
                    await repo.initialize()

                    reasoning_loop = ReasoningLoop(
                        llm_client=llm,
                        claim_repository=repo,
                        max_iterations=max_tool_iterations,
                    )

                    # Pass context claims as dicts for ReasoningLoop compatibility
                    context_for_loop = [
                        c if isinstance(c, dict) else c
                        for c in all_context_claims
                    ]

                    reasoning_result = await reasoning_loop.run(
                        query=query,
                        context_claims=context_for_loop,
                    )
                    return APIResponse(
                        success=True,
                        message="Evaluation complete (reasoning loop)",
                        data={
                            "query": query,
                            "response": reasoning_result.response,
                            "claims_used": len(claims),
                            "decomposed_claims": len(decomposed_claims),
                            "reasoning_result": {
                                "claims_created": reasoning_result.claims_created,
                                "supporting_claims": reasoning_result.supporting_claims,
                                "iterations": reasoning_result.iterations,
                                "halted_reason": reasoning_result.halted_reason,
                                "tool_calls": reasoning_result.tool_calls,
                            },
                        }
                    )
                finally:
                    await llm.close()

            # Step 4: Build claim context and enhanced prompt
            claim_context = build_claim_context(all_context_claims)
            enhanced_prompt = build_enhanced_prompt(query, claim_context)

            # A-0014: session ID for evaluation state tracking (used in finally/except)
            eval_session_id: Optional[str] = (
                self._current_session.id if self._current_session else None
            )

            # Step 5: Call LLM (tool-calling mode or direct mode)
            try:
                if use_tools:
                    # A-0010: Use tool-calling mode for traceable reasoning
                    from src.process.claim_tools import CLAIM_TOOLS, ClaimToolExecutor, ToolResult
                    from src.data.repositories import ClaimRepository

                    # Initialize tool executor with a session-scoped repository
                    # This allows tool-created claims to be tracked for this evaluation
                    repo = ClaimRepository()
                    await repo.initialize()
                    executor = ClaimToolExecutor(repo)

                    # Tool-calling loop: iterate until respond_to_user or max iterations
                    tool_calls_log = []
                    created_claim_ids: List[str] = []
                    supporting_claims: List[str] = []
                    final_response = None
                    llm_response = None

                    system_prompt = (
                        "You are a reasoning assistant that uses structured tools to build knowledge. "
                        "Use create_claim to record facts and reasoning steps. "
                        "Use update_confidence if evidence changes your certainty about a claim. "
                        "Use respond_to_user to deliver your final answer, citing supporting claims."
                    )

                    # Combine CLAIM_TOOLS + RETRIEVAL_TOOLS so the LLM can request
                    # external knowledge retrieval (A-0015).
                    from src.process.claim_tools import RETRIEVAL_TOOLS, PausedReasoningState, RetrievalRequest
                    all_tools = CLAIM_TOOLS + RETRIEVAL_TOOLS

                    # A-0014: session ID for evaluation state tracking
                    eval_session_id = (
                        self._current_session.id
                        if self._current_session
                        else generate_id("s")
                    )
                    # Claim IDs relevant to this evaluation (from context + created so far)
                    eval_claim_ids: List[str] = list(claims) if claims else []

                    for iteration in range(max_tool_iterations):
                        logger.info(
                            "Tool-calling loop: iteration %d/%d",
                            iteration + 1, max_tool_iterations
                        )
                        llm_response = await llm.generate_with_tools(
                            prompt=enhanced_prompt,
                            tools=all_tools,
                            system_prompt=system_prompt,
                            temperature=0.7,
                            max_tokens=1024
                        )

                        tool_calls = llm_response.get("tool_calls", [])
                        llm_content = llm_response.get("content")

                        # A-0014: publish evaluation state after LLM call
                        self.publish_evaluation_state(
                            session_id=eval_session_id,
                            query=query,
                            iteration=iteration + 1,
                            max_iterations=max_tool_iterations,
                            claims_being_evaluated=eval_claim_ids,
                            tool_calls_so_far=tool_calls_log,
                            created_claim_ids=created_claim_ids,
                            status="in_progress",
                            llm_content=llm_content,
                        )

                        if not tool_calls:
                            # No tool calls — LLM responded with plain text
                            logger.info(
                                "Tool-calling loop: no tool calls in iteration %d, "
                                "treating content as final response",
                                iteration + 1
                            )
                            final_response = llm_response.get("content", "")
                            break

                        # A-0015: Guard — only pause if retrieve_knowledge is in
                        # this iteration's tool names (backward compatibility).
                        iteration_tool_names = {tc.get("name", "") for tc in tool_calls}
                        if "retrieve_knowledge" in iteration_tool_names:
                            # Find the first retrieve_knowledge call
                            rk_tc = next(
                                tc for tc in tool_calls
                                if tc.get("name") == "retrieve_knowledge"
                            )
                            rk_args = rk_tc.get("arguments", {})
                            retrieval_request = RetrievalRequest(
                                query=rk_args.get("query", ""),
                                tool_hint=rk_args.get("tool_hint"),
                                claim_ids=list(created_claim_ids),
                            )
                            pause_id = str(uuid.uuid4())
                            # A-0014: publish paused state before returning
                            self.publish_evaluation_state(
                                session_id=eval_session_id,
                                query=query,
                                iteration=iteration + 1,
                                max_iterations=max_tool_iterations,
                                claims_being_evaluated=eval_claim_ids,
                                tool_calls_so_far=tool_calls_log,
                                created_claim_ids=created_claim_ids,
                                status="paused",
                                llm_content=llm_content,
                            )
                            paused_state = PausedReasoningState(
                                session_id=eval_session_id,
                                iteration=iteration,
                                messages=[],  # LLM message history not available at this layer
                                pending_retrieval=retrieval_request,
                                created_claim_ids=list(created_claim_ids),
                            )
                            # Store paused state so resume_evaluation() can retrieve it
                            self._paused_states[pause_id] = paused_state
                            logger.info(
                                "retrieve_knowledge called at iteration %d — "
                                "pausing session %s with pause_id %s",
                                iteration + 1, eval_session_id, pause_id,
                            )
                            return APIResponse(
                                success=True,
                                message="Evaluation paused — awaiting retrieval results",
                                data={
                                    "status": "paused",
                                    "pause_id": pause_id,
                                    "session_id": eval_session_id,
                                    "retrieval_request": {
                                        "query": retrieval_request.query,
                                        "tool_hint": retrieval_request.tool_hint,
                                        "claim_ids": retrieval_request.claim_ids,
                                    },
                                    "query": query,
                                    "tool_calls_so_far": tool_calls_log,
                                    "created_claim_ids": list(created_claim_ids),
                                }
                            )

                        # Execute each tool call in order (steps 19.4 + 19.5)
                        halted = False
                        for tc in tool_calls:
                            tool_name = tc.get("name", "")
                            tool_args = tc.get("arguments", {})
                            logger.info(
                                "Executing tool '%s' (iteration %d)",
                                tool_name, iteration + 1
                            )
                            # A-0014: publish state with current_tool during execution
                            self.publish_evaluation_state(
                                session_id=eval_session_id,
                                query=query,
                                iteration=iteration + 1,
                                max_iterations=max_tool_iterations,
                                claims_being_evaluated=eval_claim_ids,
                                tool_calls_so_far=tool_calls_log,
                                created_claim_ids=created_claim_ids,
                                status="in_progress",
                                current_tool=tool_name,
                                llm_content=llm_content,
                            )
                            result = await executor.execute_tool(tool_name, tool_args)
                            tool_calls_log.append({
                                "name": tool_name,
                                "arguments": tool_args,
                                "success": result.success,
                                "claim_ids": result.claim_ids,
                                "error": result.error,
                                "iteration": iteration + 1,
                            })

                            if result.success:
                                created_claim_ids.extend(result.claim_ids)

                            # respond_to_user halts the loop (A-0012 foundation)
                            if tool_name == "respond_to_user" and result.success:
                                payload = result.result or {}
                                if isinstance(payload, dict):
                                    final_response = payload.get("response", "")
                                    supporting_claims = list(payload.get("supporting_claims", []))
                                else:
                                    final_response = str(payload)
                                logger.info(
                                    "respond_to_user called — halting tool loop after %d iterations",
                                    iteration + 1
                                )
                                halted = True
                                break

                        if halted or final_response is not None:
                            break

                    # Return tool-based response
                    response_text = (
                        final_response
                        if final_response is not None
                        else (llm_response.get("content", "") if llm_response else "")
                    )
                    # A-0014: clear ephemeral evaluation state on completion
                    self.clear_evaluation_state(eval_session_id)
                    return APIResponse(
                        success=True,
                        message="Evaluation complete (tool mode)",
                        data={
                            "status": "complete",
                            "query": query,
                            "response": response_text,
                            "claims_used": len(claims),
                            "decomposed_claims": len(decomposed_claims),
                            "tool_calls": tool_calls_log,
                            "tool_iterations": len(tool_calls_log),
                            "created_claim_ids": created_claim_ids,
                            "supporting_claims": supporting_claims,
                            "claim_context": claim_context if include_reasoning else None,
                            "model": llm_response.get("model", "unknown") if llm_response else "unknown",
                            "usage": llm_response.get("usage", {}) if llm_response else {}
                        }
                    )

                else:
                    # Direct mode (default): single LLM call without tools
                    llm_response = await llm.generate(
                        prompt=enhanced_prompt,
                        temperature=0.7,
                        max_tokens=1024
                    )
            finally:
                await llm.close()

            # Step 6: Return response with decomposition metadata (direct mode)
            # A-0014: clear ephemeral evaluation state on completion
            self.clear_evaluation_state(eval_session_id)
            return APIResponse(
                success=True,
                message="Evaluation complete",
                data={
                    "query": query,
                    "query_type": query_type.value,  # O-0009 routing decision
                    "response": llm_response.get("content", ""),
                    "claims_used": len(claims),
                    "decomposed_claims": len(decomposed_claims),
                    "claim_context": claim_context if include_reasoning else None,
                    "enhanced_prompt": enhanced_prompt if include_reasoning else None,
                    "model": llm_response.get("model", "unknown"),
                    "usage": llm_response.get("usage", {})
                }
            )

        except ValueError as e:
            # A-0014: clear ephemeral evaluation state on error
            self.clear_evaluation_state(eval_session_id)
            return APIResponse(
                success=False,
                message="LLM not configured",
                errors=[str(e), "Set CHUTES_API_KEY environment variable"]
            )
        except Exception as e:
            logger.error(f"Failed to evaluate query: {e}")
            # A-0014: clear ephemeral evaluation state on error
            self.clear_evaluation_state(eval_session_id)
            return APIResponse(
                success=False,
                message="Failed to evaluate query",
                errors=[str(e)]
            )

    async def resume_evaluation(
        self,
        pause_id: str,
        retrieval_results: List[str],
        max_tool_iterations: int = 5,
        include_reasoning: bool = True,
    ) -> APIResponse:
        """Resume a paused evaluation after the caller has performed retrieval.

        Per A-0015, when evaluate() returns status="paused", the caller performs
        the requested knowledge retrieval and calls this method with the results.
        Each result string is decomposed into claims via decompose_input() and
        appended to the reasoning context, then the tool-calling loop continues
        from where it left off.

        Args:
            pause_id: The pause_id returned by the paused evaluate() call.
            retrieval_results: List of retrieved text strings (e.g. passages,
                facts, document excerpts) to incorporate as evidence claims.
            max_tool_iterations: Maximum additional tool iterations to run
                after resuming (default 5).
            include_reasoning: Whether to include reasoning metadata in response.

        Returns:
            APIResponse with status="complete" and the final response, or
            another status="paused" if a second retrieval is requested, or
            an error response if the pause_id is not found.
        """
        # Look up the paused state
        paused_state = self._paused_states.pop(pause_id, None)
        if paused_state is None:
            return APIResponse(
                success=False,
                message=f"No paused session found for pause_id: {pause_id}",
                errors=["PAUSE_ID_NOT_FOUND"]
            )

        try:
            from src.endpoint.llm_client import (
                LLMClient, build_claim_context, build_enhanced_prompt,
                TOOL_CAPABLE_MODEL
            )
            from src.process.claim_tools import (
                CLAIM_TOOLS, RETRIEVAL_TOOLS, ClaimToolExecutor,
                PausedReasoningState, RetrievalRequest
            )
            from src.data.repositories import ClaimRepository

            llm = LLMClient(model=TOOL_CAPABLE_MODEL)

            try:
                # Step 1: Decompose each retrieval result into evidence claims
                evidence_claims = []
                for result_text in retrieval_results:
                    if not result_text or not result_text.strip():
                        continue
                    try:
                        new_claims = await decompose_input(result_text, llm_client=llm)
                        evidence_claims.extend(new_claims)
                        logger.info(
                            "resume_evaluation: decomposed retrieval result into %d claim(s)",
                            len(new_claims)
                        )
                    except Exception as decomp_err:
                        logger.warning(
                            "resume_evaluation: decomposition of retrieval result failed "
                            "(%s) — using heuristic fallback",
                            decomp_err
                        )
                        # Heuristic fallback: wrap the raw text as-is
                        from src.process.input_decomposer import _heuristic_decompose
                        evidence_claims.extend(_heuristic_decompose(result_text))

                # Step 2: Build a new claim context incorporating the evidence
                evidence_dicts = [
                    c.model_dump() if hasattr(c, "model_dump") else c
                    for c in evidence_claims
                ]
                claim_context = build_claim_context(evidence_dicts)

                # Reconstruct the original query from the paused state's
                # retrieval request (best available proxy for the original prompt)
                original_query = paused_state.pending_retrieval.query
                enhanced_prompt = build_enhanced_prompt(original_query, claim_context)

                # Step 3: Continue the tool-calling loop
                repo = ClaimRepository()
                await repo.initialize()
                executor = ClaimToolExecutor(repo)

                all_tools = CLAIM_TOOLS + RETRIEVAL_TOOLS

                tool_calls_log = []
                created_claim_ids: List[str] = list(paused_state.created_claim_ids)
                supporting_claims: List[str] = []
                final_response = None
                llm_response = None

                system_prompt = (
                    "You are a reasoning assistant that uses structured tools to build knowledge. "
                    "Use create_claim to record facts and reasoning steps. "
                    "Use update_confidence if evidence changes your certainty about a claim. "
                    "Use respond_to_user to deliver your final answer, citing supporting claims."
                )

                for iteration in range(max_tool_iterations):
                    logger.info(
                        "resume_evaluation: tool-calling loop iteration %d/%d",
                        iteration + 1, max_tool_iterations
                    )
                    llm_response = await llm.generate_with_tools(
                        prompt=enhanced_prompt,
                        tools=all_tools,
                        system_prompt=system_prompt,
                        temperature=0.7,
                        max_tokens=1024
                    )

                    tool_calls = llm_response.get("tool_calls", [])
                    llm_content = llm_response.get("content")

                    # A-0014: publish evaluation state after LLM call
                    self.publish_evaluation_state(
                        session_id=paused_state.session_id,
                        query=original_query,
                        iteration=iteration + 1,
                        max_iterations=max_tool_iterations,
                        claims_being_evaluated=[c.get("id") for c in evidence_dicts],
                        tool_calls_so_far=tool_calls_log,
                        created_claim_ids=created_claim_ids,
                        status="in_progress",
                        llm_content=llm_content,
                    )

                    if not tool_calls:
                        logger.info(
                            "resume_evaluation: no tool calls in iteration %d, "
                            "treating content as final response",
                            iteration + 1
                        )
                        final_response = llm_response.get("content", "")
                        break

                    # A-0015: Guard — detect another retrieve_knowledge call
                    iteration_tool_names = {tc.get("name", "") for tc in tool_calls}
                    if "retrieve_knowledge" in iteration_tool_names:
                        rk_tc = next(
                            tc for tc in tool_calls
                            if tc.get("name") == "retrieve_knowledge"
                        )
                        rk_args = rk_tc.get("arguments", {})
                        retrieval_request = RetrievalRequest(
                            query=rk_args.get("query", ""),
                            tool_hint=rk_args.get("tool_hint"),
                            claim_ids=list(created_claim_ids),
                        )
                        new_pause_id = str(uuid.uuid4())
                        session_id = paused_state.session_id
                        new_paused_state = PausedReasoningState(
                            session_id=session_id,
                            iteration=paused_state.iteration + iteration,
                            messages=[],
                            pending_retrieval=retrieval_request,
                            created_claim_ids=list(created_claim_ids),
                        )
                        self._paused_states[new_pause_id] = new_paused_state
                        logger.info(
                            "resume_evaluation: second retrieve_knowledge at iteration %d — "
                            "pausing again with pause_id %s",
                            iteration + 1, new_pause_id,
                        )
                        return APIResponse(
                            success=True,
                            message="Evaluation paused again — awaiting retrieval results",
                            data={
                                "status": "paused",
                                "pause_id": new_pause_id,
                                "retrieval_request": {
                                    "query": retrieval_request.query,
                                    "tool_hint": retrieval_request.tool_hint,
                                    "claim_ids": retrieval_request.claim_ids,
                                },
                                "tool_calls_so_far": tool_calls_log,
                                "created_claim_ids": list(created_claim_ids),
                            }
                        )

                    halted = False
                    for tc in tool_calls:
                        tool_name = tc.get("name", "")
                        tool_args = tc.get("arguments", {})
                        logger.info(
                            "resume_evaluation: executing tool '%s' (iteration %d)",
                            tool_name, iteration + 1
                        )
                        # A-0014: publish state with current_tool during execution
                        self.publish_evaluation_state(
                            session_id=paused_state.session_id,
                            query=original_query,
                            iteration=iteration + 1,
                            max_iterations=max_tool_iterations,
                            claims_being_evaluated=[c.get("id") for c in evidence_dicts],
                            tool_calls_so_far=tool_calls_log,
                            created_claim_ids=created_claim_ids,
                            status="in_progress",
                            current_tool=tool_name,
                            llm_content=llm_content,
                        )
                        result = await executor.execute_tool(tool_name, tool_args)
                        tool_calls_log.append({
                            "name": tool_name,
                            "arguments": tool_args,
                            "success": result.success,
                            "claim_ids": result.claim_ids,
                            "error": result.error,
                            "iteration": paused_state.iteration + iteration + 1,
                        })

                        if result.success:
                            created_claim_ids.extend(result.claim_ids)

                        if tool_name == "respond_to_user" and result.success:
                            payload = result.result or {}
                            if isinstance(payload, dict):
                                final_response = payload.get("response", "")
                                supporting_claims = list(payload.get("supporting_claims", []))
                            else:
                                final_response = str(payload)
                            logger.info(
                                "resume_evaluation: respond_to_user called — halting after %d iterations",
                                iteration + 1
                            )
                            halted = True
                            break

                    if halted or final_response is not None:
                        break

                # Return completed response
                response_text = (
                    final_response
                    if final_response is not None
                    else (llm_response.get("content", "") if llm_response else "")
                )
                # A-0014: clear ephemeral evaluation state on completion
                self.clear_evaluation_state(paused_state.session_id)
                return APIResponse(
                    success=True,
                    message="Evaluation complete (resumed)",
                    data={
                        "status": "complete",
                        "response": response_text,
                        "tool_calls": tool_calls_log,
                        "created_claim_ids": created_claim_ids,
                        "supporting_claims": supporting_claims,
                        "evidence_claims_count": len(evidence_claims),
                        "claim_context": claim_context if include_reasoning else None,
                        "model": llm_response.get("model", "unknown") if llm_response else "unknown",
                        "usage": llm_response.get("usage", {}) if llm_response else {}
                    }
                )

            finally:
                await llm.close()

        except ValueError as e:
            self.clear_evaluation_state(paused_state.session_id)
            return APIResponse(
                success=False,
                message="LLM not configured",
                errors=[str(e), "Set CHUTES_API_KEY environment variable"]
            )
        except Exception as e:
            logger.error(f"resume_evaluation failed: {e}")
            self.clear_evaluation_state(paused_state.session_id)
            return APIResponse(
                success=False,
                message="Failed to resume evaluation",
                errors=[str(e)]
            )

    async def search_claims(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_confidence: float = 0.0,
        max_confidence: float = 1.0,
        limit: int = 100
    ) -> APIResponse:
        """Search for claims matching criteria.

        This extends the core three methods with search capability per F-0003.

        Args:
            query: Optional text to search for in claim content
            tags: Optional tags to filter by
            min_confidence: Minimum confidence threshold
            max_confidence: Maximum confidence threshold
            limit: Maximum results to return (1-1000)

        Returns:
            APIResponse with list of matching claims
        """
        error_response = await self._ensure_initialized()
        if error_response:
            return error_response

        try:
            claims = []

            # Use vector search if available and query provided
            if query and self._vector_store and self._vector_store.count() > 0:
                # Semantic search via FAISS
                results = self._vector_store.search(query, k=limit)
                for claim_id, score in results:
                    claim = await self._data_manager.get_claim(claim_id)
                    if claim:
                        claims.append(claim)
            elif query and hasattr(self._data_manager, 'search_claims'):
                # Fall back to text search
                claims = await self._data_manager.search_claims(query=query, limit=limit)
            elif hasattr(self._data_manager, 'list_claims'):
                # Fallback: list all claims
                claims = await self._data_manager.list_claims()

            # Filter by confidence if needed
            if claims and (min_confidence > 0.0 or max_confidence < 1.0):
                claims = [
                    c for c in claims
                    if min_confidence <= getattr(c, 'confidence', 0.5) <= max_confidence
                ]

            # Filter by tags if provided
            if claims and tags:
                claims = [
                    c for c in claims
                    if any(t in getattr(c, 'tags', []) for t in tags)
                ]

            # Apply limit
            claims = claims[:limit]

            return APIResponse(
                success=True,
                message=f"Found {len(claims)} claims",
                data={
                    "claims": [c.model_dump() if hasattr(c, 'model_dump') else c for c in claims],
                    "count": len(claims),
                    "search_method": "vector" if (query and self._vector_store) else "text"
                }
            )

        except Exception as e:
            logger.error(f"Failed to search claims: {e}")
            return APIResponse(
                success=False,
                message="Failed to search claims",
                errors=[str(e)]
            )
