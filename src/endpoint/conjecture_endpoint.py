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
from pydantic import BaseModel, Field

from src.data.data_manager import DataManager
from src.data.models import Claim, ClaimType, ClaimState
from src.core.models import DirtyReason

logger = logging.getLogger(__name__)


class Session(BaseModel):
    """Session for scoped claim storage per M-0007.

    Sessions persist claims locally during a conversation or benchmark run.
    Claims can be elevated to project/team scope by the LLM.
    """
    id: str = Field(default_factory=lambda: f"s{uuid.uuid4().hex[:8]}")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    claim_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


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

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


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
            id=session_id or f"s{uuid.uuid4().hex[:8]}",
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
        include_reasoning: bool = True
    ) -> APIResponse:
        """Evaluate claims using LLM reasoning.

        Per A-0003, this is one of the three core endpoint methods.
        This method builds context from relevant claims and uses the
        Process layer for LLM-based reasoning.

        Args:
            query: Natural language query to evaluate
            max_claims: Maximum claims to include in context (default 10)
            min_confidence: Minimum confidence for claim inclusion (default 0.5)
            include_reasoning: Whether to include reasoning steps (default True)

        Returns:
            APIResponse with evaluation result and reasoning chain
        """
        error_response = await self._ensure_initialized()
        if error_response:
            return error_response

        try:
            from src.endpoint.llm_client import LLMClient, build_claim_context, build_enhanced_prompt

            # Step 1: Search for relevant claims
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

            # Step 2: Build claim context
            claim_context = build_claim_context(claims)

            # Step 3: Build enhanced prompt
            enhanced_prompt = build_enhanced_prompt(query, claim_context)

            # Step 4: Call LLM
            llm = LLMClient()
            try:
                llm_response = await llm.generate(
                    prompt=enhanced_prompt,
                    temperature=0.7,
                    max_tokens=1024
                )
            finally:
                await llm.close()

            # Step 5: Return response with context
            return APIResponse(
                success=True,
                message="Evaluation complete",
                data={
                    "query": query,
                    "response": llm_response.get("content", ""),
                    "claims_used": len(claims),
                    "claim_context": claim_context if include_reasoning else None,
                    "enhanced_prompt": enhanced_prompt if include_reasoning else None,
                    "model": llm_response.get("model", "unknown"),
                    "usage": llm_response.get("usage", {})
                }
            )

        except ValueError as e:
            # API key not configured
            return APIResponse(
                success=False,
                message="LLM not configured",
                errors=[str(e), "Set CHUTES_API_KEY environment variable"]
            )
        except Exception as e:
            logger.error(f"Failed to evaluate query: {e}")
            return APIResponse(
                success=False,
                message="Failed to evaluate query",
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
