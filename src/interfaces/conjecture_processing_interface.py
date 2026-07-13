# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Conjecture Processing Interface Implementation

Concrete implementation of ProcessingInterface providing claim processing,
evaluation, and session management through the Data Layer.

See CHOICES.md A-0001 (4-Layer Architecture) and A-0002 (Process Layer).
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, AsyncGenerator, Callable

from src.core.models import Claim, ClaimFilter, ClaimState, ClaimType, ClaimScope
from src.data.data_manager import DataManager
from src.utils.id_utils import generate_id
from src.interfaces.processing_interface import (
    ProcessingInterface,
    ProcessingEvent,
    EventType,
    EvaluationResult,
    ToolResult,
    Context,
    Session,
    SessionState,
)

logger = logging.getLogger(__name__)


class ConjectureProcessingInterface(ProcessingInterface):
    """
    Concrete implementation of ProcessingInterface.

    Bridges the presentation layer to the data layer through clean async methods.
    Provides claim CRUD, evaluation, session management, and event streaming.
    """

    def __init__(self, db_path: str = "data/conjecture.db"):
        """
        Initialize the processing interface.

        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = db_path
        self._data_manager = DataManager(db_path)
        self._sessions: Dict[str, Session] = {}
        self._event_queue: asyncio.Queue[ProcessingEvent] = asyncio.Queue()
        self._subscriptions: Dict[str, Dict[str, Any]] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the processing interface."""
        if not self._initialized:
            await self._data_manager.initialize()
            self._initialized = True
            logger.info("ConjectureProcessingInterface initialized")

    async def close(self) -> None:
        """Close the processing interface."""
        if self._initialized:
            await self._data_manager.close()
            self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure interface is initialized."""
        if not self._initialized:
            await self.initialize()

    async def _emit_event(self, event: ProcessingEvent) -> None:
        """Emit a processing event."""
        await self._event_queue.put(event)
        # Notify subscribers
        for sub_id, sub in self._subscriptions.items():
            if sub.get("event_types") is None or event.event_type in sub["event_types"]:
                if sub.get("session_id") is None or event.session_id == sub["session_id"]:
                    callback = sub.get("callback")
                    if callback:
                        try:
                            callback(event)
                        except Exception as e:
                            logger.error(f"Event callback error: {e}")

    # Claim CRUD Methods

    async def create_claim(
        self,
        content: str,
        confidence: Optional[float] = None,
        tags: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> Claim:
        """Create a new claim with automatic evaluation."""
        await self._ensure_initialized()

        claim_id = await self._data_manager.create_claim(
            content=content,
            confidence=confidence or 0.5,
            tags=tags or [],
            **kwargs
        )

        claim = await self._data_manager.get_claim(claim_id)

        await self._emit_event(ProcessingEvent(
            event_type=EventType.CLAIM_CREATED,
            claim_id=claim_id,
            session_id=session_id,
            message=f"Created claim: {claim_id}",
            data={"content": content[:100]}
        ))

        return claim

    async def get_claim(
        self,
        claim_id: str,
        session_id: Optional[str] = None
    ) -> Claim:
        """Retrieve a specific claim by ID."""
        await self._ensure_initialized()

        claim = await self._data_manager.get_claim(claim_id)
        if not claim:
            raise ValueError(f"Claim not found: {claim_id}")
        return claim

    async def update_claim(
        self,
        claim_id: str,
        updates: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Claim:
        """Update an existing claim."""
        await self._ensure_initialized()

        success = await self._data_manager.update_claim(claim_id, updates)
        if not success:
            raise ValueError(f"Claim not found: {claim_id}")

        claim = await self._data_manager.get_claim(claim_id)

        await self._emit_event(ProcessingEvent(
            event_type=EventType.CLAIM_UPDATED,
            claim_id=claim_id,
            session_id=session_id,
            message=f"Updated claim: {claim_id}",
            data={"updates": list(updates.keys())}
        ))

        return claim

    async def delete_claim(
        self,
        claim_id: str,
        session_id: Optional[str] = None
    ) -> bool:
        """Delete a claim by ID."""
        await self._ensure_initialized()
        return await self._data_manager.delete_claim(claim_id)

    async def search_claims(
        self,
        query: str,
        filters: Optional[ClaimFilter] = None,
        session_id: Optional[str] = None
    ) -> List[Claim]:
        """Search for claims using semantic search and filters."""
        await self._ensure_initialized()
        return await self._data_manager.search_claims(query)

    # Evaluation Methods

    async def evaluate_claim(
        self,
        claim_id: str,
        session_id: Optional[str] = None
    ) -> EvaluationResult:
        """
        Evaluate a claim using LLM and context.

        Note: Full LLM integration pending (GAP-3).
        Currently returns basic evaluation based on claim state.
        """
        await self._ensure_initialized()

        claim = await self.get_claim(claim_id, session_id)

        # Basic evaluation (full LLM integration pending GAP-3)
        result = EvaluationResult(
            success=True,
            claim_id=claim_id,
            original_confidence=claim.confidence,
            new_confidence=claim.confidence,
            state=claim.state,
            evaluation_summary=f"Claim '{claim.content[:50]}...' evaluated",
            supporting_evidence=[],
            counter_evidence=[],
            recommendations=["Consider adding supporting claims"],
            processing_time=0.01,
        )

        await self._emit_event(ProcessingEvent(
            event_type=EventType.CLAIM_EVALUATED,
            claim_id=claim_id,
            session_id=session_id,
            message=f"Evaluated claim: {claim_id}",
            data={"confidence": claim.confidence}
        ))

        return result

    # Tool Execution

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> ToolResult:
        """Execute a tool with specified parameters."""
        await self._ensure_initialized()

        start_time = datetime.now(timezone.utc)

        await self._emit_event(ProcessingEvent(
            event_type=EventType.TOOL_CALLED,
            tool_name=tool_name,
            session_id=session_id,
            message=f"Tool called: {tool_name}",
            data=parameters
        ))

        # Basic tool execution (full tool framework pending)
        result = ToolResult(
            success=True,
            tool_name=tool_name,
            parameters=parameters,
            outcome=f"Tool '{tool_name}' executed (stub)",
            duration=(datetime.now(timezone.utc) - start_time).total_seconds(),
        )

        await self._emit_event(ProcessingEvent(
            event_type=EventType.TOOL_COMPLETED,
            tool_name=tool_name,
            session_id=session_id,
            message=f"Tool completed: {tool_name}",
            data={"success": result.success}
        ))

        return result

    async def get_available_tools(
        self,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get list of available tools."""
        return [
            {"name": "search", "description": "Search claims by content"},
            {"name": "evaluate", "description": "Evaluate claim confidence"},
            {"name": "create", "description": "Create a new claim"},
        ]

    # Context Building

    async def get_context(
        self,
        claim_ids: List[str],
        max_skills: int = 5,
        max_samples: int = 10,
        session_id: Optional[str] = None
    ) -> Context:
        """Build context for specified claims."""
        await self._ensure_initialized()

        related_claims = []
        for claim_id in claim_ids:
            claim = await self._data_manager.get_claim(claim_id)
            if claim:
                related_claims.append(claim)

        context = Context(
            claim_ids=claim_ids,
            related_claims=related_claims,
            context_string="\n".join([c.content for c in related_claims]),
            total_tokens=sum(len(c.content) // 4 for c in related_claims),
        )

        await self._emit_event(ProcessingEvent(
            event_type=EventType.CONTEXT_BUILT,
            session_id=session_id,
            message=f"Context built for {len(claim_ids)} claims",
            data={"total_tokens": context.total_tokens}
        ))

        return context

    # Session Management

    async def create_session(
        self,
        user_data: Optional[Dict[str, Any]] = None
    ) -> Session:
        """Create a new processing session."""
        session_id = generate_id("s")
        now = datetime.now(timezone.utc)

        session = Session(
            session_id=session_id,
            state=SessionState.ACTIVE,
            created=now,
            updated=now,
            user_data=user_data or {},
        )

        self._sessions[session_id] = session

        await self._emit_event(ProcessingEvent(
            event_type=EventType.SESSION_CREATED,
            session_id=session_id,
            message=f"Session created: {session_id}",
        ))

        return session

    async def resume_session(
        self,
        session_id: str
    ) -> Session:
        """Resume an existing session."""
        if session_id not in self._sessions:
            raise ValueError(f"Session not found: {session_id}")

        session = self._sessions[session_id]
        session.state = SessionState.ACTIVE
        session.updated = datetime.now(timezone.utc)

        await self._emit_event(ProcessingEvent(
            event_type=EventType.SESSION_RESUMED,
            session_id=session_id,
            message=f"Session resumed: {session_id}",
        ))

        return session

    # Event Streaming

    async def stream_events(
        self,
        session_id: Optional[str] = None,
        event_types: Optional[List[EventType]] = None,
        since: Optional[datetime] = None
    ) -> AsyncGenerator[ProcessingEvent, None]:
        """Stream processing events in real-time."""
        while True:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=30.0
                )
                if event_types and event.event_type not in event_types:
                    continue
                if session_id and event.session_id != session_id:
                    continue
                if since and event.timestamp < since:
                    continue
                yield event
            except asyncio.TimeoutError:
                # Yield heartbeat event periodically
                yield ProcessingEvent(
                    event_type=EventType.PROGRESS_UPDATE,
                    message="heartbeat",
                )

    async def subscribe_to_events(
        self,
        callback: Callable[[ProcessingEvent], None],
        session_id: Optional[str] = None,
        event_types: Optional[List[EventType]] = None
    ) -> str:
        """Subscribe to processing events with callback."""
        sub_id = generate_id("sub_")
        self._subscriptions[sub_id] = {
            "callback": callback,
            "session_id": session_id,
            "event_types": event_types,
        }
        return sub_id

    async def unsubscribe_from_events(
        self,
        subscription_id: str
    ) -> bool:
        """Unsubscribe from event notifications."""
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            return True
        return False

    # Batch Operations

    async def batch_create_claims(
        self,
        claims_data: List[Dict[str, Any]],
        session_id: Optional[str] = None
    ) -> List[Claim]:
        """Create multiple claims in batch."""
        await self._ensure_initialized()

        claims = []
        for data in claims_data:
            claim = await self.create_claim(
                content=data.get("content", ""),
                confidence=data.get("confidence"),
                tags=data.get("tags"),
                session_id=session_id,
            )
            claims.append(claim)
        return claims

    async def batch_evaluate_claims(
        self,
        claim_ids: List[str],
        session_id: Optional[str] = None
    ) -> List[EvaluationResult]:
        """Evaluate multiple claims in batch."""
        results = []
        for claim_id in claim_ids:
            try:
                result = await self.evaluate_claim(claim_id, session_id)
                results.append(result)
            except Exception as e:
                results.append(EvaluationResult(
                    success=False,
                    claim_id=claim_id,
                    original_confidence=0.0,
                    new_confidence=0.0,
                    state=ClaimState.EXPLORE,
                    evaluation_summary=f"Evaluation failed: {e}",
                    errors=[str(e)],
                ))
        return results

    # Health and Status

    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of processing layer."""
        await self._ensure_initialized()

        return {
            "status": "healthy",
            "initialized": self._initialized,
            "db_path": self.db_path,
            "active_sessions": len(self._sessions),
            "subscriptions": len(self._subscriptions),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def get_statistics(
        self,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get processing statistics."""
        await self._ensure_initialized()

        claim_count = await self._data_manager.count()

        return {
            "total_claims": claim_count,
            "active_sessions": len(self._sessions),
            "subscriptions": len(self._subscriptions),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def cleanup_resources(
        self,
        session_id: Optional[str] = None
    ) -> bool:
        """Clean up resources and temporary data."""
        if session_id:
            if session_id in self._sessions:
                del self._sessions[session_id]
        else:
            self._sessions.clear()
            self._subscriptions.clear()
        return True
