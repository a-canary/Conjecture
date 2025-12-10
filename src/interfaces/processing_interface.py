"""
Processing Interface for Conjecture

Clean architecture abstraction layer separating presentation from processing.
Defines the contract for all processing layer operations with async support
and event streaming capabilities for real-time feedback.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import (
    Any, Dict, List, Optional, AsyncGenerator, Callable, Union
)
from dataclasses import dataclass, field
from pydantic import BaseModel

from ..core.models import Claim, ClaimFilter, ClaimState

class EventType(str, Enum):
    """Event types for processing layer events"""
    
    CLAIM_CREATED = "claim_created"
    CLAIM_EVALUATED = "claim_evaluated"
    CLAIM_UPDATED = "claim_updated"
    TOOL_CALLED = "tool_called"
    TOOL_COMPLETED = "tool_completed"
    RESPONSE_GENERATED = "response_generated"
    SESSION_CREATED = "session_created"
    SESSION_RESUMED = "session_resumed"
    CONTEXT_BUILT = "context_built"
    ERROR_OCCURRED = "error_occurred"
    PROGRESS_UPDATE = "progress_update"

class SessionState(str, Enum):
    """Session state enumeration"""
    
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class ProcessingEvent:
    """Event emitted during processing operations"""
    
    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    claim_id: Optional[str] = None
    tool_name: Optional[str] = None
    message: Optional[str] = None
    progress: Optional[float] = None  # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "session_id": self.session_id,
            "claim_id": self.claim_id,
            "tool_name": self.tool_name,
            "message": self.message,
            "progress": self.progress,
        }

@dataclass
class EvaluationResult:
    """Result of claim evaluation"""
    
    success: bool
    claim_id: str
    original_confidence: float
    new_confidence: float
    state: ClaimState
    evaluation_summary: str
    supporting_evidence: List[str] = field(default_factory=list)
    counter_evidence: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation result to dictionary"""
        return {
            "success": self.success,
            "claim_id": self.claim_id,
            "original_confidence": self.original_confidence,
            "new_confidence": self.new_confidence,
            "state": self.state.value,
            "evaluation_summary": self.evaluation_summary,
            "supporting_evidence": self.supporting_evidence,
            "counter_evidence": self.counter_evidence,
            "recommendations": self.recommendations,
            "processing_time": self.processing_time,
            "metadata": self.metadata,
            "errors": self.errors,
        }

@dataclass
class ToolResult:
    """Result of tool execution"""
    
    success: bool
    tool_name: str
    parameters: Dict[str, Any]
    outcome: str
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)  # File paths or other artifacts
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool result to dictionary"""
        return {
            "success": self.success,
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "outcome": self.outcome,
            "duration": self.duration,
            "metadata": self.metadata,
            "artifacts": self.artifacts,
            "errors": self.errors,
        }

@dataclass
class Context:
    """Context data for claim processing"""
    
    claim_ids: List[str]
    skills: List[Claim] = field(default_factory=list)
    samples: List[Claim] = field(default_factory=list)
    related_claims: List[Claim] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    context_string: str = ""
    relevance_scores: Dict[str, float] = field(default_factory=dict)
    total_tokens: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary"""
        return {
            "claim_ids": self.claim_ids,
            "skills": [claim.to_dict() for claim in self.skills],
            "samples": [claim.to_dict() for claim in self.samples],
            "related_claims": [claim.to_dict() for claim in self.related_claims],
            "metadata": self.metadata,
            "context_string": self.context_string,
            "relevance_scores": self.relevance_scores,
            "total_tokens": self.total_tokens,
        }

class Session(BaseModel):
    """Session for tracking processing operations"""
    
    session_id: str
    state: SessionState = SessionState.ACTIVE
    created: datetime = field(default_factory=datetime.utcnow)
    updated: datetime = field(default_factory=datetime.utcnow)
    claims: List[str] = field(default_factory=list)
    context: Optional[Context] = None
    metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    user_data: Dict[str, Any] = field(default_factory=dict)
    
    model_config = {
        "arbitrary_types_allowed": True,
        "protected_namespaces": ()
    }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary"""
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "created": self.created.isoformat(),
            "updated": self.updated.isoformat(),
            "claims": self.claims,
            "context": self.context.to_dict() if self.context else None,
            "metadata": self.metadata,
            "user_data": self.user_data,
        }

class ProcessingInterface(ABC):
    """
    Abstract interface defining all processing layer operations.
    
    This interface serves as the clean boundary between presentation and processing
    layers, enabling separation of concerns and testability. All methods are async
    to support non-blocking operations and event streaming for real-time feedback.
    """
    
    @abstractmethod
    async def create_claim(
        self, 
        content: str, 
        confidence: Optional[float] = None,
        tags: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> Claim:
        """
        Create a new claim with automatic evaluation.
        
        Args:
            content: Claim content text
            confidence: Initial confidence score (0.0-1.0)
            tags: Optional tags for categorization
            session_id: Optional session ID for context
            **kwargs: Additional claim attributes
            
        Returns:
            Created claim with generated ID
            
        Raises:
            InvalidClaimError: If claim content is invalid
            DataLayerError: If claim creation fails
        """
        pass
    
    @abstractmethod
    async def evaluate_claim(
        self, 
        claim_id: str,
        session_id: Optional[str] = None
    ) -> EvaluationResult:
        """
        Evaluate a claim using LLM and context.
        
        Args:
            claim_id: ID of claim to evaluate
            session_id: Optional session ID for context
            
        Returns:
            Evaluation result with updated confidence and state
            
        Raises:
            ClaimNotFoundError: If claim doesn't exist
            DataLayerError: If evaluation fails
        """
        pass
    
    @abstractmethod
    async def search_claims(
        self, 
        query: str, 
        filters: Optional[ClaimFilter] = None,
        session_id: Optional[str] = None
    ) -> List[Claim]:
        """
        Search for claims using semantic search and filters.
        
        Args:
            query: Search query text
            filters: Optional claim filters
            session_id: Optional session ID for context
            
        Returns:
            List of matching claims ranked by relevance
            
        Raises:
            DataLayerError: If search fails
        """
        pass
    
    @abstractmethod
    async def execute_tool(
        self, 
        tool_name: str, 
        parameters: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> ToolResult:
        """
        Execute a tool with specified parameters.
        
        Args:
            tool_name: Name of tool to execute
            parameters: Tool parameters
            session_id: Optional session ID for context
            
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If tool not found or parameters invalid
            DataLayerError: If tool execution fails
        """
        pass
    
    @abstractmethod
    async def get_context(
        self, 
        claim_ids: List[str],
        max_skills: int = 5,
        max_samples: int = 10,
        session_id: Optional[str] = None
    ) -> Context:
        """
        Build context for specified claims.
        
        Args:
            claim_ids: List of claim IDs to build context for
            max_skills: Maximum number of skills to include
            max_samples: Maximum number of samples to include
            session_id: Optional session ID for context
            
        Returns:
            Context object with relevant claims and metadata
            
        Raises:
            ClaimNotFoundError: If any claim doesn't exist
            DataLayerError: If context building fails
        """
        pass
    
    @abstractmethod
    async def create_session(
        self, 
        user_data: Optional[Dict[str, Any]] = None
    ) -> Session:
        """
        Create a new processing session.
        
        Args:
            user_data: Optional user-specific data
            
        Returns:
            Created session with unique ID
            
        Raises:
            DataLayerError: If session creation fails
        """
        pass
    
    @abstractmethod
    async def resume_session(
        self, 
        session_id: str
    ) -> Session:
        """
        Resume an existing session.
        
        Args:
            session_id: ID of session to resume
            
        Returns:
            Resumed session
            
        Raises:
            ValueError: If session doesn't exist
            DataLayerError: If session resume fails
        """
        pass
    
    @abstractmethod
    async def get_claim(
        self, 
        claim_id: str,
        session_id: Optional[str] = None
    ) -> Claim:
        """
        Retrieve a specific claim by ID.
        
        Args:
            claim_id: ID of claim to retrieve
            session_id: Optional session ID for context
            
        Returns:
            Claim object
            
        Raises:
            ClaimNotFoundError: If claim doesn't exist
            DataLayerError: If retrieval fails
        """
        pass
    
    @abstractmethod
    async def update_claim(
        self, 
        claim_id: str,
        updates: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Claim:
        """
        Update an existing claim.
        
        Args:
            claim_id: ID of claim to update
            updates: Dictionary of fields to update
            session_id: Optional session ID for context
            
        Returns:
            Updated claim
            
        Raises:
            ClaimNotFoundError: If claim doesn't exist
            InvalidClaimError: If updates are invalid
            DataLayerError: If update fails
        """
        pass
    
    @abstractmethod
    async def delete_claim(
        self, 
        claim_id: str,
        session_id: Optional[str] = None
    ) -> bool:
        """
        Delete a claim by ID.
        
        Args:
            claim_id: ID of claim to delete
            session_id: Optional session ID for context
            
        Returns:
            True if claim was deleted
            
        Raises:
            DataLayerError: If deletion fails
        """
        pass
    
    @abstractmethod
    async def get_available_tools(
        self,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get list of available tools.
        
        Args:
            session_id: Optional session ID for context
            
        Returns:
            List of tool information dictionaries
            
        Raises:
            DataLayerError: If tool listing fails
        """
        pass
    
    # Event Streaming Methods
    
    @abstractmethod
    async def stream_events(
        self,
        session_id: Optional[str] = None,
        event_types: Optional[List[EventType]] = None,
        since: Optional[datetime] = None
    ) -> AsyncGenerator[ProcessingEvent, None]:
        """
        Stream processing events in real-time.
        
        Args:
            session_id: Optional session ID to filter events
            event_types: Optional event types to filter
            since: Optional timestamp to filter events
            
        Yields:
            ProcessingEvent objects as they occur
            
        Raises:
            DataLayerError: If event streaming fails
        """
        pass
    
    @abstractmethod
    async def subscribe_to_events(
        self,
        callback: Callable[[ProcessingEvent], None],
        session_id: Optional[str] = None,
        event_types: Optional[List[EventType]] = None
    ) -> str:
        """
        Subscribe to processing events with callback.
        
        Args:
            callback: Function to call for each event
            session_id: Optional session ID to filter events
            event_types: Optional event types to filter
            
        Returns:
            Subscription ID for unsubscribing later
            
        Raises:
            ValueError: If callback is invalid
            DataLayerError: If subscription fails
        """
        pass
    
    @abstractmethod
    async def unsubscribe_from_events(
        self,
        subscription_id: str
    ) -> bool:
        """
        Unsubscribe from event notifications.
        
        Args:
            subscription_id: ID of subscription to cancel
            
        Returns:
            True if subscription was cancelled
            
        Raises:
            ValueError: If subscription ID is invalid
            DataLayerError: If unsubscription fails
        """
        pass
    
    # Batch Processing Methods
    
    @abstractmethod
    async def batch_create_claims(
        self,
        claims_data: List[Dict[str, Any]],
        session_id: Optional[str] = None
    ) -> List[Claim]:
        """
        Create multiple claims in batch for better performance.
        
        Args:
            claims_data: List of claim creation data
            session_id: Optional session ID for context
            
        Returns:
            List of created claims
            
        Raises:
            InvalidClaimError: If any claim data is invalid
            DataLayerError: If batch creation fails
        """
        pass
    
    @abstractmethod
    async def batch_evaluate_claims(
        self,
        claim_ids: List[str],
        session_id: Optional[str] = None
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple claims in batch.
        
        Args:
            claim_ids: List of claim IDs to evaluate
            session_id: Optional session ID for context
            
        Returns:
            List of evaluation results
            
        Raises:
            ClaimNotFoundError: If any claim doesn't exist
            DataLayerError: If batch evaluation fails
        """
        pass
    
    # Health and Status Methods
    
    @abstractmethod
    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of processing layer.
        
        Returns:
            Dictionary containing health information
            
        Raises:
            DataLayerError: If health check fails
        """
        pass
    
    @abstractmethod
    async def get_statistics(
        self,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Args:
            session_id: Optional session ID for session-specific stats
            
        Returns:
            Dictionary containing statistics
            
        Raises:
            DataLayerError: If statistics retrieval fails
        """
        pass
    
    @abstractmethod
    async def cleanup_resources(
        self,
        session_id: Optional[str] = None
    ) -> bool:
        """
        Clean up resources and temporary data.
        
        Args:
            session_id: Optional session ID for session-specific cleanup
            
        Returns:
            True if cleanup was successful
            
        Raises:
            DataLayerError: If cleanup fails
        """
        pass