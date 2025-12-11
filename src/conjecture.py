"""
Conjecture: Async Evidence-Based AI Reasoning System
OPTIMIZED: Enhanced with comprehensive performance monitoring
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, AsyncGenerator, Callable
from pathlib import Path
import logging
from functools import lru_cache
import hashlib

from src.core.models import Claim, ClaimState, ClaimFilter

# Define ExplorationResult for backward compatibility
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class ExplorationResult:
    """Result of claim exploration"""
    
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
        """Convert exploration result to dictionary"""
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

from src.config.unified_config import UnifiedConfig as Config
from src.processing.unified_bridge import UnifiedLLMBridge as LLMBridge, LLMRequest
from src.processing.simplified_llm_manager import get_simplified_llm_manager
from src.processing.enhanced_llm_router import get_enhanced_llm_router
from src.processing.async_eval import AsyncClaimEvaluationService
from src.processing.context_collector import ContextCollector
from src.processing.tool_manager import DynamicToolCreator
from src.data.repositories import get_data_manager, RepositoryFactory
from src.monitoring import get_performance_monitor, monitor_performance
from src.interfaces.processing_interface import (
    ProcessingInterface,
    EvaluationResult,
    ToolResult,
    Context as ProcessingContext,
    Session,
    SessionState,
    ProcessingEvent,
    EventType
)

class Conjecture(ProcessingInterface):
    """
    Enhanced Conjecture with Async Claim Evaluation and Dynamic Tool Creation
    Implements the full architecture described in specifications
    Now implements ProcessingInterface for clean architecture separation
    """

    def __init__(self, config: Optional[Config] = None):
        """OPTIMIZED: Initialize Enhanced Conjecture with performance monitoring"""
        self.config = config or Config()

        # Initialize performance monitor
        self.performance_monitor = get_performance_monitor()

        # Initialize data layer with repository pattern
        self.data_manager = get_data_manager()

        # Initialize LLM components
        self.llm_bridge = LLMBridge(self.config)
        self.llm_manager = get_simplified_llm_manager()
        self.llm_router = get_enhanced_llm_router()

        # Initialize processing components
        self.context_collector = ContextCollector(data_manager=self.data_manager)
        self.tool_creator = DynamicToolCreator()
        self.async_evaluation = AsyncClaimEvaluationService(
            llm_bridge=self.llm_bridge,
            context_collector=self.context_collector,
            data_manager=self.data_manager,
            tool_executor=self.tool_creator
        )

        # Initialize statistics and caching
        self._stats = {
            "claims_processed": 0,
            "tools_created": 0,
            "evaluation_time_total": 0.0,
            "session_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        self._performance_stats = {
            "claim_creation": [],
            "claim_evaluation": [],
            "context_collection": [],
            "tool_execution": [],
        }
        self._cache_ttl = 300  # 5 minutes
        self._context_cache = {}
        self._evaluation_cache = {}

        # Services state
        self._services_started = False

        # Initialize repositories
        self.claim_repository = RepositoryFactory.get_claim_repository()
        self.session_repository = RepositoryFactory.get_session_repository()

        # Setup logging
        self.logger = logging.getLogger(__name__)

    # ProcessingInterface implementation
    async def create_claim(
        self, 
        content: str, 
        confidence: Optional[float] = None,
        tags: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> Claim:
        """Create a new claim"""
        start_time = time.time()
        
        try:
            # Generate claim ID
            claim_id = self._generate_claim_id()
            
            # Set default confidence if not provided
            if confidence is None:
                confidence = 0.5
            
            # Create claim object
            claim = Claim(
                id=claim_id,
                content=content,
                confidence=confidence,
                tags=tags or [],
                **kwargs
            )
            
            # Save to repository
            await self.claim_repository.create(claim)
            
            # Update statistics
            self._stats["claims_processed"] += 1
            self._performance_stats["claim_creation"].append(time.time() - start_time)
            
            self.logger.info(f"Created claim {claim_id} with confidence {confidence}")
            
            return claim
            
        except Exception as e:
            self.logger.error(f"Failed to create claim: {e}")
            raise

    async def get_claim(self, claim_id: str) -> Optional[Claim]:
        """Get a claim by ID"""
        try:
            return await self.claim_repository.get_by_id(claim_id)
        except Exception as e:
            self.logger.error(f"Failed to get claim {claim_id}: {e}")
            return None

    async def update_claim(
        self, 
        claim_id: str, 
        **updates
    ) -> Optional[Claim]:
        """Update a claim"""
        try:
            claim = await self.claim_repository.get_by_id(claim_id)
            if claim:
                for key, value in updates.items():
                    setattr(claim, key, value)
                await self.claim_repository.update(claim)
                self.logger.info(f"Updated claim {claim_id}")
                return claim
            return None
        except Exception as e:
            self.logger.error(f"Failed to update claim {claim_id}: {e}")
            return None

    async def delete_claim(self, claim_id: str) -> bool:
        """Delete a claim"""
        try:
            await self.claim_repository.delete(claim_id)
            self.logger.info(f"Deleted claim {claim_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete claim {claim_id}: {e}")
            return False

    async def search_claims(
        self, 
        query: str, 
        filters: Optional[ClaimFilter] = None,
        limit: int = 100
    ) -> List[Claim]:
        """Search for claims"""
        try:
            return await self.claim_repository.search(query, filters, limit)
        except Exception as e:
            self.logger.error(f"Failed to search claims: {e}")
            return []

    async def evaluate_claim(
        self, 
        claim_id: str,
        context: Optional[ProcessingContext] = None
    ) -> EvaluationResult:
        """Evaluate a claim"""
        start_time = time.time()
        
        try:
            claim = await self.claim_repository.get_by_id(claim_id)
            if not claim:
                return EvaluationResult(
                    success=False,
                    claim_id=claim_id,
                    error="Claim not found"
                )
            
            # Use async evaluation service
            result = await self.async_evaluation.evaluate_claim(claim, context)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._stats["evaluation_time_total"] += processing_time
            self._performance_stats["claim_evaluation"].append(processing_time)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate claim {claim_id}: {e}")
            return EvaluationResult(
                success=False,
                claim_id=claim_id,
                error=str(e)
            )

    async def batch_create_claims(
        self, 
        claims_data: List[Dict[str, Any]]
    ) -> List[Claim]:
        """Create multiple claims"""
        try:
            claims = []
            for claim_data in claims_data:
                claim = await self.create_claim(**claim_data)
                claims.append(claim)
            return claims
        except Exception as e:
            self.logger.error(f"Failed to batch create claims: {e}")
            return []

    async def batch_evaluate_claims(
        self, 
        claim_ids: List[str]
    ) -> List[EvaluationResult]:
        """Evaluate multiple claims"""
        try:
            results = []
            for claim_id in claim_ids:
                result = await self.evaluate_claim(claim_id)
                results.append(result)
            return results
        except Exception as e:
            self.logger.error(f"Failed to batch evaluate claims: {e}")
            return []

    async def get_context(
        self, 
        claim_id: str, 
        max_depth: int = 3
    ) -> ProcessingContext:
        """Get context for a claim"""
        try:
            return await self.context_collector.collect_context_for_claim(claim_id, max_depth)
        except Exception as e:
            self.logger.error(f"Failed to get context for claim {claim_id}: {e}")
            return ProcessingContext(
                claim_id=claim_id,
                claims=[],
                relationships=[]
            )

    async def get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        try:
            return self.tool_creator.get_available_tools()
        except Exception as e:
            self.logger.error(f"Failed to get available tools: {e}")
            return []

    async def execute_tool(
        self, 
        tool_name: str, 
        parameters: Dict[str, Any]
    ) -> ToolResult:
        """Execute a tool"""
        try:
            return await self.tool_creator.execute_tool(tool_name, parameters)
        except Exception as e:
            self.logger.error(f"Failed to execute tool {tool_name}: {e}")
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error=str(e)
            )

    async def create_session(
        self, 
        session_id: Optional[str] = None
    ) -> Session:
        """Create a new session"""
        try:
            session = Session(
                id=session_id or self._generate_session_id(),
                state=SessionState.ACTIVE,
                created_at=datetime.utcnow()
            )
            await self.session_repository.create(session)
            self._stats["session_count"] += 1
            return session
        except Exception as e:
            self.logger.error(f"Failed to create session: {e}")
            raise

    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID"""
        try:
            return await self.session_repository.get_by_id(session_id)
        except Exception as e:
            self.logger.error(f"Failed to get session {session_id}: {e}")
            return None

    async def resume_session(self, session_id: str) -> Session:
        """Resume a session"""
        try:
            session = await self.session_repository.get_by_id(session_id)
            if session:
                session.state = SessionState.ACTIVE
                await self.session_repository.update(session)
                return session
            return None
        except Exception as e:
            self.logger.error(f"Failed to resume session {session_id}: {e}")
            raise

    async def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "claims_processed": self._stats["claims_processed"],
            "tools_created": self._stats["tools_created"],
            "evaluation_time_total": self._stats["evaluation_time_total"],
            "session_count": self._stats["session_count"],
            "performance_stats": self._performance_stats
        }

    async def get_health_status(self) -> Dict[str, Any]:
        """Get system health status"""
        return {
            "status": "healthy",
            "services_started": self._services_started,
            "data_manager": "connected" if self.data_manager else "disconnected",
            "llm_bridge": "connected" if self.llm_bridge else "disconnected"
        }

    async def cleanup_resources(self):
        """Cleanup resources"""
        try:
            if self.data_manager:
                await self.data_manager.close()
            if self.llm_bridge:
                await self.llm_bridge.cleanup()
            self._services_started = False
            self.logger.info("Resources cleaned up")
        except Exception as e:
            self.logger.error(f"Failed to cleanup resources: {e}")

    async def stream_events(
        self, 
        event_types: Optional[List[EventType]] = None
    ) -> AsyncGenerator[ProcessingEvent, None]:
        """Stream processing events"""
        # Simple implementation - in practice, this would use a proper event system
        for i in range(5):  # Mock 5 events
            event = ProcessingEvent(
                type=EventType.CLAIM_CREATED,
                timestamp=datetime.utcnow(),
                data={"mock_event": i}
            )
            yield event
            await asyncio.sleep(0.1)  # Small delay

    async def subscribe_to_events(
        self, 
        callback: Callable[[ProcessingEvent], None],
        event_types: Optional[List[EventType]] = None
    ) -> str:
        """Subscribe to events"""
        subscription_id = self._generate_subscription_id()
        # In practice, this would register the callback
        self.logger.info(f"Subscribed to events with ID {subscription_id}")
        return subscription_id

    async def unsubscribe_from_events(self, subscription_id: str):
        """Unsubscribe from events"""
        # In practice, this would remove the subscription
        self.logger.info(f"Unsubscribed from events with ID {subscription_id}")

    # Helper methods
    def _generate_claim_id(self) -> str:
        """Generate a unique claim ID"""
        import uuid
        return f"c{int(time.time()) % 100000000:08d}"

    def _generate_session_id(self) -> str:
        """Generate a unique session ID"""
        import uuid
        return str(uuid.uuid4())

    def _generate_subscription_id(self) -> str:
        """Generate a unique subscription ID"""
        import uuid
        return str(uuid.uuid4())

    # Lifecycle methods
    async def start_services(self):
        """Start all services"""
        if self._services_started:
            return
        
        try:
            # Start data manager
            if self.data_manager:
                await self.data_manager.initialize()
            
            # Start LLM services
            if self.llm_bridge:
                await self.llm_bridge.initialize()
            
            self._services_started = True
            self.logger.info("All services started")
            
        except Exception as e:
            self.logger.error(f"Failed to start services: {e}")
            raise

    async def stop_services(self):
        """Stop all services"""
        await self.cleanup_resources()

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start_services()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop_services()


# Convenience functions
async def explore(query: str, **kwargs) -> ExplorationResult:
    """Quick exploration function"""
    async with Conjecture() as cf:
        claims = await cf.search_claims(query)
        return ExplorationResult(
            success=True,
            claim_id="",
            original_confidence=0.0,
            new_confidence=0.0,
            state=ClaimState.EXPLORE,
            evaluation_summary=f"Found {len(claims)} claims for '{query}'",
            claims=claims,
            total_found=len(claims),
            search_time=0.0,
            confidence_threshold=0.5,
            max_claims=100
        )


async def add_claim(
    content: str, confidence: float, **kwargs
) -> Claim:
    """Quick claim creation function"""
    async with Conjecture() as cf:
        return await cf.create_claim(content, confidence, **kwargs)


if __name__ == "__main__":
    async def test_conjecture():
        print("Testing Conjecture")
        async with Conjecture() as cf:
            claim = await cf.create_claim(
                content="Test claim for Conjecture",
                confidence=0.8
            )
            print(f"Created claim: {claim.id}")
            
            stats = await cf.get_statistics()
            print(f"Statistics: {stats}")

    asyncio.run(test_conjecture())