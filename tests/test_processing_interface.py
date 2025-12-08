"""
Test ProcessingInterface abstraction layer
"""

import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from src.interfaces.processing_interface import (
    ProcessingInterface,
    EvaluationResult,
    ToolResult,
    Context,
    Session,
    ProcessingEvent,
    EventType,
    SessionState
)
from src.core.models import Claim, ClaimState


class MockProcessingInterface(ProcessingInterface):
    """Mock implementation of ProcessingInterface for testing"""
    
    def __init__(self):
        self.claims = {}
        self.sessions = {}
        self.events = []
    
    async def create_claim(self, content: str, confidence: float = None, 
                         tags: list = None, session_id: str = None, **kwargs) -> Claim:
        claim = Claim(
            id="c0000001",
            content=content,
            confidence=confidence or 0.8,
            tags=tags or [],
            **kwargs
        )
        self.claims[claim.id] = claim
        return claim
    
    async def evaluate_claim(self, claim_id: str, session_id: str = None) -> EvaluationResult:
        if claim_id not in self.claims:
            from src.core.models import ClaimNotFoundError
            raise ClaimNotFoundError(f"Claim {claim_id} not found")
        
        claim = self.claims[claim_id]
        return EvaluationResult(
            success=True,
            claim_id=claim_id,
            original_confidence=claim.confidence,
            new_confidence=0.9,
            state=ClaimState.VALIDATED,
            evaluation_summary="Mock evaluation completed"
        )
    
    async def search_claims(self, query: str, filters=None, session_id: str = None) -> list:
        return list(self.claims.values())
    
    async def execute_tool(self, tool_name: str, parameters: dict, session_id: str = None) -> ToolResult:
        return ToolResult(
            success=True,
            tool_name=tool_name,
            parameters=parameters,
            outcome="Mock tool execution completed",
            duration=0.1
        )
    
    async def get_context(self, claim_ids: list, max_skills: int = 5, 
                        max_samples: int = 10, session_id: str = None) -> Context:
        return Context(
            claim_ids=claim_ids,
            context_string="Mock context"
        )
    
    async def create_session(self, user_data: dict = None) -> Session:
        session = Session(
            session_id="session_001",
            user_data=user_data or {}
        )
        self.sessions[session.session_id] = session
        return session
    
    async def resume_session(self, session_id: str) -> Session:
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        return self.sessions[session_id]
    
    async def get_claim(self, claim_id: str, session_id: str = None) -> Claim:
        if claim_id not in self.claims:
            from src.core.models import ClaimNotFoundError
            raise ClaimNotFoundError(f"Claim {claim_id} not found")
        return self.claims[claim_id]
    
    async def update_claim(self, claim_id: str, updates: dict, session_id: str = None) -> Claim:
        if claim_id not in self.claims:
            from src.core.models import ClaimNotFoundError
            raise ClaimNotFoundError(f"Claim {claim_id} not found")
        
        claim = self.claims[claim_id]
        for key, value in updates.items():
            if hasattr(claim, key):
                setattr(claim, key, value)
        return claim
    
    async def delete_claim(self, claim_id: str, session_id: str = None) -> bool:
        if claim_id in self.claims:
            del self.claims[claim_id]
            return True
        return False
    
    async def get_available_tools(self, session_id: str = None) -> list:
        return [{"name": "mock_tool", "description": "Mock tool for testing"}]
    
    async def stream_events(self, session_id: str = None, event_types: list = None, 
                          since: datetime = None):
        for event in self.events:
            yield event
    
    async def subscribe_to_events(self, callback, session_id: str = None, event_types: list = None) -> str:
        return "subscription_001"
    
    async def unsubscribe_from_events(self, subscription_id: str) -> bool:
        return True
    
    async def batch_create_claims(self, claims_data: list, session_id: str = None) -> list:
        claims = []
        for data in claims_data:
            claim = await self.create_claim(**data)
            claims.append(claim)
        return claims
    
    async def batch_evaluate_claims(self, claim_ids: list, session_id: str = None) -> list:
        results = []
        for claim_id in claim_ids:
            result = await self.evaluate_claim(claim_id)
            results.append(result)
        return results
    
    async def get_health_status(self) -> dict:
        return {"status": "healthy", "interface": "mock"}
    
    async def get_statistics(self, session_id: str = None) -> dict:
        return {"claims_count": len(self.claims), "sessions_count": len(self.sessions)}
    
    async def cleanup_resources(self, session_id: str = None) -> bool:
        return True


class TestProcessingInterface:
    """Test ProcessingInterface abstraction layer"""
    
    @pytest.fixture
    def processing_interface(self):
        """Create mock processing interface for testing"""
        return MockProcessingInterface()
    
    @pytest.mark.asyncio
    async def test_create_claim(self, processing_interface):
        """Test claim creation through interface"""
        claim = await processing_interface.create_claim(
            content="Test claim content",
            confidence=0.8,
            tags=["test"]
        )
        
        assert claim.content == "Test claim content"
        assert claim.confidence == 0.8
        assert claim.tags == ["test"]
        assert claim.id == "c0000001"
    
    @pytest.mark.asyncio
    async def test_evaluate_claim(self, processing_interface):
        """Test claim evaluation through interface"""
        # First create a claim
        claim = await processing_interface.create_claim(
            content="Test claim for evaluation",
            confidence=0.7
        )
        
        # Then evaluate it
        result = await processing_interface.evaluate_claim(claim.id)
        
        assert result.success is True
        assert result.claim_id == claim.id
        assert result.original_confidence == 0.7
        assert result.new_confidence == 0.9
        assert result.state == ClaimState.VALIDATED
        assert "Mock evaluation completed" in result.evaluation_summary
    
    @pytest.mark.asyncio
    async def test_search_claims(self, processing_interface):
        """Test claim search through interface"""
        # Create some claims
        await processing_interface.create_claim("First claim", 0.8)
        await processing_interface.create_claim("Second claim", 0.9)
        
        # Search claims
        results = await processing_interface.search_claims("test query")
        
        assert len(results) == 2
        assert all(isinstance(claim, Claim) for claim in results)
    
    @pytest.mark.asyncio
    async def test_execute_tool(self, processing_interface):
        """Test tool execution through interface"""
        result = await processing_interface.execute_tool(
            "test_tool",
            {"param1": "value1"}
        )
        
        assert result.success is True
        assert result.tool_name == "test_tool"
        assert result.parameters == {"param1": "value1"}
        assert "Mock tool execution completed" in result.outcome
        assert result.duration == 0.1
    
    @pytest.mark.asyncio
    async def test_get_context(self, processing_interface):
        """Test context retrieval through interface"""
        context = await processing_interface.get_context(["c0000001"])
        
        assert context.claim_ids == ["c0000001"]
        assert context.context_string == "Mock context"
        assert isinstance(context, Context)
    
    @pytest.mark.asyncio
    async def test_session_management(self, processing_interface):
        """Test session creation and resumption"""
        # Create session
        session = await processing_interface.create_session(
            user_data={"user": "test_user"}
        )
        
        assert session.session_id == "session_001"
        assert session.user_data == {"user": "test_user"}
        assert session.state == SessionState.ACTIVE
        
        # Resume session
        resumed_session = await processing_interface.resume_session(session.session_id)
        
        assert resumed_session.session_id == session.session_id
        assert resumed_session.user_data == session.user_data
    
    @pytest.mark.asyncio
    async def test_claim_crud_operations(self, processing_interface):
        """Test claim CRUD operations"""
        # Create claim
        claim = await processing_interface.create_claim("CRUD test claim", 0.8)
        
        # Get claim
        retrieved_claim = await processing_interface.get_claim(claim.id)
        assert retrieved_claim.id == claim.id
        assert retrieved_claim.content == claim.content
        
        # Update claim
        updated_claim = await processing_interface.update_claim(
            claim.id,
            {"confidence": 0.9, "tags": ["updated"]}
        )
        assert updated_claim.confidence == 0.9
        assert updated_claim.tags == ["updated"]
        
        # Delete claim
        deleted = await processing_interface.delete_claim(claim.id)
        assert deleted is True
        
        # Verify deletion
        with pytest.raises(Exception):  # Should raise ClaimNotFoundError
            await processing_interface.get_claim(claim.id)
    
    @pytest.mark.asyncio
    async def test_batch_operations(self, processing_interface):
        """Test batch operations"""
        # Batch create claims
        claims_data = [
            {"content": "Batch claim 1", "confidence": 0.8},
            {"content": "Batch claim 2", "confidence": 0.9},
            {"content": "Batch claim 3", "confidence": 0.7}
        ]
        
        created_claims = await processing_interface.batch_create_claims(claims_data)
        assert len(created_claims) == 3
        assert all(isinstance(claim, Claim) for claim in created_claims)
        
        # Batch evaluate claims
        claim_ids = [claim.id for claim in created_claims]
        evaluation_results = await processing_interface.batch_evaluate_claims(claim_ids)
        assert len(evaluation_results) == 3
        assert all(isinstance(result, EvaluationResult) for result in evaluation_results)
    
    @pytest.mark.asyncio
    async def test_event_streaming(self, processing_interface):
        """Test event streaming"""
        # Add some mock events
        processing_interface.events = [
            ProcessingEvent(
                event_type=EventType.CLAIM_CREATED,
                claim_id="c0000001",
                message="Claim created"
            ),
            ProcessingEvent(
                event_type=EventType.CLAIM_EVALUATED,
                claim_id="c0000001",
                message="Claim evaluated"
            )
        ]
        
        # Stream events
        events = []
        async for event in processing_interface.stream_events():
            events.append(event)
            if len(events) >= 2:
                break
        
        assert len(events) == 2
        assert events[0].event_type == EventType.CLAIM_CREATED
        assert events[1].event_type == EventType.CLAIM_EVALUATED
    
    @pytest.mark.asyncio
    async def test_health_and_statistics(self, processing_interface):
        """Test health status and statistics"""
        # Create some data
        await processing_interface.create_claim("Health test claim", 0.8)
        await processing_interface.create_session()
        
        # Get health status
        health = await processing_interface.get_health_status()
        assert health["status"] == "healthy"
        assert "interface" in health
        
        # Get statistics
        stats = await processing_interface.get_statistics()
        assert stats["claims_count"] == 1
        assert stats["sessions_count"] == 1


class TestDataModels:
    """Test supporting data models"""
    
    def test_evaluation_result(self):
        """Test EvaluationResult data model"""
        result = EvaluationResult(
            success=True,
            claim_id="c0000001",
            original_confidence=0.7,
            new_confidence=0.9,
            state=ClaimState.VALIDATED,
            evaluation_summary="Test evaluation"
        )
        
        assert result.success is True
        assert result.claim_id == "c0000001"
        assert result.original_confidence == 0.7
        assert result.new_confidence == 0.9
        assert result.state == ClaimState.VALIDATED
        
        # Test to_dict conversion
        result_dict = result.to_dict()
        assert result_dict["success"] is True
        assert result_dict["claim_id"] == "c0000001"
        assert result_dict["state"] == "Validated"
    
    def test_tool_result(self):
        """Test ToolResult data model"""
        result = ToolResult(
            success=True,
            tool_name="test_tool",
            parameters={"param": "value"},
            outcome="Tool executed successfully",
            duration=0.5
        )
        
        assert result.success is True
        assert result.tool_name == "test_tool"
        assert result.parameters == {"param": "value"}
        assert result.outcome == "Tool executed successfully"
        assert result.duration == 0.5
        
        # Test to_dict conversion
        result_dict = result.to_dict()
        assert result_dict["success"] is True
        assert result_dict["tool_name"] == "test_tool"
        assert result_dict["parameters"] == {"param": "value"}
    
    def test_context(self):
        """Test Context data model"""
        context = Context(
            claim_ids=["c0000001", "c0000002"],
            context_string="Test context",
            total_tokens=100
        )
        
        assert context.claim_ids == ["c0000001", "c0000002"]
        assert context.context_string == "Test context"
        assert context.total_tokens == 100
        
        # Test to_dict conversion
        context_dict = context.to_dict()
        assert context_dict["claim_ids"] == ["c0000001", "c0000002"]
        assert context_dict["context_string"] == "Test context"
        assert context_dict["total_tokens"] == 100
    
    def test_session(self):
        """Test Session data model"""
        session = Session(
            session_id="session_001",
            user_data={"user": "test_user"}
        )
        
        assert session.session_id == "session_001"
        assert session.user_data == {"user": "test_user"}
        assert session.state == SessionState.ACTIVE
        
        # Test to_dict conversion
        session_dict = session.to_dict()
        assert session_dict["session_id"] == "session_001"
        assert session_dict["user_data"] == {"user": "test_user"}
        assert session_dict["state"] == "active"
    
    def test_processing_event(self):
        """Test ProcessingEvent data model"""
        event = ProcessingEvent(
            event_type=EventType.CLAIM_CREATED,
            claim_id="c0000001",
            message="Claim created successfully"
        )
        
        assert event.event_type == EventType.CLAIM_CREATED
        assert event.claim_id == "c0000001"
        assert event.message == "Claim created successfully"
        assert isinstance(event.timestamp, datetime)
        
        # Test to_dict conversion
        event_dict = event.to_dict()
        assert event_dict["event_type"] == "claim_created"
        assert event_dict["claim_id"] == "c0000001"
        assert event_dict["message"] == "Claim created successfully"
        assert "timestamp" in event_dict


class TestEventTypes:
    """Test event type enumerations"""
    
    def test_event_types(self):
        """Test EventType enumeration"""
        assert EventType.CLAIM_CREATED.value == "claim_created"
        assert EventType.CLAIM_EVALUATED.value == "claim_evaluated"
        assert EventType.TOOL_CALLED.value == "tool_called"
        assert EventType.RESPONSE_GENERATED.value == "response_generated"
        assert EventType.SESSION_CREATED.value == "session_created"
        assert EventType.ERROR_OCCURRED.value == "error_occurred"
    
    def test_session_states(self):
        """Test SessionState enumeration"""
        assert SessionState.ACTIVE.value == "active"
        assert SessionState.PAUSED.value == "paused"
        assert SessionState.COMPLETED.value == "completed"
        assert SessionState.ERROR.value == "error"