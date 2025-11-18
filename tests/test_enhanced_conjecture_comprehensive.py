"""
Comprehensive Test Suite for Enhanced Conjecture
Tests all major components including async evaluation and dynamic tool creation
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
import time

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from core.unified_models import Claim, ClaimState, ClaimType
from processing.async_claim_evaluation import (
    AsyncClaimEvaluationService,
    EvaluationEvent,
    EvaluationTask,
)
from processing.context_collector import ContextCollector
from processing.dynamic_tool_creator import DynamicToolCreator, ToolValidator
from enhanced_conjecture import EnhancedConjecture, ExplorationResult
from processing.llm_bridge import LLMBridge, LLMRequest
from config.simple_config import Config


class TestAsyncClaimEvaluation:
    """Test Async Claim Evaluation Service"""

    @pytest.fixture
    def mock_llm_bridge(self):
        """Mock LLM bridge for testing"""
        bridge = Mock(spec=LLMBridge)
        bridge.is_available.return_value = True
        bridge.process.return_value = Mock(
            success=True,
            content="Confidence(0.85)\nComplete()",
            generated_claims=[],
            errors=[],
        )
        return bridge

    @pytest.fixture
    def mock_context_collector(self):
        """Mock context collector for testing"""
        collector = Mock(spec=ContextCollector)
        collector.build_context.return_value = []
        return collector

    @pytest.fixture
    async def evaluation_service(self, mock_llm_bridge, mock_context_collector):
        """Create evaluation service for testing"""
        service = AsyncClaimEvaluationService(
            llm_bridge=mock_llm_bridge,
            context_collector=mock_context_collector,
            max_concurrent_evaluations=2,
        )
        await service.start()
        yield service
        await service.stop()

    @pytest.mark.asyncio
    async def test_service_start_stop(self, evaluation_service):
        """Test service start and stop"""
        assert evaluation_service._running is True

        await evaluation_service.stop()
        assert evaluation_service._running is False

    @pytest.mark.asyncio
    async def test_submit_claim(self, evaluation_service):
        """Test claim submission"""
        claim = Claim(
            id="test_001",
            content="Test claim for evaluation",
            confidence=0.5,
            type=[ClaimType.CONCEPT],
            state=ClaimState.EXPLORE,
        )

        await evaluation_service.submit_claim(claim)

        # Check claim was added to queue
        assert len(evaluation_service._evaluation_queue) > 0

        # Check statistics
        stats = evaluation_service.get_statistics()
        assert stats["queue_depth"] > 0

    @pytest.mark.asyncio
    async def test_priority_calculation(self, evaluation_service):
        """Test priority calculation for claims"""
        high_confidence_claim = Claim(
            id="high_001",
            content="High confidence claim",
            confidence=0.9,
            type=[ClaimType.CONCEPT],
            state=ClaimState.EXPLORE,
        )

        low_confidence_claim = Claim(
            id="low_001",
            content="Low confidence claim",
            confidence=0.3,
            type=[ClaimType.CONCEPT],
            state=ClaimState.EXPLORE,
        )

        high_priority = evaluation_service._calculate_priority(high_confidence_claim)
        low_priority = evaluation_service._calculate_priority(low_confidence_claim)

        # Lower priority number = higher priority
        assert high_priority < low_priority

    @pytest.mark.asyncio
    async def test_event_system(self, evaluation_service):
        """Test event emission and subscription"""
        events_received = []

        def event_handler(event):
            events_received.append(event)

        evaluation_service.subscribe_to_events(event_handler)

        # Emit test event
        test_event = EvaluationEvent(claim_id="test_001", event_type="test_event")
        await evaluation_service._emit_event(test_event)

        # Wait for event processing
        await asyncio.sleep(0.1)

        assert len(events_received) > 0
        assert events_received[0].claim_id == "test_001"
        assert events_received[0].event_type == "test_event"


class TestContextCollector:
    """Test Context Collector"""

    @pytest.fixture
    def mock_data_manager(self):
        """Mock data manager for testing"""
        dm = Mock()
        dm.filter_claims.return_value = []
        dm.get_claim.return_value = None
        return dm

    @pytest.fixture
    def context_collector(self, mock_data_manager):
        """Create context collector for testing"""
        return ContextCollector(mock_data_manager)

    @pytest.mark.asyncio
    async def test_build_context_empty(self, context_collector):
        """Test context building with no related claims"""
        claim = Claim(
            id="test_001",
            content="Test claim",
            confidence=0.8,
            type=[ClaimType.CONCEPT],
        )

        context = await context_collector.build_context(claim)
        assert isinstance(context, list)

    @pytest.mark.asyncio
    async def test_collect_context_for_claim(self, context_collector):
        """Test context collection for a specific claim"""
        result = await context_collector.collect_context_for_claim(
            "test claim content", {"task": "testing"}, max_skills=2, max_samples=3
        )

        assert "claim_content" in result
        assert "skills" in result
        assert "samples" in result
        assert result["claim_content"] == "test claim content"
        assert isinstance(result["skills"], list)
        assert isinstance(result["samples"], list)

    def test_extract_keywords(self, context_collector):
        """Test keyword extraction from content"""
        content = "Machine learning algorithms require substantial training data"
        keywords = context_collector._extract_keywords(content)

        assert isinstance(keywords, list)
        assert "machine" in keywords
        assert "learning" in keywords
        assert "algorithms" in keywords
        assert "training" in keywords

    def test_relevance_scoring(self, context_collector):
        """Test relevance scoring for claims"""
        skill = Claim(
            id="skill_001",
            content="Machine learning research methodology",
            confidence=0.9,
            type=[ClaimType.CONCEPT],
            tags=["machine learning", "research"],
        )

        score = context_collector.relevance_scorer.score_skill_relevance(
            skill, "machine learning algorithms", {}
        )

        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be relevant


class TestDynamicToolCreator:
    """Test Dynamic Tool Creation System"""

    @pytest.fixture
    def mock_llm_bridge(self):
        """Mock LLM bridge for testing"""
        bridge = Mock(spec=LLMBridge)
        bridge.is_available.return_value = True
        return bridge

    @pytest.fixture
    def temp_tools_dir(self):
        """Create temporary tools directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def tool_creator(self, mock_llm_bridge, temp_tools_dir):
        """Create tool creator for testing"""
        return DynamicToolCreator(llm_bridge=mock_llm_bridge, tools_dir=temp_tools_dir)

    @pytest.mark.asyncio
    async def test_discover_tool_need(self, tool_creator):
        """Test tool need discovery"""
        claim = Claim(
            id="test_001",
            content="I need to calculate complex mathematical formulas",
            confidence=0.8,
            type=[ClaimType.CONCEPT],
        )

        # Mock LLM response indicating no tool needed
        tool_creator.llm_bridge.process.return_value = Mock(
            success=True, content="NO_TOOL_NEEDED"
        )

        tool_need = await tool_creator.discover_tool_need(claim)
        assert tool_need is None

        # Mock LLM response indicating tool needed
        tool_creator.llm_bridge.process.return_value = Mock(
            success=True,
            content="A calculator tool would be helpful for mathematical operations",
        )

        tool_need = await tool_creator.discover_tool_need(claim)
        assert tool_need is not None
        assert "calculator" in tool_need.lower()

    @pytest.mark.asyncio
    async def test_websearch_tool_methods(self, tool_creator):
        """Test searching for tool implementation methods"""
        with patch("tools.webSearch.WebSearch") as mock_search:
            mock_search.return_value.search.return_value = [
                {"content": "Method 1: Use Python's math library"},
                {"content": "Method 2: Implement custom functions"},
            ]

            methods = await tool_creator.websearch_tool_methods("calculator")

            assert len(methods) == 2
            assert "math library" in methods[0]
            assert "custom functions" in methods[1]

    @pytest.mark.asyncio
    async def test_create_tool_file(self, tool_creator):
        """Test tool file creation"""
        # Mock LLM response for code generation
        tool_creator.llm_bridge.process.return_value = Mock(
            success=True,
            content='''```python
"""
Test calculator tool
"""

def execute(a: float, b: float, operation: str = "add") -> dict:
    """Perform mathematical operations"""
    try:
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        else:
            return {"success": False, "error": "Unknown operation"}
        
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
```''',
        )

        tool_path = await tool_creator.create_tool_file(
            "calculator",
            "Mathematical calculator tool",
            ["Use math library", "Custom implementation"],
        )

        assert tool_path is not None
        assert Path(tool_path).exists()

        # Check tool was tracked
        assert "calculator" in tool_creator.created_tools

    def test_tool_validator(self):
        """Test tool code validation"""
        validator = ToolValidator()

        # Valid code
        valid_code = '''
def execute(param: str) -> dict:
    """Execute tool"""
    return {"success": True, "result": param}
'''

        is_valid, issues = validator.validate_tool_code(valid_code)
        assert is_valid is True
        assert len(issues) == 0

        # Invalid code with dangerous import
        invalid_code = '''
import os

def execute(param: str) -> dict:
    """Execute tool"""
    return {"success": True, "result": param}
'''

        is_valid, issues = validator.validate_tool_code(invalid_code)
        assert is_valid is False
        assert len(issues) > 0
        assert any("dangerous import" in issue.lower() for issue in issues)


class TestEnhancedConjecture:
    """Test Enhanced Conjecture Integration"""

    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration"""
        config = Config()
        config.database_type = "mock"
        config.llm_provider = "mock"
        return config

    @pytest.fixture
    async def enhanced_conjecture(self, temp_config):
        """Create enhanced conjecture for testing"""
        with patch("data.data_manager.get_data_manager") as mock_dm:
            mock_dm.return_value = Mock()
            mock_dm.return_value.create_claim = AsyncMock(
                return_value=Mock(
                    id="test_claim_001",
                    content="Test claim",
                    confidence=0.8,
                    type=[ClaimType.CONCEPT],
                    state=ClaimState.EXPLORE,
                    tags=[],
                )
            )

            conjecture = EnhancedConjecture(config=temp_config)
            await conjecture.start_services()
            yield conjecture
            await conjecture.stop_services()

    @pytest.mark.asyncio
    async def test_initialization(self, enhanced_conjecture):
        """Test enhanced conjecture initialization"""
        assert enhanced_conjecture.config is not None
        assert enhanced_conjecture.llm_bridge is not None
        assert enhanced_conjecture.context_collector is not None
        assert enhanced_conjecture.async_evaluation is not None
        assert enhanced_conjecture.tool_creator is not None
        assert enhanced_conjecture._services_started is True

    @pytest.mark.asyncio
    async def test_enhanced_exploration(self, enhanced_conjecture):
        """Test enhanced exploration functionality"""
        # Mock LLM response
        enhanced_conjecture.llm_bridge.process.return_value = Mock(
            success=True,
            content="""Claim: "Machine learning requires training data" Confidence: 0.9 Type: concept
Claim: "Neural networks are a type of ML algorithm" Confidence: 0.85 Type: concept""",
            generated_claims=[],
            errors=[],
        )

        result = await enhanced_conjecture.explore(
            "machine learning basics",
            max_claims=5,
            auto_evaluate=False,  # Disable evaluation for simpler testing
        )

        assert isinstance(result, ExplorationResult)
        assert result.query == "machine learning basics"
        assert len(result.claims) > 0
        assert result.search_time > 0
        assert result.evaluation_pending is False

    @pytest.mark.asyncio
    async def test_enhanced_claim_creation(self, enhanced_conjecture):
        """Test enhanced claim creation"""
        claim = await enhanced_conjecture.add_claim(
            content="Test claim with enhanced features",
            confidence=0.85,
            claim_type="concept",
            tags=["test", "enhanced"],
            auto_evaluate=False,
        )

        assert claim is not None
        assert claim.content == "Test claim with enhanced features"
        assert claim.confidence == 0.85
        assert ClaimType.CONCEPT in claim.type

    @pytest.mark.asyncio
    async def test_statistics(self, enhanced_conjecture):
        """Test statistics collection"""
        stats = enhanced_conjecture.get_statistics()

        assert "config" in stats
        assert "services_running" in stats
        assert "claims_processed" in stats
        assert "tools_created" in stats
        assert "evaluation_service" in stats
        assert "created_tools" in stats

        assert stats["services_running"] is True

    @pytest.mark.asyncio
    async def test_context_manager(self, temp_config):
        """Test async context manager"""
        with patch("data.data_manager.get_data_manager") as mock_dm:
            mock_dm.return_value = Mock()
            mock_dm.return_value.create_claim = AsyncMock()

            async with EnhancedConjecture(config=temp_config) as cf:
                assert cf._services_started is True

            # Services should be stopped after context exit
            assert cf._services_started is False


class TestIntegration:
    """Integration tests for the complete system"""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete workflow from exploration to evaluation"""
        # This would be a comprehensive integration test
        # For now, we'll test the basic flow

        config = Config()
        config.database_type = "mock"
        config.llm_provider = "mock"

        with patch("data.data_manager.get_data_manager") as mock_dm:
            # Mock data manager
            mock_dm.return_value = Mock()
            mock_dm.return_value.create_claim = AsyncMock(
                return_value=Mock(
                    id="integration_test_001",
                    content="Integration test claim",
                    confidence=0.8,
                    type=[ClaimType.CONCEPT],
                    state=ClaimState.EXPLORE,
                    tags=[],
                )
            )
            mock_dm.return_value.get_claim = AsyncMock(return_value=None)

            async with EnhancedConjecture(config=config) as cf:
                # Test exploration
                cf.llm_bridge.process.return_value = Mock(
                    success=True,
                    content='Claim: "Integration test successful" Confidence: 0.9 Type: concept',
                    generated_claims=[],
                    errors=[],
                )

                result = await cf.explore("integration test", auto_evaluate=False)
                assert len(result.claims) > 0

                # Test claim creation
                claim = await cf.add_claim(
                    content="Integration test claim",
                    confidence=0.8,
                    claim_type="concept",
                    auto_evaluate=False,
                )
                assert claim is not None

                # Test statistics
                stats = cf.get_statistics()
                assert stats["claims_processed"] > 0


# Performance and stress tests
class TestPerformance:
    """Performance tests for the enhanced system"""

    @pytest.mark.asyncio
    async def test_concurrent_claim_processing(self):
        """Test processing multiple claims concurrently"""
        config = Config()
        config.database_type = "mock"
        config.llm_provider = "mock"

        with patch("data.data_manager.get_data_manager") as mock_dm:
            mock_dm.return_value = Mock()
            mock_dm.return_value.create_claim = AsyncMock(
                return_value=Mock(
                    id="perf_test_001",
                    content="Performance test claim",
                    confidence=0.8,
                    type=[ClaimType.CONCEPT],
                    state=ClaimState.EXPLORE,
                    tags=[],
                )
            )

            async with EnhancedConjecture(config=config) as cf:
                # Create multiple claims concurrently
                tasks = []
                for i in range(10):
                    task = cf.add_claim(
                        content=f"Performance test claim {i}",
                        confidence=0.8,
                        claim_type="concept",
                        auto_evaluate=False,
                    )
                    tasks.append(task)

                start_time = time.time()
                claims = await asyncio.gather(*tasks)
                end_time = time.time()

                # Verify all claims were created
                assert len(claims) == 10

                # Check performance (should complete within reasonable time)
                processing_time = end_time - start_time
                assert processing_time < 5.0  # Should complete within 5 seconds

                print(f"Processed 10 claims concurrently in {processing_time:.2f}s")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])
