"""
Comprehensive unit tests for Process Layer functionality
Tests context building, LLM processing, and instruction identification
"""

import pytest
import asyncio
import tempfile
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, patch

# Test imports
from src.process.models import (
    InstructionType, ProcessingStatus, ContextResult, Instruction,
    ProcessingResult, ProcessingConfig
)
from src.process.context_builder import ProcessContextBuilder
from src.process.llm_processor import ProcessLLMProcessor
from src.data.models import (
    Claim, ClaimState, ClaimType, ClaimFilter,
    ProcessingResult as DataProcessingResult
)
from src.data.lancedb_repositories import LanceDBClaimRepository, LanceDBRelationshipRepository, create_lancedb_repositories
from src.core.models import ClaimScope

# Skip tests if LanceDB not available
lancedb_available = True
try:
    import lancedb
    import pandas as pd
    import numpy as np
except ImportError:
    lancedb_available = False

pytestmark = pytest.mark.skipif(not lancedb_available, reason="LanceDB not available")

class TestProcessLayerModels:
    """Test process layer model classes"""

    def test_instruction_type_enum(self):
        """Test InstructionType enum values"""
        assert InstructionType.CREATE_CLAIM == "create_claim"
        assert InstructionType.UPDATE_CLAIM == "update_claim"
        assert InstructionType.VALIDATE_CLAIM == "validate_claim"
        assert InstructionType.SEARCH_CLAIMS == "search_claims"

    def test_processing_status_enum(self):
        """Test ProcessingStatus enum values"""
        assert ProcessingStatus.PENDING == "pending"
        assert ProcessingStatus.IN_PROGRESS == "in_progress"
        assert ProcessingStatus.COMPLETED == "completed"
        assert ProcessingStatus.FAILED == "failed"

    def test_context_result_model(self):
        """Test ContextResult model creation and validation"""
        claim = Claim(
            id="c0000001",
            content="Test claim",
            confidence=0.8,
            type=[ClaimType.CONCEPT],
            tags=["test"],
            state=ClaimState.EXPLORE,
            scope=ClaimScope.USER_WORKSPACE
        )

        result = ContextResult(
            claim_id="c0000001",
            context_claims=[claim],
            context_size=100,
            traversal_depth=2,
            build_time_ms=50,
            metadata={"test": "value"}
        )

        assert result.claim_id == "c0000001"
        assert len(result.context_claims) == 1
        assert result.context_claims[0].id == "c0000001"
        assert result.context_size == 100
        assert result.traversal_depth == 2
        assert result.build_time_ms == 50
        assert result.metadata["test"] == "value"

    def test_instruction_model(self):
        """Test Instruction model creation and validation"""
        instruction = Instruction(
            instruction_type=InstructionType.CREATE_CLAIM,
            description="Create a new claim about AI",
            parameters={"content": "AI will transform society", "confidence": 0.9},
            confidence=0.85,
            priority=1,
            source_claim_id="c0000001"
        )

        assert instruction.instruction_type == InstructionType.CREATE_CLAIM
        assert "Create a new claim" in instruction.description
        assert instruction.parameters["content"] == "AI will transform society"
        assert instruction.confidence == 0.85
        assert instruction.priority == 1
        assert instruction.source_claim_id == "c0000001"
        assert isinstance(instruction.created_at, datetime)

    def test_processing_result_model(self):
        """Test ProcessingResult model creation and validation"""
        instruction = Instruction(
            instruction_type=InstructionType.VALIDATE_CLAIM,
            description="Validate the claim",
            confidence=0.8,
            priority=2
        )

        result = ProcessingResult(
            claim_id="c0000001",
            status=ProcessingStatus.COMPLETED,
            instructions=[instruction],
            evaluation_score=0.9,
            reasoning="Strong evidence supports this claim",
            processing_time_ms=150,
            metadata={"model": "gpt-4"}
        )

        assert result.claim_id == "c0000001"
        assert result.status == ProcessingStatus.COMPLETED
        assert len(result.instructions) == 1
        assert result.instructions[0].instruction_type == InstructionType.VALIDATE_CLAIM
        assert result.evaluation_score == 0.9
        assert "Strong evidence" in result.reasoning
        assert result.processing_time_ms == 150
        assert result.metadata["model"] == "gpt-4"

    def test_processing_config_model(self):
        """Test ProcessingConfig model creation and validation"""
        config = ProcessingConfig(
            max_context_size=15,
            max_traversal_depth=5,
            instruction_confidence_threshold=0.8,
            enable_parallel_processing=False
        )

        assert config.max_context_size == 15
        assert config.max_traversal_depth == 5
        assert config.instruction_confidence_threshold == 0.8
        assert config.enable_parallel_processing is False

class TestProcessContextBuilder:
    """Test ProcessContextBuilder functionality"""

    @pytest.fixture
    async def repositories(self):
        """Create temporary repositories for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_context.lance")
            claim_repo, relationship_repo = await create_lancedb_repositories(db_path)
            yield claim_repo, relationship_repo

    @pytest.fixture
    async def sample_claims(self):
        """Create sample claims for testing"""
        main_claim = Claim(
            id="c0000001",
            content="Machine learning algorithms require large datasets for training",
            confidence=0.8,
            type=[ClaimType.THESIS],
            tags=["ml", "datasets", "training"],
            state=ClaimState.EXPLORE,
            scope=ClaimScope.USER_WORKSPACE
        )

        supporting_claim = Claim(
            id="c0000002",
            content="Deep learning models have been trained on millions of images",
            confidence=0.9,
            type=[ClaimType.REFERENCE],
            tags=["deep-learning", "images", "training"],
            state=ClaimState.VALIDATED,
            scope=ClaimScope.USER_WORKSPACE
        )

        related_claim = Claim(
            id="c0000003",
            content="Data preprocessing is essential for ML model performance",
            confidence=0.85,
            type=[ClaimType.CONCEPT],
            tags=["preprocessing", "ml", "performance"],
            state=ClaimState.VALIDATED,
            scope=ClaimScope.USER_WORKSPACE
        )

        return main_claim, supporting_claim, related_claim

    @pytest.fixture
    async def context_builder(self, repositories):
        """Create context builder for testing"""
        claim_repo, relationship_repo = repositories
        config = ProcessingConfig(
            max_context_size=10,
            max_traversal_depth=3,
            instruction_confidence_threshold=0.7
        )
        return ProcessContextBuilder(claim_repo, config)

    @pytest.mark.asyncio
    async def test_build_context_success(self, repositories, sample_claims, context_builder):
        """Test successful context building"""
        claim_repo, relationship_repo = repositories
        main_claim, supporting_claim, related_claim = sample_claims

        # Create claims
        await claim_repo.create_claim(main_claim)
        await claim_repo.create_claim(supporting_claim)
        await claim_repo.create_claim(related_claim)

        # Create relationships
        await relationship_repo.create_relationship(supporting_claim.id, main_claim.id)
        await relationship_repo.create_relationship(related_claim.id, main_claim.id)

        # Build context
        result = await context_builder.build_context(main_claim.id)

        assert isinstance(result, ContextResult)
        assert result.claim_id == main_claim.id
        assert len(result.context_claims) >= 1  # Should include main claim
        assert result.traversal_depth <= 3
        assert result.build_time_ms >= 0
        assert "build_strategy" in result.metadata

    @pytest.mark.asyncio
    async def test_build_context_claim_not_found(self, context_builder):
        """Test context building with non-existent claim"""
        with pytest.raises(ValueError, match="Claim not found"):
            await context_builder.build_context("c9999999")

    @pytest.mark.asyncio
    async def test_build_context_with_hints(self, repositories, sample_claims, context_builder):
        """Test context building with hints"""
        claim_repo, relationship_repo = repositories
        main_claim, supporting_claim, related_claim = sample_claims

        # Create claims
        await claim_repo.create_claim(main_claim)
        await claim_repo.create_claim(supporting_claim)
        await claim_repo.create_claim(related_claim)

        # Build context with hints
        hints = ["deep-learning", "preprocessing"]
        result = await context_builder.build_context(main_claim.id, context_hints=hints)

        assert result.claim_id == main_claim.id
        assert result.metadata["hints_used"] == hints

    @pytest.mark.asyncio
    async def test_build_context_depth_limit(self, repositories, sample_claims):
        """Test context building with depth limit"""
        claim_repo, relationship_repo = repositories
        main_claim, supporting_claim, related_claim = sample_claims

        # Create claims
        await claim_repo.create_claim(main_claim)
        await claim_repo.create_claim(supporting_claim)
        await claim_repo.create_claim(related_claim)

        # Create relationships
        await relationship_repo.create_relationship(supporting_claim.id, main_claim.id)
        await relationship_repo.create_relationship(related_claim.id, main_claim.id)

        # Build context with depth limit
        config = ProcessingConfig(max_traversal_depth=1)
        context_builder = ProcessContextBuilder(claim_repo, config)
        result = await context_builder.build_context(main_claim.id)

        assert result.traversal_depth <= 1

    @pytest.mark.asyncio
    async def test_context_caching(self, repositories, sample_claims, context_builder):
        """Test context result caching"""
        claim_repo, relationship_repo = repositories
        main_claim, supporting_claim, related_claim = sample_claims

        # Create claims
        await claim_repo.create_claim(main_claim)
        await claim_repo.create_claim(supporting_claim)

        # Build context first time
        result1 = await context_builder.build_context(main_claim.id)
        assert result1.metadata["cache_hit"] is False

        # Build context second time (should use cache)
        result2 = await context_builder.build_context(main_claim.id)
        assert result2.metadata["cache_hit"] is True

    @pytest.mark.asyncio
    async def test_clear_cache(self, context_builder):
        """Test clearing context cache"""
        # Add something to cache manually
        claim_id = "c0000001"
        context_builder._context_cache[claim_id] = ContextResult(
            claim_id=claim_id,
            context_claims=[],
            context_size=0,
            traversal_depth=0
        )

        # Clear cache
        context_builder.clear_cache()
        assert len(context_builder._context_cache) == 0

class TestProcessLLMProcessor:
    """Test ProcessLLMProcessor functionality"""

    @pytest.fixture
    async def repositories(self):
        """Create temporary repositories for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_processor.lance")
            claim_repo, relationship_repo = await create_lancedb_repositories(db_path)
            yield claim_repo, relationship_repo

    @pytest.fixture
    def mock_llm_bridge(self):
        """Create a mock LLM bridge for testing"""
        bridge = AsyncMock()

        # Mock response for claim evaluation
        bridge.generate_response.return_value = {
            "content": "This claim is well-supported with strong evidence.",
            "confidence": 0.9,
            "reasoning": "Multiple sources confirm this assertion.",
            "tokens_used": 150
        }

        return bridge

    @pytest.fixture
    def llm_processor(self, mock_llm_bridge):
        """Create LLM processor for testing"""
        config = ProcessingConfig(
            instruction_confidence_threshold=0.7,
            enable_parallel_processing=True
        )
        return ProcessLLMProcessor(mock_llm_bridge, config)

    @pytest.fixture
    def sample_claim(self):
        """Create sample claim for testing"""
        return Claim(
            id="c0000001",
            content="Regular exercise improves mental health and cognitive function",
            confidence=0.75,
            type=[ClaimType.THESIS],
            tags=["exercise", "health", "cognition"],
            state=ClaimState.EXPLORE,
            scope=ClaimScope.USER_WORKSPACE
        )

    @pytest.mark.asyncio
    async def test_evaluate_claim_success(self, llm_processor, sample_claim):
        """Test successful claim evaluation"""
        result = await llm_processor.evaluate_claim(sample_claim)

        assert isinstance(result, ProcessingResult)
        assert result.claim_id == sample_claim.id
        assert result.status == ProcessingStatus.COMPLETED
        assert result.evaluation_score is not None
        assert result.reasoning is not None
        assert result.processing_time_ms >= 0

    @pytest.mark.asyncio
    async def test_evaluate_claim_with_context(self, llm_processor, sample_claim):
        """Test claim evaluation with context"""
        # Create context with related claims
        context_claims = [
            Claim(
                id="c0000002",
                content="Studies show exercise increases brain-derived neurotrophic factor",
                confidence=0.9,
                type=[ClaimType.REFERENCE],
                tags=["exercise", "bdnf", "neuroscience"],
                state=ClaimState.VALIDATED,
                scope=ClaimScope.USER_WORKSPACE
            )
        ]

        context = ContextResult(
            claim_id=sample_claim.id,
            context_claims=context_claims,
            context_size=200,
            traversal_depth=1
        )

        result = await llm_processor.evaluate_claim(sample_claim, context)

        assert result.status == ProcessingStatus.COMPLETED
        assert result.evaluation_score is not None

    @pytest.mark.asyncio
    async def test_identify_instructions_success(self, llm_processor, sample_claim):
        """Test successful instruction identification"""
        result = await llm_processor.identify_instructions(sample_claim)

        assert isinstance(result, ProcessingResult)
        assert result.claim_id == sample_claim.id
        assert result.status == ProcessingStatus.COMPLETED
        assert isinstance(result.instructions, list)

    @pytest.mark.asyncio
    async def test_identify_instructions_with_types(self, llm_processor, sample_claim):
        """Test instruction identification with specific types"""
        instruction_types = [InstructionType.VALIDATE_CLAIM, InstructionType.SEARCH_CLAIMS]

        result = await llm_processor.identify_instructions(
            sample_claim,
            instruction_types=instruction_types
        )

        assert result.status == ProcessingStatus.COMPLETED
        # Check that returned instructions are of the requested types
        for instruction in result.instructions:
            assert instruction.instruction_type in instruction_types

    @pytest.mark.asyncio
    async def test_process_claim_complete_workflow(self, llm_processor, sample_claim):
        """Test complete claim processing workflow"""
        result = await llm_processor.process_claim(sample_claim)

        assert isinstance(result, ProcessingResult)
        assert result.claim_id == sample_claim.id
        assert result.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]

        if result.status == ProcessingStatus.COMPLETED:
            assert result.evaluation_score is not None
            assert isinstance(result.instructions, list)

    @pytest.mark.asyncio
    async def test_process_batch_claims(self, llm_processor):
        """Test batch processing of multiple claims"""
        claims = [
            Claim(
                id=f"c000000{i:02d}",
                content=f"Test claim {i} about data science",
                confidence=0.7 + (i * 0.05),
                type=[ClaimType.CONCEPT],
                tags=["test", "data-science"],
                state=ClaimState.EXPLORE,
                scope=ClaimScope.USER_WORKSPACE
            )
            for i in range(1, 6)
        ]

        results = await llm_processor.process_batch(claims)

        assert len(results) == len(claims)
        for i, result in enumerate(results):
            assert result.claim_id == claims[i].id
            assert isinstance(result, ProcessingResult)

    @pytest.mark.asyncio
    async def test_llm_bridge_error_handling(self, sample_claim):
        """Test handling of LLM bridge errors"""
        # Create mock bridge that raises error
        error_bridge = AsyncMock()
        error_bridge.generate_response.side_effect = Exception("LLM service unavailable")

        llm_processor = ProcessLLMProcessor(error_bridge)

        result = await llm_processor.evaluate_claim(sample_claim)

        assert result.status == ProcessingStatus.FAILED
        assert "LLM service unavailable" in result.error_message

    @pytest.mark.asyncio
    async def test_instruction_confidence_filtering(self, llm_processor, sample_claim):
        """Test filtering of instructions by confidence threshold"""
        # Mock bridge to return low confidence instructions
        llm_processor.llm_bridge.generate_response.return_value = {
            "content": "Analysis complete",
            "instructions": [
                {
                    "type": "validate_claim",
                    "description": "Low priority validation",
                    "confidence": 0.5  # Below threshold
                },
                {
                    "type": "search_claims",
                    "description": "High priority search",
                    "confidence": 0.9  # Above threshold
                }
            ]
        }

        result = await llm_processor.identify_instructions(sample_claim)

        # Should only include high confidence instruction
        assert len(result.instructions) == 1
        assert result.instructions[0].instruction_type == InstructionType.SEARCH_CLAIMS

    @pytest.mark.asyncio
    async def test_processing_metrics(self, llm_processor, sample_claim):
        """Test processing metrics and performance tracking"""
        result = await llm_processor.evaluate_claim(sample_claim)

        assert result.processing_time_ms >= 0
        assert "performance" in result.metadata

        # Get processor stats
        stats = await llm_processor.get_processing_stats()

        assert "total_processed" in stats
        assert "success_rate" in stats
        assert "average_processing_time" in stats

class TestProcessLayerIntegration:
    """Integration tests for complete process layer workflows"""

    @pytest.fixture
    async def full_system(self):
        """Create complete system with repositories and processors"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_integration.lance")
            claim_repo, relationship_repo = await create_lancedb_repositories(db_path)

            # Create mock LLM bridge
            llm_bridge = AsyncMock()
            llm_bridge.generate_response.return_value = {
                "content": "Claim evaluation complete",
                "confidence": 0.85,
                "reasoning": "Strong supporting evidence found",
                "instructions": [
                    {
                        "type": "validate_claim",
                        "description": "Validate this claim with additional sources",
                        "confidence": 0.8
                    }
                ]
            }

            # Create components
            config = ProcessingConfig(
                max_context_size=5,
                max_traversal_depth=2,
                instruction_confidence_threshold=0.7
            )

            context_builder = ProcessContextBuilder(claim_repo, config)
            llm_processor = ProcessLLMProcessor(llm_bridge, config)

            yield claim_repo, relationship_repo, context_builder, llm_processor

    @pytest.mark.asyncio
    async def test_complete_claim_processing_workflow(self, full_system):
        """Test complete workflow from claim creation to processing"""
        claim_repo, relationship_repo, context_builder, llm_processor = full_system

        # Create main claim
        main_claim = Claim(
            id="c0000001",
            content="Renewable energy sources will replace fossil fuels by 2050",
            confidence=0.7,
            type=[ClaimType.THESIS],
            tags=["renewable", "energy", "future"],
            state=ClaimState.EXPLORE,
            scope=ClaimScope.USER_WORKSPACE
        )

        result = await claim_repo.create_claim(main_claim)
        assert result.success is True

        # Create supporting claims
        supporting_claims = []
        for i, content in enumerate([
            "Solar energy costs have decreased by 90% in the last decade",
            "Wind energy is now the cheapest source of new electricity",
            "Battery storage technology is improving exponentially"
        ]):
            claim = Claim(
                id=f"c000000{i+2}",
                content=content,
                confidence=0.9,
                type=[ClaimType.REFERENCE],
                tags=["renewable", "evidence", f"source{i}"],
                state=ClaimState.VALIDATED,
                scope=ClaimScope.USER_WORKSPACE
            )

            await claim_repo.create_claim(claim)
            supporting_claims.append(claim)

            # Create relationship
            await relationship_repo.create_relationship(claim.id, main_claim.id)

        # Build context
        context = await context_builder.build_context(main_claim.id)
        assert len(context.context_claims) >= 1

        # Process claim with context
        processing_result = await llm_processor.process_claim(main_claim, context)
        assert processing_result.status == ProcessingStatus.COMPLETED
        assert processing_result.evaluation_score is not None
        assert len(processing_result.instructions) > 0

        # Verify instructions were identified correctly
        instructions = processing_result.instructions
        assert all(instr.confidence >= 0.7 for instr in instructions)

    @pytest.mark.asyncio
    async def test_batch_processing_with_relationships(self, full_system):
        """Test batch processing of related claims"""
        claim_repo, relationship_repo, context_builder, llm_processor = full_system

        # Create claim network
        claims = []
        for i in range(1, 6):
            claim = Claim(
                id=f"c000000{i:02d}",
                content=f"AI research claim {i} about machine learning ethics",
                confidence=0.6 + (i * 0.05),
                type=[ClaimType.THESIS if i % 2 == 0 else ClaimType.CONCEPT],
                tags=["ai", "ethics", f"claim{i}"],
                state=ClaimState.EXPLORE,
                scope=ClaimScope.USER_WORKSPACE
            )

            await claim_repo.create_claim(claim)
            claims.append(claim)

        # Create relationships between claims
        for i in range(len(claims) - 1):
            await relationship_repo.create_relationship(claims[i].id, claims[i + 1].id)

        # Process batch
        results = await llm_processor.process_batch(claims)

        # Verify all were processed
        assert len(results) == len(claims)
        for i, result in enumerate(results):
            assert result.claim_id == claims[i].id
            assert result.status == ProcessingStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_context_building_performance(self, full_system):
        """Test context building performance with large claim networks"""
        claim_repo, relationship_repo, context_builder, llm_processor = full_system

        # Create large claim network
        claims = []
        for i in range(50):
            claim = Claim(
                id=f"c{i:08d}",
                content=f"Research claim {i} about data science and analytics",
                confidence=0.5 + (i % 50) / 100,
                type=[ClaimType.CONCEPT],
                tags=["research", f"topic{i}", "data-science"],
                state=ClaimState.EXPLORE if i % 3 == 0 else ClaimState.VALIDATED,
                scope=ClaimScope.USER_WORKSPACE
            )

            await claim_repo.create_claim(claim)
            claims.append(claim)

        # Create some relationships
        for i in range(0, 40, 2):
            await relationship_repo.create_relationship(claims[i].id, claims[i + 1].id)

        # Test context building performance
        start_time = datetime.utcnow()
        context = await context_builder.build_context(claims[0].id, max_depth=2)
        end_time = datetime.utcnow()

        processing_time = (end_time - start_time).total_seconds() * 1000

        assert context.claim_id == claims[0].id
        assert processing_time < 5000  # Should complete within 5 seconds
        assert context.build_time_ms >= 0

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])