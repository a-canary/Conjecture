# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Tests for Process Layer Models

Tests the Pydantic models used by the Process Layer for claim evaluation,
instruction identification, and processing workflow.
"""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from src.process.models import (
    InstructionType,
    ProcessingStatus,
    ContextResult,
    Instruction,
    ProcessingResult,
    ProcessingConfig,
    ProcessingRequest,
    ProcessingBatch,
)
from src.core.models import Claim, ClaimType, ClaimScope


class TestInstructionType:
    """Tests for InstructionType enum."""

    def test_all_instruction_types_defined(self):
        """Verify all expected instruction types exist."""
        expected = [
            "create_claim",
            "update_claim",
            "delete_claim",
            "search_claims",
            "analyze_claim",
            "validate_claim",
            "connect_claims",
            "generate_context",
            "custom_action",
        ]
        actual = [t.value for t in InstructionType]
        assert set(expected) == set(actual)

    def test_instruction_type_is_string_enum(self):
        """Verify InstructionType values are strings."""
        for inst_type in InstructionType:
            assert isinstance(inst_type.value, str)

    def test_instruction_type_from_string(self):
        """Test creating InstructionType from string value."""
        assert InstructionType("create_claim") == InstructionType.CREATE_CLAIM
        assert InstructionType("delete_claim") == InstructionType.DELETE_CLAIM


class TestProcessingStatus:
    """Tests for ProcessingStatus enum."""

    def test_all_statuses_defined(self):
        """Verify all expected statuses exist."""
        expected = ["pending", "in_progress", "completed", "failed", "cancelled", "retrying"]
        actual = [s.value for s in ProcessingStatus]
        assert set(expected) == set(actual)

    def test_status_is_string_enum(self):
        """Verify ProcessingStatus values are strings."""
        for status in ProcessingStatus:
            assert isinstance(status.value, str)


class TestContextResult:
    """Tests for ContextResult model."""

    def test_minimal_context_result(self):
        """Test creating ContextResult with minimal fields."""
        result = ContextResult(claim_id="c123")
        assert result.claim_id == "c123"
        assert result.context_claims == []
        assert result.context_size == 0
        assert result.traversal_depth == 0
        assert result.build_time_ms == 0
        assert result.metadata == {}

    def test_full_context_result(self):
        """Test creating ContextResult with all fields."""
        claim = Claim(
            id="related_1",
            content="Related claim for context",
            confidence=0.8,
            type=[ClaimType.CONCEPT],
            scope=ClaimScope.USER_WORKSPACE,
        )
        result = ContextResult(
            claim_id="c123",
            context_claims=[claim],
            context_size=150,
            traversal_depth=2,
            build_time_ms=45,
            metadata={"source": "test"},
        )
        assert result.claim_id == "c123"
        assert len(result.context_claims) == 1
        assert result.context_size == 150
        assert result.traversal_depth == 2
        assert result.build_time_ms == 45
        assert result.metadata["source"] == "test"

    def test_context_result_serialization(self):
        """Test ContextResult serialization with context_claims."""
        claim = Claim(
            id="related_1",
            content="Related claim for context",
            confidence=0.8,
            type=[ClaimType.CONCEPT],
            scope=ClaimScope.USER_WORKSPACE,
        )
        result = ContextResult(
            claim_id="c123",
            context_claims=[claim],
            context_size=100,
            traversal_depth=2,
        )
        data = result.model_dump()
        assert data["claim_id"] == "c123"
        assert data["context_size"] == 100
        assert len(data["context_claims"]) == 1
        assert data["context_claims"][0]["id"] == "related_1"


class TestInstruction:
    """Tests for Instruction model."""

    def test_minimal_instruction(self):
        """Test creating Instruction with minimal fields."""
        inst = Instruction(
            instruction_type=InstructionType.CREATE_CLAIM,
            description="Create a new claim",
        )
        assert inst.instruction_type == InstructionType.CREATE_CLAIM
        assert inst.description == "Create a new claim"
        assert inst.parameters == {}
        assert inst.confidence == 0.0
        assert inst.priority == 0
        assert inst.source_claim_id is None

    def test_full_instruction(self):
        """Test creating Instruction with all fields."""
        inst = Instruction(
            instruction_type=InstructionType.UPDATE_CLAIM,
            description="Update confidence score",
            parameters={"confidence": 0.9, "claim_id": "c123"},
            confidence=0.85,
            priority=10,
            source_claim_id="source_001",
        )
        assert inst.instruction_type == InstructionType.UPDATE_CLAIM
        assert inst.parameters["confidence"] == 0.9
        assert inst.confidence == 0.85
        assert inst.priority == 10
        assert inst.source_claim_id == "source_001"

    def test_instruction_confidence_validation(self):
        """Test confidence must be between 0 and 1."""
        # Valid confidence
        inst = Instruction(
            instruction_type=InstructionType.CREATE_CLAIM,
            description="Test",
            confidence=0.5,
        )
        assert inst.confidence == 0.5

        # Invalid confidence (too high)
        with pytest.raises(ValidationError):
            Instruction(
                instruction_type=InstructionType.CREATE_CLAIM,
                description="Test",
                confidence=1.5,
            )

        # Invalid confidence (negative)
        with pytest.raises(ValidationError):
            Instruction(
                instruction_type=InstructionType.CREATE_CLAIM,
                description="Test",
                confidence=-0.1,
            )

    def test_instruction_serialization(self):
        """Test Instruction serialization."""
        inst = Instruction(
            instruction_type=InstructionType.DELETE_CLAIM,
            description="Delete old claim",
            confidence=0.95,
        )
        data = inst.model_dump()
        assert data["instruction_type"] == "delete_claim"
        assert data["description"] == "Delete old claim"
        assert "created_at" in data


class TestProcessingResult:
    """Tests for ProcessingResult model."""

    def test_minimal_processing_result(self):
        """Test creating ProcessingResult with minimal fields."""
        result = ProcessingResult(
            claim_id="c123",
            status=ProcessingStatus.COMPLETED,
        )
        assert result.claim_id == "c123"
        assert result.status == ProcessingStatus.COMPLETED
        assert result.instructions == []
        assert result.evaluation_score is None
        assert result.reasoning is None
        assert result.error_message is None

    def test_full_processing_result(self):
        """Test creating ProcessingResult with all fields."""
        inst = Instruction(
            instruction_type=InstructionType.ANALYZE_CLAIM,
            description="Analyze claim content",
        )
        result = ProcessingResult(
            claim_id="c123",
            status=ProcessingStatus.COMPLETED,
            instructions=[inst],
            evaluation_score=0.87,
            reasoning="Claim is well-supported",
            processing_time_ms=250,
            metadata={"model": "test-model"},
        )
        assert result.evaluation_score == 0.87
        assert result.reasoning == "Claim is well-supported"
        assert result.processing_time_ms == 250
        assert len(result.instructions) == 1

    def test_failed_processing_result(self):
        """Test ProcessingResult for failed operation."""
        result = ProcessingResult(
            claim_id="c123",
            status=ProcessingStatus.FAILED,
            error_message="LLM provider unavailable",
            processing_time_ms=50,
        )
        assert result.status == ProcessingStatus.FAILED
        assert result.error_message == "LLM provider unavailable"


class TestProcessingConfig:
    """Tests for ProcessingConfig model."""

    def test_default_config(self):
        """Test ProcessingConfig with default values."""
        config = ProcessingConfig()
        assert config.max_context_size == 10
        assert config.max_traversal_depth == 3
        assert config.instruction_confidence_threshold == 0.7
        assert config.enable_parallel_processing is True
        assert config.timeout_seconds == 300
        assert config.retry_attempts == 3

    def test_custom_config(self):
        """Test ProcessingConfig with custom values."""
        config = ProcessingConfig(
            max_context_size=20,
            max_traversal_depth=5,
            instruction_confidence_threshold=0.5,
            enable_parallel_processing=False,
            timeout_seconds=600,
            retry_attempts=5,
        )
        assert config.max_context_size == 20
        assert config.max_traversal_depth == 5
        assert config.instruction_confidence_threshold == 0.5
        assert config.enable_parallel_processing is False

    def test_config_threshold_validation(self):
        """Test instruction_confidence_threshold must be 0-1."""
        # Valid threshold
        config = ProcessingConfig(instruction_confidence_threshold=0.5)
        assert config.instruction_confidence_threshold == 0.5

        # Invalid threshold
        with pytest.raises(ValidationError):
            ProcessingConfig(instruction_confidence_threshold=1.5)

    def test_config_allows_extra_fields(self):
        """Test ProcessingConfig allows extra fields."""
        config = ProcessingConfig(custom_field="custom_value")
        assert config.custom_field == "custom_value"


class TestProcessingRequest:
    """Tests for ProcessingRequest model."""

    def test_minimal_request(self):
        """Test creating ProcessingRequest with minimal fields."""
        request = ProcessingRequest(claim_id="c123")
        assert request.claim_id == "c123"
        assert request.config is None
        assert request.context_hints == []
        assert request.instruction_types == []
        assert request.metadata == {}

    def test_full_request(self):
        """Test creating ProcessingRequest with all fields."""
        config = ProcessingConfig(max_context_size=5)
        request = ProcessingRequest(
            claim_id="c123",
            config=config,
            context_hints=["mathematics", "algebra"],
            instruction_types=[InstructionType.ANALYZE_CLAIM, InstructionType.VALIDATE_CLAIM],
            metadata={"priority": "high"},
        )
        assert request.config.max_context_size == 5
        assert request.context_hints == ["mathematics", "algebra"]
        assert len(request.instruction_types) == 2
        assert request.metadata["priority"] == "high"

    def test_request_serialization(self):
        """Test ProcessingRequest serialization."""
        request = ProcessingRequest(
            claim_id="c123",
            instruction_types=[InstructionType.CREATE_CLAIM],
        )
        data = request.model_dump()
        assert data["claim_id"] == "c123"
        assert data["instruction_types"] == ["create_claim"]
        assert "requested_at" in data


class TestProcessingBatch:
    """Tests for ProcessingBatch model."""

    def test_minimal_batch(self):
        """Test creating ProcessingBatch with minimal fields."""
        request = ProcessingRequest(claim_id="c123")
        batch = ProcessingBatch(
            requests=[request],
            batch_id="batch_001",
        )
        assert batch.batch_id == "batch_001"
        assert len(batch.requests) == 1
        assert batch.config is None

    def test_batch_with_multiple_requests(self):
        """Test ProcessingBatch with multiple requests."""
        requests = [
            ProcessingRequest(claim_id="c1"),
            ProcessingRequest(claim_id="c2"),
            ProcessingRequest(claim_id="c3"),
        ]
        config = ProcessingConfig(enable_parallel_processing=True)
        batch = ProcessingBatch(
            requests=requests,
            batch_id="batch_multi",
            config=config,
        )
        assert len(batch.requests) == 3
        assert batch.config.enable_parallel_processing is True

    def test_batch_serialization(self):
        """Test ProcessingBatch serialization."""
        request = ProcessingRequest(claim_id="c123")
        batch = ProcessingBatch(
            requests=[request],
            batch_id="batch_001",
        )
        data = batch.model_dump()
        assert data["batch_id"] == "batch_001"
        assert len(data["requests"]) == 1
        assert "created_at" in data
