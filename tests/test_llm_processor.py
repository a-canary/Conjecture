"""
Tests for Process LLM Processor

Tests the ProcessLLMProcessor class which handles claim evaluation and
instruction identification using language models.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
import json

from src.process.llm_processor import ProcessLLMProcessor, _utc_now
from src.process.models import (
    ProcessingConfig,
    ProcessingRequest,
    ProcessingResult,
    ProcessingStatus,
    Instruction,
    InstructionType,
    ContextResult,
)
from src.processing.llm_bridge import LLMBridge
from src.core.models import Claim


class TestUtcNow:
    """Tests for _utc_now helper function."""

    def test_utc_now_returns_datetime(self):
        """Test _utc_now returns datetime object."""
        result = _utc_now()
        assert isinstance(result, datetime)

    def test_utc_now_has_timezone(self):
        """Test _utc_now returns timezone-aware datetime."""
        result = _utc_now()
        assert result.tzinfo is not None
        assert result.tzinfo == timezone.utc


class TestProcessLLMProcessorInit:
    """Tests for ProcessLLMProcessor initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default config."""
        bridge = MagicMock(spec=LLMBridge)
        processor = ProcessLLMProcessor(bridge)

        assert processor.llm_bridge == bridge
        assert processor.config is not None
        assert isinstance(processor.config, ProcessingConfig)
        assert processor._processing_cache == {}

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        bridge = MagicMock(spec=LLMBridge)
        config = ProcessingConfig(
            max_context_size=20,
            instruction_confidence_threshold=0.5,
        )
        processor = ProcessLLMProcessor(bridge, config)

        assert processor.config.max_context_size == 20
        assert processor.config.instruction_confidence_threshold == 0.5


class TestCacheOperations:
    """Tests for processing cache operations."""

    def test_clear_cache(self):
        """Test clearing the processing cache."""
        bridge = MagicMock(spec=LLMBridge)
        processor = ProcessLLMProcessor(bridge)

        # Add fake cache entries
        processor._processing_cache["key1"] = MagicMock()
        processor._processing_cache["key2"] = MagicMock()

        assert len(processor._processing_cache) == 2

        processor.clear_cache()

        assert len(processor._processing_cache) == 0

    def test_get_cache_stats_empty(self):
        """Test cache stats with empty cache."""
        bridge = MagicMock(spec=LLMBridge)
        processor = ProcessLLMProcessor(bridge)

        stats = processor.get_cache_stats()

        assert stats["cached_results"] == 0
        assert stats["cache_keys"] == []

    def test_get_cache_stats_with_entries(self):
        """Test cache stats with cached entries."""
        bridge = MagicMock(spec=LLMBridge)
        processor = ProcessLLMProcessor(bridge)

        processor._processing_cache["key1"] = MagicMock()
        processor._processing_cache["key2"] = MagicMock()

        stats = processor.get_cache_stats()

        assert stats["cached_results"] == 2
        assert set(stats["cache_keys"]) == {"key1", "key2"}


class TestGenerateCacheKey:
    """Tests for cache key generation."""

    def test_generate_cache_key_basic(self):
        """Test basic cache key generation."""
        bridge = MagicMock(spec=LLMBridge)
        processor = ProcessLLMProcessor(bridge)

        request = ProcessingRequest(
            claim_id="c123",
            context_hints=[],
            instruction_types=[],
        )

        key = processor._generate_cache_key(request)

        assert "c123" in key
        assert "|" in key

    def test_generate_cache_key_with_hints(self):
        """Test cache key with context hints."""
        bridge = MagicMock(spec=LLMBridge)
        processor = ProcessLLMProcessor(bridge)

        request = ProcessingRequest(
            claim_id="c123",
            context_hints=["math", "algebra"],
            instruction_types=[],
        )

        key = processor._generate_cache_key(request)

        assert "c123" in key
        # Hints should be sorted
        assert "algebra" in key or "math" in key

    def test_generate_cache_key_deterministic(self):
        """Test cache key is deterministic."""
        bridge = MagicMock(spec=LLMBridge)
        processor = ProcessLLMProcessor(bridge)

        request = ProcessingRequest(
            claim_id="c123",
            context_hints=["z", "a"],
            instruction_types=[InstructionType.CREATE_CLAIM],
        )

        key1 = processor._generate_cache_key(request)
        key2 = processor._generate_cache_key(request)

        assert key1 == key2


class TestBuildEvaluationPrompt:
    """Tests for evaluation prompt building."""

    @pytest.mark.asyncio
    async def test_build_basic_prompt(self):
        """Test building basic evaluation prompt."""
        bridge = MagicMock(spec=LLMBridge)
        processor = ProcessLLMProcessor(bridge)

        request = ProcessingRequest(claim_id="c123")

        prompt = await processor._build_evaluation_prompt(request, None)

        assert "c123" in prompt
        assert "evaluation score" in prompt.lower()
        assert "reasoning" in prompt.lower()

    @pytest.mark.asyncio
    async def test_build_prompt_with_context(self):
        """Test building prompt with context."""
        bridge = MagicMock(spec=LLMBridge)
        processor = ProcessLLMProcessor(bridge)

        request = ProcessingRequest(claim_id="c123")
        context = ContextResult(
            claim_id="c123",
            context_claims=[
                Claim(id="r1", content="Related", confidence=0.8)
            ],
            context_size=100,
        )

        prompt = await processor._build_evaluation_prompt(request, context)

        assert "Context" in prompt
        assert "Related claims: 1" in prompt

    @pytest.mark.asyncio
    async def test_build_prompt_with_instruction_types(self):
        """Test building prompt with instruction types."""
        bridge = MagicMock(spec=LLMBridge)
        processor = ProcessLLMProcessor(bridge)

        request = ProcessingRequest(
            claim_id="c123",
            instruction_types=[InstructionType.CREATE_CLAIM, InstructionType.ANALYZE_CLAIM],
        )

        prompt = await processor._build_evaluation_prompt(request, None)

        assert "create_claim" in prompt
        assert "analyze_claim" in prompt

    @pytest.mark.asyncio
    async def test_build_prompt_with_hints(self):
        """Test building prompt with context hints."""
        bridge = MagicMock(spec=LLMBridge)
        processor = ProcessLLMProcessor(bridge)

        request = ProcessingRequest(
            claim_id="c123",
            context_hints=["mathematics", "algebra"],
        )

        prompt = await processor._build_evaluation_prompt(request, None)

        assert "mathematics" in prompt
        assert "algebra" in prompt


class TestExtractEvaluationScore:
    """Tests for evaluation score extraction."""

    @pytest.mark.asyncio
    async def test_extract_valid_score(self):
        """Test extracting valid evaluation score."""
        bridge = MagicMock(spec=LLMBridge)
        processor = ProcessLLMProcessor(bridge)

        response = json.dumps({"evaluation_score": 0.85})

        score = await processor._extract_evaluation_score(response)

        assert score == 0.85

    @pytest.mark.asyncio
    async def test_extract_score_boundary_low(self):
        """Test extracting score at lower boundary."""
        bridge = MagicMock(spec=LLMBridge)
        processor = ProcessLLMProcessor(bridge)

        response = json.dumps({"evaluation_score": 0.0})

        score = await processor._extract_evaluation_score(response)

        assert score == 0.0

    @pytest.mark.asyncio
    async def test_extract_score_boundary_high(self):
        """Test extracting score at upper boundary."""
        bridge = MagicMock(spec=LLMBridge)
        processor = ProcessLLMProcessor(bridge)

        response = json.dumps({"evaluation_score": 1.0})

        score = await processor._extract_evaluation_score(response)

        assert score == 1.0

    @pytest.mark.asyncio
    async def test_extract_score_invalid_json(self):
        """Test extracting score from invalid JSON."""
        bridge = MagicMock(spec=LLMBridge)
        processor = ProcessLLMProcessor(bridge)

        response = "not valid json"

        score = await processor._extract_evaluation_score(response)

        assert score is None

    @pytest.mark.asyncio
    async def test_extract_score_out_of_range(self):
        """Test extracting out-of-range score returns None."""
        bridge = MagicMock(spec=LLMBridge)
        processor = ProcessLLMProcessor(bridge)

        response = json.dumps({"evaluation_score": 1.5})

        score = await processor._extract_evaluation_score(response)

        assert score is None

    @pytest.mark.asyncio
    async def test_extract_score_missing_key(self):
        """Test extracting score when key is missing."""
        bridge = MagicMock(spec=LLMBridge)
        processor = ProcessLLMProcessor(bridge)

        response = json.dumps({"other_key": "value"})

        score = await processor._extract_evaluation_score(response)

        assert score is None


class TestExtractReasoning:
    """Tests for reasoning extraction."""

    @pytest.mark.asyncio
    async def test_extract_valid_reasoning(self):
        """Test extracting valid reasoning."""
        bridge = MagicMock(spec=LLMBridge)
        processor = ProcessLLMProcessor(bridge)

        response = json.dumps({"reasoning": "This claim is well-supported."})

        reasoning = await processor._extract_reasoning(response)

        assert reasoning == "This claim is well-supported."

    @pytest.mark.asyncio
    async def test_extract_reasoning_invalid_json(self):
        """Test extracting reasoning from invalid JSON."""
        bridge = MagicMock(spec=LLMBridge)
        processor = ProcessLLMProcessor(bridge)

        response = "not valid json"

        reasoning = await processor._extract_reasoning(response)

        assert reasoning is None

    @pytest.mark.asyncio
    async def test_extract_reasoning_missing_key(self):
        """Test extracting reasoning when key is missing."""
        bridge = MagicMock(spec=LLMBridge)
        processor = ProcessLLMProcessor(bridge)

        response = json.dumps({"other_key": "value"})

        reasoning = await processor._extract_reasoning(response)

        assert reasoning is None

    @pytest.mark.asyncio
    async def test_extract_reasoning_non_string(self):
        """Test extracting reasoning when value is not string."""
        bridge = MagicMock(spec=LLMBridge)
        processor = ProcessLLMProcessor(bridge)

        response = json.dumps({"reasoning": 123})

        reasoning = await processor._extract_reasoning(response)

        assert reasoning is None


class TestParseInstructions:
    """Tests for instruction parsing."""

    @pytest.mark.asyncio
    async def test_parse_valid_instructions(self):
        """Test parsing valid instructions."""
        bridge = MagicMock(spec=LLMBridge)
        config = ProcessingConfig(instruction_confidence_threshold=0.5)
        processor = ProcessLLMProcessor(bridge, config)

        request = ProcessingRequest(claim_id="c123")
        response = json.dumps({
            "instructions": [
                {
                    "type": "create_claim",
                    "description": "Create new claim",
                    "confidence": 0.8,
                    "priority": 1,
                }
            ]
        })

        instructions = await processor._parse_instructions(response, request)

        assert len(instructions) == 1
        assert instructions[0].instruction_type == InstructionType.CREATE_CLAIM
        assert instructions[0].confidence == 0.8

    @pytest.mark.asyncio
    async def test_parse_instructions_below_threshold(self):
        """Test that instructions below threshold are filtered."""
        bridge = MagicMock(spec=LLMBridge)
        config = ProcessingConfig(instruction_confidence_threshold=0.9)
        processor = ProcessLLMProcessor(bridge, config)

        request = ProcessingRequest(claim_id="c123")
        response = json.dumps({
            "instructions": [
                {"type": "create_claim", "description": "Low conf", "confidence": 0.5},
            ]
        })

        instructions = await processor._parse_instructions(response, request)

        assert len(instructions) == 0

    @pytest.mark.asyncio
    async def test_parse_instructions_invalid_type(self):
        """Test parsing with invalid instruction type defaults to custom_action."""
        bridge = MagicMock(spec=LLMBridge)
        config = ProcessingConfig(instruction_confidence_threshold=0.0)
        processor = ProcessLLMProcessor(bridge, config)

        request = ProcessingRequest(claim_id="c123")
        response = json.dumps({
            "instructions": [
                {"type": "invalid_type", "description": "Test", "confidence": 0.8},
            ]
        })

        instructions = await processor._parse_instructions(response, request)

        assert len(instructions) == 1
        assert instructions[0].instruction_type == InstructionType.CUSTOM_ACTION

    @pytest.mark.asyncio
    async def test_parse_instructions_invalid_json(self):
        """Test parsing invalid JSON returns empty list."""
        bridge = MagicMock(spec=LLMBridge)
        processor = ProcessLLMProcessor(bridge)

        request = ProcessingRequest(claim_id="c123")

        instructions = await processor._parse_instructions("not json", request)

        assert instructions == []

    @pytest.mark.asyncio
    async def test_parse_instructions_empty_list(self):
        """Test parsing empty instructions list."""
        bridge = MagicMock(spec=LLMBridge)
        processor = ProcessLLMProcessor(bridge)

        request = ProcessingRequest(claim_id="c123")
        response = json.dumps({"instructions": []})

        instructions = await processor._parse_instructions(response, request)

        assert instructions == []


class TestProcessClaim:
    """Tests for claim processing."""

    @pytest_asyncio.fixture
    async def mock_bridge(self):
        """Create a mock LLM bridge."""
        bridge = AsyncMock(spec=LLMBridge)
        bridge.generate_response = AsyncMock(return_value=json.dumps({
            "evaluation_score": 0.85,
            "reasoning": "Well-supported claim",
            "instructions": []
        }))
        return bridge

    @pytest.mark.asyncio
    async def test_process_claim_success(self, mock_bridge):
        """Test successful claim processing."""
        processor = ProcessLLMProcessor(mock_bridge)

        request = ProcessingRequest(claim_id="c123")
        result = await processor.process_claim(request)

        assert isinstance(result, ProcessingResult)
        assert result.claim_id == "c123"
        assert result.status == ProcessingStatus.COMPLETED
        assert result.evaluation_score == 0.85

    @pytest.mark.asyncio
    async def test_process_claim_missing_id(self, mock_bridge):
        """Test processing claim with missing ID."""
        processor = ProcessLLMProcessor(mock_bridge)

        request = ProcessingRequest(claim_id="")
        result = await processor.process_claim(request)

        assert result.status == ProcessingStatus.FAILED
        assert "required" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_process_claim_caches_result(self, mock_bridge):
        """Test that successful processing caches result."""
        processor = ProcessLLMProcessor(mock_bridge)

        request = ProcessingRequest(claim_id="c123")
        await processor.process_claim(request)

        cache_key = processor._generate_cache_key(request)
        assert cache_key in processor._processing_cache

    @pytest.mark.asyncio
    async def test_process_claim_returns_cached(self, mock_bridge):
        """Test that cached result is returned."""
        processor = ProcessLLMProcessor(mock_bridge)

        request = ProcessingRequest(claim_id="c123")

        # First call
        result1 = await processor.process_claim(request)

        # Second call should return cached
        result2 = await processor.process_claim(request)

        # LLM should only be called once
        assert mock_bridge.generate_response.call_count == 1
        assert result1.claim_id == result2.claim_id

    @pytest.mark.asyncio
    async def test_process_claim_with_context(self, mock_bridge):
        """Test processing with context."""
        processor = ProcessLLMProcessor(mock_bridge)

        request = ProcessingRequest(claim_id="c123")
        context = ContextResult(
            claim_id="c123",
            context_claims=[
                Claim(id="r1", content="Related", confidence=0.8)
            ],
            context_size=100,
        )

        result = await processor.process_claim(request, context)

        assert result.status == ProcessingStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_process_claim_llm_failure(self):
        """Test handling LLM failure."""
        bridge = AsyncMock(spec=LLMBridge)
        bridge.generate_response = AsyncMock(
            side_effect=RuntimeError("LLM unavailable")
        )
        processor = ProcessLLMProcessor(bridge)

        request = ProcessingRequest(claim_id="c123")
        result = await processor.process_claim(request)

        assert result.status == ProcessingStatus.FAILED
        assert "LLM" in result.error_message


class TestProcessBatch:
    """Tests for batch processing."""

    @pytest_asyncio.fixture
    async def mock_bridge(self):
        """Create a mock LLM bridge."""
        bridge = AsyncMock(spec=LLMBridge)
        bridge.generate_response = AsyncMock(return_value=json.dumps({
            "evaluation_score": 0.85,
            "reasoning": "Well-supported",
            "instructions": []
        }))
        return bridge

    @pytest.mark.asyncio
    async def test_process_batch_sequential(self, mock_bridge):
        """Test batch processing with sequential execution."""
        config = ProcessingConfig(enable_parallel_processing=False)
        processor = ProcessLLMProcessor(mock_bridge, config)

        requests = [
            ProcessingRequest(claim_id="c1"),
            ProcessingRequest(claim_id="c2"),
        ]

        results = await processor.process_batch(requests)

        assert len(results) == 2
        assert all(isinstance(r, ProcessingResult) for r in results)

    @pytest.mark.asyncio
    async def test_process_batch_parallel(self, mock_bridge):
        """Test batch processing with parallel execution."""
        config = ProcessingConfig(enable_parallel_processing=True)
        processor = ProcessLLMProcessor(mock_bridge, config)

        requests = [
            ProcessingRequest(claim_id="c1"),
            ProcessingRequest(claim_id="c2"),
        ]

        results = await processor.process_batch(requests)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_process_batch_empty(self, mock_bridge):
        """Test batch processing with empty list."""
        processor = ProcessLLMProcessor(mock_bridge)

        results = await processor.process_batch([])

        assert results == []
