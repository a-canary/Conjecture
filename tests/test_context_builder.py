"""
Tests for Process Context Builder

Tests the ProcessContextBuilder class which traverses claim graphs
and builds processing contexts for claim evaluation.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

from src.process.context_builder import ProcessContextBuilder
from src.process.models import ProcessingConfig, ContextResult
from src.data.repositories import ClaimRepository
from src.core.models import Claim, ClaimType, ClaimScope, ClaimState


class TestProcessContextBuilderInit:
    """Tests for ProcessContextBuilder initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default config."""
        repo = ClaimRepository()
        builder = ProcessContextBuilder(repo)

        assert builder.claim_repository == repo
        assert builder.config is not None
        assert isinstance(builder.config, ProcessingConfig)
        assert builder._context_cache == {}

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        repo = ClaimRepository()
        config = ProcessingConfig(
            max_context_size=20,
            max_traversal_depth=5,
        )
        builder = ProcessContextBuilder(repo, config)

        assert builder.config.max_context_size == 20
        assert builder.config.max_traversal_depth == 5


class TestEstimateContextSize:
    """Tests for context size estimation."""

    def test_estimate_context_size_empty(self):
        """Test estimation with empty list."""
        repo = ClaimRepository()
        builder = ProcessContextBuilder(repo)

        size = builder._estimate_context_size([])
        assert size == 0

    def test_estimate_context_size_single_claim(self):
        """Test estimation with single claim."""
        repo = ClaimRepository()
        builder = ProcessContextBuilder(repo)

        claim = Claim(
            id="c001",
            content="This is a test claim with some content.",
            confidence=0.8,
        )

        size = builder._estimate_context_size([claim])
        # ~40 chars / 4 = ~10 tokens
        assert size == len(claim.content) // 4

    def test_estimate_context_size_multiple_claims(self):
        """Test estimation with multiple claims."""
        repo = ClaimRepository()
        builder = ProcessContextBuilder(repo)

        claims = [
            Claim(id="c001", content="Short claim", confidence=0.8),
            Claim(id="c002", content="Another claim with more content here", confidence=0.7),
            Claim(id="c003", content="Third claim", confidence=0.9),
        ]

        total_chars = sum(len(c.content) for c in claims)
        expected_size = total_chars // 4

        size = builder._estimate_context_size(claims)
        assert size == expected_size


class TestCacheOperations:
    """Tests for context cache operations."""

    def test_clear_cache(self):
        """Test clearing the context cache."""
        repo = ClaimRepository()
        builder = ProcessContextBuilder(repo)

        # Add some fake cache entries
        builder._context_cache["c001"] = MagicMock()
        builder._context_cache["c002"] = MagicMock()

        assert len(builder._context_cache) == 2

        builder.clear_cache()

        assert len(builder._context_cache) == 0

    def test_get_cache_stats_empty(self):
        """Test cache stats with empty cache."""
        repo = ClaimRepository()
        builder = ProcessContextBuilder(repo)

        stats = builder.get_cache_stats()

        assert stats["cached_contexts"] == 0
        assert stats["cache_keys"] == []

    def test_get_cache_stats_with_entries(self):
        """Test cache stats with cached entries."""
        repo = ClaimRepository()
        builder = ProcessContextBuilder(repo)

        # Add fake cache entries
        builder._context_cache["c001"] = MagicMock()
        builder._context_cache["c002"] = MagicMock()

        stats = builder.get_cache_stats()

        assert stats["cached_contexts"] == 2
        assert set(stats["cache_keys"]) == {"c001", "c002"}


class TestBuildContext:
    """Tests for context building."""

    @pytest_asyncio.fixture
    async def mock_repo(self):
        """Create a mock claim repository."""
        repo = AsyncMock(spec=ClaimRepository)
        return repo

    @pytest.mark.asyncio
    async def test_build_context_claim_not_found(self, mock_repo):
        """Test build_context when claim is not found."""
        mock_repo.get_by_id = AsyncMock(return_value=None)

        builder = ProcessContextBuilder(mock_repo)

        with pytest.raises(RuntimeError, match="Context building failed"):
            await builder.build_context("nonexistent")

    @pytest.mark.asyncio
    async def test_build_context_success(self, mock_repo):
        """Test successful context building."""
        primary_claim = Claim(
            id="c001",
            content="Primary claim content",
            confidence=0.8,
            supers=[],
            subs=[],
        )

        mock_repo.get_by_id = AsyncMock(return_value=primary_claim)

        builder = ProcessContextBuilder(mock_repo)
        result = await builder.build_context("c001")

        assert isinstance(result, ContextResult)
        assert result.claim_id == "c001"
        assert len(result.context_claims) >= 1
        assert result.context_claims[0].id == "c001"
        assert result.build_time_ms >= 0

    @pytest.mark.asyncio
    async def test_build_context_caches_result(self, mock_repo):
        """Test that build_context caches the result."""
        primary_claim = Claim(
            id="c001",
            content="Primary claim content",
            confidence=0.8,
            supers=[],
            subs=[],
        )

        mock_repo.get_by_id = AsyncMock(return_value=primary_claim)

        builder = ProcessContextBuilder(mock_repo)
        result = await builder.build_context("c001")

        assert "c001" in builder._context_cache
        assert builder._context_cache["c001"] == result

    @pytest.mark.asyncio
    async def test_build_context_with_custom_depth(self, mock_repo):
        """Test build_context with custom max_depth."""
        primary_claim = Claim(
            id="c001",
            content="Primary claim",
            confidence=0.8,
            supers=[],
            subs=[],
        )

        mock_repo.get_by_id = AsyncMock(return_value=primary_claim)

        builder = ProcessContextBuilder(mock_repo)
        result = await builder.build_context("c001", max_depth=5)

        assert result.traversal_depth == 5

    @pytest.mark.asyncio
    async def test_build_context_with_hints(self, mock_repo):
        """Test build_context with context hints."""
        primary_claim = Claim(
            id="c001",
            content="Primary claim",
            confidence=0.8,
            supers=[],
            subs=[],
        )

        mock_repo.get_by_id = AsyncMock(return_value=primary_claim)

        builder = ProcessContextBuilder(mock_repo)
        result = await builder.build_context(
            "c001",
            context_hints=["mathematics", "algebra"]
        )

        assert result.metadata["hints_used"] == ["mathematics", "algebra"]


class TestTraverseClaimGraph:
    """Tests for claim graph traversal."""

    @pytest_asyncio.fixture
    async def mock_repo(self):
        """Create a mock claim repository."""
        repo = AsyncMock(spec=ClaimRepository)
        return repo

    @pytest.mark.asyncio
    async def test_traverse_single_claim_no_relations(self, mock_repo):
        """Test traversal with single claim and no relationships."""
        primary_claim = Claim(
            id="c001",
            content="Primary claim",
            confidence=0.8,
            supers=[],
            subs=[],
        )

        mock_repo.get_by_id = AsyncMock(return_value=None)

        builder = ProcessContextBuilder(mock_repo)
        result = await builder._traverse_claim_graph(
            primary_claim, max_depth=3, context_hints=[]
        )

        assert len(result) == 1
        assert result[0].id == "c001"

    @pytest.mark.asyncio
    async def test_traverse_with_supers(self, mock_repo):
        """Test traversal with super claims."""
        primary_claim = Claim(
            id="c001",
            content="Primary claim",
            confidence=0.8,
            supers=["c002"],
            subs=[],
        )

        super_claim = Claim(
            id="c002",
            content="Super claim",
            confidence=0.9,
            supers=[],
            subs=["c001"],
        )

        mock_repo.get_by_id = AsyncMock(side_effect=lambda cid:
            super_claim if cid == "c002" else None
        )

        builder = ProcessContextBuilder(mock_repo)
        result = await builder._traverse_claim_graph(
            primary_claim, max_depth=3, context_hints=[]
        )

        assert len(result) == 2
        assert result[0].id == "c001"
        assert result[1].id == "c002"

    @pytest.mark.asyncio
    async def test_traverse_respects_max_context_size(self, mock_repo):
        """Test that traversal respects max_context_size."""
        # Create a chain of claims
        claims = {}
        for i in range(10):
            claims[f"c{i:03d}"] = Claim(
                id=f"c{i:03d}",
                content=f"Claim {i}",
                confidence=0.8,
                supers=[f"c{i+1:03d}"] if i < 9 else [],
                subs=[f"c{i-1:03d}"] if i > 0 else [],
            )

        mock_repo.get_by_id = AsyncMock(
            side_effect=lambda cid: claims.get(cid)
        )

        config = ProcessingConfig(max_context_size=5, max_traversal_depth=10)
        builder = ProcessContextBuilder(mock_repo, config)

        result = await builder._traverse_claim_graph(
            claims["c000"], max_depth=10, context_hints=[]
        )

        # Should stop at max_context_size
        assert len(result) <= 5


class TestGetRelatedClaims:
    """Tests for getting related claims."""

    @pytest_asyncio.fixture
    async def mock_repo(self):
        """Create a mock claim repository."""
        repo = AsyncMock(spec=ClaimRepository)
        return repo

    @pytest.mark.asyncio
    async def test_get_related_claims_empty(self, mock_repo):
        """Test getting related claims when none exist."""
        claim = Claim(
            id="c001",
            content="Claim content",
            confidence=0.8,
            supers=[],
            subs=[],
        )

        mock_repo.get_by_id = AsyncMock(return_value=None)

        builder = ProcessContextBuilder(mock_repo)
        related = await builder._get_related_claims(claim, [])

        assert related == []

    @pytest.mark.asyncio
    async def test_get_related_claims_from_supers(self, mock_repo):
        """Test getting related claims from supers."""
        claim = Claim(
            id="c001",
            content="Claim content",
            confidence=0.8,
            supers=["c002"],
            subs=[],
        )

        super_claim = Claim(
            id="c002",
            content="Super claim",
            confidence=0.9,
        )

        mock_repo.get_by_id = AsyncMock(
            side_effect=lambda cid: super_claim if cid == "c002" else None
        )

        builder = ProcessContextBuilder(mock_repo)
        related = await builder._get_related_claims(claim, [])

        assert len(related) == 1
        assert related[0].id == "c002"

    @pytest.mark.asyncio
    async def test_get_related_claims_from_subs(self, mock_repo):
        """Test getting related claims from subs."""
        claim = Claim(
            id="c001",
            content="Claim content",
            confidence=0.8,
            supers=[],
            subs=["c003"],
        )

        sub_claim = Claim(
            id="c003",
            content="Sub claim",
            confidence=0.7,
        )

        mock_repo.get_by_id = AsyncMock(
            side_effect=lambda cid: sub_claim if cid == "c003" else None
        )

        builder = ProcessContextBuilder(mock_repo)
        related = await builder._get_related_claims(claim, [])

        assert len(related) == 1
        assert related[0].id == "c003"


class TestBuildBatchContexts:
    """Tests for batch context building."""

    @pytest_asyncio.fixture
    async def mock_repo(self):
        """Create a mock claim repository."""
        repo = AsyncMock(spec=ClaimRepository)
        return repo

    @pytest.mark.asyncio
    async def test_build_batch_contexts_sequential(self, mock_repo):
        """Test batch context building with sequential processing."""
        claims = {
            "c001": Claim(id="c001", content="Claim 1", confidence=0.8, supers=[], subs=[]),
            "c002": Claim(id="c002", content="Claim 2", confidence=0.9, supers=[], subs=[]),
        }

        mock_repo.get_by_id = AsyncMock(
            side_effect=lambda cid: claims.get(cid)
        )

        config = ProcessingConfig(enable_parallel_processing=False)
        builder = ProcessContextBuilder(mock_repo, config)

        results = await builder.build_batch_contexts(["c001", "c002"])

        assert len(results) == 2
        assert all(isinstance(r, ContextResult) for r in results)

    @pytest.mark.asyncio
    async def test_build_batch_contexts_parallel(self, mock_repo):
        """Test batch context building with parallel processing."""
        claims = {
            "c001": Claim(id="c001", content="Claim 1", confidence=0.8, supers=[], subs=[]),
            "c002": Claim(id="c002", content="Claim 2", confidence=0.9, supers=[], subs=[]),
        }

        mock_repo.get_by_id = AsyncMock(
            side_effect=lambda cid: claims.get(cid)
        )

        config = ProcessingConfig(enable_parallel_processing=True)
        builder = ProcessContextBuilder(mock_repo, config)

        results = await builder.build_batch_contexts(["c001", "c002"])

        assert len(results) == 2
