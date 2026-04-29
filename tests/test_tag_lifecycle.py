"""Tests for D-0004: Tag Lifecycle Management.

Tag Lifecycle:
- Split trigger: tag >20% usage (fraction of total claims)
- Process: sample ≤100 claims → LLM suggests ≤8 replacement tags
  → batch claims (20) → LLM assigns replacement per claim
- If total claims >500 → merge similar tags
- Max 20 tags per claim
"""

import pytest
from unittest.mock import AsyncMock
from src.data.models import Claim, ClaimState, ClaimType
from src.data.repositories import ClaimRepository
from src.process.tag_lifecycle import (
    TagLifecycleManager,
    TagUsageStats,
    SplitTriggerResult,
)


@pytest.fixture
def mock_llm_processor():
    """Mock LLM processor for tag suggestion."""
    processor = AsyncMock()
    processor.call = AsyncMock(return_value='["machine-learning", "nlp", "transformers"]')
    return processor


@pytest.fixture
def sample_claims():
    """Create sample claims with varying tag distributions."""
    def make(tags, cid):
        return Claim(
            id=cid,
            content=f"Test claim {cid}",
            confidence=0.8,
            state=ClaimState.EXPLORE,
            type=[ClaimType.CONJECTURE],
            tags=list(tags),
            supers=[],
            subs=[],
        )
    return [
        make(("ml",), "c1"),
        make(("ml",), "c2"),
        make(("ml",), "c3"),
        make(("ml",), "c4"),
        make(("ml",), "c5"),
        # ml is 5/8 = 62.5% — triggers split (>20%)
        make(("nlp",), "c6"),
        make(("reasoning",), "c7"),
        make(("reasoning",), "c8"),
        # reasoning is 2/8 = 25% — also triggers split
    ]


@pytest.fixture
def mock_repository(sample_claims):
    """Mock repository with sample claims."""
    repo = AsyncMock(spec=ClaimRepository)
    repo.list_all = AsyncMock(return_value=sample_claims)
    repo.get_by_id = AsyncMock(side_effect=lambda cid: next(
        (c for c in sample_claims if c.id == cid), None
    ))
    return repo


class TestTagUsageStats:
    """Test tag usage statistics computation."""

    @pytest.mark.asyncio
    async def test_compute_usage_stats_single_tag(self, mock_repository):
        """High-usage tag shows correct percentage."""
        stats = await TagUsageStats.compute(mock_repository)

        ml_stat = stats.get_stat("ml")
        assert ml_stat is not None
        assert ml_stat.claim_count == 5
        assert ml_stat.total_claims == 8
        assert abs(ml_stat.usage_fraction - 0.625) < 0.01

    @pytest.mark.asyncio
    async def test_compute_usage_stats_low_usage_tag(self, mock_repository):
        """Low-usage tag shows correct fraction."""
        stats = await TagUsageStats.compute(mock_repository)

        nlp_stat = stats.get_stat("nlp")
        assert nlp_stat is not None
        assert nlp_stat.claim_count == 1
        assert nlp_stat.usage_fraction == 0.125  # 1/8

    @pytest.mark.asyncio
    async def test_compute_usage_stats_missing_tag(self, mock_repository):
        """Non-existent tag returns zero count."""
        stats = await TagUsageStats.compute(mock_repository)

        assert stats.get_stat("nonexistent") is None

    @pytest.mark.asyncio
    async def test_total_claims_count(self, mock_repository):
        """Total claims is correctly counted."""
        stats = await TagUsageStats.compute(mock_repository)
        assert stats.total_claims == 8


class TestSplitTriggerDetection:
    """Test split trigger detection (tag >20% usage)."""

    @pytest.mark.asyncio
    async def test_detects_split_trigger_high_usage(self, mock_repository):
        """Tag >20% triggers split."""
        result = await SplitTriggerResult.detect(mock_repository, threshold=0.20)

        trigger_tags = result.triggered_tags
        assert "ml" in trigger_tags  # 62.5% > 20%
        assert "reasoning" in trigger_tags  # 25% > 20%

    @pytest.mark.asyncio
    async def test_no_trigger_below_threshold(self, mock_repository):
        """Tag ≤20% does not trigger split."""
        result = await SplitTriggerResult.detect(mock_repository, threshold=0.20)

        assert "nlp" not in result.triggered_tags  # 12.5% < 20%

    @pytest.mark.asyncio
    async def test_empty_result_no_triggers(self):
        """No triggers when all tags below threshold."""
        repo = AsyncMock(spec=ClaimRepository)
        # 6 unique tags = 1/6 ≈ 16.7% each < 20% threshold → no triggers
        repo.list_all = AsyncMock(return_value=[
            Claim(id=f"c{i:08d}", content=f"rare-tag-{i}-claim-content", confidence=0.8,
                  state=ClaimState.EXPLORE, type=[ClaimType.CONCEPT], tags=[f"rare-tag-{i}"],
                  supers=[], subs=[])
            for i in range(1, 7)
        ])

        result = await SplitTriggerResult.detect(repo, threshold=0.20)
        assert len(result.triggered_tags) == 0

    @pytest.mark.asyncio
    async def test_trigger_result_contains_stats(self, mock_repository):
        """Trigger result includes usage statistics."""
        result = await SplitTriggerResult.detect(mock_repository, threshold=0.20)

        assert "ml" in result.usage_stats.tag_stats
        ml_stat = result.usage_stats.get_stat("ml")
        assert ml_stat.usage_fraction > 0.20


class TestTagLifecycleManager:
    """Test tag lifecycle manager end-to-end."""

    @pytest.mark.asyncio
    async def test_check_and_suggest_replacements(self, mock_repository, mock_llm_processor):
        """Manager detects overused tag and triggers replacement suggestion."""
        manager = TagLifecycleManager(
            repository=mock_repository,
            llm_processor=mock_llm_processor,
            split_threshold=0.20,
        )

        result = await manager.check_and_suggest(mock_repository)

        assert result.split_needed is True
        assert "ml" in result.split_tags
        # LLM was called for each overused tag (ml=62.5%, reasoning=25%)
        assert mock_llm_processor.call.call_count >= 1

    @pytest.mark.asyncio
    async def test_no_split_when_under_threshold(self, mock_repository, mock_llm_processor):
        """No split suggestion when all tags under threshold."""
        repo = AsyncMock(spec=ClaimRepository)
        # 6 unique tags = 1/6 ≈ 16.7% each < 20% threshold → no split
        repo.list_all = AsyncMock(return_value=[
            Claim(id=f"c{i:08d}", content=f"unique-tag-{i}-claim-content", confidence=0.8,
                  state=ClaimState.EXPLORE, type=[ClaimType.CONCEPT], tags=[f"unique-tag-{i}"],
                  supers=[], subs=[])
            for i in range(1, 7)
        ])

        manager = TagLifecycleManager(
            repository=repo,
            llm_processor=mock_llm_processor,
            split_threshold=0.20,
        )

        result = await manager.check_and_suggest(repo)
        assert result.split_needed is False

    @pytest.mark.asyncio
    async def test_batch_replacement(self, mock_llm_processor):
        """Batch replacement processes claims in groups of 20 and removes overused tag."""
        # Create 25 claims with the overused "ml" tag
        claims = []
        for i in range(25):
            claims.append(Claim(
                id=f"c{i:08d}",
                content=f"ML claim content {i:03d}",
                confidence=0.8,
                state=ClaimState.EXPLORE,
                type=[ClaimType.CONJECTURE],
                tags=["ml"],
                supers=[],
                subs=[],
            ))

        repo = AsyncMock(spec=ClaimRepository)
        repo.list_all = AsyncMock(return_value=claims)

        updated_claims = []
        async def capture_update(claim):
            updated_claims.append(claim)
        repo.update = capture_update

        manager = TagLifecycleManager(
            repository=repo,
            llm_processor=mock_llm_processor,
            split_threshold=0.20,
        )

        result = await manager.check_and_suggest(repo)

        # Should process all 25 claims
        assert len(updated_claims) == 25
        # Each claim should NOT have "ml" anymore (replaced with suggested tags)
        for claim in updated_claims:
            assert "ml" not in claim.tags


class TestTagMergeOnLargeDataset:
    """Test D-0004 merge trigger when total claims >500."""

    @pytest.mark.asyncio
    async def test_merge_trigger_over_500_claims(self):
        """When claims >500, merge_needed flag is set."""
        claims = []
        for i in range(510):
            # Two similar but different tags
            tag = f"ml-variant-{i % 2}"
            claims.append(Claim(
                id=f"c{i:08d}",
                content=f"ML claim content {i:03d}",
                confidence=0.8,
                state=ClaimState.EXPLORE,
                type=[ClaimType.CONJECTURE],
                tags=[tag],
                supers=[],
                subs=[],
            ))

        repo = AsyncMock(spec=ClaimRepository)
        repo.list_all = AsyncMock(return_value=claims)

        mock_llm = AsyncMock()
        mock_llm.call = AsyncMock(return_value='["ml-consolidated"]')

        manager = TagLifecycleManager(
            repository=repo,
            llm_processor=mock_llm,
            split_threshold=0.20,
        )

        result = await manager.check_and_suggest(repo)

        # Over 500 claims should trigger merge consideration
        assert result.merge_needed is True
        assert result.total_claims == 510
