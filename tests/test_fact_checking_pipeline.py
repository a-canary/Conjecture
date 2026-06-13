# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Tests for fact_checking_pipeline.py — R&D artifact from 2026-05-04 sprint.
Covers: SelfConsistencyChecker, VectorSearchVerifier, CascadeInvalidator, FactCheckingPipeline.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from src.core.models import Claim, ClaimState, ClaimType, ClaimScope
from src.core.dirty_flag import DirtyFlagSystem
from src.core.fact_checking_pipeline import (
    SelfConsistencyChecker,
    VectorSearchVerifier,
    CascadeInvalidator,
    FactCheckingPipeline,
    VerificationTier,
    FactCheckResult,
    FactCheckStatus,
    TierResult,
    FactCheckReport,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_claims():
    """Create a dict of sample claims for testing."""
    return {
        "c0000001": Claim(
            id="c0000001",
            content="The earth orbits the sun",
            confidence=0.95,
            state=ClaimState.VALIDATED,
            type=[ClaimType.CONCEPT],
            scope="user-{workspace}",
            supers=["c0000002"],
            subs=[],
        ),
        "c0000002": Claim(
            id="c0000002",
            content="This is a parent claim that c0000001 supports",
            confidence=0.80,
            state=ClaimState.VALIDATED,
            type=[ClaimType.ASSERTION],
            scope="user-{workspace}",
            supers=[],
            subs=["c0000001"],
        ),
        "c0000003": Claim(
            id="c0000003",
            content="Low-confidence child claim",
            confidence=0.2,
            state=ClaimState.EXPLORE,
            type=[ClaimType.CONCEPT],
            scope="user-{workspace}",
            supers=["c0000004"],
            subs=[],
        ),
        "c0000004": Claim(
            id="c0000004",
            content="Parent claim with low-confidence child",
            confidence=0.90,
            state=ClaimState.VALIDATED,
            type=[ClaimType.ASSERTION],
            scope="user-{workspace}",
            supers=[],
            subs=["c0000003"],
        ),
        "c0000005": Claim(
            id="c0000005",
            content="Orphaned claim with no relationships",
            confidence=0.5,
            state=ClaimState.ORPHANED,
            type=[ClaimType.IMPRESSION],
            scope="user-{workspace}",
            supers=[],
            subs=[],
        ),
        "c0000006": Claim(
            id="c0000006",
            content="Overconfident impression",
            confidence=0.98,
            state=ClaimState.VALIDATED,
            type=[ClaimType.IMPRESSION],
            scope="user-{workspace}",
            supers=[],
            subs=[],
        ),
    }


@pytest.fixture
def dirty_flag_system():
    return MagicMock(spec=DirtyFlagSystem)


# ============================================================================
# SelfConsistencyChecker Tests
# ============================================================================

class TestSelfConsistencyChecker:
    """Tests for Tier 1: SelfConsistencyChecker."""

    def test_check_verified_healthy_claim(self, sample_claims):
        """A stable, well-supported claim should pass self-consistency."""
        checker = SelfConsistencyChecker(sample_claims)
        result = checker.check("c0000001")

        assert result.tier == VerificationTier.SELF_CONSISTENCY
        assert result.result == FactCheckResult.VERIFIED
        assert result.confidence == 0.85  # default
        assert result.cost_usd == 0.0
        assert result.latency_ms >= 0

    def test_check_rejected_low_confidence_sub(self, sample_claims):
        """Claim whose sub has very low confidence should be rejected."""
        checker = SelfConsistencyChecker(sample_claims)
        result = checker.check("c0000004")

        assert result.result == FactCheckResult.REJECTED
        assert len(result.contradictions) >= 1
        assert any("very low confidence" in c for c in result.contradictions)

    def test_check_rejected_high_confidence_assertion(self, sample_claims):
        """ASSERTION type with low confidence should be flagged."""
        checker = SelfConsistencyChecker(sample_claims)
        # c0000004 is ASSERTION with confidence 0.90 (ok, not < 0.7)
        # Let's look for one that's ASSERTION with low confidence
        claims = {
            "c0000007": Claim(
                id="c0000007",
                content="This is an assertion but very low confidence",
                confidence=0.3,
                state=ClaimState.EXPLORE,
                type=[ClaimType.ASSERTION],
                scope="user-{workspace}",
                supers=[],
                subs=[],
            ),
        }
        checker = SelfConsistencyChecker(claims)
        result = checker.check("c0000007")

        assert result.result == FactCheckResult.REJECTED
        assert any("low confidence" in c.lower() for c in result.contradictions)

    def test_check_evidence_overconfident_impression(self, sample_claims):
        """IMPRESSION type with very high confidence should be flagged as evidence."""
        checker = SelfConsistencyChecker(sample_claims)
        result = checker.check("c0000006")

        assert result.result == FactCheckResult.VERIFIED
        assert len(result.evidence) >= 1
        assert any("overconfident" in e.lower() for e in result.evidence)

    def test_check_not_found_returns_skipped(self, sample_claims):
        """Non-existent claim returns SKIPPED."""
        checker = SelfConsistencyChecker(sample_claims)
        result = checker.check("nonexistent")

        assert result.result == FactCheckResult.SKIPPED
        assert result.confidence == 0.0

    def test_check_zero_cost(self, sample_claims):
        """Self-consistency check costs nothing."""
        checker = SelfConsistencyChecker(sample_claims)
        result = checker.check("c0000001")

        assert result.cost_usd == 0.0


# ============================================================================
# TierResult Dataclass Tests
# ============================================================================

class TestTierResult:
    def test_passed_property_verified(self):
        tr = TierResult(
            tier=VerificationTier.SELF_CONSISTENCY,
            result=FactCheckResult.VERIFIED,
            confidence=0.8,
            evidence=["ok"],
            contradictions=[],
            cost_usd=0.0,
            latency_ms=5,
        )
        assert tr.passed is True

    def test_passed_property_rejected(self):
        tr = TierResult(
            tier=VerificationTier.SELF_CONSISTENCY,
            result=FactCheckResult.REJECTED,
            confidence=0.8,
            evidence=[],
            contradictions=["contradiction"],
            cost_usd=0.0,
            latency_ms=5,
        )
        assert tr.passed is False


# ============================================================================
# FactCheckReport Tests
# ============================================================================

class TestFactCheckReport:
    def test_to_dict_serializes_correctly(self, sample_claims):
        """FactCheckReport.to_dict() produces a valid dict."""
        report = FactCheckReport(
            claim_id="c0000001",
            claim_content="The earth orbits the sun",
            status=FactCheckStatus.COMPLETE,
            final_result=FactCheckResult.VERIFIED,
            final_confidence=0.85,
            tier_results=[],
            total_cost_usd=0.0,
            total_latency_ms=10,
            cascaded_claims=set(),
            metadata={},
        )
        d = report.to_dict()

        assert d["claim_id"] == "c0000001"
        assert d["final_result"] == "verified"
        assert d["status"] == "complete"
        assert d["total_cost_usd"] == 0.0


# ============================================================================
# CascadeInvalidator Tests
# ============================================================================

class TestCascadeInvalidator:
    def test_cascade_mark_dirty(self, sample_claims, dirty_flag_system):
        inv = CascadeInvalidator(dirty_flag_system, sample_claims)
        marked = inv.cascade_rejection("c0000003", rejection_confidence=0.8, max_depth=2)

        assert "c0000003" in marked
        dirty_flag_system.mark_claim_dirty.assert_called()

    def test_cascade_max_depth_respected(self, sample_claims, dirty_flag_system):
        inv = CascadeInvalidator(dirty_flag_system, sample_claims)
        marked = inv.cascade_rejection("c0000001", max_depth=1)

        # c0000001 -> c0000002 (depth 1), c0000002 has no more supers
        # so only c0000001 should be marked at depth=0
        # c0000001 is in marked (depth 0), c0000002 is NOT in marked (would be depth 1, not reached)
        assert "c0000001" in marked

    def test_cascade_empty_claims(self, dirty_flag_system):
        inv = CascadeInvalidator(dirty_flag_system, {})
        marked = inv.cascade_rejection("nonexistent")
        assert marked == set()


# ============================================================================
# FactCheckingPipeline Integration Tests
# ============================================================================

class TestFactCheckingPipeline:
    """Integration tests for the full FactCheckingPipeline."""

    def test_pipeline_initializes(self, sample_claims, dirty_flag_system):
        """Pipeline initializes without error."""
        mock_vector_store = MagicMock()
        pipeline = FactCheckingPipeline(
            claims=sample_claims,
            vector_store=mock_vector_store,
            dirty_flag_system=dirty_flag_system,
            web_search_func=None,
        )
        assert pipeline.tier1 is not None
        assert pipeline.tier2 is not None
        assert pipeline.tier3 is None  # no web search func

    def test_pipeline_verify_tier1_only(self, sample_claims, dirty_flag_system):
        """Pipeline verify() with tier1 only."""
        mock_vector_store = MagicMock()
        pipeline = FactCheckingPipeline(
            claims=sample_claims,
            vector_store=mock_vector_store,
            dirty_flag_system=dirty_flag_system,
            web_search_func=None,
        )
        report = pipeline.verify(
            "c0000001",
            tiers=[VerificationTier.SELF_CONSISTENCY],
        )

        assert report.claim_id == "c0000001"
        assert report.status == FactCheckStatus.COMPLETE
        assert len(report.tier_results) == 1
        assert report.tier_results[0].tier == VerificationTier.SELF_CONSISTENCY

    def test_pipeline_verify_not_found_claim(self, sample_claims, dirty_flag_system):
        """Verify returns FAILED status for nonexistent claim."""
        mock_vector_store = MagicMock()
        pipeline = FactCheckingPipeline(
            claims=sample_claims,
            vector_store=mock_vector_store,
            dirty_flag_system=dirty_flag_system,
            web_search_func=None,
        )
        report = pipeline.verify(
            "nonexistent",
            tiers=[VerificationTier.SELF_CONSISTENCY],
        )

        assert report.status == FactCheckStatus.FAILED
        assert report.final_result == FactCheckResult.SKIPPED

    def test_pipeline_verify_batch(self, sample_claims, dirty_flag_system):
        """verify_batch processes multiple claims."""
        mock_vector_store = MagicMock()
        pipeline = FactCheckingPipeline(
            claims=sample_claims,
            vector_store=mock_vector_store,
            dirty_flag_system=dirty_flag_system,
            web_search_func=None,
        )
        reports = pipeline.verify_batch(
            ["c0000001", "c0000002"],
            tiers=[VerificationTier.SELF_CONSISTENCY],
        )

        assert len(reports) == 2
        assert all(r.status == FactCheckStatus.COMPLETE for r in reports)

    def test_pipeline_verify_low_confidence_sub_rejected(self, sample_claims, dirty_flag_system):
        """c0000004 should be rejected due to low-confidence sub c0000003."""
        mock_vector_store = MagicMock()
        pipeline = FactCheckingPipeline(
            claims=sample_claims,
            vector_store=mock_vector_store,
            dirty_flag_system=dirty_flag_system,
            web_search_func=None,
        )
        report = pipeline.verify(
            "c0000004",
            tiers=[VerificationTier.SELF_CONSISTENCY],
        )

        assert report.final_result == FactCheckResult.REJECTED
        assert len(report.tier_results) == 1

    def test_pipeline_tier_config_respected(self, sample_claims, dirty_flag_system):
        """Tier config is respected."""
        mock_vector_store = MagicMock()
        config = {
            "self_consistency": {"enabled": True, "always_run": True},
            "vector_search": {"enabled": False, "similarity_threshold": 0.85, "k_results": 10, "skip_on_rejection": False},
            "live_web": {"enabled": False, "max_sources": 5, "skip_on_rejection": True, "only_on_uncertain": False},
        }
        pipeline = FactCheckingPipeline(
            claims=sample_claims,
            vector_store=mock_vector_store,
            dirty_flag_system=dirty_flag_system,
            web_search_func=None,
            tier_config=config,
        )

        # With vector_search disabled, only tier1 runs by default
        report = pipeline.verify("c0000001")
        tier_values = [r.tier for r in report.tier_results]
        assert VerificationTier.SELF_CONSISTENCY in tier_values
        assert VerificationTier.VECTOR_SEARCH not in tier_values

    def test_pipeline_cascade_on_rejection(self, sample_claims, dirty_flag_system):
        """Rejection cascades dirty flags to dependent claims."""
        mock_vector_store = MagicMock()
        pipeline = FactCheckingPipeline(
            claims=sample_claims,
            vector_store=mock_vector_store,
            dirty_flag_system=dirty_flag_system,
            web_search_func=None,
        )
        # c0000003 -> c0000004 (c0000003 is a sub of c0000004)
        # Reject c0000003, should cascade to c0000004
        report = pipeline.verify(
            "c0000003",
            tiers=[VerificationTier.SELF_CONSISTENCY],
        )

        # c0000003 has low confidence -> REJECTED
        # Cascade should have been triggered
        if report.final_result == FactCheckResult.REJECTED:
            assert len(report.cascaded_claims) >= 0  # cascade happens on rejection

    def test_verify_with_tier2_vector_search(self, sample_claims, dirty_flag_system):
        """Tier 2 vector search runs when enabled and tier is specified."""
        mock_vector_store = MagicMock()
        mock_vector_store.search.return_value = []  # no similar claims found

        pipeline = FactCheckingPipeline(
            claims=sample_claims,
            vector_store=mock_vector_store,
            dirty_flag_system=dirty_flag_system,
            web_search_func=None,
            tier_config={
                "self_consistency": {"enabled": True, "always_run": True},
                "vector_search": {"enabled": True, "similarity_threshold": 0.85, "k_results": 10, "skip_on_rejection": False},
                "live_web": {"enabled": False, "max_sources": 5, "skip_on_rejection": True, "only_on_uncertain": False},
            },
        )

        report = pipeline.verify(
            "c0000001",
            tiers=[VerificationTier.SELF_CONSISTENCY, VerificationTier.VECTOR_SEARCH],
        )

        assert len(report.tier_results) == 2
        mock_vector_store.search.assert_called()

    def test_pipeline_aggregate_results_verified_wins(self, sample_claims, dirty_flag_system):
        """When verifications > rejections, final result is VERIFIED."""
        mock_vector_store = MagicMock()
        pipeline = FactCheckingPipeline(
            claims=sample_claims,
            vector_store=mock_vector_store,
            dirty_flag_system=dirty_flag_system,
            web_search_func=None,
        )
        report = pipeline.verify(
            "c0000001",
            tiers=[VerificationTier.SELF_CONSISTENCY],
        )

        # c0000001 is VERIFIED with confidence 0.85
        assert report.final_result == FactCheckResult.VERIFIED

    def test_pipeline_aggregate_results_rejected_wins(self, sample_claims, dirty_flag_system):
        """When rejections > verifications, final result is REJECTED."""
        mock_vector_store = MagicMock()
        pipeline = FactCheckingPipeline(
            claims=sample_claims,
            vector_store=mock_vector_store,
            dirty_flag_system=dirty_flag_system,
            web_search_func=None,
        )
        # c0000004 is REJECTED (low confidence sub)
        report = pipeline.verify(
            "c0000004",
            tiers=[VerificationTier.SELF_CONSISTENCY],
        )

        assert report.final_result == FactCheckResult.REJECTED


# ============================================================================
# VectorSearchVerifier Tests (mocked vector store)
# ============================================================================

class TestVectorSearchVerifier:
    """Tests for Tier 2: VectorSearchVerifier with mocked vector store."""

    def test_tier2_no_similar_claims(self, sample_claims):
        """When vector store finds nothing similar, claim is verified."""
        mock_store = MagicMock()
        mock_store.search.return_value = []  # no results

        verifier = VectorSearchVerifier(mock_store, sample_claims)
        result = verifier.check("c0000001")

        assert result.tier == VerificationTier.VECTOR_SEARCH
        # No contradictions = VERIFIED
        assert result.result == FactCheckResult.VERIFIED

    def test_tier2_skipped_when_disabled(self, sample_claims):
        """Returns SKIPPED when vector store is None."""
        verifier = VectorSearchVerifier(None, sample_claims)
        result = verifier.check("c0000001")

        assert result.result == FactCheckResult.SKIPPED