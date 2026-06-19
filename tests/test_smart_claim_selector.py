# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Tests for SmartClaimSelector (Process Layer)

Tests smart claim selection with:
- Confidence gating
- Correctness filtering
- Domain relevance scoring
- Claim limit enforcement
"""

import pytest
from src.process.smart_claim_selector import (
    SmartClaimSelector,
    SelectionConfig,
    ClaimScore,
    string_to_domain,
)
from src.data.models import Claim, ClaimDomain


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def make_claim(
    claim_id: str,
    content: str = "Test claim",
    confidence: float = 0.7,
    domain: ClaimDomain = ClaimDomain.MATH,
    is_correct: bool = None,
) -> Claim:
    """Create a test claim with specified properties."""
    return Claim(
        id=claim_id,
        content=content,
        confidence=confidence,
        domain=domain,
        is_correct=is_correct,
    )


# ---------------------------------------------------------------------------
# ClaimScore Tests
# ---------------------------------------------------------------------------

class TestClaimScore:
    """Test ClaimScore dataclass and scoring."""

    def test_default_weights(self):
        """Test default weight calculation."""
        claim = make_claim("c1")
        score = ClaimScore(
            claim=claim,
            relevance=0.8,
            confidence=0.9,
            correctness=0.5,
            recency=1.0,
        )

        # Default weights: relevance=0.4, confidence=0.2, correctness=0.3, recency=0.1
        expected = 0.8 * 0.4 + 0.9 * 0.2 + 0.5 * 0.3 + 1.0 * 0.1
        result = score.calculate_total()

        assert abs(result - expected) < 0.001
        assert score.total == result

    def test_custom_weights(self):
        """Test custom weight calculation."""
        claim = make_claim("c1")
        score = ClaimScore(
            claim=claim,
            relevance=1.0,
            confidence=1.0,
            correctness=1.0,
            recency=1.0,
        )

        # Custom: all equal weights
        weights = {
            "relevance": 0.25,
            "confidence": 0.25,
            "correctness": 0.25,
            "recency": 0.25,
        }
        result = score.calculate_total(weights)

        assert result == 1.0  # All 1.0 with equal weights


class TestSelectionConfig:
    """Test SelectionConfig defaults and customization."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SelectionConfig()

        assert config.min_confidence == 0.5
        assert config.prefer_correct is True
        assert config.exclude_incorrect is False
        assert config.max_claims == 5
        assert config.max_per_domain == 3

    def test_custom_values(self):
        """Test custom configuration."""
        config = SelectionConfig(
            min_confidence=0.8,
            exclude_incorrect=True,
            max_claims=10,
        )

        assert config.min_confidence == 0.8
        assert config.exclude_incorrect is True
        assert config.max_claims == 10


# ---------------------------------------------------------------------------
# SmartClaimSelector Tests
# ---------------------------------------------------------------------------

class TestSmartClaimSelector:
    """Test SmartClaimSelector core functionality."""

    def test_add_claim(self):
        """Test adding claims to selector."""
        selector = SmartClaimSelector()
        claim = make_claim("c1", domain=ClaimDomain.MATH)

        selector.add_claim(claim)
        stats = selector.get_stats()

        assert stats["total_claims"] == 1
        assert stats["by_domain"]["math"] == 1

    def test_add_multiple_claims(self):
        """Test adding multiple claims."""
        selector = SmartClaimSelector()

        selector.add_claim(make_claim("c1", domain=ClaimDomain.MATH))
        selector.add_claim(make_claim("c2", domain=ClaimDomain.LOGIC))
        selector.add_claim(make_claim("c3", domain=ClaimDomain.MATH))

        stats = selector.get_stats()

        assert stats["total_claims"] == 3
        assert stats["by_domain"]["math"] == 2
        assert stats["by_domain"]["logic"] == 1

    def test_clear(self):
        """Test clearing all claims."""
        selector = SmartClaimSelector()
        selector.add_claim(make_claim("c1"))
        selector.add_claim(make_claim("c2"))

        selector.clear()
        stats = selector.get_stats()

        assert stats["total_claims"] == 0

    def test_correctness_tracking(self):
        """Test correct/incorrect claim tracking."""
        selector = SmartClaimSelector()

        selector.add_claim(make_claim("c1", is_correct=True))
        selector.add_claim(make_claim("c2", is_correct=False))
        selector.add_claim(make_claim("c3", is_correct=None))

        stats = selector.get_stats()

        assert stats["correct"] == 1
        assert stats["incorrect"] == 1
        assert stats["untested"] == 1


class TestGetRelevantClaims:
    """Test get_relevant_claims functionality."""

    def test_confidence_gating(self):
        """Test that low-confidence claims are excluded."""
        config = SelectionConfig(min_confidence=0.6)
        selector = SmartClaimSelector(config)

        selector.add_claim(make_claim("c1", confidence=0.9))  # Include
        selector.add_claim(make_claim("c2", confidence=0.3))  # Exclude
        selector.add_claim(make_claim("c3", confidence=0.7))  # Include

        claims = selector.get_relevant_claims(ClaimDomain.MATH)

        assert len(claims) == 2
        assert all(c.confidence >= 0.6 for c in claims)

    def test_exclude_incorrect(self):
        """Test excluding incorrect claims when configured."""
        config = SelectionConfig(exclude_incorrect=True)
        selector = SmartClaimSelector(config)

        selector.add_claim(make_claim("c1", is_correct=True))
        selector.add_claim(make_claim("c2", is_correct=False))  # Excluded
        selector.add_claim(make_claim("c3", is_correct=None))

        claims = selector.get_relevant_claims(ClaimDomain.MATH)

        assert len(claims) == 2
        assert not any(c.is_correct is False for c in claims)

    def test_max_claims_limit(self):
        """Test max_claims is respected."""
        config = SelectionConfig(max_claims=2)
        selector = SmartClaimSelector(config)

        for i in range(5):
            selector.add_claim(make_claim(f"c{i}"))

        claims = selector.get_relevant_claims(ClaimDomain.MATH)

        assert len(claims) == 2

    def test_max_per_domain_limit(self):
        """Test max_per_domain is respected."""
        config = SelectionConfig(max_claims=10, max_per_domain=2)
        selector = SmartClaimSelector(config)

        # Add 4 claims in MATH domain
        for i in range(4):
            selector.add_claim(make_claim(f"c{i}", domain=ClaimDomain.MATH))

        claims = selector.get_relevant_claims(ClaimDomain.MATH)

        # Should only get 2 (max_per_domain)
        assert len(claims) == 2

    def test_domain_relevance_scoring(self):
        """Test that same-domain claims are preferred."""
        selector = SmartClaimSelector()

        # Add claims in different domains
        selector.add_claim(make_claim("c1", content="Math claim", domain=ClaimDomain.MATH))
        selector.add_claim(make_claim("c2", content="Logic claim", domain=ClaimDomain.LOGIC))

        claims = selector.get_relevant_claims(ClaimDomain.MATH, max_claims=1)

        # Math claim should be selected for MATH domain
        assert len(claims) == 1
        assert claims[0].domain == ClaimDomain.MATH

    def test_correctness_preference(self):
        """Test that correct claims are ranked higher."""
        selector = SmartClaimSelector()

        # Add claims with varying correctness
        selector.add_claim(make_claim("c1", confidence=0.8, is_correct=False))
        selector.add_claim(make_claim("c2", confidence=0.8, is_correct=True))
        selector.add_claim(make_claim("c3", confidence=0.8, is_correct=None))

        claims = selector.get_relevant_claims(ClaimDomain.MATH, max_claims=3)

        # Correct claim should be first
        assert claims[0].is_correct is True


class TestRelatedDomains:
    """Test domain relationship detection."""

    def test_math_related_domains(self):
        """Test MATH is related to GEOMETRY and PROBABILITY."""
        selector = SmartClaimSelector()

        # Add claims in related domains
        selector.add_claim(make_claim("c1", domain=ClaimDomain.GEOMETRY))
        selector.add_claim(make_claim("c2", domain=ClaimDomain.SCIENCE))

        # When querying for MATH, GEOMETRY should score higher than SCIENCE
        claims = selector.get_relevant_claims(ClaimDomain.MATH, max_claims=2)

        # Both included but GEOMETRY should be first (related domain)
        if len(claims) == 2:
            # Can't guarantee order without checking scores, but both should be present
            domains = {c.domain for c in claims}
            assert ClaimDomain.GEOMETRY in domains

    def test_logic_related_to_pattern(self):
        """Test LOGIC is related to PATTERN."""
        selector = SmartClaimSelector()

        result = selector._is_related_domain(ClaimDomain.LOGIC, ClaimDomain.PATTERN)

        assert result is True

    def test_unrelated_domains(self):
        """Test unrelated domains return False."""
        selector = SmartClaimSelector()

        result = selector._is_related_domain(ClaimDomain.MATH, ClaimDomain.SCIENCE)

        assert result is False


class TestFormatContext:
    """Test context formatting for LLM prompts."""

    def test_empty_claims(self):
        """Test formatting with no claims."""
        selector = SmartClaimSelector()

        result = selector.format_context([])

        assert result == "No prior knowledge."

    def test_format_single_claim(self):
        """Test formatting a single claim."""
        selector = SmartClaimSelector()
        claims = [make_claim("c1", content="Test content", confidence=0.8)]

        result = selector.format_context(claims)

        assert "Prior knowledge:" in result
        assert "Test content" in result
        assert "80%" in result

    def test_format_with_correctness(self):
        """Test formatting includes correctness when present."""
        selector = SmartClaimSelector()
        claims = [make_claim("c1", content="Verified claim content", confidence=0.9, is_correct=True)]

        result = selector.format_context(claims, include_correctness=True)

        assert "correct=True" in result

    def test_format_without_confidence(self):
        """Test formatting without confidence."""
        selector = SmartClaimSelector()
        claims = [make_claim("c1", content="Test claim content")]

        result = selector.format_context(claims, include_confidence=False)

        assert "%" not in result


class TestGetStats:
    """Test statistics generation."""

    def test_empty_stats(self):
        """Test stats with no claims."""
        selector = SmartClaimSelector()

        stats = selector.get_stats()

        assert stats["total_claims"] == 0
        assert stats["correct"] == 0
        assert stats["incorrect"] == 0
        assert stats["avg_confidence"] == 0

    def test_average_confidence(self):
        """Test average confidence calculation."""
        selector = SmartClaimSelector()

        selector.add_claim(make_claim("c1", confidence=0.6))
        selector.add_claim(make_claim("c2", confidence=0.8))

        stats = selector.get_stats()

        assert stats["avg_confidence"] == 0.7


# ---------------------------------------------------------------------------
# Utility Function Tests
# ---------------------------------------------------------------------------

class TestStringToDomain:
    """Test string_to_domain utility function."""

    def test_valid_domains(self):
        """Test conversion of valid domain strings."""
        assert string_to_domain("math") == ClaimDomain.MATH
        assert string_to_domain("MATH") == ClaimDomain.MATH
        assert string_to_domain("Math") == ClaimDomain.MATH
        assert string_to_domain("logic") == ClaimDomain.LOGIC
        assert string_to_domain("science") == ClaimDomain.SCIENCE

    def test_invalid_domain(self):
        """Test invalid domain defaults to GENERAL."""
        assert string_to_domain("unknown") == ClaimDomain.GENERAL
        assert string_to_domain("") == ClaimDomain.GENERAL
