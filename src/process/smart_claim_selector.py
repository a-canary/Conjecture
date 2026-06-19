# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Smart Claim Selector (Phase 3: Smart Claim Selection)

Implements intelligent claim filtering to reduce context pollution:
- Domain tagging (3.1): Filter claims by knowledge domain
- Confidence gating (3.3): Exclude claims below threshold
- Correctness tracking (3.4): Prefer claims that led to correct answers
- Relevance filtering (3.5): Combine filters for optimal context

Goal: Make accumulated claims outperform fresh claims by filtering noise.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from src.data.models import Claim, ClaimDomain, ClaimFilter


@dataclass
class ClaimScore:
    """Score for claim relevance ranking."""
    claim: Claim
    relevance: float = 0.0  # Domain match score (0-1)
    confidence: float = 0.0  # Claim confidence (0-1)
    correctness: float = 0.0  # Correctness bonus (0-1)
    recency: float = 0.0  # Recency bonus (0-1)
    total: float = 0.0  # Weighted total

    def calculate_total(
        self,
        weights: Dict[str, float] = None
    ) -> float:
        """Calculate weighted total score."""
        weights = weights or {
            "relevance": 0.4,
            "confidence": 0.2,
            "correctness": 0.3,
            "recency": 0.1
        }
        self.total = (
            self.relevance * weights["relevance"] +
            self.confidence * weights["confidence"] +
            self.correctness * weights["correctness"] +
            self.recency * weights["recency"]
        )
        return self.total


@dataclass
class SelectionConfig:
    """Configuration for smart claim selection."""
    # Confidence gating (Step 3.3)
    min_confidence: float = 0.5  # Exclude claims below 50%

    # Correctness filtering (Step 3.4)
    prefer_correct: bool = True  # Rank correct claims higher
    exclude_incorrect: bool = False  # Whether to exclude incorrect claims

    # Domain filtering (Step 3.1)
    domain_match_weight: float = 0.4  # Weight for same-domain claims
    cross_domain_penalty: float = 0.5  # Penalty for different domain

    # Limits
    max_claims: int = 5  # Maximum claims to return
    max_per_domain: int = 3  # Maximum claims per domain

    # Recency
    recency_decay: float = 0.1  # Decay per older claim index


class SmartClaimSelector:
    """
    Intelligent claim selector that filters claims by domain, confidence,
    and correctness to reduce context pollution.

    Phase 3 Goal: Make accumulated claims outperform fresh claims (68% → 72%+).
    """

    def __init__(self, config: Optional[SelectionConfig] = None):
        self.config = config or SelectionConfig()
        self._claims: List[Claim] = []
        self._domain_index: Dict[ClaimDomain, List[Claim]] = defaultdict(list)
        self._correct_claims: List[Claim] = []
        self._incorrect_claims: List[Claim] = []

    def add_claim(self, claim: Claim) -> None:
        """Add a claim to the selector's memory."""
        self._claims.append(claim)
        self._domain_index[claim.domain].append(claim)

        if claim.is_correct is True:
            self._correct_claims.append(claim)
        elif claim.is_correct is False:
            self._incorrect_claims.append(claim)

    def clear(self) -> None:
        """Clear all claims from memory."""
        self._claims.clear()
        self._domain_index.clear()
        self._correct_claims.clear()
        self._incorrect_claims.clear()

    def get_relevant_claims(
        self,
        target_domain: ClaimDomain,
        max_claims: Optional[int] = None
    ) -> List[Claim]:
        """
        Get relevant claims for a target domain using smart selection.

        Implements:
        - Confidence gating (3.3): Exclude claims below threshold
        - Correctness preference (3.4): Rank correct claims higher
        - Domain filtering (3.1): Prioritize same-domain claims

        Args:
            target_domain: The domain of the current question
            max_claims: Override for max claims to return

        Returns:
            List of relevant claims, sorted by relevance score
        """
        max_claims = max_claims or self.config.max_claims

        # Score all claims
        scored_claims: List[ClaimScore] = []

        for i, claim in enumerate(self._claims):
            # Skip low-confidence claims (Step 3.3)
            if claim.confidence < self.config.min_confidence:
                continue

            # Skip incorrect claims if configured (Step 3.4)
            if self.config.exclude_incorrect and claim.is_correct is False:
                continue

            score = self._score_claim(claim, target_domain, i)
            scored_claims.append(score)

        # Sort by total score descending
        scored_claims.sort(key=lambda s: s.total, reverse=True)

        # Apply domain limits
        selected: List[Claim] = []
        domain_counts: Dict[ClaimDomain, int] = defaultdict(int)

        for score in scored_claims:
            if len(selected) >= max_claims:
                break

            domain = score.claim.domain
            if domain_counts[domain] < self.config.max_per_domain:
                selected.append(score.claim)
                domain_counts[domain] += 1

        return selected

    def _score_claim(
        self,
        claim: Claim,
        target_domain: ClaimDomain,
        index: int
    ) -> ClaimScore:
        """Score a claim for relevance to target domain."""
        score = ClaimScore(claim=claim)

        # Domain relevance (Step 3.1)
        if claim.domain == target_domain:
            score.relevance = 1.0
        elif self._is_related_domain(claim.domain, target_domain):
            score.relevance = 0.5
        else:
            score.relevance = 1.0 - self.config.cross_domain_penalty

        # Confidence score
        score.confidence = claim.confidence

        # Correctness score (Step 3.4)
        if claim.is_correct is True:
            score.correctness = 1.0
        elif claim.is_correct is None:
            score.correctness = 0.5  # Unknown
        else:
            score.correctness = 0.0

        # Recency score (newer claims rank higher)
        total_claims = len(self._claims)
        if total_claims > 0:
            # Older claims have lower recency score
            score.recency = 1.0 - (index / total_claims) * self.config.recency_decay
        else:
            score.recency = 1.0

        score.calculate_total()
        return score

    def _is_related_domain(
        self,
        domain1: ClaimDomain,
        domain2: ClaimDomain
    ) -> bool:
        """Check if two domains are related."""
        related_groups = [
            {ClaimDomain.MATH, ClaimDomain.GEOMETRY, ClaimDomain.PROBABILITY},
            {ClaimDomain.LOGIC, ClaimDomain.PATTERN},
            {ClaimDomain.SCIENCE},
            {ClaimDomain.CALENDAR, ClaimDomain.PATTERN},
        ]

        for group in related_groups:
            if domain1 in group and domain2 in group:
                return True

        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored claims."""
        return {
            "total_claims": len(self._claims),
            "by_domain": {
                d.value: len(claims) for d, claims in self._domain_index.items()
            },
            "correct": len(self._correct_claims),
            "incorrect": len(self._incorrect_claims),
            "untested": len(self._claims) - len(self._correct_claims) - len(self._incorrect_claims),
            "avg_confidence": sum(c.confidence for c in self._claims) / len(self._claims) if self._claims else 0,
        }

    def format_context(
        self,
        claims: List[Claim],
        include_confidence: bool = True,
        include_correctness: bool = True
    ) -> str:
        """Format claims into a context string for LLM prompt."""
        if not claims:
            return "No prior knowledge."

        lines = ["Prior knowledge:"]
        for claim in claims:
            line = f"- {claim.content[:100]}"
            if include_confidence:
                line += f" (conf={claim.confidence:.0%}"
            if include_correctness and claim.is_correct is not None:
                line += f", correct={claim.is_correct}"
            if include_confidence or include_correctness:
                line += ")"
            lines.append(line)

        return "\n".join(lines)


def string_to_domain(domain_str: str) -> ClaimDomain:
    """Convert string domain to ClaimDomain enum."""
    mapping = {
        "math": ClaimDomain.MATH,
        "logic": ClaimDomain.LOGIC,
        "science": ClaimDomain.SCIENCE,
        "geometry": ClaimDomain.GEOMETRY,
        "probability": ClaimDomain.PROBABILITY,
        "pattern": ClaimDomain.PATTERN,
        "calendar": ClaimDomain.CALENDAR,
        "general": ClaimDomain.GENERAL,
    }
    return mapping.get(domain_str.lower(), ClaimDomain.GENERAL)
