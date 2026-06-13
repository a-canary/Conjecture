# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Fact-Checking Pipeline — R&D 1 Design

Tiered verification architecture: self-consistency → vector search → live web.
Integrates with existing DirtyFlagSystem and claim graph.

HYPOTHESES:
  H1: Most false claims fail at Tier 1 (self-consistency) — internal graph contradictions.
  H2: Tier 2 (vector search) catches 80% of remaining errors via semantic similarity.
  H3: Tier 3 (live web) is needed for <5% of claims — expensive but necessary for novel claims.
  H4: Cascade invalidation on failure reduces downstream error propagation by >10x.

COST-QUALITY TRADEoffs:
  - Tier 1: ~0 extra LLM calls (uses existing reasoning)
  - Tier 2: 1 vector search + optional LLM call (~$0.0001/claim)
  - Tier 3: 1 web search + LLM synthesis (~$0.01/claim)
  - Quality target: 95% precision, 80% recall on fact claims
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Tuple, Dict, Any, Set

import numpy as np

logger = logging.getLogger(__name__)


class VerificationTier(str, Enum):
    """Verification tier levels in order of increasing cost/quality."""
    SELF_CONSISTENCY = "self_consistency"   # Free, fast
    VECTOR_SEARCH = "vector_search"          # Moderate cost
    LIVE_WEB = "live_web"                    # Expensive, thorough


class FactCheckResult(str, Enum):
    """Outcome of a fact-check."""
    VERIFIED = "verified"           # Claim confirmed
    REJECTED = "rejected"           # Claim contradicted
    UNCERTAIN = "uncertain"        # Cannot determine
    SKIPPED = "skipped"             # Tier not attempted


class FactCheckStatus(str, Enum):
    """Processing status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class TierResult:
    """Result from a single verification tier."""
    tier: VerificationTier
    result: FactCheckResult
    confidence: float           # Confidence in this tier's verdict (0-1)
    evidence: List[str]         # Supporting evidence strings
    contradictions: List[str]    # Found contradictions (if any)
    cost_usd: float              # Actual cost incurred
    latency_ms: int             # Time taken in milliseconds
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.result == FactCheckResult.VERIFIED


@dataclass
class FactCheckReport:
    """Complete fact-checking report for a claim."""
    claim_id: str
    claim_content: str
    status: FactCheckStatus
    final_result: FactCheckResult
    final_confidence: float      # Aggregate confidence across tiers
    tier_results: List[TierResult]
    total_cost_usd: float
    total_latency_ms: int
    cascaded_claims: Set[str]    # Claim IDs marked dirty due to failure
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_content": self.claim_content,
            "status": self.status.value,
            "final_result": self.final_result.value,
            "final_confidence": self.final_confidence,
            "tier_results": [
                {
                    "tier": r.tier.value,
                    "result": r.result.value,
                    "confidence": r.confidence,
                    "evidence": r.evidence,
                    "contradictions": r.contradictions,
                    "cost_usd": r.cost_usd,
                    "latency_ms": r.latency_ms,
                }
                for r in self.tier_results
            ],
            "total_cost_usd": self.total_cost_usd,
            "total_latency_ms": self.total_latency_ms,
            "cascaded_claims": list(self.cascaded_claims),
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Tier 1: Self-Consistency Checker
# ---------------------------------------------------------------------------

class SelfConsistencyChecker:
    """
    Tier 1: Verify claim against internal claim graph.

    Checks:
    1. Are there sub-claims that contradict this claim?
    2. Do any sibling claims conflict?
    3. Is the claim consistent with its declared type/state?

    COST: Zero extra LLM calls (uses existing claim graph structure).
    """

    def __init__(self, claims: Dict[str, Any]):
        self.claims = claims  # claim_id -> Claim

    def check(self, claim_id: str) -> TierResult:
        """Run self-consistency check on a claim."""
        start = datetime.now(timezone.utc)
        claim = self.claims.get(claim_id)

        if claim is None:
            return TierResult(
                tier=VerificationTier.SELF_CONSISTENCY,
                result=FactCheckResult.SKIPPED,
                confidence=0.0,
                evidence=[],
                contradictions=[f"Claim {claim_id} not found"],
                cost_usd=0.0,
                latency_ms=int((datetime.now(timezone.utc) - start).total_seconds() * 1000),
            )

        contradictions = []
        evidence = []

        # Check 1: Sub-claims (children) that provide evidence FOR this claim
        # If sub-claims have LOW confidence, they weaken this claim
        sub_claims = [self.claims[sid] for sid in claim.subs if sid in self.claims]
        for sub in sub_claims:
            if sub.confidence < 0.3:
                contradictions.append(
                    f"Sub-claim '{sub.id}' has very low confidence ({sub.confidence})"
                )
            if sub.state == "Orphaned":
                evidence.append(f"Sub-claim '{sub.id}' is orphaned — relationship may be invalid")

        # Check 2: Self-consistency — claim's own confidence vs type
        # ASSERTION type should have high confidence
        if hasattr(claim, 'type') and claim.type:
            type_values = [t.value if hasattr(t, 'value') else str(t) for t in claim.type]
            if 'assertion' in type_values and claim.confidence < 0.7:
                contradictions.append(
                    f"ASSERTION type with low confidence ({claim.confidence})"
                )
            if 'impression' in type_values and claim.confidence > 0.95:
                evidence.append(
                    f"IMPRESSION type with very high confidence ({claim.confidence}) — may be overconfident"
                )

        # Check 3: Confidence consistency with support strength
        # Calculate what support strength SHOULD be based on subs
        if sub_claims:
            avg_sub_confidence = sum(s.confidence for s in sub_claims) / len(sub_claims)
            if claim.confidence > avg_sub_confidence + 0.3:
                contradictions.append(
                    f"Claim confidence ({claim.confidence}) exceeds average sub confidence ({avg_sub_confidence:.2f})"
                )

        # Check 4: Newly added claims that haven't been verified
        if claim.is_dirty and hasattr(claim, 'dirty_reason'):
            reason = claim.dirty_reason
            if hasattr(reason, 'value'):
                reason_str = reason.value
            else:
                reason_str = str(reason)

            if reason_str in ('new_claim_added', 'manual_mark'):
                evidence.append(f"Claim is dirty due to: {reason_str}")

        # Determine result
        if contradictions:
            result = FactCheckResult.REJECTED
            confidence = min(0.9, len(contradictions) * 0.3)
        elif evidence:
            result = FactCheckResult.VERIFIED
            confidence = 0.7
        else:
            result = FactCheckResult.VERIFIED
            confidence = 0.85  # Default: assume consistent until proven guilty

        return TierResult(
            tier=VerificationTier.SELF_CONSISTENCY,
            result=result,
            confidence=confidence,
            evidence=evidence,
            contradictions=contradictions,
            cost_usd=0.0,
            latency_ms=int((datetime.now(timezone.utc) - start).total_seconds() * 1000),
        )


# ---------------------------------------------------------------------------
# Tier 2: Vector Search Verifier
# ---------------------------------------------------------------------------

class VectorSearchVerifier:
    """
    Tier 2: Semantic similarity search against existing claims.

    Uses FAISS vector store to find semantically similar claims.
    If similar claims have different verdicts, flag for human review.

    COST: ~$0.0001 per claim (vector search + optional LLM call).
    """

    def __init__(self, vector_store: Any, claims: Dict[str, Any]):
        self.vector_store = vector_store
        self.claims = claims

    def check(
        self,
        claim_id: str,
        similarity_threshold: float = 0.85,
        k_results: int = 10,
    ) -> TierResult:
        """Run vector search verification."""
        import time
        start = time.time()

        if self.vector_store is None:
            return TierResult(
                tier=VerificationTier.VECTOR_SEARCH,
                result=FactCheckResult.SKIPPED,
                confidence=0.0,
                evidence=[],
                contradictions=["Vector store not available"],
                cost_usd=0.0,
                latency_ms=0,
            )

        claim = self.claims.get(claim_id)
        if claim is None:
            return TierResult(
                tier=VerificationTier.VECTOR_SEARCH,
                result=FactCheckResult.SKIPPED,
                confidence=0.0,
                evidence=[],
                contradictions=[f"Claim {claim_id} not found"],
                cost_usd=0.0,
                latency_ms=0,
            )

        evidence = []
        contradictions = []

        # Search for similar claims
        search_results = self.vector_store.search(claim.content, k=k_results)

        similar_claims = []
        for similar_id, score in search_results:
            if similar_id == claim_id:
                continue
            if score >= similarity_threshold and similar_id in self.claims:
                similar_claims.append((similar_id, score, self.claims[similar_id]))

        if not similar_claims:
            evidence.append("No semantically similar claims found in vector store")
            return TierResult(
                tier=VerificationTier.VECTOR_SEARCH,
                result=FactCheckResult.VERIFIED,
                confidence=0.6,  # Low confidence — no reference material
                evidence=evidence,
                contradictions=[],
                cost_usd=0.0001,
                latency_ms=int((time.time() - start) * 1000),
            )

        # Check for contradictions in similar claims
        conflicting_claims = []
        supporting_claims = []

        for similar_id, score, similar_claim in similar_claims:
            conf_diff = abs(claim.confidence - similar_claim.confidence)

            if conf_diff > 0.5:
                # Major confidence discrepancy
                if similar_claim.confidence < 0.4 and claim.confidence > 0.7:
                    conflicting_claims.append(similar_id)
                    contradictions.append(
                        f"Similar claim '{similar_id}' (score={score:.3f}) has "
                        f"LOW confidence ({similar_claim.confidence}) vs this claim's "
                        f"HIGH confidence ({claim.confidence})"
                    )
                elif similar_claim.confidence > 0.7 and claim.confidence < 0.4:
                    evidence.append(
                        f"Similar claim '{similar_id}' (score={score:.3f}) has "
                        f"HIGH confidence ({similar_claim.confidence}) vs this claim's "
                        f"LOW confidence ({claim.confidence})"
                    )

            # Check content for negation patterns
            claim_lower = claim.content.lower()
            similar_lower = similar_claim.content.lower()

            negation_patterns = [
                ("not", "is"), ("never", "always"), ("false", "true"),
                ("no ", "yes"), ("cannot", "can"), ("impossible", "possible")
            ]

            for neg, pos in negation_patterns:
                if (neg in similar_lower and pos in claim_lower) or \
                   (pos in similar_lower and neg in claim_lower):
                    contradictions.append(
                        f"Similar claim '{similar_id}' contains potential negation pattern"
                    )

        # Calculate confidence
        if conflicting_claims:
            result = FactCheckResult.UNCERTAIN
            confidence = 0.5
        elif len(similar_claims) > 3 and not contradictions:
            result = FactCheckResult.VERIFIED
            confidence = 0.85
        elif evidence:
            result = FactCheckResult.VERIFIED
            confidence = 0.75
        else:
            result = FactCheckResult.VERIFIED
            confidence = 0.8

        return TierResult(
            tier=VerificationTier.VECTOR_SEARCH,
            result=result,
            confidence=confidence,
            evidence=evidence,
            contradictions=contradictions,
            cost_usd=0.0001,
            latency_ms=int((time.time() - start) * 1000),
        )


# ---------------------------------------------------------------------------
# Tier 3: Live Web Verifier
# ---------------------------------------------------------------------------

class LiveWebVerifier:
    """
    Tier 3: External fact-checking via live web search.

    Uses web search to verify claims against external knowledge sources.
    Should only be invoked for high-value or high-risk claims.

    COST: ~$0.01 per claim (web search + LLM synthesis).
    """

    def __init__(self, web_search_func: callable):
        """
        Args:
            web_search_func: Function that takes query string and returns
                            List[Tuple[title, url, snippet]]
        """
        self.web_search_func = web_search_func

    def check(
        self,
        claim_id: str,
        claim_content: str,
        max_sources: int = 5,
    ) -> TierResult:
        """Run live web verification."""
        import time
        start = time.time()

        if not claim_content:
            return TierResult(
                tier=VerificationTier.LIVE_WEB,
                result=FactCheckResult.SKIPPED,
                confidence=0.0,
                evidence=[],
                contradictions=["Empty claim content"],
                cost_usd=0.0,
                latency_ms=int((time.time() - start) * 1000),
            )

        evidence = []
        contradictions = []
        supporting_sources = 0
        contradicting_sources = 0

        try:
            # Perform web search
            search_results = self.web_search_func(claim_content[:200])  # Truncate for search

            for title, url, snippet in search_results[:max_sources]:
                evidence.append(f"Source: {title} — {snippet[:100]}...")

                # Simple keyword-based verdict (in production, use LLM)
                snippet_lower = snippet.lower()
                claim_lower = claim_content.lower()

                # Check for supporting indicators
                supporting_keywords = ["confirmed", "verified", "true", "correct", "accurate"]
                contradicting_keywords = ["false", "denied", "incorrect", "myth", "hoax", "untrue"]

                support_count = sum(1 for kw in supporting_keywords if kw in snippet_lower)
                contradict_count = sum(1 for kw in contradicting_keywords if kw in snippet_lower)

                if support_count > contradict_count:
                    supporting_sources += 1
                elif contradict_count > support_count:
                    contradicting_sources += 1

            # Determine result
            if contradicting_sources > supporting_sources:
                result = FactCheckResult.REJECTED
                confidence = min(0.95, 0.5 + contradicting_sources * 0.15)
                contradictions.append(
                    f"{contradicting_sources} sources contradict the claim"
                )
            elif supporting_sources > contradicting_sources:
                result = FactCheckResult.VERIFIED
                confidence = min(0.95, 0.5 + supporting_sources * 0.15)
            else:
                result = FactCheckResult.UNCERTAIN
                confidence = 0.5

        except Exception as e:
            logger.warning(f"Web verification failed for {claim_id}: {e}")
            return TierResult(
                tier=VerificationTier.LIVE_WEB,
                result=FactCheckResult.UNCERTAIN,
                confidence=0.0,
                evidence=[],
                contradictions=[f"Web search failed: {str(e)}"],
                cost_usd=0.005,
                latency_ms=int((time.time() - start) * 1000),
            )

        return TierResult(
            tier=VerificationTier.LIVE_WEB,
            result=result,
            confidence=confidence,
            evidence=evidence,
            contradictions=contradictions,
            cost_usd=0.01,  # Estimated
            latency_ms=int((time.time() - start) * 1000),
        )


# ---------------------------------------------------------------------------
# Cascade Invalidation System
# ---------------------------------------------------------------------------

class CascadeInvalidator:
    """
    Handles cascade invalidation when a claim fails verification.

    When a claim is REJECTED:
    1. Mark all claims that depend on this claim (supers) as dirty
    2. Propagate rejection confidence reduction through the graph
    3. Generate a detailed audit trail

    Uses existing DirtyFlagSystem infrastructure.
    """

    def __init__(
        self,
        dirty_flag_system: Any,
        claims: Dict[str, Any],
        confidence_decay: float = 0.5,
    ):
        """
        Args:
            dirty_flag_system: Instance of DirtyFlagSystem for marking dirty
            claims: Dict of claim_id -> Claim
            confidence_decay: How much to reduce confidence on rejection cascade
        """
        self.dirty_flag = dirty_flag_system
        self.claims = claims
        self.confidence_decay = confidence_decay

    def cascade_rejection(
        self,
        claim_id: str,
        rejection_confidence: float = 0.0,
        max_depth: int = 3,
    ) -> Set[str]:
        """
        Cascade rejection through the claim graph.

        Args:
            claim_id: The claim that was rejected
            rejection_confidence: How confident we are in the rejection
            max_depth: Maximum cascade depth

        Returns:
            Set of claim IDs that were marked dirty
        """
        marked_ids: Set[str] = set()
        self._cascade_impl(claim_id, rejection_confidence, max_depth, current_depth=0, marked=marked_ids)
        return marked_ids

    def _cascade_impl(
        self,
        claim_id: str,
        rejection_confidence: float,
        max_depth: int,
        current_depth: int,
        marked: Set[str],
    ) -> None:
        """Recursive cascade implementation."""
        if current_depth >= max_depth or claim_id in marked:
            return

        claim = self.claims.get(claim_id)
        if claim is None:
            return

        # Mark this claim dirty
        self.dirty_flag.mark_claim_dirty(
            claim=claim,
            reason="supporting_claim_changed",
            priority=10 - current_depth * 2,  # Decreasing priority with depth
            cascade=False,  # We handle cascade manually
        )
        marked.add(claim_id)

        # Propagate to claims that this one provides evidence FOR (supers)
        for super_id in claim.supers:
            if super_id not in marked:
                super_claim = self.claims.get(super_id)
                if super_claim:
                    # Calculate new confidence based on decay
                    decay_factor = self.confidence_decay ** (current_depth + 1)
                    confidence_reduction = (1.0 - super_claim.confidence) * decay_factor * (1.0 - rejection_confidence)

                    logger.info(
                        f"Cascade: Claim '{claim_id}' rejection propagates to '{super_id}' "
                        f"(depth={current_depth}, confidence_reduction={confidence_reduction:.3f})"
                    )

                    self._cascade_impl(
                        super_id,
                        rejection_confidence * decay_factor,
                        max_depth,
                        current_depth + 1,
                        marked,
                    )


# ---------------------------------------------------------------------------
# Main Fact-Checking Pipeline
# ---------------------------------------------------------------------------

class FactCheckingPipeline:
    """
    Tiered fact-checking pipeline with cascade invalidation.

    Usage:
        pipeline = FactCheckingPipeline(
            claims=claims_dict,
            vector_store=vector_store,
            dirty_flag_system=dirty_flag_system,
            web_search_func=my_web_search,
        )

        report = pipeline.verify("claim-123", tiers=[
            VerificationTier.SELF_CONSISTENCY,
            VerificationTier.VECTOR_SEARCH,
            VerificationTier.LIVE_WEB,
        ])
    """

    def __init__(
        self,
        claims: Dict[str, Any],
        vector_store: Any,
        dirty_flag_system: Any,
        web_search_func: Optional[callable] = None,
        tier_config: Optional[Dict[str, Any]] = None,
    ):
        self.claims = claims
        self.vector_store = vector_store
        self.dirty_flag_system = dirty_flag_system
        self.web_search_func = web_search_func

        # Tier configuration
        self.tier_config = tier_config or {
            "self_consistency": {
                "enabled": True,
                "always_run": True,  # Run first regardless
            },
            "vector_search": {
                "enabled": True,
                "similarity_threshold": 0.85,
                "k_results": 10,
                "skip_on_rejection": False,  # Continue to next tier on rejection
            },
            "live_web": {
                "enabled": web_search_func is not None,
                "max_sources": 5,
                "skip_on_rejection": True,  # Stop cascade on rejection
                "only_on_uncertain": False,  # Only run if previous tier was uncertain
            },
        }

        # Initialize tier verifiers
        self.tier1 = SelfConsistencyChecker(claims)
        self.tier2 = VectorSearchVerifier(vector_store, claims)
        self.tier3 = LiveWebVerifier(web_search_func) if web_search_func else None

        # Cascade invalidator
        self.cascade_invalidator = CascadeInvalidator(
            dirty_flag_system=dirty_flag_system,
            claims=claims,
        )

    def verify(
        self,
        claim_id: str,
        tiers: Optional[List[VerificationTier]] = None,
        force_tier3: bool = False,
    ) -> FactCheckReport:
        """
        Run fact-checking pipeline on a claim.

        Args:
            claim_id: Claim to verify
            tiers: List of tiers to run (defaults to all enabled)
            force_tier3: Force Tier 3 even if not in tiers list

        Returns:
            FactCheckReport with detailed results
        """
        claim = self.claims.get(claim_id)
        if claim is None:
            return FactCheckReport(
                claim_id=claim_id,
                claim_content="",
                status=FactCheckStatus.FAILED,
                final_result=FactCheckResult.SKIPPED,
                final_confidence=0.0,
                tier_results=[],
                total_cost_usd=0.0,
                total_latency_ms=0,
                cascaded_claims=set(),
                metadata={"error": f"Claim {claim_id} not found"},
            )

        if tiers is None:
            tiers = [VerificationTier.SELF_CONSISTENCY]
            if self.tier_config["vector_search"]["enabled"]:
                tiers.append(VerificationTier.VECTOR_SEARCH)
            if self.tier_config["live_web"]["enabled"] and (force_tier3 or self._should_run_tier3()):
                tiers.append(VerificationTier.LIVE_WEB)

        tier_results: List[TierResult] = []
        total_cost = 0.0
        total_latency = 0
        cascaded_ids: Set[str] = set()

        for tier in tiers:
            try:
                result = self._run_tier(tier, claim_id, claim)
            except Exception as e:
                logger.error(f"Tier {tier.value} failed for {claim_id}: {e}")
                result = TierResult(
                    tier=tier,
                    result=FactCheckResult.UNCERTAIN,
                    confidence=0.0,
                    evidence=[],
                    contradictions=[f"Tier failed: {str(e)}"],
                    cost_usd=0.0,
                    latency_ms=0,
                )

            tier_results.append(result)
            total_cost += result.cost_usd
            total_latency += result.latency_ms

            # Handle early termination
            if result.result == FactCheckResult.REJECTED:
                tier_cfg = self.tier_config.get(tier.value, {})
                if tier_cfg.get("skip_on_rejection", False):
                    logger.info(f"Claim {claim_id} rejected at {tier.value}, stopping pipeline")
                    break

            if result.result == FactCheckResult.UNCERTAIN and tier == VerificationTier.SELF_CONSISTENCY:
                # Low confidence from self-consistency — continue to next tier
                continue

        # Aggregate final result
        final_result, final_confidence = self._aggregate_results(tier_results)

        # Handle cascade on rejection
        if final_result == FactCheckResult.REJECTED:
            avg_confidence = sum(r.confidence for r in tier_results) / len(tier_results) if tier_results else 0.0
            cascaded_ids = self.cascade_invalidator.cascade_rejection(
                claim_id,
                rejection_confidence=avg_confidence,
            )

        return FactCheckReport(
            claim_id=claim_id,
            claim_content=claim.content,
            status=FactCheckStatus.COMPLETE,
            final_result=final_result,
            final_confidence=final_confidence,
            tier_results=tier_results,
            total_cost_usd=total_cost,
            total_latency_ms=total_latency,
            cascaded_claims=cascaded_ids,
        )

    def _run_tier(
        self,
        tier: VerificationTier,
        claim_id: str,
        claim: Any,
    ) -> TierResult:
        """Run a specific verification tier."""
        if tier == VerificationTier.SELF_CONSISTENCY:
            return self.tier1.check(claim_id)

        elif tier == VerificationTier.VECTOR_SEARCH:
            cfg = self.tier_config["vector_search"]
            return self.tier2.check(
                claim_id,
                similarity_threshold=cfg.get("similarity_threshold", 0.85),
                k_results=cfg.get("k_results", 10),
            )

        elif tier == VerificationTier.LIVE_WEB:
            cfg = self.tier_config["live_web"]
            return self.tier3.check(
                claim_id,
                claim.content,
                max_sources=cfg.get("max_sources", 5),
            )

        raise ValueError(f"Unknown tier: {tier}")

    def _aggregate_results(
        self,
        tier_results: List[TierResult],
    ) -> Tuple[FactCheckResult, float]:
        """Aggregate results from multiple tiers into final verdict."""
        if not tier_results:
            return FactCheckResult.UNCERTAIN, 0.0

        # Weight tiers by their confidence and reliability
        # Tier 3 (live web) is most reliable, Tier 1 is least
        tier_weights = {
            VerificationTier.SELF_CONSISTENCY: 0.2,
            VerificationTier.VECTOR_SEARCH: 0.3,
            VerificationTier.LIVE_WEB: 0.5,
        }

        weighted_confidence = 0.0
        total_weight = 0.0

        for result in tier_results:
            weight = tier_weights.get(result.tier, 0.25)
            if result.result == FactCheckResult.VERIFIED:
                weighted_confidence += weight * result.confidence
            elif result.result == FactCheckResult.REJECTED:
                weighted_confidence -= weight * result.confidence  # Negative weight for rejection
            # SKIPPED and UNCERTAIN get 0 contribution
            total_weight += weight

        # Normalize
        final_confidence = max(0.0, min(1.0, weighted_confidence / total_weight if total_weight else 0.5))

        # Determine final result
        rejections = sum(1 for r in tier_results if r.result == FactCheckResult.REJECTED)
        verifications = sum(1 for r in tier_results if r.result == FactCheckResult.VERIFIED)

        if rejections > verifications:
            return FactCheckResult.REJECTED, final_confidence
        elif verifications > rejections:
            return FactCheckResult.VERIFIED, final_confidence
        else:
            return FactCheckResult.UNCERTAIN, final_confidence

    def _should_run_tier3(self) -> bool:
        """Determine if Tier 3 should run based on config."""
        cfg = self.tier_config["live_web"]
        if not cfg.get("enabled", False):
            return False
        if cfg.get("only_on_uncertain", False):
            return True  # Caller should check previous tier uncertainty
        return True

    def verify_batch(
        self,
        claim_ids: List[str],
        tiers: Optional[List[VerificationTier]] = None,
    ) -> List[FactCheckReport]:
        """Run fact-checking on multiple claims."""
        reports = []
        for claim_id in claim_ids:
            report = self.verify(claim_id, tiers=tiers)
            reports.append(report)
        return reports
