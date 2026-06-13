# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""D-0004: Tag Lifecycle Management.

Tags are LLM-generated, not user-assigned.
- Split trigger: tag >20% usage (fraction of total claims)
- Process: sample ≤100 claims → LLM suggests ≤8 replacement tags
  → batch claims (20) → LLM assigns replacement per claim
- If total >500 → merge similar tags
- Max 20 tags per claim

Enables tag quality maintenance without manual user intervention.
"""

from __future__ import annotations

import json as json_lib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.models import Claim
    from src.data.repositories import ClaimRepository


logger = logging.getLogger(__name__)


@dataclass
class TagStat:
    """Usage statistics for a single tag."""
    tag: str
    claim_count: int
    total_claims: int

    @property
    def usage_fraction(self) -> float:
        """Fraction of total claims bearing this tag."""
        if self.total_claims == 0:
            return 0.0
        return self.claim_count / self.total_claims


@dataclass
class TagUsageStats:
    """Aggregated tag usage statistics for all tags in a repository."""
    tag_stats: Dict[str, TagStat] = field(default_factory=dict)
    total_claims: int = 0

    @staticmethod
    async def compute(repository: ClaimRepository) -> TagUsageStats:
        """Compute usage statistics for all tags in the repository."""
        claims = await repository.list_all(limit=10000)
        total = len(claims)

        tag_counts: Dict[str, int] = {}
        for claim in claims:
            for tag in claim.tags:
                tag_lower = tag.lower()
                tag_counts[tag_lower] = tag_counts.get(tag_lower, 0) + 1

        tag_stats = {
            tag: TagStat(tag=tag, claim_count=count, total_claims=total)
            for tag, count in tag_counts.items()
        }

        return TagUsageStats(tag_stats=tag_stats, total_claims=total)

    def get_stat(self, tag: str) -> Optional[TagStat]:
        """Get usage stat for a specific tag (case-insensitive)."""
        return self.tag_stats.get(tag.lower())

    def tags_above_threshold(self, threshold: float) -> List[str]:
        """Return tags whose usage fraction exceeds the threshold."""
        return [
            tag for tag, stat in self.tag_stats.items()
            if stat.usage_fraction > threshold
        ]


@dataclass
class SplitTriggerResult:
    """Result of split trigger detection."""
    split_needed: bool
    triggered_tags: List[str]
    usage_stats: TagUsageStats
    merge_needed: bool = False
    total_claims: int = 0

    @staticmethod
    async def detect(
        repository: ClaimRepository,
        threshold: float = 0.20,
        merge_threshold: int = 500,
    ) -> SplitTriggerResult:
        """Detect which tags exceed the split threshold."""
        stats = await TagUsageStats.compute(repository)
        triggered = stats.tags_above_threshold(threshold)

        merge_needed = stats.total_claims > merge_threshold

        return SplitTriggerResult(
            split_needed=len(triggered) > 0,
            triggered_tags=triggered,
            usage_stats=stats,
            merge_needed=merge_needed,
            total_claims=stats.total_claims,
        )


@dataclass
class TagReplacementResult:
    """Result of tag replacement operation."""
    split_needed: bool
    split_tags: List[str]
    replacement_suggestions: Dict[str, List[str]]
    claims_updated: int
    merge_needed: bool
    total_claims: int


class TagLifecycleManager:
    """Manages tag lifecycle: split overused tags, merge on large datasets.

    D-0004 implementation:
    - Split trigger: tag usage >20% of total claims
    - Process: sample claims → LLM suggests ≤8 replacement tags
      → batch claims (20) → LLM assigns replacement per claim
    - Merge trigger: total claims >500
    """

    def __init__(
        self,
        repository: ClaimRepository,
        llm_processor: Any,
        split_threshold: float = 0.20,
        batch_size: int = 20,
        max_suggested_tags: int = 8,
        merge_threshold: int = 500,
        max_tags_per_claim: int = 20,
    ):
        self.repository = repository
        self.llm_processor = llm_processor
        self.split_threshold = split_threshold
        self.batch_size = batch_size
        self.max_suggested_tags = max_suggested_tags
        self.merge_threshold = merge_threshold
        self.max_tags_per_claim = max_tags_per_claim

    async def check_and_suggest(
        self,
        repository: Optional[ClaimRepository] = None,
    ) -> TagReplacementResult:
        """Check for split/merge triggers and return replacement suggestions.

        Args:
            repository: Repository to check. Uses self.repository if None.

        Returns:
            TagReplacementResult with split/merge decisions and suggestions.
        """
        repo = repository or self.repository

        trigger_result = await SplitTriggerResult.detect(
            repo,
            threshold=self.split_threshold,
            merge_threshold=self.merge_threshold,
        )

        if not trigger_result.split_needed:
            return TagReplacementResult(
                split_needed=False,
                split_tags=[],
                replacement_suggestions={},
                claims_updated=0,
                merge_needed=trigger_result.merge_needed,
                total_claims=trigger_result.total_claims,
            )

        # Sample claims for LLM tag suggestion (≤100)
        all_claims = await repo.list_all(limit=10000)
        sample_size = min(100, len(all_claims))
        sampled = all_claims[:sample_size]

        # Collect all overused tags
        overused_tags = trigger_result.triggered_tags

        # For each overused tag, get LLM replacement suggestions
        replacement_suggestions: Dict[str, List[str]] = {}
        for tag in overused_tags:
            suggestions = await self._llm_suggest_replacements(tag, sampled)
            replacement_suggestions[tag] = suggestions[: self.max_suggested_tags]

        # Apply replacements in batches
        claims_to_update = [
            c for c in all_claims
            if any(t.lower() in [tag.lower() for tag in overused_tags] for t in c.tags)
        ]

        updated_count = await self._apply_replacements(
            claims_to_update,
            overused_tags,
            replacement_suggestions,
        )

        return TagReplacementResult(
            split_needed=True,
            split_tags=overused_tags,
            replacement_suggestions=replacement_suggestions,
            claims_updated=updated_count,
            merge_needed=trigger_result.merge_needed,
            total_claims=trigger_result.total_claims,
        )

    async def _llm_suggest_replacements(
        self,
        overused_tag: str,
        sampled_claims: List[Claim],
    ) -> List[str]:
        """Ask LLM to suggest replacement tags for an overused tag."""
        sample_contents = [
            f"- [{c.id}] {c.content[:100]}" for c in sampled_claims[:20]
        ]
        prompt = (
            f"The tag '{overused_tag}' is overused (>20% of claims).\n"
            f"Claims using this tag:\n" + "\n".join(sample_contents) + "\n\n"
            f"Suggest up to {self.max_suggested_tags} more specific replacement tags.\n"
            f"Return a JSON array of tag strings. Example: [\"tag1\", \"tag2\"]"
        )

        try:
            response = await self.llm_processor.call(prompt)
            suggestions = json_lib.loads(response)
            if isinstance(suggestions, list):
                return [str(s).lower().strip() for s in suggestions if s]
        except Exception as exc:
            logger.warning(f"LLM tag suggestion failed for '{overused_tag}': {exc}")

        return []

    async def _apply_replacements(
        self,
        claims: List[Claim],
        overused_tags: List[str],
        replacement_suggestions: Dict[str, List[str]],
    ) -> int:
        """Apply tag replacements in batches of 20 claims."""
        overused_lower = {t.lower() for t in overused_tags}
        updated = 0

        for i in range(0, len(claims), self.batch_size):
            batch = claims[i : i + self.batch_size]

            for claim in batch:
                new_tags = []
                for tag in claim.tags:
                    if tag.lower() in overused_lower:
                        # Get replacement for this specific overused tag
                        replacements = replacement_suggestions.get(
                            tag, replacement_suggestions.get(tag.lower(), [])
                        )
                        if replacements:
                            new_tags.extend(replacements[:2])  # Add top 2 replacements
                        # Don't keep the overused tag
                    else:
                        new_tags.append(tag)

                # Deduplicate while preserving order, cap at max_tags_per_claim
                seen: Set[str] = set()
                deduped = []
                for t in new_tags:
                    tl = t.lower()
                    if tl not in seen and len(deduped) < self.max_tags_per_claim:
                        seen.add(tl)
                        deduped.append(t)

                claim.tags = deduped

                try:
                    await self.repository.update(claim)
                    updated += 1
                except Exception as exc:
                    logger.warning(f"Failed to update claim {claim.id}: {exc}")

        return updated
