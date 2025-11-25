"""
Complete Context Builder for Simplified Universal Claim Architecture
Builds comprehensive contexts with complete relationship coverage and optimized token management
"""

from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import re

from ..core.models import Claim, create_claim_index
from ..core.support_relationship_manager import SupportRelationshipManager
from ..tools.registry import ToolRegistry


@dataclass
class ContextAllocation:
    """Token allocation for context building"""

    upward_chain_tokens: int  # 40% - supporting claims to root
    downward_chain_tokens: int  # 30% - supported claims (descendants)
    semantic_tokens: int  # 30% - semantically similar claims
    total_tokens: int


@dataclass
class ContextMetrics:
    """Performance and coverage metrics for context building"""

    total_claims_considered: int
    upward_chain_claims: int
    downward_chain_claims: int
    semantic_claims: int
    tokens_used: int
    token_efficiency: float
    build_time_ms: float
    coverage_completeness: float


@dataclass
class BuiltContext:
    """Complete built context with metadata"""

    context_text: str
    allocation: ContextAllocation
    metrics: ContextMetrics
    included_claims: List[str]
    target_claim_id: str


class CompleteContextBuilder:
    """
    Builds complete contexts with comprehensive relationship coverage.

    Ensures ALL supporting claims to root are included (40% allocation),
    ALL supported claims are included (30% allocation),
    and fills remaining with semantically similar claims (30% allocation).
    """

    def __init__(self, claims: List[Claim], include_core_tools: bool = True):
        """Initialize context builder with claim collection"""
        self.claims = claims
        self.claim_index = create_claim_index(claims)
        self.relationship_manager = SupportRelationshipManager(claims)
        self.include_core_tools = include_core_tools

        # Ensure tools are loaded
        if include_core_tools:
            self.tool_registry = ToolRegistry()
        else:
            self.tool_registry = None

        # Token estimation settings
        self.estimated_tokens_per_claim = 100  # Rough estimate
        self.overhead_tokens = 500  # For headers, separators, etc.

        # Additional overhead for core tools section
        if include_core_tools:
            self.overhead_tokens += 1000  # Extra space for tools section

    def build_complete_context(
        self,
        target_claim_id: str,
        max_tokens: int = 8000,
        include_metadata: bool = True,
    ) -> BuiltContext:
        """
        Build complete context for a target claim.

        Args:
            target_claim_id: The claim to build context for
            max_tokens: Maximum tokens to use (default: 8000)
            include_metadata: Whether to include claim metadata in context

        Returns:
            BuiltContext with complete relationship coverage
        """
        start_time = datetime.utcnow()

        if target_claim_id not in self.claim_index:
            raise ValueError(f"Target claim {target_claim_id} not found")

        # Calculate token allocation (account for tools section)
        actual_overhead = self.overhead_tokens
        if self.include_core_tools:
            # Estimate tools section size
            tools_context = self.tool_registry.get_core_tools_context()
            actual_overhead += len(tools_context) // 4  # Rough token estimate

        usable_tokens = max_tokens - actual_overhead
        allocation = ContextAllocation(
            upward_chain_tokens=int(usable_tokens * 0.4),
            downward_chain_tokens=int(usable_tokens * 0.3),
            semantic_tokens=int(usable_tokens * 0.3),
            total_tokens=max_tokens,
        )

        # Build upward chain (all supporting claims to root)
        upward_claims, upward_tokens = self._build_upward_chain(
            target_claim_id, allocation.upward_chain_tokens
        )

        # Build downward chain (all supported claims)
        downward_claims, downward_tokens = self._build_downward_chain(
            target_claim_id, allocation.downward_chain_tokens
        )

        # Build semantic similar claims
        semantic_claims, semantic_tokens = self._build_semantic_claims(
            target_claim_id, upward_claims + downward_claims, allocation.semantic_tokens
        )

        # Combine all claims (remove duplicates by ID)
        included_by_id = {}
        for claim in upward_claims + downward_claims + semantic_claims:
            included_by_id[claim.id] = claim
        all_included_claims = list(included_by_id.values())

        # Format the context
        context_text = self._format_context(
            target_claim_id,
            upward_claims,
            downward_claims,
            semantic_claims,
            include_metadata,
        )

        # Calculate metrics
        end_time = datetime.utcnow()
        build_time_ms = (end_time - start_time).total_seconds() * 1000
        actual_tokens_used = self._estimate_tokens(context_text)
        token_efficiency = actual_tokens_used / max_tokens

        metrics = ContextMetrics(
            total_claims_considered=len(self.claims),
            upward_chain_claims=len(upward_claims),
            downward_chain_claims=len(downward_claims),
            semantic_claims=len(semantic_claims),
            tokens_used=actual_tokens_used,
            token_efficiency=token_efficiency,
            build_time_ms=build_time_ms,
            coverage_completeness=self._calculate_coverage_completeness(
                target_claim_id, all_included_claims
            ),
        )

        return BuiltContext(
            context_text=context_text,
            allocation=allocation,
            metrics=metrics,
            included_claims=[claim.id for claim in all_included_claims],
            target_claim_id=target_claim_id,
        )

    def _build_upward_chain(
        self, target_claim_id: str, max_tokens: int
    ) -> Tuple[List[Claim], int]:
        """
        Build upward chain - ALL supporting claims to root.
        This is the highest priority and gets 40% of tokens.
        """
        # Get all supporting ancestors (complete upward traversal)
        ancestors_result = self.relationship_manager.get_all_supporting_ancestors(
            target_claim_id
        )
        ancestor_ids = ancestors_result.visited_claims

        # Convert to claim objects
        ancestor_claims = []
        for claim_id in ancestor_ids:
            claim = self.claim_index.get(claim_id)
            if claim:
                ancestor_claims.append(claim)

        # Sort by confidence (higher confidence first)
        ancestor_claims.sort(key=lambda c: c.confidence, reverse=True)

        # Apply token limit
        included_claims, tokens_used = self._apply_token_limit(
            ancestor_claims, max_tokens
        )

        return included_claims, tokens_used

    def _build_downward_chain(
        self, target_claim_id: str, max_tokens: int
    ) -> Tuple[List[Claim], int]:
        """
        Build downward chain - DIRECT supported claims only (not recursive).
        Gets 30% of tokens.
        """
        # Get only direct supported claims (single level, not all descendants)
        descendant_claims = self.relationship_manager.get_supported_claims(target_claim_id)

        # Sort by confidence (higher confidence first)
        descendant_claims.sort(key=lambda c: c.confidence, reverse=True)

        # Apply token limit
        included_claims, tokens_used = self._apply_token_limit(
            descendant_claims, max_tokens
        )

        return included_claims, tokens_used

    def _build_semantic_claims(
        self, target_claim_id: str, excluded_claims: List[Claim], max_tokens: int
    ) -> Tuple[List[Claim], int]:
        """
        Build semantic similar claims to fill remaining tokens.
        Fills 30% of tokens with claims not in relationship chains.
        """
        target_claim = self.claim_index[target_claim_id]
        excluded_ids = {claim.id for claim in excluded_claims}
        excluded_ids.add(target_claim_id)

        # Find semantically similar claims
        candidate_claims = []
        target_tags = set(target_claim.tags)
        target_words = set(target_claim.content.lower().split())

        for claim in self.claims:
            if claim.id in excluded_ids:
                continue

            # Calculate semantic similarity
            similarity = self._calculate_semantic_similarity(
                target_claim, claim, target_tags, target_words
            )

            if similarity > 0.1:  # Minimum similarity threshold
                candidate_claims.append((claim, similarity))

        # Sort by similarity and confidence
        candidate_claims.sort(key=lambda x: (x[1], x[0].confidence), reverse=True)

        # Select top candidates within token limit
        semantic_claims = []
        tokens_used = 0

        for claim, similarity in candidate_claims:
            claim_tokens = self._estimate_claim_tokens(claim)
            if tokens_used + claim_tokens <= max_tokens:
                semantic_claims.append(claim)
                tokens_used += claim_tokens
            else:
                break

        return semantic_claims, tokens_used

    def _calculate_semantic_similarity(
        self,
        target_claim: Claim,
        candidate_claim: Claim,
        target_tags: Set[str],
        target_words: Set[str],
    ) -> float:
        """Calculate semantic similarity between two claims"""
        # Tag similarity (40% weight)
        candidate_tags = set(candidate_claim.tags)
        tag_intersection = len(target_tags.intersection(candidate_tags))
        tag_union = len(target_tags.union(candidate_tags))
        tag_similarity = tag_intersection / tag_union if tag_union > 0 else 0.0

        # Content similarity (40% weight)
        candidate_words = set(candidate_claim.content.lower().split())
        word_intersection = len(target_words.intersection(candidate_words))
        word_union = len(target_words.union(candidate_words))
        content_similarity = word_intersection / word_union if word_union > 0 else 0.0

        # Confidence similarity (20% weight)
        confidence_diff = abs(target_claim.confidence - candidate_claim.confidence)
        confidence_similarity = 1.0 - confidence_diff

        # Weighted combination
        overall_similarity = (
            tag_similarity * 0.4
            + content_similarity * 0.4
            + confidence_similarity * 0.2
        )

        return overall_similarity

    def _apply_token_limit(
        self, claims: List[Claim], max_tokens: int
    ) -> Tuple[List[Claim], int]:
        """Apply token limit to a list of claims"""
        included_claims = []
        tokens_used = 0

        for claim in claims:
            claim_tokens = self._estimate_claim_tokens(claim)
            if tokens_used + claim_tokens <= max_tokens:
                included_claims.append(claim)
                tokens_used += claim_tokens
            else:
                break

        return included_claims, tokens_used

    def _estimate_claim_tokens(self, claim: Claim) -> int:
        """Estimate token count for a single claim"""
        # Rough estimation: ~4 characters per token + metadata
        content_tokens = len(claim.content) // 4
        metadata_tokens = 20  # For ID, confidence, tags
        format_overhead = 10  # For formatting

        return content_tokens + metadata_tokens + format_overhead

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Simple estimation: ~4 characters per token
        return len(text) // 4

    def _format_context(
        self,
        target_claim_id: str,
        upward_claims: List[Claim],
        downward_claims: List[Claim],
        semantic_claims: List[Claim],
        include_metadata: bool,
    ) -> str:
        """Format the complete context using the approved LLM template with Core Tools"""
        target_claim = self.claim_index[target_claim_id]

        context_parts = []

        # Core Tools section (added at the top as requested)
        if self.include_core_tools:
            tools_context = self.tool_registry.get_core_tools_context()
            context_parts.append(tools_context)
            context_parts.append("")
            context_parts.append("---")
            context_parts.append("")

        # Relevant Claims section (semantic claims first for LLM attention)
        if semantic_claims:
            context_parts.append("# Relevant Claims")
            for claim in semantic_claims:
                context_parts.append(claim.format_for_context())
            context_parts.append("")

        # Chain From User Claim section (supporting claims)
        if upward_claims:
            context_parts.append("# Chain From User Claim")
            for claim in upward_claims:
                context_parts.append(claim.format_for_context())
            context_parts.append("")

        # Supported_by Claims section (direct supported claims)
        if downward_claims:
            context_parts.append("# Supported_by Claims")
            for claim in downward_claims:
                context_parts.append(claim.format_for_context())
            context_parts.append("")

        # Target Claim (last for LLM attention, confidence redacted)
        context_parts.append("# Target Claim")
        # Redact confidence for target claim as required
        target_formatted = f"[c{target_claim.id} | {target_claim.content} | / confidence_redacted]"
        context_parts.append(target_formatted)

        return "\n".join(context_parts)

    def _calculate_coverage_completeness(
        self, target_claim_id: str, included_claims: List[Claim]
    ) -> float:
        """Calculate how completely the context covers all relationships"""
        target_claim = self.claim_index[target_claim_id]

        # Get all possible related claims
        ancestors_result = self.relationship_manager.get_all_supporting_ancestors(
            target_claim_id
        )
        descendants_result = self.relationship_manager.get_all_supported_descendants(
            target_claim_id
        )

        all_ancestor_ids = set(ancestors_result.visited_claims)
        all_descendant_ids = set(descendants_result.visited_claims)
        all_related_ids = all_ancestor_ids.union(all_descendant_ids)

        # Calculate coverage
        covered_ids = {claim.id for claim in included_claims}

        if not all_related_ids:
            return 1.0  # No relationships to cover

        coverage = len(covered_ids.intersection(all_related_ids)) / len(all_related_ids)
        return min(1.0, coverage)  # Cap at 1.0

    def build_batch_contexts(
        self, target_claim_ids: List[str], max_tokens_per_context: int = 8000
    ) -> List[BuiltContext]:
        """Build contexts for multiple target claims"""
        contexts = []

        for claim_id in target_claim_ids:
            try:
                context = self.build_complete_context(claim_id, max_tokens_per_context)
                contexts.append(context)
            except ValueError as e:
                # Skip invalid claim IDs
                continue

        return contexts

    def get_context_statistics(self) -> Dict[str, any]:
        """Get statistics about the claim collection for context building"""
        metrics = self.relationship_manager.get_relationship_metrics()

        return {
            "total_claims": metrics.total_claims,
            "total_relationships": metrics.total_relationships,
            "relationship_density": metrics.relationship_density,
            "max_depth": metrics.max_depth,
            "orphaned_claims": metrics.orphaned_claims,
            "cycles_detected": metrics.cycles_detected,
            "avg_tokens_per_claim": self.estimated_tokens_per_claim,
            "estimated_context_size": len(self.claims)
            * self.estimated_tokens_per_claim,
        }

    def refresh(self, new_claims: List[Claim]) -> None:
        """Refresh the context builder with new claim data"""
        self.claims = new_claims
        self.claim_index = create_claim_index(new_claims)
        self.relationship_manager.refresh(new_claims)

    def build_simple_context(
        self,
        max_tokens: int = 8000,
        include_core_tools: bool = True,
        additional_sections: Optional[List[str]] = None
    ) -> str:
        """
        Build a simple context with just Core Tools and optional additional sections.
        This is useful when you don't need the full claim relationship system.

        Args:
            max_tokens: Maximum tokens to use (default: 8000)
            include_core_tools: Whether to include Core Tools section
            additional_sections: Additional context sections to include

        Returns:
            Formatted context string
        """
        context_parts = []

        # Core Tools section
        if include_core_tools:
            if self.tool_registry:
                tools_context = self.tool_registry.get_core_tools_context()
            else:
                tools_context = "# Core Tools\n\nNo core tools available."
            context_parts.append(tools_context)
            context_parts.append("")
            context_parts.append("---")
            context_parts.append("")

        # Additional sections
        if additional_sections:
            for section in additional_sections:
                context_parts.append(section)
                context_parts.append("")

        # Add instructions for LLM response format
        context_parts.append("# Instructions")
        context_parts.append("Please respond with only JSON tool calls in this format:")
        context_parts.append('```json')
        context_parts.append('{"tool_calls": [{"name": "ToolName", "arguments": {...}}]}')
        context_parts.append('```')
        context_parts.append("")

        return "\n".join(context_parts)

    def get_tools_summary(self) -> Dict[str, Any]:
        """
        Get a summary of available tools for context building.

        Returns:
            Dictionary with tools information
        """
        if self.tool_registry:
            core_tools = self.tool_registry.core_tools
            optional_tools = self.tool_registry.optional_tools
            tools_context_length = len(self.tool_registry.get_core_tools_context())
        else:
            core_tools = {}
            optional_tools = {}
            tools_context_length =  0

        return {
            'core_tools_count': len(core_tools),
            'optional_tools_count': len(optional_tools),
            'total_tools': len(core_tools) + len(optional_tools),
            'core_tools_list': list(core_tools.keys()),
            'optional_tools_list': list(optional_tools.keys()),
            'tools_context_length': tools_context_length
        }
