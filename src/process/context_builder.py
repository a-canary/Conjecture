"""
Process Context Builder

The ProcessContextBuilder is responsible for traversing claim graphs and building
processing contexts for claim evaluation and instruction identification.
"""

from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import asyncio
import logging

from src.core.models import Claim, ClaimType, ClaimState
from src.data.repositories import ClaimRepository
from .models import ContextResult, ProcessingConfig

logger = logging.getLogger(__name__)


class ProcessContextBuilder:
    """
    Builds processing contexts by traversing claim graphs and collecting
    relevant claims for evaluation and instruction identification.
    
    This class serves as the bridge between the Data Layer (claim storage)
    and the Process Layer (claim evaluation), providing intelligent context
    building capabilities that consider claim relationships, types, and states.
    """
    
    def __init__(
        self,
        claim_repository: ClaimRepository,
        config: Optional[ProcessingConfig] = None
    ):
        """
        Initialize the ProcessContextBuilder.
        
        Args:
            claim_repository: Repository for accessing claim data
            config: Processing configuration (optional)
        """
        self.claim_repository = claim_repository
        self.config = config or ProcessingConfig()
        self._context_cache: Dict[str, ContextResult] = {}
        
    async def build_context(
        self,
        claim_id: str,
        max_depth: Optional[int] = None,
        context_hints: Optional[List[str]] = None
    ) -> ContextResult:
        """
        Build a processing context for the given claim.
        
        Args:
            claim_id: ID of the primary claim to build context for
            max_depth: Maximum traversal depth (overrides config)
            context_hints: Optional hints for context building
            
        Returns:
            ContextResult containing the built context and metadata
            
        Raises:
            ValueError: If claim_id is not found
            RuntimeError: If context building fails
        """
        start_time = datetime.utcnow()
        
        try:
            # Get the primary claim
            primary_claim = await self.claim_repository.get_by_id(claim_id)
            if not primary_claim:
                raise ValueError(f"Claim not found: {claim_id}")
            
            # Determine traversal parameters
            max_depth = max_depth or self.config.max_traversal_depth
            
            # Build context through graph traversal
            context_claims = await self._traverse_claim_graph(
                primary_claim, 
                max_depth, 
                context_hints or []
            )
            
            # Calculate context size (simplified token estimation)
            context_size = self._estimate_context_size(context_claims)
            
            # Create result
            build_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            result = ContextResult(
                claim_id=claim_id,
                context_claims=context_claims,
                context_size=context_size,
                traversal_depth=max_depth,
                build_time_ms=build_time_ms,
                metadata={
                    "hints_used": context_hints or [],
                    "cache_hit": claim_id in self._context_cache,
                    "build_strategy": "graph_traversal"
                }
            )
            
            # Cache the result
            self._context_cache[claim_id] = result
            
            logger.info(f"Built context for claim {claim_id}: {len(context_claims)} claims, {context_size} tokens")
            return result
            
        except Exception as e:
            logger.error(f"Failed to build context for claim {claim_id}: {str(e)}")
            raise RuntimeError(f"Context building failed: {str(e)}")
    
    async def _traverse_claim_graph(
        self,
        primary_claim: Claim,
        max_depth: int,
        context_hints: List[str]
    ) -> List[Claim]:
        """
        Traverse the claim graph to collect relevant context claims.
        
        Args:
            primary_claim: The primary claim to build context for
            max_depth: Maximum traversal depth
            context_hints: Hints for guiding the traversal
            
        Returns:
            List of claims in the context
        """
        context_claims = [primary_claim]
        visited_ids: Set[str] = {primary_claim.id}
        
        # Implement breadth-first traversal
        current_level = [primary_claim]
        
        for depth in range(max_depth):
            if not current_level:
                break
                
            next_level = []
            
            for claim in current_level:
                # Get related claims (implementation depends on data layer)
                related_claims = await self._get_related_claims(claim, context_hints)
                
                for related_claim in related_claims:
                    if related_claim.id not in visited_ids:
                        visited_ids.add(related_claim.id)
                        context_claims.append(related_claim)
                        next_level.append(related_claim)
                        
                        # Check if we've reached the maximum context size
                        if len(context_claims) >= self.config.max_context_size:
                            return context_claims
            
            current_level = next_level
        
        return context_claims
    
    async def _get_related_claims(
        self,
        claim: Claim,
        context_hints: List[str]
    ) -> List[Claim]:
        """
        Get claims related to the given claim.
        
        Args:
            claim: The claim to find related claims for
            context_hints: Hints for finding relevant claims
            
        Returns:
            List of related claims
        """
        # This is a skeleton implementation
        # In a full implementation, this would consider:
        # - Direct references/connections
        # - Similar claim types
        # - Text similarity
        # - Context hints
        
        related_claims = []
        
        # Example: Find claims by type similarity
        try:
            # This would be implemented based on the actual data layer API
            # For now, return empty list as skeleton
            pass
        except Exception as e:
            logger.warning(f"Failed to get related claims for {claim.id}: {str(e)}")
        
        return related_claims
    
    def _estimate_context_size(self, claims: List[Claim]) -> int:
        """
        Estimate the context size in tokens for the given claims.
        
        Args:
            claims: List of claims to estimate size for
            
        Returns:
            Estimated token count
        """
        # Simplified token estimation (rough approximation)
        # In practice, this would use actual tokenization
        total_chars = sum(len(claim.text) for claim in claims)
        # Rough approximation: ~4 characters per token
        return total_chars // 4
    
    def clear_cache(self) -> None:
        """Clear the context cache."""
        self._context_cache.clear()
        logger.info("Context cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the context cache.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "cached_contexts": len(self._context_cache),
            "cache_keys": list(self._context_cache.keys())
        }
    
    async def build_batch_contexts(
        self,
        claim_ids: List[str],
        max_depth: Optional[int] = None
    ) -> List[ContextResult]:
        """
        Build contexts for multiple claims in parallel.
        
        Args:
            claim_ids: List of claim IDs to build contexts for
            max_depth: Maximum traversal depth
            
        Returns:
            List of ContextResult objects
        """
        if not self.config.enable_parallel_processing:
            # Sequential processing
            results = []
            for claim_id in claim_ids:
                result = await self.build_context(claim_id, max_depth)
                results.append(result)
            return results
        
        # Parallel processing
        tasks = [
            self.build_context(claim_id, max_depth) 
            for claim_id in claim_ids
        ]
        
        return await asyncio.gather(*tasks, return_exceptions=True)