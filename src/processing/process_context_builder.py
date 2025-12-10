"""
ProcessContextBuilder - Context building for Process Layer

This module implements the ProcessContextBuilder class which handles
context building logic for the Process Layer, integrating with the
existing DataManager and following the architecture spec for graph traversal.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from ..core.models import Claim, ClaimFilter, ClaimState
from ..data.data_manager import DataManager

logger = logging.getLogger(__name__)

class ProcessContextBuilder:
    """
    Context builder for Process Layer operations.
    
    This class handles building context for claim processing by
    retrieving relevant claims from the data layer and organizing
    them for optimal LLM processing.
    """
    
    def __init__(self, data_manager: DataManager):
        """
        Initialize the ProcessContextBuilder.
        
        Args:
            data_manager: The DataManager instance for data operations
        """
        self.data_manager = data_manager
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the context builder"""
        if not self.data_manager:
            raise ValueError("DataManager is required")
        
        self._initialized = True
        logger.info("ProcessContextBuilder initialized")
    
    async def build_context_for_claim_creation(
        self,
        content: str,
        confidence: float,
        tags: Optional[List[str]] = None,
        max_context_size: int = 10,
        similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Build context for creating a new claim.
        
        This method retrieves relevant existing claims to provide context
        for the new claim creation process.
        
        Args:
            content: The content of the new claim
            confidence: The confidence score for the new claim
            tags: Optional tags for the new claim
            max_context_size: Maximum number of related claims to include
            similarity_threshold: Minimum similarity threshold for related claims
            
        Returns:
            Dictionary containing the built context
        """
        if not self._initialized:
            raise RuntimeError("ProcessContextBuilder not initialized")
        
        try:
            # Search for similar existing claims
            similar_claims = await self.data_manager.search_claims(
                query=content,
                limit=max_context_size,
                use_vector_search=True
            )
            
            # Filter by confidence threshold
            high_confidence_claims = [
                claim for claim in similar_claims
                if claim.get("confidence", 0) >= similarity_threshold
            ]
            
            # Get recent claims for additional context
            recent_filter = ClaimFilter(
                limit=max_context_size // 2,
                confidence_min=0.5
            )
            recent_claims = await self.data_manager.filter_claims(recent_filter)
            
            # Format claims for context
            formatted_similar = [
                self._format_claim_for_context(claim)
                for claim in high_confidence_claims[:max_context_size // 2]
            ]
            
            formatted_recent = [
                self._format_claim_for_context(claim)
                for claim in recent_claims[:max_context_size // 2]
            ]
            
            context = {
                "operation": "create_claim",
                "input_content": content,
                "input_confidence": confidence,
                "input_tags": tags or [],
                "similar_claims": formatted_similar,
                "recent_claims": formatted_recent,
                "total_context_claims": len(formatted_similar) + len(formatted_recent),
                "built_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Built context for claim creation: {context['total_context_claims']} related claims")
            return context
            
        except Exception as e:
            logger.error(f"Failed to build context for claim creation: {e}")
            raise
    
    async def build_context_for_claim_analysis(
        self,
        claim_id: str,
        max_context_size: int = 10
    ) -> Dict[str, Any]:
        """
        Build context for analyzing an existing claim.
        
        Args:
            claim_id: The ID of the claim to analyze
            max_context_size: Maximum number of related claims to include
            
        Returns:
            Dictionary containing the built context
        """
        if not self._initialized:
            raise RuntimeError("ProcessContextBuilder not initialized")
        
        try:
            # Get the target claim
            target_claim = await self.data_manager.get_claim(claim_id)
            if not target_claim:
                raise ValueError(f"Claim {claim_id} not found")
            
            # Find similar claims
            similar_claims = await self.data_manager.find_similar_claims(
                claim_id=claim_id,
                limit=max_context_size,
                similarity_threshold=0.6
            )
            
            # Get related claims (supports/supported_by)
            related_claim_ids = set(target_claim.supports + target_claim.supported_by)
            related_claims = []
            
            for related_id in related_claim_ids:
                related_claim = await self.data_manager.get_claim(related_id)
                if related_claim:
                    related_claims.append(related_claim.model_dump())
            
            # Format claims for context
            formatted_target = self._format_claim_for_context(target_claim.model_dump())
            formatted_similar = [
                self._format_claim_for_context(claim)
                for claim in similar_claims[:max_context_size // 2]
            ]
            formatted_related = [
                self._format_claim_for_context(claim)
                for claim in related_claims[:max_context_size // 2]
            ]
            
            context = {
                "operation": "analyze_claim",
                "target_claim": formatted_target,
                "similar_claims": formatted_similar,
                "related_claims": formatted_related,
                "total_context_claims": len(formatted_similar) + len(formatted_related),
                "built_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Built context for claim analysis: {context['total_context_claims']} related claims")
            return context
            
        except Exception as e:
            logger.error(f"Failed to build context for claim analysis: {e}")
            raise
    
    def _format_claim_for_context(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format claim data for LLM context.
        
        Args:
            claim_data: Raw claim data from data layer
            
        Returns:
            Formatted claim data suitable for LLM processing
        """
        return {
            "id": claim_data.get("id"),
            "content": claim_data.get("content"),
            "confidence": claim_data.get("confidence"),
            "state": claim_data.get("state"),
            "tags": claim_data.get("tags", []),
            "supports": claim_data.get("supports", []),
            "supported_by": claim_data.get("supported_by", [])
        }
    
    async def validate_context(self, context: Dict[str, Any]) -> bool:
        """
        Validate that a context dictionary is properly formed.
        
        Args:
            context: The context dictionary to validate
            
        Returns:
            True if context is valid, False otherwise
        """
        required_fields = ["operation", "built_at"]
        
        for field in required_fields:
            if field not in context:
                logger.error(f"Missing required field in context: {field}")
                return False
        
        # Validate operation-specific fields
        operation = context["operation"]
        
        if operation == "create_claim":
            required_create_fields = ["input_content", "input_confidence"]
            for field in required_create_fields:
                if field not in context:
                    logger.error(f"Missing required field for create_claim: {field}")
                    return False
        
        elif operation == "analyze_claim":
            required_analyze_fields = ["target_claim"]
            for field in required_analyze_fields:
                if field not in context:
                    logger.error(f"Missing required field for analyze_claim: {field}")
                    return False
        
        return True
    
    def get_context_stats(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get statistics about a context dictionary.
        
        Args:
            context: The context dictionary to analyze
            
        Returns:
            Dictionary containing context statistics
        """
        stats = {
            "operation": context.get("operation", "unknown"),
            "total_claims": context.get("total_context_claims", 0),
            "similar_claims": len(context.get("similar_claims", [])),
            "recent_claims": len(context.get("recent_claims", [])),
            "related_claims": len(context.get("related_claims", [])),
            "built_at": context.get("built_at")
        }
        
        return stats