"""
Hierarchical Context Processor for Multi-Level Context Summarization

Implements progressive disclosure mechanisms for large contexts
with hierarchical summarization levels.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.core.models import Claim
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ContextLevel(Enum):
    """Context processing levels"""
    EXECUTIVE = "summary"          # 2-3 sentences, highest level
    KEY_CLAIMS = "key_claims"     # Key claims with confidence
    DETAILED_EVIDENCE = "detailed_evidence"  # Evidence with sources
    FULL_CONTEXT = "full_context"      # Complete context (accessed on-demand)


@dataclass
class LevelConfig:
    """Configuration for each context level"""
    max_claims: int           # Maximum claims at this level
    min_confidence: float     # Minimum confidence threshold
    compression_ratio: float     # Compression ratio for this level
    description: str          # Human-readable description


class HierarchicalContextProcessor:
    """
    Processes large contexts using hierarchical summarization
    with progressive disclosure mechanisms.
    """
    
    def __init__(self):
        self.level_configs = {
            ContextLevel.EXECUTIVE: LevelConfig(3, 0.9, 0.1, "Executive summary (2-3 sentences)"),
            ContextLevel.KEY_CLAIMS: LevelConfig(8, 0.8, 0.3, "Key claims with confidence scores"),
            ContextLevel.DETAILED_EVIDENCE: LevelConfig(15, 0.7, 0.6, "Detailed evidence with sources"),
            ContextLevel.FULL_CONTEXT: LevelConfig(50, 0.6, 0.9, "Complete context (on-demand access)")
        }
        
        # Performance tracking
        self.processing_stats = {
            'contexts_processed': 0,
            'levels_generated': 0,
            'on_demand_accesses': 0,
            'compression_achieved': 0
        }
    
    def process_large_context(self, context_claims: List[Claim], task_complexity: str) -> Dict[str, Any]:
        """
        Process large context using hierarchical summarization.
        
        Returns:
        Dictionary with processed levels and metadata
        """
        try:
            # Sort claims by confidence and relevance
            sorted_claims = self._sort_claims_by_importance(context_claims)
            
            # Generate hierarchical levels
            processed_context = {}
            
            for level in ContextLevel:
                config = self.level_configs[level]
                level_claims = self._generate_level_claims(
                    sorted_claims, level, config, task_complexity
                )
                
                processed_context[level.value] = {
                    'claims': [self._claim_to_dict(claim) for claim in level_claims],
                    'metadata': {
                        'level': level.value,
                        'claim_count': len(level_claims),
                        'avg_confidence': sum(c.confidence for c in level_claims) / len(level_claims) if level_claims else 0,
                        'compression_ratio': config.compression_ratio,
                        'description': config.description
                    }
                }
            
            # Add navigation and access metadata
            processed_context['navigation'] = {
                'total_claims': len(context_claims),
                'levels_available': list(ContextLevel),
                'access_pattern': 'progressive_disclosure',
                'task_complexity': task_complexity
            }
            
            # Update statistics
            self.processing_stats['contexts_processed'] += 1
            self.processing_stats['levels_generated'] += len(ContextLevel)
            
            logger.info(f"Hierarchical context processed: {len(context_claims)} claims → "
                       f"{len(ContextLevel)} levels")
            
            return processed_context
            
        except Exception as e:
            logger.error(f"Hierarchical context processing failed: {e}")
            return {'error': str(e)}
    
    def access_level(self, processed_context: Dict[str, Any], level: ContextLevel, 
                  start_index: int = 0, count: int = 5) -> List[Dict[str, Any]]:
        """
        Access specific level with progressive disclosure.
        
        Args:
        processed_context: Previously processed hierarchical context
        level: Context level to access
        start_index: Starting index for claims
        count: Number of claims to return
        
        Returns:
        List of claim dictionaries for requested level
        """
        try:
            if level.value not in processed_context:
                logger.warning(f"Level {level.value} not found in processed context")
                return []
            
            level_data = processed_context[level.value]
            available_claims = level_data['claims']
            
            # Progressive disclosure
            end_index = min(start_index + count, len(available_claims))
            selected_claims = available_claims[start_index:end_index]
            
            # Update on-demand access statistics
            self.processing_stats['on_demand_accesses'] += 1
            
            logger.info(f"Level {level.value} accessed: {len(selected_claims)} claims "
                       f"(indices {start_index}-{end_index-1})")
            
            return selected_claims
            
        except Exception as e:
            logger.error(f"Level access failed: {e}")
            return []
    
    def _sort_claims_by_importance(self, claims: List[Claim]) -> List[Claim]:
        """Sort claims by confidence, recency, and evidence strength"""
        def claim_score(claim):
            score = 0.0
            
            # Confidence score (40% weight)
            score += claim.confidence * 0.4
            
            # Evidence strength (30% weight)
            if hasattr(claim, 'supports') and claim.supports:
                score += min(1.0, len(claim.supports) * 0.1)
            
            # Content quality (20% weight)
            if len(claim.content) > 100:
                score += 0.1
            elif len(claim.content) > 50:
                score += 0.05
            
            # Recency (10% weight)
            if hasattr(claim, 'created'):
                # Simple recency scoring
                score += 0.1
            
            return score
        
        return sorted(claims, key=claim_score, reverse=True)
    
    def _generate_level_claims(self, sorted_claims: List[Claim], level: ContextLevel, 
                           config: LevelConfig, task_complexity: str) -> List[Claim]:
        """Generate claims for specific hierarchical level"""
        # Filter by confidence threshold
        eligible_claims = [
            claim for claim in sorted_claims 
            if claim.confidence >= config.min_confidence
        ]
        
        # Apply compression ratio
        target_count = min(config.max_claims, int(len(eligible_claims) * config.compression_ratio))
        
        # Task complexity adjustment
        if task_complexity.lower() in ['complex', 'enterprise']:
            target_count = max(1, int(target_count * 0.8))  # Reduce for complex tasks
        
        return eligible_claims[:target_count]
    
    def _claim_to_dict(self, claim: Claim) -> Dict[str, Any]:
        """Convert claim to dictionary for JSON serialization"""
        return {
            'id': claim.id,
            'content': claim.content,
            'confidence': claim.confidence,
            'type': [t.value for t in claim.type] if claim.type else [],
            'state': claim.state.value if claim.state else None,
            'tags': claim.tags if claim.tags else [],
            'supports': claim.supports if hasattr(claim, 'supports') else [],
            'created': claim.created.isoformat() if hasattr(claim, 'created') else None
        }
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get hierarchical processing statistics"""
        return self.processing_stats.copy()
    
    def reset_statistics(self):
        """Reset processing statistics"""
        self.processing_stats = {
            'contexts_processed': 0,
            'levels_generated': 0,
            'on_demand_accesses': 0,
            'compression_achieved': 0
        }


# Global instance for reuse
_hierarchical_processor = None

def get_hierarchical_processor() -> HierarchicalContextProcessor:
    """Get or create hierarchical processor instance"""
    global _hierarchical_processor
    if _hierarchical_processor is None:
        _hierarchical_processor = HierarchicalContextProcessor()
    return _hierarchical_processor


def process_context_hierarchical(context_claims: List[Claim], task_complexity: str = "medium") -> Dict[str, Any]:
    """
    Convenience function for hierarchical context processing.
    
    Args:
        context_claims: List of claims to process
        task_complexity: Complexity level for processing decisions
    
    Returns:
        Dictionary with processed hierarchical context
    """
    processor = get_hierarchical_processor()
    return processor.process_large_context(context_claims, task_complexity)


def access_context_level(processed_context: Dict[str, Any], level: str, 
                     start_index: int = 0, count: int = 5) -> List[Dict[str, Any]]:
    """
    Convenience function for accessing specific context level.
    
    Args:
        processed_context: Previously processed hierarchical context
        level: Context level to access ('summary', 'key_claims', 'detailed_evidence', 'full_context')
        start_index: Starting index for claims
        count: Number of claims to return
    
    Returns:
        List of claim dictionaries for requested level
    """
    from .hierarchical_context_processor import ContextLevel
    
    try:
        context_level = ContextLevel(level)
        processor = get_hierarchical_processor()
        return processor.access_level(processed_context, context_level, start_index, count)
    except ValueError:
        # Fallback to key_claims if invalid level
        processor = get_hierarchical_processor()
        return processor.access_level(processed_context, ContextLevel.KEY_CLAIMS, start_index, count)