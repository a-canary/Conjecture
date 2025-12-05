"""
Adaptive Compression Engine for Context Window Optimization

Implements dynamic context compression based on task complexity,
maintaining high reasoning quality while reducing token usage.
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from src.core.models import Claim
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels for compression decisions"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    ENTERPRISE = "enterprise"


@dataclass
class CompressionConfig:
    """Configuration for compression levels"""
    target_ratio: float  # Target compression ratio (0.0-1.0)
    min_confidence: float  # Minimum confidence threshold
    max_context_size: int  # Maximum context size in tokens


class AdaptiveCompressionEngine:
    """
    Dynamic context compression engine that adapts based on task complexity
    and context size to optimize token usage while maintaining reasoning quality.
    """
    
    def __init__(self):
        self.compression_configs = {
            TaskComplexity.SIMPLE: CompressionConfig(0.8, 0.9, 8000),
            TaskComplexity.MEDIUM: CompressionConfig(0.6, 0.8, 16000),
            TaskComplexity.COMPLEX: CompressionConfig(0.4, 0.7, 32000),
            TaskComplexity.ENTERPRISE: CompressionConfig(0.3, 0.6, 50000)
        }
        
        # Performance tracking
        self.compression_stats = {
            'tasks_analyzed': 0,
            'tokens_saved': 0,
            'quality_maintained': 0,
            'compression_applied': 0
        }
    
    def analyze_task_complexity(self, task: str, context_size: int) -> TaskComplexity:
        """
        Analyze task complexity based on multiple factors:
        - Context size (>8K, >20K, >50K tokens)
        - Task type indicators (factual, analytical, creative, technical)
        - Domain specificity (general vs specialized)
        - Question complexity (single vs multi-part)
        """
        
        complexity_score = 0
        
        # Factor 1: Context size
        if context_size > 50000:
            complexity_score += 3
        elif context_size > 20000:
            complexity_score += 2
        elif context_size > 8000:
            complexity_score += 1
        
        # Factor 2: Task type indicators
        task_lower = task.lower()
        
        # Analytical indicators
        analytical_words = ['analyze', 'compare', 'evaluate', 'assess', 'examine']
        if any(word in task_lower for word in analytical_words):
            complexity_score += 1
        
        # Technical indicators
        technical_words = ['implement', 'design', 'optimize', 'debug', 'architect']
        if any(word in task_lower for word in technical_words):
            complexity_score += 1
        
        # Creative/complex indicators
        creative_words = ['create', 'design', 'innovate', 'synthesize', 'integrate']
        if any(word in task_lower for word in creative_words):
            complexity_score += 1
        
        # Factor 3: Question complexity
        question_indicators = ['?', 'how', 'what', 'why', 'explain', 'describe']
        multi_part_indicators = ['and', 'or', 'but', 'however', 'additionally']
        
        question_count = sum(1 for word in question_indicators if word in task_lower)
        multi_part_count = sum(1 for word in multi_part_indicators if word in task_lower)
        
        if question_count > 1 or multi_part_count > 1:
            complexity_score += 1
        
        # Determine complexity level
        if complexity_score >= 5:
            return TaskComplexity.ENTERPRISE
        elif complexity_score >= 3:
            return TaskComplexity.COMPLEX
        elif complexity_score >= 2:
            return TaskComplexity.MEDIUM
        else:
            return TaskComplexity.SIMPLE
    
    def compress_context(self, context_claims: List[Claim], task: str, 
                       max_tokens: Optional[int] = None) -> Tuple[List[Claim], Dict[str, Any]]:
        """
        Compress context claims based on task complexity and available token budget.
        
        Returns:
        - Tuple of (compressed_claims, compression_metadata)
        """
        try:
            context_size = sum(len(claim.content) for claim in context_claims)
            task_complexity = self.analyze_task_complexity(task, context_size)
            
            config = self.compression_configs[task_complexity]
            
            # Calculate target compression
            target_claims = max(1, int(len(context_claims) * config.target_ratio))
            
            # Intelligent claim selection
            compressed_claims = self._select_optimal_claims(
                context_claims, task, target_claims, config.min_confidence
            )
            
            # Compression metadata
            compression_metadata = {
                'original_claim_count': len(context_claims),
                'compressed_claim_count': len(compressed_claims),
                'compression_ratio': len(compressed_claims) / len(context_claims),
                'task_complexity': task_complexity.value,
                'target_ratio': config.target_ratio,
                'min_confidence': config.min_confidence,
                'tokens_saved_estimate': self._estimate_tokens_saved(context_claims, compressed_claims),
                'quality_preservation': self._estimate_quality_preservation(context_claims, compressed_claims)
            }
            
            # Update statistics
            self.compression_stats['tasks_analyzed'] += 1
            self.compression_stats['compression_applied'] += 1
            self.compression_stats['tokens_saved'] += compression_metadata['tokens_saved_estimate']
            
            logger.info(f"Context compressed: {len(context_claims)} → {len(compressed_claims)} claims "
                       f"(ratio: {compression_metadata['compression_ratio']:.2f}, "
                       f"complexity: {task_complexity.value})")
            
            return compressed_claims, compression_metadata
            
        except Exception as e:
            logger.error(f"Context compression failed: {e}")
            return context_claims, {'error': str(e)}
    
    def _select_optimal_claims(self, claims: List[Claim], task: str, 
                           target_count: int, min_confidence: float) -> List[Claim]:
        """
        Select optimal claims using multi-factor scoring:
        - Relevance to task (40% weight)
        - Confidence score (20% weight)
        - Evidence strength (20% weight)
        - Recency (10% weight)
        - Cross-task transfer potential (10% weight)
        """
        
        if not claims:
            return []
        
        # Score each claim
        scored_claims = []
        task_lower = task.lower()
        
        for claim in claims:
            score = 0.0
            
            # Relevance scoring (40% weight)
            relevance_score = self._calculate_relevance(claim.content, task_lower)
            score += relevance_score * 0.4
            
            # Confidence scoring (20% weight)
            if claim.confidence >= min_confidence:
                score += (claim.confidence / 1.0) * 0.2
            
            # Evidence strength scoring (20% weight)
            evidence_score = self._calculate_evidence_strength(claim)
            score += evidence_score * 0.2
            
            # Recency scoring (10% weight)
            recency_score = self._calculate_recency_score(claim)
            score += recency_score * 0.1
            
            # Cross-task transfer scoring (10% weight)
            transfer_score = self._calculate_transfer_potential(claim, task_lower)
            score += transfer_score * 0.1
            
            scored_claims.append((claim, score))
        
        # Sort by score and select top claims
        scored_claims.sort(key=lambda x: x[1], reverse=True)
        selected_claims = [claim for claim, score in scored_claims[:target_count]]
        
        logger.info(f"Selected {len(selected_claims)} optimal claims from {len(claims)} available")
        
        return selected_claims
    
    def _calculate_relevance(self, claim_content: str, task: str) -> float:
        """Calculate relevance score between claim and task"""
        claim_words = set(claim_content.lower().split())
        task_words = set(task.split())
        
        # Jaccard similarity
        intersection = len(claim_words & task_words)
        union = len(claim_words | task_words)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _calculate_evidence_strength(self, claim: Claim) -> float:
        """Calculate evidence strength based on claim metadata"""
        strength = 0.0
        
        # Check for supporting evidence
        if hasattr(claim, 'supports') and claim.supports:
            strength += min(1.0, len(claim.supports) * 0.2)
        
        # Check for tags indicating evidence quality
        if hasattr(claim, 'tags') and claim.tags:
            evidence_tags = ['verified', 'sourced', 'validated', 'evidence_based']
            evidence_count = sum(1 for tag in evidence_tags if tag in claim.tags)
            strength += evidence_count * 0.1
        
        # Check content length (longer claims may have more evidence)
        if len(claim.content) > 200:
            strength += 0.1
        elif len(claim.content) > 100:
            strength += 0.05
        
        return min(1.0, strength)
    
    def _calculate_recency_score(self, claim: Claim) -> float:
        """Calculate recency score (newer claims get higher scores)"""
        if hasattr(claim, 'created'):
            # Simple recency scoring based on creation date
            # This would need actual date comparison in real implementation
            return 0.5  # Neutral score for now
        return 0.3  # Default for claims without timestamp
    
    def _calculate_transfer_potential(self, claim: Claim, task: str) -> float:
        """Calculate cross-task transfer potential"""
        claim_lower = claim.content.lower()
        
        # General concepts have higher transfer potential
        general_indicators = ['principle', 'concept', 'method', 'approach', 'framework']
        if any(indicator in claim_lower for indicator in general_indicators):
            return 0.8
        
        # Specific concepts have lower transfer potential
        specific_indicators = ['specific', 'particular', 'unique', 'specialized']
        if any(indicator in claim_lower for indicator in specific_indicators):
            return 0.2
        
        return 0.5  # Default
    
    def _estimate_tokens_saved(self, original_claims: List[Claim], 
                            compressed_claims: List[Claim]) -> int:
        """Estimate tokens saved through compression"""
        original_tokens = sum(len(claim.content.split()) for claim in original_claims)
        compressed_tokens = sum(len(claim.content.split()) for claim in compressed_claims)
        return max(0, original_tokens - compressed_tokens)
    
    def _estimate_quality_preservation(self, original_claims: List[Claim], 
                                   compressed_claims: List[Claim]) -> float:
        """Estimate quality preservation ratio"""
        if not original_claims:
            return 1.0
        
        # Quality based on confidence scores
        original_quality = sum(claim.confidence for claim in original_claims) / len(original_claims)
        compressed_quality = sum(claim.confidence for claim in compressed_claims) / len(compressed_claims)
        
        return compressed_quality / max(0.01, original_quality)
    
    def get_compression_statistics(self) -> Dict[str, Any]:
        """Get compression performance statistics"""
        return self.compression_stats.copy()
    
    def reset_statistics(self):
        """Reset compression statistics"""
        self.compression_stats = {
            'tasks_analyzed': 0,
            'tokens_saved': 0,
            'quality_maintained': 0,
            'compression_applied': 0
        }


# Global instance for reuse
_adaptive_compressor = None

def get_adaptive_compressor() -> AdaptiveCompressionEngine:
    """Get or create the adaptive compressor instance"""
    global _adaptive_compressor
    if _adaptive_compressor is None:
        _adaptive_compressor = AdaptiveCompressionEngine()
    return _adaptive_compressor


def compress_context_adaptive(context_claims: List[Claim], task: str, 
                           max_tokens: Optional[int] = None) -> Tuple[List[Claim], Dict[str, Any]]:
    """
    Convenience function for adaptive context compression.
    
    Args:
        context_claims: List of claims to compress
        task: Task description for complexity analysis
        max_tokens: Maximum token budget (optional)
    
    Returns:
        Tuple of (compressed_claims, compression_metadata)
    """
    compressor = get_adaptive_compressor()
    return compressor.compress_context(context_claims, task, max_tokens)