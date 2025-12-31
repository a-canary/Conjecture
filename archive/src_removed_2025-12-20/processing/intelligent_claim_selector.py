"""Intelligent Claim Selector for Context-Aware Claim Ranking

Implements multi-factor importance scoring for optimal claim selection
with context-aware ranking and dynamic selection algorithms.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
from datetime import datetime, timedelta

from src.core.models import Claim
from src.utils.logging import get_logger

logger = get_logger(__name__)

class ClaimImportance(Enum):
    """Claim importance categories"""
    CRITICAL = "critical"      # Essential for task completion
    HIGH = "high"           # Strongly relevant, high confidence
    MEDIUM = "medium"         # Moderately relevant
    LOW = "low"             # Weakly relevant, low confidence
    BACKGROUND = "background"    # Contextual information only

@dataclass
class SelectionCriteria:
    """Configuration for claim selection"""
    task_relevance_weight: float = 0.4      # Relevance to current task
    confidence_weight: float = 0.2        # Confidence score
    evidence_weight: float = 0.2           # Supporting evidence
    recency_weight: float = 0.1           # Temporal relevance
    transfer_weight: float = 0.1           # Cross-task transfer potential

@dataclass
class SelectionResult:
    """Result of claim selection process"""
    selected_claims: List[Claim]
    selection_metadata: Dict[str, Any]
    performance_metrics: Dict[str, float]

class IntelligentClaimSelector:
    """
    Intelligent claim selector using multi-factor importance scoring
    with context-aware ranking and dynamic selection algorithms.
    """
    
    def __init__(self):
        self.criteria = SelectionCriteria()
        self.selection_stats = {
            'selections_performed': 0,
            'claims_analyzed': 0,
            'avg_selection_time': 0.0,
            'top_claim_accuracy': 0.0
        }
    
    def select_optimal_claims(self, context_claims: List[Claim], task: str, 
                           context_limit: int, selection_strategy: str = "balanced") -> SelectionResult:
        """
        Select optimal claims using multi-factor scoring and context-aware ranking.
        
        Args:
            context_claims: Available claims to select from
            task: Current task description
            context_limit: Maximum number of claims to select
            selection_strategy: Strategy for selection ('balanced', 'quality_first', 'diverse')
        
        Returns:
            SelectionResult with selected claims and metadata
        """
        try:
            start_time = datetime.now()
            
            # Score all claims
            scored_claims = self._score_claims(context_claims, task)
            
            # Apply selection strategy
            if selection_strategy == "quality_first":
                selected_claims = self._select_quality_first(scored_claims, context_limit)
            elif selection_strategy == "diverse":
                selected_claims = self._select_diverse(scored_claims, context_limit)
            else:  # balanced (default)
                selected_claims = self._select_balanced(scored_claims, context_limit)
            
            # Calculate performance metrics
            selection_time = (datetime.now() - start_time).total_seconds()
            top_claim_accuracy = self._calculate_top_claim_accuracy(selected_claims, task)
            
            # Create selection metadata
            selection_metadata = {
                'strategy': selection_strategy,
                'total_claims_analyzed': len(context_claims),
                'claims_selected': len(selected_claims),
                'selection_time_seconds': selection_time,
                'avg_score': sum(score for _, score in scored_claims) / len(scored_claims) if scored_claims else 0,
                'top_claim_accuracy': top_claim_accuracy,
                'diversity_score': self._calculate_diversity_score(selected_claims),
                'quality_distribution': self._analyze_quality_distribution(selected_claims)
            }
            
            # Update statistics
            self.selection_stats['selections_performed'] += 1
            self.selection_stats['claims_analyzed'] += len(context_claims)
            self.selection_stats['avg_selection_time'] = (
                (self.selection_stats['avg_selection_time'] * (self.selection_stats['selections_performed'] - 1) + selection_time) 
                / self.selection_stats['selections_performed']
            )
            self.selection_stats['top_claim_accuracy'] = (
                (self.selection_stats['top_claim_accuracy'] * (self.selection_stats['selections_performed'] - 1) + top_claim_accuracy) 
                / self.selection_stats['selections_performed']
            )
            
            logger.info(f"Selected {len(selected_claims)} claims using {selection_strategy} strategy "
                       f"(from {len(context_claims)} available)")
            
            return SelectionResult(selected_claims, selection_metadata, {
                'selection_time': selection_time,
                'top_claim_accuracy': top_claim_accuracy,
                'diversity_score': selection_metadata['diversity_score']
            })
            
        except Exception as e:
            logger.error(f"Claim selection failed: {e}")
            return SelectionResult([], {'error': str(e)}, {})
    
    def _score_claims(self, claims: List[Claim], task: str) -> List[Tuple[Claim, float]]:
        """Score claims using multi-factor importance scoring"""
        scored_claims = []
        task_lower = task.lower()
        task_words = set(task.split())
        
        for claim in claims:
            total_score = 0.0
            
            # Task relevance scoring (40% weight)
            relevance_score = self._calculate_task_relevance(claim.content, task_lower, task_words)
            total_score += relevance_score * self.criteria.task_relevance_weight
            
            # Confidence scoring (20% weight)
            confidence_score = claim.confidence
            total_score += confidence_score * self.criteria.confidence_weight
            
            # Evidence strength scoring (20% weight)
            evidence_score = self._calculate_evidence_strength(claim)
            total_score += evidence_score * self.criteria.evidence_weight
            
            # Recency scoring (10% weight)
            recency_score = self._calculate_recency_score(claim)
            total_score += recency_score * self.criteria.recency_weight
            
            # Cross-task transfer scoring (10% weight)
            transfer_score = self._calculate_transfer_potential(claim.content, task_lower)
            total_score += transfer_score * self.criteria.transfer_weight
            
            scored_claims.append((claim, total_score))
        
        return scored_claims
    
    def _calculate_task_relevance(self, claim_content: str, task: str, task_words: set) -> float:
        """Calculate relevance score between claim and task"""
        claim_words = set(claim_content.lower().split())
        
        # Jaccard similarity for word overlap
        intersection = len(claim_words & task_words)
        union = len(claim_words | task_words)
        
        if union == 0:
            return 0.0
        
        jaccard_similarity = intersection / union
        
        # Boost for exact phrase matches
        claim_lower = claim_content.lower()
        for word in task.split():
            if word.lower() in claim_lower:
                jaccard_similarity += 0.2
        
        # Boost for domain-specific terms
        domain_terms = ['principle', 'method', 'approach', 'framework', 'algorithm', 'strategy']
        domain_boost = sum(0.1 for term in domain_terms if term in claim_lower)
        
        return min(1.0, jaccard_similarity + domain_boost)
    
    def _calculate_evidence_strength(self, claim: Claim) -> float:
        """Calculate evidence strength based on claim metadata"""
        strength = 0.0
        
        # Check for supporting evidence
        if hasattr(claim, 'supports') and claim.supports:
            strength += min(1.0, len(claim.supports) * 0.15)
        
        # Check for evidence tags
        if hasattr(claim, 'tags') and claim.tags:
            evidence_tags = ['verified', 'sourced', 'validated', 'evidence_based', 'proven']
            evidence_count = sum(1 for tag in evidence_tags if tag in claim.tags)
            strength += evidence_count * 0.1
        
        # Check content quality indicators
        content_indicators = ['data shows', 'research indicates', 'analysis reveals', 'study confirms']
        for indicator in content_indicators:
            if indicator in claim.content.lower():
                strength += 0.05
        
        # Check for quantitative evidence
        if re.search(r'\d+%', claim.content) or re.search(r'\d+\.\d+', claim.content):
            strength += 0.1
        
        return min(1.0, strength)
    
    def _calculate_recency_score(self, claim: Claim) -> float:
        """Calculate recency score (newer claims get higher scores)"""
        if hasattr(claim, 'created') and claim.created:
            # Calculate days since creation
            days_old = (datetime.now() - claim.created).days
            
            # Exponential decay for recency
            if days_old <= 1:
                return 1.0  # Very recent
            elif days_old <= 7:
                return 0.8  # Recent
            elif days_old <= 30:
                return 0.6  # Moderately recent
            elif days_old <= 90:
                return 0.4  # Old
            else:
                return 0.2  # Very old
        
        return 0.5  # Default for claims without timestamp
    
    def _calculate_transfer_potential(self, claim_content: str, task: str) -> float:
        """Calculate cross-task transfer potential"""
        claim_lower = claim_content.lower()
        
        # General concepts have high transfer potential
        general_indicators = ['principle', 'concept', 'method', 'approach', 'framework', 'theory', 'model']
        if any(indicator in claim_lower for indicator in general_indicators):
            return 0.8
        
        # Specific implementations have medium transfer potential
        specific_indicators = ['implementation', 'technique', 'algorithm', 'procedure', 'process']
        if any(indicator in claim_lower for indicator in specific_indicators):
            return 0.6
        
        # Domain-specific knowledge has lower transfer potential
        domain_indicators = ['specific', 'particular', 'specialized', 'unique', 'custom']
        if any(indicator in claim_lower for indicator in domain_indicators):
            return 0.4
        
        return 0.5  # Default
    
    def _select_quality_first(self, scored_claims: List[Tuple[Claim, float]], limit: int) -> List[Claim]:
        """Select claims prioritizing quality over diversity"""
        # Sort by score and take top claims
        scored_claims.sort(key=lambda x: x[1], reverse=True)
        return [claim for claim, score in scored_claims[:limit]]
    
    def _select_diverse(self, scored_claims: List[Tuple[Claim, float]], limit: int) -> List[Claim]:
        """Select diverse claims covering different aspects"""
        if limit <= 1:
            # Single claim - take highest scored
            scored_claims.sort(key=lambda x: x[1], reverse=True)
            return [claim for claim, score in scored_claims[:limit]]
        
        # Group by claim types for diversity
        type_groups = {}
        for claim, score in scored_claims:
            claim_type = claim.type[0].value if claim.type else 'unknown'
            if claim_type not in type_groups:
                type_groups[claim_type] = []
            type_groups[claim_type].append((claim, score))
        
        selected_claims = []
        claims_per_type = max(1, limit // len(type_groups))
        
        # Select top claims from each type group
        for claim_type, claims in type_groups.items():
            claims.sort(key=lambda x: x[1], reverse=True)
            selected_claims.extend([claim for claim, score in claims[:claims_per_type]])
        
        # Fill remaining slots with highest scored claims
        while len(selected_claims) < limit and scored_claims:
            best_claim, best_score = max(scored_claims, key=lambda x: x[1])
            if best_claim not in selected_claims:
                selected_claims.append(best_claim)
                scored_claims.remove((best_claim, best_score))
        
        return selected_claims[:limit]
    
    def _select_balanced(self, scored_claims: List[Tuple[Claim, float]], limit: int) -> List[Claim]:
        """Select claims with balance between quality and diversity"""
        # Sort by score but ensure diversity
        scored_claims.sort(key=lambda x: x[1], reverse=True)
        selected_claims = []
        
        # Take top claim (highest quality)
        if scored_claims:
            selected_claims.append(scored_claims[0][0])
            scored_claims.pop(0)
        
        # Select remaining claims ensuring type diversity
        used_types = set()
        if selected_claims:
            used_types.add(selected_claims[0].type[0].value if selected_claims[0].type else 'unknown')
        
        while len(selected_claims) < limit and scored_claims:
            best_claim, best_score = None, 0
            
            # Find best claim that adds type diversity
            for i, (claim, score) in enumerate(scored_claims):
                claim_type = claim.type[0].value if claim.type else 'unknown'
                
                # Prefer claims with new types
                if claim_type not in used_types:
                    diversity_bonus = 0.3
                else:
                    diversity_bonus = 0.0
                
                adjusted_score = score + diversity_bonus
                
                if adjusted_score > best_score:
                    best_claim, best_score = claim, adjusted_score
            
            if best_claim:
                selected_claims.append(best_claim)
                scored_claims.remove((best_claim, best_score))
                used_types.add(best_claim.type[0].value if best_claim.type else 'unknown')
        
        return selected_claims
    
    def _calculate_top_claim_accuracy(self, selected_claims: List[Claim], task: str) -> float:
        """Calculate accuracy of top selected claim"""
        if not selected_claims:
            return 0.0
        
        # Simple relevance check for top claim
        top_claim = selected_claims[0]
        task_words = set(task.lower().split())
        claim_words = set(top_claim.content.lower().split())
        
        intersection = len(claim_words & task_words)
        union = len(claim_words | task_words)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_diversity_score(self, claims: List[Claim]) -> float:
        """Calculate diversity score of selected claims"""
        if not claims:
            return 0.0
        
        # Type diversity
        types = set()
        for claim in claims:
            if claim.type:
                types.add(claim.type[0].value)
        
        type_diversity = len(types) / len(claims)
        
        # Confidence range diversity
        confidences = [claim.confidence for claim in claims]
        confidence_range = max(confidences) - min(confidences)
        confidence_diversity = confidence_range / 1.0  # Normalized
        
        # Content length diversity
        lengths = [len(claim.content) for claim in claims]
        length_range = max(lengths) - min(lengths)
        length_diversity = length_range / 100.0  # Normalized
        
        return (type_diversity + confidence_diversity + length_diversity) / 3.0
    
    def _analyze_quality_distribution(self, claims: List[Claim]) -> Dict[str, Any]:
        """Analyze quality distribution of selected claims"""
        if not claims:
            return {'error': 'No claims provided'}
        
        confidences = [claim.confidence for claim in claims]
        types = [claim.type[0].value if claim.type else 'unknown' for claim in claims]
        
        return {
            'avg_confidence': sum(confidences) / len(confidences),
            'min_confidence': min(confidences),
            'max_confidence': max(confidences),
            'confidence_std': (sum((c - sum(confidences)/len(confidences))**2 for c in confidences) / len(confidences))**0.5,
            'type_distribution': {t: types.count(t) for t in set(types)},
            'type_diversity': len(set(types)) / len(types),
            'total_claims': len(claims)
        }
    
    def get_selection_statistics(self) -> Dict[str, Any]:
        """Get claim selection performance statistics"""
        return self.selection_stats.copy()
    
    def reset_statistics(self):
        """Reset selection statistics"""
        self.selection_stats = {
            'selections_performed': 0,
            'claims_analyzed': 0,
            'avg_selection_time': 0.0,
            'top_claim_accuracy': 0.0
        }

# Global instance for reuse
_intelligent_selector = None

def get_intelligent_selector() -> IntelligentClaimSelector:
    """Get or create intelligent selector instance"""
    global _intelligent_selector
    if _intelligent_selector is None:
        _intelligent_selector = IntelligentClaimSelector()
    return _intelligent_selector

def select_claims_intelligent(context_claims: List[Claim], task: str, 
                         context_limit: int = 5, strategy: str = "balanced") -> SelectionResult:
    """
    Convenience function for intelligent claim selection.
    
    Args:
        context_claims: List of claims to select from
        task: Current task description
        context_limit: Maximum number of claims to select
        strategy: Selection strategy ('balanced', 'quality_first', 'diverse')
    
    Returns:
        SelectionResult with selected claims and metadata
    """
    selector = get_intelligent_selector()
    return selector.select_optimal_claims(context_claims, task, context_limit, strategy)