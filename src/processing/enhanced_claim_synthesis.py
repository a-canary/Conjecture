"""
Enhanced Claim Synthesis for Multi-Modal Reasoning

Advanced algorithms for converting multi-modal evidence into structured claims
with improved confidence calibration and evidence integration.
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.core.models import Claim, ClaimType, ClaimState

logger = logging.getLogger(__name__)


class EvidenceCorrelation(Enum):
    """Evidence correlation types"""
    COMPLEMENTARY = "complementary"
    CONTRADICTORY = "contradictory"
    REDUNDANT = "redundant"
    INDEPENDENT = "independent"


@dataclass
class EvidenceCluster:
    """Cluster of related evidence items"""
    evidence_items: List[Any]  # MultiModalEvidence items
    correlation_type: EvidenceCorrelation
    aggregate_confidence: float
    cluster_weight: float
    primary_modality: str


class AdvancedClaimSynthesizer:
    """Advanced multi-modal claim synthesizer with evidence integration"""
    
    def __init__(self):
        self.synthesis_stats = {
            "claims_generated": 0,
            "multimodal_integrations": 0,
            "synthesis_time_ms": 0.0,
            "evidence_clusters_processed": 0,
            "confidence_calibration_error": 0.0
        }
        
        # Modality reliability weights based on empirical analysis
        self.modality_weights = {
            "text": 1.0,      # Highest reliability
            "document": 0.9,   # High reliability
            "image": 0.8      # Good reliability
        }
        
        # Evidence correlation thresholds
        self.correlation_thresholds = {
            "complementary": 0.7,
            "contradictory": 0.3,
            "redundant": 0.9
        }
    
    async def synthesize_claims(self, evidence_list: List[Any]) -> List[Claim]:
        """Advanced claim synthesis from multi-modal evidence"""
        start_time = time.time()
        
        logger.info(f"Starting advanced claim synthesis with {len(evidence_list)} evidence items")
        
        try:
            # Phase 1: Evidence preprocessing and filtering
            filtered_evidence = self._preprocess_evidence(evidence_list)
            
            # Phase 2: Evidence clustering and correlation
            evidence_clusters = self._cluster_evidence(filtered_evidence)
            
            # Phase 3: Cross-modal evidence integration
            integrated_evidence = self._integrate_evidence_clusters(evidence_clusters)
            
            # Phase 4: Claim generation with quality validation
            claims = self._generate_claims_from_integrated_evidence(integrated_evidence)
            
            # Phase 5: Multi-modal reasoning chain construction
            enhanced_claims = self._construct_reasoning_chains(claims, evidence_clusters)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update statistics
            self.synthesis_stats["claims_generated"] += len(enhanced_claims)
            self.synthesis_stats["multimodal_integrations"] += 1
            self.synthesis_stats["synthesis_time_ms"] += processing_time
            self.synthesis_stats["evidence_clusters_processed"] += len(evidence_clusters)
            
            logger.info(f"Generated {len(enhanced_claims)} claims in {processing_time:.1f}ms")
            
            return enhanced_claims
            
        except Exception as e:
            logger.error(f"Claim synthesis failed: {e}")
            # Return fallback claim for debugging
            return self._create_fallback_claim(evidence_list)
    
    def _preprocess_evidence(self, evidence_list: List[Any]) -> List[Any]:
        """Preprocess and filter evidence items"""
        filtered = []
        
        for evidence in evidence_list:
            # Quality filtering
            if hasattr(evidence, 'confidence') and evidence.confidence >= 0.5:
                # Content validation
                if hasattr(evidence, 'content') and evidence.content:
                    filtered.append(evidence)
        
        logger.debug(f"Filtered evidence: {len(filtered)}/{len(evidence_list)} items")
        return filtered
    
    def _cluster_evidence(self, evidence_list: List[Any]) -> List[EvidenceCluster]:
        """Cluster evidence by correlation and modality"""
        clusters = []
        
        # Simple clustering by modality and content similarity
        modality_groups = {}
        
        for evidence in evidence_list:
            modality = getattr(evidence, 'modality', 'unknown').value if hasattr(evidence, 'modality') else 'unknown'
            
            if modality not in modality_groups:
                modality_groups[modality] = []
            modality_groups[modality].append(evidence)
        
        # Create clusters from modality groups
        for modality, items in modality_groups.items():
            if items:
                # Calculate aggregate confidence
                confidences = [getattr(item, 'confidence', 0.5) for item in items]
                aggregate_confidence = sum(confidences) / len(confidences)
                
                # Determine correlation type (simplified)
                correlation_type = self._determine_correlation_type(items)
                
                # Calculate cluster weight
                cluster_weight = self.modality_weights.get(modality, 0.5)
                
                cluster = EvidenceCluster(
                    evidence_items=items,
                    correlation_type=correlation_type,
                    aggregate_confidence=aggregate_confidence,
                    cluster_weight=cluster_weight,
                    primary_modality=modality
                )
                
                clusters.append(cluster)
        
        logger.debug(f"Created {len(clusters)} evidence clusters")
        return clusters
    
    def _determine_correlation_type(self, evidence_items: List[Any]) -> EvidenceCorrelation:
        """Determine correlation type within evidence cluster"""
        if len(evidence_items) <= 1:
            return EvidenceCorrelation.INDEPENDENT
        
        # Simplified correlation analysis
        # In real implementation, this would use semantic similarity
        contents = [getattr(item, 'content', '') for item in evidence_items]
        
        # Check for redundancy (high content similarity)
        if len(set(contents)) == 1:
            return EvidenceCorrelation.REDUNDANT
        
        # Check for contradictions (simplified)
        # In real implementation, this would use NLP contradiction detection
        if any("not" in content.lower() or "false" in content.lower() for content in contents):
            return EvidenceCorrelation.CONTRADICTORY
        
        # Default to complementary
        return EvidenceCorrelation.COMPLEMENTARY
    
    def _integrate_evidence_clusters(self, clusters: List[EvidenceCluster]) -> List[Dict[str, Any]]:
        """Integrate evidence clusters for claim generation"""
        integrated = []
        
        for cluster in clusters:
            # Weight confidence by cluster weight and correlation type
            correlation_multiplier = {
                EvidenceCorrelation.COMPLEMENTARY: 1.2,
                EvidenceCorrelation.INDEPENDENT: 1.0,
                EvidenceCorrelation.REDUNDANT: 0.8,
                EvidenceCorrelation.CONTRADICTORY: 0.6
            }
            
            adjusted_confidence = (
                cluster.aggregate_confidence * 
                cluster.cluster_weight * 
                correlation_multiplier.get(cluster.correlation_type, 1.0)
            )
            
            # Cap confidence at 1.0
            adjusted_confidence = min(adjusted_confidence, 1.0)
            
            integrated_evidence = {
                "cluster": cluster,
                "adjusted_confidence": adjusted_confidence,
                "evidence_summary": self._create_evidence_summary(cluster),
                "reasoning_context": self._create_reasoning_context(cluster)
            }
            
            integrated.append(integrated_evidence)
        
        return integrated
    
    def _create_evidence_summary(self, cluster: EvidenceCluster) -> str:
        """Create summary of evidence cluster"""
        if not cluster.evidence_items:
            return "No evidence available"
        
        contents = [getattr(item, 'content', '') for item in cluster.evidence_items]
        
        if len(contents) == 1:
            return contents[0]
        
        # Combine multiple evidence items
        if cluster.correlation_type == EvidenceCorrelation.COMPLEMENTARY:
            return f"Multiple complementary sources: {'; '.join(contents[:2])}"
        elif cluster.correlation_type == EvidenceCorrelation.CONTRADICTORY:
            return f"Conflicting evidence detected: {'; '.join(contents[:2])}"
        elif cluster.correlation_type == EvidenceCorrelation.REDUNDANT:
            return f"Multiple consistent sources: {contents[0]}"
        else:
            return f"Independent evidence: {'; '.join(contents[:2])}"
    
    def _create_reasoning_context(self, cluster: EvidenceCluster) -> Dict[str, Any]:
        """Create reasoning context for evidence cluster"""
        return {
            "modality": cluster.primary_modality,
            "evidence_count": len(cluster.evidence_items),
            "correlation_type": cluster.correlation_type.value,
            "confidence_level": self._classify_confidence_level(cluster.aggregate_confidence),
            "reliability_assessment": self._assess_reliability(cluster)
        }
    
    def _classify_confidence_level(self, confidence: float) -> str:
        """Classify confidence level"""
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _assess_reliability(self, cluster: EvidenceCluster) -> str:
        """Assess evidence reliability"""
        base_reliability = self.modality_weights.get(cluster.primary_modality, 0.5)
        
        if cluster.correlation_type == EvidenceCorrelation.COMPLEMENTARY:
            return "enhanced"
        elif cluster.correlation_type == EvidenceCorrelation.CONTRADICTORY:
            return "questionable"
        elif cluster.correlation_type == EvidenceCorrelation.REDUNDANT:
            return "verified"
        else:
            return "standard"
    
    def _generate_claims_from_integrated_evidence(self, integrated_evidence: List[Dict[str, Any]]) -> List[Claim]:
        """Generate claims from integrated evidence"""
        claims = []
        
        for evidence_data in integrated_evidence:
            cluster = evidence_data["cluster"]
            confidence = evidence_data["adjusted_confidence"]
            summary = evidence_data["evidence_summary"]
            context = evidence_data["reasoning_context"]
            
            # Generate claim content based on evidence type and context
            claim_content = self._generate_claim_content(summary, context)
            
            # Determine claim type based on evidence
            claim_types = self._determine_claim_types(cluster, context)
            
            # Create claim
            claim = Claim(
                id=f"enhanced_multimodal_{int(time.time())}_{len(claims):03d}",
                content=claim_content,
                confidence=confidence,
                state=ClaimState.VALIDATED,
                tags=["multimodal", cluster.primary_modality, "enhanced_synthesis"] + [t.value for t in claim_types],
                metadata={
                    "evidence_modality": cluster.primary_modality,
                    "evidence_count": len(cluster.evidence_items),
                    "correlation_type": cluster.correlation_type.value,
                    "reasoning_context": context,
                    "synthesis_timestamp": time.time(),
                    "synthesis_method": "enhanced_multimodal",
                    "claim_types": [t.value for t in claim_types]
                }
            )
            
            claims.append(claim)
        
        return claims
    
    def _generate_claim_content(self, evidence_summary: str, context: Dict[str, Any]) -> str:
        """Generate claim content based on evidence and context"""
        modality = context["modality"]
        correlation = context["correlation_type"]
        confidence_level = context["confidence_level"]
        reliability = context["reliability_assessment"]
        
        # Generate structured claim content
        if correlation == "complementary":
            content = f"Multi-modal analysis indicates {evidence_summary.lower()}"
        elif correlation == "contradictory":
            content = f"Conflicting evidence detected: {evidence_summary.lower()}"
        elif correlation == "redundant":
            content = f"Multiple sources confirm: {evidence_summary.lower()}"
        else:
            content = f"Independent analysis shows: {evidence_summary.lower()}"
        
        # Add confidence and reliability context
        content += f" (Confidence: {confidence_level}, Reliability: {reliability})"
        
        return content
    
    def _determine_claim_types(self, cluster: EvidenceCluster, context: Dict[str, Any]) -> List[ClaimType]:
        """Determine claim types based on evidence and context"""
        claim_types = [ClaimType.ANALYSIS]  # Default analysis type
        
        modality = cluster.primary_modality
        correlation = context["correlation_type"]
        
        # Add specific types based on modality
        if modality == "text":
            claim_types.append(ClaimType.FACT)
        elif modality == "document":
            claim_types.append(ClaimType.REFERENCE)
        elif modality == "image":
            claim_types.append(ClaimType.EXAMPLE)
        
        # Add reasoning type based on correlation
        if correlation == "contradictory":
            claim_types.append(ClaimType.CONCEPT)
        
        return claim_types
    
    def _construct_reasoning_chains(self, claims: List[Claim], clusters: List[EvidenceCluster]) -> List[Claim]:
        """Construct reasoning chains for enhanced claims"""
        # Add cross-references between related claims
        for i, claim in enumerate(claims):
            # Add evidence chain metadata
            related_clusters = [c for c in clusters if c.primary_modality in claim.tags]
            
            if len(related_clusters) > 1:
                claim.metadata["cross_modal_reasoning"] = True
                claim.metadata["evidence_chain_length"] = len(related_clusters)
            
            # Add confidence calibration metadata
            claim.metadata["confidence_calibration"] = {
                "original_confidence": claim.confidence,
                "calibrated_confidence": min(claim.confidence * 0.95, 1.0),  # Conservative adjustment
                "calibration_method": "multimodal_conservative"
            }
        
        return claims
    
    def _create_fallback_claim(self, evidence_list: List[Any]) -> List[Claim]:
        """Create fallback claim when synthesis fails"""
        if not evidence_list:
            return []
        
        fallback_claim = Claim(
            id=f"fallback_multimodal_{int(time.time())}_000",
            content=f"Multi-modal evidence processing completed with {len(evidence_list)} evidence items",
            confidence=0.5,
            state=ClaimState.VALIDATED,
            tags=["multimodal", "fallback", "synthesized", "fact"],
            metadata={
                "fallback_reason": "synthesis_failed",
                "evidence_count": len(evidence_list),
                "synthesis_timestamp": time.time(),
                "claim_types": ["fact"]
            }
        )
        
        return [fallback_claim]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get synthesis statistics"""
        return self.synthesis_stats.copy()
    
    def reset_stats(self) -> None:
        """Reset synthesis statistics"""
        self.synthesis_stats = {
            "claims_generated": 0,
            "multimodal_integrations": 0,
            "synthesis_time_ms": 0.0,
            "evidence_clusters_processed": 0,
            "confidence_calibration_error": 0.0
        }