"""
Enhanced Context Builder with Primed Knowledge Integration

Integrates DynamicPrimingEngine-generated foundational claims into context
building for improved reasoning quality and evidence utilization.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json

from src.core.models import Claim, ClaimState
from src.data.data_manager import DataManager
from src.processing.dynamic_priming_engine import DynamicPrimingEngine, PrimingDomain
from src.processing.context_collector import ContextCollector

@dataclass
class ContextMetrics:
    """Metrics for context building performance"""
    
    total_context_time: float = 0.0
    primed_claims_used: int = 0
    regular_claims_used: int = 0
    context_size_tokens: int = 0
    quality_score: float = 0.0
    evidence_utilization: float = 0.0
    cross_task_knowledge_transfer: int = 0

class EnhancedContextBuilder:
    """
    Enhanced context builder that integrates primed foundational knowledge
    with existing context collection for improved reasoning quality.
    """
    
    def __init__(
        self,
        data_manager: DataManager,
        priming_engine: Optional[DynamicPrimingEngine] = None,
        base_context_collector: Optional[ContextCollector] = None
    ):
        """Initialize Enhanced Context Builder"""
        self.data_manager = data_manager
        self.priming_engine = priming_engine
        self.base_context_collector = base_context_collector
        
        # Context configuration
        self.context_config = {
            "max_context_claims": 15,
            "primed_claim_ratio": 0.4,  # 40% of context should be primed claims
            "min_similarity_threshold": 0.85,
            "max_tokens": 8000,
            "quality_weight": 0.3,
            "recency_weight": 0.2,
            "relevance_weight": 0.5,
        }
        
        # Performance tracking
        self.context_metrics = ContextMetrics()
        self.context_history = []
        
        # Caching for performance
        self._context_cache = {}
        self._cache_ttl = 300  # 5 minutes
        
        self.logger = logging.getLogger(__name__)
    
    async def build_enhanced_context(
        self,
        query: str,
        max_tokens: Optional[int] = None,
        include_primed: bool = True,
        force_refresh: bool = False
    ) -> Tuple[str, ContextMetrics]:
        """
        Build enhanced context with primed knowledge integration.
        
        Args:
            query: User query or task description
            max_tokens: Maximum context tokens (default from config)
            include_primed: Whether to include primed foundational claims
            force_refresh: Force cache refresh
            
        Returns:
            Tuple of (context_string, context_metrics)
        """
        start_time = datetime.utcnow()
        metrics = ContextMetrics()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query, include_primed)
            if not force_refresh and cache_key in self._context_cache:
                cached_result = self._context_cache[cache_key]
                if (datetime.utcnow() - cached_result["timestamp"]).seconds < self._cache_ttl:
                    self.logger.debug(f"Using cached context for query: {query[:50]}...")
                    return cached_result["context"], cached_result["metrics"]
            
            # Set max tokens
            max_tokens = max_tokens or self.context_config["max_tokens"]
            
            # Collect base context
            base_context_claims = []
            if self.base_context_collector:
                base_context = await self.base_context_collector.collect_context(query)
                base_context_claims = self._extract_claims_from_context(base_context)
            
            # Collect primed context
            primed_context_claims = []
            if include_primed and self.priming_engine:
                primed_context_claims = await self._collect_primed_context(query)
            
            # Merge and optimize context
            merged_claims = await self._merge_context_claims(
                base_context_claims, 
                primed_context_claims,
                query
            )
            
            # Format context for LLM
            context_string = self._format_context_for_llm(merged_claims, query)
            
            # Calculate metrics
            metrics.total_context_time = (datetime.utcnow() - start_time).total_seconds()
            metrics.primed_claims_used = len([c for c in merged_claims if "primed" in c.tags])
            metrics.regular_claims_used = len(merged_claims) - metrics.primed_claims_used
            metrics.context_size_tokens = self._estimate_tokens(context_string)
            metrics.quality_score = self._calculate_context_quality(merged_claims)
            metrics.evidence_utilization = self._calculate_evidence_utilization(merged_claims)
            metrics.cross_task_knowledge_transfer = self._count_cross_task_references(merged_claims)
            
            # Cache result
            self._context_cache[cache_key] = {
                "context": context_string,
                "metrics": metrics,
                "timestamp": datetime.utcnow()
            }
            
            # Store metrics
            self.context_metrics = metrics
            self.context_history.append({
                "query": query,
                "metrics": metrics,
                "timestamp": datetime.utcnow()
            })
            
            self.logger.info(
                f"Built enhanced context: {len(merged_claims)} claims "
                f"({metrics.primed_claims_used} primed, {metrics.regular_claims_used} regular), "
                f"{metrics.context_size_tokens} tokens"
            )
            
            return context_string, metrics
            
        except Exception as e:
            self.logger.error(f"Failed to build enhanced context: {e}")
            # Fallback to base context
            if self.base_context_collector:
                fallback_context = await self.base_context_collector.collect_context(query)
                return fallback_context, metrics
            raise
    
    async def _collect_primed_context(self, query: str) -> List[Claim]:
        """Collect primed foundational claims relevant to query"""
        try:
            primed_claims = []
            
            # Search for primed claims across all domains
            for domain in PrimingDomain:
                # Search domain-specific primed claims
                domain_claims = await self.data_manager.search_claims(
                    query=f"{query} {domain.value}",
                    limit=10,
                    confidence_threshold=0.7,
                    use_vector_search=True
                )
                
                # Filter for primed claims
                primed_domain_claims = [
                    claim_data for claim_data in domain_claims
                    if "primed" in claim_data.get("tags", [])
                ]
                
                # Convert to Claim objects
                for claim_data in primed_domain_claims:
                    try:
                        claim = Claim(
                            id=claim_data["id"],
                            content=claim_data["content"],
                            confidence=claim_data["confidence"],
                            tags=claim_data.get("tags", []),
                            state=ClaimState(claim_data.get("state", "Validated"))
                        )
                        claim.similarity_score = claim_data.get("similarity", 0.0)
                        primed_claims.append(claim)
                    except Exception as e:
                        self.logger.warning(f"Failed to convert claim data: {e}")
                        continue
            
            # Sort by relevance and confidence
            primed_claims.sort(
                key=lambda c: (
                    c.similarity_score * self.context_config["relevance_weight"] +
                    c.confidence * self.context_config["quality_weight"]
                ),
                reverse=True
            )
            
            return primed_claims[:self.context_config["max_context_claims"]]
            
        except Exception as e:
            self.logger.error(f"Failed to collect primed context: {e}")
            return []
    
    def _extract_claims_from_context(self, base_context: str) -> List[Claim]:
        """Extract claims from base context string"""
        claims = []
        
        try:
            # Try to parse as JSON first
            if base_context.strip().startswith('{'):
                context_data = json.loads(base_context)
                if "claims" in context_data:
                    for claim_data in context_data["claims"]:
                        claim = Claim(
                            id=claim_data.get("id", "unknown"),
                            content=claim_data.get("content", ""),
                            confidence=claim_data.get("confidence", 0.5),
                            tags=claim_data.get("tags", [])
                        )
                        claims.append(claim)
            
            # Try to extract XML claims
            elif "<claim" in base_context:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(f"<root>{base_context}</root>")
                
                for claim_elem in root.findall(".//claim"):
                    claim = Claim(
                        id=claim_elem.get("id", "unknown"),
                        content=claim_elem.findtext("content", "").strip(),
                        confidence=float(claim_elem.get("confidence", 0.5)),
                        tags=claim_elem.get("tags", "").split(",") if claim_elem.get("tags") else []
                    )
                    claims.append(claim)
            
            # Fallback: treat as plain text with claim markers
            else:
                import re
                claim_pattern = r'\[c(\w+)\s*\|\s*([^|]+)\s*\|\s*/\s*([\d.]+)\]'
                matches = re.findall(claim_pattern, base_context)
                
                for claim_id, content, confidence in matches:
                    claim = Claim(
                        id=f"c{claim_id}",
                        content=content.strip(),
                        confidence=float(confidence),
                        tags=[]
                    )
                    claims.append(claim)
        
        except Exception as e:
            self.logger.warning(f"Failed to extract claims from base context: {e}")
        
        return claims
    
    async def _merge_context_claims(
        self,
        base_claims: List[Claim],
        primed_claims: List[Claim],
        query: str
    ) -> List[Claim]:
        """Merge base and primed claims with optimal balance"""
        try:
            # Calculate target counts
            max_claims = self.context_config["max_context_claims"]
            target_primed_count = int(max_claims * self.context_config["primed_claim_ratio"])
            target_regular_count = max_claims - target_primed_count
            
            # Select best primed claims
            selected_primed = primed_claims[:target_primed_count]
            
            # Select best regular claims
            # Sort by relevance (similarity to query) and quality
            base_claims_with_relevance = []
            for claim in base_claims:
                # Calculate relevance score
                relevance = self._calculate_relevance(claim, query)
                claim.relevance_score = relevance
                base_claims_with_relevance.append(claim)
            
            base_claims_with_relevance.sort(
                key=lambda c: (
                    c.relevance_score * self.context_config["relevance_weight"] +
                    c.confidence * self.context_config["quality_weight"]
                ),
                reverse=True
            )
            
            selected_regular = base_claims_with_relevance[:target_regular_count]
            
            # Merge and sort by combined score
            merged_claims = selected_primed + selected_regular
            merged_claims.sort(
                key=lambda c: (
                    getattr(c, 'similarity_score', getattr(c, 'relevance_score', 0.0)) * 
                    self.context_config["relevance_weight"] +
                    c.confidence * self.context_config["quality_weight"]
                ),
                reverse=True
            )
            
            # Deduplicate using fuzzy matching
            unique_claims = []
            similarity_threshold = self.context_config["min_similarity_threshold"]
            
            import difflib
            
            for claim in merged_claims:
                is_duplicate = False
                content_lower = claim.content.lower().strip()
                
                for existing_claim in unique_claims:
                    existing_lower = existing_claim.content.lower().strip()
                    
                    # Check exact match first (fast)
                    if content_lower == existing_lower:
                        is_duplicate = True
                        break
                    
                    # Check fuzzy match
                    similarity = difflib.SequenceMatcher(None, content_lower, existing_lower).ratio()
                    if similarity >= similarity_threshold:
                        is_duplicate = True
                        # If existing claim has lower confidence, potentially swap?
                        # For now, keep the one that was sorted higher (first one)
                        break
                
                if not is_duplicate:
                    unique_claims.append(claim)
            
            return unique_claims[:max_claims]
            
        except Exception as e:
            self.logger.error(f"Failed to merge context claims: {e}")
            # Fallback: return all claims limited by max
            all_claims = primed_claims + base_claims
            return all_claims[:max_claims]
    
    def _calculate_relevance(self, claim: Claim, query: str) -> float:
        """Calculate relevance score of claim to query"""
        try:
            # Simple keyword-based relevance
            query_words = set(query.lower().split())
            claim_words = set(claim.content.lower().split())
            
            if not query_words:
                return 0.0
            
            # Jaccard similarity
            intersection = query_words.intersection(claim_words)
            union = query_words.union(claim_words)
            
            return len(intersection) / len(union) if union else 0.0
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate relevance: {e}")
            return 0.0
    
    def _format_context_for_llm(self, claims: List[Claim], query: str) -> str:
        """Format claims for LLM consumption"""
        try:
            context_parts = [
                f"<enhanced_context for_query='{query}'>",
                f"<priming_info>",
                f"This context includes {len([c for c in claims if 'primed' in c.tags])} primed foundational claims "
                f"and {len([c for c in claims if 'primed' not in c.tags])} regular claims.",
                f"Primed claims provide foundational knowledge in fact-checking, programming, "
                f"scientific method, and critical thinking domains.",
                f"</priming_info>",
                "",
                "<claims>"
            ]
            
            for i, claim in enumerate(claims, 1):
                claim_type = "primed" if "primed" in claim.tags else "regular"
                domains = [tag for tag in claim.tags if tag in [d.value for d in PrimingDomain]]
                domain_str = f" domain={domains[0]}" if domains else ""
                
                formatted_claim = (
                    f"  <claim id='{claim.id}' index='{i}' type='{claim_type}'{domain_str} "
                    f"confidence='{claim.confidence:.2f}'>"
                    f"\n    <content>{claim.content}</content>"
                )
                
                # Add metadata for primed claims
                if "primed" in claim.tags:
                    formatted_claim += f"\n    <metadata>primed_foundational_knowledge</metadata>"
                
                formatted_claim += "\n  </claim>"
                context_parts.append(formatted_claim)
            
            context_parts.extend([
                "</claims>",
                "",
                f"<context_summary>",
                f"Total claims: {len(claims)}",
                f"Primed foundational claims: {len([c for c in claims if 'primed' in c.tags])}",
                f"Regular claims: {len([c for c in claims if 'primed' not in c.tags])}",
                f"Average confidence: {sum(c.confidence for c in claims) / len(claims):.2f}",
                f"</context_summary>",
                "</enhanced_context>"
            ])
            
            return "\n".join(context_parts)
            
        except Exception as e:
            self.logger.error(f"Failed to format context for LLM: {e}")
            # Fallback: simple claim list
            return "\n".join([f"[{claim.id} | {claim.content} | / {claim.confidence:.2f}]" for claim in claims])
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Rough estimation: ~4 characters per token
        return len(text) // 4
    
    def _calculate_context_quality(self, claims: List[Claim]) -> float:
        """Calculate overall quality score for context"""
        if not claims:
            return 0.0
        
        # Weight by confidence and claim type
        total_score = 0.0
        total_weight = 0.0
        
        for claim in claims:
            weight = 1.0
            if "primed" in claim.tags:
                weight = 1.2  # Boost primed claims
            
            total_score += claim.confidence * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_evidence_utilization(self, claims: List[Claim]) -> float:
        """Calculate evidence utilization rate"""
        if not claims:
            return 0.0
        
        claims_with_evidence = len([
            claim for claim in claims 
            if "has_evidence" in claim.tags or "evidence" in claim.tags
        ])
        
        return claims_with_evidence / len(claims)
    
    def _count_cross_task_references(self, claims: List[Claim]) -> int:
        """Count cross-task knowledge transfer instances"""
        cross_task_count = 0
        
        for claim in claims:
            # Check for cross-domain indicators
            domains = [tag for tag in claim.tags if tag in [d.value for d in PrimingDomain]]
            if len(domains) > 1:
                cross_task_count += 1
            
            # Check for transfer indicators
            if any(indicator in claim.content.lower() for indicator in 
                   ["applies to", "transfers to", "cross-domain", "universal"]):
                cross_task_count += 1
        
        return cross_task_count
    
    def _generate_cache_key(self, query: str, include_primed: bool) -> str:
        """Generate cache key for context"""
        import hashlib
        content = f"{query}:{include_primed}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def get_context_statistics(self) -> Dict[str, Any]:
        """Get comprehensive context building statistics"""
        try:
            # Calculate averages from history
            if self.context_history:
                avg_quality = sum(
                    h["metrics"].quality_score for h in self.context_history
                ) / len(self.context_history)
                
                avg_evidence_util = sum(
                    h["metrics"].evidence_utilization for h in self.context_history
                ) / len(self.context_history)
                
                avg_cross_task = sum(
                    h["metrics"].cross_task_knowledge_transfer for h in self.context_history
                ) / len(self.context_history)
                
                avg_time = sum(
                    h["metrics"].total_context_time for h in self.context_history
                ) / len(self.context_history)
            else:
                avg_quality = avg_evidence_util = avg_cross_task = avg_time = 0.0
            
            return {
                "context_builder": {
                    "configuration": self.context_config,
                    "cache_size": len(self._context_cache),
                    "total_contexts_built": len(self.context_history)
                },
                "performance_metrics": {
                    "average_quality_score": avg_quality,
                    "average_evidence_utilization": avg_evidence_util,
                    "average_cross_task_transfer": avg_cross_task,
                    "average_build_time": avg_time,
                    "current_session": {
                        "primed_claims_used": self.context_metrics.primed_claims_used,
                        "regular_claims_used": self.context_metrics.regular_claims_used,
                        "context_size_tokens": self.context_metrics.context_size_tokens,
                        "quality_score": self.context_metrics.quality_score,
                        "evidence_utilization": self.context_metrics.evidence_utilization,
                        "cross_task_knowledge_transfer": self.context_metrics.cross_task_knowledge_transfer
                    }
                },
                "priming_integration": {
                    "priming_engine_available": self.priming_engine is not None,
                    "primed_claim_ratio": self.context_config["primed_claim_ratio"],
                    "max_context_claims": self.context_config["max_context_claims"]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get context statistics: {e}")
            return {"error": str(e)}
    
    def clear_cache(self) -> None:
        """Clear context cache"""
        self._context_cache.clear()
        self.logger.info("Context cache cleared")
    
    def update_configuration(self, new_config: Dict[str, Any]) -> None:
        """Update context building configuration"""
        self.context_config.update(new_config)
        self.clear_cache()  # Clear cache when config changes
        self.logger.info(f"Context configuration updated: {new_config}")