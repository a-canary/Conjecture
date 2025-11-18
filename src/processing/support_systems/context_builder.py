"""
Context Building System for Agent Harness
Builds relevant, optimized context for LLM consumption
"""

import asyncio
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import logging
import math

from .models import (
    ContextItem, ContextItemType, ContextResult, OptimizedContext, TokenUsage
)
from .data_collection import DataCollector, DataItem
from ..utils.id_generator import generate_context_id


logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Builds and optimizes context from collected data for LLM consumption
    """

    def __init__(self, data_collector: DataCollector, default_token_limit: int = 4000):
        self.data_collector = data_collector
        self.default_token_limit = default_token_limit
        
        # Configuration
        self.relevance_weights = {
            'keyword_match': 0.3,
            'semantic_similarity': 0.2,
            'recency': 0.2,
            'confidence': 0.15,
            'usage_frequency': 0.15
        }
        
        # Token estimation (rough approximation)
        self.token_ratio = 1.0  # characters per token (approximation)
        
        # Cache optimization results
        self.optimization_cache: Dict[str, OptimizedContext] = {}
        self.cache_ttl_seconds = 300  # 5 minutes

    async def build_context(self, query: str, sources: List[str],
                           max_tokens: Optional[int] = None,
                           context_type: str = "default",
                           filters: Optional[Dict[str, Any]] = None) -> ContextResult:
        """
        Build context from multiple sources
        
        Args:
            query: Query string
            sources: List of source names to collect from
            max_tokens: Maximum token limit
            context_type: Type of context being built
            filters: Optional filters for data selection
            
        Returns:
            Context building result
        """
        start_time = time.time()
        max_tokens = max_tokens or self.default_token_limit
        filters = filters or {}
        
        try:
            logger.debug(f"Building context for query: {query[:100]}...")
            
            # Collect data from all sources
            context_items = []
            
            for source in sources:
                try:
                    source_items = await self._collect_from_source(source, query, filters)
                    context_items.extend(source_items)
                except Exception as e:
                    logger.warning(f"Failed to collect from source {source}: {e}")
                    continue
            
            # Score relevance of all items
            scored_items = await self._score_relevance(context_items, query, context_type)
            
            # Filter by relevance threshold
            relevance_threshold = filters.get('relevance_threshold', 0.1)
            scored_items = [item for item in scored_items if item.relevance_score >= relevance_threshold]
            
            # Sort by relevance
            scored_items.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Estimate tokens and optimize
            total_tokens = sum(item.token_estimate for item in scored_items)
            
            if total_tokens > max_tokens:
                # Apply optimization
                optimized_context = await self._optimize_context(scored_items, max_tokens, context_type)
                context_items = optimized_context.items
                optimization_applied = True
            else:
                # No optimization needed
                context_items = scored_items
                optimization_applied = False
            
            # Calculate final token count
            final_tokens = sum(item.token_estimate for item in context_items)
            
            # Build result
            processing_time = int((time.time() - start_time) * 1000)
            result = ContextResult(
                query=query,
                context_items=context_items,
                total_tokens=final_tokens,
                processing_time_ms=processing_time,
                collection_method=context_type,
                optimization_applied=optimization_applied,
                metadata={
                    'sources': sources,
                    'max_tokens': max_tokens,
                    'original_items': len(context_items),
                    'filters': filters
                }
            )
            
            logger.debug(f"Context built: {len(context_items)} items, {final_tokens} tokens in {processing_time}ms")
            return result

        except Exception as e:
            logger.error(f"Failed to build context: {e}")
            # Return empty context result
            return ContextResult(
                query=query,
                context_items=[],
                total_tokens=0,
                processing_time_ms=int((time.time() - start_time) * 1000),
                collection_method=context_type,
                optimization_applied=False,
                metadata={'error': str(e)}
            )

    async def score_relevance(self, item: ContextItem, query: str, 
                            context_type: str) -> float:
        """
        Score relevance of a context item
        
        Args:
            item: Context item to score
            query: Original query
            context_type: Context building type
            
        Returns:
            Relevance score (0.0-1.0)
        """
        scores = {}
        
        # Keyword matching
        scores['keyword_match'] = self._score_keyword_match(item.content, query)
        
        # Semantic similarity (simplified)
        scores['semantic_similarity'] = self._score_semantic_similarity(item.content, query)
        
        # Recency (more recent is more relevant)
        scores['recency'] = self._score_recency(item.created_at)
        
        # Confidence
        scores['confidence'] = item.metadata.get('confidence', 1.0)
        
        # Usage frequency (more used items might be more relevant)
        scores['usage_frequency'] = min(1.0, item.metadata.get('usage_count', 0) / 10.0)
        
        # Calculate weighted score
        total_score = sum(
            scores[factor] * weight
            for factor, weight in self.relevance_weights.items()
        )
        
        # Apply context type modifiers
        context_modifier = self._get_context_type_modifier(item.item_type, context_type)
        total_score *= context_modifier
        
        return min(1.0, max(0.0, total_score))

    async def optimize_for_tokens(self, items: List[ContextItem], 
                                limit: int) -> OptimizedContext:
        """
        Optimize context items within token limits
        
        Args:
            items: List of context items
            limit: Token limit
            
        Returns:
            Optimized context
        """
        cache_key = self._generate_optimization_cache_key(items, limit)
        
        # Check cache first
        cached_result = self.optimization_cache.get(cache_key)
        if cached_result and not self._is_cache_expired(cached_result, cache_key):
            return cached_result
        
        # Perform optimization
        optimized = await self._optimize_context(items, limit, "default")
        
        # Cache result
        self.optimization_cache[cache_key] = optimized
        
        return optimized

    async def format_context_item(self, item: ContextItem, 
                                format_type: str = "default") -> str:
        """
        Format a context item for LLM consumption
        
        Args:
            item: Context item to format
            format_type: Format type
            
        Returns:
            Formatted context string
        """
        try:
            if format_type == "compact":
                return f"[{item.item_type.value}] {item.content}"
            
            elif format_type == "detailed":
                metadata_str = ", ".join([f"{k}={v}" for k, v in item.metadata.items()])
                return (
                    f"{item.item_type.value.upper()} (id={item.id}, "
                    f"relevance={item.relevance_score:.2f}, "
                    f"{metadata_str}):\n{item.content}"
                )
            
            else:  # default
                prefix = {
                    ContextItemType.EXAMPLE: "EXAMPLE",
                    ContextItemType.CLAIM: "CLAIM",
                    ContextItemType.TOOL: "TOOL",
                    ContextItemType.DATA: "DATA",
                    ContextItemType.KNOWLEDGE: "KNOWLEDGE"
                }.get(item.item_type, "INFO")
                
                return f"- [{prefix}] {item.content}"

        except Exception as e:
            logger.error(f"Failed to format context item {item.id}: {e}")
            return f"-[{item.item_type.value}] {item.content}"

    async def calculate_token_usage(self, text: str) -> TokenUsage:
        """
        Calculate token usage for text
        
        Args:
            text: Input text
            
        Returns:
            Token usage information
        """
        try:
            # Rough token estimation (can be replaced with actual tokenizer)
            char_count = len(text)
            estimated_tokens = math.ceil(char_count / self.token_ratio)
            
            return TokenUsage(
                input_tokens=estimated_tokens,
                output_tokens=0,
                total_tokens=estimated_tokens,
                context_tokens=estimated_tokens
            )

        except Exception as e:
            logger.error(f"Failed to calculate token usage: {e}")
            return TokenUsage()

    async def track_context_effectiveness(self, context_result: ContextResult,
                                        outcome: Dict[str, Any]) -> None:
        """
        Track effectiveness of context based on outcome
        
        Args:
            context_result: Context that was used
            outcome: Outcome metrics
        """
        try:
            # Update usage statistics for context items
            usage_scores = outcome.get('item_usage_scores', {})
            
            for item in context_result.context_items:
                item_score = usage_scores.get(item.id, 0.5)  # Default neutral score
                usage_count = item.metadata.get('usage_count', 0)
                
                # Update item metadata
                item.metadata.update({
                    'usage_count': usage_count + 1,
                    'last_score': item_score,
                    'average_score': ((item.metadata.get('average_score', 0.5) * usage_count) + item_score) / (usage_count + 1)
                })
            
            # Track overall context effectiveness
            avg_item_score = sum(usage_scores.values()) / len(usage_scores) if usage_scores else 0.5
            
            logger.debug(f"Context effectiveness tracked: avg_score={avg_item_score:.2f}")

        except Exception as e:
            logger.error(f"Failed to track context effectiveness: {e}")

    async def _collect_from_source(self, source: str, query: str,
                                 filters: Dict[str, Any]) -> List[ContextItem]:
        """Collect context items from a specific source"""
        try:
            # Map source names to DataSource enum
            from .models import DataSource
            
            source_mapping = {
                'user_input': DataSource.USER_INPUT,
                'existing_claims': DataSource.EXISTING_CLAIMS,
                'tool_result': DataSource.TOOL_RESULT,
                'knowledge_base': DataSource.KNOWLEDGE_BASE
            }
            
            data_source = source_mapping.get(source)
            if not data_source:
                logger.warning(f"Unknown source: {source}")
                return []
            
            # Collect raw data
            data_items = await self.data_collector.collect_from_source(
                data_source, query, filters
            )
            
            # Convert to context items
            context_items = []
            for data_item in data_items:
                context_item = ContextItem(
                    id=data_item.id,
                    item_type=self._determine_item_type(data_item),
                    content=self._extract_content(data_item),
                    relevance_score=0.5,  # Will be scored later
                    metadata=data_item.metadata,
                    token_estimate=self._estimate_tokens(data_item),
                    source=source,
                    created_at=data_item.timestamp
                )
                context_items.append(context_item)
            
            return context_items

        except Exception as e:
            logger.error(f"Failed to collect from source {source}: {e}")
            return []

    async def _score_relevance(self, items: List[ContextItem], query: str,
                             context_type: str) -> List[ContextItem]:
        """Score relevance for all items"""
        for item in items:
            try:
                item.relevance_score = await self.score_relevance(item, query, context_type)
            except Exception as e:
                logger.warning(f"Failed to score item {item.id}: {e}")
                item.relevance_score = 0.1  # Low default score
        
        return items

    async def _optimize_context(self, items: List[ContextItem], limit: int,
                              optimization_strategy: str) -> OptimizedContext:
        """Optimize context items within token limit"""
        try:
            current_tokens = sum(item.token_estimate for item in items)
            
            if current_tokens <= limit:
                return OptimizedContext(
                    items=items,
                    total_tokens=current_tokens,
                    token_limit=limit,
                    optimization_strategy=optimization_strategy,
                    optimization_score=1.0
                )
            
            # Apply optimization strategy
            if optimization_strategy == "relevance":
                optimized_items = self._optimize_by_relevance(items, limit)
            elif optimization_strategy == "balanced":
                optimized_items = self._optimize_balanced(items, limit)
            else:
                optimized_items = self._optimize_by_relevance(items, limit)
            
            # Calculate optimization score
            optimization_score = min(1.0, limit / current_tokens)
            
            return OptimizedContext(
                items=optimized_items,
                excluded_items=[item for item in items if item not in optimized_items],
                total_tokens=sum(item.token_estimate for item in optimized_items),
                token_limit=limit,
                optimization_strategy=optimization_strategy,
                optimization_score=optimization_score
            )

        except Exception as e:
            logger.error(f"Context optimization failed: {e}")
            # Return original items if optimization fails
            return OptimizedContext(
                items=items[:10],  # Truncate to reasonable size
                excluded_items=items[10:],
                total_tokens=sum(item.token_estimate for item in items[:10]),
                token_limit=limit,
                optimization_strategy="fallback",
                optimization_score=0.5
            )

    def _optimize_by_relevance(self, items: List[ContextItem], limit: int) -> List[ContextItem]:
        """Optimize by keeping most relevant items"""
        sorted_items = sorted(items, key=lambda x: x.relevance_score, reverse=True)
        
        optimized_items = []
        current_tokens = 0
        
        for item in sorted_items:
            if current_tokens + item.token_estimate <= limit:
                optimized_items.append(item)
                current_tokens += item.token_estimate
            else:
                break
        
        return optimized_items

    def _optimize_balanced(self, items: List[ContextItem], limit: int) -> List[ContextItem]:
        """Optimize with balanced item type distribution"""
        # Group by item type
        type_groups = {}
        for item in items:
            item_type = item.item_type
            if item_type not in type_groups:
                type_groups[item_type] = []
            type_groups[item_type].append(item)
        
        # Sort each group by relevance
        for item_type, group_items in type_groups.items():
            group_items.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Select items with balanced distribution
        optimized_items = []
        current_tokens = 0
        type_pointers = {item_type: 0 for item_type in type_groups}
        active_types = set(type_groups.keys())
        
        while active_types and current_tokens < limit:
            added_items = []
            
            for item_type in list(active_types):
                pointer = type_pointers[item_type]
                group_items = type_groups[item_type]
                
                if pointer < len(group_items):
                    item = group_items[pointer]
                    
                    if current_tokens + item.token_estimate <= limit:
                        optimized_items.append(item)
                        added_items.append(item)
                        current_tokens += item.token_estimate
                        type_pointers[item_type] += 1
                    else:
                        # Can't fit more from this type
                        active_types.remove(item_type)
                else:
                    # No more items in this type
                    active_types.remove(item_type)
            
            if not added_items:
                break
        
        return optimized_items

    def _determine_item_type(self, data_item: DataItem) -> ContextItemType:
        """Determine context item type from data item"""
        metadata = data_item.metadata
        
        if metadata.get('is_claim'):
            return ContextItemType.CLAIM
        elif metadata.get('is_example'):
            return ContextItemType.EXAMPLE
        elif metadata.get('is_claim'):
            return ContextItemType.CLAIM
        elif metadata.get('is_tool'):
            return ContextItemType.TOOL
        elif metadata.get('is_knowledge'):
            return ContextItemType.KNOWLEDGE
        else:
            return ContextItemType.DATA

    def _extract_content(self, data_item: DataItem) -> str:
        """Extract text content from data item"""
        content = data_item.content
        
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            # Try to find meaningful text content
            if 'text' in content:
                return content['text']
            elif 'content' in content:
                return content['content']
            elif 'description' in content:
                return content['description']
            else:
                # Convert dict to string representation
                return str(content)
        else:
            return str(content)

    def _estimate_tokens(self, data_item: DataItem) -> int:
        """Estimate token count for data item"""
        content = self._extract_content(data_item)
        return math.ceil(len(content) / self.token_ratio)

    def _score_keyword_match(self, text1: str, text2: str) -> float:
        """Score keyword matching between two texts"""
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

    def _score_semantic_similarity(self, text1: str, text2: str) -> float:
        """Score semantic similarity (simplified implementation)"""
        # Pattern-based similarity as semantic substitute
        common_patterns = [
            r'\b(weather|temperature|forecast|climate)\b',
            r'\b(calculate|compute|math|arithmetic)\b',
            r'\b(search|find|lookup|query)\b',
            r'\b(get|fetch|retrieve|obtain)\b',
            r'\b(code|programming|function|algorithm)\b',
            r'\b(test|validate|verify|check)\b'
        ]
        
        score = 0.0
        for pattern in common_patterns:
            match1 = bool(re.search(pattern, text1, re.IGNORECASE))
            match2 = bool(re.search(pattern, text2, re.IGNORECASE))
            
            if match1 and match2:
                score += 1.0 / len(common_patterns)
        
        return min(1.0, score)

    def _score_recency(self, timestamp: datetime) -> float:
        """Score recency (more recent is higher)"""
        age_hours = (datetime.utcnow() - timestamp).total_seconds() / 3600
        
        # Full score for items less than 1 hour old
        if age_hours < 1:
            return 1.0
        
        # Linear decay over 24 hours
        if age_hours < 24:
            return 1.0 - (age_hours - 1) / 23
        
        # Minimal score for older items
        return 0.1

    def _get_context_type_modifier(self, item_type: ContextItemType, 
                                 context_type: str) -> float:
        """Get context type modifier for item type"""
        modifiers = {
            ('example', 'research'): 1.1,
            ('claim', 'research'): 1.3,
            ('example', 'coding'): 1.2,
            ('tool', 'coding'): 1.1,
            ('example', 'testing'): 1.3,
            ('claim', 'evaluation'): 1.4,
            ('example', 'evaluation'): 1.2
        }
        
        return modifiers.get((item_type.value, context_type), 1.0)

    def _generate_optimization_cache_key(self, items: List[ContextItem], limit: int) -> str:
        """Generate cache key for optimization result"""
        item_ids = [item.id for item in items]
        item_ids.sort()
        
        import hashlib
        key_data = f"{item_ids}:{limit}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _is_cache_expired(self, cached_result: OptimizedContext, cache_key: str) -> bool:
        """Check if cached optimization result is expired"""
        # For simplicity, using a simple time-based approach
        # In a real implementation, store timestamps with cache entries
        return False  # Disable cache expiration for now