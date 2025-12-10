"""
Context Collector for the Conjecture skill-based agency system.
OPTIMIZED: Enhanced caching, parallel processing, query optimization
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import re
import hashlib
from functools import lru_cache

try:
    from ..core.models import Claim
    from ..data.data_manager import DataManager
except ImportError:
    # Handle relative import issues for test compatibility
    import sys
    import os
    # Add src directory to path for absolute imports
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.core.models import Claim
    from src.data.data_manager import DataManager

logger = logging.getLogger(__name__)

class ContextRelevanceScorer:
    """Scores relevance of skills and samples to a given claim."""

    def __init__(self):
        self.relevance_weights = {
            "keyword_match": 0.4,
            "semantic_similarity": 0.3,
            "tool_match": 0.2,
            "usage_frequency": 0.1,
        }

    def score_skill_relevance(
        self, skill: Claim, claim_content: str, context: Dict[str, Any]
    ) -> float:
        """
        Score how relevant a skill is to a claim.

        Args:
            skill: Claim to score
            claim_content: Content of the claim being evaluated
            context: Additional context

        Returns:
            Relevance score between 0.0 and 1.0
        """
        scores = {}

        # Keyword matching
        scores["keyword_match"] = self._score_keyword_match(
            skill.content, claim_content
        )

        # Tool name matching (simplified)
        scores["tool_match"] = 0.5

        # Usage frequency (simplified)
        scores["usage_frequency"] = 0.5

        # Semantic similarity (simplified - could use embeddings)
        scores["semantic_similarity"] = self._score_semantic_similarity(
            skill.content, claim_content
        )

        # Calculate weighted score
        total_score = sum(
            scores[factor] * weight for factor, weight in self.relevance_weights.items()
        )

        return min(1.0, max(0.0, total_score))

    def score_sample_relevance(
        self, sample: Claim, claim_content: str, context: Dict[str, Any]
    ) -> float:
        """
        Score how relevant a sample is to a claim.

        Args:
            sample: Claim to score
            claim_content: Content of the claim being evaluated
            context: Additional context

        Returns:
            Relevance score between 0.0 and 1.0
        """
        scores = {}

        # Tool name matching
        scores["tool_match"] = 0.5

        # Keyword matching in content
        scores["keyword_match"] = self._score_keyword_match(
            sample.content, claim_content
        )

        # Success preference (prefer successful samples)
        scores["success_preference"] = sample.confidence

        # Quality score
        scores["quality_score"] = sample.confidence

        # Usage frequency
        scores["usage_frequency"] = 0.5

        # Calculate weighted score (different weights for samples)
        sample_weights = {
            "tool_match": 0.3,
            "keyword_match": 0.3,
            "success_preference": 0.2,
            "quality_score": 0.1,
            "usage_frequency": 0.1,
        }

        total_score = sum(
            scores[factor] * weight for factor, weight in sample_weights.items()
        )

        return min(1.0, max(0.0, total_score))

    def _score_keyword_match(self, text1: str, text2: str) -> float:
        """Score keyword matching between two texts."""
        words1 = set(re.findall(r"\b\w+\b", text1.lower()))
        words2 = set(re.findall(r"\b\w+\b", text2.lower()))

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def _score_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Score semantic similarity (simplified implementation).
        In a full implementation, this would use embeddings.
        """
        # Simple heuristic based on shared patterns
        patterns = [
            r"\b(weather|temperature|forecast|climate)\b",
            r"\b(calculate|compute|math|arithmetic)\b",
            r"\b(search|find|lookup|query)\b",
            r"\b(get|fetch|retrieve|obtain)\b",
        ]

        score = 0.0
        for pattern in patterns:
            if re.search(pattern, text1, re.IGNORECASE) and re.search(
                pattern, text2, re.IGNORECASE
            ):
                score += 0.25

        return min(1.0, score)

class ContextCollector:
    """
    OPTIMIZED: Enhanced context collector with intelligent caching and parallel processing
    """

    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.relevance_scorer = ContextRelevanceScorer()
        
        # OPTIMIZATION: Multi-level caching system
        self.collection_cache = {}
        self.query_cache = {}
        self.relevance_cache = {}
        self.cache_ttl_minutes = 10
        self.max_cache_size = 500
        
        # OPTIMIZATION: Performance monitoring
        self._performance_stats = {
            "query_time": [],
            "scoring_time": [],
            "filtering_time": [],
            "cache_hits": 0,
            "cache_misses": 0,
        }
        
        # OPTIMIZATION: Pre-computed query patterns
        self._query_patterns = self._initialize_query_patterns()
        
        # OPTIMIZATION: Batch processing configuration
        self.batch_size = 50
        self.max_concurrent_queries = 10

    def _initialize_query_patterns(self) -> Dict[str, List[str]]:
        """Initialize common query patterns for optimization"""
        return {
            "weather": ["weather", "temperature", "forecast", "climate"],
            "calculation": ["calculate", "compute", "math", "arithmetic"],
            "search": ["search", "find", "lookup", "query"],
            "data": ["get", "fetch", "retrieve", "obtain"],
            "code": ["code", "programming", "development", "software"],
            "analysis": ["analyze", "evaluate", "assess", "review"],
        }

    async def collect_context_for_claim(
        self,
        claim_content: str,
        context: Dict[str, Any],
        max_skills: int = 5,
        max_samples: int = 10,
    ) -> Dict[str, Any]:
        """
        OPTIMIZED: Collect relevant skills and samples with enhanced caching and parallel processing
        """
        start_time = datetime.utcnow()
        
        try:
            # OPTIMIZATION: Enhanced cache checking with multiple levels
            cache_key = self._generate_cache_key(claim_content, context, max_skills, max_samples)
            cached_result = self._get_from_enhanced_cache(cache_key)
            if cached_result:
                self._performance_stats["cache_hits"] += 1
                return cached_result

            self._performance_stats["cache_misses"] += 1

            # OPTIMIZATION: Parallel skill and sample collection
            skills_task = self.collect_relevant_skills_optimized(
                claim_content, context, max_skills
            )
            samples_task = self.collect_relevant_samples_optimized(
                claim_content, context, max_samples
            )

            # Execute both collections in parallel
            relevant_skills, relevant_samples = await asyncio.gather(
                skills_task, samples_task
            )

            # Build context
            context_result = {
                "claim_content": claim_content,
                "skills": relevant_skills,
                "samples": relevant_samples,
                "collection_timestamp": datetime.utcnow().isoformat(),
                "total_skills": len(relevant_skills),
                "total_samples": len(relevant_samples),
                "query_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
            }

            # Cache the result
            self._add_to_enhanced_cache(cache_key, context_result)

            return context_result

        except Exception as e:
            logger.error(f"Error collecting context: {e}")
            return {
                "claim_content": claim_content,
                "skills": [],
                "samples": [],
                "error": str(e),
                "query_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
            }

    async def collect_relevant_skills_optimized(
        self, claim_content: str, context: Dict[str, Any], max_skills: int = 5
    ) -> List[Dict[str, Any]]:
        """
        OPTIMIZED: Collect relevant skills with batch processing and caching
        """
        try:
            # OPTIMIZATION: Query optimization with pattern matching
            query_start = datetime.utcnow()
            
            # Pre-filter using query patterns
            relevant_tags = self._extract_relevant_tags(claim_content)
            
            # Get skill claims with optimized filtering
            skill_claims = await self._query_skills_optimized(relevant_tags)
            
            query_time = (datetime.utcnow() - query_start).total_seconds()
            self._performance_stats["query_time"].append(query_time)

            # OPTIMIZATION: Batch relevance scoring
            scoring_start = datetime.utcnow()
            
            # Convert to Claim objects in batch
            skill_objects = []
            for claim_dict in skill_claims:
                if self._is_skill_claim(claim_dict):
                    try:
                        skill = Claim(**claim_dict)
                        skill_objects.append((skill, claim_dict))
                    except Exception as e:
                        logger.warning(
                            f"Failed to process skill claim {claim_dict.get('id', 'unknown')}: {e}"
                        )

            # Batch relevance scoring
            skills = await self._batch_score_skills(
                skill_objects, claim_content, context, max_skills
            )
            
            scoring_time = (datetime.utcnow() - scoring_start).total_seconds()
            self._performance_stats["scoring_time"].append(scoring_time)

            return skills

        except Exception as e:
            logger.error(f"Error collecting relevant skills: {e}")
            return []

    async def collect_relevant_skills(
        self, claim_content: str, context: Dict[str, Any], max_skills: int = 5
    ) -> List[Dict[str, Any]]:
        """Legacy method for backward compatibility"""
        return await self.collect_relevant_skills_optimized(claim_content, context, max_skills)

    async def collect_relevant_samples_optimized(
        self, claim_content: str, context: Dict[str, Any], max_samples: int = 10
    ) -> List[Dict[str, Any]]:
        """
        OPTIMIZED: Collect relevant samples with batch processing and caching
        """
        try:
            # OPTIMIZATION: Query optimization
            query_start = datetime.utcnow()
            
            # Pre-filter using query patterns
            relevant_tags = self._extract_relevant_tags(claim_content)
            
            # Get sample claims with optimized filtering
            sample_claims = await self._query_samples_optimized(relevant_tags)
            
            query_time = (datetime.utcnow() - query_start).total_seconds()
            self._performance_stats["query_time"].append(query_time)

            # OPTIMIZATION: Batch relevance scoring
            scoring_start = datetime.utcnow()
            
            # Convert to Claim objects in batch
            sample_objects = []
            for claim_dict in sample_claims:
                if self._is_sample_claim(claim_dict):
                    try:
                        sample = Claim(**claim_dict)
                        sample_objects.append((sample, claim_dict))
                    except Exception as e:
                        logger.warning(
                            f"Failed to process sample claim {claim_dict.get('id', 'unknown')}: {e}"
                        )

            # Batch relevance scoring
            samples = await self._batch_score_samples(
                sample_objects, claim_content, context, max_samples
            )
            
            scoring_time = (datetime.utcnow() - scoring_start).total_seconds()
            self._performance_stats["scoring_time"].append(scoring_time)

            return samples

        except Exception as e:
            logger.error(f"Error collecting relevant samples: {e}")
            return []

    async def collect_relevant_samples(
        self, claim_content: str, context: Dict[str, Any], max_samples: int = 10
    ) -> List[Dict[str, Any]]:
        """Legacy method for backward compatibility"""
        return await self.collect_relevant_samples_optimized(claim_content, context, max_samples)

    def build_llm_context_string(self, context_result: Dict[str, Any]) -> str:
        """
        Build a formatted context string for LLM consumption.

        Args:
            context_result: Result from collect_context_for_claim

        Returns:
            Formatted context string
        """
        context_parts = []

        # Add header
        context_parts.append("=== RELEVANT CONTEXT ===")
        context_parts.append(f"Claim: {context_result.get('claim_content', '')}")
        context_parts.append("")

        # Add skills
        skills = context_result.get("skills", [])
        if skills:
            context_parts.append("RELEVANT SKILLS:")
            for i, skill_data in enumerate(skills, 1):
                context_parts.append(
                    f"Skill {i} (relevance: {skill_data['relevance_score']:.2f}):"
                )
                context_parts.append(skill_data["context_format"])
                context_parts.append("")

        # Add samples
        samples = context_result.get("samples", [])
        if samples:
            context_parts.append("RELEVANT SAMPLES:")
            for i, sample_data in enumerate(samples, 1):
                context_parts.append(
                    f"Sample {i} (relevance: {sample_data['relevance_score']:.2f}):"
                )
                context_parts.append(sample_data["context_format"])
                context_parts.append("")

        # Add footer
        context_parts.append("=== END CONTEXT ===")

        return "\n".join(context_parts)

    async def update_usage_stats(
        self, context_result: Dict[str, Any], feedback: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update usage statistics based on context usage.

        Args:
            context_result: Context that was used
            feedback: Optional feedback on usefulness
        """
        try:
            # Update skill usage stats
            for concept_data in context_result.get("concepts", []):
                concept = concept_data["concept"]
                # Simplified - no usage tracking

                if feedback:
                    # Simplified - no specialized feedback
                    # Simplified - no update methods
                    pass

                # Update in database
                await self.data_manager.update_claim(
                    skill.id, confidence=concept.confidence
                )

            # Update sample usage stats
            for sample_data in context_result.get("samples", []):
                sample = sample_data["sample"]
                helpfulness = None

                if feedback:
                    helpfulness = feedback.get("sample_helpfulness", {}).get(sample.id)

                sample.update_usage_stats(helpfulness)

                # Update in database
                # Simplified update for base Claim model
                await self.data_manager.update_claim(
                    sample.id, confidence=sample.confidence
                )

        except Exception as e:
            logger.error(f"Error updating usage stats: {e}")

    def _extract_relevant_tags(self, claim_content: str) -> List[str]:
        """OPTIMIZATION: Extract relevant tags based on query patterns"""
        content_lower = claim_content.lower()
        relevant_tags = []
        
        for pattern_name, keywords in self._query_patterns.items():
            if any(keyword in content_lower for keyword in keywords):
                relevant_tags.append(pattern_name)
        
        return relevant_tags

    def _is_skill_claim(self, claim_dict: Dict[str, Any]) -> bool:
        """OPTIMIZATION: Check if claim is a skill claim"""
        tags = claim_dict.get("tags", [])
        return "type.concept" in tags or "concept" in tags

    def _is_sample_claim(self, claim_dict: Dict[str, Any]) -> bool:
        """OPTIMIZATION: Check if claim is a sample claim"""
        tags = claim_dict.get("tags", [])
        return "type.sample" in tags or "sample" in tags

    async def _query_skills_optimized(self, relevant_tags: List[str]) -> List[Dict[str, Any]]:
        """OPTIMIZATION: Query skills with optimized filtering"""
        try:
            # Use optimized query if tags are available
            if relevant_tags:
                # This would be implemented in DataManager for tag-based filtering
                filter_obj = {"tags": relevant_tags, "limit": 100}
            else:
                filter_obj = {"limit": 100}
            
            skill_claims = await self.data_manager.filter_claims(filter_obj)
            return skill_claims
            
        except Exception as e:
            logger.error(f"Error in optimized skills query: {e}")
            # Fallback to basic query
            return await self.data_manager.filter_claims({"limit": 100})

    async def _query_samples_optimized(self, relevant_tags: List[str]) -> List[Dict[str, Any]]:
        """OPTIMIZATION: Query samples with optimized filtering"""
        try:
            # Use optimized query if tags are available
            if relevant_tags:
                filter_obj = {"tags": relevant_tags, "limit": 200}
            else:
                filter_obj = {"limit": 200}
            
            sample_claims = await self.data_manager.filter_claims(filter_obj)
            return sample_claims
            
        except Exception as e:
            logger.error(f"Error in optimized samples query: {e}")
            # Fallback to basic query
            return await self.data_manager.filter_claims({"limit": 200})

    async def _batch_score_skills(
        self, skill_objects: List[Tuple[Claim, Dict[str, Any]]],
        claim_content: str, context: Dict[str, Any], max_skills: int
    ) -> List[Dict[str, Any]]:
        """OPTIMIZATION: Batch score skills with parallel processing"""
        if not skill_objects:
            return []
        
        # Create scoring tasks for parallel processing
        scoring_tasks = []
        for skill, claim_dict in skill_objects:
            # Check relevance cache first
            relevance_key = self._generate_relevance_cache_key(skill.id, claim_content)
            cached_score = self._get_from_relevance_cache(relevance_key)
            
            if cached_score is not None:
                if cached_score > 0.1:  # Only include relevant skills
                    scoring_tasks.append(
                        self._create_skill_result(skill, cached_score, claim_dict)
                    )
            else:
                # Create scoring task
                scoring_tasks.append(
                    self._score_and_cache_skill(skill, claim_content, context, claim_dict)
                )
        
        # Execute scoring tasks in batches
        skills = []
        batch_size = min(self.batch_size, len(scoring_tasks))
        
        for i in range(0, len(scoring_tasks), batch_size):
            batch = scoring_tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Error in batch scoring: {result}")
                elif result and result.get("relevance_score", 0) > 0.1:
                    skills.append(result)
        
        # Sort by relevance score and return top skills
        skills.sort(key=lambda x: x["relevance_score"], reverse=True)
        return skills[:max_skills]

    async def _batch_score_samples(
        self, sample_objects: List[Tuple[Claim, Dict[str, Any]]],
        claim_content: str, context: Dict[str, Any], max_samples: int
    ) -> List[Dict[str, Any]]:
        """OPTIMIZATION: Batch score samples with parallel processing"""
        if not sample_objects:
            return []
        
        # Create scoring tasks for parallel processing
        scoring_tasks = []
        for sample, claim_dict in sample_objects:
            # Check relevance cache first
            relevance_key = self._generate_relevance_cache_key(sample.id, claim_content)
            cached_score = self._get_from_relevance_cache(relevance_key)
            
            if cached_score is not None:
                if cached_score > 0.1:  # Only include relevant samples
                    scoring_tasks.append(
                        self._create_sample_result(sample, cached_score, claim_dict)
                    )
            else:
                # Create scoring task
                scoring_tasks.append(
                    self._score_and_cache_sample(sample, claim_content, context, claim_dict)
                )
        
        # Execute scoring tasks in batches
        samples = []
        batch_size = min(self.batch_size, len(scoring_tasks))
        
        for i in range(0, len(scoring_tasks), batch_size):
            batch = scoring_tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Error in batch scoring: {result}")
                elif result and result.get("relevance_score", 0) > 0.1:
                    samples.append(result)
        
        # Sort by relevance score and return top samples
        samples.sort(key=lambda x: x["relevance_score"], reverse=True)
        return samples[:max_samples]

    async def _score_and_cache_skill(
        self, skill: Claim, claim_content: str, context: Dict[str, Any], claim_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Score skill and cache result"""
        relevance_score = self.relevance_scorer.score_skill_relevance(
            skill, claim_content, context
        )
        
        # Cache the relevance score
        relevance_key = self._generate_relevance_cache_key(skill.id, claim_content)
        self._add_to_relevance_cache(relevance_key, relevance_score)
        
        return self._create_skill_result(skill, relevance_score, claim_dict)

    async def _score_and_cache_sample(
        self, sample: Claim, claim_content: str, context: Dict[str, Any], claim_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Score sample and cache result"""
        relevance_score = self.relevance_scorer.score_sample_relevance(
            sample, claim_content, context
        )
        
        # Cache the relevance score
        relevance_key = self._generate_relevance_cache_key(sample.id, claim_content)
        self._add_to_relevance_cache(relevance_key, relevance_score)
        
        return self._create_sample_result(sample, relevance_score, claim_dict)

    def _create_skill_result(
        self, skill: Claim, relevance_score: float, claim_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create skill result dictionary"""
        return {
            "concept": skill,
            "relevance_score": relevance_score,
            "context_format": skill.to_llm_context(),
        }

    def _create_sample_result(
        self, sample: Claim, relevance_score: float, claim_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create sample result dictionary"""
        return {
            "sample": sample,
            "relevance_score": relevance_score,
            "context_format": sample.format_for_llm_context(),
        }

    def _generate_relevance_cache_key(self, claim_id: str, claim_content: str) -> str:
        """Generate cache key for relevance scoring"""
        key_data = f"{claim_id}:{hashlib.md5(claim_content.encode()).hexdigest()}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_from_relevance_cache(self, cache_key: str) -> Optional[float]:
        """Get relevance score from cache"""
        if cache_key in self.relevance_cache:
            cached_item = self.relevance_cache[cache_key]
            timestamp = cached_item.get("timestamp", 0)
            
            # Check if cache is still valid
            if time.time() - timestamp < (self.cache_ttl_minutes * 60):
                return cached_item.get("score")
            else:
                # Remove expired cache item
                del self.relevance_cache[cache_key]
        
        return None

    def _add_to_relevance_cache(self, cache_key: str, score: float) -> None:
        """Add relevance score to cache"""
        self.relevance_cache[cache_key] = {
            "score": score,
            "timestamp": time.time(),
        }
        
        # Maintain cache size
        if len(self.relevance_cache) > self.max_cache_size:
            # Remove oldest items
            oldest_keys = sorted(
                self.relevance_cache.keys(),
                key=lambda k: self.relevance_cache[k]["timestamp"],
            )[:self.max_cache_size // 4]  # Remove 25% of items
            
            for key in oldest_keys:
                del self.relevance_cache[key]

    def _generate_cache_key(self, claim_content: str, context: Dict[str, Any], max_skills: int, max_samples: int) -> str:
        """OPTIMIZATION: Generate enhanced cache key for context collection"""
        context_hash = hashlib.md5(str(sorted(context.items())).hexdigest()
)
        key_data = f"{hashlib.md5(claim_content.encode()).hexdigest()}:{context_hash}:{max_skills}:{max_samples}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_from_enhanced_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """OPTIMIZATION: Get cached context result with enhanced validation"""
        # Check collection cache first
        if cache_key in self.collection_cache:
            cached_item = self.collection_cache[cache_key]
            timestamp = datetime.fromisoformat(cached_item["timestamp"])

            # Check if cache is still valid
            if datetime.utcnow() - timestamp < timedelta(
                minutes=self.cache_ttl_minutes
            ):
                return cached_item["data"]
            else:
                # Remove expired cache item
                del self.collection_cache[cache_key]

        return None

    def _add_to_enhanced_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """OPTIMIZATION: Add context result to enhanced cache with size management"""
        self.collection_cache[cache_key] = {
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        }

        # Maintain cache size with LRU eviction
        if len(self.collection_cache) > self.max_cache_size:
            # Remove oldest items (25% of cache)
            oldest_keys = sorted(
                self.collection_cache.keys(),
                key=lambda k: self.collection_cache[k]["timestamp"],
            )[:self.max_cache_size // 4]

            for key in oldest_keys:
                del self.collection_cache[key]

    def get_performance_stats(self) -> Dict[str, Any]:
        """OPTIMIZATION: Get detailed performance statistics"""
        stats = {
            "cache_performance": {
                "hits": self._performance_stats["cache_hits"],
                "misses": self._performance_stats["cache_misses"],
                "hit_rate": (
                    self._performance_stats["cache_hits"] /
                    max(1, self._performance_stats["cache_hits"] + self._performance_stats["cache_misses"])
                ) * 100
            },
            "timing_breakdown": {},
            "cache_sizes": {
                "collection_cache": len(self.collection_cache),
                "relevance_cache": len(self.relevance_cache),
                "max_cache_size": self.max_cache_size,
            }
        }
        
        # Calculate timing statistics
        timing_metrics = ["query_time", "scoring_time", "filtering_time"]
        for metric in timing_metrics:
            times = self._performance_stats.get(metric, [])
            if times:
                stats["timing_breakdown"][metric] = {
                    "count": len(times),
                    "average": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "total": sum(times)
                }
        
        return stats

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about context collection."""
        try:
            # Get skill and sample counts
            skill_claims = await self.data_manager.filter_claims(
                filters=None  # Will be implemented to filter by tags
            )

            skill_count = sum(
                1 for claim in skill_claims if "type.concept" in claim.get("tags", [])
            )
            sample_count = sum(
                1 for claim in skill_claims if "type.sample" in claim.get("tags", [])
            )

            return {
                "total_skills": skill_count,
                "total_samples": sample_count,
                "cache_size": len(self.collection_cache),
                "cache_ttl_minutes": self.cache_ttl_minutes,
            }

        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {
                "total_skills": 0,
                "total_samples": 0,
                "cache_size": len(self.collection_cache),
                "cache_ttl_minutes": self.cache_ttl_minutes,
            }

    def clear_cache(self) -> None:
        """Clear the context collection cache."""
        self.collection_cache.clear()
        logger.info("Context collection cache cleared")
