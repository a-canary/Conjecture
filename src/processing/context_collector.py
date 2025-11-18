"""
Context Collector for the Conjecture skill-based agency system.
Collects relevant skills and samples for LLM context building.
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import re

from core.unified_models import Claim
from data.data_manager import DataManager


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
    Collects relevant skills and samples for LLM context building.
    """

    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.relevance_scorer = ContextRelevanceScorer()
        self.collection_cache = {}
        self.cache_ttl_minutes = 10

    async def collect_context_for_claim(
        self,
        claim_content: str,
        context: Dict[str, Any],
        max_skills: int = 5,
        max_samples: int = 10,
    ) -> Dict[str, Any]:
        """
        Collect relevant skills and samples for a claim.

        Args:
            claim_content: Content of the claim being evaluated
            context: Additional context for collection
            max_skills: Maximum number of skills to collect
            max_samples: Maximum number of samples to collect

        Returns:
            Dictionary with collected context
        """
        try:
            # Check cache first
            cache_key = self._generate_cache_key(claim_content, context)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return cached_result

            # Collect skills
            relevant_skills = await self.collect_relevant_skills(
                claim_content, context, max_skills
            )

            # Collect samples
            relevant_samples = await self.collect_relevant_samples(
                claim_content, context, max_samples
            )

            # Build context
            context_result = {
                "claim_content": claim_content,
                "skills": relevant_skills,
                "samples": relevant_samples,
                "collection_timestamp": datetime.utcnow().isoformat(),
                "total_skills": len(relevant_skills),
                "total_samples": len(relevant_samples),
            }

            # Cache the result
            self._add_to_cache(cache_key, context_result)

            return context_result

        except Exception as e:
            logger.error(f"Error collecting context: {e}")
            return {
                "claim_content": claim_content,
                "skills": [],
                "samples": [],
                "error": str(e),
            }

    async def collect_relevant_skills(
        self, claim_content: str, context: Dict[str, Any], max_skills: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Collect relevant skills for a claim.

        Args:
            claim_content: Content of the claim being evaluated
            context: Additional context
            max_skills: Maximum number of skills to return

        Returns:
            List of relevant skills with scores
        """
        try:
            # Get all skill claims
            skill_claims = await self.data_manager.filter_claims(
                filters=None  # Will be implemented to filter by tags
            )

            # Filter for skill claims
            skills = []
            for claim_dict in skill_claims:
                if "type.concept" in claim_dict.get("tags", []):
                    try:
                        skill = Claim(**claim_dict)
                        relevance_score = self.relevance_scorer.score_skill_relevance(
                            skill, claim_content, context
                        )

                        if relevance_score > 0.1:  # Only include relevant skills
                            skills.append(
                                {
                                    "concept": skill,
                                    "relevance_score": relevance_score,
                                    "context_format": skill.to_llm_context(),
                                }
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to process skill claim {claim_dict.get('id', 'unknown')}: {e}"
                        )

            # Sort by relevance score
            skills.sort(key=lambda x: x["relevance_score"], reverse=True)

            # Return top skills
            return skills[:max_skills]

        except Exception as e:
            logger.error(f"Error collecting relevant skills: {e}")
            return []

    async def collect_relevant_samples(
        self, claim_content: str, context: Dict[str, Any], max_samples: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Collect relevant samples for a claim.

        Args:
            claim_content: Content of the claim being evaluated
            context: Additional context
            max_samples: Maximum number of samples to return

        Returns:
            List of relevant samples with scores
        """
        try:
            # Get all sample claims
            sample_claims = await self.data_manager.filter_claims(
                filters=None  # Will be implemented to filter by tags
            )

            # Filter for sample claims
            samples = []
            for claim_dict in sample_claims:
                if "type.sample" in claim_dict.get("tags", []):
                    try:
                        sample = Claim(**claim_dict)
                        relevance_score = self.relevance_scorer.score_sample_relevance(
                            sample, claim_content, context
                        )

                        if relevance_score > 0.1:  # Only include relevant samples
                            samples.append(
                                {
                                    "sample": sample,
                                    "relevance_score": relevance_score,
                                    "context_format": sample.format_for_llm_context(),
                                }
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to process sample claim {claim_dict.get('id', 'unknown')}: {e}"
                        )

            # Sort by relevance score
            samples.sort(key=lambda x: x["relevance_score"], reverse=True)

            # Return top samples
            return samples[:max_samples]

        except Exception as e:
            logger.error(f"Error collecting relevant samples: {e}")
            return []

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

    def _generate_cache_key(self, claim_content: str, context: Dict[str, Any]) -> str:
        """Generate cache key for context collection."""
        import hashlib

        key_data = f"{claim_content}:{str(sorted(context.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached context result."""
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

    def _add_to_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Add context result to cache."""
        self.collection_cache[cache_key] = {
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        }

        # Maintain cache size
        if len(self.collection_cache) > 100:
            # Remove oldest items
            oldest_keys = sorted(
                self.collection_cache.keys(),
                key=lambda k: self.collection_cache[k]["timestamp"],
            )[:20]

            for key in oldest_keys:
                del self.collection_cache[key]

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
