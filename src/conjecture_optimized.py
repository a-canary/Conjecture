"""
Optimized Conjecture: Async Evidence-Based AI Reasoning System
PERFORMANCE OPTIMIZED: Minimal startup time and memory footprint
"""

import asyncio
import time
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
from functools import lru_cache
import hashlib

# Suppress heavy library imports during initialization
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

# Minimal core imports only
from src.core.models import Claim, ClaimState
from src.config.unified_config import UnifiedConfig as Config

class OptimizedConjecture:
    """
    Performance-optimized Conjecture with lazy loading and minimal startup overhead
    """

    def __init__(self, config: Optional[Config] = None):
        """OPTIMIZED: Initialize with minimal dependencies"""
        self.config = config or Config()

        # Defer all heavy imports
        self._llm_manager = None
        self._data_manager = None
        self._claim_repository = None
        self._context_collector = None
        self._async_evaluation = None
        self._tool_creator = None
        self._performance_monitor = None

        # Track what's been initialized
        self._initialized_components = set()

        # Performance stats
        self._performance_stats = {
            "startup_time": 0.0,
            "component_init_times": {},
            "api_call_times": [],
            "cache_hits": 0,
            "cache_misses": 0,
        }

        self.logger = logging.getLogger(__name__)

        # Simple caches with memory limits
        self._simple_cache = {}
        self._cache_max_size = 20
        self._cache_ttl = 300  # 5 minutes

        print(f"Optimized Conjecture initialized with lazy loading")

    @property
    def llm_manager(self):
        """Lazy load LLM manager"""
        if self._llm_manager is None:
            start_time = time.time()
            self._load_llm_bridge()
            self._performance_stats["component_init_times"]["llm_manager"] = time.time() - start_time
        return self._llm_manager

    @property
    def data_manager(self):
        """Lazy load data manager"""
        if self._data_manager is None:
            start_time = time.time()
            self._load_data_manager()
            self._performance_stats["component_init_times"]["data_manager"] = time.time() - start_time
        return self._data_manager

    @property
    def claim_repository(self):
        """Lazy load claim repository"""
        if self._claim_repository is None:
            start_time = time.time()
            self._load_claim_repository()
            self._performance_stats["component_init_times"]["claim_repository"] = time.time() - start_time
        return self._claim_repository

    def _load_llm_bridge(self):
        """Load LLM bridge components"""
        try:
            # Use simplified LLM manager directly for better performance
            from src.processing.simplified_llm_manager import get_simplified_llm_manager
            self._llm_manager = get_simplified_llm_manager()
            self._initialized_components.add("llm_manager")

            if self._llm_manager.get_available_providers():
                print(f"LLM Manager: Connected to {len(self._llm_manager.get_available_providers())} providers")
            else:
                print("LLM Manager: No providers available, using fallback mode")

        except Exception as e:
            print(f"LLM Manager initialization failed: {e}")
            self._llm_manager = None

    def _load_data_manager(self):
        """Load data manager with minimal dependencies"""
        try:
            from src.data.repositories import get_data_manager
            self._data_manager = get_data_manager(use_mock_embeddings=False)
            self._initialized_components.add("data_manager")
        except Exception as e:
            print(f"Data manager initialization failed: {e}")
            self._data_manager = None

    def _load_claim_repository(self):
        """Load claim repository"""
        if self.data_manager is None:
            return
        try:
            from src.data.repositories import RepositoryFactory
            self._claim_repository = RepositoryFactory.create_claim_repository(self.data_manager)
            self._initialized_components.add("claim_repository")
        except Exception as e:
            print(f"Claim repository initialization failed: {e}")
            self._claim_repository = None

    def _get_context_collector(self):
        """Get context collector with lazy loading"""
        if self._context_collector is None:
            try:
                from src.processing.context_collector import ContextCollector
                if self.data_manager:
                    self._context_collector = ContextCollector(self.data_manager)
                else:
                    self._context_collector = None
            except Exception as e:
                print(f"Context collector initialization failed: {e}")
                self._context_collector = None
        return self._context_collector

    async def start_services(self):
        """Start background services with minimal overhead"""
        if "services_started" in self._initialized_components:
            return

        # Initialize data manager if not already done
        if self.data_manager:
            try:
                await self.data_manager.initialize()
            except Exception as e:
                print(f"Services initialization failed: {e}")

        self._initialized_components.add("services_started")

    async def stop_services(self):
        """Stop background services and cleanup"""
        self._clear_cache()
        print("Optimized Conjecture services stopped")

    def _get_cache_key(self, *args) -> str:
        """Generate cache key"""
        key_str = ":".join(str(arg) for arg in args)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str):
        """Get from cache with TTL check"""
        if cache_key in self._simple_cache:
            item = self._simple_cache[cache_key]
            if time.time() - item["timestamp"] < self._cache_ttl:
                self._performance_stats["cache_hits"] += 1
                return item["data"]
            else:
                del self._simple_cache[cache_key]

        self._performance_stats["cache_misses"] += 1
        return None

    def _add_to_cache(self, cache_key: str, data: Any):
        """Add to cache with size management"""
        if len(self._simple_cache) >= self._cache_max_size:
            # Remove oldest item
            oldest_key = min(self._simple_cache.keys(),
                           key=lambda k: self._simple_cache[k]["timestamp"])
            del self._simple_cache[oldest_key]

        self._simple_cache[cache_key] = {
            "data": data,
            "timestamp": time.time()
        }

    def _clear_cache(self):
        """Clear all caches"""
        self._simple_cache.clear()

    async def explore(
        self,
        query: str,
        max_claims: int = 10,
        claim_types: Optional[List[str]] = None,
        confidence_threshold: Optional[float] = None,
        auto_evaluate: bool = False,  # Default to False for performance
    ) -> "ExplorationResult":
        """
        Optimized exploration with minimal overhead and caching
        """
        start_time = time.time()

        if not query or len(query.strip()) < 5:
            raise ValueError("Query must be at least 5 characters long")

        confidence_threshold = confidence_threshold or self.config.confidence_threshold

        print(f"OPTIMIZED exploration: '{query}'")

        try:
            # Check cache first
            cache_key = self._get_cache_key(query, max_claims, confidence_threshold)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                print(f"Returning cached result for: '{query}'")
                return cached_result

            # Start services if needed
            if "services_started" not in self._initialized_components:
                await self.start_services()

            # Generate claims with minimal LLM overhead
            claim_gen_start = time.time()
            claims = await self._generate_claims_optimized(query, max_claims)
            claim_gen_time = time.time() - claim_gen_start

            # Filter by confidence threshold
            filtered_claims = [
                claim for claim in claims
                if claim.confidence >= confidence_threshold
            ]

            # Store claims efficiently
            stored_claims = []
            if self.claim_repository:
                for claim in filtered_claims:
                    try:
                        claim_data = {
                            "content": claim.content,
                            "confidence": claim.confidence,
                            "tags": claim.tags,
                            "state": ClaimState.EXPLORE,
                        }
                        stored_claim = await self.claim_repository.create(claim_data)
                        stored_claims.append(stored_claim)
                    except Exception as e:
                        self.logger.warning(f"Failed to store claim: {e}")
                        # Include original claim if storage fails
                        stored_claims.append(claim)
            else:
                stored_claims = filtered_claims

            processing_time = time.time() - start_time

            result = OptimizedExplorationResult(
                query=query,
                claims=stored_claims,
                total_found=len(filtered_claims),
                search_time=processing_time,
                confidence_threshold=confidence_threshold,
                max_claims=max_claims,
                evaluation_pending=auto_evaluate,
                optimization_stats={
                    "claim_generation_time": claim_gen_time,
                    "cache_hit": False,
                    "components_used": list(self._initialized_components)
                }
            )

            # Cache the result
            self._add_to_cache(cache_key, result)

            print(f"OPTIMIZED exploration completed: {len(result.claims)} claims in {result.search_time:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"Error in optimized exploration: {e}")
            raise

    async def _generate_claims_optimized(self, query: str, max_claims: int) -> List[Claim]:
        """Generate claims with optimized LLM usage"""
        if not self.llm_manager:
            # Fallback: create simple claims
            return self._create_fallback_claims(query, max_claims)

        try:
            # Use minimal context to reduce processing time
            prompt = f"""Generate {max_claims} high-quality claims about: {query}

Requirements:
- Use XML format: <claim type="[fact|concept|example]" confidence="[0.0-1.0]">content</claim>
- Clear, specific statements
- Realistic confidence scores
- Cover different aspects

<claims>
  <claim type="fact" confidence="0.8">Your factual claim here</claim>
  <claim type="concept" confidence="0.7">Your conceptual claim here</claim>
</claims>"""

            # Import GenerationConfig for proper parameter passing
            from src.processing.llm.common import GenerationConfig

            config = GenerationConfig(
                max_tokens=1500,  # Reduced for performance
                temperature=0.7,
                top_p=0.8
            )

            response = self.llm_manager.generate_response(
                prompt=prompt,
                config=config
            )

            if response.success:
                return self._parse_claims_optimized(response.content)
            else:
                print(f"LLM processing failed: {response.errors}")
                return self._create_fallback_claims(query, max_claims)

        except Exception as e:
            print(f"Error generating claims: {e}")
            return self._create_fallback_claims(query, max_claims)

    def _parse_claims_optimized(self, response: str) -> List[Claim]:
        """Optimized claim parsing"""
        claims = []

        try:
            # Simple XML parsing for performance
            import re
            claim_pattern = r'<claim\s+type="([^"]+)"\s+confidence="([^"]+)">([^<]+)</claim>'
            matches = re.findall(claim_pattern, response)

            for i, (claim_type, confidence_str, content) in enumerate(matches[:10]):  # Limit to 10
                try:
                    confidence = float(confidence_str)
                    claim = Claim(
                        id=f"optimized_{int(time.time())}_{i}",
                        content=content.strip(),
                        confidence=confidence,
                        tags=[claim_type, "auto_generated", "optimized"],
                        state=ClaimState.EXPLORE,
                    )
                    claims.append(claim)
                except (ValueError, TypeError):
                    continue

        except Exception as e:
            self.logger.warning(f"Optimized parsing failed: {e}")

        if not claims:
            return self._create_fallback_claims(response, 3)

        return claims

    def _create_fallback_claims(self, content: str, max_claims: int) -> List[Claim]:
        """Create fallback claims when LLM is unavailable"""
        claims = []

        # Split content into meaningful chunks
        sentences = content.split('.')
        for i, sentence in enumerate(sentences[:max_claims]):
            sentence = sentence.strip()
            if len(sentence) > 20:  # Only substantial sentences
                claim = Claim(
                    id=f"fallback_{int(time.time())}_{i}",
                    content=sentence,
                    confidence=0.6,  # Lower confidence for fallback
                    tags=["fallback", "auto_generated"],
                    state=ClaimState.EXPLORE,
                )
                claims.append(claim)

        return claims

    async def add_claim(
        self,
        content: str,
        confidence: float,
        tags: Optional[List[str]] = None,
        auto_evaluate: bool = False,  # Default to False for performance
        **kwargs,
    ) -> Claim:
        """Optimized claim creation"""
        if len(content.strip()) < 10:
            raise ValueError("Content must be at least 10 characters long")
        if not (0.0 <= confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")

        claim = Claim(
            id=f"manual_{int(time.time())}",
            content=content.strip(),
            confidence=confidence,
            tags=tags or [],
            state=ClaimState.EXPLORE,
            **kwargs,
        )

        # Store if repository available
        if self.claim_repository:
            try:
                claim_data = {
                    "content": claim.content,
                    "confidence": claim.confidence,
                    "tags": claim.tags,
                    "state": claim.state,
                }
                stored_claim = await self.claim_repository.create(claim_data)
                return stored_claim
            except Exception as e:
                self.logger.warning(f"Failed to store claim: {e}")

        return claim

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            "performance": self._performance_stats,
            "cache_stats": {
                "size": len(self._simple_cache),
                "hits": self._performance_stats["cache_hits"],
                "misses": self._performance_stats["cache_misses"],
                "hit_rate": (
                    self._performance_stats["cache_hits"] /
                    max(1, self._performance_stats["cache_hits"] + self._performance_stats["cache_misses"])
                ) * 100
            },
            "initialized_components": list(self._initialized_components),
            "memory_optimized": True
        }

        return stats

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start_services()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop_services()


class OptimizedExplorationResult:
    """Optimized exploration result with performance metrics"""

    def __init__(
        self,
        query: str,
        claims: List[Claim],
        total_found: int,
        search_time: float,
        confidence_threshold: float,
        max_claims: int,
        evaluation_pending: bool = False,
        optimization_stats: Optional[Dict[str, Any]] = None,
    ):
        self.query = query
        self.claims = claims
        self.total_found = total_found
        self.search_time = search_time
        self.confidence_threshold = confidence_threshold
        self.max_claims = max_claims
        self.evaluation_pending = evaluation_pending
        self.timestamp = datetime.utcnow()
        self.optimization_stats = optimization_stats or {}

    def summary(self) -> str:
        """Provide performance-focused summary"""
        if not self.claims:
            return f"No claims found for '{self.query}' above confidence threshold {self.confidence_threshold}"

        lines = [
            f"🚀 Optimized Exploration: '{self.query}'",
            f"📊 Found: {len(self.claims)} claims (of {self.total_found} total)",
            f"⚡ Time: {self.search_time:.2f}s",
            f"🎚️  Confidence: ≥{self.confidence_threshold}",
            f"🧠 Components: {', '.join(self.optimization_stats.get('components_used', []))}",
            "",
            "📋 Top Claims:",
        ]

        for i, claim in enumerate(self.claims[:5], 1):
            tags_str = ",".join(claim.tags) if claim.tags else "none"
            lines.append(
                f"  {i}. [{claim.confidence:.2f}, {tags_str}] {claim.content[:80]}{'...' if len(claim.content) > 80 else ''}"
            )

        if len(self.claims) > 5:
            lines.append(f"  ... and {len(self.claims) - 5} more claims")

        return "\n".join(lines)


# Convenience functions for optimized usage
async def explore_optimized(query: str, **kwargs) -> OptimizedExplorationResult:
    """Quick optimized exploration function"""
    async with OptimizedConjecture() as cf:
        return await cf.explore(query, **kwargs)


if __name__ == "__main__":

    async def test_optimized_conjecture():
        print("🧪 Testing Optimized Conjecture")
        print("=" * 40)

        start_time = time.time()

        async with OptimizedConjecture() as cf:
            init_time = time.time() - start_time
            print(f"✅ Initialization time: {init_time:.3f}s")

            # Test optimized exploration
            print("\n🔍 Testing optimized exploration...")
            explore_start = time.time()
            result = await cf.explore("quantum computing applications", max_claims=3)
            explore_time = time.time() - explore_start

            print(f"✅ Exploration time: {explore_time:.3f}s")
            print(result.summary())

            # Test performance stats
            print("\n📊 Performance stats:")
            stats = cf.get_performance_stats()
            print(f"  Cache hit rate: {stats['cache_stats']['hit_rate']:.1f}%")
            print(f"  Components initialized: {len(stats['initialized_components'])}")

        total_time = time.time() - start_time
        print(f"\n🎉 Total test time: {total_time:.3f}s")

    asyncio.run(test_optimized_conjecture())