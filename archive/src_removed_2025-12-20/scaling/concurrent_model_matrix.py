"""
Concurrent Model Matrix Implementation for Scaling Analysis - Phase 2
Enables parallel execution across multiple LLM providers with load balancing
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict

from ..processing.simplified_llm_manager import SimplifiedLLMManager, get_simplified_llm_manager
from ..processing.unified_bridge import UnifiedLLMBridge as LLMBridge, LLMRequest
from ..core.models import Claim, ClaimState
from ..monitoring import get_performance_monitor

logger = logging.getLogger(__name__)

class ProviderStatus(Enum):
    """Provider status enumeration"""
    AVAILABLE = "available"
    BUSY = "busy"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"

@dataclass
class ProviderMetrics:
    """Metrics for an individual provider"""
    name: str
    status: ProviderStatus = ProviderStatus.AVAILABLE
    requests_completed: int = 0
    requests_failed: int = 0
    average_response_time: float = 0.0
    last_request_time: Optional[datetime] = None
    rate_limit_reset: Optional[datetime] = None
    concurrent_requests: int = 0
    max_concurrent: int = 3

@dataclass
class ModelMatrixResult:
    """Result from concurrent model matrix execution"""
    query: str
    provider_results: Dict[str, Dict[str, Any]]
    execution_time: float
    consensus_confidence: float
    best_provider: str
    consensus_found: bool
    error_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

class ConcurrentModelMatrix:
    """
    Enhanced Model Matrix with concurrent execution capabilities
    Supports parallel testing across multiple LLM providers with intelligent load balancing
    """

    def __init__(
        self,
        llm_manager: Optional[SimplifiedLLMManager] = None,
        max_concurrent_per_provider: int = 3,
        consensus_threshold: float = 0.7,
        timeout_per_request: int = 30,
        enable_load_balancing: bool = True
    ):
        self.llm_manager = llm_manager or get_simplified_llm_manager()
        self.max_concurrent_per_provider = max_concurrent_per_provider
        self.consensus_threshold = consensus_threshold
        self.timeout_per_request = timeout_per_request
        self.enable_load_balancing = enable_load_balancing

        # Provider management
        self.provider_metrics: Dict[str, ProviderMetrics] = {}
        self.provider_semaphores: Dict[str, asyncio.Semaphore] = {}
        self._initialize_providers()

        # Performance tracking
        self.performance_monitor = get_performance_monitor()
        self.execution_history: List[ModelMatrixResult] = []
        self.matrix_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "consensus_found": 0,
            "average_execution_time": 0.0,
            "provider_utilization": defaultdict(float)
        }

        # Thread pool for CPU-intensive tasks
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Lock for thread safety
        self._stats_lock = threading.RLock()

        logger.info(f"ConcurrentModelMatrix initialized with {len(self.provider_metrics)} providers")

    def _initialize_providers(self):
        """Initialize provider metrics and semaphores"""
        provider_info = self.llm_manager.get_provider_info()

        for provider_name in provider_info["available_providers"]:
            self.provider_metrics[provider_name] = ProviderMetrics(
                name=provider_name,
                max_concurrent=self.max_concurrent_per_provider
            )
            self.provider_semaphores[provider_name] = asyncio.Semaphore(
                self.max_concurrent_per_provider
            )

        logger.info(f"Initialized {len(self.provider_metrics)} providers: {list(self.provider_metrics.keys())}")

    async def execute_concurrent_query(
        self,
        query: str,
        providers: Optional[List[str]] = None,
        max_results: Optional[int] = None,
        analyze_consensus: bool = True
    ) -> ModelMatrixResult:
        """
        Execute a query concurrently across multiple providers

        Args:
            query: The query to execute
            providers: Specific providers to use (None = all available)
            max_results: Maximum number of results to wait for
            analyze_consensus: Whether to analyze consensus between providers

        Returns:
            ModelMatrixResult with execution details
        """
        execution_start = time.time()

        # Determine which providers to use
        available_providers = self._get_available_providers(providers)

        if not available_providers:
            raise RuntimeError("No available providers for concurrent execution")

        logger.info(f"Executing concurrent query across {len(available_providers)} providers: {available_providers}")

        # Create concurrent tasks
        tasks = []
        for provider_name in available_providers:
            task = asyncio.create_task(
                self._execute_provider_query(provider_name, query)
            )
            tasks.append((provider_name, task))

        # Wait for results (with timeout)
        results = {}
        timeout_occurred = False

        try:
            # Wait for all tasks or timeout
            completed_tasks = await asyncio.wait(
                [task for _, task in tasks],
                timeout=self.timeout_per_request,
                return_when=asyncio.ALL_COMPLETED
            )[0]

            # Collect results from completed tasks
            for provider_name, original_task in tasks:
                if original_task in completed_tasks:
                    try:
                        result = await original_task
                        results[provider_name] = result
                    except Exception as e:
                        logger.error(f"Provider {provider_name} failed: {e}")
                        results[provider_name] = {"error": str(e), "success": False}
                else:
                    # Task didn't complete within timeout
                    results[provider_name] = {"error": "timeout", "success": False}
                    timeout_occurred = True

        except asyncio.TimeoutError:
            logger.warning(f"Concurrent execution timed out after {self.timeout_per_request}s")
            # Cancel remaining tasks
            for _, task in tasks:
                task.cancel()

        execution_time = time.time() - execution_start

        # Analyze results
        if analyze_consensus:
            consensus_confidence, consensus_found, best_provider = self._analyze_consensus(results)
        else:
            consensus_confidence = 0.0
            consensus_found = False
            best_provider = max(results.keys(), key=lambda k: results[k].get("confidence", 0.0))

        # Update provider metrics
        self._update_provider_metrics(available_providers, results, execution_time)

        # Create result
        result = ModelMatrixResult(
            query=query,
            provider_results=results,
            execution_time=execution_time,
            consensus_confidence=consensus_confidence,
            best_provider=best_provider,
            consensus_found=consensus_found,
            error_count=sum(1 for r in results.values() if not r.get("success", False)),
            metadata={
                "providers_attempted": len(available_providers),
                "providers_successful": sum(1 for r in results.values() if r.get("success", False)),
                "timeout_occurred": timeout_occurred,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

        # Update statistics
        self._update_matrix_stats(result)

        # Store in history
        self.execution_history.append(result)

        logger.info(f"Concurrent execution completed in {execution_time:.2f}s, "
                   f"success: {result.metadata['providers_successful']}/{len(available_providers)}, "
                   f"consensus: {consensus_found}")

        return result

    def _get_available_providers(self, providers: Optional[List[str]]) -> List[str]:
        """Get list of available providers for execution"""
        if providers:
            # Filter requested providers by availability
            available = []
            for provider in providers:
                if provider in self.provider_metrics:
                    metrics = self.provider_metrics[provider]
                    if (metrics.status == ProviderStatus.AVAILABLE and
                        metrics.concurrent_requests < metrics.max_concurrent):
                        available.append(provider)
            return available
        else:
            # Return all available providers
            available = []
            for provider, metrics in self.provider_metrics.items():
                if (metrics.status == ProviderStatus.AVAILABLE and
                    metrics.concurrent_requests < metrics.max_concurrent):
                    available.append(provider)
            return available

    async def _execute_provider_query(self, provider_name: str, query: str) -> Dict[str, Any]:
        """Execute query on a specific provider with semaphore control"""
        provider_metrics = self.provider_metrics[provider_name]
        semaphore = self.provider_semaphores[provider_name]

        # Update provider status
        provider_metrics.concurrent_requests += 1
        provider_metrics.last_request_time = datetime.utcnow()

        try:
            async with semaphore:
                # Create LLM request
                llm_request = LLMRequest(
                    prompt=query,
                    temperature=0.7,
                    task_type="matrix_query"
                )

                # Execute with monitoring
                with self.performance_monitor.timer("provider_query", {"provider": provider_name}):
                    response = self.llm_manager.generate_response(
                        prompt=query,
                        provider=provider_name
                    )

                # Process response
                if response:
                    return {
                        "success": True,
                        "response": response,
                        "response_length": len(response),
                        "provider": provider_name,
                        "timestamp": datetime.utcnow().isoformat(),
                        "confidence": 0.8  # Default confidence - could be enhanced
                    }
                else:
                    return {
                        "success": False,
                        "error": "Empty response",
                        "provider": provider_name,
                        "timestamp": datetime.utcnow().isoformat()
                    }

        except Exception as e:
            logger.error(f"Provider {provider_name} query failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "provider": provider_name,
                "timestamp": datetime.utcnow().isoformat()
            }

        finally:
            # Update provider status
            provider_metrics.concurrent_requests -= 1

    def _analyze_consensus(self, results: Dict[str, Dict[str, Any]]) -> Tuple[float, bool, str]:
        """Analyze consensus between provider results"""
        successful_results = {k: v for k, v in results.items() if v.get("success", False)}

        if len(successful_results) < 2:
            return 0.0, False, ""

        # Extract responses
        responses = [result["response"] for result in successful_results.values()]
        providers = list(successful_results.keys())

        # Simple consensus analysis based on response similarity
        # In a production system, this would use more sophisticated NLP
        consensus_score = self._calculate_response_similarity(responses)

        consensus_found = consensus_score >= self.consensus_threshold
        best_provider = providers[0]  # Could be enhanced to select best performer

        return consensus_score, consensus_found, best_provider

    def _calculate_response_similarity(self, responses: List[str]) -> float:
        """Calculate similarity between responses (simplified implementation)"""
        if len(responses) < 2:
            return 0.0

        # Simple word overlap similarity
        # In production, would use embeddings or semantic similarity
        def tokenize(text: str) -> set:
            return set(text.lower().split())

        base_tokens = tokenize(responses[0])
        similarities = []

        for response in responses[1:]:
            tokens = tokenize(response)
            if base_tokens and tokens:
                intersection = len(base_tokens.intersection(tokens))
                union = len(base_tokens.union(tokens))
                similarity = intersection / union if union > 0 else 0.0
                similarities.append(similarity)

        if similarities:
            return sum(similarities) / len(similarities)
        return 0.0

    def _update_provider_metrics(
        self,
        providers: List[str],
        results: Dict[str, Dict[str, Any]],
        execution_time: float
    ):
        """Update metrics for each provider"""
        for provider in providers:
            metrics = self.provider_metrics[provider]

            if provider in results:
                result = results[provider]
                if result.get("success", False):
                    metrics.requests_completed += 1
                    # Update average response time
                    if metrics.average_response_time == 0:
                        metrics.average_response_time = execution_time
                    else:
                        metrics.average_response_time = (
                            metrics.average_response_time * 0.8 + execution_time * 0.2
                        )
                else:
                    metrics.requests_failed += 1

                    # Check for rate limiting
                    if "rate limit" in result.get("error", "").lower():
                        metrics.status = ProviderStatus.RATE_LIMITED
                        metrics.rate_limit_reset = datetime.utcnow() + timedelta(minutes=1)

            # Update utilization
            with self._stats_lock:
                self.matrix_stats["provider_utilization"][provider] += execution_time

    def _update_matrix_stats(self, result: ModelMatrixResult):
        """Update overall matrix statistics"""
        with self._stats_lock:
            self.matrix_stats["total_executions"] += 1

            if result.metadata["providers_successful"] > 0:
                self.matrix_stats["successful_executions"] += 1

            if result.consensus_found:
                self.matrix_stats["consensus_found"] += 1

            # Update average execution time
            if self.matrix_stats["average_execution_time"] == 0:
                self.matrix_stats["average_execution_time"] = result.execution_time
            else:
                self.matrix_stats["average_execution_time"] = (
                    self.matrix_stats["average_execution_time"] * 0.9 + result.execution_time * 0.1
                )

    async def stress_test(
        self,
        query_count: int = 50,
        concurrent_queries: int = 5,
        query_template: str = "Analyze the claim: {topic}"
    ) -> Dict[str, Any]:
        """
        Perform stress testing of the concurrent model matrix

        Args:
            query_count: Total number of queries to execute
            concurrent_queries: Maximum concurrent queries at any time
            query_template: Template for generating test queries

        Returns:
            Stress test results with performance metrics
        """
        logger.info(f"Starting stress test: {query_count} queries, {concurrent_queries} concurrent")

        stress_start = time.time()

        # Generate test queries
        test_queries = [
            query_template.format(topic=f"test topic {i+1}")
            for i in range(query_count)
        ]

        # Execute with concurrency control
        semaphore = asyncio.Semaphore(concurrent_queries)

        async def execute_with_semaphore(query: str, index: int) -> Tuple[int, ModelMatrixResult]:
            async with semaphore:
                start_time = time.time()
                try:
                    result = await self.execute_concurrent_query(query)
                    execution_time = time.time() - start_time
                    return index, result
                except Exception as e:
                    logger.error(f"Query {index} failed: {e}")
                    # Return a failed result
                    return index, ModelMatrixResult(
                        query=query,
                        provider_results={},
                        execution_time=time.time() - start_time,
                        consensus_confidence=0.0,
                        best_provider="",
                        consensus_found=False,
                        error_count=1,
                        metadata={"error": str(e)}
                    )

        # Execute all queries
        tasks = [
            execute_with_semaphore(query, i)
            for i, query in enumerate(test_queries)
        ]

        completed_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        successful_results = []
        failed_results = []
        execution_times = []
        consensus_count = 0

        for index, result in completed_results:
            if isinstance(result, Exception):
                failed_results.append((index, str(result)))
            elif result.error_count == 0:
                successful_results.append((index, result))
                execution_times.append(result.execution_time)
                if result.consensus_found:
                    consensus_count += 1
            else:
                failed_results.append((index, f"Errors: {result.error_count}"))

        total_time = time.time() - stress_start

        # Compile stress test results
        stress_results = {
            "test_configuration": {
                "total_queries": query_count,
                "concurrent_queries": concurrent_queries,
                "providers_available": len(self.provider_metrics)
            },
            "execution_summary": {
                "total_time": total_time,
                "successful_queries": len(successful_results),
                "failed_queries": len(failed_results),
                "success_rate": len(successful_results) / query_count,
                "queries_per_second": query_count / total_time,
                "consensus_rate": consensus_count / max(1, len(successful_results))
            },
            "performance_metrics": {
                "average_execution_time": sum(execution_times) / max(1, len(execution_times)),
                "min_execution_time": min(execution_times) if execution_times else 0,
                "max_execution_time": max(execution_times) if execution_times else 0,
                "provider_utilization": dict(self.matrix_stats["provider_utilization"])
            },
            "current_matrix_stats": self.get_matrix_statistics(),
            "provider_health": self.get_provider_health()
        }

        logger.info(f"Stress test completed: {len(successful_results)}/{query_count} successful, "
                   f"{query_count / total_time:.1f} queries/sec")

        return stress_results

    def get_matrix_statistics(self) -> Dict[str, Any]:
        """Get comprehensive matrix statistics"""
        with self._stats_lock:
            return {
                **self.matrix_stats,
                "provider_metrics": {
                    name: {
                        "status": metrics.status.value,
                        "requests_completed": metrics.requests_completed,
                        "requests_failed": metrics.requests_failed,
                        "success_rate": metrics.requests_completed / max(1, metrics.requests_completed + metrics.requests_failed),
                        "average_response_time": metrics.average_response_time,
                        "concurrent_requests": metrics.concurrent_requests,
                        "max_concurrent": metrics.max_concurrent
                    }
                    for name, metrics in self.provider_metrics.items()
                }
            }

    def get_provider_health(self) -> Dict[str, Any]:
        """Get health status of all providers"""
        health = {}
        current_time = datetime.utcnow()

        for provider, metrics in self.provider_metrics.items():
            # Reset rate-limited providers if timeout has passed
            if (metrics.status == ProviderStatus.RATE_LIMITED and
                metrics.rate_limit_reset and
                current_time >= metrics.rate_limit_reset):
                metrics.status = ProviderStatus.AVAILABLE
                metrics.rate_limit_reset = None

            health[provider] = {
                "status": metrics.status.value,
                "healthy": metrics.status == ProviderStatus.AVAILABLE,
                "load_percentage": (metrics.concurrent_requests / metrics.max_concurrent) * 100,
                "last_request": metrics.last_request_time.isoformat() if metrics.last_request_time else None
            }

        return health

    def reset_statistics(self):
        """Reset all statistics"""
        with self._stats_lock:
            self.matrix_stats = {
                "total_executions": 0,
                "successful_executions": 0,
                "consensus_found": 0,
                "average_execution_time": 0.0,
                "provider_utilization": defaultdict(float)
            }

        for metrics in self.provider_metrics.values():
            metrics.requests_completed = 0
            metrics.requests_failed = 0
            metrics.average_response_time = 0.0

        logger.info("Matrix statistics reset")

    async def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        logger.info("ConcurrentModelMatrix cleanup completed")

# Convenience function for easy access
async def get_concurrent_model_matrix(**kwargs) -> ConcurrentModelMatrix:
    """Get a configured ConcurrentModelMatrix instance"""
    matrix = ConcurrentModelMatrix(**kwargs)
    return matrix