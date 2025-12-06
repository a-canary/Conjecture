"""
Context Optimization System - Complete Integration

Main orchestrator for advanced context window optimization in tiny LLMs.
Integrates multiple optimization strategies:

1. Information-Theoretic Context Optimization
2. Dynamic Resource Allocation
3. Adaptive Compression
4. Performance-Based Learning
5. Real-time Monitoring and Tuning

This system provides a unified interface for maximizing tiny LLM performance
through intelligent context management.
"""

import json
import time
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import asyncio
from pathlib import Path

from src.core.models import Claim, ClaimState, ClaimType
from src.processing.advanced_context_optimizer import (
    TinyLLMContextOptimizer,
    TaskType,
    ContextMetrics,
    ContextPerformanceEvaluator,
    create_tiny_llm_optimizer
)
from src.processing.dynamic_context_allocator import (
    DynamicContextAllocator,
    ComponentType,
    AllocationStrategy,
    ResourceMonitor,
    create_dynamic_allocator
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizationRequest:
    """Request for context optimization"""
    context_text: str
    task_type: TaskType
    task_keywords: List[str]
    performance_requirements: Dict[str, float]
    active_components: List[ComponentType]
    custom_parameters: Dict[str, Any] = None


@dataclass
class OptimizationResult:
    """Result of context optimization"""
    optimized_context: str
    original_tokens: int
    optimized_tokens: int
    compression_ratio: float
    allocation: Dict[ComponentType, int]
    metrics: ContextMetrics
    performance_prediction: Dict[str, float]
    processing_time_ms: float
    recommendations: List[str]


@dataclass
class SystemConfiguration:
    """Configuration for the optimization system"""
    model_name: str = "ibm/granite-4-h-tiny"
    default_token_budget: int = 2048
    allocation_strategy: AllocationStrategy = AllocationStrategy.PERFORMANCE_ADAPTIVE
    enable_learning: bool = True
    learning_window_hours: int = 24
    performance_threshold: float = 0.7
    cache_optimizations: bool = True
    cache_size_limit: int = 1000
    monitoring_enabled: bool = True


class ContextOptimizationSystem:
    """
    Complete context optimization system for tiny LLM enhancement.
    Integrates multiple optimization strategies with adaptive learning.
    """

    def __init__(self, config: SystemConfiguration = None):
        self.config = config or SystemConfiguration()
        self.context_optimizer = create_tiny_llm_optimizer(self.config.model_name)
        self.dynamic_allocator = create_dynamic_allocator(self.config.default_token_budget)
        self.performance_evaluator = ContextPerformanceEvaluator()

        # Performance tracking
        self.optimization_history = []
        self.performance_cache = {}
        self.optimization_cache = {}

        # System state
        self.system_start_time = datetime.now()
        self.total_optimizations = 0
        self.successful_optimizations = 0

        logger.info(f"Context Optimization System initialized for {self.config.model_name}")

    async def optimize_context(self, request: OptimizationRequest) -> OptimizationResult:
        """
        Main optimization function that processes context optimization request
        """
        start_time = time.time()

        logger.info(f"Starting context optimization for {request.task_type.value}")

        try:
            # Convert keywords to set
            task_keywords = set(request.task_keywords) if request.task_keywords else set()

            # Check cache first
            cache_key = self._generate_cache_key(request)
            if self.config.cache_optimizations and cache_key in self.optimization_cache:
                cached_result = self.optimization_cache[cache_key]
                logger.info("Returning cached optimization result")
                return cached_result

            # Stage 1: Analyze and Optimize Context Content
            optimized_context, content_metrics = await self._optimize_content(
                request.context_text, request.task_type, task_keywords
            )

            # Stage 2: Dynamic Resource Allocation
            resource_allocation = await self._allocate_resources(
                request, content_metrics
            )

            # Stage 3: Apply Allocation to Content
            final_context, final_metrics = await self._apply_allocation(
                optimized_context, resource_allocation, request.task_type
            )

            # Stage 4: Performance Prediction
            performance_prediction = await self._predict_performance(
                final_metrics, resource_allocation, request.performance_requirements
            )

            # Calculate final metrics
            original_tokens = len(request.context_text.split())
            optimized_tokens = len(final_context.split())
            compression_ratio = optimized_tokens / max(original_tokens, 1)

            processing_time_ms = (time.time() - start_time) * 1000

            # Generate recommendations
            recommendations = await self._generate_recommendations(
                final_metrics, performance_prediction, resource_allocation
            )

            result = OptimizationResult(
                optimized_context=final_context,
                original_tokens=original_tokens,
                optimized_tokens=optimized_tokens,
                compression_ratio=compression_ratio,
                allocation=resource_allocation,
                metrics=final_metrics,
                performance_prediction=performance_prediction,
                processing_time_ms=processing_time_ms,
                recommendations=recommendations
            )

            # Cache result
            if self.config.cache_optimizations:
                self._cache_result(cache_key, result)

            # Update statistics
            self._update_statistics(result)

            # Record allocation metrics for learning
            self._record_allocation_metrics(resource_allocation, final_metrics)

            self.total_optimizations += 1
            self.successful_optimizations += 1

            logger.info(f"Context optimization completed in {processing_time_ms:.2f}ms")
            return result

        except Exception as e:
            logger.error(f"Context optimization failed: {str(e)}")
            self.total_optimizations += 1
            raise

    async def _optimize_content(self, context_text: str, task_type: TaskType,
                              task_keywords: set) -> Tuple[str, ContextMetrics]:
        """Optimize content using information-theoretic approaches"""
        logger.debug("Starting content optimization")

        optimized_context, metrics = self.context_optimizer.optimize_context(
            context_text=context_text,
            task_type=task_type,
            task_keywords=task_keywords
        )

        logger.debug(f"Content optimization: {metrics.token_efficiency:.2%} token efficiency")
        return optimized_context, metrics

    async def _allocate_resources(self, request: OptimizationRequest,
                                content_metrics: ContextMetrics) -> Dict[ComponentType, int]:
        """Allocate resources dynamically based on requirements"""
        logger.debug("Starting resource allocation")

        # Calculate task complexity
        task_complexity = await self._calculate_task_complexity(
            request.context_text, request.task_type, content_metrics
        )

        # Perform allocation
        allocation = self.dynamic_allocator.allocate_context_resources(
            task_complexity=task_complexity,
            performance_requirements=request.performance_requirements,
            active_components=request.active_components
        )

        logger.debug(f"Resource allocation: {sum(allocation.values())} tokens allocated")
        return allocation

    async def _apply_allocation(self, optimized_context: str,
                             allocation: Dict[ComponentType],
                             task_type: TaskType) -> Tuple[str, ContextMetrics]:
        """Apply resource allocation to optimized context"""
        logger.debug("Applying resource allocation")

        # This would integrate the allocation into the actual context structure
        # For now, return the optimized context with allocation-aware metrics
        tokens_used = sum(allocation.values())
        context_tokens = len(optimized_context.split())

        # Adjust if context exceeds allocation
        if context_tokens > tokens_used:
            # Apply additional compression to fit allocation
            compression_ratio = tokens_used / context_tokens
            final_context = await self._compress_to_fit(
                optimized_context, compression_ratio
            )
        else:
            final_context = optimized_context

        # Recalculate metrics with allocation
        final_metrics = self.context_optimizer.information_optimizer.calculate_semantic_density(
            final_context
        )

        return final_context, ContextMetrics(
            semantic_density=final_metrics,
            token_efficiency=tokens_used / max(len(optimized_context.split()), 1)
        )

    async def _compress_to_fit(self, context: str, target_ratio: float) -> str:
        """Compress context to fit target ratio"""
        if target_ratio >= 1.0:
            return context

        # Simple truncation-based compression (could be enhanced)
        sentences = context.split('. ')
        target_sentences = max(1, int(len(sentences) * target_ratio))

        return '. '.join(sentences[:target_sentences])

    async def _predict_performance(self, metrics: ContextMetrics,
                                 allocation: Dict[ComponentType],
                                 requirements: Dict[str, float]) -> Dict[str, float]:
        """Predict performance based on optimization metrics"""
        predictions = {}

        # Base performance from quality metrics
        base_performance = (
            metrics.semantic_density * 0.3 +
            metrics.compression_quality * 0.4 +
            metrics.relevance_score * 0.3
        )

        # Adjust based on allocation efficiency
        total_allocated = sum(allocation.values())
        allocation_efficiency = min(total_allocated / self.config.default_token_budget, 1.0)

        # Component-specific predictions
        for req_name, req_value in requirements.items():
            component_performance = base_performance * allocation_efficiency

            # Adjust based on component type
            if 'reasoning' in req_name.lower():
                component_performance *= 1.1  # Reasoning gets boost
            elif 'synthesis' in req_name.lower():
                component_performance *= 1.05  # Synthesis gets small boost

            predictions[req_name] = min(component_performance, 1.0)

        return predictions

    async def _generate_recommendations(self, metrics: ContextMetrics,
                                      performance: Dict[str, float],
                                      allocation: Dict[ComponentType]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        # Content-based recommendations
        if metrics.semantic_density < 0.6:
            recommendations.append(
                "Consider improving semantic density through better information selection"
            )

        if metrics.compression_quality < 0.8:
            recommendations.append(
                "Current compression may lose important information"
            )

        # Performance-based recommendations
        avg_performance = sum(performance.values()) / max(len(performance), 1)
        if avg_performance < self.config.performance_threshold:
            recommendations.append(
                "Performance below threshold - consider increasing token budget"
            )

        # Allocation-based recommendations
        total_allocated = sum(allocation.values())
        if total_allocated < self.config.default_token_budget * 0.7:
            recommendations.append(
                "Underutilizing token budget - could allocate more resources"
            )

        return recommendations

    async def _calculate_task_complexity(self, context_text: str, task_type: TaskType,
                                       metrics: ContextMetrics) -> float:
        """Calculate task complexity score"""
        # Base complexity from task type
        task_complexities = {
            TaskType.REASONING: 0.8,
            TaskType.SYNTHESIS: 0.7,
            TaskType.ANALYSIS: 0.6,
            TaskType.DECISION: 0.5,
            TaskType.CREATION: 0.4,
            TaskType.COMPARISON: 0.3
        }

        base_complexity = task_complexities.get(task_type, 0.5)

        # Adjust based on content
        content_complexity = metrics.complexity_score

        # Adjust based on length
        length_factor = min(len(context_text) / 5000, 1.0)

        # Combined complexity
        final_complexity = (
            base_complexity * 0.4 +
            content_complexity * 0.4 +
            length_factor * 0.2
        )

        return min(final_complexity, 1.0)

    def _generate_cache_key(self, request: OptimizationRequest) -> str:
        """Generate cache key for optimization request"""
        key_data = {
            "task_type": request.task_type.value,
            "keywords": sorted(request.task_keywords),
            "components": sorted([c.value for c in request.active_components]),
            "text_hash": hash(request.context_text[:1000])  # First 1000 chars for hashing
        }
        return json.dumps(key_data, sort_keys=True)

    def _cache_result(self, cache_key: str, result: OptimizationResult) -> None:
        """Cache optimization result"""
        if len(self.optimization_cache) >= self.config.cache_size_limit:
            # Remove oldest entry
            oldest_key = next(iter(self.optimization_cache))
            del self.optimization_cache[oldest_key]

        self.optimization_cache[cache_key] = result

    def _update_statistics(self, result: OptimizationResult) -> None:
        """Update system statistics"""
        self.optimization_history.append({
            "timestamp": datetime.now(),
            "compression_ratio": result.compression_ratio,
            "processing_time_ms": result.processing_time_ms,
            "performance_prediction": result.performance_prediction,
            "metrics": asdict(result.metrics)
        })

        # Keep history manageable
        if len(self.optimization_history) > 1000:
            self.optimization_history = self.optimization_history[-500:]

    def _record_allocation_metrics(self, allocation: Dict[ComponentType],
                                 metrics: ContextMetrics) -> None:
        """Record allocation metrics for adaptive learning"""
        if not self.config.enable_learning:
            return

        # Record metrics for each component
        for comp_type, allocated_tokens in allocation.items():
            self.dynamic_allocator.record_allocation_metrics(
                component_type=comp_type,
                allocated_tokens=allocated_tokens,
                actual_tokens=int(allocated_tokens * metrics.token_efficiency),
                performance_score=metrics.compression_quality,
                quality_score=metrics.relevance_score
            )

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        success_rate = (self.successful_optimizations / max(self.total_optimizations, 1))

        # Calculate average metrics
        if self.optimization_history:
            avg_compression = sum(h["compression_ratio"] for h in self.optimization_history) / len(self.optimization_history)
            avg_processing_time = sum(h["processing_time_ms"] for h in self.optimization_history) / len(self.optimization_history)
        else:
            avg_compression = 0.0
            avg_processing_time = 0.0

        return {
            "system_info": {
                "model_name": self.config.model_name,
                "uptime_hours": (datetime.now() - self.system_start_time).total_seconds() / 3600,
                "total_optimizations": self.total_optimizations,
                "success_rate": success_rate,
                "cache_size": len(self.optimization_cache)
            },
            "performance_metrics": {
                "average_compression_ratio": avg_compression,
                "average_processing_time_ms": avg_processing_time,
                "default_token_budget": self.config.default_token_budget,
                "allocation_strategy": self.config.allocation_strategy.value
            },
            "resource_status": self.dynamic_allocator.get_resource_status(),
            "learning_enabled": self.config.enable_learning,
            "recommendations": self._get_system_recommendations()
        }

    def _get_system_recommendations(self) -> List[str]:
        """Get system-level recommendations"""
        recommendations = []

        if self.total_optimizations == 0:
            return ["No optimizations performed yet"]

        success_rate = self.successful_optimizations / max(self.total_optimizations, 1)

        if success_rate < 0.8:
            recommendations.append(
                "Low success rate - review system configuration and error handling"
            )

        if len(self.optimization_cache) > self.config.cache_size_limit * 0.9:
            recommendations.append(
                "Cache approaching limit - consider increasing cache size limit"
            )

        # Get allocator recommendations
        allocator_recs = self.dynamic_allocator.get_allocation_recommendations()
        recommendations.extend(allocator_recs)

        return recommendations

    def update_configuration(self, config_updates: Dict[str, Any]) -> None:
        """Update system configuration"""
        for key, value in config_updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated configuration: {key} = {value}")

        # Reinitialize components if needed
        if "default_token_budget" in config_updates:
            self.dynamic_allocator = create_dynamic_allocator(self.config.default_token_budget)

    def export_performance_data(self, file_path: Union[str, Path]) -> None:
        """Export performance data for analysis"""
        data = {
            "system_config": asdict(self.config),
            "optimization_history": self.optimization_history,
            "system_status": self.get_system_status(),
            "export_timestamp": datetime.now().isoformat()
        }

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Performance data exported to {file_path}")


# Factory function for easy instantiation
def create_context_optimization_system(config: SystemConfiguration = None) -> ContextOptimizationSystem:
    """Create complete context optimization system"""
    return ContextOptimizationSystem(config)


# Utility function for quick optimization
async def optimize_context_for_tiny_llm(
    context_text: str,
    task_type: str = "reasoning",
    task_keywords: List[str] = None,
    token_budget: int = 2048
) -> OptimizationResult:
    """
    Quick utility function for optimizing context for tiny LLMs
    """
    system = create_context_optimization_system()

    request = OptimizationRequest(
        context_text=context_text,
        task_type=TaskType(task_type),
        task_keywords=task_keywords or [],
        performance_requirements={},
        active_components=list(ComponentType)
    )

    return await system.optimize_context(request)