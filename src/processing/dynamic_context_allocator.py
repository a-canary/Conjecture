"""
Dynamic Context Allocator for Adaptive Resource Management

Implements intelligent, real-time context allocation that adapts to:
- Task complexity and requirements
- Model capabilities and constraints
- Performance feedback and optimization
- Memory and token budget constraints

Key Features:
- Real-time resource monitoring
- Adaptive budget allocation
- Performance-based tuning
- Context-aware scaling
"""

import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import statistics

from src.core.models import Claim
from src.utils.logging import get_logger

logger = get_logger(__name__)


class AllocationStrategy(Enum):
    """Context allocation strategies"""
    EQUAL_DISTRIBUTION = "equal"          # Equal distribution across components
    PRIORITY_BASED = "priority"          # Priority-based allocation
    PERFORMANCE_ADAPTIVE = "performance"  # Adaptive based on performance
    HYBRID = "hybrid"                    # Combination of strategies


class ComponentType(Enum):
    """Context component types requiring allocation"""
    CLAIM_PROCESSING = "claim_processing"
    EVIDENCE_SYNTHESIS = "evidence_synthesis"
    REASONING_ENGINE = "reasoning_engine"
    TASK_INSTRUCTIONS = "task_instructions"
    EXAMPLES = "examples"
    OUTPUT_FORMAT = "output_format"
    WORKING_MEMORY = "working_memory"


@dataclass
class ResourceMonitor:
    """Monitor for resource usage and performance"""
    total_tokens: int = 0
    used_tokens: int = 0
    processing_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    success_rate: float = 1.0
    quality_score: float = 1.0

    def utilization_rate(self) -> float:
        """Calculate token utilization rate"""
        return self.used_tokens / max(self.total_tokens, 1)

    def efficiency_score(self) -> float:
        """Calculate overall efficiency score"""
        return (self.success_rate * 0.4 +
                self.quality_score * 0.4 +
                (1.0 - self.utilization_rate()) * 0.2)


@dataclass
class AllocationMetrics:
    """Metrics for allocation effectiveness"""
    allocated_tokens: int
    actual_usage: int
    utilization_efficiency: float
    performance_impact: float
    quality_preservation: float
    allocation_timestamp: datetime


@dataclass
class ComponentRequirement:
    """Requirements for a context component"""
    component_type: ComponentType
    min_tokens: int
    preferred_tokens: int
    max_tokens: int
    priority: float
    importance_weight: float
    performance_critical: bool = False
    flexible: bool = True


class DynamicContextAllocator:
    """
    Dynamically allocates context resources based on real-time requirements
    and performance feedback for optimal tiny LLM performance.
    """

    def __init__(self, total_token_budget: int = 2048):
        self.total_token_budget = total_token_budget
        self.current_allocation = {}
        self.performance_history = []
        self.resource_monitor = ResourceMonitor(total_tokens=total_token_budget)
        self.component_requirements = self._initialize_component_requirements()
        self.allocation_strategy = AllocationStrategy.PERFORMANCE_ADAPTIVE
        self.adaptation_window = timedelta(minutes=30)  # Learning window

    def _initialize_component_requirements(self) -> Dict[ComponentType, ComponentRequirement]:
        """Initialize baseline requirements for each component type"""
        return {
            ComponentType.CLAIM_PROCESSING: ComponentRequirement(
                component_type=ComponentType.CLAIM_PROCESSING,
                min_tokens=200,
                preferred_tokens=400,
                max_tokens=600,
                priority=0.9,
                importance_weight=0.25,
                performance_critical=True,
                flexible=True
            ),
            ComponentType.EVIDENCE_SYNTHESIS: ComponentRequirement(
                component_type=ComponentType.EVIDENCE_SYNTHESIS,
                min_tokens=150,
                preferred_tokens=300,
                max_tokens=500,
                priority=0.8,
                importance_weight=0.2,
                performance_critical=False,
                flexible=True
            ),
            ComponentType.REASONING_ENGINE: ComponentRequirement(
                component_type=ComponentType.REASONING_ENGINE,
                min_tokens=300,
                preferred_tokens=500,
                max_tokens=800,
                priority=1.0,
                importance_weight=0.3,
                performance_critical=True,
                flexible=False
            ),
            ComponentType.TASK_INSTRUCTIONS: ComponentRequirement(
                component_type=ComponentType.TASK_INSTRUCTIONS,
                min_tokens=100,
                preferred_tokens=150,
                max_tokens=200,
                priority=0.7,
                importance_weight=0.1,
                performance_critical=False,
                flexible=False
            ),
            ComponentType.EXAMPLES: ComponentRequirement(
                component_type=ComponentType.EXAMPLES,
                min_tokens=50,
                preferred_tokens=100,
                max_tokens=200,
                priority=0.5,
                importance_weight=0.05,
                performance_critical=False,
                flexible=True
            ),
            ComponentType.OUTPUT_FORMAT: ComponentRequirement(
                component_type=ComponentType.OUTPUT_FORMAT,
                min_tokens=50,
                preferred_tokens=100,
                max_tokens=150,
                priority=0.6,
                importance_weight=0.05,
                performance_critical=False,
                flexible=True
            ),
            ComponentType.WORKING_MEMORY: ComponentRequirement(
                component_type=ComponentType.WORKING_MEMORY,
                min_tokens=100,
                preferred_tokens=200,
                max_tokens=400,
                priority=0.7,
                importance_weight=0.05,
                performance_critical=False,
                flexible=True
            )
        }

    def allocate_context_resources(self,
                                 task_complexity: float,
                                 performance_requirements: Dict[str, float],
                                 active_components: List[ComponentType]) -> Dict[ComponentType, int]:
        """
        Allocate context resources based on current requirements and strategy
        """
        logger.info(f"Allocating context resources for complexity {task_complexity}")

        # Filter active components
        active_requirements = {
            comp_type: req for comp_type, req in self.component_requirements.items()
            if comp_type in active_components
        }

        # Apply allocation strategy
        if self.allocation_strategy == AllocationStrategy.EQUAL_DISTRIBUTION:
            allocation = self._allocate_equal(active_requirements)
        elif self.allocation_strategy == AllocationStrategy.PRIORITY_BASED:
            allocation = self._allocate_priority_based(active_requirements)
        elif self.allocation_strategy == AllocationStrategy.PERFORMANCE_ADAPTIVE:
            allocation = self._allocate_performance_adaptive(
                active_requirements, task_complexity, performance_requirements
            )
        else:  # HYBRID
            allocation = self._allocate_hybrid(
                active_requirements, task_complexity, performance_requirements
            )

        # Validate and adjust allocation
        allocation = self._validate_and_adjust_allocation(allocation)

        # Update current allocation
        self.current_allocation = allocation

        return allocation

    def _allocate_equal(self, requirements: Dict[ComponentType, ComponentRequirement]) -> Dict[ComponentType, int]:
        """Equal distribution allocation strategy"""
        num_components = len(requirements)
        if num_components == 0:
            return {}

        tokens_per_component = self.total_token_budget // num_components

        allocation = {}
        for comp_type, req in requirements.items():
            allocated = min(max(tokens_per_component, req.min_tokens), req.max_tokens)
            allocation[comp_type] = allocated

        return allocation

    def _allocate_priority_based(self, requirements: Dict[ComponentType, ComponentRequirement]) -> Dict[ComponentType, int]:
        """Priority-based allocation strategy"""
        # Sort by priority and importance
        sorted_components = sorted(
            requirements.items(),
            key=lambda x: (x[1].priority, x[1].importance_weight),
            reverse=True
        )

        allocation = {}
        remaining_budget = self.total_token_budget

        for comp_type, req in sorted_components:
            if remaining_budget <= 0:
                break

            # Allocate based on priority
            if req.performance_critical:
                allocated = min(req.preferred_tokens, remaining_budget, req.max_tokens)
            else:
                allocated = min(
                    max(req.min_tokens, int(remaining_budget * req.importance_weight)),
                    req.preferred_tokens
                )

            allocation[comp_type] = allocated
            remaining_budget -= allocated

        return allocation

    def _allocate_performance_adaptive(self,
                                     requirements: Dict[ComponentType, ComponentRequirement],
                                     task_complexity: float,
                                     performance_requirements: Dict[str, float]) -> Dict[ComponentType, int]:
        """Performance-adaptive allocation based on historical performance"""
        # Analyze recent performance
        recent_performance = self._get_recent_performance()

        # Adjust requirements based on performance feedback
        adjusted_requirements = self._adjust_requirements_based_on_performance(
            requirements, recent_performance
        )

        # Factor in task complexity
        complexity_multiplier = 0.5 + (task_complexity * 0.5)  # 0.5 to 1.0

        allocation = {}
        total_importance = sum(req.importance_weight for req in adjusted_requirements.values())

        for comp_type, req in adjusted_requirements.items():
            # Base allocation proportional to importance
            base_allocation = int(
                (req.importance_weight / total_importance) * self.total_token_budget
            )

            # Apply complexity scaling
            scaled_allocation = int(base_allocation * complexity_multiplier)

            # Apply performance adjustments
            performance_adjustment = self._calculate_performance_adjustment(
                comp_type, recent_performance
            )

            final_allocation = int(scaled_allocation * performance_adjustment)

            # Ensure within bounds
            allocated = max(req.min_tokens, min(final_allocation, req.max_tokens))
            allocation[comp_type] = allocated

        return self._balance_allocation(allocation)

    def _allocate_hybrid(self,
                        requirements: Dict[ComponentType, ComponentRequirement],
                        task_complexity: float,
                        performance_requirements: Dict[str, float]) -> Dict[ComponentType, int]:
        """Hybrid allocation combining multiple strategies"""
        # Get allocations from different strategies
        equal_allocation = self._allocate_equal(requirements)
        priority_allocation = self._allocate_priority_based(requirements)
        performance_allocation = self._allocate_performance_adaptive(
            requirements, task_complexity, performance_requirements
        )

        # Weight the strategies based on conditions
        weights = {
            'equal': 0.2,
            'priority': 0.4,
            'performance': 0.4
        }

        # Adjust weights based on task complexity
        if task_complexity > 0.8:
            weights['performance'] = 0.6
            weights['priority'] = 0.3
            weights['equal'] = 0.1
        elif task_complexity < 0.3:
            weights['equal'] = 0.4
            weights['priority'] = 0.4
            weights['performance'] = 0.2

        allocation = {}
        for comp_type in requirements.keys():
            weighted_allocation = (
                equal_allocation.get(comp_type, 0) * weights['equal'] +
                priority_allocation.get(comp_type, 0) * weights['priority'] +
                performance_allocation.get(comp_type, 0) * weights['performance']
            )

            req = requirements[comp_type]
            allocated = max(req.min_tokens, min(int(weighted_allocation), req.max_tokens))
            allocation[comp_type] = allocated

        return self._balance_allocation(allocation)

    def _get_recent_performance(self) -> Dict[ComponentType, List[float]]:
        """Get recent performance metrics for each component"""
        recent_time = datetime.now() - self.adaptation_window
        recent_data = [
            metric for metric in self.performance_history
            if metric.allocation_timestamp > recent_time
        ]

        performance_by_component = {}
        for metric in recent_data:
            # Group by component type (this would need component tracking in actual implementation)
            pass  # Placeholder - would track component-specific performance

        return performance_by_component

    def _adjust_requirements_based_on_performance(self,
                                                 requirements: Dict[ComponentType, ComponentRequirement],
                                                 recent_performance: Dict[ComponentType, List[float]]) -> Dict[ComponentType, ComponentRequirement]:
        """Adjust component requirements based on performance feedback"""
        adjusted_requirements = {}

        for comp_type, req in requirements.items():
            adjusted_req = ComponentRequirement(
                component_type=req.component_type,
                min_tokens=req.min_tokens,
                preferred_tokens=req.preferred_tokens,
                max_tokens=req.max_tokens,
                priority=req.priority,
                importance_weight=req.importance_weight,
                performance_critical=req.performance_critical,
                flexible=req.flexible
            )

            # Adjust based on performance
            if comp_type in recent_performance:
                perf_scores = recent_performance[comp_type]
                avg_performance = statistics.mean(perf_scores)

                if avg_performance < 0.7 and adjusted_req.flexible:
                    # Poor performance - increase allocation
                    adjusted_req.preferred_tokens = min(
                        int(adjusted_req.preferred_tokens * 1.2),
                        adjusted_req.max_tokens
                    )
                    adjusted_req.importance_weight = min(
                        adjusted_req.importance_weight * 1.1,
                        1.0
                    )
                elif avg_performance > 0.9 and adjusted_req.flexible:
                    # Excellent performance - can potentially reduce
                    adjusted_req.preferred_tokens = max(
                        int(adjusted_req.preferred_tokens * 0.9),
                        adjusted_req.min_tokens
                    )

            adjusted_requirements[comp_type] = adjusted_req

        return adjusted_requirements

    def _calculate_performance_adjustment(self,
                                        component_type: ComponentType,
                                        recent_performance: Dict[ComponentType, List[float]]) -> float:
        """Calculate performance-based adjustment factor"""
        if component_type not in recent_performance:
            return 1.0

        perf_scores = recent_performance[component_type]
        if not perf_scores:
            return 1.0

        avg_performance = statistics.mean(perf_scores)

        # Map performance to adjustment factor
        if avg_performance < 0.5:
            return 1.3  # Increase allocation significantly
        elif avg_performance < 0.7:
            return 1.15  # Increase allocation moderately
        elif avg_performance < 0.85:
            return 1.0  # No change
        elif avg_performance < 0.95:
            return 0.9  # Slight reduction possible
        else:
            return 0.8  # Can reduce allocation

    def _balance_allocation(self, allocation: Dict[ComponentType, int]) -> Dict[ComponentType, int]:
        """Balance allocation to fit within token budget"""
        total_allocated = sum(allocation.values())

        if total_allocated <= self.total_token_budget:
            return allocation

        # Need to reduce allocation
        excess = total_allocated - self.total_token_budget

        # Sort components by flexibility (reduce flexible components first)
        flexible_components = [
            (comp_type, tokens) for comp_type, tokens in allocation.items()
            if self.component_requirements[comp_type].flexible
        ]

        # Sort by importance (reduce less important first)
        flexible_components.sort(
            key=lambda x: self.component_requirements[x[0]].importance_weight
        )

        # Reduce allocations
        for comp_type, current_tokens in flexible_components:
            if excess <= 0:
                break

            req = self.component_requirements[comp_type]
            max_reduction = current_tokens - req.min_tokens

            if max_reduction > 0:
                reduction = min(max_reduction, excess)
                allocation[comp_type] -= reduction
                excess -= reduction

        return allocation

    def _validate_and_adjust_allocation(self, allocation: Dict[ComponentType, int]) -> Dict[ComponentType, int]:
        """Validate allocation meets minimum requirements and adjust if needed"""
        # Check minimum requirements
        for comp_type, allocated in allocation.items():
            req = self.component_requirements.get(comp_type)
            if req and allocated < req.min_tokens:
                allocation[comp_type] = req.min_tokens

        # Check total budget
        total_allocated = sum(allocation.values())
        if total_allocated > self.total_token_budget:
            # Emergency rebalancing
            return self._emergency_rebalance(allocation)

        return allocation

    def _emergency_rebalance(self, allocation: Dict[ComponentType, int]) -> Dict[ComponentType, int]:
        """Emergency rebalancing when over budget"""
        total_allocated = sum(allocation.values())
        excess = total_allocated - self.total_token_budget

        if excess <= 0:
            return allocation

        # Calculate proportional reduction
        reduction_ratio = (self.total_token_budget - excess) / total_allocated

        for comp_type in allocation:
            allocation[comp_type] = int(allocation[comp_type] * reduction_ratio)

        return allocation

    def record_allocation_metrics(self,
                                component_type: ComponentType,
                                allocated_tokens: int,
                                actual_tokens: int,
                                performance_score: float,
                                quality_score: float) -> None:
        """Record allocation performance for adaptive learning"""
        metrics = AllocationMetrics(
            allocated_tokens=allocated_tokens,
            actual_usage=actual_tokens,
            utilization_efficiency=actual_tokens / max(allocated_tokens, 1),
            performance_impact=performance_score,
            quality_preservation=quality_score,
            allocation_timestamp=datetime.now()
        )

        self.performance_history.append(metrics)

        # Update resource monitor
        self.resource_monitor.used_tokens += actual_tokens
        self.resource_monitor.success_rate = performance_score
        self.resource_monitor.quality_score = quality_score

        # Trim old performance data
        cutoff_time = datetime.now() - self.adaptation_window
        self.performance_history = [
            m for m in self.performance_history
            if m.allocation_timestamp > cutoff_time
        ]

    def get_allocation_recommendations(self) -> List[str]:
        """Get recommendations for allocation optimization"""
        recommendations = []

        if not self.performance_history:
            return ["No performance data available for recommendations"]

        # Analyze utilization efficiency
        utilizations = [m.utilization_efficiency for m in self.performance_history]
        avg_utilization = statistics.mean(utilizations)

        if avg_utilization < 0.6:
            recommendations.append(
                "Low utilization efficiency - consider reducing allocated tokens"
            )
        elif avg_utilization > 0.95:
            recommendations.append(
                "High utilization - consider increasing token budget"
            )

        # Analyze performance impact
        performances = [m.performance_impact for m in self.performance_history]
        avg_performance = statistics.mean(performances)

        if avg_performance < 0.7:
            recommendations.append(
                "Low performance impact - review allocation strategy"
            )

        # Strategy recommendations
        if avg_performance > 0.85 and avg_utilization > 0.8:
            recommendations.append(
                "Current allocation strategy working well - consider current settings"
            )

        return recommendations

    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status and metrics"""
        return {
            "total_budget": self.total_token_budget,
            "current_allocation": self.current_allocation,
            "utilization_rate": self.resource_monitor.utilization_rate(),
            "efficiency_score": self.resource_monitor.efficiency_score(),
            "allocation_strategy": self.allocation_strategy.value,
            "performance_history_size": len(self.performance_history),
            "recommendations": self.get_allocation_recommendations()
        }


# Factory function for easy instantiation
def create_dynamic_allocator(token_budget: int = 2048) -> DynamicContextAllocator:
    """Create dynamic context allocator with specified token budget"""
    return DynamicContextAllocator(total_token_budget=token_budget)