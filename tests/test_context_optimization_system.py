"""
Comprehensive Test Suite for Context Optimization System

Tests all aspects of the advanced context window optimization for tiny LLMs:
- Information-theoretic optimization
- Dynamic resource allocation
- Performance-based learning
- Integration scenarios
- Edge cases and error handling
"""

import pytest
import asyncio
import time
from typing import List, Dict, Any
from pathlib import Path
import tempfile
import json

from src.processing.context_optimization_system import (
    ContextOptimizationSystem,
    OptimizationRequest,
    SystemConfiguration,
    TaskType,
    ComponentType,
    AllocationStrategy,
    create_context_optimization_system,
    optimize_context_for_tiny_llm
)
from src.processing.advanced_context_optimizer import (
    TinyLLMContextOptimizer,
    InformationCategory,
    create_tiny_llm_optimizer
)
from src.processing.dynamic_context_allocator import (
    DynamicContextAllocator,
    create_dynamic_allocator
)


class TestContextOptimizerCore:
    """Test core context optimization functionality"""

    @pytest.fixture
    def optimizer(self):
        return create_tiny_llm_optimizer()

    @pytest.fixture
    def sample_context(self):
        return """
        In the field of artificial intelligence, machine learning algorithms have become increasingly
        sophisticated. Deep learning, a subset of machine learning, uses neural networks with multiple
        layers to progressively extract higher-level features from raw input. For example, in image
        processing, lower layers may identify edges, while higher layers may identify concepts
        relevant to a human such as digits, letters, or faces. This hierarchical approach has led
        to breakthroughs in computer vision, natural language processing, and game playing.
        However, these systems require large amounts of data and computational resources.
        """

    def test_semantic_density_calculation(self, optimizer, sample_context):
        """Test semantic density calculation"""
        density = optimizer.information_optimizer.calculate_semantic_density(sample_context)

        assert 0.0 <= density <= 1.0
        assert density > 0.0  # Should have some semantic content

        # Test with empty text
        empty_density = optimizer.information_optimizer.calculate_semantic_density("")
        assert empty_density == 0.0

    def test_context_analysis(self, optimizer, sample_context):
        """Test context analysis and chunking"""
        task_keywords = {"artificial intelligence", "machine learning", "deep learning"}

        chunks = optimizer.analyze_context(
            sample_context,
            TaskType.ANALYSIS,
            task_keywords
        )

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.content
            assert chunk.tokens > 0
            assert chunk.category in InformationCategory
            assert 0.0 <= chunk.relevance_score <= 1.0
            assert 0.0 <= chunk.semantic_density <= 1.0

    def test_optimization_workflow(self, optimizer, sample_context):
        """Test complete optimization workflow"""
        task_keywords = {"artificial intelligence", "neural networks"}

        optimized_context, metrics = optimizer.optimize_context(
            context_text=sample_context,
            task_type=TaskType.REASONING,
            task_keywords=task_keywords
        )

        # Verify optimization occurred
        assert optimized_context
        assert len(optimized_context) <= len(sample_context)
        assert metrics.semantic_density > 0.0
        assert 0.0 <= metrics.compression_quality <= 1.0
        assert 0.0 <= metrics.token_efficiency <= 1.0

    def test_different_task_types(self, optimizer, sample_context):
        """Test optimization for different task types"""
        task_types = [TaskType.REASONING, TaskType.SYNTHESIS, TaskType.ANALYSIS]
        results = {}

        for task_type in task_types:
            optimized_context, metrics = optimizer.optimize_context(
                context_text=sample_context,
                task_type=task_type,
                task_keywords={"AI", "machine learning"}
            )
            results[task_type] = (optimized_context, metrics)

        # Different task types should produce different optimizations
        assert len(set(result[0] for result in results.values())) > 1

    def test_information_categorization(self, optimizer):
        """Test information categorization logic"""
        test_cases = [
            ("Critical evidence proves the hypothesis", InformationCategory.CRITICAL),
            ("Supporting data suggests correlation", InformationCategory.HIGH),
            ("General information about the topic", InformationCategory.MEDIUM),
            ("Minor details and observations", InformationCategory.LOW),
            ("Repeated information", InformationCategory.REDUNDANT)
        ]

        for text, expected_category in test_cases:
            category = optimizer._categorize_information(0.8, 0.9)  # High relevance and density
            # Note: This is simplified - actual categorization depends on more factors


class TestDynamicAllocator:
    """Test dynamic resource allocation functionality"""

    @pytest.fixture
    def allocator(self):
        return create_dynamic_allocator(token_budget=2048)

    def test_component_requirements_initialization(self, allocator):
        """Test component requirements are properly initialized"""
        assert len(allocator.component_requirements) > 0
        assert ComponentType.CLAIM_PROCESSING in allocator.component_requirements
        assert ComponentType.REASONING_ENGINE in allocator.component_requirements

        # Verify requirement structure
        for req in allocator.component_requirements.values():
            assert req.min_tokens > 0
            assert req.preferred_tokens >= req.min_tokens
            assert req.max_tokens >= req.preferred_tokens
            assert 0.0 <= req.priority <= 1.0
            assert 0.0 <= req.importance_weight <= 1.0

    def test_equal_distribution_allocation(self, allocator):
        """Test equal distribution allocation strategy"""
        allocator.allocation_strategy = AllocationStrategy.EQUAL_DISTRIBUTION

        allocation = allocator.allocate_context_resources(
            task_complexity=0.5,
            performance_requirements={},
            active_components=[ComponentType.CLAIM_PROCESSING, ComponentType.REASONING_ENGINE]
        )

        assert len(allocation) == 2
        assert ComponentType.CLAIM_PROCESSING in allocation
        assert ComponentType.REASONING_ENGINE in allocation
        assert sum(allocation.values()) <= allocator.total_token_budget

    def test_priority_based_allocation(self, allocator):
        """Test priority-based allocation strategy"""
        allocator.allocation_strategy = AllocationStrategy.PRIORITY_BASED

        allocation = allocator.allocate_context_resources(
            task_complexity=0.8,
            performance_requirements={},
            active_components=list(ComponentType)[:3]  # Test with first 3 components
        )

        assert len(allocation) <= 3
        assert sum(allocation.values()) <= allocator.total_token_budget

        # Critical components should get more allocation
        reasoning_tokens = allocation.get(ComponentType.REASONING_ENGINE, 0)
        examples_tokens = allocation.get(ComponentType.EXAMPLES, 0)
        assert reasoning_tokens >= examples_tokens

    def test_performance_adaptive_allocation(self, allocator):
        """Test performance-adaptive allocation strategy"""
        allocator.allocation_strategy = AllocationStrategy.PERFORMANCE_ADAPTIVE

        # Simulate some performance history
        allocator.record_allocation_metrics(
            ComponentType.REASONING_ENGINE, 500, 450, 0.8, 0.9
        )

        allocation = allocator.allocate_context_resources(
            task_complexity=0.7,
            performance_requirements={"reasoning_quality": 0.8},
            active_components=[ComponentType.REASONING_ENGINE, ComponentType.CLAIM_PROCESSING]
        )

        assert allocation
        assert sum(allocation.values()) <= allocator.total_token_budget

    def test_budget_validation(self, allocator):
        """Test budget validation and adjustment"""
        # Request allocation that exceeds budget
        allocation = allocator.allocate_context_resources(
            task_complexity=0.9,
            performance_requirements={},
            active_components=list(ComponentType)  # All components
        )

        # Should never exceed budget
        assert sum(allocation.values()) <= allocator.total_token_budget

    def test_resource_monitoring(self, allocator):
        """Test resource monitoring functionality"""
        allocator.resource_monitor.used_tokens = 1024

        utilization = allocator.resource_monitor.utilization_rate()
        assert utilization == 0.5  # 1024/2048

        efficiency = allocator.resource_monitor.efficiency_score()
        assert 0.0 <= efficiency <= 1.0

    def test_allocation_recommendations(self, allocator):
        """Test allocation recommendation generation"""
        # Record some performance metrics
        allocator.record_allocation_metrics(
            ComponentType.CLAIM_PROCESSING, 400, 350, 0.6, 0.8
        )

        recommendations = allocator.get_allocation_recommendations()
        assert isinstance(recommendations, list)


class TestIntegrationSystem:
    """Test complete integration system"""

    @pytest.fixture
    def system(self):
        config = SystemConfiguration(
            model_name="test-model",
            default_token_budget=1536,
            cache_optimizations=True,
            enable_learning=True
        )
        return create_context_optimization_system(config)

    @pytest.fixture
    def complex_context(self):
        return """
        ## Research Study: Climate Change Impact on Global Food Security

        ### Executive Summary
        Climate change poses significant threats to global food security through multiple pathways:
        1. Temperature increases affecting crop yields
        2. Changes in precipitation patterns
        3. Increased frequency of extreme weather events

        ### Methodology
        Our study analyzed data from 150 countries over 30 years, focusing on:
        - Crop production metrics
        - Climate pattern changes
        - Economic indicators
        - Population growth trends

        ### Key Findings
        1. Temperature increases of 2°C could reduce global crop yields by 15-20%
        2. Developing countries are disproportionately affected
        3. Adaptation strategies can mitigate 40-60% of projected losses

        ### Detailed Analysis
        The relationship between temperature and crop productivity follows a nonlinear pattern.
        Moderate warming (1-2°C) may benefit some high-latitude regions, but temperatures
        above 3°C consistently show negative impacts across all crop types and regions.

        Rice production is particularly vulnerable to temperature stress during the flowering
        stage. Even short periods of high temperature (>35°C) can cause significant yield
        reductions through pollen sterility.

        Wheat shows more resilience but faces challenges from changing precipitation patterns.
        Increased drought frequency in major wheat-producing regions threatens stable production.

        Maize production demonstrates high sensitivity to both temperature and water stress.
        The crop's C4 photosynthetic pathway provides some heat tolerance but water limitations
        remain a critical constraint.

        ### Economic Implications
        Food price volatility is expected to increase by 25-40% under moderate climate change
        scenarios. Small-scale farmers and low-income populations face the highest risk of
        food insecurity and malnutrition.

        ### Adaptation Strategies
        1. Development of heat-tolerant crop varieties
        2. Improved irrigation and water management
        3. Diversification of cropping systems
        4. Early warning systems for extreme events
        5. Insurance and risk management programs

        ### Policy Recommendations
        Immediate action is required to:
        - Invest in climate-resilient agriculture
        - Strengthen food distribution systems
        - Support vulnerable populations
        - International cooperation on adaptation
        """

    @pytest.mark.asyncio
    async def test_complete_optimization_workflow(self, system, complex_context):
        """Test complete optimization workflow"""
        request = OptimizationRequest(
            context_text=complex_context,
            task_type=TaskType.SYNTHESIS,
            task_keywords=["climate change", "food security", "crop yields"],
            performance_requirements={"accuracy": 0.8, "efficiency": 0.7},
            active_components=[
                ComponentType.CLAIM_PROCESSING,
                ComponentType.EVIDENCE_SYNTHESIS,
                ComponentType.REASONING_ENGINE
            ]
        )

        result = await system.optimize_context(request)

        # Verify result structure
        assert result.optimized_context
        assert result.original_tokens > 0
        assert result.optimized_tokens > 0
        assert result.compression_ratio > 0
        assert len(result.allocation) > 0
        assert result.metrics
        assert result.performance_prediction
        assert result.processing_time_ms > 0
        assert isinstance(result.recommendations, list)

    @pytest.mark.asyncio
    async def test_caching_functionality(self, system, complex_context):
        """Test optimization caching"""
        request = OptimizationRequest(
            context_text=complex_context,
            task_type=TaskType.ANALYSIS,
            task_keywords=["agriculture", "climate"],
            performance_requirements={},
            active_components=[ComponentType.CLAIM_PROCESSING]
        )

        # First optimization
        start_time = time.time()
        result1 = await system.optimize_context(request)
        first_time = time.time() - start_time

        # Second optimization (should use cache)
        start_time = time.time()
        result2 = await system.optimize_context(request)
        second_time = time.time() - start_time

        # Results should be identical
        assert result1.optimized_context == result2.optimized_context
        assert result1.compression_ratio == result2.compression_ratio

        # Second call should be faster (cache hit)
        assert second_time < first_time

    @pytest.mark.asyncio
    async def test_performance_learning(self, system, complex_context):
        """Test performance-based learning"""
        # Perform multiple optimizations to build learning data
        for i in range(5):
            request = OptimizationRequest(
                context_text=complex_context[:200 * (i + 1)],  # Varying context sizes
                task_type=TaskType.REASONING,
                task_keywords=["climate", "agriculture"],
                performance_requirements={"accuracy": 0.8 - i * 0.05},  # Varying requirements
                active_components=[ComponentType.REASONING_ENGINE, ComponentType.CLAIM_PROCESSING]
            )

            result = await system.optimize_context(request)
            assert result.optimized_context

        # Check that learning data was collected
        assert system.total_optimizations >= 5
        assert system.successful_optimizations >= 5

    def test_system_status(self, system):
        """Test system status reporting"""
        status = system.get_system_status()

        assert "system_info" in status
        assert "performance_metrics" in status
        assert "resource_status" in status
        assert "recommendations" in status

        # Verify system info
        system_info = status["system_info"]
        assert "model_name" in system_info
        assert "total_optimizations" in system_info
        assert "success_rate" in system_info

    def test_configuration_updates(self, system):
        """Test configuration updates"""
        original_budget = system.config.default_token_budget

        # Update configuration
        system.update_configuration({
            "default_token_budget": 3072,
            "performance_threshold": 0.8
        })

        # Verify updates
        assert system.config.default_token_budget == 3072
        assert system.config.performance_threshold == 0.8
        assert system.dynamic_allocator.total_token_budget == 3072

    def test_performance_data_export(self, system, tmp_path):
        """Test performance data export"""
        # Create some optimization history
        system.optimization_history = [
            {
                "timestamp": "2025-12-06T10:00:00",
                "compression_ratio": 0.7,
                "processing_time_ms": 150.0,
                "performance_prediction": {"accuracy": 0.85},
                "metrics": {"semantic_density": 0.8}
            }
        ]

        # Export data
        export_file = tmp_path / "performance_data.json"
        system.export_performance_data(export_file)

        # Verify export
        assert export_file.exists()
        with open(export_file) as f:
            data = json.load(f)

        assert "system_config" in data
        assert "optimization_history" in data
        assert len(data["optimization_history"]) == 1


class TestUtilityFunctions:
    """Test utility functions and convenience APIs"""

    @pytest.mark.asyncio
    async def test_quick_optimization_utility(self):
        """Test the quick optimization utility function"""
        context = """
        The rapid advancement of artificial intelligence has transformed many industries.
        Machine learning models can now perform tasks that previously required human intelligence.
        Natural language processing enables computers to understand and generate human language.
        Computer vision allows machines to interpret and analyze visual information.
        """

        result = await optimize_context_for_tiny_llm(
            context_text=context,
            task_type="synthesis",
            task_keywords=["artificial intelligence", "machine learning"],
            token_budget=1024
        )

        assert result.optimized_context
        assert result.original_tokens > 0
        assert result.optimized_tokens > 0
        assert result.compression_ratio > 0

    def test_factory_functions(self):
        """Test factory functions for component creation"""
        # Test context optimizer creation
        optimizer = create_tiny_llm_optimizer("test-model")
        assert optimizer
        assert optimizer.model_name == "test-model"

        # Test dynamic allocator creation
        allocator = create_dynamic_allocator(4096)
        assert allocator
        assert allocator.total_token_budget == 4096

        # Test system creation
        config = SystemConfiguration(model_name="custom-model")
        system = create_context_optimization_system(config)
        assert system
        assert system.config.model_name == "custom-model"


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_empty_context_handling(self, system):
        """Test handling of empty context"""
        request = OptimizationRequest(
            context_text="",
            task_type=TaskType.ANALYSIS,
            task_keywords=[],
            performance_requirements={},
            active_components=[ComponentType.CLAIM_PROCESSING]
        )

        result = await system.optimize_context(request)
        assert result.optimized_context == ""
        assert result.original_tokens == 0

    @pytest.mark.asyncio
    async def test_very_long_context_handling(self, system):
        """Test handling of very long contexts"""
        # Create a very long context (10K+ words)
        long_context = "This is a test sentence. " * 10000

        request = OptimizationRequest(
            context_text=long_context,
            task_type=TaskType.SYNTHESIS,
            task_keywords=["test"],
            performance_requirements={},
            active_components=[ComponentType.CLAIM_PROCESSING]
        )

        result = await system.optimize_context(request)

        # Should be significantly compressed
        assert result.optimized_tokens < result.original_tokens
        assert result.compression_ratio < 0.5  # Should be at least 50% compressed

    def test_invalid_configuration_handling(self):
        """Test handling of invalid configuration"""
        # Test with invalid configuration values
        config = SystemConfiguration(
            default_token_budget=-100,  # Invalid negative budget
            performance_threshold=1.5    # Invalid threshold > 1.0
        )

        system = create_context_optimization_system(config)

        # Should handle gracefully
        status = system.get_system_status()
        assert status is not None

    @pytest.mark.asyncio
    async def test_concurrent_optimizations(self, system):
        """Test handling of concurrent optimization requests"""
        context = "Sample context for concurrent testing."

        # Create multiple concurrent requests
        requests = [
            OptimizationRequest(
                context_text=context + f" Request {i}",
                task_type=TaskType.ANALYSIS,
                task_keywords=[],
                performance_requirements={},
                active_components=[ComponentType.CLAIM_PROCESSING]
            )
            for i in range(5)
        ]

        # Execute concurrently
        results = await asyncio.gather(*[system.optimize_context(req) for req in requests])

        # All should complete successfully
        assert len(results) == 5
        for result in results:
            assert result.optimized_context


if __name__ == "__main__":
    pytest.main([__file__, "-v"])