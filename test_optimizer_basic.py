#!/usr/bin/env python3
"""
Basic test for context optimization system components
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_semantic_density():
    """Test semantic density calculation"""
    try:
        from processing.advanced_context_optimizer import InformationTheoreticOptimizer

        optimizer = InformationTheoreticOptimizer()

        test_text = """
        Artificial intelligence and machine learning are transforming technology.
        Neural networks can learn complex patterns from data. Deep learning uses
        multiple layers to extract features progressively. This hierarchical approach
        has led to breakthroughs in computer vision and natural language processing.
        """

        density = optimizer.calculate_semantic_density(test_text)
        print(f"Semantic density: {density:.3f}")

        # Test with empty text
        empty_density = optimizer.calculate_semantic_density("")
        print(f"Empty text density: {empty_density:.3f}")

        # Test with simple text
        simple_density = optimizer.calculate_semantic_density("This is a test.")
        print(f"Simple text density: {simple_density:.3f}")

        return True

    except Exception as e:
        print(f"Semantic density test failed: {str(e)}")
        return False

def test_entity_extraction():
    """Test entity extraction"""
    try:
        from processing.advanced_context_optimizer import InformationTheoreticOptimizer

        optimizer = InformationTheoreticOptimizer()

        test_text = "IBM Granite Tiny model processes 42,000 tokens at 300K parameters."
        entities = optimizer.extract_key_entities(test_text)
        print(f"Extracted entities: {entities}")

        return len(entities) > 0

    except Exception as e:
        print(f"Entity extraction test failed: {str(e)}")
        return False

def test_relevance_calculation():
    """Test relevance calculation"""
    try:
        from processing.advanced_context_optimizer import InformationTheoreticOptimizer

        optimizer = InformationTheoreticOptimizer()

        text = "Machine learning algorithms use neural networks for pattern recognition."
        task_keywords = {"machine learning", "artificial intelligence", "algorithms"}
        context_keywords = {"neural networks", "pattern recognition"}

        relevance = optimizer.calculate_relevance(text, task_keywords, context_keywords)
        print(f"Relevance score: {relevance:.3f}")

        return 0.0 <= relevance <= 1.0

    except Exception as e:
        print(f"Relevance calculation test failed: {str(e)}")
        return False

def test_dynamic_allocation():
    """Test dynamic resource allocation"""
    try:
        from processing.dynamic_context_allocator import DynamicContextAllocator, ComponentType

        allocator = DynamicContextAllocator(token_budget=2048)

        # Test allocation for different scenarios
        allocation1 = allocator.allocate_context_resources(
            task_complexity=0.3,
            performance_requirements={},
            active_components=[ComponentType.CLAIM_PROCESSING]
        )

        allocation2 = allocator.allocate_context_resources(
            task_complexity=0.8,
            performance_requirements={},
            active_components=[ComponentType.REASONING_ENGINE, ComponentType.CLAIM_PROCESSING]
        )

        print(f"Simple allocation: {allocation1}")
        print(f"Complex allocation: {allocation2}")

        # Verify allocations are within budget
        total1 = sum(allocation1.values())
        total2 = sum(allocation2.values())

        print(f"Simple total: {total1} tokens")
        print(f"Complex total: {total2} tokens")

        return total1 <= 2048 and total2 <= 2048

    except Exception as e:
        print(f"Dynamic allocation test failed: {str(e)}")
        return False

def test_resource_monitoring():
    """Test resource monitoring"""
    try:
        from processing.dynamic_context_allocator import DynamicContextAllocator

        allocator = DynamicContextAllocator(token_budget=1024)

        # Simulate resource usage
        allocator.resource_monitor.used_tokens = 512
        allocator.resource_monitor.success_rate = 0.85
        allocator.resource_monitor.quality_score = 0.9

        utilization = allocator.resource_monitor.utilization_rate()
        efficiency = allocator.resource_monitor.efficiency_score()

        print(f"Resource utilization: {utilization:.2f}")
        print(f"Resource efficiency: {efficiency:.2f}")

        return 0.0 <= utilization <= 1.0 and 0.0 <= efficiency <= 1.0

    except Exception as e:
        print(f"Resource monitoring test failed: {str(e)}")
        return False

def main():
    """Run all basic tests"""
    print("Testing Context Optimization System - Basic Components")
    print("=" * 60)

    tests = [
        ("Semantic Density Calculation", test_semantic_density),
        ("Entity Extraction", test_entity_extraction),
        ("Relevance Calculation", test_relevance_calculation),
        ("Dynamic Resource Allocation", test_dynamic_allocation),
        ("Resource Monitoring", test_resource_monitoring)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\nTesting: {test_name}")
        print("-" * 40)

        if test_func():
            print(f"PASSED: {test_name}")
            passed += 1
        else:
            print(f"FAILED: {test_name}")

    print(f"\n{'='*60}")
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("SUCCESS: All basic tests passed!")
        print("Context optimization system components are working correctly.")
    else:
        print(f"WARNING: {total - passed} tests failed.")
        print("Some components may need adjustment.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)