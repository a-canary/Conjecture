#!/usr/bin/env python3
"""
Simple test for context optimization system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_functionality():
    """Test basic context optimization functionality"""
    try:
        # Test semantic density calculation
        from processing.advanced_context_optimizer import InformationTheoreticOptimizer

        optimizer = InformationTheoreticOptimizer()

        test_text = """
        Artificial intelligence and machine learning are transforming technology.
        Neural networks can learn complex patterns from data. Deep learning uses
        multiple layers to extract features progressively.
        """

        density = optimizer.calculate_semantic_density(test_text)
        print(f"✅ Semantic density calculation: {density:.2f}")

        # Test entity extraction
        entities = optimizer.extract_key_entities(test_text)
        print(f"✅ Entity extraction: {len(entities)} entities found")

        # Test relevance calculation
        task_keywords = {"artificial intelligence", "machine learning"}
        relevance = optimizer.calculate_relevance(test_text, task_keywords, set())
        print(f"✅ Relevance calculation: {relevance:.2f}")

        return True

    except Exception as e:
        print(f"❌ Basic functionality test failed: {str(e)}")
        return False

def test_dynamic_allocator():
    """Test dynamic resource allocation"""
    try:
        from processing.dynamic_context_allocator import DynamicContextAllocator, ComponentType

        allocator = DynamicContextAllocator(token_budget=2048)

        # Test allocation
        allocation = allocator.allocate_context_resources(
            task_complexity=0.7,
            performance_requirements={},
            active_components=[ComponentType.CLAIM_PROCESSING, ComponentType.REASONING_ENGINE]
        )

        print(f"✅ Resource allocation: {allocation}")
        print(f"✅ Total allocated: {sum(allocation.values())} tokens")

        # Test resource monitoring
        utilization = allocator.resource_monitor.utilization_rate()
        print(f"✅ Resource utilization: {utilization:.2f}")

        return True

    except Exception as e:
        print(f"❌ Dynamic allocator test failed: {str(e)}")
        return False

def main():
    """Run simple tests"""
    print("🚀 Testing Context Optimization System")
    print("=" * 50)

    success_count = 0
    total_tests = 2

    if test_basic_functionality():
        success_count += 1

    if test_dynamic_allocator():
        success_count += 1

    print(f"\n📊 Results: {success_count}/{total_tests} tests passed")

    if success_count == total_tests:
        print("🎉 All tests passed! Context optimization system is working.")
    else:
        print("⚠️  Some tests failed. Check the implementation.")

if __name__ == "__main__":
    main()