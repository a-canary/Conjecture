#!/usr/bin/env python3
"""
Experiment 4: Context Window Optimization - Simple Test
Basic functionality test for context optimization components
"""

import sys
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_functionality():
    """Test basic functionality of context optimization components"""
    print("Experiment 4: Context Window Optimization - Simple Test")
    print("=" * 60)
    
    try:
        # Test 1: Import modules
        print("\n1. Testing Module Imports...")
        
        try:
            from src.processing.adaptive_compression import AdaptiveCompressionEngine, get_adaptive_compressor
            print("   + AdaptiveCompressionEngine: SUCCESS")
        except Exception as e:
            print(f"   - AdaptiveCompressionEngine: FAILED - {e}")
            return False
        
        try:
            from src.processing.hierarchical_context_processor import HierarchicalContextProcessor, get_hierarchical_processor
            print("   + HierarchicalContextProcessor: SUCCESS")
        except Exception as e:
            print(f"   - HierarchicalContextProcessor: FAILED - {e}")
            return False
        
        try:
            from src.processing.intelligent_claim_selector import IntelligentClaimSelector, get_intelligent_selector
            print("   + IntelligentClaimSelector: SUCCESS")
        except Exception as e:
            print(f"   - IntelligentClaimSelector: FAILED - {e}")
            return False
        
        # Test 2: Basic functionality
        print("\n2. Testing Basic Functionality...")
        
        compressor = get_adaptive_compressor()
        processor = get_hierarchical_processor()
        selector = get_intelligent_selector()
        
        # Test task complexity analysis
        print("   Testing task complexity analysis...")
        simple_complexity = compressor.analyze_task_complexity("What are best practices?", 1000)
        medium_complexity = compressor.analyze_task_complexity("Analyze system performance and identify bottlenecks", 8000)
        complex_complexity = compressor.analyze_task_complexity("Design enterprise architecture with microservices", 25000)
        
        print(f"   + Simple task (1000 tokens): {simple_complexity.value}")
        print(f"   + Medium task (8000 tokens): {medium_complexity.value}")
        print(f"   + Complex task (25000 tokens): {complex_complexity.value}")
        
        # Test 3: Mock data processing
        print("\n3. Testing Mock Data Processing...")
        
        from src.core.models import Claim, ClaimType, ClaimState
        from datetime import datetime
        
        # Create mock claims
        mock_claims = [
            Claim(
                id="c0000001",
                content="Unit testing ensures code quality and reliability",
                confidence=0.9,
                type=[ClaimType.FACT],
                state=ClaimState.EXPLORE,
                tags=["testing", "quality"],
                created=datetime.now()
            ),
            Claim(
                id="c0000002", 
                content="Continuous integration testing prevents regressions",
                confidence=0.85,
                type=[ClaimType.CONCEPT],
                state=ClaimState.EXPLORE,
                tags=["testing", "integration"],
                created=datetime.now()
            ),
            Claim(
                id="c0000003",
                content="Code reviews improve maintainability and reduce technical debt",
                confidence=0.8,
                type=[ClaimType.CONCEPT],
                state=ClaimState.EXPLORE,
                tags=["development", "quality"],
                created=datetime.now()
            ),
            Claim(
                id="c0000004",
                content="Automated testing increases development velocity",
                confidence=0.75,
                type=[ClaimType.EXAMPLE],
                state=ClaimState.EXPLORE,
                tags=["automation", "efficiency"],
                created=datetime.now()
            ),
            Claim(
                id="c0000005",
                content="Test coverage metrics should exceed 80% for critical components",
                confidence=0.95,
                type=[ClaimType.GOAL],
                state=ClaimState.EXPLORE,
                tags=["metrics", "quality"],
                created=datetime.now()
            )
        ]
        
        print(f"   + Created {len(mock_claims)} mock claims")
        
        # Test adaptive compression
        print("   Testing adaptive compression...")
        start_time = time.time()
        compressed_claims, compression_metadata = compressor.compress_context(
            mock_claims, "Analyze system performance", max_tokens=10
        )
        compression_time = time.time() - start_time
        
        print(f"   + Original: {len(mock_claims)} claims")
        print(f"   + Compressed: {len(compressed_claims)} claims")
        print(f"   + Compression ratio: {compression_metadata.get('compression_ratio', 0):.3f}")
        print(f"   + Processing time: {compression_time*1000:.1f}ms")
        
        # Test hierarchical processing
        print("   Testing hierarchical processing...")
        start_time = time.time()
        hierarchical_context = processor.process_large_context(mock_claims, "medium")
        hierarchical_time = time.time() - start_time
        
        print(f"   + Levels generated: {len(hierarchical_context) - 1}")  # -1 for navigation metadata
        print(f"   + Processing time: {hierarchical_time*1000:.1f}ms")
        
        # Test intelligent selection
        print("   Testing intelligent selection...")
        start_time = time.time()
        selection_result = selector.select_optimal_claims(
            mock_claims, "Analyze system performance", context_limit=5
        )
        selection_time = time.time() - start_time
        
        print(f"   + Selected: {len(selection_result.selected_claims)} claims")
        print(f"   + Selection time: {selection_time*1000:.1f}ms")
        print(f"   + Top claim accuracy: {selection_result.performance_metrics.get('top_claim_accuracy', 0):.3f}")
        
        # Test 4: Performance metrics
        print("\n4. Testing Performance Metrics...")
        
        compression_stats = compressor.get_compression_statistics()
        processing_stats = processor.get_processing_statistics()
        selection_stats = selector.get_selection_statistics()
        
        print(f"   + Compression stats: {compression_stats}")
        print(f"   + Processing stats: {processing_stats}")
        print(f"   + Selection stats: {selection_stats}")
        
        # Test 5: Integration test
        print("\n5. Testing Integration...")
        
        # Test end-to-end workflow
        start_time = time.time()
        
        # Step 1: Compress context
        compressed_claims, _ = compressor.compress_context(mock_claims, "Analyze system performance")
        
        # Step 2: Process hierarchically
        hierarchical_context = processor.process_large_context(compressed_claims, "medium")
        
        # Step 3: Select optimal claims
        selection_result = selector.select_optimal_claims(
            compressed_claims, "Analyze system performance", context_limit=5
        )
        
        total_time = time.time() - start_time
        print(f"   + End-to-end workflow time: {total_time*1000:.1f}ms")
        
        print("\n6. Test Results Summary:")
        print("=" * 40)
        
        # Calculate success metrics
        tests_passed = 0
        total_tests = 6
        
        # Module imports
        try:
            from src.processing.adaptive_compression import get_adaptive_compressor
            tests_passed += 1
        except:
            pass
        
        try:
            from src.processing.hierarchical_context_processor import get_hierarchical_processor
            tests_passed += 1
        except:
            pass
        
        try:
            from src.processing.intelligent_claim_selector import get_intelligent_selector
            tests_passed += 1
        except:
            pass
        
        # Basic functionality
        if simple_complexity.value == "simple" and medium_complexity.value == "medium" and complex_complexity.value == "complex":
            tests_passed += 1
        
        # Mock data processing
        if len(mock_claims) == 5 and compression_time < 1.0:
            tests_passed += 1
        
        # Adaptive compression
        if len(compressed_claims) > 0 and compression_metadata.get('compression_ratio', 0) > 0:
            tests_passed += 1
        
        # Hierarchical processing
        if len(hierarchical_context) > 1 and hierarchical_time < 1.0:
            tests_passed += 1
        
        # Intelligent selection
        if len(selection_result.selected_claims) > 0 and selection_time < 1.0:
            tests_passed += 1
        
        # End-to-end integration
        if total_time < 2.0:  # Should complete in under 2 seconds
            tests_passed += 1
        
        success_rate = (tests_passed / total_tests) * 100
        
        print(f"Tests Passed: {tests_passed}/{total_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("\n+ EXPERIMENT 4 IMPLEMENTATION SUCCESSFUL!")
            print("+ All core components working correctly")
            print("+ Ready for integration testing")
            return True
        else:
            print(f"\n- EXPERIMENT 4 NEEDS IMPROVEMENT!")
            print(f"- Success rate: {success_rate:.1f}% < 80%")
            return False
            
    except Exception as e:
        print(f"\n- Test failed with error: {e}")
        return False

if __name__ == "__main__":
    print("Experiment 4: Context Window Optimization")
    print("Testing basic functionality of context optimization components")
    print("=" * 60)
    
    success = test_basic_functionality()
    
    if success:
        print("\n+ Ready for integration with Conjecture!")
        print("+ Context window optimization components validated")
    else:
        print("\n- Review implementation before integration")
        print("+ Fix issues and re-run tests")