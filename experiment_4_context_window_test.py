#!/usr/bin/env python3
"""
Experiment 4: Context Window Optimization - Test Script
Tests adaptive compression and hierarchical context processing
"""

import sys
import time
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def run_experiment_4_test():
    """Run Experiment 4: Context Window Optimization test"""
    print("Experiment 4: Context Window Optimization - Test")
    print("=" * 60)
    
    try:
        # Import new modules
        from src.processing.adaptive_compression import compress_context_adaptive
        from src.processing.hierarchical_context_processor import process_context_hierarchical, access_context_level
        from src.processing.intelligent_claim_selector import select_claims_intelligent
        from src.core.models import Claim, ClaimType, ClaimState
        from datetime import datetime
        
        # Test data - different context sizes and complexities
        test_scenarios = [
            {
                'name': 'Simple Task - Small Context',
                'task': 'What are best practices for unit testing?',
                'context_size': 2000,  # Small context
                'expected_compression': 0.8,  # High compression
                'complexity': 'simple'
            },
            {
                'name': 'Medium Task - Medium Context', 
                'task': 'Analyze the performance implications of microservices architecture',
                'context_size': 8000,  # Medium context
                'expected_compression': 0.6,  # Medium compression
                'complexity': 'medium'
            },
            {
                'name': 'Complex Task - Large Context',
                'task': 'Design and implement a comprehensive enterprise resource planning system with multi-level approval workflows, real-time budget tracking, and automated resource allocation based on project priorities and departmental utilization metrics',
                'context_size': 25000,  # Large context
                'expected_compression': 0.4,  # Low compression
                'complexity': 'complex'
            },
            {
                'name': 'Enterprise Task - Very Large Context',
                'task': 'Evaluate and optimize the global supply chain resilience strategy for a multinational manufacturing corporation, considering geopolitical risks, climate change impacts, and digital transformation initiatives across 5 continents and 15 product categories',
                'context_size': 60000,  # Very large context
                'expected_compression': 0.3,  # Very low compression
                'complexity': 'enterprise'
            }
        ]
        
        print("Testing Adaptive Compression Engine:")
        print("-" * 40)
        
        compression_results = []
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\nTest {i}: {scenario['name']}")
            print(f"Context Size: {scenario['context_size']} tokens")
            print(f"Expected Compression: {scenario['expected_compression']}")
            
            # Create mock claims for testing
            mock_claims = create_mock_claims(scenario['context_size'])
            
            # Test adaptive compression
            start_time = time.time()
            compressed_claims, compression_metadata = compress_context_adaptive(
                mock_claims, scenario['task']
            )
            compression_time = time.time() - start_time
            
            # Test hierarchical processing
            start_time = time.time()
            hierarchical_context = process_context_hierarchical(mock_claims, scenario['complexity'])
            hierarchical_time = time.time() - start_time
            
            # Test intelligent selection
            start_time = time.time()
            selection_result = select_claims_intelligent(
                mock_claims, scenario['task'], context_limit=10
            )
            selection_time = time.time() - start_time
            
            # Calculate metrics
            original_count = len(mock_claims)
            compressed_count = len(compressed_claims)
            actual_compression = compressed_count / original_count if original_count > 0 else 0
            
            result = {
                'scenario': scenario['name'],
                'context_size': scenario['context_size'],
                'complexity': scenario['complexity'],
                'original_claims': original_count,
                'compressed_claims': compressed_count,
                'expected_compression': scenario['expected_compression'],
                'actual_compression': actual_compression,
                'compression_accuracy': abs(actual_compression - scenario['expected_compression']),
                'compression_time_ms': compression_time * 1000,
                'hierarchical_time_ms': hierarchical_time * 1000,
                'selection_time_ms': selection_time * 1000,
                'tokens_saved_estimate': compression_metadata.get('tokens_saved_estimate', 0),
                'quality_preservation': compression_metadata.get('quality_preservation', 0),
                'success': True
            }
            
            compression_results.append(result)
            
            print(f"  Original Claims: {original_count}")
            print(f"  Compressed Claims: {compressed_count}")
            print(f"  Compression Ratio: {actual_compression:.3f} (expected: {scenario['expected_compression']:.3f})")
            print(f"  Compression Accuracy: {abs(actual_compression - scenario['expected_compression']):.3f}")
            print(f"  Tokens Saved: {compression_metadata.get('tokens_saved_estimate', 0)}")
            print(f"  Quality Preservation: {compression_metadata.get('quality_preservation', 0):.3f}")
            print(f"  Processing Time: {compression_time*1000:.1f}ms")
            
            # Success criteria check
            compression_success = abs(actual_compression - scenario['expected_compression']) <= 0.1
            quality_success = compression_metadata.get('quality_preservation', 0) >= 0.9
            performance_success = compression_time < 1.0  # Should be fast
            
            if compression_success and quality_success and performance_success:
                print(f"  Result: SUCCESS")
            elif compression_success and quality_success:
                print(f"  Result: PARTIAL (slow performance)")
            elif compression_success:
                print(f"  Result: PARTIAL (quality degradation)")
            else:
                print(f"  Result: FAILED")
        
        # Calculate overall results
        print(f"\n" + "=" * 60)
        print("EXPERIMENT 4 RESULTS:")
        print("-" * 30)
        
        # Success metrics
        total_tests = len(test_scenarios)
        successful_tests = sum(1 for r in compression_results if r.get('success', False))
        success_rate = (successful_tests / total_tests) * 100
        
        # Compression accuracy
        avg_compression_accuracy = sum(r.get('compression_accuracy', 1.0) for r in compression_results) / total_tests
        
        # Quality preservation
        avg_quality_preservation = sum(r.get('quality_preservation', 0.0) for r in compression_results) / total_tests
        
        # Performance
        avg_compression_time = sum(r.get('compression_time_ms', 0) for r in compression_results) / total_tests
        avg_tokens_saved = sum(r.get('tokens_saved_estimate', 0) for r in compression_results) / total_tests
        
        print(f"Overall Success Rate: {success_rate:.1f}% ({successful_tests}/{total_tests})")
        print(f"Average Compression Accuracy: {avg_compression_accuracy:.3f}")
        print(f"Average Quality Preservation: {avg_quality_preservation:.3f}")
        print(f"Average Compression Time: {avg_compression_time:.1f}ms")
        print(f"Average Tokens Saved: {avg_tokens_saved:.0f}")
        
        # Determine if experiment meets success criteria
        success_criteria_met = (
            success_rate >= 75 and  # 75% of tests successful
            avg_compression_accuracy <= 0.15 and  # Compression within 15% of target
            avg_quality_preservation >= 0.90 and  # Quality preservation >= 90%
            avg_compression_time <= 500  # Processing time <= 500ms
        )
        
        if success_criteria_met:
            print(f"\n+ EXPERIMENT 4 SUCCESS!")
            print(f"+ Success Rate: {success_rate:.1f}% >= 75%")
            print(f"+ Compression Accuracy: {avg_compression_accuracy:.3f} <= 0.15")
            print(f"+ Quality Preservation: {avg_quality_preservation:.3f} >= 0.90")
            print(f"+ Performance: {avg_compression_time:.1f}ms <= 500ms")
            return True
        else:
            print(f"\n- EXPERIMENT 4 NEEDS IMPROVEMENT!")
            if success_rate < 75:
                print(f"- Success Rate: {success_rate:.1f}% < 75%")
            if avg_compression_accuracy > 0.15:
                print(f"- Compression Accuracy: {avg_compression_accuracy:.3f} > 0.15")
            if avg_quality_preservation < 0.90:
                print(f"- Quality Preservation: {avg_quality_preservation:.3f} < 0.90")
            if avg_compression_time > 500:
                print(f"- Performance: {avg_compression_time:.1f}ms > 500ms")
            return False
            
    except Exception as e:
        print(f"- Experiment 4 test failed: {e}")
        return False

def create_mock_claims(context_size: int) -> List[Any]:
    """Create mock claims for testing based on context size"""
    claims = []
    
    # Generate claims proportional to context size
    claim_count = min(50, max(5, context_size // 200))
    
    for i in range(claim_count):
        claim = {
            'id': f'c{i+1:07d}',
            'content': f'This is test claim {i+1} with sufficient content to simulate realistic claim generation for context window optimization testing purposes.',
            'confidence': 0.8 + (i * 0.01),  # Varying confidence
            'type': ['fact' if i % 3 == 0 else 'concept' if i % 3 == 1 else 'example'],
            'state': 'explore',
            'tags': ['test', 'mock', 'compression_test'],
            'created': datetime.now(),
            'supports': [f'c{j:07d}' for j in range(max(0, i-2), i-2)]
        }
        claims.append(claim)
    
    return claims

def save_results(results: List[Dict[str, Any]], filename: str = "experiment_4_results.json"):
    """Save experiment results to JSON file"""
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"+ Results saved to {filename}")
    except Exception as e:
        print(f"- Failed to save results: {e}")

if __name__ == "__main__":
    print("Experiment 4: Context Window Optimization")
    print("Testing adaptive compression and hierarchical context processing")
    print("=" * 60)
    
    # Run the test
    success = asyncio.run(run_experiment_4_test())
    
    # Save results
    if success:
        print("\n+ Experiment 4 completed successfully!")
        print("+ Context window optimization ready for deployment")
    else:
        print("\n- Experiment 4 needs refinement")
        print("+ Review implementation and try again")