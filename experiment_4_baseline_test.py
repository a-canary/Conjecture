#!/usr/bin/env python3
"""
Experiment 4: Context Window Optimization - Baseline Measurements
Measure current performance before applying context optimization
"""

import sys
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def run_baseline_measurements():
    """Run baseline measurements for context window performance"""
    print("Experiment 4: Context Window Optimization - Baseline Measurements")
    print("=" * 70)
    
    try:
        from src.core.models import Claim, ClaimType, ClaimState
        from src.processing.unified_llm_manager import get_unified_llm_manager
        from src.data.data_manager import get_data_manager
        
        # Initialize components
        llm_manager = get_unified_llm_manager()
        data_manager = get_data_manager()
        
        print("\n1. System Health Check...")
        health = llm_manager.health_check()
        print(f"   + LLM Manager Status: {health['overall_status']}")
        print(f"   + Available Providers: {health['available_providers']}")
        
        print("\n2. Creating Test Claims...")
        
        # Create test claims of varying complexity
        test_claims = []
        
        # Simple claims
        for i in range(10):
            claim = Claim(
                id=f"baseline_simple_{i:03d}",
                content=f"This is a simple test claim {i} with basic content for baseline testing.",
                confidence=0.85,
                type=[ClaimType.FACT],
                state=ClaimState.VALIDATED,
                tags=["baseline", "simple", "test"]
            )
            test_claims.append(claim)
        
        # Medium complexity claims
        for i in range(10):
            claim = Claim(
                id=f"baseline_medium_{i:03d}",
                content=f"This is a medium complexity test claim {i} with more detailed content including multiple concepts and relationships that would require more processing time and memory to handle effectively in the context window.",
                confidence=0.75,
                type=[ClaimType.CONCEPT, ClaimType.EXAMPLE],
                state=ClaimState.EXPLORE,
                tags=["baseline", "medium", "test", "complex"]
            )
            test_claims.append(claim)
        
        # High complexity claims
        for i in range(5):
            claim = Claim(
                id=f"baseline_complex_{i:03d}",
                content=f"This is a high complexity test claim {i} with extensive detailed content including multiple interconnected concepts, detailed explanations, complex relationships between various ideas, comprehensive analysis of different aspects, and substantial amounts of information that would significantly impact context window usage and processing performance when included in large-scale reasoning tasks.",
                confidence=0.65,
                type=[ClaimType.GOAL, ClaimType.CONCEPT, ClaimType.REFERENCE],
                state=ClaimState.EXPLORE,
                tags=["baseline", "complex", "test", "detailed", "analysis"]
            )
            test_claims.append(claim)
        
        print(f"   + Created {len(test_claims)} test claims")
        print(f"   + Simple claims: 10")
        print(f"   + Medium claims: 10") 
        print(f"   + Complex claims: 5")
        
        # Calculate baseline metrics
        total_content = sum(len(claim.content) for claim in test_claims)
        avg_confidence = sum(claim.confidence for claim in test_claims) / len(test_claims)
        
        print(f"\n3. Baseline Metrics...")
        print(f"   + Total content length: {total_content:,} characters")
        print(f"   + Average claim length: {total_content / len(test_claims):.1f} characters")
        print(f"   + Average confidence: {avg_confidence:.3f}")
        
        print("\n4. Processing Performance Test...")
        
        # Test processing with different context sizes
        context_sizes = [5, 10, 15, 20, 25]
        baseline_results = {}
        
        for size in context_sizes:
            start_time = time.time()
            
            # Select subset of claims
            context_claims = test_claims[:size]
            
            try:
                # Process with LLM (mock processing for baseline)
                result = llm_manager.process_claims(
                    context_claims, 
                    task="analyze",
                    temperature=0.7,
                    max_tokens=1000
                )
                
                processing_time = (time.time() - start_time) * 1000
                
                baseline_results[size] = {
                    "claims_processed": len(context_claims),
                    "processing_time_ms": processing_time,
                    "success": result.success if hasattr(result, 'success') else True,
                    "content_length": sum(len(claim.content) for claim in context_claims)
                }
                
                print(f"   + Context size {size:2d}: {processing_time:.1f}ms - Success")
                
            except Exception as e:
                processing_time = (time.time() - start_time) * 1000
                baseline_results[size] = {
                    "claims_processed": len(context_claims),
                    "processing_time_ms": processing_time,
                    "success": False,
                    "error": str(e),
                    "content_length": sum(len(claim.content) for claim in context_claims)
                }
                print(f"   + Context size {size:2d}: {processing_time:.1f}ms - FAILED: {e}")
        
        print("\n5. Baseline Performance Analysis...")
        
        # Calculate performance trends
        successful_sizes = [size for size, result in baseline_results.items() if result['success']]
        
        if successful_sizes:
            # Processing time trends
            times = [baseline_results[size]['processing_time_ms'] for size in successful_sizes]
            content_lengths = [baseline_results[size]['content_length'] for size in successful_sizes]
            
            avg_time_per_claim = sum(t / baseline_results[size]['claims_processed'] 
                                   for size, t in zip(successful_sizes, times)) / len(successful_sizes)
            avg_time_per_char = sum(t / l for t, l in zip(times, content_lengths)) / len(times)
            
            print(f"   + Successful context sizes: {successful_sizes}")
            print(f"   + Average time per claim: {avg_time_per_claim:.2f}ms")
            print(f"   + Average time per character: {avg_time_per_char*1000:.3f}ms")
            
            # Performance degradation analysis
            if len(times) >= 2:
                time_growth = times[-1] / times[0] if times[0] > 0 else 1
                content_growth = content_lengths[-1] / content_lengths[0] if content_lengths[0] > 0 else 1
                
                print(f"   + Processing time growth: {time_growth:.2f}x")
                print(f"   + Content length growth: {content_growth:.2f}x")
                print(f"   + Performance efficiency: {content_growth/time_growth:.2f}")
        
        print("\n6. Baseline Summary...")
        
        baseline_summary = {
            "test_claims_created": len(test_claims),
            "total_content_characters": total_content,
            "average_claim_length": total_content / len(test_claims),
            "average_confidence": avg_confidence,
            "context_sizes_tested": context_sizes,
            "performance_results": baseline_results,
            "test_timestamp": time.time()
        }
        
        # Save baseline results
        baseline_file = Path("experiment_4_baseline_results.json")
        with open(baseline_file, 'w') as f:
            json.dump(baseline_summary, f, indent=2)
        
        print(f"   + Baseline results saved to: {baseline_file}")
        print(f"   + Test claims: {len(test_claims)}")
        print(f"   + Context sizes tested: {len(context_sizes)}")
        print(f"   + Successful tests: {len(successful_sizes)}")
        
        return baseline_summary
        
    except Exception as e:
        print(f"Baseline measurement failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    success = run_baseline_measurements()
    if success:
        print("\n+ BASELINE MEASUREMENTS COMPLETED SUCCESSFULLY!")
        print("+ Ready for Experiment 4 optimization testing")
    else:
        print("\n- BASELINE MEASUREMENTS FAILED!")
        print("+ Check system configuration and retry")
    
    sys.exit(0 if success else 1)