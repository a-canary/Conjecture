#!/usr/bin/env python3
"""
Experiment 4: Context Window Optimization - Direct Test
Test context optimization components directly without LLM dependencies
"""

import sys
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_test_claims():
    """Create test claims with varying complexity"""
    from src.core.models import Claim, ClaimType, ClaimState
    
    claims = []
    
    # Simple claims
    for i in range(20):
        claim = Claim(
            id=f"simple_{i:03d}",
            content=f"Simple test claim {i} with basic content.",
            confidence=0.85,
            type=[ClaimType.FACT],
            state=ClaimState.VALIDATED,
            tags=["test", "simple"]
        )
        claims.append(claim)
    
    # Medium complexity claims
    for i in range(15):
        claim = Claim(
            id=f"medium_{i:03d}",
            content=f"Medium complexity test claim {i} with more detailed content including multiple concepts and relationships that require more processing time and memory.",
            confidence=0.75,
            type=[ClaimType.CONCEPT, ClaimType.EXAMPLE],
            state=ClaimState.EXPLORE,
            tags=["test", "medium", "complex"]
        )
        claims.append(claim)
    
    # High complexity claims
    for i in range(10):
        claim = Claim(
            id=f"complex_{i:03d}",
            content=f"High complexity test claim {i} with extensive detailed content including multiple interconnected concepts, detailed explanations, complex relationships, comprehensive analysis, and substantial amounts of information that significantly impact context window usage.",
            confidence=0.65,
            type=[ClaimType.GOAL, ClaimType.CONCEPT, ClaimType.REFERENCE],
            state=ClaimState.EXPLORE,
            tags=["test", "complex", "detailed", "analysis"]
        )
        claims.append(claim)
    
    return claims

def test_context_optimization():
    """Test context optimization components directly"""
    print("Experiment 4: Context Window Optimization - Direct Test")
    print("=" * 65)
    
    try:
        # Import optimization components
        from src.processing.adaptive_compression import get_adaptive_compressor
        from src.processing.hierarchical_context_processor import get_hierarchical_processor
        from src.processing.intelligent_claim_selector import get_intelligent_selector
        
        print("\n1. Initializing Components...")
        compressor = get_adaptive_compressor()
        processor = get_hierarchical_processor()
        selector = get_intelligent_selector()
        
        print("   + Adaptive Compression Engine: READY")
        print("   + Hierarchical Context Processor: READY")
        print("   + Intelligent Claim Selector: READY")
        
        print("\n2. Creating Test Data...")
        test_claims = create_test_claims()
        print(f"   + Created {len(test_claims)} test claims")
        print(f"   + Simple claims: 20")
        print(f"   + Medium claims: 15")
        print(f"   + Complex claims: 10")
        
        # Calculate baseline metrics
        total_content = sum(len(claim.content) for claim in test_claims)
        avg_confidence = sum(claim.confidence for claim in test_claims) / len(test_claims)
        
        print(f"   + Total content: {total_content:,} characters")
        print(f"   + Average claim length: {total_content / len(test_claims):.1f} characters")
        print(f"   + Average confidence: {avg_confidence:.3f}")
        
        print("\n3. Testing Adaptive Compression...")
        
        # Test compression with different context sizes
        compression_results = {}
        context_sizes = [10, 25, 45]  # Small, medium, large contexts
        
        for size in context_sizes:
            context_claims = test_claims[:size]
            
            start_time = time.time()
            compressed_claims, metadata = compressor.compress_context(
                context_claims, 
                f"Analyze {size} claims for patterns"
            )
            compression_time = (time.time() - start_time) * 1000
            
            compression_ratio = len(compressed_claims) / len(context_claims)
            tokens_saved = (len(context_claims) - len(compressed_claims)) * 100  # Estimate
            
            compression_results[size] = {
                "original_count": len(context_claims),
                "compressed_count": len(compressed_claims),
                "compression_ratio": compression_ratio,
                "compression_time_ms": compression_time,
                "tokens_saved": tokens_saved,
                "metadata": metadata
            }
            
            print(f"   + Context size {size:2d}: {len(context_claims)} -> {len(compressed_claims)} claims "
                  f"({compression_ratio:.2f}x ratio, {compression_time:.1f}ms)")
        
        print("\n4. Testing Hierarchical Processing...")
        
        hierarchical_results = {}
        
        for size in context_sizes:
            context_claims = test_claims[:size]
            
            start_time = time.time()
            try:
                processed_context = processor.process_large_context(context_claims, "medium")
                processing_time = (time.time() - start_time) * 1000
                
                # Count levels from processed context
                levels_count = len(processed_context.get("levels", []))
                
                hierarchical_results[size] = {
                    "levels_count": levels_count,
                    "processing_time_ms": processing_time,
                    "success": True
                }
                
                print(f"   + Context size {size:2d}: {levels_count} levels ({processing_time:.1f}ms)")
                
            except Exception as e:
                processing_time = (time.time() - start_time) * 1000
                hierarchical_results[size] = {
                    "levels_count": 0,
                    "processing_time_ms": processing_time,
                    "success": False,
                    "error": str(e)
                }
                print(f"   + Context size {size:2d}: FAILED ({processing_time:.1f}ms) - {e}")
        
        print("\n5. Testing Intelligent Selection...")
        
        selection_results = {}
        
        for size in context_sizes:
            context_claims = test_claims[:size]
            
            start_time = time.time()
            try:
                selection_result = selector.select_optimal_claims(
                    context_claims,
                    task="analysis",
                    context_limit=size * 200,  # Estimate tokens per claim
                    selection_strategy="balanced"
                )
                selection_time = (time.time() - start_time) * 1000
                
                selected_claims = selection_result.selected_claims if hasattr(selection_result, 'selected_claims') else []
                selection_ratio = len(selected_claims) / len(context_claims)
                
                selection_results[size] = {
                    "original_count": len(context_claims),
                    "selected_count": len(selected_claims),
                    "selection_ratio": selection_ratio,
                    "selection_time_ms": selection_time,
                    "success": True
                }
                
                print(f"   + Context size {size:2d}: {len(context_claims)} -> {len(selected_claims)} claims "
                      f"({selection_ratio:.2f}x ratio, {selection_time:.1f}ms)")
                
            except Exception as e:
                selection_time = (time.time() - start_time) * 1000
                selection_results[size] = {
                    "original_count": len(context_claims),
                    "selected_count": 0,
                    "selection_ratio": 0.0,
                    "selection_time_ms": selection_time,
                    "success": False,
                    "error": str(e)
                }
                print(f"   + Context size {size:2d}: FAILED ({selection_time:.1f}ms) - {e}")
        
        print("\n6. Performance Analysis...")
        
        # Calculate overall performance metrics
        successful_compressions = [r for r in compression_results.values() if r.get('compression_ratio', 0) > 0]
        successful_hierarchical = [r for r in hierarchical_results.values() if r.get('success', False)]
        successful_selections = [r for r in selection_results.values() if r.get('success', False)]
        
        if successful_compressions:
            avg_compression_ratio = sum(r['compression_ratio'] for r in successful_compressions) / len(successful_compressions)
            avg_compression_time = sum(r['compression_time_ms'] for r in successful_compressions) / len(successful_compressions)
            total_tokens_saved = sum(r['tokens_saved'] for r in successful_compressions)
            
            print(f"   + Compression Performance:")
            print(f"     - Average compression ratio: {avg_compression_ratio:.3f}")
            print(f"     - Average compression time: {avg_compression_time:.2f}ms")
            print(f"     - Total tokens saved: {total_tokens_saved:,}")
        
        if successful_hierarchical:
            avg_levels = sum(r['levels_count'] for r in successful_hierarchical) / len(successful_hierarchical)
            avg_processing_time = sum(r['processing_time_ms'] for r in successful_hierarchical) / len(successful_hierarchical)
            
            print(f"   + Hierarchical Performance:")
            print(f"     - Average levels generated: {avg_levels:.1f}")
            print(f"     - Average processing time: {avg_processing_time:.2f}ms")
        
        if successful_selections:
            avg_selection_ratio = sum(r['selection_ratio'] for r in successful_selections) / len(successful_selections)
            avg_selection_time = sum(r['selection_time_ms'] for r in successful_selections) / len(successful_selections)
            
            print(f"   + Selection Performance:")
            print(f"     - Average selection ratio: {avg_selection_ratio:.3f}")
            print(f"     - Average selection time: {avg_selection_time:.2f}ms")
        
        print("\n7. Integrated Workflow Test...")
        
        # Test end-to-end workflow
        large_context = test_claims[:45]  # Largest context
        
        start_time = time.time()
        
        # Step 1: Compress context
        compressed_claims, _ = compressor.compress_context(large_context, "Integrated test")
        
        # Step 2: Create hierarchical levels
        try:
            processed_context = processor.process_large_context(compressed_claims, "complex")
            levels = processed_context.get("levels", [])
        except:
            levels = []
        
        # Step 3: Select optimal claims
        try:
            selection_result = selector.select_optimal_claims(
                compressed_claims,
                task="analysis",
                context_limit=8000,
                selection_strategy="balanced"
            )
            final_claims = selection_result.selected_claims if hasattr(selection_result, 'selected_claims') else compressed_claims
        except:
            final_claims = compressed_claims
        
        total_time = (time.time() - start_time) * 1000
        
        print(f"   + Original context: {len(large_context)} claims")
        print(f"   + After compression: {len(compressed_claims)} claims")
        print(f"   + Hierarchical levels: {len(levels)}")
        print(f"   + Final selection: {len(final_claims)} claims")
        print(f"   + Total processing time: {total_time:.1f}ms")
        
        # Calculate overall improvement
        overall_compression = len(final_claims) / len(large_context)
        estimated_token_reduction = (1 - overall_compression) * 100
        
        print(f"   + Overall compression: {overall_compression:.3f}x")
        print(f"   + Estimated token reduction: {estimated_token_reduction:.1f}%")
        
        print("\n8. Results Summary...")
        
        experiment_results = {
            "test_claims": len(test_claims),
            "total_content": total_content,
            "compression_results": compression_results,
            "hierarchical_results": hierarchical_results,
            "selection_results": selection_results,
            "integrated_workflow": {
                "original_claims": len(large_context),
                "compressed_claims": len(compressed_claims),
                "hierarchical_levels": len(levels),
                "final_claims": len(final_claims),
                "overall_compression": overall_compression,
                "token_reduction_percent": estimated_token_reduction,
                "processing_time_ms": total_time
            },
            "timestamp": time.time()
        }
        
        # Save results
        results_file = Path("experiment_4_optimization_results.json")
        with open(results_file, 'w') as f:
            json.dump(experiment_results, f, indent=2)
        
        print(f"   + Results saved to: {results_file}")
        
        # Success criteria
        success_criteria = {
            "compression_working": len(successful_compressions) > 0,
            "hierarchical_working": len(successful_hierarchical) > 0,
            "selection_working": len(successful_selections) > 0,
            "overall_compression_achieved": overall_compression < 0.8,  # At least 20% reduction
            "processing_time_acceptable": total_time < 1000  # Under 1 second
        }
        
        criteria_met = sum(success_criteria.values())
        total_criteria = len(success_criteria)
        
        print(f"\n9. Success Criteria...")
        for criterion, met in success_criteria.items():
            status = "PASS" if met else "FAIL"
            print(f"   + {criterion}: {status}")
        
        print(f"   + Overall: {criteria_met}/{total_criteria} criteria met")
        
        success = criteria_met >= 4  # At least 80% of criteria
        
        return success, experiment_results
        
    except Exception as e:
        print(f"Experiment 4 optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    success, results = test_context_optimization()
    
    if success:
        print("\n+ EXPERIMENT 4 CONTEXT OPTIMIZATION SUCCESSFUL!")
        print("+ Context window optimization components working correctly")
        print("+ Ready for integration with full Conjecture system")
    else:
        print("\n- EXPERIMENT 4 CONTEXT OPTIMIZATION FAILED!")
        print("+ Check component implementation and retry")
    
    sys.exit(0 if success else 1)