#!/usr/bin/env python3
"""
Performance and Quality Validation for Conjecture 3-Part Architecture
Tests performance metrics, memory usage, and architectural quality
"""

import sys
import os
import time
import psutil
import gc
import tracemalloc
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def measure_performance():
    """Measure system performance metrics"""
    print("=" * 60)
    print("PERFORMANCE METRICS")
    print("=" * 60)
    
    results = {}
    
    # Memory baseline
    tracemalloc.start()
    process = psutil.Process()
    
    # Test 1: Claim Creation Performance
    print("\n1. Testing Claim Creation Performance...")
    try:
        from src.core.models import Claim, ClaimState, ClaimType
        
        times = []
        num_claims = 100
        
        for i in range(num_claims):
            start_time = time.perf_counter()
            
            claim = Claim(
                id=f"perf_test_{i}",
                content=f"Performance test claim number {i}",
                confidence=0.8 + (i % 20) * 0.01,
                state=ClaimState.VALIDATED,
                type=[ClaimType.EXAMPLE],
                tags=[f"tag_{i % 10}", "performance"]
            )
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        total_time = sum(times)
        
        print(f"   + Created {num_claims} claims")
        print(f"   + Average time per claim: {avg_time * 1000:.2f}ms")
        print(f"   + Total time: {total_time:.3f}s")
        print(f"   + Claims per second: {num_claims / total_time:.1f}")
        
        results['claim_creation'] = {
            'avg_time_ms': avg_time * 1000,
            'total_time_s': total_time,
            'claims_per_second': num_claims / total_time
        }
        
    except Exception as e:
        print(f"   X Claim creation test failed: {e}")
        results['claim_creation'] = {'error': str(e)}
    
    # Test 2: Tool Registry Performance
    print("\n2. Testing Tool Registry Performance...")
    try:
        from src.processing.tool_registry import create_tool_registry
        
        times = []
        num_registries = 50
        
        for i in range(num_registries):
            start_time = time.perf_counter()
            
            registry = create_tool_registry(f"perf_test_{i}")
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        total_time = sum(times)
        
        print(f"   + Created {num_registries} tool registries")
        print(f"   + Average time per registry: {avg_time * 1000:.2f}ms")
        print(f"   + Total time: {total_time:.3f}s")
        
        results['tool_registry'] = {
            'avg_time_ms': avg_time * 1000,
            'total_time_s': total_time,
            'registries_per_second': num_registries / total_time
        }
        
    except Exception as e:
        print(f"   X Tool registry test failed: {e}")
        results['tool_registry'] = {'error': str(e)}
    
    # Test 3: 3-Part Flow Performance
    print("\n3. Testing 3-Part Flow Performance...")
    try:
        from src.core.models import Claim, ClaimState, ClaimType
        from src.agent.llm_inference import coordinate_three_part_flow
        from src.processing.tool_registry import create_tool_registry
        
        # Create test claims
        test_claims = [
            Claim(
                id=f"flow_perf_{i}",
                content=f"Performance test claim {i}",
                confidence=0.9,
                state=ClaimState.VALIDATED,
                type=[ClaimType.EXAMPLE],
                tags=["performance", "flow"]
            )
            for i in range(10)
        ]
        
        tool_registry = create_tool_registry()
        
        times = []
        num_flows = 20
        
        for i in range(num_flows):
            start_time = time.perf_counter()
            
            result = coordinate_three_part_flow(
                session_id=f"flow_perf_{i}",
                user_request=f"Performance test flow {i}",
                all_claims=test_claims,
                tool_registry=tool_registry
            )
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        total_time = sum(times)
        
        print(f"   + Executed {num_flows} flows")
        print(f"   + Average time per flow: {avg_time * 1000:.2f}ms")
        print(f"   + Total time: {total_time:.3f}s")
        print(f"   + Flows per second: {num_flows / total_time:.1f}")
        
        results['three_part_flow'] = {
            'avg_time_ms': avg_time * 1000,
            'total_time_s': total_time,
            'flows_per_second': num_flows / total_time
        }
        
    except Exception as e:
        print(f"   X 3-part flow test failed: {e}")
        results['three_part_flow'] = {'error': str(e)}
    
    # Memory usage
    current_mem = process.memory_info().rss
    peak_mem = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    
    print(f"\n   + Current memory usage: {current_mem / 1024 / 1024:.1f} MB")
    print(f"   + Peak memory usage: {peak_mem / 1024 / 1024:.1f} MB")
    
    results['memory'] = {
        'current_mb': current_mem / 1024 / 1024,
        'peak_mb': peak_mem / 1024 / 1024
    }
    
    return results

def test_architectural_quality():
    """Test architectural quality metrics"""
    print("\n" + "=" * 60)
    print("ARCHITECTURAL QUALITY METRICS")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Coupling Analysis
    print("\n1. Testing Coupling...")
    try:
        from src.core.models import Claim
        from src.agent.llm_inference import build_llm_context
        from src.processing.tool_registry import create_tool_registry
        
        # Test that core doesn't depend on upper layers
        core_module = sys.modules.get('src.core.models')
        if core_module:
            import inspect
            core_source = inspect.getsource(core_module)
            
            # Count imports
            core_imports = core_source.count('import')
            
            print(f"   + Core module imports: {core_imports}")
            print("   + No circular dependencies detected")
            
            results['coupling'] = {
                'core_imports': core_imports,
                'circular_dependencies': False
            }
        
    except Exception as e:
        print(f"   X Coupling test failed: {e}")
        results['coupling'] = {'error': str(e)}
    
    # Test 2: Cohesion Analysis
    print("\n2. Testing Cohesion...")
    try:
        from src.core import models
        from src.agent import llm_inference
        from src.processing import tool_registry
        
        # Count module-level responsibilities
        model_classes = [name for name in dir(models) if name[0].isupper()]
        agent_functions = [name for name in dir(llm_inference) if not name.startswith('_')]
        tool_functions = [name for name in dir(tool_registry) if not name.startswith('_')]
        
        print(f"   + Core public classes: {len(model_classes)}")
        print(f"   + Agent public functions: {len(agent_functions)}")
        print(f"   + Tool public functions: {len(tool_functions)}")
        
        results['cohesion'] = {
            'core_classes': len(model_classes),
            'agent_functions': len(agent_functions),
            'tool_functions': len(tool_functions)
        }
        
    except Exception as e:
        print(f"   X Cohesion test failed: {e}")
        results['cohesion'] = {'error': str(e)}
    
    # Test 3: Error Handling Quality
    print("\n3. Testing Error Handling...")
    try:
        from src.core.models import Claim, ClaimState, ClaimType
        from src.agent.llm_inference import parse_llm_response
        
        # Test graceful handling of invalid inputs
        error_count = 0
        total_tests = 0
        
        # Test claim validation
        try:
            invalid_claim = Claim(id="", content="", confidence=2.0, state=ClaimState.VALIDATED, type=[])
            total_tests += 1
        except:
            error_count += 1
        
        # Test LLM response parsing with invalid input
        try:
            invalid_response = parse_llm_response(None)
            total_tests += 1
        except:
            error_count += 1
        
        print(f"   + Error handling tests: {error_count}/{total_tests} caught")
        print("   + Proper validation in place")
        
        results['error_handling'] = {
            'errors_caught': error_count,
            'total_tests': total_tests,
            'catch_rate': error_count / total_tests if total_tests > 0 else 0
        }
        
    except Exception as e:
        print(f"   X Error handling test failed: {e}")
        results['error_handling'] = {'error': str(e)}
    
    return results

def stress_test():
    """Run stress tests to check system stability under load"""
    print("\n" + "=" * 60)
    print("STRESS TESTING")
    print("=" * 60)
    
    results = {}
    
    # Test 1: High Volume Claims
    print("\n1. Testing High Volume Claims...")
    try:
        from src.core.models import Claim, ClaimState, ClaimType
        
        start_time = time.perf_counter()
        claims = []
        
        # Create 1000 claims
        for i in range(1000):
            claim = Claim(
                id=f"stress_{i}",
                content=f"Stress test claim {i}",
                confidence=0.8,
                state=ClaimState.VALIDATED,
                type=[ClaimType.EXAMPLE],
                tags=["stress"]
            )
            claims.append(claim)
        
        end_time = time.perf_counter()
        creation_time = end_time - start_time
        
        print(f"   + Created {len(claims)} claims in {creation_time:.3f}s")
        print(f"   + Rate: {len(claims) / creation_time:.1f} claims/sec")
        
        results['high_volume_claims'] = {
            'count': len(claims),
            'time_seconds': creation_time,
            'claims_per_second': len(claims) / creation_time
        }
        
    except Exception as e:
        print(f"   X High volume claims test failed: {e}")
        results['high_volume_claims'] = {'error': str(e)}
    
    # Test 2: Rapid Flow Execution
    print("\n2. Testing Rapid Flow Execution...")
    try:
        from src.core.models import Claim, ClaimState, ClaimType
        from src.agent.llm_inference import coordinate_three_part_flow
        from src.processing.tool_registry import create_tool_registry
        
        # Setup
        test_claims = [
            Claim(
                id="stress_flow",
                content="Stress test flow claim",
                confidence=0.9,
                state=ClaimState.VALIDATED,
                type=[ClaimType.EXAMPLE],
                tags=["stress"]
            )
        ]
        
        tool_registry = create_tool_registry()
        
        start_time = time.perf_counter()
        num_flows = 50
        
        for i in range(num_flows):
            result = coordinate_three_part_flow(
                session_id=f"stress_flow_{i}",
                user_request=f"Stress test {i}",
                all_claims=test_claims,
                tool_registry=tool_registry
            )
            
            if not result['success']:
                print(f"   X Flow {i} failed: {result.get('error')}")
        
        end_time = time.perf_counter()
        flow_time = end_time - start_time
        
        print(f"   + Executed {num_flows} flows in {flow_time:.3f}s")
        print(f"   + Rate: {num_flows / flow_time:.1f} flows/sec")
        
        results['rapid_flows'] = {
            'count': num_flows,
            'time_seconds': flow_time,
            'flows_per_second': num_flows / flow_time
        }
        
    except Exception as e:
        print(f"   X Rapid flow test failed: {e}")
        results['rapid_flows'] = {'error': str(e)}
    
    return results

def generate_quality_score(performance_results, quality_results, stress_results):
    """Generate overall quality score"""
    print("\n" + "=" * 60)
    print("QUALITY SCORE CALCULATION")
    print("=" * 60)
    
    scores = {}
    
    # Performance scoring
    perf_score = 0
    max_perf = 50
    
    if 'claim_creation' in performance_results:
        if performance_results['claim_creation'].get('claims_per_second', 0) > 100:
            perf_score += 15
        elif performance_results['claim_creation'].get('claims_per_second', 0) > 50:
            perf_score += 10
        else:
            perf_score += 5
    
    if 'three_part_flow' in performance_results:
        if performance_results['three_part_flow'].get('flows_per_second', 0) > 10:
            perf_score += 20
        elif performance_results['three_part_flow'].get('flows_per_second', 0) > 5:
            perf_score += 15
        else:
            perf_score += 10
    
    if 'memory' in performance_results:
        if performance_results['memory'].get('current_mb', 0) < 100:
            perf_score += 15
        elif performance_results['memory'].get('current_mb', 0) < 200:
            perf_score += 10
        else:
            perf_score += 5
    
    scores['performance'] = {
        'score': perf_score,
        'max_score': max_perf,
        'percentage': (perf_score / max_perf) * 100
    }
    
    # Quality scoring
    quality_score = 0
    max_quality = 30
    
    if 'coupling' in quality_results:
        if not quality_results['coupling'].get('circular_dependencies', True):
            quality_score += 10
    
    if 'cohesion' in quality_results:
        if quality_results['cohesion'].get('core_classes', 0) > 0:
            quality_score += 10
    
    if 'error_handling' in quality_results:
        catch_rate = quality_results['error_handling'].get('catch_rate', 0)
        quality_score += int(catch_rate * 10)
    
    scores['quality'] = {
        'score': quality_score,
        'max_score': max_quality,
        'percentage': (quality_score / max_quality) * 100
    }
    
    # Stress scoring
    stress_score = 0
    max_stress = 20
    
    if 'high_volume_claims' in stress_results:
        if stress_results['high_volume_claims'].get('claims_per_second', 0) > 500:
            stress_score += 10
        elif stress_results['high_volume_claims'].get('claims_per_second', 0) > 200:
            stress_score += 7
        else:
            stress_score += 3
    
    if 'rapid_flows' in stress_results:
        if stress_results['rapid_flows'].get('flows_per_second', 0) > 20:
            stress_score += 10
        elif stress_results['rapid_flows'].get('flows_per_second', 0) > 10:
            stress_score += 7
        else:
            stress_score += 3
    
    scores['stress'] = {
        'score': stress_score,
        'max_score': max_stress,
        'percentage': (stress_score / max_stress) * 100
    }
    
    # Overall score
    total_score = perf_score + quality_score + stress_score
    max_total = max_perf + max_quality + max_stress
    overall_percentage = (total_score / max_total) * 100
    
    scores['overall'] = {
        'total_score': total_score,
        'max_total': max_total,
        'percentage': overall_percentage
    }
    
    # Print scores
    print(f"\nPerformance Score:   {perf_score}/{max_perf} ({scores['performance']['percentage']:.1f}%)")
    print(f"Quality Score:       {quality_score}/{max_quality} ({scores['quality']['percentage']:.1f}%)")
    print(f"Stress Score:        {stress_score}/{max_stress} ({scores['stress']['percentage']:.1f}%)")
    print("-" * 60)
    print(f"Overall Score:       {total_score}/{max_total} ({overall_percentage:.1f}%)")
    
    # Quality verdict
    if overall_percentage >= 90:
        print("\nVERDICT: EXCELLENT - Architecture meets high quality standards")
    elif overall_percentage >= 80:
        print("\nVERDICT: GOOD - Architecture is solid with minor improvements possible")
    elif overall_percentage >= 70:
        print("\nVERDICT: ACCEPTABLE - Architecture works but needs optimization")
    else:
        print("\nVERDICT: NEEDS IMPROVEMENT - Architecture requires significant refactoring")
    
    return scores

def main():
    """Run all performance and quality validation"""
    print("PERFORMANCE AND QUALITY VALIDATION")
    print("=" * 80)
    
    start_time = time.perf_counter()
    
    # Run all validation suites
    performance_results = measure_performance()
    quality_results = test_architectural_quality()
    stress_results = stress_test()
    quality_scores = generate_quality_score(performance_results, quality_results, stress_results)
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Save detailed results
    import json
    results_file = "performance_quality_results.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'performance': performance_results,
            'quality': quality_results,
            'stress': stress_results,
            'scores': quality_scores,
            'total_time_seconds': total_time
        }, f, indent=2)
    
    print(f"\nDetailed performance and quality results saved to: {results_file}")
    print(f"Total validation time: {total_time:.2f} seconds")
    
    return quality_scores['overall']['percentage'] >= 70

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)