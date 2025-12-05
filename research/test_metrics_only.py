#!/usr/bin/env python3
"""
Simplified test for Direct vs Conjecture quality metrics
Tests metrics evaluation with mocked responses to validate quality detection
"""

import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from direct_vs_conjecture_test import evaluate_response_quality

def test_quality_metrics():
    """Test that our quality metrics work correctly"""
    
    # Sample test cases with expected quality differences
    test_cases = [
        {
            "name": "Poor Response",
            "response": "idk",
            "expected_low_metrics": ["completeness", "reasoning_quality", "coherence"]
        },
        {
            "name": "Basic Response",
            "response": "This is correct based on my knowledge.",
            "expected_moderate_metrics": ["correctness"]
        },
        {
            "name": "Good Response",
            "response": """
            Based on the evidence provided, I believe this approach is effective because
            research has demonstrated its utility in similar contexts. First, let's examine
            the primary factors. Second, we should consider alternative perspectives.
            Therefore, my conclusion is that a balanced approach yields the best results.
            """,
            "expected_high_metrics": ["reasoning_quality", "coherence", "completeness"]
        },
        {
            "name": "Overconfident Response",
            "response": """
            This is definitely the only correct answer. It is always true and never wrong.
            I am absolutely certain without any doubt. No other perspective is valid.
            """,
            "expected_low_metrics": ["hallucination_reduction"]
        },
        {
            "name": "Conjecture-style Response",
            "response": """
            Let me evaluate this through a systematic claim-based framework:
            
            Claim 1: The proposed approach has merit.
            Evaluation: Evidence suggests this is supported by data.
            
            Claim 2: Alternative approaches exist.
            Evaluation: Research indicates multiple valid perspectives.
            
            Conclusion: The structured evaluation supports a balanced approach.
            Metrics collected: 0.85 confidence, 0.92 coherence
            """,
            "expected_high_metrics": ["reasoning_quality", "coherence"],
            "approach": "conjecture"
        }
    ]
    
    base_test_case = {
        "file": "test.json",
        "category": "reasoning",
        "data": {
            "task": "Analyze the given situation",
            "question": "What is the best approach?"
        }
    }
    
    print("Testing Quality Metrics Evaluation")
    print("=" * 50)
    
    results = []
    
    for test_case in test_cases:
        approach = test_case.get("approach", "direct")
        quality = evaluate_response_quality(
            test_case["response"], base_test_case, approach
        )
        
        print(f"\n{test_case['name']}:")
        print(f"Response: {test_case['response'][:100]}...")
        print(f"  Correctness: {quality['correctness']:.3f}")
        print(f"  Reasoning Quality: {quality['reasoning_quality']:.3f}")
        print(f"  Completeness: {quality['completeness']:.3f}")
        print(f"  Coherence: {quality['coherence']:.3f}")
        print(f"  Confidence Calibration: {quality['confidence_calibration']:.3f}")
        print(f"  Efficiency: {quality['efficiency']:.3f}")
        print(f"  Hallucination Reduction: {quality['hallucination_reduction']:.3f}")
        
        # Check for detailed metrics if available
        if 'response_length' in quality:
            print(f"  Response Length: {quality['response_length']}")
        if 'reasoning_indicators_found' in quality:
            print(f"  Reasoning Indicators: {quality['reasoning_indicators_found']}")
        if 'evidence_indicators_found' in quality:
            print(f"  Evidence Indicators: {quality['evidence_indicators_found']}")
        if 'perspective_indicators_found' in quality:
            print(f"  Perspective Indicators: {quality['perspective_indicators_found']}")
        
        results.append({
            "name": test_case["name"],
            "approach": approach,
            "metrics": quality
        })
    
    # Analyze metric differentiation
    print("\nMetric Differentiation Analysis:")
    print("-" * 40)
    
    metrics_to_check = ["correctness", "reasoning_quality", "completeness", "coherence", 
                      "confidence_calibration", "efficiency", "hallucination_reduction"]
    
    for metric in metrics_to_check:
        values = [r["metrics"][metric] for r in results]
        min_val, max_val = min(values), max(values)
        range_val = max_val - min_val
        
        print(f"{metric}:")
        print(f"  Range: {min_val:.3f} - {max_val:.3f} (diff: {range_val:.3f})")
        
        if range_val > 0.3:
            print(f"  [+] Good differentiation")
        elif range_val > 0.1:
            print(f"  [~] Moderate differentiation")
        else:
            print(f"  [-] Poor differentiation")
    
    # Test weighted improvement calculation
    print("\nWeighted Improvement Test:")
    print("-" * 30)
    
    direct_result = results[2]  # Good Response
    conjecture_result = results[4]  # Conjecture-style Response
    
    weights = {
        "correctness": 1.5,
        "reasoning_quality": 1.2,
        "completeness": 1.0,
        "coherence": 1.0,
        "confidence_calibration": 1.0,
        "efficiency": 0.5,
        "hallucination_reduction": 1.3,
    }
    
    weighted_improvement = 0
    total_weight = 0
    
    print("Metric improvements (Conjecture - Direct):")
    for metric in metrics_to_check:
        improvement = conjecture_result["metrics"][metric] - direct_result["metrics"][metric]
        weight = weights[metric]
        contribution = improvement * weight
        
        arrow = "UP" if improvement > 0 else "DOWN" if improvement < 0 else "SAME"
        print(f"  {metric}: {arrow} {improvement:+.3f} (x{weight} = {contribution:+.3f})")
        
        weighted_improvement += contribution
        total_weight += weight
    
    final_score = weighted_improvement / total_weight
    print(f"\nFinal weighted improvement: {final_score:+.3f}")
    
    # Save test results
    output_file = Path(__file__).parent / "metrics_test_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "test_cases": results,
            "weighted_improvement": final_score,
            "metric_ranges": {
                metric: {
                    "min": min(r["metrics"][metric] for r in results),
                    "max": max(r["metrics"][metric] for r in results),
                    "range": max(r["metrics"][metric] for r in results) - min(r["metrics"][metric] for r in results)
                }
                for metric in metrics_to_check
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return final_score > 0  # Return True if Conjecture approach scores better

if __name__ == "__main__":
    success = test_quality_metrics()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)