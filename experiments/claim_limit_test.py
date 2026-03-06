#!/usr/bin/env python3
"""
Test impact of claim retrieval limit on accuracy
Compares limit=10 (current) vs limit=50 (proposed)
"""
import asyncio
import json
from datetime import datetime
from src.endpoint.conjecture_endpoint import ConjectureEndpoint
from src.config.unified_config import UnifiedConfig

async def test_claim_limits():
    """Test different claim retrieval limits on sample problems"""
    
    # Sample problems from different benchmarks
    test_cases = [
        {
            "query": "If Alice has 3 apples and Bob has twice as many, and they share equally with Carol, how many does each get?",
            "expected": "3",
            "type": "math_reasoning"
        },
        {
            "query": "What is the capital of France?",
            "expected": "Paris",
            "type": "factual_recall"
        },
        {
            "query": "In a sequence where each number is the sum of the previous two, starting with 1,1, what is the 7th number?",
            "expected": "13",
            "type": "sequential_reasoning"
        }
    ]
    
    config = UnifiedConfig()
    endpoint = ConjectureEndpoint(config=config)
    
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "limits_tested": [10, 50],
        "test_cases": []
    }
    
    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"Query: {test_case['query'][:60]}...")
        print(f"Type: {test_case['type']}")
        print(f"{'='*60}")
        
        case_results = {
            "query": test_case['query'],
            "type": test_case['type'],
            "expected": test_case['expected'],
            "limits": {}
        }
        
        for limit in [10, 50]:
            print(f"\nTesting with max_claims={limit}...")
            
            try:
                response = await endpoint.evaluate(
                    query=test_case['query'],
                    max_claims=limit,
                    min_confidence=0.5,
                    include_reasoning=True
                )
                
                if response.success:
                    answer = response.data.get("response", "")
                    claims_used = response.data.get("claims_used", 0)
                    
                    case_results["limits"][limit] = {
                        "answer": answer,
                        "claims_used": claims_used,
                        "success": True
                    }
                    
                    print(f"  Claims used: {claims_used}")
                    print(f"  Answer: {answer[:100]}...")
                else:
                    case_results["limits"][limit] = {
                        "error": response.error,
                        "success": False
                    }
                    print(f"  ERROR: {response.error}")
                    
            except Exception as e:
                case_results["limits"][limit] = {
                    "error": str(e),
                    "success": False
                }
                print(f"  EXCEPTION: {e}")
        
        results["test_cases"].append(case_results)
    
    # Save results
    output_file = f"experiments/results/claim_limit_test_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")
    
    # Summary
    print("\nSUMMARY:")
    for case in results["test_cases"]:
        print(f"\n{case['type']}:")
        for limit in [10, 50]:
            if limit in case["limits"] and case["limits"][limit]["success"]:
                claims_used = case["limits"][limit]["claims_used"]
                print(f"  limit={limit}: {claims_used} claims used")

if __name__ == "__main__":
    asyncio.run(test_claim_limits())
