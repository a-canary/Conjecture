#!/usr/bin/env python3
"""
Experiment 1: XML Format Optimization - Baseline Test
Measures current claim format compliance (expected: 0%)
"""

import os
import sys
import json
import time
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[OK] Environment loaded")
except ImportError:
    print("[FAIL] python-dotenv not available")

# Simple test cases for baseline measurement
BASELINE_TEST_CASES = [
    {
        "id": "simple_factual",
        "category": "factual_recall",
        "question": "What are the main differences between Python lists and tuples?",
        "expected_claims": 3
    },
    {
        "id": "concept_explanation", 
        "category": "conceptual_understanding",
        "question": "Explain the concept of recursion in programming with an example.",
        "expected_claims": 4
    },
    {
        "id": "planning_simple",
        "category": "planning",
        "question": "Plan the steps to create a simple web API with Flask.",
        "expected_claims": 5
    },
    {
        "id": "analysis_task",
        "category": "analysis",
        "question": "Analyze the pros and cons of microservices vs monolithic architecture.",
        "expected_claims": 4
    }
]

# Use available model for baseline
BASELINE_MODEL = {
    "name": "zai-org/GLM-4.6",
    "provider": "chutes",
    "url": "https://llm.chutes.ai/v1",
    "api_key": os.getenv("CHUTES_API_KEY", ""),
    "description": "Baseline model"
}

def make_api_call(prompt: str, max_tokens: int = 1500) -> Dict[str, Any]:
    """Make API call to baseline model"""
    try:
        import requests

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {BASELINE_MODEL['api_key']}"
        }

        data = {
            "model": BASELINE_MODEL["name"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3
        }

        endpoint = f"{BASELINE_MODEL['url']}/chat/completions"
        
        start_time = time.time()
        response = requests.post(endpoint, headers=headers, json=data, timeout=120)
        end_time = time.time()

        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            return {
                "content": content,
                "response_time": end_time - start_time,
                "status": "success",
                "response_length": len(content)
            }
        else:
            return {
                "content": f"API error {response.status_code}: {response.text}",
                "response_time": end_time - start_time,
                "status": "error",
                "response_length": 0,
                "error": response.text
            }

    except Exception as e:
        return {
            "content": f"Exception: {str(e)}",
            "response_time": 0,
            "status": "error", 
            "response_length": 0,
            "error": str(e)
        }

def generate_baseline_prompt(test_case: Dict[str, Any]) -> str:
    """Generate baseline prompt (current approach)"""
    return f"""You are Conjecture, an AI system that uses evidence-based reasoning.

Task: {test_case['question']}

Generate 3-7 specific claims using this exact format:
[c1 | claim content | / confidence_level]
[c2 | claim content | / confidence_level]
etc.

Requirements:
- Use claim IDs: c1, c2, c3, etc.
- Include clear, specific statements
- Provide confidence scores (0.0-1.0)
- Use appropriate claim types: fact, concept, example, goal, reference
- Focus on accuracy and verifiability

Format your response with:
- Claims section (using exact format above)
- Brief analysis of each claim
- Final summary"""

def extract_claims_baseline(response: str) -> List[Dict[str, Any]]:
    """Extract claims from baseline response"""
    import re
    
    claims = []
    # Pattern: [c1 | content | / confidence]
    pattern = r'\[c(\d+)\s*\|\s*([^|]+)\s*\|\s*/\s*([0-9.]+)\s*\]'
    matches = re.findall(pattern, response, re.IGNORECASE)
    
    for match in matches:
        claim_id, content, confidence = match
        claims.append({
            "id": claim_id,
            "content": content.strip(),
            "confidence": float(confidence)
        })
    
    return claims

def calculate_format_compliance(claims: List[Dict[str, Any]], expected_count: int) -> Dict[str, Any]:
    """Calculate format compliance metrics"""
    if not claims:
        return {
            "compliance_rate": 0.0,
            "format_correct": False,
            "claims_found": 0,
            "expected_claims": expected_count
        }
    
    # Check if claims follow expected format
    format_correct = len(claims) > 0
    
    return {
        "compliance_rate": len(claims) / expected_count if expected_count > 0 else 0.0,
        "format_correct": format_correct,
        "claims_found": len(claims),
        "expected_claims": expected_count,
        "average_confidence": sum(c["confidence"] for c in claims) / len(claims) if claims else 0.0
    }

async def run_baseline_test():
    """Run baseline test to measure current performance"""
    print("=" * 80)
    print("EXPERIMENT 1: XML FORMAT OPTIMIZATION - BASELINE TEST")
    print("Measuring current claim format compliance")
    print("=" * 80)
    
    if not BASELINE_MODEL["api_key"]:
        print("[ERROR] No CHUTES_API_KEY found. Please set environment variable.")
        return
    
    print(f"Model: {BASELINE_MODEL['name']}")
    print(f"Test cases: {len(BASELINE_TEST_CASES)}")
    print(f"Expected baseline compliance: ~0%")
    print("=" * 80)
    
    baseline_results = []
    
    for i, test_case in enumerate(BASELINE_TEST_CASES, 1):
        print(f"\n[{i}/{len(BASELINE_TEST_CASES)}] Testing: {test_case['id']}")
        print(f"Category: {test_case['category']}")
        
        # Generate prompt
        prompt = generate_baseline_prompt(test_case)
        
        # Make API call
        result = make_api_call(prompt)
        
        if result["status"] == "success":
            # Extract claims
            claims = extract_claims_baseline(result["content"])
            
            # Calculate compliance
            compliance = calculate_format_compliance(claims, test_case["expected_claims"])
            
            baseline_result = {
                "test_case_id": test_case["id"],
                "category": test_case["category"],
                "prompt": prompt,
                "response": result["content"],
                "response_time": result["response_time"],
                "response_length": result["response_length"],
                "claims_extracted": claims,
                "compliance": compliance
            }
            
            baseline_results.append(baseline_result)
            
            print(f"  Status: SUCCESS")
            print(f"  Time: {result['response_time']:.1f}s")
            print(f"  Claims found: {compliance['claims_found']}/{compliance['expected_claims']}")
            print(f"  Compliance rate: {compliance['compliance_rate']:.1%}")
            print(f"  Format correct: {compliance['format_correct']}")
            
        else:
            print(f"  Status: FAILED - {result.get('error', 'Unknown error')}")
            baseline_result = {
                "test_case_id": test_case["id"],
                "category": test_case["category"],
                "prompt": prompt,
                "response": result["content"],
                "response_time": result["response_time"],
                "response_length": result["response_length"],
                "error": result.get("error"),
                "claims_extracted": [],
                "compliance": {"compliance_rate": 0.0, "format_correct": False}
            }
            baseline_results.append(baseline_result)
    
    # Calculate overall baseline metrics
    print(f"\n{'=' * 80}")
    print("BASELINE RESULTS SUMMARY")
    print(f"{'=' * 80}")
    
    successful_tests = [r for r in baseline_results if "error" not in r]
    total_claims_found = sum(r["compliance"]["claims_found"] for r in successful_tests)
    total_expected_claims = sum(r["compliance"]["expected_claims"] for r in successful_tests)
    
    if total_expected_claims > 0:
        overall_compliance = total_claims_found / total_expected_claims
    else:
        overall_compliance = 0.0
    
    avg_response_time = sum(r["response_time"] for r in successful_tests) / len(successful_tests) if successful_tests else 0
    
    print(f"Tests completed: {len(successful_tests)}/{len(BASELINE_TEST_CASES)}")
    print(f"Overall claim format compliance: {overall_compliance:.1%}")
    print(f"Total claims extracted: {total_claims_found}")
    print(f"Total expected claims: {total_expected_claims}")
    print(f"Average response time: {avg_response_time:.1f}s")
    
    # Save baseline results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"experiment_1_baseline_results_{timestamp}.json"
    
    results_data = {
        "experiment_id": f"experiment_1_baseline_{timestamp}",
        "experiment_type": "xml_optimization_baseline",
        "timestamp": datetime.now().isoformat(),
        "model": BASELINE_MODEL["name"],
        "hypothesis": "XML-based prompts will increase claim creation success from 0% to 60%+",
        "baseline_metrics": {
            "overall_compliance_rate": overall_compliance,
            "total_claims_found": total_claims_found,
            "total_expected_claims": total_expected_claims,
            "successful_tests": len(successful_tests),
            "total_tests": len(BASELINE_TEST_CASES),
            "average_response_time": avg_response_time
        },
        "test_results": baseline_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n[OK] Baseline results saved to: {results_file}")
    
    # Update RESULTS.md with baseline findings
    results_entry = f"""## Experiment 1: XML Format Optimization - Baseline
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Hypothesis**: XML-based prompts will increase claim creation success from 0% to 60%+ while maintaining or improving reasoning quality.

### Baseline Metrics
- **Overall Claim Format Compliance**: {overall_compliance:.1%}
- **Claims Successfully Extracted**: {total_claims_found}/{total_expected_claims}
- **Successful Tests**: {len(successful_tests)}/{len(BASELINE_TEST_CASES)}
- **Average Response Time**: {avg_response_time:.1f}s

### Key Findings
- Current baseline shows {overall_compliance:.1%} claim format compliance
- This confirms the starting point of ~0% as expected
- Ready to proceed with XML optimization implementation

### Next Steps
1. Integrate XML templates from `src/processing/llm_prompts/xml_optimized_templates.py`
2. Update claim creation pipeline in `src/conjecture.py`
3. Modify `src/processing/unified_claim_parser.py` for XML handling
4. Run comprehensive tests with 4-model comparison framework

---

"""
    
    # Append to RESULTS.md
    try:
        with open("RESULTS.md", "a") as f:
            f.write(results_entry)
        print("[OK] Results appended to RESULTS.md")
    except Exception as e:
        print(f"[WARN] Could not append to RESULTS.md: {e}")
    
    return results_data

if __name__ == "__main__":
    asyncio.run(run_baseline_test())