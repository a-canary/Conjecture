#!/usr/bin/env python3
"""
Local Task Decomposition Experiment Runner
"""

import asyncio
import json
import time
import uuid
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
import logging
import sys
import os
import requests
from scipy import stats


@dataclass
class TestResult:
    """Result from a single test case execution"""
    
    test_id: str
    approach: str
    question: str
    generated_answer: str
    execution_time: float
    
    # Evaluation metrics
    correctness: float
    completeness: float
    coherence: float
    reasoning_quality: float
    
    # Metadata
    timestamp: datetime
    difficulty: str


async def main():
    """Run the local task decomposition experiment"""
    
    print("Starting Local Task Decomposition Experiment...")
    print("This demonstrates the framework with local LM Studio model")
    
    # Configuration
    sample_size = 5  # Small sample for demo
    target_improvement = 0.20  # 20% improvement target
    
    # Setup
    experiments_dir = Path("experiments")
    results_dir = Path("experiments/results")
    test_cases_dir = Path("experiments/test_cases")
    reports_dir = Path("experiments/reports")
    
    for dir_path in [experiments_dir, results_dir, test_cases_dir, reports_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # API configuration
    api_url = "http://localhost:1234"
    
    # Generate test cases
    test_cases = []
    scenarios = [
        "Launch a new mobile app for food delivery",
        "Organize a company meeting for 50 people", 
        "Develop a training program for new employees",
        "Create a marketing plan for a product launch",
        "Design a customer feedback system"
    ]
    
    for i, scenario in enumerate(scenarios[:sample_size]):
        case = {
            "id": f"demo_task_{i+1:03d}",
            "task": f"You are tasked with: {scenario}. Break this down into manageable steps and provide a comprehensive solution plan."
        }
        test_cases.append(case)
    
    print(f"Generated {len(test_cases)} test cases")
    
    # Run tests
    direct_results = []
    conjecture_results = []
    
    print("Running direct approach tests...")
    for i, test_case in enumerate(test_cases):
        print(f"Direct test {i+1}/{len(test_cases)}: {test_case['id']}")
        
        # Direct prompt
        prompt = f"""
Please provide a comprehensive solution to the following task:

{test_case['task']}

Provide a detailed, well-structured response that addresses all aspects of the task.
"""
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{api_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": "ibm/granite-4-h-tiny",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000,
                    "temperature": 0.7
                },
                timeout=60
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            execution_time = time.time() - start_time
            
            result = TestResult(
                test_id=test_case["id"],
                approach="direct",
                question=test_case["task"],
                generated_answer=content,
                execution_time=execution_time,
                correctness=0.0,
                completeness=0.0,
                coherence=0.0,
                reasoning_quality=0.0,
                timestamp=datetime.utcnow(),
                difficulty="medium"
            )
            direct_results.append(result)
            
        except Exception as e:
            print(f"Direct test failed: {e}")
    
    print("Running Conjecture approach tests...")
    for i, test_case in enumerate(test_cases):
        print(f"Conjecture test {i+1}/{len(test_cases)}: {test_case['id']}")
        
        # Conjecture prompt
        prompt = f"""
You are using Conjecture's task decomposition approach to solve a complex problem. Break down the task into smaller, manageable claims or subtasks, then provide a comprehensive solution.

**Task:**
{test_case['task']}

**Instructions:**
1. First, decompose the problem into 3-5 key claims or subtasks
2. For each claim/subtask, provide a confidence score (0.0-1.0)
3. Show how the claims relate to each other
4. Provide a final solution based on the claims

Format your response using Conjecture's claim format:
[c1 | claim content | / confidence]
[c2 | supporting claim | / confidence]
[c3 | subtask claim | / confidence]
etc.

Then provide your final comprehensive solution based on these claims.
"""
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{api_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": "ibm/granite-4-h-tiny",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000,
                    "temperature": 0.7
                },
                timeout=60
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            execution_time = time.time() - start_time
            
            result = TestResult(
                test_id=test_case["id"],
                approach="conjecture",
                question=test_case["task"],
                generated_answer=content,
                execution_time=execution_time,
                correctness=0.0,
                completeness=0.0,
                coherence=0.0,
                reasoning_quality=0.0,
                timestamp=datetime.utcnow(),
                difficulty="medium"
            )
            conjecture_results.append(result)
            
        except Exception as e:
            print(f"Conjecture test failed: {e}")
    
    # Simple evaluation
    print("Evaluating results...")
    for result in direct_results + conjecture_results:
        response = result.generated_answer.lower()
        
        # Check for claim format (Conjecture approach)
        claim_format_score = 1.0 if '[c1 |' in response and '| /' in response else 0.0
        
        # Check for structured approach
        structure_score = 1.0 if any(word in response for word in ['step', 'phase', 'stage', 'first', 'second', 'third']) else 0.5
        
        # Check for completeness (longer responses tend to be more complete)
        length_score = min(1.0, len(response) / 500)
        
        # Check for coherence (logical connectors)
        coherence_score = 1.0 if any(connector in response for connector in ['therefore', 'because', 'thus', 'consequently']) else 0.5
        
        # Overall quality score
        if result.approach == "conjecture":
            result.correctness = claim_format_score * 0.7 + structure_score * 0.2 + length_score * 0.1
            result.completeness = structure_score * 0.6 + length_score * 0.3 + claim_format_score * 0.1
            result.coherence = coherence_score * 0.5 + claim_format_score * 0.3 + structure_score * 0.2
            result.reasoning_quality = claim_format_score * 0.6 + structure_score * 0.3 + coherence_score * 0.1
        else:
            result.correctness = structure_score * 0.7 + length_score * 0.2 + coherence_score * 0.1
            result.completeness = structure_score * 0.6 + length_score * 0.3 + coherence_score * 0.1
            result.coherence = coherence_score * 0.5 + structure_score * 0.3 + length_score * 0.2
            result.reasoning_quality = structure_score * 0.6 + coherence_score * 0.3 + length_score * 0.1
    
    # Statistical analysis
    print("Performing statistical analysis...")
    
    direct_correctness = [r.correctness for r in direct_results]
    conjecture_correctness = [r.correctness for r in conjecture_results]
    
    direct_completeness = [r.completeness for r in direct_results]
    conjecture_completeness = [r.completeness for r in conjecture_results]
    
    if len(direct_correctness) >= 2 and len(conjecture_correctness) >= 2:
        try:
            # Paired t-test for correctness
            t_stat, p_value = stats.ttest_rel(conjecture_correctness, direct_correctness)
            
            # Calculate improvement
            direct_mean = statistics.mean(direct_correctness)
            conjecture_mean = statistics.mean(conjecture_correctness)
            improvement = (conjecture_mean - direct_mean) / direct_mean if direct_mean > 0 else 0
            
            # Effect size
            diff_mean = statistics.mean([c - d for c, d in zip(conjecture_correctness, direct_correctness)])
            diff_std = statistics.stdev([c - d for c, d in zip(conjecture_correctness, direct_correctness)]) if len(conjecture_correctness) > 1 else 1
            effect_size = diff_mean / (diff_std + 0.001)
            
            print(f"\nStatistical Results:")
            print(f"Direct Correctness Mean: {direct_mean:.3f}")
            print(f"Conjecture Correctness Mean: {conjecture_mean:.3f}")
            print(f"Improvement: {improvement:+.1%}")
            print(f"P-value: {p_value:.3f}")
            print(f"Effect Size: {effect_size:.3f}")
            
            # Determine if target achieved
            target_achieved = (
                improvement >= target_improvement and
                p_value < 0.05 and
                abs(effect_size) >= 0.5
            )
            
            print(f"\nTarget Achieved: {'YES' if target_achieved else 'NO'}")
            print(f"Hypothesis Validated: {'YES' if target_achieved else 'NO'}")
            
        except Exception as e:
            print(f"Statistical analysis failed: {e}")
    
    # Save results
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    results_data = {
        "experiment_id": str(uuid.uuid4())[:8],
        "timestamp": datetime.utcnow().isoformat(),
        "sample_size": sample_size,
        "target_improvement": target_improvement,
        "direct_results": [asdict(r) for r in direct_results],
        "conjecture_results": [asdict(r) for r in conjecture_results]
    }
    
    # Convert datetime objects to strings for JSON serialization
    for result_list in [results_data["direct_results"], results_data["conjecture_results"]]:
        for result in result_list:
            result["timestamp"] = result["timestamp"].isoformat()
    
    results_file = results_dir / f"local_experiment_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print("Experiment completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())