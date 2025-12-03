#!/usr/bin/env python3
"""
Baseline Comparison Test - REAL LLM CALLS ONLY
Compares Conjecture approach vs direct prompting baseline using actual models
"""

import sys
import json
import time
import os
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def load_environment():
    """Load real environment variables from .env files"""
    env_vars = {}
    env_files = [
        Path(__file__).parent.parent / '.env',
        Path(__file__).parent / '.env'
    ]
    
    for env_file in env_files:
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
    
    return env_vars

def load_test_cases():
    """Load test cases for comparison"""
    test_case_dir = Path(__file__).parent / 'test_cases'
    test_cases = []

    for file_path in test_case_dir.glob('*.json'):
        try:
            with open(file_path, 'r') as f:
                test_case = json.load(f)
                test_cases.append(test_case)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return test_cases

def generate_conjecture_prompt(test_case):
    """Generate Conjecture-style claims-based prompt"""
    if 'question' in test_case:
        question = test_case['question']
    elif 'task' in test_case:
        question = test_case['task']
    else:
        question = str(test_case.get('id', 'Unknown task'))

    conjecture_prompt = f"""You are tasked with solving a complex problem using Conjecture's approach of breaking down the problem into smaller, manageable claims.

**Problem:**
{question}

**Instructions:**
1. Decompose the problem into 3-5 key claims or subtasks
2. For each claim/subtask, provide a confidence score (0.0-1.0)
3. Show how the claims relate to each other
4. Provide a final solution based on the claims

Format your response using Conjecture's claim format:
[c1 | claim content | / confidence]
[c2 | supporting claim | / confidence]
etc.

Then provide your final solution."""
    
    return conjecture_prompt

def generate_baseline_prompt(test_case, baseline_type="direct"):
    """Generate baseline prompt for comparison"""
    if 'question' in test_case:
        question = test_case['question']
    elif 'task' in test_case:
        question = test_case['task']
    else:
        question = str(test_case.get('id', 'Unknown task'))
    
    if baseline_type == "direct":
        return f"""Answer the following question to the best of your ability:

{question}

Provide a clear, accurate, and complete answer."""
    
    elif baseline_type == "chain_of_thought":
        return f"""Answer the following question step by step. Think through each step carefully before providing your final answer.

{question}

Show your reasoning step by step, then give your final answer."""
    
    elif baseline_type == "few_shot":
        return f"""Answer the following question by following the pattern in these examples:

Example 1:
Q: What is 2+2?
A: To solve 2+2, I add the numbers: 2+2 = 4. The answer is 4.

Example 2:
Q: What is the capital of France?
A: The capital of France is Paris.

Now answer this question:
{question}

Provide your answer with brief reasoning."""
    
    else:
        return question

def make_real_llm_call(prompt, model_name, env_vars):
    """Make REAL LLM call to specified model"""
    try:
        import requests
        
        # Map model names to actual API endpoints
        if "lmstudio" in model_name:
            # LM Studio local API
            api_url = env_vars.get('LM_STUDIO_API_URL', 'http://localhost:1234')
            endpoint = f"{api_url}/v1/chat/completions"
            
            # Extract actual model name
            if "ibm/granite-4-h-tiny" in model_name:
                model = "ibm/granite-4-h-tiny"
            elif "GLM-Z1-9B-0414" in model_name:
                model = "GLM-Z1-9B-0414"
            else:
                model = model_name.split(':')[-1]
            
        elif "chutes" in model_name:
            # Chutes API
            api_url = env_vars.get('PROVIDER_API_URL', 'https://llm.chutes.ai/v1')
            api_key = env_vars.get('CHUTES_API_KEY') or env_vars.get('PROVIDER_API_KEY')
            endpoint = f"{api_url}/chat/completions"
            
            if not api_key:
                raise ValueError("CHUTES_API_KEY not found in environment")
            
            # Extract model name
            if "GLM-4.5-Air" in model_name:
                model = "zai-org/GLM-4.5-Air"
            elif "GLM-4.6" in model_name:
                model = "zai-org/GLM-4.6"
            else:
                model = model_name.split(':')[-1]
        
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Prepare request
        headers = {
            "Content-Type": "application/json"
        }
        
        if "chutes" in model_name:
            headers["Authorization"] = f"Bearer {api_key}"
        
        data = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2000,
            "temperature": 0.3
        }
        
        # Make request
        response = requests.post(endpoint, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        # Extract response text with GLM reasoning content support
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0].get("message", {})

            # Check for reasoning_content first (GLM models)
            reasoning_content = message.get("reasoning_content")
            if reasoning_content:
                return reasoning_content.strip()

            # Fallback to standard content
            content = message.get("content")
            if content:
                return content.strip()

            # If no content found, log the response
            print(f"No content found in response: {result}")
            raise ValueError("No content found in API response")
        else:
            print(f"Unexpected response format: {result}")
            raise ValueError("Unexpected response format")
            
    except Exception as e:
        print(f"Error making LLM call to {model_name}: {e}")
        raise

def evaluate_with_real_judge(test_case, response, approach, model_name, env_vars):
    """Evaluate response using REAL GLM-4.6 judge"""
    try:
        import requests
        
        # Get judge configuration
        judge_model = env_vars.get('JUDGE_MODEL', 'chutes:GLM-4.6')
        api_url = env_vars.get('PROVIDER_API_URL', 'https://llm.chutes.ai/v1')
        api_key = env_vars.get('CHUTES_API_KEY') or env_vars.get('PROVIDER_API_KEY')
        
        if not api_key:
            raise ValueError("CHUTES_API_KEY not found for judge evaluation")
        
        # Create evaluation prompt
        question = test_case.get('question', test_case.get('task', 'Unknown task'))
        ground_truth = test_case.get('ground_truth', 'No ground truth provided')
        
        eval_prompt = f"""You are an expert evaluator assessing AI model responses. Your task is to evaluate a response based on multiple criteria.

**Question/Task:**
{question}

**Ground Truth (Reference Answer):**
{ground_truth}

**Model Response:**
{response}

**Approach Used:** {approach}
**Model:** {model_name}

**Evaluation Criteria:**
1. Correctness (0.0-1.0): Factual accuracy
2. Completeness (0.0-1.0): Coverage of all aspects
3. Coherence (0.0-1.0): Logical flow and consistency
4. Reasoning Quality (0.0-1.0): Strength of logical arguments
5. Clarity (0.0-1.0): Expression and understandability

Please evaluate the response and provide your assessment in the following format:

CORRECTNESS: [0.0-1.0]
COMPLETENESS: [0.0-1.0]
COHERENCE: [0.0-1.0]
REASONING_QUALITY: [0.0-1.0]
CLARITY: [0.0-1.0]
OVERALL_SCORE: [0.0-1.0]
REASONING: [Detailed explanation of your evaluation]
CONFIDENCE: [0.0-1.0 for your confidence in this evaluation]

Be objective, thorough, and fair in your evaluation."""

        # Make judge call
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        data = {
            "model": "zai-org/GLM-4.6",
            "messages": [
                {"role": "user", "content": eval_prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.1
        }
        
        endpoint = f"{api_url}/chat/completions"
        response = requests.post(endpoint, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0].get("message", {})

            # Check for reasoning_content first (GLM models)
            eval_text = message.get("reasoning_content")
            if not eval_text:
                # Fallback to standard content
                eval_text = message.get("content", "")
            
            # Parse evaluation
            evaluation = parse_judge_response(eval_text)
            evaluation['judge_model'] = judge_model
            evaluation['approach'] = approach
            evaluation['model_evaluated'] = model_name
            
            return evaluation
        else:
            raise ValueError("Unexpected judge response format")
            
    except Exception as e:
        print(f"Error in judge evaluation: {e}")
        raise

def parse_judge_response(eval_text):
    """Parse judge evaluation response"""
    evaluation = {
        'scores': {},
        'overall_score': 0.0,
        'reasoning': '',
        'confidence': 0.0
    }
    
    try:
        lines = eval_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('CORRECTNESS:'):
                evaluation['scores']['correctness'] = float(line.split(':')[1].strip())
            elif line.startswith('COMPLETENESS:'):
                evaluation['scores']['completeness'] = float(line.split(':')[1].strip())
            elif line.startswith('COHERENCE:'):
                evaluation['scores']['coherence'] = float(line.split(':')[1].strip())
            elif line.startswith('REASONING_QUALITY:'):
                evaluation['scores']['reasoning_quality'] = float(line.split(':')[1].strip())
            elif line.startswith('CLARITY:'):
                evaluation['scores']['clarity'] = float(line.split(':')[1].strip())
            elif line.startswith('OVERALL_SCORE:'):
                evaluation['overall_score'] = float(line.split(':')[1].strip())
            elif line.startswith('REASONING:'):
                evaluation['reasoning'] = line.split(':', 1)[1].strip()
            elif line.startswith('CONFIDENCE:'):
                evaluation['confidence'] = float(line.split(':')[1].strip())
        
        # If no overall score provided, calculate average
        if evaluation['overall_score'] == 0.0 and evaluation['scores']:
            evaluation['overall_score'] = sum(evaluation['scores'].values()) / len(evaluation['scores'])
        
        return evaluation
        
    except Exception as e:
        print(f"Error parsing judge response: {e}")
        # Return minimal evaluation
        return {
            'scores': {'correctness': 0.5},
            'overall_score': 0.5,
            'reasoning': f'Parse error: {str(e)}',
            'confidence': 0.3
        }

def run_real_baseline_comparison():
    """Run REAL baseline comparison with actual LLM calls"""
    print("REAL BASELINE COMPARISON - Using Actual LLM Calls")
    print("=" * 60)
    
    # Load environment
    env_vars = load_environment()
    print(f"Loaded {len(env_vars)} environment variables")
    
    # Verify required API keys
    if not env_vars.get('CHUTES_API_KEY'):
        print("ERROR: CHUTES_API_KEY not found in environment")
        return None
    
    # Load test cases
    test_cases = load_test_cases()
    if not test_cases:
        print("No test cases found!")
        return None
    
    print(f"Loaded {len(test_cases)} test cases")
    
    # Define models and approaches
    models = [
        'lmstudio:ibm/granite-4-h-tiny',
        'lmstudio:GLM-Z1-9B-0414', 
        'chutes:GLM-4.5-Air',
        'chutes:GLM-4.6'
    ]
    
    approaches = [
        ('conjecture', 'Conjecture Claims-Based'),
        ('direct', 'Direct Prompting'),
        ('chain_of_thought', 'Chain of Thought'),
        ('few_shot', 'Few-Shot Learning')
    ]
    
    # Run comparison
    all_results = []
    start_time = datetime.now()
    
    for model in models:
        print(f"\nTesting model: {model}")
        print("-" * 40)
        
        for approach, approach_name in approaches:
            print(f"  Approach: {approach_name}")
            
            approach_results = []
            
            # Test on subset of test cases
            for test_case in test_cases[:2]:  # Test first 2 cases
                print(f"    Test case: {test_case['id']}")
                
                try:
                    # Generate prompt
                    if approach == 'conjecture':
                        prompt = generate_conjecture_prompt(test_case)
                    else:
                        prompt = generate_baseline_prompt(test_case, approach)
                    
                    # Make REAL LLM call
                    print(f"      Making LLM call...")
                    response_start = time.time()
                    response = make_real_llm_call(prompt, model, env_vars)
                    response_time = time.time() - response_start
                    
                    print(f"      Got response in {response_time:.2f}s")
                    
                    # Evaluate with REAL judge
                    print(f"      Evaluating with GLM-4.6 judge...")
                    eval_start = time.time()
                    evaluation = evaluate_with_real_judge(test_case, response, approach, model, env_vars)
                    eval_time = time.time() - eval_start
                    
                    print(f"      Evaluation: {evaluation['overall_score']:.3f}")
                    
                    result = {
                        'model': model,
                        'approach': approach,
                        'approach_name': approach_name,
                        'test_case_id': test_case['id'],
                        'test_case_category': test_case.get('category', 'unknown'),
                        'prompt': prompt[:200] + '...' if len(prompt) > 200 else prompt,
                        'response': response[:500] + '...' if len(response) > 500 else response,
                        'response_time': response_time,
                        'evaluation_time': eval_time,
                        'evaluation': evaluation
                    }
                    
                    approach_results.append(result)
                    
                except Exception as e:
                    print(f"      ERROR: {e}")
                    # Continue with next test case
                    continue
            
            if approach_results:
                all_results.extend(approach_results)
                
                # Show approach average
                avg_score = sum(r['evaluation']['overall_score'] for r in approach_results) / len(approach_results)
                print(f"    Approach average: {avg_score:.3f}")
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Analysis
    print(f"\nComparison completed in {duration}")
    print("\nREAL RESULTS ANALYSIS:")
    print("=" * 50)
    
    if not all_results:
        print("No successful results obtained!")
        return None
    
    # Group by approach and model
    approach_performance = {}
    model_performance = {}
    
    for result in all_results:
        approach = result['approach']
        model = result['model']
        score = result['evaluation']['overall_score']
        
        if approach not in approach_performance:
            approach_performance[approach] = []
        if model not in model_performance:
            model_performance[model] = {}
        if approach not in model_performance[model]:
            model_performance[model][approach] = []
        
        approach_performance[approach].append(score)
        model_performance[model][approach].append(score)
    
    # Approach comparison
    print("\nApproach Performance (All Models):")
    for approach, scores in approach_performance.items():
        avg_score = sum(scores) / len(scores)
        print(f"  {approach}: {avg_score:.3f} ({len(scores)} tests)")
    
    # Model-specific analysis
    print("\nModel-Specific Performance:")
    for model, approaches in model_performance.items():
        print(f"\n{model}:")
        for approach, scores in approaches.items():
            avg_score = sum(scores) / len(scores)
            print(f"  {approach}: {avg_score:.3f}")
    
    # Save results
    results_data = {
        'experiment_id': f'real_baseline_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_seconds': duration.total_seconds(),
        'models_tested': models,
        'approaches_tested': [a[0] for a in approaches],
        'test_cases_used': [tc['id'] for tc in test_cases[:2]],
        'results': all_results,
        'analysis': {
            'approach_performance': {approach: {'avg_score': sum(scores)/len(scores), 'count': len(scores)} 
                                   for approach, scores in approach_performance.items()},
            'model_performance': model_performance
        },
        'note': 'REAL LLM CALLS - No simulation used'
    }
    
    # Save to file
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / f"{results_data['experiment_id']}.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nREAL results saved to: {results_file}")
    
    return results_data

def main():
    """Main function"""
    try:
        results = run_real_baseline_comparison()
        if results:
            print("\n" + "=" * 60)
            print("REAL BASELINE COMPARISON COMPLETED SUCCESSFULLY!")
            print("All results are from actual LLM calls - no simulation.")
            return True
        else:
            print("Failed to obtain results")
            return False
    except Exception as e:
        print(f"Real baseline comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)