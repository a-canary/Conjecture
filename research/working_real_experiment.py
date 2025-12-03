#!/usr/bin/env python3
"""
Working Real Experiment - Focus on Local Models
Uses REAL LLM calls to local models only
"""

import sys
import json
import time
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
    """Load test cases from the test_cases directory"""
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

def make_real_lmstudio_call(prompt, model_name, api_url):
    """Make REAL LLM call to LM Studio"""
    try:
        import requests
        
        endpoint = f"{api_url}/v1/chat/completions"
        
        # Extract actual model name
        if "ibm/granite-4-h-tiny" in model_name:
            model = "ibm/granite-4-h-tiny"
        elif "GLM-Z1-9B-0414" in model_name:
            model = "GLM-Z1-9B-0414"
        else:
            model = model_name.split(':')[-1]
        
        # Prepare request
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2000,
            "temperature": 0.3
        }
        
        # Make request
        response = requests.post(endpoint, headers=headers, json=data, timeout=120)
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
        print(f"Error making LM Studio call to {model_name}: {e}")
        raise

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

def generate_direct_prompt(test_case):
    """Generate direct baseline prompt"""
    if 'question' in test_case:
        question = test_case['question']
    elif 'task' in test_case:
        question = test_case['task']
    else:
        question = str(test_case.get('id', 'Unknown task'))
    
    return f"""Answer the following question to the best of your ability:

{question}

Provide a clear, accurate, and complete answer."""

def run_working_real_experiment():
    """Run real experiment with working local models"""
    print("WORKING REAL EXPERIMENT - Local Models Only")
    print("=" * 60)
    
    # Load environment
    env_vars = load_environment()
    print(f"Loaded {len(env_vars)} environment variables")
    
    # Load test cases
    test_cases = load_test_cases()
    if not test_cases:
        print("No test cases found!")
        return None
    
    print(f"Loaded {len(test_cases)} test cases")
    
    # Use only working local models
    models = [
        'lmstudio:ibm/granite-4-h-tiny',
        'lmstudio:GLM-Z1-9B-0414'
    ]
    
    approaches = [
        ('conjecture', 'Conjecture Claims-Based'),
        ('direct', 'Direct Prompting')
    ]
    
    # Get LM Studio URL
    api_url = env_vars.get('LM_STUDIO_API_URL', 'http://localhost:1234')
    print(f"Using LM Studio at: {api_url}")
    
    # Run experiment
    all_results = []
    start_time = datetime.now()
    
    for model in models:
        print(f"\nTesting model: {model}")
        print("-" * 40)
        
        for approach, approach_name in approaches:
            print(f"  Approach: {approach_name}")
            
            approach_results = []
            
            # Test on first 2 test cases
            for test_case in test_cases[:2]:
                print(f"    Test case: {test_case['id']}")
                
                try:
                    # Generate prompt
                    if approach == 'conjecture':
                        prompt = generate_conjecture_prompt(test_case)
                    else:
                        prompt = generate_direct_prompt(test_case)
                    
                    # Make REAL LLM call
                    print(f"      Making real LLM call...")
                    response_start = time.time()
                    response = make_real_lmstudio_call(prompt, model, api_url)
                    response_time = time.time() - response_start
                    
                    print(f"      Got response in {response_time:.2f}s ({len(response)} chars)")
                    
                    result = {
                        'model': model,
                        'approach': approach,
                        'approach_name': approach_name,
                        'test_case_id': test_case['id'],
                        'test_case_category': test_case.get('category', 'unknown'),
                        'prompt': prompt[:300] + '...' if len(prompt) > 300 else prompt,
                        'response': response[:800] + '...' if len(response) > 800 else response,
                        'response_time': response_time,
                        'response_length': len(response),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    approach_results.append(result)
                    print(f"      [SUCCESS] Real response obtained")
                    
                except Exception as e:
                    print(f"      [ERROR] {e}")
                    continue
            
            if approach_results:
                all_results.extend(approach_results)
                
                # Show approach average
                avg_time = sum(r['response_time'] for r in approach_results) / len(approach_results)
                avg_length = sum(r['response_length'] for r in approach_results) / len(approach_results)
                print(f"    Approach averages: {avg_time:.2f}s, {avg_length:.0f} chars")
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Analysis
    print(f"\nExperiment completed in {duration}")
    print("\nREAL RESULTS ANALYSIS:")
    print("=" * 50)
    
    if not all_results:
        print("No successful results obtained!")
        return None
    
    # Group by approach and model
    approach_stats = {}
    model_stats = {}
    
    for result in all_results:
        approach = result['approach']
        model = result['model']
        
        if approach not in approach_stats:
            approach_stats[approach] = []
        if model not in model_stats:
            model_stats[model] = {}
        if approach not in model_stats[model]:
            model_stats[model][approach] = []
        
        approach_stats[approach].append(result)
        model_stats[model][approach].append(result)
    
    # Approach comparison
    print("\nApproach Performance:")
    for approach, results in approach_stats.items():
        avg_time = sum(r['response_time'] for r in results) / len(results)
        avg_length = sum(r['response_length'] for r in results) / len(results)
        print(f"  {approach}: {len(results)} tests, {avg_time:.2f}s avg, {avg_length:.0f} chars avg")
    
    # Model-specific analysis
    print("\nModel-Specific Performance:")
    for model, approaches in model_stats.items():
        print(f"\n{model}:")
        for approach, results in approaches.items():
            avg_time = sum(r['response_time'] for r in results) / len(results)
            avg_length = sum(r['response_length'] for r in results) / len(results)
            print(f"  {approach}: {avg_time:.2f}s, {avg_length:.0f} chars")
    
    # Save results
    results_data = {
        'experiment_id': f'working_real_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_seconds': duration.total_seconds(),
        'models_tested': models,
        'approaches_tested': [a[0] for a in approaches],
        'test_cases_used': [tc['id'] for tc in test_cases[:2]],
        'results': all_results,
        'analysis': {
            'approach_stats': {approach: len(results) for approach, results in approach_stats.items()},
            'model_stats': {model: {approach: len(results) for approach, results in approaches.items()} 
                           for model, approaches in model_stats.items()}
        },
        'note': 'REAL LLM CALLS ONLY - Local models via LM Studio, no simulation'
    }
    
    # Save to file
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / f"{results_data['experiment_id']}.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nREAL results saved to: {results_file}")
    
    # Generate simple report
    report = generate_working_report(results_data)
    
    report_file = results_dir / f"{results_data['experiment_id']}_report.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"Report saved to: {report_file}")
    
    return results_data

def generate_working_report(results_data):
    """Generate report for working experiment"""
    report = []
    report.append("# Working Real Experiment Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("## Executive Summary")
    report.append(f"- Models tested: {len(results_data['models_tested'])}")
    report.append(f"- Approaches compared: {len(results_data['approaches_tested'])}")
    report.append(f"- Test cases: {len(results_data['test_cases_used'])}")
    report.append(f"- Total evaluations: {len(results_data['results'])}")
    report.append(f"- Duration: {results_data['duration_seconds']:.1f} seconds")
    report.append("")
    report.append("**IMPORTANT**: All results are from REAL LLM calls - no simulation used.")
    report.append("")
    
    # Results by model
    report.append("## Results by Model")
    for model, approaches in results_data['analysis']['model_stats'].items():
        report.append(f"### {model}")
        for approach, count in approaches.items():
            report.append(f"- {approach}: {count} successful calls")
        report.append("")
    
    # Sample responses
    report.append("## Sample Responses")
    for result in results_data['results'][:4]:  # Show first 4 results
        report.append(f"### {result['model']} - {result['approach_name']} - {result['test_case_id']}")
        report.append(f"**Response Time:** {result['response_time']:.2f}s")
        report.append(f"**Response Length:** {result['response_length']} characters")
        report.append("")
        report.append("**Response Preview:**")
        report.append("```")
        report.append(result['response'][:500] + '...' if len(result['response']) > 500 else result['response'])
        report.append("```")
        report.append("")
    
    # Technical details
    report.append("## Technical Details")
    report.append("- **API**: LM Studio local server")
    report.append("- **Models**: Local models running on user hardware")
    report.append("- **Prompt Types**: Conjecture claims-based vs direct prompting")
    report.append("- **No Simulation**: All responses are genuine LLM outputs")
    report.append("")
    
    return "\n".join(report)

def main():
    """Main function"""
    try:
        results = run_working_real_experiment()
        if results:
            print("\n" + "=" * 60)
            print("WORKING REAL EXPERIMENT COMPLETED SUCCESSFULLY!")
            print("All results are from actual LLM calls - no simulation.")
            print(f"Total real responses obtained: {len(results['results'])}")
            return True
        else:
            print("Failed to obtain results")
            return False
    except Exception as e:
        print(f"Real experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)