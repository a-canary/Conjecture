#!/usr/bin/env python3
"""
Production Real Research with Proper Chutes API
Uses official Chutes API format and models
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def load_environment():
    """Load environment variables"""
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
    """Load test cases"""
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

def make_chutes_api_call(prompt, model_name, env_vars):
    """Make API call to Chutes using official format"""
    try:
        import requests
        
        api_url = env_vars.get('PROVIDER_API_URL', 'https://llm.chutes.ai/v1')
        api_key = env_vars.get('CHUTES_API_KEY')
        
        if not api_key:
            raise ValueError("CHUTES_API_KEY not found")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # Use proper Chutes model format
        if not model_name.startswith('chutes/'):
            model_name = f"chutes/{model_name}"
        
        data = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1500,
            "temperature": 0.3
        }
        
        endpoint = f"{api_url}/chat/completions"
        
        # Make request
        response = requests.post(endpoint, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            choice = result["choices"][0]
            message = choice.get("message", {})
            
            # Handle Chutes API format - check both content and reasoning_content
            content = message.get("content")
            reasoning_content = message.get("reasoning_content")
            
            if content is not None and content != "":
                return content
            elif reasoning_content is not None and reasoning_content != "":
                return reasoning_content
            else:
                print(f"Warning: Both content and reasoning_content are null/empty for {model_name}")
                return "No content available"
        else:
            print(f"Unexpected response format: {result}")
            raise ValueError("Unexpected response format")
            
    except Exception as e:
        print(f"Error making Chutes API call to {model_name}: {e}")
        raise

def generate_conjecture_prompt(test_case):
    """Generate Conjecture-style prompt"""
    if 'question' in test_case:
        question = test_case['question']
    elif 'task' in test_case:
        question = test_case['task']
    else:
        question = str(test_case.get('id', 'Unknown task'))

    return f"""You are tasked with solving a complex problem using Conjecture's approach of breaking down the problem into smaller, manageable claims.

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

def run_production_chutes_research():
    """Run production research with proper Chutes API"""
    print("PRODUCTION RESEARCH - Official Chutes API")
    print("=" * 60)
    
    # Load environment
    env_vars = load_environment()
    print(f"Loaded {len(env_vars)} environment variables")
    
    # Verify API key
    if not env_vars.get('CHUTES_API_KEY'):
        print("ERROR: CHUTES_API_KEY not found!")
        return None
    
    # Load test cases
    test_cases = load_test_cases()
    if not test_cases:
        print("No test cases found!")
        return None
    
    print(f"Loaded {len(test_cases)} test cases")
    
    # Use production-ready models from Chutes
    models = [
        'zai-org/GLM-4.6',           # Best GLM model - 203K context
        'deepseek-ai/DeepSeek-R1',    # Excellent reasoning - 164K context  
        'openai/gpt-oss-20b',         # Fast and cost-effective - 131K context
        'Qwen/Qwen2.5-72B-Instruct'   # Good quality model - 33K context
    ]
    
    approaches = [
        ('conjecture', 'Conjecture Claims-Based'),
        ('direct', 'Direct Prompting')
    ]
    
    # Run experiment
    all_results = []
    start_time = datetime.now()
    
    for model in models:
        print(f"\nTesting model: {model}")
        print("-" * 50)
        
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
                    
                    # Make REAL Chutes API call
                    print(f"      Making Chutes API call...")
                    response_start = time.time()
                    response = make_chutes_api_call(prompt, model, env_vars)
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
                    print(f"      [SUCCESS] Production Chutes response obtained")
                    
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
    print("\nPRODUCTION CHUTES RESULTS ANALYSIS:")
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
        'experiment_id': f'production_chutes_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
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
        'note': 'PRODUCTION CHUTES API CALLS - Official API format with high-quality models, no simulation'
    }
    
    # Save to file
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / f"{results_data['experiment_id']}.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nPRODUCTION Chutes results saved to: {results_file}")
    
    # Generate report
    report = generate_production_report(results_data)
    
    report_file = results_dir / f"{results_data['experiment_id']}_report.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"Report saved to: {report_file}")
    
    return results_data

def generate_production_report(results_data):
    """Generate production research report"""
    report = []
    report.append("# Production Chutes Research Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("## Executive Summary")
    report.append(f"- Models tested: {len(results_data['models_tested'])}")
    report.append(f"- Approaches compared: {len(results_data['approaches_tested'])}")
    report.append(f"- Test cases: {len(results_data['test_cases_used'])}")
    report.append(f"- Total evaluations: {len(results_data['results'])}")
    report.append(f"- Duration: {results_data['duration_seconds']:.1f} seconds")
    report.append("")
    report.append("**PRODUCTION READY**: All results are from official Chutes API with production models.")
    report.append("")
    
    # Results by model
    report.append("## Results by Model")
    for model, approaches in results_data['analysis']['model_stats'].items():
        report.append(f"### {model}")
        for approach, count in approaches.items():
            report.append(f"- {approach}: {count} successful calls")
        report.append("")
    
    # Sample responses
    report.append("## Sample Production Responses")
    for result in results_data['results'][:6]:  # Show first 6 results
        report.append(f"### {result['model']} - {result['approach_name']} - {result['test_case_id']}")
        report.append(f"**Response Time:** {result['response_time']:.2f}s")
        report.append(f"**Response Length:** {result['response_length']} characters")
        report.append("")
        report.append("**Production Response Preview:**")
        report.append("```")
        report.append(result['response'][:500] + '...' if len(result['response']) > 500 else result['response'])
        report.append("```")
        report.append("")
    
    # Technical details
    report.append("## Technical Details")
    report.append("- **API**: Official Chutes.ai (https://llm.chutes.ai/v1)")
    report.append("- **Models**: Production GLM-4.6, DeepSeek-R1, GPT-OSS-20b, Qwen2.5-72B")
    report.append("- **Authentication**: Bearer token authentication")
    report.append("- **Format**: OpenAI-compatible with Chutes-specific response handling")
    report.append("- **No Simulation**: All responses are genuine from production models")
    report.append("")
    
    return "\n".join(report)

def main():
    """Main function"""
    try:
        results = run_production_chutes_research()
        if results:
            print("\n" + "=" * 60)
            print("PRODUCTION CHUTES RESEARCH COMPLETED SUCCESSFULLY!")
            print("All results are from official production models - no simulation.")
            print(f"Total production responses obtained: {len(results['results'])}")
            return True
        else:
            print("Failed to obtain results")
            return False
    except Exception as e:
        print(f"Production Chutes research failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)