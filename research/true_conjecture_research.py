#!/usr/bin/env python3
"""
True Conjecture Implementation Research
Tests the actual Conjecture claims-based approach with proper claim format and processing
"""

import sys
import json
import time
import re
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
    """Make API call to Chutes"""
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
        
        data = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2000,
            "temperature": 0.3
        }
        
        endpoint = f"{api_url}/chat/completions"
        
        response = requests.post(endpoint, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            choice = result["choices"][0]
            message = choice.get("message", {})
            
            content = message.get("content")
            reasoning_content = message.get("reasoning_content")
            
            if content is not None and content != "":
                return content
            elif reasoning_content is not None and reasoning_content != "":
                return reasoning_content
            else:
                return "No content available"
        else:
            raise ValueError("Unexpected response format")
            
    except Exception as e:
        print(f"Error making Chutes API call to {model_name}: {e}")
        raise

def parse_claims_from_response(response):
    """Parse claims from model response in proper Conjecture format"""
    claims = []
    
    # Pattern to match [c{id} | content | / confidence]
    claim_pattern = r'\[c(\d+)\s*\|\s*([^|]+)\s*\|\s*/\s*([0-9.]+)\]'
    
    matches = re.findall(claim_pattern, response)
    
    for match in matches:
        claim_id, content, confidence = match
        try:
            confidence_val = float(confidence)
            if 0.0 <= confidence_val <= 1.0:
                claims.append({
                    'id': claim_id,
                    'content': content.strip(),
                    'confidence': confidence_val
                })
        except ValueError:
            continue
    
    return claims

def generate_true_conjecture_prompt(test_case):
    """Generate proper Conjecture prompt that forces claim format"""
    if 'question' in test_case:
        question = test_case['question']
    elif 'task' in test_case:
        question = test_case['task']
    else:
        question = str(test_case.get('id', 'Unknown task'))

    return f"""You are using Conjecture's claims-based reasoning system. Your task is to break down the problem into specific claims and evaluate each one.

**Problem:**
{question}

**CRITICAL INSTRUCTIONS:**
1. You MUST output claims in the EXACT format: [c1 | claim content | / 0.85]
2. Each claim must have: ID, content, and confidence (0.0-1.0)
3. Generate 3-5 claims that decompose the problem
4. After the claims, provide your final solution

**EXAMPLE FORMAT:**
[c1 | The doctor lives in house 3 based on clue 1 | / 0.95]
[c2 | The baker lives in house 1 based on clue 5 | / 0.90]
[c3 | The engineer's house is green based on clue 3 | / 0.85]

**Final Solution:** [Your answer here]

Now solve the problem using this EXACT format:"""

def generate_claim_evaluation_prompt(claims, original_question):
    """Generate prompt to evaluate claims and reach conclusion"""
    claims_text = "\n".join([f"[c{claim['id']} | {claim['content']} | / {claim['confidence']:.2f}]" for claim in claims])
    
    return f"""You are evaluating the following claims to solve a problem:

**Original Problem:**
{original_question}

**Claims to Evaluate:**
{claims_text}

**Instructions:**
1. Analyze each claim for validity and confidence
2. Identify relationships between claims
3. Resolve any contradictions
4. Provide a final answer based on the claims

**Response Format:**
**Claim Analysis:** [Your analysis of each claim]
**Final Answer:** [Your final solution to the original problem]"""

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

def run_true_conjecture_research():
    """Run research with proper Conjecture implementation"""
    print("TRUE CONJECTURE RESEARCH - Proper Claims-Based Approach")
    print("=" * 65)
    
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
    
    # Use working models
    models = [
        'zai-org/GLM-4.6',        # High quality
        'openai/gpt-oss-20b',      # Fast
    ]
    
    approaches = [
        ('true_conjecture', 'True Conjecture Claims-Based'),
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
                    if approach == 'true_conjecture':
                        # Step 1: Generate claims
                        print(f"      Step 1: Generating claims...")
                        claims_prompt = generate_true_conjecture_prompt(test_case)
                        
                        claims_start = time.time()
                        claims_response = make_chutes_api_call(claims_prompt, model, env_vars)
                        claims_time = time.time() - claims_start
                        
                        # Parse claims
                        claims = parse_claims_from_response(claims_response)
                        print(f"      Generated {len(claims)} claims in {claims_time:.2f}s")
                        
                        if not claims:
                            print(f"      [ERROR] No valid claims found in response")
                            continue
                        
                        # Step 2: Evaluate claims and get final answer
                        print(f"      Step 2: Evaluating claims...")
                        original_question = test_case.get('question', test_case.get('task', ''))
                        eval_prompt = generate_claim_evaluation_prompt(claims, original_question)
                        
                        eval_start = time.time()
                        final_response = make_chutes_api_call(eval_prompt, model, env_vars)
                        eval_time = time.time() - eval_start
                        
                        total_time = claims_time + eval_time
                        
                        result = {
                            'model': model,
                            'approach': approach,
                            'approach_name': approach_name,
                            'test_case_id': test_case['id'],
                            'test_case_category': test_case.get('category', 'unknown'),
                            'claims_generated': len(claims),
                            'claims': claims,
                            'claims_response': claims_response[:500] + '...' if len(claims_response) > 500 else claims_response,
                            'final_response': final_response[:800] + '...' if len(final_response) > 800 else final_response,
                            'claims_time': claims_time,
                            'eval_time': eval_time,
                            'total_time': total_time,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        print(f"      Final answer in {eval_time:.2f}s (total: {total_time:.2f}s)")
                        print(f"      [SUCCESS] True Conjecture process completed")
                        
                    else:  # direct
                        prompt = generate_direct_prompt(test_case)
                        
                        print(f"      Making direct call...")
                        response_start = time.time()
                        response = make_chutes_api_call(prompt, model, env_vars)
                        response_time = time.time() - response_start
                        
                        result = {
                            'model': model,
                            'approach': approach,
                            'approach_name': approach_name,
                            'test_case_id': test_case['id'],
                            'test_case_category': test_case.get('category', 'unknown'),
                            'response': response[:800] + '...' if len(response) > 800 else response,
                            'response_time': response_time,
                            'response_length': len(response),
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        print(f"      Got response in {response_time:.2f}s")
                        print(f"      [SUCCESS] Direct response obtained")
                    
                    approach_results.append(result)
                    
                except Exception as e:
                    print(f"      [ERROR] {e}")
                    continue
            
            if approach_results:
                all_results.extend(approach_results)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Analysis
    print(f"\nExperiment completed in {duration}")
    print("\nTRUE CONJECTURE RESULTS ANALYSIS:")
    print("=" * 50)
    
    if not all_results:
        print("No successful results obtained!")
        return None
    
    # Analyze results
    conjecture_results = [r for r in all_results if r['approach'] == 'true_conjecture']
    direct_results = [r for r in all_results if r['approach'] == 'direct']
    
    print(f"\nConjecture Results: {len(conjecture_results)} successful")
    print(f"Direct Results: {len(direct_results)} successful")
    
    if conjecture_results:
        avg_claims = sum(r['claims_generated'] for r in conjecture_results) / len(conjecture_results)
        avg_total_time = sum(r['total_time'] for r in conjecture_results) / len(conjecture_results)
        print(f"Average claims per Conjecture response: {avg_claims:.1f}")
        print(f"Average total time for Conjecture: {avg_total_time:.2f}s")
    
    # Save results
    results_data = {
        'experiment_id': f'true_conjecture_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_seconds': duration.total_seconds(),
        'models_tested': models,
        'approaches_tested': [a[0] for a in approaches],
        'test_cases_used': [tc['id'] for tc in test_cases[:2]],
        'results': all_results,
        'analysis': {
            'conjecture_count': len(conjecture_results),
            'direct_count': len(direct_results),
            'avg_claims_per_response': sum(r['claims_generated'] for r in conjecture_results) / len(conjecture_results) if conjecture_results else 0
        },
        'note': 'TRUE CONJECTURE RESEARCH - Proper claims-based approach with claim parsing and evaluation'
    }
    
    # Save to file
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / f"{results_data['experiment_id']}.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nTRUE Conjecture results saved to: {results_file}")
    
    # Generate report
    report = generate_true_conjecture_report(results_data)
    
    report_file = results_dir / f"{results_data['experiment_id']}_report.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"Report saved to: {report_file}")
    
    return results_data

def generate_true_conjecture_report(results_data):
    """Generate report for true Conjecture research"""
    report = []
    report.append("# True Conjecture Research Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("## Executive Summary")
    report.append(f"- Models tested: {len(results_data['models_tested'])}")
    report.append(f"- Approaches compared: {len(results_data['approaches_tested'])}")
    report.append(f"- Test cases: {len(results_data['test_cases_used'])}")
    report.append(f"- Total evaluations: {len(results_data['results'])}")
    report.append(f"- Duration: {results_data['duration_seconds']:.1f} seconds")
    report.append("")
    report.append("**TRUE CONJECTURE**: Testing proper claims-based approach with claim parsing and evaluation.")
    report.append("")
    
    # Results by approach
    conjecture_results = [r for r in results_data['results'] if r['approach'] == 'true_conjecture']
    direct_results = [r for r in results_data['results'] if r['approach'] == 'direct']
    
    report.append("## Results by Approach")
    report.append(f"### True Conjecture: {len(conjecture_results)} successful evaluations")
    if conjecture_results:
        avg_claims = results_data['analysis']['avg_claims_per_response']
        avg_time = sum(r['total_time'] for r in conjecture_results) / len(conjecture_results)
        report.append(f"- Average claims generated: {avg_claims:.1f}")
        report.append(f"- Average total time: {avg_time:.2f}s")
    
    report.append(f"### Direct: {len(direct_results)} successful evaluations")
    if direct_results:
        avg_time = sum(r['response_time'] for r in direct_results) / len(direct_results)
        avg_length = sum(r['response_length'] for r in direct_results) / len(direct_results)
        report.append(f"- Average response time: {avg_time:.2f}s")
        report.append(f"- Average response length: {avg_length:.0f} characters")
    
    report.append("")
    
    # Sample Conjecture results
    report.append("## Sample True Conjecture Results")
    for result in conjecture_results[:3]:
        report.append(f"### {result['model']} - {result['test_case_id']}")
        report.append(f"**Claims Generated:** {result['claims_generated']}")
        report.append(f"**Total Time:** {result['total_time']:.2f}s (Claims: {result['claims_time']:.2f}s, Eval: {result['eval_time']:.2f}s)")
        report.append("")
        report.append("**Generated Claims:**")
        for claim in result['claims']:
            report.append(f"- [c{claim['id']} | {claim['content']} | / {claim['confidence']:.2f}]")
        report.append("")
        report.append("**Final Response Preview:**")
        report.append("```")
        report.append(result['final_response'][:400] + '...' if len(result['final_response']) > 400 else result['final_response'])
        report.append("```")
        report.append("")
    
    # Technical Details
    report.append("## Technical Details")
    report.append("- **Approach**: True Conjecture with proper claim format [c{id} | content | / confidence]")
    report.append("- **Process**: 1) Generate claims, 2) Parse claims, 3) Evaluate claims, 4) Final answer")
    report.append("- **Claim Parsing**: Regex-based extraction of properly formatted claims")
    report.append("- **Models**: Production GLM-4.6, GPT-OSS-20b")
    report.append("- **No Simulation**: All responses are genuine from production models")
    report.append("")
    
    return "\n".join(report)

def main():
    """Main function"""
    try:
        results = run_true_conjecture_research()
        if results:
            print("\n" + "=" * 65)
            print("TRUE CONJECTURE RESEARCH COMPLETED!")
            print("Testing proper claims-based approach with claim parsing.")
            print(f"Total evaluations: {len(results['results'])}")
            return True
        else:
            print("Failed to obtain results")
            return False
    except Exception as e:
        print(f"True Conjecture research failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)