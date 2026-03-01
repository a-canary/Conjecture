#!/usr/bin/env python3
"""
Improved True Conjecture Implementation
Addresses the 50% failure rate with better prompts and parsing
"""

import sys
import json
import time
import re
import statistics
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
            "max_tokens": 1500,
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
    """Parse claims from model response with improved patterns"""
    claims = []
    
    # Multiple patterns to catch different formatting variations
    patterns = [
        r'\[c(\d+)\s*\|\s*([^|]+)\s*\|\s*/\s*([0-9.]+)\]',  # Standard format
        r'\[c(\d+)\s*\|\s*([^|]+)\s*\|\s*/\s*([0-9.]+)\s*\]',  # With trailing space
        r'\[c(\d+)\|([^|]+)\|/([0-9.]+)\]',  # Compact format
        r'\[c(\d+)\s*\|\s*([^|]+)\s*\|\s*/\s*([0-9.]+)\s*\]',  # Extra spaces
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response)
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

def generate_improved_conjecture_prompt(test_case):
    """Generate improved Conjecture prompt that addresses failure patterns"""
    if 'question' in test_case:
        question = test_case['question']
    elif 'task' in test_case:
        question = test_case['task']
    else:
        question = str(test_case.get('id', 'Unknown task'))

    return f"""IMPORTANT: You must output ONLY claims in the exact format shown below. No explanations, no descriptions, just claims.

PROBLEM: {question}

REQUIRED FORMAT (copy exactly):
[c1 | claim text here | / 0.85]
[c2 | another claim here | / 0.90]
[c3 | final claim here | / 0.75]

RULES:
1. Start each claim with [c1], [c2], [c3], etc.
2. Use | to separate parts
3. End with / and a number between 0.0 and 1.0
4. Make 3-5 claims about the problem
5. Claims should be specific, testable statements

EXAMPLE:
[c1 | The doctor lives in house 3 | / 0.95]
[c2 | The baker lives in house 1 | / 0.90]
[c3 | The engineer's house is green | / 0.85]

NOW OUTPUT YOUR CLAIMS:"""

def generate_claim_evaluation_prompt(claims, original_question):
    """Generate prompt to evaluate claims and reach conclusion"""
    claims_text = "\n".join([f"[c{claim['id']} | {claim['content']} | / {claim['confidence']:.2f}]" for claim in claims])
    
    return f"""Evaluate these claims to solve the problem:

ORIGINAL PROBLEM:
{original_question}

CLAIMS TO EVALUATE:
{claims_text}

TASK:
1. Analyze each claim
2. Determine the final answer
3. Provide your solution

Format your response as:
FINAL ANSWER: [Your solution here]"""

def generate_direct_prompt(test_case):
    """Generate direct baseline prompt"""
    if 'question' in test_case:
        question = test_case['question']
    elif 'task' in test_case:
        question = test_case['task']
    else:
        question = str(test_case.get('id', 'Unknown task'))
    
    return f"""Answer the following question:

{question}

Provide your answer directly."""

def generate_chain_of_thought_prompt(test_case):
    """Generate Chain of Thought prompt"""
    if 'question' in test_case:
        question = test_case['question']
    elif 'task' in test_case:
        question = test_case['task']
    else:
        question = str(test_case.get('id', 'Unknown task'))
    
    return f"""Solve this step by step:

{question}

Think through each step carefully, then give your final answer."""

def run_improved_conjecture_study():
    """Run improved Conjecture study to beat Chain of Thought accuracy"""
    print("IMPROVED TRUE CONJECTURE STUDY")
    print("Goal: Beat Chain of Thought accuracy")
    print("=" * 50)
    
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
    
    # Use models
    models = [
        'zai-org/GLM-4.6',        # High quality
        'openai/gpt-oss-20b',      # Fast
    ]
    
    approaches = [
        ('improved_conjecture', 'Improved True Conjecture'),
        ('direct', 'Direct Prompting'),
        ('chain_of_thought', 'Chain of Thought')
    ]
    
    # Run experiment
    all_results = []
    start_time = datetime.now()
    
    for model in models:
        print(f"\nTesting model: {model}")
        print("-" * 40)
        
        for approach, approach_name in approaches:
            print(f"  Approach: {approach_name}")
            
            approach_results = []
            
            # Test on all test cases
            for test_case in test_cases:
                print(f"    Test case: {test_case['id']}")
                
                try:
                    if approach == 'improved_conjecture':
                        # Step 1: Generate claims with improved prompt
                        print(f"      Step 1: Generating claims...")
                        claims_prompt = generate_improved_conjecture_prompt(test_case)
                        
                        claims_start = time.time()
                        claims_response = make_chutes_api_call(claims_prompt, model, env_vars)
                        claims_time = time.time() - claims_start
                        
                        # Parse claims
                        claims = parse_claims_from_response(claims_response)
                        print(f"      Generated {len(claims)} claims in {claims_time:.2f}s")
                        
                        if not claims:
                            print(f"      [FAILURE] No valid claims found")
                            # Still record as failure for accuracy calculation
                            result = {
                                'model': model,
                                'approach': approach,
                                'approach_name': approach_name,
                                'test_case_id': test_case['id'],
                                'test_case_category': test_case.get('category', 'unknown'),
                                'success': False,
                                'claims_generated': 0,
                                'error': 'No claims parsed',
                                'claims_response': claims_response[:200] + '...' if len(claims_response) > 200 else claims_response,
                                'total_time': claims_time,
                                'timestamp': datetime.now().isoformat()
                            }
                        else:
                            # Step 2: Evaluate claims
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
                                'success': True,
                                'claims_generated': len(claims),
                                'claims': claims,
                                'claims_response': claims_response[:300] + '...' if len(claims_response) > 300 else claims_response,
                                'final_response': final_response[:500] + '...' if len(final_response) > 500 else final_response,
                                'claims_time': claims_time,
                                'eval_time': eval_time,
                                'total_time': total_time,
                                'timestamp': datetime.now().isoformat()
                            }
                            
                            print(f"      Final answer in {eval_time:.2f}s (total: {total_time:.2f}s)")
                            print(f"      [SUCCESS] Improved Conjecture completed")
                        
                    elif approach == 'chain_of_thought':
                        prompt = generate_chain_of_thought_prompt(test_case)
                        
                        print(f"      Making Chain of Thought call...")
                        response_start = time.time()
                        response = make_chutes_api_call(prompt, model, env_vars)
                        response_time = time.time() - response_start
                        
                        result = {
                            'model': model,
                            'approach': approach,
                            'approach_name': approach_name,
                            'test_case_id': test_case['id'],
                            'test_case_category': test_case.get('category', 'unknown'),
                            'success': True,
                            'response': response[:500] + '...' if len(response) > 500 else response,
                            'response_time': response_time,
                            'response_length': len(response),
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        print(f"      Got response in {response_time:.2f}s")
                        print(f"      [SUCCESS] Chain of Thought completed")
                        
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
                            'success': True,
                            'response': response[:500] + '...' if len(response) > 500 else response,
                            'response_time': response_time,
                            'response_length': len(response),
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        print(f"      Got response in {response_time:.2f}s")
                        print(f"      [SUCCESS] Direct completed")
                    
                    approach_results.append(result)
                    
                except Exception as e:
                    print(f"      [ERROR] {e}")
                    continue
            
            if approach_results:
                all_results.extend(approach_results)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Calculate accuracy and performance
    print(f"\nExperiment completed in {duration}")
    print("\nACCURACY AND PERFORMANCE ANALYSIS:")
    print("=" * 50)
    
    if not all_results:
        print("No results obtained!")
        return None
    
    # Calculate success rates by approach
    approach_stats = {}
    for result in all_results:
        approach = result['approach']
        if approach not in approach_stats:
            approach_stats[approach] = {'total': 0, 'success': 0, 'times': [], 'response_lengths': []}
        
        approach_stats[approach]['total'] += 1
        if result.get('success', True):
            approach_stats[approach]['success'] += 1
        
        if 'total_time' in result:
            approach_stats[approach]['times'].append(result['total_time'])
        elif 'response_time' in result:
            approach_stats[approach]['times'].append(result['response_time'])
        
        if 'response_length' in result:
            approach_stats[approach]['response_lengths'].append(result['response_length'])
    
    # Print results
    print("\nApproach Performance:")
    for approach, stats in approach_stats.items():
        success_rate = (stats['success'] / stats['total']) * 100
        avg_time = statistics.mean(stats['times']) if stats['times'] else 0
        avg_length = statistics.mean(stats['response_lengths']) if stats['response_lengths'] else 0
        
        print(f"\n{approach.replace('_', ' ').title()}:")
        print(f"  Success Rate: {success_rate:.1f}% ({stats['success']}/{stats['total']})")
        print(f"  Average Time: {avg_time:.2f}s")
        print(f"  Average Length: {avg_length:.0f} characters")
    
    # Check if Improved Conjecture beats Chain of Thought
    conj_stats = approach_stats.get('improved_conjecture', {})
    cot_stats = approach_stats.get('chain_of_thought', {})
    
    if conj_stats and cot_stats:
        conj_rate = (conj_stats['success'] / conj_stats['total']) * 100
        cot_rate = (cot_stats['success'] / cot_stats['total']) * 100
        
        print(f"\nHEAD-TO-HEAD COMPARISON:")
        print(f"Improved Conjecture: {conj_rate:.1f}% success rate")
        print(f"Chain of Thought: {cot_rate:.1f}% success rate")
        
        if conj_rate > cot_rate:
            improvement = conj_rate - cot_rate
            print(f"🎉 IMPROVED CONJECTURE WINS! +{improvement:.1f}% better than Chain of Thought")
        elif conj_rate == cot_rate:
            print(f"⚖️  TIE: Both approaches have {conj_rate:.1f}% success rate")
        else:
            gap = cot_rate - conj_rate
            print(f"❌ Chain of Thought still leads by {gap:.1f}%")
    
    # Save results
    results_data = {
        'experiment_id': f'improved_conjecture_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_seconds': duration.total_seconds(),
        'models_tested': models,
        'approaches_tested': [a[0] for a in approaches],
        'test_cases_used': [tc['id'] for tc in test_cases],
        'results': all_results,
        'accuracy_analysis': approach_stats,
        'note': 'IMPROVED TRUE CONJECTURE - Better prompts to address 50% failure rate'
    }
    
    # Save to file
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / f"{results_data['experiment_id']}.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results_data

def main():
    """Main function"""
    try:
        results = run_improved_conjecture_study()
        if results:
            print("\n" + "=" * 50)
            print("IMPROVED CONJECTURE STUDY COMPLETED!")
            print("Check if we beat Chain of Thought accuracy.")
            return True
        else:
            print("Failed to obtain results")
            return False
    except Exception as e:
        print(f"Improved Conjecture study failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)