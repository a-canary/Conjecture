#!/usr/bin/env python3
<arg_value>Diagnose True Conjecture Failure Patterns
Analyze why True Conjecture has 50% failure rate
"""

import sys
import json
import re
from pathlib import Path

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
            "max_tokens": 1000,
            "temperature": 0.3
        }
        
        endpoint = f"{api_url}/chat/completions"
        
        response = requests.post(endpoint, headers=headers, json=data, timeout=60)
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

def generate_diagnostic_prompt(test_case):
    """Generate diagnostic prompt to understand failure patterns"""
    if 'question' in test_case:
        question = test_case['question']
    elif 'task' in test_case:
        question = test_case['task']
    else:
        question = str(test_case.get('id', 'Unknown task'))

    prompt = f"""You are using Conjecture's claims-based reasoning system. Your task is to break down the problem into specific claims and evaluate each one.

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

    return prompt

def diagnose_failure_patterns():
    """Diagnose why True Conjecture fails 50% of the time"""
    print("DIAGNOSING TRUE CONJECTURE FAILURE PATTERNS")
    print("=" * 60)
    
    # Load environment
    env_vars = load_environment()
    print(f"Loaded {len(env_vars)} environment variables")
    
    # Load test cases that failed in previous runs
    test_cases = [
        {
            'id': 'complex_reasoning_001',
            'question': '''In a small town, there are five houses in a row, each painted a different color: red, blue, green, yellow, and white. Each house is owned by a person with a different profession: doctor, teacher, engineer, artist, and baker. Each person has a different favorite fruit: apple, banana, cherry, date, and elderberry. Using the following clues, determine who owns the red house and what is their favorite fruit?

Clues:
1. The doctor lives in the middle house.
2. The artist lives next to the person who likes apples.
3. The engineer lives in the green house.
4. The teacher likes bananas.
5. The baker lives in the first house.
6. The person who likes cherries lives next to the white house.
7. The red house is somewhere to the left of the blue house.
8. The artist does not live in the yellow house.
9. The person who likes dates lives next to the doctor.
10. The person who likes elderberries lives in the last house.''',
            'category': 'complex_reasoning'
        },
        {
            'id': 'logic_puzzle_20251202_212949',
            'question': '''In a small town, there are five houses in a row, each painted a different color: red, green, blue, white, yellow. Each house is owned by a person with a different profession: artist, engineer, baker, doctor, teacher. Each person has a different favorite fruit: elderberry, cherry, date, apple, banana.

Clues:
1. The baker lives in the middle house.
2. The artist lives in the first house.
3. The person who likes bananas lives in the last house.
4. The engineer lives in the green house.
5. The teacher likes cherries.
6. The person who likes dates lives next to the doctor.
7. The red house is somewhere to the left of the blue house.
8. The artist does not live in the yellow house.
9. The person who likes elderberries lives next to the white house.
10. The person who likes apples lives next to the baker.

Who owns the green house and what is their favorite fruit?''',
            'category': 'complex_reasoning'
        }
    ]
    
    models = ['zai-org/GLM-4.6', 'openai/gpt-oss-20b']
    
    failure_analysis = []
    
    for model in models:
        print(f"\nAnalyzing model: {model}")
        print("-" * 40)
        
        for test_case in test_cases:
            print(f"  Test case: {test_case['id']}")
            
            try:
                # Generate claims
                prompt = generate_diagnostic_prompt(test_case)
                response = make_chutes_api_call(prompt, model, env_vars)
                
                # Parse claims
                claims = parse_claims_from_response(response)
                
                print(f"    Claims found: {len(claims)}")
                
                if len(claims) == 0:
                    print(f"    [FAILURE] No claims parsed")
                    
                    # Analyze why it failed
                    analysis = {
                        'model': model,
                        'test_case': test_case['id'],
                        'failure_type': 'no_claims_parsed',
                        'response_length': len(response),
                        'response_preview': response[:300] + '...' if len(response) > 300 else response,
                        'contains_brackets': '[' in response and ']' in response,
                        'contains_c_format': '[c' in response,
                        'contains_confidence': '/ ' in response or '/0.' in response
                    }
                    
                    failure_analysis.append(analysis)
                    
                    # Check for common failure patterns
                    if not analysis['contains_brackets']:
                        print(f"    REASON: No brackets found in response")
                    elif not analysis['contains_c_format']:
                        print(f"    REASON: No [c format found")
                    elif not analysis['contains_confidence']:
                        print(f"    REASON: No confidence format found")
                    else:
                        print(f"    REASON: Format present but parsing failed")
                        
                else:
                    print(f"    [SUCCESS] {len(claims)} claims parsed")
                    for claim in claims[:2]:  # Show first 2 claims
                        print(f"      [c{claim['id']} | {claim['content'][:50]}... | / {claim['confidence']:.2f}]")
                
            except Exception as e:
                print(f"    [ERROR] {e}")
    
    # Generate failure diagnosis report
    print(f"\nFAILURE DIAGNOSIS SUMMARY")
    print("=" * 50)
    print(f"Total failures analyzed: {len(failure_analysis)}")
    
    if failure_analysis:
        # Analyze common failure patterns
        bracket_failures = sum(1 for f in failure_analysis if not f['contains_brackets'])
        format_failures = sum(1 for f in failure_analysis if not f['contains_c_format'])
        confidence_failures = sum(1 for f in failure_analysis if not f['contains_confidence'])
        
        print(f"Failures due to no brackets: {bracket_failures}")
        print(f"Failures due to no [c format: {format_failures}")
        print(f"Failures due to no confidence format: {confidence_failures}")
        
        # Show sample failure responses
        print(f"\nSAMPLE FAILURE RESPONSES:")
        for i, failure in enumerate(failure_analysis[:3]):
            print(f"\nFailure {i+1}: {failure['model']} - {failure['test_case']}")
            print(f"Response preview: {failure['response_preview']}")
    
    return failure_analysis

def main():
    """Main function"""
    try:
        failures = diagnose_failure_patterns()
        print(f"\nDiagnosis completed. Found {len(failures)} failure patterns to address.")
        return True
    except Exception as e:
        print(f"Diagnosis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)