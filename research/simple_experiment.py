#!/usr/bin/env python3
"""
Simple Real Experiment Test
Uses REAL LLM calls only - no simulation
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
                print(f"Loaded test case: {test_case['id']}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return test_cases

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
                raise ValueError("CHUTES_API_KEY or PROVIDER_API_KEY not found in environment")
            
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
            "max_tokens": 1000,
            "temperature": 0.3
        }
        
        # Make request
        response = requests.post(endpoint, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        # Extract response text with GLM reasoning content support
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0].get("message", {})
            print(f"DEBUG: Message keys: {list(message.keys())}")

            # Check for reasoning_content first (GLM models)
            reasoning_content = message.get("reasoning_content")
            print(f"DEBUG: reasoning_content length: {len(reasoning_content) if reasoning_content else 'None'}")
            if reasoning_content:
                return reasoning_content.strip()

            # Fallback to standard content
            content = message.get("content")
            print(f"DEBUG: content length: {len(content) if content else 'None'}")
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

def run_real_simple_experiment():
    """Run simple experiment with REAL LLM calls only"""
    print("Running Simple Real Experiment")
    print("=" * 50)
    
    # Load environment
    env_vars = load_environment()
    print(f"Loaded {len(env_vars)} environment variables")
    
    # Verify API keys
    if not (env_vars.get('CHUTES_API_KEY') or env_vars.get('PROVIDER_API_KEY')):
        print("ERROR: CHUTES_API_KEY or PROVIDER_API_KEY not found")
        return None
    
    # Load test cases
    test_cases = load_test_cases()
    if not test_cases:
        print("No test cases found!")
        return None
    
    print(f"Found {len(test_cases)} test cases")
    
    # Define models to test
    models = [
        'lmstudio:ibm/granite-4-h-tiny',
        'chutes:zai-org/GLM-4.6'
    ]
    
    # Run experiment
    results = []
    start_time = datetime.now()
    
    for model in models:
        print(f"\nTesting model: {model}")
        model_results = []
        
        for test_case in test_cases[:1]:  # Test with first test case only
            print(f"  Running test case: {test_case['id']}")
            
            try:
                # Create simple prompt
                if 'question' in test_case:
                    prompt = test_case['question']
                elif 'task' in test_case:
                    prompt = test_case['task']
                else:
                    prompt = f"Please address: {test_case['id']}"
                
                # Make REAL LLM call
                print(f"    Making real LLM call...")
                response_start = time.time()
                response = make_real_llm_call(prompt, model, env_vars)
                response_time = time.time() - response_start
                
                print(f"    Got response in {response_time:.2f}s")
                print(f"    Response length: {len(response)} characters")
                
                result = {
                    'model': model,
                    'test_case_id': test_case['id'],
                    'prompt': prompt,
                    'response': response,
                    'response_time': response_time,
                    'response_length': len(response)
                }
                
                model_results.append(result)
                print(f"    [SUCCESS] Real response obtained")
                
            except Exception as e:
                print(f"    [ERROR] {e}")
                continue
        
        results.extend(model_results)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Save results
    if results:
        results_data = {
            'experiment_id': f'real_simple_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'models_tested': models,
            'test_cases_used': [tc['id'] for tc in test_cases[:1]],
            'results': results,
            'note': 'REAL LLM CALLS ONLY - No simulation used'
        }
        
        # Save to file
        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / f"{results_data['experiment_id']}.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        # Simple report
        print(f"\nExperiment Summary:")
        print(f"- Duration: {duration}")
        print(f"- Successful calls: {len(results)}")
        for result in results:
            print(f"  {result['model']}: {result['response_time']:.2f}s, {result['response_length']} chars")
        
        return results_data
    else:
        print("No successful results obtained")
        return None

def main():
    """Main function"""
    try:
        results = run_real_simple_experiment()
        if results:
            print("\n" + "=" * 50)
            print("Real experiment completed successfully!")
            print("All results are from actual LLM calls.")
            return True
        else:
            print("Experiment failed")
            return False
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)