#!/usr/bin/env python3
"""
Simple benchmark test
"""
import subprocess
import sys
import json
from pathlib import Path

# Create test script
test_script = '''
import asyncio
import sys
import time
import json

async def quick_test():
    # Test with simple math and logic problems
    math_prompt = """You are Conjecture, an adaptive AI system that matches reasoning approach to problem domain.

DOMAIN-ADAPTIVE APPROACH:
For MATHEMATICAL problems:
- Focus on calculation accuracy and step-by-step reasoning
- Use mathematical knowledge and problem-solving strategies
- Work through calculations systematically and verify results

PROBLEM:
What is 17 * 24?

Please provide the answer to this problem."""

    # Simulate testing (would use actual model in real implementation)
    print("Testing math problem...")
    time.sleep(0.1)  # Simulate API call
    math_response = "The answer is 408.\\n\\n17 * 24 = 408"
    math_correct = "408" in math_response

    print(f"Math: {'PASS' if math_correct else 'FAIL'}")

    result = {
        "math_accuracy": 1.0 if math_correct else 0.0,
        "total_problems": 1,
        "correct_answers": 1 if math_correct else 0
    }

    return result

result = asyncio.run(quick_test())
print(json.dumps(result))
'''

# Write test script
test_file = Path("temp_test.py")
with open(test_file, 'w') as f:
    f.write(test_script)

# Run test
try:
    result = subprocess.run(
        [sys.executable, str(test_file)],
        capture_output=True,
        text=True,
        timeout=30
    )

    print("Return code:", result.returncode)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    if result.returncode == 0:
        try:
            data = json.loads(result.stdout.strip())
            print("\nParsed result:", data)
        except json.JSONDecodeError as e:
            print("JSON decode error:", e)

finally:
    # Clean up
    if test_file.exists():
        test_file.unlink()