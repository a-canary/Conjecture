#!/usr/bin/env python3
"""
Debug improvement cycle benchmark error
"""
import asyncio
import sys
import json
import subprocess
from pathlib import Path

# Set UTF-8 encoding for stdout
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "benchmarking"))

async def debug_benchmark():
    """Debug the benchmark step"""

    test_script = '''
import asyncio
import sys
import time
import json
from pathlib import Path
sys.path.insert(0, str(Path("src")))
sys.path.insert(0, str(Path("src/benchmarking")))

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

    return {
        "math_accuracy": 1.0 if math_correct else 0.0,
        "total_problems": 1,
        "correct_answers": 1 if math_correct else 0
    }

result = asyncio.run(quick_test())
print(json.dumps(result))
'''

    # Write test script
    test_file = Path(__file__).parent / "temp_debug_test.py"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_script)

    try:
        # Run test
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True,
            text=True,
            timeout=30
        )

        print("=== BENCHMARK DEBUG ===")
        print(f"Return code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")

        if result.returncode != 0:
            return {"success": False, "error": f"Test failed: {result.stderr}"}

        # Find last line of stdout (should be JSON)
        lines = result.stdout.strip().split('\n')
        json_line = lines[-1] if lines else ""

        print(f"\nLast line (JSON): {json_line}")

        try:
            benchmark_data = json.loads(json_line)
            print(f"\nParsed benchmark data: {benchmark_data}")
            return {"success": True, "data": benchmark_data}
        except json.JSONDecodeError as e:
            print(f"\nJSON decode error: {e}")
            return {"success": False, "error": "Could not parse benchmark results"}

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Benchmark timeout"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink()

if __name__ == "__main__":
    result = asyncio.run(debug_benchmark())
    print(f"\nFinal result: {result}")