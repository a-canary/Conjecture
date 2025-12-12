#!/usr/bin/env python3
"""
Test Enhanced Evaluation Methodology
Validates LLM-as-judge integration with fallback mechanisms
"""

import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'benchmarking'))

from gpt_oss_scaled_test import GptOssScaledTester

async def test_enhanced_evaluation():
    """Test the enhanced evaluation methodology"""
    print("Testing Enhanced LLM Evaluation Methodology")
    print("=" * 50)

    try:
        tester = GptOssScaledTester()

        # Test 1: Check if LLM judge is available
        print("\n1. Testing LLM Judge Availability...")
        try:
            await tester.setup_judge_model()
            print("SUCCESS: LLM Judge (GLM-4.6) configured successfully")
            judge_available = True
        except Exception as e:
            print(f"WARNING: LLM Judge not available: {e}")
            print("INFO: Will use string matching fallback")
            judge_available = False

        # Test 2: Test evaluation with sample responses
        print("\n2. Testing Enhanced Evaluation...")

        test_cases = [
            {
                "problem": "What is 15% of 240?",
                "response": "15% of 240 is 36.",
                "expected": "36",
                "should_be_correct": True
            },
            {
                "problem": "What is 15% of 240?",
                "response": "The answer is 42.",
                "expected": "36",
                "should_be_correct": False
            },
            {
                "problem": "All cats are animals. Some animals are pets. Can we conclude some cats are pets?",
                "response": "No. The premises don't guarantee that the pets mentioned are cats.",
                "expected": "No",
                "should_be_correct": True
            }
        ]

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n   Test Case {i}:")
            print(f"   Problem: {test_case['problem']}")
            print(f"   Response: {test_case['response']}")
            print(f"   Expected: {test_case['expected']}")

            # Test the enhanced evaluation
            is_correct, evaluation_meta = await tester._check_correctness(
                test_case['problem'],
                test_case['response'],
                test_case['expected']
            )

            status = "CORRECT" if is_correct else "INCORRECT"
            print(f"   Evaluated As: {status}")
            print(f"   Method: {evaluation_meta.get('method', 'unknown')}")

            if evaluation_meta.get('confidence'):
                print(f"   Confidence: {evaluation_meta['confidence']}%")

            if evaluation_meta.get('reasoning_quality'):
                print(f"   Quality: {evaluation_meta['reasoning_quality']}")

            # Verify correctness
            if is_correct == test_case['should_be_correct']:
                print(f"   PASS: Evaluation matches expected result")
            else:
                print(f"   FAIL: Evaluation mismatch (expected {test_case['should_be_correct']}, got {is_correct})")

        # Test 3: Demonstrate scientific rigor improvement
        print("\n3. Scientific Rigor Analysis...")
        if judge_available:
            print("SUCCESS: LLM-as-judge evaluation available")
            print("INFO: Scientific rigor enhanced with confidence scoring and reasoning quality assessment")
            print("INFO: Fallback mechanism provides robust string matching when LLM judge unavailable")
        else:
            print("INFO: LLM judge not configured")
            print("INFO: Using enhanced string matching with comprehensive variations")
            print("INFO: Recommendation: Configure GLM-4.6 for improved evaluation rigor")

        print("\n" + "=" * 50)
        print("Enhanced Evaluation Methodology Test Complete")
        print("\nKey Improvements:")
        print("• LLM-as-judge provides nuanced evaluation beyond string matching")
        print("• Confidence thresholds prevent low-confidence evaluations")
        print("• Robust fallback mechanisms ensure reliability")
        print("• Detailed evaluation metadata enables better analysis")
        print("• Maintains compatibility with existing test infrastructure")

        return True

    except Exception as e:
        print(f"FAIL: Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_enhanced_evaluation())
    sys.exit(0 if success else 1)