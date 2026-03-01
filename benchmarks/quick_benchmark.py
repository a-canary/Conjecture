"""Quick benchmark - direct answer evaluation without DeepEval's LLM evaluator"""

import os
import sys
sys.path.insert(0, '/workspace')

from deepeval.models import GPTModel

# Test problems
MATH_PROBLEMS = [
    ("What is 15 + 27?", "42"),
    ("What is 8 * 7?", "56"),
    ("What is 144 / 12?", "12"),
    ("What is 25% of 80?", "20"),
    ("If x + 5 = 12, what is x?", "7"),
]

LOGIC_PROBLEMS = [
    ("If all dogs are mammals and Fido is a dog, is Fido a mammal? Answer yes or no.", "yes"),
    ("If it's raining, the ground is wet. The ground is wet. Is it definitely raining? Answer yes or no.", "no"),
    ("Complete: 2, 4, 6, 8, __", "10"),
]

def create_model():
    return GPTModel(
        model='deepseek-ai/DeepSeek-V3-0324',
        api_key=os.environ['CHUTES_API_KEY'],
        base_url='https://llm.chutes.ai/v1'
    )

def evaluate(model, problems, name, use_conjecture=False):
    correct = 0
    for question, expected in problems:
        if use_conjecture:
            prompt = f"""Reason step-by-step. Verify your answer.

{question}

Think carefully, then give your final answer."""
        else:
            prompt = question

        response, _ = model.generate(prompt)
        response_lower = response.lower().strip()
        expected_lower = expected.lower()

        if expected_lower in response_lower:
            correct += 1
            status = "✓"
        else:
            status = "✗"
        print(f"  {status} Q: {question[:40]}... Expected: {expected}, Got: {response[:50]}...")

    accuracy = correct / len(problems) * 100
    return accuracy

def main():
    print("Quick Benchmark (Direct Evaluation)")
    print("=" * 50)

    model = create_model()

    print("\n--- MATH (Baseline) ---")
    math_baseline = evaluate(model, MATH_PROBLEMS, "Math", use_conjecture=False)
    print(f"Accuracy: {math_baseline:.0f}%")

    print("\n--- MATH (Conjecture) ---")
    math_conjecture = evaluate(model, MATH_PROBLEMS, "Math", use_conjecture=True)
    print(f"Accuracy: {math_conjecture:.0f}%")

    print("\n--- LOGIC (Baseline) ---")
    logic_baseline = evaluate(model, LOGIC_PROBLEMS, "Logic", use_conjecture=False)
    print(f"Accuracy: {logic_baseline:.0f}%")

    print("\n--- LOGIC (Conjecture) ---")
    logic_conjecture = evaluate(model, LOGIC_PROBLEMS, "Logic", use_conjecture=True)
    print(f"Accuracy: {logic_conjecture:.0f}%")

    print("\n" + "=" * 50)
    print("SUMMARY")
    print(f"Math:  {math_baseline:.0f}% -> {math_conjecture:.0f}% ({math_conjecture - math_baseline:+.0f}pp)")
    print(f"Logic: {logic_baseline:.0f}% -> {logic_conjecture:.0f}% ({logic_conjecture - logic_baseline:+.0f}pp)")

if __name__ == "__main__":
    main()
