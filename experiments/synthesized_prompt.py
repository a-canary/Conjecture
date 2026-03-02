#!/usr/bin/env python3
"""
Synthesized Optimal Prompt - Combined from Top 3 Performers

Based on benchmark results:
1. v01_baseline: 82.5% (simple, direct)
2. v16_final_first: 81.25% (answer-first pattern)
3. v10_structured: 67.5% (85% math - JSON format helps extraction)

Synthesis Strategy:
- Keep baseline simplicity (proven winner)
- Add answer-first framing (v16 insight: stating answer first helps)
- Use structured output hint without enforcing JSON (middle ground)
- Clear format specification at end

Key Finding: Simple > Complex for llama3.1-8b
"""

SYNTHESIZED_PROMPTS = {
    "math": {
        # Combines: baseline simplicity + answer-first hint + clear format
        "template": "Q: {q}\n\nAnswer = ",
        "rationale": "Baseline format with direct answer expectation. No CoT, no complex instructions."
    },
    "logic": {
        # Combines: baseline simplicity + clear options
        "template": "Q: {q}\n\nAnswer (Yes/No/Cannot determine): ",
        "rationale": "Baseline format with explicit valid options. Direct extraction."
    },
    "gsm8k": {
        # Combines: baseline structure + answer-first emphasis
        "template": "Problem: {question}\n\nSolve and state final answer:\n#### ",
        "rationale": "GSM8K format with direct answer prompt. Skip reasoning instructions."
    },
    "mmlu": {
        # Combines: baseline format + clear letter expectation
        "template": "{question}\n\n{choices}\n\nAnswer: ",
        "rationale": "Clean format, no role-playing, no CoT. Just question and answer."
    },
}

# Anti-patterns discovered (what NOT to do):
ANTI_PATTERNS = {
    "cot_explicit": "Let's think step by step - 2.5% (WORST)",
    "expert_role": "You are an expert... - 13.75%",
    "decompose": "Break into steps - 8.75%",
    "verify": "Double-check your answer - 10%",
    "units": "Track units through steps - 5%",
    "meta": "What strategy should you use? - 16.25%",
}

# What works:
PATTERNS_THAT_WORK = {
    "direct_simple": "Q: {q}\\nAnswer: - 82.5% (BEST)",
    "answer_first": "Answer = (then explain) - 81.25%",
    "structured_math": "JSON format for math extraction - 85% math accuracy",
    "confidence_request": "Rate your confidence - 43.75% (helps some)",
}


def get_optimal_prompt(task_type: str, **kwargs) -> str:
    """Get the synthesized optimal prompt for a task type."""
    if task_type not in SYNTHESIZED_PROMPTS:
        return SYNTHESIZED_PROMPTS["math"]["template"].format(**kwargs)
    return SYNTHESIZED_PROMPTS[task_type]["template"].format(**kwargs)


if __name__ == "__main__":
    print("Synthesized Optimal Prompts")
    print("=" * 60)
    for task, info in SYNTHESIZED_PROMPTS.items():
        print(f"\n{task.upper()}:")
        print(f"  Template: {info['template']!r}")
        print(f"  Rationale: {info['rationale']}")

    print("\n" + "=" * 60)
    print("Anti-Patterns (AVOID):")
    for pattern, result in ANTI_PATTERNS.items():
        print(f"  ❌ {pattern}: {result}")

    print("\n" + "=" * 60)
    print("Patterns That Work:")
    for pattern, result in PATTERNS_THAT_WORK.items():
        print(f"  ✅ {pattern}: {result}")
