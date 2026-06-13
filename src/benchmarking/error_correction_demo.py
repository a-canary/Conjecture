# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Error Correction Prompts Demonstration

Shows how to integrate error correction prompts into the reasoning system.
This is a lightweight alternative to full re-generation that helps models
catch and correct their own mistakes.

Key advantage: Minimal overhead compared to full regeneration
Strategy: Prompt model to reconsider answer when confidence is low
"""

import asyncio
from src.agent.prompt_system import PromptSystem, ProblemType


async def demonstrate_error_correction():
    """Demonstrate error correction prompt generation across problem types."""

    prompt_system = PromptSystem()

    # Example problems with different types
    examples = [
        {
            "problem": "If a rectangle has length 8 and width 5, what is its area?",
            "initial_response": "The area is 8 + 5 = 13 square units.",
            "type": ProblemType.MATHEMATICAL,
            "explanation": "Common error: addition instead of multiplication",
        },
        {
            "problem": "All humans are mortal. Socrates is human. Is Socrates mortal?",
            "initial_response": "Not necessarily. Being human doesn't guarantee mortality.",
            "type": ProblemType.LOGICAL,
            "explanation": "Common error: ignoring logical deduction",
        },
        {
            "problem": "First prepare the ingredients. Then mix them. Finally bake.",
            "initial_response": "Mix ingredients, then prepare them, then bake.",
            "type": ProblemType.SEQUENTIAL,
            "explanation": "Common error: reversing sequential order",
        },
    ]

    print("=" * 80)
    print("ERROR CORRECTION PROMPTS DEMONSTRATION")
    print("=" * 80)
    print()

    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}: {example['type'].value.upper()}")
        print(f"Issue: {example['explanation']}")
        print("-" * 80)

        # Show the correction prompt
        correction_prompt = prompt_system.get_error_correction_prompt(
            problem=example["problem"],
            initial_response=example["initial_response"],
            problem_type=example["type"],
        )

        print(f"PROBLEM:\n{example['problem']}\n")
        print(f"INITIAL ANSWER:\n{example['initial_response']}\n")
        print("ERROR CORRECTION PROMPT:")
        print(correction_prompt)
        print()

    print("\n" + "=" * 80)
    print("QUICK ERROR CORRECTION REMINDERS")
    print("=" * 80)

    # Show quick error correction for each type
    problem_types = [
        ProblemType.MATHEMATICAL,
        ProblemType.LOGICAL,
        ProblemType.SEQUENTIAL,
        ProblemType.SCIENTIFIC,
        ProblemType.DECOMPOSITION,
        ProblemType.GENERAL,
    ]

    for ptype in problem_types:
        quick_msg = prompt_system.get_quick_error_correction(ptype)
        print(f"\n{ptype.value.upper()}:")
        print(f"  {quick_msg}")

    print("\n" + "=" * 80)
    print("CONFIDENCE-BASED ERROR CORRECTION TRIGGERING")
    print("=" * 80)

    # Demonstrate confidence-based triggering
    test_confidences = [0.4, 0.65, 0.75, 0.9, 0.95]

    print("\nError correction threshold: 0.7")
    for conf in test_confidences:
        should_correct = prompt_system.should_trigger_error_correction(conf)
        status = "TRIGGER" if should_correct else "SKIP"
        print(f"  Confidence {conf:.2f}: {status} error correction")

    print("\n" + "=" * 80)
    print("ENHANCEMENT STATUS")
    print("=" * 80)

    status = prompt_system.get_enhancement_status()
    print("\nAll enhancements:")
    for name, enabled in status.items():
        status_str = "ENABLED" if enabled else "DISABLED"
        print(f"  {name.replace('_', ' ').title()}: {status_str}")

    print("\n" + "=" * 80)
    print("USAGE PATTERNS")
    print("=" * 80)

    print("""
1. INLINE ERROR CORRECTION (lightweight):
   Include quick reminder in system prompt for low-confidence answers:

   system_prompt = f'''
   {base_prompt}

   SAFETY CHECK:
   {prompt_system.get_quick_error_correction(problem_type)}
   '''

2. FULL ERROR CORRECTION (when confidence is below threshold):
   Generate detailed correction prompt for reconsideration:

   if prompt_system.should_trigger_error_correction(confidence):
       correction = prompt_system.get_error_correction_prompt(
           problem, initial_answer, problem_type
       )
       # Send correction prompt to model for reconsideration

3. AUTOMATIC CORRECTION WORKFLOW:
   - Get initial answer with confidence score
   - Check if confidence < 0.7
   - If yes, generate correction prompt
   - Ask model to reconsider and provide corrected answer
   - Compare initial vs corrected answer
   - Return best result with higher confidence
""")

    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(demonstrate_error_correction())
