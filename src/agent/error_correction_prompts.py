# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Error Correction Enhancement Module

Provides lightweight error correction prompts as an alternative to full re-generation.
This module implements domain-specific error checking guidance that helps the model
reconsider and correct its answers without requiring a full regeneration pass.

Key strategy:
- If first answer seems potentially wrong, prompt for reconsideration
- Domain-specific error patterns (mathematical, logical, sequential, etc.)
- Lightweight prompt-based correction vs expensive re-inference
- Integrates with the PromptSystem for enhanced reasoning
"""

from enum import Enum
from typing import Optional, Dict, Any


class ProblemType(Enum):
    MATHEMATICAL = "mathematical"
    LOGICAL = "logical"
    GENERAL = "general"
    SCIENTIFIC = "scientific"
    SEQUENTIAL = "sequential"
    DECOMPOSITION = "decomposition"


def get_error_correction_prompt(
    problem: str,
    initial_response: str,
    problem_type: ProblemType
) -> str:
    """
    Generate domain-specific error correction prompt.

    This lightweight prompt asks the model to reconsider its answer for common
    domain-specific errors without requiring full re-generation.

    Args:
        problem: The original problem statement
        initial_response: The model's initial answer attempt
        problem_type: The type of problem (mathematical, logical, etc.)

    Returns:
        Formatted error correction prompt with domain-specific guidance
    """

    correction_guidance = {
        ProblemType.MATHEMATICAL: """
ERROR CORRECTION STEP - Mathematical Check:
Before finalizing your answer, consider if your solution may contain errors:

1. CALCULATION REVIEW: Did you double-check arithmetic?
   - Verify order of operations (PEMDAS/BODMAS)
   - Recalculate key steps
   - Check for sign errors (-/+) or decimal placement mistakes

2. LOGIC VERIFICATION: Did you interpret the problem correctly?
   - Does your answer match what was asked?
   - Did you use correct units?
   - Are assumptions valid?

3. ALTERNATIVE CHECK: Can you verify using a different method?
   - Try working backwards from your answer
   - Use estimation to check if answer is reasonable
   - Check for off-by-one errors

If you spot an error, provide the corrected answer. If your answer appears correct, state "CONFIRMED" and restate the answer.""",

        ProblemType.LOGICAL: """
ERROR CORRECTION STEP - Logic Check:
Before finalizing your conclusion, reconsider if your reasoning may contain errors:

1. PREMISE VERIFICATION: Are your starting assumptions sound?
   - Did you misread any facts?
   - Are there hidden assumptions?
   - Did you consider all given information?

2. REASONING CHAIN: Is each logical step valid?
   - Does each conclusion follow from premises?
   - Are there logical fallacies?
   - Could a different interpretation be valid?

3. CONCLUSION VALIDITY: Is your conclusion the only one justified?
   - Are there counterexamples?
   - Could the opposite be true?
   - Does it logically follow from all steps?

If you find a logical flaw, provide the corrected reasoning. If valid, state "CONFIRMED" and restate your conclusion.""",

        ProblemType.SEQUENTIAL: """
ERROR CORRECTION STEP - Sequence Check:
Before finalizing your step sequence, reconsider if any steps are wrong or out of order:

1. STEP ORDER: Are steps in the correct sequence?
   - Is each step logically before the next?
   - Are prerequisites met?
   - Could any steps be reordered?

2. STEP COMPLETENESS: Are all necessary steps included?
   - Did you skip any required steps?
   - Are there hidden prerequisites?
   - Is the progression to the final answer complete?

3. DEPENDENCY VERIFICATION: Does each step properly support the next?
   - Would skipping a step break the sequence?
   - Are intermediate results used correctly?
   - Is the final step actually final?

If you spot a sequencing error, provide the corrected sequence. If correct, state "CONFIRMED" and restate the steps.""",

        ProblemType.SCIENTIFIC: """
ERROR CORRECTION STEP - Scientific Validity Check:
Before finalizing your analysis, reconsider if your scientific reasoning may contain errors:

1. METHODOLOGY: Did you apply the scientific method correctly?
   - Are hypotheses clearly testable?
   - Is the experimental design sound?
   - Are variables properly controlled?

2. EVIDENCE INTERPRETATION: Did you correctly interpret the data?
   - Could the data support alternative conclusions?
   - Are you confusing correlation with causation?
   - Are there biases in your interpretation?

3. CONCLUSION SUPPORT: Is your conclusion justified by evidence?
   - Does evidence actually support this conclusion?
   - Are alternative explanations ruled out?
   - Are limitations acknowledged?

If you find a scientific flaw, provide the corrected analysis. If sound, state "CONFIRMED" and restate your conclusion.""",

        ProblemType.DECOMPOSITION: """
ERROR CORRECTION STEP - Decomposition Check:
Before finalizing your decomposed analysis, reconsider if components or their integration may be wrong:

1. COMPONENT IDENTIFICATION: Did you identify all major components?
   - Are there missing components?
   - Are components properly separated?
   - Is the decomposition complete?

2. COMPONENT ANALYSIS: Did you correctly analyze each component?
   - Is each component analysis thorough?
   - Did you miss interactions between components?
   - Are component relationships clear?

3. INTEGRATION: Did you properly integrate component analyses?
   - Do component analyses combine logically?
   - Are interactions properly accounted for?
   - Does the whole equal the sum of parts?

If you find a decomposition error, provide the corrected analysis. If complete, state "CONFIRMED" and restate your answer.""",

        ProblemType.GENERAL: """
ERROR CORRECTION STEP - Answer Review:
Before finalizing your answer, take a moment to reconsider:

1. UNDERSTANDING CHECK: Did you fully understand the question?
   - Is your answer addressing exactly what was asked?
   - Did you miss any parts of the question?
   - Could the question be interpreted differently?

2. CONTENT VERIFICATION: Is your answer accurate and complete?
   - Are there any obvious errors or contradictions?
   - Does your reasoning make sense?
   - Are all relevant points covered?

3. SENSE CHECK: Does your answer make logical sense?
   - Is it reasonable in context?
   - Could there be a better answer?
   - Have you made unfounded assumptions?

If you find an error, provide the corrected answer. If your answer appears sound, state "CONFIRMED" and restate it.""",
    }

    guidance = correction_guidance.get(
        problem_type, correction_guidance[ProblemType.GENERAL]
    )

    return f"""
{guidance}

Original problem: {problem}
Your initial answer: {initial_response}"""


def get_quick_error_correction(
    problem_type: ProblemType,
    confidence_score: Optional[float] = None
) -> str:
    """
    Get a quick, inline error correction reminder.

    Minimal version for lower-confidence answers. Can be injected into
    system prompts as a light touch error correction mechanism.

    Args:
        problem_type: The type of problem
        confidence_score: Optional confidence level (0-1)

    Returns:
        Brief error correction reminder
    """

    quick_corrections = {
        ProblemType.MATHEMATICAL: "If your first answer seems wrong, reconsider: did you verify the calculation step-by-step and check for arithmetic errors?",
        ProblemType.LOGICAL: "If your conclusion seems questionable, reconsider: are all premises valid and does your reasoning logically follow?",
        ProblemType.SEQUENTIAL: "If the sequence seems off, reconsider: are all steps in correct order and are prerequisites satisfied?",
        ProblemType.SCIENTIFIC: "If your analysis seems uncertain, reconsider: is it based on evidence and have you considered alternative explanations?",
        ProblemType.DECOMPOSITION: "If your breakdown seems incomplete, reconsider: have you identified all components and properly integrated them?",
        ProblemType.GENERAL: "If your answer seems wrong, reconsider: did you fully understand the question and address all parts?",
    }

    return quick_corrections.get(
        problem_type, quick_corrections[ProblemType.GENERAL]
    )


def should_trigger_error_correction(
    confidence: float,
    threshold: float = 0.7,
    response_length: Optional[int] = None,
    contains_uncertainty_markers: bool = False
) -> bool:
    """
    Determine if error correction should be triggered.

    Simple heuristic to decide when to suggest answer reconsideration.
    Based on confidence score and other signals.

    Args:
        confidence: Model's confidence score (0-1)
        threshold: Confidence threshold below which to trigger (default 0.7)
        response_length: Length of response (optional signal)
        contains_uncertainty_markers: If response contains "maybe", "might", etc.

    Returns:
        True if error correction should be triggered
    """

    # Primary signal: low confidence
    if confidence < threshold:
        return True

    # Secondary signals
    if contains_uncertainty_markers:
        return True

    # Response too short might indicate incomplete thinking
    if response_length is not None and response_length < 50:
        return True

    return False


class ErrorCorrectionConfig:
    """Configuration for error correction behavior."""

    def __init__(
        self,
        enabled: bool = True,
        confidence_threshold: float = 0.7,
        apply_to_all_domains: bool = False,
        target_domains: Optional[list] = None,
        max_correction_attempts: int = 1,
    ):
        """
        Initialize error correction configuration.

        Args:
            enabled: Whether error correction is enabled
            confidence_threshold: Trigger error correction below this confidence
            apply_to_all_domains: Apply to all problem types
            target_domains: List of specific problem types to apply to
            max_correction_attempts: Maximum number of correction attempts
        """
        self.enabled = enabled
        self.confidence_threshold = confidence_threshold
        self.apply_to_all_domains = apply_to_all_domains
        self.target_domains = target_domains or [
            ProblemType.MATHEMATICAL,
            ProblemType.LOGICAL,
            ProblemType.SEQUENTIAL,
        ]
        self.max_correction_attempts = max_correction_attempts


def format_error_correction_prompt(
    guidance_text: str,
    problem: str,
    initial_response: str,
    attempt_number: int = 1
) -> str:
    """
    Format complete error correction prompt with attempt tracking.

    Args:
        guidance_text: Domain-specific guidance text
        problem: Original problem
        initial_response: Model's initial answer
        attempt_number: Which correction attempt this is

    Returns:
        Formatted prompt ready to send to model
    """

    prefix = f"CORRECTION ATTEMPT #{attempt_number}:\n" if attempt_number > 1 else ""

    return f"""{prefix}
{guidance_text}

Original problem: {problem}
Your previous answer: {initial_response}

Please review for errors and provide a corrected answer if needed."""
