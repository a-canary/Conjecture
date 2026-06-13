# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Self-Verification Module (Phase 4, Step 4.1)

Implements self-verification claims that catch errors before submitting answers.
Goal: Catch 20%+ of errors before final answer.

Strategy:
1. Generate initial answer
2. Create verification claim asking model to check the answer
3. If verification finds error, regenerate
4. Track verification effectiveness
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class VerificationResult(str, Enum):
    """Result of self-verification check."""
    CORRECT = "correct"      # Answer verified as correct
    ERROR_FOUND = "error"    # Error detected, needs correction
    UNCERTAIN = "uncertain"  # Verification inconclusive
    SKIPPED = "skipped"      # Verification not performed


@dataclass
class VerificationStats:
    """Track self-verification effectiveness."""
    total_verified: int = 0
    errors_caught: int = 0
    false_positives: int = 0  # Thought wrong but was right
    true_positives: int = 0   # Caught actual error
    true_negatives: int = 0   # Correctly verified as right
    false_negatives: int = 0  # Missed actual error

    @property
    def catch_rate(self) -> float:
        """Percentage of errors caught by verification."""
        actual_errors = self.true_positives + self.false_negatives
        if actual_errors == 0:
            return 0.0
        return self.true_positives / actual_errors

    @property
    def precision(self) -> float:
        """When verification says error, how often is it right?"""
        flagged = self.true_positives + self.false_positives
        if flagged == 0:
            return 0.0
        return self.true_positives / flagged

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_verified": self.total_verified,
            "errors_caught": self.errors_caught,
            "catch_rate": f"{self.catch_rate:.1%}",
            "precision": f"{self.precision:.1%}",
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
        }


@dataclass
class VerificationConfig:
    """Configuration for self-verification."""
    enabled: bool = True
    verification_prompt_template: str = """Review this answer for errors:

Question: {question}
Proposed Answer: {answer}
Reasoning: {reasoning}

Check for:
1. Calculation errors
2. Logic mistakes
3. Misread question
4. Unit errors

If you find an error, explain it briefly and provide the correct answer.
If the answer is correct, respond with "VERIFIED: [answer]"
"""
    max_retries: int = 1  # How many times to retry on error
    confidence_threshold: float = 0.7  # Below this, always verify


class SelfVerifier:
    """
    Self-verification system for catching errors before final answer.

    Phase 4 Goal: Catch 20%+ of errors to improve GSM8K 50%→60%.
    """

    def __init__(self, config: Optional[VerificationConfig] = None):
        self.config = config or VerificationConfig()
        self.stats = VerificationStats()

    def create_verification_prompt(
        self,
        question: str,
        answer: str,
        reasoning: str = ""
    ) -> str:
        """Create a verification prompt for the given answer."""
        return self.config.verification_prompt_template.format(
            question=question,
            answer=answer,
            reasoning=reasoning or "Not provided"
        )

    def parse_verification_response(
        self,
        response: str
    ) -> Tuple[VerificationResult, Optional[str]]:
        """
        Parse verification response to determine if error was found.

        Returns:
            Tuple of (result, corrected_answer if error found)
        """
        response = response.strip()

        # Check for explicit verification
        if response.upper().startswith("VERIFIED"):
            # Extract the verified answer
            parts = response.split(":", 1)
            verified_answer = parts[1].strip() if len(parts) > 1 else None
            return VerificationResult.CORRECT, verified_answer

        # Check for error indicators
        error_indicators = [
            "error", "mistake", "incorrect", "wrong",
            "should be", "correct answer is", "actually"
        ]

        response_lower = response.lower()
        found_error = any(ind in response_lower for ind in error_indicators)

        if found_error:
            # Try to extract corrected answer
            corrected = self._extract_corrected_answer(response)
            return VerificationResult.ERROR_FOUND, corrected

        # Uncertain - verification didn't clearly indicate either way
        return VerificationResult.UNCERTAIN, None

    def _extract_corrected_answer(self, response: str) -> Optional[str]:
        """Extract corrected answer from verification response."""
        # Look for common patterns
        patterns = [
            "correct answer is ",
            "should be ",
            "the answer is ",
            "= ",
        ]

        response_lower = response.lower()
        for pattern in patterns:
            if pattern in response_lower:
                idx = response_lower.find(pattern) + len(pattern)
                # Extract until end of line or punctuation
                remaining = response[idx:].strip()
                # Take first word/number
                answer = remaining.split()[0] if remaining.split() else None
                if answer:
                    return answer.rstrip(".,!?")

        return None

    def update_stats(
        self,
        verification_result: VerificationResult,
        original_was_correct: bool,
        corrected_was_correct: Optional[bool] = None
    ) -> None:
        """Update statistics based on verification outcome."""
        self.stats.total_verified += 1

        if verification_result == VerificationResult.ERROR_FOUND:
            if not original_was_correct:
                # True positive: caught actual error
                self.stats.true_positives += 1
                self.stats.errors_caught += 1
            else:
                # False positive: flagged correct answer as wrong
                self.stats.false_positives += 1

        elif verification_result == VerificationResult.CORRECT:
            if original_was_correct:
                # True negative: correctly verified as right
                self.stats.true_negatives += 1
            else:
                # False negative: missed actual error
                self.stats.false_negatives += 1

    def should_verify(self, confidence: float) -> bool:
        """Determine if answer should be verified based on confidence."""
        if not self.config.enabled:
            return False
        return confidence < self.config.confidence_threshold

    def get_effectiveness_report(self) -> str:
        """Generate report on verification effectiveness."""
        stats = self.stats.to_dict()
        lines = [
            "Self-Verification Effectiveness:",
            f"  Total verified: {stats['total_verified']}",
            f"  Errors caught: {stats['errors_caught']}",
            f"  Catch rate: {stats['catch_rate']}",
            f"  Precision: {stats['precision']}",
            "",
            "  Confusion matrix:",
            f"    True positives (caught errors): {stats['true_positives']}",
            f"    False positives (wrong flags): {stats['false_positives']}",
            f"    True negatives (verified OK): {stats['true_negatives']}",
            f"    False negatives (missed errors): {stats['false_negatives']}",
        ]
        return "\n".join(lines)


# Simplified prompts for different question types
VERIFICATION_PROMPTS = {
    "math": """Check this math answer:
Q: {question}
A: {answer}

Verify the calculation step by step. If wrong, give correct answer.
If correct, say "VERIFIED: {answer}" """,

    "logic": """Check this logic answer:
Q: {question}
A: {answer}

Verify the logical reasoning. If flawed, explain and correct.
If valid, say "VERIFIED: {answer}" """,

    "science": """Check this science answer:
Q: {question}
A: {answer}

Verify against known facts. If incorrect, give correct answer.
If correct, say "VERIFIED: {answer}" """,
}


def get_verification_prompt(
    domain: str,
    question: str,
    answer: str
) -> str:
    """Get domain-appropriate verification prompt."""
    template = VERIFICATION_PROMPTS.get(domain, VERIFICATION_PROMPTS["math"])
    return template.format(question=question, answer=answer)
