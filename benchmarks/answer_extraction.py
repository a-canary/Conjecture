# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Robust Answer Extraction for Benchmarks
========================================

Handles multiple answer formats across different benchmark types:
- GSM8K: #### number format
- AIME/Math: boxed{answer}, "answer is X", numerical
- Multiple Choice: A/B/C/D letters
- Open-ended: Full response or key concepts

Key improvements:
1. Unified normalization (lowercase, strip whitespace, remove formatting)
2. Priority-ordered extraction patterns (most specific first)
3. Type-aware matching (numerical vs categorical)
4. Decimal/fraction support
5. Boxed answer detection (LaTeX \boxed{})
6. Fallback strategy for edge cases
"""

import re
from typing import Optional, Tuple, List
from enum import Enum


class AnswerType(Enum):
    """Types of answers we handle"""
    NUMERICAL = "numerical"
    MULTIPLE_CHOICE = "multiple_choice"
    CATEGORICAL = "categorical"
    OPEN_ENDED = "open_ended"


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison.
    Handles: whitespace, commas, case, common formatting
    """
    if not answer:
        return ""

    # Strip leading/trailing whitespace
    answer = answer.strip()

    # Remove common formatting characters (but preserve structure)
    answer = answer.replace("_", "")

    # Lowercase for categorical/text answers
    answer = answer.lower()

    return answer


def extract_numerical_answer(response: str) -> Optional[str]:
    """
    Extract numerical answer from response.
    Tries patterns in order of specificity.

    Patterns (in priority order):
    1. #### number (GSM8K format)
    2. \\boxed{answer} (LaTeX boxed)
    3. "answer is X" / "answer: X" with number
    4. "final answer: X"
    5. "result: X" / "result is X"
    6. Standalone number at end of sentence
    7. Last isolated number in response
    """
    if not response:
        return None

    # Pattern 1: GSM8K format "#### 42" or "#### 42.5"
    match = re.search(r'#+\s*(\-?[\d,]+(?:\.\d+)?)', response)
    if match:
        return match.group(1).replace(",", "")

    # Pattern 2: LaTeX boxed format
    match = re.search(r'\\boxed\{([^}]+)\}', response)
    if match:
        answer = match.group(1).strip()
        # Extract number from boxed content if it contains non-numeric
        num_match = re.search(r'(\-?[\d,]+(?:\.\d+)?)', answer)
        if num_match:
            return num_match.group(1).replace(",", "")
        return answer

    # Pattern 3: "Answer is X" / "Answer: X" with flexibility
    match = re.search(
        r'answer\s*(?:is|:|\s)\s*\$?(\-?[\d,]+(?:\.\d+)?)',
        response,
        re.IGNORECASE
    )
    if match:
        return match.group(1).replace(",", "")

    # Pattern 4: "The answer is X"
    match = re.search(
        r'the\s+answer\s+(?:is|:)\s*\$?(\-?[\d,]+(?:\.\d+)?)',
        response,
        re.IGNORECASE
    )
    if match:
        return match.group(1).replace(",", "")

    # Pattern 5: "Final answer: X" or "Final answer is X"
    match = re.search(
        r'final\s+answer\s*(?:is|:|\s)\s*\$?(\-?[\d,]+(?:\.\d+)?)',
        response,
        re.IGNORECASE
    )
    if match:
        return match.group(1).replace(",", "")

    # Pattern 6: "Result: X" / "Result is X"
    match = re.search(
        r'result\s*(?:is|:|\s)\s*\$?(\-?[\d,]+(?:\.\d+)?)',
        response,
        re.IGNORECASE
    )
    if match:
        return match.group(1).replace(",", "")

    # Pattern 7: "Therefore X" or "So X" at sentence end
    match = re.search(
        r'(?:therefore|thus|so)\s*(?:the\s+answer\s+is\s+)?\$?(\-?[\d,]+(?:\.\d+)?)',
        response,
        re.IGNORECASE
    )
    if match:
        return match.group(1).replace(",", "")

    # Pattern 8: Number after equals sign (equations)
    match = re.search(r'=\s*\$?(\-?[\d,]+(?:\.\d+)?)', response)
    if match:
        return match.group(1).replace(",", "")

    # Fallback: Find all numbers and return the last one
    # But filter out numbers that appear to be part of larger context
    numbers = re.findall(r'\-?[\d,]+(?:\.\d+)?', response)
    if numbers:
        # Return last number (most likely to be final answer)
        return numbers[-1].replace(",", "")

    return None


def extract_multiple_choice(response: str) -> Optional[str]:
    """
    Extract multiple choice answer (A, B, C, D, or 1, 2, 3, 4).

    Tries patterns in order:
    1. Explicit answer prefix: "Answer: A" or "The answer is B"
    2. Letter by itself at start/end of line
    3. First letter in first 100 chars
    """
    if not response:
        return None

    # Pattern 1: Explicit answer indicators
    match = re.search(
        r'(?:answer|choice|select)[:\s]+\s*([A-D])',
        response,
        re.IGNORECASE
    )
    if match:
        return match.group(1).upper()

    match = re.search(
        r'(?:the\s+)?answer\s+is\s+([A-D])',
        response,
        re.IGNORECASE
    )
    if match:
        return match.group(1).upper()

    # Pattern 2: Single letter on own line/start of response
    match = re.search(r'^([A-D])\b', response, re.MULTILINE)
    if match:
        return match.group(1).upper()

    # Pattern 3: Letter in parentheses
    match = re.search(r'\(([A-D])\)', response)
    if match:
        return match.group(1).upper()

    # Pattern 4: First isolated letter in first portion
    match = re.search(r'\b([A-D])\b', response[:200])
    if match:
        return match.group(1).upper()

    # Pattern 5: Single letter answer (handle numeric choices 1-4)
    match = re.search(r'answer[:\s]+\s*([1-4])', response, re.IGNORECASE)
    if match:
        choice_num = int(match.group(1))
        return chr(ord('A') + choice_num - 1)

    return None


def extract_answer(response: str, expected: str = None, answer_type: AnswerType = None) -> str:
    """
    Extract answer from model response with intelligent type detection.

    Args:
        response: Model's full response text
        expected: Expected answer (for type inference)
        answer_type: Explicitly specified answer type

    Returns:
        Extracted answer (normalized and cleaned)
    """
    if not response:
        return ""

    # Determine answer type if not specified
    if answer_type is None and expected:
        expected_normalized = normalize_answer(expected)

        # Type detection
        if expected_normalized in ['a', 'b', 'c', 'd']:
            answer_type = AnswerType.MULTIPLE_CHOICE
        elif re.match(r'^\-?[\d,]+(?:\.\d+)?$', expected_normalized):
            answer_type = AnswerType.NUMERICAL
        else:
            answer_type = AnswerType.CATEGORICAL
    elif answer_type is None:
        # Default: try numerical first, then multiple choice
        answer_type = AnswerType.NUMERICAL

    # Extract based on type
    if answer_type == AnswerType.NUMERICAL:
        extracted = extract_numerical_answer(response)
        return extracted if extracted else response.strip()[:50]

    elif answer_type == AnswerType.MULTIPLE_CHOICE:
        extracted = extract_multiple_choice(response)
        return extracted if extracted else response.strip()[:1].upper()

    else:
        # For categorical/open-ended, just return cleaned response
        return response.strip()[:200]


def check_answer_match(predicted: str, expected: str, answer_type: AnswerType = None) -> bool:
    """
    Check if predicted answer matches expected answer.
    Handles numerical precision, case-insensitivity, etc.

    Args:
        predicted: Extracted predicted answer
        expected: Expected correct answer
        answer_type: Type of answer for proper comparison

    Returns:
        True if answers match, False otherwise
    """
    if not predicted or not expected:
        return False

    # Normalize both answers
    pred_norm = normalize_answer(str(predicted))
    exp_norm = normalize_answer(str(expected))

    # Direct string match
    if pred_norm == exp_norm:
        return True

    # Determine type if not specified
    if answer_type is None:
        if exp_norm in ['a', 'b', 'c', 'd']:
            answer_type = AnswerType.MULTIPLE_CHOICE
        elif re.match(r'^\-?[\d,]+(?:\.\d+)?$', exp_norm):
            answer_type = AnswerType.NUMERICAL
        else:
            answer_type = AnswerType.CATEGORICAL

    # Type-specific matching
    if answer_type == AnswerType.NUMERICAL:
        try:
            pred_num = float(pred_norm.replace(",", ""))
            exp_num = float(exp_norm.replace(",", ""))
            # Allow small floating point differences
            return abs(pred_num - exp_num) < 0.01
        except (ValueError, AttributeError):
            return False

    elif answer_type == AnswerType.MULTIPLE_CHOICE:
        # Both should be single letters
        return len(pred_norm) > 0 and len(exp_norm) > 0 and pred_norm[0] == exp_norm[0]

    else:
        # For categorical, exact match after normalization
        return pred_norm == exp_norm


def analyze_extraction_quality(responses: List[Tuple[str, str, str]]) -> dict:
    """
    Analyze quality of answer extraction across a batch.

    Args:
        responses: List of (response_text, expected_answer, answer_type_str)

    Returns:
        Quality metrics dict
    """
    metrics = {
        "total": len(responses),
        "correct": 0,
        "extraction_failures": 0,
        "type_misdetections": 0,
        "extraction_accuracy": 0.0
    }

    for response_text, expected, answer_type_str in responses:
        try:
            answer_type = AnswerType[answer_type_str.upper()] if answer_type_str else None
            extracted = extract_answer(response_text, expected, answer_type)

            if not extracted:
                metrics["extraction_failures"] += 1
                continue

            if check_answer_match(extracted, expected, answer_type):
                metrics["correct"] += 1
        except Exception as e:
            metrics["extraction_failures"] += 1

    metrics["extraction_accuracy"] = round(
        100 * metrics["correct"] / max(1, metrics["total"]),
        2
    )

    return metrics


# Backward compatibility wrappers for existing code
def extract_answer_deepeval_compatible(response: str, expected: str) -> str:
    """
    Compatibility wrapper for deepeval_suite.py
    """
    # Infer type from expected
    answer_type = None
    if expected and expected.lower() in ['a', 'b', 'c', 'd']:
        answer_type = AnswerType.MULTIPLE_CHOICE
    elif expected and re.match(r'^\-?[\d,]+(?:\.\d+)?$', normalize_answer(expected)):
        answer_type = AnswerType.NUMERICAL

    return extract_answer(response, expected, answer_type)


def extract_answer_benchmark_compatible(response: str, expected: str) -> str:
    """
    Compatibility wrapper for benchmark_framework.py
    """
    return extract_numerical_answer(response) or response.strip()[:50]


if __name__ == "__main__":
    # Test cases
    test_cases = [
        # GSM8K format
        ("The answer is #### 42", "42", "numerical"),
        ("Calculation: 5 + 10 + 27 = #### 42", "42", "numerical"),

        # Boxed format
        ("Therefore \\boxed{17}", "17", "numerical"),
        ("The answer is \\boxed{\\frac{5}{6}}", "5/6", "numerical"),

        # Answer is X
        ("The answer is 42", "42", "numerical"),
        ("Answer: 42", "42", "numerical"),
        ("answer is 3.14", "3.14", "numerical"),

        # Multiple choice
        ("The answer is B", "B", "multiple_choice"),
        ("Answer: A", "A", "multiple_choice"),
        ("(C) is correct", "C", "multiple_choice"),

        # Edge cases
        ("The ball is $0.05. The bat is $1.05", "0.05", "numerical"),
        ("Numbers: 10, 20, 30, 42", "42", "numerical"),

        # Negative numbers
        ("Answer: -5", "-5", "numerical"),

        # With commas
        ("The answer is 1,000", "1000", "numerical"),
    ]

    print("Testing Answer Extraction")
    print("=" * 60)

    for response, expected, ans_type in test_cases:
        extracted = extract_answer(response, expected, AnswerType[ans_type.upper()])
        matches = check_answer_match(extracted, expected, AnswerType[ans_type.upper()])
        status = "✓" if matches else "✗"
        print(f"{status} Response: {response[:40]:40s} | Expected: {expected:10s} | Got: {extracted}")

    print("=" * 60)
