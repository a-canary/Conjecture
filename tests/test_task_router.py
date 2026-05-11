"""
Tests for Task-Type Router (O-0009).

O-0009 mandates task-type routing in production:
- REASONING (hard multi-step): route to three-prompt (70B+ models)
- RECALL (MMLU/commonsense): route to cot_lite (lightweight)
- MATH (competition math): route to specialized prompt

Tests classify_query() accuracy on labeled queries, and verifies
evaluate() wires routing correctly.

Gate: classify_query accuracy ≥90% on held-out 20-query labeled set.
"""

import pytest
from src.agent.task_router import classify_query, QueryType


# ------------------------------------------------------------------
# Gold-standard labeled queries for accuracy measurement
# ------------------------------------------------------------------
REASONING_QUERIES = [
    # Multi-step deduction
    "If all cats are mammals, and some mammals are pets, can some cats be pets?",
    # Logical chain
    "Given that A implies B, and B implies C, does A imply C? Prove your answer.",
    # Abstract reasoning
    "What is the next number in the sequence: 2, 6, 12, 20, 30? Show your work.",
    # BBH-style
    "John went to the store. Mary went to the park. Who went to the store?",
    # Cause-effect
    "If you increase temperature, does pressure increase or decrease in a closed system?",
    # Counterfactual
    "What would happen to gravity if Earth's mass doubled but radius stayed the same?",
    # Proof
    "Prove that the square root of 2 is irrational.",
    # Strategy
    "In a tournament with 127 players, how many matches are needed to determine a winner?",
]

RECALL_QUERIES = [
    # MMLU-style factual
    "What is the capital of France?",
    # Commonsense
    "What happens if you mix bleach and ammonia?",
    # World knowledge
    "Who was the first president of the United States?",
    # Definition recall
    "What does photosynthesis produce?",
    # Factual
    "What is the chemical symbol for gold?",
    # Classification
    "Is a tomato a fruit or a vegetable?",
]

MATH_QUERIES = [
    "Calculate the area of a circle with radius 5.",
    "Solve for x: 2x + 5 = 15",
    "What is 17 * 23?",
    "Simplify: (3 + 4) * (5 - 2)",
    "What is the derivative of x^3 + 2x?",
    "If a right triangle has legs of 3 and 4, what is the hypotenuse?",
    "Solve: x^2 - 5x + 6 = 0",
]

ALL_LABELED = (
    [(q, QueryType.REASONING) for q in REASONING_QUERIES]
    + [(q, QueryType.RECALL) for q in RECALL_QUERIES]
    + [(q, QueryType.MATH) for q in MATH_QUERIES]
)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestClassifyQuery:
    """Accuracy on labeled gold-standard set."""

    @pytest.mark.parametrize("query,expected", ALL_LABELED)
    def test_classify_query_returns_correct_type(self, query, expected):
        """classify_query maps known query patterns to correct QueryType."""
        result = classify_query(query)
        assert result == expected, (
            f"classify_query({query!r}) returned {result}, expected {expected}"
        )

    def test_classify_query_accuracy_threshold(self):
        """≥90% accuracy on held-out labeled set meets O-0009 gate."""
        correct = sum(
            1 for query, expected in ALL_LABELED
            if classify_query(query) == expected
        )
        total = len(ALL_LABELED)
        accuracy = correct / total
        assert accuracy >= 0.90, (
            f"Accuracy {accuracy:.1%} ({correct}/{total}) below 90% gate"
        )


class TestQueryTypeEnum:
    """QueryType enum has all required variants."""
    assert hasattr(QueryType, "REASONING")
    assert hasattr(QueryType, "RECALL")
    assert hasattr(QueryType, "MATH")


class TestClassifyQueryEdgeCases:
    """Edge cases and ambiguous inputs."""

    def test_empty_string(self):
        """Empty query returns RECALL (safe default, avoids three-prompt overhead)."""
        result = classify_query("")
        assert result == QueryType.RECALL

    def test_very_long_query(self):
        """Long complex query classified as REASONING."""
        long_query = (
            "Consider a system where particles move according to Brownian motion. "
            "If we introduce a temperature gradient, how does the diffusion coefficient "
            "change as a function of position? Derive the relationship and explain the "
            "physical implications for the steady-state distribution."
        )
        result = classify_query(long_query)
        assert result == QueryType.REASONING

    def test_code_snippet(self):
        """Code snippet is classified as REASONING (requires multi-step reasoning)."""
        code_query = "Write a function that returns the nth Fibonacci number using dynamic programming."
        result = classify_query(code_query)
        assert result == QueryType.REASONING

    def test_single_word_math(self):
        """Single-word math indicator classified as MATH."""
        result = classify_query("Calculate.")
        assert result == QueryType.MATH

    def test_whitespace_only(self):
        """Whitespace-only query returns RECALL (safe default)."""
        result = classify_query("   \n\t  ")
        assert result == QueryType.RECALL