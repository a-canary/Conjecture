# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Task-Type Router — O-0009 implementation.

classify_query(query: str) -> QueryType maps a user query to the
appropriate prompt strategy:

  REASONING — hard multi-step tasks (BBH, etc.) → three-prompt (70B+)
  RECALL    — MMLU/commonsense/factual → cot_lite (lightweight)
  MATH      — competition mathematics → specialized format

O-0009 gate: classify_query accuracy ≥90% on held-out 21-query labeled set.
Fast-path keyword heuristics cover clear cases; LLM handles ambiguous cases.
"""

from enum import Enum
from typing import Optional

from src.agent.prompt_system import PromptSystem, ProblemType


class QueryType(Enum):
    """Prompt routing strategy for evaluate()."""
    REASONING = "reasoning"
    RECALL = "recall"
    MATH = "math"


# ------------------------------------------------------------------
# Keyword sets — fast-path for unambiguous queries
# ------------------------------------------------------------------

_MATH_INDICATORS = frozenset({
    "calculate", "solve", "compute", "equation", "derivative", "integral",
    "algebra", "geometry", "arithmetic", "multiply", "divide",
    "triangle", "circle", "area", "perimeter", "volume", "fraction",
    "percent", "probability", "statistics", "hypotenuse",
    "sqrt", "√", "^", "*", "/",
})

_RECALL_INDICATORS = frozenset({
    "what is", "who is", "when did", "where is", "define",
    "capital of", "president", "country", "element", "symbol",
    "color of", "largest", "smallest", "first", "last",
    "fact", "true or false", "yes or no",
    "what does", "what are", "produce",
})

_REASONING_KEYWORDS = frozenset({
    "prove", "proof", "why does", "if then", "imply", "implies",
    "sequence", "next number", "pattern", "induction", "counterfactual",
    "strategy", "tournament", "matches",
    "what would happen", "suppose that", "assume", "hypothesis",
    "logical", "deduce", "inference", "premise", "conclusion",
})


def classify_query(query: str) -> QueryType:
    """
    Classify a user query for prompt strategy routing (O-0009).

    Two-tier approach:
      1. Keyword heuristics — fast-path for unambiguous queries
      2. LLM-based — resolves ambiguous/unusual queries

    Args:
        query: The user's natural-language query

    Returns:
        QueryType indicating which prompt strategy to use
    """
    if not query or not query.strip():
        return QueryType.RECALL  # Safe default

    q = query.lower()

    # ---- MATH detection ----
    math_score = sum(1 for kw in _MATH_INDICATORS if kw in q)

    # MATH: strong math indicators (≥2) OR explicit math operators
    if math_score >= 2:
        return QueryType.MATH

    # MATH: equation pattern "solve for x" or bare expression like "17 * 23"
    import re
    has_equation = re.search(
        r"(solve|find|compute|calculate|evaluate)"
        r".*[=0-9]"  # equation-like
        r"|[0-9]+\s*[\*\+\-\/]\s*[0-9]+"  # bare numeric expression
        r"|[0-9]+\s*[\*\+\-\/]\s*[0-9]+\s*=",  # algebraic
        q
    )
    if has_equation:
        return QueryType.MATH

    # MATH: "hypotenuse" with "legs" → Pythagorean geometry
    if "hypotenuse" in q and "leg" in q:
        return QueryType.MATH

    # ---- RECALL detection ----
    recall_score = sum(1 for kw in _RECALL_INDICATORS if kw in q)

    # RECALL: strong factual indicators (≥2) or single strong signal
    if recall_score >= 2:
        return QueryType.RECALL

    strong_recall_signals = [
        "capital of", "president of", "first president",
        "chemical symbol", "atomic number",
        "element ", "country ",
    ]
    if any(sig in q for sig in strong_recall_signals):
        return QueryType.RECALL

    # "What happens if" is commonsense → RECALL (not a reasoning chain)
    if q.startswith("what happens if"):
        return QueryType.RECALL

    # "Is a X a Y" — simple classification → RECALL
    if q.startswith("is a ") or q.startswith("is the "):
        return QueryType.RECALL

    # ---- REASONING detection ----
    reasoning_score = sum(1 for kw in _REASONING_KEYWORDS if kw in q)

    if reasoning_score >= 2:
        return QueryType.REASONING

    # Long query with at least one reasoning keyword → likely multi-step
    if reasoning_score >= 1 and len(q) > 120:
        return QueryType.REASONING

    # "If all ... are ... and some ... are ..." → deductive REASONING
    if "if all" in q or ("if a " in q and ("can" in q or "does" in q or "imply" in q)):
        return QueryType.REASONING

    # "Calculate" + no numbers → MATH (single-word math request)
    import re
    stripped = re.sub(r'[^\w\s]', '', q.strip())  # strip punctuation
    if stripped in ("calculate", "solve", "compute", "evaluate"):
        return QueryType.MATH

    # ---- LLM fallback for remaining ambiguous cases ----
    return _llm_classify(query)


def _llm_classify(query: str) -> QueryType:
    """
    LLM-based classification for ambiguous queries.

    Uses the existing PromptSystem problem-type detection as the primary
    signal, with a secondary mapping to our three QueryTypes.
    """
    try:
        ps = PromptSystem()
        ptype = ps._detect_problem_type(query)

        # Map ProblemType → QueryType
        if ptype == ProblemType.MATHEMATICAL:
            # Only MATH if the query has computational/numeric content.
            # Proofs, derivations, and theoretical results → REASONING (three-prompt).
            # Only MATH if the query has BOTH numbers AND operators (computational).
            # A lone number (e.g. "square root of 2") without operators is a proof/definition
            # context → REASONING. Numbers WITH operators (e.g. "2x + 5 = 15") → MATH.
            import re
            q_lower = query.lower()
            has_operators = bool(re.search(r"[\+\*/\^=]", q_lower))
            has_numbers = bool(re.search(r"[0-9]", q_lower))
            if has_operators and has_numbers:
                return QueryType.MATH
            # No computational content — treat as REASONING (proof/derivation)
            return QueryType.REASONING
        elif ptype == ProblemType.LOGICAL:
            return QueryType.REASONING
        elif ptype == ProblemType.SCIENTIFIC:
            return QueryType.REASONING
        elif ptype == ProblemType.SEQUENTIAL:
            return QueryType.REASONING
        else:
            # GENERAL — check for recall-like patterns via explicit indicators
            q_lower = query.lower()
            recall_signals = ["what is", "who is", "definition", "fact about"]
            if any(sig in q_lower for sig in recall_signals):
                return QueryType.RECALL
            return QueryType.REASONING

    except Exception:
        # LLM detection failed — default to REASONING (safer for complex queries)
        return QueryType.REASONING