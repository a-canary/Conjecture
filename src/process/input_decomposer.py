# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Input Decomposer — A-0009 implementation.

The Process Layer treats all input as compound. Prompts are decomposed into
constituent claims (questions, assertions, references, context) using LLM
analysis before reasoning.
"""

import asyncio
import inspect
import json
import logging
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from src.core.models import Claim, ClaimType, ClaimState, ClaimScope

logger = logging.getLogger(__name__)

# ── LLM prompt template (per A-0009 specification) ────────────────────────────

DECOMPOSE_PROMPT_TEMPLATE = """Analyze this input and extract constituent claims:

Input: {text}

Extract each distinct claim with its type. For each claim identify:
- content: the claim text
- type: one of [GOAL, ASSERTION, ASSUMPTION, OBSERVATION, CONJECTURE, REFERENCE, CONCEPT, EXAMPLE]
- confidence: your confidence in the extraction (0.0-1.0)

Return as JSON array."""

# ── Type mapping (per F-0007 nine types) ──────────────────────────────────────

# Maps every label the LLM might return to a ClaimType.
# Accepts both upper- and lower-case variants for robustness.
_PART_TYPE_MAP: Dict[str, ClaimType] = {
    # Canonical upper-case labels (as requested in the prompt)
    "GOAL":        ClaimType.GOAL,        # What the user wants to know / achieve
    "ASSERTION":   ClaimType.ASSERTION,   # Strong statement made with confidence
    "ASSUMPTION":  ClaimType.ASSUMPTION,  # Something taken as true without proof
    "OBSERVATION": ClaimType.OBSERVATION, # Something noticed or perceived
    "CONJECTURE":  ClaimType.CONJECTURE,  # Conclusion on incomplete evidence
    "REFERENCE":   ClaimType.REFERENCE,   # Pointer to external information
    "CONCEPT":     ClaimType.CONCEPT,     # Abstract idea or general notion
    "EXAMPLE":     ClaimType.EXAMPLE,     # Specific instance or case
    # Lower-case tolerance
    "goal":        ClaimType.GOAL,
    "assertion":   ClaimType.ASSERTION,
    "assumption":  ClaimType.ASSUMPTION,
    "observation": ClaimType.OBSERVATION,
    "conjecture":  ClaimType.CONJECTURE,
    "reference":   ClaimType.REFERENCE,
    "concept":     ClaimType.CONCEPT,
    "example":     ClaimType.EXAMPLE,
    # Legacy labels — kept for backward compatibility
    "question":    ClaimType.GOAL,
    # IMPRESSION is a valid ClaimType but not in the A-0009 decomposition
    # vocabulary; map to the closest epistemic neighbour.
    "IMPRESSION":  ClaimType.CONJECTURE,
    "impression":  ClaimType.CONJECTURE,
}

# Keep the old prompt name as an alias for any existing callers
_DECOMPOSITION_PROMPT = DECOMPOSE_PROMPT_TEMPLATE


def _generate_id() -> str:
    """Generate a unique claim ID compatible with src/core/models.py format."""
    timestamp = int(time.time() * 1000)
    unique_part = str(uuid.uuid4())[:8]
    return f"c{timestamp}_{unique_part}"


# ── JSON parsing helpers ──────────────────────────────────────────────────────

def _extract_json_array(raw: str) -> str:
    """Return the first JSON array found in *raw*, stripping markdown fences."""
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if fenced:
        return fenced.group(1).strip()
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1]
    return raw.strip()


def _clamp_confidence(value: Any) -> float:
    """Clamp *value* to [0.0, 1.0], defaulting to 0.7 on invalid input."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.7
    return max(0.0, min(1.0, f))


def _build_claims_from_raw(raw_items: List[Dict[str, Any]]) -> List[Claim]:
    """Convert a list of raw JSON dicts into ``Claim`` objects."""
    claims: List[Claim] = []
    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            logger.warning("Skipping non-dict item at index %d: %r", idx, item)
            continue
        content = str(item.get("content", "")).strip()
        if len(content) < 5:
            if content:
                content = content + " (input)"
            if len(content) < 5:
                logger.warning("Skipping claim at index %d — content too short: %r", idx, content)
                continue
        raw_type = str(item.get("type", "CONCEPT"))
        claim_type = _PART_TYPE_MAP.get(raw_type) or _PART_TYPE_MAP.get(raw_type.upper(), ClaimType.CONCEPT)
        confidence = _clamp_confidence(item.get("confidence", 0.7))
        try:
            claim = Claim(
                id=_generate_id(),
                content=content,
                type=[claim_type],
                confidence=confidence,
                state=ClaimState.EXPLORE,
                scope=ClaimScope.USER_WORKSPACE,
                tags=["decomposed", "input-analysis"],
            )
            claims.append(claim)
        except Exception as exc:
            logger.warning("Skipping malformed claim at index %d (%s): %r", idx, exc, item)
    return claims


def _parse_llm_response(response_text: str, original_text: str) -> List[Claim]:
    """Parse the JSON response from the LLM into a list of Claims.

    Handles both formats:
    - JSON array (new A-0009 spec): ``[{"content": ..., "type": ..., "confidence": ...}, ...]``
    - JSON object with "parts" key (legacy): ``{"parts": [...]}``

    Args:
        response_text: Raw LLM response string.
        original_text: Original user input (used for fallback).

    Returns:
        List of Claim objects extracted from the LLM response, or a single
        fallback OBSERVATION claim if parsing fails.
    """
    # --- Try JSON array format first (A-0009 spec) ---
    array_candidate = _extract_json_array(response_text)
    try:
        parsed = json.loads(array_candidate)
        if isinstance(parsed, list):
            claims = _build_claims_from_raw(parsed)
            if claims:
                return claims
    except json.JSONDecodeError:
        pass

    # --- Try legacy object-with-"parts" format ---
    json_match = re.search(r'\{[\s\S]*\}', response_text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            parts = data.get("parts", [])
            if parts:
                claims = _build_claims_from_raw(parts)
                if claims:
                    return claims
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse JSON from LLM response (%s); using fallback.", exc)

    logger.warning("No usable JSON found in LLM response; falling back to single claim.")
    return _fallback_claim(original_text)


def _fallback_claim(text: str) -> List[Claim]:
    """Return a single OBSERVATION claim wrapping the full input text.

    Used when the LLM is unavailable or returns an unparseable response.

    Args:
        text: Original user input.

    Returns:
        List containing a single Claim of type OBSERVATION.
    """
    content = text.strip()
    # Ensure minimum length required by Claim validator (validator strips whitespace,
    # so we cannot pad with spaces — append a descriptive suffix instead).
    if len(content) < 5:
        content = content + " (input)"
    # Truncate to model max_length
    if len(content) > 1000:
        content = content[:997] + "..."

    return [
        Claim(
            id=_generate_id(),
            content=content,
            confidence=0.5,
            type=[ClaimType.OBSERVATION],
            state=ClaimState.EXPLORE,
            tags=["decomposed", "fallback"],
        )
    ]


# ── LLM invocation helpers ────────────────────────────────────────────────────

async def _invoke_llm_async(llm_client: Any, prompt: str) -> str:
    """Await the async ``generate()`` method and return the response text."""
    result = await llm_client.generate(prompt=prompt, temperature=0.3, max_tokens=1024)
    if isinstance(result, dict):
        return result.get("content", "")
    return str(result)


def _invoke_llm_sync(llm_client: Any, prompt: str) -> str:
    """Call the synchronous ``generate()`` method and return the response text."""
    result = llm_client.generate(prompt=prompt, temperature=0.3, max_tokens=1024)
    if isinstance(result, dict):
        return result.get("content", "")
    return str(result)


# ── Heuristic fallback ────────────────────────────────────────────────────────

def _heuristic_decompose(text: str) -> List[Claim]:
    """Heuristic sentence-level decomposer used when no LLM is available.

    Segments the text by sentence boundaries and assigns ``ClaimType`` values
    based on simple lexical cues.  Supports all eight A-0009 / F-0007 types.
    All outputs carry ``EXPLORE`` state and are tagged with ``"heuristic"``.
    """
    segments = re.split(r"(?<=[.!?])\s+", text.strip())
    refined: List[str] = []
    for seg in segments:
        sub_parts = re.split(r";\s+| because | since ", seg, flags=re.IGNORECASE)
        refined.extend(p.strip() for p in sub_parts if p.strip())

    claims: List[Claim] = []
    for segment in refined:
        if len(segment) < 5:
            continue
        sl = segment.lower().strip()

        if sl.endswith("?") or re.match(
            r"^(what|who|where|when|why|how|is |are |does |do )", sl
        ):
            claim_type, confidence = ClaimType.GOAL, 0.85
        elif any(kw in sl for kw in (
            "i think", "i believe", "i assume", "assuming", "suppose",
        )):
            claim_type, confidence = ClaimType.ASSUMPTION, 0.75
        elif any(kw in sl for kw in (
            "i noticed", "i observe", "i see", "it appears", "it seems",
            "according to", "i found",
        )):
            claim_type, confidence = ClaimType.OBSERVATION, 0.75
        elif any(kw in sl for kw in (
            "perhaps", "maybe", "might", "could be", "possibly",
            "i conjecture", "hypothesis",
        )):
            claim_type, confidence = ClaimType.CONJECTURE, 0.70
        elif any(kw in sl for kw in (
            "for example", "such as", "e.g.", "like ", "instance",
        )):
            claim_type, confidence = ClaimType.EXAMPLE, 0.80
        elif any(kw in sl for kw in (
            "refer to", "citation", "source", "paper", "book", "article",
        )):
            claim_type, confidence = ClaimType.REFERENCE, 0.70
        else:
            claim_type, confidence = ClaimType.ASSERTION, 0.75

        try:
            claim = Claim(
                id=_generate_id(),
                content=segment[:1000],
                type=[claim_type],
                confidence=confidence,
                state=ClaimState.EXPLORE,
                scope=ClaimScope.USER_WORKSPACE,
                tags=["decomposed", "heuristic"],
            )
            claims.append(claim)
        except Exception as exc:
            logger.warning("Heuristic claim creation failed for segment %r: %s", segment, exc)

    # Last resort: wrap the whole text as a single ASSERTION
    if not claims:
        content = text.strip()[:1000]
        if len(content) < 5:
            content = content + " (input)"
        try:
            claims.append(
                Claim(
                    id=_generate_id(),
                    content=content,
                    type=[ClaimType.ASSERTION],
                    confidence=0.5,
                    state=ClaimState.EXPLORE,
                    scope=ClaimScope.USER_WORKSPACE,
                    tags=["decomposed", "heuristic"],
                )
            )
        except Exception as exc:
            logger.error("Failed to create fallback claim: %s", exc)

    return claims


# ── Public API ────────────────────────────────────────────────────────────────

def decompose_input(text: str, llm_client: Optional[Any] = None) -> List[Claim]:
    """Decompose user input text into constituent ``Claim`` objects.

    Implements A-0009: treats all input as compound and extracts constituent
    claims (goals, assertions, assumptions, observations, conjectures,
    references, concepts, examples) via LLM analysis.

    Claim type mappings (per F-0007 nine types):
        - Questions                   -> GOAL
        - Assertions / statements     -> ASSERTION
        - Assumptions                 -> ASSUMPTION
        - Observations                -> OBSERVATION
        - Conjectures / hypotheses    -> CONJECTURE
        - References to external src  -> REFERENCE
        - Concepts being discussed    -> CONCEPT
        - Examples given              -> EXAMPLE

    Args:
        text: Raw user input text to analyse.  Must be non-empty.
        llm_client: Optional LLM client.  Accepts:

            * An instance of ``src.endpoint.llm_client.LLMClient`` (async).
            * Any object with a ``generate(prompt, ...)`` method (sync or
              async) that returns either a string or ``{"content": str}``.
            * ``None``: triggers heuristic decomposition (no LLM required).

    Returns:
        A non-empty list of ``Claim`` objects, one per extracted constituent.
        Every claim has ``state=EXPLORE`` to mark it as provisional.

    Raises:
        ValueError: If *text* is empty or whitespace-only.

    Example::

        claims = decompose_input(
            "What is 2+2? I think it's 4 because of basic addition."
        )
        # Returns (approximately):
        # [
        #   Claim(content="What is 2+2?", type=[GOAL], confidence=0.85),
        #   Claim(content="I think it's 4", type=[ASSUMPTION], confidence=0.75),
        #   Claim(content="basic addition rules apply", type=[ASSERTION], ...),
        # ]
    """
    if not text or not text.strip():
        raise ValueError("Input text must not be empty.")

    if llm_client is None:
        logger.info("No LLM client provided — using heuristic decomposition.")
        return _heuristic_decompose(text)

    prompt = DECOMPOSE_PROMPT_TEMPLATE.format(text=text)

    generate_fn = getattr(llm_client, "generate", None)
    if generate_fn is None:
        logger.error("llm_client has no 'generate' method — falling back to heuristic.")
        return _heuristic_decompose(text)

    try:
        if inspect.iscoroutinefunction(generate_fn):
            # Async client — bridge into sync context via a dedicated event loop.
            # Callers inside an async context should use decompose_input_async().
            loop = asyncio.new_event_loop()
            try:
                raw_response = loop.run_until_complete(_invoke_llm_async(llm_client, prompt))
            finally:
                loop.close()
        else:
            raw_response = _invoke_llm_sync(llm_client, prompt)

        claims = _parse_llm_response(raw_response, text)
        logger.info("decompose_input: extracted %d claim(s) via LLM.", len(claims))
        return claims

    except Exception as exc:
        logger.error(
            "LLM decomposition failed (%s: %s) — falling back to heuristic.",
            type(exc).__name__, exc,
        )
        return _heuristic_decompose(text)


async def decompose_input_async(
    text: str, llm_client: Optional[Any] = None
) -> List[Claim]:
    """Async variant of :func:`decompose_input`.

    Preferred when calling from an async context (e.g. an endpoint handler)
    with an async LLM client, to avoid nested event-loop issues.

    Args:
        text: Raw user input text to analyse.
        llm_client: Optional async or sync LLM client.

    Returns:
        A non-empty list of ``Claim`` objects, one per extracted constituent.

    Raises:
        ValueError: If *text* is empty or whitespace-only.
    """
    if not text or not text.strip():
        raise ValueError("Input text must not be empty.")

    if llm_client is None:
        logger.info("No LLM client provided — using heuristic decomposition.")
        return _heuristic_decompose(text)

    prompt = DECOMPOSE_PROMPT_TEMPLATE.format(text=text)

    generate_fn = getattr(llm_client, "generate", None)
    if generate_fn is None:
        logger.error("llm_client has no 'generate' method — falling back to heuristic.")
        return _heuristic_decompose(text)

    try:
        if inspect.iscoroutinefunction(generate_fn):
            raw_response = await _invoke_llm_async(llm_client, prompt)
        else:
            raw_response = _invoke_llm_sync(llm_client, prompt)

        claims = _parse_llm_response(raw_response, text)
        logger.info("decompose_input_async: extracted %d claim(s) via LLM.", len(claims))
        return claims

    except Exception as exc:
        logger.error(
            "Async LLM decomposition failed (%s: %s) — falling back to heuristic.",
            type(exc).__name__, exc,
        )
        return _heuristic_decompose(text)


# Maximum characters stored verbatim in a root context claim's content field.
# The src.core.models.Claim validator enforces max_length=1000; we leave a small
# buffer for the truncation suffix so the value always fits.
_ROOT_CONTEXT_MAX_CONTENT = 990


async def create_root_context(
    conversation: str,
    decomposed_claims: List[Claim] = None,
    llm_client=None,
) -> Tuple[Claim, List[Claim]]:
    """Create a root context claim from a full conversation.

    Implements D-0009: stores the entire conversation as a single OBSERVATION
    claim at USER_WORKSPACE scope.  Every decomposed sub-claim is linked to
    the root by adding the root's ID to that sub-claim's ``supers`` list.

    Args:
        conversation: Full conversation text (user + framework messages).
        decomposed_claims: Optional pre-decomposed claims.  When ``None`` the
            function calls :func:`decompose_input` with the supplied
            ``llm_client``.
        llm_client: LLM client used for decomposition when
            ``decomposed_claims`` is not provided.

    Returns:
        A ``(root_claim, sub_claims)`` tuple where:
        - ``root_claim`` — OBSERVATION claim whose ``content`` is the full
          conversation (truncated to fit the model's ``max_length`` limit).
        - ``sub_claims`` — Decomposed claims, each with ``root_claim.id``
          appended to its ``supers`` list.
    """
    # ------------------------------------------------------------------
    # 1. Build the root claim content (truncate if necessary)
    # ------------------------------------------------------------------
    content = conversation.strip() if conversation else ""
    if len(content) < 5:
        content = content + " (conversation)"
    if len(content) > _ROOT_CONTEXT_MAX_CONTENT:
        content = content[:_ROOT_CONTEXT_MAX_CONTENT - 3] + "..."

    root_claim = Claim(
        id=_generate_id(),
        content=content,
        confidence=1.0,
        type=[ClaimType.OBSERVATION],
        state=ClaimState.EXPLORE,
        scope=ClaimScope.USER_WORKSPACE,
        tags=["root_context", "conversation"],
    )

    # ------------------------------------------------------------------
    # 2. Obtain decomposed sub-claims
    # ------------------------------------------------------------------
    if decomposed_claims is None:
        # Use the async variant to avoid await issues
        sub_claims = await decompose_input_async(conversation, llm_client=llm_client)
    else:
        # Work on a shallow copy of the list so we don't mutate the caller's
        # list reference; individual Claim objects are mutable Pydantic models.
        sub_claims = list(decomposed_claims)

    # ------------------------------------------------------------------
    # 3. Link each sub-claim to the root via the supers field
    # ------------------------------------------------------------------
    for claim in sub_claims:
        if root_claim.id not in claim.supers:
            claim.supers.append(root_claim.id)

    logger.info(
        "create_root_context: created root claim %s with %d sub-claim(s).",
        root_claim.id,
        len(sub_claims),
    )
    return root_claim, sub_claims
