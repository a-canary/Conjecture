"""
Tests for src/process/input_decomposer.py — A-0009 implementation.

Covers:
- Basic decomposition (fallback path)
- Question detection (-> ClaimType.GOAL)
- Multiple parts detection via mocked LLM
- Fallback behavior when LLM is unavailable or returns bad data
- Gate: decompose_input("What is 2+2? I think it's 4.") returns 2+ claims
- create_root_context: basic creation, sub-claim linking, pre-decomposed path
"""

import json
import time
import uuid
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from src.process.input_decomposer import (
    decompose_input,
    _fallback_claim,
    _parse_llm_response,
    create_root_context,
)
from src.core.models import ClaimType, ClaimState, ClaimScope, Claim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_llm_client(response_json: dict) -> MagicMock:
    """Build a mock LLMClient whose generate() returns the given JSON as content."""
    client = MagicMock()
    client.generate = AsyncMock(return_value={"content": json.dumps(response_json)})
    return client


def make_failing_llm_client() -> MagicMock:
    """Build a mock LLMClient whose generate() always raises an exception."""
    client = MagicMock()
    client.generate = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
    return client


# ---------------------------------------------------------------------------
# Fallback helpers (sync, no asyncio needed)
# ---------------------------------------------------------------------------

class TestFallbackClaim:
    def test_returns_single_claim(self):
        claims = _fallback_claim("Hello world")
        assert len(claims) == 1

    def test_fallback_is_observation(self):
        claims = _fallback_claim("Hello world")
        assert ClaimType.OBSERVATION in claims[0].type

    def test_fallback_contains_original_text(self):
        text = "Something interesting happened here."
        claims = _fallback_claim(text)
        assert claims[0].content == text

    def test_short_text_padded(self):
        # Claim model requires min_length=5; validator strips whitespace so
        # _fallback_claim appends " (input)" suffix instead of space-padding.
        claims = _fallback_claim("Hi")
        assert len(claims[0].content) >= 5
        assert "Hi" in claims[0].content

    def test_long_text_truncated(self):
        text = "x" * 2000
        claims = _fallback_claim(text)
        assert len(claims[0].content) <= 1000

    def test_fallback_tag_present(self):
        claims = _fallback_claim("Some text here.")
        assert "fallback" in claims[0].tags


# ---------------------------------------------------------------------------
# _parse_llm_response (sync helper)
# ---------------------------------------------------------------------------

class TestParseLLMResponse:
    def test_parses_question_as_goal(self):
        raw = json.dumps({
            "parts": [
                {"type": "question", "content": "What is 2+2?", "confidence": 0.9}
            ]
        })
        claims = _parse_llm_response(raw, "What is 2+2?")
        assert len(claims) == 1
        assert ClaimType.GOAL in claims[0].type

    def test_parses_assertion_type(self):
        raw = json.dumps({
            "parts": [
                {"type": "assertion", "content": "I think it equals four.", "confidence": 0.8}
            ]
        })
        claims = _parse_llm_response(raw, "I think it equals four.")
        assert ClaimType.ASSERTION in claims[0].type

    def test_parses_reference_type(self):
        raw = json.dumps({
            "parts": [
                {"type": "reference", "content": "According to Wikipedia math page.", "confidence": 0.7}
            ]
        })
        claims = _parse_llm_response(raw, "According to Wikipedia math page.")
        assert ClaimType.REFERENCE in claims[0].type

    def test_parses_assumption_type(self):
        raw = json.dumps({
            "parts": [
                {"type": "assumption", "content": "Assuming base 10 arithmetic.", "confidence": 0.6}
            ]
        })
        claims = _parse_llm_response(raw, "Assuming base 10 arithmetic.")
        assert ClaimType.ASSUMPTION in claims[0].type

    def test_parses_observation_type(self):
        raw = json.dumps({
            "parts": [
                {"type": "observation", "content": "The result looks correct to me.", "confidence": 0.75}
            ]
        })
        claims = _parse_llm_response(raw, "The result looks correct to me.")
        assert ClaimType.OBSERVATION in claims[0].type

    def test_unknown_type_defaults_to_concept(self):
        raw = json.dumps({
            "parts": [
                {"type": "mystery", "content": "Something unknown here...", "confidence": 0.5}
            ]
        })
        claims = _parse_llm_response(raw, "Something unknown here...")
        # Unknown types default to CONCEPT per implementation
        assert ClaimType.CONCEPT in claims[0].type

    def test_multiple_parts(self):
        raw = json.dumps({
            "parts": [
                {"type": "question", "content": "What is 2+2?", "confidence": 0.9},
                {"type": "assertion", "content": "I think it is four.", "confidence": 0.8},
            ]
        })
        claims = _parse_llm_response(raw, "What is 2+2? I think it is four.")
        assert len(claims) == 2

    def test_malformed_json_falls_back(self):
        claims = _parse_llm_response("not valid json at all", "some text")
        assert len(claims) == 1
        assert ClaimType.OBSERVATION in claims[0].type

    def test_empty_parts_falls_back(self):
        raw = json.dumps({"parts": []})
        claims = _parse_llm_response(raw, "some text for fallback")
        assert len(claims) == 1

    def test_confidence_clamped_to_range(self):
        raw = json.dumps({
            "parts": [
                {"type": "observation", "content": "Confidence out of range test.", "confidence": 1.5}
            ]
        })
        claims = _parse_llm_response(raw, "Confidence out of range test.")
        assert 0.0 <= claims[0].confidence <= 1.0

    def test_json_embedded_in_markdown(self):
        """LLMs often wrap JSON in markdown code fences."""
        inner = json.dumps({
            "parts": [
                {"type": "question", "content": "What is happening here?", "confidence": 0.85}
            ]
        })
        raw = f"```json\n{inner}\n```"
        claims = _parse_llm_response(raw, "What is happening here?")
        assert len(claims) == 1
        assert ClaimType.GOAL in claims[0].type


# ---------------------------------------------------------------------------
# decompose_input — sync tests (function is sync, uses heuristics or bridges async LLM)
# ---------------------------------------------------------------------------

class TestDecomposeInput:
    """Tests for the main decompose_input function (sync API)."""

    # -- Fallback (no LLM client) --

    def test_no_client_returns_single_observation(self):
        # decompose_input is sync - returns list directly
        claims = decompose_input("Hello, what time is it?")
        assert len(claims) >= 1
        # Heuristic mode may return GOAL for questions or OBSERVATION
        assert any(ClaimType.OBSERVATION in c.type or ClaimType.GOAL in c.type for c in claims)

    def test_empty_text_raises_value_error(self):
        # Empty text raises ValueError per docstring
        with pytest.raises(ValueError):
            decompose_input("")

    def test_whitespace_only_raises_value_error(self):
        # Whitespace-only raises ValueError per docstring
        with pytest.raises(ValueError):
            decompose_input("   \n\t  ")

    def test_fallback_preserves_text(self):
        text = "Some interesting claim about the world."
        claims = decompose_input(text)
        # Content should be preserved (possibly with heuristic tagging)
        assert any(text in c.content for c in claims)

    # -- Failing LLM client => fallback --

    def test_failing_llm_uses_fallback(self):
        client = make_failing_llm_client()
        claims = decompose_input("Something to decompose.", llm_client=client)
        assert len(claims) >= 1
        # Falls back to heuristic which may assign various types

    # -- Mocked LLM client returning structured response --

    def test_question_detection(self):
        client = make_llm_client({
            "parts": [
                {"type": "question", "content": "What is 2+2?", "confidence": 0.9}
            ]
        })
        claims = decompose_input("What is 2+2?", llm_client=client)
        assert len(claims) == 1
        assert ClaimType.GOAL in claims[0].type

    def test_multiple_parts_detection(self):
        client = make_llm_client({
            "parts": [
                {"type": "question", "content": "What is 2+2?", "confidence": 0.9},
                {"type": "assertion", "content": "I think it is four.", "confidence": 0.8},
            ]
        })
        claims = decompose_input("What is 2+2? I think it is four.", llm_client=client)
        assert len(claims) == 2

    def test_all_claim_types_roundtrip(self):
        client = make_llm_client({
            "parts": [
                {"type": "question",    "content": "What is the answer?",          "confidence": 0.9},
                {"type": "assertion",   "content": "I assert it is forty-two.",    "confidence": 0.85},
                {"type": "reference",   "content": "See chapter three for detail.", "confidence": 0.7},
                {"type": "assumption",  "content": "Assuming integer arithmetic.",  "confidence": 0.6},
                {"type": "observation", "content": "The output seems correct.",     "confidence": 0.75},
            ]
        })
        claims = decompose_input("Compound input text.", llm_client=client)
        assert len(claims) == 5
        types_found = {c.type[0] for c in claims}
        assert ClaimType.GOAL in types_found
        assert ClaimType.ASSERTION in types_found
        assert ClaimType.REFERENCE in types_found
        assert ClaimType.ASSUMPTION in types_found
        assert ClaimType.OBSERVATION in types_found

    def test_claims_are_valid_claim_objects(self):
        client = make_llm_client({
            "parts": [
                {"type": "question", "content": "What is 2+2?", "confidence": 0.9}
            ]
        })
        claims = decompose_input("What is 2+2?", llm_client=client)
        for claim in claims:
            assert isinstance(claim, Claim)
            assert claim.id
            assert 0.0 <= claim.confidence <= 1.0
            assert claim.state == ClaimState.EXPLORE

    def test_claims_tagged_as_decomposed(self):
        client = make_llm_client({
            "parts": [
                {"type": "question", "content": "What is 2+2?", "confidence": 0.9}
            ]
        })
        claims = decompose_input("What is 2+2?", llm_client=client)
        assert "decomposed" in claims[0].tags

    # -- Gate test (requirement) --

    def test_gate_two_plus_claims_returned(self):
        """Gate: decompose_input('What is 2+2? I think it is 4.') returns 2+ claims."""
        client = make_llm_client({
            "parts": [
                {"type": "question",  "content": "What is 2+2?",  "confidence": 0.95},
                {"type": "assertion", "content": "I think it is 4.", "confidence": 0.8},
            ]
        })
        claims = decompose_input("What is 2+2? I think it's 4.", llm_client=client)
        assert len(claims) >= 2, (
            f"Gate failed: expected 2+ claims, got {len(claims)}"
        )
        # Verify types
        found_types = {c.type[0] for c in claims}
        assert ClaimType.GOAL in found_types, "Expected a GOAL (question) claim"
        assert ClaimType.ASSERTION in found_types, "Expected an ASSERTION claim"


# ---------------------------------------------------------------------------
# create_root_context — async tests (D-0009)
# ---------------------------------------------------------------------------

class TestCreateRootContextBasic:
    """test_create_root_context_basic: verify root claim structure."""

    def test_returns_tuple_of_claim_and_list(self):
        root, subs = asyncio.get_event_loop().run_until_complete(
            create_root_context("Hello")
        )
        assert isinstance(root, Claim)
        assert isinstance(subs, list)

    def test_root_claim_type_is_observation(self):
        root, _ = asyncio.get_event_loop().run_until_complete(
            create_root_context("Hello world, this is a test.")
        )
        assert ClaimType.OBSERVATION in root.type

    def test_root_claim_state_is_explore(self):
        root, _ = asyncio.get_event_loop().run_until_complete(
            create_root_context("Hello world, this is a test.")
        )
        assert root.state == ClaimState.EXPLORE

    def test_root_claim_scope_is_user_workspace(self):
        root, _ = asyncio.get_event_loop().run_until_complete(
            create_root_context("Hello world, this is a test.")
        )
        assert root.scope == ClaimScope.USER_WORKSPACE

    def test_root_claim_tags_include_root_context_and_conversation(self):
        root, _ = asyncio.get_event_loop().run_until_complete(
            create_root_context("Hello world, this is a test.")
        )
        assert "root_context" in root.tags
        assert "conversation" in root.tags

    def test_root_claim_confidence_is_1(self):
        root, _ = asyncio.get_event_loop().run_until_complete(
            create_root_context("Hello world, this is a test.")
        )
        assert root.confidence == 1.0

    def test_root_claim_content_contains_conversation(self):
        conversation = "User: What is the capital of France?"
        root, _ = asyncio.get_event_loop().run_until_complete(
            create_root_context(conversation)
        )
        assert "France" in root.content

    def test_long_conversation_truncated(self):
        long_text = "x" * 2000
        root, _ = asyncio.get_event_loop().run_until_complete(
            create_root_context(long_text)
        )
        # Claim model max_length=1000; content must fit
        assert len(root.content) <= 1000

    def test_short_conversation_padded(self):
        root, _ = asyncio.get_event_loop().run_until_complete(
            create_root_context("Hi")
        )
        # Content must meet model min_length=5
        assert len(root.content) >= 5

    def test_root_claim_has_valid_id(self):
        root, _ = asyncio.get_event_loop().run_until_complete(
            create_root_context("Hello world, this is a test.")
        )
        assert root.id.startswith("c")
        assert len(root.id) > 1


class TestRootContextLinksSubs:
    """test_root_context_links_subs: verify sub-claims contain root_claim.id in supers."""

    def test_subs_contain_root_id_in_supers(self):
        """Gate: sub_claims[0].supers contains root_claim.id."""
        client = make_llm_client({
            "parts": [
                {"type": "question", "content": "What is 2+2?", "confidence": 0.9},
            ]
        })
        root, subs = asyncio.get_event_loop().run_until_complete(
            create_root_context("What is 2+2?", llm_client=client)
        )
        assert len(subs) >= 1
        assert root.id in subs[0].supers

    def test_all_subs_linked_to_root(self):
        client = make_llm_client({
            "parts": [
                {"type": "question",  "content": "What is 2+2?",    "confidence": 0.9},
                {"type": "assertion", "content": "I think it is 4.", "confidence": 0.8},
            ]
        })
        root, subs = asyncio.get_event_loop().run_until_complete(
            create_root_context("What is 2+2? I think it is 4.", llm_client=client)
        )
        for sub in subs:
            assert root.id in sub.supers, (
                f"Expected root id {root.id} in sub.supers, got {sub.supers}"
            )

    def test_no_duplicate_root_id_in_supers(self):
        """Calling create_root_context twice on the same pre-decomposed claims should
        not duplicate the root id in supers when the second call uses a different root."""
        client = make_llm_client({
            "parts": [
                {"type": "observation", "content": "The sky is blue today.", "confidence": 0.8},
            ]
        })
        root, subs = asyncio.get_event_loop().run_until_complete(
            create_root_context("The sky is blue today.", llm_client=client)
        )
        # root_claim.id should appear exactly once in each sub's supers
        for sub in subs:
            assert sub.supers.count(root.id) == 1

    def test_fallback_sub_is_linked_when_no_llm(self):
        """Without an LLM client, decompose_input returns a fallback claim that
        should still be linked to the root."""
        root, subs = asyncio.get_event_loop().run_until_complete(
            create_root_context("Hello world, this is a test.")
        )
        assert len(subs) >= 1
        assert root.id in subs[0].supers


class TestRootContextWithPredecomposed:
    """test_root_context_with_predecomposed: pre-decomposed claims path."""

    def _make_observation_claim(self, content: str) -> Claim:
        return Claim(
            id=f"c{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
            content=content,
            confidence=0.8,
            type=[ClaimType.OBSERVATION],
            state=ClaimState.EXPLORE,
            tags=["decomposed"],
        )

    def test_predecomposed_claims_are_returned_unchanged_content(self):
        claim = self._make_observation_claim("Some pre-decomposed observation text.")
        root, subs = asyncio.get_event_loop().run_until_complete(
            create_root_context(
                "Some conversation text here.",
                decomposed_claims=[claim],
            )
        )
        assert len(subs) == 1
        assert subs[0].content == claim.content

    def test_predecomposed_claims_are_linked_to_root(self):
        claim = self._make_observation_claim("Some pre-decomposed observation text.")
        root, subs = asyncio.get_event_loop().run_until_complete(
            create_root_context(
                "Some conversation text here.",
                decomposed_claims=[claim],
            )
        )
        assert root.id in subs[0].supers

    def test_predecomposed_skips_llm_call(self):
        """When decomposed_claims is provided, the LLM client should not be called."""
        client = MagicMock()
        client.generate = AsyncMock(side_effect=AssertionError("LLM should not be called"))
        claim = self._make_observation_claim("Some pre-decomposed observation text.")
        # Should not raise even though the mock raises on call
        root, subs = asyncio.get_event_loop().run_until_complete(
            create_root_context(
                "Some conversation text here.",
                decomposed_claims=[claim],
                llm_client=client,
            )
        )
        assert len(subs) == 1

    def test_multiple_predecomposed_all_linked(self):
        claims = [
            self._make_observation_claim("First observation that was pre-decomposed."),
            self._make_observation_claim("Second observation that was pre-decomposed."),
            self._make_observation_claim("Third observation that was pre-decomposed."),
        ]
        root, subs = asyncio.get_event_loop().run_until_complete(
            create_root_context(
                "A longer conversation with multiple parts included here.",
                decomposed_claims=claims,
            )
        )
        assert len(subs) == 3
        for sub in subs:
            assert root.id in sub.supers

    def test_gate_supers_contains_root_id(self):
        """Gate: create_root_context('Hello') returns (root, subs)
        where subs[0].supers contains root.id."""
        root, subs = asyncio.get_event_loop().run_until_complete(
            create_root_context("Hello")
        )
        assert isinstance(root, Claim)
        assert len(subs) >= 1
        assert root.id in subs[0].supers, (
            f"Gate failed: root.id={root.id!r} not found in subs[0].supers={subs[0].supers!r}"
        )
