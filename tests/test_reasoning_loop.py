# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Tests for src/process/reasoning_loop.py

Gate requirement (Steps 20.1-20.3, A-0012):
  ReasoningLoop.run() returns ReasoningResult with claims and trace.

Tests:
  - test_loop_halts_on_respond          — LLM immediately calls respond_to_user
  - test_loop_explores_before_responding — LLM creates claims then responds
  - test_max_iterations_enforced        — safety backstop fires
  - test_claims_tracked_across_iterations — all created claim IDs in result
  - (additional coverage tests)
"""

from __future__ import annotations

import pytest

from src.process.reasoning_loop import ReasoningLoop, ReasoningResult, ReasoningState
from src.data.repositories import ClaimRepository


# ---------------------------------------------------------------------------
# Stub LLM client
# ---------------------------------------------------------------------------


class StubLLMClient:
    """Simulates an LLM with tool-calling support via a pre-programmed response queue.

    Each call to generate_with_tools pops the next canned response.
    When the queue is empty an AssertionError is raised so tests fail loudly.
    """

    def __init__(self, responses: list) -> None:
        """
        Args:
            responses: Ordered list of dicts, each returned per call.
                Each dict may contain:
                  "tool_calls": list of {"name": str, "arguments": dict}
                  "content": optional plain-text response string
        """
        self._responses = list(responses)

    async def generate_with_tools(
        self,
        prompt: str,
        tools: list,
        system_prompt: str = "",
        **kwargs,
    ) -> dict:
        if not self._responses:
            raise AssertionError("StubLLMClient response queue exhausted unexpectedly")
        return self._responses.pop(0)

    @property
    def remaining(self) -> int:
        """Number of responses still in the queue."""
        return len(self._responses)


# ---------------------------------------------------------------------------
# Helpers: build fresh repository
# ---------------------------------------------------------------------------


async def _fresh_repo() -> ClaimRepository:
    repo = ClaimRepository()
    await repo.initialize()
    return repo


# ---------------------------------------------------------------------------
# Helpers: canned tool-call payloads
# ---------------------------------------------------------------------------


def _create_claim_response(
    content: str = "An intermediate reasoning step for testing",
    claim_type: str = "observation",
    confidence: float = 0.6,
) -> dict:
    """Return a simulated LLM response that creates one claim."""
    return {
        "tool_calls": [
            {
                "name": "create_claim",
                "arguments": {
                    "content": content,
                    "type": claim_type,
                    "confidence": confidence,
                },
            }
        ],
        "content": "",
    }


def _respond_response(
    text: str = "The final answer based on my reasoning.",
    supporting: list | None = None,
) -> dict:
    """Return a simulated LLM response that calls respond_to_user."""
    args: dict = {"response": text}
    if supporting:
        args["supporting_claims"] = supporting
    return {
        "tool_calls": [{"name": "respond_to_user", "arguments": args}],
        "content": "",
    }


def _text_only_response(text: str = "Plain text answer without tools.") -> dict:
    """Return a simulated LLM response with no tool calls (raw text)."""
    return {"tool_calls": [], "content": text}


def _no_content_no_tools() -> dict:
    """Return a simulated LLM response with neither tool calls nor content."""
    return {"tool_calls": [], "content": ""}


# ---------------------------------------------------------------------------
# test_loop_halts_on_respond
# ---------------------------------------------------------------------------


class TestLoopHaltsOnRespond:
    """Gate test: loop halts when LLM calls respond_to_user."""

    @pytest.mark.asyncio
    async def test_loop_halts_on_respond(self):
        """LLM immediately calls respond_to_user — loop stops at iteration 1."""
        client = StubLLMClient([_respond_response("Paris is the capital of France.")])
        repo = await _fresh_repo()
        loop = ReasoningLoop(llm_client=client, claim_repository=repo, max_iterations=5)

        result = await loop.run(query="What is the capital of France?")

        assert isinstance(result, ReasoningResult)
        assert result.halted_reason == "respond_to_user"
        assert result.iterations == 1
        assert result.response == "Paris is the capital of France."

    @pytest.mark.asyncio
    async def test_halt_on_respond_supporting_claims_propagated(self):
        """Supporting claim IDs from respond_to_user appear in result.supporting_claims."""
        client = StubLLMClient(
            [_respond_response("Confirmed.", supporting=["c0000001", "c0000002"])]
        )
        repo = await _fresh_repo()
        loop = ReasoningLoop(llm_client=client, claim_repository=repo, max_iterations=5)

        result = await loop.run(query="Is water wet?")

        assert "c0000001" in result.supporting_claims
        assert "c0000002" in result.supporting_claims

    @pytest.mark.asyncio
    async def test_halt_on_respond_no_claims_created(self):
        """Immediate respond_to_user means no claims were created during exploration."""
        client = StubLLMClient([_respond_response("Yes.")])
        repo = await _fresh_repo()
        loop = ReasoningLoop(llm_client=client, claim_repository=repo, max_iterations=5)

        result = await loop.run(query="Is 2+2=4?")

        assert result.claims_created == []

    @pytest.mark.asyncio
    async def test_halt_on_respond_tool_calls_trace_present(self):
        """tool_calls trace contains at least the respond_to_user call."""
        client = StubLLMClient([_respond_response("Answer.")])
        repo = await _fresh_repo()
        loop = ReasoningLoop(llm_client=client, claim_repository=repo, max_iterations=5)

        result = await loop.run(query="Simple question?")

        assert len(result.tool_calls) >= 1
        tools_used = [tc["tool"] for tc in result.tool_calls]
        assert "respond_to_user" in tools_used


# ---------------------------------------------------------------------------
# test_loop_explores_before_responding
# ---------------------------------------------------------------------------


class TestLoopExploresBeforeResponding:
    """Gate test: loop runs multiple iterations before halting (A-0012)."""

    @pytest.mark.asyncio
    async def test_loop_explores_before_responding(self):
        """LLM creates a claim in iteration 1 then responds in iteration 2.

        This is the primary gate test for Steps 20.1-20.3.
        """
        client = StubLLMClient(
            [
                _create_claim_response("First reasoning step: checking premises"),
                _respond_response("Final answer after exploration."),
            ]
        )
        repo = await _fresh_repo()
        loop = ReasoningLoop(llm_client=client, claim_repository=repo, max_iterations=5)

        result = await loop.run(query="Explain photosynthesis.")

        assert result.halted_reason == "respond_to_user"
        assert result.iterations == 2
        assert result.response == "Final answer after exploration."
        # One claim created (by ID string, not Claim object)
        assert len(result.claims_created) == 1
        assert isinstance(result.claims_created[0], str)

    @pytest.mark.asyncio
    async def test_explores_three_iterations_before_halting(self):
        """LLM creates claims in iterations 1 and 2, then responds in iteration 3."""
        client = StubLLMClient(
            [
                _create_claim_response("Step 1: identify knowns"),
                _create_claim_response("Step 2: apply formula", confidence=0.75),
                _respond_response("Step 3: answer is 42."),
            ]
        )
        repo = await _fresh_repo()
        loop = ReasoningLoop(llm_client=client, claim_repository=repo, max_iterations=5)

        result = await loop.run(query="Solve a math problem.")

        assert result.halted_reason == "respond_to_user"
        assert result.iterations == 3
        assert len(result.claims_created) == 2

    @pytest.mark.asyncio
    async def test_exploration_claims_are_strings(self):
        """claims_created contains string IDs (not Claim objects)."""
        client = StubLLMClient(
            [
                _create_claim_response("Alpha exploration claim content here"),
                _respond_response("Done."),
            ]
        )
        repo = await _fresh_repo()
        loop = ReasoningLoop(llm_client=client, claim_repository=repo, max_iterations=5)

        result = await loop.run(query="Check types.")

        for claim_id in result.claims_created:
            assert isinstance(claim_id, str), (
                f"Expected str, got {type(claim_id)}: {claim_id}"
            )


# ---------------------------------------------------------------------------
# test_max_iterations_enforced
# ---------------------------------------------------------------------------


class TestMaxIterationsEnforced:
    """Gate test: max_iterations safety backstop fires when LLM never halts."""

    @pytest.mark.asyncio
    async def test_max_iterations_enforced(self):
        """Loop terminates at max_iterations when LLM never calls respond_to_user."""
        # Queue more responses than max_iterations to show the backstop fires.
        explore_responses = [
            _create_claim_response(f"Exploration step {i} for iteration test")
            for i in range(10)
        ]
        client = StubLLMClient(explore_responses)
        repo = await _fresh_repo()
        loop = ReasoningLoop(llm_client=client, claim_repository=repo, max_iterations=3)

        result = await loop.run(query="Never-ending question?")

        assert result.halted_reason == "max_iterations"
        assert result.iterations == 3

    @pytest.mark.asyncio
    async def test_max_iterations_result_has_non_empty_response(self):
        """Fallback response is non-empty when max_iterations is reached."""
        explore_responses = [
            _create_claim_response(f"Step {i} content for max iteration test")
            for i in range(5)
        ]
        client = StubLLMClient(explore_responses)
        repo = await _fresh_repo()
        loop = ReasoningLoop(llm_client=client, claim_repository=repo, max_iterations=2)

        result = await loop.run(query="Question with no halt.")

        assert result.halted_reason == "max_iterations"
        assert len(result.response) > 0

    @pytest.mark.asyncio
    async def test_max_iterations_one_is_valid(self):
        """max_iterations=1 with an exploring LLM halts after exactly 1 iteration."""
        client = StubLLMClient(
            [_create_claim_response("Single exploration step content here")]
        )
        repo = await _fresh_repo()
        loop = ReasoningLoop(llm_client=client, claim_repository=repo, max_iterations=1)

        result = await loop.run(query="Will this halt?")

        assert result.iterations == 1
        assert result.halted_reason == "max_iterations"

    def test_max_iterations_less_than_one_raises(self):
        """max_iterations < 1 raises ValueError at construction time."""
        import pytest as _pytest
        with _pytest.raises(ValueError):
            ReasoningLoop(llm_client=None, claim_repository=None, max_iterations=0)


# ---------------------------------------------------------------------------
# test_claims_tracked_across_iterations
# ---------------------------------------------------------------------------


class TestClaimsTrackedAcrossIterations:
    """Gate test: all created claim IDs are tracked across all iterations."""

    @pytest.mark.asyncio
    async def test_claims_tracked_across_iterations(self):
        """Claim IDs from all iterations are collected in result.claims_created."""
        client = StubLLMClient(
            [
                _create_claim_response("Alpha claim in iteration one"),
                _create_claim_response("Beta claim in iteration two"),
                _create_claim_response("Gamma claim in iteration three"),
                _respond_response("Done exploring."),
            ]
        )
        repo = await _fresh_repo()
        loop = ReasoningLoop(llm_client=client, claim_repository=repo, max_iterations=5)

        result = await loop.run(query="Multi-step problem.")

        assert result.halted_reason == "respond_to_user"
        assert len(result.claims_created) == 3
        # All IDs should be distinct strings.
        assert len(set(result.claims_created)) == 3

    @pytest.mark.asyncio
    async def test_claims_created_ids_persist_in_repo(self):
        """Each claim ID in result.claims_created points to a real claim in the repo."""
        client = StubLLMClient(
            [
                _create_claim_response("Persisted reasoning claim for repo test"),
                _respond_response("Done."),
            ]
        )
        repo = await _fresh_repo()
        loop = ReasoningLoop(llm_client=client, claim_repository=repo, max_iterations=5)

        result = await loop.run(query="Verify persistence.")

        for claim_id in result.claims_created:
            stored = await repo.get_by_id(claim_id)
            assert stored is not None, f"Claim {claim_id} not found in repository"

    @pytest.mark.asyncio
    async def test_tool_calls_trace_records_all_iterations(self):
        """tool_calls trace contains an entry for each tool call across all iterations."""
        client = StubLLMClient(
            [
                _create_claim_response("Trace test step one iteration claim"),
                _create_claim_response("Trace test step two iteration claim"),
                _respond_response("Trace test done."),
            ]
        )
        repo = await _fresh_repo()
        loop = ReasoningLoop(llm_client=client, claim_repository=repo, max_iterations=5)

        result = await loop.run(query="Trace test.")

        # 2 create_claim + 1 respond_to_user = 3 tool calls
        assert len(result.tool_calls) == 3
        tools_used = [tc["tool"] for tc in result.tool_calls]
        assert tools_used.count("create_claim") == 2
        assert tools_used.count("respond_to_user") == 1

    @pytest.mark.asyncio
    async def test_tool_calls_trace_has_iteration_numbers(self):
        """Each tool call trace entry carries the correct iteration number."""
        client = StubLLMClient(
            [
                _create_claim_response("Iteration one claim for trace number test"),
                _respond_response("Answer."),
            ]
        )
        repo = await _fresh_repo()
        loop = ReasoningLoop(llm_client=client, claim_repository=repo, max_iterations=5)

        result = await loop.run(query="Iteration numbering?")

        iterations_seen = [tc["iteration"] for tc in result.tool_calls]
        assert 1 in iterations_seen
        assert 2 in iterations_seen


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_query_raises(self):
        """Empty query raises ValueError."""
        client = StubLLMClient([])
        repo = await _fresh_repo()
        loop = ReasoningLoop(llm_client=client, claim_repository=repo)

        with pytest.raises(ValueError):
            await loop.run(query="")

    @pytest.mark.asyncio
    async def test_whitespace_query_raises(self):
        """Whitespace-only query raises ValueError."""
        client = StubLLMClient([])
        repo = await _fresh_repo()
        loop = ReasoningLoop(llm_client=client, claim_repository=repo)

        with pytest.raises(ValueError):
            await loop.run(query="   \t\n")

    @pytest.mark.asyncio
    async def test_no_tool_calls_halts_with_no_tools(self):
        """LLM returning no tool calls halts with halt_reason='no_tools'."""
        client = StubLLMClient([_text_only_response("Plain text.")])
        repo = await _fresh_repo()
        loop = ReasoningLoop(llm_client=client, claim_repository=repo, max_iterations=5)

        result = await loop.run(query="Implicit halt?")

        assert result.halted_reason == "no_tools"
        assert result.response == "Plain text."
        assert result.iterations == 1

    @pytest.mark.asyncio
    async def test_no_content_no_tools_halts(self):
        """LLM returning neither tool calls nor content halts with 'no_tools'."""
        client = StubLLMClient([_no_content_no_tools()])
        repo = await _fresh_repo()
        loop = ReasoningLoop(llm_client=client, claim_repository=repo, max_iterations=5)

        result = await loop.run(query="Empty response?")

        assert result.halted_reason == "no_tools"

    @pytest.mark.asyncio
    async def test_llm_error_halts_with_tool_error(self):
        """LLM raising an exception halts with halt_reason='tool_error'."""

        class ErrorClient:
            async def generate_with_tools(self, **kwargs):
                raise RuntimeError("LLM unavailable")

        repo = await _fresh_repo()
        loop = ReasoningLoop(
            llm_client=ErrorClient(),
            claim_repository=repo,
            max_iterations=3,
        )

        result = await loop.run(query="Will this error out?")

        assert result.halted_reason == "tool_error"
        assert len(result.response) > 0

    @pytest.mark.asyncio
    async def test_context_claims_included_in_first_prompt(self):
        """context_claims seed is passed to the first iteration prompt."""
        prompt_captured = []

        class CapturingClient:
            async def generate_with_tools(self, prompt, tools, system_prompt="", **kwargs):
                prompt_captured.append(prompt)
                # Respond immediately.
                return _respond_response("Done.")

        repo = await _fresh_repo()
        loop = ReasoningLoop(
            llm_client=CapturingClient(),
            claim_repository=repo,
            max_iterations=5,
        )
        context = [
            {"content": "Known fact: the sky is blue", "confidence": 0.95}
        ]

        await loop.run(query="What colour is the sky?", context_claims=context)

        assert len(prompt_captured) == 1
        assert "Known fact" in prompt_captured[0]
        assert "sky is blue" in prompt_captured[0]

    @pytest.mark.asyncio
    async def test_result_is_reasoning_result_instance(self):
        """run() always returns a ReasoningResult regardless of halt reason."""
        for responses in [
            [_respond_response("Answer.")],                 # respond_to_user
            [_text_only_response("Text.")],                 # no_tools
            [_create_claim_response("Claim content ok")],  # max_iterations (limit=1)
        ]:
            client = StubLLMClient(responses)
            repo = await _fresh_repo()
            loop = ReasoningLoop(
                llm_client=client, claim_repository=repo, max_iterations=1
            )
            result = await loop.run(query="Any query?")
            assert isinstance(result, ReasoningResult), (
                f"Expected ReasoningResult, got {type(result)}"
            )

    @pytest.mark.asyncio
    async def test_reasoning_state_dataclass_defaults(self):
        """ReasoningState initialises with sensible defaults."""
        state = ReasoningState()
        assert state.iteration == 0
        assert state.claims_created == []
        assert state.tool_calls == []
        assert state.final_response is None
        assert state.supporting_claims == []
        assert state.halted is False
        assert state.halt_reason is None
