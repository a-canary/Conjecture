"""
Reasoning Loop — A-0012

Per A-0012: "LLM decides whether to halt and respond OR explore further by
creating new claims to question or investigate.  Not a system-imposed
threshold.  If LLM is unsatisfied with confidence or evidence, it creates
new claims rather than forcing a response."

This module orchestrates the halt-or-explore decision loop.  Each iteration
calls the LLM with the full CLAIM_TOOLS set.  The loop continues until the
LLM issues a ``respond_to_user`` tool call (indicating it is satisfied) or
the configurable ``max_iterations`` ceiling is reached as a safety backstop.

All claims created during the session are tracked and returned in
``ReasoningResult`` so callers have a full reasoning trace.

Design:
  - ``ReasoningState``  — mutable state threaded through each iteration.
  - ``ReasoningResult`` — immutable result returned to the caller.
  - ``ReasoningLoop``   — async orchestrator with ``run(query)`` entry point.

Halt reasons (``ReasoningResult.halted_reason``):
  - ``"respond_to_user"``  — LLM explicitly chose to halt and respond.
  - ``"max_iterations"``   — safety backstop was hit.
  - ``"no_tools"``         — LLM returned no tool calls (implicit halt).
  - ``"tool_error"``       — an unrecoverable LLM or tool error occurred.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.process.claim_tools import CLAIM_TOOLS, ClaimToolExecutor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State and result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ReasoningState:
    """Mutable state threaded through each iteration of the reasoning loop.

    Attributes:
        iteration: Current iteration number (1-indexed; 0 before loop starts).
        claims_created: IDs of claims created during this reasoning session.
        tool_calls: Ordered log of every tool call made and its outcome.
        final_response: Text to deliver to the user once halted.
        supporting_claims: Claim IDs cited in the final respond_to_user call.
        halted: True once the loop has terminated.
        halt_reason: One of ``"respond_to_user"``, ``"max_iterations"``,
            ``"no_tools"``, or ``"tool_error"``.
    """

    iteration: int = 0
    claims_created: List[str] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    final_response: Optional[str] = None
    supporting_claims: List[str] = field(default_factory=list)
    halted: bool = False
    halt_reason: Optional[str] = None


@dataclass
class ReasoningResult:
    """Immutable result returned by ``ReasoningLoop.run()``.

    Attributes:
        response: Final response text for the user.
        claims_created: IDs of all claims created during reasoning.
        supporting_claims: Claim IDs cited as evidence in the final response.
        iterations: Number of loop iterations executed.
        halted_reason: Why the loop terminated — one of ``"respond_to_user"``,
            ``"max_iterations"``, ``"no_tools"``, or ``"tool_error"``.
        tool_calls: Full ordered trace of tool calls and their outcomes.
    """

    response: str
    claims_created: List[str]
    supporting_claims: List[str]
    iterations: int
    halted_reason: str
    tool_calls: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a careful reasoning assistant.\n"
    "\n"
    "For every query you may either:\n"
    "  - Explore further: call create_claim to record observations,\n"
    "    sub-questions, or intermediate conclusions that help build\n"
    "    your understanding.\n"
    "  - Halt and respond: call respond_to_user when you have\n"
    "    sufficient confidence and evidence to give a good answer.\n"
    "\n"
    "Rules:\n"
    "  1. Always cite supporting claim IDs in respond_to_user.\n"
    "  2. Explore only when genuinely uncertain — do not over-explore.\n"
    "  3. When you have enough information, call respond_to_user.\n"
    "  4. Every reasoning step you externalise should become a claim.\n"
    "  5. Prefer accuracy over speed; prefer brevity over verbosity."
)


# ---------------------------------------------------------------------------
# ReasoningLoop
# ---------------------------------------------------------------------------


class ReasoningLoop:
    """Orchestrates the LLM halt-or-explore decision loop.

    Per A-0012, the LLM — not a system-imposed threshold — decides when it
    has sufficient confidence to respond.  The loop continues until:

    1. The LLM calls ``respond_to_user``      → preferred exit.
    2. ``max_iterations`` is reached           → safety backstop.
    3. The LLM returns no tool calls           → implicit halt.

    Each iteration is a round-trip to the LLM.  Newly created claims are
    tracked in ``ReasoningState`` so the prompt can include the running
    reasoning trace for the next iteration.

    Args:
        llm_client: Object with an async ``generate_with_tools`` method
            matching the signature in ``src/endpoint/llm_client.py``.
        claim_repository: Initialised ``ClaimRepository`` instance.
        max_iterations: Safety ceiling on the number of LLM calls (default 5).
            The LLM is encouraged to halt when ready; this backstop prevents
            infinite loops.

    Raises:
        ValueError: If ``max_iterations < 1``.
    """

    def __init__(
        self,
        llm_client: Any,
        claim_repository: Any,
        max_iterations: int = 5,
    ) -> None:
        if max_iterations < 1:
            raise ValueError(
                f"max_iterations must be >= 1, got {max_iterations}"
            )
        self.llm_client = llm_client
        self.claim_repository = claim_repository
        self.max_iterations = max_iterations
        self._executor = ClaimToolExecutor(claim_repository)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        query: str,
        context_claims: Optional[List[Dict[str, Any]]] = None,
    ) -> ReasoningResult:
        """Run the reasoning loop until the LLM halts or max iterations reached.

        Args:
            query: The user's question or request.
            context_claims: Optional pre-loaded claims to seed the first
                prompt.  Each item should be a dict with at least ``content``
                and ``confidence`` keys.

        Returns:
            ``ReasoningResult`` with response, full claims list, and execution
            trace.

        Raises:
            ValueError: If *query* is empty or whitespace-only.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        state = ReasoningState()

        while state.iteration < self.max_iterations and not state.halted:
            state.iteration += 1
            logger.debug(
                "ReasoningLoop iteration %d / %d",
                state.iteration,
                self.max_iterations,
            )

            prompt = self._build_prompt(query, context_claims, state)

            # Call the LLM with the full claim-tool set.
            try:
                response = await self.llm_client.generate_with_tools(
                    prompt=prompt,
                    tools=CLAIM_TOOLS,
                    system_prompt=_SYSTEM_PROMPT,
                )
            except Exception as exc:
                logger.error(
                    "LLM call failed on iteration %d: %s", state.iteration, exc
                )
                state.tool_calls.append(
                    {
                        "iteration": state.iteration,
                        "tool": None,
                        "args": None,
                        "success": False,
                        "error": str(exc),
                        "created_ids": [],
                    }
                )
                state.halted = True
                state.halt_reason = "tool_error"
                break

            await self._process_response(response, state)

        # If the loop ended because we hit the iteration ceiling, record that.
        if not state.halted:
            logger.warning(
                "ReasoningLoop reached max_iterations=%d without halt; "
                "forcing stop.",
                self.max_iterations,
            )
            state.halted = True
            state.halt_reason = "max_iterations"

        return self._build_result(state)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        query: str,
        context_claims: Optional[List[Dict[str, Any]]],
        state: ReasoningState,
    ) -> str:
        """Construct the prompt for the current iteration.

        The prompt includes the original query, any seed context claims, and
        the running log of claims created so far.

        Args:
            query: The original user query.
            context_claims: Seed claims provided to the first call.
            state: Current loop state.

        Returns:
            Formatted prompt string.
        """
        parts: List[str] = []

        # Seed context (provided externally — typically from RAG / search).
        if context_claims:
            parts.append("Relevant knowledge (confidence in brackets):")
            for i, claim in enumerate(context_claims, 1):
                content = claim.get("content", "")
                confidence = claim.get("confidence", 0.5)
                parts.append(f"  {i}. [{confidence:.0%}] {content}")
            parts.append("")

        # Original query.
        parts.append(f"Query: {query}")
        parts.append("")

        # Running reasoning trace (non-empty after iteration 1).
        if state.claims_created:
            parts.append(
                f"Claims you have created so far "
                f"(after iteration {state.iteration - 1}):"
            )
            for cid in state.claims_created:
                # Include the claim content from the trace if available.
                label = cid
                for tc in state.tool_calls:
                    if (
                        tc.get("tool") == "create_claim"
                        and cid in tc.get("created_ids", [])
                    ):
                        label = (
                            f"{cid}: {tc['args'].get('content', '')}"
                        )
                        break
                parts.append(f"  - {label}")
            parts.append("")

        if state.iteration > 1:
            remaining = self.max_iterations - state.iteration
            if remaining <= 0:
                parts.append(
                    "This is your last iteration. You MUST call "
                    "respond_to_user now with your best current answer."
                )
            else:
                parts.append(
                    f"Continue reasoning ({remaining} iteration(s) remaining). "
                    "If you now have enough evidence, call respond_to_user. "
                    "Otherwise, create more claims to explore further."
                )
        else:
            parts.append(
                "Begin reasoning. Create claims to capture your thoughts, "
                "or respond directly if you already have sufficient confidence."
            )

        return "\n".join(parts)

    async def _process_response(
        self, response: Dict[str, Any], state: ReasoningState
    ) -> None:
        """Execute every tool call in the LLM response and update state.

        If the LLM issued no tool calls we treat this as an implicit halt
        (``halt_reason = "no_tools"``).  Any raw text content from the LLM
        is captured as the final response in that case.

        Args:
            response: Raw dict returned by ``llm_client.generate_with_tools``.
            state: Mutable loop state to update in place.
        """
        tool_calls: List[Dict[str, Any]] = response.get("tool_calls", [])

        if not tool_calls:
            logger.debug(
                "LLM returned no tool calls on iteration %d; "
                "halting with 'no_tools'.",
                state.iteration,
            )
            state.halted = True
            state.halt_reason = "no_tools"
            # Capture any raw text the LLM may have emitted.
            raw_content = response.get("content", "")
            if raw_content and state.final_response is None:
                state.final_response = raw_content
            return

        for tc in tool_calls:
            tool_name: str = tc.get("name", "")
            args: Dict[str, Any] = tc.get("arguments", {})

            tool_result = await self._executor.execute_tool(tool_name, args)

            trace_entry: Dict[str, Any] = {
                "iteration": state.iteration,
                "tool": tool_name,
                "args": args,
                "success": tool_result.success,
                "error": tool_result.error,
                "created_ids": [],
            }

            if tool_result.success:
                if tool_name == "create_claim":
                    created = tool_result.claim_ids
                    state.claims_created.extend(created)
                    trace_entry["created_ids"] = created
                    logger.debug(
                        "Iteration %d: created claim(s) %s",
                        state.iteration,
                        created,
                    )

                elif tool_name == "update_confidence":
                    logger.debug(
                        "Iteration %d: updated confidence on %s",
                        state.iteration,
                        tool_result.claim_ids,
                    )

                elif tool_name == "respond_to_user":
                    payload = tool_result.result or {}
                    state.final_response = payload.get("response", "")
                    state.supporting_claims = payload.get(
                        "supporting_claims", []
                    )
                    state.halted = True
                    state.halt_reason = "respond_to_user"
                    logger.debug(
                        "Iteration %d: LLM halted via respond_to_user.",
                        state.iteration,
                    )
            else:
                logger.warning(
                    "Iteration %d: tool '%s' failed: %s",
                    state.iteration,
                    tool_name,
                    tool_result.error,
                )

            state.tool_calls.append(trace_entry)

            # Stop processing further tool calls in this batch once halted.
            if state.halted:
                break

    def _build_result(self, state: ReasoningState) -> ReasoningResult:
        """Convert the final loop state into an immutable ``ReasoningResult``.

        Synthesises a fallback message if the loop ended without an explicit
        ``respond_to_user`` call so the caller always receives a non-empty
        response.

        Args:
            state: Final loop state.

        Returns:
            ``ReasoningResult`` ready for the caller.
        """
        response = state.final_response

        if not response:
            if state.halt_reason == "max_iterations":
                response = (
                    "I reached the maximum number of reasoning steps without "
                    "arriving at a confident answer. The claims I created "
                    "during exploration may still be useful context."
                )
            elif state.halt_reason == "no_tools":
                response = (
                    "The reasoning process concluded without producing a "
                    "response."
                )
            elif state.halt_reason == "tool_error":
                response = (
                    "The reasoning process was interrupted by an error. "
                    "Please try again."
                )
            else:
                response = ""

        return ReasoningResult(
            response=response,
            claims_created=list(state.claims_created),
            supporting_claims=list(state.supporting_claims),
            iterations=state.iteration,
            halted_reason=state.halt_reason or "unknown",
            tool_calls=list(state.tool_calls),
        )
