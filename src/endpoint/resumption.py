# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Resumption — Resume paused evaluations after delegated retrieval.

Per A-0015 (Delegated Tool Calling for Knowledge Retrieval) and A-0014
(Streaming Evaluation State), when evaluate() returns status="paused", the
caller performs the requested retrieval and calls resume_evaluation() with
the results. This module owns that second-leg tool-calling loop so the main
endpoint class stays focused on session/claim CRUD.

The function takes an `endpoint` duck-typed on three attributes so it can
live outside ConjectureEndpoint without inheriting or copying the class:
  - endpoint._paused_states: Dict[str, PausedReasoningState]
  - endpoint.publish_evaluation_state(...): A-0014 broadcaster
  - endpoint.clear_evaluation_state(session_id): A-0014 cleanup
"""

import logging
import uuid
from typing import Any, List

from src.endpoint.conjecture_endpoint import APIResponse
from src.process.input_decomposer import decompose_input

logger = logging.getLogger(__name__)


async def resume_evaluation(
    endpoint: Any,
    pause_id: str,
    retrieval_results: List[str],
    max_tool_iterations: int = 5,
    include_reasoning: bool = True,
) -> APIResponse:
    """Resume a paused evaluation after the caller has performed retrieval.

    Per A-0015, when evaluate() returns status="paused", the caller performs
    the requested knowledge retrieval and calls this method with the results.
    Each result string is decomposed into claims via decompose_input() and
    appended to the reasoning context, then the tool-calling loop continues
    from where it left off.

    Args:
        endpoint: Duck-typed host exposing ``_paused_states`` (dict),
            ``publish_evaluation_state(...)``, and ``clear_evaluation_state(...)``.
            Decoupled from ConjectureEndpoint so this module is testable and
            keeps the endpoint class small.
        pause_id: The pause_id returned by the paused evaluate() call.
        retrieval_results: List of retrieved text strings (e.g. passages,
            facts, document excerpts) to incorporate as evidence claims.
        max_tool_iterations: Maximum additional tool iterations to run
            after resuming (default 5).
        include_reasoning: Whether to include reasoning metadata in response.

    Returns:
        APIResponse with status="complete" and the final response, or
        another status="paused" if a second retrieval is requested, or
        an error response if the pause_id is not found.
    """
    # Look up the paused state
    paused_state = endpoint._paused_states.pop(pause_id, None)
    if paused_state is None:
        return APIResponse(
            success=False,
            message=f"No paused session found for pause_id: {pause_id}",
            errors=["PAUSE_ID_NOT_FOUND"]
        )

    try:
        from src.endpoint.llm_client import (
            LLMClient, build_claim_context, build_enhanced_prompt,
            TOOL_CAPABLE_MODEL
        )
        from src.process.claim_tools import (
            CLAIM_TOOLS, RETRIEVAL_TOOLS, ClaimToolExecutor,
            PausedReasoningState, RetrievalRequest
        )
        from src.data.repositories import ClaimRepository

        llm = LLMClient(model=TOOL_CAPABLE_MODEL)

        try:
            # Step 1: Decompose each retrieval result into evidence claims
            evidence_claims = []
            for result_text in retrieval_results:
                if not result_text or not result_text.strip():
                    continue
                try:
                    new_claims = await decompose_input(result_text, llm_client=llm)
                    evidence_claims.extend(new_claims)
                    logger.info(
                        "resume_evaluation: decomposed retrieval result into %d claim(s)",
                        len(new_claims)
                    )
                except Exception as decomp_err:
                    logger.warning(
                        "resume_evaluation: decomposition of retrieval result failed "
                        "(%s) — using heuristic fallback",
                        decomp_err
                    )
                    # Heuristic fallback: wrap the raw text as-is
                    from src.process.input_decomposer import _heuristic_decompose
                    evidence_claims.extend(_heuristic_decompose(result_text))

            # Step 2: Build a new claim context incorporating the evidence
            evidence_dicts = [
                c.model_dump() if hasattr(c, "model_dump") else c
                for c in evidence_claims
            ]
            claim_context = build_claim_context(evidence_dicts)

            # Reconstruct the original query from the paused state's
            # retrieval request (best available proxy for the original prompt)
            original_query = paused_state.pending_retrieval.query
            enhanced_prompt = build_enhanced_prompt(original_query, claim_context)

            # Step 3: Continue the tool-calling loop
            repo = ClaimRepository()
            await repo.initialize()
            executor = ClaimToolExecutor(repo)

            all_tools = CLAIM_TOOLS + RETRIEVAL_TOOLS

            tool_calls_log = []
            created_claim_ids: List[str] = list(paused_state.created_claim_ids)
            supporting_claims: List[str] = []
            final_response = None
            llm_response = None

            system_prompt = (
                "You are a reasoning assistant that uses structured tools to build knowledge. "
                "Use create_claim to record facts and reasoning steps. "
                "Use update_confidence if evidence changes your certainty about a claim. "
                "Use respond_to_user to deliver your final answer, citing supporting claims."
            )

            for iteration in range(max_tool_iterations):
                logger.info(
                    "resume_evaluation: tool-calling loop iteration %d/%d",
                    iteration + 1, max_tool_iterations
                )
                llm_response = await llm.generate_with_tools(
                    prompt=enhanced_prompt,
                    tools=all_tools,
                    system_prompt=system_prompt,
                    temperature=0.7,
                    max_tokens=1024
                )

                tool_calls = llm_response.get("tool_calls", [])
                llm_content = llm_response.get("content")

                # A-0014: publish evaluation state after LLM call
                endpoint.publish_evaluation_state(
                    session_id=paused_state.session_id,
                    query=original_query,
                    iteration=iteration + 1,
                    max_iterations=max_tool_iterations,
                    claims_being_evaluated=[c.get("id") for c in evidence_dicts],
                    tool_calls_so_far=tool_calls_log,
                    created_claim_ids=created_claim_ids,
                    status="in_progress",
                    llm_content=llm_content,
                )

                if not tool_calls:
                    logger.info(
                        "resume_evaluation: no tool calls in iteration %d, "
                        "treating content as final response",
                        iteration + 1
                    )
                    final_response = llm_response.get("content", "")
                    break

                # A-0015: Guard — detect another retrieve_knowledge call
                iteration_tool_names = {tc.get("name", "") for tc in tool_calls}
                if "retrieve_knowledge" in iteration_tool_names:
                    rk_tc = next(
                        tc for tc in tool_calls
                        if tc.get("name") == "retrieve_knowledge"
                    )
                    rk_args = rk_tc.get("arguments", {})
                    retrieval_request = RetrievalRequest(
                        query=rk_args.get("query", ""),
                        tool_hint=rk_args.get("tool_hint"),
                        claim_ids=list(created_claim_ids),
                    )
                    new_pause_id = str(uuid.uuid4())
                    session_id = paused_state.session_id
                    new_paused_state = PausedReasoningState(
                        session_id=session_id,
                        iteration=paused_state.iteration + iteration,
                        messages=[],
                        pending_retrieval=retrieval_request,
                        created_claim_ids=list(created_claim_ids),
                    )
                    endpoint._paused_states[new_pause_id] = new_paused_state
                    logger.info(
                        "resume_evaluation: second retrieve_knowledge at iteration %d — "
                        "pausing again with pause_id %s",
                        iteration + 1, new_pause_id,
                    )
                    return APIResponse(
                        success=True,
                        message="Evaluation paused again — awaiting retrieval results",
                        data={
                            "status": "paused",
                            "pause_id": new_pause_id,
                            "retrieval_request": {
                                "query": retrieval_request.query,
                                "tool_hint": retrieval_request.tool_hint,
                                "claim_ids": retrieval_request.claim_ids,
                            },
                            "tool_calls_so_far": tool_calls_log,
                            "created_claim_ids": list(created_claim_ids),
                        }
                    )

                halted = False
                for tc in tool_calls:
                    tool_name = tc.get("name", "")
                    tool_args = tc.get("arguments", {})
                    logger.info(
                        "resume_evaluation: executing tool '%s' (iteration %d)",
                        tool_name, iteration + 1
                    )
                    # A-0014: publish state with current_tool during execution
                    endpoint.publish_evaluation_state(
                        session_id=paused_state.session_id,
                        query=original_query,
                        iteration=iteration + 1,
                        max_iterations=max_tool_iterations,
                        claims_being_evaluated=[c.get("id") for c in evidence_dicts],
                        tool_calls_so_far=tool_calls_log,
                        created_claim_ids=created_claim_ids,
                        status="in_progress",
                        current_tool=tool_name,
                        llm_content=llm_content,
                    )
                    result = await executor.execute_tool(tool_name, tool_args)
                    tool_calls_log.append({
                        "name": tool_name,
                        "arguments": tool_args,
                        "success": result.success,
                        "claim_ids": result.claim_ids,
                        "error": result.error,
                        "iteration": paused_state.iteration + iteration + 1,
                    })

                    if result.success:
                        created_claim_ids.extend(result.claim_ids)

                    if tool_name == "respond_to_user" and result.success:
                        payload = result.result or {}
                        if isinstance(payload, dict):
                            final_response = payload.get("response", "")
                            supporting_claims = list(payload.get("supporting_claims", []))
                        else:
                            final_response = str(payload)
                        logger.info(
                            "resume_evaluation: respond_to_user called — halting after %d iterations",
                            iteration + 1
                        )
                        halted = True
                        break

                if halted or final_response is not None:
                    break

            # Return completed response
            response_text = (
                final_response
                if final_response is not None
                else (llm_response.get("content", "") if llm_response else "")
            )
            # A-0014: clear ephemeral evaluation state on completion
            endpoint.clear_evaluation_state(paused_state.session_id)
            return APIResponse(
                success=True,
                message="Evaluation complete (resumed)",
                data={
                    "status": "complete",
                    "response": response_text,
                    "tool_calls": tool_calls_log,
                    "created_claim_ids": created_claim_ids,
                    "supporting_claims": supporting_claims,
                    "evidence_claims_count": len(evidence_claims),
                    "claim_context": claim_context if include_reasoning else None,
                    "model": llm_response.get("model", "unknown") if llm_response else "unknown",
                    "usage": llm_response.get("usage", {}) if llm_response else {}
                }
            )

        finally:
            await llm.close()

    except ValueError as e:
        endpoint.clear_evaluation_state(paused_state.session_id)
        return APIResponse(
            success=False,
            message="LLM not configured",
            errors=[str(e), "Set CHUTES_API_KEY environment variable"]
        )
    except Exception as e:
        logger.error(f"resume_evaluation failed: {e}")
        endpoint.clear_evaluation_state(paused_state.session_id)
        return APIResponse(
            success=False,
            message="Failed to resume evaluation",
            errors=[str(e)]
        )
