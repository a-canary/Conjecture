# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Evaluator - LLM reasoning evaluation for ConjectureEndpoint.

Extracted from ConjectureEndpoint.evaluate() (slice 2 of
improve-architecture:src/endpoint/conjecture_endpoint.py split).

Per A-0003, evaluate() is one of the three core endpoint methods.
Per A-0009, the input is decomposed into constituent claims before reasoning.
Per A-0010, the LLM operates via structured claim tools so all reasoning
    is traceable through the claim graph.
Per A-0012, the LLM may use ReasoningLoop to halt-or-explore.
Per A-0014, evaluation state is published for streaming consumers.
Per A-0015, retrieve_knowledge pauses the session and stores state.
Per O-0009, classify_query() routes queries to REASONING/RECALL/MATH.
"""

import logging
import uuid
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from src.agent.task_router import QueryType
    from src.endpoint.conjecture_endpoint import APIResponse

logger = logging.getLogger(__name__)


async def run_evaluate(endpoint, query: str,
                       max_claims: int = 10,
                       min_confidence: float = 0.5,
                       include_reasoning: bool = True,
                       use_decomposition: bool = True,
                       use_tools: bool = True,
                       max_tool_iterations: int = 5,
                       use_reasoning_loop: bool = False,
                       route: Optional["QueryType"] = None) -> "APIResponse":
    """Evaluate claims using LLM reasoning.

    Delegates from ConjectureEndpoint.evaluate(); behavior is identical.
    The endpoint is passed by reference so this function can use its
    session/claim/evaluation-state methods without inheritance.
    """
    from src.endpoint.conjecture_endpoint import APIResponse

    error_response = await endpoint._ensure_initialized()
    if error_response:
        return error_response  # type: ignore[no-any-return]

    # O-0009: classify query for prompt strategy routing
    if route is None:
        from src.agent.task_router import classify_query
        query_type = classify_query(query)
    else:
        query_type = route

    logger.info(f"O-0009 routing: query_type={query_type.value}")

    if query_type.value == "recall":
        use_decomposition = False

    try:
        from src.endpoint.llm_client import (
            LLMClient, build_claim_context, build_enhanced_prompt,
            DEFAULT_MODEL, TOOL_CAPABLE_MODEL
        )

        model = TOOL_CAPABLE_MODEL if use_tools else DEFAULT_MODEL
        llm = LLMClient(model=model)

        # Step 1: A-0009 Input Decomposition (optional, non-blocking)
        decomposed_claims = []
        if use_decomposition:
            # Look up through the endpoint module so test patches of
            # `src.endpoint.conjecture_endpoint.decompose_input` still take effect.
            import src.endpoint.conjecture_endpoint as _ep
            decompose_input = _ep.decompose_input
            try:
                decomposed_claims = await decompose_input(query, llm_client=llm)
                logger.info(f"Decomposed input into {len(decomposed_claims)} claims")
            except Exception as decomp_err:
                logger.warning(
                    f"Input decomposition failed (continuing without it): {decomp_err}"
                )
                decomposed_claims = []

        # Step 2: Search for relevant existing claims
        claims_response = await endpoint.search_claims(
            query=query,
            min_confidence=min_confidence,
            limit=max_claims
        )
        claims = []
        if claims_response.success and claims_response.data:
            claims = claims_response.data.get("claims", [])

        if not claims:
            all_claims_response = await endpoint.search_claims(
                min_confidence=min_confidence,
                limit=max_claims
            )
            if all_claims_response.success and all_claims_response.data:
                claims = all_claims_response.data.get("claims", [])

        # Step 3: Merge decomposed claims into context
        all_context_claims = list(claims)
        if decomposed_claims:
            for dc in decomposed_claims:
                all_context_claims.append(
                    dc.model_dump() if hasattr(dc, "model_dump") else dc
                )

        # Step 3b: A-0012 ReasoningLoop path (optional)
        if use_reasoning_loop:
            try:
                from src.process.reasoning_loop import ReasoningLoop
                from src.data.repositories import ClaimRepository

                repo = ClaimRepository()
                await repo.initialize()

                reasoning_loop = ReasoningLoop(
                    llm_client=llm,
                    claim_repository=repo,
                    max_iterations=max_tool_iterations,
                )

                context_for_loop = [
                    c if isinstance(c, dict) else c
                    for c in all_context_claims
                ]

                reasoning_result = await reasoning_loop.run(
                    query=query,
                    context_claims=context_for_loop,
                )
                return APIResponse(
                    success=True,
                    message="Evaluation complete (reasoning loop)",
                    data={
                        "query": query,
                        "response": reasoning_result.response,
                        "claims_used": len(claims),
                        "decomposed_claims": len(decomposed_claims),
                        "reasoning_result": {
                            "claims_created": reasoning_result.claims_created,
                            "supporting_claims": reasoning_result.supporting_claims,
                            "iterations": reasoning_result.iterations,
                            "halted_reason": reasoning_result.halted_reason,
                            "tool_calls": reasoning_result.tool_calls,
                        },
                    }
                )
            finally:
                await llm.close()

        # Step 4: Build claim context and enhanced prompt
        claim_context = build_claim_context(all_context_claims)
        enhanced_prompt = build_enhanced_prompt(query, claim_context)

        # A-0014: session ID for evaluation state tracking
        eval_session_id: Optional[str] = (
            endpoint._current_session.id if endpoint._current_session else None
        )

        # Step 5: Call LLM (tool-calling mode or direct mode)
        try:
            if use_tools:
                from src.process.claim_tools import (
                    CLAIM_TOOLS, ClaimToolExecutor,
                    RETRIEVAL_TOOLS, PausedReasoningState, RetrievalRequest,
                )
                from src.data.repositories import ClaimRepository

                repo = ClaimRepository()
                await repo.initialize()
                executor = ClaimToolExecutor(repo)

                tool_calls_log: List[dict] = []
                created_claim_ids: List[str] = []
                supporting_claims: List[str] = []
                final_response = None
                llm_response = None

                system_prompt = (
                    "You are a reasoning assistant that uses structured tools to build knowledge. "
                    "Use create_claim to record facts and reasoning steps. "
                    "Use update_confidence if evidence changes your certainty about a claim. "
                    "Use respond_to_user to deliver your final answer, citing supporting claims."
                )

                all_tools = CLAIM_TOOLS + RETRIEVAL_TOOLS

                eval_session_id = (
                    endpoint._current_session.id
                    if endpoint._current_session
                    else f"s{uuid.uuid4().hex[:8]}"
                )
                eval_claim_ids: List[str] = list(claims) if claims else []

                for iteration in range(max_tool_iterations):
                    logger.info(
                        "Tool-calling loop: iteration %d/%d",
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

                    endpoint.publish_evaluation_state(
                        session_id=eval_session_id,
                        query=query,
                        iteration=iteration + 1,
                        max_iterations=max_tool_iterations,
                        claims_being_evaluated=eval_claim_ids,
                        tool_calls_so_far=tool_calls_log,
                        created_claim_ids=created_claim_ids,
                        status="in_progress",
                        llm_content=llm_content,
                    )

                    if not tool_calls:
                        logger.info(
                            "Tool-calling loop: no tool calls in iteration %d, "
                            "treating content as final response",
                            iteration + 1
                        )
                        final_response = llm_response.get("content", "")
                        break

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
                        pause_id = str(uuid.uuid4())
                        endpoint.publish_evaluation_state(
                            session_id=eval_session_id,
                            query=query,
                            iteration=iteration + 1,
                            max_iterations=max_tool_iterations,
                            claims_being_evaluated=eval_claim_ids,
                            tool_calls_so_far=tool_calls_log,
                            created_claim_ids=created_claim_ids,
                            status="paused",
                            llm_content=llm_content,
                        )
                        paused_state = PausedReasoningState(
                            session_id=eval_session_id,
                            iteration=iteration,
                            messages=[],
                            pending_retrieval=retrieval_request,
                            created_claim_ids=list(created_claim_ids),
                        )
                        endpoint._paused_states[pause_id] = paused_state
                        logger.info(
                            "retrieve_knowledge called at iteration %d — "
                            "pausing session %s with pause_id %s",
                            iteration + 1, eval_session_id, pause_id,
                        )
                        return APIResponse(
                            success=True,
                            message="Evaluation paused — awaiting retrieval results",
                            data={
                                "status": "paused",
                                "pause_id": pause_id,
                                "session_id": eval_session_id,
                                "retrieval_request": {
                                    "query": retrieval_request.query,
                                    "tool_hint": retrieval_request.tool_hint,
                                    "claim_ids": retrieval_request.claim_ids,
                                },
                                "query": query,
                                "tool_calls_so_far": tool_calls_log,
                                "created_claim_ids": list(created_claim_ids),
                            }
                        )

                    halted = False
                    for tc in tool_calls:
                        tool_name = tc.get("name", "")
                        tool_args = tc.get("arguments", {})
                        logger.info(
                            "Executing tool '%s' (iteration %d)",
                            tool_name, iteration + 1
                        )
                        endpoint.publish_evaluation_state(
                            session_id=eval_session_id,
                            query=query,
                            iteration=iteration + 1,
                            max_iterations=max_tool_iterations,
                            claims_being_evaluated=eval_claim_ids,
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
                            "iteration": iteration + 1,
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
                                "respond_to_user called — halting tool loop after %d iterations",
                                iteration + 1
                            )
                            halted = True
                            break

                    if halted or final_response is not None:
                        break

                response_text = (
                    final_response
                    if final_response is not None
                    else (llm_response.get("content", "") if llm_response else "")
                )
                endpoint.clear_evaluation_state(eval_session_id)
                return APIResponse(
                    success=True,
                    message="Evaluation complete (tool mode)",
                    data={
                        "status": "complete",
                        "query": query,
                        "response": response_text,
                        "claims_used": len(claims),
                        "decomposed_claims": len(decomposed_claims),
                        "tool_calls": tool_calls_log,
                        "tool_iterations": len(tool_calls_log),
                        "created_claim_ids": created_claim_ids,
                        "supporting_claims": supporting_claims,
                        "claim_context": claim_context if include_reasoning else None,
                        "model": llm_response.get("model", "unknown") if llm_response else "unknown",
                        "usage": llm_response.get("usage", {}) if llm_response else {}
                    }
                )

            else:
                llm_response = await llm.generate(
                    prompt=enhanced_prompt,
                    temperature=0.7,
                    max_tokens=1024
                )
        finally:
            await llm.close()

        endpoint.clear_evaluation_state(eval_session_id)
        return APIResponse(
            success=True,
            message="Evaluation complete",
            data={
                "query": query,
                "query_type": query_type.value,
                "response": llm_response.get("content", ""),
                "claims_used": len(claims),
                "decomposed_claims": len(decomposed_claims),
                "claim_context": claim_context if include_reasoning else None,
                "enhanced_prompt": enhanced_prompt if include_reasoning else None,
                "model": llm_response.get("model", "unknown"),
                "usage": llm_response.get("usage", {})
            }
        )

    except ValueError as e:
        endpoint.clear_evaluation_state(eval_session_id)
        return APIResponse(
            success=False,
            message="LLM not configured",
            errors=[str(e), "Set CHUTES_API_KEY environment variable"]
        )
    except Exception as e:
        logger.error(f"Failed to evaluate query: {e}")
        endpoint.clear_evaluation_state(eval_session_id)
        return APIResponse(
            success=False,
            message="Failed to evaluate query",
            errors=[str(e)]
        )


__all__ = ["run_evaluate"]
