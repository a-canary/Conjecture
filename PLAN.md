# Plan: A-0015 Delegated Tool Calling for Knowledge Retrieval

## Objective

Implement the request/resume protocol where the Conjecture LLM endpoint emits
structured retrieval tool-call requests to the calling system rather than
executing them directly. The endpoint suspends its reasoning loop, the caller
performs the retrieval, and the caller resumes the loop by appending results as
evidence claims. The core loop is: retrieve → decompose to claims → reason with
evidence. This unblocks the 8B small-model hypothesis: the -32pp BBH regression
(p=0.0007) may reflect missing retrieved evidence, not an architecture flaw.

## Scope

**In Scope:**
- `RETRIEVAL_TOOLS` schema definitions in `src/process/claim_tools.py`
- `PausedReasoningState` Pydantic model for serializing a suspended loop
- Suspend/resume protocol added to `src/endpoint/conjecture_endpoint.py`
- HTTP request/resume route pair in `src/endpoint/http_server.py`
- Backward compatibility: existing direct-claims mode unchanged
- BBH re-validation on 8B model after protocol is operative

**Out of Scope:**
- Implementing any specific retrieval tool (web search, database query, etc.)
- Changing the existing `CLAIM_TOOLS` schema or `ClaimToolExecutor`
- Streaming SSE responses
- Authentication or multi-tenant isolation

---

## Phase 1: Retrieval Tool Schema

### Steps
- [x] 1.1 Add `RETRIEVAL_TOOLS` list to `src/process/claim_tools.py` with a
      single generic `retrieve_knowledge` tool definition in OpenAI
      function-calling format (name, description, parameters: query string,
      tool_hint optional string)
- [x] 1.2 Add a `RetrievalRequest` Pydantic model (query, tool_hint, claim_ids
      that triggered the request) to `src/process/claim_tools.py`
- [x] 1.3 Add a `PausedReasoningState` Pydantic model to
      `src/process/claim_tools.py` capturing: session_id, iteration, messages
      list, pending_retrieval as `RetrievalRequest`, created_claim_ids so far
- [x] 1.4 Extend `ClaimToolExecutor.execute_tool` dispatch to recognise
      `retrieve_knowledge` and return a `ToolResult` with
      `paused=True` and the `RetrievalRequest` in `result`

### Gates
- [x] `python -c "from src.process.claim_tools import RETRIEVAL_TOOLS, RetrievalRequest, PausedReasoningState; print('schema OK')"` exits 0
- [x] `python -m pytest tests/ -k "claim_tools" -v` passes with no new failures

---

## Phase 2: Suspend/Resume in ConjectureEndpoint

### Steps
- [x] 2.1 Add `_paused_states: Dict[str, PausedReasoningState]` store to
      `ConjectureEndpoint.__init__`
- [x] 2.2 Modify the tool-calling loop in `evaluate()` to detect when
      `retrieve_knowledge` is called: capture `PausedReasoningState`, store it
      keyed by `session_id + ":" + pause_id`, and return an `APIResponse` with
      `success=True`, `data.status="paused"`, `data.retrieval_request=...`,
      and `data.pause_id=...`
- [x] 2.3 Add `resume_evaluation(pause_id, retrieval_results: List[str])` method
      to `ConjectureEndpoint`: look up paused state, decompose each result
      string into claims via `decompose_input`, append claims to context,
      continue the tool-calling loop from the saved iteration, return final
      `APIResponse`
- [x] 2.4 Ensure the existing `evaluate()` path with no retrieval tool calls is
      completely unaffected (add guard: only pause if `retrieve_knowledge` in
      tool names from this iteration)

### Gates
- [x] Mock test: call `evaluate()` with a stub LLM that emits
      `retrieve_knowledge`; assert response has `status="paused"` and a
      `pause_id`
- [x] Mock test: call `resume_evaluation(pause_id, ["The capital is Paris"])`;
      assert response has `status="complete"` and non-empty `response`
- [x] `python -m pytest tests/ -m "unit" -v` passes with no regressions

---

## Phase 3: HTTP Request/Resume Routes

### Steps
- [ ] 3.1 Add `POST /v1/chat/completions/resume` route to
      `src/endpoint/http_server.py` accepting JSON body:
      `{pause_id: str, results: List[str]}`; delegates to
      `endpoint.resume_evaluation()` and returns OpenAI-compatible response
- [ ] 3.2 Modify `POST /v1/chat/completions` response to include custom header
      `X-Conjecture-Pause-ID` when `data.status="paused"`, and populate the
      assistant message content with a structured JSON payload describing the
      retrieval request (so OpenAI-compatible clients see a well-formed response)
- [ ] 3.3 Add `GET /v1/pause/{pause_id}` route that returns the current
      `PausedReasoningState` as JSON (enables callers to inspect pending state)
- [ ] 3.4 Add integration smoke test: start server in-process, POST a request
      that triggers a pause, POST to `/resume`, assert final response

### Gates
- [ ] `python -m pytest tests/ -k "http" -v` passes with no regressions
- [ ] Manual curl: `POST /v1/chat/completions` with a prompt that triggers
      retrieval returns HTTP 200 with `X-Conjecture-Pause-ID` header set

---

## Phase 4: 8B BBH Re-Validation

### Steps
- [ ] 4.1 Create `experiments/bbh_delegated_retrieval_8b.py` benchmark script
      that acts as the caller: sends BBH problems to the HTTP server, detects
      pause responses, supplies a mock retrieval result (the problem statement
      re-stated as evidence), and resumes
- [ ] 4.2 Run the benchmark against Llama-3.1-8B via OpenRouter with n=50 BBH
      problems; record direct baseline, three-prompt without retrieval, and
      three-prompt with delegated retrieval
- [ ] 4.3 Compute p-values (two-proportion z-test) comparing delegated-retrieval
      vs baseline and vs three-prompt-no-retrieval
- [ ] 4.4 Record results in `experiments/results/bbh_delegated_8b_<timestamp>.json`
      and append row to `experiments/results/benchmark_results.csv`
- [ ] 4.5 Update `CHOICES.md` O-0008 note with finding: if p<0.05 improvement,
      note retrieval restored performance; if still regressed, document that
      architecture (not missing retrieval) is the cause

### Gates
- [ ] Benchmark script completes for n=50 without crash
- [ ] Result JSON contains `direct_accuracy`, `three_prompt_accuracy`,
      `delegated_accuracy`, `p_value_delegated_vs_direct`
- [ ] If `delegated_accuracy > direct_accuracy` with `p < 0.05`: note confirms
      hypothesis; if not: note disproves it — either outcome satisfies the gate

---

## Current Phase: 3
## Status: in-progress

---

## Success Criteria

- [x] `from src.process.claim_tools import RETRIEVAL_TOOLS, PausedReasoningState` imports cleanly
- [x] `evaluate()` returns `status="paused"` when LLM requests retrieval
- [x] `resume_evaluation()` completes the loop and returns a final response
- [ ] HTTP `/resume` route works end-to-end with no existing route regressions
- [ ] 8B BBH re-validation produces a statistically interpretable result (either
      confirms or disproves the missing-retrieval hypothesis with p-value)
- [x] Existing direct-claims mode (`use_tools=True`, no retrieval) is unchanged
