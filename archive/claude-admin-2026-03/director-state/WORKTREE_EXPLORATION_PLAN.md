# 8 Worktree Parallel Exploration Plan

**Date:** 2026-03-08
**Context:** Goldilocks Principle validated on LFM-2.5. Now scaling exploration across 8 dimensions.

---

## Worktree 1: Multi-Model Validation (Llama-3.2-1B)

**Goal:** Validate Goldilocks Principle generalizes beyond LFM-2.5

**Experiments:**
- Baseline BBH on Llama-3.2-1B (1.3B params)
- Test 0, 1, 2, 3, 5 claims on same problems
- Compare Goldilocks curve to LFM-2.5

**Success criteria:** 1-3 claims optimal on different model
**Branch:** `worktree/multi-model-llama`

---

## Worktree 2: Claim Selection Strategies

**Goal:** Optimize which claims to retrieve from database

**Experiments:**
- Random selection (baseline)
- Semantic similarity (FAISS)
- Relevance scoring (custom)
- Confidence-weighted selection

**Success criteria:** +5pp over random selection
**Branch:** `worktree/claim-selection`

---

## Worktree 3: Optimal Word Count Tuning

**Goal:** Fine-tune the <15 word per claim limit

**Experiments:**
- Test limits: 5, 10, 15, 20, 25 words
- Measure accuracy vs cognitive load
- Find optimal brevity threshold

**Success criteria:** Identify exact word count sweet spot
**Branch:** `worktree/word-count-optimization`

---

## Worktree 4: Production Task Router

**Goal:** Build and validate the task-type routing system

**Implementation:**
- Pattern detection: keywords, structure, verbs
- LLM-based classification (lightweight)
- Rule-based heuristics
- Hybrid approach

**Success criteria:** >90% correct routing across benchmarks
**Branch:** `worktree/task-router-prod`

---

## Worktree 5: Learning Loop (Claim Promotion)

**Goal:** Dynamic claim ranking based on success/failure

**Implementation:**
- Track which claims used in successful responses
- Promote claims from correct answers
- Demote claims from incorrect answers
- Test if learning improves over time

**Success criteria:** +3-5pp improvement after 100 samples
**Branch:** `worktree/learning-loop`

---

## Worktree 6: Statistical Confidence (Large Samples)

**Goal:** Re-run key experiments with n=100 for confidence

**Experiments:**
- BBH Goldilocks with n=100 (vs n=10)
- MMLU format test with n=100 (vs n=20)
- GSM8K strategies with n=100 (vs n=10)

**Success criteria:** Confirm patterns hold with p<0.05
**Branch:** `worktree/large-sample-validation`

---

## Worktree 7: Hybrid Task Strategies

**Goal:** Explore boundary cases between reasoning/calculation

**Experiments:**
- Multi-step reasoning with calculation (ARC-Challenge)
- Word problems requiring both logic and math
- Test when to use claims vs format vs both

**Success criteria:** Decision tree for hybrid cases
**Branch:** `worktree/hybrid-strategies`

---

## Worktree 8: Production API Prototype

**Goal:** Build tiny-model optimization endpoint

**Implementation:**
- `POST /optimize` endpoint
- Input: query, task_type (optional)
- Process: task detection → claim retrieval → formatting
- Output: optimized prompt

**Success criteria:** Working API with all strategies integrated
**Branch:** `worktree/production-api`

---

## Execution Strategy

### Phase 1: Setup (5 min)
1. Create 8 git worktrees with clean branches
2. Copy base experiment infrastructure to each
3. Create exploration-specific scripts

### Phase 2: Parallel Execution (30-60 min)
- Launch all 8 explorations simultaneously
- Independent background processes
- Results written to separate directories

### Phase 3: Synthesis (15 min)
- Collect results from all 8 worktrees
- Identify breakthrough findings
- Update architectural guidance
- Merge successful branches

---

## Expected Outcomes

**High-value discoveries (likely):**
- Multi-model validation confirms or refutes generalization
- Task router enables production deployment
- Large samples provide statistical confidence

**Medium-value discoveries (possible):**
- Claim selection optimization adds +3-5pp
- Word count tuning identifies exact threshold
- Learning loop shows improvement over time

**Exploratory findings (uncertain):**
- Hybrid strategies reveal edge cases
- Production API exposes integration challenges

---

## Risk Mitigation

- Each worktree is isolated (no cross-contamination)
- Failures in one worktree don't block others
- Can abandon low-value explorations early
- Results aggregated incrementally

---

## Success Definition

**Minimum viable:** 3/8 worktrees yield actionable insights
**Target:** 5/8 worktrees produce improvements or validation
**Stretch:** 7/8 worktrees advance the architecture

**Time budget:** 2 hours total (setup + execution + synthesis)
