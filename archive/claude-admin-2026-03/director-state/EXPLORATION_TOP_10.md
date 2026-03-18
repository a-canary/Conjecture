# Top 10 Experimental Variations for Validation

**Selection Date:** 2026-03-08T02:00:00Z
**Context:** A-0015 validated that 8B models have architectural limitations (meta-reasoning, confidence calibration, multi-prompt context), not just missing retrieval. 70B+ models succeed with three-prompt architecture.

**Selection Criteria:**
- Directly addresses validated 8B failure modes
- Amplifies validated 70B+ success patterns
- High expected information value per implementation effort
- Testable with n=50 BBH benchmark (statistical rigor)

---

## Validation Status

| Variation | Status | Result | P-value | Outcome |
|-----------|--------|--------|---------|---------|
| #2: 5-claim context limit | ✅ Complete | 48% (+8pp vs 40%) | p=0.42 | ❌ FAILED (not significant) |
| #3: Single-step confidence | ✅ Complete | 58% (+18pp vs 40%) | p=0.072 | ⚠️ MARGINAL (not p<0.05) |
| #1: Three-model ensemble | 🚫 Blocked | - | - | Mistral-7B unavailable (404) |

**Key Findings:**
- Context size NOT the bottleneck (Variation #2)
- Iteration overhead contributes but insufficient (Variation #3, p=0.072)
- 8B architectural incompatibility validated across all attempts
- Direct prompting recommended for <32B models

**Architectural Constraint:** Three-prompt requires 70B+ models (validated in O-0008)

**Last Updated:** 2026-03-08T03:00:00Z

---

## Tier 1: High Impact, Easy Implementation (Run First)

### 1. Three-Model Ensemble Vote (8B)
**Category:** Model Size Optimization (Variation 2.1)
**Hypothesis:** Multiple 8B models voting covers individual failure modes
**Implementation:** Run 3 diverse 8B models (Llama-3.1-8B, Qwen-14B downscaled, Mistral-8B) on same problem, majority vote wins
**Expected Impact:** +15-20pp if ensemble compensates for architectural gaps
**Test:** BBH logical_deduction n=50, compare to 8B baseline (72%)
**Effort:** 2-3 hours (reuse existing benchmark infrastructure)

### 2. Hard 5-Claim Context Limit (8B)
**Category:** Context Management (Variation 5.3)
**Hypothesis:** 8B models fail due to context overload (50 claims → 5 claims)
**Implementation:** Modify context_builder.py max_claims=5 for 8B models
**Expected Impact:** +10-15pp if context reduction improves focus
**Test:** BBH n=50 with 5-claim vs 50-claim context
**Effort:** 1 hour (configuration change + benchmark run)

### 3. Single-Step Confidence Threshold (8B)
**Category:** Prompt Architecture (Variation 1.5)
**Hypothesis:** 8B fails at multi-prompt iteration, single high-confidence step better
**Implementation:** confidence_threshold=0.9, max_iterations=1 (force direct answer)
**Expected Impact:** +8-12pp if iteration overhead causes cascading errors
**Test:** BBH n=50, single-step vs three-prompt baseline
**Effort:** 30 minutes (parameter change + benchmark)

---

## Tier 2: Moderate Impact, Addresses 70B+ Optimization

### 4. Adaptive Iteration Depth by Confidence
**Category:** Prompt Architecture (Variation 1.4)
**Hypothesis:** 70B+ wastes tokens on easy problems, needs dynamic depth
**Implementation:** If confidence > 0.85 after iteration 1, SKIP to final response
**Expected Impact:** -30% token usage on mixed-difficulty tasks, maintain accuracy
**Test:** GSM8K n=50 (high baseline), measure tokens and accuracy
**Effort:** 2 hours (modify evaluate() loop + benchmark)

### 5. Cascade 8B → 70B on Low Confidence
**Category:** Model Size Optimization (Variation 2.4)
**Hypothesis:** Route hard problems to large models, save cost on easy ones
**Implementation:** Run 8B first, if confidence < 0.6 after 2 iterations, escalate to 70B
**Expected Impact:** -50% cost vs pure 70B, maintain 70B accuracy
**Test:** BBH n=100 mixed difficulty, measure cost and accuracy
**Effort:** 3-4 hours (routing logic + dual-model benchmark)

### 6. Temperature-Based Confidence Calibration
**Category:** Confidence Calibration (Variation 4.3)
**Hypothesis:** Lower temperature (0.3) improves confidence accuracy vs default (0.7)
**Implementation:** Run same benchmark at temp=[0.3, 0.5, 0.7, 1.0], measure confidence vs correctness correlation
**Expected Impact:** Better confidence scores enable better routing/escalation
**Test:** BBH n=50 at 4 temps, calculate Brier score for confidence calibration
**Effort:** 2 hours (parameter sweep + analysis)

---

## Tier 3: Exploratory, Longer-Term Validation

### 7. Parallel Claim Evaluation (Multi-Branch)
**Category:** Claim Decomposition (Variation 6.4)
**Hypothesis:** Explore multiple decomposition paths simultaneously, merge best claims
**Implementation:** Generate 3 parallel claim sets at iteration 1, continue with highest-confidence set
**Expected Impact:** +5-8pp on ambiguous problems, no benefit on clear problems
**Test:** BBH subset with known ambiguous problems (boolean_expressions, causal_judgment)
**Effort:** 4-5 hours (parallel execution logic + selective benchmark)

### 8. Self-Consistency Voting (Single Model)
**Category:** Verification (Variation 8.1)
**Hypothesis:** Generate 5 solutions with temp=0.8, majority vote improves reliability
**Implementation:** Run same problem 5 times, count answer frequency, return mode
**Expected Impact:** +3-5pp on problems with near-threshold difficulty
**Test:** BBH n=50, compare to deterministic baseline
**Effort:** 2 hours (rerun logic + vote aggregation)

### 9. Eager Retrieval (All Tools Upfront)
**Category:** Knowledge Retrieval (Variation 3.1)
**Hypothesis:** Pre-fetch all potential knowledge before reasoning loop starts
**Implementation:** Call retrieve_knowledge with broad query at iteration 0, load results as claims
**Expected Impact:** -1 iteration average, +2-4pp if initial context is richer
**Test:** BBH n=50 with mock retrieval strategy
**Effort:** 2-3 hours (modify endpoint to fetch pre-reasoning)

### 10. Bayesian Confidence Updates
**Category:** Confidence Calibration (Variation 4.1)
**Hypothesis:** Update confidence using Bayes' rule based on claim agreement/conflict
**Implementation:** Track P(claim|evidence), update with new claims using Bayesian posterior
**Expected Impact:** More accurate confidence scores → better stopping criteria
**Test:** Compare confidence calibration (Brier score) on BBH n=50
**Effort:** 3-4 hours (Bayesian update logic + analysis)

---

## Implementation Priority Order

**Week 1 (High ROI):**
1. Variation #2 (5-claim limit) - 1 hour
2. Variation #3 (single-step) - 30 min
3. Variation #1 (ensemble) - 3 hours

**Week 2 (Optimization):**
4. Variation #6 (temperature calibration) - 2 hours
5. Variation #4 (adaptive depth) - 2 hours
6. Variation #5 (cascade routing) - 4 hours

**Week 3 (Exploratory):**
7. Variation #8 (self-consistency) - 2 hours
8. Variation #9 (eager retrieval) - 3 hours
9. Variation #7 (parallel branches) - 5 hours
10. Variation #10 (Bayesian confidence) - 4 hours

---

## Success Criteria

- **Variation succeeds:** +5pp improvement (p<0.05) OR -30% cost with <3pp accuracy loss
- **Variation fails:** No significant difference (p>0.05) OR regression >3pp
- **Update CHOICES.md:** Add new choice if breakthrough (>10pp improvement or >50% cost reduction)
- **Document all results:** experiments/results/ with statistical analysis

---

## Next Action

Autonomously implement **Variation #2 (5-claim context limit)** as the quickest validation test (30 min implementation + 40 min benchmark runtime).
