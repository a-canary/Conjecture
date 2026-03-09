# Variation #2 Results: 5-Claim Context Limit

**Experiment Date:** 2026-03-08
**Status:** ❌ **FAILED** (Hypothesis DISPROVED)

---

## Hypothesis

8B models regress (-32pp: 72% → 40%) due to **context overload** during iterative reasoning. Limiting claim context to 5 (vs unlimited) should improve performance by reducing cognitive load.

---

## Implementation

- Modified `bbh_three_prompt_benchmark.py` → `bbh_context_limit_5claims.py`
- Set `MAX_CLAIMS = 5` (enforced after each claim creation)
- Pruning logic: Keep only top 5 claims by confidence after each iteration
- Model: `meta-llama/llama-3.1-8b-instruct`
- Test: BBH logical_deduction_three_objects (n=50)

---

## Results

| Method | Accuracy | Correct | Tokens | Avg Time |
|--------|----------|---------|--------|----------|
| **Direct baseline** | **78.0%** | 39/50 | 14,646 | 3.54s |
| **5-claim limit** | **48.0%** | 24/50 | 148,264 | 15.43s |
| **Prior unlimited** | 40.0% | 20/50 | 146,705 | 14.91s |

**Delta vs unlimited:** +8.0pp (48% vs 40%)

---

## Statistical Analysis

```
Two-proportion z-test (5-claim vs unlimited):
- p1 = 0.48 (n=50), p2 = 0.40 (n=50)
- Z-score: 0.8058
- P-value: 0.4203
- Significant? NO (p > 0.05)
```

**Conclusion:** The +8pp improvement is **NOT statistically significant** (p=0.42). Within random variation for n=50.

**Regression vs direct:** -30pp (48% vs 78%), p=0.0011 (highly significant)

---

## Findings

### ❌ Hypothesis DISPROVED

Context size (5 vs unlimited claims) does **NOT** significantly impact 8B performance. The regression persists regardless of claim count.

### Key Insights

1. **Context overload is NOT the root cause** of 8B regression
2. **Iteration overhead likely the issue**: 3-prompt architecture still causes massive regression even with minimal context
3. **Random variation observed**: Direct baseline 78% (current) vs 72% (prior) = +6pp noise with n=50
4. **Architecture fundamentally incompatible with 8B**: -30pp regression remains (p=0.0011)

### Implications for Next Variations

- ✅ **Test Variation #3 next (single-step)**: If iteration overhead is the issue, forcing max_iterations=1 should help
- ⏭️ **Skip further context variations**: #9 (eager retrieval) and similar context strategies unlikely to help
- 🎯 **Focus on iteration/architecture**: The problem is HOW the model reasons, not WHAT it reasons over

---

## Next Steps (Autonomous)

Proceeding to **Variation #3: Single-Step Confidence** to test if iteration overhead (not context size) causes the regression.

**Hypothesis #3:** Force single-step reasoning (max_iterations=1, confidence_threshold=0.9) to eliminate iteration overhead.
**Expected:** If iteration is the problem, single-step should restore performance closer to direct baseline.

---

## Files

- Benchmark: `experiments/bbh_context_limit_5claims.py`
- Results: `experiments/results/bbh_context_limit_5claim_20260308_021130.json`
- Baseline: `experiments/results/bbh_delegated_8b_20260308_012332.json`
