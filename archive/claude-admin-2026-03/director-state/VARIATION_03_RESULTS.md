# Variation #3 Results: Single-Step Forcing

**Experiment Date:** 2026-03-08
**Status:** ⚠️ **MARGINAL** (p=0.072, just above significance threshold)

---

## Hypothesis

8B models regress due to **iteration overhead** (cascading errors across multiple prompts). Forcing single-step reasoning (max_iterations=1) should restore performance closer to direct baseline.

---

## Implementation

- Modified `bbh_three_prompt_benchmark.py` → `bbh_single_step_8b.py`
- Set `MAX_ITERATIONS = 1` (force single reasoning cycle)
- Set `CONFIDENCE_THRESHOLD = 0.9` (high bar to ensure quality)
- Model: `meta-llama/llama-3.1-8b-instruct`
- Test: BBH logical_deduction_three_objects (n=50)

---

## Results

| Method | Accuracy | Correct | Tokens | Avg Time |
|--------|----------|---------|--------|----------|
| **Direct baseline** | **90.0%** | 45/50 | 15,149 | 2.92s |
| **Single-step** | **58.0%** | 29/50 | 34,108 | 5.18s |
| **Prior unlimited** | 40.0% | 20/50 | 146,705 | 14.91s |
| **Prior 5-claim** | 48.0% | 24/50 | 148,264 | 15.43s |

**Delta vs unlimited:** +18.0pp (58% vs 40%)
**Delta vs 5-claim:** +10.0pp (58% vs 48%)

---

## Statistical Analysis

### Single-Step vs Unlimited Iterations
```
Two-proportion z-test:
- p1 = 0.58 (n=50), p2 = 0.40 (n=50)
- Z-score: 1.8004
- P-value: 0.0718
- Significant? MARGINAL (p < 0.10 but NOT p < 0.05)
```

### Single-Step vs 5-Claim Limit
```
Two-proportion z-test:
- p1 = 0.58 (n=50), p2 = 0.48 (n=50)
- Z-score: 1.0020
- P-value: 0.3140
- Significant? NO
```

### Single-Step vs Direct Baseline
```
Regression: -32pp (58% vs 90%)
P-value: 0.000089 (highly significant)
```

---

## Findings

### ⚠️ Hypothesis PARTIALLY SUPPORTED (Not Statistically Significant)

Iteration overhead **contributes** to the regression (+18pp improvement, p=0.072), but:
1. **Not statistically significant** at p<0.05 threshold (marginal at p<0.10)
2. **Architecture still regresses massively** vs direct (-32pp, p<0.001)
3. **Improvement insufficient** to restore production viability

### Key Insights

1. **Iteration overhead is A factor, not THE factor:**
   - Single-step: 58% (better than unlimited 40%)
   - But still far below direct: 90%
   - Suggests multiple compounding issues

2. **Progressive improvement pattern:**
   - Unlimited iterations: 40%
   - 5-claim limit: 48% (+8pp, p=0.42)
   - Single-step: 58% (+18pp, p=0.072)
   - Direct: 90% (target)

3. **Three-prompt architecture fundamentally incompatible:**
   - Even with optimal configuration (single-step, high threshold)
   - 8B models cannot reach direct baseline performance
   - -32pp gap persists across all variations

### Critical Methodological Issue: Direct Baseline Variance

**Direct baseline across 3 runs (all n=50):**
- Run 1 (Variation #2 baseline): 72%
- Run 2 (Variation #2 current): 78% (+6pp)
- Run 3 (Variation #3 current): 90% (+18pp from Run 1)

**Range: 18pp — HIGH VARIANCE**

**Implication:** n=50 is **insufficient** for stable baseline measurements. True effect sizes may be masked by sampling noise. Future validations should use n≥100 or compare within-run deltas only.

---

## Next Steps (Autonomous Decision)

### Proceed to Variation #1: Three-Model Ensemble

**Rationale:**
- Variations #2 and #3 both failed to achieve p<0.05 significance
- Ensemble (multiple 8B models voting) is the final Tier 1 attempt
- Tests if model diversity can compensate for architectural incompatibility

**If Ensemble Succeeds (+5pp, p<0.05):**
- Multi-model voting compensates for 8B limitations
- Practical solution for 8B deployment (higher latency, 3x API cost)

**If Ensemble Fails (p>0.05):**
- **Conclude:** Three-prompt architecture requires 70B+ models
- Update CHOICES.md with architectural constraint
- Proceed to Tier 2 variations (focus on 70B+ optimizations)

### Alternative: Address Baseline Variance

**Option:** Re-run all variations with n=100 for stable baselines
**Cost:** 2x runtime per benchmark (~70 min each)
**Benefit:** Eliminate 18pp baseline noise, clarify true effect sizes
**Decision:** Defer until Tier 1 complete (ensemble test first)

---

## Implications for Architecture

### ✅ Validated Findings
1. Iteration overhead contributes to regression (marginal evidence)
2. Context size not primary factor (Variation #2)
3. Multiple issues compound (no single fix restores performance)

### ❌ 8B Architectural Limitations
- Meta-reasoning across prompts
- Confidence calibration accuracy
- Multi-step logical consistency
- Claims integration and weighting

### 🎯 Production Recommendation (Preliminary)
**Until ensemble tested:**
- Use direct prompting for 8B models (72-90% accuracy)
- Reserve three-prompt for 70B+ models (100% BBH accuracy validated)
- Cost-accuracy tradeoff: 8B direct < 8B three-prompt ≪ 70B three-prompt

---

## Files

- Benchmark: `experiments/bbh_single_step_8b.py`
- Results: `experiments/results/bbh_single_step_8b_20260308_022125.json`
- Baseline: `experiments/results/bbh_context_limit_5claim_20260308_021130.json`
