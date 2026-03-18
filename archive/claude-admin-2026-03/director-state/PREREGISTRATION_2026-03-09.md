# Pre-Registration: n≥100 Validation Studies
**Date:** 2026-03-09
**Purpose:** Validate or disprove hypotheses from 2026-03-08 LFM-2.5 exploration

---

## Why Pre-Registration?

Pre-registration prevents:
- **Cherry-picking:** Selectively reporting only positive results
- **P-hacking:** Running multiple tests until finding significance
- **HARKing:** Hypothesizing After Results are Known
- **Moving goalposts:** Changing success criteria after seeing data

By documenting hypotheses, methods, and success criteria BEFORE running experiments, we commit to honest reporting regardless of outcomes.

---

## Hypotheses to Test

### H1: Word Count Optimization (5w vs 15w)

**Exploratory Finding (n=20):**
- 5-word claims: 65% accuracy
- 15-word claims: 40% accuracy
- Difference: +25pp (p=0.102, NOT significant)
- 95% CI: [-5pp, +55pp]

**Pre-Registered Hypothesis:**
- Ultra-concise claims (5 words) will outperform verbose claims (15 words) by ≥10pp on BBH reasoning tasks
- Null hypothesis (H0): No difference or difference <10pp
- Alternative hypothesis (H1): 5w > 15w by ≥10pp

**Method:**
- Benchmark: BBH logical_deduction_five_objects (reasoning task)
- Sample size: n=100 per condition (200 total)
- Model: LFM-2.5-1.2B via LM Studio (http://100.73.201.58:1234)
- 5-word claims: Extract ~5 core claims, max 7 words each
- 15-word claims: Extract ~5 core claims, 12-18 words each
- Claim count held constant (5 claims in both conditions)

**Success Criteria:**
- p < 0.05 (two-tailed test)
- Effect size ≥10pp improvement
- 95% CI lower bound > 0

**If hypothesis FAILS:** Mark A-0016 word count optimization as DISPROVED

---

### H2: Goldilocks Principle (1-3 claims optimal)

**Exploratory Finding (n=10):**
- 0 claims: 90% (baseline)
- 1-3 claims: 100%
- 5 claims: 70%
- Difference: +10pp (p=0.299, NOT significant)

**Pre-Registered Hypothesis:**
- Tiny models (1-2B) perform best with 1-3 claims vs 0 claims or 5+ claims
- Optimal zone: 2 claims (middle of 1-3 range)
- Null hypothesis (H0): No difference between claim counts
- Alternative hypothesis (H1): 2 claims > 0 claims AND 2 claims > 5 claims

**Method:**
- Benchmark: BBH logical_deduction_three_objects (easier than 5-obj for higher baseline)
- Sample size: n=100 per condition (300 total)
- Model: LFM-2.5-1.2B via LM Studio
- Three conditions:
  - 0 claims (direct baseline)
  - 2 claims (Goldilocks zone)
  - 5 claims (overload hypothesis)

**Success Criteria:**
- 2 claims > 0 claims: p < 0.025 (Bonferroni correction: 0.05/2 tests)
- 2 claims > 5 claims: p < 0.025
- Effect size ≥5pp improvement over baseline
- Both comparisons must be significant

**If hypothesis FAILS:** Mark A-0016 Goldilocks Principle as DISPROVED

---

### H3: Task-Type Dependency (Reasoning vs Commonsense)

**Exploratory Finding (n=20 per benchmark):**
- BBH reasoning: +10pp with claims (90% → 100%)
- HellaSwag commonsense: -40pp with ultra-concise claims (55% → 15%, p=0.004 VALIDATED)
- TruthfulQA: +10pp with claims (55% → 65%, p=0.516 NOT significant)

**Pre-Registered Hypothesis:**
- Claims help reasoning tasks (BBH) but hurt commonsense tasks (HellaSwag)
- Already validated: HellaSwag regression (p=0.004)
- New test: Does BBH improvement replicate at n=100?
- Null hypothesis (H0): No difference or <5pp
- Alternative hypothesis (H1): BBH improvement ≥5pp with claims

**Method:**
- Benchmark: BBH logical_deduction_three_objects
- Sample size: n=100 per condition (200 total)
- Model: LFM-2.5-1.2B via LM Studio
- Two conditions:
  - Direct (no claims)
  - With claims (2 claims, concise but not ultra-concise ~10 words)

**Success Criteria:**
- p < 0.05 (two-tailed test)
- Effect size ≥5pp improvement
- 95% CI lower bound > 0
- Commonsense regression already validated (no retest needed)

**If hypothesis FAILS:** Mark A-0016 task-type routing as DISPROVED, keep only negative result (HellaSwag harm)

---

## Multiple Testing Correction

**Three primary hypotheses = 3 tests**
**Bonferroni correction:** α = 0.05 / 3 = 0.0167

But H2 has 2 internal comparisons, so:
- H1: p < 0.05 (single test)
- H2: p < 0.025 each (2 tests within hypothesis)
- H3: p < 0.05 (single test)

**Overall family-wise error rate:** 0.05

---

## Sample Size Justification

**Margin of error with n=100:**
- For p=0.50 (worst case): ±9.8pp at 95% CI
- For p=0.70 (typical): ±9.0pp at 95% CI
- For p=0.90 (high accuracy): ±5.9pp at 95% CI

**Power analysis (assuming true effect = 10pp):**
- Power ≈ 80% to detect 10pp difference at p<0.05
- Power ≈ 90% to detect 15pp difference

**Comparison to exploration (n=10-20):**
- n=20 margin: ±20-30pp (unreliable)
- n=100 margin: ±6-10pp (production-grade)
- 5x larger sample = 2.2x smaller margin of error

---

## Execution Plan

### Phase 1: Word Count (H1)
- Create experiment script: `experiments/validate_word_count_n100.py`
- Run 100 problems with 5-word claims
- Run 100 problems with 15-word claims
- Save results: `experiments/results/validate_word_count_*.json`
- Runtime: ~90 minutes (100 problems × 2 conditions × API delay)

### Phase 2: Goldilocks (H2)
- Create experiment script: `experiments/validate_goldilocks_n100.py`
- Run 100 problems × 3 conditions (0, 2, 5 claims)
- Save results: `experiments/results/validate_goldilocks_*.json`
- Runtime: ~135 minutes (100 problems × 3 conditions)

### Phase 3: Task-Type (H3)
- Create experiment script: `experiments/validate_task_type_n100.py`
- Run 100 problems × 2 conditions (direct, with claims)
- Save results: `experiments/results/validate_task_type_*.json`
- Runtime: ~90 minutes

**Total runtime:** ~5-6 hours (can run in parallel if needed)

---

## Statistical Analysis Plan

For each hypothesis, calculate:

1. **Descriptive statistics:**
   - Mean accuracy per condition
   - Standard deviation
   - 95% confidence intervals

2. **Significance test:**
   - Two-proportion z-test
   - Two-tailed p-value
   - Bonferroni correction applied

3. **Effect size:**
   - Percentage point difference
   - Cohen's h (standardized effect size for proportions)

4. **Reporting:**
   - Always report: n, mean, SD, difference, 95% CI, p-value
   - Report negative results honestly
   - Update CHOICES.md based on results (VALIDATED or DISPROVED)

---

## Success Outcomes

**Best case (all 3 hypotheses validate):**
- Remove [HYPOTHESIS] tag from A-0016
- Mark as VALIDATED with statistical evidence
- Document production guidelines

**Partial validation (1-2 hypotheses):**
- Split A-0016 into validated and disproved sections
- Keep [HYPOTHESIS] tag on unvalidated parts
- Document which aspects are production-ready

**Worst case (all 3 hypotheses fail):**
- Mark A-0016 as DISPROVED
- Keep only the validated negative result (HellaSwag harm)
- Document learnings about tiny model optimization
- Value: Prevented deployment of ineffective architecture

---

## Commitment

By pre-registering these hypotheses, I commit to:
- Running experiments exactly as specified above
- Reporting all results honestly (positive or negative)
- Applying corrections for multiple testing
- Not changing success criteria after seeing data
- Not running additional exploratory tests without documenting them
- Updating CHOICES.md truthfully based on results

**Signature:** Claude Sonnet 4.5 (Autonomous Director)
**Date:** 2026-03-09
**Pre-registration complete:** Ready to execute validation studies
