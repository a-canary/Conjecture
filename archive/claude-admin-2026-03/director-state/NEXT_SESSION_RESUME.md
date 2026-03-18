# Resume for Next Session

**Last Session:** 2026-03-08 LFM-2.5 Exploration
**Status:** Research complete, awaiting validation decision

---

## Critical Context

**READ FIRST:**
1. `.director/STATISTICAL_REALITY_CHECK.md` - Why "breakthrough findings" aren't validated
2. `.director/FINAL_SESSION_REPORT.md` - Comprehensive session summary
3. `CHOICES.md` A-0016 - Now marked [HYPOTHESIS] with caveats

**Key Insight:** Statistical analysis revealed only 1/7 findings significant (p<0.05), and it was NEGATIVE. All claimed improvements (5-word optimization, Goldilocks Principle) are not statistically validated with n=10-20 samples.

---

## What Was Accomplished

**Phase 1 (12 experiments):** Explored tiny model optimization, generated hypotheses
**Phase 2 (4 worktrees):** Tested claim count, word count, selection, routing
**Phase 3 (3 multi-benchmark):** Tested generalization (failed - 1/3 success)

**Documentation:** 9 comprehensive reports (1900+ lines)
**Architecture:** Updated CLAUDE.md with learnings, CHOICES.md with caveats

---

## Statistical Reality

| Finding | Claimed | P-value | Significant? | Status |
|---------|---------|---------|--------------|--------|
| 5w vs 15w claims | +25pp | 0.102 | ❌ NO | Hypothesis only |
| Goldilocks (1-3 claims) | +10pp | 0.299 | ❌ NO | Hypothesis only |
| Claim selection | +5pp | 0.751 | ❌ NO | Hypothesis only |
| HellaSwag regression | -40pp | 0.004 | ✅ YES | **VALIDATED** |

**Only validated finding:** Ultra-concise claims HARM commonsense reasoning

---

## Uncommitted Work

**Modified:**
- `CLAUDE.md` - Added 5 learnings (statistical requirements, LM Studio workarounds, etc.)
- `CHOICES.md` - A-0016 now marked [HYPOTHESIS] with validation requirements
- `.director/SESSION_FINDINGS.md` - Updated with all results
- `.director/state.json` - Session state

**New files (13 reports in .director/):**
- Statistical reality check documentation
- Comprehensive session summaries
- Production guides (NOT validated)
- Pattern analysis across experiments

---

## Decision Point for Next Session

**Option 1: Commit as Hypotheses**
- Commit all documentation with [HYPOTHESIS] tags
- Mark findings as "requiring n≥100 validation"
- Value: Preserve exploration work for future validation

**Option 2: Run n≥100 Validation First**
- Re-run key experiments with proper sample sizes
- Only commit if findings replicate with p<0.05
- Timeline: 2-4 weeks
- Value: Only commit validated findings

**Option 3: Document Negative Result Only**
- Commit only the validated finding (commonsense harm)
- Discard unvalidated hypotheses
- Value: Scientific honesty, avoid misleading claims

**Recommendation:** Option 1 (commit as hypotheses) - preserves valuable exploration work while being honest about validation requirements

---

## If Resuming Validation Work

**High Priority Validation:**
1. Word count effect (n=100): Test if 5w > 15w replicates
2. Goldilocks claim count (n=100): Test if 1-3 claims optimal
3. Multi-benchmark (n=100): Which task types benefit

**Required:**
- Pre-register hypotheses (avoid cherry-picking)
- Use holdout validation sets
- Apply Bonferroni correction for multiple testing
- Report effect sizes and confidence intervals

**Timeline:** 2-3 days of experiment runtime for n=100 validation

---

## Key Files to Review

1. `.director/STATISTICAL_REALITY_CHECK.md` - Full analysis
2. `.director/FINAL_SESSION_REPORT.md` - Complete summary
3. `CLAUDE.md` - Updated with learnings (lines 186-195, 350-358, 424-495)
4. `CHOICES.md` - A-0016 with [HYPOTHESIS] tag (line 357)
5. `experiments/statistical_significance_check.py` - Test script

---

## Warnings

- **Don't make production claims** from this work without n≥100 validation
- **All positive findings are tentative** - could be noise with n=10-20
- **Only negative finding is validated** - avoid ultra-concise claims on commonsense tasks
- **High variance in BBH** - different 10-problem samples show 20-90% baseline
- **LM Studio requires curl** - Python HTTP libraries fail (see CLAUDE.md)

---

**Next Action:** Review documentation and decide commit strategy (Option 1, 2, or 3 above)
