# Multi-Model Three-Prompt Architecture Validation (Draft)

## Update for CLAUDE.md

### Section to Add/Update: Three-Prompt Architecture Validation

**Status:** VALIDATED across model sizes [SMALL/MEDIUM/LARGE/XL]

The three-prompt architecture has been validated across multiple model sizes to assess generalization:

| Model Size | Model | BBH (Hard Reasoning) | GSM8K (Saturated) | Notes |
|------------|-------|---------------------|-------------------|-------|
| **Small (8B)** | Llama-3.1-8B | [TBD] | [TBD] | Baseline capability |
| **Medium (32B)** | DeepSeek-R1-Qwen-32B | [TBD] | [TBD] | Reasoning-optimized |
| **Large (70B)** | Llama-3.1-70B | [TBD] | [TBD] | Standard large model |
| **XL (670B)** | DeepSeek-V3 | +10.0pp (p=0.018) ✅ | -2.0pp (p=0.695) ≈ | Validated 2026-03-07 |

### Key Findings (Template)

**BBH (Hard Reasoning):**
- [X]/4 models show significant improvement (p<0.05)
- Average improvement: [TBD]pp
- Pattern: [Smaller models benefit more / Consistent across sizes / Larger models benefit more]

**GSM8K (Saturated Math):**
- [X]/4 models show significant changes
- Pattern: [Statistically equivalent across all sizes / Some regressions / Mixed results]
- Interpretation: [High baseline leaves no room / Architecture neutral on saturated tasks]

**Model Size Dependency:**
- [✅ / ❌] Architecture effectiveness varies with model size
- Small models (<14B): [Benefit significantly / Show regression / Neutral]
- Large models (>70B): [Benefit significantly / Show regression / Neutral]

**Production Implications:**
- [✅ / ❌] Safe to deploy across all model sizes
- Recommended routing: [Always use three-prompt / Size-aware routing needed / Baseline-aware only]
- Critical limitations: [Multi-provider testing still needed / Benchmark coverage gaps]

### Updated CLAUDE.md Text (to replace existing section)

## Three-Prompt Architecture Validation (2026-03-07)

**Status:** VALIDATED for hard reasoning across [X] model sizes (8B to 670B)

The three-prompt architecture splits single prompts into 3 focused stages: update confidence → create claim or SKIP → final response. Validated with statistical analysis across model sizes.

| Model Size | Model | BBH | GSM8K | Pattern |
|------------|-------|-----|-------|---------|
| Small (8B) | Llama-3.1-8B | [Δ +X.Xpp, p=X.XXX] | [Δ +X.Xpp, p=X.XXX] | [TBD] |
| Medium (32B) | DeepSeek-R1-Qwen-32B | [Δ +X.Xpp, p=X.XXX] | [Δ +X.Xpp, p=X.XXX] | [TBD] |
| Large (70B) | Llama-3.1-70B | [Δ +X.Xpp, p=X.XXX] | [Δ +X.Xpp, p=X.XXX] | [TBD] |
| XL (670B) | DeepSeek-V3 | Δ +10.0pp (p=0.018) ✅ | Δ -2.0pp (p=0.695) ≈ | SIG improvement / Equivalent |

### Multi-Model Findings

**[Key Pattern 1]:**
- [Description of pattern across models]
- [Implications for production use]

**[Key Pattern 2]:**
- [Description]
- [Implications]

**Production Recommendations:**
- ✅ Use task-type routing (see `experiments/task_type_router.py`)
- ✅ Route to three-prompt when baseline <90% and hard reasoning expected
- ✅ Route to direct when baseline ≥90% or simple recall/calculation
- [✅ / ⚠️] [New recommendation based on multi-model results]
- ⚠️ Single-provider validation (OpenRouter only) - multi-provider testing recommended

**Remaining Limitations:**
- Tested on OpenRouter models only (Claude, GPT-4, Gemini, local models not tested)
- [Other limitations discovered during multi-model validation]

**See `experiments/THREE_PROMPT_ARCHITECTURE.md` and `.director/THREE_PROMPT_VALIDATION_COMPLETE.md` for full details**

---

## Analysis Plan

Once all 6 benchmarks complete:

1. **Run analysis script:**
   ```bash
   .venv/bin/python experiments/analyze_multimodel_validation.py
   ```

2. **Extract key findings:**
   - Model size dependency (small vs large)
   - BBH pattern (consistent improvement?)
   - GSM8K pattern (consistent equivalence?)
   - Any unexpected regressions

3. **Update documentation:**
   - Fill in table with actual results
   - Write key findings sections
   - Update production recommendations
   - Add any new limitations discovered

4. **Update related files:**
   - THREE_PROMPT_ARCHITECTURE.md - add multi-model section
   - THREE_PROMPT_VALIDATION_COMPLETE.md - expand findings
   - CLAUDE.md - replace current section with multi-model version

5. **Commit with detailed message:**
   - List all models tested
   - Summarize key findings
   - Note implications for production use
