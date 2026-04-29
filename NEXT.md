# NEXT.md — Status: 2026-04-29

## ✅ Completed This Session
- **UX-0007 Phase 1-2**: Claim tree visualization (CLI + Web API)
  - Added `conjecture tree <id>` command with `--depth` and `--confidence` flags
  - Added `conjecture trace <id>` command showing chain to root
  - Added Web API endpoints: `/v1/claims/{id}/tree`, `/trace`, `/graph`
  - Created `src/utils/visualization.py` with tree/trace/graph builders
  - 22 new tests in `test_claim_visualization.py` — all pass
  - 963 total tests pass (+22 new)

## 🔧 Known Issues
- (none)

## 📋 Pending / Next Steps
- UX-0007 Phase 3: TUI Interactive Browser (optional enhancement)
- O-0008: 3 more benchmarks needed (DROP, MATH, HumanEval) for full validation

---

# Legacy: Ideas to Follow Up (2026-03-01)

Notable research findings and implementation ideas from R&D (2026-03-01).

## Validated Findings to Implement

### 1. Position Primacy (+10pp)
Claims at prompt START beat MIDDLE. Leverage attention primacy bias.
```python
# Current (bad)
prompt = f"Problem: {q}\nHints: {claims}\nSolve..."

# Better
prompt = f"KEY PATTERNS:\n{claims}\n\nProblem: {q}\nSolve..."
```

### 2. Optimal Confidence Threshold: 0.5 (not 0.8)
Too strict (0.9) rejects useful claims. 0.5-0.8 range optimal.

### 3. No Semantic Filtering Needed
Simple inclusion of all correct claims (86%) beats semantic filtering (84%).
Counter-intuitive but validated.

### 4. Direct Prompting for Standard Benchmarks
GSM8K: Direct 96% vs Conjecture 65%. Decomposition adds overhead without benefit on well-formed problems.

## Research Hypotheses to Test

### Category-Based Learning Effect
Category filtering showed best learning (+12pp) despite lower accuracy.
Worth investigating: Accuracy vs learning tradeoff.

### Window Size Optimization
Experiment completed but output truncated. Re-run with explicit sizes:
- Window=5, 10, 20, 50, unlimited

### Model-Specific Accumulation
Small models (8B) may be hurt by accumulation.
Test: At what model size does accumulation become beneficial?

## Literature to Explore

1. **Lost in the Middle** (Liu et al. 2023) - Primacy/recency bias
2. **The Few-shot Dilemma** (2025) - Over-prompting effects
3. **Context Rot** (Chroma 2024) - Context degradation
4. **Cluster-based Adaptive Retrieval** (arXiv 2511.14769)

## Infrastructure Ideas

### 1. Adaptive Claim Selection
Select strategy based on:
- Problem complexity (simple → direct, complex → conjecture)
- Model size (small → no accumulation)
- Problem category (match to claim categories)

### 2. Claim Quality Scoring
Track claim usage → correctness correlation.
Prune claims that don't help.

### 3. Production Benchmarking
Use lm-evaluation-harness for official benchmarks.
Wrapper created: `src/evaluation/conjecture_lm.py`

## Key Learnings (from archived .agent)

### What Worked
1. **Removing unused code** cleaner than creating stubs (LanceDB removal eliminated 43 skipped tests)
2. **Subprocess isolation** provides adequate sandboxing without Docker overhead
3. **Syntax error resolution** unblocks static analysis systematically

### What Didn't Work
1. **SWE-Bench "100% accuracy"** was on SYNTHETIC tasks, not real SWE-Bench-lite
2. **Baseline 0%** suggests LLM never called or silent exception (bug in test, not model)
3. **"SWE-bench-bash-only"** is misleading - it's synthetic bash test, NOT official SWE-bench methodology

### Test Validity Lessons
- Real SWE-Bench IDs are `repo__repo-number` format (django__django-11133)
- Synthetic fallback tasks are generic bash exercises, NOT evidence of real capability
- Always verify test is measuring what you think it's measuring

## Testing Framework Ideas (from archived .trae)

### No-Mock Testing Approach
- Use real components only, no mocking of dependencies
- Local/test configurations to avoid external dependencies
- In-memory databases for speed
- Target < 2 seconds per test
- 20 unit tests + 3 E2E tests covers core functionality
