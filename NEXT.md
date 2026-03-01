# NEXT.md - Ideas to Follow Up

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
