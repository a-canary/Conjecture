# SWE-Bench-Bash-Only + GraniteTiny Optimization Analysis

**Date**: December 30, 2025  
**Status**: Analysis Complete  
**Target**: >70% accuracy on SWE-Bench-Bash-Only with GraniteTiny (1B params, 4K context)

---

## 🎯 Executive Summary

The combination of **Technical Constraints** (1B params, 4K context, bash-only subset) + **Performance Criteria** (token efficiency, latency <2s, accuracy >70%) + **Scalability & Maintainability** (reusable patterns, reproducible results) requires a fundamentally different approach than naive LLM evaluation.

**Key Insight**: GraniteTiny cannot compete on raw reasoning capacity. Success requires **aggressive optimization** across three dimensions:
1. **Context Compression** - Reduce problem statements to <500 tokens
2. **Structured Output** - Use JSON to eliminate parsing errors
3. **Early Exit** - Run tests immediately, retry only on failure

---

## 🔍 Problem Analysis

### Constraints
- **Model Size**: 1B active parameters (vs 7B-70B for typical models)
- **Context Window**: 4K tokens (vs 8K-128K for larger models)
- **Domain**: Bash-only subset (deterministic, testable, clear success/failure)
- **Latency Target**: <2s per task (requires efficient processing)

### Why Bash-Only is Ideal
✅ **Deterministic test execution** - Tests pass or fail, no ambiguity  
✅ **Clear success signals** - Test output is unambiguous  
✅ **Minimal context needed** - Bash scripts are typically short  
✅ **Reusable patterns** - Common bash patterns repeat across tasks  
✅ **Efficient evaluation** - Can run tests in sandbox with timeout  

### Why GraniteTiny Struggles
❌ **Limited reasoning capacity** - 1B params insufficient for complex logic  
❌ **Small context window** - 4K tokens limits problem understanding  
❌ **No specialized training** - Not optimized for code generation  
❌ **Parsing errors** - Unstructured output leads to failures  

---

## 💡 Solution Strategy

### 8-Step Optimization Approach

#### 1. **Context Compression** (<500 tokens)
Extract only essential information:
- Repo name
- File path
- Error message
- Test command
- Relevant code snippet (50 lines max)
- Test output (20 lines max)

#### 2. **Structured Output** (JSON Frontmatter)
Enforce JSON format to eliminate parsing errors

#### 3. **Limited Context** (Problem + Code + Output)
Load only what's necessary

#### 4. **Early Exit** (Test-Driven)
Run tests immediately, retry only on failure

#### 5. **Prompt Caching** (Batch Processing)
Group similar tasks for efficiency

#### 6. **Bash-Specific Optimization** (Domain Knowledge)
Include bash-specific examples and patterns

#### 7. **Confidence Calibration** (Tiny Model Adjustment)
Adjust confidence scores based on task complexity

#### 8. **Conjecture Integration** (Intermediate Reasoning)
Use Conjecture's claim system for guidance

---

## 📊 Expected Outcomes

### Performance Targets
| Metric | Target | Rationale |
|--------|--------|-----------|
| **Accuracy** | >70% | Primary success criterion |
| **Latency** | <2s/task | Real-time feedback requirement |
| **Token Efficiency** | <1000 tokens/task | Context window constraint |
| **Reproducibility** | 100% deterministic | Validation requirement |

---

## 🚀 Implementation Roadmap

### Phase 1: Baseline (Week 1)
- [ ] Verify GraniteTiny configuration
- [ ] Run baseline evaluation (5-10 tasks)
- [ ] Document current metrics
- [ ] Establish performance baseline

### Phase 2: Optimization (Week 2-3)
- [ ] Implement context compression
- [ ] Add structured output format
- [ ] Run comprehensive comparison
- [ ] Analyze results

### Phase 3: Enhancement (Week 4)
- [ ] Implement early exit strategy
- [ ] Add prompt caching
- [ ] Optimize bash-specific prompts
- [ ] Achieve 65% accuracy

### Phase 4: Target Achievement (Month 1)
- [ ] Achieve >70% accuracy
- [ ] Maintain <2s latency
- [ ] Document techniques
- [ ] Create reusable patterns

---

## 📁 Key Files

### Production-Ready Components
- **SWE-Bench Evaluator**: `benchmarks/benchmarking/swe_bench_evaluator.py` (895 lines)
- **GraniteTiny Integration**: `docs/ibm_granite_tiny_integration_guide.md` (385 lines)
- **Quick Reference**: `.agent/plan/swebench_quick_reference.md` (415 lines)

### Success Criteria
- **SC-FEAT-001**: SWE-Bench-Bash-Only accuracy target (>70%)
- **Location**: `.agent/backlog.md` (Line 273-281)

---

## 📝 Next Steps

1. Verify GraniteTiny configuration
2. Run baseline evaluation (5-10 tasks)
3. Document current metrics
4. Implement context compression
5. Add structured output format
6. Run comprehensive comparison
7. Analyze results and iterate
8. Achieve >70% accuracy target

---

**Analysis Generated**: December 30, 2025  
**Status**: Ready for Implementation  
**Target Achievement**: Month 1 (4 weeks)  
**Success Criteria**: >70% accuracy on SWE-Bench-Bash-Only with <2s latency
