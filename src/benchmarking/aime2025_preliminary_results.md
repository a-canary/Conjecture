# AIME2025 Benchmark - Preliminary Results

**Date**: December 11, 2025
**Model**: IBM Granite-4-H-Tiny via LM Studio
**Problems Tested**: First 5 of 30 AIME2025 problems

## Executive Summary

### Key Findings:
- **GraniteTiny-Direct**: 1/5 = 20% accuracy
- **GraniteTiny+Conjecture**: 0/3 so far (still running)
- **Performance**: Conjecture enhancement appears to be hurting rather than helping

### Problem-by-Problem Results

#### GraniteTiny-Direct
| Problem | Expected | Result | Time |
|---------|----------|--------|------|
| aime25_i_001 | 70 | ✅ CORRECT | 17.8s |
| aime25_i_002 | 588 | ❌ INCORRECT | 14.9s |
| aime25_i_003 | 16 | ❌ INCORRECT | 19.5s |
| aime25_i_004 | ??? | ❌ INCORRECT | 21.5s |
| aime25_i_005 | ??? | ❌ INCORRECT | 22.9s |

#### GraniteTiny+Conjecture (Enhanced Prompts)
| Problem | Expected | Result | Time |
|---------|----------|--------|------|
| aime25_i_001 | 70 | ❌ INCORRECT | 20.4s |
| aime25_i_002 | 588 | ❌ INCORRECT | 23.8s |
| aime25_i_003 | ??? | 🔄 IN PROGRESS | ~?s |

## Critical Issues Identified

### 1. Conjecture Enhancement Counterproductive
- **Problem**: The enhanced prompt engineering is causing regression
- **Evidence**: Problem aime25_i_001 was solved correctly by basic version but wrong by enhanced version
- **Impact**: -20% accuracy drop on tested problems

### 2. Model Performance Gap
- **Current**: ~20% accuracy on AIME2025
- **Target**: ≥70% accuracy (per TODO.md)
- **SOTA**: ~50% accuracy (top models like DeepSeek-R1)
- **Gap**: Need 50 percentage points improvement

### 3. Mathematical Reasoning Challenges
- AIME problems require advanced mathematical reasoning
- Current model struggles with multi-step mathematical proofs
- Answer extraction/evaluation may need improvement

## Immediate Next Steps

### High Priority
1. **Debug Conjecture Prompts**: Understand why enhanced prompts hurt performance
2. **Fix Answer Evaluation**: Improve regex patterns for mathematical answer extraction
3. **Model Fine-tuning**: Consider mathematical reasoning specialization

### Medium Priority
1. **Chain-of-Thought Enhancement**: Better step-by-step reasoning prompts
2. **Verification Mechanisms**: Add self-checking for mathematical computations
3. **Alternative Models**: Test with larger mathematical models

### Low Priority
1. **Full Benchmark Run**: Complete 30-problem evaluation once fixes are implemented
2. **Comparative Analysis**: Test against other benchmarks (GPQA, LiveCodeBench)
3. **Documentation**: Create detailed methodology guide

## Root Cause Analysis

### Why is Conjecture Hurting Performance?

The current enhanced prompt adds:
```
Solve this step-by-step with maximum accuracy:

1. Analyze the problem thoroughly
2. Consider multiple approaches
3. Select the best method
4. Provide a complete solution with clear reasoning
5. State the final answer clearly
```

**Potential Issues:**
- **Over-constrained**: Too many requirements may confuse the model
- **Cognitive Load**: Multi-step instructions may overwhelm smaller models
- **Focus Dilution**: Model focuses on process rather than accuracy
- ** verbosity**: Longer prompts may lead to more complex, error-prone responses

### Recommendations for Conjecture v2

1. **Simplify Enhancement**: Focus on accuracy rather than process
2. **Math-Specific Prompts**: Tailor to mathematical reasoning patterns
3. **Confidence Scoring**: Add internal confidence mechanisms
4. **Iterative Refinement**: Allow the model to check and revise answers

## Timeline for Improvement

- **Week 1**: Debug and fix Conjecture prompt issues
- **Week 2**: Implement mathematical reasoning improvements
- **Week 3**: Run full 30-problem benchmark
- **Week 4**: Analyze and document results

## Conclusion

While we successfully established the AIME2025 benchmark infrastructure, initial results show significant room for improvement. The current 20% accuracy is well below both our 70% target and SOTA performance. The surprising regression from Conjecture enhancement suggests our prompt engineering needs refinement for mathematical reasoning tasks.

This baseline provides a solid foundation for systematic improvement toward our SWEBench goals.