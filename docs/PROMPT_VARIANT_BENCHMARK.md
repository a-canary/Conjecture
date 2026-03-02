# Prompt Variant Benchmark Results

## Experiment Setup
- **Model**: llama3.1-8b (via Cerebras)
- **Test Size**: 40 math + 20 logic = 60 problems per variant
- **Variants Tested**: 16 different prompt strategies

## Final Rankings

| Rank | Variant | Combined | Math | Logic | Key Insight |
|------|---------|----------|------|-------|-------------|
| 🥇 | v01_baseline | **82.5%** | 85.0% | 80.0% | Simple direct prompts WIN |
| 🥈 | v16_final_first | 81.25% | 82.5% | 80.0% | Answer-first nearly matches |
| 🥉 | v10_structured | 67.5% | 85.0% | 50.0% | JSON helps math, hurts logic |
| 4 | v08_confidence | 43.75% | 37.5% | 50.0% | Confidence request helps some |
| 5 | v11_reframe | 41.25% | 82.5% | 0.0% | Good math, kills logic |
| 6 | v06_few_shot | 35.0% | 10.0% | 60.0% | Examples hurt math |
| 7 | v02_answer_only | 33.75% | 47.5% | 20.0% | Too minimal |
| 8 | v09_concise | 23.75% | 17.5% | 30.0% | Ultra-short bad |
| 9 | v12_negative | 21.25% | 22.5% | 20.0% | "Don't make errors" backfires |
| 10 | v14_meta | 16.25% | 32.5% | 0.0% | Meta-cognition fails |
| 11 | v13_positive | 15.0% | 30.0% | 0.0% | Affirmation doesn't help |
| 12 | v04_expert_role | 13.75% | 22.5% | 5.0% | Role-play hurts |
| 13 | v07_verify | 10.0% | 20.0% | 0.0% | Verification step bad |
| 14 | v05_decompose | 8.75% | 17.5% | 0.0% | Decomposition kills it |
| 15 | v15_units | 5.0% | 10.0% | 0.0% | Unit tracking fails |
| 16 | v03_cot_explicit | **2.5%** | 5.0% | 0.0% | **CoT is WORST** |

## Key Findings

### What Works (Top 3)
1. **Simple Direct Prompts** - Baseline wins at 82.5%
   - `Q: {q}\nAnswer (number only):`
   
2. **Answer-First Pattern** - 81.25%
   - `{q}\n\nState the answer first, then explain: Answer =`
   
3. **Structured Output (Math Only)** - 85% math accuracy
   - `{q}\n\nRespond: {"answer": <number>}`

### What Fails (Bottom 5)
1. **Explicit CoT** - 2.5% (33x worse than baseline!)
2. **Unit Emphasis** - 5%
3. **Decomposition** - 8.75%
4. **Verification** - 10%
5. **Role-Playing** - 13.75%

## Synthesis Conclusion

**The baseline IS the optimal prompt for llama3.1-8b.**

Attempts to "improve" it with:
- Chain-of-thought → Catastrophic failure
- Decomposition → Massive accuracy loss
- Role-playing → Significant degradation
- Verification steps → No benefit, harms performance

## Optimal Prompt Templates

### Math
```
Q: {question}
Answer (number only):
```

### Logic
```
Q: {statement}
Answer (Yes/No/Cannot determine):
```

### GSM8K
```
Solve this math problem. Show your work and end with #### followed by the answer.

Problem: {question}

Solution:
```

## Implications

1. **Model-specific optimization matters** - What works for GPT-4 (CoT) fails for smaller models
2. **Simplicity beats complexity** - Direct prompts outperform elaborate scaffolding
3. **Don't over-engineer** - Each added instruction is a potential failure point
4. **Test, don't assume** - Common wisdom (CoT always helps) can be wrong
