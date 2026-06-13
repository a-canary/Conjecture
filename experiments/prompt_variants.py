#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
16 Prompt Variant Definitions for Parallel Benchmark Experiment

Each variant rephrases the upstream prompt and context build differently.
"""

VARIANTS = {
    "v17_synthesized": {
        "name": "Synthesized Optimal (Top 3 Combined)",
        # Keep baseline format exactly, it works
        "math": "Q: {q}\nAnswer (number only):",
        "logic": "Q: {q}\nAnswer (Yes/No/Cannot determine):",
        "gsm8k": "Solve this math problem. Show your work and end with #### followed by the answer.\n\nProblem: {question}\n\nSolution:",
        "mmlu": "Question: {question}\n\n{choices}\n\nAnswer with just the letter (A, B, C, or D):",
    },
    "v01_baseline": {
        "name": "Baseline",
        "math": "Q: {q}\nAnswer (number only):",
        "logic": "Q: {q}\nAnswer (Yes/No/Cannot determine):",
        "gsm8k": "Solve this math problem. Show your work and end with #### followed by the answer.\n\nProblem: {question}\n\nSolution:",
        "mmlu": "Question: {question}\n\n{choices}\n\nAnswer with just the letter (A, B, C, or D):",
    },
    "v02_answer_only": {
        "name": "Answer Only",
        "math": "{q}\n\n=",
        "logic": "{q}",
        "gsm8k": "{question}\n\nFinal answer:",
        "mmlu": "{question}\n{choices}\n\nLetter:",
    },
    "v03_cot_explicit": {
        "name": "Explicit Chain-of-Thought",
        "math": "Q: {q}\n\nLet's work through this step by step:\n",
        "logic": "Q: {q}\n\nLet's reason through this step by step:\n",
        "gsm8k": "Problem: {question}\n\nLet's think step by step, then give the final answer after ####:",
        "mmlu": "Question: {question}\n{choices}\n\nLet me think through each option step by step, then give my answer:",
    },
    "v04_expert_role": {
        "name": "Expert Role",
        "math": "You are an expert mathematician. {q}\nAnswer:",
        "logic": "You are a logic expert. {q}\nAnswer:",
        "gsm8k": "You are a math professor. Solve this problem for a student.\n\n{question}\n\n#### Answer:",
        "mmlu": "You are a domain expert. {question}\n{choices}\n\nAs an expert, the answer is:",
    },
    "v05_decompose": {
        "name": "Decompose First",
        "math": "Problem: {q}\n\n1. What are the given values?\n2. What operation is needed?\n3. Calculate:\n\nAnswer:",
        "logic": "Logical statement: {q}\n\n1. Identify premises\n2. Apply rules\n3. Conclusion:",
        "gsm8k": "Problem: {question}\n\nBreak it down:\n- Key facts:\n- Steps needed:\n- Calculation:\n\n####",
        "mmlu": "{question}\n{choices}\n\nAnalysis:\n- Topic:\n- Key concept:\n- Answer:",
    },
    "v06_few_shot": {
        "name": "Few-Shot Examples",
        "math": "Example: A car goes 60 miles in 2 hours. Speed? Answer: 30\n\nNow: {q}\nAnswer:",
        "logic": "Example: All A are B. All B are C. Is every A a C? Answer: Yes\n\nNow: {q}\nAnswer:",
        "gsm8k": "Example: John has 5 apples and buys 3 more. How many? #### 8\n\nProblem: {question}\n####",
        "mmlu": "Example: What is 2+2? A.3 B.4 C.5 D.6 Answer: B\n\n{question}\n{choices}\nAnswer:",
    },
    "v07_verify": {
        "name": "Verification Step",
        "math": "{q}\n\nSolve, then verify your answer by checking the work:\nAnswer:",
        "logic": "{q}\n\nState your reasoning, verify it, then answer:",
        "gsm8k": "Problem: {question}\n\nSolve this, then verify your answer makes sense.\n\n####",
        "mmlu": "{question}\n{choices}\n\nChoose an answer and verify why the others are wrong:",
    },
    "v08_confidence": {
        "name": "Confidence Request",
        "math": "{q}\n\nProvide your answer and confidence (0-100):\nAnswer:",
        "logic": "{q}\n\nState answer with confidence level:",
        "gsm8k": "Problem: {question}\n\nProvide solution and rate your confidence (high/medium/low):\n####",
        "mmlu": "{question}\n{choices}\n\nAnswer (with confidence rating):",
    },
    "v09_concise": {
        "name": "Ultra Concise",
        "math": "{q} =",
        "logic": "{q}?",
        "gsm8k": "{question} ####",
        "mmlu": "{question}\n{choices}\n:",
    },
    "v10_structured": {
        "name": "Structured JSON",
        "math": '{q}\n\nRespond: {{"answer": <number>}}',
        "logic": '{q}\n\nRespond: {{"answer": "Yes/No/Cannot determine"}}',
        "gsm8k": 'Problem: {question}\n\nRespond with JSON: {{"steps": [...], "answer": <number>}}',
        "mmlu": '{question}\n{choices}\n\nRespond: {{"answer": "A/B/C/D"}}',
    },
    "v11_reframe": {
        "name": "Reframed Problem",
        "math": "Calculate the following and provide just the number: {q}",
        "logic": "Evaluate this logical statement as True, False, or Indeterminate: {q}",
        "gsm8k": "Word problem to solve (numeric answer required): {question}\n\nResult ####",
        "mmlu": "Knowledge question:\n{question}\n\nOptions:\n{choices}\n\nCorrect option:",
    },
    "v12_negative": {
        "name": "Negative Guidance",
        "math": "{q}\n\nDo not make arithmetic errors. Answer:",
        "logic": "{q}\n\nAvoid logical fallacies. Answer:",
        "gsm8k": "Problem: {question}\n\nAvoid calculation errors. Show work carefully.\n####",
        "mmlu": "{question}\n{choices}\n\nDo not pick wrong answers. Correct answer:",
    },
    "v13_positive": {
        "name": "Positive Affirmation",
        "math": "You're excellent at arithmetic. {q}\nAnswer:",
        "logic": "You have strong logical reasoning. {q}\nAnswer:",
        "gsm8k": "You're great at math word problems.\n\nProblem: {question}\n####",
        "mmlu": "You excel at knowledge questions.\n{question}\n{choices}\nAnswer:",
    },
    "v14_meta": {
        "name": "Meta-Cognitive",
        "math": "{q}\n\nFirst, what approach should you use? Then solve:\nAnswer:",
        "logic": "{q}\n\nWhat logical rule applies here? Apply it:\nAnswer:",
        "gsm8k": "Problem: {question}\n\nWhat strategy works best? Apply it:\n####",
        "mmlu": "{question}\n{choices}\n\nWhat knowledge domain? Apply it:\nAnswer:",
    },
    "v15_units": {
        "name": "Unit Emphasis",
        "math": "{q}\n\nInclude units in your reasoning. Final numeric answer:",
        "logic": "{q}\n\nIdentify the logical components. Conclusion:",
        "gsm8k": "Problem: {question}\n\nTrack units through each step.\n####",
        "mmlu": "{question}\n{choices}\n\nIdentify the subject domain, then answer:",
    },
    "v16_final_first": {
        "name": "Answer-First",
        "math": "{q}\n\nState the answer first, then explain: Answer =",
        "logic": "{q}\n\nAnswer first, then justify:",
        "gsm8k": "Problem: {question}\n\n#### [answer first, then show work]",
        "mmlu": "{question}\n{choices}\n\nAnswer: [letter first, then why]",
    },
}

def get_variant(variant_id: str) -> dict:
    """Get a specific variant by ID"""
    return VARIANTS.get(variant_id, VARIANTS["v01_baseline"])

def list_variants() -> list:
    """List all variant IDs"""
    return list(VARIANTS.keys())

if __name__ == "__main__":
    print(f"Defined {len(VARIANTS)} prompt variants:")
    for vid, v in VARIANTS.items():
        print(f"  {vid}: {v['name']}")
