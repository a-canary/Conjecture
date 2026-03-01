#!/usr/bin/env python3
"""
Fixed Benchmark with Proper Answer Extraction

Issues with previous benchmarks:
1. Using 0-shot instead of 5-shot
2. No proper MMLU prompt format  
3. Regex extraction from long text vs constrained choice

Fixes:
1. Use 5-shot prompting with examples
2. Use standard MMLU prompt format
3. Extract only first character/choice from response
4. Constrain max_tokens to 1-5

Expected: Llama 3.1 8B should get ~65-69% on MMLU (Meta official)
"""
import asyncio
import json
import os
import re
import httpx
from pathlib import Path
from typing import List, Dict, Tuple

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "csk-hpr4pjyd895p4ktvpnn436exx49rr925f6dptjvmee5ycrx8")
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
MODEL = "llama3.1-8b"

# 5-shot examples by subject
EXAMPLES = {
    "math": """Question: What is 2 + 2?
A. 3  B. 4  C. 5  D. 6
Answer: B

Question: What is the square root of 16?
A. 2  B. 3  C. 4  D. 5
Answer: C

Question: If x = 3, what is 2x + 1?
A. 5  B. 6  C. 7  D. 8
Answer: C

Question: What is 15% of 100?
A. 10  B. 15  C. 20  D. 25
Answer: B

Question: How many sides does a hexagon have?
A. 5  B. 6  C. 7  D. 8
Answer: B

""",
    "science": """Question: What is the chemical symbol for water?
A. H2O  B. CO2  C. NaCl  D. O2
Answer: A

Question: What planet is closest to the Sun?
A. Venus  B. Mercury  C. Mars  D. Earth
Answer: B

Question: What is the atomic number of Carbon?
A. 4  B. 5  C. 6  D. 7
Answer: C

Question: What is the boiling point of water in Celsius?
A. 90  B. 100  C. 110  D. 120
Answer: B

Question: How many chromosomes do humans have?
A. 23  B. 46  C. 48  D. 64
Answer: B

""",
    "logic": """Question: All A are B, all B are C. Are all A also C?
A. Yes  B. No  C. Cannot determine  D. Sometimes
Answer: A

Question: If P implies Q, and Q is false, what about P?
A. P is true  B. P is false  C. Cannot determine  D. Might be true
Answer: B

Question: All dogs are mammals. Some mammals are pets. Therefore:
A. All dogs are pets  B. Some dogs are pets  C. No dogs are pets  D. Cannot determine
Answer: D

Question: If rain then wet ground. Ground is wet. Did it rain?
A. Yes  B. No  C. Cannot determine  D. Probably
Answer: C

Question: A > B, B > C. Is A > C?
A. Yes  B. No  C. Cannot determine  D. Sometimes
Answer: A

"""
}

# Test questions
TEST_QUESTIONS = [
    {"subject": "math", "q": "What is 7 * 8?", "choices": ["54", "56", "58", "64"], "answer": "B"},
    {"subject": "math", "q": "Area of rectangle length 5, width 3?", "choices": ["8", "12", "15", "18"], "answer": "C"},
    {"subject": "math", "q": "Train at 60 mph for 2 hours goes how far?", "choices": ["100 mi", "120 mi", "140 mi", "160 mi"], "answer": "B"},
    {"subject": "math", "q": "What is 25% of 80?", "choices": ["15", "20", "25", "30"], "answer": "B"},
    {"subject": "math", "q": "What is 9 squared?", "choices": ["72", "81", "90", "99"], "answer": "B"},
    {"subject": "math", "q": "If x + 5 = 12, what is x?", "choices": ["5", "6", "7", "8"], "answer": "C"},
    {"subject": "math", "q": "What is 144 / 12?", "choices": ["10", "11", "12", "13"], "answer": "C"},
    {"subject": "math", "q": "Perimeter of square with side 4?", "choices": ["8", "12", "16", "20"], "answer": "C"},
    {"subject": "math", "q": "What is 3^3?", "choices": ["9", "18", "27", "36"], "answer": "C"},
    {"subject": "math", "q": "8 slices, eat 3, what fraction left?", "choices": ["3/8", "4/8", "5/8", "6/8"], "answer": "C"},
    {"subject": "science", "q": "Gas plants release in photosynthesis?", "choices": ["CO2", "N2", "O2", "H2"], "answer": "C"},
    {"subject": "science", "q": "Largest planet in solar system?", "choices": ["Saturn", "Jupiter", "Neptune", "Uranus"], "answer": "B"},
    {"subject": "science", "q": "pH of pure water?", "choices": ["5", "6", "7", "8"], "answer": "C"},
    {"subject": "science", "q": "Chemical symbol for gold?", "choices": ["Go", "Gd", "Au", "Ag"], "answer": "C"},
    {"subject": "science", "q": "Bones in adult human body?", "choices": ["186", "196", "206", "216"], "answer": "C"},
    {"subject": "science", "q": "Speed of light approx?", "choices": ["300k m/s", "300k km/s", "3M m/s", "30k km/s"], "answer": "B"},
    {"subject": "science", "q": "Atomic number of Oxygen?", "choices": ["6", "7", "8", "9"], "answer": "C"},
    {"subject": "science", "q": "Water freezing point in F?", "choices": ["0F", "32F", "100F", "212F"], "answer": "B"},
    {"subject": "science", "q": "Which planet is the Red Planet?", "choices": ["Venus", "Mars", "Jupiter", "Saturn"], "answer": "B"},
    {"subject": "science", "q": "What type of animal is a dolphin?", "choices": ["Fish", "Amphibian", "Reptile", "Mammal"], "answer": "D"},
    {"subject": "logic", "q": "All cats are animals. Animals need food. Do cats need food?", "choices": ["Yes", "No", "Cannot determine", "Sometimes"], "answer": "A"},
    {"subject": "logic", "q": "Today Monday, what day 3 days ago?", "choices": ["Thursday", "Friday", "Saturday", "Sunday"], "answer": "B"},
    {"subject": "logic", "q": "Some birds fly. Penguins are birds. Can all penguins fly?", "choices": ["Yes", "No", "Cannot determine", "Probably"], "answer": "B"},
    {"subject": "logic", "q": "X > Y, Y > Z. Relationship of X and Z?", "choices": ["X > Z", "X < Z", "X = Z", "Cannot determine"], "answer": "A"},
    {"subject": "logic", "q": "All squares are rectangles. Every rectangle a square?", "choices": ["Yes", "No", "Cannot determine", "Sometimes"], "answer": "B"},
    {"subject": "logic", "q": "Not P implies not Q. Q true. What about P?", "choices": ["P true", "P false", "Cannot determine", "Might be true"], "answer": "A"},
    {"subject": "logic", "q": "Mon before Tue, Wed after Tue. What first?", "choices": ["Monday", "Tuesday", "Wednesday", "Cannot determine"], "answer": "A"},
    {"subject": "logic", "q": "No fish are mammals. Whales are mammals. Are whales fish?", "choices": ["Yes", "No", "Cannot determine", "Sometimes"], "answer": "B"},
    {"subject": "logic", "q": "A = B, B = C. Does A = C?", "choices": ["Yes", "No", "Cannot determine", "Sometimes"], "answer": "A"},
    {"subject": "logic", "q": "Opposite of 'always'?", "choices": ["Sometimes", "Never", "Often", "Rarely"], "answer": "B"},
]


class CerebrasLLM:
    def __init__(self, client):
        self.client = client
        self.total_tokens = 0

    async def generate(self, prompt: str, max_tokens: int = 5) -> Tuple[str, int]:
        for attempt in range(3):
            try:
                resp = await self.client.post(
                    CEREBRAS_URL,
                    headers={"Authorization": f"Bearer {CEREBRAS_API_KEY}"},
                    json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": max_tokens, "temperature": 0.0,
                          "stop": ["\n", "Question"]},
                    timeout=60.0
                )
                if resp.status_code == 429:
                    await asyncio.sleep(3 * (attempt + 1))
                    continue
                if resp.status_code != 200:
                    return "", 0
                data = resp.json()
                tokens = data.get("usage", {}).get("total_tokens", 0)
                self.total_tokens += tokens
                return data["choices"][0]["message"]["content"].strip(), tokens
            except:
                await asyncio.sleep(1)
        return "", 0


def extract_choice(resp: str) -> str:
    if not resp:
        return ""
    resp = resp.strip().upper()
    # Direct letter
    if resp and resp[0] in 'ABCD':
        return resp[0]
    # Pattern match
    m = re.match(r'^[\(\s]*([ABCD])[\)\.\s]*', resp)
    if m:
        return m.group(1)
    m = re.search(r'\b([ABCD])\b', resp)
    if m:
        return m.group(1)
    return ""


def build_5shot_prompt(q: Dict) -> str:
    examples = EXAMPLES.get(q["subject"], EXAMPLES["math"])
    return f"""The following are multiple choice questions about {q['subject']}.

{examples}Question: {q['q']}
A. {q['choices'][0]}  B. {q['choices'][1]}  C. {q['choices'][2]}  D. {q['choices'][3]}
Answer:"""


async def run_old(llm, questions):
    print("\n--- OLD METHOD (0-shot, 200 tokens) ---")
    correct = 0
    for i, q in enumerate(questions):
        prompt = f"Q: {q['q']}\nA. {q['choices'][0]} B. {q['choices'][1]} C. {q['choices'][2]} D. {q['choices'][3]}\nAnswer:"
        resp, _ = await llm.generate(prompt, 200)
        ans = extract_choice(resp)
        if ans == q["answer"]:
            correct += 1
        if (i+1) % 10 == 0:
            print(f"  [{i+1:2d}/{len(questions)}] acc={100*correct/(i+1):.1f}%")
        await asyncio.sleep(0.1)
    return {"correct": correct, "accuracy": round(100*correct/len(questions), 1), "tokens": llm.total_tokens}


async def run_fixed(llm, questions):
    print("\n--- FIXED METHOD (5-shot, 5 tokens) ---")
    correct = 0
    for i, q in enumerate(questions):
        prompt = build_5shot_prompt(q)
        resp, _ = await llm.generate(prompt, 5)
        ans = extract_choice(resp)
        if ans == q["answer"]:
            correct += 1
        if (i+1) % 10 == 0:
            print(f"  [{i+1:2d}/{len(questions)}] acc={100*correct/(i+1):.1f}%")
        await asyncio.sleep(0.1)
    return {"correct": correct, "accuracy": round(100*correct/len(questions), 1), "tokens": llm.total_tokens}


async def main():
    print("="*60)
    print("BENCHMARK: OLD vs FIXED EXTRACTION")
    print(f"Model: {MODEL} | Questions: {len(TEST_QUESTIONS)}")
    print("Expected: ~65-69% (Meta official)")
    print("="*60)

    async with httpx.AsyncClient() as client:
        llm1 = CerebrasLLM(client)
        old = await run_old(llm1, TEST_QUESTIONS)

        await asyncio.sleep(2)

        llm2 = CerebrasLLM(client)
        fixed = await run_fixed(llm2, TEST_QUESTIONS)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"OLD (0-shot):   {old['accuracy']:5.1f}% ({old['tokens']} tokens)")
    print(f"FIXED (5-shot): {fixed['accuracy']:5.1f}% ({fixed['tokens']} tokens)")
    print(f"Improvement:    {fixed['accuracy']-old['accuracy']:+.1f}pp")
    print(f"Expected:       65-69%")
    print("="*60)

    Path("/workspace/data").mkdir(exist_ok=True)
    with open("/workspace/data/fixed_extraction_results.json", "w") as f:
        json.dump({"old": old, "fixed": fixed}, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
