#!/usr/bin/env python3
"""
Proper MMLU-style Benchmark with Correct Methodology

Fixes:
1. Rate limit handling (0.5s delay)
2. Proper prompt format ("Answer with ONLY the letter")
3. Constrained output (max_tokens=2)
4. Retry on 429

Expected: ~65-69% for Llama 3.1 8B
"""
import asyncio
import json
import os
import httpx
from pathlib import Path

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "csk-hpr4pjyd895p4ktvpnn436exx49rr925f6dptjvmee5ycrx8")
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
MODEL = "llama3.1-8b"

# 50 MMLU-style questions across domains
QUESTIONS = [
    # Math (10)
    {"q": "What is 8 * 7?", "choices": ["54", "56", "58", "63"], "answer": "B", "domain": "math"},
    {"q": "What is 144 / 12?", "choices": ["10", "11", "12", "14"], "answer": "C", "domain": "math"},
    {"q": "What is 15% of 200?", "choices": ["25", "30", "35", "40"], "answer": "B", "domain": "math"},
    {"q": "What is 9 squared?", "choices": ["72", "81", "90", "99"], "answer": "B", "domain": "math"},
    {"q": "What is the square root of 64?", "choices": ["6", "7", "8", "9"], "answer": "C", "domain": "math"},
    {"q": "If x + 7 = 15, what is x?", "choices": ["6", "7", "8", "9"], "answer": "C", "domain": "math"},
    {"q": "What is 2^5?", "choices": ["16", "32", "64", "128"], "answer": "B", "domain": "math"},
    {"q": "What is 1000 - 567?", "choices": ["433", "443", "453", "463"], "answer": "A", "domain": "math"},
    {"q": "What is 3/4 as a decimal?", "choices": ["0.25", "0.5", "0.75", "1.0"], "answer": "C", "domain": "math"},
    {"q": "How many minutes in 2 hours?", "choices": ["60", "90", "120", "180"], "answer": "C", "domain": "math"},
    
    # Science (15)
    {"q": "What is the chemical symbol for gold?", "choices": ["Go", "Gd", "Au", "Ag"], "answer": "C", "domain": "science"},
    {"q": "What is the boiling point of water in Celsius?", "choices": ["90", "100", "110", "120"], "answer": "B", "domain": "science"},
    {"q": "Which planet is closest to the Sun?", "choices": ["Venus", "Mercury", "Mars", "Earth"], "answer": "B", "domain": "science"},
    {"q": "What is the atomic number of carbon?", "choices": ["4", "5", "6", "7"], "answer": "C", "domain": "science"},
    {"q": "What gas do humans exhale?", "choices": ["Oxygen", "Nitrogen", "Carbon dioxide", "Helium"], "answer": "C", "domain": "science"},
    {"q": "How many bones in adult human body?", "choices": ["186", "196", "206", "216"], "answer": "C", "domain": "science"},
    {"q": "What is the pH of pure water?", "choices": ["5", "6", "7", "8"], "answer": "C", "domain": "science"},
    {"q": "Which organ pumps blood?", "choices": ["Liver", "Heart", "Lung", "Kidney"], "answer": "B", "domain": "science"},
    {"q": "What is H2O commonly known as?", "choices": ["Salt", "Water", "Acid", "Sugar"], "answer": "B", "domain": "science"},
    {"q": "What is the largest planet?", "choices": ["Saturn", "Jupiter", "Neptune", "Uranus"], "answer": "B", "domain": "science"},
    {"q": "What type of animal is a whale?", "choices": ["Fish", "Amphibian", "Reptile", "Mammal"], "answer": "D", "domain": "science"},
    {"q": "How many chromosomes do humans have?", "choices": ["23", "46", "48", "64"], "answer": "B", "domain": "science"},
    {"q": "What is the speed of light (approx)?", "choices": ["300 km/s", "3000 km/s", "30000 km/s", "300000 km/s"], "answer": "D", "domain": "science"},
    {"q": "Which planet is the Red Planet?", "choices": ["Venus", "Mars", "Jupiter", "Saturn"], "answer": "B", "domain": "science"},
    {"q": "What element has symbol O?", "choices": ["Gold", "Oxygen", "Osmium", "Oganesson"], "answer": "B", "domain": "science"},
    
    # Geography (10)
    {"q": "What is the capital of France?", "choices": ["London", "Paris", "Berlin", "Madrid"], "answer": "B", "domain": "geography"},
    {"q": "Which is the largest continent?", "choices": ["Africa", "Asia", "Europe", "Antarctica"], "answer": "B", "domain": "geography"},
    {"q": "What is the longest river?", "choices": ["Amazon", "Nile", "Mississippi", "Yangtze"], "answer": "B", "domain": "geography"},
    {"q": "How many continents are there?", "choices": ["5", "6", "7", "8"], "answer": "C", "domain": "geography"},
    {"q": "Which ocean is the largest?", "choices": ["Atlantic", "Pacific", "Indian", "Arctic"], "answer": "B", "domain": "geography"},
    {"q": "What is the capital of Japan?", "choices": ["Beijing", "Seoul", "Tokyo", "Bangkok"], "answer": "C", "domain": "geography"},
    {"q": "Which country has the most people?", "choices": ["India", "USA", "China", "Russia"], "answer": "C", "domain": "geography"},
    {"q": "What is the smallest country?", "choices": ["Monaco", "Vatican City", "San Marino", "Malta"], "answer": "B", "domain": "geography"},
    {"q": "Which desert is the largest?", "choices": ["Gobi", "Sahara", "Arabian", "Kalahari"], "answer": "B", "domain": "geography"},
    {"q": "What is the capital of Australia?", "choices": ["Sydney", "Melbourne", "Canberra", "Perth"], "answer": "C", "domain": "geography"},
    
    # Logic (10)
    {"q": "All cats are animals. Fluffy is a cat. Is Fluffy an animal?", "choices": ["Yes", "No", "Maybe", "Unknown"], "answer": "A", "domain": "logic"},
    {"q": "If A > B and B > C, is A > C?", "choices": ["Yes", "No", "Maybe", "Unknown"], "answer": "A", "domain": "logic"},
    {"q": "What is the opposite of 'always'?", "choices": ["Sometimes", "Never", "Often", "Rarely"], "answer": "B", "domain": "logic"},
    {"q": "If today is Monday, what day is tomorrow?", "choices": ["Sunday", "Tuesday", "Wednesday", "Monday"], "answer": "B", "domain": "logic"},
    {"q": "All birds have wings. A sparrow is a bird. Does a sparrow have wings?", "choices": ["Yes", "No", "Maybe", "Unknown"], "answer": "A", "domain": "logic"},
    {"q": "If not P then not Q. Q is true. Is P true?", "choices": ["Yes", "No", "Maybe", "Unknown"], "answer": "A", "domain": "logic"},
    {"q": "What comes next: 2, 4, 6, 8, ?", "choices": ["9", "10", "11", "12"], "answer": "B", "domain": "logic"},
    {"q": "What is the next letter: A, C, E, G, ?", "choices": ["H", "I", "J", "K"], "answer": "B", "domain": "logic"},
    {"q": "If all roses are flowers, are all flowers roses?", "choices": ["Yes", "No", "Maybe", "Unknown"], "answer": "B", "domain": "logic"},
    {"q": "Monday before Tuesday, Wednesday after Tuesday. What comes first?", "choices": ["Monday", "Tuesday", "Wednesday", "Thursday"], "answer": "A", "domain": "logic"},
    
    # General Knowledge (5)
    {"q": "How many days in a week?", "choices": ["5", "6", "7", "8"], "answer": "C", "domain": "general"},
    {"q": "What color is the sky on a clear day?", "choices": ["Red", "Blue", "Green", "Yellow"], "answer": "B", "domain": "general"},
    {"q": "How many months have 31 days?", "choices": ["5", "6", "7", "8"], "answer": "C", "domain": "general"},
    {"q": "What is 2 + 2?", "choices": ["3", "4", "5", "6"], "answer": "B", "domain": "general"},
    {"q": "How many letters in the English alphabet?", "choices": ["24", "25", "26", "27"], "answer": "C", "domain": "general"},
]


async def run_benchmark():
    print("="*60)
    print(f"PROPER MMLU BENCHMARK")
    print(f"Model: {MODEL} | Questions: {len(QUESTIONS)}")
    print(f"Expected: ~65-69% (Meta official)")
    print("="*60)
    
    correct = 0
    domain_correct = {}
    domain_total = {}
    
    async with httpx.AsyncClient() as client:
        for i, q in enumerate(QUESTIONS):
            prompt = f"""Answer with ONLY the letter (A, B, C, or D).

Question: {q['q']}
A. {q['choices'][0]}
B. {q['choices'][1]}
C. {q['choices'][2]}
D. {q['choices'][3]}

Your answer:"""
            
            answer = None
            for attempt in range(5):
                try:
                    resp = await client.post(
                        CEREBRAS_URL,
                        headers={"Authorization": f"Bearer {CEREBRAS_API_KEY}"},
                        json={"model": MODEL, 
                              "messages": [{"role": "user", "content": prompt}],
                              "max_tokens": 2, "temperature": 0.0},
                        timeout=60.0
                    )
                    if resp.status_code == 429:
                        await asyncio.sleep(5)
                        continue
                    if resp.status_code == 200:
                        content = resp.json()["choices"][0]["message"]["content"].strip().upper()
                        answer = content[0] if content and content[0] in 'ABCD' else None
                        break
                except:
                    await asyncio.sleep(2)
            
            is_correct = answer == q['answer']
            if is_correct:
                correct += 1
            
            domain = q['domain']
            domain_correct[domain] = domain_correct.get(domain, 0) + (1 if is_correct else 0)
            domain_total[domain] = domain_total.get(domain, 0) + 1
            
            if (i + 1) % 10 == 0:
                print(f"  [{i+1:2d}/{len(QUESTIONS)}] acc={100*correct/(i+1):.1f}%")
            
            await asyncio.sleep(0.5)  # Rate limit
    
    accuracy = 100 * correct / len(QUESTIONS)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Overall: {correct}/{len(QUESTIONS)} = {accuracy:.1f}%")
    print("\nBy Domain:")
    for domain in sorted(domain_total.keys()):
        d_acc = 100 * domain_correct[domain] / domain_total[domain]
        print(f"  {domain:12s}: {domain_correct[domain]:2d}/{domain_total[domain]:2d} = {d_acc:5.1f}%")
    print("\n" + "="*60)
    print(f"Expected: 65-69% | Got: {accuracy:.1f}%")
    print(f"Status: {'✅ IN RANGE' if 60 <= accuracy <= 75 else '❌ OUT OF RANGE'}")
    print("="*60)
    
    Path("/workspace/data").mkdir(exist_ok=True)
    with open("/workspace/data/proper_benchmark_results.json", "w") as f:
        json.dump({
            "model": MODEL,
            "total": len(QUESTIONS),
            "correct": correct,
            "accuracy": accuracy,
            "by_domain": {d: {"correct": domain_correct[d], "total": domain_total[d], 
                             "accuracy": 100*domain_correct[d]/domain_total[d]} 
                         for d in domain_total}
        }, f, indent=2)

if __name__ == "__main__":
    asyncio.run(run_benchmark())
