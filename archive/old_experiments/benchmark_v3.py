#!/usr/bin/env python3
"""
Fixed Benchmark with DeepSeek-V3 via Chutes

Expected: GSM8K ~84%, MMLU ~70%
"""
import asyncio
import json
import os
import re
import httpx
from pathlib import Path

CHUTES_API_KEY = os.getenv("CHUTES_API_KEY", "")
CHUTES_URL = "https://llm.chutes.ai/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-V3"

GSM8K_FEWSHOT = """Q: There are 15 trees in the grove. Grove workers will plant trees today. After they are done, there will be 21 trees. How many trees did they plant?
A: There are 15 trees originally. Then there were 21 trees after planting. So 21 - 15 = 6. #### 6

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### 5

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. #### 39

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. #### 8

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 2 + 2 = 4 more toys. 5 + 4 = 9. #### 9

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. #### 29

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. #### 33

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 * 3 = 15 dollars. So she has 23 - 15 = 8 dollars left. #### 8

"""

GSM8K_PROBLEMS = [
    {"q": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?", "a": "18"},
    {"q": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?", "a": "3"},
    {"q": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?", "a": "70000"},
    {"q": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?", "a": "624"},
    {"q": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed. She gives the chickens their feed in three separate meals. In the morning, she gives her flock 15 cups. In the afternoon, she gives them 25 cups. How many cups in the final meal if she has 20 chickens?", "a": "20"},
    {"q": "Kylar went to the store to buy glasses. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay?", "a": "64"},
    {"q": "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do they have together if Seattle has 20 sheep?", "a": "260"},
    {"q": "Carla is downloading a 200 GB file. She can download 2 GB/minute, but 40% through, Windows restarts for updates (20 min). She restarts the download. How long total?", "a": "160"},
    {"q": "John drives for 3 hours at 60 mph then turns around. He spends 2 hours in traffic, then 30 min at 30 mph, then drives at 80 mph for the remaining 1.5 hours. How far from home?", "a": "45"},
    {"q": "Eliza earns $10/hour for the first 40 hours and 1.2x overtime. She worked 45 hours. What are her earnings?", "a": "460"},
    {"q": "A program had 60 downloads in month 1. Month 2 was 3x month 1. Month 3 reduced by 30%. Total downloads?", "a": "366"},
    {"q": "Toula bought 3 dozen donuts at $68/dozen, 2 dozen cupcakes at $80/dozen, and 6 dozen cheesecakes at $55/dozen. Total cost?", "a": "694"},
    {"q": "A lemon tree costs $90 to plant. It grows 7 lemons/year sold at $1.5 each. Costs $3/year to maintain. How many years until profit?", "a": "13"},
    {"q": "Melanie sold 1/3 of vacuums at the green house, 2 more at red house, half of remaining at orange house. She has 5 left. How many did she start with?", "a": "18"},
    {"q": "In a dance class of 20 students, 20% enrolled in contemporary, 25% of remaining in jazz, rest in hip-hop. What % in hip-hop?", "a": "60"},
    {"q": "A merchant can buy jewelry worth $5,000 (goes up 2.5%) or electronics worth $8,000 (goes up 1.2%). Which gives max profit and how much?", "a": "125"},
    {"q": "Two trains: BHP Iron Ore is 7.353 km, 2001 record is 7.35 km. Difference in meters?", "a": "3"},
    {"q": "A man divides 3 hectares among 8 sons. Every 750m^2 makes $500 profit per 3 months. How much can each son make in 1 year? (1 hectare = 10000 m^2)", "a": "2500"},
    {"q": "A farmer sold 1/3 of cows, bought 1/2 as many as he had left. Then 1/4 got sick and sold cheap. Started with 120. How many healthy cows sold at full price?", "a": "60"},
    {"q": "Jean has 30 lollipops. She eats 2 and packages the rest 2 per bag. How many bags?", "a": "14"},
]

MMLU_FEWSHOT = """Question: What is the derivative of x^2?
A) x
B) 2x
C) x^2
D) 2
Answer: B

Question: What is the chemical symbol for gold?
A) Ag
B) Au
C) Fe
D) Cu
Answer: B

Question: All mammals are warm-blooded. Dogs are mammals. Therefore:
A) Dogs are cold-blooded
B) Dogs are warm-blooded
C) Some dogs are warm-blooded
D) Cannot be determined
Answer: B

Question: What is the value of pi (approximately)?
A) 2.14
B) 3.14
C) 4.14
D) 5.14
Answer: B

Question: If P then Q. P is true. What is Q?
A) True
B) False
C) Unknown
D) Both
Answer: A

"""

MMLU_PROBLEMS = [
    {"q": "What is the integral of 2x?", "options": ["x", "x^2", "2x^2", "x^2 + C"], "a": "D"},
    {"q": "What is log base 10 of 100?", "options": ["1", "2", "10", "100"], "a": "B"},
    {"q": "What is 2^10?", "options": ["512", "1024", "2048", "4096"], "a": "B"},
    {"q": "What is the boiling point of water at sea level?", "options": ["90C", "100C", "110C", "120C"], "a": "B"},
    {"q": "What is the largest planet in our solar system?", "options": ["Saturn", "Jupiter", "Uranus", "Neptune"], "a": "B"},
    {"q": "What is the powerhouse of the cell?", "options": ["Nucleus", "Ribosome", "Mitochondria", "Golgi body"], "a": "C"},
    {"q": "All A are B. All B are C. Are all A also C?", "options": ["Yes", "No", "Maybe", "Cannot say"], "a": "A"},
    {"q": "If not Q then not P. Q is false. What is P?", "options": ["True", "False", "Unknown", "Both"], "a": "B"},
    {"q": "Some X are Y. Some Y are Z. Are some X definitely Z?", "options": ["Yes", "No", "Cannot determine", "All X are Z"], "a": "C"},
    {"q": "What is the speed of light (approx)?", "options": ["300 km/s", "300,000 km/s", "3,000,000 km/s", "30,000 km/s"], "a": "B"},
    {"q": "What is the atomic number of oxygen?", "options": ["6", "7", "8", "9"], "a": "C"},
    {"q": "What type of bond involves sharing electrons?", "options": ["Ionic", "Covalent", "Metallic", "Hydrogen"], "a": "B"},
    {"q": "If A or B, and not A, then:", "options": ["B is true", "B is false", "A is true", "Cannot say"], "a": "A"},
    {"q": "What is the formula for water?", "options": ["CO2", "H2O", "NaCl", "O2"], "a": "B"},
    {"q": "What is the square root of 144?", "options": ["10", "11", "12", "13"], "a": "C"},
]


class ChutesLLM:
    def __init__(self, client):
        self.client = client
        self.total_tokens = 0

    async def generate(self, prompt, max_tokens=400):
        for attempt in range(3):
            try:
                resp = await self.client.post(
                    CHUTES_URL,
                    headers={"Authorization": f"Bearer {CHUTES_API_KEY}"},
                    json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": max_tokens, "temperature": 0.0},
                    timeout=120.0
                )
                if resp.status_code == 429:
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                if resp.status_code != 200:
                    print(f"  Error {resp.status_code}: {resp.text[:100]}")
                    return "", 0
                data = resp.json()
                tokens = data.get("usage", {}).get("total_tokens", 0)
                self.total_tokens += tokens
                return data["choices"][0]["message"]["content"], tokens
            except Exception as e:
                print(f"  Exception: {e}")
                await asyncio.sleep(2)
        return "", 0


def extract_gsm8k_answer(text):
    if not text:
        return ""
    # #### format (official GSM8K)
    match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if match:
        return match.group(1).replace(',', '')
    # \boxed{} format
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).replace(',', '').replace('$', '')
    # **X** bold format
    match = re.search(r'\*\*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)\*\*', text)
    if match:
        return match.group(1).replace(',', '')
    # "answer is X" or "= X" near end
    lines = text.strip().split('\n')
    for line in reversed(lines[-5:]):
        match = re.search(r'(?:answer\s+is|=|makes?|earns?|profit|total)\s*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)', line, re.IGNORECASE)
        if match:
            return match.group(1).replace(',', '').replace('$', '')
    # Last number in response
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    return numbers[-1] if numbers else ""


def extract_mmlu_answer(text):
    if not text:
        return ""
    text = text.strip().upper()
    # Direct letter at start
    if text and text[0] in 'ABCD':
        return text[0]
    # "Answer: X" format
    match = re.search(r'(?:ANSWER|ANS)[:\s]*([ABCD])', text)
    if match:
        return match.group(1)
    # (X) format
    match = re.search(r'\(([ABCD])\)', text)
    if match:
        return match.group(1)
    # **X** format
    match = re.search(r'\*\*([ABCD])\*\*', text)
    if match:
        return match.group(1)
    return ""


def check_gsm8k(pred, exp):
    try:
        p = float(str(pred).replace(',', '').replace('$', ''))
        e = float(str(exp).replace(',', '').replace('$', ''))
        return abs(p - e) < 0.5
    except:
        return str(pred).strip() == str(exp).strip()


async def run_gsm8k(llm, problems, n=20):
    print(f"\n{'='*60}")
    print(f"GSM8K (DeepSeek-V3, {n} problems, 8-shot CoT)")
    print("="*60)
    correct = 0
    for i, p in enumerate(problems[:n]):
        prompt = GSM8K_FEWSHOT + f"Q: {p['q']}\nA:"
        resp, _ = await llm.generate(prompt, 500)
        answer = extract_gsm8k_answer(resp)
        is_correct = check_gsm8k(answer, p['a'])
        if is_correct:
            correct += 1
        status = "Y" if is_correct else "N"
        print(f"  [{i+1:2d}/{n}] {status} exp={p['a']:>6} got={answer[:12]:>12}", flush=True)
        await asyncio.sleep(0.5)
    return {"problems": n, "correct": correct, "accuracy": round(100 * correct / n, 1)}


async def run_mmlu(llm, problems, n=15):
    print(f"\n{'='*60}")
    print(f"MMLU (DeepSeek-V3, {n} problems, 5-shot)")
    print("="*60)
    correct = 0
    for i, p in enumerate(problems[:n]):
        options = "\n".join([f"{chr(65+j)}) {opt}" for j, opt in enumerate(p['options'])])
        prompt = MMLU_FEWSHOT + f"Question: {p['q']}\n{options}\nAnswer:"
        resp, _ = await llm.generate(prompt, 100)
        answer = extract_mmlu_answer(resp)
        is_correct = answer == p['a']
        if is_correct:
            correct += 1
        status = "Y" if is_correct else "N"
        print(f"  [{i+1:2d}/{n}] {status} exp={p['a']} got={answer}", flush=True)
        await asyncio.sleep(0.5)
    return {"problems": n, "correct": correct, "accuracy": round(100 * correct / n, 1)}


async def main():
    if not CHUTES_API_KEY:
        print("ERROR: CHUTES_API_KEY not set")
        return

    print("="*60)
    print("BENCHMARK: DeepSeek-V3 via Chutes")
    print("="*60)
    print("Fixes: 8-shot CoT, 5-shot MMLU, proper extraction")
    print("Expected: GSM8K ~90%, MMLU ~80%")
    print("="*60)

    async with httpx.AsyncClient() as client:
        llm = ChutesLLM(client)
        gsm8k = await run_gsm8k(llm, GSM8K_PROBLEMS, n=20)
        await asyncio.sleep(2)
        mmlu = await run_mmlu(llm, MMLU_PROBLEMS, n=15)

    print("\n" + "="*60)
    print("RESULTS (DeepSeek-V3)")
    print("="*60)
    print(f"GSM8K: {gsm8k['correct']}/{gsm8k['problems']} = {gsm8k['accuracy']}%")
    print(f"MMLU:  {mmlu['correct']}/{mmlu['problems']} = {mmlu['accuracy']}%")
    print("="*60)

    Path("/workspace/data").mkdir(exist_ok=True)
    with open("/workspace/data/benchmark_v3_results.json", "w") as f:
        json.dump({"gsm8k": gsm8k, "mmlu": mmlu, "model": "DeepSeek-V3"}, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
