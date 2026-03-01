#!/usr/bin/env python3
"""10x Benchmark with DeepSeek-V3"""
import asyncio
import json
import os
import re
import random
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

"""

def generate_gsm8k_problems(n=200):
    problems = []
    templates = [
        ("A store sells apples for ${p} each. If you buy {n} apples, how much do you pay?", lambda p,n: p*n),
        ("A train travels {s} mph for {t} hours. How many miles does it travel?", lambda s,t: s*t),
        ("If {w1} workers can do a job in {d} days, how many days for {w2} workers?", lambda w1,d,w2: (w1*d)//w2),
        ("{n} items cost ${c}. How much do {m} items cost?", lambda n,c,m: round(c*m/n, 2)),
        ("What is {p}% of {n}?", lambda p,n: p*n//100),
        ("A rectangle has length {l} and width {w}. What is its area?", lambda l,w: l*w),
        ("{name} has ${m}. They spend {p}% on food. How much is left?", lambda m,p: m - m*p//100),
        ("A factory makes {n} units per hour. How many in {h} hours?", lambda n,h: n*h),
        ("{name} earns ${r}/hour. How much for {h} hours of work?", lambda r,h: r*h),
        ("If {a} + {b} = x, what is x?", lambda a,b: a+b),
    ]
    names = ["John", "Mary", "Tom", "Lisa", "Alex", "Sarah"]
    
    i = 0
    while len(problems) < n:
        t_idx = i % len(templates)
        template, calc = templates[t_idx]
        try:
            if "apples" in template:
                p, n_val = random.choice([(2,5), (3,4), (5,6), (4,8), (6,3)])
                q = template.format(p=p, n=n_val)
                a = str(calc(p, n_val))
            elif "train" in template:
                s, t = random.choice([(60,3), (50,4), (80,2), (40,5), (70,3)])
                q = template.format(s=s, t=t)
                a = str(calc(s, t))
            elif "workers" in template:
                w1, d, w2 = random.choice([(4,12,6), (5,10,10), (3,15,5), (6,8,4)])
                q = template.format(w1=w1, d=d, w2=w2)
                a = str(calc(w1, d, w2))
            elif "items cost" in template:
                n_val, c, m = random.choice([(3,6,5), (4,8,7), (5,10,8), (2,4,6)])
                q = template.format(n=n_val, c=c, m=m)
                a = str(calc(n_val, c, m))
            elif "% of" in template:
                p, n_val = random.choice([(10,200), (15,80), (20,150), (25,120)])
                q = template.format(p=p, n=n_val)
                a = str(calc(p, n_val))
            elif "rectangle" in template:
                l, w = random.choice([(8,5), (10,6), (12,4), (7,9)])
                q = template.format(l=l, w=w)
                a = str(calc(l, w))
            elif "spend" in template:
                m, p = random.choice([(100,20), (80,25), (50,40), (200,15)])
                name = random.choice(names)
                q = template.format(name=name, m=m, p=p)
                a = str(calc(m, p))
            elif "factory" in template:
                n_val, h = random.choice([(50,8), (30,6), (40,5), (25,4)])
                q = template.format(n=n_val, h=h)
                a = str(calc(n_val, h))
            elif "earns" in template:
                r, h = random.choice([(10,8), (15,6), (20,5), (12,10)])
                name = random.choice(names)
                q = template.format(name=name, r=r, h=h)
                a = str(calc(r, h))
            elif "+" in template:
                a_val, b_val = random.choice([(23,17), (45,55), (33,67), (12,88)])
                q = template.format(a=a_val, b=b_val)
                a = str(calc(a_val, b_val))
            else:
                i += 1
                continue
            problems.append({"q": q, "a": a})
        except:
            pass
        i += 1
    return problems[:n]


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
                    await asyncio.sleep(10)
                    continue
                if resp.status_code != 200:
                    return "", 0
                data = resp.json()
                return data["choices"][0]["message"]["content"], data.get("usage", {}).get("total_tokens", 0)
            except:
                await asyncio.sleep(2)
        return "", 0


def extract_answer(text):
    if not text:
        return ""
    match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if match:
        return match.group(1).replace(',', '')
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).replace(',', '')
    match = re.search(r'\*\*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)\*\*', text)
    if match:
        return match.group(1).replace(',', '')
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    return numbers[-1] if numbers else ""


def check(pred, exp):
    try:
        return abs(float(pred) - float(exp)) < 0.5
    except:
        return str(pred).strip() == str(exp).strip()


async def main():
    if not CHUTES_API_KEY:
        print("ERROR: CHUTES_API_KEY not set")
        return

    print("="*60)
    print("10x BENCHMARK: DeepSeek-V3 (200 problems)")
    print("="*60)

    problems = generate_gsm8k_problems(200)

    async with httpx.AsyncClient() as client:
        llm = ChutesLLM(client)
        correct = 0
        
        for i, p in enumerate(problems):
            prompt = GSM8K_FEWSHOT + f"Q: {p['q']}\nA:"
            resp, _ = await llm.generate(prompt, 300)
            answer = extract_answer(resp)
            is_correct = check(answer, p['a'])
            if is_correct:
                correct += 1
            
            if (i+1) % 20 == 0:
                acc = 100 * correct / (i+1)
                print(f"  [{i+1:3d}/200] acc={acc:.1f}%", flush=True)
            
            await asyncio.sleep(0.3)

    acc = round(100 * correct / 200, 1)
    print("\n" + "="*60)
    print(f"RESULT: {correct}/200 = {acc}%")
    print("="*60)

    Path("/workspace/data").mkdir(exist_ok=True)
    with open("/workspace/data/benchmark_v3_10x_results.json", "w") as f:
        json.dump({"problems": 200, "correct": correct, "accuracy": acc}, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
