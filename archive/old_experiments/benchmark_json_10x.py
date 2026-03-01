#!/usr/bin/env python3
"""10x JSON Benchmark - DeepSeek-V3"""
import asyncio
import json
import os
import re
import random
import time
import httpx
from pathlib import Path
from datetime import datetime

CHUTES_API_KEY = os.getenv("CHUTES_API_KEY", "")
CHUTES_URL = "https://llm.chutes.ai/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-V3"

GSM8K_FEWSHOT = """Q: There are 15 trees. After planting, there are 21. How many planted?
A: 21 - 15 = 6. #### 6

Q: 3 cars, 2 more arrive. Total?
A: 3 + 2 = 5. #### 5

Q: 32 + 42 chocolates, ate 35. Left?
A: 74 - 35 = 39. #### 39

Q: Had 20 lollipops, now 12. How many given?
A: 20 - 12 = 8. #### 8

"""

def generate_problems(n=100):
    problems = []
    templates = [
        ("A store sells items for ${p} each. How much for {n} items?", lambda p,n: p*n),
        ("A train goes {s} mph for {t} hours. Distance?", lambda s,t: s*t),
        ("{w1} workers do job in {d} days. Days for {w2} workers?", lambda w1,d,w2: (w1*d)//w2),
        ("What is {p}% of {n}?", lambda p,n: p*n//100),
        ("Rectangle: length {l}, width {w}. Area?", lambda l,w: l*w),
        ("${m}, spend {p}%. Left?", lambda m,p: m - m*p//100),
        ("Factory: {n} units/hour. Units in {h} hours?", lambda n,h: n*h),
        ("${r}/hour for {h} hours. Total?", lambda r,h: r*h),
        ("{a} + {b} = ?", lambda a,b: a+b),
        ("{a} * {b} = ?", lambda a,b: a*b),
    ]
    
    i = 0
    while len(problems) < n:
        t_idx = i % len(templates)
        template, calc = templates[t_idx]
        try:
            if "items" in template:
                p, n_val = random.choice([(2,5), (3,4), (5,6), (4,8), (6,3), (7,4), (8,5), (9,3)])
                q = template.format(p=p, n=n_val)
                a = str(calc(p, n_val))
            elif "train" in template:
                s, t = random.choice([(60,3), (50,4), (80,2), (40,5), (70,3), (90,2), (55,4), (65,3)])
                q = template.format(s=s, t=t)
                a = str(calc(s, t))
            elif "workers" in template:
                w1, d, w2 = random.choice([(4,12,6), (5,10,10), (3,15,5), (6,8,4), (8,6,12), (10,5,5)])
                q = template.format(w1=w1, d=d, w2=w2)
                a = str(calc(w1, d, w2))
            elif "% of" in template:
                p, n_val = random.choice([(10,200), (15,80), (20,150), (25,120), (30,100), (50,80)])
                q = template.format(p=p, n=n_val)
                a = str(calc(p, n_val))
            elif "Rectangle" in template:
                l, w = random.choice([(8,5), (10,6), (12,4), (7,9), (15,3), (9,8), (11,7)])
                q = template.format(l=l, w=w)
                a = str(calc(l, w))
            elif "spend" in template:
                m, p = random.choice([(100,20), (80,25), (50,40), (200,15), (150,30), (120,10)])
                q = template.format(m=m, p=p)
                a = str(calc(m, p))
            elif "Factory" in template:
                n_val, h = random.choice([(50,8), (30,6), (40,5), (25,4), (60,10), (45,9)])
                q = template.format(n=n_val, h=h)
                a = str(calc(n_val, h))
            elif "/hour" in template:
                r, h = random.choice([(10,8), (15,6), (20,5), (12,10), (25,4), (18,7)])
                q = template.format(r=r, h=h)
                a = str(calc(r, h))
            elif "+" in template:
                a_val, b_val = random.choice([(23,17), (45,55), (33,67), (12,88), (56,44), (78,22)])
                q = template.format(a=a_val, b=b_val)
                a = str(calc(a_val, b_val))
            elif "*" in template:
                a_val, b_val = random.choice([(7,8), (9,6), (12,5), (15,4), (11,7), (13,6)])
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


async def main():
    if not CHUTES_API_KEY:
        print(json.dumps({"error": "No CHUTES_API_KEY"}))
        return

    problems = generate_problems(100)
    results = []
    correct = 0
    total_tokens = 0
    start_time = time.time()

    async with httpx.AsyncClient() as client:
        for i, p in enumerate(problems):
            prompt = GSM8K_FEWSHOT + f"Q: {p['q']}\nA:"
            
            try:
                resp = await client.post(
                    CHUTES_URL,
                    headers={"Authorization": f"Bearer {CHUTES_API_KEY}"},
                    json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": 200, "temperature": 0.0},
                    timeout=60.0
                )
                
                if resp.status_code != 200:
                    results.append({"id": i, "error": f"status_{resp.status_code}"})
                    continue
                
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                tokens = data.get("usage", {}).get("total_tokens", 0)
                total_tokens += tokens
                
                # Extract answer
                m = re.search(r'####\s*(-?\d+)', content)
                answer = m.group(1) if m else re.findall(r'-?\d+', content)[-1] if re.findall(r'-?\d+', content) else ""
                
                is_correct = False
                try:
                    is_correct = abs(float(answer) - float(p["a"])) < 1
                except:
                    pass
                
                if is_correct:
                    correct += 1
                
                results.append({
                    "id": i,
                    "expected": p["a"],
                    "predicted": answer,
                    "correct": is_correct
                })
                
            except Exception as e:
                results.append({"id": i, "error": str(e)})
            
            if (i + 1) % 20 == 0:
                print(json.dumps({"progress": i + 1, "accuracy": round(100 * correct / (i + 1), 1)}), flush=True)
            
            await asyncio.sleep(0.2)

    elapsed = time.time() - start_time
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "benchmark": "gsm8k_10x",
        "problems": len(problems),
        "correct": correct,
        "accuracy": round(100 * correct / len(problems), 1),
        "total_tokens": total_tokens,
        "total_time": round(elapsed, 2),
        "avg_time_per_q": round(elapsed / len(problems), 3),
        "results": results
    }
    
    print(json.dumps(output, indent=2))
    
    Path("/workspace/data").mkdir(exist_ok=True)
    with open("/workspace/data/benchmark_v3_10x_json.json", "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
