#!/usr/bin/env python3
"""
Unified Benchmark Suite - JSON Output

All results output as structured JSON for easy parsing.
Supports: Cerebras, Chutes (DeepSeek-V3)
"""
import asyncio
import json
import os
import re
import sys
import time
import httpx
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# API Configuration
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "csk-hpr4pjyd895p4ktvpnn436exx49rr925f6dptjvmee5ycrx8")
CHUTES_API_KEY = os.getenv("CHUTES_API_KEY", "")

PROVIDERS = {
    "cerebras": {
        "url": "https://api.cerebras.ai/v1/chat/completions",
        "key": CEREBRAS_API_KEY,
        "model": "llama3.1-8b",
        "expected": {"gsm8k": 84, "mmlu": 70}
    },
    "chutes": {
        "url": "https://llm.chutes.ai/v1/chat/completions",
        "key": CHUTES_API_KEY,
        "model": "deepseek-ai/DeepSeek-V3",
        "expected": {"gsm8k": 90, "mmlu": 85}
    }
}

# Few-shot prompts
GSM8K_FEWSHOT = """Q: There are 15 trees in the grove. Grove workers will plant trees today. After they are done, there will be 21 trees. How many trees did they plant?
A: 21 - 15 = 6. #### 6

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: 3 + 2 = 5. #### 5

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: 32 + 42 = 74. 74 - 35 = 39. #### 39

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: 20 - 12 = 8. #### 8

"""

MMLU_FEWSHOT = """Question: What is the derivative of x^2?
A) x  B) 2x  C) x^2  D) 2
Answer: B

Question: What is the chemical symbol for gold?
A) Ag  B) Au  C) Fe  D) Cu
Answer: B

Question: All mammals are warm-blooded. Dogs are mammals. Therefore:
A) Dogs are cold-blooded  B) Dogs are warm-blooded  C) Some dogs are warm-blooded  D) Cannot be determined
Answer: B

Question: What is the value of pi (approximately)?
A) 2.14  B) 3.14  C) 4.14  D) 5.14
Answer: B

Question: If P then Q. P is true. What is Q?
A) True  B) False  C) Unknown  D) Both
Answer: A

"""

# Test problems
GSM8K_PROBLEMS = [
    {"q": "Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes four into muffins. She sells the rest at $2 each. How much does she make daily?", "a": "18"},
    {"q": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total?", "a": "3"},
    {"q": "Josh buys a house for $80,000 and puts in $50,000 in repairs. This increased the value by 150%. How much profit did he make?", "a": "70000"},
    {"q": "James writes a 3-page letter to 2 friends twice a week. How many pages does he write a year?", "a": "624"},
    {"q": "Wendi feeds each of her 20 chickens 3 cups of feed daily in 3 meals. Morning: 15 cups. Afternoon: 25 cups. How many cups in the final meal?", "a": "20"},
    {"q": "One glass costs $5, every second glass costs 60% of the price. How much for 16 glasses?", "a": "64"},
    {"q": "Toulouse has 2x sheep as Charleston. Charleston has 4x sheep as Seattle. Seattle has 20 sheep. Total sheep?", "a": "260"},
    {"q": "Carla downloads 200GB at 2GB/min. At 40%, Windows restarts (20 min) and she restarts download. Total time?", "a": "160"},
    {"q": "John drives 3 hours at 60mph then turns around. Spends 2 hours in traffic, 30min at 30mph, then 80mph for 1.5 hours. How far from home?", "a": "45"},
    {"q": "Eliza earns $10/hour for first 40 hours and 1.2x overtime. She worked 45 hours. Total earnings?", "a": "460"},
]

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
]


class LLMClient:
    def __init__(self, provider: str, client: httpx.AsyncClient):
        self.provider = provider
        self.config = PROVIDERS[provider]
        self.client = client
        self.total_tokens = 0
        self.total_time = 0
        self.calls = 0

    async def generate(self, prompt: str, max_tokens: int = 400) -> Dict:
        start = time.time()
        for attempt in range(3):
            try:
                resp = await self.client.post(
                    self.config["url"],
                    headers={"Authorization": f"Bearer {self.config['key']}"},
                    json={
                        "model": self.config["model"],
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0.0
                    },
                    timeout=120.0
                )
                elapsed = time.time() - start
                
                if resp.status_code == 429:
                    return {"error": "rate_limited", "status": 429}
                if resp.status_code != 200:
                    return {"error": resp.text[:100], "status": resp.status_code}
                
                data = resp.json()
                tokens = data.get("usage", {}).get("total_tokens", 0)
                self.total_tokens += tokens
                self.total_time += elapsed
                self.calls += 1
                
                return {
                    "content": data["choices"][0]["message"]["content"],
                    "tokens": tokens,
                    "time": round(elapsed, 3)
                }
            except Exception as e:
                if attempt == 2:
                    return {"error": str(e)}
                await asyncio.sleep(2)
        return {"error": "max_retries"}


def extract_gsm8k(text: str) -> str:
    if not text:
        return ""
    # #### format
    m = re.search(r'####\s*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if m: return m.group(1).replace(',', '')
    # \boxed{}
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    if m: return m.group(1).replace(',', '').replace('$', '')
    # **X**
    m = re.search(r'\*\*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)\*\*', text)
    if m: return m.group(1).replace(',', '')
    # Last number
    nums = re.findall(r'-?\d+(?:\.\d+)?', text)
    return nums[-1] if nums else ""


def extract_mmlu(text: str) -> str:
    if not text:
        return ""
    text = text.strip().upper()
    if text and text[0] in 'ABCD':
        return text[0]
    m = re.search(r'(?:ANSWER|ANS)[:\s]*([ABCD])', text)
    if m: return m.group(1)
    m = re.search(r'\*\*([ABCD])\*\*', text)
    if m: return m.group(1)
    return ""


def check_numeric(pred: str, exp: str) -> bool:
    try:
        return abs(float(pred) - float(exp)) < 1
    except:
        return pred.strip() == exp.strip()


async def run_gsm8k(llm: LLMClient, problems: List[Dict]) -> Dict:
    results = []
    correct = 0
    
    for i, p in enumerate(problems):
        prompt = GSM8K_FEWSHOT + f"Q: {p['q']}\nA:"
        resp = await llm.generate(prompt, 400)
        
        if "error" in resp:
            results.append({"id": i, "error": resp["error"]})
            continue
            
        answer = extract_gsm8k(resp["content"])
        is_correct = check_numeric(answer, p["a"])
        if is_correct:
            correct += 1
            
        results.append({
            "id": i,
            "expected": p["a"],
            "predicted": answer,
            "correct": is_correct,
            "tokens": resp["tokens"],
            "time": resp["time"]
        })
        await asyncio.sleep(0.3)
    
    return {
        "benchmark": "gsm8k",
        "problems": len(problems),
        "correct": correct,
        "accuracy": round(100 * correct / len(problems), 1),
        "results": results
    }


async def run_mmlu(llm: LLMClient, problems: List[Dict]) -> Dict:
    results = []
    correct = 0
    
    for i, p in enumerate(problems):
        opts = " ".join([f"{chr(65+j)}) {o}" for j, o in enumerate(p["options"])])
        prompt = MMLU_FEWSHOT + f"Question: {p['q']}\n{opts}\nAnswer:"
        resp = await llm.generate(prompt, 50)
        
        if "error" in resp:
            results.append({"id": i, "error": resp["error"]})
            continue
            
        answer = extract_mmlu(resp["content"])
        is_correct = answer == p["a"]
        if is_correct:
            correct += 1
            
        results.append({
            "id": i,
            "expected": p["a"],
            "predicted": answer,
            "correct": is_correct,
            "tokens": resp["tokens"],
            "time": resp["time"]
        })
        await asyncio.sleep(0.3)
    
    return {
        "benchmark": "mmlu",
        "problems": len(problems),
        "correct": correct,
        "accuracy": round(100 * correct / len(problems), 1),
        "results": results
    }


async def main():
    # Select provider
    provider = sys.argv[1] if len(sys.argv) > 1 else "chutes"
    if provider not in PROVIDERS:
        print(json.dumps({"error": f"Unknown provider: {provider}"}))
        return
    
    if not PROVIDERS[provider]["key"]:
        print(json.dumps({"error": f"No API key for {provider}"}))
        return

    output = {
        "timestamp": datetime.now().isoformat(),
        "provider": provider,
        "model": PROVIDERS[provider]["model"],
        "expected": PROVIDERS[provider]["expected"],
        "benchmarks": {}
    }

    async with httpx.AsyncClient() as client:
        llm = LLMClient(provider, client)
        
        # Run GSM8K
        gsm8k = await run_gsm8k(llm, GSM8K_PROBLEMS)
        output["benchmarks"]["gsm8k"] = gsm8k
        
        await asyncio.sleep(1)
        
        # Run MMLU
        mmlu = await run_mmlu(llm, MMLU_PROBLEMS)
        output["benchmarks"]["mmlu"] = mmlu
        
        # Summary
        output["summary"] = {
            "gsm8k_accuracy": gsm8k["accuracy"],
            "mmlu_accuracy": mmlu["accuracy"],
            "total_tokens": llm.total_tokens,
            "total_time": round(llm.total_time, 2),
            "total_calls": llm.calls
        }

    # Output JSON
    print(json.dumps(output, indent=2))
    
    # Save to file
    Path("/workspace/data").mkdir(exist_ok=True)
    with open(f"/workspace/data/benchmark_{provider}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
