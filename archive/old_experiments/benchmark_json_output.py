#!/usr/bin/env python3
"""
Benchmark Suite - JSON LLM Output

Request structured JSON responses from LLM for reliable answer extraction.
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
from typing import List, Dict

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "csk-hpr4pjyd895p4ktvpnn436exx49rr925f6dptjvmee5ycrx8")
CHUTES_API_KEY = os.getenv("CHUTES_API_KEY", "")

PROVIDERS = {
    "cerebras": {"url": "https://api.cerebras.ai/v1/chat/completions", "key": CEREBRAS_API_KEY, "model": "llama3.1-8b"},
    "chutes": {"url": "https://llm.chutes.ai/v1/chat/completions", "key": CHUTES_API_KEY, "model": "deepseek-ai/DeepSeek-V3"}
}

GSM8K_FEWSHOT_JSON = """Solve math problems. Respond with JSON only: {"reasoning": "...", "answer": <number>}

Q: There are 15 trees. Workers plant more. Now 21 trees. How many planted?
{"reasoning": "21 - 15 = 6", "answer": 6}

Q: 3 cars in parking lot, 2 more arrive. Total cars?
{"reasoning": "3 + 2 = 5", "answer": 5}

Q: Leah had 32 chocolates, sister had 42. They ate 35. How many left?
{"reasoning": "32 + 42 = 74, then 74 - 35 = 39", "answer": 39}

Q: Jason had 20 lollipops, gave some to Denny, now has 12. How many given?
{"reasoning": "20 - 12 = 8", "answer": 8}

"""

MMLU_FEWSHOT_JSON = """Answer with JSON only: {"reasoning": "...", "answer": "X"} where X is A/B/C/D.

Q: What is the derivative of x^2?
A) x  B) 2x  C) x^2  D) 2
{"reasoning": "d/dx(x^2) = 2x", "answer": "B"}

Q: Chemical symbol for gold?
A) Ag  B) Au  C) Fe  D) Cu
{"reasoning": "Gold is Au from Latin aurum", "answer": "B"}

Q: All mammals are warm-blooded. Dogs are mammals. Therefore:
A) cold-blooded  B) warm-blooded  C) some warm-blooded  D) cannot determine
{"reasoning": "Dogs are mammals, mammals are warm-blooded", "answer": "B"}

"""

GSM8K_PROBLEMS = [
    {"q": "Janet's ducks lay 16 eggs/day. She eats 3 and bakes 4. Sells rest at $2 each. Daily earnings?", "a": 18},
    {"q": "Robe takes 2 bolts blue fiber, half that white. Total bolts?", "a": 3},
    {"q": "Josh buys house $80k, repairs $50k. Value increased 150%. Profit?", "a": 70000},
    {"q": "James writes 3-page letter to 2 friends, twice weekly. Pages per year?", "a": 624},
    {"q": "20 chickens, 3 cups each daily, 3 meals. Morning 15 cups, afternoon 25. Final meal?", "a": 20},
    {"q": "Glass $5, every 2nd glass 60% price. Cost for 16 glasses?", "a": 64},
    {"q": "Toulouse 2x sheep as Charleston. Charleston 4x as Seattle. Seattle has 20. Total?", "a": 260},
    {"q": "Download 200GB at 2GB/min. At 40%, restart takes 20min, then restart download. Total time?", "a": 160},
    {"q": "Drive 3hr at 60mph, turn around. 2hr traffic, 30min at 30mph, 1.5hr at 80mph. Distance from home?", "a": 45},
    {"q": "Earn $10/hr first 40hrs, 1.2x overtime. Worked 45hrs. Total?", "a": 460},
]

MMLU_PROBLEMS = [
    {"q": "Integral of 2x?", "opts": "A) x  B) x^2  C) 2x^2  D) x^2+C", "a": "D"},
    {"q": "log10(100)?", "opts": "A) 1  B) 2  C) 10  D) 100", "a": "B"},
    {"q": "2^10?", "opts": "A) 512  B) 1024  C) 2048  D) 4096", "a": "B"},
    {"q": "Boiling point of water?", "opts": "A) 90C  B) 100C  C) 110C  D) 120C", "a": "B"},
    {"q": "Largest planet?", "opts": "A) Saturn  B) Jupiter  C) Uranus  D) Neptune", "a": "B"},
    {"q": "Powerhouse of cell?", "opts": "A) Nucleus  B) Ribosome  C) Mitochondria  D) Golgi", "a": "C"},
    {"q": "All A are B. All B are C. All A are C?", "opts": "A) Yes  B) No  C) Maybe  D) Cannot say", "a": "A"},
    {"q": "If not Q then not P. Q false. P?", "opts": "A) True  B) False  C) Unknown  D) Both", "a": "B"},
    {"q": "Some X are Y. Some Y are Z. Some X are Z?", "opts": "A) Yes  B) No  C) Cannot determine  D) All", "a": "C"},
    {"q": "Speed of light?", "opts": "A) 300km/s  B) 300,000km/s  C) 3,000,000km/s  D) 30,000km/s", "a": "B"},
]


class LLM:
    def __init__(self, provider, client):
        self.cfg = PROVIDERS[provider]
        self.client = client
        self.stats = {"tokens": 0, "calls": 0, "errors": 0}

    async def ask(self, prompt, max_tokens=150):
        for attempt in range(3):
            try:
                r = await self.client.post(
                    self.cfg["url"],
                    headers={"Authorization": f"Bearer {self.cfg['key']}"},
                    json={"model": self.cfg["model"], "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": max_tokens, "temperature": 0.0},
                    timeout=120.0
                )
                if r.status_code == 429:
                    return {"error": "rate_limited"}
                if r.status_code != 200:
                    return {"error": f"http_{r.status_code}"}
                data = r.json()
                self.stats["tokens"] += data.get("usage", {}).get("total_tokens", 0)
                self.stats["calls"] += 1
                return {"content": data["choices"][0]["message"]["content"]}
            except Exception as e:
                if attempt == 2:
                    self.stats["errors"] += 1
                    return {"error": str(e)[:30]}
                await asyncio.sleep(2)
        return {"error": "max_retry"}


def parse_json_answer(text, typ="num"):
    if not text:
        return ""
    # Extract JSON
    if "```" in text:
        m = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if m: text = m.group(1)
    m = re.search(r'\{[^{}]*"answer"[^{}]*\}', text)
    if m: text = m.group(0)
    try:
        ans = json.loads(text).get("answer", "")
        return str(ans).upper() if typ == "letter" else str(ans)
    except:
        m = re.search(r'"answer"\s*:\s*"?([^",}\s]+)', text)
        if m:
            return m.group(1).upper() if typ == "letter" else m.group(1)
        if typ == "num":
            nums = re.findall(r'-?\d+', text)
            return nums[-1] if nums else ""
        return ""


async def bench_gsm8k(llm, problems):
    results = []
    correct = 0
    for i, p in enumerate(problems):
        r = await llm.ask(GSM8K_FEWSHOT_JSON + f"Q: {p['q']}\n", 150)
        if "error" in r:
            results.append({"id": i, "error": r["error"], "expected": p["a"]})
            continue
        ans = parse_json_answer(r["content"], "num")
        try:
            ok = abs(float(ans) - p["a"]) < 1
        except:
            ok = False
        if ok: correct += 1
        results.append({"id": i, "exp": p["a"], "got": ans, "ok": ok, "raw": r["content"][:80]})
        await asyncio.sleep(0.3)
    return {"bench": "gsm8k", "n": len(problems), "correct": correct, "acc": round(100*correct/len(problems), 1), "results": results}


async def bench_mmlu(llm, problems):
    results = []
    correct = 0
    for i, p in enumerate(problems):
        r = await llm.ask(MMLU_FEWSHOT_JSON + f"Q: {p['q']}\n{p['opts']}\n", 100)
        if "error" in r:
            results.append({"id": i, "error": r["error"], "expected": p["a"]})
            continue
        ans = parse_json_answer(r["content"], "letter")
        ok = ans == p["a"]
        if ok: correct += 1
        results.append({"id": i, "exp": p["a"], "got": ans, "ok": ok, "raw": r["content"][:80]})
        await asyncio.sleep(0.3)
    return {"bench": "mmlu", "n": len(problems), "correct": correct, "acc": round(100*correct/len(problems), 1), "results": results}


async def main():
    prov = sys.argv[1] if len(sys.argv) > 1 else "chutes"
    if prov not in PROVIDERS or not PROVIDERS[prov]["key"]:
        print(json.dumps({"error": f"Invalid provider or no key: {prov}"}))
        return

    out = {"ts": datetime.now().isoformat(), "provider": prov, "model": PROVIDERS[prov]["model"], "method": "json_output"}

    async with httpx.AsyncClient() as client:
        llm = LLM(prov, client)
        out["gsm8k"] = await bench_gsm8k(llm, GSM8K_PROBLEMS)
        await asyncio.sleep(1)
        out["mmlu"] = await bench_mmlu(llm, MMLU_PROBLEMS)
        out["stats"] = llm.stats
        out["summary"] = {"gsm8k": out["gsm8k"]["acc"], "mmlu": out["mmlu"]["acc"]}

    print(json.dumps(out, indent=2))
    Path("/workspace/data").mkdir(exist_ok=True)
    with open(f"/workspace/data/bench_jsonout_{prov}.json", "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
