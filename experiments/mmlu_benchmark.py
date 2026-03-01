#!/usr/bin/env python3
"""
MMLU-Pro Benchmark: Compare 7 models on 100 questions
With retry logic for 429 errors
"""
import asyncio
import json
import time
import os
import re
import httpx
from pathlib import Path

CHUTES_API_KEY = os.getenv("CHUTES_API_KEY", "")
CHUTES_URL = "https://llm.chutes.ai/v1/chat/completions"

MODELS = [
    "deepseek-ai/DeepSeek-V3-0324",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    "deepseek-ai/DeepSeek-R1-0528",
    "deepseek-ai/DeepSeek-V3",
]

def format_question(q: dict) -> str:
    """Format MMLU question for LLM"""
    opts = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(q["options"])])
    return f"""Question: {q["question"]}

Options:
{opts}

Answer with ONLY the letter (A, B, C, etc.) of the correct option."""

def extract_answer(response: str) -> str:
    """Extract single letter answer from response"""
    # Look for standalone letter at start or after common patterns
    patterns = [
        r'^([A-J])[.\s:)]',  # Letter at start followed by punctuation
        r'answer[:\s]+([A-J])\b',  # "answer: X"
        r'\b([A-J])\b',  # Any standalone letter
    ]
    for pattern in patterns:
        match = re.search(pattern, response.upper())
        if match:
            return match.group(1)
    return ""

async def query_model(client: httpx.AsyncClient, model: str, question: str, max_retries: int = 5) -> tuple[str, float, int]:
    """Query a model with retry logic for 429 errors"""
    for attempt in range(max_retries):
        start = time.time()
        try:
            resp = await client.post(
                CHUTES_URL,
                headers={"Authorization": f"Bearer {CHUTES_API_KEY}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": question}],
                    "max_tokens": 100,
                    "temperature": 0.0
                },
                timeout=60.0
            )
            elapsed = time.time() - start

            if resp.status_code == 429:
                wait_time = min(30, 5 * (attempt + 1))  # 5, 10, 15, 20, 25 seconds
                print(f"    [429 retry {attempt+1}/{max_retries}, waiting {wait_time}s]")
                await asyncio.sleep(wait_time)
                continue

            if resp.status_code != 200:
                return f"ERROR:{resp.status_code}", elapsed, 0

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens", 0)
            return extract_answer(content), elapsed, tokens

        except httpx.TimeoutException:
            print(f"    [Timeout retry {attempt+1}/{max_retries}]")
            await asyncio.sleep(5)
            continue
        except Exception as e:
            return f"ERROR:{str(e)[:30]}", time.time() - start, 0

    return "ERROR:max_retries", 0, 0

async def benchmark_model(client: httpx.AsyncClient, model: str, questions: list) -> dict:
    """Run benchmark on a single model"""
    print(f"\n{'='*60}", flush=True)
    print(f"Testing: {model}", flush=True)
    print(f"{'='*60}", flush=True)

    correct = 0
    total_time = 0
    total_tokens = 0
    errors = 0

    for i, q in enumerate(questions):
        prompt = format_question(q)
        answer, elapsed, tokens = await query_model(client, model, prompt)

        expected = q["answer"]
        is_correct = answer == expected
        if is_correct:
            correct += 1
        elif answer.startswith("ERROR"):
            errors += 1

        total_time += elapsed
        total_tokens += tokens

        status = "O" if is_correct else ("X" if not answer.startswith("ERROR") else "E")
        print(f"  [{i+1:3d}/100] {status} exp={expected} got={answer} ({elapsed:.1f}s)", flush=True)

        # Rate limiting between requests
        await asyncio.sleep(1.0)

    accuracy = correct / len(questions) * 100 if questions else 0
    avg_time = total_time / len(questions) if questions else 0

    result = {
        "model": model,
        "correct": correct,
        "total": len(questions),
        "accuracy": accuracy,
        "total_time": total_time,
        "avg_time": avg_time,
        "total_tokens": total_tokens,
        "errors": errors
    }

    print(f"\n  RESULT: {correct}/{len(questions)} ({accuracy:.1f}%) in {total_time:.1f}s ({avg_time:.2f}s/q)", flush=True)
    return result

async def main():
    print("MMLU-Pro Benchmark Starting...", flush=True)

    # Load questions
    with open("/workspace/data/mmlu_pro/sample_100.json") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} MMLU-Pro questions", flush=True)
    print(f"Testing {len(MODELS)} models", flush=True)

    results = []

    async with httpx.AsyncClient() as client:
        for model in MODELS:
            result = await benchmark_model(client, model, questions)
            results.append(result)

            # Save intermediate results
            with open("/workspace/data/mmlu_pro/results.json", "w") as f:
                json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "="*80, flush=True)
    print("MMLU-Pro Benchmark Results (100 questions)", flush=True)
    print("="*80, flush=True)
    print(f"{'Model':<45} {'Accuracy':>10} {'Avg Time':>10} {'Tokens':>10}", flush=True)
    print("-"*80, flush=True)

    for r in sorted(results, key=lambda x: x["accuracy"], reverse=True):
        model_short = r["model"].split("/")[-1][:40]
        print(f"{model_short:<45} {r['accuracy']:>9.1f}% {r['avg_time']:>9.2f}s {r['total_tokens']:>10}", flush=True)

    print("="*80, flush=True)

    # Save final results
    with open("/workspace/data/mmlu_pro/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to /workspace/data/mmlu_pro/results.json", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
