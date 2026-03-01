#!/usr/bin/env python3
"""
MMLU-Pro Benchmark: Cerebras models (50 questions each)
Uses Cerebras' ultra-fast inference
"""
import asyncio
import json
import time
import os
import re
import httpx
from pathlib import Path

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "csk-hpr4pjyd895p4ktvpnn436exx49rr925f6dptjvmee5ycrx8")
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"

MODELS = [
    "llama3.1-8b",
    "qwen-3-235b-a22b-instruct-2507",
    "zai-glm-4.7",
    "gpt-oss-120b",
]

NUM_QUESTIONS = 50

def format_question(q: dict) -> str:
    """Format MMLU question for LLM"""
    opts = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(q["options"])])
    return f"""Question: {q["question"]}

Options:
{opts}

Answer with ONLY the letter (A, B, C, etc.) of the correct option."""

def extract_answer(response: str) -> str:
    """Extract single letter answer from response"""
    patterns = [
        r'^([A-J])[.\s:)]',
        r'answer[:\s]+([A-J])\b',
        r'\b([A-J])\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, response.upper())
        if match:
            return match.group(1)
    return ""

async def query_model(client: httpx.AsyncClient, model: str, question: str, q_num: int, max_retries: int = 3) -> tuple[str, float, str, int]:
    """Query a model with retry logic"""
    for attempt in range(max_retries):
        start = time.time()
        try:
            resp = await client.post(
                CEREBRAS_URL,
                headers={"Authorization": f"Bearer {CEREBRAS_API_KEY}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": question}],
                    "max_tokens": 50,
                    "temperature": 0.0
                },
                timeout=60.0
            )
            elapsed = time.time() - start

            if resp.status_code == 429:
                wait = 3 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...", flush=True)
                await asyncio.sleep(wait)
                continue

            if resp.status_code != 200:
                print(f"    HTTP {resp.status_code}: {resp.text[:100]}", flush=True)
                return f"ERR{resp.status_code}", elapsed, "", 0

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens", 0)
            return extract_answer(content), elapsed, content[:100], tokens

        except httpx.TimeoutException:
            await asyncio.sleep(2)
            continue
        except Exception as e:
            return "ERR", time.time() - start, str(e)[:50], 0

    return "RETRY", 0, "", 0

async def benchmark_model(client: httpx.AsyncClient, model: str, questions: list) -> dict:
    """Run benchmark on a single model"""
    print(f"\n{'='*60}", flush=True)
    print(f"Testing: {model}", flush=True)
    print(f"{'='*60}", flush=True)

    correct = 0
    total_time = 0
    errors = 0

    for i, q in enumerate(questions[:NUM_QUESTIONS]):
        prompt = format_question(q)
        answer, elapsed, raw, tokens = await query_model(client, model, prompt, i)

        expected = q["answer"]
        is_correct = answer == expected
        if is_correct:
            correct += 1
        elif answer.startswith("ERR") or answer == "RETRY":
            errors += 1

        total_time += elapsed

        status = "✓" if is_correct else ("✗" if not answer.startswith("ERR") else "E")
        print(f"  [{i+1:2d}/{NUM_QUESTIONS}] {status} exp={expected} got={answer} ({elapsed:.2f}s)", flush=True)

        # Brief delay
        await asyncio.sleep(0.3)

    accuracy = correct / NUM_QUESTIONS * 100
    avg_time = total_time / NUM_QUESTIONS

    result = {
        "model": model,
        "provider": "cerebras",
        "correct": correct,
        "total": NUM_QUESTIONS,
        "accuracy": round(accuracy, 1),
        "total_time": round(total_time, 1),
        "avg_time": round(avg_time, 3),
        "errors": errors
    }

    print(f"\n  RESULT: {correct}/{NUM_QUESTIONS} ({accuracy:.1f}%) avg={avg_time:.3f}s/q", flush=True)
    return result

async def main():
    print("MMLU-Pro Benchmark - Cerebras Models", flush=True)
    print(f"Testing {len(MODELS)} models x {NUM_QUESTIONS} questions\n", flush=True)

    # Load questions
    with open("/workspace/data/mmlu_pro/sample_100.json") as f:
        questions = json.load(f)

    results = []

    # Load previous results if any
    results_file = Path("/workspace/data/mmlu_pro/cerebras_results.json")

    async with httpx.AsyncClient() as client:
        for model in MODELS:
            try:
                result = await benchmark_model(client, model, questions)
                results.append(result)

                # Save after each model
                with open(results_file, "w") as f:
                    json.dump(results, f, indent=2)

            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                results.append({"model": model, "provider": "cerebras", "error": str(e)})

    # Print summary
    print("\n" + "="*80, flush=True)
    print("MMLU-Pro Benchmark Results - Cerebras", flush=True)
    print("="*80, flush=True)
    print(f"{'Model':<45} {'Accuracy':>10} {'Avg Time':>10} {'Errors':>8}", flush=True)
    print("-"*80, flush=True)

    for r in sorted([x for x in results if 'accuracy' in x], key=lambda x: x["accuracy"], reverse=True):
        print(f"{r['model']:<45} {r['accuracy']:>9.1f}% {r['avg_time']:>9.3f}s {r['errors']:>8}", flush=True)

    print("="*80, flush=True)
    print(f"\nResults saved to {results_file}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
