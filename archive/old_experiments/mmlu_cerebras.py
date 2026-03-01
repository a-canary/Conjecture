#!/usr/bin/env python3
"""MMLU-Pro Benchmark on Cerebras"""
import asyncio
import json
import time
import os
import re
import httpx

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "csk-hpr4pjyd895p4ktvpnn436exx49rr925f6dptjvmee5ycrx8")
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"

MODELS = [
    "llama3.1-8b",
    "qwen-3-235b-a22b-instruct-2507",
    "gpt-oss-120b",
    "zai-glm-4.7",
]

NUM_QUESTIONS = 50

def format_question(q: dict) -> str:
    opts = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(q["options"])])
    return f"""Question: {q["question"]}

Options:
{opts}

Answer with ONLY the letter (A, B, C, etc.) of the correct option."""

def extract_answer(response: str) -> str:
    patterns = [r'^([A-J])[.\s:)]', r'answer[:\s]+([A-J])\b', r'\b([A-J])\b']
    for pattern in patterns:
        match = re.search(pattern, response.upper())
        if match:
            return match.group(1)
    return ""

async def query_model(client: httpx.AsyncClient, model: str, question: str) -> tuple[str, float]:
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
            timeout=30.0
        )
        elapsed = time.time() - start

        if resp.status_code != 200:
            return f"ERR{resp.status_code}", elapsed

        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return extract_answer(content), elapsed

    except Exception as e:
        return f"ERR:{str(e)[:20]}", time.time() - start

async def benchmark_model(client: httpx.AsyncClient, model: str, questions: list) -> dict:
    print(f"\n{'='*60}", flush=True)
    print(f"Testing: {model}", flush=True)
    print(f"{'='*60}", flush=True)

    correct = 0
    total_time = 0
    errors = 0

    for i, q in enumerate(questions[:NUM_QUESTIONS]):
        prompt = format_question(q)
        answer, elapsed = await query_model(client, model, prompt)

        expected = q["answer"]
        is_correct = answer == expected
        if is_correct:
            correct += 1
        elif answer.startswith("ERR"):
            errors += 1

        total_time += elapsed
        status = "✓" if is_correct else ("✗" if not answer.startswith("ERR") else "E")
        print(f"  [{i+1:2d}/{NUM_QUESTIONS}] {status} exp={expected} got={answer} ({elapsed:.3f}s)", flush=True)

    accuracy = correct / NUM_QUESTIONS * 100
    avg_time = total_time / NUM_QUESTIONS

    result = {
        "model": model,
        "correct": correct,
        "total": NUM_QUESTIONS,
        "accuracy": round(accuracy, 1),
        "total_time": round(total_time, 2),
        "avg_time": round(avg_time, 3),
        "errors": errors
    }

    print(f"\n  RESULT: {correct}/{NUM_QUESTIONS} ({accuracy:.1f}%) avg={avg_time:.3f}s/q", flush=True)
    return result

async def main():
    print("MMLU-Pro Benchmark on Cerebras", flush=True)
    print(f"Testing {len(MODELS)} models x {NUM_QUESTIONS} questions\n", flush=True)

    with open("/workspace/data/mmlu_pro/sample_100.json") as f:
        questions = json.load(f)

    results = []

    async with httpx.AsyncClient() as client:
        for model in MODELS:
            try:
                result = await benchmark_model(client, model, questions)
                results.append(result)

                # Save after each
                with open("/workspace/data/mmlu_pro/cerebras_results.json", "w") as f:
                    json.dump(results, f, indent=2)
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                results.append({"model": model, "error": str(e)})

    # Summary
    print("\n" + "="*80, flush=True)
    print("Cerebras MMLU-Pro Results", flush=True)
    print("="*80, flush=True)
    print(f"{'Model':<40} {'Accuracy':>10} {'Avg Time':>12}", flush=True)
    print("-"*80, flush=True)

    for r in sorted([x for x in results if 'accuracy' in x], key=lambda x: x["accuracy"], reverse=True):
        print(f"{r['model']:<40} {r['accuracy']:>9.1f}% {r['avg_time']:>11.3f}s", flush=True)

    print("="*80, flush=True)

if __name__ == "__main__":
    asyncio.run(main())
