#!/usr/bin/env python3
"""
MMLU-Pro Benchmark (Short): Compare 7 models on 20 questions
Saves results immediately after each model
"""
import asyncio
import json
import time
import os
import re
import sys
import httpx

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

NUM_QUESTIONS = 20  # Reduced for faster results

def format_question(q: dict) -> str:
    """Format MMLU question for LLM"""
    opts = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(q["options"])])
    return f"""Question: {q["question"]}

Options:
{opts}

Respond with ONLY a single letter (A, B, C, etc.) for your answer. No explanation."""

def extract_answer(response: str) -> str:
    """Extract single letter answer from response"""
    response = response.strip().upper()
    # First char if it's a letter
    if response and response[0] in "ABCDEFGHIJ":
        return response[0]
    # Look for pattern "Answer: X" or similar
    match = re.search(r'(?:answer|option)[:\s]*([A-J])', response, re.I)
    if match:
        return match.group(1).upper()
    # Any standalone letter
    match = re.search(r'\b([A-J])\b', response)
    if match:
        return match.group(1)
    return ""

async def query_model(client: httpx.AsyncClient, model: str, question: str) -> tuple[str, float, int, str]:
    """Query a model with retry logic. Returns (answer, time, tokens, raw_response)"""
    for attempt in range(5):
        start = time.time()
        try:
            resp = await client.post(
                CHUTES_URL,
                headers={"Authorization": f"Bearer {CHUTES_API_KEY}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": question}],
                    "max_tokens": 50,
                    "temperature": 0.0
                },
                timeout=45.0
            )
            elapsed = time.time() - start

            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"    [429, wait {wait}s]", flush=True)
                await asyncio.sleep(wait)
                continue

            if resp.status_code != 200:
                return f"ERR{resp.status_code}", elapsed, 0, resp.text[:100]

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens", 0)
            answer = extract_answer(content)
            return answer, elapsed, tokens, content[:100]

        except Exception as e:
            if attempt < 4:
                await asyncio.sleep(3)
                continue
            return "ERR", time.time() - start, 0, str(e)[:100]

    return "ERR", 0, 0, "max_retries"

async def benchmark_model(client: httpx.AsyncClient, model: str, questions: list) -> dict:
    """Run benchmark on a single model"""
    print(f"\n{'='*50}", flush=True)
    print(f"Model: {model}", flush=True)
    print(f"{'='*50}", flush=True)

    correct = 0
    total_time = 0
    total_tokens = 0
    errors = 0
    details = []

    for i, q in enumerate(questions):
        prompt = format_question(q)
        answer, elapsed, tokens, raw = await query_model(client, model, prompt)

        expected = q["answer"]
        is_correct = answer == expected
        if is_correct:
            correct += 1
        elif answer.startswith("ERR"):
            errors += 1

        total_time += elapsed
        total_tokens += tokens

        icon = "O" if is_correct else ("X" if not answer.startswith("ERR") else "E")
        print(f"[{i+1:2d}/{NUM_QUESTIONS}] {icon} exp={expected} got={answer} ({elapsed:.1f}s)", flush=True)

        details.append({
            "q": i+1,
            "expected": expected,
            "got": answer,
            "correct": is_correct,
            "time": elapsed
        })

        await asyncio.sleep(1.5)  # Rate limit

    accuracy = correct / len(questions) * 100
    avg_time = total_time / len(questions)

    result = {
        "model": model,
        "correct": correct,
        "total": len(questions),
        "accuracy": round(accuracy, 1),
        "total_time": round(total_time, 1),
        "avg_time": round(avg_time, 2),
        "total_tokens": total_tokens,
        "errors": errors,
        "details": details
    }

    print(f"\nRESULT: {correct}/{len(questions)} = {accuracy:.1f}%", flush=True)
    print(f"Time: {total_time:.1f}s total, {avg_time:.2f}s/q", flush=True)

    return result

async def main():
    print("="*60, flush=True)
    print("MMLU-Pro Benchmark (Short Version)", flush=True)
    print(f"Questions: {NUM_QUESTIONS}, Models: {len(MODELS)}", flush=True)
    print("="*60, flush=True)

    # Load and sample questions
    with open("/workspace/data/mmlu_pro/sample_100.json") as f:
        all_questions = json.load(f)
    questions = all_questions[:NUM_QUESTIONS]
    print(f"Using first {NUM_QUESTIONS} questions", flush=True)

    results = []

    async with httpx.AsyncClient() as client:
        for model in MODELS:
            try:
                result = await benchmark_model(client, model, questions)
                results.append(result)

                # Save after each model
                with open("/workspace/data/mmlu_pro/results_short.json", "w") as f:
                    json.dump(results, f, indent=2)
                print(f"[Saved results: {len(results)} models]", flush=True)

            except Exception as e:
                print(f"ERROR with {model}: {e}", flush=True)
                results.append({"model": model, "error": str(e)})

    # Final summary
    print("\n" + "="*70, flush=True)
    print("MMLU-Pro Benchmark Results", flush=True)
    print("="*70, flush=True)
    print(f"{'Model':<40} {'Acc':>8} {'Avg/Q':>8} {'Tokens':>8}", flush=True)
    print("-"*70, flush=True)

    valid_results = [r for r in results if "error" not in r]
    for r in sorted(valid_results, key=lambda x: x["accuracy"], reverse=True):
        name = r["model"].split("/")[-1][:35]
        print(f"{name:<40} {r['accuracy']:>7.1f}% {r['avg_time']:>7.2f}s {r['total_tokens']:>8}", flush=True)

    print("="*70, flush=True)
    print(f"Results saved to /workspace/data/mmlu_pro/results_short.json", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
