#!/usr/bin/env python3
"""
MMLU-Pro Benchmark: Cerebras llama3.1-8b + Conjecture Framework

Tests whether claim-based reasoning improves llama3.1-8b accuracy.
Baseline: 26% @ 0.31s/q
Target: Show improvement through structured reasoning.
"""
import asyncio
import json
import time
import os
import re
import httpx
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "csk-hpr4pjyd895p4ktvpnn436exx49rr925f6dptjvmee5ycrx8")
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
MODEL = "llama3.1-8b"

NUM_QUESTIONS = 50


@dataclass
class Claim:
    """A claim in the reasoning process"""
    id: str
    content: str
    confidence: float
    claim_type: str  # "question", "option_analysis", "evidence", "synthesis"
    parent_id: Optional[str] = None


@dataclass
class ConjectureSession:
    """Session for claim-based reasoning"""
    root_claim: Claim
    sub_claims: List[Claim] = field(default_factory=list)
    final_answer: Optional[str] = None
    total_tokens: int = 0
    total_time: float = 0


class CerebrasLLM:
    """Cerebras LLM client"""

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def generate(self, prompt: str, max_tokens: int = 200) -> tuple[str, int, float]:
        """Generate response from Cerebras llama3.1-8b"""
        start = time.time()

        for attempt in range(3):
            try:
                resp = await self.client.post(
                    CEREBRAS_URL,
                    headers={"Authorization": f"Bearer {CEREBRAS_API_KEY}"},
                    json={
                        "model": MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0.1  # Low temp for reasoning
                    },
                    timeout=30.0
                )
                elapsed = time.time() - start

                if resp.status_code == 429:
                    await asyncio.sleep(2 * (attempt + 1))
                    continue

                if resp.status_code != 200:
                    return "", 0, elapsed

                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                tokens = data.get("usage", {}).get("total_tokens", 0)
                return content, tokens, elapsed

            except Exception as e:
                await asyncio.sleep(1)
                continue

        return "", 0, time.time() - start


class ConjectureMMLP:
    """Conjecture Framework adapted for MMLU-Pro"""

    def __init__(self, llm: CerebrasLLM):
        self.llm = llm

    async def process_question(self, q: dict) -> ConjectureSession:
        """Process MMLU question with claim-based reasoning"""

        # Phase 1: Create root claim
        root = Claim(
            id="root",
            content=q["question"],
            confidence=0.0,
            claim_type="question"
        )
        session = ConjectureSession(root_claim=root)

        # Phase 2: Analyze options - create sub-claims
        options = q["options"]
        option_letters = [chr(65 + i) for i in range(len(options))]

        # Step 1: Break down the question
        decompose_prompt = f"""Analyze this question step by step.

Question: {q["question"]}

Options:
{chr(10).join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))}

List 2-3 key facts or concepts needed to answer this question. Be brief."""

        decomp_resp, tokens1, time1 = await self.llm.generate(decompose_prompt, max_tokens=150)
        session.total_tokens += tokens1
        session.total_time += time1

        session.sub_claims.append(Claim(
            id="decomp",
            content=decomp_resp[:200] if decomp_resp else "Unable to decompose",
            confidence=0.7 if decomp_resp else 0.3,
            claim_type="evidence",
            parent_id="root"
        ))

        # Step 2: Evaluate each option
        eval_prompt = f"""Based on the analysis:
{decomp_resp[:300] if decomp_resp else "Consider each option carefully."}

Question: {q["question"]}

Options:
{chr(10).join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))}

Rate each option's likelihood (High/Medium/Low):"""

        eval_resp, tokens2, time2 = await self.llm.generate(eval_prompt, max_tokens=200)
        session.total_tokens += tokens2
        session.total_time += time2

        session.sub_claims.append(Claim(
            id="eval",
            content=eval_resp[:200] if eval_resp else "Unable to evaluate",
            confidence=0.8 if eval_resp else 0.4,
            claim_type="option_analysis",
            parent_id="root"
        ))

        # Phase 3: Synthesize final answer
        synth_prompt = f"""Based on this reasoning:
{decomp_resp[:200] if decomp_resp else ""}
{eval_resp[:200] if eval_resp else ""}

Question: {q["question"]}

Options:
{chr(10).join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))}

What is the correct answer? Reply with ONLY the letter (A, B, C, etc.)."""

        synth_resp, tokens3, time3 = await self.llm.generate(synth_prompt, max_tokens=10)
        session.total_tokens += tokens3
        session.total_time += time3

        # Extract answer
        session.final_answer = self._extract_answer(synth_resp)

        # Update root confidence
        validated_claims = [c for c in session.sub_claims if c.confidence >= 0.7]
        session.root_claim.confidence = min(0.9, 0.5 + 0.2 * len(validated_claims))

        return session

    def _extract_answer(self, response: str) -> str:
        """Extract single letter answer"""
        if not response:
            return ""
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


def format_question_bare(q: dict) -> str:
    """Format question for bare LLM (single-shot)"""
    opts = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(q["options"])])
    return f"""Question: {q["question"]}

Options:
{opts}

Answer with ONLY the letter (A, B, C, etc.) of the correct option."""


async def benchmark_bare(client: httpx.AsyncClient, questions: list) -> dict:
    """Run bare llama3.1-8b benchmark"""
    print(f"\n{'='*60}")
    print("Testing: Bare llama3.1-8b (baseline)")
    print(f"{'='*60}")

    correct = 0
    total_time = 0
    total_tokens = 0
    errors = 0

    for i, q in enumerate(questions[:NUM_QUESTIONS]):
        prompt = format_question_bare(q)

        start = time.time()
        try:
            resp = await client.post(
                CEREBRAS_URL,
                headers={"Authorization": f"Bearer {CEREBRAS_API_KEY}"},
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 50,
                    "temperature": 0.0
                },
                timeout=30.0
            )
            elapsed = time.time() - start

            if resp.status_code == 429:
                await asyncio.sleep(3)
                errors += 1
                print(f"  [{i+1:2d}/{NUM_QUESTIONS}] E (rate limit)", flush=True)
                continue

            if resp.status_code != 200:
                errors += 1
                print(f"  [{i+1:2d}/{NUM_QUESTIONS}] E (HTTP {resp.status_code})", flush=True)
                continue

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens", 0)

            # Extract answer
            answer = ""
            for pattern in [r'^([A-J])[.\s:)]', r'answer[:\s]+([A-J])\b', r'\b([A-J])\b']:
                match = re.search(pattern, content.upper())
                if match:
                    answer = match.group(1)
                    break

            expected = q["answer"]
            is_correct = answer == expected
            if is_correct:
                correct += 1

            total_time += elapsed
            total_tokens += tokens

            status = "✓" if is_correct else "✗"
            print(f"  [{i+1:2d}/{NUM_QUESTIONS}] {status} exp={expected} got={answer} ({elapsed:.2f}s)", flush=True)

            await asyncio.sleep(0.3)

        except Exception as e:
            errors += 1
            print(f"  [{i+1:2d}/{NUM_QUESTIONS}] E ({str(e)[:20]})", flush=True)

    accuracy = correct / NUM_QUESTIONS * 100
    avg_time = total_time / NUM_QUESTIONS if NUM_QUESTIONS > 0 else 0

    print(f"\n  RESULT: {correct}/{NUM_QUESTIONS} ({accuracy:.1f}%) avg={avg_time:.3f}s/q")

    return {
        "model": f"{MODEL} (bare)",
        "correct": correct,
        "total": NUM_QUESTIONS,
        "accuracy": round(accuracy, 1),
        "avg_time": round(avg_time, 3),
        "total_tokens": total_tokens,
        "errors": errors
    }


async def benchmark_conjecture(client: httpx.AsyncClient, questions: list) -> dict:
    """Run llama3.1-8b + Conjecture benchmark"""
    print(f"\n{'='*60}")
    print("Testing: llama3.1-8b + Conjecture Framework")
    print(f"{'='*60}")

    llm = CerebrasLLM(client)
    framework = ConjectureMMLP(llm)

    correct = 0
    total_time = 0
    total_tokens = 0
    total_claims = 0
    errors = 0

    for i, q in enumerate(questions[:NUM_QUESTIONS]):
        try:
            session = await framework.process_question(q)

            answer = session.final_answer or ""
            expected = q["answer"]
            is_correct = answer == expected

            if is_correct:
                correct += 1
            elif not answer:
                errors += 1

            total_time += session.total_time
            total_tokens += session.total_tokens
            total_claims += len(session.sub_claims)

            status = "✓" if is_correct else ("E" if not answer else "✗")
            claims = len(session.sub_claims)
            conf = session.root_claim.confidence
            print(f"  [{i+1:2d}/{NUM_QUESTIONS}] {status} exp={expected} got={answer} ({session.total_time:.2f}s) claims={claims} conf={conf:.2f}", flush=True)

            await asyncio.sleep(0.3)

        except Exception as e:
            errors += 1
            print(f"  [{i+1:2d}/{NUM_QUESTIONS}] E ({str(e)[:30]})", flush=True)

    accuracy = correct / NUM_QUESTIONS * 100
    avg_time = total_time / NUM_QUESTIONS if NUM_QUESTIONS > 0 else 0
    avg_claims = total_claims / NUM_QUESTIONS if NUM_QUESTIONS > 0 else 0

    print(f"\n  RESULT: {correct}/{NUM_QUESTIONS} ({accuracy:.1f}%) avg={avg_time:.3f}s/q claims={avg_claims:.1f}/q")

    return {
        "model": f"{MODEL} + Conjecture",
        "correct": correct,
        "total": NUM_QUESTIONS,
        "accuracy": round(accuracy, 1),
        "avg_time": round(avg_time, 3),
        "total_tokens": total_tokens,
        "avg_claims": round(avg_claims, 1),
        "errors": errors
    }


async def main():
    print("MMLU-Pro Benchmark: Cerebras + Conjecture")
    print(f"Model: {MODEL}")
    print(f"Questions: {NUM_QUESTIONS}\n")

    # Load questions
    with open("/workspace/data/mmlu_pro/sample_100.json") as f:
        questions = json.load(f)

    results = []

    async with httpx.AsyncClient() as client:
        # Run bare baseline
        bare_result = await benchmark_bare(client, questions)
        results.append(bare_result)

        # Brief pause between benchmarks
        await asyncio.sleep(2)

        # Run Conjecture-enhanced
        conj_result = await benchmark_conjecture(client, questions)
        results.append(conj_result)

    # Summary
    print("\n" + "="*80)
    print("COMPARISON: Bare vs Conjecture")
    print("="*80)

    bare = results[0]
    conj = results[1]

    improvement = conj["accuracy"] - bare["accuracy"]
    speedup = bare["avg_time"] / conj["avg_time"] if conj["avg_time"] > 0 else 0

    print(f"{'Configuration':<30} {'Accuracy':>10} {'Avg Time':>10} {'Tokens':>10}")
    print("-"*80)
    print(f"{'Bare llama3.1-8b':<30} {bare['accuracy']:>9.1f}% {bare['avg_time']:>9.3f}s {bare['total_tokens']:>10}")
    print(f"{'llama3.1-8b + Conjecture':<30} {conj['accuracy']:>9.1f}% {conj['avg_time']:>9.3f}s {conj['total_tokens']:>10}")
    print("-"*80)
    print(f"{'Improvement':<30} {improvement:>+9.1f}pp {speedup:>9.2f}x")
    print("="*80)

    # Save results
    results_file = Path("/workspace/data/mmlu_pro/conjecture_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "bare": bare,
            "conjecture": conj,
            "improvement_pp": round(improvement, 1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    asyncio.run(main())
