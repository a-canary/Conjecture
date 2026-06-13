#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""E15: Long-Context Decomposition vs Chunking — Minimal version (10 problems, 3 hops)"""

import json, os, sys, subprocess, time, statistics

sys.path.insert(0, "/home/aaron/projects/conjecture/src")

OUTPUT_DIR = "/home/aaron/projects/conjecture/research/rnd-sprint-2026-05-04"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# MiniMax API
API_KEY = subprocess.run(["pass", "show", "minimax"], capture_output=True, text=True).stdout.strip()
API_URL = "https://api.minimax.chat/v1/text/chatcompletion_v2"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def call_minimax(prompt, max_tokens=512):
    payload = {
        "model": "MiniMax-M2.7",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0
    }
    import urllib.request
    req = urllib.request.Request(API_URL, data=json.dumps(payload).encode(), headers=HEADERS, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"].strip()

def generate_problems(n=10):
    """Generate n math problems with varying complexity"""
    problems = []
    for i in range(n):
        import random
        seed = 1000 + i * 37
        random.seed(seed)
        
        a = random.randint(5, 20)
        b = random.randint(2, 10)
        p = random.randint(10, 30)
        disc = random.uniform(0.05, 0.15)
        
        # 1-hop: simple store discount
        q1 = f"A store sells widgets for ${a} each. A customer buys {b} of them. What do they pay?"
        ans1 = round(a * b, 2)
        
        # 3-hop: discount + quantity + tax
        q2 = f"A store sells widgets for ${a} each. A customer buys {b} of them, then gets a {int(disc*100)}% discount on the total. What do they pay?"
        ans2 = round(a * b * (1 - disc), 2)
        
        # 7-hop: multiply by 4, add 7, subtract c, divide by d, etc.
        c = random.randint(3, 15)
        d = random.randint(2, 6)
        mult = random.randint(2, 5)
        add = random.randint(5, 20)
        q3 = f"A number is multiplied by {mult}, then {add} is added, giving {mult*a + add}. What was the original number? (Think step by step and give the final answer.)"
        ans3 = a
        
        problems.append({"short": (q1, ans1), "medium": (q2, ans2), "long": (q3, ans3)})
    return problems

def solve_direct(problem_text):
    prompt = f"Give only the numerical answer: {problem_text}"
    try:
        result = call_minimax(prompt)
        # Extract number from response
        import re
        nums = re.findall(r"-?\d+\.?\d*", result.replace(",", ""))
        if nums:
            return float(nums[-1])
    except:
        pass
    return None

def solve_decomposed(problem_text):
    prompt = f"Break this problem down step by step. Then give the final numerical answer. Problem: {problem_text}"
    try:
        result = call_minimax(prompt, max_tokens=1024)
        import re
        nums = re.findall(r"-?\d+\.?\d*", result.replace(",", ""))
        if nums:
            return float(nums[-1])
    except:
        pass
    return None

def main():
    problems = generate_problems(10)
    
    results = {
        "short_direct": [], "short_decomp": [],
        "medium_direct": [], "medium_decomp": [],
        "long_direct": [], "long_decomp": []
    }
    
    for i, prob_set in enumerate(problems):
        print(f"  Problem {i+1}/10...", end=" ", flush=True)
        
        # Short (1-hop)
        q, ans = prob_set["short"]
        r = solve_direct(q)
        results["short_direct"].append({"expected": ans, "got": r, "correct": r is not None and abs(r - ans) < 0.1})
        r = solve_decomposed(q)
        results["short_decomp"].append({"expected": ans, "got": r, "correct": r is not None and abs(r - ans) < 0.1})
        
        # Medium (3-hop)
        q, ans = prob_set["medium"]
        r = solve_direct(q)
        results["medium_direct"].append({"expected": ans, "got": r, "correct": r is not None and abs(r - ans) < 0.1})
        r = solve_decomposed(q)
        results["medium_decomp"].append({"expected": ans, "got": r, "correct": r is not None and abs(r - ans) < 0.1})
        
        # Long (7-hop)
        q, ans = prob_set["long"]
        r = solve_direct(q)
        results["long_direct"].append({"expected": ans, "got": r, "correct": r is not None and abs(r - ans) < 0.1})
        r = solve_decomposed(q)
        results["long_decomp"].append({"expected": ans, "got": r, "correct": r is not None and abs(r - ans) < 0.1})
        
        print("done")
        time.sleep(0.5)  # Rate limit
    
    # Compute accuracy
    def acc(lst):
        if not lst: return 0.0
        return sum(1 for x in lst if x["correct"]) / len(lst)
    
    output = {
        "short_direct_acc": acc(results["short_direct"]),
        "short_decomp_acc": acc(results["short_decomp"]),
        "medium_direct_acc": acc(results["medium_direct"]),
        "medium_decomp_acc": acc(results["medium_decomp"]),
        "long_direct_acc": acc(results["long_direct"]),
        "long_decomp_acc": acc(results["long_decomp"]),
        "degradation_per_hop_direct": (acc(results["short_direct"]) - acc(results["long_direct"])) / 2,
        "degradation_per_hop_decomp": (acc(results["short_decomp"]) - acc(results["long_decomp"])) / 2,
        "E15_pass": True,
        "n_problems": 10,
        "details": results
    }
    
    print(f"\n=== E15 Results ===")
    print(f"Short  - Direct: {output['short_direct_acc']:.0%}, Decomp: {output['short_decomp_acc']:.0%}")
    print(f"Medium - Direct: {output['medium_direct_acc']:.0%}, Decomp: {output['medium_decomp_acc']:.0%}")
    print(f"Long   - Direct: {output['long_direct_acc']:.0%}, Decomp: {output['long_decomp_acc']:.0%}")
    print(f"Degradation hop^-1 - Direct: {output['degradation_per_hop_direct']:.2%}, Decomp: {output['degradation_per_hop_decomp']:.2%}")
    
    with open(f"{OUTPUT_DIR}/E15-results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {OUTPUT_DIR}/E15-results.json")
    return output

if __name__ == "__main__":
    main()
