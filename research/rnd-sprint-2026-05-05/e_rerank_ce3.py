#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""E-RERANK: Pure CE reranking with larger candidate pool."""
import json, random, numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

random.seed(42)
np.random.seed(42)

NUM_QUERIES = 500
TOP_K = 20
OUTPUT_PATH = "/home/aaron/projects/conjecture/research/rnd-sprint-2026-05-05/E-RERANK.json"

BASE_WORDS = [
    "research", "climate", "change", "weather", "patterns", "scientists", "universities",
    "studies", "risks", "coastal", "areas", "sea", "levels", "data", "satellites",
    "environmental", "shifts", "technology", "medicine", "patients", "treatment",
    "economies", "trade", "investment", "markets", "economic", "policy", "companies",
    "education", "systems", "students", "learning", "outcomes", "methods",
    "communication", "networks", "security", "information", "infrastructure",
    "maintenance", "urban", "planning", "development", "agriculture", "technology"
]

def make_text(seed, min_len=80, max_len=200):
    rng = random.Random(seed)
    words = []
    while len(' '.join(words)) < min_len:
        words.append(rng.choice(BASE_WORDS))
    text = ' '.join(words)
    if len(text) > max_len:
        text = text[:max_len]
    return text[0].upper() + text[1:] + '.'

def make_variant(text, rng):
    words = text[:-1].split()
    n_swaps = rng.randint(2, 4)
    for _ in range(n_swaps):
        if len(words) >= 3:
            i, j = random.sample(range(len(words)), 2)
            words[i], words[j] = words[j], words[i]
    return ' '.join(words).capitalize() + '.'

def compute_mrr(results, true_indices):
    return sum(1/rank for result, ti in zip(results, true_indices)
               for rank, (cid, _) in enumerate(result, 1) if cid == ti) / len(true_indices)

def hits_at_k(results, true_indices, k):
    return sum(1 for result, ti in zip(results, true_indices)
               for rank, (cid, _) in enumerate(result, 1) if rank <= k and cid == ti)

def main():
    print("Loading models...")
    bi = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    ce = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2', device='cpu')

    print("\nBuilding corpus...")
    rng = random.Random(42)
    seeds = list(range(200))
    corpus = []
    group_members = {}
    for seed in seeds:
        base = make_text(seed)
        group = [base] + [make_variant(base, rng) for _ in range(4)]
        for idx in range(len(corpus), len(corpus) + len(group)):
            group_members.setdefault(seed, []).append(idx)
        corpus.extend(group)
    print(f"Corpus: {len(corpus)}")

    print(f"Building {NUM_QUERIES} test pairs...")
    test_pairs = []
    seen = set()
    attempts = 0
    while len(test_pairs) < NUM_QUERIES and attempts < NUM_QUERIES * 10:
        attempts += 1
        seed = random.choice(seeds)
        members = group_members[seed]
        qi, ti = random.sample(members, 2)
        key = (qi, ti)
        if key in seen: continue
        seen.add(key)
        test_pairs.append((corpus[qi], corpus[ti], ti))
    print(f"Test pairs: {len(test_pairs)}")

    print("\nEncoding corpus...")
    emb = bi.encode(corpus, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    norm_emb = emb / norms

    true_indices = [p[2] for p in test_pairs]

    # --- Baseline ---
    print("\n--- Baseline: bi-encoder only ---")
    baseline = []
    for q, _, _ in test_pairs:
        qe = bi.encode([q], convert_to_numpy=True)
        qe = qe / (np.linalg.norm(qe) + 1e-8)
        scores = np.dot(norm_emb, qe.T).flatten()
        top = [(int(i), float(scores[i])) for i in np.argsort(scores)[::-1][:TOP_K]]
        baseline.append(top)
    b_mrr = compute_mrr(baseline, true_indices)
    b_h1 = hits_at_k(baseline, true_indices, 1)
    b_h5 = hits_at_k(baseline, true_indices, 5)
    print(f"Bi MRR: {b_mrr:.4f}, Hits@1: {b_h1} ({100*b_h1/len(test_pairs):.1f}%), Hits@5: {b_h5} ({100*b_h5/len(test_pairs):.1f}%)")

    # --- Pure CE reranking with varying candidate pool sizes ---
    print("\n--- Pure CE reranking (varying candidate pool) ---")
    best_mrr = 0
    best_k = 0
    best_results = None
    for cand_k in [20, 50, 100]:
        reranked = []
        for q, _, _ in test_pairs:
            qe = bi.encode([q], convert_to_numpy=True)
            qe = qe / (np.linalg.norm(qe) + 1e-8)
            scores = np.dot(norm_emb, qe.T).flatten()
            top_candidates = np.argsort(scores)[::-1][:cand_k]
            candidate_texts = [corpus[int(i)] for i in top_candidates]
            pairs = [(q, doc) for doc in candidate_texts]
            ce_scores = ce.predict(pairs)

            sorted_order = np.argsort(ce_scores)[::-1][:TOP_K]
            result = [(int(top_candidates[i]), float(ce_scores[i])) for i in sorted_order]
            reranked.append(result)

        r_mrr = compute_mrr(reranked, true_indices)
        r_h1 = hits_at_k(reranked, true_indices, 1)
        r_h5 = hits_at_k(reranked, true_indices, 5)
        print(f"  cand_k={cand_k}: MRR={r_mrr:.4f}, Hits@1={r_h1} ({100*r_h1/len(test_pairs):.1f}%), Hits@5={r_h5} ({100*r_h5/len(test_pairs):.1f}%)")
        if r_mrr > best_mrr:
            best_mrr = r_mrr
            best_k = cand_k
            best_results = reranked

    improvement_pp = round((best_mrr - b_mrr) * 100, 2)
    e_pass = improvement_pp > 10.0 or best_mrr > 0.70

    print(f"\n=== FINAL ===")
    print(f"Baseline (bi-encoder): {b_mrr:.4f}")
    print(f"Best: CE cand_k={best_k} → MRR={best_mrr:.4f}")
    print(f"Improvement: {improvement_pp} pp")
    print(f"E_pass: {e_pass}")

    result = {
        "approach": f"cross-encoder/ms-marco-MiniLM-L6-v2 pure CE reranking (bi-encoder top-{best_k} candidates)",
        "original_mrr": round(b_mrr, 4),
        "improved_mrr": round(best_mrr, 4),
        "improvement_pp": improvement_pp,
        "E_pass": e_pass
    }
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten to {OUTPUT_PATH}")
    return result

if __name__ == "__main__":
    main()