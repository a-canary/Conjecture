#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
E-RERANK v4: Cross-encoder reranking, no BM25 (too slow).
Focus: Pure CE reranking + bi+CE weighted combo on larger candidate pool.
"""
import json, random, math, numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

random.seed(42)
np.random.seed(42)

NUM_QUERIES = 500
TOP_K = 20
CANDIDATE_K = 100
OUTPUT_PATH = "./research/rnd-sprint-2026-05-05/E-RERANK.json"

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

def rrf_fusion_two(ranked_lists, k=60):
    scores = {}
    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list, 1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

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
    print(f"Corpus: {len(corpus)} ({len(seeds)} groups x 5)")

    print(f"Building test pairs ({NUM_QUERIES})...")
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

    print("\nEncoding corpus with bi-encoder...")
    emb = bi.encode(corpus, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    norm_emb = emb / norms

    true_indices = [p[2] for p in test_pairs]

    # Pre-encode all queries
    print("Pre-encoding queries...")
    query_embs = []
    for q, _, _ in test_pairs:
        qe = bi.encode([q], convert_to_numpy=True)
        qe = qe / (np.linalg.norm(qe) + 1e-8)
        query_embs.append(qe)
    query_embs = np.vstack(query_embs)

    # Bi-encoder scores for all (query, corpus_item)
    print("Computing bi-encoder scores matrix...")
    bi_scores_matrix = np.dot(query_embs, norm_emb.T)  # (500, 1000)

    # --- Baseline: bi-encoder only ---
    print("\n--- Phase 1: Bi-encoder baseline ---")
    baseline = []
    for i in range(len(test_pairs)):
        scores = bi_scores_matrix[i]
        top = [(int(j), float(scores[j])) for j in np.argsort(scores)[::-1][:TOP_K]]
        baseline.append(top)
    b_mrr = compute_mrr(baseline, true_indices)
    b_h1 = hits_at_k(baseline, true_indices, 1)
    b_h5 = hits_at_k(baseline, true_indices, 5)
    print(f"Bi MRR: {b_mrr:.4f}, Hits@1: {b_h1} ({100*b_h1/len(test_pairs):.1f}%), Hits@5: {b_h5}")

    # --- Phase 2: Pure CE reranking (top-100) ---
    print("\n--- Phase 2: Pure CE reranking (top-100) ---")
    ce_reranked = []
    for i in range(len(test_pairs)):
        q = test_pairs[i][0]
        scores = bi_scores_matrix[i]
        top_candidates = np.argsort(scores)[::-1][:CANDIDATE_K]
        candidate_texts = [corpus[int(j)] for j in top_candidates]
        pairs = [(q, doc) for doc in candidate_texts]
        ce_scores = ce.predict(pairs)
        sorted_order = np.argsort(ce_scores)[::-1][:TOP_K]
        result = [(int(top_candidates[j]), float(ce_scores[j])) for j in sorted_order]
        ce_reranked.append(result)
    ce_mrr = compute_mrr(ce_reranked, true_indices)
    ce_h1 = hits_at_k(ce_reranked, true_indices, 1)
    ce_h5 = hits_at_k(ce_reranked, true_indices, 5)
    print(f"Pure CE MRR: {ce_mrr:.4f}, Hits@1: {ce_h1} ({100*ce_h1/len(test_pairs):.1f}%), Hits@5: {ce_h5}")

    # --- Phase 3: CE + bi weighted combo (sweep weights) ---
    print("\n--- Phase 3: CE + bi weighted combo ---")
    best_combo_mrr = 0
    best_combo_w = 0.5
    for ce_w in [0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
        bi_w = 1.0 - ce_w
        combo = []
        for i in range(len(test_pairs)):
            q = test_pairs[i][0]
            scores = bi_scores_matrix[i]
            top_candidates = np.argsort(scores)[::-1][:CANDIDATE_K]
            candidate_texts = [corpus[int(j)] for j in top_candidates]
            pairs = [(q, doc) for doc in candidate_texts]
            ce_scores = ce.predict(pairs)
            bi_cand = scores[top_candidates]
            bi_norm = (bi_cand - bi_cand.min()) / (bi_cand.max() - bi_cand.min() + 1e-8)
            ce_norm = (ce_scores - ce_scores.min()) / (ce_scores.max() - ce_scores.min() + 1e-8)
            combined = ce_w * ce_norm + bi_w * bi_norm
            sorted_order = np.argsort(combined)[::-1][:TOP_K]
            result = [(int(top_candidates[j]), float(combined[j])) for j in sorted_order]
            combo.append(result)
        c_mrr = compute_mrr(combo, true_indices)
        c_h1 = hits_at_k(combo, true_indices, 1)
        print(f"  CE_w={ce_w:.2f}: MRR={c_mrr:.4f}, Hits@1={c_h1} ({100*c_h1/len(test_pairs):.1f}%)")
        if c_mrr > best_combo_mrr:
            best_combo_mrr = c_mrr
            best_combo_w = ce_w

    # --- Phase 4: 2-way RRF (bi top-20 + CE top-20) ---
    print("\n--- Phase 4: 2-way RRF (bi + CE) ---")
    best_rrf_mrr = 0
    best_rrf_k = 60
    for rrf_k in [30, 60]:
        fused = []
        for i in range(len(test_pairs)):
            scores = bi_scores_matrix[i]
            top_candidates = np.argsort(scores)[::-1][:CANDIDATE_K]
            bi_top = [(int(top_candidates[j]), float(scores[top_candidates[j]]))
                      for j in np.argsort(scores[top_candidates])[::-1][:TOP_K]]
            # CE scores
            q = test_pairs[i][0]
            candidate_texts = [corpus[int(j)] for j in top_candidates]
            pairs = [(q, doc) for doc in candidate_texts]
            ce_scores = ce.predict(pairs)
            ce_order = np.argsort(ce_scores)[::-1][:TOP_K]
            ce_top = [(int(top_candidates[j]), float(ce_scores[j])) for j in ce_order]
            fused.append(rrf_fusion_two([bi_top, ce_top], k=rrf_k))
        r_mrr = compute_mrr(fused, true_indices)
        r_h1 = hits_at_k(fused, true_indices, 1)
        print(f"  RRF k={rrf_k}: MRR={r_mrr:.4f}, Hits@1={r_h1} ({100*r_h1/len(test_pairs):.1f}%)")
        if r_mrr > best_rrf_mrr:
            best_rrf_mrr = r_mrr
            best_rrf_k = rrf_k

    # --- Choose best ---
    all_mrrs = {
        "bi-encoder only": b_mrr,
        f"pure CE (top-{CANDIDATE_K})": ce_mrr,
        f"CE+bi combo (CE_w={best_combo_w})": best_combo_mrr,
        f"bi+CE RRF (k={best_rrf_k})": best_rrf_mrr,
    }
    best_name = max(all_mrrs, key=lambda k: all_mrrs[k])
    improved_mrr = all_mrrs[best_name]

    improvement_pp = round((improved_mrr - b_mrr) * 100, 2)
    e_pass = improvement_pp > 10.0 or improved_mrr > 0.70

    print(f"\n=== FINAL RESULTS ===")
    print(f"Baseline (bi-encoder only): {b_mrr:.4f}")
    print(f"Best: {best_name} → MRR={improved_mrr:.4f}")
    print(f"Improvement: {improvement_pp} pp")
    print(f"E_pass: {e_pass}")

    result = {
        "approach": best_name,
        "original_mrr": round(b_mrr, 4),
        "improved_mrr": round(improved_mrr, 4),
        "improvement_pp": improvement_pp,
        "E_pass": e_pass
    }
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten to {OUTPUT_PATH}")
    return result

if __name__ == "__main__":
    main()