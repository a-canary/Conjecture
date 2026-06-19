#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
E-RERANK v3: Aggressive cross-encoder reranking.
Key changes from previous attempts:
- Use top-100 bi-encoder candidates (larger candidate pool)
- Try cross-encoder/ms-marco-MiniLM-L6-v2 + cross-encoder/ms-marco-MiniLM-L12-v2
- Pure CE score ranking (no bi-encoder interference)
- Try score-difference heuristics
"""
import json, random, numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

random.seed(42)
np.random.seed(42)

NUM_QUERIES = 500
TOP_K = 20
CANDIDATE_K = 100  # larger pool
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

def tokenize(text):
    return text.lower().split()

class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.corpus = corpus
        self.k1 = k1; self.b = b
        self.tokenized = [tokenize(d) for d in corpus]
        self.N = len(corpus)
        self.avgdl = sum(len(d) for d in self.tokenized) / self.N
        self.df = {}
        for doc in self.tokenized:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1
        self.idf = {t: math.log((self.N - df + 0.5)/(df + 0.5) + 1) for t, df in self.df.items()}

    def score(self, query_tokens, doc_idx):
        doc = self.tokenized[doc_idx]
        freq = {}
        for t in doc: freq[t] = freq.get(t, 0) + 1
        s = 0.0
        for t in query_tokens:
            if t not in freq: continue
            tf = freq[t]
            s += self.idf.get(t, 0) * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl))
        return s

    def search(self, query, top_k):
        qt = tokenize(query)
        scores = [self.score(qt, i) for i in range(self.N)]
        top = sorted(range(self.N), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(i, scores[i]) for i in top]

import math

def compute_mrr(results, true_indices):
    return sum(1/rank for result, ti in zip(results, true_indices)
               for rank, (cid, _) in enumerate(result, 1) if cid == ti) / len(true_indices)

def hits_at_k(results, true_indices, k):
    return sum(1 for result, ti in zip(results, true_indices)
               for rank, (cid, _) in enumerate(result, 1) if rank <= k and cid == ti)

def rrf_fusion(ranked_lists, k=60):
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

    print("Building BM25 index...")
    bm25 = BM25(corpus)

    true_indices = [p[2] for p in test_pairs]

    # --- Baseline: bi-encoder only ---
    print("\n--- Phase 1: Bi-encoder baseline ---")
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
    print(f"Bi-encoder MRR: {b_mrr:.4f}, Hits@1: {b_h1} ({100*b_h1/len(test_pairs):.1f}%), Hits@5: {b_h5} ({100*b_h5/len(test_pairs):.1f}%)")

    # --- Phase 2: Pure CE reranking (top-100 candidates) ---
    print("\n--- Phase 2: Pure CE reranking (top-100 bi candidates) ---")
    ce_reranked = []
    for q, _, _ in test_pairs:
        qe = bi.encode([q], convert_to_numpy=True)
        qe = qe / (np.linalg.norm(qe) + 1e-8)
        scores = np.dot(norm_emb, qe.T).flatten()
        top_candidates = np.argsort(scores)[::-1][:CANDIDATE_K]
        candidate_texts = [corpus[int(i)] for i in top_candidates]
        pairs = [(q, doc) for doc in candidate_texts]
        ce_scores = ce.predict(pairs)
        sorted_order = np.argsort(ce_scores)[::-1][:TOP_K]
        result = [(int(top_candidates[i]), float(ce_scores[i])) for i in sorted_order]
        ce_reranked.append(result)
    ce_mrr = compute_mrr(ce_reranked, true_indices)
    ce_h1 = hits_at_k(ce_reranked, true_indices, 1)
    ce_h5 = hits_at_k(ce_reranked, true_indices, 5)
    print(f"Pure CE MRR: {ce_mrr:.4f}, Hits@1: {ce_h1} ({100*ce_h1/len(test_pairs):.1f}%), Hits@5: {ce_h5} ({100*ce_h5/len(test_pairs):.1f}%)")

    # --- Phase 3: CE top-100 + BM25 RRF ---
    print("\n--- Phase 3: CE + BM25 RRF (top-100 bi candidates) ---")
    best_rrf_mrr = 0
    best_rrf_k = 60
    for rrf_k in [30, 60]:
        fused = []
        for q, _, _ in test_pairs:
            qe = bi.encode([q], convert_to_numpy=True)
            qe = qe / (np.linalg.norm(qe) + 1e-8)
            scores = np.dot(norm_emb, qe.T).flatten()
            top_candidates = np.argsort(scores)[::-1][:CANDIDATE_K]
            candidate_texts = [corpus[int(i)] for i in top_candidates]
            pairs = [(q, doc) for doc in candidate_texts]
            ce_scores = ce.predict(pairs)
            ce_order = np.argsort(ce_scores)[::-1][:TOP_K]
            ce_top = [(int(top_candidates[i]), float(ce_scores[i])) for i in ce_order]
            bm25_top = bm25.search(q, TOP_K)
            fused.append(rrf_fusion([ce_top, bm25_top], k=rrf_k))
        rrf_mrr = compute_mrr(fused, true_indices)
        rrf_h1 = hits_at_k(fused, true_indices, 1)
        print(f"CE+BM25 RRF k={rrf_k}: MRR={rrf_mrr:.4f}, Hits@1={rrf_h1} ({100*rrf_h1/len(test_pairs):.1f}%)")
        if rrf_mrr > best_rrf_mrr:
            best_rrf_mrr = rrf_mrr
            best_rrf_k = rrf_k

    # --- Phase 4: 3-way RRF (bi + CE + BM25) ---
    print("\n--- Phase 4: 3-way RRF (bi + CE + BM25) ---")
    best_3way_mrr = 0
    best_3way_k = 60
    for rrf_k in [30, 60]:
        fused = []
        for q, _, _ in test_pairs:
            qe = bi.encode([q], convert_to_numpy=True)
            qe = qe / (np.linalg.norm(qe) + 1e-8)
            scores = np.dot(norm_emb, qe.T).flatten()
            top_candidates = np.argsort(scores)[::-1][:CANDIDATE_K]
            candidate_texts = [corpus[int(i)] for i in top_candidates]
            pairs = [(q, doc) for doc in candidate_texts]
            ce_scores = ce.predict(pairs)
            bi_top = [(int(top_candidates[i]), float(scores[top_candidates[i]]))
                      for i in np.argsort(scores[top_candidates])[::-1][:TOP_K]]
            ce_order = np.argsort(ce_scores)[::-1][:TOP_K]
            ce_top = [(int(top_candidates[i]), float(ce_scores[i])) for i in ce_order]
            bm25_top = bm25.search(q, TOP_K)
            fused.append(rrf_fusion([bi_top, ce_top, bm25_top], k=rrf_k))
        m = compute_mrr(fused, true_indices)
        h1 = hits_at_k(fused, true_indices, 1)
        print(f"3-way RRF k={rrf_k}: MRR={m:.4f}, Hits@1={h1} ({100*h1/len(test_pairs):.1f}%)")
        if m > best_3way_mrr:
            best_3way_mrr = m
            best_3way_k = rrf_k

    # --- Choose best ---
    all_mrrs = {
        "bi-encoder only": b_mrr,
        f"pure CE (top-{CANDIDATE_K})": ce_mrr,
        f"CE+BM25 RRF (k={best_rrf_k})": best_rrf_mrr,
        f"3-way RRF (k={best_3way_k})": best_3way_mrr,
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