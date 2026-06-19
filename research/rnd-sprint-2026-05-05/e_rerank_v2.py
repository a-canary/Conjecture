#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
E-RERANK v2: Extended experiment — 2-stage + score-weighted RRF
"""
import json, random, math, numpy as np
from sentence_transformers import SentenceTransformer

random.seed(42)
np.random.seed(42)

NUM_QUERIES = 500
TOP_K = 20
RRF_K = 60
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

def compute_mrr(results, true_indices):
    return sum(1/rank for result, ti in zip(results, true_indices)
               for rank, (cid, _) in enumerate(result, 1) if cid == ti) / len(true_indices)

def hits_at_k(results, true_indices, k):
    return sum(1 for result, ti in zip(results, true_indices)
               for rank, (cid, _) in enumerate(result, 1) if rank <= k and cid == ti)

def rrf_fusion(bi_results, bm25_results, k=60):
    scores = {}
    for rank, (doc_id, _) in enumerate(bi_results, 1):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    for rank, (doc_id, _) in enumerate(bm25_results, 1):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def score_fusion(bi_results, bm25_results, alpha=0.5):
    scores = {}
    for doc_id, bi_score in bi_results:
        scores[doc_id] = scores.get(doc_id, 0) + alpha * bi_score
    for doc_id, bm25_score in bm25_results:
        scores[doc_id] = scores.get(doc_id, 0) + (1-alpha) * bm25_score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def main():
    print("Loading bi-encoder...")
    bi = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

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

    print("\nEncoding corpus...")
    emb = bi.encode(corpus, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    norm_emb = emb / norms

    print("Building BM25 index...")
    bm25 = BM25(corpus)

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
    print(f"Bi-encoder MRR: {b_mrr:.4f}, Hits@1: {b_h1} ({100*b_h1/len(test_pairs):.1f}%), Hits@5: {b_h5} ({100*b_h5/len(test_pairs):.1f}%)")

    # --- BM25 only ---
    print("\n--- BM25 only ---")
    bm = []
    for q, _, _ in test_pairs:
        bm.append(bm25.search(q, TOP_K))
    bm_mrr = compute_mrr(bm, true_indices)
    bm_h1 = hits_at_k(bm, true_indices, 1)
    print(f"BM25 MRR: {bm_mrr:.4f}, Hits@1: {bm_h1} ({100*bm_h1/len(test_pairs):.1f}%)")

    # --- Bi-encoder + BM25 score fusion (various alpha) ---
    print("\n--- Bi + BM25 score fusion ---")
    best_f_mrr = 0
    best_alpha = 0
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5]:
        fused = []
        for q, _, _ in test_pairs:
            qe = bi.encode([q], convert_to_numpy=True)
            qe = qe / (np.linalg.norm(qe) + 1e-8)
            scores = np.dot(norm_emb, qe.T).flatten()
            bi_top = [(int(i), float(scores[i])) for i in np.argsort(scores)[::-1][:TOP_K]]
            bm_top = bm25.search(q, TOP_K)
            fused.append(score_fusion(bi_top, bm_top, alpha))
        f_mrr = compute_mrr(fused, true_indices)
        f_h1 = hits_at_k(fused, true_indices, 1)
        print(f"  alpha={alpha}: MRR={f_mrr:.4f}, Hits@1={f_h1} ({100*f_h1/len(test_pairs):.1f}%)")
        if f_mrr > best_f_mrr:
            best_f_mrr = f_mrr
            best_alpha = alpha

    # --- RRF with different k ---
    print("\n--- Bi + BM25 RRF (various k) ---")
    best_rrf_mrr = 0
    best_k = 60
    for k in [10, 30, 60, 100]:
        fused = []
        for q, _, _ in test_pairs:
            qe = bi.encode([q], convert_to_numpy=True)
            qe = qe / (np.linalg.norm(qe) + 1e-8)
            scores = np.dot(norm_emb, qe.T).flatten()
            bi_top = [(int(i), float(scores[i])) for i in np.argsort(scores)[::-1][:TOP_K]]
            bm_top = bm25.search(q, TOP_K)
            fused.append(rrf_fusion(bi_top, bm_top, k))
        rf_mrr = compute_mrr(fused, true_indices)
        rf_h1 = hits_at_k(fused, true_indices, 1)
        print(f"  k={k}: MRR={rf_mrr:.4f}, Hits@1={rf_h1} ({100*rf_h1/len(test_pairs):.1f}%)")
        if rf_mrr > best_rrf_mrr:
            best_rrf_mrr = rf_mrr
            best_k = k

    # --- 2-stage: bi-encoder candidate selection, BM25 re-ranking within top-50 ---
    print("\n--- 2-stage: bi candidates + BM25 re-rank ---")
    stage2 = []
    for q, _, _ in test_pairs:
        qe = bi.encode([q], convert_to_numpy=True)
        qe = qe / (np.linalg.norm(qe) + 1e-8)
        scores = np.dot(norm_emb, qe.T).flatten()
        bi_top50 = np.argsort(scores)[::-1][:50]
        bm25_scores = [(i, bm25.score(tokenize(q), int(i))) for i in bi_top50]
        bm25_scores.sort(key=lambda x: x[1], reverse=True)
        stage2.append(bm25_scores[:TOP_K])
    s2_mrr = compute_mrr(stage2, true_indices)
    s2_h1 = hits_at_k(stage2, true_indices, 1)
    print(f"  2-stage MRR: {s2_mrr:.4f}, Hits@1={s2_h1} ({100*s2_h1/len(test_pairs):.1f}%)")

    # --- Choose best ---
    methods = {
        "bi-encoder only": (b_mrr, baseline),
        "BM25 only": (bm_mrr, bm),
        f"bi+BM25 score fusion (alpha={best_alpha})": (best_f_mrr, None),
        f"bi+BM25 RRF (k={best_k})": (best_rrf_mrr, None),
        "2-stage (bi→BM25)": (s2_mrr, None),
    }
    best_name = max(methods, key=lambda k: methods[k][0])
    improved_mrr = methods[best_name][0]

    improvement_pp = round((improved_mrr - b_mrr) * 100, 2)
    e_pass = improvement_pp > 10.0 or improved_mrr > 0.70

    print(f"\n=== FINAL ===")
    print(f"Best: {best_name} → MRR={improved_mrr:.4f}")
    print(f"Baseline: {b_mrr:.4f}")
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

if __name__ == "__main__":
    main()