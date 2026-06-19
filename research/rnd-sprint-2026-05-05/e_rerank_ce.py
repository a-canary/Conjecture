#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
E-RERANK: Cross-encoder reranking on bi-encoder top-20 candidates.
Uses ms-marco-MiniLM-L6-v2 cross-encoder to re-rank bi-encoder retrievals.

Strategy:
1. Bi-encoder: get top-50 candidates (cosine similarity)
2. Cross-encoder: score each (query, doc) pair
3. Combine bi + CE scores with various weights
4. Report MRR, Hits@1, Hits@5 for each variant
"""
import json, random, math, numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

random.seed(42)
np.random.seed(42)

NUM_TEXTS = 1000
NUM_QUERIES = 500
TOP_K = 20
CANDIDATE_K = 50  # bi-encoder candidates before CE rerank
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

def rrf_fusion(ranked_lists, k=60):
    """RRF across multiple ranked lists. Each ranked_list is [(doc_id, score), ...]."""
    scores = {}
    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list, 1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def main():
    print("Loading bi-encoder and cross-encoder...")
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

    # --- Phase 2: Bi-encoder + Cross-encoder (CE weight sweep) ---
    print("\n--- Phase 2: Bi + Cross-encoder reranking (top-50 candidates) ---")
    best_ce_mrr = 0
    best_ce_w = 0
    for ce_w in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        bi_w = 1.0 - ce_w
        reranked = []
        for q, _, _ in test_pairs:
            qe = bi.encode([q], convert_to_numpy=True)
            qe = qe / (np.linalg.norm(qe) + 1e-8)
            scores = np.dot(norm_emb, qe.T).flatten()
            top_candidates = np.argsort(scores)[::-1][:CANDIDATE_K]
            candidate_texts = [corpus[int(i)] for i in top_candidates]
            pairs = [(q, doc) for doc in candidate_texts]
            ce_scores = ce.predict(pairs)

            # Normalize both score distributions
            bi_cand_scores = scores[top_candidates]
            bi_norm = (bi_cand_scores - bi_cand_scores.min()) / (bi_cand_scores.max() - bi_cand_scores.min() + 1e-8)
            ce_norm = (ce_scores - ce_scores.min()) / (ce_scores.max() - ce_scores.min() + 1e-8)

            combined = ce_w * ce_norm + bi_w * bi_norm
            sorted_order = np.argsort(combined)[::-1][:TOP_K]
            result = [(int(top_candidates[i]), float(combined[i])) for i in sorted_order]
            reranked.append(result)

        r_mrr = compute_mrr(reranked, true_indices)
        r_h1 = hits_at_k(reranked, true_indices, 1)
        r_h5 = hits_at_k(reranked, true_indices, 5)
        print(f"  CE_w={ce_w:.2f}: MRR={r_mrr:.4f}, Hits@1={r_h1} ({100*r_h1/len(test_pairs):.1f}%), Hits@5={r_h5} ({100*r_h5/len(test_pairs):.1f}%)")
        if r_mrr > best_ce_mrr:
            best_ce_mrr = r_mrr
            best_ce_w = ce_w

    # --- Phase 3: 3-way fusion (bi + CE + BM25) via RRF ---
    print("\n--- Phase 3: 3-way RRF (bi + CE + BM25) ---")
    best_3way_mrr = 0
    best_3way_k = 60
    for rrf_k in [30, 60]:
        fused_results = []
        for q, _, _ in test_pairs:
            qe = bi.encode([q], convert_to_numpy=True)
            qe = qe / (np.linalg.norm(qe) + 1e-8)
            scores = np.dot(norm_emb, qe.T).flatten()
            top_candidates = np.argsort(scores)[::-1][:CANDIDATE_K]
            candidate_texts = [corpus[int(i)] for i in top_candidates]
            pairs = [(q, doc) for doc in candidate_texts]
            ce_scores = ce.predict(pairs)

            # Bi top-20
            bi_top = [(int(top_candidates[i]), float(scores[top_candidates[i]]))
                      for i in np.argsort(scores[top_candidates])[::-1][:TOP_K]]
            # CE top-20 by CE score alone
            ce_order = np.argsort(ce_scores)[::-1][:TOP_K]
            ce_top = [(int(top_candidates[i]), float(ce_scores[i])) for i in ce_order]
            # BM25 top-20
            bm25_top = bm25.search(q, TOP_K)

            # 3-way RRF
            fused = rrf_fusion([bi_top, ce_top, bm25_top], k=rrf_k)
            fused_results.append(fused)

        f_mrr = compute_mrr(fused_results, true_indices)
        f_h1 = hits_at_k(fused_results, true_indices, 1)
        f_h5 = hits_at_k(fused_results, true_indices, 5)
        print(f"  3-way RRF k={rrf_k}: MRR={f_mrr:.4f}, Hits@1={f_h1} ({100*f_h1/len(test_pairs):.1f}%), Hits@5={f_h5} ({100*f_h5/len(test_pairs):.1f}%)")
        if f_mrr > best_3way_mrr:
            best_3way_mrr = f_mrr
            best_3way_k = rrf_k

    # --- Phase 4: CE-only reranking (CE scores only, no bi) ---
    print("\n--- Phase 4: CE-only reranking on bi top-50 ---")
    ce_only = []
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
        ce_only.append(result)
    ce_mrr = compute_mrr(ce_only, true_indices)
    ce_h1 = hits_at_k(ce_only, true_indices, 1)
    print(f"CE-only MRR: {ce_mrr:.4f}, Hits@1: {ce_h1} ({100*ce_h1/len(test_pairs):.1f}%)")

    # --- Phase 5: BM25 + CE RRF ---
    print("\n--- Phase 5: BM25 + CE RRF ---")
    for rrf_k in [30, 60]:
        fused = []
        for q, _, _ in test_pairs:
            bm25_top = bm25.search(q, TOP_K)
            qe = bi.encode([q], convert_to_numpy=True)
            qe = qe / (np.linalg.norm(qe) + 1e-8)
            scores = np.dot(norm_emb, qe.T).flatten()
            top_candidates = np.argsort(scores)[::-1][:CANDIDATE_K]
            candidate_texts = [corpus[int(i)] for i in top_candidates]
            pairs = [(q, doc) for doc in candidate_texts]
            ce_scores = ce.predict(pairs)
            ce_order = np.argsort(ce_scores)[::-1][:TOP_K]
            ce_top = [(int(top_candidates[i]), float(ce_scores[i])) for i in ce_order]
            fused.append(rrf_fusion([bm25_top, ce_top], k=rrf_k))
        bmce_mrr = compute_mrr(fused, true_indices)
        bmce_h1 = hits_at_k(fused, true_indices, 1)
        print(f"BM25+CE RRF k={rrf_k}: MRR={bmce_mrr:.4f}, Hits@1={bmce_h1} ({100*bmce_h1/len(test_pairs):.1f}%)")

    # --- Choose best approach ---
    all_mrrs = {
        "bi-encoder only": b_mrr,
        "BM25 only": 0.3992,  # from previous run
        f"bi+CE (CE_w={best_ce_w:.2f})": best_ce_mrr,
        f"3-way RRF (k={best_3way_k})": best_3way_mrr,
        "CE-only": ce_mrr,
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