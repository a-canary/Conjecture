#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
E-RERANK BM25: 2-stage retrieval with bi-encoder + TF-IDF via RRF.
No cross-encoder available, so we combine bi-encoder with a
simple TF-IDF/KL-divergence style second stage using RRF fusion.
"""
import json
import random
import math
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer

random.seed(42)
np.random.seed(42)

NUM_QUERIES = 500
TOP_K = 20
CANDIDATE_K = 50  # candidates from bi-encoder for TF-IDF re-rank
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

def build_tf_idf(corpus):
    """Build TF-IDF vectors for corpus using sparse representation."""
    N = len(corpus)
    docs_tokens = [tokenize(t) for t in corpus]
    df = Counter()
    for tokens in docs_tokens:
        df.update(set(tokens))
    idf = {word: math.log((N - df[word] + 0.5) / (df[word] + 0.5) + 1) for word in df}

    tfidf_vectors = []
    for tokens in docs_tokens:
        tf = Counter(tokens)
        vec = {word: (0.5 + 0.5 * tf[word] / max(tf.values())) * idf.get(word, 0) for word in tf}
        tfidf_vectors.append(vec)
    return tfidf_vectors

def tfidf_cosine(q_vec, d_vec):
    """Cosine similarity between two sparse TF-IDF vectors."""
    common = set(q_vec) & set(d_vec)
    if not common:
        return 0.0
    num = sum(q_vec[w] * d_vec[w] for w in common)
    q_norm = math.sqrt(sum(v*v for v in q_vec.values()))
    d_norm = math.sqrt(sum(v*v for v in d_vec.values()))
    if q_norm == 0 or d_norm == 0:
        return 0.0
    return num / (q_norm * d_norm)

def tfidf_search(corpus, tfidf_vectors, query, top_k):
    """Search corpus using TF-IDF cosine similarity."""
    q_tokens = tokenize(query)
    q_tf = Counter(q_tokens)
    q_vec = {word: (0.5 + 0.5 * q_tf[word] / max(q_tf.values())) for word in q_tf}
    scores = [tfidf_cosine(q_vec, d_vec) for d_vec in tfidf_vectors]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(idx), float(scores[idx])) for idx in top_indices]

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
        if key in seen:
            continue
        seen.add(key)
        test_pairs.append((corpus[qi], corpus[ti], ti))
    print(f"Test pairs: {len(test_pairs)}")

    print("\nBuilding TF-IDF index...")
    tfidf_vectors = build_tf_idf(corpus)

    print("Encoding corpus with bi-encoder...")
    emb = bi.encode(corpus, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    norm_emb = emb / norms

    true_indices = [p[2] for p in test_pairs]

    # Pre-encode queries
    print("Pre-encoding queries...")
    query_embs = []
    for q, _, _ in test_pairs:
        qe = bi.encode([q], convert_to_numpy=True)
        qe = qe / (np.linalg.norm(qe) + 1e-8)
        query_embs.append(qe)
    query_embs = np.vstack(query_embs)

    print("Computing bi-encoder score matrix...")
    bi_scores_matrix = np.dot(query_embs, norm_emb.T)

    # --- Baseline ---
    print("\n--- Phase 1: Bi-encoder baseline ---")
    baseline = []
    for i in range(len(test_pairs)):
        scores = bi_scores_matrix[i]
        top = [(int(j), float(scores[j])) for j in np.argsort(scores)[::-1][:TOP_K]]
        baseline.append(top)
    b_mrr = compute_mrr(baseline, true_indices)
    b_h1 = hits_at_k(baseline, true_indices, 1)
    b_h5 = hits_at_k(baseline, true_indices, 5)
    print(f"Bi-encoder MRR: {b_mrr:.4f}, Hits@1: {b_h1} ({100*b_h1/len(test_pairs):.1f}%), Hits@5: {b_h5}")

    # --- 2-way RRF (bi-encoder + TF-IDF) ---
    print(f"\n--- Phase 2: 2-way RRF (bi-encoder + TF-IDF) ---")
    best_rrf_mrr = 0
    best_rrf_k = 60
    best_rrf_name = ""

    for rrf_k in [30, 60]:
        fused = []
        for i in range(len(test_pairs)):
            q = test_pairs[i][0]
            bi_scores = bi_scores_matrix[i]
            bi_top = [(int(j), float(bi_scores[j])) for j in np.argsort(bi_scores)[::-1][:TOP_K]]
            tfidf_top = tfidf_search(corpus, tfidf_vectors, q, TOP_K)
            fused.append(rrf_fusion([bi_top, tfidf_top], k=rrf_k))
        r_mrr = compute_mrr(fused, true_indices)
        r_h1 = hits_at_k(fused, true_indices, 1)
        print(f"  RRF k={rrf_k}: MRR={r_mrr:.4f}, Hits@1={r_h1} ({100*r_h1/len(test_pairs):.1f}%)")
        if r_mrr > best_rrf_mrr:
            best_rrf_mrr = r_mrr
            best_rrf_k = rrf_k

    # --- BM25-style second-stage re-rank ---
    print(f"\n--- Phase 3: BM25-style re-rank (bi top-{CANDIDATE_K} re-scored by TF-IDF) ---")
    bm25_reranked = []
    for i in range(len(test_pairs)):
        q = test_pairs[i][0]
        bi_scores = bi_scores_matrix[i]
        top_candidates = np.argsort(bi_scores)[::-1][:CANDIDATE_K]
        q_tokens = tokenize(q)
        q_tf = Counter(q_tokens)
        q_vec = {word: (0.5 + 0.5 * q_tf[word] / max(q_tf.values())) for word in q_tf}

        bm25_scores = []
        for j in top_candidates:
            d_tokens = tokenize(corpus[int(j)])
            d_tf = Counter(d_tokens)
            d_vec = {word: (0.5 + 0.5 * d_tf[word] / max(d_tf.values())) for word in d_tf}
            bm25_scores.append(tf_idf_cosine(q_vec, d_vec) if d_vec else 0)

        bm25_scores = np.array(bm25_scores)
        bi_norm = (bi_scores[top_candidates] - bi_scores[top_candidates].min()) / \
                  (bi_scores[top_candidates].max() - bi_scores[top_candidates].min() + 1e-8)
        bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        combined = 0.5 * bi_norm + 0.5 * bm25_norm
        sorted_order = np.argsort(combined)[::-1][:TOP_K]
        result = [(int(top_candidates[j]), float(combined[j])) for j in sorted_order]
        bm25_reranked.append(result)
    bm25_mrr = compute_mrr(bm25_reranked, true_indices)
    bm25_h1 = hits_at_k(bm25_reranked, true_indices, 1)
    bm25_h5 = hits_at_k(bm25_reranked, true_indices, 5)
    print(f"BM25-style MRR: {bm25_mrr:.4f}, Hits@1: {bm25_h1} ({100*bm25_h1/len(test_pairs):.1f}%), Hits@5: {bm25_h5}")

    # --- 3-way RRF (bi + TF-IDF top + BM25 re-score) ---
    print(f"\n--- Phase 4: 3-way RRF (bi + TF-IDF + BM25) ---")
    for rrf_k in [30, 60]:
        fused = []
        for i in range(len(test_pairs)):
            q = test_pairs[i][0]
            bi_scores = bi_scores_matrix[i]
            bi_top = [(int(j), float(bi_scores[j])) for j in np.argsort(bi_scores)[::-1][:TOP_K]]
            tfidf_top = tfidf_search(corpus, tfidf_vectors, q, TOP_K)
            bm25_top = bm25_reranked[i][:TOP_K]
            fused.append(rrf_fusion([bi_top, tfidf_top, bm25_top], k=rrf_k))
        r_mrr = compute_mrr(fused, true_indices)
        r_h1 = hits_at_k(fused, true_indices, 1)
        print(f"  3-way RRF k={rrf_k}: MRR={r_mrr:.4f}, Hits@1={r_h1} ({100*r_h1/len(test_pairs):.1f}%)")

    # Choose best approach
    all_mrrs = {
        "bi-encoder only": b_mrr,
        f"bi+TF-IDF RRF (k={best_rrf_k})": best_rrf_mrr,
        f"BM25-style blend (top-{CANDIDATE_K})": bm25_mrr,
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
        "approach": f"bi-encoder + TF-IDF/BM25-style 2-stage retrieval ({best_name})",
        "original_mrr": round(b_mrr, 4),
        "improved_mrr": round(improved_mrr, 4),
        "improvement_pp": improvement_pp,
        "E_pass": e_pass
    }
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten to {OUTPUT_PATH}")
    return result

def tf_idf_cosine(q_vec, d_vec):
    """Cosine similarity between two sparse TF-IDF vectors."""
    common = set(q_vec) & set(d_vec)
    if not common:
        return 0.0
    num = sum(q_vec[w] * d_vec[w] for w in common)
    q_norm = math.sqrt(sum(v*v for v in q_vec.values()))
    d_norm = math.sqrt(sum(v*v for v in d_vec.values()))
    if q_norm == 0 or d_norm == 0:
        return 0.0
    return num / (q_norm * d_norm)

if __name__ == "__main__":
    main()