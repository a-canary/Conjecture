#!/usr/bin/env python3
"""
E-RERANK: Cross-Encoder Re-Ranking — Expanded Test Set

The variant-based test set shows a clear signal (MRR 0.3193→0.3467, Hits@1 0→2)
but with only 50 queries the variance is too high. Increase to 500 queries
for a more reliable MRR estimate and try higher CE weights.
"""
import json
import random
import time
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

NUM_TEXTS = 1000   # will be overridden by actual corpus size
NUM_QUERIES = 500
TOP_K = 20
OUTPUT_PATH = "/home/aaron/projects/conjecture/research/rnd-sprint-2026-05-05/E-RERANK.json"

random.seed(42)
np.random.seed(42)

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

def cosine_search(embeddings, query_emb, k):
    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)
    scores = np.dot(embeddings, query_emb.T).flatten()
    top_k_indices = np.argsort(scores)[::-1][:k]
    top_k_scores = scores[top_k_indices]
    return top_k_scores, top_k_indices

def compute_mrr(results, true_indices):
    rr_sum = 0.0
    for result_list, true_idx in zip(results, true_indices):
        rr = 0.0
        for rank, (claim_id, score) in enumerate(result_list, 1):
            if claim_id == true_idx:
                rr = 1.0 / rank
                break
        rr_sum += rr
    return rr_sum / len(true_indices)

def hits_at_k(results, true_indices, k):
    hits = 0
    for result_list, true_idx in zip(results, true_indices):
        for rank, (claim_id, score) in enumerate(result_list, 1):
            if rank > k:
                break
            if claim_id == true_idx:
                hits += 1
                break
    return hits

def main():
    print("Loading models...")
    bi_encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2', device='cpu')

    # Build corpus
    print("\nBuilding corpus with variant groups...")
    rng = random.Random(42)
    seeds = list(range(200))
    corpus = []
    group_members = {}

    for seed in seeds:
        base_text = make_text(seed)
        group = [base_text]
        for _ in range(4):
            group.append(make_variant(base_text, rng))
        for idx in range(len(corpus), len(corpus) + len(group)):
            group_members.setdefault(seed, []).append(idx)
        corpus.extend(group)

    print(f"Corpus: {len(corpus)} ({len(seeds)} groups x 5)")

    # Test pairs: generate NUM_QUERIES pairs with potential seed reuse
    test_pairs = []
    seen_pairs = set()
    attempts = 0
    while len(test_pairs) < NUM_QUERIES and attempts < NUM_QUERIES * 10:
        attempts += 1
        seed = random.choice(seeds)
        members = group_members[seed]
        q_idx, t_idx = random.sample(members, 2)
        pair_key = (q_idx, t_idx)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        test_pairs.append((corpus[q_idx], corpus[t_idx], t_idx))

    print(f"Test pairs: {len(test_pairs)}")

    # Encode
    print(f"\nEncoding {len(corpus)} items...")
    embeddings = bi_encoder.encode(corpus, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    normalized_embeddings = embeddings / norms

    # Baseline
    print("\n--- Phase 1: Bi-encoder baseline ---")
    baseline_results = []
    for query_claim, _, true_idx in test_pairs:
        query_emb = bi_encoder.encode([query_claim], convert_to_numpy=True)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        scores, indices = cosine_search(normalized_embeddings, query_emb, TOP_K)
        result_list = [(int(idx), float(scores[i])) for i, idx in enumerate(indices)]
        baseline_results.append(result_list)

    baseline_mrr = compute_mrr(baseline_results, [p[2] for p in test_pairs])
    baseline_h1 = hits_at_k(baseline_results, [p[2] for p in test_pairs], 1)
    baseline_h5 = hits_at_k(baseline_results, [p[2] for p in test_pairs], 5)
    print(f"Baseline MRR@{TOP_K}: {baseline_mrr:.4f}")
    print(f"Baseline Hits@1: {baseline_h1} ({100*baseline_h1/len(test_pairs):.1f}%)")
    print(f"Baseline Hits@5: {baseline_h5} ({100*baseline_h5/len(test_pairs):.1f}%)")

    # Re-ranking with best blend
    print("\n--- Phase 2: Bi-encoder + Cross-encoder (CE weight=0.8) ---")
    reranked_results = []
    for query_claim, _, true_idx in test_pairs:
        query_emb = bi_encoder.encode([query_claim], convert_to_numpy=True)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        scores, indices = cosine_search(normalized_embeddings, query_emb, TOP_K)

        candidate_indices = list(indices)
        candidate_claims = [corpus[int(idx)] for idx in candidate_indices]

        query_doc_pairs = [(query_claim, doc) for doc in candidate_claims]
        cross_scores = cross_encoder.predict(query_doc_pairs)

        bi_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        ce_norm = (cross_scores - cross_scores.min()) / (cross_scores.max() - cross_scores.min() + 1e-8)

        combined = 0.8 * ce_norm + 0.2 * bi_norm
        sorted_pairs = sorted(zip(candidate_indices, combined), key=lambda x: x[1], reverse=True)
        reranked_results.append(sorted_pairs)

    reranked_mrr = compute_mrr(reranked_results, [p[2] for p in test_pairs])
    reranked_h1 = hits_at_k(reranked_results, [p[2] for p in test_pairs], 1)
    reranked_h5 = hits_at_k(reranked_results, [p[2] for p in test_pairs], 5)
    print(f"Re-ranked MRR@{TOP_K}: {reranked_mrr:.4f}")
    print(f"Re-ranked Hits@1: {reranked_h1} ({100*reranked_h1/len(test_pairs):.1f}%)")
    print(f"Re-ranked Hits@5: {reranked_h5} ({100*reranked_h5/len(test_pairs):.1f}%)")

    improvement_pp = round((reranked_mrr - baseline_mrr) * 100, 2)
    e_pass = improvement_pp > 10.0 or reranked_mrr > 0.70

    print(f"\n=== RESULTS ===")
    print(f"Original MRR:  {baseline_mrr:.4f}")
    print(f"Improved MRR:  {reranked_mrr:.4f}")
    print(f"Improvement:   {improvement_pp} pp")
    print(f"E_pass:        {e_pass}")

    result = {
        "approach": "cross-encoder/ms-marco-MiniLM-L6-v2 re-ranking on top-20 bi-encoder candidates (80% CE + 20% bi)",
        "original_mrr": round(baseline_mrr, 4),
        "improved_mrr": round(reranked_mrr, 4),
        "improvement_pp": improvement_pp,
        "E_pass": e_pass
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten to {OUTPUT_PATH}")
    return result

if __name__ == "__main__":
    main()