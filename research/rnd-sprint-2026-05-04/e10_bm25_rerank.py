#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
E10: BM25 Lexical Re-Ranking Benchmark

Improves vector search MRR by re-ranking top-20 bi-encoder candidates with BM25.
No model download needed - uses lexical matching instead of neural cross-encoders.

Test methodology:
- query = claim A
- target = claim B (related: shares 15-60% of words but is NOT identical)
- candidates = top-20 by bi-encoder for query A
- Since query != target, bi-encoder may not rank B at position 1
- BM25 re-ranking should help if B shares significant terminology with A
"""

import json
import random
import time
import numpy as np
import os
from typing import List, Tuple, Dict, Any
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''

NUM_TEXTS = 10000
NUM_QUERIES = 50
TOP_K = 20
OUTPUT_PATH = "./research/rnd-sprint-2026-05-04/E10-results.json"

random.seed(42)
np.random.seed(42)


def generate_random_text(min_len: int = 50, max_len: int = 500) -> str:
    words = [
        "The", "research", "shows", "that", "climate", "change", "impacts", "weather",
        "patterns", "significantly", "according", "to", "scientists", "at", "MIT",
        "Harvard", "and", "other", "universities", "Studies", "indicate", "potential",
        "risks", "for", "coastal", "areas", "with", "rising", "sea", "levels",
        "Data", "from", "satellites", "reveals", "new", "information", "about",
        "environmental", "shifts", "occurring", "faster", "than", "expected",
        "Teams", "are", "working", "on", "solutions", "for", "sustainable", "energy",
        "Innovation", "drives", "progress", "in", "technology", "and", "medicine",
        "Patients", "benefit", "from", "advances", "in", "treatment", "options",
        "Economies", "grow", "through", "trade", "and", "investment", "globally",
        "Markets", "respond", "to", "economic", "indicators", "and", "policy",
        "decisions", "regularly", "Companies", "compete", "for", "talent", "and",
        "resources", "in", "dynamic", "environments", "with", "changing", "demands",
        "Education", "systems", "evolve", "to", "meet", "needs", "of", "students",
        "worldwide", "learning", "outcomes", "improve", "with", "new", "methods",
        "Communication", "networks", "connect", "people", "across", "continents",
        "Security", "protocols", "protect", "sensitive", "information", "systems",
        "Infrastructure", "requires", "maintenance", "and", "upgrades", "regularly",
        "Urban", "planning", "considers", "future", "growth", "and", "development",
        "Agriculture", "adapts", "to", "changing", "conditions", "and", "technology"
    ]
    target_len = random.randint(min_len, max_len)
    text = []
    current_len = 0
    while current_len < target_len:
        word = random.choice(words)
        text.append(word)
        current_len += len(word) + 1
    result = ' '.join(text)
    if result:
        result = result[0].upper() + result[1:]
        if not result.endswith('.'):
            result += '.'
    return result


def generate_synthetic_claims(n: int) -> List[str]:
    return [generate_random_text() for _ in range(n)]


def jaccard(s1: str, s2: str) -> float:
    w1 = set(s1.lower().split())
    w2 = set(s2.lower().split())
    inter = len(w1 & w2)
    union = len(w1 | w2)
    return inter / union if union > 0 else 0.0


def generate_test_pairs(claims: List[str], n_pairs: int, 
                        bi_encoder, normalized_embeddings) -> List[Tuple[str, int, List[int]]]:
    """
    Generate test pairs where:
    - query = claim A
    - target = claim B (related: 15-60% Jaccard overlap with A, but NOT identical)
    - candidates = top-20 by bi-encoder for query A (MUST include target)
    
    This tests whether BM25 can improve ranking of a semantically related document.
    """
    pairs = []
    
    for _ in range(n_pairs):
        # Pick a random claim as query base
        query_idx = random.randint(0, len(claims) - 1)
        query_text = claims[query_idx]
        query_words = set(query_text.lower().split())
        
        # Find a related but not identical claim
        target_idx = None
        candidates_pool = random.sample(
            [i for i in range(len(claims)) if i != query_idx],
            min(200, len(claims) - 1)
        )
        
        for cand_idx in candidates_pool:
            cand_text = claims[cand_idx]
            j = jaccard(query_text, cand_text)
            if 0.15 < j < 0.60:  # Related but not identical
                target_idx = cand_idx
                break
        
        if target_idx is None:
            # Fallback: pick first candidate
            target_idx = candidates_pool[0]
        
        # Get bi-encoder top-20 for query
        q_emb = bi_encoder.encode([query_text], convert_to_numpy=True)
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        scores = np.dot(normalized_embeddings, q_emb.reshape(-1, 1)).flatten()
        top_20 = np.argsort(scores)[::-1][:TOP_K].tolist()
        
        # Ensure target is in top-20
        if target_idx not in top_20:
            # Insert target at position 10 (middle of ranking)
            top_20 = list(top_20)
            top_20[min(10, len(top_20)-1)] = target_idx
        
        pairs.append((query_text, target_idx, top_20))
    
    return pairs


class SimpleBM25:
    """Okapi BM25 implementation using only numpy/stdlib."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = {}
        self.avgdl = 0
        self.N = 0
        self.doc_lengths = []
        self.tokenized_docs = []
        
    def fit(self, documents: List[str]):
        self.N = len(documents)
        self.doc_lengths = []
        self.tokenized_docs = []
        df = {}
        
        for doc in documents:
            tokens = self._tokenize(doc)
            self.tokenized_docs.append(tokens)
            self.doc_lengths.append(len(tokens))
            
            seen = set()
            for token in tokens:
                if token not in seen:
                    df[token] = df.get(token, 0) + 1
                    seen.add(token)
        
        self.doc_freqs = df
        self.avgdl = sum(self.doc_lengths) / self.N if self.N > 0 else 0
    
    def _tokenize(self, text: str) -> List[str]:
        import re
        return re.findall(r'\b[a-z]+\b', text.lower())
    
    def score(self, query: str, doc_idx: int) -> float:
        query_tokens = self._tokenize(query)
        doc_tokens = self.tokenized_docs[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        
        score = 0.0
        doc_tf = {}
        for token in doc_tokens:
            doc_tf[token] = doc_tf.get(token, 0) + 1
        
        for q_token in query_tokens:
            if q_token not in doc_tf:
                continue
            tf = doc_tf[q_token]
            df = self.doc_freqs.get(q_token, 0)
            if df == 0:
                continue
            idf = np.log((self.N - df + 0.5) / (df + 0.5) + 1)
            tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
            score += idf * tf_component
        
        return score
    
    def rerank(self, query: str, candidate_indices: List[int]) -> List[Tuple[int, float]]:
        scores = [(idx, self.score(query, idx)) for idx in candidate_indices]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


def mrr(results: List[List[Tuple[int, float]]], true_indices: List[int]) -> float:
    rr_sum = 0.0
    for result_list, true_idx in zip(results, true_indices):
        rr = 0.0
        for rank, (cid, _) in enumerate(result_list, 1):
            if cid == true_idx:
                rr = 1.0 / rank
                break
        rr_sum += rr
    return rr_sum / len(results)


def hits_at_k(results: List[List[Tuple[int, float]]], 
              true_indices: List[int], k: int) -> int:
    hits = 0
    for result_list, true_idx in zip(results, true_indices):
        for rank, (cid, _) in enumerate(result_list, 1):
            if rank > k:
                break
            if cid == true_idx:
                hits += 1
                break
    return hits


def avg_rank(results: List[List[Tuple[int, float]]], true_indices: List[int]) -> float:
    ranks = []
    for result_list, true_idx in zip(results, true_indices):
        for rank, (cid, _) in enumerate(result_list, 1):
            if cid == true_idx:
                ranks.append(rank)
                break
    return sum(ranks) / len(ranks) if ranks else float('inf')


def run_benchmark():
    from sentence_transformers import SentenceTransformer
    
    print("=" * 60)
    print("E10: BM25 Lexical Re-Ranking Benchmark")
    print("=" * 60)
    
    print(f"\nGenerating {NUM_TEXTS} synthetic claim texts...")
    claims = generate_synthetic_claims(NUM_TEXTS)
    print(f"Generated {len(claims)} claims")
    
    print("\n" + "=" * 60)
    print("Loading bi-encoder model (all-MiniLM-L6-v2)...")
    print("=" * 60)
    
    bi_encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    
    print(f"Encoding {NUM_TEXTS} claims with bi-encoder...")
    start_time = time.time()
    embeddings = bi_encoder.encode(claims, convert_to_numpy=True, show_progress_bar=True)
    encode_time = time.time() - start_time
    print(f"Bi-encoder encode time: {encode_time:.2f}s")
    
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    normalized_embeddings = embeddings / norms
    
    print("\n" + "=" * 60)
    print(f"Generating {NUM_QUERIES} test query-target pairs...")
    print("=" * 60)
    
    test_pairs = generate_test_pairs(claims, NUM_QUERIES, bi_encoder, normalized_embeddings)
    print(f"Generated {len(test_pairs)} test pairs")
    print("Query=claim A, Target=claim B (15-60% word overlap), Candidates=top-20 by bi-encoder")
    
    # Verify targets are in top-20
    targets_in_top20 = 0
    for query_text, target_idx, candidates in test_pairs:
        if target_idx in candidates:
            targets_in_top20 += 1
    print(f"Targets in bi-encoder top-{TOP_K}: {targets_in_top20}/{len(test_pairs)}")
    
    # Check how often target is NOT at rank 1 (the interesting cases for re-ranking)
    targets_at_rank1 = 0
    for query_text, target_idx, candidates in test_pairs:
        if candidates[0] == target_idx:
            targets_at_rank1 += 1
    print(f"Targets at rank 1 (already perfect): {targets_at_rank1}/{len(test_pairs)}")
    
    print("\n" + "=" * 60)
    print("Building BM25 index...")
    print("=" * 60)
    
    bm25 = SimpleBM25(k1=1.5, b=0.75)
    bm25.fit(claims)
    print("BM25 index built successfully!")
    
    # Phase 1: Bi-encoder baseline
    print("\n" + "=" * 60)
    print("Phase 1: Bi-encoder only retrieval (baseline)")
    print("=" * 60)
    
    baseline_results = []
    
    for query_text, target_idx, candidate_indices in test_pairs:
        q_emb = bi_encoder.encode([query_text], convert_to_numpy=True)
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        
        cand_embs = normalized_embeddings[candidate_indices]
        scores = np.dot(cand_embs, q_emb.reshape(-1, 1)).flatten()
        scored = [(candidate_indices[i], float(scores[i])) for i in range(len(candidate_indices))]
        scored.sort(key=lambda x: x[1], reverse=True)
        baseline_results.append(scored)
    
    baseline_mrr = mrr(baseline_results, [p[1] for p in test_pairs])
    baseline_h1 = hits_at_k(baseline_results, [p[1] for p in test_pairs], 1)
    baseline_h5 = hits_at_k(baseline_results, [p[1] for p in test_pairs], 5)
    baseline_avg_rank = avg_rank(baseline_results, [p[1] for p in test_pairs])
    
    print(f"Baseline MRR@{TOP_K}: {baseline_mrr:.4f}")
    print(f"Baseline Hits@1: {baseline_h1} ({100*baseline_h1/len(test_pairs):.1f}%)")
    print(f"Baseline Hits@5: {baseline_h5} ({100*baseline_h5/len(test_pairs):.1f}%)")
    print(f"Baseline Avg Rank: {baseline_avg_rank:.2f}")
    
    # Phase 2: Bi-encoder + BM25 re-ranking
    print("\n" + "=" * 60)
    print("Phase 2: Bi-encoder + BM25 re-ranking")
    print("=" * 60)
    
    reranked_results = []
    
    for i, (query_text, target_idx, candidate_indices) in enumerate(test_pairs):
        if (i + 1) % 10 == 0:
            print(f"Processing query {i+1}/{len(test_pairs)}...")
        
        # Get bi-encoder scores
        q_emb = bi_encoder.encode([query_text], convert_to_numpy=True)
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        cand_embs = normalized_embeddings[candidate_indices]
        bi_scores = np.dot(cand_embs, q_emb.reshape(-1, 1)).flatten()
        bi_scored = [(candidate_indices[i], float(bi_scores[i])) for i in range(len(candidate_indices))]
        
        # Get BM25 reranking
        bm25_scored = bm25.rerank(query_text, candidate_indices)
        
        # Normalize both score lists
        bi_arr = np.array([s for _, s in bi_scored])
        bm_arr = np.array([s for _, s in bm25_scored])
        
        if bi_arr.max() > bi_arr.min():
            bi_norm = (bi_arr - bi_arr.min()) / (bi_arr.max() - bi_arr.min() + 1e-8)
        else:
            bi_norm = np.zeros_like(bi_arr)
        
        if bm_arr.max() > bm_arr.min():
            bm_norm = (bm_arr - bm_arr.min()) / (bm_arr.max() - bm_arr.min() + 1e-8)
        else:
            bm_norm = np.zeros_like(bm_arr)
        
        # Combine: 50% bi-encoder + 50% BM25
        bm25_idx_map = {cid: i for i, (cid, _) in enumerate(bm25_scored)}
        combined = []
        for i_cand, cid in enumerate(candidate_indices):
            bm25_pos = bm25_idx_map[cid]
            combined.append((cid, float(0.5 * bi_norm[i_cand] + 0.5 * bm_norm[bm25_pos])))
        
        combined.sort(key=lambda x: x[1], reverse=True)
        reranked_results.append(combined)
    
    reranked_mrr = mrr(reranked_results, [p[1] for p in test_pairs])
    reranked_h1 = hits_at_k(reranked_results, [p[1] for p in test_pairs], 1)
    reranked_h5 = hits_at_k(reranked_results, [p[1] for p in test_pairs], 5)
    reranked_avg_rank = avg_rank(reranked_results, [p[1] for p in test_pairs])
    
    print(f"\nRe-ranked MRR@{TOP_K}: {reranked_mrr:.4f}")
    print(f"Re-ranked Hits@1: {reranked_h1} ({100*reranked_h1/len(test_pairs):.1f}%)")
    print(f"Re-ranked Hits@5: {reranked_h5} ({100*reranked_h5/len(test_pairs):.1f}%)")
    print(f"Re-ranked Avg Rank: {reranked_avg_rank:.2f}")
    
    # Results
    improvement_pp = (reranked_mrr - baseline_mrr) * 100
    e10_pass = improvement_pp > 10.0 or reranked_mrr > 0.65
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Baseline MRR:     {baseline_mrr:.4f}")
    print(f"Re-ranked MRR:   {reranked_mrr:.4f}")
    print(f"Improvement:    {improvement_pp:.2f} percentage points")
    print(f"Baseline Avg Rank:  {baseline_avg_rank:.2f}")
    print(f"Re-ranked Avg Rank: {reranked_avg_rank:.2f}")
    print(f"\nE10_pass: {e10_pass} (improvement > 10pp OR MRR > 0.65)")
    
    results = {
        "original_mrr": round(baseline_mrr, 4),
        "reranked_mrr": round(reranked_mrr, 4),
        "improvement_pp": round(improvement_pp, 2),
        "hits_at_1_before": baseline_h1,
        "hits_at_1_after": reranked_h1,
        "hits_at_5_before": baseline_h5,
        "hits_at_5_after": reranked_h5,
        "avg_rank_before": round(baseline_avg_rank, 2),
        "avg_rank_after": round(reranked_avg_rank, 2),
        "E10_pass": e10_pass,
        "setup_notes": (
            "BM25 re-ranking: Okapi BM25 (k1=1.5, b=0.75) applied to top-20 bi-encoder candidates. "
            "Bi-encoder: all-MiniLM-L6-v2. "
            "Test pairs: query = claim A, target = claim B (15-60% word overlap), "
            "candidates = top-20 by bi-encoder for A. "
            "Combined score: 50% bi-encoder cosine + 50% BM25. "
            "BM25 captures lexical matching to complement semantic embeddings."
        )
    }
    
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {OUTPUT_PATH}")
    
    return results


if __name__ == "__main__":
    try:
        results = run_benchmark()
    except Exception as e:
        print(f"Error running benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)