#!/usr/bin/env python3
"""
E10: Cross-Encoder Re-Ranking Benchmark

Improves vector search MRR from 0.53 to 0.70+ by re-ranking top-20 candidates
with a cross-encoder (cross-encoder/ms-marco-MiniLM-L6-v2).

This benchmark follows E6's methodology but focuses on the gap between 
Hits@1 (26%) and Hits@5 (96%). The cross-encoder should help improve 
rank-1 accuracy by better evaluating query-document relevance.
"""

import json
import random
import time
import numpy as np
from typing import List, Tuple, Dict, Any
import sys

# Constants
NUM_TEXTS = 10000
NUM_QUERIES = 50
TOP_K = 20
OUTPUT_PATH = "/home/aaron/projects/conjecture/research/rnd-sprint-2026-05-04/E10-results.json"

# Seed for reproducibility
random.seed(42)
np.random.seed(42)


def generate_random_text(min_len: int = 50, max_len: int = 500) -> str:
    """Generate a random sentence of random length."""
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
    """Generate n synthetic claim texts."""
    return [generate_random_text() for _ in range(n)]


def generate_test_pairs_v2(claims: List[str], n_pairs: int) -> List[Tuple[str, int, str]]:
    """
    Generate test pairs where query is similar but not identical to target.
    This creates the scenario where bi-encoder finds the neighborhood
    but cross-encoder is needed to identify the correct match.
    
    Returns: (query_claim, target_idx, original_claim)
    """
    pairs = []
    
    # Pre-compute embeddings would help but we'll use word overlap heuristics
    for _ in range(n_pairs):
        # Pick a random claim as base
        base_idx = random.randint(0, len(claims) - 1)
        base_claim = claims[base_idx]
        
        # Find claims with significant word overlap (30-60%) to simulate
        # "related but not identical" documents
        base_words = set(base_claim.lower().split())
        best_target_idx = None
        best_overlap = 0
        
        # Sample 100 candidates
        candidates = random.sample([i for i in range(len(claims)) if i != base_idx], 
                                  min(100, len(claims) - 1))
        for cand_idx in candidates:
            cand_words = set(claims[cand_idx].lower().split())
            intersection = len(base_words & cand_words)
            union = len(base_words | cand_words)
            jaccard = intersection / union if union > 0 else 0
            
            # We want moderate overlap (related documents)
            if 0.15 < jaccard < 0.5:
                best_overlap = jaccard
                best_target_idx = cand_idx
                break
        
        if best_target_idx is None:
            # Fallback - just pick a random one
            best_target_idx = random.choice(candidates)
        
        # Now pick the query - it's the base claim, but target is the related claim
        # Query = base claim (which we search for)
        # Target = best_target_idx (the related claim we want to find)
        query_claim = base_claim
        target_claim = claims[best_target_idx]
        
        pairs.append((query_claim, best_target_idx, target_claim))
    
    return pairs


def cosine_search(embeddings: np.ndarray, query_emb: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pure numpy cosine similarity search."""
    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)
    
    scores = np.dot(embeddings, query_emb.T).flatten()
    top_k_indices = np.argsort(scores)[::-1][:k]
    top_k_scores = scores[top_k_indices]
    
    return top_k_scores, top_k_indices


def mean_reciprocal_rank(results: List[List[Tuple[int, float]]], 
                         true_indices: List[int]) -> float:
    """Calculate Mean Reciprocal Rank."""
    reciprocal_ranks = []
    for result_list, true_idx in zip(results, true_indices):
        rr = 0.0
        for rank, (claim_id, score) in enumerate(result_list, 1):
            if claim_id == true_idx:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def hits_at_k(results: List[List[Tuple[int, float]]], 
              true_indices: List[int], k: int) -> int:
    """Count how many times the true item appears in top-k."""
    hits = 0
    for result_list, true_idx in zip(results, true_indices):
        for rank, (claim_id, score) in enumerate(result_list, 1):
            if rank > k:
                break
            if claim_id == true_idx:
                hits += 1
                break
    return hits


def benchmark_cross_encoder_reranking():
    """Main benchmark comparing bi-encoder vs bi-encoder + cross-encoder re-ranking."""
    from sentence_transformers import SentenceTransformer, CrossEncoder
    
    print("=" * 60)
    print("E10: Cross-Encoder Re-Ranking Benchmark")
    print("=" * 60)
    
    # Generate synthetic claims
    print(f"\nGenerating {NUM_TEXTS} synthetic claim texts...")
    claims = generate_synthetic_claims(NUM_TEXTS)
    print(f"Generated {len(claims)} claims")
    
    # Generate test pairs
    print(f"\nGenerating {NUM_QUERIES} test queries (related but not identical targets)...")
    test_pairs = generate_test_pairs_v2(claims, NUM_QUERIES)
    print(f"Generated {len(test_pairs)} test pairs")
    
    # =========================================================================
    # Load bi-encoder model
    # =========================================================================
    print("\n" + "=" * 60)
    print("Loading bi-encoder model (all-MiniLM-L6-v2)...")
    print("=" * 60)
    bi_encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    dimension = 384
    
    # Encode all claims
    print(f"Encoding {NUM_TEXTS} claims with bi-encoder...")
    start_time = time.time()
    embeddings = bi_encoder.encode(claims, convert_to_numpy=True, show_progress_bar=True)
    encode_time = time.time() - start_time
    print(f"Bi-encoder encode time: {encode_time:.2f}s")
    
    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    normalized_embeddings = embeddings / norms
    
    # =========================================================================
    # Load cross-encoder model
    # =========================================================================
    print("\n" + "=" * 60)
    print("Loading cross-encoder model (cross-encoder/ms-marco-MiniLM-L6-v2)...")
    print("=" * 60)
    
    cross_encoder_loaded = False
    cross_encoder = None
    
    try:
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2', device='cpu')
        cross_encoder_loaded = True
        print("Cross-encoder loaded successfully!")
    except Exception as e:
        print(f"Failed to load cross-encoder: {e}")
    
    # =========================================================================
    # Phase 1: Bi-encoder only retrieval (baseline)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Phase 1: Bi-encoder only retrieval (baseline)")
    print("=" * 60)
    
    baseline_results = []
    baseline_mrr_sum = 0.0
    
    for query_claim, true_idx, _ in test_pairs:
        # Encode query
        query_emb = bi_encoder.encode([query_claim], convert_to_numpy=True)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        
        # Search top-20 using numpy cosine similarity
        scores, indices = cosine_search(normalized_embeddings, query_emb, TOP_K)
        
        # Build result list
        result_list = [(int(idx), float(scores[i])) 
                       for i, idx in enumerate(indices)]
        baseline_results.append(result_list)
        
        # Calculate RR for this query
        rr = 0.0
        for rank, (claim_id, score) in enumerate(result_list, 1):
            if claim_id == true_idx:
                rr = 1.0 / rank
                break
        baseline_mrr_sum += rr
    
    baseline_mrr = baseline_mrr_sum / len(test_pairs)
    baseline_hits_at_1 = hits_at_k(baseline_results, [p[1] for p in test_pairs], 1)
    baseline_hits_at_5 = hits_at_k(baseline_results, [p[1] for p in test_pairs], 5)
    baseline_hits_at_20 = hits_at_k(baseline_results, [p[1] for p in test_pairs], 20)
    
    print(f"Baseline MRR@{TOP_K}: {baseline_mrr:.4f}")
    print(f"Baseline Hits@1: {baseline_hits_at_1} ({100*baseline_hits_at_1/len(test_pairs):.1f}%)")
    print(f"Baseline Hits@5: {baseline_hits_at_5} ({100*baseline_hits_at_5/len(test_pairs):.1f}%)")
    print(f"Baseline Hits@20: {baseline_hits_at_20} ({100*baseline_hits_at_20/len(test_pairs):.1f}%)")
    
    # =========================================================================
    # Phase 2: Bi-encoder + Cross-encoder re-ranking
    # =========================================================================
    print("\n" + "=" * 60)
    print("Phase 2: Bi-encoder + Cross-encoder re-ranking")
    print("=" * 60)
    
    reranked_results = []
    reranked_mrr_sum = 0.0
    
    for i, (query_claim, true_idx, target_claim) in enumerate(test_pairs):
        if (i + 1) % 10 == 0:
            print(f"Processing query {i+1}/{len(test_pairs)}...")
        
        # Encode query
        query_emb = bi_encoder.encode([query_claim], convert_to_numpy=True)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        
        # Search top-20 candidates using bi-encoder
        scores, indices = cosine_search(normalized_embeddings, query_emb, TOP_K)
        
        # Prepare candidate documents for cross-encoder
        candidate_indices = list(indices)
        candidate_claims = [claims[int(idx)] for idx in candidate_indices]
        
        if cross_encoder_loaded and cross_encoder is not None:
            # Use cross-encoder to score query-document pairs
            query_doc_pairs = [(query_claim, doc) for doc in candidate_claims]
            cross_scores = cross_encoder.predict(query_doc_pairs)
            
            # Normalize bi-encoder scores
            bi_scores = scores
            bi_scores_normalized = (bi_scores - bi_scores.min()) / (bi_scores.max() - bi_scores.min() + 1e-8)
            # Normalize cross-encoder scores
            cross_scores_normalized = (cross_scores - cross_scores.min()) / (cross_scores.max() - cross_scores.min() + 1e-8)
            
            # Combine: 80% cross-encoder, 20% bi-encoder
            combined_scores = 0.8 * cross_scores_normalized + 0.2 * bi_scores_normalized
            
            # Sort by combined scores
            sorted_pairs = sorted(
                zip(candidate_indices, combined_scores),
                key=lambda x: x[1],
                reverse=True
            )
        else:
            # Fallback: use bi-encoder scores only
            sorted_pairs = sorted(
                zip(candidate_indices, scores),
                key=lambda x: x[1],
                reverse=True
            )
        
        reranked_results.append(sorted_pairs)
        
        # Calculate RR for this query
        rr = 0.0
        for rank, (claim_id, score) in enumerate(sorted_pairs, 1):
            if claim_id == true_idx:
                rr = 1.0 / rank
                break
        reranked_mrr_sum += rr
    
    reranked_mrr = reranked_mrr_sum / len(test_pairs)
    reranked_hits_at_1 = hits_at_k(reranked_results, [p[1] for p in test_pairs], 1)
    reranked_hits_at_5 = hits_at_k(reranked_results, [p[1] for p in test_pairs], 5)
    reranked_hits_at_20 = hits_at_k(reranked_results, [p[1] for p in test_pairs], 20)
    
    print(f"\nRe-ranked MRR@{TOP_K}: {reranked_mrr:.4f}")
    print(f"Re-ranked Hits@1: {reranked_hits_at_1} ({100*reranked_hits_at_1/len(test_pairs):.1f}%)")
    print(f"Re-ranked Hits@5: {reranked_hits_at_5} ({100*reranked_hits_at_5/len(test_pairs):.1f}%)")
    print(f"Re-ranked Hits@20: {reranked_hits_at_20} ({100*reranked_hits_at_20/len(test_pairs):.1f}%)")
    
    # =========================================================================
    # Calculate improvement
    # =========================================================================
    improvement_pp = (reranked_mrr - baseline_mrr) * 100
    e10_pass = improvement_pp > 15.0 or reranked_mrr > 0.70
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Baseline MRR:   {baseline_mrr:.4f}")
    print(f"Re-ranked MRR:  {reranked_mrr:.4f}")
    print(f"Improvement:    {improvement_pp:.2f} percentage points")
    print(f"Baseline Hits@1:  {baseline_hits_at_1} ({100*baseline_hits_at_1/len(test_pairs):.1f}%)")
    print(f"Re-ranked Hits@1: {reranked_hits_at_1} ({100*reranked_hits_at_1/len(test_pairs):.1f}%)")
    print(f"\nE10_pass: {e10_pass} (improvement > 15pp OR MRR > 0.70)")
    
    # =========================================================================
    # Write results
    # =========================================================================
    setup_notes = "Used cross-encoder/ms-marco-MiniLM-L6-v2 for re-ranking. " \
                  "Combined scores: 80% cross-encoder + 20% bi-encoder cosine similarity. " \
                  f"Cross-encoder loaded: {cross_encoder_loaded}. " \
                  "Test pairs: query is original claim, target is related claim with 15-50% word overlap."
    
    results = {
        "original_mrr": round(baseline_mrr, 4),
        "reranked_mrr": round(reranked_mrr, 4),
        "improvement_pp": round(improvement_pp, 2),
        "hits_at_1_before": baseline_hits_at_1,
        "hits_at_1_after": reranked_hits_at_1,
        "hits_at_5_before": baseline_hits_at_5,
        "hits_at_5_after": reranked_hits_at_5,
        "E10_pass": e10_pass,
        "setup_notes": setup_notes
    }
    
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {OUTPUT_PATH}")
    
    return results


if __name__ == "__main__":
    try:
        results = benchmark_cross_encoder_reranking()
    except Exception as e:
        print(f"Error running benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)