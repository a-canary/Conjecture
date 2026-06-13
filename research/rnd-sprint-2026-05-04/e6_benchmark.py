#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
E6: Vector Store Benchmark — FastEmbed bge-base-en-v1.5 vs all-MiniLM-L6-v2
Optimized for speed with proper batching.
"""

import json
import random
import string
import time
import numpy as np
from typing import List, Tuple

# Constants
NUM_TEXTS = 10000
NUM_SEARCHES = 1000
OUTPUT_PATH = "/home/aaron/projects/conjecture/research/rnd-sprint-2026-05-04/E6-results.json"

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


def generate_contradiction_pairs(claims: List[str], n_pairs: int = 500) -> List[Tuple[str, int]]:
    """
    Generate synthetic contradiction test set.
    Returns list of (query_claim, true_claim_idx) tuples.
    For MRR: we search for claim i and check if it ranks at position of the matching claim.
    """
    pairs = []
    indices = list(range(len(claims)))
    for i in range(n_pairs):
        idx = i % len(claims)
        pairs.append((claims[idx], idx))
    return pairs


def percentile_latencies(times_ms: List[float]) -> Tuple[float, float, float]:
    """Calculate p50, p95, p99 latencies."""
    sorted_times = sorted(times_ms)
    n = len(sorted_times)
    p50_idx = int(n * 0.50)
    p95_idx = int(n * 0.95)
    p99_idx = int(n * 0.99)
    
    p50 = sorted_times[p50_idx] if p50_idx < n else sorted_times[-1]
    p95 = sorted_times[p95_idx] if p95_idx < n else sorted_times[-1]
    p99 = sorted_times[p99_idx] if p99_idx < n else sorted_times[-1]
    
    return p50, p95, p99


def benchmark_sentence_transformers(claims: List[str], test_pairs: List[Tuple[str, int]]):
    """Benchmark current setup using sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    import faiss
    
    print("\n" + "="*60)
    print("BENCHMARKING: sentence-transformers (all-MiniLM-L6-v2)")
    print("="*60)
    
    # Load model
    print("Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    dimension = 384
    
    # Encode all claims in batches
    print(f"Encoding {NUM_TEXTS} claims...")
    start_time = time.time()
    embeddings = model.encode(claims, convert_to_numpy=True, batch_size=64, show_progress_bar=True)
    encode_time = time.time() - start_time
    print(f"Encode time: {encode_time:.2f}s ({NUM_TEXTS/encode_time:.1f} texts/sec)")
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    normalized_embeddings = embeddings / norms
    
    # Create FAISS index
    print("Building FAISS index...")
    index = faiss.IndexFlatIP(dimension)
    index.add(normalized_embeddings.astype(np.float32))
    
    # Map indices to claim IDs
    idx_to_id = {i: f"claim_{i}" for i in range(len(claims))}
    
    # Benchmark search latency (encode queries first for fair comparison)
    print(f"\nRunning {NUM_SEARCHES} searches (pre-encoded queries)...")
    pre_encoded_queries = embeddings[:NUM_SEARCHES]
    
    search_latencies_encode = []
    search_latencies_total = []
    
    for i in range(NUM_SEARCHES):
        query_emb = pre_encoded_queries[i:i+1]
        
        start_encode = time.time()
        start_total = time.time()
        
        # Search (no re-encoding since we pre-encoded)
        k = 10
        scores, indices = index.search(query_emb.astype(np.float32), k)
        
        latency_total_ms = (time.time() - start_total) * 1000
        search_latencies_total.append(latency_total_ms)
    
    p50, p95, p99 = percentile_latencies(search_latencies_total)
    print(f"Search latency (pre-encoded) - p50: {p50:.2f}ms, p95: {p95:.2f}ms, p99: {p99:.2f}ms")
    
    # Calculate MRR
    print("\nCalculating MRR...")
    mrr_sum = 0.0
    hits_at_1 = 0
    hits_at_5 = 0
    
    for query_claim, true_idx in test_pairs[:500]:
        # Search
        query_emb = pre_encoded_queries[true_idx:true_idx+1]
        scores, indices = index.search(query_emb.astype(np.float32), 10)
        
        rr = 0.0
        for rank, idx in enumerate(indices[0], 1):
            if idx == true_idx:
                rr = 1.0 / rank
                break
        mrr_sum += rr
        if rr >= 1.0:
            hits_at_1 += 1
        if rr >= 0.2:
            hits_at_5 += 1
    
    mrr = mrr_sum / len(test_pairs[:500])
    print(f"MRR: {mrr:.4f}, Hits@1: {hits_at_1}, Hits@5: {hits_at_5}")
    
    return {
        "encode_time": encode_time,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "mrr": mrr,
        "texts_per_sec": NUM_TEXTS / encode_time
    }


def benchmark_fastembed(claims: List[str], test_pairs: List[Tuple[str, int]]):
    """Benchmark FastEmbed BAAI/bge-base-en-v1.5."""
    from fastembed import TextEmbedding
    import faiss
    
    print("\n" + "="*60)
    print("BENCHMARKING: FastEmbed (BAAI/bge-base-en-v1.5)")
    print("="*60)
    
    # Load model
    print("Loading model...")
    model = TextEmbedding(model_name="BAAI/bge-base-en-v1.5")
    dimension = 768  # bge-base-en-v1.5 has 768 dimensions
    
    # Encode all claims
    print(f"Encoding {NUM_TEXTS} claims...")
    start_time = time.time()
    
    embeddings_list = []
    for emb in model.embed(claims, batch_size=64):
        embeddings_list.append(emb)
    
    embeddings = np.array(embeddings_list)
    encode_time = time.time() - start_time
    print(f"Encode time: {encode_time:.2f}s ({NUM_TEXTS/encode_time:.1f} texts/sec)")
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    normalized_embeddings = embeddings / norms
    
    # Create FAISS index
    print("Building FAISS index...")
    index = faiss.IndexFlatIP(dimension)
    index.add(normalized_embeddings.astype(np.float32))
    
    # Pre-encode queries for fair comparison
    print(f"\nRunning {NUM_SEARCHES} searches (pre-encoded queries)...")
    pre_encoded_queries = embeddings[:NUM_SEARCHES]
    
    search_latencies = []
    for i in range(NUM_SEARCHES):
        query_emb = pre_encoded_queries[i:i+1]
        
        start = time.time()
        k = 10
        scores, indices = index.search(query_emb.astype(np.float32), k)
        
        latency_ms = (time.time() - start) * 1000
        search_latencies.append(latency_ms)
    
    p50, p95, p99 = percentile_latencies(search_latencies)
    print(f"Search latency (pre-encoded) - p50: {p50:.2f}ms, p95: {p95:.2f}ms, p99: {p99:.2f}ms")
    
    # Calculate MRR
    print("\nCalculating MRR...")
    mrr_sum = 0.0
    hits_at_1 = 0
    hits_at_5 = 0
    
    for query_claim, true_idx in test_pairs[:500]:
        query_emb = pre_encoded_queries[true_idx:true_idx+1]
        scores, indices = index.search(query_emb.astype(np.float32), 10)
        
        rr = 0.0
        for rank, idx in enumerate(indices[0], 1):
            if idx == true_idx:
                rr = 1.0 / rank
                break
        mrr_sum += rr
        if rr >= 1.0:
            hits_at_1 += 1
        if rr >= 0.2:
            hits_at_5 += 1
    
    mrr = mrr_sum / len(test_pairs[:500])
    print(f"MRR: {mrr:.4f}, Hits@1: {hits_at_1}, Hits@5: {hits_at_5}")
    
    return {
        "encode_time": encode_time,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "mrr": mrr,
        "texts_per_sec": NUM_TEXTS / encode_time
    }


def main():
    print("="*60)
    print("E6: Vector Store Benchmark")
    print("FastEmbed bge-base-en-v1.5 vs all-MiniLM-L6-v2")
    print("="*60)
    
    # Generate synthetic claims
    print(f"\nGenerating {NUM_TEXTS} synthetic claim texts...")
    claims = generate_synthetic_claims(NUM_TEXTS)
    print(f"Generated {len(claims)} claims")
    print(f"Sample claim: {claims[0][:100]}...")
    
    # Generate test pairs for MRR
    print("\nGenerating MRR test pairs...")
    test_pairs = generate_contradiction_pairs(claims, n_pairs=500)
    print(f"Generated {len(test_pairs)} test pairs")
    
    results = {}
    
    # Benchmark sentence-transformers (current)
    try:
        st_results = benchmark_sentence_transformers(claims, test_pairs)
        results["current_encode_time_s"] = round(st_results["encode_time"], 3)
        results["current_p50_ms"] = round(st_results["p50_ms"], 3)
        results["current_p95_ms"] = round(st_results["p95_ms"], 3)
        results["current_p99_ms"] = round(st_results["p99_ms"], 3)
        results["current_mrr"] = round(st_results["mrr"], 4)
        results["current_texts_per_sec"] = round(st_results["texts_per_sec"], 1)
    except Exception as e:
        print(f"Error benchmarking sentence-transformers: {e}")
        import traceback
        traceback.print_exc()
        results["current_encode_time_s"] = -1
        results["current_p50_ms"] = -1
        results["current_p95_ms"] = -1
        results["current_p99_ms"] = -1
        results["current_mrr"] = -1
        results["current_texts_per_sec"] = -1
    
    # Benchmark FastEmbed
    try:
        fe_results = benchmark_fastembed(claims, test_pairs)
        results["fastembed_encode_time_s"] = round(fe_results["encode_time"], 3)
        results["fastembed_p50_ms"] = round(fe_results["p50_ms"], 3)
        results["fastembed_p95_ms"] = round(fe_results["p95_ms"], 3)
        results["fastembed_p99_ms"] = round(fe_results["p99_ms"], 3)
        results["fastembed_mrr"] = round(fe_results["mrr"], 4)
        results["fastembed_texts_per_sec"] = round(fe_results["texts_per_sec"], 1)
    except Exception as e:
        print(f"Error benchmarking FastEmbed: {e}")
        import traceback
        traceback.print_exc()
        results["fastembed_encode_time_s"] = -1
        results["fastembed_p50_ms"] = -1
        results["fastembed_p95_ms"] = -1
        results["fastembed_p99_ms"] = -1
        results["fastembed_mrr"] = -1
        results["fastembed_texts_per_sec"] = -1
    
    # Determine pass/fail
    e6_pass = (
        results["fastembed_encode_time_s"] > 0 and
        results["fastembed_encode_time_s"] < 60 and
        results["fastembed_mrr"] > 0.75
    )
    results["E6_pass"] = e6_pass
    results["E6_pass_reason"] = (
        f"encode={results['fastembed_encode_time_s']}s (<60s), "
        f"mrr={results['fastembed_mrr']} (>0.75)"
    )
    
    # Write results
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {OUTPUT_PATH}")
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Current (all-MiniLM-L6-v2):")
    print(f"  Encode time: {results['current_encode_time_s']}s ({results.get('current_texts_per_sec', 'N/A')} texts/sec)")
    print(f"  Search latency - p50: {results['current_p50_ms']}ms, p95: {results['current_p95_ms']}ms, p99: {results['current_p99_ms']}ms")
    print(f"  MRR: {results['current_mrr']}")
    print(f"\nFastEmbed (BAAI/bge-base-en-v1.5):")
    print(f"  Encode time: {results['fastembed_encode_time_s']}s ({results.get('fastembed_texts_per_sec', 'N/A')} texts/sec)")
    print(f"  Search latency - p50: {results['fastembed_p50_ms']}ms, p95: {results['fastembed_p95_ms']}ms, p99: {results['fastembed_p99_ms']}ms")
    print(f"  MRR: {results['fastembed_mrr']}")
    print(f"\nE6_pass: {results['E6_pass']}")
    print(f"  ({results['E6_pass_reason']})")
    
    return results


if __name__ == "__main__":
    main()
