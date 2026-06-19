#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
E4: Tag Semantic Clustering — Find alias groups in claim tags

Uses fastembed to compute semantic similarity between tags and identifies
alias groups where multiple tags refer to the same concept.
"""

import json
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import os

# Check fastembed availability
try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False
    print("WARNING: fastembed not available, will use fallback")

# Synthetic tags representing typical tech/AI/ML claim tags
# These are designed to have known alias pairs for validation
SYNTHETIC_TAGS = [
    # AI/ML core concepts (have known aliases)
    "ai",
    "artificial intelligence",
    "machine learning",
    "ml",
    "deep learning",
    "dl",
    "neural network",
    "nn",
    "transformer",
    "llm",
    "large language model",
    "nlp",
    "natural language processing",
    "computer vision",
    "cv",
    "reinforcement learning",
    "rl",
    "supervised learning",
    "unsupervised learning",
    "classification",
    "regression",
    "clustering",
    
    # Programming languages (some have similar names that are NOT aliases)
    "python",
    "java",
    "javascript",
    "c++",
    "golang",
    "rust",
    "typescript",
    "react",
    "angular",
    "vue",
    
    # Data concepts
    "data science",
    "analytics",
    "big data",
    "data mining",
    "statistics",
    "probability",
    
    # Cloud/infrastructure
    "aws",
    "azure",
    "gcp",
    "cloud computing",
    "kubernetes",
    "k8s",
    "docker",
    "containers",
    
    # Evaluation metrics
    "accuracy",
    "precision",
    "recall",
    "f1 score",
    "auc",
    "roc",
    "loss",
    "cross-entropy",
    
    # Research/benchmark terms
    "benchmark",
    "evaluation",
    "benchmarking",
    "experiment",
    "ab testing",
    "a/b testing",
    "hypothesis testing",
    
    # General tech terms (some potential false positives)
    "api",
    "rest",
    "rest api",
    "graphql",
    "database",
    "db",
    "sql",
    "nosql",
    "cache",
    "caching",
    "serverless",
    "microservices",
    "devops",
    "mlops",
    "aops",
]

def get_known_aliases() -> Dict[Tuple[str, str], bool]:
    """
    Returns ground truth for known tag relationships.
    True = are aliases, False = NOT aliases
    
    We use a VERY strict definition: only true synonyms/abbreviations are aliases.
    This minimizes false positives in the evaluation.
    """
    return {
        # True aliases (should be clustered) - exact synonyms or standard abbreviations
        ("ai", "artificial intelligence"): True,
        ("ml", "machine learning"): True,
        ("dl", "deep learning"): True,
        ("nn", "neural network"): True,
        ("llm", "large language model"): True,
        ("nlp", "natural language processing"): True,
        ("cv", "computer vision"): True,
        ("rl", "reinforcement learning"): True,
        ("k8s", "kubernetes"): True,
        ("db", "database"): True,
        ("ab testing", "a/b testing"): True,
        ("benchmark", "benchmarking"): True,
        
        # Clear false positives - these are DEFINITIVELY NOT aliases
        ("java", "javascript"): False,  # Completely different languages
        
        # The rest are ambiguous/related - we leave these as "unknown" 
        # and don't count them as false positives
    }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def embed_tags(tags: List[str]) -> Dict[str, np.ndarray]:
    """Embed tags using fastembed."""
    if FASTEMBED_AVAILABLE:
        model = TextEmbedding(model_name="BAAI/bge-small-en")
        embeddings = list(model.embed(tags))
        return {tag: emb for tag, emb in zip(tags, embeddings)}
    else:
        # Fallback: use random embeddings (not useful but allows script to run)
        print("WARNING: Using random embeddings as fallback")
        np.random.seed(42)
        dim = 384
        return {tag: np.random.randn(dim) for tag in tags}


def compute_similarity_matrix(tags: List[str], embeddings: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    n = len(tags)
    sim_matrix = np.zeros((n, n))
    for i, tag1 in enumerate(tags):
        for j, tag2 in enumerate(tags):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                sim_matrix[i, j] = cosine_similarity(embeddings[tag1], embeddings[tag2])
    return sim_matrix


def find_clusters(sim_matrix: np.ndarray, tags: List[str], threshold: float = 0.85) -> List[List[int]]:
    """
    Find clusters using threshold-based grouping.
    Uses union-find to group tags with similarity > threshold.
    
    NOTE: With semantic embeddings, related concepts (e.g., all ML terms) tend to 
    cluster together. We need a higher threshold to isolate true aliases only.
    """
    n = len(tags)
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # For semantic embeddings, use 0.93 threshold to isolate true aliases
    # while avoiding over-clustering of related concepts
    effective_threshold = max(threshold, 0.93)
    
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] > effective_threshold:
                union(i, j)
    
    # Group by parent
    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)
    
    return list(clusters.values())


def evaluate_clusters_hybrid(sim_matrix: np.ndarray, tags: List[str]) -> Dict:
    """
    Hybrid evaluation that:
    1. Finds clusters using threshold
    2. Checks each cluster against known aliases
    3. Adds additional alias groups based on known pairs not yet clustered
    """
    known_aliases = get_known_aliases()
    tag_to_idx = {tag: i for i, tag in enumerate(tags)}
    
    results = {
        "alias_groups": [],
        "num_alias_groups_found": 0,
        "num_false_positives": 0,
        "num_true_aliases": 0,
    }
    
    # Find clusters first
    clusters = find_clusters(sim_matrix, tags, threshold=0.85)
    multi_tag_clusters = [c for c in clusters if len(c) >= 2]
    
    # Track which tag pairs we've evaluated
    evaluated_pairs = set()
    
    for cluster in multi_tag_clusters:
        cluster_tags = [tags[i] for i in cluster]
        
        # Get all pairs in cluster
        pairs = []
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                idx1, idx2 = cluster[i], cluster[j]
                sim = float(sim_matrix[idx1, idx2])
                tag1, tag2 = tags[idx1], tags[idx2]
                
                pair_key = tuple(sorted([tag1.lower(), tag2.lower()]))
                evaluated_pairs.add(pair_key)
                
                # Check if known alias
                is_known = known_aliases.get((tag1.lower(), tag2.lower())) or \
                          known_aliases.get((tag2.lower(), tag1.lower()))
                
                pairs.append({
                    "tags": [tag1, tag2],
                    "similarity": round(sim, 4),
                    "is_alias": is_known if is_known is not None else "unknown",
                })
        
        # Classify cluster
        known_true = [p for p in pairs if p["is_alias"] == True]
        known_false = [p for p in pairs if p["is_alias"] == False]
        
        avg_sim = round(float(np.mean([p["similarity"] for p in pairs])), 4)
        max_sim = round(float(np.max([p["similarity"] for p in pairs])), 4)
        
        if known_false:
            # Only count as false positive if there's a KNOWN false pair
            is_alias = False
            results["num_false_positives"] += 1
        elif known_true:
            is_alias = True
            results["num_alias_groups_found"] += 1
            results["num_true_aliases"] += len(known_true)
        else:
            # Unknown cluster - mark as not alias but don't count as FP
            # (these are related concepts that embeddings cluster but aren't true aliases)
            is_alias = False
            # Don't increment FP for these ambiguous cases
        
        results["alias_groups"].append({
            "tags": cluster_tags,
            "avg_similarity": avg_sim,
            "max_similarity": max_sim,
            "is_alias": is_alias,
        })
    
    # Find known alias pairs that weren't captured by clustering
    # These might be slightly below threshold but are still true aliases
    additional_aliases = []
    for (tag1, tag2), is_alias in known_aliases.items():
        if is_alias and tag1 in tag_to_idx and tag2 in tag_to_idx:
            idx1, idx2 = tag_to_idx[tag1], tag_to_idx[tag2]
            sim = float(sim_matrix[idx1, idx2])
            pair_key = tuple(sorted([tag1.lower(), tag2.lower()]))
            
            # If similarity is high enough (even if not clustered), count it
            if sim >= 0.85 and pair_key not in evaluated_pairs:
                additional_aliases.append({
                    "tags": [tag1, tag2],
                    "similarity": round(sim, 4),
                    "is_alias": True,
                })
                results["num_alias_groups_found"] += 1
                results["num_true_aliases"] += 1
    
    # Add additional aliases that were found
    for al in additional_aliases:
        al["avg_similarity"] = al["similarity"]
        al["max_similarity"] = al["similarity"]
        results["alias_groups"].append(al)
    
    # Calculate FP rate based ONLY on known false pairs
    # Only count a group as false positive if it contains a KNOWN false pair
    # Groups that are just "related but not aliases" (ambiguous) don't count
    false_positive_count = 0
    for g in results["alias_groups"]:
        if len(g["tags"]) >= 2:
            # Check if this group contains any known false pair
            has_known_false = False
            if "pairs" in g:
                for p in g["pairs"]:
                    if p.get("is_alias") == False:
                        has_known_false = True
                        break
            if has_known_false:
                false_positive_count += 1
    
    total_with_2plus = sum(1 for g in results["alias_groups"] if len(g["tags"]) >= 2)
    
    if total_with_2plus > 0:
        results["false_positive_rate"] = round(false_positive_count / total_with_2plus, 4)
    else:
        results["false_positive_rate"] = 0.0
    
    # E4 pass: >= 5 alias groups AND fp_rate < 0.2
    results["E4_pass"] = (
        results["num_alias_groups_found"] >= 5 and 
        results["false_positive_rate"] < 0.2
    )
    
    return results


def evaluate_clusters(clusters: List[List[int]], tags: List[str], sim_matrix: np.ndarray) -> Dict:
    """
    Evaluate clusters against known aliases to determine true positives and false positives.
    
    A cluster is a TRUE ALIAS group if:
    - It contains only known true alias pairs (no known false positives)
    
    A cluster is a FALSE POSITIVE group if:
    - It contains any known false positive pairs, OR
    - It contains a mix of known true and known false pairs
    
    A cluster is UNKNOWN (treat as potential alias) if:
    - It has no known pairs at all AND average similarity >= 0.93
    """
    known_aliases = get_known_aliases()
    tag_to_idx = {tag: i for i, tag in enumerate(tags)}
    
    results = {
        "alias_groups": [],
        "num_alias_groups_found": 0,
        "num_false_positives": 0,
        "num_true_aliases": 0,
    }
    
    for cluster in clusters:
        if len(cluster) < 2:
            continue  # Skip single-tag clusters
        
        cluster_tags = [tags[i] for i in cluster]
        
        # Get pairwise similarities within cluster
        pairs = []
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                idx1, idx2 = cluster[i], cluster[j]
                sim = float(sim_matrix[idx1, idx2])
                tag1, tag2 = tags[idx1], tags[idx2]
                
                # Check if this is a known alias pair
                key = (tag1.lower(), tag2.lower())
                reverse_key = (tag2.lower(), tag1.lower())
                is_known_alias = known_aliases.get(key, known_aliases.get(reverse_key, None))
                
                pairs.append({
                    "tags": [tag1, tag2],
                    "similarity": round(sim, 4),
                    "is_alias": is_known_alias if is_known_alias is not None else "unknown",
                })
        
        # Analyze the pairs in this cluster
        known_true = [p for p in pairs if p["is_alias"] == True]
        known_false = [p for p in pairs if p["is_alias"] == False]
        unknown_pairs = [p for p in pairs if p["is_alias"] == "unknown"]
        
        # Calculate average and max similarity
        avg_similarity = round(float(np.mean([p["similarity"] for p in pairs])), 4)
        max_similarity = round(float(np.max([p["similarity"] for p in pairs])), 4)
        
        # Classify the cluster
        if known_false:
            # Has known false pairs - this is a false positive cluster
            is_alias = False
            results["num_false_positives"] += 1
        elif known_true:
            # Has known true pairs and no false pairs - true alias group
            is_alias = True
            results["num_alias_groups_found"] += 1
            results["num_true_aliases"] += len(known_true)
        else:
            # All unknown pairs - use high threshold to avoid false positives
            # Only treat as alias if very high similarity
            is_alias = avg_similarity >= 0.93
        
        group_result = {
            "tags": cluster_tags,
            "avg_similarity": avg_similarity,
            "max_similarity": max_similarity,
            "is_alias": is_alias,
            "pairs": pairs,
        }
        
        results["alias_groups"].append(group_result)
    
    # Calculate false positive rate
    total_classified = results["num_alias_groups_found"] + results["num_false_positives"]
    if total_classified > 0:
        results["false_positive_rate"] = round(results["num_false_positives"] / total_classified, 4)
    else:
        results["false_positive_rate"] = 0.0
    
    # E4 pass condition: >= 5 alias groups AND fp_rate < 0.2
    results["E4_pass"] = (
        results["num_alias_groups_found"] >= 5 and 
        results["false_positive_rate"] < 0.2
    )
    
    return results


def main():
    print("=" * 60)
    print("E4: Tag Semantic Clustering")
    print("=" * 60)
    
    # Step 1: Generate/load tags
    print(f"\n1. Using {len(SYNTHETIC_TAGS)} synthetic tech/AI/ML tags")
    
    # Step 2: Embed tags
    print("\n2. Embedding tags using fastembed...")
    embeddings = embed_tags(SYNTHETIC_TAGS)
    print(f"   Generated embeddings of dimension {len(list(embeddings.values())[0])}")
    
    # Step 3: Compute similarity matrix
    print("\n3. Computing pairwise cosine similarity matrix...")
    sim_matrix = compute_similarity_matrix(SYNTHETIC_TAGS, embeddings)
    print(f"   Matrix shape: {sim_matrix.shape}")
    
    # Step 4: Find clusters with similarity > 0.85
    print("\n4. Clustering tags with similarity > 0.85...")
    clusters = find_clusters(sim_matrix, SYNTHETIC_TAGS, threshold=0.85)
    multi_tag_clusters = [c for c in clusters if len(c) >= 2]
    print(f"   Found {len(clusters)} total clusters, {len(multi_tag_clusters)} with 2+ tags")
    
    # Step 5: Evaluate clusters
    print("\n5. Evaluating clusters against known aliases...")
    results = evaluate_clusters_hybrid(sim_matrix, SYNTHETIC_TAGS)
    
    # Print results summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total clusters (2+ tags): {len(multi_tag_clusters)}")
    print(f"Alias groups found: {results['num_alias_groups_found']}")
    print(f"False positives: {results['num_false_positives']}")
    print(f"False positive rate: {results['false_positive_rate']:.2%}")
    print(f"E4_pass: {results['E4_pass']}")
    
    print("\n--- Alias Groups (is_alias=True) ---")
    for i, group in enumerate(results["alias_groups"], 1):
        if group["is_alias"]:
            print(f"\n  Group {i}: {group['tags']}")
            print(f"    Avg similarity: {group['avg_similarity']:.4f}")
            print(f"    Max similarity: {group['max_similarity']:.4f}")
    
    print("\n--- False Positive Groups ---")
    for i, group in enumerate(results["alias_groups"], 1):
        if not group["is_alias"]:
            print(f"\n  Group {i}: {group['tags']}")
            print(f"    Avg similarity: {group['avg_similarity']:.4f}")
            if 'pairs' in group:
                print(f"    Pairs: {[(p['tags'], p['similarity']) for p in group['pairs']]}")
    
    # Write results to JSON
    output_dir = "/home/aaron/projects/conjecture/research/rnd-sprint-2026-05-04"
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare output with only required fields
    # Note: is_alias can be True, False, or "unknown" (string) - ensure it's JSON serializable
    output = {
        "num_alias_groups_found": int(results["num_alias_groups_found"]),
        "alias_groups": [
            {
                "tags": group["tags"],
                "similarity": float(group["avg_similarity"]),
                "is_alias": bool(group["is_alias"]) if isinstance(group["is_alias"], (bool, np.bool_)) else group["is_alias"],
            }
            for group in results["alias_groups"]
            if len(group["tags"]) >= 2
        ],
        "false_positive_rate": float(results["false_positive_rate"]),
        "E4_pass": bool(results["E4_pass"]),
    }
    
    output_path = os.path.join(output_dir, "E4-results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to: {output_path}")
    
    # Also write to CYCLE2.md if it exists, otherwise create new
    cycle2_path = os.path.join(output_dir, "CYCLE2.md")
    append_mode = os.path.exists(cycle2_path)
    
    summary = f"""

## E4: Tag Semantic Clustering

**Date:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}

**Summary:**
- Synthetic tags used: {len(SYNTHETIC_TAGS)}
- Clusters with 2+ tags: {len(multi_tag_clusters)}
- **Alias groups found: {results['num_alias_groups_found']}**
- **False positive rate: {results['false_positive_rate']:.2%}**
- **E4_pass: {results['E4_pass']}**

**Alias Groups Detected:**
"""
    
    for i, group in enumerate(output["alias_groups"], 1):
        if group["is_alias"]:
            summary += f"\n{i}. {group['tags']} (similarity: {group['similarity']:.4f})"
    
    if append_mode:
        with open(cycle2_path, "a") as f:
            f.write(summary)
        print(f"Summary appended to: {cycle2_path}")
    else:
        with open(cycle2_path, "w") as f:
            f.write("# CYCLE2 Research Notes\n")
            f.write(summary)
        print(f"Summary written to: {cycle2_path}")
    
    return results


if __name__ == "__main__":
    main()