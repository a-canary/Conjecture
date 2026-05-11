#!/usr/bin/env python3
"""Debug E6's MRR calculation."""
import random
import numpy as np
from sentence_transformers import SentenceTransformer

random.seed(42)
np.random.seed(42)

WORDS = [
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

def generate_random_text(min_len=50, max_len=500):
    target_len = random.randint(min_len, max_len)
    text, current_len = [], 0
    while current_len < target_len:
        word = random.choice(WORDS)
        text.append(word)
        current_len += len(word) + 1
    result = ' '.join(text)
    if result:
        result = result[0].upper() + result[1:]
        if not result.endswith('.'):
            result += '.'
    return result

claims = [generate_random_text() for _ in range(1000)]

def generate_contradiction_pairs(claims, n_pairs=200):
    pairs = []
    used_pairs = set()
    while len(pairs) < n_pairs:
        idx1, idx2 = random.sample(range(len(claims)), 2)
        key = tuple(sorted([idx1, idx2]))
        if key in used_pairs:
            continue
        used_pairs.add(key)
        pairs.append((claims[idx1], claims[idx2], idx2))
    return pairs

pairs = generate_contradiction_pairs(claims, 200)
print(f"First 5 true_idx = {[p[2] for p in pairs[:5]]}")

# Encode
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
embeddings = model.encode(claims, convert_to_numpy=True, show_progress_bar=False)
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norms[norms == 0] = 1e-8
norm_emb = embeddings / norms

# Check ranks
for qi, (q, t, ti) in enumerate(pairs[:10]):
    qe = model.encode([q], convert_to_numpy=True)
    qe = qe / (np.linalg.norm(qe) + 1e-8)
    scores = np.dot(norm_emb, qe.T).flatten()
    idx = np.argsort(scores)[::-1]
    rank = list(idx).index(ti) + 1
    print(f"Pair {qi}: true_idx={ti}, rank={rank}, top5={idx[:5].tolist()}")

# Compute MRR over all 200
mrr_sum = 0.0
for q, t, ti in pairs:
    qe = model.encode([q], convert_to_numpy=True)
    qe = qe / (np.linalg.norm(qe) + 1e-8)
    scores = np.dot(norm_emb, qe.T).flatten()
    idx = np.argsort(scores)[::-1]
    rank = list(idx).index(ti) + 1
    mrr_sum += 1.0 / rank
mrr = mrr_sum / len(pairs)
print(f"\nMRR@1000 (all): {mrr:.4f}")

# E6 searches k=10
mrr_sum10 = 0.0
hits1 = 0
for q, t, ti in pairs:
    qe = model.encode([q], convert_to_numpy=True)
    qe = qe / (np.linalg.norm(qe) + 1e-8)
    scores = np.dot(norm_emb, qe.T).flatten()
    idx = np.argsort(scores)[::-1][:10]
    if ti in idx:
        rank = list(idx).index(ti) + 1
        mrr_sum10 += 1.0 / rank
        if rank == 1:
            hits1 += 1
mrr10 = mrr_sum10 / len(pairs)
print(f"MRR@10: {mrr10:.4f}, Hits@1: {hits1} ({100*hits1/len(pairs):.1f}%)")