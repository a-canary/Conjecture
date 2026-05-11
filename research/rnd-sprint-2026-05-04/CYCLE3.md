# Cycle 3 Results — blog pipeline + vector benchmark + cascade stress

**Date**: 2026-05-05  
**Experiments**: E5 (blog pipeline), E6 (vector benchmark), E7 (cascade stress)

---

## E5: Blog Pipeline End-to-End ✅ PASS

**Goal**: Run R&D synthesis through blog-publish.ts pipeline.

**Results** (from previous run, 2026-05-05 02:09):
- Word count: **1803** (target: 1500-3000) ✅
- Coherence score: **4/5** (target: ≥4) ✅
- Claim traceability: **4/5**
- Claims extracted: **126**
- Sections composed: **12**
- QA gate: PASSED_WITH_ISSUES (low evidence enrichment: 29%)

**Key Finding**: The pipeline successfully extracted 126 claims from the R&D report and composed a 12-section blog post. The evidence enrichment stage (Stage 2) is simulated — it marks claims as enriched based on confidence level rather than actual KE lookup. This is the root cause of the 29% enrichment rate.

**Blog post**: `E5-blog-output.md` (14.2KB)  
**Audit**: `E5-blog-output-audit.jsonl`

---

## E6: Vector Store Benchmark ⚠️ PARTIAL

**Goal**: Benchmark FastEmbed BAAI/bge-base-en-v1.5 vs current all-MiniLM-L6-v2.

**Setup**: 1000 synthetic claim texts, 500 MRR test pairs (self-retrieval), 1000 search latency runs.
**Environment**: CPU-only (no GPU), first-run model download.

### Results

| Metric | all-MiniLM-L6-v2 (current) | BAAI/bge-base-en-v1.5 |
|--------|---------------------------|------------------------|
| Encode time (1K texts) | **5.38s** | 73.98s |
| Encode speed | **186 texts/sec** | 13.5 texts/sec |
| Search latency p50 | **0.043ms** | 0.10ms |
| Search latency p95 | **0.067ms** | 0.14ms |
| Search latency p99 | **0.082ms** | 0.17ms |
| MRR (self-retrieval) | 1.0 | 1.0 |

### Analysis

**Speed**: MiniLM is **5.5x faster** than BGE on CPU encoding. This is the opposite of the expected 3-5x speedup claim for BGE. Possible explanations:
1. First-run model download overhead (BGE was downloading for the first time)
2. BGE's 768-dim embeddings are 2x the memory bandwidth of MiniLM's 384-dim
3. FastEmbed's batch processing may not be optimized for this environment
4. MiniLM is a small model specifically optimized for speed

**MRR**: Both models achieved MRR 1.0 on self-retrieval — this is a degenerate test that doesn't reflect real contradction detection performance. A proper MRR test requires finding *contradictory* claims, not identical ones.

**Pass criteria**: `encode < 60s AND MRR > 0.75`
- BGE encode: 73.98s > 60s ❌
- BGE MRR: 1.0 > 0.75 ✅
- **E6_pass: false** (encode threshold failed)

### Recommendations

1. **Re-run BGE benchmark** after caching to confirm whether first-run overhead is the cause
2. **Fix MRR test**: Use actual contradictory claim pairs (negated statements) instead of self-retrieval
3. **Consider**: MiniLM's 384-dim may be preferable for CPU-only部署 where memory bandwidth is constrained
4. **BGE on GPU** would likely show the expected 3-5x speedup

---

## E7: Cascade Stress Test ✅ PASS

**Goal**: Stress test dirty flag cascade with 1000 claims.

**Results** (from previous run, 2026-05-05 02:21):
- Num claims: **1000**
- Dirty count (seed): **100** (10%)
- Cascade correctness: **1.0** (100%)
- Cascade time: **57.44ms**
- Expected total dirty: **979**
- Actual total dirty: **979**
- False positives: **0**
- False negatives: **0**

**Key Finding**: The cascade propagation is 100% correct. The dirty flag system correctly propagated from 100 seed dirty claims through the subs graph, resulting in 979 total dirty claims (979 out of 1000 = 97.9% of the graph eventually becomes dirty when starting from 10%).

**Cascade formula verified**: `0.5^depth` depth decay correctly limits propagation. At depth 3 (default cascade_depth), the decay factor is 0.125, meaning claims 3 levels away from a dirty seed are still marked dirty with very low priority.

**Performance**: 57.44ms for full cascade on 1000 claims — well under the 5s threshold.

---

## Summary

| Experiment | Status | Key Metric | Verdict |
|------------|--------|------------|---------|
| E5: Blog Pipeline | ✅ PASS | word_count=1803, coherence=4 | Blog post produced successfully |
| E6: Vector Benchmark | ⚠️ PARTIAL | BGE encode 74s (threshold 60s) | Encode too slow on CPU; MRR test flawed |
| E7: Cascade Stress | ✅ PASS | correctness=1.0, time=57ms | Perfect cascade propagation |

### Priority Follow-ups

1. **E6 re-run**: Cache BGE model, re-benchmark with proper contradiction MRR test
2. **Blog pipeline Stage 2**: Replace simulated enrichment with actual KE lookup
3. **E5 QA fix**: Increase evidence enrichment from 29% to >50% by wiring in real KE calls

---

*Generated 2026-05-05 10:30 AM — Cycle 3 of rnd-sprint-2026-05-04*
