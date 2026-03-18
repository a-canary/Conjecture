# Conjecture — Agent Instructions

## Core Philosophy

Claims are NOT facts. All knowledge is provisional with confidence scores, subject to revision.
This project implements a claim-based reasoning system and benchmarks LLM prompt architectures.

## Essential Commands

```bash
# Tests
.venv/bin/python -m pytest tests/ -m "unit" -v
.venv/bin/python -m pytest tests/ --cov=src --cov-report=term-missing

# Validation
python conjecture validate
python conjecture stats

# Benchmarks (100 samples, ~30-40 min, run in background)
.venv/bin/python experiments/gsm8k_standard_benchmark.py -n 100
.venv/bin/python experiments/bbh_benchmark.py -n 100

# Check results
ls -lt experiments/results/*.json | head -5
.venv/bin/python experiments/analyze_benchmark_results.py
```

Always use `.venv/bin/python`, not `python`.

## Architecture

4-layer: Presentation (`src/cli/`) → Endpoint (`src/endpoint/`) → Process (`src/process/`) → Data (`src/data/`)

Core models: `Claim`, `ClaimType` (CONCEPT/REFERENCE/THESIS/SKILL), `ClaimState` (EXPLORE/VALIDATED/ORPHANED/QUEUED)

## Key Findings (as of 2026-03)

- Three-prompt architecture: +10pp on BBH (p=0.018), equivalent on GSM8K (p=0.695)
- Task-type routing required: decomposition helps hard reasoning, hurts recall/commonsense
- Multi-model validation still needed (all results are DeepSeek V3 only)
- Validation experiments H1/H2/H3 started 2026-03-09, status unknown — check `.director/state.json`

## Statistical Requirements

- n≥100 for production claims; n=10-20 for exploration only
- Always report p-values and 95% CI
- p<0.05 for claims, p<0.01 for strong claims
- Bonferroni correction for multiple hypotheses

## Choices & Planning

- CHOICES.md is the source of truth — read before proposing direction changes
- Use `/choose-wisely` to modify (never edit CHOICES.md directly)
- PLAN.md tracks implementation; NEXT.md tracks backlog
