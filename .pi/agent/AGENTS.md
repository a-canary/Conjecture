# Conjecture — Project Guidelines

## Scientific Rigour (non-negotiable)

- n≥100 samples for any production or published benchmark claim
- Always calculate p-values before claiming improvement or regression
- Report 95% CI alongside point estimates
- Apply Bonferroni correction when testing multiple hypotheses simultaneously
- Single-model results (e.g. DeepSeek V3) are exploratory — state model explicitly in every result
- Negative results are valid outcomes — report them honestly

## Choices Governance

- Read CHOICES.md before any change affecting project direction
- Modify choices only via `/choose-wisely` (triggers cascading review)
- Never edit CHOICES.md directly
- Contradictions between a spec doc and CHOICES.md: choice wins, spec is updated

## Development Rules

- Always use `.venv/bin/python`, not bare `python`
- Run unit tests before committing: `pytest tests/ -m "unit"`
- Background benchmarks take 30-40 min — do not poll; check result files instead
- FAISS + SQLite only (ChromaDB rejected: heavy deps)
- No worktree creation without documenting purpose in git commit

## Known Working Patterns

- Core reasoning enhancements (prompt system): 5/5 success rate
- Three-prompt architecture: validated for hard reasoning (BBH +10pp p=0.018)
- Task-type routing: required for production (decomposition hurts recall tasks)

## Known Failing Patterns (do not retry)

- Surface-level prompt changes (formatting, confidence tuning): 0/3 success
- Knowledge infrastructure changes without core reasoning basis

## Troubleshooting

- LM Studio connection errors: use `curl` subprocess, not `requests`/`httpx`
- HuggingFace legacy datasets (piqa, social_i_qa, winogrande): use alternatives
- SSH not configured in container: commits accumulate locally, push via host
