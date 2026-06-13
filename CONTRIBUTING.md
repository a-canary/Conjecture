# Contributing to Conjecture

## Setup

```bash
git clone git@github.com:a-canary/Conjecture.git
cd Conjecture
pip install -r requirements.txt
python -m pytest tests/ --cov=src  # smoke test
```

## Dev Workflow

1. Read [CHOICES.md](CHOICES.md) before proposing changes
2. One small change at a time; test before commit
3. Use `/choose-wisely` to modify project choices
4. No mocking — test against real behavior

## Running Tests

```bash
python -m pytest tests/                        # all
python -m pytest tests/ -m "unit"              # unit only
python -m pytest tests/ -m "integration"        # integration only
```

## Filing Issues

Open an issue on GitHub. Note that issues and PRs may be triaged slowly (solo dev).

## Code Style

- Type hints throughout (Pydantic v2)
- Module-scoped loggers: `get_logger(__name__)`
- 85% test coverage minimum; no commit if coverage drops