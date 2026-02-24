# Conjecture: Claim-Based Reasoning Harness

An agent harness that improves LLM reasoning, verifies assumptions, minimizes hallucinations, and handles complex tasks through claim-based knowledge management.

## Core Philosophy: Claims Are NOT Facts

All knowledge is provisional claims with confidence scores — never facts. Claims are impressions, assumptions, observations, and conjectures that may be wrong and subject to revision.

## What It Does

- **Claim Management**: Create, search, and analyze evidence-based claims
- **Confidence Scoring**: Track uncertainty with 0.0-1.0 confidence scores
- **Semantic Search**: Find relevant claims using natural language
- **Multi-Provider LLM**: Claude Agent SDK (primary), plus local/custom endpoints
- **Provenance Tracking**: Trace any assertion back through its evidence chain

## Quick Start

```bash
# Clone and install
git clone <repository-url>
cd Conjecture
pip install -r requirements.txt

# Run Conjecture
python conjecture

# Test setup
python conjecture validate
```

## Configuration

Conjecture uses hierarchical JSON configuration:

1. **Workspace**: `.conjecture/config.json`
2. **User**: `~/.conjecture/config.json`
3. **Default**: `src/config/default_config.json`

Claude Agent SDK handles Claude authentication. Custom OpenAI/Anthropic-compatible endpoints configured via JSON.

See [CHOICES.md](CHOICES.md) for all configuration choices.

## Architecture

**4-Layer Architecture**:
1. **Presentation** (`src/cli/`): User interaction (CLI, TUI, Web, MCP)
2. **Endpoint** (`src/endpoint/`): Public API
3. **Process** (`src/process/`): Intelligence and context building
4. **Data** (`src/data/`): Universal claim storage (SQLite + ChromaDB)

See [specs/architecture.md](specs/architecture.md) for details.

## Testing

```bash
# Run all tests
python -m pytest tests/ --cov=src

# By category
python -m pytest tests/ -m "unit"
python -m pytest tests/ -m "integration"
```

## Project Structure

```
Conjecture/
├── CHOICES.md           # Source of Plan — all project choices
├── CLAUDE.md            # Agent instructions
├── README.md            # This file
├── src/
│   ├── cli/             # Presentation layer
│   ├── endpoint/        # Public API
│   ├── process/         # Intelligence layer
│   └── data/            # Storage layer
├── specs/               # Specifications
├── docs/                # User guides
└── tests/               # Test suites
```

## Key Documentation

| Document | Purpose |
|----------|---------|
| [CHOICES.md](CHOICES.md) | Source of Plan — all project choices |
| [CLAUDE.md](CLAUDE.md) | Agent instructions |
| [specs/architecture.md](specs/architecture.md) | 4-layer architecture |
| [specs/requirements.md](specs/requirements.md) | System requirements |
| [docs/claims_philosophy.md](docs/claims_philosophy.md) | Claims philosophy |

## Contributing

1. Read [CHOICES.md](CHOICES.md) before proposing changes
2. Use `/choose-wisely` to modify project choices
3. Run tests before committing
4. Follow existing code patterns

## License

[Add license information here]

---

**Conjecture** — Making evidence-based reasoning accessible and transparent.
