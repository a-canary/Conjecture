# Conjecture Examples

Runnable quickstart scripts to get started with Conjecture.

## Prerequisites

```bash
# Install Conjecture
pip install -e .
```

## Quickstart Scripts

| Script | Description |
|--------|-------------|
| [01-ledger-quickstart.sh](01-ledger-quickstart.sh) | Core claim management commands |
| [02-provider-setup.sh](02-provider-setup.sh) | Configure LLM providers |
| [03-search-analysis.sh](03-search-analysis.sh) | Search and analyze claims |

## Running Examples

```bash
# Make scripts executable
chmod +x examples/*.sh

# Run the first quickstart
./examples/01-ledger-quickstart.sh

# Run provider setup (optional, for LLM features)
./examples/02-provider-setup.sh

# Run search examples (after creating some claims)
./examples/03-search-analysis.sh
```

## Order of Operations

1. **01-ledger-quickstart.sh** - Learn basic claim operations (create, stats, health)
2. **02-provider-setup.sh** - Configure an LLM provider for analysis features
3. **03-search-analysis.sh** - Search and analyze claims (run after creating claims)

## What's Next?

- Read the [full documentation](../docs/)
- Explore the [API reference](../src/)
- Check [CHOICES.md](../CHOICES.md) for configuration options