# AGENTS.md - Guide for Working with Conjecture

This document provides essential information for AI agents working in the Conjecture codebase.

## Project Overview

Conjecture is an AI-Powered Evidence-Based Reasoning System that enables users to create, search, and analyze evidence-based claims using local or cloud AI providers.

**Architecture**: CLI Interface → Core Engine → Data Layer (SQLite + ChromaDB)

## Essential Docs
- ANALYSIS.md: most recent comprehensive analysis of tests and metrics, rewrite after each comprehensive test results
- RESULTS.md: After experiment, log the hypothesis and result in a single short paragraph. Use this to guide replanning 
- TODO.md: breakdown of remaining work to persist between agents

IMPORTANT: organize each feature development or experiement in a git branch based on "Base" Branch. When metrics and analysis improves, rebase "Base" to current work branch. If hypothesis or feature dev failed, revert to Base and retry or do next todo item.

## Essential Commands

### Running the Application
```bash
# Main entry point
python conjecture

# Test setup and configuration
python conjecture validate
python conjecture config
python conjecture backends

# Basic usage
python conjecture create "Your claim here" --confidence 0.95
python conjecture search "search terms"
python conjecture analyze c0000001
python conjecture stats
```

### Testing
```bash
# Run all tests (cross-platform)
./run_tests.sh          # Unix/Linux/macOS
./run_tests.bat          # Windows

# Or directly with pytest
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_core_tools.py
python -m pytest tests/test_data_layer.py
python -m pytest tests/test_basic_functionality.py
```

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configuration directory
mkdir -p ~/.conjecture
# Edit: ~/.conjecture/config.json
```

## Code Organization and Structure

### Key Directories
```
src/
├── cli/           # CLI interface and backends
├── core/          # Core models and utilities  
├── config/        # Configuration management
├── processing/    # LLM integration and evaluation
├── tools/         # Tool registry and management
├── data/          # Data layer components
└── utils/         # Utility functions
```

### Entry Points
- `conjecture` - Main script entry point (sets UTF-8 encoding, path)
- `src/cli/modular_cli.py` - Unified CLI with backend auto-detection
- `src/conjecture.py` - Core Conjecture class with async evaluation
- `src/core/models.py` - Single source of truth for data models

### Core Models
- **Claim**: Main data structure with states (Explore, Validated, Orphaned, Queued)
- **ClaimType**: Types (fact, concept, example, goal, reference, etc.)
- **ClaimScope**: Scopes (user-workspace, team-workspace, team-wide, public)
- **Config**: Unified configuration with provider settings

## Code Conventions and Patterns

### General Patterns
- **Async/await**: Used extensively for LLM interactions and processing
- **Pydantic models**: All data structures use Pydantic for validation
- **Type hints**: Full type annotations throughout
- **Error handling**: Custom exception classes for different error types
- **Logging**: Structured logging with appropriate levels

### CLI Patterns
- **Typer**: Used for CLI interface with rich markup
- **Rich**: Beautiful terminal output with progress indicators
- **Encoding**: UTF-8 environment setup for emoji/Unicode support
- **Cross-platform**: Windows (.bat) and Unix (.sh) scripts

### Backend Architecture
- **Provider pattern**: Pluggable LLM providers (local/cloud)
- **Auto-detection**: Intelligent backend selection
- **Bridge pattern**: LLMBridge for unified provider access
- **Repository pattern**: Clean data access layer

## Testing Approach

### Test Structure
- **pytest**: Primary testing framework
- **Markers**: Tests categorized with markers (unit, integration, performance, etc.)
- **Async tests**: pytest-asyncio for async code testing
- **Mock providers**: Mock LLM providers for reliable testing

### Key Test Files
- `test_basic_functionality.py` - Core functionality tests
- `test_data_layer.py` - SQLite and ChromaDB tests
- `test_core_tools.py` - Tool management tests
- `test_emoji.py` - Unicode/emoji support tests

### Test Configuration
- **pytest.ini**: Test discovery and configuration
- **Markers**: unit, integration, performance, slow, asyncio, models, sqlite, chroma
- **Timeout**: 300 seconds default
- **Coverage**: 80% minimum requirement (commented out)

## Important Gotchas and Non-Obvious Patterns

### Critical Setup Requirements
1. **PYTHONPATH**: Must include project root (`export PYTHONPATH=.`)
2. **UTF-8 Encoding**: Required for emoji support (set in environment and Rich console)
3. **Config Directory**: `~/.conjecture/` must exist with valid `config.json`
4. **Provider Availability**: At least one LLM provider must be configured

### Common Issues
- **"Provider not found"**: Check `~/.conjecture/config.json` and provider URLs
- **"Database locked"**: Ensure only one Conjecture instance is running
- **Import errors**: Run with correct PYTHONPATH or use entry script
- **Unicode issues**: Verify UTF-8 encoding is set before Rich console usage

### Recent Major Refactoring
The codebase underwent 87% complexity reduction:
- **Data models**: Consolidated from 40 to 5 classes
- **Context models**: Simplified for LLM processing
- **CLI commands**: All functional (create, get, search, analyze, prompt)
- **Configuration**: Unified ProviderConfig across the system

## Configuration

### Configuration

#### Config File Hierarchy
Conjecture uses a hierarchical configuration system:

1. **Workspace Config**: `{project}/.conjecture/config.json` (highest priority)
2. **User Config**: `~/.conjecture/config.json` (overrides default)
3. **Default Config**: `src/config/default_config.json` (fallback)

#### User Configuration Setup
```bash
# Create user config with your API keys
mkdir -p ~/.conjecture
cp src/config/default_config.json ~/.conjecture/config.json
# Then edit ~/.conjecture/config.json to add your API keys
```

#### Configuration Format
```json
{
  "providers": [
    {
      "url": "https://llm.chutes.ai/v1",
      "api": "cpk_your-api-key-here",
      "model": "zai-org/GLM-4.6-FP8",
      "name": "chutes"
    },
    {
      "url": "http://localhost:11434",
      "api": "",
      "model": "llama2",
      "name": "ollama"
    }
  ],
  "confidence_threshold": 0.95,
  "max_context_size": 10,
  "debug": false,
  "database_path": "data/conjecture.db",
  "user": "user",
  "team": "default"
}
```

#### Supported Providers
- **Local**: Ollama, LM Studio (no API keys required)
- **Cloud**: Chutes.ai, OpenRouter, OpenAI, Anthropic (API keys in config)

#### Configuration Migration
When upgrading from environment variables:
```bash
# Run migration script to convert .env to JSON config
python scripts/migrate_to_config.py

# Script will:
# - Create ~/.conjecture/config.json from env vars
# - Backup existing config to ~/.conjecture/config.json.backup
# - Preserve all settings and providers
```

## Development Notes

### When Making Changes
1. **Read first**: Always read files before editing them
2. **Exact matching**: Edit tool requires exact whitespace/indentation matches
3. **Test after**: Run tests immediately after modifications
4. **Cross-platform**: Consider Windows and Unix compatibility

### Complexity Prevention Rules
1. **No duplicate implementations**: Before creating new files, search existing code. If similar functionality exists, improve it instead of creating duplicates
2. **Keep single source of truth**: Only maintain the most recent/valuable version of each component. Archive or delete outdated versions immediately
3. **Post-feature cleanup**: After each feature git commit, perform a "purge-organize-rename-fixup" commit to remove duplicates, organize files, and fix references

### Key Dependencies
- **Pydantic 2.5.2**: Data validation and models
- **Typer 0.9.0+**: CLI framework
- **Rich 13.0.0+**: Terminal output
- **ChromaDB 0.4.15**: Vector storage
- **aiosqlite 0.19.0+**: Async SQLite
- **tenacity 8.2.0+**: Retry logic

### Performance Considerations
- **Async evaluation**: All LLM calls are async
- **Context collection**: Optimized for minimal context windows
- **Caching**: Results cached where appropriate
- **Resource management**: Proper cleanup of connections and resources

## Troubleshooting Commands

```bash
# Check configuration
python conjecture config

# Validate setup
python conjecture validate

# Test provider connectivity
python conjecture backends

# Verify database
ls -la data/conjecture.db

# Check environment
echo $PYTHONPATH
python -c "import sys; print(sys.path)"
```

## Security Notes

- **No API keys in repository**: All credentials via environment variables
- **Local providers**: Keep data completely private
- **Scope-based access**: User-workspace is most restrictive (default)
- **Input validation**: All inputs validated through Pydantic models

---

This document covers the essential patterns and conventions for working effectively in the Conjecture codebase. Always reference the actual source code for implementation details.