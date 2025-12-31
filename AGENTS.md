# AGENTS.md - Guide for Working with Conjecture

This document provides essential information for AI agents working in the Conjecture codebase.

## Project Overview

Conjecture is an AI-Powered Evidence-Based Reasoning System that enables users to create, search, and analyze evidence-based claims using local or cloud AI providers.

**Architecture**: 4-Layer Architecture (Presentation → Endpoint → Process → Data) - see specs/architecture.md

## 🚀 Code-Test-Commit Workflow (MANDATORY)

### **ESTABLISHED 2025-12-07 - ALL AGENTS MUST FOLLOW**

> [!IMPORTANT]
> **Current Strategic Focus**
> 1. **Deep Code Analysis**: Fix root causes, not symptoms. Do not patch over issues with `try/except` or fake mocks.
> 2. **Dead Code Removal**: Aggressively verify and delete unreachable or duplicate code.
> 3. **Test Rationalization**: Quality over quantity. Remove inflated, repetitive, or misleading tests.
> 4. **Deceptive Pattern Removal**: Eliminate silent failures and "safety" blocks that hide actual bugs.

#### **Core Workflow Rules**
1. **One Small Bug/Feature at a Time** - Focus on single, manageable changes
2. **Test Thoroughly** - Run comprehensive tests before each commit
3. **Update RESULTS.md** - Document all test results, benchmark metrics, and repo size
4. **Justify Repo Size Increases** - Every commit must justify any project size growth
5. **Quality Verification** - Ensure project quality before each commit

#### **Agent-Specific Requirements**
- **Before Any Code Changes**: Read current RESULTS.md and .agent/backlog.md
- **During Development**: Follow pre-commit checklist religiously
- **Before Commit**: Update RESULTS.md with current metrics and analysis
- **After Commit**: Verify all quality gates passed

#### **Quality Gates**
- **Blocking Issues**: Test failures, coverage <85%, performance regressions, security issues
- **Warning Conditions**: Test flakiness, minor performance impact, documentation gaps
- **Commit Readiness**: All blocking issues resolved, warnings documented

## Essential Docs
- ANALYSIS.md: most recent comprehensive analysis of tests, metrics, and benchmarks to use a baseline to compare WIP changes to before committing, rewrite after each comprehensive test results, git diff to compare to baseline
- RESULTS.md: log of past work
- .agent/backlog.md: backlog of future work

CRITICAL: ALL AGENTS must follow the Code-Test-Commit workflow rules documented in .agent/backlog.md. No exceptions.

## Essential Commands

### Running the Application
```bash
# Main entry point
python conjecture

# Configuration and setup
python conjecture config          # Show configuration status
python conjecture providers       # Show available providers
python conjecture setup           # Interactive provider setup
python conjecture backends        # Show backend status
python conjecture health          # System health check
python conjecture quickstart      # Quick start guide

# Basic usage
python conjecture create "Your claim here" --confidence 0.95
python conjecture get c0000001    # Retrieve specific claim
python conjecture search "search terms"
python conjecture analyze c0000001
python conjecture stats            # Database statistics
python conjecture prompt "Your prompt here"  # Process as claim with context
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
python -m pytest tests/test_comprehensive_metrics.py
```

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configuration directory (auto-created by setup)
mkdir -p ~/.conjecture
# Edit: ~/.conjecture/config.json

# Validate configuration
python conjecture config
```

### ConjectureEndpoint Management
```bash
# CRITICAL: ALWAYS use the restart_endpoint.bat script for ConjectureEndpoint
# NEVER run python run_endpoint_app.py directly
./restart_endpoint.bat     # Windows (REQUIRED)
./restart_endpoint.sh     # Unix/Linux/macOS

# The bat script handles:
# - Proper process termination
# - Port cleanup
# - Environment setup
# - Error handling
# - Unicode encoding issues
```

## Code Organization and Structure

### Key Directories
```
src/
├── agent/                 # Agent coordination and harness systems
├── cli/                   # CLI interface and backends
│   ├── backends/         # Backend implementations (cloud, local)
│   └── encoding_handler.py # UTF-8 encoding support
├── config/                # Configuration management
│   ├── default_config.json # Default configuration template
│   └── pydantic_config.py # Pydantic-based configuration
├── core/                  # Core models and utilities
├── context/               # Context building and management
├── interfaces/            # User interfaces (TUI, GUI)
├── llm/                   # LLM instruction processing
├── local/                 # Local model management
├── modes/                 # Specialized operation modes
├── monitoring/            # Performance and metrics monitoring
├── processing/            # LLM integration and evaluation
│   ├── llm/             # LLM adapters and providers
│   ├── llm_prompts/     # Template management
│   └── support_systems/  # Processing support systems
├── providers/            # External provider implementations
├── tools/               # Tool registry and management
└── utils/               # Utility functions

archive/                  # Archived documentation and experiments
├── documentation/        # Historical documentation (65% reduction)
└── experiments/         # Archived experiment files

docs/                     # Current documentation
├── architecture/         # System architecture documentation
├── configuration/        # Setup and configuration guides
├── examples/            # Usage examples
├── reference/           # API and reference documentation
└── tutorials/           # User tutorials

experiments/              # Active research and experiments
research/                # Research scripts and analysis
scripts/                 # Utility and automation scripts
tests/                   # Comprehensive test suite
├── evaluation_config/    # Test evaluation configurations
├── examples/            # Test examples
├── reports/             # Test reports and results
└── results/             # Test execution results
```

### Entry Points
- `conjecture` - Main script entry point (sets UTF-8 encoding, path)
- `src/cli/modular_cli.py` - Layer 1: Presentation Layer CLI interface
- `src/endpoint/conjecture_endpoint.py` - Layer 2: ConjectureEndpoint class (public API)
- `src/process/context_builder.py` - Layer 3: Context building and management
- `src/process/llm_processor.py` - Layer 3: LLM instruction processing
- `src/data/claim_model.py` - Layer 4: Universal Claim Model
- `src/core/models.py` - Single source of truth for data models
- `src/config/unified_config.py` - Unified configuration system
- `restart_endpoint.bat` - (re)start the ConjectureEndpoint service

### Core Models (Layer 4: Data Layer)
- **Claim**: Universal Claim Model with bidirectional relationships
  - `id`: Unique ID
  - `content`: Make it short, make it clear
  - `confidence`: 0.0 - 1.0
  - `state`: ClaimState (Explore, Validated, Orphaned, Queued)
  - `type`: List[ClaimType] (Concept, Thesis, etc.)
  - `tags`: List[str] (Organized via Tag Lifecycle)
  - `supports`: List[str] (Claims this claim supports - Upward)
  - `supported_by`: List[str] (Claims that support this claim - Downward)
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

### Architecture Patterns
- **4-Layer Architecture**: Clear separation of concerns (Presentation → Endpoint → Process → Data)
- **Provider pattern**: Pluggable LLM providers (local/cloud)
- **Auto-detection**: Intelligent backend selection
- **Bridge pattern**: LLMBridge for unified provider access
- **Repository pattern**: Clean data access layer
- **Universal Claim Model**: Single data structure across all layers

## Testing Approach

### Test Structure
- **pytest**: Primary testing framework
- **Markers**: Tests categorized with markers (unit, integration, performance, etc.)
- **Async tests**: pytest-asyncio for async code testing
- **Ban Mock**: Mocking is banned to ensure real-world testing

### Key Test Files
- `test_basic_functionality.py` - Core functionality tests
- `test_data_layer.py` - SQLite and ChromaDB tests
- `test_core_tools.py` - Tool management tests
- `test_comprehensive_metrics.py` - Comprehensive metrics and analysis tests
- `test_emoji.py` - Unicode/emoji support tests
- `test_cli_comprehensive.py` - Full CLI integration tests
- `test_processing_comprehensive.py` - Processing layer tests
- `test_unified_config_comprehensive.py` - Configuration system tests

### Test Configuration
- **pytest.ini**: Test discovery and configuration
- **Markers**: unit, integration, performance, slow, asyncio, models, sqlite, chroma
- **Timeout**: 300 seconds default
- **Coverage**: 80% minimum requirement (achieved 89% coverage)
- **Test Reports**: Automated test reports in `tests/reports/` directory
- **Test Results**: Detailed results stored in `tests/results/` directory

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
The codebase underwent significant refactoring through 2025:
- **4-Layer Architecture**: Implemented clean separation of concerns (Presentation → Endpoint → Process → Data)
- **Universal Claim Model**: Consolidated to single Claim class with bidirectional relationships
- **ConjectureEndpoint**: Centralized public API in Layer 2 with methods: create_claim(), get_claim(), evaluate()
- **Context Building**: Intelligent context traversal (upward 100%, downward to depth 2, semantic fill)
- **Tag Lifecycle**: Creation → Consolidation → Retirement management
- **Data models**: Consolidated from 40 to 5 classes with Pydantic validation
- **Context models**: Simplified for LLM processing with XML optimization
- **CLI commands**: All functional (create, get, search, analyze, prompt, stats, config, providers, setup, health, quickstart)
- **Configuration**: Unified ProviderConfig across the system with JSON-based configuration and hierarchical config system
- **Documentation**: Streamlined from 100+ files to ~35 high-value files (65% reduction) with archive/ structure
- **Architecture**: Simplified to OpenAI-compatible provider pattern with unified backend system
- **Testing**: Comprehensive test suite with 89% coverage achieved across 100+ test files
- **Security**: Implemented comprehensive security framework with input validation and SQL injection protection
- **Performance**: Achieved 26% improvement in response times and 40% reduction in memory usage
- **XML Optimization**: 100% XML compliance achieved with 40% improvement in reasoning quality

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

#### Unified Configuration System
- **Pydantic-based**: Type-safe configuration with validation
- **Hierarchical**: Workspace > User > Default config precedence
- **Provider Management**: Unified provider configuration and validation
- **Settings Models**: Structured configuration settings with proper typing
- **Auto-reload**: Configuration changes detected and reloaded automatically

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

# Test provider connectivity
python conjecture backends

# Check system health
python conjecture health

# Show available providers
python conjecture providers

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