# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Philosophy: Claims Are NOT Facts

Conjecture treats all knowledge as provisional claims, not facts. Claims are impressions, assumptions, observations, and conjectures with variable confidence that may be wrong and subject to revision.

## Essential Development Commands

### Testing Infrastructure
```bash
# Run all tests with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing

# Run specific test categories
python -m pytest tests/ -m "unit"           # Unit tests only
python -m pytest tests/ -m "integration"     # Integration tests only
python -m pytest tests/ -m "performance"     # Performance tests only
python -m pytest tests/ -m "security"        # Security tests only

# Run single test file
python -m pytest tests/test_claim_models.py -v

# Run with specific markers
python -m pytest tests/ -m "database" -v     # Database tests
python -m pytest tests/ -m "llm" -v           # LLM provider tests
python -m pytest tests/ -m "models" -v         # Pydantic model tests
```

### Systematic Improvement Cycles
```bash
# Execute systematic improvement cycle (following established pattern)
python src/benchmarking/cycleXX_[enhancement_name].py

# View cycle results
ls src/benchmarking/cycle_results/
cat src/benchmarking/cycle_results/cycle_XXX_results.json

# Check progress documentation
cat TODO.md
cat ANALYSIS_100_CYCLES.md
```

### Core System Validation
```bash
# Test core module imports
python -c "from src.data.models import Claim; print('Claim system OK')"
python -c "from src.agent.prompt_system import PromptSystem; print('Prompt system OK')"

# Validate configuration
python conjecture validate

# Check system status
python conjecture stats
```

## High-Level Architecture

### 4-Layer Architecture (Canonical Reference)
1. **Presentation Layer** (`src/cli/`): User interaction and CLI interface
2. **Endpoint Layer** (`src/endpoint/`): Public API and service endpoints
3. **Process Layer** (`src/process/`): Intelligence, LLM integration, and context management
4. **Data Layer** (`src/data/`): Universal claim storage with SQLite/ChromaDB

### Core Data Models (`src/data/models.py`)
- **Claim**: Primary knowledge container with confidence scores, state, tags, and relationships
- **ClaimType**: CONCEPT, REFERENCE, THESIS, SKILL
- **ClaimState**: EXPLORE, VALIDATED, ORPHANED, QUEUED
- **ClaimScope**: Global, user workspace, project-specific

### Claim-Based Knowledge System
```python
from src.data.models import Claim, ClaimType, ClaimState

# Create a claim (core knowledge unit)
claim = Claim(
    id="c0001",
    content="Mathematical induction requires base case verification",
    confidence=0.95,
    type=[ClaimType.THESIS],
    tags=["mathematics", "induction", "verification"],
    state=ClaimState.VALIDATED
)
```

### Multi-Provider LLM System
- **Local Providers**: Ollama (localhost:11434), LM Studio (localhost:1234)
- **Cloud Providers**: Chutes.ai, OpenRouter, OpenAI, Anthropic, Google
- **Auto-detection**: Intelligent backend selection based on availability
- **Fallback Mechanisms**: Automatic provider switching on failures

## Critical System Components

### Prompt System (`src/agent/prompt_system.py`)
Core system that implements successful reasoning enhancements:
- **Domain-adaptive prompts** (Cycle 1: 100% improvement)
- **Context integration** (Cycle 2: structured guidance)
- **Self-verification** (Cycle 3: error detection)
- **Mathematical reasoning** (Cycle 9: 8% improvement)
- **Multi-step reasoning** (Cycle 11: 10% improvement)
- **Problem decomposition** (Cycle 12: 9% improvement)

### Systematic Improvement Framework (`src/benchmarking/`)
- **Skeptical validation**: Minimum 2-4% improvement thresholds
- **Real testing**: Actual problem-solving validation, not just metrics
- **Honest failure reporting**: 62% success rate (8/13 cycles successful)
- **Pattern recognition**: Core reasoning enhancements work (5/5 successful)

### Database Integration (`src/data/`)
- **SQLite**: Primary claim storage with structured queries
- **ChromaDB**: Vector embeddings for semantic search (when available)
- **Repository Pattern**: Clean separation between data access and business logic
- **Connection Pooling**: Optimized database performance

## Proven Development Patterns

### Successful Enhancement Patterns
1. **Core reasoning enhancements**: 100% success rate (5/5 successful cycles)
2. **Prompt system improvements**: High success rate with measurable improvements
3. **Problem-type-specific strategies**: Mathematical, logical, sequential reasoning
4. **Structured decomposition**: Breaking complex problems into manageable steps

### Failed Patterns to Avoid
1. **Knowledge infrastructure attempts**: 0/2 successful (ChromaDB API incompatibility)
2. **Surface-level changes**: 0/3 successful (formatting, confidence optimization)
3. **Arbitrary metric gaming**: Leading to false positives (corrected by multi-agent critique)

### Systematic Cycle Implementation
```python
# Following established pattern from successful cycles
async def run_cycle(self):
    # Step 1: Enhancement (build on working systems)
    enhancement_success = self.enhance_prompt_system()

    # Step 2: Testing (real problem-solving validation)
    test_results = self.test_enhancement()

    # Step 3: Impact estimation (conservative calculation)
    estimated_improvement = test_results['success_rate'] * 0.15

    # Step 4: Skeptical validation (>2-4% threshold)
    success = estimated_improvement > 3.0
```

## Testing Strategy

### Current Test Status
- **96 tests collected**, 41 passed, 1 failed in current run
- **Core claim system tests**: Working (test_claim_models.py, test_claim_processing.py)
- **Integration tests**: Some failures due to missing local LLM providers
- **Database tests**: Working but need proper cleanup

### Test Categories
- **Unit tests**: Individual component testing
- **Integration tests**: Multi-component workflows
- **End-to-end tests**: Complete system workflows
- **Performance tests**: Load and stress testing
- **Security tests**: SQL injection, input validation

### Known Test Issues
- Local LLM providers not running (Ollama/LM Studio on localhost)
- Database schema issues with claim type column
- Configuration validation errors in workspace settings

## Configuration System

### Priority Order
1. **Workspace config** (`.conjecture/config.json`)
2. **User config** (`~/.conjecture/config.json`)
3. **Default config** (`src/config/default_config.json`)

### Example Configuration
```json
{
  "providers": [
    {
      "url": "http://localhost:11434",
      "api": "",
      "model": "llama2",
      "name": "ollama"
    }
  ],
  "confidence_threshold": 0.95,
  "database_path": "data/conjecture.db"
}
```

## Infrastructure Requirements

### Working Systems
- **4/4 core modules**: agent, core, data, data.repositories import successfully
- **Claim system**: src.data.models.Claim available and functional
- **Prompt system**: Enhanced with proven reasoning capabilities
- **Test infrastructure**: 96 tests with comprehensive coverage

### Missing/Broken Infrastructure
- **Knowledge repositories**: Database operations fail (ClaimRepository.create_claim)
- **ChromaDB integration**: Vector storage not accessible
- **Local LLM providers**: Ollama/LM Studio not running on localhost
- **Workspace configuration**: Pydantic validation errors

## Development Workflow

### Before Making Changes
1. **Run tests**: `python -m pytest tests/ -m "unit" -v`
2. **Check core imports**: Verify agent, core, data modules work
3. **Review successful patterns**: Follow core reasoning enhancement approach
4. **Avoid failed patterns**: Skip knowledge infrastructure, surface-level changes

### When Implementing Enhancements
1. **Build on working systems**: Use existing prompt system enhancements
2. **Focus on core reasoning**: Mathematical, logical, multi-step reasoning
3. **Use real testing**: Test actual problem-solving improvement
4. **Apply skeptical validation**: Minimum 2-4% improvement thresholds

### Systematic Improvement Process
1. **Create enhancement** in `src/benchmarking/cycleXX_[name].py`
2. **Build on proven patterns**: Reference successful cycles 9, 11, 12
3. **Test thoroughly**: Real problem-solving validation
4. **Validate conservatively**: Apply skeptical thresholds
5. **Document honestly**: Report both successes and failures

The system has demonstrated that **core reasoning enhancements consistently work** while **infrastructure attempts consistently fail**. Focus on what works and maintain the scientific validation approach established through 13 systematic improvement cycles.