# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Philosophy: Claims Are NOT Facts

Conjecture treats all knowledge as provisional claims, not facts. Claims are impressions, assumptions, observations, and conjectures with variable confidence that may be wrong and subject to revision.

## Choice Management

This project uses CHOICES.md as the Source of Plan. All project choices are recorded in priority order — higher choices constrain lower ones (gravity rule). Each choice lists explicit `Supports:` references forming a dependency graph.

- Read CHOICES.md before proposing changes that affect project direction
- Use `/choose-wisely` to add, change, remove, or reorder choices (triggers cascading review + independent commit)
- Use `/choose-wisely audit` to check for contradictions and structural issues
- Never edit CHOICES.md directly without running cascading review
- If a specification in another doc contradicts CHOICES.md, the choice wins and the spec should be updated
- Architecture choices are tool-agnostic; Technology choices name specific tools
- Git diff is the change record — no status annotations in the file

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
cat .agent/backlog.md
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
4. **Data Layer** (`src/data/`): Universal claim storage with SQLite/FAISS

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
- **FAISS**: Vector embeddings for semantic search (ChromaDB deprecated — slow, heavy deps)
- **Repository Pattern**: Clean separation between data access and business logic
- **Connection Pooling**: Optimized database performance

## Proven Development Patterns

### Architecture Validation Requirements
- **Multi-model testing required** - Single model (e.g., DeepSeek V3) insufficient for production claims
- **Document model explicitly** - Always note which model(s) tested in results
- **Test multiple task types** - Hard reasoning (BBH) + saturated (GSM8K) + recall (MMLU) minimum
- **Statistical validation** - Include p-values in result tables, not just accuracy numbers
- **Test positive AND negative cases** - Where architecture should help AND where it shouldn't
- **Limitations section required** - Document model coverage, sample size, benchmark gaps

### Successful Enhancement Patterns
1. **Core reasoning enhancements**: 100% success rate (5/5 successful cycles)
2. **Prompt system improvements**: High success rate with measurable improvements
3. **Problem-type-specific strategies**: Mathematical, logical, sequential reasoning
4. **Structured decomposition**: Breaking complex problems into manageable steps

### Failed Patterns to Avoid
1. **Heavy external dependencies**: ChromaDB rejected (slow, heavy deps) — use FAISS+SQLite instead
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

### Statistical Significance for Benchmarks
- **Don't eyeball results** - Calculate p-values using scipy.stats before claiming improvements/regressions
- **Sample size n=50** provides ±10pp margin of error (95% CI) - differences <10pp may be noise
- **Example lesson** - Reported -2pp "regression" was p=0.695 (not significant), actually equivalent
- **Significance threshold** - p<0.05 for claims, p<0.01 for strong claims
- **Test code** - `from scipy import stats; import math; se = math.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2); z = diff/se; p = 2*stats.norm.cdf(-abs(z))`

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

## Standard Benchmark Validation (2026-03-06)

**O-0008 Validation Complete (7/10 benchmarks)**: Task-type dependency confirmed across reasoning, recall, and commonsense tasks.

| Benchmark | Type | Direct | Decomposition | Delta | Model |
|-----------|------|--------|---------------|-------|-------|
| **BBH** | Hard Reasoning | 84.0% | 93.0% | **+9.0pp** | DeepSeek-V3 |
| **GSM8K** | Math | 92.0% | 93.0% | **+1.0pp** | DeepSeek-V3 |
| **ARC-Challenge** | Science | 93.0% | 92.0% | **-1.0pp** | DeepSeek-V3 |
| **MMLU** | Knowledge | 62.0% | 45.0% | **-17.0pp** | DeepSeek-V3 |
| **TruthfulQA** | Truthfulness | 79.0% | 66.0% | **-13.0pp** | DeepSeek-V3 |
| **HellaSwag** | Commonsense | 83.0% | 73.0% | **-10.0pp** | DeepSeek-V3 |
| **Synthetic** | Math | 79.0% | 97.0% | **+18.0pp** | DeepSeek-V3 |

**Alternative Method:** cot_lite (lightweight scaffolding) achieved +2pp on MMLU without regression.

### When Decomposition Works
- ✅ Multi-step math/reasoning with baseline <85%
- ✅ Hard reasoning tasks (BBH +9pp, Synthetic +18pp)
- ✅ Novel problems (not in training data)
- ❌ Factual recall tasks (MMLU: -17pp)
- ❌ Commonsense/intuition tasks (HellaSwag: -10pp)
- ❌ Truthfulness (TruthfulQA: -13pp, decomposition increases false confidence)
- ❌ High baseline (>90%, GSM8K: +1pp)

**Critical Finding:** Task-type routing or confidence-based exploration REQUIRED for production.

**See `experiments/O-0008_VALIDATION_REPORT.md` for comprehensive analysis**

## Three-Prompt Architecture Validation (2026-03-07)

**Status:** VALIDATED for hard reasoning, neutral on saturated tasks (with statistical analysis)

The three-prompt architecture splits single prompts into 3 focused stages: update confidence → create claim or SKIP → final response. Validated on DeepSeek V3 with 50-problem benchmarks and p-value analysis.

| Benchmark | Type | Direct | Three-Prompt | Delta | P-value | Status |
|-----------|------|--------|--------------|-------|---------|--------|
| **BBH** | Hard Reasoning | 90.0% | 100.0% | **+10.0pp** | **0.018** | ✅ **SIGNIFICANT** |
| **GSM8K** | Math (saturated) | 94.0% | 92.0% | -2.0pp | 0.695 | ≈ **Equivalent** |

### Key Findings with Statistical Analysis

**Significant Improvement on Hard Reasoning:**
- BBH: +10pp improvement (p=0.018) - Perfect accuracy achieved
- Matches O-0008 decomposition performance (+9pp)
- More efficient than GSM8K (4.9x tokens vs 8.7x)
- Cost justified by accuracy gains

**Statistically Equivalent on Saturated Tasks:**
- GSM8K: -2pp difference (p=0.695) - NOT significant, within random variation
- No evidence of harm - architecture safe on all tested task types
- High baseline (94%) leaves little room for improvement
- Decision is economic (8.7x tokens), not accuracy-based

**Production Recommendations:**
- ✅ Use task-type routing (see `experiments/task_type_router.py`)
- ✅ Route to three-prompt when baseline <90% and hard reasoning expected
- ✅ Route to direct when baseline ≥90% or simple recall/calculation
- ⚠️ Single-model validation only (DeepSeek V3) - multi-model testing REQUIRED for production

**Critical Limitation:** Only tested on DeepSeek V3. Results may not generalize across Claude/GPT-4/Gemini/Llama. Multi-model validation required before production deployment.

**See `experiments/THREE_PROMPT_ARCHITECTURE.md` and `.director/THREE_PROMPT_VALIDATION_COMPLETE.md` for full details**

## Benchmark Execution Best Practices

### Running Standard Benchmarks
```bash
# Standard benchmarks in experiments/ directory (100 samples each)
.venv/bin/python experiments/gsm8k_standard_benchmark.py -n 100
.venv/bin/python experiments/mmlu_standard_benchmark.py -n 100
.venv/bin/python experiments/bbh_benchmark.py -n 100

# Run in parallel (background) for efficiency
PYTHONUNBUFFERED=1 .venv/bin/python experiments/arc_challenge_benchmark.py -n 100 &
PYTHONUNBUFFERED=1 .venv/bin/python experiments/hellaswag_benchmark.py -n 100 &
PYTHONUNBUFFERED=1 .venv/bin/python experiments/truthfulqa_benchmark.py -n 100 &

# Monitor progress (check for new result files)
ls -lt experiments/results/*.json | head -5

# Update CSV with all results
.venv/bin/python experiments/analyze_benchmark_results.py
```

### Benchmark Timing Expectations
- **100 problems**: 30-40 minutes typical (API rate limiting)
- **Three-prompt architecture**: 35-45 minutes for n=50 (more API calls per problem)
- **Parallel execution**: Run 5 benchmarks simultaneously for efficiency
- **Background monitoring**: Check every 2-3 minutes, don't poll actively
- **Process may show 0% CPU**: Normal when waiting on API calls (check results file instead)
- **Results location**: `experiments/results/[benchmark]_[timestamp].json`
- **CSV tracking**: `experiments/results/benchmark_results.csv`

### Result Analysis
- Run `analyze_benchmark_results.py` after benchmarks complete
- Automatically updates CSV and generates pattern analysis
- Task types: reasoning (+3pp avg), recall (-15pp avg), commonsense (-10pp avg)

## Three-Prompt Architecture (Experimental - 2026-03-06)

**Status:** Implemented, ready for real LLM testing
**Goal:** Confidence-based exploration without hard-coded task routing

### Design
Split single prompt into 3 focused prompts with shared context:
1. **Update claim confidence** (0-1.0) - Evaluate evidence quality
2. **Create claim or SKIP** - Explore OR signal completion
3. **Final response** (when confidence > 0.7 and SKIP) - Synthesize answer

### Key Features
- Same 50-claim context for all prompts (retrieved once)
- Iterative loop until confidence > 0.7 AND action == SKIP
- Each prompt outputs structured JSON
- Self-regulating complexity (no task-type routing needed)

### Testing
```bash
# Mock LLM test (architecture validation)
.venv/bin/python experiments/three_prompt_test.py

# Real LLM test (3 benchmark problems)
.venv/bin/python experiments/three_prompt_real_test.py
```

**See `experiments/THREE_PROMPT_ARCHITECTURE.md` for full design**

## Key File Locations

### Prompt System
- **Main**: `src/agent/prompt_system.py` (simple_mode vs full mode)
- **Error correction**: `src/agent/error_correction_prompts.py`
- **Mode selection**: simple_mode for <14B models, full mode for >14B

### Claim Retrieval
- **Endpoint**: `src/endpoint/conjecture_endpoint.py` (default max_claims=10)
- **Context builder**: `src/process/context_builder.py` (hint limit=5)
- **Test increasing to 50** when database populated

### Database
- **Location**: `data/conjecture.db` (may not exist in test environment)
- **Note**: Benchmarks test prompts directly, not full endpoint with claims
- Empty database is normal during prompt-only testing

## R&D Key Findings (2026-03-01)

- **Direct prompting beats decomposition** on standard benchmarks (GSM8K: 96% vs 65%)
- **Position primacy**: Claims at prompt START (+10pp improvement)
- **Confidence threshold 0.5** is optimal (not 0.8)
- **No semantic filtering needed**: Simple inclusion of all correct claims works best
- See `NEXT.md` for follow-up ideas, `docs/RND_COMPREHENSIVE_REPORT.md` for full report

## Troubleshooting

### Common Environment Issues
- **Always use `.venv/bin/python`** not just `python` (command not found)
- **Database may be empty** in test environment (benchmarks test prompts directly)
- **SSH not configured**: Git push blocked, commits accumulate locally (137+ unpushed common)
- **API rate limiting**: Benchmarks take 2-3x expected time (30-40min for 100 problems)
- **Long-running tasks**: Use `run_in_background=true`, monitor with 2-3min intervals

### Background Task and Process Monitoring

**TaskOutput inconsistencies:**
- Background tasks with `run_in_background=true` return `task_id` in result
- Use `TaskOutput` tool to check status, NOT `Bash` with `tail`
- Long-running benchmarks (30-40 min): Check every 2-3 minutes, don't poll actively
- Process monitoring: Use `ls -lt experiments/results/*.json | head -5` to detect completion

**Three-prompt benchmarks:**
- Expect 35-45 minutes for 50 problems (3.88 avg iterations × 4 prompts × API delays)
- Direct baseline: 10-15 minutes for comparison
- Don't assume completion - check result file timestamps
- Background execution recommended for parallel benchmark runs

**Autonomous agent patterns:**
- State in single JSON file (`state.json`) NOT database
- Check `state` field: IDLE, WAITING, BLOCKED
- BLOCKED requires human intervention - post clear blocker description
- WAITING with background tasks: Don't poll, wait for notification

```bash
python conjecture config      # Check configuration
python conjecture backends    # Test provider connectivity
python conjecture health      # Check system health
python conjecture providers   # Show available providers
```