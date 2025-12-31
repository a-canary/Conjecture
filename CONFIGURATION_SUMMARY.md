# Conjecture Configuration Files Summary
## LLM Temperature Settings, GraniteTiny, and LM Studio

**Generated**: December 30, 2025  
**Scope**: Comprehensive search of all configuration files related to temperature settings, GraniteTiny model, and LM Studio integration

---

## 📋 Executive Summary

This document provides a complete inventory of all configuration files in the Conjecture project that contain:
- **Temperature settings** for LLM inference
- **GraniteTiny model** (IBM Granite 4-H-Tiny) configurations
- **LM Studio** integration settings
- **SWE-Bench** specific configurations

### Key Findings

| Category | Count | Status |
|----------|-------|--------|
| **Config Files with Temperature** | 18+ | ✅ Documented |
| **GraniteTiny References** | 25+ | ✅ Documented |
| **LM Studio Configs** | 8+ | ✅ Documented |
| **SWE-Bench Files** | 3+ | ✅ Documented |

---

## 🔧 Core Configuration Files

### 1. **Default Configuration** (`src/config/default_config.json`)
**Purpose**: Default provider configuration template  
**Status**: ✅ Primary configuration source

```json
{
  "providers": [
    {
      "url": "http://localhost:11434",
      "api_key": "",
      "model": "llama2",
      "name": "ollama"
    },
    {
      "url": "http://localhost:1234",
      "api_key": "",
      "model": "ibm/granite-4-h-tiny",
      "name": "lm_studio"
    }
  ],
  "confidence_threshold": 0.95,
  "max_context_size": 10,
  "batch_size": 10,
  "debug": false,
  "database_path": "data/conjecture.db"
}
```

**Key Settings**:
- **LM Studio URL**: `http://localhost:1234`
- **Model**: `ibm/granite-4-h-tiny`
- **Confidence Threshold**: 0.95
- **Max Context Size**: 10

---

### 2. **Research Configuration** (`research/config.json`)
**Purpose**: Research experiments configuration with environment variable support  
**Status**: ✅ Active research configuration

```json
{
  "providers": [
    {
      "url": "${LM_STUDIO_API_URL:-http://localhost:1234}",
      "api_key": "${LM_STUDIO_API_KEY:-}",
      "model": "${LM_STUDIO_MODEL:-ibm/granite-4-h-tiny}"
    }
  ],
  "judge_model": "${JUDGE_MODEL:-chutes:zai-org/GLM-4.6-FP8}",
  "judge_provider": "${JUDGE_PROVIDER:-chutes}",
  "judge_temperature": "${JUDGE_TEMPERATURE:-0.1}",
  "judge_max_tokens": "${JUDGE_MAX_TOKENS:-1000}",
  "experiments": {
    "hypothesis_validation": true,
    "model_comparison": true,
    "baseline_comparison": true,
    "generate_test_cases": true
  }
}
```

**Key Settings**:
- **Judge Temperature**: `0.1` (low for consistency)
- **Judge Max Tokens**: `1000`
- **Models**: GraniteTiny + GLM-4.6 comparison

---

### 3. **LM Studio Configuration** (`research/lm_studio_config.json`)
**Purpose**: Dedicated LM Studio experiment configuration  
**Status**: ✅ Experiment-specific

```json
{
  "providers": [
    {
      "url": "http://localhost:1234",
      "api_key": "",
      "model": "ibm/granite-4-h-tiny"
    },
    {
      "url": "http://localhost:1234",
      "api_key": "",
      "model": "glm-z1-9b-0414"
    }
  ],
  "judge_model": "chutes:zai-org/GLM-4.6",
  "experiments": {
    "hypothesis_validation": true,
    "model_comparison": true,
    "baseline_comparison": true,
    "generate_test_cases": true
  }
}
```

**Key Settings**:
- **Primary Model**: `ibm/granite-4-h-tiny`
- **Comparison Model**: `glm-z1-9b-0414`
- **Judge Model**: `chutes:zai-org/GLM-4.6`

---

### 4. **Provider Registry** (`src/config/providers.json`)
**Purpose**: Provider processor mapping and configuration  
**Status**: ✅ Provider registry

```json
{
  "chutes": {"processor": "ChutesProcessor", "default_model": "zai-org/GLM-4.6"},
  "openrouter": {"processor": "OpenRouterProcessor", "default_model": "openai/gpt-3.5-turbo"},
  "ollama": {"processor": "LocalProviderProcessor", "provider_type": "ollama"},
  "lm_studio": {"processor": "LocalProviderProcessor", "provider_type": "lm_studio"}
}
```

**Key Settings**:
- **LM Studio Processor**: `LocalProviderProcessor`
- **Provider Type**: `lm_studio`

---

## 🎯 Temperature Settings Across Project

### Temperature Values Used

| Temperature | Use Case | Files | Count |
|-------------|----------|-------|-------|
| **0.1** | Judge evaluation, consistent output | research/config.json, benchmarks | 8+ |
| **0.2** | Benchmark consistency | cycle14, cycle15 benchmarks | 6+ |
| **0.3** | LM Studio experiments, reasoning | run_lm_studio_experiment.py, baseline_comparison.py | 12+ |
| **0.5** | Exploration, varied responses | research experiments | 2+ |
| **0.7** | Creative tasks, exploration | conjecture_explore_method_clean.py | 3+ |

### Temperature Configuration Locations

#### Low Temperature (0.1) - Consistency
- **research/config.json**: `"judge_temperature": "${JUDGE_TEMPERATURE:-0.1}"`
- **validate_hypothesis_direct.py**: `"temperature": 0.1`
- **check_chutes_models.py**: `"temperature": 0.1`
- **benchmarks/benchmarking/cycle14_real_deepeval_verification.py**: `temperature=0.2`

#### Medium Temperature (0.3) - Reasoning
- **research/run_lm_studio_experiment.py**: `"temperature": 0.3`
- **research/baseline_comparison.py**: `"temperature": 0.3`
- **research/comprehensive_comparison_study.py**: `"temperature": 0.3`
- **src/config/tiny_model_config.py**: `temperature: float = 0.3`

#### High Temperature (0.7) - Exploration
- **research/conjecture_explore_method_clean.py**: `temperature=0.7`
- **research/generate_initial_claims_clean.py**: `temperature=0.7`
- **benchmarks/benchmarking/comprehensive_benchmark.py**: `temperature: float = 0.7`

---

## 🤖 GraniteTiny Configuration

### Tiny Model Configuration File (`src/config/tiny_model_config.py`)
**Purpose**: Optimized settings for IBM Granite Tiny model  
**Status**: ✅ Production-ready

```python
@dataclass
class TinyModelConfig:
    # Model-specific parameters
    model_name: str = "ibm/granite-4-h-tiny"
    max_tokens: int = 42000
    temperature: float = 0.3
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    
    # Context optimization
    max_context_size: int = 5
    max_context_concepts: int = 3
    max_context_references: int = 2
    max_context_goals: int = 1
    
    # Processing parameters
    batch_size: int = 3
    confidence_threshold: float = 0.90
    confident_threshold: float = 0.75
    
    # Prompt optimization
    use_simplified_prompts: bool = True
    include_examples: bool = True
    max_examples: int = 2
    
    # Error handling
    max_retries: int = 2
    retry_delay: float = 0.5
    timeout: int = 15
    
    # Special handling
    enable_two_step_processing: bool = True
    use_json_frontmatter: bool = True
    enable_confidence_boosting: bool = True
```

### Key GraniteTiny Settings

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Temperature** | 0.3 | Lower for consistent reasoning |
| **Max Tokens** | 42000 | Proper evaluation window |
| **Max Context Size** | 5 | Reduced from 10 for tiny models |
| **Confidence Threshold** | 0.90 | Slightly lower for tiny models |
| **Batch Size** | 3 | Reduced from 10 |
| **Simplified Prompts** | True | Tiny models need simpler instructions |
| **JSON Frontmatter** | True | Structured output format |
| **Two-Step Processing** | True | Break down complex tasks |

### GraniteTiny References in Codebase

**Files Containing GraniteTiny Configuration**:
1. `src/config/tiny_model_config.py` - Primary configuration
2. `src/config/default_config.json` - Default provider
3. `research/lm_studio_config.json` - Research experiments
4. `research/config.json` - Research configuration
5. `.agent/plan/swebench_quick_reference.md` - Quick reference guide

**Research Files Using GraniteTiny**:
- `research/run_lm_studio_experiment.py` - Main LM Studio experiment
- `research/analyze_lm_studio_results.py` - Results analysis
- `research/baseline_comparison.py` - Baseline comparison
- `benchmarks/benchmarking/final_baseline_test.py` - Baseline testing
- `research/simple_experiment.py` - Simple experiments
- `research/working_real_experiment.py` - Real experiments

---

## 🏗️ LM Studio Integration

### LM Studio Adapter (`src/processing/llm/lm_studio_adapter.py`)
**Purpose**: Compatibility layer for LM Studio  
**Status**: ✅ Backward compatible

```python
from .local_providers_adapter import LocalProviderProcessor as LMStudioAdapter

def create_lm_studio_adapter_from_config(config):
    """Create LM Studio adapter from configuration"""
    return LMStudioAdapter(config)
```

### LM Studio Configuration Details

**Connection Settings**:
- **URL**: `http://localhost:1234/v1`
- **API Key**: Empty (local model)
- **Model**: `ibm/granite-4-h-tiny`

**Performance Parameters**:
- **Max Tokens**: 512 (prevent rambling)
- **Temperature**: 0.3 (consistency)
- **Top P**: 0.9 (nucleus sampling)
- **Repetition Penalty**: 1.1 (reduce repetition)

**Optimization Settings**:
- **Max Context Size**: 5 (focus)
- **Confidence Threshold**: 0.90 (tiny model calibration)
- **Batch Size**: 3 (reduced for tiny models)

### LM Studio Configuration in Files

**Primary Configuration**:
- `src/config/default_config.json` - Default provider
- `research/lm_studio_config.json` - Research configuration
- `src/config/tiny_model_config.py` - Tiny model optimization

**Integration Points**:
- `src/processing/llm/lm_studio_adapter.py` - Adapter layer
- `src/processing/llm/local_providers_adapter.py` - Local provider processor
- `src/processing/llm_bridge.py` - LLM bridge for unified access

**Research Scripts**:
- `research/run_lm_studio_experiment.py` - Main experiment runner
- `research/analyze_lm_studio_results.py` - Results analysis
- `benchmarks/benchmarking/final_baseline_test.py` - Baseline testing

---

## 📊 SWE-Bench Configuration

### SWE-Bench Evaluator (`benchmarks/benchmarking/swe_bench_evaluator.py`)
**Purpose**: Real SWE-bench-lite evaluation framework  
**Status**: ✅ Production-ready (895 lines)

**Key Classes**:
- `RealSWEBenchEvaluator` - Main evaluator
- `SWETask` - Task representation
- `EvaluationOutput` - Results container
- `EvaluationResult` - Status enum

**Key Methods**:
```python
async def load_swe_tasks(num_tasks: int = 5) -> List[SWETask]
async def evaluate_direct_approach(task: SWETask) -> EvaluationOutput
async def evaluate_conjecture_approach(task: SWETask) -> EvaluationOutput
async def evaluate_models_on_tasks(models: List[str], tasks: List[SWETask])
```

### SWE-Bench Quick Reference (``.agent/plan/swebench_quick_reference.md`)
**Purpose**: Quick reference guide for SWE-Bench and GraniteTiny  
**Status**: ✅ Ready for implementation (415 lines)

**Configuration Example**:
```json
{
  "url": "http://localhost:1234/v1",
  "model": "ibm/granite-4-h-tiny",
  "name": "lm_studio",
  "max_tokens": 512,
  "temperature": 0.3
}
```

**Optimized Parameters**:
- `max_tokens`: 512 (prevent rambling)
- `temperature`: 0.3 (consistency)
- `max_context_size`: 5 (focus)
- `confidence_threshold`: 0.90 (tiny model calibration)

### SWE-Bench Success Criteria

**SC-FEAT-001: SWE-Bench-Bash-Only Accuracy**
- **Target**: >70% accuracy on SWE-Bench-Bash-Only
- **Status**: Promoted to success criteria
- **Plan**: `.agent/plan/swebench_enhancement.md`

**Approach**:
1. Context engineering for bash-specific tasks
2. Prompt refinement for shell scripting
3. GraniteTiny + Conjecture optimization
4. Comparison with direct LLM approach

---

## 📁 Complete File Inventory

### Configuration Files (JSON)

| File | Purpose | Temperature | GraniteTiny | LM Studio | SWE-Bench |
|------|---------|-------------|-------------|-----------|-----------|
| `src/config/default_config.json` | Default provider config | ❌ | ✅ | ✅ | ❌ |
| `research/config.json` | Research experiments | ✅ (0.1) | ✅ | ✅ | ❌ |
| `research/lm_studio_config.json` | LM Studio experiments | ❌ | ✅ | ✅ | ❌ |
| `src/config/providers.json` | Provider registry | ❌ | ❌ | ✅ | ❌ |
| `.phased_testing_config.json` | Testing configuration | ❌ | ❌ | ❌ | ❌ |

### Python Configuration Files

| File | Purpose | Temperature | GraniteTiny | LM Studio | SWE-Bench |
|------|---------|-------------|-------------|-----------|-----------|
| `src/config/tiny_model_config.py` | Tiny model optimization | ✅ (0.3) | ✅ | ✅ | ❌ |
| `src/processing/llm/lm_studio_adapter.py` | LM Studio adapter | ❌ | ❌ | ✅ | ❌ |
| `src/processing/llm/local_providers_adapter.py` | Local provider processor | ❌ | ❌ | ✅ | ❌ |
| `src/processing/llm_bridge.py` | LLM bridge | ❌ | ❌ | ✅ | ❌ |

### Research Scripts

| File | Purpose | Temperature | GraniteTiny | LM Studio | SWE-Bench |
|------|---------|-------------|-------------|-----------|-----------|
| `research/run_lm_studio_experiment.py` | Main LM Studio experiment | ✅ (0.3) | ✅ | ✅ | ❌ |
| `research/analyze_lm_studio_results.py` | Results analysis | ❌ | ✅ | ✅ | ❌ |
| `research/baseline_comparison.py` | Baseline comparison | ✅ (0.3) | ✅ | ❌ | ❌ |
| `research/check_chutes_models.py` | Model checking | ✅ (0.1) | ❌ | ❌ | ❌ |
| `validate_hypothesis_direct.py` | Hypothesis validation | ✅ (0.1) | ❌ | ❌ | ❌ |

### Benchmark Files

| File | Purpose | Temperature | GraniteTiny | LM Studio | SWE-Bench |
|------|---------|-------------|-------------|-----------|-----------|
| `benchmarks/benchmarking/swe_bench_evaluator.py` | SWE-Bench evaluator | ❌ | ❌ | ❌ | ✅ |
| `benchmarks/benchmarking/comprehensive_benchmark.py` | Comprehensive benchmark | ✅ (0.7) | ❌ | ❌ | ❌ |
| `benchmarks/benchmarking/cycle14_real_deepeval_verification.py` | DeepEval verification | ✅ (0.2) | ❌ | ❌ | ❌ |
| `benchmarks/benchmarking/cycle15_deepeval_comparison.py` | DeepEval comparison | ✅ (0.2) | ❌ | ❌ | ❌ |

### Documentation Files

| File | Purpose | Temperature | GraniteTiny | LM Studio | SWE-Bench |
|------|---------|-------------|-------------|-----------|-----------|
| `.agent/plan/swebench_quick_reference.md` | Quick reference | ✅ (0.3) | ✅ | ✅ | ✅ |
| `docs/ibm_granite_tiny_integration_guide.md` | Integration guide | ✅ (0.3) | ✅ | ✅ | ❌ |
| `SWEBENCH_EXPLORATION_REPORT.md` | Exploration report | ❌ | ❌ | ❌ | ✅ |

---

## 🔍 Temperature Settings Summary

### By Use Case

**Evaluation & Judging (Temperature: 0.1)**
- Consistent, deterministic output
- Used for LLM-as-a-Judge evaluation
- Files: `research/config.json`, benchmark verification scripts

**Reasoning & Analysis (Temperature: 0.3)**
- Balanced creativity and consistency
- Used for GraniteTiny and LM Studio
- Files: `src/config/tiny_model_config.py`, `research/run_lm_studio_experiment.py`

**Exploration & Generation (Temperature: 0.7)**
- More creative, varied responses
- Used for claim generation and exploration
- Files: `research/conjecture_explore_method_clean.py`

**Benchmark Testing (Temperature: 0.2)**
- Slightly more consistent than 0.3
- Used for benchmark comparisons
- Files: `benchmarks/benchmarking/cycle14_*.py`, `cycle15_*.py`

---

## 🚀 Implementation Recommendations

### For GraniteTiny Usage
1. **Use Temperature 0.3** for reasoning tasks
2. **Enable JSON Frontmatter** for structured output
3. **Reduce Context Size to 5** for tiny models
4. **Use Simplified Prompts** for better understanding
5. **Enable Two-Step Processing** for complex tasks

### For LM Studio Integration
1. **URL**: `http://localhost:1234/v1`
2. **Model**: `ibm/granite-4-h-tiny`
3. **Max Tokens**: 512
4. **Temperature**: 0.3
5. **Batch Size**: 3

### For SWE-Bench Evaluation
1. **Use RealSWEBenchEvaluator** class
2. **Load tasks** from HuggingFace dataset
3. **Compare direct vs Conjecture** approaches
4. **Target >70% accuracy** on bash-only subset
5. **Track execution time** and confidence scores

---

## 📚 Related Documentation

### Primary References
- `src/config/tiny_model_config.py` - Tiny model configuration
- `research/lm_studio_config.json` - LM Studio configuration
- `.agent/plan/swebench_quick_reference.md` - Quick reference guide
- `docs/ibm_granite_tiny_integration_guide.md` - Integration guide

### Architecture
- `specs/architecture.md` - 4-Layer architecture
- `docs/architecture/main.md` - Architecture specification

### Backlog & Planning
- `.agent/backlog.md` - SC-FEAT-001 (SWE-Bench target)
- `.agent/plan/swebench_enhancement.md` - Enhancement plan

---

## ✅ Verification Checklist

- [x] All configuration files identified
- [x] Temperature settings documented
- [x] GraniteTiny configurations catalogued
- [x] LM Studio integration points mapped
- [x] SWE-Bench configurations documented
- [x] File inventory completed
- [x] Implementation recommendations provided

---

**Document Status**: ✅ Complete  
**Last Updated**: December 30, 2025  
**Scope**: Comprehensive configuration inventory  
**Coverage**: 100% of temperature, GraniteTiny, and LM Studio configurations
