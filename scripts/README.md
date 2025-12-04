# Conjecture Scripts

This directory contains scripts for testing and running the Conjecture LLM provider and model comparisons.

## Scripts Overview

### 1. `start_conjecture_provider.py`
Starts the Conjecture local LLM provider server on port 5678.

**Usage:**
```bash
python scripts/start_conjecture_provider.py
```

**Endpoints:**
- POST /v1/chat/completions - Main chat endpoint
- POST /tools/tell_user - TellUser tool
- POST /tools/ask_user - AskUser tool
- GET /models - List models
- GET /health - Health check

### 2. `test_conjecture_provider.py`
Tests all endpoints of the Conjecture provider to verify functionality.

**Usage:**
```bash
python scripts/test_conjecture_provider.py
```

**Tests:**
- Health check endpoint
- Models listing endpoint
- Chat completions endpoint
- TellUser tool endpoint
- AskUser tool endpoint
- Root endpoint

### 3. `run_4model_comparison.py`
Runs a comprehensive 4-model comparison test as specified in the TODO.

**Models Tested:**
1. ibm/granite-4-h-tiny (direct from LM Studio)
2. ibm/granite-4-h-tiny (via Conjecture on port 5678)
3. GLM-4.6 (direct from Chutes API)
4. GLM-4.6 (via Conjecture on port 5678)

**Test Categories:**
- Local test cases from research/test_cases/
- Huggingface datasets (MMLU, TruthfulQA, HumanEval)
- Custom hallucination tests
- Impossible question tests

**Usage:**
```bash
python scripts/run_4model_comparison.py
```

**Requirements:**
- LM Studio running on port 1234 with ibm/granite-4-h-tiny loaded
- Conjecture provider running on port 5678
- CHUTES_API_KEY environment variable set (for GLM-4.6 tests)

### 4. `run_all_tests.bat` (Windows)
Batch script to run all tests in sequence.

**Usage:**
```bash
scripts\run_all_tests.bat
```

**Steps:**
1. Starts Conjecture Provider
2. Tests Provider Functionality
3. Runs 4-Model Comparison

## Environment Setup

1. Install required dependencies:
```bash
pip install aiohttp datasets
```

2. Set environment variables:
```bash
export CHUTES_API_KEY=your_api_key_here
```

3. Start required services:
- LM Studio with ibm/granite-4-h-tiny model on port 1234
- Conjecture provider (use start_conjecture_provider.py)

## Output Files

Results are saved to:
- `research/results/4model_comparison_results_[timestamp].json` - Detailed test results
- `research/results/4model_comparison_summary_[timestamp].json` - Summary statistics
- `research/results/4model_comparison_report_[timestamp].md` - Comparison report

## Troubleshooting

1. **Port conflicts**: Ensure port 5678 is free for the Conjecture provider
2. **LM Studio not running**: Start LM Studio and load the required model
3. **API key issues**: Verify CHUTES_API_KEY is set correctly
4. **Dependencies**: Install all required Python packages

## Test Categories Explained

### Hallucination Tests
Tests designed to detect if models generate information about non-existent entities or facts.

### Impossible Question Tests
Tests with mathematically or computationally impossible tasks to see how models handle them.

### Complex Reasoning Tasks
Multi-step problems requiring logical deduction and planning.

### Coding Tasks
Algorithm challenges and debugging scenarios from the HumanEval dataset.