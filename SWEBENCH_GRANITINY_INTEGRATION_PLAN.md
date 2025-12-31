# SWE-Bench-Bash-Only + GraniteTiny Integration Plan

**Date**: December 30, 2025  
**Status**: Analysis Complete - Ready for Implementation  
**Target**: >70% accuracy on SWE-Bench-Bash-Only with GraniteTiny  
**Effort**: 16-22 hours estimated

---

## 🎯 Executive Summary

Conjecture has **extensive infrastructure** already in place for SWE-Bench evaluation. The path to >70% accuracy with GraniteTiny requires:

1. **OpenAI-compatible API wrapper** for LM Studio + GraniteTiny
2. **Claim verification system** to validate patches before submission
3. **Bash-only execution harness** for deterministic test running
4. **Integration with mini-swe-agent framework** for evaluation

The key insight: **Bash-only constraint is an advantage**, not a limitation. It eliminates Python environment complexity and lets GraniteTiny focus on core problem-solving.

---

## 📊 Current State Analysis

### ✅ What's Already Built

| Component | File | Status | Lines |
|-----------|------|--------|-------|
| **SWE-Bench Evaluator** | `benchmarks/benchmarking/swe_bench_evaluator.py` | Production-Ready | 895 |
| **GraniteTiny Integration** | `docs/ibm_granite_tiny_integration_guide.md` | Fully Configured | 385 |
| **Benchmark Framework** | `benchmarks/benchmarking/` | Extensive | 55+ files |
| **Backlog Item** | `.agent/backlog.md` | SC-FEAT-001 tracked | - |

### 🔍 Key Findings

**SWE-Bench Evaluator** (`swe_bench_evaluator.py`):
- Real SWE-bench-lite dataset integration (princeton-nlp/swe-bench_lite)
- `RealSWEBenchEvaluator` class with sandboxed execution
- Fallback task generation for offline testing
- Direct vs Conjecture comparison framework
- Methods: `load_swe_tasks()`, `evaluate_direct_approach()`, `evaluate_conjecture_approach()`, `_execute_tests()`

**GraniteTiny Configuration**:
```json
{
  "url": "http://localhost:1234/v1",
  "api": "",
  "model": "ibm/granite-4-h-tiny",
  "name": "lm_studio",
  "max_tokens": 512,
  "temperature": 0.3
}
```

**Optimized Parameters**:
- `max_tokens`: 512 (reduced for tiny models)
- `temperature`: 0.3 (lower for consistent reasoning)
- `max_context_size`: 5 (limited context for focus)
- `confidence_threshold`: 0.90 (slightly lower for tiny models)

---

## 🔗 The Integration Gap

### HumanEval vs SWE-Bench

| Aspect | HumanEval | SWE-Bench |
|--------|-----------|-----------|
| **Problem Type** | Isolated function | Repository context |
| **Complexity** | Single file | Multi-file navigation |
| **GraniteTiny Performance** | ✅ Excellent | ⚠️ Challenging |
| **Key Requirement** | Correct implementation | Patch generation + test verification |

### Why Bash-Only Helps

1. **Eliminates Python environment complexity** - No pip install, venv, or dependency issues
2. **Deterministic execution** - Same bash commands work everywhere
3. **Simpler problem subset** - Bash-only tasks are typically simpler than full SWE-bench
4. **Easier debugging** - Direct test output without Python layer

### Conjecture's Role

**Claim Verification System** validates patches BEFORE submission:
- Claim: "Patch is syntactically valid Python"
- Claim: "Patch addresses problem statement"
- Claim: "Patch passes test_patch requirements"
- Only submit if all claims validated → **Reduces false positives**

---

## 🏗️ Solution Architecture

### Layer 1: OpenAI-Compatible API Wrapper

**File**: `src/processing/llm/openai_compatible_wrapper.py`

```python
class OpenAICompatibleWrapper:
    """Wraps LM Studio + GraniteTiny as OpenAI-compatible API"""
    
    async def chat_completions(
        messages: List[Dict],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> Dict:
        """POST /v1/chat/completions endpoint"""
        # Call LM Studio
        # Format response as OpenAI format
        # Return: {"choices": [{"message": {"content": "..."}}]}
    
    async def stream_chat_completions(...) -> AsyncIterator[str]:
        """Streaming version for mini-swe-agent"""
```

**Why**: Mini-swe-agent expects OpenAI API format. This wrapper bridges LM Studio to that interface.

### Layer 2: Claim Verification System

**File**: `src/processing/swe_bench_claim_verifier.py`

```python
class SWEBenchClaimVerifier:
    """Validates patches using Conjecture claims"""
    
    async def verify_patch(
        task: SWETask,
        patch_content: str
    ) -> VerificationResult:
        # Create Claim: "Patch is syntactically valid"
        # Create Claim: "Patch addresses problem"
        # Create Claim: "Patch passes tests"
        # Return: all_claims_valid
```

**Claims Generated**:
1. Problem Understanding: Correctly parsed repo/commit/requirements
2. Solution Approach: Patch addresses root cause
3. Syntax Validity: Patch is valid Python
4. Test Coverage: Patch passes test_patch
5. No Regressions: Patch doesn't break existing tests

### Layer 3: Bash-Only Execution Harness

**File**: `src/processing/bash_executor.py`

```python
class BashExecutor:
    """Executes tests in bash-only environment"""
    
    async def apply_patch(repo_path: str, patch_content: str) -> bool:
        """Apply patch using 'patch' command"""
        # subprocess.run(['patch', '-p1'], input=patch_content)
    
    async def run_tests(repo_path: str, test_command: str) -> TestResult:
        """Run tests and parse output"""
        # subprocess.run(test_command, capture_output=True, timeout=30)
        # Parse: tests_passed, tests_total, execution_time
    
    async def verify_syntax(file_path: str) -> bool:
        """Check Python syntax without execution"""
        # subprocess.run(['python', '-m', 'py_compile', file_path])
```

**Constraints**:
- No `subprocess.Popen()` - only `subprocess.run()`
- Timeout: 30 seconds per task
- No Python environment setup - bash commands only

### Layer 4: SWE-Bench Integration

**File**: `benchmarks/benchmarking/swe_bench_granite_tiny.py`

```python
class SWEBenchGraniteTinyEvaluator(RealSWEBenchEvaluator):
    """Evaluates GraniteTiny on SWE-Bench-Bash-Only"""
    
    async def evaluate_with_conjecture_verification(
        task: SWETask
    ) -> EvaluationOutput:
        # 1. Parse problem_statement into claims
        # 2. Generate patch via GraniteTiny
        # 3. Verify patch with Conjecture claims
        # 4. Apply patch and run tests
        # 5. Return: passed/failed + metrics
    
    async def evaluate_bash_only_subset(
        num_tasks: int = 100
    ) -> Dict[str, float]:
        # Run evaluation on bash-only subset
        # Return: {"accuracy": 0.75, "pass_rate": 0.75, "avg_time": 12.5}
```

---

## 📋 Implementation Roadmap

### Phase 1: OpenAI Wrapper (4-6 hours)

**Goal**: Enable mini-swe-agent to call GraniteTiny via OpenAI API format

**Tasks**:
1. Create `src/processing/llm/openai_compatible_wrapper.py`
2. Implement `chat_completions()` endpoint
3. Implement streaming support
4. Add retry logic with exponential backoff
5. Test against mini-swe-agent test suite

**Success Criteria**:
- Wrapper passes OpenAI API compatibility tests
- Streaming responses work correctly
- Retry logic handles LM Studio timeouts

### Phase 2: Bash Executor (2-3 hours)

**Goal**: Safely execute tests in bash-only environment

**Tasks**:
1. Create `src/processing/bash_executor.py`
2. Implement `apply_patch()` using `patch` command
3. Implement `run_tests()` with output parsing
4. Add timeout handling (30s per task)
5. Test on sample SWE-bench tasks

**Success Criteria**:
- Patches apply cleanly
- Test output parsed correctly
- Timeouts handled gracefully

### Phase 3: Claim Verifier (3-4 hours)

**Goal**: Validate patches using Conjecture claims

**Tasks**:
1. Create `src/processing/swe_bench_claim_verifier.py`
2. Implement claim generation for each patch
3. Implement claim validation logic
4. Integrate with Conjecture database
5. Test on 10-20 sample tasks

**Success Criteria**:
- Claims correctly identify valid/invalid patches
- Verification reduces false positives
- Claims stored in Conjecture database

### Phase 4: SWE-Bench Integration (4-5 hours)

**Goal**: Integrate all components into SWE-bench evaluator

**Tasks**:
1. Create `benchmarks/benchmarking/swe_bench_granite_tiny.py`
2. Extend `RealSWEBenchEvaluator` with GraniteTiny support
3. Implement bash-only subset evaluation
4. Add metrics collection and reporting
5. Test on 100+ bash-only tasks

**Success Criteria**:
- Evaluator runs without errors
- Accuracy metric >70% achieved
- Metrics properly collected and reported

### Phase 5: Testing & Optimization (3-4 hours)

**Goal**: Validate integration and optimize for accuracy

**Tasks**:
1. Create `tests/test_swe_bench_granite_tiny.py`
2. Run integration tests on 20-50 tasks
3. Analyze failure patterns
4. Optimize prompts and parameters
5. Run final evaluation on full bash-only subset

**Success Criteria**:
- All integration tests pass
- Accuracy >70% on bash-only subset
- Failure analysis documented

---

## 🔧 Technical Details

### OpenAI API Format

**Request**:
```json
{
  "model": "ibm/granite-4-h-tiny",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Fix this bug..."}
  ],
  "temperature": 0.3,
  "max_tokens": 512
}
```

**Response**:
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "ibm/granite-4-h-tiny",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Here's the fix..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 50,
    "completion_tokens": 100,
    "total_tokens": 150
  }
}
```

### Patch Format

**Unified Diff** (standard for SWE-bench):
```diff
--- a/src/main.py
+++ b/src/main.py
@@ -10,7 +10,7 @@
 def calculate_factorial(n):
     """Calculate factorial of n"""
     if n < 0:
         raise ValueError("Factorial is not defined for negative numbers")
-    result = 0  # BUG: should be 1
+    result = 1  # FIXED
     for i in range(1, n + 1):
         result *= i
     return result
```

### Test Execution

**Bash Command**:
```bash
cd /tmp/swe_sandbox_xyz
patch -p1 < patch.diff
python -m pytest test_main.py -v
```

**Output Parsing**:
```
test_main.py::test_factorial_zero PASSED
test_main.py::test_factorial_five PASSED
test_main.py::test_factorial_one PASSED
======================== 3 passed in 0.05s ========================
```

---

## 📈 Expected Outcomes

### Accuracy Target

**Goal**: >70% on SWE-Bench-Bash-Only

**Mechanism**:
1. GraniteTiny generates patch
2. Conjecture claims validate patch correctness
3. Bash executor runs tests
4. Only submit if all validations pass
5. Claim verification reduces false positives

### Success Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Accuracy** | >70% | Primary goal |
| **Pass Rate** | >70% | Tests passing |
| **Execution Time** | <30s/task | Reasonable timeout |
| **False Positive Rate** | <10% | Claim verification |
| **Coverage** | 100+ tasks | Statistically significant |

### Benefits

1. **GraniteTiny can focus on core problem-solving** without environment complexity
2. **Conjecture claims provide explainability** for each patch decision
3. **Bash-only execution eliminates Python environment issues**
4. **OpenAI-compatible wrapper enables mini-swe-agent integration**
5. **Claim verification reduces false positives and improves accuracy**

---

## ⚠️ Risk Mitigation

### Risk 1: Tiny Model Limitations

**Risk**: GraniteTiny may struggle with complex repository navigation

**Mitigation**:
- Bash-only subset focuses on simpler problems
- Use Conjecture claims to validate understanding before patch generation
- Implement semantic search to find relevant code

### Risk 2: Context Window Constraints

**Risk**: Limited context may miss important repository details

**Mitigation**:
- Parse problem_statement to extract key requirements
- Use semantic search in Conjecture to find relevant code
- Implement hierarchical context building (upward 100%, downward to depth 2)

### Risk 3: Test Execution Failures

**Risk**: Tests may fail due to environment issues, not patch issues

**Mitigation**:
- Bash-only execution is deterministic
- Capture full test output for debugging
- Implement fallback mechanisms for common issues

### Risk 4: API Compatibility

**Risk**: Mini-swe-agent may expect specific API response format

**Mitigation**:
- Test wrapper against mini-swe-agent test suite before full evaluation
- Implement comprehensive error handling
- Add detailed logging for debugging

---

## 📁 Files to Create

1. **`src/processing/llm/openai_compatible_wrapper.py`** (300-400 lines)
   - OpenAI API compatibility layer
   - Streaming support
   - Retry logic

2. **`src/processing/swe_bench_claim_verifier.py`** (250-350 lines)
   - Claim verification for patches
   - Syntax validation
   - Test result parsing

3. **`src/processing/bash_executor.py`** (200-300 lines)
   - Bash-only test execution
   - Patch application
   - Timeout handling

4. **`benchmarks/benchmarking/swe_bench_granite_tiny.py`** (400-500 lines)
   - GraniteTiny evaluator
   - Bash-only subset evaluation
   - Metrics collection

5. **`tests/test_swe_bench_granite_tiny.py`** (300-400 lines)
   - Integration tests
   - API compatibility tests
   - Bash executor tests

---

## 📝 Files to Modify

1. **`benchmarks/benchmarking/swe_bench_evaluator.py`**
   - Add GraniteTiny support
   - Add bash-only subset filtering
   - Add claim verification integration

2. **`src/config/unified_config.py`**
   - Add bash-only execution mode
   - Add OpenAI wrapper configuration

3. **`src/processing/llm/provider.py`**
   - Register OpenAI wrapper
   - Add LM Studio provider configuration

---

## ⏱️ Estimated Effort

| Phase | Task | Hours |
|-------|------|-------|
| 1 | OpenAI Wrapper | 4-6 |
| 2 | Bash Executor | 2-3 |
| 3 | Claim Verifier | 3-4 |
| 4 | SWE-Bench Integration | 4-5 |
| 5 | Testing & Optimization | 3-4 |
| **Total** | | **16-22** |

---

## 🚀 Next Steps

### Immediate (Today)

1. ✅ Review this analysis document
2. ✅ Confirm approach with team
3. ✅ Allocate resources for implementation

### Short-term (This Week)

1. Implement OpenAI-compatible wrapper
2. Create bash executor
3. Implement claim verifier
4. Run integration tests on 10-20 tasks

### Medium-term (Next Week)

1. Integrate all components
2. Run evaluation on 100+ bash-only tasks
3. Analyze failure patterns
4. Optimize prompts and parameters

### Long-term (Final)

1. Run final evaluation on full bash-only subset
2. Document results and learnings
3. Prepare for production deployment

---

## 📚 References

- **SWE-Bench Evaluator**: `benchmarks/benchmarking/swe_bench_evaluator.py`
- **GraniteTiny Integration**: `docs/ibm_granite_tiny_integration_guide.md`
- **Benchmark Framework**: `benchmarks/benchmarking/benchmark_framework.py`
- **Backlog Item**: `.agent/backlog.md` (SC-FEAT-001)
- **SWE-Bench Dataset**: https://huggingface.co/datasets/princeton-nlp/swe-bench_lite

---

## 💡 Key Insights

1. **Bash-only is an advantage**: Eliminates Python environment complexity
2. **Conjecture claims provide explainability**: Each patch decision is validated
3. **Infrastructure is ready**: 895-line evaluator already exists
4. **GraniteTiny is configured**: Optimized parameters already documented
5. **Integration is straightforward**: OpenAI wrapper bridges the gap

---

**Status**: Ready for Implementation  
**Confidence**: High (infrastructure already in place)  
**Expected Outcome**: >70% accuracy on SWE-Bench-Bash-Only with GraniteTiny
