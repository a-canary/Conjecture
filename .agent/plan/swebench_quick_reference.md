# SWE-Bench & GraniteTiny Quick Reference Guide

**Last Updated**: December 30, 2025  
**Status**: Ready for Implementation

---

## 🚀 Quick Start

### 1. Verify GraniteTiny Configuration

```bash
# Check config
python -c "
import json
with open('.conjecture/config.json') as f:
    config = json.load(f)
    lm_studio = next((p for p in config['providers'] if p['name'] == 'lm_studio'), None)
    if lm_studio:
        print('✅ LM Studio configured:', lm_studio['model'])
    else:
        print('❌ LM Studio not found')
"
```

### 2. Start LM Studio

```bash
# Install LM Studio from https://lmstudio.ai/
# Load model: ibm/granite-4-h-tiny
# Verify running on http://localhost:1234
curl http://localhost:1234/v1/models
```

### 3. Run SWE-Bench Evaluator

```bash
# Basic evaluation
python benchmarks/benchmarking/swe_bench_evaluator.py

# With custom task count
python -c "
import asyncio
from benchmarks.benchmarking.swe_bench_evaluator import RealSWEBenchEvaluator

async def main():
    evaluator = RealSWEBenchEvaluator()
    await evaluator.initialize_conjecture()
    
    # Load 5 tasks
    tasks = await evaluator.load_swe_tasks(num_tasks=5)
    
    # Evaluate
    results = await evaluator.evaluate_models_on_tasks(['gpt-4'], tasks)
    
    # Print results
    print(json.dumps(results, indent=2, default=str))
    
    await evaluator.cleanup()

asyncio.run(main())
"
```

---

## 📁 Key Files

### SWE-Bench Evaluator
**Path**: `benchmarks/benchmarking/swe_bench_evaluator.py`  
**Lines**: 895  
**Status**: ✅ Production-ready

**Key Classes**:
- `RealSWEBenchEvaluator` - Main evaluator
- `SWETask` - Task representation
- `EvaluationOutput` - Results
- `EvaluationResult` - Status enum

**Key Methods**:
```python
# Load tasks
tasks = await evaluator.load_swe_tasks(num_tasks=5)

# Evaluate direct approach
direct = await evaluator.evaluate_direct_approach(task)

# Evaluate Conjecture approach
conjecture = await evaluator.evaluate_conjecture_approach(task)

# Compare models
results = await evaluator.evaluate_models_on_tasks(models, tasks)
```

### GraniteTiny Integration Guide
**Path**: `docs/ibm_granite_tiny_integration_guide.md`  
**Lines**: 385  
**Status**: ✅ Fully configured

**Configuration**:
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

---

## 🎯 Success Criteria

### SC-FEAT-001: SWE-Bench-Bash-Only Accuracy

**Target**: >70% accuracy on SWE-Bench-Bash-Only  
**Status**: Promoted to success criteria  
**Plan**: `.agent/plan/swebench_enhancement.md`

**Approach**:
1. Context engineering for bash-specific tasks
2. Prompt refinement for shell scripting
3. GraniteTiny + Conjecture optimization
4. Comparison with direct LLM approach

**Metrics**:
- Accuracy on bash-only subset
- Execution time
- Confidence scores
- Error analysis

---

## 📊 Benchmark Framework

### Available Benchmarks

| Benchmark | File | Status |
|-----------|------|--------|
| AIME 2025 | `benchmark_framework.py` | ✅ Ready |
| GPQA | `benchmark_framework.py` | ✅ Ready |
| SWE-Bench | `swe_bench_evaluator.py` | ✅ Ready |
| LiveCodeBench | `benchmark_framework.py` | ✅ Ready |
| DeepEval | `deepeval_integration.py` | ✅ Ready |
| HumanEval | `cycle16_multi_benchmark_framework.py` | ✅ Ready |
| ARC Easy | `cycle16_multi_benchmark_framework.py` | ✅ Ready |

### Evaluation Approaches

```python
# Direct LLM
direct_result = await evaluator.evaluate_direct_approach(task)

# Conjecture-enhanced
conjecture_result = await evaluator.evaluate_conjecture_approach(task)

# LLM Judge
judge_result = await evaluator.evaluate_with_llm_judge(task)

# Multi-model
results = await evaluator.evaluate_models_on_tasks(models, tasks)
```

---

## 🔧 Configuration

### Provider Configuration

**File**: `.conjecture/config.json`

```json
{
  "providers": [
    {
      "url": "http://localhost:1234/v1",
      "api": "",
      "model": "ibm/granite-4-h-tiny",
      "name": "lm_studio",
      "priority": 1,
      "is_local": true,
      "max_tokens": 512,
      "temperature": 0.3
    }
  ],
  "confidence_threshold": 0.95,
  "max_context_size": 10,
  "debug": false,
  "database_path": "data/conjecture.db"
}
```

### Tiny Model Configuration

**File**: `src/config/tiny_model_config.py`

```python
@dataclass
class TinyModelConfig:
    model_name: str = "ibm/granite-4-h-tiny"
    max_tokens: int = 512
    temperature: float = 0.3
    max_context_size: int = 5
    confidence_threshold: float = 0.90
    use_json_frontmatter: bool = True
    enable_confidence_boosting: bool = True
```

---

## 📈 Performance Targets

### GraniteTiny Expected Performance

| Metric | Target | Status |
|--------|--------|--------|
| Claim Generation Success | 90%+ | ✅ Expected |
| Response Time | <5s | ✅ Expected |
| JSON Parsing Rate | 95%+ | ✅ Expected |
| Confidence Quality | 0.8-0.95 | ✅ Expected |

### SWE-Bench Targets

| Benchmark | Target | Status |
|-----------|--------|--------|
| SWE-Bench-Bash-Only | >70% | 🔄 In Progress |
| AIME2025 | Maintain/Improve | 🔄 In Progress |
| LiveCodeBench v6 | Maintain/Improve | 🔄 In Progress |

---

## 🔍 Troubleshooting

### LM Studio Connection Issues

```bash
# Check if LM Studio is running
curl http://localhost:1234/v1/models

# Expected response:
# {"object":"list","data":[{"id":"ibm/granite-4-h-tiny",...}]}
```

### Configuration Issues

```bash
# Verify configuration
python conjecture config

# Check provider availability
python conjecture backends

# Test provider connectivity
python conjecture health
```

### Generation Problems

```bash
# Test simple generation
curl -X POST http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ibm/granite-4-h-tiny",
    "messages": [{"role": "user", "content": "Test"}],
    "max_tokens": 50,
    "temperature": 0.3
  }'
```

---

## 📚 Related Documentation

### Primary Docs
- `docs/ibm_granite_tiny_integration_guide.md` - GraniteTiny setup
- `SWEBENCH_EXPLORATION_REPORT.md` - Comprehensive exploration report

### Backlog
- `.agent/backlog.md` - SC-FEAT-001 (SWE-Bench target)
- `.agent/plan/swebench_enhancement.md` - Enhancement plan

### Architecture
- `specs/architecture.md` - 4-Layer architecture
- `docs/architecture/main.md` - Architecture specification

---

## 🎓 Code Examples

### Basic SWE-Bench Evaluation

```python
import asyncio
from benchmarks.benchmarking.swe_bench_evaluator import RealSWEBenchEvaluator

async def evaluate_swe_bench():
    evaluator = RealSWEBenchEvaluator()
    await evaluator.initialize_conjecture()
    
    # Load tasks
    tasks = await evaluator.load_swe_tasks(num_tasks=5)
    
    # Evaluate
    results = await evaluator.evaluate_models_on_tasks(['gpt-4'], tasks)
    
    # Print statistics
    stats = evaluator.get_statistics()
    print(f"Completed: {stats['evaluations_completed']}")
    print(f"Success Rate: {stats['success_rate']:.1%}")
    
    await evaluator.cleanup()

asyncio.run(evaluate_swe_bench())
```

### Custom Benchmark

```python
from benchmarks.benchmarking.benchmark_framework import Benchmark, BenchmarkTask

class CustomBenchmark(Benchmark):
    async def load_tasks(self):
        return [
            BenchmarkTask(
                task_id="task_1",
                prompt="Your prompt here",
                expected_answer="Expected answer"
            )
        ]
    
    def evaluate_response(self, task, response):
        return response.strip() == task.expected_answer

# Run benchmark
benchmark = CustomBenchmark("custom")
summary = await benchmark.run_benchmark(model_func, "model_name")
print(f"Accuracy: {summary.accuracy:.1%}")
```

### Direct vs Conjecture Comparison

```python
async def compare_approaches(task):
    evaluator = RealSWEBenchEvaluator()
    
    # Direct approach
    direct = await evaluator.evaluate_direct_approach(task)
    
    # Conjecture approach
    conjecture = await evaluator.evaluate_conjecture_approach(task)
    
    # Compare
    improvement = evaluator._calculate_improvement(direct, conjecture)
    
    print(f"Direct: {direct.result.value} ({direct.execution_time:.2f}s)")
    print(f"Conjecture: {conjecture.result.value} ({conjecture.execution_time:.2f}s)")
    print(f"Improvement: {improvement:.1%}")
```

---

## 🚀 Implementation Roadmap

### Phase 1: Baseline (Week 1)
- [ ] Verify GraniteTiny configuration
- [ ] Run baseline SWE-Bench evaluation
- [ ] Document current performance
- [ ] Establish metrics baseline

### Phase 2: Optimization (Week 2-3)
- [ ] Implement context engineering
- [ ] Refine prompt templates
- [ ] Run comprehensive comparison
- [ ] Analyze results

### Phase 3: Enhancement (Week 4)
- [ ] Achieve >70% accuracy target
- [ ] Maintain other benchmark scores
- [ ] Document techniques
- [ ] Create reusable patterns

### Phase 4: Scaling (Month 2)
- [ ] Extend to other benchmarks
- [ ] Optimize for different domains
- [ ] Build knowledge foundation
- [ ] Implement advanced techniques

---

## 📞 Support

### Key Contacts
- **SWE-Bench Evaluator**: `benchmarks/benchmarking/swe_bench_evaluator.py`
- **GraniteTiny Integration**: `docs/ibm_granite_tiny_integration_guide.md`
- **Benchmark Framework**: `benchmarks/benchmarking/benchmark_framework.py`

### Resources
- HuggingFace SWE-Bench: https://huggingface.co/datasets/princeton-nlp/swe-bench_lite
- LM Studio: https://lmstudio.ai/
- IBM Granite Models: https://github.com/ibm-granite/granite-code-models

---

**Last Updated**: December 30, 2025  
**Status**: Ready for Implementation  
**Next Review**: After Phase 1 completion
