# SWE-Bench Bash-Only Evaluator - Quick Start Guide

## 🚀 Get Started in 2 Minutes

### 1. Verify Installation
```bash
# Check Python syntax
python -m py_compile benchmarks/benchmarking/swe_bench_bash_only_evaluator.py
# ✅ Should complete without errors
```

### 2. Configure Conjecture
```bash
# Check configuration
python conjecture config

# If needed, set up config
mkdir -p ~/.conjecture
cp src/config/default_config.json ~/.conjecture/config.json
# Edit with your API keys
```

### 3. Run Evaluation
```bash
# Interactive menu (recommended)
python benchmarks/benchmarking/run_bash_only_evaluator.py

# Or direct evaluation
python -c "
import asyncio
from benchmarks.benchmarking.swe_bench_bash_only_evaluator import BashOnlySWEBenchEvaluator

async def main():
    evaluator = BashOnlySWEBenchEvaluator()
    await evaluator.initialize_conjecture()
    tasks = await evaluator.load_swe_tasks(10)
    results = await evaluator.evaluate_batch(tasks, 10)
    print(f'Success Rate: {results[\"summary\"][\"success_rate\"]:.1f}%')
    await evaluator.cleanup()

asyncio.run(main())
"
```

## 📊 What You Get

### Results JSON
```json
{
  "results": [
    {
      "task_id": "bash_task_0001_file_processing",
      "result": "passed",
      "execution_time": 12.34,
      "react_iterations": 3,
      "success": true
    }
  ],
  "summary": {
    "total": 10,
    "passed": 8,
    "failed": 2,
    "success_rate": 80.0,
    "average_time": 11.5,
    "total_react_iterations": 28,
    "average_iterations": 2.8
  }
}
```

## 🎯 Key Features

| Feature | Details |
|---------|---------|
| **Temperature** | 0.0 (deterministic) |
| **Loop** | ReAct (Observe → Diagnose → Patch → Verify) |
| **Iterations** | Max 5 per task |
| **Timeout** | 30 seconds per bash command |
| **Context** | <500 tokens (GraniteTiny optimized) |
| **Dataset** | 500 real SWE-bench instances |
| **Time** | ~2-3 min for 10 tasks, ~30-60 min for 500 |

## 📚 Documentation

- **BASH_ONLY_EVALUATOR_README.md** - Comprehensive guide
- **BASH_ONLY_IMPLEMENTATION_SUMMARY.md** - Technical details
- **swe_bench_bash_only_evaluator.py** - Source code (716 lines)

## 🔧 Common Tasks

### Evaluate 10 Tasks (Quick Test)
```python
evaluator = BashOnlySWEBenchEvaluator()
await evaluator.initialize_conjecture()
tasks = await evaluator.load_swe_tasks(10)
results = await evaluator.evaluate_batch(tasks, 10)
```

### Evaluate 500 Tasks (Full Benchmark)
```python
evaluator = BashOnlySWEBenchEvaluator()
await evaluator.initialize_conjecture()
tasks = await evaluator.load_swe_tasks(500)
results = await evaluator.evaluate_batch(tasks, 50)  # Batch size 50
```

### Custom Task
```python
from benchmarks.benchmarking.swe_bench_bash_only_evaluator import SWETask

task = SWETask(
    instance_id="custom_001",
    repo="bash/example",
    base_commit="abc123",
    problem_statement="Your problem here...",
    hints="Helpful hints...",
    test_patch="# Test patch",
    version="1.0"
)

result = await evaluator.evaluate_bash_react(task)
```

## ✅ Verification

```bash
# All files present
ls -lh benchmarks/benchmarking/swe_bench_bash_only_evaluator.py
ls -lh benchmarks/benchmarking/run_bash_only_evaluator.py
ls -lh benchmarks/benchmarking/BASH_ONLY_EVALUATOR_README.md
ls -lh benchmarks/benchmarking/BASH_ONLY_IMPLEMENTATION_SUMMARY.md

# Syntax valid
python -m py_compile benchmarks/benchmarking/swe_bench_bash_only_evaluator.py
python -m py_compile benchmarks/benchmarking/run_bash_only_evaluator.py

# Ready to use
python benchmarks/benchmarking/run_bash_only_evaluator.py
```

## 🆘 Troubleshooting

**"No LLM providers available"**
```bash
python conjecture config
python conjecture backends
```

**"Import errors"**
```bash
export PYTHONPATH=.
pip install -r requirements.txt
```

**"Timeout errors"**
- Bash command took >30 seconds
- Check for infinite loops
- Increase timeout if needed

## 📞 Support

See **BASH_ONLY_EVALUATOR_README.md** for:
- Detailed configuration
- Advanced usage
- Performance tuning
- Integration details
- Complete troubleshooting

---

**Status**: ✅ Production-Ready
**Version**: 1.0.0
**Created**: 2025-12-30
