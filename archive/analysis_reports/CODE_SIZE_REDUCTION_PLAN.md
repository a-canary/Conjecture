# Code Size Reduction Plan
**Date**: 2025-12-17  
**Status**: CRITICAL - Currently 8.4x OVER BUDGET

## Current State vs. Targets

| Category | Current | Target | Over Budget | % Over |
|----------|---------|--------|-------------|--------|
| **Product Source** | **83,734 lines** | **10,000 lines** | **+73,734** | **+837%** 🚨 |
| **Test Code** | **12,022 lines** | **10,000 lines** | **+2,022** | **+20%** ⚠️ |
| **Benchmark Code** | **30,691 lines** | **10,000 lines** | **+20,691** | **+307%** 🚨 |
| **TOTAL** | **126,447 lines** | **30,000 lines** | **+96,447** | **+422%** 🚨 |

## Critical Analysis

### Product Source Code: 83,734 lines → 10,000 target

**Reduction Needed**: 73,734 lines (88% reduction required!)

#### Top Bloat Categories

**1. Benchmarking in src/ (Should be in benchmarks/)**: ~5,000+ lines
- `src/benchmarking/scaled_50_test_framework.py` (1,022 lines)
- `src/benchmarking/gpt_oss_scaled_test.py` (968 lines)
- `src/benchmarking/swe_bench_evaluator.py` (895 lines)
- `src/benchmarking/improvement_cycle_agent.py` (840 lines)
- **Action**: Move to `benchmarks/` or `research/`, remove from src/

**2. Scaling/Monitoring Infrastructure**: ~3,000+ lines
- `src/scaling/scaling_benchmarks.py` (916 lines)
- `src/scaling/resource_monitor.py` (899 lines)
- `src/scaling/database_isolation.py` (882 lines)
- `src/monitoring/metrics_visualization.py` (895 lines)
- `src/monitoring/retry_tracker.py` (818 lines)
- `src/monitoring/metrics_analysis.py` (792 lines)
- **Action**: Critical path only, archive rest

**3. Duplicate/Redundant CLIs**: ~2,500+ lines
- `src/cli/modular_cli.py` (913 lines)
- `src/cli/enhanced_modular_cli.py` (885 lines)
- `src/cli/ui_enhancements.py` (773 lines)
- **Action**: Consolidate to ONE CLI implementation

**4. Multiple SQLite Managers**: ~1,500+ lines
- `src/data/enhanced_sqlite_manager.py` (833 lines)
- `src/data/optimized_sqlite_manager.py` (662 lines)
- **Action**: Keep ONE, archive others

**5. Excessive Prompt Templates**: ~3,500+ lines
- `src/processing/llm_prompts/xml_optimized_templates.py` (964 lines)
- `src/processing/llm_prompts/tiny_llm_optimizer.py` (854 lines)
- `src/processing/llm_prompts/template_evolution.py` (747 lines)
- `src/processing/llm_prompts/template_manager.py` (743 lines)
- **Action**: Consolidate to 2-3 essential templates

**6. Large Monolithic Files**: ~3,000+ lines
- `src/endpoint_app.py` (1,103 lines - should be modular)
- `src/conjecture.py` (740 lines)
- `src/stats/main.py` (853 lines)
- **Action**: Break into focused modules

**7. Dead Code (86% reported)**: ~50,000+ lines estimated
- Archive directory analysis suggests massive duplication
- 214 source files but only ~30 actively used
- **Action**: Aggressive purge of unused code

#### Reduction Strategy (Product Source)

**Phase 1: Quick Wins (70% reduction, -58,000 lines)**
1. Move benchmarking from src/ to benchmarks/ (-5,000 lines)
2. Archive scaling/monitoring non-critical code (-2,500 lines)
3. Delete duplicate CLI implementations (-1,500 lines)
4. Delete duplicate SQLite managers (-800 lines)
5. Archive dead code (86% of files) (-48,000 lines)

**Phase 2: Consolidation (15% reduction, -12,000 lines)**
1. Consolidate prompt templates to 3 files max (-2,500 lines)
2. Break up monolithic files into focused modules (-1,500 lines)
3. Remove unused processing/support systems (-3,000 lines)
4. Consolidate agent/harness code (-2,000 lines)
5. Remove experimental features not in production (-3,000 lines)

**Phase 3: Final Optimization (3% reduction, -3,700 lines)**
1. Optimize remaining code for brevity
2. Remove redundant error handling
3. Consolidate similar functions
4. Simplify complex implementations

**Target Result**: 10,000 lines of focused, production-ready code

### Test Code: 12,022 lines → 10,000 target

**Reduction Needed**: 2,022 lines (17% reduction)

**Strategy**:
1. Consolidate redundant test utilities (-500 lines)
2. Remove obsolete test files for deleted code (-1,000 lines)
3. Optimize verbose test implementations (-500 lines)
4. Keep comprehensive coverage for core modules

**Lower Priority**: Only 20% over budget

### Benchmark Code: 30,691 lines → 10,000 target

**Reduction Needed**: 20,691 lines (67% reduction)

**Strategy**:
1. Move benchmark code from src/ to benchmarks/ (+5,000 lines initially)
2. Consolidate duplicate experiment scripts (-10,000 lines)
3. Archive old/obsolete benchmarks (-8,000 lines)
4. Keep only active, maintained benchmark suites (-2,691 lines)

## Implementation Plan

### Week 1: Emergency Reduction (Priority 1)

**Goal**: Reduce product source from 83,734 → 30,000 lines (-53,734)

**Actions**:
1. **Day 1**: Identify and archive 86% dead code
   - Run dependency analysis
   - Archive unused files
   - Target: -40,000 lines

2. **Day 2**: Move benchmarking out of src/
   - Move to benchmarks/ or research/
   - Update imports
   - Target: -5,000 lines

3. **Day 3**: Delete duplicate implementations
   - Choose ONE CLI (delete others)
   - Choose ONE SQLite manager (delete others)
   - Choose ONE of each component
   - Target: -3,000 lines

4. **Day 4**: Archive non-critical infrastructure
   - Archive scaling benchmarks
   - Archive monitoring visualizations
   - Keep only essential monitoring
   - Target: -3,000 lines

5. **Day 5**: Consolidate templates and large files
   - Merge prompt templates
   - Break up monolithic files
   - Target: -2,734 lines

**Checkpoint**: Should be at ~30,000 lines (10k over budget)

### Week 2: Deep Optimization (Priority 2)

**Goal**: Reduce product source from 30,000 → 15,000 lines (-15,000)

**Actions**:
1. Analyze remaining code for duplication
2. Consolidate similar functionality
3. Simplify complex implementations
4. Remove experimental features
5. Optimize for brevity and clarity

**Checkpoint**: Should be at ~15,000 lines (5k over budget)

### Week 3: Final Polish (Priority 3)

**Goal**: Reduce product source from 15,000 → 10,000 lines (-5,000)

**Actions**:
1. Critical path analysis - keep only essential
2. Inline small helper modules
3. Remove redundant error handling
4. Optimize imports and structure
5. Final cleanup and documentation

**Target**: 10,000 lines of focused, production code

### Week 4: Test & Benchmark Cleanup

**Goal**: Reduce tests to 10,000 and benchmarks to 10,000

**Actions**:
1. Clean up test files for deleted code (-2,000)
2. Consolidate benchmark scripts (-20,691)
3. Verify all tests still pass
4. Update documentation

## Success Metrics

| Metric | Current | Week 1 | Week 2 | Week 3 | Week 4 (Target) |
|--------|---------|--------|--------|--------|-----------------|
| Product Source | 83,734 | 30,000 | 15,000 | 10,000 | ≤10,000 ✅ |
| Test Code | 12,022 | 12,022 | 11,000 | 10,500 | ≤10,000 ✅ |
| Benchmark Code | 30,691 | 25,000 | 15,000 | 12,000 | ≤10,000 ✅ |
| **TOTAL** | 126,447 | 67,022 | 41,000 | 32,500 | ≤30,000 ✅ |

## Risk Assessment

**High Risk**: Accidentally deleting critical code
- **Mitigation**: Git commits at each step, comprehensive testing after each phase

**Medium Risk**: Breaking functionality during consolidation
- **Mitigation**: Run full test suite after each major change

**Low Risk**: Missing reduction targets
- **Mitigation**: Aggressive approach to dead code removal

## Quick Start (First Actions)

**Immediate Next Steps** (can do in 1 hour):

1. **Run dead code analysis** (10 min):
   ```bash
   # Find files not imported anywhere
   # Find unused functions
   # Generate deletion candidates
   ```

2. **Archive obvious duplicates** (20 min):
   - Move one CLI to archive, keep the other
   - Move benchmarking from src/ to research/
   - Archive old experiment files

3. **Delete confirmed dead code** (20 min):
   - Remove files with 0 imports
   - Remove obsolete experiment scripts
   - Remove archived documentation duplicates

4. **Measure progress** (10 min):
   ```bash
   find src -name "*.py" | xargs wc -l | tail -1
   ```

**Expected Result**: 83,734 → ~60,000 lines (first hour)

## Long-Term Maintenance

**Code Size Enforcement**:
1. Add pre-commit hook to check line counts
2. Fail CI if thresholds exceeded
3. Require justification for any size increases
4. Regular quarterly audits for bloat

**Culture Change**:
- Prefer deletion over addition
- One implementation per concept
- Archive > Delete (but move out of src/)
- Quality over quantity
- Focus over features

---

## Appendix: File Inventory

### Largest Files (Top 30)

```
83,734  TOTAL
 1,103  src/endpoint_app.py
 1,022  src/benchmarking/scaled_50_test_framework.py
   968  src/benchmarking/gpt_oss_scaled_test.py
   964  src/processing/llm_prompts/xml_optimized_templates.py
   916  src/scaling/scaling_benchmarks.py
   913  src/cli/modular_cli.py
   899  src/scaling/resource_monitor.py
   895  src/monitoring/metrics_visualization.py
   895  src/benchmarking/swe_bench_evaluator.py
   885  src/cli/enhanced_modular_cli.py
   882  src/scaling/database_isolation.py
   856  src/processing/async_eval.py
   854  src/processing/llm_prompts/tiny_llm_optimizer.py
   853  src/stats/main.py
   840  src/benchmarking/improvement_cycle_agent.py
   833  src/data/enhanced_sqlite_manager.py
   818  src/monitoring/retry_tracker.py
   803  src/processing/context_collector.py
   792  src/monitoring/metrics_analysis.py
   792  src/agent/prompt_system.py
   773  src/cli/ui_enhancements.py
   747  src/processing/llm_prompts/template_evolution.py
   743  src/processing/llm_prompts/template_manager.py
   740  src/conjecture.py
   695  src/processing/error_handling.py
   691  src/processing/tool_executor.py
   669  src/processing/support_systems/persistence_layer.py
   662  src/data/optimized_sqlite_manager.py
   651  src/processing/enhanced_llm_router.py
```

**Action Required**: IMMEDIATE code size reduction plan execution
