# Conjecture Hypothesis Validation Test Suite

## Overview

This comprehensive test suite expands Conjecture's hypothesis validation to **50-100 test cases per category** for statistical significance. The suite validates the core hypothesis that:

> *"By decomposing tasks and concepts, and providing relevant context through claims-based representations that include in-context learning examples of task breakdown strategies, research-plan-work-validate phases, scientific method, critical thinking, and fact-checking best practices, small LLMs can achieve performance comparable to larger models on complex reasoning tasks."*

## Architecture

The test suite consists of 6 integrated components:

### 1. **Test Case Generation** (`test_hypothesis_validation.py`)
- Generates 50-100 diverse test cases per category
- 6 core reasoning categories with varying difficulty levels
- Systematic test case metadata and categorization
- Proper controls and randomization

### 2. **A/B Testing Framework** (`test_ab_testing_framework.py`)
- Direct vs Conjecture approach comparisons
- Randomized test assignment and execution
- Comprehensive result collection
- Integrated LLM-as-a-Judge evaluation

### 3. **LLM-as-a-Judge System** (`test_llm_judge.py`)
- GLM-4.6 for consistent evaluation
- 7 weighted evaluation criteria
- Calibration and quality control
- Structured feedback and improvement suggestions

### 4. **Statistical Validation** (`test_statistical_validation.py`)
- Paired t-tests and Wilcoxon signed-rank tests
- Effect size calculations (Cohen's d, Hedges' g)
- Power analysis and confidence intervals
- Multiple comparison corrections

### 5. **Performance Monitoring** (`test_performance_monitoring.py`)
- Real-time resource tracking
- Execution timing and token usage
- Cost estimation and efficiency metrics
- Anomaly detection and alerting

### 6. **Integration Suite** (`test_hypothesis_validation_suite.py`)
- Orchestrates all components
- Progress tracking and reporting
- Configuration management
- Result synthesis and analysis

## Test Categories

### 1. **Complex Reasoning** (75 test cases)
- **Multi-step logic puzzles**: Deductive reasoning, logical fallacies
- **Causal inference**: Confounding variables, correlation vs causation
- **Analytical reasoning**: Problem decomposition, systematic analysis

### 2. **Mathematical Reasoning** (75 test cases)
- **Algebra word problems**: Equation setup and solving
- **Geometric calculations**: Spatial reasoning, formula application
- **Rate/proportion problems**: Ratio analysis, proportional thinking

### 3. **Context Compression** (75 test cases)
- **Long document QA**: Information extraction, relevance filtering
- **Multi-source synthesis**: Evidence integration, conflict resolution
- **Research paper analysis**: Academic comprehension, critical evaluation

### 4. **Evidence Evaluation** (75 test cases)
- **Conflicting evidence assessment**: Source reliability, bias detection
- **Scientific claim evaluation**: Data interpretation, methodology assessment
- **Argument analysis**: Logical structure, evidence quality

### 5. **Task Decomposition** (75 test cases)
- **Project planning scenarios**: Resource allocation, timeline creation
- **Multi-step problem solving**: Sequential reasoning, dependency analysis
- **Strategic planning**: Goal decomposition, milestone setting

### 6. **Coding Tasks** (75 test cases)
- **Agenting capabilities**: System design, decision processes
- **Code generation**: Algorithm design, implementation quality
- **Debugging challenges**: Error analysis, solution optimization

## Evaluation Criteria

### Primary Metrics (Weighted)

1. **Correctness** (Weight: 1.5, Threshold: 0.70)
   - Factual accuracy and correctness
   - Absence of factual errors

2. **Reasoning Quality** (Weight: 1.2, Threshold: 0.65)
   - Depth and rigor of logical reasoning
   - Quality of analytical thinking

3. **Hallucination Reduction** (Weight: 1.3, Threshold: 0.80)
   - Grounding in provided information
   - Absence of fabricated claims

4. **Completeness** (Weight: 1.0, Threshold: 0.75)
   - Thoroughness in addressing all aspects
   - Coverage of relevant subtopics

5. **Coherence** (Weight: 1.0, Threshold: 0.70)
   - Logical flow and structure
   - Consistency in reasoning

6. **Confidence Calibration** (Weight: 1.0, Threshold: 0.60)
   - Appropriate confidence levels
   - Avoidance of overconfidence

7. **Efficiency** (Weight: 0.5, Threshold: 0.60)
   - Conciseness and focus
   - Optimal resource usage

## Statistical Framework

### Significance Testing
- **Alpha Level**: 0.05 (Type I error control)
- **Power Target**: 0.8 (80% statistical power)
- **Effect Size Thresholds**: Small=0.2, Medium=0.5, Large=0.8
- **Multiple Comparison Correction**: Bonferroni method

### Statistical Tests
- **Paired t-test**: For normally distributed differences
- **Wilcoxon signed-rank**: Non-parametric alternative
- **Cohen's d**: Effect size for paired comparisons
- **Hedges' g**: Bias-corrected effect size
- **Confidence Intervals**: 95% CI for all estimates

### Power Analysis
- **Sample Size Calculation**: Required N for target power
- **Achieved Power**: Post-hoc power analysis
- **Effect Size Sensitivity**: Power vs effect size curves

## Usage

### Quick Start
```bash
# Run with default settings (75 test cases per category)
python tests/test_hypothesis_validation_suite.py

# Run with specific sample size
python tests/test_hypothesis_validation_suite.py --sample-size 100

# Run specific categories only
python tests/test_hypothesis_validation_suite.py --categories complex_reasoning mathematical_reasoning

# Quick validation (25 test cases per category)
python tests/test_hypothesis_validation_suite.py --quick

# Use existing test cases
python tests/test_hypothesis_validation_suite.py --use-existing
```

### Configuration
Create a configuration file (`config.json`):
```json
{
  "sample_size_per_category": 75,
  "categories": ["complex_reasoning", "mathematical_reasoning"],
  "tiny_model": "ibm/granite-4-h-tiny",
  "baseline_model": "zai-org/GLM-4.6",
  "judge_model": "zai-org/GLM-4.6",
  "alpha_level": 0.05,
  "target_power": 0.8,
  "generate_plots": true,
  "save_intermediate_results": true,
  "create_comprehensive_report": true,
  "integrate_with_research_framework": true,
  "use_existing_test_cases": false
}
```

### Provider Setup
Configure your providers in `~/.conjecture/config.json`:
```json
{
  "providers": [
    {
      "url": "http://localhost:1234",
      "api_key": "",
      "model": "ibm/granite-4-h-tiny"
    },
    {
      "url": "https://llm.chutes.ai/v1",
      "api_key": "your-api-key",
      "model": "zai-org/GLM-4.6"
    }
  ]
}
```

## Output Structure

### Directory Organization
```
tests/hypothesis_validation/
├── results/                 # All test results and metrics
│   ├── execution_metrics_*.json
│   ├── statistical_validation_*.json
│   └── final_validation_results_*.json
├── reports/                 # Generated reports
│   ├── comprehensive_validation_report_*.md
│   ├── performance_report_*.md
│   └── statistical_validation_report_*.md
├── plots/                   # Visualization plots
│   ├── effect_sizes_*.png
│   ├── comparison_means_*.png
│   └── power_analysis_*.png
└── test_cases/              # Generated test cases
    ├── complex_reasoning_test_cases.json
    ├── mathematical_reasoning_test_cases.json
    └── ...
```

### Result Files

#### Test Results (`results/`)
- **Individual test results**: JSON with complete execution data
- **Statistical analysis**: Paired comparisons and effect sizes
- **Performance metrics**: Resource usage and efficiency data
- **Final summary**: Consolidated results across all categories

#### Reports (`reports/`)
- **Comprehensive validation report**: Executive summary and detailed analysis
- **Statistical validation report**: Detailed statistical analysis
- **Performance monitoring report**: Resource utilization and efficiency
- **Category-specific reports**: Individual category analysis

#### Visualizations (`plots/`)
- **Effect size plots**: Cohen's d by metric and category
- **Comparison plots**: Approach means with confidence intervals
- **Significance plots**: P-values and statistical significance
- **Power analysis plots**: Power curves and sample size requirements

## Integration with Research Framework

### Existing Components
The test suite integrates with existing research infrastructure:

- **`research/hypothesis_testing_framework.py`**: Core hypothesis definitions
- **`research/statistical_analyzer.py`**: Statistical analysis tools
- **`research/enhanced_test_generator.py`**: Test case generation
- **`research/experiments/hypothesis_experiments.py`**: Experiment orchestration

### Data Flow
1. **Test Generation** → Enhanced test case generator
2. **A/B Testing** → Direct vs Conjecture comparison
3. **LLM Evaluation** → GLM-4.6 judge evaluation
4. **Statistical Analysis** → Paired t-tests and effect sizes
5. **Performance Monitoring** → Resource tracking and optimization
6. **Result Integration** → Research framework consolidation

### API Integration
```python
# Import research components
from research.hypothesis_testing_framework import ConjectureTestingFramework
from research.statistical_analyzer import ConjectureStatisticalAnalyzer

# Initialize with existing framework
framework = ConjectureTestingFramework()
await framework.initialize(providers)

# Run validation through research framework
results = await framework.run_hypothesis_testing_cycle(
    hypothesis_ids=core_hypotheses,
    max_iterations=3
)
```

## Validation Criteria

### Success Thresholds
- **Performance Claims**: ≥20% improvement over baseline, p<0.05
- **Efficiency Claims**: ≥15% cost reduction, ≥90% accuracy maintained
- **Parity Claims**: ≤10% performance gap between small+Conjecture vs large models
- **Quality Claims**: ≥0.90 score on relevant metrics
- **Reduction Claims**: ≥25% reduction in hallucinations

### Statistical Requirements
- **Minimum Sample Size**: 20 test cases per hypothesis per model
- **Preferred Sample Size**: 50+ test cases for high-stakes hypotheses
- **Power Analysis**: Target 80% statistical power, α=0.05
- **Effect Size**: Cohen's d > 0.5 for practical significance

### Quality Standards
- **Strong Evidence**: Effect size > 0.8, p < 0.01
- **Reproducibility**: Results stable across ≥3 runs
- **Generalizability**: Validated across multiple test case categories
- **Practical Significance**: Meaningful improvements in real-world scenarios

## Troubleshooting

### Common Issues

#### Provider Connection Failures
```bash
# Check provider configuration
python conjecture config

# Test individual providers
python conjecture backends

# Verify API keys and URLs
```

#### Insufficient Statistical Power
```bash
# Increase sample size
python tests/test_hypothesis_validation_suite.py --sample-size 100

# Check effect sizes
python tests/test_statistical_validation.py --analyze-power
```

#### Memory/Performance Issues
```bash
# Reduce concurrent tests
python tests/test_hypothesis_validation_suite.py --max-concurrent 2

# Monitor resources
python tests/test_performance_monitoring.py --monitor-only
```

#### Test Case Quality Issues
```bash
# Regenerate test cases
python tests/test_hypothesis_validation_suite.py --regenerate-cases

# Validate test case difficulty
python tests/test_hypothesis_validation.py --validate-cases
```

## Performance Optimization

### Resource Management
- **Concurrent Execution**: Configurable parallel test execution
- **Memory Monitoring**: Real-time memory usage tracking
- **Token Optimization**: Efficient prompt engineering and response parsing
- **Cost Control**: Token usage monitoring and estimation

### Execution Strategies
- **Batch Processing**: Group similar test cases for efficiency
- **Caching**: Persist results and intermediate computations
- **Progressive Loading**: Load test cases in chunks for large datasets
- **Early Stopping**: Stop testing when significance is achieved

### Monitoring and Alerting
- **Performance Thresholds**: Configurable alerting for slow/failed tests
- **Resource Limits**: Automatic throttling when resources are constrained
- **Error Tracking**: Comprehensive error logging and analysis
- **Progress Reporting**: Real-time progress updates and ETA calculations

## Research Integration

### Hypothesis Validation Pipeline
1. **Baseline Establishment**: Measure current performance levels
2. **Initial Testing**: Run all hypothesis experiments
3. **Analysis & Refinement**: Identify failures and improvements
4. **Iteration Testing**: Re-test refined approaches
5. **Consolidation**: Final validation and proof generation

### Data Management
- **Centralized Storage**: All results in standardized format
- **Version Control**: Track changes to test cases and methodology
- **Metadata Enrichment**: Comprehensive context for all results
- **Cross-Validation**: Compare with existing research findings

### Continuous Validation
- **Automated Pipelines**: Regular execution of validation suite
- **Trend Analysis**: Track performance over time
- **Adaptive Testing**: Adjust test cases based on results
- **Integration Testing**: Validate with new model versions

## Contributing

### Adding New Test Categories
1. Define category in `SuiteConfiguration`
2. Implement test case generation in `_generate_category_test_cases()`
3. Add category-specific evaluation guidance
4. Update statistical analysis parameters
5. Add visualization and reporting

### Extending Evaluation Criteria
1. Add criterion to evaluation rubrics
2. Define weight and threshold parameters
3. Implement scoring logic in LLM judge
4. Update statistical analysis methods
5. Validate with calibration samples

### Integration with New Models
1. Add model configuration to provider setup
2. Update cost estimation parameters
3. Test model compatibility with framework
4. Update performance benchmarks
5. Document model-specific considerations

## License and Support

This test suite is part of the Conjecture project and follows the same licensing terms. For support, questions, or contributions:

- **Documentation**: See `docs/` directory for detailed guides
- **Issues**: Report via GitHub issues with detailed reproduction steps
- **Discussions**: Use GitHub discussions for questions and ideas
- **Research Papers**: See `research/papers/` for theoretical background

---

**Status**: ✅ **PRODUCTION READY** - Comprehensive test suite for statistical hypothesis validation