# Model Comparison Experiment

## Overview

The Model Comparison Experiment tests the core hypothesis that **"Small models (3-9B) with Conjecture match/exceed larger models (30B+) without Conjecture"**.

This is a critical experiment for validating Conjecture's value proposition - that structured, claims-based reasoning can enable smaller, more efficient models to achieve performance comparable to much larger models.

## Experiment Design

### Models Compared

1. **Model A (Small+Conjecture)**: IBM Granite Tiny (3B parameters) with Conjecture methods
2. **Model B (Large without Conjecture)**: GLM-4.6 (30B+ parameters) with direct prompting  
3. **Model C (Large+Conjecture)**: GLM-4.6 (30B+ parameters) with Conjecture (optional comparison)

### Evaluation Framework

- **Judge Model**: GLM-4.6 for consistent, high-quality evaluation
- **Evaluation Criteria**: Correctness, Completeness, Coherence, Reasoning Quality, Confidence Calibration, Efficiency, Hallucination Reduction
- **Statistical Analysis**: Paired t-tests, effect sizes, power analysis
- **Sample Size**: 50-100 test cases across reasoning categories

## Usage

### Prerequisites

1. **Conjecture Environment**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Model Access**: Configure access to required models:
   - **Local Small Model**: Ollama or LM Studio running IBM Granite Tiny
   - **Large Model**: API access to GLM-4.6 (Chutes.ai or compatible)
   - **Judge Model**: Same GLM-4.6 access for evaluation

3. **Configuration**: Set up provider configurations in your environment or config files

### Running the Experiment

#### Basic Execution
```bash
cd experiments
python run_model_comparison_experiment.py
```

#### Custom Configuration
```python
from experiments.run_model_comparison_experiment import ModelComparisonExperiment, ExperimentConfig

# Create custom configuration
config = ExperimentConfig(
    sample_size=75,           # Number of test cases
    alpha_level=0.05,         # Statistical significance level
    power_target=0.8,         # Statistical power target
    small_model="ibm/granite-4-h-tiny",
    large_model="zai-org/GLM-4.6",
    judge_model="zai-org/GLM-4.6"
)

# Run experiment
experiment = ModelComparisonExperiment(config)
await experiment.run_experiment()
```

### Environment Variables

Set these environment variables for cloud provider access:

```bash
# For Chutes.ai (GLM-4.6)
export CHUTES_API_KEY="your-api-key-here"

# For other providers as needed
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

## Test Cases

The experiment loads test cases from `research/test_cases/` directory:

### Supported Categories

1. **Complex Reasoning**: Multi-step logical puzzles and deduction
2. **Mathematical Reasoning**: Algebra, geometry, and word problems
3. **Context Compression**: Long document QA and synthesis
4. **Evidence Evaluation**: Conflicting evidence assessment
5. **Task Decomposition**: Complex planning and organization
6. **Claims Reasoning**: Evidence-based claim analysis

### Test Case Format

Each test case follows this JSON structure:

```json
{
  "id": "unique_test_id",
  "category": "complex_reasoning",
  "difficulty": "hard",
  "description": "Test description",
  "question": "The actual question/task",
  "ground_truth": "Expected answer or evaluation criteria",
  "expected_approach": "break_down_problem",
  "metadata": {
    "type": "logic_puzzle",
    "estimated_time_minutes": 15,
    "claims_based_approach_beneficial": true
  }
}
```

## Outputs and Results

### Generated Files

1. **Experiment Results**: `experiments/results/{experiment_id}.json`
   - Raw execution data for all test cases
   - Model responses and timing information
   - Evaluation scores and qualitative feedback

2. **Statistical Analysis**: `experiments/results/{experiment_id}_statistical_analysis.json`
   - Statistical test results (t-tests, ANOVA)
   - Effect sizes and confidence intervals
   - Power analysis and assumptions checking

3. **Comprehensive Report**: `experiments/reports/{experiment_id}_report.md`
   - Executive summary and key findings
   - Hypothesis validation results
   - Performance comparisons by approach
   - Detailed statistical analysis
   - Recommendations and next steps

4. **Execution Log**: `experiments/results/{experiment_id}.log`
   - Detailed execution timeline
   - Error messages and debugging information
   - Progress tracking

### Key Metrics

#### Primary Metrics
- **Overall Score**: Weighted combination of all evaluation criteria
- **Correctness**: Factual accuracy of responses
- **Reasoning Quality**: Strength and validity of logical arguments

#### Secondary Metrics
- **Response Time**: Execution time per test case
- **Response Length**: Token/character count of responses
- **Claim Format Success**: Percentage of responses using proper claims format
- **Reasoning Steps**: Number of explicit reasoning steps identified

#### Statistical Measures
- **Effect Size**: Cohen's d for practical significance
- **P-value**: Statistical significance of differences
- **Confidence Intervals**: Range estimates for true effects
- **Power Analysis**: Adequacy of sample size

## Hypothesis Validation

### Success Criteria

The hypothesis is **VALIDATED** if:
1. **Statistical Significance**: p < 0.05 for primary comparison
2. **Practical Significance**: Effect size ≥ 0.5 (medium effect)
3. **Performance Threshold**: Small+Conjecture ≥ Large without Conjecture
4. **Adequate Power**: Statistical power ≥ 0.8

### Interpretation Guide

| Result | Interpretation | Action |
|--------|---------------|--------|
| **Validated** | Small+Conjecture significantly outperforms Large without Conjecture | Proceed to production scaling |
| **Not Validated** | No significant difference found | Investigate effect size, increase sample size |
| **Rejected** | Small+Conjecture significantly underperforms | Revisit Conjecture methodology |

## Troubleshooting

### Common Issues

#### 1. Model Connection Failures
```
Error: Failed to connect to model
```
**Solution**: 
- Verify model is running (local) or API key is valid (cloud)
- Check network connectivity
- Confirm model names match exactly

#### 2. Test Case Loading Issues
```
Warning: No test case files found
```
**Solution**:
- Ensure test cases exist in `research/test_cases/`
- Run test case generator: `python research/test_cases/test_case_generator.py`
- Check file permissions and paths

#### 3. Evaluation Failures
```
Error: Evaluation failed
```
**Solution**:
- Verify judge model access
- Check API rate limits
- Ensure sufficient tokens for evaluation

#### 4. Statistical Analysis Errors
```
Error: Insufficient data for test
```
**Solution**:
- Increase sample size (minimum 20 per condition)
- Check for missing evaluation results
- Verify data format consistency

### Debug Mode

Enable detailed logging by setting environment variable:
```bash
export CONJECTURE_DEBUG=1
python experiments/run_model_comparison_experiment.py
```

## Advanced Configuration

### Custom Evaluation Criteria

Modify evaluation criteria and weights:

```python
config = ExperimentConfig()
config.evaluation_criteria = ["accuracy", "efficiency", "clarity"]
config.criterion_weights = {
    "accuracy": 2.0,
    "efficiency": 1.0,
    "clarity": 1.5
}
```

### Custom Models

Use different models in the comparison:

```python
config = ExperimentConfig(
    small_model="microsoft/DialoGPT-small",  # Different small model
    large_model="anthropic/claude-3-sonnet",  # Different large model
    judge_model="openai/gpt-4"               # Different judge
)
```

### Batch Processing

Run multiple experiments with different configurations:

```python
import asyncio

async def run_batch():
    configs = [
        ExperimentConfig(sample_size=50),
        ExperimentConfig(sample_size=75),
        ExperimentConfig(sample_size=100)
    ]
    
    for i, config in enumerate(configs):
        print(f"Running batch {i+1}/3")
        experiment = ModelComparisonExperiment(config)
        await experiment.run_experiment()

asyncio.run(run_batch())
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Model Comparison Experiment
on: [push, pull_request]

jobs:
  experiment:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run experiment
      env:
        CHUTES_API_KEY: ${{ secrets.CHUTES_API_KEY }}
      run: |
        cd experiments
        python run_model_comparison_experiment.py
    
    - name: Upload results
      uses: actions/upload-artifact@v2
      with:
        name: experiment-results
        path: experiments/results/
```

## Contributing

### Adding New Test Cases

1. Create JSON files in `research/test_cases/`
2. Follow the standard test case format
3. Include appropriate metadata and difficulty levels
4. Test with sample execution before adding

### Extending Evaluation Criteria

1. Modify `JudgeConfiguration` in `tests/test_llm_judge.py`
2. Update evaluation prompts
3. Adjust criterion weights
4. Re-calibrate judge system

### New Statistical Tests

1. Extend `ConjectureStatisticalAnalyzer` class
2. Add new test methods
3. Update report generation
4. Document interpretation guidelines

## Research Impact

This experiment provides critical evidence for:

1. **Model Efficiency**: Can smaller models replace larger ones?
2. **Methodology Value**: Does structured reasoning improve performance?
3. **Cost Optimization**: Can we reduce computational costs?
4. **Scalability**: How well does Conjecture scale across model sizes?

### Publication Support

The experiment generates publication-ready data:
- Statistical significance testing
- Effect size calculations
- Comprehensive methodology documentation
- Reproducible experimental setup

## Next Steps

After running the experiment:

1. **Review Results**: Analyze the comprehensive report
2. **Validate Findings**: Cross-check with manual evaluation
3. **Plan Follow-up**: Based on outcomes, design next experiments
4. **Share Insights**: Contribute to research community
5. **Update Documentation**: Incorporate findings into Conjecture guides

---

**For questions or support, refer to the main Conjecture documentation or create an issue in the project repository.**