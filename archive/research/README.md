# Conjecture Research Suite

Comprehensive research framework for validating Conjecture's core hypotheses through controlled experiments.

## 🎯 Research Objectives

This research suite validates the central hypothesis of Conjecture:

**"By decomposing tasks and concepts, and compressing the context using claims-based representations, small LLMs can achieve performance comparable to larger models on complex reasoning tasks."**

## 📁 Directory Structure

```
research/
├── experiments/           # Core experiment implementations
│   ├── experiment_framework.py    # Main experiment framework
│   ├── hypothesis_experiments.py  # Hypothesis validation experiments
│   ├── model_comparison.py        # Model comparison studies
│   └── llm_judge.py              # LLM-as-a-Judge evaluation system
├── test_cases/            # Test case definitions and generators
│   ├── *.json                   # Individual test cases
│   └── test_case_generator.py    # Automated test case generation
├── results/               # Experiment results and outputs
├── analysis/              # Analysis and reporting tools
│   ├── experiment_analyzer.py    # Statistical analysis
│   └── *.png                    # Generated visualizations
├── run_research.py        # Main research orchestrator
└── README.md              # This file
```

## 🧪 Core Experiments

### 1. Task Decomposition Experiment
**Hypothesis**: Small LLMs will show 20%+ improvement in correctness when using task decomposition vs direct approach.

**Test Cases**: Complex reasoning problems that benefit from step-by-step breakdown
**Models**: All specified models (granite-4-h-tiny, GLM-Z1-9B, GLM-4.5-Air, GLM-4.6)
**Metrics**: Correctness, Completeness, Coherence

### 2. Context Compression Experiment  
**Hypothesis**: Models will maintain 90%+ performance with 50%+ context reduction using claims format.

**Test Cases**: Long document QA and synthesis tasks
**Models**: All specified models
**Metrics**: Correctness, Efficiency, Completeness

### 3. Model Comparison Experiment
**Hypothesis**: Small models (3-9B) with Conjecture prompting will match/exceed larger models (30B+) on reasoning tasks.

**Test Cases**: Logical, mathematical, and ethical reasoning problems
**Models**: All specified models
**Metrics**: Correctness, Coherence, Confidence Calibration

### 4. Claims-Based Reasoning Experiment
**Hypothesis**: Claims-based reasoning will show 15%+ improvement in correctness and confidence calibration.

**Test Cases**: Evidence evaluation and argument analysis
**Models**: All specified models  
**Metrics**: Correctness, Confidence Calibration, Coherence

### 5. End-to-End Pipeline Experiment
**Hypothesis**: Full pipeline will show 25%+ improvement over baseline for small models on complex tasks.

**Test Cases**: Multi-step research and analysis tasks
**Models**: All specified models
**Metrics**: All core metrics

## 🤖 LLM-as-a-Judge System

Uses GLM-4.6 as a consistent judge to evaluate model responses across multiple criteria:

- **Correctness**: Factual accuracy
- **Completeness**: Coverage of all aspects  
- **Coherence**: Logical flow and consistency
- **Reasoning Quality**: Strength of logical arguments
- **Depth**: Insight and analysis quality
- **Clarity**: Expression and understandability
- **Confidence Calibration**: Alignment of confidence with accuracy
- **Efficiency**: Conciseness and effectiveness

## 📊 Test Case Categories

### Complex Reasoning
- Multi-step logic puzzles
- Causal inference problems
- Analytical reasoning tasks

### Mathematical Reasoning  
- Algebra word problems
- Geometric calculations
- Rate and proportion problems

### Context Compression
- Long document QA
- Multi-source synthesis
- Research paper analysis

### Evidence Evaluation
- Conflicting evidence assessment
- Scientific claim evaluation
- Risk-benefit analysis

### Task Decomposition
- Project planning scenarios
- Multi-step problem solving
- Strategic decision making

## 🚀 Getting Started

### 1. Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Additional research dependencies
pip install matplotlib seaborn pandas scipy
```

### 2. Configuration
Create or edit `research/config.json`:

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
      "model": "GLM-Z1-9B-0414"
    },
    {
      "url": "https://llm.chutes.ai/v1",
      "api_key": "your-api-key",
      "model": "GLM-4.5-Air"
    },
    {
      "url": "https://llm.chutes.ai/v1",
      "api_key": "your-api-key", 
      "model": "GLM-4.6"
    }
  ],
  "judge_model": "chutes:GLM-4.6",
  "experiments": {
    "hypothesis_validation": true,
    "model_comparison": true,
    "generate_test_cases": true
  },
  "output": {
    "save_results": true,
    "generate_visualizations": true,
    "create_reports": true
  }
}
```

### 3. Run Experiments

#### Full Research Suite
```bash
python research/run_research.py --full
```

#### Individual Components
```bash
# Generate test cases only
python research/run_research.py --generate-tests

# Run hypothesis validation only
python research/run_research.py --hypothesis

# Run model comparison only
python research/run_research.py --comparison

# Analyze existing results only
python research/run_research.py --analyze

# Run custom experiment
python research/run_research.py --custom path/to/experiment.json
```

### 4. Custom Experiments

Create a custom experiment configuration:

```json
{
  "experiment_id": "custom_001",
  "experiment_type": "model_comparison",
  "name": "My Custom Experiment",
  "description": "Testing specific hypothesis",
  "hypothesis": "Custom hypothesis to test",
  "models_to_test": ["lmstudio:ibm/granite-4-h-tiny", "chutes:GLM-4.6"],
  "test_cases": ["complex_reasoning_001", "mathematical_reasoning_001"],
  "metrics": ["correctness", "coherence"],
  "parameters": {
    "max_tokens": 2000,
    "temperature": 0.3
  }
}
```

## 📈 Analysis and Reporting

### Automated Analysis
- Statistical significance testing
- Effect size calculations  
- Performance visualizations
- Cross-experiment comparisons

### Generated Reports
- **Individual Experiment Reports**: Detailed analysis per experiment
- **Hypothesis Validation Report**: Assessment of core hypotheses
- **Model Comparison Report**: Comprehensive model performance analysis
- **Comprehensive Analysis Report**: Overall research findings

### Visualizations
- Model performance comparison charts
- Metric breakdown graphs
- Success rate visualizations
- Statistical significance plots

## 🎯 Success Criteria

### Hypothesis Validation
- **Task Decomposition**: ≥20% improvement in correctness
- **Context Compression**: ≥90% performance with ≥50% context reduction
- **Model Comparison**: Small models match/exceed larger models
- **Claims Reasoning**: ≥15% improvement in correctness and calibration
- **End-to-End Pipeline**: ≥25% improvement over baseline

### Statistical Significance
- p-value < 0.05 for key comparisons
- Effect size > 0.5 for meaningful differences
- Consistent results across multiple test cases

### Practical Impact
- Demonstrable cost/performance improvements
- Clear guidance for model selection
- Actionable insights for Conjecture development

## 🔬 Research Methodology

### Experimental Design
- **Controlled Variables**: Consistent prompts, temperature settings, evaluation criteria
- **Randomization**: Randomized test case order to reduce bias
- **Multiple Judges**: Consensus evaluation using GLM-4.6
- **Statistical Validation**: Proper significance testing and effect size calculation

### Evaluation Process
1. **Test Case Generation**: Automated generation of diverse test cases
2. **Model Execution**: Run each model on each test case
3. **Judge Evaluation**: LLM-as-a-Judge evaluates responses
4. **Statistical Analysis**: Calculate significance and effect sizes
5. **Report Generation**: Comprehensive analysis and visualization

### Quality Assurance
- **Test Case Validation**: Ensure ground truth accuracy
- **Judge Consistency**: Multiple evaluation rounds for reliability
- **Result Verification**: Cross-check statistical calculations
- **Reproducibility**: Full logging and configuration tracking

## 📋 Expected Outcomes

### Primary Outcomes
1. **Validation Evidence**: Statistical support for Conjecture's core hypotheses
2. **Performance Benchmarks**: Clear model performance comparisons
3. **Optimization Guidance**: Data-driven recommendations for Conjecture development
4. **Research Infrastructure**: Reusable experiment framework

### Secondary Outcomes
1. **Test Case Library**: Comprehensive collection of evaluation test cases
2. **Evaluation Standards**: Established methodology for LLM evaluation
3. **Publication Materials**: Research papers and technical reports
4. **Tool Development**: Enhanced analysis and visualization tools

## 🛠️ Development Notes

### Extending the Framework
- Add new experiment types in `experiments/`
- Create new test case categories in `test_cases/`
- Extend evaluation criteria in `llm_judge.py`
- Add new analysis methods in `experiment_analyzer.py`

### Troubleshooting
- Ensure all providers are accessible before running experiments
- Check API keys and model availability
- Monitor token usage and costs
- Verify test case format and ground truth accuracy

### Performance Optimization
- Parallel experiment execution where possible
- Efficient result caching and storage
- Optimized prompt engineering
- Resource usage monitoring

---

This research suite provides the rigorous experimental validation needed to demonstrate Conjecture's effectiveness and guide its future development.