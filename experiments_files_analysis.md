# Conjecture Project Experiments Directory Analysis

## Overview
This document provides a comprehensive analysis of all files in the experiments/ directory of the Conjecture project, rated according to the standards developed in rating_standards.md. The experiments directory contains critical research scripts and results that validate the core hypotheses of the Conjecture AI-Powered Evidence-Based Reasoning System.

## Rating Standards Applied
- **Value Rating (0-10)**: Overall importance and contribution to the project
- **Functional Description**: Brief 2-3 word description of the file's purpose
- **Contribution**: One-sentence explanation of the file's value to the project
- **Dependencies**: Key relationships or dependencies with other files

---

## experiments/ Directory Structure

### Root Level Files

#### experiments/run_end_to_end_experiment.py
- **Value Rating**: 10/10
- **Functional Description**: End-to-end pipeline validation
- **Contribution**: Critical experiment testing if full Conjecture pipeline shows 25%+ improvement over baseline, demonstrating the core value proposition of the entire system.
- **Dependencies**: Depends on core Conjecture modules, GLM-4.6 judge model, IBM Granite Tiny model

#### experiments/run_claims_based_reasoning_experiment.py
- **Value Rating**: 9/10
- **Functional Description**: Claims-based reasoning validation
- **Contribution**: Tests if claims-based reasoning shows 15%+ improvement in correctness and confidence calibration, validating a fundamental approach of the Conjecture system.
- **Dependencies**: Depends on claim processing modules, evaluation framework, statistical analysis tools

#### experiments/run_context_compression_experiment.py
- **Value Rating**: 8/10
- **Functional Description**: Context compression testing
- **Contribution**: Tests if models maintain 90%+ performance with 50%+ context reduction using claims format, validating a key optimization for handling large contexts.
- **Dependencies**: Depends on context processing, claims compression algorithms, evaluation metrics

#### experiments/run_model_comparison_experiment.py
- **Value Rating**: 9/10
- **Functional Description**: Model performance comparison
- **Contribution**: Tests hypothesis that small models (3-9B) with Conjecture match/exceed larger models (30B+), crucial for demonstrating efficiency gains.
- **Dependencies**: Depends on multiple model providers, evaluation framework, statistical analysis

#### experiments/run_task_decomposition_experiment.py
- **Value Rating**: 8/10
- **Functional Description**: Task decomposition validation
- **Contribution**: Tests if Conjecture methods provide 20%+ improvement with task decomposition vs direct approach, validating a core methodology.
- **Dependencies**: Depends on task processing modules, decomposition algorithms, evaluation framework

#### experiments/README_model_comparison.md
- **Value Rating**: 7/10
- **Functional Description**: Model comparison documentation
- **Contribution**: Comprehensive documentation for model comparison experiments, providing setup instructions and troubleshooting guidance for researchers.
- **Dependencies**: References run_model_comparison_experiment.py, configuration files

#### experiments/run_claims_experiment_standalone.py
- **Value Rating**: 7/10
- **Functional Description**: Standalone claims testing
- **Contribution**: Simplified version of claims-based reasoning experiment that avoids complex import issues, enabling easier testing and validation.
- **Dependencies**: Minimal dependencies, self-contained implementation

#### experiments/run_coding_capabilities_experiment.py
- **Value Rating**: 8/10
- **Functional Description**: Coding capabilities evaluation
- **Contribution**: Tests hypothesis that small models with Conjecture can achieve near SOTA performance on coding tasks, expanding applicability to technical domains.
- **Dependencies**: Depends on coding evaluation framework, specialized metrics, model providers

#### experiments/run_end_to_end_standalone.py
- **Value Rating**: 6/10
- **Functional Description**: Simplified end-to-end testing
- **Contribution**: Simplified version of end-to-end experiment for testing without complex dependencies, providing framework for validation.
- **Dependencies**: Minimal dependencies, self-contained implementation

#### experiments/run_local_experiment.py
- **Value Rating**: 6/10
- **Functional Description**: Local model testing
- **Contribution**: Demonstrates framework with local LM Studio model, enabling offline testing and development.
- **Dependencies**: Depends on local model setup, basic evaluation framework

#### experiments/simple_context_compression_experiment.py
- **Value Rating**: 7/10
- **Functional Description**: Simplified context compression
- **Contribution**: Streamlined version of context compression experiment that works directly with existing codebase without complex dependencies.
- **Dependencies**: Minimal dependencies, uses local and cloud models

#### experiments/simple_task_decomposition_experiment.py
- **Value Rating**: 7/10
- **Functional Description**: Simplified task decomposition
- **Contribution**: Streamlined task decomposition experiment avoiding complex import issues while maintaining core functionality.
- **Dependencies**: Minimal dependencies, self-contained implementation

---

### experiments/results/ Directory

#### experiments/results/end_to_end_results_end_to_end_20251204_182000.json
- **Value Rating**: 9/10
- **Functional Description**: End-to-end experiment results
- **Contribution**: Contains successful results showing 52.47% improvement (exceeding 25% target), validating the core effectiveness of the Conjecture pipeline.
- **Dependencies**: Generated by run_end_to_end_experiment.py

#### experiments/results/end_to_end_results_end_to_end_20251204_182050.json
- **Value Rating**: 9/10
- **Functional Description**: End-to-end experiment validation
- **Contribution**: Confirms reproducibility of end-to-end results with 25 test cases, providing statistical validation of pipeline effectiveness.
- **Dependencies**: Generated by run_end_to_end_experiment.py

#### experiments/results/experiment_2_simple_results_20251205_160556.json
- **Value Rating**: 8/10
- **Functional Description**: Enhanced prompt engineering results
- **Contribution**: Shows 66.67% improvement in claims generation and 57.14% improvement in confidence calibration, validating prompt optimization strategies.
- **Dependencies**: Generated by enhanced prompt engineering experiment

#### experiments/results/experiment_3_real_results_20251205_172633.json
- **Value Rating**: 7/10
- **Functional Description**: Database priming results
- **Contribution**: Mixed results showing some improvements but failing to meet overall success criteria, providing insights for further optimization.
- **Dependencies**: Generated by database priming experiment

#### experiments/results/experiment_3_standalone_results_20251205_162740.json
- **Value Rating**: 8/10
- **Functional Description**: Standalone database priming results
- **Contribution**: Successful validation of database priming with 23.78% reasoning quality improvement and 32.5% evidence utilization increase.
- **Dependencies**: Generated by standalone database priming experiment

---

### experiments/experiments/results/ Directory

#### experiments/experiments/results/local_experiment_results_20251204_221953.json
- **Value Rating**: 6/10
- **Functional Description**: Local experiment results
- **Contribution**: Demonstrates local model testing capabilities, enabling offline development and validation of the framework.
- **Dependencies**: Generated by run_local_experiment.py

#### experiments/experiments/results/local_experiment_results_20251204_222238.json
- **Value Rating**: 6/10
- **Functional Description**: Local experiment validation
- **Contribution**: Provides reproducibility validation for local model testing, confirming framework reliability.
- **Dependencies**: Generated by run_local_experiment.py

#### experiments/experiments/results/simple_task_decomposition_experiment_71c542bd_20251204_221548.json
- **Value Rating**: 7/10
- **Functional Description**: Task decomposition results
- **Contribution**: Contains results from simplified task decomposition experiment, validating the approach with reduced complexity.
- **Dependencies**: Generated by simple_task_decomposition_experiment.py

---

### experiments/experiments/test_cases/ Directory

#### experiments/experiments/test_cases/simple_context_compression_cases_15.json
- **Value Rating**: 6/10
- **Functional Description**: Context compression test cases
- **Contribution**: Provides 15 test cases using Renaissance text for context compression experiments, ensuring consistent evaluation across runs.
- **Dependencies**: Used by context compression experiments

#### experiments/experiments/test_cases/simple_task_decomposition_cases_10.json
- **Value Rating**: 6/10
- **Functional Description**: Task decomposition test cases
- **Contribution**: Contains 10 complex scenarios for task decomposition testing, covering project planning and problem-solving domains.
- **Dependencies**: Used by task decomposition experiments

---

### experiments/test_cases/ Directory

#### experiments/test_cases/claims_based_reasoning_cases_75.json
- **Value Rating**: 7/10
- **Functional Description**: Claims reasoning test cases
- **Contribution**: Provides 75 comprehensive test cases for claims-based reasoning experiments, covering evidence evaluation and argument analysis scenarios.
- **Dependencies**: Used by claims-based reasoning experiments

---

### experiments/reports/ Directory

#### experiments/reports/claims_based_reasoning_report_9cfb87b9_2025-12-04 23-08-17.md
- **Value Rating**: 7/10
- **Functional Description**: Claims reasoning experiment report
- **Contribution**: Documents experiment results showing hypothesis not validated (-4.0% correctness improvement), providing critical feedback for approach refinement.
- **Dependencies**: Generated by claims-based reasoning experiment

---

## Summary by Category

### Core Experiment Scripts (Value 8-10)
1. **run_end_to_end_experiment.py** (10/10) - Most critical validation
2. **run_claims_based_reasoning_experiment.py** (9/10) - Fundamental approach validation
3. **run_model_comparison_experiment.py** (9/10) - Efficiency demonstration
4. **run_context_compression_experiment.py** (8/10) - Optimization validation
5. **run_task_decomposition_experiment.py** (8/10) - Methodology validation
6. **run_coding_capabilities_experiment.py** (8/10) - Domain expansion validation

### Supporting Scripts (Value 6-7)
1. **simple_context_compression_experiment.py** (7/10) - Streamlined testing
2. **simple_task_decomposition_experiment.py** (7/10) - Simplified validation
3. **run_claims_experiment_standalone.py** (7/10) - Import-free testing
4. **run_end_to_end_standalone.py** (6/10) - Basic framework
5. **run_local_experiment.py** (6/10) - Local testing

### Results Files (Value 6-9)
1. **end_to_end_results_*.json** (9/10) - Critical success validation
2. **experiment_3_standalone_results_*.json** (8/10) - Successful priming validation
3. **experiment_2_simple_results_*.json** (8/10) - Prompt optimization validation
4. **experiment_3_real_results_*.json** (7/10) - Mixed results for learning
5. **Local experiment results** (6/10) - Development support

### Test Cases and Documentation (Value 6-7)
1. **claims_based_reasoning_cases_75.json** (7/10) - Comprehensive test coverage
2. **README_model_comparison.md** (7/10) - Research guidance
3. **Simple test cases** (6/10) - Basic validation support

## Key Insights

### Successful Validations
- **End-to-end pipeline**: 52.47% improvement achieved (target: 25%)
- **Enhanced prompt engineering**: 66.67% claims generation improvement
- **Database priming**: 23.78% reasoning quality improvement
- **Evidence utilization**: 32.5% increase achieved

### Areas Needing Improvement
- **Claims-based reasoning**: -4.0% correctness improvement (target: 15%)
- **Context compression**: Need more comprehensive testing
- **Task decomposition**: Requires further optimization

### Research Value Distribution
- **High Value (8-10)**: 6 files - Core research experiments
- **Medium Value (6-7)**: 9 files - Supporting experiments and results
- **Documentation**: Essential for reproducibility and knowledge transfer

## Recommendations

### Priority for Preservation
1. **All core experiment scripts** (Value 8-10) - Critical research assets
2. **Successful results files** - Validation of core hypotheses
3. **Comprehensive test cases** - Essential for reproducibility
4. **Documentation files** - Knowledge transfer and onboarding

### Consolidation Opportunities
1. **Merge standalone versions** with main experiments where appropriate
2. **Standardize results format** across all experiments
3. **Create unified test case format** for consistency
4. **Document experiment dependencies** more explicitly

### Future Development
1. **Expand successful approaches** (end-to-end, prompt engineering)
2. **Refine struggling approaches** (claims-based reasoning)
3. **Add new domain experiments** (coding, security, etc.)
4. **Improve statistical validation** across all experiments

---

## Conclusion

The experiments directory contains high-value research assets that validate the core Conjecture hypotheses. The end-to-end pipeline validation is particularly successful, showing 52.47% improvement against a 25% target. The experiments demonstrate both the strengths of the Conjecture approach and areas needing further refinement. This directory represents a comprehensive research foundation that should be preserved and extended for continued development of the Conjecture AI-Powered Evidence-Based Reasoning System.