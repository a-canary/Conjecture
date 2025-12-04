
# Scripts Archive

This directory contains legacy test and utility scripts that were moved from the project root to improve organization.

## Overview

These scripts were primarily used for development, debugging, and testing during various phases of the Conjecture project. They are preserved here for historical reference and potential future use, but are not part of the active development workflow.

## Script Categories

### Configuration and Setup
- `setup_wizard.py` - Initial configuration wizard for the project
- `compare_env_loading.py` - Environment loading comparison utility

### Chutes API Testing
- `debug_chutes_endpoint.py` - Debugging tool for Chutes API endpoints
- `query_chutes_models.py` - Utility to query available models on Chutes API
- `test_chutes_api_connectivity.py` - Connectivity test for Chutes API

### Research Testing
- `debug_research_call.py` - Debugging tool for research function calls
- `minimal_research_runner.py` - Minimal implementation of research runner
- `simple_research_test.py` - Simple test cases for research functionality
- `test_exact_research_call.py` - Test for exact research call implementations
- `test_experiment.py` - Experiment testing utilities
- `test_glm_integration.py` - GLM model integration testing

### Import and Testing Utilities
- `test_imports.py` - Import validation and testing
- `test_iteration_2.py` - Testing utilities for iteration 2 development
- `test_with_same_imports.py` - Import testing with consistent modules
- `test_wizard.py` - Testing utilities for configuration wizard

### Miscellaneous
- `test_diag_simple.py` - Simple diagnostic testing
- `usage_examples.py` - Usage examples and demonstrations
- `validation_report.py` - Validation reporting utilities

## Modern Equivalents

Many of these scripts have been superseded by more comprehensive tools in the main scripts directory:

- For provider testing: `../test_conjecture_provider.py`
- For model comparisons: `../run_4model_comparison.py`
- For startup tasks: `../start_conjecture_provider.py`

## Usage

These scripts are provided as-is for reference. To use them:

1. Navigate to this directory
2. Run with Python 3.8+: `python script_name.py`

Note that some scripts may have dependencies that are no longer actively maintained in the project.

## Project Structure Impact

Moving these scripts from the project root has:
- Reduced clutter in the main project directory
- Improved project organization
- Separated legacy tools from active development scripts
- Made the project structure more navigable
