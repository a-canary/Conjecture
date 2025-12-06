# Coverage Measurement Infrastructure for Conjecture

This document describes the comprehensive coverage measurement infrastructure setup for the Conjecture project, designed to help track progress toward the 80% coverage goal.

## Overview

The coverage infrastructure includes:

- **Configuration**: `.coveragerc` and enhanced `tests/pytest.ini`
- **Scripts**: Automated coverage reporting and analysis tools
- **Baseline Tracking**: Progress monitoring over time
- **Reports**: Multiple output formats for different needs

## Quick Start

### 1. Install Dependencies

```bash
# Install coverage dependencies
pip install pytest-cov coverage

# Or install with dev dependencies
pip install -e ".[dev]"
```

### 2. Run Coverage Analysis

```bash
# Unix/Linux/macOS
./scripts/run_coverage.sh

# Windows
scripts\run_coverage.bat

# Or directly with pytest
python -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html
```

### 3. View Results

- **Terminal**: Summary shows immediately
- **HTML Report**: Open `htmlcov/index.html` in your browser
- **XML/JSON**: Machine-readable reports for CI/CD

## Configuration Files

### `.coveragerc`

The main coverage configuration file with:

- **Source Focus**: Only measures `src/` directory
- **Branch Coverage**: Enabled for comprehensive measurement
- **Exclusions**: Test files, `__init__.py`, migrations, examples, etc.
- **Reports**: HTML, XML, and JSON output configured

Key exclusions:
```
*/tests/*          # Test files
*/__init__.py       # Import-only files
*/migrations/*      # Database migrations
examples/*          # Example code
generated_code/*   # Generated source files
```

### `tests/pytest.ini`

Enhanced pytest configuration with:

- **Coverage Integration**: Built-in coverage reporting
- **Multiple Formats**: Terminal, HTML, XML, JSON
- **Markers**: Additional test categorization for coverage analysis
- **Thresholds**: Progressive coverage goals (commented out initially)

## Scripts and Tools

### `scripts/run_coverage.sh` / `scripts/run_coverage.bat`

Main coverage execution script that:

1. **Cleans** previous coverage data
2. **Runs** comprehensive coverage analysis
3. **Generates** multiple report formats
4. **Extracts** key metrics and progress indicators
5. **Timestamps** reports for historical tracking

**Usage:**
```bash
./scripts/run_coverage.sh    # Unix/macOS
scripts\run_coverage.bat      # Windows
```

**Output:**
- Terminal summary with progress indicators
- HTML report in `htmlcov/` directory
- XML/JSON reports in project root
- Timestamped copies in `coverage_reports/`

### `scripts/compare_coverage.py`

Compares coverage between different runs to identify progress or regression.

**Features:**
- Compare current vs. previous coverage
- Compare any two coverage files
- Compare latest two saved reports
- Visual indicators for improvement/regression
- Progress assessment toward goals

**Usage:**
```bash
# Compare current coverage with latest saved
python scripts/compare_coverage.py

# Compare specific files
python scripts/compare_coverage.py --old old_coverage.json --new new_coverage.json

# Compare latest two saved reports
python scripts/compare_coverage.py --latest

# Custom goal percentage
python scripts/compare_coverage.py --goal 85.0
```

**Output Examples:**
```
📊 Previous Coverage
====================
  Line Coverage: 45.2% (234/518)
  Branch Coverage: 32.1% (67/209)
  Missing Lines: 284

📊 Current Coverage
==================
  Line Coverage: 52.8% (274/518)
  Branch Coverage: 38.3% (80/209)
  Missing Lines: 244

📈 Coverage Comparison
=====================
  Line Coverage: 45.2% → 52.8% ↑ 7.6% 🟢
  Branch Coverage: 32.1% → 38.3% ↑ 6.2% 🟢
  Lines Covered: 234 → 274 ↑ 40 🟢
  Missing Lines: 284 → 244 ↓ 40 🟢

🟡 ALMOST THERE! You're within 10% of the 80% goal!
✅ Coverage improved by 7.6%!
```

### `scripts/coverage_baseline.py`

Establishes and tracks coverage baselines over time.

**Features:**
- Set initial coverage baseline
- Track progress against baseline
- Milestone tracking (40%, 60%, 80%)
- Historical trend analysis
- Comprehensive reporting

**Usage:**
```bash
# Set current coverage as baseline
python scripts/coverage_baseline.py --set-baseline

# Check current coverage against baseline
python scripts/coverage_baseline.py --check

# Show baseline status and history
python scripts/coverage_baseline.py --status

# Generate comprehensive report
python scripts/coverage_baseline.py --report coverage_report.json

# Force overwrite existing baseline
python scripts/coverage_baseline.py --set-baseline --force
```

**Milestones:**
- 🏆 **40%**: Initial baseline established
- 🏆 **60%**: Good progress achieved
- 🏆 **80%**: Target coverage reached

## Workflow Guide

### Initial Setup

1. **Install Dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

2. **Establish Baseline:**
   ```bash
   ./scripts/run_coverage.sh
   python scripts/coverage_baseline.py --set-baseline
   ```

3. **Review Initial Coverage:**
   ```bash
   open htmlcov/index.html  # macOS
   start htmlcov/index.html  # Windows
   ```

### Daily Development Workflow

1. **Run Tests with Coverage:**
   ```bash
   ./scripts/run_coverage.sh
   ```

2. **Check Progress:**
   ```bash
   python scripts/coverage_baseline.py --check
   ```

3. **Compare with Previous:**
   ```bash
   python scripts/compare_coverage.py
   ```

### Weekly Review

1. **Generate Comprehensive Report:**
   ```bash
   python scripts/coverage_baseline.py --report weekly_report_$(date +%Y%m%d).json
   ```

2. **Review Trends:**
   ```bash
   python scripts/coverage_baseline.py --status
   ```

3. **Analyze HTML Report:**
   - Open `htmlcov/index.html`
   - Focus on modules with low coverage
   - Identify missing test cases

### CI/CD Integration

**GitHub Actions Example:**
```yaml
- name: Run coverage
  run: ./scripts/run_coverage.sh

- name: Check coverage regression
  run: python scripts/compare_coverage.py --old coverage_baseline.json --new coverage.json

- name: Upload coverage reports
  uses: actions/upload-artifact@v3
  with:
    name: coverage-reports
    path: |
      htmlcov/
      coverage.xml
      coverage.json
```

## Interpreting Results

### Coverage Metrics

- **Line Coverage**: Percentage of executable lines covered
- **Branch Coverage**: Percentage of conditional branches covered
- **Missing Lines**: Specific lines not covered by tests

### Progress Indicators

- 🎯 **GOAL ACHIEVED**: 80%+ coverage reached
- 🟡 **ALMOST THERE**: 70-79% coverage
- 🟠 **GOOD PROGRESS**: 60-69% coverage
- 🔶 **HALF WAY**: 40-59% coverage
- 🔴 **JUST STARTING**: <40% coverage

### Trend Analysis

- ↑ **Improving**: Coverage increasing over time
- ↓ **Declining**: Coverage decreasing (investigate!)
- → **Stable**: Coverage unchanged

### HTML Report Navigation

1. **Overview**: Overall project coverage
2. **Module View**: Coverage by source file
3. **Source View**: Line-by-line coverage details
4. **Branch View**: Conditional branch coverage

## Best Practices

### Writing Tests for Coverage

1. **Focus on Critical Paths**: Test core functionality first
2. **Edge Cases**: Test error handling and boundary conditions
3. **Integration Points**: Test component interactions
4. **Branch Coverage**: Ensure all conditional paths are tested

### Coverage Quality vs. Quantity

1. **Meaningful Tests**: Focus on test quality, not just coverage numbers
2. **Avoid Test Pollution**: Don't write tests just to increase coverage
3. **Test Scenarios**: Cover real-world use cases
4. **Regular Review**: Periodically review and improve tests

### Maintenance

1. **Update Baselines**: When major features are added
2. **Review Exclusions**: Regularly check `.coveragerc` exclusions
3. **Monitor Trends**: Watch for coverage regression
4. **Adjust Goals**: Update coverage targets as needed

## Troubleshooting

### Common Issues

**Issue**: "No data to report" coverage error
**Solution**: Ensure tests are actually running and importing source code

**Issue**: Coverage not including new files
**Solution**: Check `.coveragerc` source configuration and file patterns

**Issue**: Slow coverage execution
**Solution**: Use `--cov-branch` only when needed, consider parallel execution

**Issue**: Inconsistent coverage between runs
**Solution**: Check for test order dependencies, clean environment before running

### Debug Commands

```bash
# Check coverage configuration
coverage debug config

# Check source discovery
coverage debug sys

# Run with verbose output
coverage run -m pytest tests/ -v --cov=src

# Check specific file coverage
coverage report --include="src/core/models.py"
```

## Integration with Development Tools

### VS Code Integration

Install these extensions:
- **Python**: Microsoft's Python extension
- **Coverage Gutters**: Visual coverage indicators
- **Test Explorer**: Test discovery and running

**Settings (.vscode/settings.json):**
```json
{
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/", "--cov=src"],
    "coverage-gutters.coverageFileNames": [".coverage"],
    "coverage-gutters.coverageReportFileName": "htmlcov/index.html"
}
```

### Pre-commit Hooks

```bash
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest-cov
        name: pytest with coverage
        entry: python -m pytest
        args: ["tests/", "--cov=src", "--cov-fail-under=40"]
        language: system
        pass_filenames: false
```

## Goals and Milestones

### Current Target: 80% Line Coverage

**Rationale:**
- **Good Balance**: Comprehensive testing without excessive overhead
- **Industry Standard**: Recognized as quality threshold
- **Achievable**: Realistic goal for complex codebase
- **Maintainable**: Sustainable level for ongoing development

**Progression Plan:**
1. **Phase 1**: Establish baseline (current state)
2. **Phase 2**: Reach 40% (critical paths)
3. **Phase 3**: Reach 60% (core functionality)
4. **Phase 4**: Reach 80% (comprehensive coverage)

### Success Metrics

- **Line Coverage**: ≥80% of executable lines
- **Branch Coverage**: ≥70% of conditional branches
- **Module Coverage**: All core modules ≥60%
- **Trend**: Positive or stable over time
- **Regression**: Zero coverage loss in existing code

## Advanced Usage

### Selective Coverage

```bash
# Coverage for specific module
python -m pytest tests/test_core_models.py --cov=src/core/models

# Coverage with specific markers
python -m pytest tests/ -m "unit" --cov=src

# Exclude slow tests from coverage
python -m pytest tests/ -m "not slow" --cov=src
```

### Combination with Other Tools

```bash
# Coverage with profiling
python -m pytest tests/ --cov=src --profile

# Coverage with mutation testing
python -m pytest tests/ --cov=src && mutmut run --paths-to-mutate src/

# Coverage with type checking
mypy src/ && python -m pytest tests/ --cov=src
```

This comprehensive coverage infrastructure provides the tools and workflows needed to systematically improve test coverage toward the 80% goal while maintaining code quality and development velocity.