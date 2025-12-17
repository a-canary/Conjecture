---
description: Run tests with coverage analysis
agent: build
model: chutes/zai-org/GLM-4.6-FP8
---

Run comprehensive test suite with coverage:

1. Run full test suite:
!`python -m pytest tests/ -v --cov=src --cov-report=term-missing`

2. Analyze results:
   - Identify failing tests and suggest fixes
   - Review coverage gaps
   - Check for test quality issues
   - Look for flaky tests

3. Report:
   - Test pass/fail summary
   - Coverage percentage and gaps
   - Recommended improvements
   - Priority fixes needed

Focus on maintaining/improving coverage while ensuring all tests pass.
