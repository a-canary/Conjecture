---
description: Run infinite iteration cycle for improvement
agent: build
model: chutes/zai-org/GLM-4.6-FP8
subtask: false
---

Execute the infinite improvement cycle:

## Cycle Process
1. Analyze project status using git diff and read .agent/backlog.md
2. Consider 5 possible high-impact work priorities from backlog
3. Choose a focused goal that takes ~10 minutes
4. Research and analysis:
   - Analyze related project code
   - Review past cycles in completed tasks (backlog items 001-015)
   - Define hypothesis/design with expected outcomes
   - Research 2 alternatives, select best of 3
   - Ask user for confirmation if conflicts/concerns arise
5. Complete minimal changes to achieve goal
6. Validate improvement:
   - Run full test suite: `python -m pytest tests/ -v`
   - Check code coverage
   - Gather metrics
7. Update state files:
   - Update .agent/backlog.md with current quality metrics
   - Update ANALYSIS.md with comprehensive assessment
   - Update STATS.yaml with current metrics
8. If confident this is net positive (compare ANALYSIS.md to main branch):
   - Delete dead code and deprecated files
   - Update .agent/backlog.md existing items (mark as AI tested/COMMITTED)
   - Add new backlog items for next steps
   - Revert temporary changes, debug logging
   - Check docs are accurate
   - Git commit relevant changes
   Else:
   - If issues are small: fix them
   - If issues are significant: git revert all changes, mark cycle as failure in .agent/backlog.md

## Key Principles
- Focus on transparency, maintenance, verification
- Keep project small, simple with high quality code
- Don't fix dead code
- Don't fix code until it has coverage
- Don't allow mock code or synthetic results
- Tell me when I'm wrong

Run this cycle continuously until told to stop.
