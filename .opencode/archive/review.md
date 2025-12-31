---
description: Comprehensive code review workflow
agent: plan
model: anthropic/claude-sonnet-4-20250514
---

Execute comprehensive code review:

## Review Process
1. Start full test suite in background (if not already running)
2. Summarize major local changes and determine logical impacts
3. Have you proven completion/progress to current goal?
4. Criticize changes to high standard:
   - Improve organization
   - All elements named accurately (no "*_fixed" qualifiers)
   - All changes in changelists with minimal dependencies
   - Prove functionality and stability
5. Consider planned future work:
   - Look for over-engineered code
   - Ways to simplify design
   - Lean external libraries to use
6. Wait for test results
7. Return comprehensive summary of:
   - Critical changes required
   - High-priority recommendations
   
## Standards
- Code must be production-ready and extremely functional
- Code is not expected to be perfect in every way
- Codebase will improve slowly in auto-testing, naming, data structures

Current git changes:
!`git status`
!`git diff`

Recent commits:
!`git log --oneline -10`

Analyze these changes and provide actionable feedback.
