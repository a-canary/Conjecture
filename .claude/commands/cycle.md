# cycle.md

Generic work cycle for fixing bugs, exploring new features, or simplifing complexities.

## Steps
1. repeat infinite times:
  - start a Task to perform [Cycle Steps](#cycle-steps)

### Cycle Steps
1. analyse project status, git diff files, read TODO.md 
2. Consider 5 possible high-impact work priorieties, and make a focused goal for this cycle that should take about 10 minutes. These can be defined tasks/goals or investigations to learn more about something that your unsure about. 
3. Complete research and analysis
  - analyse related project code
  - analyse related past cycles in log of completed tasks
  - define a hypothesis or design changes with expected outcomes and value or benefit
  - research and consider 2 alternatives, then select 1 of the 3
  - If you have conflicts or concerns or need clarity on the goal, ask user to confrim your 10-minute task at the concept level.
4. Complete minimal changes to achieve goal improvement
5. validate improvement: run full test suite, run code coverage, and gather significant metrics and `/review.md` workflow
6. Update ANALYSIS.md to report current quality of code, docs, test results, and benchmarks and other metrics.  
7. IF you are confident this change is a net positive compared to main branch (diff ANALYSIS.md to main branch to check progression), 
  THEN:
    1. delete dead code and deprecated files
    2. update TODO.md items
    3. Adding next steps and new identified work or explorations to TODO.md
    4. revert temporary changes, debug logging, cycle specific plans and reports
    5. check important and relevant docs are accurate
    6. `git commit` relevant changes and Cycle State Files    
  ELSE:
    - If issues are small and addressable, goto [this step](#cycle-steps/4)
    - If issues are significant and not addressable:
      1. git revert all changes
      2. set cycle as failure in TODO.md, and git commit
8. At this point all files have been commited or reverted, Cycle State Files have been updated, and you start next cycle.

While in this cycle, remember to:
- Keep a big focus on transparency, maintanence, verifications, 
- Keep project small, simple with high quality code, high value features and file content
- Don't fix Dead code. 
- Don't fix code until it has code coverage.
- Don't allow mock code or synthetic results.
- Tell me when I'm wrong.
- see file formats in [Cycle State Files](#cycle-state-files)

### Cycle State Files
#### TODO.md
Concise inventory of task concepts.

#### ANALYSIS.md
**Intent**: Comprehensive assessment of project quality, including code, documentation, tests, and benchmarks.

**Format**:
Simple Dictionary: `metric_name: value`
Plus a 3 sentence summary of the analysis, improvements and concerns.

**Content**:
Metrics collection should reflect the quality of current code, documentation, tests, benchmarks, and any other aspect that the design or user declares as important. Diff-ing this file with previous commits can help identify trends and areas for improvement.
