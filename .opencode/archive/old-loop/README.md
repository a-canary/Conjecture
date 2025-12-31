# InvisibleHand - Systematic Agentic Iteration Framework

**Purpose**: Drive steady, reliable progress toward all success criteria through systematic iteration, hyper-skeptical analysis, and truthful implementation.

## Quick Start

```bash
# Run continuous autonomous iteration loop
python .loop/main.py

# Run single task
python .loop/main.py "implement feature X"

# Use opencode command
opencode -p "/loop"

# Run just the reviewer (hyper-skeptical analysis)
opencode -p "/loop-review"

# Run just the developer (implementation)
opencode -p "/loop-developer"
```

## Overview

InvisibleHand is a systematic framework that orchestrates continuous improvement through:

```
ASSESS → PLAN → IMPLEMENT → TEST → VALIDATE → LOOP
   ↓         ↓          ↓        ↓         ↓
 STATUS   REVIEW   DEVELOPER  EVIDENCE  UPDATE
```

### Core Components

1. **InvisibleHand Orchestrator** (`agent/invisiblyhand.md`)
   - Coordinates the entire iteration loop
   - Makes progress decisions based on evidence
   - Only asks user when 100% complete or truly blocked

2. **Loop-Review** (`command/loop-review.md`)
   - Hyper-skeptical analysis with extreme scrutiny
   - Root cause analysis (not symptom fixing)
   - Generates simple, atomic tasks
   - Demands evidence for all claims

3. **Loop-Developer** (`command/loop-developer.md`)
   - Implements minimal, focused changes
   - Tests thoroughly with real evidence
   - Reports truthfully (what worked, what didn't)
   - Admits limitations and unknowns

4. **Persistent Core Loop** (`.loop/main.py`)
   - Python framework for continuous iteration
   - Status assessment and task selection
   - Timeout enforcement (10 minutes per iteration)
   - State file management

## The InvisibleHand Loop

### Iteration Cycle (10 minutes maximum)

```
┌─────────────────────────────────────────────────┐
│ 1. ASSESS STATUS                             │
│    - Load success_criteria.json                  │
│    - Load backlog.md                            │
│    - Calculate completion percentage               │
│    - Identify incomplete criteria                │
└──────────────┬────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────┐
│ 2. SELECT NEXT TASK                          │
│    - Priority: CRITICAL > HIGH > MEDIUM > LOW  │
│    - Choose first incomplete high-priority item    │
└──────────────┬────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────┐
│ 3. LOOP-REVIEW                             │
│    - Hyper-skeptical analysis                  │
│    - Root cause identification                 │
│    - Simple task generation                    │
│    - Evidence-based confidence                │
└──────────────┬────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────┐
│ 4. LOOP-DEVELOPER                          │
│    - Minimal implementation                    │
│    - Thorough testing                        │
│    - Real evidence collection                 │
│    - Truthful reporting                     │
└──────────────┬────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────┐
│ 5. VALIDATE RESULTS                          │
│    - Compare to success criteria              │
│    - Verify evidence (no mocks)              │
│    - Check for regressions                   │
│    - Update documentation                  │
└──────────────┬────────────────────────────────┘
               │
         ┌───────┴───────┐
         │ Complete?      │
         │               │
        Yes              No
         │               │
         ▼               ▼
     ┌─────────┐   ┌─────────────────┐
     │ STOP:   │   │ 6. UPDATE STATE │
     │ Report  │   │    - backlog.md  │
     │         │   │    - ANALYSIS.md  │
     │         │   │    - criteria.json│
     │         │   │    - git commit │
     └────┬────┘   └────────┬────────┘
          │                 │
          └─────────────────┘
                  │
                  ▼
            ┌─────────────────┐
            │ 7. LOOP BACK   │
            └─────────────────┘
```

## When InvisibleHand Asks for User Input

InvisibleHand operates autonomously and **only** asks for user direction in these situations:

### 1. All Success Criteria Met (100%)

```
🎉 ALL SUCCESS CRITERIA ACHIEVED!

Completion Report:
- Total Criteria: 42
- Met Criteria: 42
- Progress: 100%
- Iterations: 17
- Total Time: 2h 45m

Next Steps - User Decision Required:
Please choose one:
1. Set new success criteria for next phase
2. Refine existing criteria for deeper work
3. Stop iteration and review results
```

### 2. Blocked on All Tasks (After 3+ Attempts)

```
🚨 BLOCKED - USER INTERVENTION REQUIRED

Situation Analysis:
- All high-priority tasks attempted multiple times
- Each approach has failed with documented reasons
- Root cause appears to be: architectural constraint

Attempts Made:
1. SC-151-3 (Delete duplicate CLIs) - Import errors persist
2. SC-151-4 (Delete duplicate SQLite managers) - Test failures
3. SC-151-5 (Consolidate prompt templates) - Runtime errors

Root Cause:
Deep coupling between CLI and data layers prevents independent removal.

Proposed Resolutions:
1. [Technical workaround] - Refactor coupling first (-2h estimated)
2. [Architectural change] - Implement factory pattern (-4h estimated)
3. **Scope increase** - Add success criteria for refactoring (+2 days)

User Decision Required:
Do you want to:
1. Try resolution option 1 (technical workaround)?
2. Try resolution option 2 (architectural change)?
3. Approve scope increase with new success criteria?
4. Stop iteration and review status?
```

### 3. Scope Increase Required

```
📋 SCOPE INCREASE PROPOSED

Situation:
- Current scope insufficient to achieve code size reduction
- Evidence shows: src/ still 42k lines after dead code removal
- Additional capabilities needed: automated dependency analysis

Proposed Scope Increase:
1. Add new success criteria: SC-151-7 (Automated dependency analysis)
2. Requires changes to: core_tools/, src/data/
3. Estimated effort: 6-8 hours
4. Benefits: Identify 85%+ of dead code automatically

User Approval Required:
Do you approve this scope increase?
- If YES: Add new criteria and continue iteration
- If NO: Stop iteration and review current progress
```

## Key Principles

### Transparency
- Always show evidence, never claims
- Report failures honestly with details
- Admit unknowns and uncertainties
- Document what worked AND what didn't work
- Never hide partial successes

### Simplicity
- One task at a time (10-minute focus)
- Atomic changes independently completable
- Clear, measurable success criteria
- No over-engineering
- Minimal viable changes

### Evidence-Based
- Real test outputs, not "tests pass"
- Actual metrics, not "improved by X%"
- Code changes with line numbers
- Exact error messages and stack traces
- No mocks or simulations

### Timeout Enforcement
- Strict 10-minute timeout per iteration
- Prioritize: assess > plan > implement > test > validate
- Stop and report if time exceeded
- Don't compromise quality for speed

## Required Tools

### OpenCode Commands
- `read`: Read any file in the project
- `write`: Create new files
- `edit`: Edit existing files
- `bash`: Execute commands (with timeout)

### Python Loop Framework
- `.loop/main.py`: Persistent core loop
- Success criteria parsing
- Task selection algorithm
- Iteration orchestration
- State file management

## State Files (Records of Truth)

These structured documents must always be accurate:

### `.agent/success_criteria.json`
All success criteria with:
- ID and name
- Description and purpose
- Target (measurable outcome)
- Test method (verifiable command)
- Status (pending, completed, blocked)
- Confidence level (0-100%)

### `.agent/backlog.md`
Task history with:
- Completed items (with results, files, learning)
- Current improvement cycles
- Pending work items
- Status tracking (open, started, AI tested, COMMITTED)

### `ANALYSIS.md`
Metrics and trends:
- Test coverage progress
- Code size measurements
- Benchmark results
- Issue tracking
- Recent findings

### `CODE_SIZE_REDUCTION_PLAN.md`
Strategic plan for:
- Current state vs targets
- Reduction strategy phases
- Success metrics tracking
- Risk assessment

### `RESULTS.md`
Session results:
- Cycles completed
- Tests passing/failing
- Files created/modified
- Key metrics achieved
- Lessons learned

## Example Workflow

### Starting the Loop

```bash
python .loop/main.py
```

**Output**:
```
================================================================================
INVISIBLE HAND - SYSTEMATIC AGENTIC ITERATION
================================================================================
Project Root: D:\projects\Conjecture
Max Iterations: 1000
Timeout per Iteration: 10 minutes
================================================================================

================================================================================
ITERATION 1
================================================================================

================================================================================
STATUS ASSESSMENT
================================================================================

Success Criteria Progress:
  Total: 42
  Completed: 12
  Incomplete: 30
  Progress: 28.6%

Top Incomplete Criteria:
  1. SC-151-3 - Delete duplicate CLI implementations
  2. SC-151-4 - Delete duplicate SQLite managers
  3. SC-151-5 - Consolidate prompt templates
  4. SC-152-1 - Test coverage to 15%
  5. SC-152-2 - Create test suite for adaptive_compression.py

================================================================================
LOOP-REVIEW: Hyper-skeptic analysis and planning
================================================================================

[Hyper-skeptical analysis output...]

================================================================================
LOOP-DEVELOPER: Implementation and testing
================================================================================

[Implementation and testing output...]

================================================================================
UPDATE BACKLOG
================================================================================

  Task ID: SC-151-3
  New Status: AI tested
  Result: {"success": true, "tests_passed": 42, "coverage": 85.2%}

[Continue to iteration 2...]
```

### Running Single Task

```bash
python .loop/main.py "Add logging to SQLite manager"
```

## Testing the System

### Test Loop Framework
```bash
# Run loop with limited iterations
python .loop/main.py --max-iterations 3

# Verify status assessment works
python -c "from .loop.main import InvisibleHand; h = InvisibleHand(Path('.')); print(h.assess_status())"
```

### Test Individual Commands
```bash
# Test loop-review
opencode -p "/loop-review Analyze SC-151-3"

# Test loop-developer
opencode -p "/loop-developer Implement SC-151-3"
```

### Test Integration
```bash
# Run full loop cycle with verbose output
python .loop/main.py

# Check that all state files are updated
ls -la .agent/ ANALYSIS.md RESULTS.md
```

## Troubleshooting

### Loop Not Starting
```bash
# Check Python version
python --version  # Should be 3.8+

# Check dependencies
pip list | grep -E "(pytest|pathlib)"

# Verify file structure
ls -la .loop/ .opencode/
```

### State Files Not Updating
```bash
# Check file permissions
ls -la .agent/backlog.md .agent/success_criteria.json

# Verify JSON format
python -c "import json; json.load(open('.agent/success_criteria.json'))"

# Check loop framework output
python .loop/main.py --max-iterations 1 2>&1 | grep -E "(UPDATE|STATUS)"
```

### Commands Not Found
```bash
# Verify opencode config
cat .opencode/opencode.json | grep -A5 "command"

# Test opencode installation
opencode --version

# Check command files
ls -la .opencode/command/loop*.md
```

## Success Indicators

The InvisibleHand system is working correctly when:

1. ✅ Loop runs continuously without user intervention
2. ✅ Progress made each iteration (even small)
3. ✅ Evidence is real (no mocks or simulations)
4. ✅ Documentation stays accurate and up-to-date
5. ✅ User only asked at 100% completion or blockage
6. ✅ Quality never compromised for speed
7. ✅ Failures are learned from, not hidden
8. ✅ Timeout enforced (never exceeds 10 minutes/iteration)

## Architecture

```
.opencode/
├── agent/
│   ├── invisiblyhand.md     # Orchestrator (you are here)
│   ├── planner.md           # Strategic planning agent
│   └── coder.md             # Implementation agent
├── command/
│   ├── loop.md              # Legacy cycle command
│   ├── loop-review.md        # Hyper-skeptical reviewer
│   └── loop-developer.md     # Developer
└── opencode.json         # Configuration

.loop/
└── main.py              # Persistent core loop framework

.agent/
├── success_criteria.json # All objectives
├── backlog.md          # Task history
└── [other state files]

[Root]
├── ANALYSIS.md         # Metrics and trends
├── CODE_SIZE_REDUCTION_PLAN.md  # Strategic plan
└── RESULTS.md          # Session results
```

## Maintenance

### Adding New Success Criteria
1. Add to `.agent/success_criteria.json`
2. Ensure test_method is runnable and verifiable
3. Set status to "pending" or "started"
4. Set appropriate priority (CRITICAL, HIGH, MEDIUM, LOW)

### Updating Loop Framework
1. Edit `.loop/main.py`
2. Test with `--max-iterations 1`
3. Verify state file updates
4. Check timeout enforcement
5. Document changes

### Customizing Agents
1. Edit `.opencode/agent/invisiblyhand.md` for orchestrator
2. Edit `.opencode/command/loop-review.md` for reviewer
3. Edit `.opencode/command/loop-developer.md` for developer
4. Test changes with limited iterations
5. Monitor for regressions

## License

Part of the Conjecture project. See project LICENSE for details.
