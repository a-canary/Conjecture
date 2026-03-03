# Pulse Schedule

Scheduled tasks executed by the pulse loop (every 5 minutes).
Edit via `/director:pulse` skill.

## Tasks

### watchdog
- interval: 5
- type: inline
- description: Detect stalled states, crashed PIDs, dirty stale workspaces

### github
- interval: 60
- type: subagent
- description: Check GitHub for new PRs, issues, and activity

### upskill
- interval: 60
- type: subagent
- trigger: 10+ new memories since last run
- description: Extract reusable skills from memory files

### plan_review
- interval: 180
- type: subagent
- description: Review progress against PLAN.md, post status update

### screen-check
- interval: 10
- type: inline
- description: Check on running processes, detect stalls, manage server costs
