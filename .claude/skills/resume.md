# /resume — Session Orientation

Run at the start of every Director session. Orient before acting.

## Steps

### 1. Read Inbox
```bash
python3 -c "
import json
from pathlib import Path
entries = [json.loads(l) for l in Path('/data/inbox.jsonl').read_text().splitlines() if l.strip()]
pending = [e for e in entries if not e.get('processed')]
print(f'{len(pending)} unprocessed, {len(entries)} total')
for e in pending:
    print(f'  [{e.get(\"author\",\"?\")}] {e.get(\"content\",\"\")[:120]}')
"
```

### 2. Check Running Background Tasks
```bash
python3 -c "
import json, os
from pathlib import Path
f = Path('/tmp/director-tasks.json')
if not f.exists():
    print('No background tasks tracked')
else:
    tasks = json.loads(f.read_text())
    for name, info in tasks.items():
        pid = info.get('pid')
        alive = True
        try: os.kill(pid, 0)
        except OSError: alive = False
        result = Path(f'/tmp/task-{name}.result').exists()
        print(f'  {name}: pid={pid} alive={alive} result={result}')
"
```

### 3. Check Workspace State
```bash
git -C /workspace status --short
git -C /workspace log --oneline -5
```

### 4. Read Memory
```bash
# Read project memory if it exists
python3 -c "
from pathlib import Path
mem = Path('/workspace/.claude/memory')
if mem.exists():
    for f in sorted(mem.glob('*.md')):
        print(f'=== {f.name} ===')
        print(f.read_text()[:500])
        print()
"
```

### 5. Gap Analysis

Read PLAN.md and check progress against memory:
```bash
cat /workspace/PLAN.md 2>/dev/null || echo "No PLAN.md"
```

- Which steps are done `[x]`? Which are in-flight `[~]`? Which are pending `[ ]`?
- Are all gates for the current phase met? If yes → advance Current Phase.
- Does MEMORY.md contradict PLAN.md? (e.g. step marked done but gate not ticked)
- Determine plan status: `on-track | replanning | waiting`

Output: `PLAN STATUS: {on-track | replanning | waiting}`

### 6. Synthesize

Based on the above, write a terse orientation summary to stdout:

```
RESUME:
  Inbox: <N pending messages>
  Tasks: <running task names, or "none">
  Git: <clean|N uncommitted files>
  Last commit: <message>
  Plan: <current phase, N pending steps, status>
  Status: <what was I doing? what's next?>
```

Then process pending inbox entries in priority order:
1. Task completions (`author: "system"`) — read result, continue dev loop
2. User requests — respond or plan
3. Pulse / alerts — check and report

## When to Use

- **Start of every session** — before reading any user message
- **After a long gap** — re-orient if returning after >10 min of no activity
- **After `thread_state: "blocked"`** — check if blocker was resolved

## Self-Improvement

If you discover better orientation checks during a session, update this file at `/workspace/.claude/skills/resume.md`.
