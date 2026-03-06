# /create-task — Launch Background Task

Use when work takes >15 tool calls. Delegate it, stay responsive.

## Template

Replace `{name}` (short slug, e.g. `tests`, `build`, `lint`) and `{command}`:

```bash
TASK_NAME="{name}"
TASK_CMD="{command}"
TASK_TIMEOUT={seconds}  # default 600

nohup bash -c "
  ${TASK_CMD} > /tmp/task-${TASK_NAME}.result 2>&1
  python3 -c \"
import json, time, fcntl
entry = json.dumps({
  'author': 'system',
  'content': '[task:${TASK_NAME}] completed. Results in /tmp/task-${TASK_NAME}.result',
  'timestamp': '$(date -u +%Y-%m-%dT%H:%M:%SZ)',
  'message_id': int(time.time() * 1000),
  'processed': False,
})
lock = open('/data/inbox.lock', 'w')
fcntl.flock(lock, fcntl.LOCK_EX)
open('/data/inbox.jsonl', 'a').write(entry + chr(10))
fcntl.flock(lock, fcntl.LOCK_UN)
\"
" > /dev/null 2>&1 &

TASK_PID=$!

# Track it
python3 -c "
import json, time
from pathlib import Path
f = Path('/tmp/director-tasks.json')
tasks = json.loads(f.read_text()) if f.exists() else {}
tasks['${TASK_NAME}'] = {
  'pid': ${TASK_PID},
  'started': '$(date -u +%Y-%m-%dT%H:%M:%SZ)',
  'timeout': ${TASK_TIMEOUT},
}
f.write_text(json.dumps(tasks))
"

echo "Task '${TASK_NAME}' started (PID ${TASK_PID})"
```

## After Launching

1. Post to outbox: `"Starting {name} in background — will resume on completion"`
2. Continue other inbox work — do NOT wait for the task
3. When `[task:{name}]` arrives in inbox, read `/tmp/task-{name}.result` and act

## Reading Results

```bash
cat /tmp/task-{name}.result
# Then clean up from tracking:
python3 -c "
import json
from pathlib import Path
f = Path('/tmp/director-tasks.json')
tasks = json.loads(f.read_text()) if f.exists() else {}
tasks.pop('{name}', None)
f.write_text(json.dumps(tasks))
"
```

## Common Tasks

| Work | Command |
|------|---------|
| Run tests | `cd /workspace && python -m pytest -q` |
| Build | `cd /workspace && make build` |
| Lint | `cd /workspace && ruff check .` |
| Long git op | `cd /workspace && git fetch --all` |

## Self-Improvement

Found a better pattern? Update `/workspace/.claude/skills/create-task.md`.
