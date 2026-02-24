# /delegate — Dispatch Subagent

Use to delegate focused work to a Developer or Explorer subagent.

## Developer — code changes

For: code edits, tests, commits in `/workspace`. Cannot browse web or install packages.

```bash
TASK_NAME="{name}"

nohup bash -c "
  claude --agent developer \
    --print --output-format text \
    --dangerously-skip-permissions \
    -p '{task_description}' \
    > /tmp/task-${TASK_NAME}.result 2>&1

  python3 -c \"
import json, time, fcntl
entry = json.dumps({
  'author': 'system',
  'content': '[task:${TASK_NAME}] Developer done. Results in /tmp/task-${TASK_NAME}.result',
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

echo "Developer subagent started (PID $!)"
```

## Explorer — read-only research

For: web research, codebase exploration, external API reads. Cannot write or execute.

```bash
TASK_NAME="{name}"

nohup bash -c "
  claude --agent explorer \
    --print --output-format text \
    --dangerously-skip-permissions \
    -p '{research_question}' \
    > /tmp/task-${TASK_NAME}.result 2>&1

  python3 -c \"
import json, time, fcntl
entry = json.dumps({
  'author': 'system',
  'content': '[task:${TASK_NAME}] Explorer done. Results in /tmp/task-${TASK_NAME}.result',
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

echo "Explorer subagent started (PID $!)"
```

## Writing a Good Task Description

Include in the prompt:
- **Goal**: What should be produced?
- **Constraints**: What files/dirs are in scope?
- **Output**: Where to write results (e.g. `/tmp/task-{name}.result` or a file in `/workspace`)
- **Context**: Key facts the subagent needs (don't assume it has memory)

## When to Use Which

| Task | Agent |
|------|-------|
| Implement a feature | Developer |
| Fix a failing test | Developer |
| Research a library or API | Explorer |
| Audit external code for security | Explorer |
| Write a commit | Developer |
| Find examples online | Explorer |

## After Delegating

1. Track the PID in `/tmp/director-tasks.json` (same as `/create-task`)
2. Post to outbox: `"Delegated {task} to Developer/Explorer — will resume on completion"`
3. Continue other inbox work — do NOT wait

## Self-Improvement

Found a better delegation pattern? Update `/workspace/.claude/skills/delegate.md`.
