# cron: */2 * * * *
# timeout: 15
# approval: auto
"""Director pulse — monitor background tasks, alert on stalls.

Reads /tmp/director-tasks.json, checks PIDs, writes status to inbox.
Only writes if there's something to report (no spam on idle).
"""

import fcntl
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

TASKS_FILE = Path("/tmp/director-tasks.json")
INBOX = Path("/data/inbox.jsonl")
INBOX_LOCK = Path("/data/inbox.lock")
DEFAULT_TIMEOUT = 600  # 10 min


def write_inbox(content: str):
    """Append an entry to inbox with file locking."""
    entry = json.dumps({
        "author": "pulse",
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message_id": int(time.time() * 1000),
        "processed": False,
    })
    with open(INBOX_LOCK, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            with INBOX.open("a") as f:
                f.write(entry + "\n")
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)


def pid_alive(pid: int) -> bool:
    """Check if a PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    if not TASKS_FILE.exists():
        return  # No tasks tracked — nothing to do

    try:
        tasks = json.loads(TASKS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return

    if not tasks:
        return

    now = time.time()
    alerts = []
    completed = []

    for name, info in tasks.items():
        pid = info.get("pid")
        started = info.get("started", "")
        timeout = info.get("timeout", DEFAULT_TIMEOUT)

        if not pid:
            continue

        alive = pid_alive(pid)
        result_file = Path(f"/tmp/task-{name}.result")

        if not alive:
            if result_file.exists():
                completed.append(name)
            else:
                # Dead PID, no result file — crashed
                alerts.append(f"[task:{name}] PID {pid} died with no result file")
        else:
            # Still running — check for timeout
            try:
                start_time = datetime.fromisoformat(started).timestamp()
                elapsed = now - start_time
                if elapsed > timeout:
                    alerts.append(
                        f"[task:{name}] stalled — running {int(elapsed)}s (timeout {timeout}s)"
                    )
            except (ValueError, OSError):
                pass

    # Check for orphaned result files (result exists but no task entry)
    for f in Path("/tmp").glob("task-*.result"):
        task_name = f.stem.removeprefix("task-")
        if task_name not in tasks:
            alerts.append(f"[task:{task_name}] orphaned result file: {f}")

    if alerts:
        write_inbox("Pulse: " + "; ".join(alerts))

    # Clean up completed tasks from tracking file
    if completed:
        for name in completed:
            tasks.pop(name, None)
        TASKS_FILE.write_text(json.dumps(tasks))


if __name__ == "__main__":
    main()
