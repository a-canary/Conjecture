# cron: */15 * * * *
# timeout: 30
# approval: auto
"""Session failsafe — detect crashed Director sessions mid-task (F-0063).

Checks for signs of an interrupted session:
1. Dirty workspace (uncommitted changes)
2. No recent Claude CLI activity (session file / outbox stale)

If both are true, the session likely crashed mid-task. Posts an alert
to Discord via outbox so the operator can re-engage.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "/app")

OUTBOX = Path("/data/outbox.jsonl")
SESSION_FILE = Path("/data/session.json")
STALE_THRESHOLD = 1800  # 30 min — no activity after this = likely stale
ALERT_COOLDOWN_FILE = Path("/tmp/.session-failsafe-last-alert")
ALERT_COOLDOWN = 3600  # Don't alert more than once per hour


def is_workspace_dirty() -> bool:
    """Check if /workspace has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
            cwd="/workspace",
        )
        return bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, OSError):
        return False


def last_activity_age() -> float:
    """Seconds since the most recent session-related file was modified."""
    candidates = [SESSION_FILE, OUTBOX]
    newest = 0.0
    for f in candidates:
        try:
            mtime = f.stat().st_mtime
            if mtime > newest:
                newest = mtime
        except OSError:
            continue

    # Also check git log for recent commits
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ct"],
            capture_output=True, text=True, timeout=10,
            cwd="/workspace",
        )
        if result.stdout.strip():
            commit_time = float(result.stdout.strip())
            if commit_time > newest:
                newest = commit_time
    except (subprocess.TimeoutExpired, OSError, ValueError):
        pass

    if newest == 0.0:
        return float("inf")
    return time.time() - newest


def should_alert() -> bool:
    """Check cooldown — don't spam alerts."""
    try:
        last = float(ALERT_COOLDOWN_FILE.read_text().strip())
        return (time.time() - last) >= ALERT_COOLDOWN
    except (OSError, ValueError):
        return True


def record_alert():
    """Record that we sent an alert."""
    ALERT_COOLDOWN_FILE.write_text(str(time.time()))


def main():
    dirty = is_workspace_dirty()
    age = last_activity_age()

    if not dirty:
        return  # Clean workspace — nothing to worry about

    if age < STALE_THRESHOLD:
        return  # Recent activity — session is probably still active

    if not should_alert():
        return  # Already alerted recently

    # Stale dirty workspace — session likely crashed mid-task
    entry = json.dumps({
        "message": (
            f"Session failsafe: workspace has uncommitted changes "
            f"with no activity for {int(age / 60)}min. "
            f"Director session may have crashed mid-task."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    with OUTBOX.open("a") as f:
        f.write(entry + "\n")

    record_alert()


if __name__ == "__main__":
    main()
