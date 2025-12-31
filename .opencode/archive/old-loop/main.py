#!/usr/bin/env python3
"""
InvisibleHand - Systematic Agentic Iteration Framework

Runs automated analyse/work loop until all success criteria pass.

Phases:
  1. ANALYSE - Verify work, check criteria, identify next tasks
  2. WORK - Execute ONE high-priority task

Repeats until:
  - All success criteria pass (100% complete), OR
  - Max iterations reached, OR
  - User input required (scope expansion needed)

The loop is persistent - it keeps trying even when stuck, exploring failed
attempts until finding a solution. Only stops for user input when scope
expansion is truly needed.

Usage:
    python .loop/main.py [--max-iterations N] [--dry-run] [--status]
"""

import json
import subprocess
import sys
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
AGENT_DIR = PROJECT_ROOT / ".agent"
LOOP_DIR = PROJECT_ROOT / ".loop"
SUCCESS_CRITERIA = AGENT_DIR / "success_criteria.json"
BACKLOG = AGENT_DIR / "backlog.md"
STATE_FILE = LOOP_DIR / "state.json"
LOOP_LOG = LOOP_DIR / "loop.log"

MAX_ITERATIONS = 50
ANALYSE_TIMEOUT = 180  # 3 minutes for analyse
WORK_TIMEOUT = 600  # 10 minutes for work
VALIDATE_TIMEOUT = 60  # 1 minute for validation


def log(msg: str, level: str = "INFO"):
    """Log message with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    print(line)

    # Also append to log file
    LOOP_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOOP_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_json(path: Path) -> dict:
    """Load JSON file, return empty dict if not found."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        log(f"Could not load {path}: {e}", "WARN")
        return {}


def save_json(path: Path, data: dict):
    """Save JSON file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        log(f"Could not save {path}: {e}", "ERROR")


def load_state() -> dict:
    """Load persistent state."""
    state = load_json(STATE_FILE)
    if not state:
        state = {
            "total_iterations": 0,
            "successful_iterations": 0,
            "failed_iterations": 0,
            "tasks_completed": [],
            "consecutive_failures": 0,
            "last_run": None,
        }
    return state


def save_state(state: dict):
    """Save persistent state."""
    state["last_run"] = datetime.now().isoformat()
    save_json(STATE_FILE, state)


def get_success_criteria() -> List[Dict[str, Any]]:
    """Load success criteria from file."""
    data = load_json(SUCCESS_CRITERIA)
    return data.get("criteria", [])


def get_incomplete_criteria() -> List[Dict[str, Any]]:
    """Get incomplete success criteria."""
    all_criteria = get_success_criteria()
    return [
        c for c in all_criteria if c.get("status") not in ["completed", "AI tested"]
    ]


def check_all_criteria_pass() -> bool:
    """Check if all success criteria are complete."""
    incomplete = get_incomplete_criteria()
    return len(incomplete) == 0


def get_completion_percentage() -> float:
    """Get completion percentage."""
    all_criteria = get_success_criteria()
    if not all_criteria:
        return 0.0
    incomplete = get_incomplete_criteria()
    completed = len(all_criteria) - len(incomplete)
    return (completed / len(all_criteria)) * 100


def run_opencode(command: str, timeout: int = 300) -> int:
    """Run opencode with given command, return exit code."""
    cmd = f"opencode run --command {command}"
    log(f"Running: {cmd} (timeout={timeout}s)")

    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            cwd=PROJECT_ROOT,
            capture_output=False,  # Let output flow to terminal
            shell=True,
        )
        return result.returncode
    except subprocess.TimeoutExpired:
        log(f"Command timed out after {timeout}s", "ERROR")
        return -1
    except Exception as e:
        log(f"Command failed: {e}", "ERROR")
        return -1


def run_command(cmd: str, timeout: int = 60) -> Tuple[bool, str, str]:
    """Run a shell command and return (success, stdout, stderr)."""
    log(f"Running: {cmd}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=PROJECT_ROOT,
        )
        success = result.returncode == 0
        return success, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        log(f"Command timed out after {timeout}s", "ERROR")
        return False, "", f"Timeout after {timeout}s"
    except Exception as e:
        log(f"Command failed: {e}", "ERROR")
        return False, "", str(e)


def extract_command(test_method: str) -> str:
    """Extract the actual command from test_method, removing comments."""
    if not test_method:
        return ""
    cmd = test_method
    # Remove trailing parenthetical comments like "(should be ≤5)"
    cmd = re.sub(r"\s*\([^)]*\)\s*$", "", cmd)
    # Remove inline comments starting with #
    cmd = re.sub(r"\s*#.*$", "", cmd)
    cmd = cmd.strip()

    # Windows compatibility: convert single quotes to double quotes for python -c
    if sys.platform == "win32":
        if "python -c '" in cmd:
            cmd = cmd.replace("python -c '", 'python -c "')
            last_quote = cmd.rfind("'")
            if last_quote > 0:
                cmd = cmd[:last_quote] + '"' + cmd[last_quote + 1 :]

    return cmd


def validate_criterion(criterion: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate a single success criterion using its test_method."""
    test_method = criterion.get("test_method", "")
    if not test_method:
        return False, "No test_method defined"

    cmd = extract_command(test_method)
    if not cmd:
        return False, "No valid command in test_method"

    log(f"Validating {criterion.get('id')}: {criterion.get('name')}")

    success, stdout, stderr = run_command(cmd, timeout=VALIDATE_TIMEOUT)
    output = stdout + stderr

    # Check for error patterns
    error_patterns = [
        "No such file or directory",
        "command not found",
        "ModuleNotFoundError",
        "ImportError",
        "FileNotFoundError",
        "Error:",
        "error:",
        "FAILED",
        "Traceback (most recent call last)",
    ]

    has_error = any(pattern in output for pattern in error_patterns)
    if has_error:
        success = False

    return success, output[:500]


def update_criterion_status(criterion_id: str, new_status: str, result: str = ""):
    """Update the status of a criterion in success_criteria.json."""
    try:
        data = load_json(SUCCESS_CRITERIA)
        for criterion in data.get("criteria", []):
            if criterion.get("id") == criterion_id:
                criterion["status"] = new_status
                if result:
                    criterion["last_result"] = result[:500]
                criterion["last_validated"] = datetime.now().isoformat()
                break
        save_json(SUCCESS_CRITERIA, data)
        log(f"Updated {criterion_id} status to: {new_status}")
    except Exception as e:
        log(f"Failed to update criterion status: {e}", "ERROR")


def select_next_task(skip_ids: List[str] = None) -> Optional[Dict[str, Any]]:
    """Select next high-priority task from success criteria."""
    skip_ids = skip_ids or []
    incomplete = get_incomplete_criteria()

    # Filter out tasks we've already tried
    available = [c for c in incomplete if c.get("id") not in skip_ids]

    if not available:
        return None

    # Priority order: CRITICAL > HIGH > MEDIUM > LOW
    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    available.sort(
        key=lambda x: (
            priority_order.get(x.get("priority", "MEDIUM"), 99),
            x.get("id", ""),
        )
    )

    return available[0]


def run_analyse() -> bool:
    """Run ANALYSE phase - verify work and check criteria."""
    log("=" * 60)
    log("ANALYSE PHASE")
    log("=" * 60)

    # Check if loop-review command exists
    exit_code = run_opencode("loop-review", ANALYSE_TIMEOUT)

    if exit_code != 0:
        log(f"Analyse failed (code {exit_code})", "WARN")
        # Don't fail hard - continue to work phase
        return True  # Allow work to continue

    return True


def run_work(
    skip_ids: List[str] = None, validate_only: bool = False
) -> Tuple[bool, Optional[str]]:
    """Run WORK phase - execute one task. Returns (success, task_id)."""
    log("=" * 60)
    log("WORK PHASE" + (" (validate-only)" if validate_only else ""))
    log("=" * 60)

    # Select next task
    task = select_next_task(skip_ids=skip_ids)
    if not task:
        log("No available tasks to work on")
        return True, None

    task_id = task.get("id", "UNKNOWN")
    log(f"Selected task: {task_id} - {task.get('name')}")
    log(f"Priority: {task.get('priority')}")

    # First, validate if it's already complete
    is_valid, output = validate_criterion(task)

    if is_valid:
        log(f"Task {task_id} already meets criteria!")
        update_criterion_status(task_id, "completed", output)
        return True, task_id

    log(f"Task {task_id} needs implementation")

    # In validate-only mode, just report and skip
    if validate_only:
        log(f"Skipping {task_id} (validate-only mode)")
        return False, task_id

    # Try to invoke loop-developer to implement
    exit_code = run_opencode("loop-developer", WORK_TIMEOUT)

    if exit_code != 0:
        log(f"Work failed for {task_id} (code {exit_code})", "WARN")
        return False, task_id

    # Re-validate after work
    is_valid, output = validate_criterion(task)
    if is_valid:
        log(f"Task {task_id} completed successfully!")
        update_criterion_status(task_id, "completed", output)
        return True, task_id
    else:
        log(f"Task {task_id} still not complete after work", "WARN")
        return False, task_id


def summarize_iteration(
    iteration: int,
    analyse_ok: bool,
    work_ok: bool,
    task_id: Optional[str],
    start_time: float,
) -> None:
    """Log iteration summary."""
    duration = time.time() - start_time
    completion = get_completion_percentage()
    incomplete = len(get_incomplete_criteria())

    status = "OK" if work_ok else "FAIL"
    task_info = task_id if task_id else "none"

    log("-" * 60)
    log(
        f"ITER {iteration}: {duration:.0f}s | {status} {task_info} | {completion:.1f}% ({incomplete} remaining)"
    )
    log("-" * 60)


def show_status():
    """Show current progress status."""
    log("=" * 60)
    log("STATUS")
    log("=" * 60)

    all_criteria = get_success_criteria()
    incomplete = get_incomplete_criteria()
    completed = len(all_criteria) - len(incomplete)
    completion = get_completion_percentage()

    log(f"Total criteria: {len(all_criteria)}")
    log(f"Completed: {completed}")
    log(f"Incomplete: {len(incomplete)}")
    log(f"Progress: {completion:.1f}%")

    if incomplete:
        log("\nTop 5 incomplete (by priority):")
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        incomplete.sort(
            key=lambda x: (
                priority_order.get(x.get("priority", "MEDIUM"), 99),
                x.get("id", ""),
            )
        )
        for i, c in enumerate(incomplete[:5], 1):
            log(f"  {i}. [{c.get('priority')}] {c.get('id')} - {c.get('name')}")

    state = load_state()
    log(f"\nTotal iterations: {state.get('total_iterations', 0)}")
    log(f"Successful: {state.get('successful_iterations', 0)}")
    log(f"Failed: {state.get('failed_iterations', 0)}")
    log(f"Tasks completed: {len(state.get('tasks_completed', []))}")
    log(f"Last run: {state.get('last_run', 'Never')}")


def main():
    """Main loop."""
    import argparse

    parser = argparse.ArgumentParser(
        description="InvisibleHand - Systematic Agentic Iteration"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=MAX_ITERATIONS,
        help=f"Maximum iterations (default: {MAX_ITERATIONS})",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would run without executing"
    )
    parser.add_argument(
        "--status", action="store_true", help="Show current progress and exit"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate criteria, don't invoke agents",
    )
    args = parser.parse_args()

    # Show status and exit if requested
    if args.status:
        show_status()
        return 0

    log("=" * 60)
    log(f"INVISIBLE HAND LOOP START (max={args.max_iterations})")
    log("=" * 60)

    # Check initial state
    if check_all_criteria_pass():
        log("SUCCESS! All criteria already pass!")
        return 0

    state = load_state()
    skip_ids = []  # Track tasks that failed this session

    for iteration in range(1, args.max_iterations + 1):
        iter_start = time.time()
        state["total_iterations"] = state.get("total_iterations", 0) + 1

        log("")
        log(f"### ITERATION {iteration}/{args.max_iterations} ###")

        if args.dry_run:
            log("[DRY RUN] Would run analyse + work")
            continue

        # Phase 1: Analyse (optional - verify work)
        if not args.validate_only:
            analyse_ok = run_analyse()
            if not analyse_ok:
                state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
                if state["consecutive_failures"] >= 5:
                    log("Too many consecutive failures, waiting 60s...")
                    time.sleep(60)
                    state["consecutive_failures"] = 0
                else:
                    time.sleep(10)
                save_state(state)
                continue

        # Check completion after analyse
        if check_all_criteria_pass():
            log("SUCCESS! All criteria pass!")
            save_state(state)
            return 0

        # Phase 2: Work (execute one task)
        work_ok, task_id = run_work(skip_ids=skip_ids, validate_only=args.validate_only)

        if work_ok:
            state["consecutive_failures"] = 0
            state["successful_iterations"] = state.get("successful_iterations", 0) + 1
            if task_id:
                tasks_completed = state.get("tasks_completed", [])
                if task_id not in tasks_completed:
                    tasks_completed.append(task_id)
                    state["tasks_completed"] = tasks_completed
                # Remove from skip list if it was there
                if task_id in skip_ids:
                    skip_ids.remove(task_id)
        else:
            state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
            state["failed_iterations"] = state.get("failed_iterations", 0) + 1
            # Add to skip list to try different tasks
            if task_id and task_id not in skip_ids:
                skip_ids.append(task_id)
                log(f"Skipping {task_id} for rest of session")

        # Summarize iteration
        summarize_iteration(iteration, True, work_ok, task_id, iter_start)
        save_state(state)

        # Check completion after work
        if check_all_criteria_pass():
            log("SUCCESS! All criteria pass!")
            return 0

        # Check if we've tried all available tasks
        if len(skip_ids) >= len(get_incomplete_criteria()):
            log("All available tasks attempted this session")
            log(f"Tasks needing implementation: {len(skip_ids)}")
            return 2  # Exit code for "needs implementation"

        # Brief pause between iterations
        time.sleep(2)

    log(f"COMPLETED: All {args.max_iterations} iterations finished")
    completion = get_completion_percentage()
    log(f"Final progress: {completion:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
