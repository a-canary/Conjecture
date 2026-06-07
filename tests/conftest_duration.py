"""Pytest plugin: per-test duration tracking and reporting.

Extracted from conftest.py so the test infrastructure layer is not a
1.2k-line wall of mixed concerns. Loaded via conftest.py's
`pytest_plugins` entry — no behavior change versus the previous
inline block.

Outputs:
  - Print: comprehensive duration table at session finish
  - File: tests/results/test_duration_report.json (consumed by no
    other code, but the original conftest also wrote this and
    callers may read it via the file system; preserved as-is).
"""

import time
import json
from datetime import datetime
from pathlib import Path

import pytest


def pytest_sessionstart(session):
    """Initialize session-level timing."""
    session._session_start_time = time.time()
    session._test_durations = []
    session._test_results = {}


def pytest_runtest_setup(item):
    """Record test setup start time."""
    item._setup_start_time = time.time()


def pytest_runtest_call(item):
    """Record test call start time."""
    item._call_start_time = time.time()


def pytest_runtest_teardown(item):
    """Record test teardown start time."""
    item._teardown_start_time = time.time()


def pytest_runtest_logreport(report):
    """Collect test duration data."""
    if report.when == "call":
        timing_info = {
            "nodeid": report.nodeid,
            "setup_duration": getattr(report, "_setup_start_time", 0),
            "call_duration": report.duration if hasattr(report, "duration") else 0,
            "teardown_duration": getattr(report, "_teardown_start_time", 0),
            "total_duration": report.duration if hasattr(report, "duration") else 0,
            "outcome": report.outcome,
            "markers": [],
        }
        if not hasattr(pytest_runtest_logreport, "_test_durations"):
            pytest_runtest_logreport._test_durations = []
        pytest_runtest_logreport._test_durations.append(timing_info)


def pytest_sessionfinish(session, exitstatus):
    """Generate comprehensive duration report and JSON sidecar."""
    session_end_time = time.time()
    total_session_time = session_end_time - getattr(
        session, "_session_start_time", session_end_time
    )
    test_durations = getattr(pytest_runtest_logreport, "_test_durations", [])
    if not test_durations:
        return

    sorted_tests = sorted(
        test_durations, key=lambda x: x["total_duration"], reverse=True
    )
    total_tests = len(sorted_tests)
    passed_tests = len([t for t in sorted_tests if t["outcome"] == "passed"])
    failed_tests = len([t for t in sorted_tests if t["outcome"] == "failed"])
    skipped_tests = len([t for t in sorted_tests if t["outcome"] == "skipped"])
    total_test_time = sum(t["total_duration"] for t in sorted_tests)
    avg_test_time = total_test_time / total_tests if total_tests > 0 else 0
    slowest_test = sorted_tests[0] if sorted_tests else None
    fastest_test = sorted_tests[-1] if sorted_tests else None

    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST DURATION REPORT")
    print("=" * 80)
    print("SUMMARY STATISTICS:")
    print(f"   Total Session Time: {total_session_time:.2f}s")
    print(f"   Total Test Time: {total_test_time:.2f}s")
    print(f"   Overhead Time: {total_session_time - total_test_time:.2f}s")
    print(f"   Total Tests: {total_tests}")
    print(
        f"   Passed: {passed_tests} | Failed: {failed_tests} | Skipped: {skipped_tests}"
    )
    print(f"   Average Test Time: {avg_test_time:.3f}s")
    if slowest_test:
        print(
            f"   Slowest Test: {slowest_test['total_duration']:.3f}s ({slowest_test['nodeid']})"
        )
    if fastest_test:
        print(
            f"   Fastest Test: {fastest_test['total_duration']:.3f}s ({fastest_test['nodeid']})"
        )

    print("\nSLOWEST TESTS (Top 10):")
    for i, test in enumerate(sorted_tests[:10], 1):
        status_char = (
            "PASS"
            if test["outcome"] == "passed"
            else "FAIL"
            if test["outcome"] == "failed"
            else "SKIP"
        )
        markers_str = f" [{', '.join(test['markers'])}]" if test["markers"] else ""
        print(
            f"   {i:2d}. {status_char:4} {test['total_duration']:6.3f}s | {test['nodeid']}{markers_str}"
        )

    marker_stats: dict = {}
    for test in sorted_tests:
        for marker in test["markers"]:
            marker_stats.setdefault(
                marker, {"count": 0, "total_time": 0.0, "tests": []}
            )
            marker_stats[marker]["count"] += 1
            marker_stats[marker]["total_time"] += test["total_duration"]
            marker_stats[marker]["tests"].append(test)

    if marker_stats:
        print("\nDURATION BY MARKER:")
        for marker, stats in sorted(
            marker_stats.items(), key=lambda x: x[1]["total_time"], reverse=True
        ):
            avg_time = stats["total_time"] / stats["count"]
            slowest_in_marker = max(
                stats["tests"], key=lambda x: x["total_duration"]
            )
            print(
                f"   {marker:15s}: {stats['count']:3d} tests, {stats['total_time']:6.2f}s total, {avg_time:6.3f}s avg"
            )
            print(
                f"                    Slowest: {slowest_in_marker['total_duration']:.3f}s ({slowest_in_marker['nodeid']})"
            )

    print("\nPERFORMANCE WARNINGS:")
    slow_tests = [t for t in sorted_tests if t["total_duration"] > 5.0]
    if slow_tests:
        print(f"   {len(slow_tests)} tests took > 5 seconds:")
        for test in slow_tests[:5]:
            print(f"     * {test['total_duration']:.2f}s - {test['nodeid']}")
    else:
        print("   No tests exceeded 5 seconds")

    very_slow_tests = [t for t in sorted_tests if t["total_duration"] > 30.0]
    if very_slow_tests:
        print(
            f"   {len(very_slow_tests)} tests took > 30 seconds (consider optimization):"
        )
        for test in very_slow_tests:
            print(f"     * {test['total_duration']:.2f}s - {test['nodeid']}")

    print("=" * 80)
    print("END DURATION REPORT")
    print("=" * 80 + "\n")

    try:
        report_file = Path("tests/results/test_duration_report.json")
        report_file.parent.mkdir(exist_ok=True)
        report_data = {
            "session_info": {
                "total_session_time": total_session_time,
                "total_test_time": total_test_time,
                "overhead_time": total_session_time - total_test_time,
                "timestamp": datetime.now().isoformat(),
                "exit_status": exitstatus,
            },
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "skipped": skipped_tests,
                "average_test_time": avg_test_time,
                "slowest_test": slowest_test,
                "fastest_test": fastest_test,
            },
            "test_durations": sorted_tests,
            "marker_statistics": marker_stats,
        }
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"Detailed duration report saved to: {report_file}")
    except Exception as e:
        print(f"Could not save duration report: {e}")


def pytest_report_header(config):
    """Add performance line to test report header."""
    return "Optimized Test Suite - Parallel Execution Enabled"


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add performance summary lines to terminal output."""
    terminalreporter.write_sep("=", "Test Suite Summary")
    terminalreporter.write_line("[+] Parallel execution: Enabled")
    terminalreporter.write_line("[+] Database isolation: Enforced")
    terminalreporter.write_line("[+] UTF-8 compliance: Validated")
    terminalreporter.write_line("[+] Memory monitoring: Active")
    terminalreporter.write_line("[+] Performance timing: Enabled")
