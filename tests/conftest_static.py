"""Pytest plugin: optional static-analysis tool execution.

Extracted from conftest.py. Registers four markers (ruff/mypy/vulture/bandit),
auto-marks tests whose nodeid matches a tool name, and -- when the
``--static-analysis`` flag is set or ``[pytest-static-analysis]`` exists
in pytest.ini with ``auto_discovery = true`` -- runs the matching tool
for each marked test and fails it on a non-zero return code.

Behavior parity: this block is byte-for-byte the same logic that was
inlined in conftest.py. No test currently opts in (no test uses these
markers, and no test name contains the tool names), so in practice
this plugin is a no-op for the current test suite. Kept as-is so the
optional integration still works for users who enable it.
"""

import subprocess
import configparser
from pathlib import Path
from typing import Any, Dict, List

import pytest


STATIC_ANALYSIS_CONFIG = {
    "ruff": {
        "enabled": True,
        "command": "ruff check . --format=json",
        "config_file": ".ruff.toml",
        "marker": "ruff",
    },
    "mypy": {
        "enabled": True,
        "command": "mypy src/ --json-report /tmp/mypy-report",
        "config_file": "mypy.ini",
        "marker": "mypy",
    },
    "vulture": {
        "enabled": True,
        "command": "vulture src/ tests/ --min-confidence 80 --format json",
        "config_file": "vulture.cfg",
        "marker": "vulture",
    },
    "bandit": {
        "enabled": True,
        "command": "bandit -r src/ -f json -o /tmp/bandit-report.json",
        "config_file": ".bandit",
        "marker": "bandit",
    },
}


def pytest_addoption(parser):
    parser.addoption(
        "--static-analysis",
        action="store_true",
        default=False,
        help="Enable static analysis tool execution for marked tests",
    )


def pytest_configure(config):
    """Initialize static analysis state on the config object."""
    config._static_analysis_results = {}
    config._static_analysis_enabled = config.getoption(
        "--static-analysis", default=False
    )

    pytest_ini_path = Path(__file__).parent.parent / "pytest.ini"
    if pytest_ini_path.exists():
        parser = configparser.ConfigParser()
        try:
            parser.read(pytest_ini_path, encoding="utf-8")
            if "pytest-static-analysis" in parser:
                config._static_analysis_auto_discovery = parser.getboolean(
                    "pytest-static-analysis", "auto_discovery", fallback=True
                )
            else:
                config._static_analysis_auto_discovery = True
        except (configparser.Error, UnicodeDecodeError):
            config._static_analysis_auto_discovery = False
    else:
        config._static_analysis_auto_discovery = False


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests whose nodeid/name matches a static-analysis tool.

    Skipped unless ``--static-analysis`` is set or auto-discovery is
    enabled in pytest.ini's ``[pytest-static-analysis]`` section.
    """
    if not (
        config.getoption("--static-analysis", default=False)
        or getattr(config, "_static_analysis_auto_discovery", False)
    ):
        return

    for item in items:
        if "static_analysis" in item.nodeid:
            item.add_marker(pytest.mark.static_analysis)
        for tool_name in STATIC_ANALYSIS_CONFIG:
            if tool_name in item.nodeid or tool_name in item.name:
                item.add_marker(getattr(pytest.mark, tool_name))
                item.add_marker(pytest.mark.static_analysis)


def pytest_report_header(config):
    """Add static-analysis line to test report header."""
    lines = ["Static Analysis Integration: Configured"]
    if getattr(config, "_static_analysis_enabled", False):
        lines.append("Static Analysis: Active")
    return lines


def pytest_runtest_setup(item):
    """Skip static-analysis tests whose tool is not on PATH."""
    if any(mark.name == "static_analysis" for mark in item.iter_markers()):
        for tool_name, tool_config in STATIC_ANALYSIS_CONFIG.items():
            if tool_config["enabled"] and any(
                mark.name == tool_name for mark in item.iter_markers()
            ):
                try:
                    subprocess.run(
                        [tool_name, "--version"],
                        capture_output=True,
                        check=True,
                        timeout=10,
                    )
                except (
                    subprocess.CalledProcessError,
                    FileNotFoundError,
                    subprocess.TimeoutExpired,
                ):
                    pytest.skip(f"Static analysis tool '{tool_name}' not available")


def pytest_runtest_call(item):
    """Run static analysis tools for marked tests, fail on non-zero exit."""
    if any(mark.name == "static_analysis" for mark in item.iter_markers()):
        for tool_name, tool_config in STATIC_ANALYSIS_CONFIG.items():
            if tool_config["enabled"] and any(
                mark.name == tool_name for mark in item.iter_markers()
            ):
                try:
                    result = subprocess.run(
                        tool_config["command"].split(),
                        capture_output=True,
                        text=True,
                        timeout=300,
                        cwd=Path(__file__).parent.parent,
                    )
                    if not hasattr(item.config, "_static_analysis_results"):
                        item.config._static_analysis_results = {}
                    item.config._static_analysis_results[tool_name] = {
                        "returncode": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "success": result.returncode == 0,
                    }
                    if result.returncode != 0:
                        pytest.fail(
                            f"{tool_name} static analysis failed:\n{result.stderr}"
                        )
                except subprocess.TimeoutExpired:
                    pytest.fail(
                        f"{tool_name} static analysis timed out after 300 seconds"
                    )
                except Exception as e:
                    pytest.fail(f"Error running {tool_name} static analysis: {str(e)}")


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add static-analysis summary section to terminal output."""
    if not getattr(config, "_static_analysis_results", None):
        return
    terminalreporter.write_sep("-", "Static Analysis Results")
    for tool_name, results in config._static_analysis_results.items():
        status = "[PASS]" if results["success"] else "[FAIL]"
        terminalreporter.write_line(f"• {tool_name}: {status}")
        if not results["success"] and results["stderr"]:
            error_lines = results["stderr"].strip().split("\n")[:3]
            for line in error_lines:
                terminalreporter.write_line(f"  {line}")
            if len(results["stderr"].strip().split("\n")) > 3:
                terminalreporter.write_line("  ... (truncated)")


# Static analysis fixtures
@pytest.fixture(scope="session")
def static_analysis_config():
    """Fixture providing static analysis configuration."""
    return STATIC_ANALYSIS_CONFIG.copy()


@pytest.fixture(scope="function")
def static_analysis_runner():
    """Fixture for running static analysis tools programmatically."""

    class StaticAnalysisRunner:
        def run_tool(
            self, tool_name: str, extra_args: List[str] = None
        ) -> Dict[str, Any]:
            """Run a specific static analysis tool."""
            if tool_name not in STATIC_ANALYSIS_CONFIG:
                raise ValueError(f"Unknown static analysis tool: {tool_name}")
            tool_config = STATIC_ANALYSIS_CONFIG[tool_name]
            if not tool_config["enabled"]:
                return {"success": False, "error": f"Tool {tool_name} is disabled"}
            command = tool_config["command"].split()
            if extra_args:
                command.extend(extra_args)
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=Path(__file__).parent.parent,
                )
                return {
                    "success": result.returncode == 0,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "command": " ".join(command),
                }
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "error": f"Tool {tool_name} timed out after 300 seconds",
                    "command": " ".join(command),
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Error running {tool_name}: {str(e)}",
                    "command": " ".join(command),
                }

        def run_all_tools(self) -> Dict[str, Dict[str, Any]]:
            """Run all enabled static analysis tools."""
            return {tool: self.run_tool(tool) for tool in STATIC_ANALYSIS_CONFIG}

    return StaticAnalysisRunner()
