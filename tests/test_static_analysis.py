import subprocess
import sys
from pathlib import Path
import pytest

@pytest.mark.static_analysis
def test_ruff_check():
    """Run ruff linter and ensure it passes."""
    result = subprocess.run(
        ["ruff", "check", "."],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=Path(__file__).parent.parent
    )
    
    if result.returncode != 0:
        print("Ruff check failed:")
        print(result.stdout or "")
        print(result.stderr or "")
        pytest.fail(f"Ruff check failed with exit code {result.returncode}")
    
    assert result.returncode == 0, "Ruff check should pass"

@pytest.mark.static_analysis
def test_mypy_check():
    """Run mypy type checker and ensure it passes."""
    result = subprocess.run(
        ["mypy", "."],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=Path(__file__).parent.parent
    )
    
    if result.returncode != 0:
        print("Mypy check failed:")
        print(result.stdout or "")
        print(result.stderr or "")
        pytest.fail(f"Mypy check failed with exit code {result.returncode}")
    
    assert result.returncode == 0, "Mypy check should pass"

@pytest.mark.static_analysis
def test_vulture_check():
    """Run vulture dead code finder and ensure it passes."""
    result = subprocess.run(
        ["vulture", "."],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=Path(__file__).parent.parent
    )
    
    if result.returncode != 0:
        print("Vulture check failed:")
        print(result.stdout or "")
        print(result.stderr or "")
        pytest.fail(f"Vulture check failed with exit code {result.returncode}")
    
    assert result.returncode == 0, "Vulture check should pass"

@pytest.mark.static_analysis
def test_bandit_check():
    """Run bandit security scanner and ensure it passes."""
    result = subprocess.run(
        ["bandit", "-r", "src"],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=Path(__file__).parent.parent
    )
    
    if result.returncode != 0:
        print("Bandit check failed:")
        print(result.stdout or "")
        print(result.stderr or "")
        pytest.fail(f"Bandit check failed with exit code {result.returncode}")
    
    assert result.returncode == 0, "Bandit check should pass"

@pytest.mark.static_analysis
def test_static_analysis_comprehensive():
    """
    Comprehensive test that runs all static analysis tools together.
    This provides a single point of failure for all static analysis checks.
    """
    root_dir = Path(__file__).parent.parent
    
    tools = [
        (["ruff", "check", "."], "Ruff"),
        (["mypy", "."], "Mypy"),
        (["vulture", "."], "Vulture"),
        (["bandit", "-r", "src"], "Bandit")
    ]
    
    failed_tools = []
    
    for command, name in tools:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            cwd=root_dir
        )
        
        if result.returncode != 0:
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            output = stdout[:200] if stdout else stderr[:200] if stderr else "No output"
            failed_tools.append(f"{name}: {output}...")
    
    if failed_tools:
        pytest.fail(f"Static analysis failed for: {'; '.join(failed_tools)}")
