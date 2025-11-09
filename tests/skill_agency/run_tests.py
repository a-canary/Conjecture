#!/usr/bin/env python3
"""
Test runner script for skill-based agency tests.
Provides categorized test execution and reporting.
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path


def run_command(cmd, cwd=None, timeout=300):
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False
        )
        return result
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout} seconds: {cmd}")
        return None
    except Exception as e:
        print(f"Error running command: {e}")
        return None


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Run skill-based agency tests")
    parser.add_argument("--category", "-c", 
                       choices=["unit", "integration", "security", "performance", "edge_case", "all"],
                       default="all",
                       help="Test category to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--html", action="store_true", help="Generate HTML coverage report")
    parser.add_argument("--parallel", "-p", action="store_true", help="Run tests in parallel")
    parser.add_argument("--failfast", "-x", action="store_true", help="Stop on first failure")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke tests only")
    parser.add_argument("--timeout", "-t", type=int, default=300, help="Test timeout in seconds")
    
    args = parser.parse_args()
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    # Build pytest command
    pytest_cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        pytest_cmd.append("-vv")
    else:
        pytest_cmd.append("-v")
    
    # Add category filters
    if args.category != "all":
        pytest_cmd.extend(["-m", args.category])
    
    # Add smoke test filter
    if args.smoke:
        if args.category != "all":
            pytest_cmd[-1] = f"{args.category} or smoke"
        else:
            pytest_cmd.extend(["-m", "smoke"])
    
    # Add coverage options
    if args.coverage:
        coverage_args = [
            "--cov=src",
            "--cov=tests/skill_agency",
            "--cov-report=term-missing"
        ]
        if args.html:
            coverage_args.append("--cov-report=html")
            coverage_args.append("--cov-report=xml")
        pytest_cmd.extend(coverage_args)
    
    # Add parallel execution
    if args.parallel and not args.coverage:
        pytest_cmd.extend(["-n", "auto"])
    
    # Add failfast
    if args.failfast:
        pytest_cmd.append("--failfast")
    
    # Add timeout
    pytest_cmd.extend(["--timeout", str(args.timeout)])
    
    # Add test directory
    pytest_cmd.append(str(script_dir))
    
    # Print test configuration
    print("=" * 60)
    print("Skill-Based Agency Test Runner")
    print("=" * 60)
    print(f"Category: {args.category}")
    print(f"Verbose: {args.verbose}")
    print(f"Coverage: {args.coverage}")
    print(f"HTML Report: {args.html}")
    print(f"Parallel: {args.parallel}")
    print(f"Smoke Tests: {args.smoke}")
    print(f"Timeout: {args.timeout}s")
    print(f"Command: {' '.join(pytest_cmd)}")
    print("=" * 60)
    
    # Run tests
    result = run_command(" ".join(pytest_cmd), cwd=script_dir, timeout=args.timeout)
    
    if result is None:
        print("Test execution failed or timed out")
        return 1
    
    # Print results
    if args.verbose:
        print("\nSTDOUT:")
        print(result.stdout)
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
    
    # Check exit code
    if result.returncode == 0:
        print("\n✅ All tests passed!")
        if args.coverage and args.html:
            print(f"📊 HTML coverage report generated in {script_dir / 'htmlcov' / 'index.html'}")
    else:
        print(f"\n❌ Tests failed with exit code {result.returncode}")
        if not args.verbose:
            print("\nRun with --verbose for detailed output")
    
    return result.returncode


def run_category_tests():
    """Run tests by category with summary."""
    categories = ["unit", "integration", "security", "performance", "edge_case"]
    results = {}
    
    print("Running comprehensive test suite by category...")
    print("=" * 60)
    
    for category in categories:
        print(f"\n🧪 Running {category} tests...")
        
        cmd = f"python -m pytest -m {category} -v --tb=short"
        result = run_command(cmd, cwd=Path(__file__).parent, timeout=180)
        
        if result:
            results[category] = {
                "exit_code": result.returncode,
                "passed": "passed" in result.stdout.lower() or result.returncode == 0
            }
            
            if result.returncode == 0:
                print(f"✅ {category.title()} tests passed")
            else:
                print(f"❌ {category.title()} tests failed")
        else:
            results[category] = {"exit_code": -1, "passed": False}
            print(f"⚠️ {category.title()} tests timed out or crashed")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Suite Summary")
    print("=" * 60)
    
    total_passed = sum(1 for r in results.values() if r["passed"])
    total_failed = len(results) - total_passed
    
    for category, result in results.items():
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"{category.title():15s}: {status}")
    
    print("-" * 60)
    print(f"Total: {total_passed} passed, {total_failed} failed")
    
    if total_failed > 0:
        print("\n⚠️ Some test categories failed. Run with verbose output for details.")
        return 1
    else:
        print("\n🎉 All test categories passed!")
        return 0


if __name__ == "__main__":
    # Handle special cases
    if len(sys.argv) > 1 and sys.argv[1] == "--comprehensive":
        sys.exit(run_category_tests())
    else:
        sys.exit(main())