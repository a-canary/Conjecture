#!/usr/bin/env python3
"""
Test runner script for the Conjecture data layer test suite.
Provides convenient ways to run different test categories and generate reports.
"""
import sys
import os
import argparse
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False


def check_environment():
    """Check if the test environment is properly set up."""
    print("Checking test environment...")
    
    # Check Python path
    project_root = Path(__file__).parent.parent
    src_path = project_root / "src"
    
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
        print(f"Added {src_path} to Python path")
    
    # Check if pytest is available
    try:
        import pytest
        print(f"✅ pytest version {pytest.__version__} found")
    except ImportError:
        print("❌ pytest not found. Please install it with: pip install pytest")
        return False
    
    return True


def install_dependencies():
    """Install test dependencies if needed."""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if requirements_file.exists():
        print("Installing test dependencies...")
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
        return run_command(cmd, "Installing dependencies")
    else:
        print("Requirements file not found, skipping dependency installation")
        return True


def run_unit_tests():
    """Run unit tests only."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-m", "unit",
        "-v",
        "--tb=short"
    ]
    return run_command(cmd, "Unit Tests")


def run_integration_tests():
    """Run integration tests only."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-m", "integration",
        "-v",
        "--tb=short"
    ]
    return run_command(cmd, "Integration Tests")


def run_performance_tests():
    """Run performance tests with benchmarks."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-m", "performance",
        "-v",
        "--tb=short"
    ]
    return run_command(cmd, "Performance Tests")


def run_error_handling_tests():
    """Run error handling and edge case tests."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-m", "error_handling",
        "-v",
        "--tb=short"
    ]
    return run_command(cmd, "Error Handling Tests")


def run_all_tests():
    """Run all tests with coverage."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--cov=src/data",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "--cov-fail-under=80"
    ]
    return run_command(cmd, "All Tests with Coverage")


def run_quick_tests():
    """Run quick tests (unit tests only, no coverage)."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-m", "unit and not slow",
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure
    ]
    return run_command(cmd, "Quick Tests")


def run_coverage_only():
    """Generate coverage report only."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "--cov=src/data",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov"
    ]
    return run_command(cmd, "Coverage Report")


def run_parallel_tests():
    """Run tests in parallel (requires pytest-xdist)."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-n", "auto",
        "-v",
        "--tb=short"
    ]
    return run_command(cmd, "Parallel Tests")


def run_smoke_tests():
    """Run critical smoke tests only."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_models.py::TestClaimModel::test_valid_claim_creation",
        "tests/test_sqlite_manager.py::TestSQLiteManagerClaimsCRUD::test_create_claim",
        "tests/test_chroma_manager.py::TestChromaManagerEmbeddingOperations::test_add_embedding",
        "tests/test_data_manager_integration.py::TestDataManagerClaimCRUD::test_create_claim_complete_workflow",
        "-v"
    ]
    return run_command(cmd, "Smoke Tests")


def run_specific_file(filename):
    """Run tests from a specific file."""
    if not filename.startswith("test_"):
        filename = f"test_{filename}"
    if not filename.endswith(".py"):
        filename = f"{filename}.py"
    
    filepath = Path("tests") / filename
    if not filepath.exists():
        print(f"❌ Test file {filepath} not found")
        return False
    
    cmd = [
        sys.executable, "-m", "pytest",
        str(filepath),
        "-v",
        "--tb=short"
    ]
    return run_command(cmd, f"Tests from {filename}")


def run_specific_marker(marker):
    """Run tests with a specific marker."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-m", marker,
        "-v",
        "--tb=short"
    ]
    return run_command(cmd, f"Tests with marker {marker}")


def list_available_markers():
    """List all available pytest markers."""
    cmd = [sys.executable, "-m", "pytest", "--markers"]
    return run_command(cmd, "Available Markers")


def generate_test_report():
    """Generate a comprehensive test report."""
    print("Generating comprehensive test report...")
    
    # Run tests with multiple report formats
    base_cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "--cov=src/data",
        "--cov-report=html:htmlcov",
        "--cov-report=xml",
        "--junit-xml=test-results.xml",
        "-v"
    ]
    
    success = run_command(base_cmd, "Comprehensive Test Report")
    
    if success:
        print("\n📊 Test Report Generated:")
        print("  - HTML Coverage: htmlcov/index.html")
        print("  - XML Coverage: coverage.xml")
        print("  - Test Results: test-results.xml")
    
    return success


def cleanup_test_artifacts():
    """Clean up test artifacts and temporary files."""
    print("Cleaning up test artifacts...")
    
    artifacts = [
        ".pytest_cache",
        "__pycache__",
        "htmlcov",
        ".coverage",
        "coverage.xml",
        "test-results.xml",
        "test_*.db",
        "chroma_*",
        "logs"
    ]
    
    import glob
    import shutil
    
    cleaned = 0
    for pattern in artifacts:
        for path in glob.glob(pattern):
            try:
                if os.path.isfile(path):
                    os.remove(path)
                    print(f"  Removed file: {path}")
                elif os.path.isdir(path):
                    shutil.rmtree(path)
                    print(f"  Removed directory: {path}")
                cleaned += 1
            except Exception as e:
                print(f"  Could not remove {path}: {e}")
    
    print(f"✅ Cleaned up {cleaned} artifacts")
    return True


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(
        description="Conjecture Data Layer Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Test type options
    parser.add_argument(
        "command",
        nargs="?",
        choices=[
            "all", "unit", "integration", "performance", "error",
            "quick", "coverage", "parallel", "smoke", "clean",
            "report", "markers"
        ],
        help="Test category to run"
    )
    
    # Specific test options
    parser.add_argument(
        "--file", "-f",
        help="Run tests from specific file (e.g., models, sqlite_manager)"
    )
    
    parser.add_argument(
        "--marker", "-m",
        help="Run tests with specific marker"
    )
    
    # Configuration options
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install test dependencies before running"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Install dependencies if requested
    if args.install_deps:
        if not install_dependencies():
            sys.exit(1)
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Execute the requested command
    success = True
    
    if args.command == "all" or not args.command:
        success = run_all_tests()
    elif args.command == "unit":
        success = run_unit_tests()
    elif args.command == "integration":
        success = run_integration_tests()
    elif args.command == "performance":
        success = run_performance_tests()
    elif args.command == "error":
        success = run_error_handling_tests()
    elif args.command == "quick":
        success = run_quick_tests()
    elif args.command == "coverage":
        success = run_coverage_only()
    elif args.command == "parallel":
        success = run_parallel_tests()
    elif args.command == "smoke":
        success = run_smoke_tests()
    elif args.command == "clean":
        success = cleanup_test_artifacts()
    elif args.command == "report":
        success = generate_test_report()
    elif args.command == "markers":
        success = list_available_markers()
    
    # Handle specific file or marker
    elif args.file:
        success = run_specific_file(args.file)
    elif args.marker:
        success = run_specific_marker(args.marker)
    
    else:
        parser.print_help()
        sys.exit(1)
    
    # Exit with appropriate code
    if success:
        print("\n✅ Test execution completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Test execution failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()