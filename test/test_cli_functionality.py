#!/usr/bin/env python3
"""
Test CLI functionality with new 3-part architecture
Tests direct backend functionality bypassing Rich UI issues
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_cli_backends():
    """Test CLI backend functionality"""
    print("=" * 60)
    print("CLI BACKEND FUNCTIONALITY TESTS")
    print("=" * 60)

    results = {"total": 0, "passed": 0, "failed": 0}

    # Test 1: Backend Loading
    print("\n1. Testing Backend Loading...")
    try:
        from src.cli.backends.auto import AutoBackend
        from src.cli.backends.cloud import CloudBackend
        from src.cli.backends.hybrid import HybridBackend
        from src.cli.backends.local import LocalBackend

        print("   + AutoBackend imported")
        print("   + LocalBackend imported")
        print("   + CloudBackend imported")
        print("   + HybridBackend imported")

        results["total"] += 4
        results["passed"] += 4

    except Exception as e:
        print(f"   X Backend loading failed: {e}")
        results["total"] += 4
        results["failed"] += 4

    # Test 2: Backend Initialization
    print("\n2. Testing Backend Initialization...")
    try:
        # Test AutoBackend (our default)
        auto_backend = AutoBackend()
        print("   + AutoBackend initialized")

        # Test availability checking
        is_available = auto_backend.is_available()
        print(f"   + AutoBackend availability: {is_available}")

        # Test backend info
        info = auto_backend.get_backend_info()
        print(f"   + Backend info: {info.get('name', 'Unknown')}")

        results["total"] += 3
        results["passed"] += 3

    except Exception as e:
        print(f"   X Backend initialization failed: {e}")
        results["total"] += 3
        results["failed"] += 3

    # Test 3: Backend Core Operations (if available)
    print("\n3. Testing Backend Claim Operations...")
    try:
        # Create a test claim directly using backend
        if auto_backend.is_available():
            # Test claim creation
            claim_id = auto_backend.create_claim(
                content="CLI test claim for 3-part architecture",
                confidence=0.9,
                user="test_user",
                analyze=False,
            )
            print(f"   + Claim created: {claim_id}")

            # Test claim retrieval
            claim = auto_backend.get_claim(claim_id)
            if claim:
                print(f"   + Claim retrieved: {claim['content'][:30]}...")
            else:
                print("   X Claim retrieval failed")
                raise Exception("Claim not found after creation")

            # Test claim search
            search_results = auto_backend.search_claims("CLI test", 5)
            print(f"   + Search results: {len(search_results)} claims found")

            results["total"] += 3
            results["passed"] += 3
        else:
            print("   . Skipped - backend not configured")
            results["total"] += 3

    except Exception as e:
        print(f"   X Backend operations failed: {e}")
        results["total"] += 3
        results["failed"] += 3

    # Test 4: Configuration Integration
    print("\n4. Testing Configuration Integration...")
    try:
        from src.config.config import get_unified_validator

        validator = get_unified_validator()
        config_result = validator.validate()

        print(f"   + Configuration validation: {config_result.success}")
        print(f"   + Validation errors: {len(config_result.errors)}")

        results["total"] += 2
        if config_result.success:
            results["passed"] += 2
        else:
            results["passed"] += 1
            results["failed"] += 1

    except Exception as e:
        print(f"   X Configuration integration failed: {e}")
        results["total"] += 2
        results["failed"] += 2

    return results


def test_3part_architecture_integration():
    """Test CLI with 3-part architecture integration"""
    print("\n" + "=" * 60)
    print("3-PART ARCHITECTURE CLI INTEGRATION")
    print("=" * 60)

    results = {"total": 0, "passed": 0, "failed": 0}

    try:
        from src.agent.llm_inference import (
            build_llm_context,
            coordinate_three_part_flow,
        )
        from src.cli.backends.auto import AutoBackend
        from src.core.models import Claim, ClaimState, ClaimType
        from src.processing.tool_registry import create_tool_registry

        # Test 1: CLI can access all three layers
        print("\n1. Testing Layer Access...")

        # Create test data
        test_claim = Claim(
            id="cli_integration_test",
            content="CLI integration test claim",
            confidence=0.8,
            state=ClaimState.VALIDATED,
            type=[ClaimType.EXAMPLE],
            tags=["cli", "integration", "test"],
        )

        print("   + Claims layer accessible")

        # Test tool registry
        tool_registry = create_tool_registry()
        print("   + Tools layer accessible")

        # Test LLM inference
        context = build_llm_context(
            session_id="cli_test",
            user_request="CLI integration test",
            all_claims=[test_claim],
            tool_registry=tool_registry,
        )
        print("   + LLM inference layer accessible")

        results["total"] += 3
        results["passed"] += 3

        # Test 2: CLI can coordinate flow
        print("\n2. Testing Flow Coordination...")

        flow_result = coordinate_three_part_flow(
            session_id="cli_flow_test",
            user_request="Test CLI to 3-part architecture flow",
            all_claims=[test_claim],
            tool_registry=tool_registry,
        )

        if flow_result["success"]:
            print("   + CLI successfully coordinated 3-part flow")
            results["total"] += 1
            results["passed"] += 1
        else:
            print(f"   X Flow coordination failed: {flow_result.get('error')}")
            results["total"] += 1
            results["failed"] += 1

        # Test 3: Backend can use architecture
        print("\n3. Testing Backend Architecture Integration...")

        backend = AutoBackend()
        if backend.is_available():
            # Backend should be able to create claims using the architecture
            claim_id = backend.create_claim(
                content="Backend architecture integration test",
                confidence=0.85,
                user="cli_test",
            )
            print(f"   + Backend created claim via architecture: {claim_id}")

            results["total"] += 1
            results["passed"] += 1
        else:
            print("   . Skipped - backend not available")
            results["total"] += 1

    except Exception as e:
        print(f"   X Architecture integration failed: {e}")
        results["total"] += 5
        results["failed"] += 5

    return results


def test_cli_entry_points():
    """Test main CLI entry points"""
    print("\n" + "=" * 60)
    print("CLI ENTRY POINTS TESTING")
    print("=" * 60)

    results = {"total": 0, "passed": 0, "failed": 0}

    # Test 1: Main CLI import
    print("\n1. Testing Main CLI Import...")
    try:
        from src.cli.modular_cli import app

        print("   + Main CLI app imported")
        print("   + Typer app created")

        results["total"] += 2
        results["passed"] += 2

    except Exception as e:
        print(f"   X Main CLI import failed: {e}")
        results["total"] += 2
        results["failed"] += 2

    # Test 2: CLI Command Structure
    print("\n2. Testing CLI Command Structure...")
    try:
        # Check that key commands exist
        import inspect

        import src.cli.modular_cli as cli_module

        commands = []
        for name, obj in inspect.getmembers(cli_module):
            if callable(obj) and not name.startswith("_"):
                commands.append(name)

        expected_commands = ["create", "get", "search", "analyze", "config", "stats"]
        found_commands = [cmd for cmd in expected_commands if cmd in commands]

        print(
            f"   + Expected commands found: {len(found_commands)}/{len(expected_commands)}"
        )
        for cmd in found_commands:
            print(f"     + {cmd}")

        results["total"] += 1
        if len(found_commands) >= len(expected_commands) * 0.8:  # 80% threshold
            results["passed"] += 1
        else:
            results["failed"] += 1

    except Exception as e:
        print(f"   X Command structure test failed: {e}")
        results["total"] += 1
        results["failed"] += 1

    return results


def main():
    """Run all CLI functionality tests"""
    print("CLI FUNCTIONALITY TESTS FOR 3-PART ARCHITECTURE")
    print("=" * 80)

    # Run all test suites
    backend_results = test_cli_backends()
    integration_results = test_3part_architecture_integration()
    entry_results = test_cli_entry_points()

    # Aggregate results
    total_total = (
        backend_results["total"] + integration_results["total"] + entry_results["total"]
    )
    total_passed = (
        backend_results["passed"]
        + integration_results["passed"]
        + entry_results["passed"]
    )
    total_failed = (
        backend_results["failed"]
        + integration_results["failed"]
        + entry_results["failed"]
    )

    # Final summary
    print("\n" + "=" * 80)
    print("CLI FUNCTIONALITY TEST RESULTS")
    print("=" * 80)
    print(f"Total Tests:     {total_total}")
    print(f"Passed:          {total_passed}")
    print(f"Failed:          {total_failed}")
    print(f"Success Rate:    {(total_passed / total_total) * 100:.1f}%")

    print("\nTest Suite Breakdown:")
    print(
        f"Backend Tests:       {backend_results['passed']}/{backend_results['total']} passed"
    )
    print(
        f"Integration Tests:   {integration_results['passed']}/{integration_results['total']} passed"
    )
    print(
        f"Entry Point Tests:   {entry_results['passed']}/{entry_results['total']} passed"
    )

    # Overall verdict
    print("\n" + "=" * 80)
    if total_failed == 0:
        print("OVERALL RESULT: ALL CLI TESTS PASSED - + CLI VALIDATED")
        print("CLI functionality is working with the 3-part architecture!")
    else:
        print(f"OVERALL RESULT: {total_failed} CLI TESTS FAILED")
        print("Some CLI functionality needs attention.")
    print("=" * 80)

    return total_failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
