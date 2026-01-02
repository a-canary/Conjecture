#!/usr/bin/env python
"""
==============================================================================
GENERATED CODE - SC-FEAT-001 - TEST BRANCH
Quick-start script for SWE-Bench Bash-Only Evaluator

Modified 2025-12-30:
  - Added Docker sandbox support
  - Added sandbox control options
  - All generated code marked for test branch
  - Added argparse for non-interactive CLI execution (2025-12-31)
==============================================================================
Run immediately with: python run_bash_only_evaluator.py --quick
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.benchmarking.swe_bench_bash_only_evaluator import (
    BashOnlySWEBenchEvaluator,
    EvaluationResult,
)


async def run_quick_evaluation(
    use_sandbox: bool = True, docker_image: str = "ubuntu:22.04"
):
    """
    SC-FEAT-001: Run quick evaluation with 10 tasks.

    Args:
        use_sandbox: Whether to use Docker sandbox
        docker_image: Docker image to use for sandbox
    """
    print("\n" + "=" * 70)
    print("[ROCKET] SWE-Bench Bash-Only Evaluator - Quick Start")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # SC-FEAT-001: Pass sandbox options to evaluator
    evaluator = BashOnlySWEBenchEvaluator(
        max_iterations=5,
        use_sandbox=use_sandbox,
        docker_image=docker_image,
    )

    try:
        # Initialize Conjecture
        print("[PACKAGE] Initializing Conjecture system...")
        await evaluator.initialize_conjecture()
        print("[OK] Conjecture initialized\n")

        # Load tasks
        print("[INBOX] Loading SWE-bench tasks...")
        tasks = await evaluator.load_swe_tasks(num_tasks=500)
        print(f"[OK] Loaded {len(tasks)} tasks\n")

        # Run evaluation
        print("[MICROSCOPE] Starting evaluation (first 10 tasks)...")
        print("-" * 70)
        results = await evaluator.evaluate_batch(tasks, batch_size=10)

        # Print summary
        print("\n" + "=" * 70)
        print("[CHART] EVALUATION SUMMARY")
        print("=" * 70)

        summary = results["summary"]
        print(f"Total Tasks Evaluated: {summary['total']}")
        print(f"Passed: {summary['passed']} [OK]")
        print(f"Failed: {summary['failed']} [FAIL]")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Average Execution Time: {summary['average_time']:.2f}s")
        print(f"Total ReAct Iterations: {summary['total_react_iterations']}")
        print(f"Average Iterations per Task: {summary['average_iterations']:.1f}")

        # Print detailed results
        print("\n" + "=" * 70)
        print("[LIST] DETAILED RESULTS")
        print("=" * 70)

        for i, result in enumerate(results["results"], 1):
            status = "[OK]" if result["success"] else "[FAIL]"
            print(
                f"{i:2d}. {result['task_id']:<40} {status} "
                f"({result['execution_time']:6.2f}s, {result['react_iterations']} iter)"
            )

        # Get statistics
        stats = evaluator.get_statistics()
        print("\n" + "=" * 70)
        print("[GRAPH] STATISTICS")
        print("=" * 70)
        print(f"Evaluations Completed: {stats['evaluations_completed']}")
        print(f"Successful Evaluations: {stats['successful_evaluations']}")
        print(f"Total Execution Time: {stats['total_execution_time']:.2f}s")
        print(f"Average Time per Evaluation: {stats['average_execution_time']:.2f}s")
        print(f"Success Rate: {stats['success_rate']:.1f}%")
        print(f"Total ReAct Iterations: {stats['total_react_iterations']}")
        print(f"Average Iterations: {stats['average_react_iterations']:.1f}")

        # Save results
        output_file = Path("swe_bench_bash_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[SAVE] Results saved to {output_file}")

        # Print configuration info
        sandbox_health = evaluator.sandbox_executor.health_check()
        print("\n" + "=" * 70)
        print("[CONFIG] CONFIGURATION")
        print("=" * 70)
        print(
            f"Sandbox: {'ENABLED' if sandbox_health['sandbox_enabled'] and sandbox_health['docker_available'] else 'DISABLED (direct execution)'}"
        )
        if sandbox_health["docker_available"]:
            print(f"Docker Image: {sandbox_health['docker_image']}")
        print("Temperature: 0.0 (deterministic)")
        print("Max Iterations: 5")
        print("Command Timeout: 30 seconds")
        print("Context Budget: <500 tokens")
        print("Model: GraniteTiny optimized")
        print("Dataset: SWE-bench-lite (HuggingFace)")

        print("\n" + "=" * 70)
        print(
            f"[OK] Evaluation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print("=" * 70 + "\n")

        return results

    except KeyboardInterrupt:
        print("\n\n[WARN] Evaluation interrupted by user")
        return None

    except Exception as e:
        print(f"\n[ERROR] Error during evaluation: {e}")
        import traceback

        traceback.print_exc()
        return None

    finally:
        print("[CLEANUP] Cleaning up resources...")
        await evaluator.cleanup()
        print("[OK] Cleanup complete")


async def run_full_evaluation(
    use_sandbox: bool = True, docker_image: str = "ubuntu:22.04"
):
    """
    SC-FEAT-001: Run full evaluation with 500 tasks (production).

    Args:
        use_sandbox: Whether to use Docker sandbox
        docker_image: Docker image to use for sandbox
    """
    print("\n" + "=" * 70)
    print("[ROCKET] SWE-Bench Bash-Only Evaluator - Full Evaluation")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("[WARN] This will take 30-60 minutes depending on system performance")
    print()

    # SC-FEAT-001: Pass sandbox options to evaluator
    evaluator = BashOnlySWEBenchEvaluator(
        max_iterations=5,
        use_sandbox=use_sandbox,
        docker_image=docker_image,
    )

    try:
        await evaluator.initialize_conjecture()
        tasks = await evaluator.load_swe_tasks(num_tasks=500)

        print(f"[MICROSCOPE] Starting full evaluation ({len(tasks)} tasks)...")
        print("This will be run in batches of 50 tasks\n")

        all_results = []
        total_passed = 0
        total_failed = 0

        # Process in batches
        for batch_start in range(0, len(tasks), 50):
            batch_end = min(batch_start + 50, len(tasks))
            batch_num = (batch_start // 50) + 1
            total_batches = (len(tasks) + 49) // 50

            print(
                f"[PACKAGE] Batch {batch_num}/{total_batches} (tasks {batch_start + 1}-{batch_end})..."
            )

            batch_tasks = tasks[batch_start:batch_end]
            batch_results = await evaluator.evaluate_batch(batch_tasks, batch_size=50)

            all_results.append(batch_results)
            total_passed += batch_results["summary"]["passed"]
            total_failed += batch_results["summary"]["failed"]

            print(
                f"   [OK] Batch complete: {batch_results['summary']['passed']} passed, "
                f"{batch_results['summary']['failed']} failed\n"
            )

        # Aggregate results
        print("\n" + "=" * 70)
        print("[CHART] FULL EVALUATION SUMMARY")
        print("=" * 70)
        print(f"Total Tasks: {len(tasks)}")
        print(f"Total Passed: {total_passed}")
        print(f"Total Failed: {total_failed}")
        print(f"Overall Success Rate: {(total_passed / len(tasks) * 100):.1f}%")

        # Save comprehensive results
        output_file = Path("swe_bench_bash_full_results.json")
        with open(output_file, "w") as f:
            json.dump(
                {
                    "batches": all_results,
                    "summary": {
                        "total_tasks": len(tasks),
                        "total_passed": total_passed,
                        "total_failed": total_failed,
                        "success_rate": (total_passed / len(tasks) * 100),
                    },
                },
                f,
                indent=2,
            )

        print(f"\n[SAVE] Results saved to {output_file}")
        print(
            f"[OK] Full evaluation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

    finally:
        await evaluator.cleanup()


def main():
    """
    SC-FEAT-001: Main entry point with sandbox options.

    Generated for SC-FEAT-001 test branch.
    Supports both interactive and command-line execution.
    """
    import argparse

    # SC-FEAT-001: Parse command-line arguments first (non-interactive mode)
    parser = argparse.ArgumentParser(
        description="SWE-Bench Bash-Only Evaluator", add_help=False
    )
    parser.add_argument(
        "--use-sandbox",
        action="store_true",
        default=None,
        help="Enable Docker sandbox (default: yes)",
    )
    parser.add_argument(
        "--no-sandbox",
        action="store_true",
        help="Disable Docker sandbox, use direct execution",
    )
    parser.add_argument(
        "--docker-image",
        type=str,
        default="ubuntu:22.04",
        help="Docker image to use (default: ubuntu:22.04)",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick evaluation (10 tasks)"
    )
    parser.add_argument(
        "--full", action="store_true", help="Run full evaluation (500 tasks)"
    )
    parser.add_argument("-h", "--help", action="store_true", help="Show help message")

    # Parse known args, ignore unknown ones
    args, _ = parser.parse_known_args()

    # Handle help
    if args.help:
        parser.print_help()
        sys.exit(0)

    # Determine if we're in interactive or CLI mode
    # CLI mode if any of --quick, --full, --use-sandbox, --no-sandbox specified
    cli_mode = args.quick or args.full or args.use_sandbox or args.no_sandbox

    # Parse sandbox settings
    if args.no_sandbox:
        use_sandbox = False
    elif args.use_sandbox:
        use_sandbox = True
    else:
        use_sandbox = None  # Will prompt in interactive mode

    docker_image = args.docker_image

    # Interactive mode (default, no CLI args)
    if not cli_mode:
        print("\n[TARGET] SWE-Bench Bash-Only Evaluator")
        print("=" * 70)
        print("\nOptions:")
        print("  1. Quick evaluation (10 tasks) - ~2-3 minutes")
        print("  2. Full evaluation (500 tasks) - ~30-60 minutes")
        print("  3. Exit")
        print()

        # SC-FEAT-001: Sandbox options
        if use_sandbox is None:
            sandbox_choice = (
                input("Use Docker sandbox? (recommended) [Y/n]: ").strip().lower()
            )
            use_sandbox = sandbox_choice != "n"

        if use_sandbox:
            docker_input = input(f"Docker image [{docker_image}]: ").strip()
            if docker_input:
                docker_image = docker_input
        else:
            print("[WARN] Direct execution mode - commands will run on HOST system!")

        choice = input("Select option (1-3): ").strip()

        if choice == "1":
            asyncio.run(
                run_quick_evaluation(use_sandbox=use_sandbox, docker_image=docker_image)
            )
        elif choice == "2":
            confirm = (
                input(
                    "\n[WARN] Full evaluation will take 30-60 minutes. Continue? (y/n): "
                )
                .strip()
                .lower()
            )
            if confirm == "y":
                asyncio.run(
                    run_full_evaluation(
                        use_sandbox=use_sandbox, docker_image=docker_image
                    )
                )
            else:
                print("Cancelled.")
        elif choice == "3":
            print("Exiting.")
        else:
            print("Invalid option.")
    else:
        # CLI mode: non-interactive execution
        if use_sandbox is None:
            use_sandbox = True  # Default to sandbox in CLI mode

        if not use_sandbox:
            print("[WARN] Direct execution mode - commands will run on HOST system!")

        if args.quick:
            print("\n[TARGET] SWE-Bench Bash-Only Evaluator (Quick Mode)")
            print("=" * 70)
            asyncio.run(
                run_quick_evaluation(use_sandbox=use_sandbox, docker_image=docker_image)
            )
        elif args.full:
            print("\n[TARGET] SWE-Bench Bash-Only Evaluator (Full Mode)")
            print("=" * 70)
            asyncio.run(
                run_full_evaluation(use_sandbox=use_sandbox, docker_image=docker_image)
            )
        else:
            print("Error: Must specify --quick or --full")
            parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARN] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
