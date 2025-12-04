#!/usr/bin/env python3
"""
Conjecture Usage Examples
Demonstrates how to use the simplified Conjecture system
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def example_1_basic_usage():
    """Example 1: Basic Conjecture usage"""
    print("=== Example 1: Basic Usage ===")

    try:
        from conjecture import Conjecture

        # Initialize Conjecture
        cf = Conjecture()

        # Process a simple request
        result = cf.process_request("Research machine learning basics")

        print(f"Skill used: {result.get('skill_used', 'Unknown')}")
        print(f"Tools executed: {len(result.get('tool_results', []))}")
        print("✅ Basic usage example completed")

    except Exception as e:
        print(f"❌ Error: {e}")


def example_2_emoji_system():
    """Example 2: Using the emoji system"""
    print("\n=== Example 2: Emoji System ===")

    try:
        from utils.terminal_emoji import success, error, warning, info, target

        # Basic emoji functions
        success("Operation completed successfully!")
        error("Something went wrong")
        warning("Configuration needs attention")
        info("Processing started...")
        target("Goal achieved!")

        print("✅ Emoji system example completed")

    except Exception as e:
        print(f"❌ Error: {e}")


def example_3_verbose_logging():
    """Example 3: Verbose logging with emojis"""
    print("\n=== Example 3: Verbose Logging ===")

    try:
        from utils.verbose_logger import VerboseLogger, VerboseLevel

        # Create logger
        logger = VerboseLogger(VerboseLevel.USER)

        # Log various events
        logger.claim_assessed_confident("c0000001", 0.9, 0.8)
        logger.claim_needs_evaluation("c0000002", 0.6, 0.8)
        logger.claim_resolved("c0000001", 0.9)
        logger.user_message("Hello from user!")
        logger.final_response("Here is the response with emoji support!")

        print("✅ Verbose logging example completed")

    except Exception as e:
        print(f"❌ Error: {e}")


def example_4_core_tools():
    """Example 4: Using core tools directly"""
    print("\n=== Example 4: Core Tools ===")

    try:
        from core_tools.registry import get_tool_registry

        # Get tool registry
        registry = get_tool_registry()

        # List available tools
        tools_info = registry.get_tools_info()
        print(f"Available tools: {len(tools_info)}")

        # Execute a simple tool
        result = registry.execute_tool(
            "Reason", {"thought_process": "Testing the core tools system"}
        )

        if result.get("success"):
            print("✅ Core tools example completed")
        else:
            print(f"❌ Tool execution failed: {result.get('error')}")

    except Exception as e:
        print(f"❌ Error: {e}")


def example_5_context_builder():
    """Example 5: Building context with tools"""
    print("\n=== Example 5: Context Builder ===")

    try:
        from context.complete_context_builder import CompleteContextBuilder
        from core.models import Claim, ClaimState, ClaimType

        # Create sample claims
        claims = [
            Claim(
                id="c0000001",
                content="Python is a popular programming language",
                confidence=0.9,
                state=ClaimState.VALIDATED,
                type=[ClaimType.FACT],
                tags=["python", "programming"],
            ),
            Claim(
                id="c0000002",
                content="Machine learning often uses Python",
                confidence=0.8,
                state=ClaimState.EXPLORE,
                type=[ClaimType.CONCEPT],
                tags=["ml", "python"],
            ),
        ]

        # Build context
        builder = CompleteContextBuilder(claims)
        context = builder.build_simple_context(include_core_tools=True)

        print(f"Context built with {len(context)} characters")
        print("✅ Context builder example completed")

    except Exception as e:
        print(f"❌ Error: {e}")


def example_6_simple_workflows():
    """Example 6: Simple workflow demonstrations"""
    print("\n=== Example 6: Simple Workflows ===")

    workflows = [
        {
            "name": "Research Workflow",
            "description": "Search → Read → Create Claim",
            "tools": ["WebSearch", "ReadFiles", "ClaimCreate"],
        },
        {
            "name": "Code Development",
            "description": "Read → Write Code → Test",
            "tools": ["ReadFiles", "WriteCodeFile", "ClaimCreate"],
        },
        {
            "name": "Validation",
            "description": "Search → Validate → Document",
            "tools": ["WebSearch", "ClaimCreate", "ClaimAddSupport"],
        },
    ]

    for workflow in workflows:
        print(f"🔄 {workflow['name']}: {workflow['description']}")
        print(f"   Tools: {', '.join(workflow['tools'])}")

    print("✅ Workflow examples completed")


def main():
    """Run all examples"""
    print("Conjecture Usage Examples")
    print("=" * 50)

    examples = [
        example_1_basic_usage,
        example_2_emoji_system,
        example_3_verbose_logging,
        example_4_core_tools,
        example_5_context_builder,
        example_6_simple_workflows,
    ]

    for example in examples:
        try:
            example()
        except KeyboardInterrupt:
            print("\nExample interrupted")
        except Exception as e:
            print(f"Unexpected error: {e}")

    print("\nAll examples completed!")
    print("\nFor more information:")
    print("   • README.md - Project overview")
    print("   • EMOJI_USAGE.md - Emoji system documentation")
    print("   • CORE_TOOLS_IMPLEMENTATION_SUMMARY.md - Tools documentation")


if __name__ == "__main__":
    main()
