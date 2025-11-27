#!/usr/bin/env python3
"""
Simplified Conjecture - Evidence-Based AI Reasoning System
Main entry point for the simplified architecture

Usage:
    python simple_conjecture.py
"""

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.conjecture import Conjecture


def main():
    """Main entry point for Simplified Conjecture"""
    print("Simplified Conjecture - Evidence-Based AI Reasoning")
    print("=" * 60)
    print("Features:")
    print("  • 90% of functionality with 10% of the complexity")
    print("  • Research, Code, Test, and Evaluate workflows")
    print("  • Tool integration (WebSearch, ReadFiles, WriteCodeFile)")
    print("  • Skill-based guidance (4-step processes)")
    print("  • Basic claim management and persistence")
    print()

    # Initialize Conjecture
    cf = Conjecture()

    # Show statistics
    stats = cf.get_statistics()
    print(f"System ready:")
    print(f"  • Available tools: {stats['available_tools']}")
    print(f"  • Available skills: {stats['available_skills']}")
    print(f"  • Total claims: {stats['total_claims']}")
    print()

    # Interactive demo
    print("Interactive Demo")
    print("-" * 30)
    print("Try these commands:")
    print("  • Research machine learning")
    print("  • Write Python code for data analysis")
    print("  • Test the application")
    print("  • Evaluate system performance")
    print("  • 'quit' to exit")
    print()

    while True:
        try:
            user_input = input(">>> ").strip()

if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not user_input:
                continue

            print(f"\n🔍 Processing: '{user_input}'")

            # Process the request
            result = cf.process_request(user_input)

            if result["success"]:
                print(f"✅ Completed using {result['skill_used']} skill")
                print(f"📊 Context: {len(result['context_claims'])} claims")
                print(f"🔧 Tools executed: {len(result['tool_results'])}")

                if result["tool_results"]:
                    print("\nTool results:")
                    for tool_result in result["tool_results"]:
                        tool_name = tool_result["tool"]
                        if tool_result["result"]["success"]:
                            print(f"  ✓ {tool_name}: Success")
                            if "message" in tool_result["result"]:
                                print(f"    {tool_result['result']['message']}")
                        else:
                            print(f"  ✗ {tool_name}: {tool_result['result']['error']}")

else:
                print(f"Error: {result.get('error', 'Unknown error')}")

            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")


def demo_workflows():
    """Run automated demo of all workflows"""
    print("🎬 Automated Workflow Demo")
    print("=" * 40)

    cf = Conjecture()

    workflows = [
        "Research artificial intelligence basics",
        "Write a simple Python function",
        "Test the implementation",
        "Evaluate the performance",
    ]

    for i, workflow in enumerate(workflows, 1):
        print(f"\n🔧 Workflow {i}: {workflow}")
        print("-" * 30)

        result = cf.process_request(workflow)

        if result["success"]:
            print(f"✅ Success using {result['skill_used']} skill")
            print(f"📊 Tool results: {len(result['tool_results'])}")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")

    print("\n🎉 Demo completed!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_workflows()
    else:
        main()
