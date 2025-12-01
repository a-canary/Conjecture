#!/usr/bin/env python3
"""
Simple CLI demonstration script
Shows the unified API in action across different interface styles
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from conjecture import Conjecture


def safe_print(text):
    """Print text safely, handling encoding issues on Windows."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Remove problematic characters for Windows console
        safe_text = text.encode("ascii", "ignore").decode("ascii")
        print(safe_text)


def demonstrate_unified_api():
    """Demonstrate the unified Conjecture API."""
    safe_print("Conjecture Unified API Demonstration")
    safe_print("=" * 50)

    # Initialize the unified API
    cf = Conjecture()
    safe_print(f"[OK] Initialized Conjecture: {cf.config}")
    safe_print("")

    # Demonstrate exploration
    safe_print("Demonstrating Exploration")
    safe_print("-" * 30)

    queries = ["machine learning", "artificial intelligence", "data science"]

    for query in queries:
        safe_print(f"\nSearching: '{query}'")
        try:
            result = cf.explore(query, max_claims=3)
            safe_print(
                f"  Found {len(result.claims)} claims in {result.search_time:.2f}s"
            )
            for claim in result.claims:
                type_str = ", ".join([t.value for t in claim.type])
                safe_print(f"  - {claim.id}: {claim.content[:50]}... [{type_str}]")
        except Exception as e:
            safe_print(f"  Error: {e}")

    # Demonstrate claim creation
    print("\n\nDemonstrating Claim Creation")
    print("-" * 30)

    test_claims = [
        ("Python is excellent for data science", 0.9, "concept", ["python", "data"]),
        ("Machine learning requires quality data", 0.85, "thesis", ["ml", "data"]),
        ("Deep learning uses neural networks", 0.95, "concept", ["deep", "neural"]),
    ]

    for content, confidence, claim_type, tags in test_claims:
        safe_print(f"\nAdding claim: {content[:40]}...")
        try:
            claim = cf.add_claim(content, confidence, claim_type, tags)
            safe_print(f"  [OK] Created: {claim.id}")
            safe_print(f"  Type: {', '.join([t.value for t in claim.type])}")
            safe_print(f"  Confidence: {claim.confidence:.2f}")
        except Exception as e:
            safe_print(f"  Error: {e}")

    # Demonstrate statistics
    print("\n\nDemonstrating Statistics")
    print("-" * 30)

    try:
        stats = cf.get_statistics()
        print("System Statistics:")
        for key, value in stats.items():
            display_key = key.replace("_", " ").title()
            print(f"  {display_key}: {value}")
    except Exception as e:
        print(f"Error: {e}")

    safe_print("\n\nUnified API demonstration completed!")


def demonstrate_interface_patterns():
    """Demonstrate how different interfaces use the same API."""
    safe_print("\n\nInterface Pattern Demonstration")
    safe_print("=" * 50)

    # CLI Pattern
    print("\nCLI Pattern:")
    print("```python")
    print("from conjecture import Conjecture")
    print("cf = Conjecture()")
    print("result = cf.explore('machine learning')")
    print("claim = cf.add_claim('content', 0.8, 'concept')")
    print("```")

    # TUI Pattern
    print("\nTUI Pattern:")
    print("```python")
    print("from conjecture import Conjecture")
    print("class TUIApp:")
    print("    def __init__(self):")
    print("        self.cf = Conjecture()")
    print("    def search(self, query):")
    print("        return self.cf.explore(query)")
    print("```")

    # GUI Pattern
    print("\nGUI Pattern:")
    print("```python")
    print("from conjecture import Conjecture")
    print("class GUIApp:")
    print("    def __init__(self):")
    print("        self.cf = Conjecture()")
    print("    def on_search(self):")
    print("        results = self.cf.explore(self.query.get())")
    print("        self.display_results(results)")
    print("```")

    print("\nAll interfaces use the same simple pattern!")


def demonstrate_error_handling():
    """Demonstrate consistent error handling."""
    print("\n\nError Handling Demonstration")
    print("=" * 50)

    cf = Conjecture()

    # Test validation errors
    print("\n1. Testing query validation (too short):")
    try:
        result = cf.explore("hi")  # Too short
    except ValueError as e:
        print(f"   [OK] Caught expected error: {e}")

    print("\n2. Testing confidence validation (out of range):")
    try:
        claim = cf.add_claim("Valid content", 1.5, "concept")  # Too high
    except ValueError as e:
        print(f"   [OK] Caught expected error: {e}")

    print("\n3. Testing content validation (too short):")
    try:
        claim = cf.add_claim("Short", 0.8, "concept")  # Too short
    except ValueError as e:
        print(f"   [OK] Caught expected error: {e}")

    print("\n4. Testing claim type validation (invalid type):")
    try:
        claim = cf.add_claim(
            "Valid content with sufficient length", 0.8, "invalid_type"
        )
    except ValueError as e:
        print(f"   [OK] Caught expected error: {e}")

    print("\n[OK] Error handling works consistently across all interfaces!")


if __name__ == "__main__":
    try:
        demonstrate_unified_api()
        demonstrate_interface_patterns()
        demonstrate_error_handling()

        print("\n\n" + "=" * 60)
        print("🎯 KEY TAKEAWAYS:")
        print("  • Single Conjecture class serves all interfaces")
        print("  • No service layers or complex abstractions needed")
        print("  • Direct API usage across CLI, TUI, and GUI")
        print("  • Consistent error handling everywhere")
        print("  • Maximum power with minimum complexity")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        sys.exit(1)
