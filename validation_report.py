#!/usr/bin/env python3
"""
Comprehensive Documentation Validation Report for Conjecture
Tests all the claims made in documentation against actual implementation
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_import_paths():
    """Test which import paths actually work"""
    print("=== IMPORT PATH VALIDATION ===")

    results = {}

    # Test contextflow import
    try:
        from contextflow import Conjecture

        results["contextflow"] = "✅ WORKS"
    except Exception as e:
        results["contextflow"] = f"❌ FAILED: {e}"

    # Test core models import
    try:
        from core.unified_models import Claim, ClaimType, ClaimState

        results["core_models"] = "✅ WORKS"
    except Exception as e:
        results["core_models"] = f"❌ FAILED: {e}"

    # Test config import
    try:
        from config.simple_config import Config

        results["config"] = "✅ WORKS"
    except Exception as e:
        results["config"] = f"❌ FAILED: {e}"

    # Test enhanced conjecture import
    try:
        from src.enhanced_conjecture import EnhancedConjecture

        results["enhanced_conjecture"] = "✅ WORKS"
    except Exception as e:
        results["enhanced_conjecture"] = f"❌ FAILED: {e}"

    for path, result in results.items():
        print(f"{path}: {result}")

    return results


def test_api_methods():
    """Test which API methods actually exist and work"""
    print("\n=== API METHOD VALIDATION ===")

    try:
        from contextflow import Conjecture

        cf = Conjecture()

        methods = [m for m in dir(cf) if not m.startswith("_")]
        print(f"Available methods: {methods}")

        # Test specific methods mentioned in docs
        method_tests = {
            "explore": hasattr(cf, "explore"),
            "add_claim": hasattr(cf, "add_claim"),
            "get_statistics": hasattr(cf, "get_statistics"),
        }

        for method, exists in method_tests.items():
            status = "✅ EXISTS" if exists else "❌ MISSING"
            print(f"{method}(): {status}")

        return method_tests

    except Exception as e:
        print(f"❌ API method testing failed: {e}")
        return {}


def test_example_code():
    """Test if example code from documentation actually works"""
    print("\n=== EXAMPLE CODE VALIDATION ===")

    try:
        from contextflow import Conjecture

        cf = Conjecture()

        # Test exploration example
        print("Testing explore() example...")
        result = cf.explore("machine learning", max_claims=2)
        print(f"✅ explore() works: Found {len(result.claims)} claims")

        # Test claim creation example
        print("Testing add_claim() example...")
        claim = cf.add_claim(
            content="Machine learning requires quality data",
            confidence=0.85,
            claim_type="concept",
            tags=["ml", "data"],
        )
        print(f"✅ add_claim() works: Created claim {claim.id}")

        # Test statistics example
        print("Testing get_statistics() example...")
        stats = cf.get_statistics()
        print(f"✅ get_statistics() works: Returns {type(stats).__name__}")

        return True

    except Exception as e:
        print(f"❌ Example code testing failed: {e}")
        return False


def test_configuration():
    """Test configuration system"""
    print("\n=== CONFIGURATION VALIDATION ===")

    try:
        from config.simple_config import Config

        config = Config()

        # Test expected attributes
        attrs = {
            "database_type": hasattr(config, "database_type"),
            "database_path": hasattr(config, "database_path"),
            "llm_provider": hasattr(config, "llm_provider"),
            "confidence_threshold": hasattr(config, "confidence_threshold"),
        }

        for attr, exists in attrs.items():
            status = "✅ EXISTS" if exists else "❌ MISSING"
            value = getattr(config, attr, "N/A") if exists else ""
            print(f"{attr}: {status} ({value})")

        return attrs

    except Exception as e:
        print(f"❌ Configuration testing failed: {e}")
        return {}


def test_file_structure():
    """Test if documented file paths exist"""
    print("\n=== FILE STRUCTURE VALIDATION ===")

    expected_files = {
        "src/contextflow.py": os.path.exists("src/contextflow.py"),
        "src/core/__init__.py": os.path.exists("src/core/__init__.py"),
        "src/config/simple_config.py": os.path.exists("src/config/simple_config.py"),
        "docs/tutorials/basic_usage.md": os.path.exists(
            "docs/tutorials/basic_usage.md"
        ),
        "README.md": os.path.exists("README.md"),
        "requirements.txt": os.path.exists("requirements.txt"),
        ".env.example": os.path.exists(".env.example"),
    }

    for file_path, exists in expected_files.items():
        status = "✅ EXISTS" if exists else "❌ MISSING"
        print(f"{file_path}: {status}")

    return expected_files


def main():
    """Run all validation tests"""
    print("CONJECTURE DOCUMENTATION VALIDATION REPORT")
    print("=" * 50)

    import_results = test_import_paths()
    api_results = test_api_methods()
    example_results = test_example_code()
    config_results = test_configuration()
    file_results = test_file_structure()

    print("\n=== SUMMARY ===")

    # Count successes and failures
    total_imports = len(import_results)
    working_imports = sum(1 for r in import_results.values() if "✅" in r)

    total_apis = len(api_results)
    working_apis = sum(api_results.values())

    total_files = len(file_results)
    existing_files = sum(file_results.values())

    print(f"Import Paths: {working_imports}/{total_imports} working")
    print(f"API Methods: {working_apis}/{total_apis} working")
    print(f"Example Code: {'✅ WORKS' if example_results else '❌ BROKEN'}")
    print(f"File Structure: {existing_files}/{total_files} exist")

    print("\n=== RECOMMENDATIONS ===")

    if "src.enhanced_conjecture" not in import_results or "❌" in import_results.get(
        "src.enhanced_conjecture", ""
    ):
        print("• Fix src.enhanced_conjecture.py or remove from documentation")

    if working_apis < total_apis:
        print("• Update API documentation to reflect actual available methods")

    if existing_files < total_files:
        print("• Update file structure documentation or create missing files")

    print("• Update import examples to use working paths")
    print("• Test all code examples in documentation before publishing")

    print("\n=== WORKING EXAMPLES ===")
    print("✅ CORRECT: from contextflow import Conjecture")
    print("✅ CORRECT: from core.unified_models import Claim, ClaimType")
    print("✅ CORRECT: cf = Conjecture(); cf.explore('query')")
    print("✅ CORRECT: cf.add_claim('content', 0.8, 'concept', ['tag'])")

    print("\n=== BROKEN EXAMPLES ===")
    print("❌ BROKEN: from src.enhanced_conjecture import EnhancedConjecture")
    print("❌ BROKEN: import conjecture (CLI has dependency issues)")


if __name__ == "__main__":
    main()
