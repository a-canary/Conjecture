#!/usr/bin/env python3
import sys

sys.path.insert(0, "src")

print("Testing imports...")

# Test contextflow import
try:
    from conjecture import Conjecture

    print("SUCCESS: contextflow import works")
    print(f"Conjecture class: {Conjecture}")
except Exception as e:
    print(f"FAILED: contextflow import failed: {e}")

# Test direct conjecture import
try:
    from conjecture import Conjecture

    print("SUCCESS: direct conjecture import works")
    print(f"Conjecture class: {Conjecture}")
except Exception as e:
    print(f"FAILED: direct conjecture import failed: {e}")

# Test package import
try:
    import conjecture

    print("SUCCESS: package import works")
    print(f"Conjecture class: {conjecture.Conjecture}")
except Exception as e:
    print(f"FAILED: package import failed: {e}")
