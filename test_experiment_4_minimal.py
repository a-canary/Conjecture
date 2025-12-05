#!/usr/bin/env python3
"""
Minimal test for Experiment 4 components
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test importing Experiment 4 components"""
    print("Testing Experiment 4 component imports...")
    
    try:
        from src.processing.adaptive_compression import AdaptiveCompressionEngine
        print("+ AdaptiveCompressionEngine imported successfully")
    except Exception as e:
        print(f"- AdaptiveCompressionEngine failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        from src.processing.hierarchical_context_processor import HierarchicalContextProcessor
        print("+ HierarchicalContextProcessor imported successfully")
    except Exception as e:
        print(f"- HierarchicalContextProcessor failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        from src.processing.intelligent_claim_selector import IntelligentClaimSelector
        print("+ IntelligentClaimSelector imported successfully")
    except Exception as e:
        print(f"- IntelligentClaimSelector failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("All imports successful!")
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)