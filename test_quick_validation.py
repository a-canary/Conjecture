"""
Quick validation test for local services
Tests basic import and initialization
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        # Configuration
        from config.local_config import LocalConfig
        print("✓ Local config imported")
        
        # Local services
        from local.embeddings import LocalEmbeddingManager, MockEmbeddingManager
        print("✓ Embedding managers imported")
        
        from local.vector_store import LocalVectorStore, MockVectorStore
        print("✓ Vector stores imported")
        
        from local.ollama_client import OllamaClient
        print("✓ Ollama client imported")
        
        from local.local_manager import LocalServicesManager
        print("✓ Local services manager imported")
        
        from local.unified_manager import UnifiedServiceManager
        print("✓ Unified manager imported")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_config_creation():
    """Test configuration creation."""
    print("\nTesting configuration...")
    
    try:
        from config.local_config import LocalConfig
        config = LocalConfig()
        
        # Check basic properties
        assert config.embedding_mode is not None
        assert config.vector_store_mode is not None
        assert config.llm_mode is not None
        
        print("✓ Configuration created successfully")
        return True
        
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
        return False

def test_mock_services():
    """Test mock service instantiation."""
    print("\nTesting mock services...")
    
    try:
        from local.embeddings import MockEmbeddingManager
        from local.vector_store import MockVectorStore
        
        # Create mock services
        mock_embeddings = MockEmbeddingManager()
        mock_vector_store = MockVectorStore()
        
        print("✓ Mock services created successfully")
        return True
        
    except Exception as e:
        print(f"✗ Mock services failed: {e}")
        return False

def main():
    """Run quick validation tests."""
    print("Conjecture Local Services - Quick Validation")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config_creation,
        test_mock_services
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All validation tests passed!")
        print("\nThe local services integration is ready to use.")
        print("\nNext step: Run the full integration test:")
        print("python test_local_integration.py")
        return True
    else:
        print("✗ Some validation tests failed.")
        print("Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)