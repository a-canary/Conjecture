"""
Comprehensive test script for local services integration
Tests all components of the local-first approach
"""

import asyncio
import sys
import os
import time
import pytest
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config.local_config import LocalConfig
from local.embeddings import LocalEmbeddingManager
from local.vector_store import LocalVectorStore
# from local.unified_manager import UnifiedServiceManager, create_unified_manager  # Module does not exist

class TestColors:
    """Colors for test output"""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"

def print_test(test_name: str, status: str, message: str = ""):
    """Print test result with color"""
    if status == "PASS":
        print(f"{TestColors.GREEN}✓ PASS{TestColors.ENDC} {test_name}")
    elif status == "FAIL":
        print(f"{TestColors.RED}✗ FAIL{TestColors.ENDC} {test_name}")
        if message:
            print(f"       {message}")
    elif status == "SKIP":
        print(f"{TestColors.YELLOW}- SKIP{TestColors.ENDC} {test_name}")
        if message:
            print(f"       {message}")
    elif status == "INFO":
        print(f"{TestColors.BLUE}ℹ INFO{TestColors.ENDC} {test_name}")
        if message:
            print(f"       {message}")

@pytest.mark.asyncio
async def test_local_embeddings(real_embedding_service):
    """Test local embedding manager."""
    print_test("Testing Local Embeddings", "INFO", "Starting")
    
    try:
        # Test with real embeddings
        print_test("Real Embedding Manager", "INFO", "Testing")
        
        # Test embedding generation
        test_text = "Test quantum encryption for hospital networks"
        embedding = await real_embedding_service.generate_embedding(test_text)
        
        if len(embedding) == real_embedding_service.embedding_dimension:
            print_test("Embedding Generation", "PASS", f"Generated {len(embedding)}-dimension embedding")
        else:
            print_test("Embedding Generation", "FAIL", f"Expected {real_embedding_service.embedding_dimension}, got {len(embedding)}")
            return False
        
        # Test batch embedding generation
        test_texts = [
            "First test claim about quantum cryptography",
            "Second test claim about hospital security",
            "Third test claim about data protection"
        ]
        batch_embeddings = await real_embedding_service.generate_embeddings_batch(test_texts)
        
        if len(batch_embeddings) == len(test_texts):
            print_test("Batch Embedding Generation", "PASS", f"Generated {len(batch_embeddings)} embeddings")
        else:
            print_test("Batch Embedding Generation", "FAIL", f"Expected {len(test_texts)}, got {len(batch_embeddings)}")
            return False
        
        # Test similarity computation
        similarity = await real_embedding_service.compute_similarity(embedding, batch_embeddings[1])
        if 0.0 <= similarity <= 1.0:
            print_test("Similarity Computation", "PASS", f"Similarity: {similarity:.3f}")
        else:
            print_test("Similarity Computation", "FAIL", f"Invalid similarity: {similarity}")
            return False
        
        print_test("Local Embeddings", "PASS", "All tests completed")
        return True
        
    except Exception as e:
        print_test("Local Embeddings", "FAIL", f"Error: {str(e)}")
        return False
        
@pytest.mark.asyncio
async def test_local_vector_store(real_vector_store, real_embedding_service):
    """Test local vector store with real embeddings."""
    print_test("Testing Local Vector Store", "INFO", "Starting")
    
    try:
        # Test adding vectors
        test_claims = [
            ("claim_001", "Quantum encryption provides security", [0.1, 0.2, 0.3]),
            ("claim_002", "Hospital networks need encryption", [0.4, 0.5, 0.6]),
            ("claim_003", "Data protection is essential", [0.7, 0.8, 0.9])
        ]
        
        # Generate embeddings for test claims
        for claim_id, content, embedding in test_claims:
            success = await real_vector_store.add_vector(claim_id, content, embedding)
            if success:
                print_test(f"Vector Store Add {claim_id}", "PASS", "Added vector to store")
            else:
                print_test(f"Vector Store Add {claim_id}", "FAIL", "Failed to add vector")
                return False
        
        # Test similarity search
        query_embedding = await real_embedding_service.generate_embedding("quantum security")
        results = await real_vector_store.search_similar(query_embedding, limit=2)
        
        if len(results) >= 1:
            print_test("Vector Store Search", "PASS", f"Found {len(results)} similar vectors")
        else:
            print_test("Vector Store Search", "FAIL", "No similar vectors found")
            return False
        
        print_test("Local Vector Store", "PASS", "All tests completed")
        return True
        
    except Exception as e:
        print_test("Local Vector Store", "FAIL", f"Error: {str(e)}")
        return False

@pytest.mark.asyncio
async def test_local_services_integration(real_embedding_service, real_vector_store):
    """Test integration of all local services."""
    print_test("Testing Local Services Integration", "INFO", "Starting")
    
    try:
        # Test embedding -> vector store integration
        test_text = "Integration test for quantum hospital security"
        embedding = await real_embedding_service.generate_embedding(test_text)
        
        # Add to vector store
        await real_vector_store.add_vector("integration_test", test_text, embedding)
        
        # Search for similar content
        search_results = await real_vector_store.search_similar(embedding, limit=5)
        
        if len(search_results) >= 1:
            print_test("Local Services Integration", "PASS", f"Integration successful, found {len(search_results)} results")
        else:
            print_test("Local Services Integration", "FAIL", "Integration failed, no results found")
            return False
        
        print_test("Local Services Integration", "PASS", "All integration tests completed")
        return True
        
    except Exception as e:
        print_test("Local Services Integration", "FAIL", f"Error: {str(e)}")
import asyncio

if __name__ == "__main__":
    """Run all local integration tests."""
    print("=" * 60)
    print("LOCAL SERVICES INTEGRATION TEST SUITE")
    print("=" * 60)
    
    # Test individual components
    async def run_all_tests():
        results = []
        results.append(await test_local_embeddings())
        results.append(await test_local_vector_store())
        results.append(await test_local_services_integration())
        return results
    
    # Run tests
    results = asyncio.run(run_all_tests())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 60)
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    print("=" * 60)
    
    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)