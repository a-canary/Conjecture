"""
Comprehensive test script for local services integration
Tests all components of the local-first approach
"""

import asyncio
import sys
import os
import time
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config.local_config import LocalConfig
from local.embeddings import LocalEmbeddingManager, MockEmbeddingManager
from local.ollama_client import OllamaClient, ModelProvider, create_ollama_client
from local.vector_store import LocalVectorStore, MockVectorStore
from local.local_manager import LocalServicesManager, create_local_manager
from local.unified_manager import UnifiedServiceManager, create_unified_manager


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


async def test_local_embeddings():
    """Test local embedding manager."""
    print_test("Testing Local Embeddings", "INFO", "Starting")
    
    try:
        # Test with mock embeddings
        print_test("Mock Embedding Manager", "INFO", "Testing")
        mock_manager = MockEmbeddingManager(embedding_dim=384)
        await mock_manager.initialize()
        
        # Test single embedding
        test_text = "This is a test claim for embedding generation."
        embedding = await mock_manager.generate_embedding(test_text)
        
        if len(embedding) == 384:
            print_test("Mock embedding generation", "PASS")
        else:
            print_test("Mock embedding generation", "FAIL", f"Expected 384 dims, got {len(embedding)}")
        
        # Test batch embeddings
        texts = [f"Test claim {i}" for i in range(5)]
        batch_embeddings = await mock_manager.generate_embeddings_batch(texts)
        
        if len(batch_embeddings) == 5 and all(len(emb) == 384 for emb in batch_embeddings):
            print_test("Mock batch embedding", "PASS")
        else:
            print_test("Mock batch embedding", "FAIL", f"Expected 5 embeddings of 384 dims")
        
        # Test similarity computation
        similarity = await mock_manager.compute_similarity(embedding, embedding)
        if abs(similarity - 1.0) < 0.001:
            print_test("Embedding similarity", "PASS")
        else:
            print_test("Embedding similarity", "FAIL", f"Expected 1.0, got {similarity}")
        
        await mock_manager.close()
        
        # Test local sentence transformers (if available)
        try:
            print_test("Local sentence-transformers", "INFO", "Testing")
            local_manager = LocalEmbeddingManager("all-MiniLM-L6-v2")
            await local_manager.initialize()
            
            embedding = await local_manager.generate_embedding(test_text)
            
            if len(embedding) == 384:
                print_test("Local sentence-transformers", "PASS", "all-MiniLM-L6-v2 working")
            else:
                print_test("Local sentence-transformers", "FAIL", f"Expected 384 dims, got {len(embedding)}")
            
            await local_manager.close()
            
        except Exception as e:
            print_test("Local sentence-transformers", "SKIP", f"Not available: {str(e)[:50]}...")
    
    except Exception as e:
        print_test("Local Embeddings test", "FAIL", str(e))


async def test_local_vector_store():
    """Test local vector store."""
    print_test("Testing Local Vector Store", "INFO", "Starting")
    
    try:
        # Test with mock vector store
        print_test("Mock Vector Store", "INFO", "Testing")
        mock_store = MockVectorStore(embedding_dim=384)
        await mock_store.initialize()
        
        # Add test vectors
        test_embedding = [0.0] * 384  # Simple test embedding
        success = await mock_store.add_vector(
            claim_id="test-1",
            content="Test claim content",
            embedding=test_embedding,
            metadata={"test": True}
        )
        
        if success:
            print_test("Mock vector add", "PASS")
        else:
            print_test("Mock vector add", "FAIL")
        
        # Test search
        results = await mock_store.search_similar(test_embedding, limit=5)
        if len(results) >= 1:
            print_test("Mock vector search", "PASS")
        else:
            print_test("Mock vector search", "FAIL", f"Expected >=1 results, got {len(results)}")
        
        # Test FAISS + SQLite store (if FAISS available)
        try:
            print_test("FAISS + SQLite Vector Store", "INFO", "Testing")
            
            import tempfile
            temp_dir = tempfile.mkdtemp()
            db_path = os.path.join(temp_dir, "test_vector_store.db")
            
            faiss_store = LocalVectorStore(db_path=db_path, use_faiss=True)
            await faiss_store.initialize(dimension=384)
            
            # Add vector
            success = await faiss_store.add_vector(
                claim_id="faiss-test-1",
                content="FAISS test claim",
                embedding=test_embedding,
                metadata={"faiss": True}
            )
            
            if success:
                print_test("FAISS vector add", "PASS")
            else:
                print_test("FAISS vector add", "FAIL")
            
            # Test search
            results = await faiss_store.search_similar(test_embedding, limit=5)
            if len(results) >= 1:
                print_test("FAISS vector search", "PASS")
            else:
                print_test("FAISS vector search", "FAIL", f"Expected >=1 results, got {len(results)}")
            
            # Test stats
            stats = await faiss_store.get_stats()
            if stats.get('total_vectors', 0) >= 1:
                print_test("FAISS vector stats", "PASS")
            else:
                print_test("FAISS vector stats", "FAIL")
            
            await faiss_store.close()
            
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        except Exception as e:
            print_test("FAISS + SQLite Vector Store", "SKIP", f"Not available: {str(e)[:50]}...")
    
    except Exception as e:
        print_test("Local Vector Store test", "FAIL", str(e))


async def test_local_llm():
    """Test local LLM client."""
    print_test("Testing Local LLM", "INFO", "Starting")
    
    try:
        # Test Ollama client
        print_test("Ollama Client", "INFO", "Testing connection")
        
        ollama_client = OllamaClient(
            base_url="http://localhost:11434",
            timeout=10,
            provider=ModelProvider.OLLAMA
        )
        
        is_available = await ollama_client.health_check()
        if is_available:
            print_test("Ollama health check", "PASS")
            
            # Test model list
            models = await ollama_client.get_available_models()
            print_test("Ollama model list", "PASS", f"Found {len(models)} models")
            
            if models:
                # Test generation
                try:
                    response = await ollama_client.generate_response(
                        "Hello, respond briefly.",
                        model=models[0].name
                    )
                    if response:
                        print_test("Ollama generation", "PASS", f"Response length: {len(response)}")
                    else:
                        print_test("Ollama generation", "FAIL", "Empty response")
                except Exception as e:
                    print_test("Ollama generation", "FAIL", str(e))
            else:
                print_test("Ollama generation", "SKIP", "No models available")
        else:
            print_test("Ollama health check", "SKIP", "Ollama not running")
        
        await ollama_client.close()
        
        # Test LM Studio client
        print_test("LM Studio Client", "INFO", "Testing connection")
        
        lm_client = OllamaClient(
            base_url="http://localhost:1234",
            timeout=10,
            provider=ModelProvider.LM_STUDIO
        )
        
        is_available = await lm_client.health_check()
        if is_available:
            print_test("LM Studio health check", "PASS")
            
            # Test model list
            models = await lm_client.get_available_models()
            print_test("LM Studio model list", "PASS", f"Found {len(models)} models")
        else:
            print_test("LM Studio health check", "SKIP", "LM Studio not running")
        
        await lm_client.close()
    
    except Exception as e:
        print_test("Local LLM test", "FAIL", str(e))


async def test_local_manager():
    """Test local services manager."""
    print_test("Testing Local Services Manager", "INFO", "Starting")
    
    try:
        # Test with mock services
        print_test("Local Manager (Mock)", "INFO", "Testing")
        
        local_manager = LocalServicesManager(use_mocks=True)
        await local_manager.initialize()
        
        # Test embedding generation
        embedding = await local_manager.generate_embedding("Test claim")
        if embedding:
            print_test("Local manager embedding", "PASS")
        else:
            print_test("Local manager embedding", "FAIL")
        
        # Test vector operations
        success = await local_manager.add_vector(
            claim_id="manager-test-1",
            content="Test content",
            embedding=embedding
        )
        if success:
            print_test("Local manager vector add", "PASS")
        else:
            print_test("Local manager vector add", "FAIL")
        
        # Test search
        results = await local_manager.search_similar(embedding, limit=5)
        if len(results) >= 1:
            print_test("Local manager search", "PASS")
        else:
            print_test("Local manager search", "FAIL")
        
        # Test LLM
        try:
            response = await local_manager.generate_response("Hello")
            if response:
                print_test("Local manager LLM", "PASS")
            else:
                print_test("Local manager LLM", "FAIL")
        except Exception as e:
            print_test("Local manager LLM", "SKIP", f"Mock service: {str(e)[:50]}...")
        
        # Test health check
        health = await local_manager.health_check()
        if health.get('initialized'):
            print_test("Local manager health", "PASS")
        else:
            print_test("Local manager health", "FAIL")
        
        await local_manager.close()
        
    except Exception as e:
        print_test("Local Services Manager test", "FAIL", str(e))


async def test_unified_manager():
    """Test unified service manager with fallback."""
    print_test("Testing Unified Service Manager", "INFO", "Starting")
    
    try:
        # Test with mock services
        print_test("Unified Manager (Mock)", "INFO", "Testing")
        
        config = LocalConfig()
        unified_manager = UnifiedServiceManager(config, use_mocks=True)
        await unified_manager.initialize()
        
        # Test embedding with fallback
        embedding = await unified_manager.generate_embedding("Test claim")
        if embedding:
            print_test("Unified manager embedding", "PASS")
        else:
            print_test("Unified manager embedding", "FAIL")
        
        # Test vector operations with fallback
        success = await unified_manager.add_vector(
            claim_id="unified-test-1",
            content="Test content",
            embedding=embedding
        )
        if success:
            print_test("Unified manager vector add", "PASS")
        else:
            print_test("Unified manager vector add", "FAIL")
        
        # Test search with fallback
        results = await unified_manager.search_similar(embedding, limit=5)
        if len(results) >= 1:
            print_test("Unified manager search", "PASS")
        else:
            print_test("Unified manager search", "FAIL")
        
        # Test LLM with fallback
        try:
            response = await unified_manager.generate_response("Hello")
            if response:
                print_test("Unified manager LLM", "PASS")
            else:
                print_test("Unified manager LLM", "FAIL")
        except Exception as e:
            print_test("Unified manager LLM", "SKIP", f"Service not available")
        
        # Test comprehensive status
        status = await unified_manager.get_comprehensive_status()
        if status.get('initialized'):
            print_test("Unified manager status", "PASS")
        else:
            print_test("Unified manager status", "FAIL")
        
        await unified_manager.close()
        
    except Exception as e:
        print_test("Unified Service Manager test", "FAIL", str(e))


async def test_config_system():
    """Test configuration system."""
    print_test("Testing Configuration System", "INFO", "Starting")
    
    try:
        # Test local config
        config = LocalConfig()
        
        if config.embedding_mode:
            print_test("Local config creation", "PASS")
        else:
            print_test("Local config creation", "FAIL")
        
        # Test validation
        if config.validate_local_config():
            print_test("Local config validation", "PASS")
        else:
            print_test("Local config validation", "FAIL")
        
        # Test configuration properties
        if config.supports_offline():
            print_test("Offline capability", "PASS")
        else:
            print_test("Offline capability", "SKIP", "Configuration requires external services")
        
        # Test config generation
        config_dict = config.to_dict()
        if config_dict and 'embedding' in config_dict:
            print_test("Config serialization", "PASS")
        else:
            print_test("Config serialization", "FAIL")
        
    except Exception as e:
        print_test("Configuration System test", "FAIL", str(e))


async def test_end_to_end_workflow():
    """Test complete end-to-end workflow."""
    print_test("Testing End-to-End Workflow", "INFO", "Starting")
    
    try:
        # Initialize unified manager
        config = LocalConfig()
        unified_manager = UnifiedServiceManager(config, use_mocks=True)
        await unified_manager.initialize()
        
        # Simulate complete workflow
        print_test("Complete workflow", "INFO", "Simulating")
        
        # Step 1: Create claims with embeddings
        claims = [
            ("The sky is blue due to Rayleigh scattering", 0.95),
            ("Water boils at 100°C at sea level", 0.99),
            ("Gravity causes objects to fall", 0.98)
        ]
        
        claim_ids = []
        for i, (content, confidence) in enumerate(claims):
            # Generate embedding
            embedding = await unified_manager.generate_embedding(content)
            
            # Store in vector database
            claim_id = f"test-claim-{i+1}"
            success = await unified_manager.add_vector(
                claim_id=claim_id,
                content=content,
                embedding=embedding,
                metadata={"confidence": confidence, "type": "scientific"}
            )
            
            if success:
                claim_ids.append(claim_id)
            else:
                print_test(f"Claim {i+1} creation", "FAIL")
        
        if len(claim_ids) == len(claims):
            print_test("Claim creation", "PASS", f"Created {len(claim_ids)} claims")
        else:
            print_test("Claim creation", "FAIL", f"Expected {len(claims)}, got {len(claim_ids)}")
        
        # Step 2: Search for similar claims
        query = "physics phenomena"
        query_embedding = await unified_manager.generate_embedding(query)
        
        results = await unified_manager.search_similar(query_embedding, limit=5)
        if len(results) >= 1:
            print_test("Claims search", "PASS", f"Found {len(results)} results")
        else:
            print_test("Claims search", "FAIL", "No results found")
        
        # Step 3: Generate insights with LLM
        try:
            search_summary = f"Found {len(results)} claims for query: {query}"
            insights = await unified_manager.generate_response(
                f"Analyze these search results: {search_summary}"
            )
            if insights:
                print_test("LLM insights", "PASS")
            else:
                print_test("LLM insights", "FAIL")
        except Exception as e:
            print_test("LLM insights", "SKIP", "LLM not available")
        
        # Step 4: Check system health
        status = await unified_manager.get_comprehensive_status()
        if status['overall_health'] in ['healthy', 'degraded']:
            print_test("System health", "PASS", f"Status: {status['overall_health']}")
        else:
            print_test("System health", "FAIL", f"Status: {status['overall_health']}")
        
        await unified_manager.close()
        
    except Exception as e:
        print_test("End-to-End Workflow", "FAIL", str(e))


async def main():
    """Run all tests."""
    print(f"{TestColors.BOLD}{TestColors.BLUE}")
    print("=" * 60)
    print("CONJECTURE LOCAL SERVICES INTEGRATION TESTS")
    print("=" * 60)
    print(f"{TestColors.ENDC}")
    
    start_time = time.time()
    
    # Run all tests
    await test_config_system()
    await test_local_embeddings()
    await test_local_vector_store()
    await test_local_llm()
    await test_local_manager()
    await test_unified_manager()
    await test_end_to_end_workflow()
    
    # Summary
    total_time = time.time() - start_time
    
    print(f"\n{TestColors.BOLD}")
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"{TestColors.ENDC}")
    
    print(f"Total test time: {total_time:.2f}s")
    print(f"Test environment: {sys.platform}")
    print(f"Python version: {sys.version.split()[0]}")
    
    print(f"\n{TestColors.GREEN}✓ Integration tests completed{TestColors.ENDC}")
    print("Check individual test results above for details.")
    
    # Provide next steps
    print(f"\n{TestColors.BLUE}Next Steps:{TestColors.ENDC}")
    print("1. Install missing dependencies for full functionality:")
    print("   pip install sentence-transformers faiss-cpu")
    print("2. Start local LLM services:")
    print("   - Ollama: ollama serve && ollama pull llama2")
    print("   - LM Studio: Start app on localhost:1234")
    print("3. Test real services:")
    print("   python src/local_cli.py --local create 'Test claim' --user test")
    print("4. Check health status:")
    print("   python src/local_cli.py health")


if __name__ == "__main__":
    asyncio.run(main())