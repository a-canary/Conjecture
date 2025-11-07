"""
Quick Vector Database Evaluation with Graceful Dependency Handling
Tests available databases and provides recommendations
"""

import os
import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_dependency(name):
    """Check if a dependency is available"""
    try:
        if name == "chromadb":
            import chromadb

            return True, chromadb.__version__
        elif name == "faiss":
            import faiss

            return True, getattr(faiss, "__version__", "unknown")
        elif name == "sentence_transformers":
            import sentence_transformers

            return True, sentence_transformers.__version__
        elif name == "numpy":
            import numpy

            return True, numpy.__version__
        else:
            return False, None
    except ImportError:
        return False, None


def test_mock_implementation():
    """Test the MockChromaDB implementation"""
    print("Testing MockChromaDB Implementation...")

    try:
        from src.core.basic_models import BasicClaim, ClaimState, ClaimType
        from src.data.mock_chroma import MockChromaDB

        # Initialize database
        db = MockChromaDB("./data/quick_eval_mock.json")

        # Test basic operations
        claim = BasicClaim(
            id="test_1",
            content="Quantum entanglement enables instantaneous correlations",
            confidence=0.95,
            type=[ClaimType.CONCEPT],
            tags=["quantum-physics", "science"],
            state=ClaimState.EXPLORE,
        )

        # Test add
        start = time.time()
        add_success = db.add_claim(claim)
        add_time = time.time() - start

        # Test retrieve
        start = time.time()
        retrieved = db.get_claim("test_1")
        retrieve_time = time.time() - start

        # Test search
        start = time.time()
        search_results = db.search_by_content("quantum", limit=5)
        search_time = time.time() - start

        # Performance test with 1000 mock claims
        start = time.time()
        for i in range(1000):
            test_claim = BasicClaim(
                id=f"perf_{i}",
                content=f"Performance test claim {i} about physics and computing",
                confidence=0.8,
                type=[ClaimType.EXAMPLE],
                tags=["test", "performance"],
                state=ClaimState.EXPLORE,
            )
            db.add_claim(test_claim)
        batch_time = time.time() - start

        # Search performance
        start = time.time()
        results = db.search_by_content("performance", limit=10)
        large_search_time = time.time() - start

        stats = {
            "add_time": add_time,
            "retrieve_time": retrieve_time,
            "search_time": search_time,
            "batch_1000_time": batch_time,
            "large_search_time": large_search_time,
            "total_claims": db.get_claim_count(),
        }

        db.clear_all()  # Cleanup

        return True, stats, "MockChromaDB working correctly"

    except Exception as e:
        return False, {}, f"MockChromaDB error: {e}"


def test_chromadb_implementation():
    """Test ChromaDB implementation if available"""
    chroma_available, version = check_dependency("chromadb")

    if not chroma_available:
        return False, {}, "ChromaDB not installed"

    print(f"Testing ChromaDB {version}...")

    try:
        # Import after checking availability
        sys.path.insert(0, str(project_root / "evaluation"))
        from src.core.basic_models import BasicClaim, ClaimState, ClaimType
        from src.data.vectors.chromadb_integration import (
            ChromaDBIntegration,
            create_chromadb_config,
        )

        # Initialize database
        config = create_chromadb_config(
            persist_directory="./data/quick_eval_chroma", collection_name="quick_eval"
        )
        db = ChromaDBIntegration(config)

        # Test basic operations
        claim = BasicClaim(
            id="test_chroma_1",
            content="Machine learning enables pattern recognition from data",
            confidence=0.87,
            type=[ClaimType.CONCEPT],
            tags=["ai", "machine-learning"],
            state=ClaimState.VALIDATED,
        )

        # Test add
        start = time.time()
        add_success = db.add_claim(claim)
        add_time = time.time() - start

        # Test retrieve
        start = time.time()
        retrieved = db.get_claim("test_chroma_1")
        retrieve_time = time.time() - start

        # Test search
        start = time.time()
        search_results = db.search_similar("machine learning", limit=5)
        search_time = time.time() - start

        # Performance test with batch
        claims = []
        for i in range(100):
            test_claim = BasicClaim(
                id=f"chroma_perf_{i}",
                content=f"ChromaDB performance test {i} about vector databases and embeddings",
                confidence=0.85,
                type=[ClaimType.EXAMPLE],
                tags=["test", "vector-db"],
                state=ClaimState.EXPLORE,
            )
            claims.append(test_claim)

        start = time.time()
        batch_success = db.batch_add_claims(claims)
        batch_time = time.time() - start

        # Search performance
        start = time.time()
        results = db.search_similar("vector database", limit=10)
        large_search_time = time.time() - start

        stats = db.get_stats()
        stats.update(
            {
                "add_time": add_time,
                "retrieve_time": retrieve_time,
                "search_time": search_time,
                "batch_100_time": batch_time,
                "large_search_time": large_search_time,
                "total_claims": stats.get("total_claims", 0),
            }
        )

        db.clear_all()
        db.close()

        return True, stats, "ChromaDB working correctly"

    except Exception as e:
        return False, {}, f"ChromaDB error: {e}"


def test_faiss_implementation():
    """Test Faiss implementation if available"""
    faiss_available, faiss_version = check_dependency("faiss")
    st_available, st_version = check_dependency("sentence_transformers")

    if not faiss_available:
        return False, {}, "Faiss not installed"

    if not st_available:
        print("Warning: Faiss available but sentence-transformers missing")
        print("  Sentence transformers recommended for better performance")

    print(f"Testing Faiss {faiss_version}...")

    try:
        # Import after checking availability
        sys.path.insert(0, str(project_root / "evaluation"))
        from src.core.basic_models import BasicClaim, ClaimState, ClaimType
        from src.data.vectors.faiss_integration import (
            FaissIntegration,
            create_faiss_config,
        )

        # Initialize database
        config = create_faiss_config(
            index_path="./data/quick_eval_faiss.index",
            index_type="flat",  # Use flat for exact comparison
            embedding_dim=384,
        )
        db = FaissIntegration(config)

        # Test basic operations
        claim = BasicClaim(
            id="test_faiss_1",
            content="Neural networks use backpropagation for learning",
            confidence=0.90,
            type=[ClaimType.CONCEPT],
            tags=["ai", "neural-networks"],
            state=ClaimState.VALIDATED,
        )

        # Test add
        start = time.time()
        add_success = db.add_claim(claim)
        add_time = time.time() - start

        # Test retrieve
        start = time.time()
        retrieved = db.get_claim("test_faiss_1")
        retrieve_time = time.time() - start

        # Test search
        start = time.time()
        search_results = db.search_similar("neural networks", limit=5)
        search_time = time.time() - start

        # Performance test with batch
        claims = []
        for i in range(100):
            test_claim = BasicClaim(
                id=f"faiss_perf_{i}",
                content=f"Faiss performance test {i} about nearest neighbor search and embeddings",
                confidence=0.88,
                type=[ClaimType.EXAMPLE],
                tags=["test", "similarity-search"],
                state=ClaimState.EXPLORE,
            )
            claims.append(test_claim)

        start = time.time()
        batch_success = db.batch_add_claims(claims)
        batch_time = time.time() - start

        # Search performance
        start = time.time()
        results = db.search_similar("nearest neighbor", limit=10)
        large_search_time = time.time() - start

        stats = db.get_stats()
        stats.update(
            {
                "add_time": add_time,
                "retrieve_time": retrieve_time,
                "search_time": search_time,
                "batch_100_time": batch_time,
                "large_search_time": large_search_time,
                "total_claims": stats.get("total_claims", 0),
            }
        )

        db.clear_all()
        db.close()

        return True, stats, "Faiss working correctly"

    except Exception as e:
        return False, {}, f"Faiss error: {e}"


def calculate_score(success, stats, criteria):
    """Calculate score based on test results"""
    if not success:
        return 0

    score = 0

    # Basic functionality (20 points)
    if success:
        score += 20

    # Performance metrics (20 points)
    if "search_time" in stats and stats["search_time"] < 0.1:  # <100ms
        score += 10
    elif "search_time" in stats and stats["search_time"] < 0.2:  # <200ms
        score += 5

    if (
        "batch_100_time" in stats and stats["batch_100_time"] < 2.0
    ):  # <2s for 100 claims
        score += 10

    return score


def run_quick_evaluation():
    """Run quick evaluation of available databases"""
    print("=" * 80)
    print("Conjecture QUICK VECTOR DATABASE EVALUATION")
    print("=" * 80)
    print("Testing available implementations...")

    # Check dependencies first
    print("\n📦 DEPENDENCY CHECK:")
    print("-" * 40)

    deps = ["chromadb", "faiss", "sentence_transformers", "numpy"]
    for dep in deps:
        available, version = check_dependency(dep)
        status = "✅" if available else "❌"
        version_str = f" ({version})" if version else ""
        print(f"{status} {dep}{version_str}")

    # Test implementations
    print("\n🧪 TESTING IMPLEMENTATIONS:")
    print("-" * 40)

    results = []

    # Test 1: MockChromaDB (always available)
    mock_success, mock_stats, mock_msg = test_mock_implementation()
    mock_score = calculate_score(mock_success, mock_stats, 40)
    results.append(
        {
            "name": "MockChromaDB",
            "success": mock_success,
            "score": mock_score,
            "stats": mock_stats,
            "message": mock_msg,
        }
    )

    # Test 2: ChromaDB
    chroma_success, chroma_stats, chroma_msg = test_chromadb_implementation()
    chroma_score = calculate_score(chroma_success, chroma_stats, 40)
    results.append(
        {
            "name": "ChromaDB",
            "success": chroma_success,
            "score": chroma_score,
            "stats": chroma_stats,
            "message": chroma_msg,
        }
    )

    # Test 3: Faiss
    faiss_success, faiss_stats, faiss_msg = test_faiss_implementation()
    faiss_score = calculate_score(faiss_success, faiss_stats, 40)
    results.append(
        {
            "name": "Faiss",
            "success": faiss_success,
            "score": faiss_score,
            "stats": faiss_stats,
            "message": faiss_msg,
        }
    )

    # Results summary
    print("\n📊 RESULTS SUMMARY:")
    print("-" * 40)

    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"{result['name']:<15} | {result['score']:>2}/40 | {status}")
        if not result["success"]:
            print(f"{'':17} | {result['message']}")
        else:
            search_time = result["stats"].get("search_time", 0)
            search_ms = f"{search_time * 1000:.1f}ms" if search_time > 0 else "N/A"
            print(f"{'':17} | Search: {search_ms}")

    # Find best option
    successful = [r for r in results if r["success"]]

    if successful:
        best = max(successful, key=lambda x: x["score"])
        print(f"\n🏆 BEST OPTION: {best['name']}")
        print(f"   Score: {best['score']}/40")

        if best["score"] >= 35:
            print("   Status: ✅ MEETS SUCCESS CRITERIA")
        elif best["score"] >= 25:
            print("   Status: ⚠️  CLOSE TO CRITERIA")
        else:
            print("   Status: ❌ BELOW CRITERIA")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")

    if chroma_success:
        print("✅ ChromaDB is available and working")
        print("   Recommended for production use")
    else:
        print("❌ ChromaDB not available")
        print("   Install with: pip install chromadb")

    if faiss_success:
        print("✅ Faiss is available and working")
        print("   Recommended for high-performance scenarios")
    else:
        print("❌ Faiss not available")
        print("   Install with: pip install faiss-cpu sentence-transformers")

    print("✅ MockChromaDB always available")
    print("   Recommended for development and testing")

    # Next steps
    best_available = max(
        [r for r in results if r["success"]], key=lambda x: x["score"], default=None
    )

    if best_available:
        print(f"\n🚀 NEXT STEPS:")
        print(f"1. Use {best_available['name']} for development")
        print(f"2. Test with real Conjecture data")
        print(f"3. Move to Phase 2: LLM API Integration")

    # Performance comparison
    working_results = [r for r in results if r["success"]]
    if len(working_results) > 1:
        print(f"\n⚡ PERFORMANCE COMPARISON:")
        print("-" * 40)
        for result in working_results:
            stats = result["stats"]
            search_time = stats.get("search_time", 0)
            search_ms = f"{search_time * 1000:.1f}ms" if search_time > 0 else "N/A"
            batch_time = stats.get("batch_100_time", stats.get("batch_1000_time", 0))
            batch_ms = f"{batch_time * 1000:.1f}ms" if batch_time > 0 else "N/A"
            print(
                f"{result['name']:<15} | Search: {search_ms:>6} | Batch: {batch_ms:>8}"
            )

    print("\n" + "=" * 80)
    print("QUICK EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_quick_evaluation()
