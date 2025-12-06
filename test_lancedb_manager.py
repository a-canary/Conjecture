#!/usr/bin/env python3
"""
Test LanceDB Manager functionality
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_lancedb_manager():
    """Test LanceDB manager basic functionality."""
    try:
        from local.lancedb_manager import LanceDBManager
        print("SUCCESS: LanceDB manager imported successfully")

        # Test initialization
        manager = LanceDBManager("test_data/test_conjecture.lance")
        await manager.initialize(dimension=384)
        print("✅ LanceDB manager initialized")

        # Test adding embeddings
        texts = [
            "This is a test document about AI research",
            "Another document about machine learning",
            "A third document about data science"
        ]

        # Mock embeddings (384 dimensions)
        vectors = [
            [0.1] * 384,
            [0.2] * 384,
            [0.3] * 384
        ]

        metadata = [
            {"category": "ai", "source": "test"},
            {"category": "ml", "source": "test"},
            {"category": "ds", "source": "test"}
        ]

        claim_ids = ["claim_1", "claim_2", "claim_3"]

        ids = await manager.add_embeddings(texts, vectors, metadata, claim_ids)
        print(f"✅ Added {len(ids)} embeddings")

        # Test search
        results = await manager.search([0.15] * 384, limit=2)
        print(f"✅ Search found {len(results)} results")
        for result in results:
            print(f"   - {result['text'][:50]}... (score: {result['score']:.3f})")

        # Test get by claim ID
        claim_results = await manager.get_by_claim_id("claim_1")
        print(f"✅ Found {len(claim_results)} documents for claim_1")

        # Test stats
        stats = await manager.get_stats()
        print(f"✅ Database stats: {stats}")

        # Cleanup
        await manager.close()
        print("✅ LanceDB manager test completed successfully")

    except Exception as e:
        print(f"❌ LanceDB manager test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_lancedb_manager())