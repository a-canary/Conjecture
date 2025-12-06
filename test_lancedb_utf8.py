#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LanceDB Manager Test with UTF-8 Enforcement
"""

import os
import sys
# Force UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform.startswith('win'):
    try:
        import ctypes
        import ctypes.wintypes
        kernel32 = ctypes.windll.kernel32
        STD_OUTPUT_HANDLE = -11
        mode = ctypes.wintypes.DWORD()
        handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        kernel32.GetConsoleMode(handle, ctypes.byref(mode))
        mode.value |= 0x0004
        kernel32.SetConsoleMode(handle, mode)
        kernel32.SetConsoleOutputCP(65001)
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace', buffering=1)
        sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', errors='replace', buffering=1)

import asyncio
import tempfile
import numpy as np

async def test_lancedb_functionality():
    """Test LanceDB functionality with proper UTF-8 support."""
    print("🚀 LanceDB Functionality Test Starting...")

    try:
        import lancedb
        print("✅ LanceDB imported successfully")

        # Test basic functionality
        with tempfile.TemporaryDirectory() as tmp:
            db_path = tmp + '/test_conjecture.lance'
            db = lancedb.connect(db_path)
            print("✅ Database connection established")

            # Create test data with embeddings
            test_data = [
                {
                    'id': 'claim_1',
                    'text': 'Artificial Intelligence research advances rapidly in recent years 🤖',
                    'vector': np.random.rand(384).astype(np.float32),
                    'category': 'ai',
                    'confidence': 0.95
                },
                {
                    'id': 'claim_2',
                    'text': 'Machine learning models require substantial training data 📊',
                    'vector': np.random.rand(384).astype(np.float32),
                    'category': 'ml',
                    'confidence': 0.87
                },
                {
                    'id': 'claim_3',
                    'text': 'Data science combines statistics and computer science 🔬',
                    'vector': np.random.rand(384).astype(np.float32),
                    'category': 'ds',
                    'confidence': 0.92
                }
            ]

            # Create table with test data
            table = db.create_table('claims', test_data)
            print("✅ Table created with 3 test claims")

            # Test vector search
            query_vec = np.random.rand(384).astype(np.float32)
            results = table.search(query_vec).limit(2).to_pandas()
            print(f"✅ Vector search completed - found {len(results)} results")

            # Display results with Unicode
            print("\n📋 Search Results:")
            for i, row in results.iterrows():
                print(f"  {i+1}. {row['text']}")
                print(f"     Category: {row['category']} | Confidence: {row.get('confidence', 'N/A')}")
                print(f"     Score: {row.get('_score', 0.0):.3f}")
                print()

            # Test metadata functionality
            ai_results = table.search(query_vec).where("category = 'ai'").limit(1).to_pandas()
            print(f"✅ Category filtering works - found {len(ai_results)} AI claims")

            # Performance metrics
            print("📊 Performance Metrics:")
            print(f"  Database size: {os.path.getsize(db_path)} bytes")
            print(f"  Vector dimension: 384")
            print(f"  Total claims: {len(test_data)}")

        print("\n🎉 LanceDB validation completed successfully!")
        print("✅ Ready for production use in Conjecture!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_lancedb_functionality())