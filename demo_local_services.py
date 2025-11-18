#!/usr/bin/env python3
"""
Simple Local Services Demo
Demonstrates local embeddings and Ollama integration
"""

import sys
import os
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_local_embeddings():
    """Test local sentence embeddings"""
    print("=== Testing Local Embeddings ===")
    
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        # Load a small model
        print("Loading sentence transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test embedding
        texts = [
            "Machine learning is a subset of artificial intelligence",
            "Deep learning uses neural networks with multiple layers",
            "Natural language processing helps computers understand text"
        ]
        
        print("Generating embeddings...")
        embeddings = model.encode(texts)
        
        print(f"[OK] Generated {len(embeddings)} embeddings")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Sample embedding (first 5 dims): {embeddings[0][:5]}")
        
        # Test similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([embeddings[0]], embeddings[1:])
        print(f"   Similarities to first text: {similarities[0]}")
        
        return True
        
    except ImportError as e:
        print(f"[ERROR] Missing dependencies: {e}")
        print("   Install with: pip install sentence-transformers scikit-learn")
        return False
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return False

async def test_ollama_connection():
    """Test Ollama connection"""
    print("\n=== Testing Ollama Connection ===")
    
    try:
        import aiohttp
        import asyncio
        
        # Test Ollama API
        base_url = "http://localhost:11434"
        
        async with aiohttp.ClientSession() as session:
            # Check if Ollama is running
            try:
                async with session.get(f"{base_url}/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [model['name'] for model in data.get('models', [])]
                        print(f"[OK] Ollama is running with {len(models)} models:")
                        for model in models[:5]:  # Show first 5
                            print(f"   - {model}")
                        return True
                    else:
                        print(f"[ERROR] Ollama returned status {response.status}")
                        return False
            except asyncio.TimeoutError:
                print("[ERROR] Ollama connection timeout")
                return False
            except Exception as e:
                print(f"[ERROR] Ollama connection failed: {e}")
                return False
                
    except ImportError as e:
        print(f"[ERROR] Missing aiohttp: {e}")
        print("   Install with: pip install aiohttp")
        return False

def test_simple_vector_storage():
    """Test simple vector storage with SQLite"""
    print("\n=== Testing Simple Vector Storage ===")
    
    try:
        import sqlite3
        import numpy as np
        import json
        
        # Create in-memory database
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()
        
        # Create table for vectors
        cursor.execute('''
            CREATE TABLE vectors (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                metadata TEXT
            )
        ''')
        
        # Sample data
        texts = [
            "Python is a programming language",
            "JavaScript runs in web browsers",
            "SQL is used for databases"
        ]
        
        # Generate simple mock embeddings (normally you'd use sentence-transformers)
        embeddings = [np.random.rand(384).astype(np.float32) for _ in texts]
        
        # Store vectors
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            embedding_bytes = embedding.tobytes()
            metadata = json.dumps({"length": len(text), "index": i})
            
            cursor.execute(
                "INSERT INTO vectors (id, text, embedding, metadata) VALUES (?, ?, ?, ?)",
                (f"vec_{i}", text, embedding_bytes, metadata)
            )
        
        conn.commit()
        
        # Retrieve and verify
        cursor.execute("SELECT id, text, metadata FROM vectors")
        rows = cursor.fetchall()
        
        print(f"[OK] Stored and retrieved {len(rows)} vectors")
        for row in rows:
            print(f"   {row[0]}: {row[1][:30]}...")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"[ERROR] Vector storage error: {e}")
        return False

def test_faiss_if_available():
    """Test FAISS if available"""
    print("\n=== Testing FAISS (if available) ===")
    
    try:
        import faiss
        import numpy as np
        
        # Create simple index
        dimension = 384
        n_vectors = 100
        
        # Generate random vectors
        vectors = np.random.random((n_vectors, dimension)).astype('float32')
        
        # Create FAISS index
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)
        
        # Test search
        query = np.random.random((1, dimension)).astype('float32')
        k = 5
        distances, indices = index.search(query, k)
        
        print(f"[OK] FAISS working with {n_vectors} vectors")
        print(f"   Search results: indices {indices[0]}, distances {distances[0][:2]}")
        
        return True
        
    except ImportError:
        print("[WARNING] FAISS not available (install with: pip install faiss-cpu)")
        return False
    except Exception as e:
        print(f"[ERROR] FAISS error: {e}")
        return False

async def main():
    """Run all tests"""
    print("Conjecture Local Services Demo")
    print("=" * 50)
    
    results = {
        "embeddings": test_local_embeddings(),
        "vector_storage": test_simple_vector_storage(),
        "faiss": test_faiss_if_available()
    }
    
    # Test Ollama (async)
    results["ollama"] = await test_ollama_connection()
    
    print("\n" + "=" * 50)
    print("Results Summary:")
    
    for service, success in results.items():
        status = "[OK] Working" if success else "[ERROR] Failed"
        print(f"   {service:15}: {status}")
    
    working_count = sum(results.values())
    total_count = len(results)
    
    print(f"\nOverall: {working_count}/{total_count} services working")
    
    if working_count >= 2:
        print("[OK] Local services are ready for use!")
        print("\nNext steps:")
        print("1. Install Ollama: https://ollama.ai/")
        print("2. Pull a model: ollama pull llama2")
        print("3. Run Ollama server: ollama serve")
        print("4. Use the local CLI: python src/local_cli.py")
    else:
        print("[WARNING] Some services need setup")

if __name__ == "__main__":
    asyncio.run(main())