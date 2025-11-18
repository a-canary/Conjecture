# Conjecture Local Services Setup

This guide explains how to set up and use the local services integration for Conjecture CLI. The local services provide fast, offline-capable operation without heavy dependencies.

## 🚀 Quick Start

### 1. Install Core Dependencies

```bash
# Basic requirements
pip install -r requirements.txt

# Additional local services dependencies
pip install sentence-transformers faiss-cpu aiokafka
```

### 2. Test Basic Integration

```bash
# Quick validation
python test_quick_validation.py

# Full integration test
python test_local_integration.py
```

### 3. Start Using Local CLI

```bash
# Use local services with --local flag
python src/local_cli.py --local create "The Earth orbits the Sun" --user scientist --confidence 0.99

# Search claims
python src/local_cli.py search "astronomy concepts" --limit 5

# Check system health
python src/local_cli.py health

# View statistics
python src/local_cli.py stats
```

## 🔧 Local Services Configuration

### Environment Variables

Create a `.env` file in your project root:

```bash
# === Embedding Service ===
Conjecture_EMBEDDING_MODE=local
Conjecture_LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2
Conjecture_EMBEDDING_CACHE_DIR=~/.conjecture/models

# === Vector Storage ===
Conjecture_VECTOR_STORE_MODE=local
Conjecture_VECTOR_STORE_TYPE=faiss_sqlite
Conjecture_VECTOR_STORE_PATH=data/local_vector_store.db
Conjecture_FAISS_INDEX_TYPE=flat
Conjecture_USE_FAISS=true

# === Local LLM (Ollama/LM Studio) ===
Conjecture_LLM_MODE=auto
Conjecture_OLLAMA_URL=http://localhost:11434
Conjecture_LM_STUDIO_URL=http://localhost:1234
Conjecture_LLM_TIMEOUT=60

# === Performance ===
Conjecture_FALLBACK_ENABLED=true
Conjecture_ENABLE_CACHING=true
Conjecture_MAX_MEMORY_MB=1024
```

### Configuration Modes

- **`local`**: Use only local services
- **`external`**: Use only external services (ChromaDB, APIs)
- **`auto`**: Try local first, fallback to external
- **`disabled`**: Disable the service

## 🤖 Local LLM Setup

### Option 1: Ollama (Recommended)

1. **Install Ollama**:
   ```bash
   # macOS
   brew install ollama
   
   # Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Windows
   # Download from https://ollama.ai/download
   ```

2. **Start Ollama**:
   ```bash
   ollama serve
   ```

3. **Pull a Model**:
   ```bash
   ollama pull llama2
   ollama pull codellama
   ```

4. **Test with Conjecture**:
   ```bash
   python src/local_cli.py llm "What is the scientific method?" --model llama2
   python src/local_cli.py models
   ```

### Option 2: LM Studio

1. **Download LM Studio**: https://lmstudio.ai/

2. **Start LM Studio** on `localhost:1234`

3. **Load a Model**: Choose any model from the built-in browser

4. **Test with Conjecture**:
   ```bash
   python src/local_cli.py llm "Explain quantum computing"
   ```

## 📊 Vector Storage Options

### FAISS + SQLite (Default)

- **Fast**: Optimized vector search with FAISS
- **Lightweight**: SQLite for metadata storage
- **Scalable**: Handles thousands of vectors efficiently

### SQLite Only

- **Simpler**: No FAISS dependency
- **Reliable**: Pure SQLite implementation
- **Portable**: Single file database

## 🧪 Testing

### Run All Tests

```bash
# Quick validation
python test_quick_validation.py

# Comprehensive test suite
python test_local_integration.py
```

### Test Individual Components

```bash
# Test embeddings only
python -c "
import asyncio
from local.embeddings import MockEmbeddingManager
async def test():
    mgr = MockEmbeddingManager()
    await mgr.initialize()
    emb = await mgr.generate_embedding('test')
    print(f'Embedding: {len(emb)} dimensions')
asyncio.run(test())
"

# Test vector store only
python -c "
import asyncio
from local.vector_store import MockVectorStore
async def test():
    store = MockVectorStore()
    await store.initialize()
    await store.add_vector('test-1', 'test', [0.0]*384)
    results = await store.search_similar([0.0]*384)
    print(f'Search results: {len(results)}')
asyncio.run(test())
"
```

## 📁 Directory Structure

```
Conjecture/
├── src/
│   ├── local/
│   │   ├── __init__.py
│   │   ├── embeddings.py          # Local sentence-transformers
│   │   ├── vector_store.py        # FAISS + SQLite storage
│   │   ├── ollama_client.py       # Ollama/LM Studio client
│   │   ├── local_manager.py       # Services coordinator
│   │   └── unified_manager.py     # Fallback manager
│   ├── config/
│   │   ├── simple_config.py       # Original config
│   │   └── local_config.py        # Enhanced config
│   ├── local_cli.py               # Local-first CLI
│   └── full_cli.py                # CLI with --local option
├── data/
│   ├── local_vector_store.db      # SQLite vector storage
│   └── .conjecture/
│       └── models/                # Embedding model cache
├── requirements.txt
├── .env                           # Configuration
└── test_local_integration.py
```

## 🔄 Fallback Behavior

When `Conjecture_LLM_MODE=auto` (default):

1. **Primary Service**: Local service (Ollama/FAISS/etc.)
2. **Fallback**: External services (ChromaDB/APIs)
3. **Final Fallback**: Mock services (for testing)

Metrics are tracked for service usage and fallback events.

## 📈 Performance

### Startup Time
- **Local CLI**: ~2-3 seconds (model caching)
- **Original CLI**: ~10-15 seconds (TensorFlow/ChromaDB)
- **Mock Mode**: <1 second

### Memory Usage
- **Local Services**: ~200-500MB
- **Original**: ~1-2GB
- **Mock Mode**: ~50MB

### Search Performance
- **FAISS**: <50ms for 1000 vectors
- **SQLite**: ~200ms for 1000 vectors
- **ChromaDB**: ~100ms for 1000 vectors

## 🚨 Troubleshooting

### Common Issues

1. **FAISS Installation**:
   ```bash
   # If faiss-cpu fails, try:
   pip install faiss-cpu --no-cache-dir
   # or conda install -c conda-forge faiss-cpu
   ```

2. **Sentence Transformers Slow**:
   ```bash
   # Clear cache and retry
   rm -rf ~/.cache/torch/sentence_transformers
   ```

3. **Ollama Not Responding**:
   ```bash
   # Check if running
   curl http://localhost:11434/api/tags
   
   # Restart Ollama
   ollama serve
   ```

4. **Model Download Issues**:
   ```bash
   # Check embedding cache permissions
   ls -la ~/.conjecture/models/
   
   # Set correct permissions
   chmod -R 755 ~/.conjecture/
   ```

### Debug Mode

Enable debug logging:

```bash
export Conjecture_DEBUG=true
python src/local_cli.py health
```

### Service Health Check

```bash
# Check all services
python src/local_cli.py health

# Detailed stats
python src/local_cli.py stats

# Test individual components
python test_local_integration.py
```

## 🎯 Best Practices

### Production Deployment

1. **Configuration**:
   ```bash
   Conjecture_EMBEDDING_MODE=local
   Conjecture_VECTOR_STORE_MODE=local
   Conjecture_LLM_MODE=auto
   ```

2. **Resource Limits**:
   ```bash
   Conjecture_MAX_MEMORY_MB=2048
   Conjecture_EMBEDDING_BATCH_SIZE=16
   ```

3. **Monitoring**:
   ```bash
   python src/local_cli.py stats
   python src/local_cli.py health
   ```

### Development

1. **Use Mock Services**:
   ```python
   manager = LocalServicesManager(use_mocks=True)
   ```

2. **Enable Debug Mode**:
   ```bash
   Conjecture_DEBUG=true python src/local_cli.py ...
   ```

3. **Run Integration Tests**:
   ```bash
   python test_local_integration.py
   ```

## 📚 API Reference

### Local Services Manager

```python
from local.local_manager import LocalServicesManager

manager = LocalServicesManager()
await manager.initialize()

# Generate embedding
embedding = await manager.generate_embedding("text")

# Add vector
await manager.add_vector(id, content, embedding, metadata)

# Search similar
results = await manager.search_similar(embedding, limit=10)

# Generate LLM response
response = await manager.generate_response("prompt")
```

### Unified Manager with Fallback

```python
from local.unified_manager import UnifiedServiceManager

manager = UnifiedServiceManager()
await manager.initialize()

# Automatic fallback between local/external services
embedding = await manager.generate_embedding("text")
response = await manager.generate_response("prompt")

# Check status
status = await manager.get_comprehensive_status()
```

## 🤝 Contributing

To contribute to local services:

1. Run all tests: `python test_local_integration.py`
2. Follow existing code patterns
3. Update documentation
4. Test with both local and external services

## 📄 License

Same license as the main Conjecture project.