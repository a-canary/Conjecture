# Local Subsystem

## Overview

The Local subsystem provides lightweight local services for embeddings, LLM inference, and vector storage. It enables offline capability and reduces dependency on external services by implementing local alternatives.

This subsystem is designed to be modular and configurable, allowing users to choose between local and external services based on their needs and environment.

## Key Components and Responsibilities

### LocalServicesManager
- Unified manager for all local services
- Coordinates local embeddings, LLM, and vector storage services
- Handles initialization, health checks, and fallback logic
- Provides a single interface for all local services
- Implements fallback mechanisms to external services when local services are unavailable

### LocalEmbeddingManager
- Manages local text embeddings using sentence-transformers
- Provides embedding generation and batch processing
- Implements caching for improved performance
- Supports multiple embedding models
- Provides health check functionality

### OllamaClient
- Client for interacting with Ollama and LM Studio for local LLM inference
- Supports both streaming and non-streaming generation
- Handles connection to Ollama (localhost:11434) and LM Studio (localhost:1234)
- Provides model discovery and management
- Implements health checking and connection management

### LocalVectorStore
- Local vector storage implementation using FAISS or SQLite
- Provides vector addition and similarity search
- Supports configurable index types
- Implements health checking and persistence
- Provides statistics and metrics

### UnifiedManager
- Unified service manager with fallback support
- Coordinates local and external services
- Implements intelligent fallback mechanisms
- Provides comprehensive health monitoring
- Tracks service metrics and performance

## Integration with the Rest of the System

The Local subsystem integrates with other components through well-defined interfaces:

- **Agent Subsystem**: Provides local LLM inference and embeddings to the agent
- **Processing Subsystem**: Provides local LLM inference and embeddings for tool creation and analysis
- **LLM Subsystem**: Provides local LLM inference for instruction identification and support relationship creation
- **Data Layer**: Provides local vector storage for claim embeddings

The local subsystem acts as the foundation for offline capability, allowing the system to function without external dependencies.

## Example Usage

```python
from src.local import LocalServicesManager

# Initialize local services manager
manager = LocalServicesManager()
await manager.initialize()

# Generate embedding
embedding = await manager.generate_embedding("Hello world")

# Add vector to store
await manager.add_vector("claim-123", "Hello world", embedding)

# Search similar vectors
results = await manager.search_similar(embedding, limit=5)

# Generate LLM response
response = await manager.generate_response("What is the meaning of life?")

# Check health status
health = await manager.health_check()
print(f"Health status: {health['overall_status']}")
```

## Configuration Requirements

- **Embeddings**: Configuration for local embeddings (enabled/disabled) and model selection
- **Vector Store**: Configuration for local vector store (enabled/disabled) and storage type (FAISS/SQLite)
- **LLM**: Configuration for local LLM (enabled/disabled) and provider (Ollama/LM Studio)
- **Fallback**: Configuration for fallback to external services when local services are unavailable
- **Health Check**: Configuration for health check interval and timeout
- **Batch Size**: Configuration for embedding batch processing size

All configuration is handled through the LocalConfig system in the config/ directory. The system defaults to local services being enabled with fallback to external services, providing a robust offline-capable configuration.