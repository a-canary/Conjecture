"""
Local services for Conjecture CLI.
Provides lightweight local embeddings, LLM inference, and vector storage.
"""

from .embeddings import LocalEmbeddingManager
from .ollama_client import OllamaClient
from .vector_store import LocalVectorStore

__all__ = ['LocalEmbeddingManager', 'OllamaClient', 'LocalVectorStore']