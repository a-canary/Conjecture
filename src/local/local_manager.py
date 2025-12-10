"""
Local Services Manager
Coordinates local embeddings, LLM, and vector storage services
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import time
import warnings

from .embeddings import LocalEmbeddingManager, LocalEmbeddingManager
from .ollama_client import OllamaClient, ModelProvider
from .vector_store import LocalVectorStore, LocalVectorStore
from src.config.local_config import LocalConfig, ServiceMode

logger = logging.getLogger(__name__)

class LocalServicesManager:
    """
    Unified manager for all local services.
    Provides a single interface for embeddings, LLM, and vector storage.
    Handles initialization, health checks, and fallback logic.
    """

    def __init__(self, config: Optional[LocalConfig] = None):
        self.config = config or LocalConfig()
        
        # Service instances
        self.embedding_manager: Optional[LocalEmbeddingManager] = None
        self.vector_store: Optional[LocalVectorStore] = None
        self.llm_client: Optional[OllamaClient] = None
        
        # Service state
        self.initialized = False
        self.health_status = {}
        self.last_health_check = None
        
        # Fallback state
        self.embeddings_fallback = False
        self.vector_store_fallback = False
        self.llm_fallback = False

    async def initialize(self) -> None:
        """Initialize all configured local services."""
        if self.initialized:
            return

        logger.info("Initializing local services manager...")
        start_time = time.time()

        try:
            # Initialize services based on configuration
            await self._init_embeddings()
            await self._init_vector_store()
            await self._init_llm()

            # Perform initial health checks
            await self._check_all_services_health()

            self.initialized = True
            init_time = time.time() - start_time
            
            logger.info(f"Local services initialized in {init_time:.2f}s")
            await self._log_initialization_summary(init_time)

        except Exception as e:
            logger.error(f"Failed to initialize local services: {e}")
            raise RuntimeError(f"Local services initialization failed: {e}")

    async def _init_embeddings(self) -> None:
        """Initialize embedding service."""
        try:
            if self.config.use_local_embeddings:
                logger.info(f"Initializing local embeddings: {self.config.local_embedding_model}")
                self.embedding_manager = LocalEmbeddingManager(
                    model_name=self.config.local_embedding_model,
                    cache_dir=self.config.embedding_cache_dir
                )
                await self.embedding_manager.initialize()
            else:
                logger.info("Local embeddings disabled")
                self.embedding_manager = None

        except Exception as e:
            logger.warning(f"Failed to initialize local embeddings: {e}")
            self.embedding_manager = None
            if self.config.fallback_enabled:
                self.embeddings_fallback = True
                logger.info("Will fall back to external embeddings")

    async def _init_vector_store(self) -> None:
        """Initialize vector storage service."""
        try:
            if self.config.use_local_vector_store:
                logger.info(f"Initializing local vector store: {self.config.vector_store_type}")
                self.vector_store = LocalVectorStore(
                    db_path=self.config.vector_store_path,
                    index_type=self.config.faiss_index_type,
                    use_faiss=self.config.use_faiss
                )
                await self.vector_store.initialize()
            else:
                logger.info("Local vector store disabled")
                self.vector_store = None

        except Exception as e:
            logger.warning(f"Failed to initialize local vector store: {e}")
            self.vector_store = None
            if self.config.fallback_enabled:
                self.vector_store_fallback = True
                logger.info("Will fall back to external vector store")

    async def _init_llm(self) -> None:
        """Initialize LLM service."""
        try:
            if self.config.use_local_llm:
                logger.info(f"Initializing local LLM: {self.config.llm_endpoint}")
                self.llm_client = OllamaClient(
                    base_url=self.config.llm_endpoint,
                    model=self.config.llm_model
                )
                # Test connection
                models = await self.llm_client.list_models()
                if self.config.llm_model not in [model.name for model in models]:
                    logger.warning(f"Model {self.config.llm_model} not available")
            else:
                logger.info("Local LLM disabled")
                self.llm_client = None
                
        except Exception as e:
            logger.warning(f"Failed to initialize local LLM: {e}")
            self.llm_client = None
            if self.config.fallback_enabled:
                self.llm_fallback = True
                logger.info("Will fall back to external LLM")  