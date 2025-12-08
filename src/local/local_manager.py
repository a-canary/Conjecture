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

from .embeddings import LocalEmbeddingManager, MockEmbeddingManager
from .ollama_client import OllamaClient, ModelProvider
from .vector_store import LocalVectorStore, MockVectorStore
from src.config.local_config import LocalConfig, ServiceMode

logger = logging.getLogger(__name__)


class LocalServicesManager:
    """
    Unified manager for all local services.
    Provides a single interface for embeddings, LLM, and vector storage.
    Handles initialization, health checks, and fallback logic.
    """

    def __init__(self, config: Optional[LocalConfig] = None, use_mocks: bool = False):
        self.config = config or LocalConfig()
        self.use_mocks = use_mocks
        
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
            if self.use_mocks:
                logger.info("Using mock embedding service")
                self.embedding_manager = MockEmbeddingManager()
            elif self.config.use_local_embeddings:
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
            if self.use_mocks:
                logger.info("Using mock vector store")
                self.vector_store = MockVectorStore()
            elif self.config.use_local_vector_store:
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
            if self.use_mocks:
                logger.info("Using mock LLM service")
                self.llm_client = None  # Mock responses handled in manager
            elif self.config.use_local_llm:
                logger.info(f"Initializing local LLM client")
                
                # Try Ollama first, then LM Studio
                for url, provider in [
                    (self.config.ollama_base_url, ModelProvider.OLLAMA),
                    (self.config.lm_studio_url, ModelProvider.LM_STUDIO)
                ]:
                    try:
                        logger.info(f"Trying {provider.value} at {url}")
                        self.llm_client = OllamaClient(base_url=url, provider=provider)
                        await self.llm_client.initialize()
                        
                        # Check if service is actually available
                        if await self.llm_client.health_check():
                            logger.info(f"Connected to {provider.value}")
                            break
                        else:
                            await self.llm_client.close()
                            self.llm_client = None
                            logger.warning(f"{provider.value} not available")
                    
                    except Exception as e:
                        logger.warning(f"Failed to connect to {provider.value}: {e}")
                        continue

                if self.llm_client is None:
                    raise ConnectionError("No local LLM service available")

            else:
                logger.info("Local LLM disabled")
                self.llm_client = None

        except Exception as e:
            logger.warning(f"Failed to initialize local LLM: {e}")
            self.llm_client = None
            if self.config.fallback_enabled:
                self.llm_fallback = True
                logger.info("Will fall back to external LLM")

    async def _check_all_services_health(self) -> None:
        """Check health of all initialized services."""
        self.health_status = {}
        
        if self.embedding_manager:
            try:
                self.health_status['embeddings'] = await self.embedding_manager.health_check()
            except Exception as e:
                self.health_status['embeddings'] = {'status': 'unhealthy', 'error': str(e)}
        
        if self.vector_store:
            try:
                self.health_status['vector_store'] = await self.vector_store.health_check()
            except Exception as e:
                self.health_status['vector_store'] = {'status': 'unhealthy', 'error': str(e)}
        
        if self.llm_client:
            try:
                healthy = await self.llm_client.health_check()
                self.health_status['llm'] = {'status': 'healthy' if healthy else 'unhealthy'}
            except Exception as e:
                self.health_status['llm'] = {'status': 'unhealthy', 'error': str(e)}
        
        self.last_health_check = datetime.utcnow()

    async def _log_initialization_summary(self, init_time: float) -> None:
        """Log initialization summary."""
        logger.info("=== Local Services Initialization Summary ===")
        logger.info(f"Initialization time: {init_time:.2f}s")
        
        # Embeddings
        if self.embedding_manager:
            info = self.embedding_manager.get_model_info()
            logger.info(f"Embeddings: {info.get('model_name', 'unknown')} - {info.get('embedding_dimension', 'unknown')} dims")
        else:
            logger.info(f"Embeddings: {'Disabled' if not self.config.use_local_embeddings else 'Failed'}")
        
        # Vector Store
        if self.vector_store:
            stats = await self.vector_store.get_stats()
            logger.info(f"Vector Store: {stats.get('total_vectors', 0)} vectors, FAISS: {stats.get('use_faiss', False)}")
        else:
            logger.info(f"Vector Store: {'Disabled' if not self.config.use_local_vector_store else 'Failed'}")
        
        # LLM
        if self.llm_client:
            info = await self.llm_client.get_service_info()
            logger.info(f"LLM: {info.get('provider', 'unknown')} - {info.get('model_count', 0)} models")
        else:
            logger.info(f"LLM: {'Disabled' if not self.config.use_local_llm else 'Failed'}")

    # === Embedding Operations ===

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate text embedding using local or fallback service."""
        if not self.initialized:
            await self.initialize()

        if self.embedding_manager:
            try:
                embedding = await self.embedding_manager.generate_embedding(text)
                if self.embeddings_fallback:
                    logger.info("Successfully used local embeddings (was falling back)")
                    self.embeddings_fallback = False
                return embedding
            except Exception as e:
                logger.error(f"Local embedding generation failed: {e}")
                if self.config.fallback_enabled and not self.embeddings_fallback:
                    self.embeddings_fallback = True
                    logger.warning("Falling back to external embeddings")

        # Fallback logic would be handled by the calling layer
        raise RuntimeError("No embedding service available")

    async def generate_embeddings_batch(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not self.initialized:
            await self.initialize()

        batch_size = batch_size or self.config.embedding_batch_size

        if self.embedding_manager:
            try:
                embeddings = await self.embedding_manager.generate_embeddings_batch(texts, batch_size)
                return embeddings
            except Exception as e:
                logger.error(f"Local batch embedding generation failed: {e}")
                if self.config.fallback_enabled and not self.embeddings_fallback:
                    self.embeddings_fallback = True

        raise RuntimeError("No embedding service available")

    # === Vector Store Operations ===

    async def add_vector(self, claim_id: str, content: str, 
                        embedding: List[float], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a vector to the store."""
        if not self.initialized:
            await self.initialize()

        if self.vector_store:
            try:
                return await self.vector_store.add_vector(claim_id, content, embedding, metadata)
            except Exception as e:
                logger.error(f"Failed to add vector: {e}")
                if self.config.fallback_enabled and not self.vector_store_fallback:
                    self.vector_store_fallback = True

        return False

    async def search_similar(self, query_embedding: List[float], 
                           limit: int = 10, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        if not self.initialized:
            await self.initialize()

        if self.vector_store:
            try:
                return await self.vector_store.search_similar(query_embedding, limit, threshold)
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                if self.config.fallback_enabled and not self.vector_store_fallback:
                    self.vector_store_fallback = True

        return []

    async def get_vector(self, claim_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific vector."""
        if not self.initialized:
            await self.initialize()

        if self.vector_store:
            return await self.vector_store.get_vector(claim_id)
        return None

    # === LLM Operations ===

    async def generate_response(self, prompt: str, 
                              model: Optional[str] = None,
                              system_prompt: Optional[str] = None) -> str:
        """Generate a response using local LLM."""
        if not self.initialized:
            await self.initialize()

        if self.use_mocks:
            # Mock response for testing
            return f"Mock response for: {prompt[:50]}..."

        if self.llm_client:
            try:
                response = await self.llm_client.generate_response(
                    prompt=prompt,
                    model=model,
                    system_prompt=system_prompt
                )
                if self.llm_fallback:
                    logger.info("Successfully used local LLM (was falling back)")
                    self.llm_fallback = False
                return response
            except Exception as e:
                logger.error(f"Local LLM generation failed: {e}")
                if self.config.fallback_enabled and not self.llm_fallback:
                    self.llm_fallback = True
                    logger.warning("Falling back to external LLM")

        raise RuntimeError("No LLM service available")

    async def check_llm_health(self) -> bool:
        """Check if LLM service is healthy."""
        if self.llm_client:
            return await self.llm_client.health_check()
        return False

    async def get_available_models(self) -> List[str]:
        """Get list of available local models."""
        if self.llm_client:
            models = await self.llm_client.get_available_models()
            return [model.name for model in models]
        return []

    # === Health and Diagnostics ===

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        if not self.initialized:
            return {'initialized': False, 'status': 'not_initialized'}

        await self._check_all_services_health()

        return {
            'initialized': True,
            'timestamp': datetime.utcnow().isoformat(),
            'services': self.health_status,
            'fallback_status': {
                'embeddings': self.embeddings_fallback,
                'vector_store': self.vector_store_fallback,
                'llm': self.llm_fallback
            },
            'overall_status': self._calculate_overall_health()
        }

    def _calculate_overall_health(self) -> str:
        """Calculate overall health status."""
        if not self.health_status:
            return 'unknown'

        statuses = [service.get('status', 'unknown') for service in self.health_status.values()]
        
        if all(status == 'healthy' for status in statuses):
            return 'healthy'
        elif any(status == 'unhealthy' for status in statuses):
            return 'degraded'
        else:
            return 'partial'

    async def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all services."""
        if not self.initialized:
            return {'initialized': False}

        stats = {
            'initialized': True,
            'timestamp': datetime.utcnow().isoformat(),
            'services': {},
            'configuration': {
                'embeddings': self.config.embedding_config,
                'vector_store': self.config.vector_store_config,
                'llm': self.config.llm_config
            }
        }

        # Embedding stats
        if self.embedding_manager:
            stats['services']['embeddings'] = self.embedding_manager.get_model_info()

        # Vector store stats
        if self.vector_store:
            stats['services']['vector_store'] = await self.vector_store.get_stats()

        # LLM stats
        if self.llm_client:
            stats['services']['llm'] = await self.llm_client.get_service_info()

        return stats

    async def start_health_monitoring(self) -> None:
        """Start background health monitoring."""
        if not self.config.health_check_interval:
            return

        async def monitor():
            while self.initialized:
                await asyncio.sleep(self.config.health_check_interval)
                try:
                    await self._check_all_services_health()
                except Exception as e:
                    logger.error(f"Health check failed: {e}")

        # Start monitoring task
        asyncio.create_task(monitor())
        logger.info(f"Health monitoring started (interval: {self.config.health_check_interval}s)")

    async def close(self) -> None:
        """Close all local services."""
        logger.info("Closing local services manager...")

        try:
            if self.embedding_manager:
                await self.embedding_manager.close()
            
            if self.vector_store:
                await self.vector_store.close()
            
            if self.llm_client:
                await self.llm_client.close()
            
            self.initialized = False
            logger.info("Local services manager closed")

        except Exception as e:
            logger.error(f"Error closing local services: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Factory function
async def create_local_manager(config: Optional[LocalConfig] = None, 
                              use_mocks: bool = False) -> LocalServicesManager:
    """
    Create and initialize a local services manager.
    
    Args:
        config: Configuration instance (uses defaults if None)
        use_mocks: Use mock services for testing
        
    Returns:
        Initialized LocalServicesManager
    """
    manager = LocalServicesManager(config, use_mocks)
    await manager.initialize()
    return manager


# Test function
async def test_local_services():
    """Test local services functionality."""
    print("Testing local services...")
    
    config = LocalConfig()
    
    try:
        async with await create_local_manager(config, use_mocks=True) as manager:
            # Test embeddings
            print("Testing embeddings...")
            embedding = await manager.generate_embedding("Hello world")
            print(f"Embedding generated: {len(embedding)} dimensions")
            
            # Test vector store
            print("Testing vector store...")
            await manager.add_vector("test-1", "Hello world", embedding, {"test": True})
            results = await manager.search_similar(embedding, limit=5)
            print(f"Search results: {len(results)} items")
            
            # Test LLM
            print("Testing LLM...")
            response = await manager.generate_response("Hello, how are you?")
            print(f"LLM response: {response}")
            
            # Health check
            health = await manager.health_check()
            print(f"Health status: {health['overall_status']}")
            
            print("Local services test completed successfully!")
    
    except Exception as e:
        print(f"Local services test failed: {e}")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    asyncio.run(test_local_services())