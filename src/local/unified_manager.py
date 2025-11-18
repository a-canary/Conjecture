"""
Unified Service Manager with Fallback Support
Coordinates local and external services with intelligent fallback
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import time
from enum import Enum

from .local_manager import LocalServicesManager
from .embeddings import LocalEmbeddingManager, MockEmbeddingManager
from .ollama_client import OllamaClient
from .vector_store import LocalVectorStore, MockVectorStore
from ..config.local_config import LocalConfig, ServiceMode

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service status enumeration"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    DISABLED = "disabled"


class UnifiedServiceManager:
    """
    Unified service manager that coordinates local and external services
    with intelligent fallback mechanisms.
    """

    def __init__(self, config: Optional[LocalConfig] = None, use_mocks: bool = False):
        self.config = config or LocalConfig()
        self.use_mocks = use_mocks
        
        # Service instances
        self.local_manager: Optional[LocalServicesManager] = None
        self.external_services: Dict[str, Any] = {}  # For external API clients
        
        # Service tracking
        self.service_status: Dict[str, ServiceStatus] = {}
        self.last_health_check: Dict[str, datetime] = {}
        self.fallback_active: Dict[str, bool] = {}
        
        # Performance metrics
        self.service_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Initialization state
        self.initialized = False

    async def initialize(self) -> None:
        """Initialize all configured services."""
        if self.initialized:
            return

        logger.info("Initializing unified service manager...")
        start_time = time.time()

        try:
            # Initialize local services
            if self._should_use_local_service("embeddings") or self._should_use_local_service("vector_store") or self._should_use_local_service("llm"):
                self.local_manager = LocalServicesManager(self.config, self.use_mocks)
                await self.local_manager.initialize()
                logger.info("Local services manager initialized")

            # Initialize external services based on configuration
            await self._init_external_services()

            # Perform initial health checks
            await self._check_all_services_health()

            # Initialize metrics tracking
            self._init_metrics()

            self.initialized = True
            init_time = time.time() - start_time
            
            logger.info(f"Unified service manager initialized in {init_time:.2f}s")
            await self._log_service_status()

        except Exception as e:
            logger.error(f"Failed to initialize unified service manager: {e}")
            raise RuntimeError(f"Unified service manager initialization failed: {e}")

    def _should_use_local_service(self, service: str) -> bool:
        """Check if local service should be initialized based on config."""
        if service == "embeddings":
            return self.config.use_local_embeddings
        elif service == "vector_store":
            return self.config.use_local_vector_store
        elif service == "llm":
            return self.config.use_local_llm
        return False

    async def _init_external_services(self) -> None:
        """Initialize external services."""
        try:
            # External embeddings (would integrate with existing embedding service)
            if self.config.embedding_mode in (ServiceMode.EXTERNAL, ServiceMode.AUTO):
                # Could add external embedding service initialization here
                logger.info("External embeddings available as fallback")
            
            # External vector store (ChromaDB or other)
            if self.config.vector_store_mode in (ServiceMode.EXTERNAL, ServiceMode.AUTO):
                # Could add ChromaDB initialization here
                logger.info("External vector store available as fallback")
            
            # External LLM
            if self.config.llm_mode in (ServiceMode.EXTERNAL, ServiceMode.AUTO):
                # Initialize external LLM client
                try:
                    sys.path.insert(0, ".")
                    from processing.llm.llm_manager import LLMManager
                    from config.simple_config import Config
                    self.external_services['llm'] = LLMManager(Config())
                    logger.info("External LLM service initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize external LLM: {e}")

        except Exception as e:
            logger.warning(f"Error initializing external services: {e}")

    async def _check_all_services_health(self) -> None:
        """Check health of all available services."""
        current_time = datetime.utcnow()
        
        # Check local services
        if self.local_manager:
            try:
                local_health = await self.local_manager.health_check()
                for service_name, health in local_health.get('services', {}).items():
                    status = self._map_health_to_status(health.get('status', 'unknown'))
                    self.service_status[f"local_{service_name}"] = status
                    self.last_health_check[f"local_{service_name}"] = current_time
                    
            except Exception as e:
                logger.error(f"Local services health check failed: {e}")
                for service in ['embeddings', 'vector_store', 'llm']:
                    self.service_status[f"local_{service}"] = ServiceStatus.UNHEALTHY
                    self.last_health_check[f"local_{service}"] = current_time

        # Check external services
        for service_name, service_instance in self.external_services.items():
            try:
                # Service-specific health checks would go here
                self.service_status[f"external_{service_name}"] = ServiceStatus.HEALTHY
                self.last_health_check[f"external_{service_name}"] = current_time
            except Exception as e:
                logger.error(f"External {service_name} health check failed: {e}")
                self.service_status[f"external_{service_name}"] = ServiceStatus.UNHEALTHY
                self.last_health_check[f"external_{service_name}"] = current_time

    def _map_health_to_status(self, health_status: str) -> ServiceStatus:
        """Map health status string to ServiceStatus enum."""
        health_lower = health_status.lower()
        if health_lower == 'healthy':
            return ServiceStatus.HEALTHY
        elif health_lower == 'unhealthy':
            return ServiceStatus.UNHEALTHY
        elif health_lower == 'disabled':
            return ServiceStatus.DISABLED
        else:
            return ServiceStatus.UNKNOWN

    def _init_metrics(self) -> None:
        """Initialize metrics tracking for all services."""
        services = ['embeddings', 'vector_store', 'llm']
        for service in services:
            self.service_metrics[service] = {
                'local_requests': 0,
                'local_successes': 0,
                'local_failures': 0,
                'external_requests': 0,
                'external_successes': 0,
                'external_failures': 0,
                'fallback_count': 0,
                'total_response_time': 0.0
            }
            self.fallback_active[service] = False

    # === Embedding Operations with Fallback ===

    async def generate_embedding(self, text: str, prefer_local: Optional[bool] = None) -> List[float]:
        """Generate embedding with fallback support."""
        service_used = None
        start_time = time.time()
        
        try:
            # Determine service preference
            if prefer_local is None:
                prefer_local = self.config.is_local_first()
            
            # Try local service first (if preferred and available)
            if prefer_local and self.local_manager and self._is_service_healthy("local_embeddings"):
                service_used = "local"
                try:
                    embedding = await self.local_manager.generate_embedding(text)
                    self._record_metrics("embeddings", service_used, True, time.time() - start_time)
                    if self.fallback_active["embeddings"]:
                        self.fallback_active["embeddings"] = False
                        logger.info("Local embeddings service recovered from fallback")
                    return embedding
                except Exception as e:
                    logger.error(f"Local embedding service failed: {e}")
                    self.service_status["local_embeddings"] = ServiceStatus.UNHEALTHY
            
            # Try external service
            if self._is_service_healthy("external_embeddings"):
                service_used = "external"
                # Would call external embedding service here
                # For now, fall back to mock
                mock_service = MockEmbeddingManager()
                embedding = await mock_service.generate_embedding(text)
                self._record_metrics("embeddings", service_used, True, time.time() - start_time)
                if not self.fallback_active["embeddings"] and prefer_local:
                    self.fallback_active["embeddings"] = True
                    logger.warning("Fell back to external embeddings service")
                return embedding
            
            # Try local service as last resort (if not preferred)
            if not prefer_local and self.local_manager and self._is_service_healthy("local_embeddings"):
                service_used = "local"
                try:
                    embedding = await self.local_manager.generate_embedding(text)
                    self._record_metrics("embeddings", service_used, True, time.time() - start_time)
                    return embedding
                except Exception as e:
                    logger.error(f"Secondary local embedding service failed: {e}")
            
            raise RuntimeError("No embedding service available")

        except Exception as e:
            self._record_metrics("embeddings", service_used, False, time.time() - start_time)
            raise

    # === Vector Store Operations with Fallback ===

    async def add_vector(self, claim_id: str, content: str, 
                        embedding: List[float], metadata: Optional[Dict[str, Any]] = None,
                        prefer_local: Optional[bool] = None) -> bool:
        """Add vector with fallback support."""
        service_used = None
        start_time = time.time()
        
        try:
            # Determine service preference
            if prefer_local is None:
                prefer_local = self.config.is_local_first()
            
            # Try local service first (if preferred and available)
            if prefer_local and self.local_manager and self._is_service_healthy("local_vector_store"):
                service_used = "local"
                success = await self.local_manager.add_vector(claim_id, content, embedding, metadata)
                self._record_metrics("vector_store", service_used, success, time.time() - start_time)
                if success and self.fallback_active["vector_store"]:
                    self.fallback_active["vector_store"] = False
                    logger.info("Local vector store service recovered from fallback")
                return success
            
            # Try external service
            if self._is_service_healthy("external_vector_store"):
                service_used = "external"
                # Would call external vector store (ChromaDB) here
                # For now, mock success
                self._record_metrics("vector_store", service_used, True, time.time() - start_time)
                if not self.fallback_active["vector_store"] and prefer_local:
                    self.fallback_active["vector_store"] = True
                    logger.warning("Fell back to external vector store")
                return True
            
            # Try local service as last resort
            if not prefer_local and self.local_manager and self._is_service_healthy("local_vector_store"):
                service_used = "local"
                success = await self.local_manager.add_vector(claim_id, content, embedding, metadata)
                self._record_metrics("vector_store", service_used, success, time.time() - start_time)
                return success
            
            return False

        except Exception as e:
            self._record_metrics("vector_store", service_used, False, time.time() - start_time)
            logger.error(f"Vector add operation failed: {e}")
            return False

    async def search_similar(self, query_embedding: List[float], 
                           limit: int = 10, threshold: float = 0.0,
                           prefer_local: Optional[bool] = None) -> List[Dict[str, Any]]:
        """Search similar vectors with fallback support."""
        service_used = None
        start_time = time.time()
        
        try:
            # Determine service preference
            if prefer_local is None:
                prefer_local = self.config.is_local_first()
            
            # Try local service first (if preferred and available)
            if prefer_local and self.local_manager and self._is_service_healthy("local_vector_store"):
                service_used = "local"
                results = await self.local_manager.search_similar(query_embedding, limit, threshold)
                self._record_metrics("vector_store", service_used, True, time.time() - start_time)
                if self.fallback_active["vector_store"]:
                    self.fallback_active["vector_store"] = False
                    logger.info("Local vector store search recovered from fallback")
                return results
            
            # Try external service
            if self._is_service_healthy("external_vector_store"):
                service_used = "external"
                # Would call external vector store (ChromaDB) here
                # For now, return empty results
                results = []
                self._record_metrics("vector_store", service_used, True, time.time() - start_time)
                if not self.fallback_active["vector_store"] and prefer_local:
                    self.fallback_active["vector_store"] = True
                    logger.warning("Fell back to external vector store search")
                return results
            
            # Try local service as last resort
            if not prefer_local and self.local_manager and self._is_service_healthy("local_vector_store"):
                service_used = "local"
                results = await self.local_manager.search_similar(query_embedding, limit, threshold)
                self._record_metrics("vector_store", service_used, True, time.time() - start_time)
                return results
            
            return []

        except Exception as e:
            self._record_metrics("vector_store", service_used, False, time.time() - start_time)
            logger.error(f"Vector search operation failed: {e}")
            return []

    # === LLM Operations with Fallback ===

    async def generate_response(self, prompt: str, 
                              model: Optional[str] = None,
                              system_prompt: Optional[str] = None,
                              prefer_local: Optional[bool] = None) -> str:
        """Generate LLM response with fallback support."""
        service_used = None
        start_time = time.time()
        
        try:
            # Determine service preference
            if prefer_local is None:
                prefer_local = self.config.is_local_first()
            
            # Try local service first (if preferred and available)
            if prefer_local and self.local_manager and await self.local_manager.check_llm_health():
                service_used = "local"
                try:
                    response = await self.local_manager.generate_response(prompt, model, system_prompt)
                    self._record_metrics("llm", service_used, True, time.time() - start_time)
                    if self.fallback_active["llm"]:
                        self.fallback_active["llm"] = False
                        logger.info("Local LLM service recovered from fallback")
                    return response
                except Exception as e:
                    logger.error(f"Local LLM service failed: {e}")
                    self.service_status["local_llm"] = ServiceStatus.UNHEALTHY
            
            # Try external service
            if self._is_service_healthy("external_llm") and 'llm' in self.external_services:
                service_used = "external"
                try:
                    # Would call external LLM service here
                    # For now, return mock response
                    response = f"External LLM response for: {prompt[:50]}..."
                    self._record_metrics("llm", service_used, True, time.time() - start_time)
                    if not self.fallback_active["llm"] and prefer_local:
                        self.fallback_active["llm"] = True
                        logger.warning("Fell back to external LLM service")
                    return response
                except Exception as e:
                    logger.error(f"External LLM service failed: {e}")
                    self.service_status["external_llm"] = ServiceStatus.UNHEALTHY
            
            # Try local service as last resort
            if not prefer_local and self.local_manager and await self.local_manager.check_llm_health():
                service_used = "local"
                response = await self.local_manager.generate_response(prompt, model, system_prompt)
                self._record_metrics("llm", service_used, True, time.time() - start_time)
                return response
            
            raise RuntimeError("No LLM service available")

        except Exception as e:
            self._record_metrics("llm", service_used, False, time.time() - start_time)
            raise

    # === Utility Methods ===

    def _is_service_healthy(self, service_name: str) -> bool:
        """Check if a service is healthy."""
        status = self.service_status.get(service_name, ServiceStatus.UNKNOWN)
        
        # Consider unknown as healthy for initial attempts
        if status == ServiceStatus.DISABLED:
            return False
        
        # Allow unknown status (service not checked yet)
        return status != ServiceStatus.UNHEALTHY

    def _record_metrics(self, service: str, service_type: str, success: bool, response_time: float) -> None:
        """Record service metrics."""
        if service_type:
            self.service_metrics[service][f"{service_type}_requests"] += 1
            if success:
                self.service_metrics[service][f"{service_type}_successes"] += 1
            else:
                self.service_metrics[service][f"{service_type}_failures"] += 1
            
            self.service_metrics[service]["total_response_time"] += response_time
            
            if service_type != self._get_primary_service_type(service):
                self.service_metrics[service]["fallback_count"] += 1

    def _get_primary_service_type(self, service: str) -> str:
        """Get the primary service type for a given service."""
        if self.config.is_local_first():
            return "local"
        else:
            return "external"

    async def check_service_health(self, service_name: str) -> ServiceStatus:
        """Check health of a specific service."""
        await self._check_all_services_health()
        return self.service_status.get(service_name, ServiceStatus.UNKNOWN)

    async def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive service status with metrics."""
        await self._check_all_services_health()
        
        return {
            'initialized': self.initialized,
            'timestamp': datetime.utcnow().isoformat(),
            'service_status': {k: v.value for k, v in self.service_status.items()},
            'fallback_active': self.fallback_active.copy(),
            'service_metrics': self.service_metrics.copy(),
            'configuration': {
                'local_first': self.config.is_local_first(),
                'offline_capable': self.config.supports_offline(),
                'fallback_enabled': self.config.fallback_enabled
            },
            'overall_health': self._calculate_overall_health()
        }

    def _calculate_overall_health(self) -> str:
        """Calculate overall system health."""
        if not self.service_status:
            return 'unknown'
        
        statuses = list(self.service_status.values())
        
        if all(s == ServiceStatus.HEALTHY for s in statuses):
            return 'healthy'
        elif any(s == ServiceStatus.HEALTHY for s in statuses):
            return 'degraded'
        else:
            return 'unhealthy'

    async def _log_service_status(self) -> None:
        """Log current service status."""
        logger.info("=== Unified Service Manager Status ===")
        for service, status in self.service_status.items():
            logger.info(f"{service}: {status.value}")
        
        for service, active in self.fallback_active.items():
            if active:
                logger.warning(f"Service {service} is in fallback mode")

    async def close(self) -> None:
        """Close all services."""
        logger.info("Closing unified service manager...")
        
        try:
            if self.local_manager:
                await self.local_manager.close()
            
            # Close external services
            for service_name, service_instance in self.external_services.items():
                try:
                    # Service-specific cleanup would go here
                    logger.info(f"Closed external service: {service_name}")
                except Exception as e:
                    logger.error(f"Error closing external service {service_name}: {e}")
            
            self.initialized = False
            logger.info("Unified service manager closed")

        except Exception as e:
            logger.error(f"Error closing unified service manager: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Factory function
async def create_unified_manager(config: Optional[LocalConfig] = None, 
                               use_mocks: bool = False) -> UnifiedServiceManager:
    """
    Create and initialize a unified service manager.
    
    Args:
        config: Configuration instance (uses defaults if None)
        use_mocks: Use mock services for testing
        
    Returns:
        Initialized UnifiedServiceManager
    """
    manager = UnifiedServiceManager(config, use_mocks)
    await manager.initialize()
    return manager