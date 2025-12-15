"""
Enhanced LLM Router for Conjecture EndPoint App
Provides intelligent routing, health monitoring, and failover for multiple LLM providers
"""

import asyncio
import time
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from .llm.openai_compatible_provider import OpenAICompatibleProcessor, create_openai_compatible_processor
from .llm.common import GenerationConfig, LLMProcessingResult
from src.config.unified_config import get_config
from src.config.settings_models import ProviderConfig
from src.core.models import Claim

class ProviderStatus(str, Enum):
    """Provider status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DISABLED = "disabled"

class RoutingStrategy(str, Enum):
    """Routing strategy enumeration"""
    PRIORITY = "priority"  # Use highest priority available provider
    ROUND_ROBIN = "round_robin"  # Rotate through available providers
    LOAD_BALANCED = "load_balanced"  # Route to least loaded provider
    FASTEST_RESPONSE = "fastest_response"  # Route to fastest responding provider

@dataclass
class ProviderMetrics:
    """Metrics for a provider"""
    provider_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    last_success_time: Optional[datetime] = None
    last_failure_time: Optional[datetime] = None
    consecutive_failures: int = 0
    average_response_time: float = 0.0
    success_rate: float = 0.0
    last_health_check: Optional[datetime] = None
    status: ProviderStatus = ProviderStatus.UNHEALTHY
    current_load: int = 0  # Number of concurrent requests
    
    def update_success(self, response_time: float):
        """Update metrics after successful request"""
        self.total_requests += 1
        self.successful_requests += 1
        self.total_response_time += response_time
        self.last_success_time = datetime.utcnow()
        self.consecutive_failures = 0
        self._calculate_derived_metrics()
    
    def update_failure(self):
        """Update metrics after failed request"""
        self.total_requests += 1
        self.failed_requests += 1
        self.last_failure_time = datetime.utcnow()
        self.consecutive_failures += 1
        self._calculate_derived_metrics()
    
    def _calculate_derived_metrics(self):
        """Calculate derived metrics"""
        if self.total_requests > 0:
            self.success_rate = self.successful_requests / self.total_requests
            self.average_response_time = self.total_response_time / self.total_requests

@dataclass
class ProviderConfig:
    """Enhanced provider configuration"""
    name: str
    url: str
    api_key: str = ""
    model: str = "gpt-3.5-turbo"
    priority: int = 999
    max_retries: int = 3
    timeout: int = 60
    is_local: bool = False
    enabled: bool = True
    routing_strategy: RoutingStrategy = RoutingStrategy.PRIORITY
    max_concurrent_requests: int = 10
    health_check_interval: int = 300  # seconds
    health_check_timeout: int = 10  # seconds
    
    # Provider-specific settings
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "url": self.url,
            "api_key": self.api_key,
            "model": self.model,
            "priority": self.priority,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "is_local": self.is_local,
            "enabled": self.enabled,
            "routing_strategy": self.routing_strategy.value,
            "max_concurrent_requests": self.max_concurrent_requests,
            "health_check_interval": self.health_check_interval,
            "health_check_timeout": self.health_check_timeout,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p
        }

class EnhancedLLMRouter:
    """Enhanced LLM router with intelligent routing and health monitoring"""
    
    def __init__(self, providers: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize enhanced LLM router
        
        Args:
            providers: List of provider configurations. If None, loads from config.
        """
        self.providers: Dict[str, OpenAICompatibleProcessor] = {}
        self.provider_configs: Dict[str, ProviderConfig] = {}
        self.provider_metrics: Dict[str, ProviderMetrics] = {}
        self.routing_strategy = RoutingStrategy.PRIORITY
        self.round_robin_index = 0
        self.failed_providers = set()
        self.health_check_task = None
        self.logger = logging.getLogger(__name__)
        
        if providers is None:
            providers = self._load_providers_from_config()
        
        self._initialize_providers(providers)
        self._start_health_monitoring()
    
    def _load_providers_from_config(self) -> List[Dict[str, Any]]:
        """Load provider configurations from unified config"""
        try:
            config = get_config()
            providers_data = []
            
            # Handle both old format (dict) and new format (list)
            if isinstance(config.providers, dict):
                for name, provider_data in config.providers.items():
                    provider_dict = {
                        "name": name,
                        "url": provider_data.get("url", ""),
                        "api_key": provider_data.get("key", provider_data.get("api", "")),
                        "model": provider_data.get("model", ""),
                        "priority": provider_data.get("priority", 999)
                    }
                    providers_data.append(provider_dict)
            else:
                # New format - list of providers
                for provider_data in config.providers:
                    providers_data.append(provider_data)
            
            return providers_data
            
        except Exception as e:
            self.logger.error(f"Failed to load providers from config: {e}")
            return []
    
    def _initialize_providers(self, providers: List[ProviderConfig]):
        """Initialize providers with enhanced configuration"""
        self.logger.info(f"Initializing {len(providers)} providers...")
        
        for provider_config in providers:
            try:
                # Use the provider config directly
                config = provider_config
                
                # Create processor
                processor = create_openai_compatible_processor(
                    provider_name=config.name,
                    api_url=config.url,
                    api_key=config.api,
                    model=config.model
                )
                
                self.providers[config.name] = processor
                self.provider_configs[config.name] = config
                self.provider_metrics[config.name] = ProviderMetrics(provider_name=config.name)
                
                self.logger.info(f"Initialized provider: {config.name} (priority: {config.priority})")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize provider {provider_config.name}: {e}")
        
        if not self.providers:
            self.logger.warning("No providers initialized")
        else:
            self.logger.info(f"Successfully initialized {len(self.providers)} providers")
    
    def _is_local_provider(self, url: str) -> bool:
        """Check if provider is local"""
        local_indicators = ["localhost", "127.0.0.1", "0.0.0.0"]
        return any(indicator in url.lower() for indicator in local_indicators)
    
    def _start_health_monitoring(self):
        """Start background health monitoring if in async context"""
        try:
            # Only start if we're in an async context with an event loop
            if self.health_check_task is None and asyncio.get_running_loop():
                self.health_check_task = asyncio.create_task(self._health_monitoring_loop())
                self.logger.info("Health monitoring started")
        except RuntimeError:
            # No event loop running - health monitoring will be started manually
            self.logger.info("Health monitoring deferred (no event loop)")
        except Exception as e:
            self.logger.error(f"Failed to start health monitoring: {e}")
    
    async def start_health_monitoring_async(self):
        """Manually start health monitoring in async context"""
        if self.health_check_task is None:
            self.health_check_task = asyncio.create_task(self._health_monitoring_loop())
            self.logger.info("Health monitoring started manually")
    
    async def _health_monitoring_loop(self):
        """Background health monitoring loop"""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _perform_health_checks(self):
        """Perform health checks on all providers"""
        current_time = datetime.utcnow()
        
        for provider_name, config in self.provider_configs.items():
            if not config.enabled:
                continue
            
            metrics = self.provider_metrics[provider_name]
            
            # Check if health check is due
            if (metrics.last_health_check and 
                (current_time - metrics.last_health_check).seconds < config.health_check_interval):
                continue
            
            try:
                # Perform health check
                processor = self.providers[provider_name]
                health_result = await self._async_health_check(processor, config.health_check_timeout)
                
                if health_result["status"] == "healthy":
                    metrics.status = ProviderStatus.HEALTHY
                    if provider_name in self.failed_providers:
                        self.failed_providers.remove(provider_name)
                        self.logger.info(f"Provider {provider_name} recovered")
                else:
                    metrics.status = ProviderStatus.UNHEALTHY
                    self.failed_providers.add(provider_name)
                    self.logger.warning(f"Provider {provider_name} unhealthy: {health_result.get('error')}")
                
                metrics.last_health_check = current_time
                
            except Exception as e:
                metrics.status = ProviderStatus.UNHEALTHY
                self.failed_providers.add(provider_name)
                metrics.last_health_check = current_time
                self.logger.error(f"Health check failed for {provider_name}: {e}")
    
    async def _async_health_check(self, processor: OpenAICompatibleProcessor, timeout: int) -> Dict[str, Any]:
        """Perform async health check on provider"""
        try:
            # Run health check in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            health_result = await loop.run_in_executor(
                None, processor.health_check
            )
            return health_result
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def _select_provider(self, preferred_provider: Optional[str] = None) -> Optional[str]:
        """Select provider based on routing strategy"""
        available_providers = [
            name for name in self.providers.keys()
            if (name not in self.failed_providers and 
                self.provider_configs[name].enabled and
                self.provider_metrics[name].current_load < self.provider_configs[name].max_concurrent_requests)
        ]
        
        if not available_providers:
            return None
        
        # If specific provider requested and available, use it
        if preferred_provider and preferred_provider in available_providers:
            return preferred_provider
        
        # Apply routing strategy
        if self.routing_strategy == RoutingStrategy.PRIORITY:
            return self._select_by_priority(available_providers)
        elif self.routing_strategy == RoutingStrategy.ROUND_ROBIN:
            return self._select_by_round_robin(available_providers)
        elif self.routing_strategy == RoutingStrategy.LOAD_BALANCED:
            return self._select_by_load(available_providers)
        elif self.routing_strategy == RoutingStrategy.FASTEST_RESPONSE:
            return self._select_by_fastest_response(available_providers)
        else:
            # Default to priority
            return self._select_by_priority(available_providers)
    
    def _select_by_priority(self, available_providers: List[str]) -> str:
        """Select provider by priority (lowest number = highest priority)"""
        sorted_providers = sorted(
            available_providers,
            key=lambda name: self.provider_configs[name].priority
        )
        return sorted_providers[0]
    
    def _select_by_round_robin(self, available_providers: List[str]) -> str:
        """Select provider using round-robin"""
        if self.round_robin_index >= len(available_providers):
            self.round_robin_index = 0
        
        provider = available_providers[self.round_robin_index]
        self.round_robin_index = (self.round_robin_index + 1) % len(available_providers)
        return provider
    
    def _select_by_load(self, available_providers: List[str]) -> str:
        """Select provider with lowest current load"""
        sorted_providers = sorted(
            available_providers,
            key=lambda name: self.provider_metrics[name].current_load
        )
        return sorted_providers[0]
    
    def _select_by_fastest_response(self, available_providers: List[str]) -> str:
        """Select provider with fastest average response time"""
        # Filter providers that have successful requests
        providers_with_stats = [
            name for name in available_providers
            if self.provider_metrics[name].average_response_time > 0
        ]
        
        if providers_with_stats:
            sorted_providers = sorted(
                providers_with_stats,
                key=lambda name: self.provider_metrics[name].average_response_time
            )
            return sorted_providers[0]
        else:
            # Fallback to priority if no stats available
            return self._select_by_priority(available_providers)
    
    async def generate_response(
        self,
        prompt: str,
        preferred_provider: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> LLMProcessingResult:
        """Generate response with intelligent routing and failover"""
        attempted_providers = set()
        
        while True:
            provider_name = self._select_provider(preferred_provider)
            if not provider_name:
                raise RuntimeError("No providers available")
            
            if provider_name in attempted_providers:
                raise RuntimeError("All available providers failed")
            
            attempted_providers.add(provider_name)
            
            try:
                # Update load
                self.provider_metrics[provider_name].current_load += 1
                
                # Get provider and config
                processor = self.providers[provider_name]
                provider_config = self.provider_configs[provider_name]
                
                # Merge generation configs
                merged_config = self._merge_generation_configs(config, provider_config)
                
                # Generate response
                start_time = time.time()
                result = await self._async_generate_response(processor, prompt, merged_config)
                response_time = time.time() - start_time
                
                # Update metrics
                if result.success:
                    self.provider_metrics[provider_name].update_success(response_time)
                    self.logger.debug(f"Response generated successfully by {provider_name} in {response_time:.2f}s")
                else:
                    self.provider_metrics[provider_name].update_failure()
                    self.failed_providers.add(provider_name)
                    self.logger.warning(f"Response generation failed for {provider_name}")
                
                return result
                
            except Exception as e:
                self.provider_metrics[provider_name].update_failure()
                self.failed_providers.add(provider_name)
                self.logger.error(f"Provider {provider_name} failed: {e}")
                
                # If this was a preferred provider, allow fallback
                if preferred_provider:
                    preferred_provider = None
                    continue
                
                # Try next provider
                if len(attempted_providers) < len(self.providers):
                    continue
                else:
                    raise
            finally:
                # Update load
                self.provider_metrics[provider_name].current_load -= 1
    
    async def _async_generate_response(
        self, processor: OpenAICompatibleProcessor, prompt: str, config: GenerationConfig
    ) -> LLMProcessingResult:
        """Generate response asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, processor.generate_response, prompt, config
        )
    
    def _merge_generation_configs(
        self, request_config: Optional[GenerationConfig], provider_config: ProviderConfig
    ) -> GenerationConfig:
        """Merge request and provider generation configs"""
        # Start with request config or defaults
        if request_config:
            merged = GenerationConfig(
                temperature=request_config.temperature,
                max_tokens=request_config.max_tokens,
                top_p=request_config.top_p,
                frequency_penalty=getattr(request_config, 'frequency_penalty', None),
                presence_penalty=getattr(request_config, 'presence_penalty', None),
                stop_sequences=getattr(request_config, 'stop_sequences', None),
                stream=getattr(request_config, 'stream', False)
            )
        else:
            merged = GenerationConfig()
        
        # Override with provider-specific settings
        if provider_config.temperature is not None:
            merged.temperature = provider_config.temperature
        if provider_config.max_tokens is not None:
            merged.max_tokens = provider_config.max_tokens
        if provider_config.top_p is not None:
            merged.top_p = provider_config.top_p
        
        return merged
    
    async def process_claims(
        self,
        claims: List[Claim],
        task: str = "analyze",
        preferred_provider: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> LLMProcessingResult:
        """Process claims with intelligent routing and failover"""
        attempted_providers = set()
        
        while True:
            provider_name = self._select_provider(preferred_provider)
            if not provider_name:
                raise RuntimeError("No providers available")
            
            if provider_name in attempted_providers:
                raise RuntimeError("All available providers failed")
            
            attempted_providers.add(provider_name)
            
            try:
                # Update load
                self.provider_metrics[provider_name].current_load += 1
                
                # Get provider and config
                processor = self.providers[provider_name]
                provider_config = self.provider_configs[provider_name]
                
                # Merge generation configs
                merged_config = self._merge_generation_configs(config, provider_config)
                
                # Process claims
                start_time = time.time()
                result = await self._async_process_claims(processor, claims, task, merged_config)
                response_time = time.time() - start_time
                
                # Update metrics
                if result.success:
                    self.provider_metrics[provider_name].update_success(response_time)
                    self.logger.debug(f"Claims processed successfully by {provider_name} in {response_time:.2f}s")
                else:
                    self.provider_metrics[provider_name].update_failure()
                    self.failed_providers.add(provider_name)
                    self.logger.warning(f"Claims processing failed for {provider_name}")
                
                return result
                
            except Exception as e:
                self.provider_metrics[provider_name].update_failure()
                self.failed_providers.add(provider_name)
                self.logger.error(f"Provider {provider_name} failed: {e}")
                
                # If this was a preferred provider, allow fallback
                if preferred_provider:
                    preferred_provider = None
                    continue
                
                # Try next provider
                if len(attempted_providers) < len(self.providers):
                    continue
                else:
                    raise
            finally:
                # Update load
                self.provider_metrics[provider_name].current_load -= 1
    
    async def _async_process_claims(
        self, processor: OpenAICompatibleProcessor, claims: List[Claim], task: str, config: GenerationConfig
    ) -> LLMProcessingResult:
        """Process claims asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, processor.process_claims, claims, task, config
        )
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get comprehensive provider status"""
        status = {
            "total_providers": len(self.providers),
            "enabled_providers": len([p for p in self.provider_configs.values() if p.enabled]),
            "healthy_providers": len([m for m in self.provider_metrics.values() if m.status == ProviderStatus.HEALTHY]),
            "failed_providers": list(self.failed_providers),
            "routing_strategy": self.routing_strategy.value,
            "providers": {}
        }
        
        for name, config in self.provider_configs.items():
            metrics = self.provider_metrics[name]
            status["providers"][name] = {
                "enabled": config.enabled,
                "priority": config.priority,
                "status": metrics.status.value,
                "current_load": metrics.current_load,
                "max_concurrent_requests": config.max_concurrent_requests,
                "total_requests": metrics.total_requests,
                "success_rate": metrics.success_rate,
                "average_response_time": metrics.average_response_time,
                "consecutive_failures": metrics.consecutive_failures,
                "last_success_time": metrics.last_success_time.isoformat() if metrics.last_success_time else None,
                "last_failure_time": metrics.last_failure_time.isoformat() if metrics.last_failure_time else None,
                "last_health_check": metrics.last_health_check.isoformat() if metrics.last_health_check else None
            }
        
        return status
    
    def get_provider_metrics(self) -> Dict[str, Any]:
        """Get detailed provider metrics"""
        metrics = {}
        
        for name, provider_metrics in self.provider_metrics.items():
            config = self.provider_configs[name]
            metrics[name] = {
                "provider_name": provider_metrics.provider_name,
                "total_requests": provider_metrics.total_requests,
                "successful_requests": provider_metrics.successful_requests,
                "failed_requests": provider_metrics.failed_requests,
                "success_rate": provider_metrics.success_rate,
                "average_response_time": provider_metrics.average_response_time,
                "total_response_time": provider_metrics.total_response_time,
                "current_load": provider_metrics.current_load,
                "consecutive_failures": provider_metrics.consecutive_failures,
                "status": provider_metrics.status.value,
                "enabled": config.enabled,
                "priority": config.priority,
                "is_local": config.is_local,
                "model": config.model,
                "url": config.url
            }
        
        return metrics
    
    def set_routing_strategy(self, strategy: RoutingStrategy):
        """Set routing strategy"""
        self.routing_strategy = strategy
        self.logger.info(f"Routing strategy changed to: {strategy.value}")
    
    def enable_provider(self, provider_name: str) -> bool:
        """Enable a provider"""
        if provider_name in self.provider_configs:
            self.provider_configs[provider_name].enabled = True
            if provider_name in self.failed_providers:
                self.failed_providers.remove(provider_name)
            self.logger.info(f"Provider {provider_name} enabled")
            return True
        return False
    
    def disable_provider(self, provider_name: str) -> bool:
        """Disable a provider"""
        if provider_name in self.provider_configs:
            self.provider_configs[provider_name].enabled = False
            self.failed_providers.add(provider_name)
            self.logger.info(f"Provider {provider_name} disabled")
            return True
        return False
    
    def reset_provider_metrics(self, provider_name: Optional[str] = None):
        """Reset metrics for specified provider or all providers"""
        if provider_name:
            if provider_name in self.provider_metrics:
                self.provider_metrics[provider_name] = ProviderMetrics(provider_name=provider_name)
                self.logger.info(f"Metrics reset for provider {provider_name}")
        else:
            for name in self.provider_metrics:
                self.provider_metrics[name] = ProviderMetrics(provider_name=name)
            self.logger.info("All provider metrics reset")
    
    async def shutdown(self):
        """Shutdown the router and cleanup resources"""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Enhanced LLM router shutdown complete")

# Global instance for easy access
_enhanced_llm_router = None

def get_enhanced_llm_router() -> EnhancedLLMRouter:
    """Get global enhanced LLM router instance"""
    global _enhanced_llm_router
    if _enhanced_llm_router is None:
        _enhanced_llm_router = EnhancedLLMRouter()
    return _enhanced_llm_router