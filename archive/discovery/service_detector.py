"""
Service Detection Module

Detects available LLM providers through:
1. Local service scanning (Ollama, LM Studio)
2. Environment variable scanning for API keys
3. Service availability and model enumeration
"""

import os
import asyncio
import aiohttp
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from urllib.parse import urljoin
import logging

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class DetectedProvider:
    """Represents a detected LLM provider"""
    name: str
    type: str  # 'local' or 'cloud'
    endpoint: Optional[str] = None
    api_key_env_var: Optional[str] = None
    models: List[str] = None
    priority: int = 0  # Lower number = higher priority
    status: str = 'available'  # 'available', 'unavailable', 'error'
    info: Dict[str, Any] = None

    def __post_init__(self):
        if self.models is None:
            self.models = []
        if self.info is None:
            self.info = {}

class ServiceDetector:
    """Detects available LLM services and providers"""
    
    def __init__(self, timeout: int = 3):
        self.timeout = timeout
        self.session = None
        
        # Provider configurations
        self.local_providers = {
            'ollama': {
                'endpoints': ['http://localhost:11434'],
                'models_path': '/api/tags',
                'health_path': '/api/tags',
                'priority': 1
            },
            'lm_studio': {
                'endpoints': ['http://localhost:1234'],
                'models_path': '/v1/models',
                'health_path': '/v1/models',
                'priority': 2
            }
        }
        
        # Cloud provider API key patterns
        self.cloud_providers = {
            'openai': {
                'env_vars': ['OPENAI_API_KEY'],
                'key_pattern': r'^sk-[A-Za-z0-9]{48}$',
                'models_endpoint': 'https://api.openai.com/v1/models',
                'priority': 10
            },
            'anthropic': {
                'env_vars': ['ANTHROPIC_API_KEY'],
                'key_pattern': r'^sk-ant-api[0-9]{2}-[A-Za-z0-9_-]{95}$',
                'models_endpoint': None,  # No public models endpoint
                'priority': 11
            },
            'google': {
                'env_vars': ['GOOGLE_API_KEY', 'GEMINI_API_KEY'],
                'key_pattern': r'^[A-Za-z0-9_-]{39}$',
                'models_endpoint': None,
                'priority': 12
            },
            'chutes': {
                'env_vars': ['CHUTES_API_KEY', 'Conjecture_LLM_API_KEY'],
                'key_pattern': r'^[A-Za-z0-9_-]{20,}$',
                'models_endpoint': None,
                'priority': 5
            },
            'openrouter': {
                'env_vars': ['OPENROUTER_API_KEY'],
                'key_pattern': r'^sk-or-v1-[A-Za-z0-9]{48}$',
                'models_endpoint': 'https://openrouter.ai/api/v1/models',
                'priority': 13
            }
        }

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def detect_all(self) -> List[DetectedProvider]:
        """Detect all available providers"""
        detected = []
        
        # Detect local services first (preferred)
        local_providers = await self.detect_local_services()
        detected.extend(local_providers)
        
        # Detect cloud services via API keys
        cloud_providers = await self.detect_cloud_services()
        detected.extend(cloud_providers)
        
        # Sort by priority
        detected.sort(key=lambda p: p.priority)
        
        return detected

    async def detect_local_services(self) -> List[DetectedProvider]:
        """Detect local LLM services"""
        providers = []
        
        for service_name, config in self.local_providers.items():
            for endpoint in config['endpoints']:
                provider = await self._detect_local_service(
                    service_name, endpoint, config
                )
                if provider:
                    providers.append(provider)
                    break  # Found working endpoint for this service
        
        return providers

    async def _detect_local_service(
        self, 
        service_name: str, 
        endpoint: str, 
        config: Dict
    ) -> Optional[DetectedProvider]:
        """Detect a specific local service"""
        try:
            # Check health
            health_url = urljoin(endpoint, config['health_path'])
            async with self.session.get(health_url) as response:
                if response.status != 200:
                    logger.debug(f"{service_name} at {endpoint} returned {response.status}")
                    return None
                
                # Get models
                models_url = urljoin(endpoint, config['models_path'])
                models = await self._get_local_models(models_url, service_name)
                
                return DetectedProvider(
                    name=service_name.title(),
                    type='local',
                    endpoint=endpoint,
                    models=models,
                    priority=config['priority'],
                    status='available',
                    info={'service_type': service_name, 'response_time': 'fast'}
                )
                
        except Exception as e:
            logger.debug(f"Failed to detect {service_name} at {endpoint}: {e}")
            return None

    async def _get_local_models(self, models_url: str, service_name: str) -> List[str]:
        """Get available models from local service"""
        try:
            async with self.session.get(models_url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if service_name == 'ollama':
                        return [model['name'] for model in data.get('models', [])]
                    elif service_name == 'lm_studio':
                        return [model['id'] for model in data.get('data', [])]
                        
        except Exception as e:
            logger.debug(f"Failed to get models from {models_url}: {e}")
            
        return []

    async def detect_cloud_services(self) -> List[DetectedProvider]:
        """Detect cloud services via API keys in environment"""
        providers = []
        
        for provider_name, config in self.cloud_providers.items():
            provider = await self._detect_cloud_service(provider_name, config)
            if provider:
                providers.append(provider)
        
        return providers

    async def _detect_cloud_service(self, provider_name: str, config: Dict) -> Optional[DetectedProvider]:
        """Detect a specific cloud service"""
        # Check environment variables
        api_key = None
        api_key_env_var = None
        
        for env_var in config['env_vars']:
            if os.getenv(env_var):
                api_key = os.getenv(env_var)
                api_key_env_var = env_var
                break
        
        if not api_key:
            return None
        
        # Validate API key format
        if not self._validate_api_key_format(api_key, config.get('key_pattern')):
            logger.debug(f"Invalid API key format for {provider_name}")
            return None
        
        # Try to get models if endpoint available
        models = []
        if config.get('models_endpoint'):
            models = await self._get_cloud_models(
                config['models_endpoint'], 
                api_key,
                provider_name
            )
        
        return DetectedProvider(
            name=provider_name.title(),
            type='cloud',
            api_key_env_var=api_key_env_var,
            models=models,
            priority=config['priority'],
            status='available',
            info={'provider_type': provider_name}
        )

    def _validate_api_key_format(self, api_key: str, pattern: Optional[str]) -> bool:
        """Validate API key format"""
        if not pattern:
            return True  # No validation pattern specified
        
        try:
            return bool(re.match(pattern, api_key))
        except re.error:
            logger.warning(f"Invalid regex pattern: {pattern}")
            return True  # Fall back to no validation

    async def _get_cloud_models(self, models_url: str, api_key: str, provider_name: str) -> List[str]:
        """Get available models from cloud service"""
        try:
            headers = {}
            
            # Set appropriate authorization header
            if provider_name == 'openai':
                headers['Authorization'] = f'Bearer {api_key}'
            elif provider_name == 'openrouter':
                headers['Authorization'] = f'Bearer {api_key}'
            else:
                headers['Authorization'] = f'Bearer {api_key}'
            
            async with self.session.get(models_url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if provider_name == 'openai':
                        return [model['id'] for model in data.get('data', [])]
                    elif provider_name == 'openrouter':
                        return [model['id'] for model in data.get('data', [])]
                        
        except Exception as e:
            logger.debug(f"Failed to get models from {models_url}: {e}")
            
        return []

    def mask_api_key(self, api_key: str) -> str:
        """Mask API key for safe display"""
        if len(api_key) <= 8:
            return '*' * len(api_key)
        return api_key[:4] + '*' * (len(api_key) - 8) + api_key[-4:]

    def get_detected_summary(self, providers: List[DetectedProvider]) -> Dict[str, Any]:
        """Get summary of detected providers"""
        local_count = sum(1 for p in providers if p.type == 'local')
        cloud_count = sum(1 for p in providers if p.type == 'cloud')
        total_models = sum(len(p.models) for p in providers)
        
        return {
            'total_providers': len(providers),
            'local_providers': local_count,
            'cloud_providers': cloud_count,
            'total_models': total_models,
            'providers': [
                {
                    'name': p.name,
                    'type': p.type,
                    'models_count': len(p.models),
                    'priority': p.priority,
                    'endpoint': p.endpoint,
                    'has_api_key': bool(p.api_key_env_var)
                }
                for p in providers
            ]
        }

# Convenience function for quick detection
async def discover_providers(timeout: int = 3) -> List[DetectedProvider]:
    """Quick discovery of all available providers"""
    async with ServiceDetector(timeout=timeout) as detector:
        return await detector.detect_all()