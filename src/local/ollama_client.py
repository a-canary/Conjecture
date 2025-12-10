"""
Ollama client for local LLM inference.
Provides integration with Ollama and LM Studio for local model inference.
"""

import asyncio
import logging
import json
import aiohttp
from typing import List, Dict, Any, Optional, AsyncGenerator
import time
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ModelProvider(Enum):
    """Available local model providers."""
    OLLAMA = "ollama"
    LM_STUDIO = "lm_studio"

@dataclass
class ModelInfo:
    """Model information structure."""
    name: str
    provider: ModelProvider
    size: Optional[str] = None
    modified_at: Optional[str] = None
    digest: Optional[str] = None
    parameters: Optional[str] = None
    quantization: Optional[str] = None

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 2048
    stream: bool = False
    stop: Optional[List[str]] = None

class OllamaClient:
    """
    Client for interacting with Ollama and LM Studio for local LLM inference.
    Supports both streaming and non-streaming generation.
    """

    def __init__(self, 
                 base_url: str = "http://localhost:11434",  # Default Ollama port
                 timeout: int = 60,
                 provider: ModelProvider = ModelProvider.OLLAMA):
        self.base_url = base_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.provider = provider
        self.session: Optional[aiohttp.ClientSession] = None
        self._available_models: List[ModelInfo] = []
        self._health_status = "unknown"

    async def initialize(self) -> None:
        """Initialize the client session and check connection."""
        try:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
            
            # Check if service is available
            await self.health_check()
            
            if self._health_status == "healthy":
                # Fetch available models
                await self.refresh_models()
                logger.info(f"Connected to {self.provider.value} at {self.base_url}")
            else:
                logger.warning(f"Service at {self.base_url} is not available")

        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            raise

    async def close(self) -> None:
        """Close the client session."""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info("Ollama client closed")

    async def health_check(self) -> bool:
        """Check if the service is available and responsive."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession(timeout=self.timeout)

            # Try different endpoints based on provider
            if self.provider == ModelProvider.OLLAMA:
                endpoint = "/api/tags"
            else:  # LM Studio
                endpoint = "/v1/models"

            url = f"{self.base_url}{endpoint}"

            async with self.session.get(url) as response:
                if response.status == 200:
                    self._health_status = "healthy"
                    return True
                else:
                    self._health_status = "unhealthy"
                    return False

        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            self._health_status = "unhealthy"
            return False

    async def refresh_models(self) -> List[ModelInfo]:
        """Fetch and cache list of available models."""
        try:
            if not self.session:
                raise RuntimeError("Client not initialized")

            if self.provider == ModelProvider.OLLAMA:
                models = await self._get_ollama_models()
            else:  # LM Studio
                models = await self._get_lm_studio_models()

            self._available_models = models
            return models

        except Exception as e:
            logger.error(f"Failed to refresh models: {e}")
            return []

    async def _get_ollama_models(self) -> List[ModelInfo]:
        """Get models from Ollama API."""
        url = f"{self.base_url}/api/tags"
        
        async with self.session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                models = []
                for model_data in data.get('models', []):
                    model_info = ModelInfo(
                        name=model_data.get('name', ''),
                        provider=ModelProvider.OLLAMA,
                        size=model_data.get('size'),
                        modified_at=model_data.get('modified_at'),
                        digest=model_data.get('digest'),
                        parameters=model_data.get('details', {}).get('parameter_size'),
                        quantization=model_data.get('details', {}).get('quantization_level')
                    )
                    models.append(model_info)
                return models
            else:
                return []

    async def _get_lm_studio_models(self) -> List[ModelInfo]:
        """Get models from LM Studio API (OpenAI compatible)."""
        url = f"{self.base_url}/v1/models"
        
        async with self.session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                models = []
                for model_data in data.get('data', []):
                    model_info = ModelInfo(
                        name=model_data.get('id', ''),
                        provider=ModelProvider.LM_STUDIO,
                        size=model_data.get('object', '')
                    )
                    models.append(model_info)
                return models
            else:
                return []

    async def get_available_models(self) -> List[ModelInfo]:
        """Get cached list of available models."""
        return self._available_models

    async def generate_response(self, 
                               prompt: str, 
                               model: Optional[str] = None,
                               config: Optional[GenerationConfig] = None,
                               system_prompt: Optional[str] = None) -> str:
        """
        Generate a response from the selected model.
        
        Args:
            prompt: The input prompt
            model: Model name (uses first available if None)
            config: Generation configuration
            system_prompt: Optional system prompt
            
        Returns:
            Generated text response
        """
        if not self.session:
            raise RuntimeError("Client not initialized")

        # Select model if not specified
        if not model:
            models = await self.get_available_models()
            if not models:
                raise RuntimeError("No models available")
            model = models[0].name

        try:
            if self.provider == ModelProvider.OLLAMA:
                response = await self._generate_ollama_response(
                    prompt, model, config, system_prompt
                )
            else:  # LM Studio
                response = await self._generate_lm_studio_response(
                    prompt, model, config, system_prompt
                )

            return response

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Failed to generate response: {e}")

    async def _generate_ollama_response(self, 
                                      prompt: str, 
                                      model: str,
                                      config: Optional[GenerationConfig],
                                      system_prompt: Optional[str]) -> str:
        """Generate response using Ollama API."""
        url = f"{self.base_url}/api/generate"
        
        # Prepare request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }

        # Add system prompt if provided
        if system_prompt:
            payload["system"] = system_prompt

        # Add generation config
        if config:
            payload.update({
                "options": {
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                    "num_predict": config.max_tokens,
                    "stop": config.stop if config.stop else []
                }
            })

        async with self.session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Ollama API error: {response.status} - {error_text}")

            data = await response.json()
            return data.get('response', '')

    async def _generate_lm_studio_response(self, 
                                         prompt: str, 
                                         model: str,
                                         config: Optional[GenerationConfig],
                                         system_prompt: Optional[str]) -> str:
        """Generate response using LM Studio (OpenAI compatible) API."""
        url = f"{self.base_url}/v1/chat/completions"
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Prepare request payload
        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }

        # Add generation config
        if config:
            payload.update({
                "temperature": config.temperature,
                "top_p": config.top_p,
                "max_tokens": config.max_tokens,
                "stop": config.stop if config.stop else None
            })

        async with self.session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"LM Studio API error: {response.status} - {error_text}")

            data = await response.json()
            return data['choices'][0]['message']['content']

    async def generate_stream(self, 
                             prompt: str, 
                             model: Optional[str] = None,
                             config: Optional[GenerationConfig] = None,
                             system_prompt: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        Generate response as a stream.
        
        Yields:
            Chunks of generated text
        """
        if not self.session:
            raise RuntimeError("Client not initialized")

        # Select model if not specified
        if not model:
            models = await self.get_available_models()
            if not models:
                raise RuntimeError("No models available")
            model = models[0].name

        try:
            if self.provider == ModelProvider.OLLAMA:
                async for chunk in self._generate_ollama_stream(
                    prompt, model, config, system_prompt
                ):
                    yield chunk
            else:  # LM Studio
                async for chunk in self._generate_lm_studio_stream(
                    prompt, model, config, system_prompt
                ):
                    yield chunk

        except Exception as e:
            logger.error(f"Stream generation failed: {e}")
            raise RuntimeError(f"Failed to generate stream: {e}")

    async def _generate_ollama_stream(self, 
                                    prompt: str, 
                                    model: str,
                                    config: Optional[GenerationConfig],
                                    system_prompt: Optional[str]) -> AsyncGenerator[str, None]:
        """Generate streaming response using Ollama API."""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True
        }

        if system_prompt:
            payload["system"] = system_prompt

        if config:
            payload.update({
                "options": {
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                    "num_predict": config.max_tokens,
                    "stop": config.stop if config.stop else []
                }
            })

        async with self.session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Ollama streaming error: {response.status} - {error_text}")

            async for line in response.content:
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        chunk = data.get('response', '')
                        if chunk:
                            yield chunk
                        
                        # Check if generation is complete
                        if data.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue

    async def _generate_lm_studio_stream(self, 
                                       prompt: str, 
                                       model: str,
                                       config: Optional[GenerationConfig],
                                       system_prompt: Optional[str]) -> AsyncGenerator[str, None]:
        """Generate streaming response using LM Studio API."""
        url = f"{self.base_url}/v1/chat/completions"
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "stream": True
        }

        if config:
            payload.update({
                "temperature": config.temperature,
                "top_p": config.top_p,
                "max_tokens": config.max_tokens,
                "stop": config.stop if config.stop else None
            })

        async with self.session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"LM Studio streaming error: {response.status} - {error_text}")

            async for line in response.content:
                if line:
                    try:
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith('data: '):
                            line_str = line_str[6:]
                        
                        if line_str == '[DONE]':
                            break
                        
                        data = json.loads(line_str)
                        chunk = data['choices'][0]['delta'].get('content', '')
                        if chunk:
                            yield chunk
                    except (json.JSONDecodeError, KeyError):
                        continue

    async def get_service_info(self) -> Dict[str, Any]:
        """Get information about the service."""
        health = await self.health_check()
        
        return {
            "provider": self.provider.value,
            "base_url": self.base_url,
            "health_status": self._health_status,
            "available_models": [
                {
                    "name": model.name,
                    "size": model.size,
                    "parameters": model.parameters,
                    "quantization": model.quantization
                }
                for model in self._available_models
            ],
            "model_count": len(self._available_models),
            "healthy": health
        }

    async def test_connection(self) -> bool:
        """Test connection with a simple generation request."""
        try:
            models = await self.get_available_models()
            if not models:
                return False

            # Try a simple generation
            response = await self.generate_response(
                test_prompt="Hello",
                model=models[0].name
            )
            return len(response) > 0

        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

async def create_ollama_client(base_url: str = "http://localhost:11434",
                              timeout: int = 60,
                              provider: ModelProvider = ModelProvider.OLLAMA) -> OllamaClient:
    """
    Factory function to create and initialize an Ollama client.
    
    Args:
        base_url: Base URL for the service
        timeout: Request timeout in seconds
        provider: Model provider (Ollama or LM Studio)
        
    Returns:
        Initialized OllamaClient
    """
    client = OllamaClient(base_url, timeout, provider)
    await client.initialize()
    return client

# Test functions
async def test_ollama_client():
    """Test the Ollama client functionality."""
    print("Testing Ollama client...")
    
    try:
        # Test with Ollama
        async with await create_ollama_client("http://localhost:11434", ModelProvider.OLLAMA) as client:
            info = await client.get_service_info()
            print(f"Ollama Info: {info}")
            
            if info['healthy']:
                response = await client.generate_response("Hello, how are you?")
                print(f"Ollama Response: {response}")
    
    except Exception as e:
        print(f"Ollama test failed: {e}")
    
    try:
        # Test with LM Studio
        async with await create_ollama_client("http://localhost:1234", ModelProvider.LM_STUDIO) as client:
            info = await client.get_service_info()
            print(f"LM Studio Info: {info}")
            
            if info['healthy']:
                response = await client.generate_response("Hello, how are you?")
                print(f"LM Studio Response: {response}")
    
    except Exception as e:
        print(f"LM Studio test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_ollama_client())