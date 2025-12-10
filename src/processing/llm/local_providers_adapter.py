"""
Local Providers Adapter for Conjecture
Provides unified interface for local LLM providers (Ollama, LM Studio, etc.)
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .common import GenerationConfig as BaseGenerationConfig, LLMProcessingResult

@dataclass
class GenerationConfig(BaseGenerationConfig):
    """Extended generation config for local providers"""
    # Local provider specific parameters can be added here
    pass

@dataclass
class LocalClaim:
    """Local provider claim representation"""
    id: str
    content: str
    confidence: float
    type: Any  # ClaimType enum
    state: Any  # ClaimState enum
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LocalProcessingResult:
    """Result from local provider processing"""
    success: bool
    processed_claims: List[LocalClaim]
    model_used: str
    tokens_used: int
    processing_time_ms: int
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class LocalProviderProcessor:
    """
    Unified processor for local LLM providers
    Supports Ollama, LM Studio, and other OpenAI-compatible local providers
    """

    def __init__(self, provider_type: str, base_url: str, model_name: str):
        self.provider_type = provider_type.lower()
        self.base_url = base_url
        self.model_name = model_name
        self._stats = {
            "requests_processed": 0,
            "successful_requests": 0,
            "total_tokens": 0,
            "total_time": 0.0,
        }

    def generate_response(self, prompt: str, config: GenerationConfig) -> LocalProcessingResult:
        """Generate response from local provider"""
        start_time = time.time()
        self._stats["requests_processed"] += 1

        try:
            # Make API request to local provider
            import requests
            
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": config.max_tokens if hasattr(config, 'max_tokens') else 1000,
                "temperature": config.temperature if hasattr(config, 'temperature') else 0.7
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                headers=headers,
                json=data,
                timeout=config.timeout if hasattr(config, 'timeout') else 30
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract response content
            content = result.get("response", "")
            tokens_used = len(content.split())  # Rough estimate
            
            # Update stats
            self._stats["successful_requests"] += 1
            self._stats["total_tokens"] += tokens_used
            
            processing_time = (time.time() - start_time) * 1000
            self._stats["total_time"] += processing_time
            
            return LocalProcessingResult(
                success=True,
                processed_claims=[LocalClaim(
                    id="generated",
                    content=content,
                    confidence=0.8,
                    type=None,
                    state=None
                )],
                model_used=self.model_name,
                tokens_used=tokens_used,
                processing_time_ms=int(processing_time)
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return LocalProcessingResult(
                success=False,
                processed_claims=[],
                model_used=self.model_name,
                tokens_used=0,
                processing_time_ms=int(processing_time),
                errors=[str(e)]
            )