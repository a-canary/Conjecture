"""
Chutes.ai Integration for Conjecture
Provides direct integration with Chutes.ai API for claim processing
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .common import GenerationConfig as BaseGenerationConfig, LLMProcessingResult

@dataclass
class GenerationConfig(BaseGenerationConfig):
    """Extended generation config for Chutes.ai"""
    # Chutes-specific parameters can be added here
    pass

@dataclass
class ChutesClaim:
    """Chutes.ai claim representation"""
    id: str
    content: str
    confidence: float
    type: Any  # ClaimType enum
    state: Any  # ClaimState enum
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ChutesProcessingResult:
    """Result from Chutes.ai processing"""
    success: bool
    processed_claims: List[ChutesClaim]
    model_used: str
    tokens_used: int
    processing_time_ms: int
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ChutesProcessor:
    """
    Direct Chutes.ai processor for claim generation and processing
    Maintains compatibility with existing Conjecture architecture
    """

    def __init__(self, api_key: str, api_url: str, model_name: str):
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name
        self._stats = {
            "requests_processed": 0,
            "successful_requests": 0,
            "total_tokens": 0,
            "total_time": 0.0,
        }

    def generate_response(self, prompt: str, config: GenerationConfig) -> ChutesProcessingResult:
        """Generate response from Chutes.ai"""
        start_time = time.time()
        self._stats["requests_processed"] += 1

        try:
            # Make API request to Chutes.ai
            import requests
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": config.max_tokens if hasattr(config, 'max_tokens') else 1000,
                "temperature": config.temperature if hasattr(config, 'temperature') else 0.7
            }
            
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=config.timeout if hasattr(config, 'timeout') else 30
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract response content
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            tokens_used = result.get("usage", {}).get("total_tokens", 0)
            
            # Update stats
            self._stats["successful_requests"] += 1
            self._stats["total_tokens"] += tokens_used
            
            processing_time = (time.time() - start_time) * 1000
            self._stats["total_time"] += processing_time
            
            return ChutesProcessingResult(
                success=True,
                processed_claims=[ChutesClaim(
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
            return ChutesProcessingResult(
                success=False,
                processed_claims=[],
                model_used=self.model_name,
                tokens_used=0,
                processing_time_ms=int(processing_time),
                errors=[str(e)]
            )