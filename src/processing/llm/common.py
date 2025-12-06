"""
Common data classes for LLM processing
Consolidated configuration classes to reduce duplication
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class GenerationConfig:
    """Unified configuration for LLM generation across all providers"""

    # Common parameters
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.8

    # Optional parameters (not all providers support these)
    top_k: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

    # Provider-specific parameters
    stop_sequences: Optional[List[str]] = None
    stream: bool = False

    def to_provider_dict(self, provider: str) -> Dict[str, Any]:
        """Convert to provider-specific parameter dictionary"""
        base_params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }

        # Add provider-specific parameters
        if provider.lower() in ["openai", "anthropic"]:
            if self.frequency_penalty is not None:
                base_params["frequency_penalty"] = self.frequency_penalty
            if self.presence_penalty is not None:
                base_params["presence_penalty"] = self.presence_penalty

        if provider.lower() in ["anthropic", "cohere", "groq"]:
            if self.top_k is not None:
                base_params["top_k"] = self.top_k

        if self.stop_sequences:
            base_params["stop"] = self.stop_sequences

        return base_params


@dataclass
class LLMProcessingResult:
    """Unified result structure for LLM processing"""

    success: bool
    processed_claims: List[Any] = None
    errors: List[str] = None
    processing_time: float = 0.0
    tokens_used: int = 0
    model_used: str = ""
    content: str = ""
    processing_time_ms: Optional[int] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.processed_claims is None:
            self.processed_claims = []
        if self.errors is None:
            self.errors = []
        if self.metadata is None:
            self.metadata = {}
