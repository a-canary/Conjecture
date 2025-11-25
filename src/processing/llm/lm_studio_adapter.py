"""
LM Studio Adapter for LLM Bridge
Implements LLMProvider interface for LM Studio integration
Maintains clean separation with minimal complexity
"""

import time
import os
from typing import Any, Dict, List, Optional

from ..bridge import LLMProvider, LLMRequest, LLMResponse
from .local_providers_adapter import LocalProviderProcessor
from ...core.models import Claim, ClaimType, ClaimState
from ...config.simple_config import Config


class LMStudioAdapter(LLMProvider):
    """
    LM Studio adapter implementing LLMProvider interface
    Provides clean bridge between standardized requests and LM Studio API
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processor = None
        self._stats = {
            "requests_processed": 0,
            "successful_requests": 0,
            "total_tokens": 0,
            "total_time": 0.0,
        }

        # Initialize the LocalProviderProcessor for LM Studio
        self._initialize_processor()

    def _initialize(self):
        """Initialize LM Studio adapter (placeholder - processor initialized separately)"""
        pass

    def _initialize_processor(self):
        """Initialize LM Studio processor with configuration"""
        try:
            base_url = self.config.get("base_url", "http://localhost:1234")
            model_name = self.config.get("model", "ibm/granite-4-h-tiny")

            # Initialize LM Studio processor
            self.processor = LocalProviderProcessor(
                provider_type="lm_studio", base_url=base_url, model_name=model_name
            )

            print(
                f"LM Studio adapter initialized with model: {model_name} at {base_url}"
            )

        except Exception as e:
            print(f"Failed to initialize LM Studio adapter: {e}")
            self.processor = None

    def process_request(self, request: LLMRequest) -> LLMResponse:
        """Process standardized request using LM Studio"""
        if not self.is_available():
            return LLMResponse(
                success=False,
                content="",
                generated_claims=[],
                metadata={},
                errors=["LM Studio not available"],
                processing_time=0.0,
                tokens_used=0,
            )

        start_time = time.time()
        self._stats["requests_processed"] += 1

        try:
            # Convert standardized request to LM Studio format
            lm_studio_request = self._convert_request(request)

            # Process with LM Studio
            if request.task_type == "explore":
                from ..llm.local_providers_adapter import LLMProcessingResult

                # Create a simple prompt for exploration
                prompt = f"""You are an expert knowledge explorer. Analyze the topic "{request.prompt}" and generate relevant claims.

Please provide claims in this format:
<claim type="concept" confidence="0.8">A factual claim about {request.prompt}</claim>
<claim type="thesis" confidence="0.7">An analytical insight about {request.prompt}</claim>
<claim type="example" confidence="0.6">A specific example related to {request.prompt}</claim>

Focus on:
- Accuracy and factual correctness
- Appropriate confidence scores (0.0-1.0)
- Clear, concise claims
- Different claim types for comprehensive coverage

Generate up to 5 claims with confidence ≥ 0.5."""

                # Create generation config
                from ..local.ollama_client import (
                    GenerationConfig as LocalGenerationConfig,
                )

                config = LocalGenerationConfig(
                    temperature=request.temperature, max_tokens=request.max_tokens
                )

                result = self.processor.generate_response(prompt, config=config)

                # Create a LLMProcessingResult-like object
                processing_result = LLMProcessingResult(
                    success=True,
                    processed_claims=[],  # We'll parse claims from content
                    errors=[],
                    processing_time=time.time() - start_time,
                    tokens_used=0,  # Local providers don't provide detailed token usage
                    model_used=self.processor.model_name
                    if hasattr(self.processor, "model_name")
                    else "unknown",
                )

            else:  # For validation, analysis, etc.
                # Create a simple prompt for other tasks
                prompt = f"Analyze and respond to: {request.prompt}"
                from ..local.ollama_client import (
                    GenerationConfig as LocalGenerationConfig,
                )

                config = LocalGenerationConfig(
                    temperature=request.temperature, max_tokens=request.max_tokens
                )

                result = self.processor.generate_response(prompt, config=config)

                processing_result = LLMProcessingResult(
                    success=True,
                    processed_claims=[],  # We'll parse claims from content
                    errors=[],
                    processing_time=time.time() - start_time,
                    tokens_used=0,
                    model_used=self.processor.model_name
                    if hasattr(self.processor, "model_name")
                    else "unknown",
                )

            # Convert response to standardized format
            processing_time = time.time() - start_time
            self._stats["total_time"] += processing_time

            if processing_result.success:
                self._stats["successful_requests"] += 1

                # Parse claims from the response content
                generated_claims = self._parse_claims_from_content(
                    processing_result.processed_claims,
                    result if isinstance(result, str) else "",
                )

                return LLMResponse(
                    success=True,
                    content=result if isinstance(result, str) else result,
                    generated_claims=generated_claims,
                    metadata={
                        "model": processing_result.model_used,
                        "provider": "lm_studio",
                        "task_type": request.task_type,
                    },
                    errors=processing_result.errors,
                    processing_time=processing_time,
                    tokens_used=processing_result.tokens_used,
                )
            else:
                return LLMResponse(
                    success=False,
                    content="",
                    generated_claims=[],
                    metadata={"provider": "lm_studio"},
                    errors=processing_result.errors,
                    processing_time=processing_time,
                    tokens_used=0,
                )

        except Exception as e:
            processing_time = time.time() - start_time
            self._stats["total_time"] += processing_time

            return LLMResponse(
                success=False,
                content="",
                generated_claims=[],
                metadata={"provider": "lm_studio"},
                errors=[f"LM Studio processing error: {e}"],
                processing_time=processing_time,
                tokens_used=0,
            )

    def _convert_request(self, request: LLMRequest) -> Dict[str, Any]:
        """Convert standardized request to LM Studio format"""
        # Build context from claims if provided
        context_parts = []
        if request.context_claims:
            context_parts.append("# Context Claims:")
            for claim in request.context_claims:
                type_str = ", ".join([t.value for t in claim.type])
                context_parts.append(
                    f"- [{claim.confidence:.2f}, {type_str}] {claim.content}"
                )
            context_parts.append("")

        # Build full prompt
        full_prompt = "\n".join(context_parts + [request.prompt])

        return {"prompt": full_prompt, "context_claims": request.context_claims}

    def _parse_claims_from_content(self, processed_claims, content: str) -> List[Claim]:
        """Parse claims from LM Studio response content"""
        unified_claims = []

        # If we have processed claims from the processor, use them
        if processed_claims:
            for basic_claim in processed_claims:
                try:
                    # Convert BasicClaim to unified Claim model
                    unified_claim = Claim(
                        id=basic_claim.id,
                        content=basic_claim.content,
                        confidence=basic_claim.confidence,
                        type=basic_claim.type,
                        state=basic_claim.state,
                        tags=getattr(basic_claim, "tags", []),
                        supported_by=getattr(basic_claim, "supported_by", []),
                        supports=getattr(basic_claim, "supports", []),
                        created=basic_claim.created,
                        updated=basic_claim.updated,
                    )
                    unified_claims.append(unified_claim)
                except Exception as e:
                    print(f"Error converting claim: {e}")
                    continue
        else:
            # If no processed claims, try to parse from raw content
            # Look for claim patterns in the response
            import re

            # Pattern for <claim type="..." confidence="...">...</claim> format
            pattern = r'<claim\s+type="([^"]+)"\s+confidence="([^"]+)">([^<]+)</claim>'
            matches = re.findall(pattern, content, re.DOTALL)

            for claim_type_str, confidence_str, claim_content in matches:
                try:
                    claim_type = ClaimType(claim_type_str.lower())
                    confidence = float(confidence_str)

                    unified_claim = Claim(
                        id=f"lm_studio_{int(time.time() * 1000)}_{len(unified_claims)}",
                        content=claim_content.strip(),
                        confidence=confidence,
                        type=[claim_type],
                        state=ClaimState.EXPLORE,
                    )
                    unified_claims.append(unified_claim)
                except Exception as e:
                    print(f"Error parsing claim from content: {e}")
                    continue

        return unified_claims

    def is_available(self) -> bool:
        """Check if LM Studio adapter is available"""
        if not self.processor:
            return False

        # Perform a simple health check on the processor
        try:
            health = self.processor.health_check()
            if health.get("status") == "healthy":
                return True
        except Exception:
            pass  # Continue to additional checks if health check fails

        # If health check failed, try to get available models as a basic functionality test
        try:
            models = self.processor.get_available_models()
            if len(models) > 0:
                return True
        except Exception:
            pass  # Continue to generation test

        # As a final test, try a simple generation request
        try:
            from ..local.ollama_client import GenerationConfig as LocalGenerationConfig

            config = LocalGenerationConfig(
                temperature=0.1,
                max_tokens=10,  # Very small response to test connectivity
            )

            # Test with a simple prompt
            test_result = self.processor.generate_response("Hello", config=config)
            # If no exception was raised, consider it functional
            return True
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics"""
        base_stats = super().get_stats()

        if self._stats["requests_processed"] > 0:
            success_rate = (
                self._stats["successful_requests"] / self._stats["requests_processed"]
            )
            avg_time = self._stats["total_time"] / self._stats["requests_processed"]
        else:
            success_rate = 0.0
            avg_time = 0.0

        base_stats.update(
            {
                "requests_processed": self._stats["requests_processed"],
                "successful_requests": self._stats["successful_requests"],
                "success_rate": success_rate,
                "total_tokens": self._stats["total_tokens"],
                "average_processing_time": avg_time,
                "total_processing_time": self._stats["total_time"],
            }
        )

        # Add processor stats if available
        if self.processor and hasattr(self.processor, "get_stats"):
            processor_stats = self.processor.get_stats()
            base_stats["processor_stats"] = processor_stats

        return base_stats


def create_lm_studio_adapter_from_config() -> LMStudioAdapter:
    """
    Factory function to create LMStudioAdapter from environment configuration
    Simplifies adapter creation with standard config
    """
    config = Config()

    # Check if LM Studio is configured
    if (
        config.llm_provider.lower() != "lm_studio"
        and not config.llm_api_url.lower().startswith("http://localhost:1234")
    ):
        # Check if using LM Studio through unified provider config
        provider_url = os.getenv("PROVIDER_API_URL", "")
        if (
            "lmstudio" not in provider_url.lower()
            and "localhost:1234" not in provider_url
        ):
            return LMStudioAdapter({"enabled": False})  # Return disabled adapter

    lm_studio_config = {
        "base_url": getattr(
            config,
            "llm_api_url",
            os.getenv("PROVIDER_API_URL", "http://localhost:1234/v1"),
        ),
        "model": getattr(
            config, "llm_model", os.getenv("PROVIDER_MODEL", "ibm/granite-4-h-tiny")
        ),
    }

    return LMStudioAdapter(lm_studio_config)
