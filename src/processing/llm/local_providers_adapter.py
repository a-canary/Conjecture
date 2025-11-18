"""
Local Providers Adapter for Conjecture LLM Processing
Integrates Ollama and LM Studio clients into the unified LLM Manager framework
"""

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from core.basic_models import BasicClaim, ClaimState, ClaimType
from local.ollama_client import OllamaClient, ModelProvider, ModelInfo, GenerationConfig as LocalGenerationConfig
from .error_handling import with_error_handling, LLMErrorHandler, RetryConfig


@dataclass
class LLMProcessingResult:
    """Result from LLM processing operation"""

    success: bool
    processed_claims: List[BasicClaim]
    errors: List[str]
    processing_time: float
    tokens_used: int
    model_used: str


class LocalProviderProcessor:
    """Unified adapter for local LLM providers (Ollama and LM Studio)"""

    def __init__(self, provider_type: str, base_url: str, model_name: Optional[str] = None):
        """
        Initialize local provider processor
        
        Args:
            provider_type: "ollama" or "lm_studio"
            base_url: Base URL for the local service
            model_name: Specific model to use (will use first available if None)
        """
        if provider_type not in ["ollama", "lm_studio"]:
            raise ValueError("provider_type must be 'ollama' or 'lm_studio'")

        self.provider_type = provider_type
        self.base_url = base_url
        self.model_name = model_name
        self.client: Optional[OllamaClient] = None
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_processing_time": 0.0,
        }

        # Initialize enhanced error handling for local providers (more aggressive retry)
        self.error_handler = LLMErrorHandler(
            retry_config=RetryConfig(
                max_attempts=3,
                base_delay=0.5,  # Faster retry for local
                max_delay=10.0,  # Shorter max delay
                exponential_base=1.5,  # Less aggressive backoff for local
                jitter=True
            )
        )

        self._initialize_client()

    def _initialize_client(self):
        """Initialize the appropriate local client"""
        try:
            # Convert provider type to ModelProvider enum
            if self.provider_type == "ollama":
                model_provider = ModelProvider.OLLAMA
            else:  # lm_studio
                model_provider = ModelProvider.LM_STUDIO

            # Create and initialize client
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                self.client = loop.run_until_complete(
                    self._create_and_init_client(self.base_url, model_provider)
                )
            finally:
                loop.close()

            print(f"Local {self.provider_type} client initialized at {self.base_url}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize {self.provider_type} client: {e}")

    async def _create_and_init_client(self, base_url: str, provider: ModelProvider) -> OllamaClient:
        """Create and initialize client asynchronously"""
        client = OllamaClient(base_url=base_url, provider=provider)
        await client.initialize()
        return client

    def _get_model_name(self) -> str:
        """Get model name, using configured or first available"""
        if self.model_name:
            return self.model_name

        # Get first available model
        if self.client:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                models = loop.run_until_complete(self.client.get_available_models())
                if models:
                    return models[0].name
            finally:
                loop.close()

        raise RuntimeError("No models available and no specific model configured")

    @with_error_handling("generation")
    def _make_generation_request(self, prompt: str, config: Optional[LocalGenerationConfig] = None) -> str:
        """Make generation request to local provider"""
        if not self.client:
            raise RuntimeError("Client not initialized")

        if config is None:
            config = LocalGenerationConfig()

        # Run async generation in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            model_name = self._get_model_name()
            response = loop.run_until_complete(
                self.client.generate_response(prompt, model_name, config)
            )
            return response
        finally:
            loop.close()

    def _extract_usage_stats(self, response_text: str) -> Tuple[int, int]:
        """Extract token usage statistics (local providers don't provide detailed usage)"""
        # Local providers don't typically provide detailed token usage
        # We'll estimate based on character count
        estimated_tokens = len(response_text) // 4  # Rough estimate
        return estimated_tokens, estimated_tokens

    def _format_claim_for_processing(self, claim: BasicClaim) -> str:
        """Format a claim for LLM processing"""
        claim_type_desc = ClaimType(claim.claim_type).name if claim.claim_type else "UNKNOWN"
        claim_state_desc = ClaimState(claim.state).name if claim.state else "UNKNOWN"
        
        return f"""
Claim ID: {claim.claim_id}
Type: {claim_type_desc}
State: {claim_state_desc}
Content: {claim.content}
Confidence: {claim.confidence}
Evidence: {'Available' if claim.evidence else 'None'}
"""

    def _parse_claims_from_response(self, response_text: str, original_claims: List[BasicClaim]) -> List[BasicClaim]:
        """Parse processed claims from LLM response"""
        processed_claims = []
        
        try:
            # Try to parse as JSON first
            if response_text.strip().startswith('{') or response_text.strip().startswith('['):
                data = json.loads(response_text)
                if isinstance(data, list):
                    claims_data = data
                else:
                    claims_data = data.get("claims", [])
            else:
                # Parse from text format
                claims_data = self._parse_text_claims(response_text)
            
            for claim_data in claims_data:
                # Find original claim
                original_claim = None
                for orig_claim in original_claims:
                    if orig_claim.claim_id == claim_data.get("claim_id"):
                        original_claim = orig_claim
                        break
                
                if original_claim:
                    # Update original claim with processed data
                    if "state" in claim_data:
                        try:
                            original_claim.state = ClaimState[claim_data["state"].upper()]
                        except KeyError:
                            pass
                    
                    if "confidence" in claim_data:
                        original_claim.confidence = float(claim_data["confidence"])
                    
                    if "analysis" in claim_data:
                        original_claim.analysis = claim_data["analysis"]
                    
                    if "verification" in claim_data:
                        original_claim.verification = claim_data["verification"]
                    
                    processed_claims.append(original_claim)
                else:
                    # Create new claim if original not found
                    new_claim = BasicClaim(
                        claim_id=claim_data.get("claim_id", f"generated_{len(processed_claims)}"),
                        content=claim_data.get("content", ""),
                        claim_type=claim_data.get("claim_type", ClaimType.ASSERTION.value)
                    )
                    processed_claims.append(new_claim)

        except Exception as e:
            # If parsing fails, return original claims unchanged
            processed_claims = original_claims.copy()

        return processed_claims

    def _parse_text_claims(self, text: str) -> List[Dict[str, Any]]:
        """Parse claims from text-based response"""
        claims = []
        lines = text.strip().split('\n')
        
        current_claim = {}
        for line in lines:
            line = line.strip()
            if not line:
                if current_claim:
                    claims.append(current_claim)
                    current_claim = {}
                continue
            
            if ':' in line:
                key, value = line.split(':', 1)
                current_claim[key.strip().lower()] = value.strip()
        
        if current_claim:
            claims.append(current_claim)
        
        return claims

    def generate_response(self, prompt: str, config: Optional[LocalGenerationConfig] = None) -> LLMProcessingResult:
        """Generate a response from local provider"""
        start_time = time.time()
        
        try:
            response_text = self._make_generation_request(prompt, config)
            content = response_text.strip()
            total_tokens, completion_tokens = self._extract_usage_stats(content)
            
            processing_time = time.time() - start_time
            
            # Update stats
            self.stats["total_requests"] += 1
            self.stats["successful_requests"] += 1
            self.stats["total_tokens"] += total_tokens
            self.stats["total_processing_time"] += processing_time
            
            return LLMProcessingResult(
                success=True,
                processed_claims=[],
                errors=[],
                processing_time=processing_time,
                tokens_used=total_tokens,
                model_used=self._get_model_name()
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self.stats["total_requests"] += 1
            self.stats["failed_requests"] += 1
            self.stats["total_processing_time"] += processing_time
            
            return LLMProcessingResult(
                success=False,
                processed_claims=[],
                errors=[str(e)],
                processing_time=processing_time,
                tokens_used=0,
                model_used=self._get_model_name() if self.client else "Unknown"
            )

    def process_claims(self, claims: List[BasicClaim], task: str = "analyze", 
                      config: Optional[LocalGenerationConfig] = None, **kwargs) -> LLMProcessingResult:
        """Process claims using local provider"""
        start_time = time.time()
        
        try:
            # Format claims for processing
            claims_text = "\n".join([self._format_claim_for_processing(claim) for claim in claims])
            
            # Create prompt based on task
            if task == "analyze":
                prompt = f"""You are an analytical AI assistant. Analyze the following claims systematically.

Claims to analyze:
{claims_text}

For each claim, provide:
1. State classification: VERIFIED (evidence supports), UNVERIFIED (insufficient evidence), or DEBUNKED (contradicted)
2. Confidence score (0.0-1.0)
3. Brief analysis explaining your reasoning
4. Key evidence or reasoning for your conclusion

Return your response as JSON:
{{
    "claims": [
        {{
            "claim_id": "claim_id_here",
            "state": "VERIFIED|UNVERIFIED|DEBUNKED",
            "confidence": 0.85,
            "analysis": "Brief analysis explaining reasoning",
            "verification": "Key evidence or supporting reasoning"
        }}
    ]
}}"""
            elif task == "categorize":
                prompt = f"""Categorize each claim by its primary communicative purpose.

Claims to categorize:
{claims_text}

Categories:
- ASSERTION: Statement presented as fact or truth claim
- HYPOTHESIS: Proposed explanation needing investigation
- PREDICTION: Forecast about future events
- QUESTION: Inquiry seeking information
- OPINION: Subjective viewpoint or judgment

Return your response as JSON:
{{
    "claims": [
        {{
            "claim_id": "claim_id_here",
            "claim_type": "ASSERTION|HYPOTHESIS|PREDICTION|QUESTION|OPINION",
            "confidence": 0.85
        }}
    ]
}}"""
            else:
                # Default to general processing
                prompt = f"Analyze these claims:\n{claims_text}\n\nReturn JSON response."

            response_text = self._make_generation_request(prompt, config)
            content = response_text.strip()
            total_tokens, completion_tokens = self._extract_usage_stats(content)
            
            # Parse processed claims
            processed_claims = self._parse_claims_from_response(content, claims)
            
            processing_time = time.time() - start_time
            
            # Update stats
            self.stats["total_requests"] += 1
            self.stats["successful_requests"] += 1
            self.stats["total_tokens"] += total_tokens
            self.stats["total_processing_time"] += processing_time
            
            return LLMProcessingResult(
                success=True,
                processed_claims=processed_claims,
                errors=[],
                processing_time=processing_time,
                tokens_used=total_tokens,
                model_used=self._get_model_name()
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self.stats["total_requests"] += 1
            self.stats["failed_requests"] += 1
            self.stats["total_processing_time"] += processing_time
            
            return LLMProcessingResult(
                success=False,
                processed_claims=claims,  # Return original claims on error
                errors=[str(e)],
                processing_time=processing_time,
                tokens_used=0,
                model_used=self._get_model_name() if self.client else "Unknown"
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = self.stats.copy()
        
        if stats["total_requests"] > 0:
            stats["success_rate"] = stats["successful_requests"] / stats["total_requests"]
            stats["average_processing_time"] = stats["total_processing_time"] / stats["total_requests"]
            stats["average_tokens_per_request"] = stats["total_tokens"] / stats["total_requests"]
        else:
            stats["success_rate"] = 0.0
            stats["average_processing_time"] = 0.0
            stats["average_tokens_per_request"] = 0.0
        
        return stats

    def reset_stats(self):
        """Reset processing statistics"""
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_processing_time": 0.0,
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the local provider"""
        try:
            if not self.client:
                return {
                    "status": "unhealthy",
                    "model": "Unknown",
                    "last_check": datetime.now().isoformat(),
                    "error": "Client not initialized"
                }

            # Check health via async call
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                is_healthy = loop.run_until_complete(self.client.health_check())
                model_name = self._get_model_name()
                
                return {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "model": model_name,
                    "last_check": datetime.now().isoformat(),
                    "error": None if is_healthy else "Health check failed"
                }
            finally:
                loop.close()

        except Exception as e:
            return {
                "status": "unhealthy",
                "model": self._get_model_name() if self.client else "Unknown",
                "last_check": datetime.now().isoformat(),
                "error": str(e)
            }

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        try:
            if not self.client:
                return []

            # Get models via async call
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                models = loop.run_until_complete(self.client.get_available_models())
                return [
                    {
                        "name": model.name,
                        "provider": model.provider.value,
                        "size": model.size,
                        "parameters": model.parameters,
                        "quantization": model.quantization
                    }
                    for model in models
                ]
            finally:
                loop.close()

        except Exception as e:
            print(f"Failed to get available models: {e}")
            return []

    def close(self):
        """Close the client connection"""
        if self.client:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                loop.run_until_complete(self.client.close())
            finally:
                loop.close()
            
            self.client = None