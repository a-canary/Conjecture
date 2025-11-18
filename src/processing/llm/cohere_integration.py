"""
Cohere API Integration for Conjecture LLM Processing
Implements LLM processing using Cohere API with enterprise-grade models
"""

import json
import time
import requests
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ...core.basic_models import BasicClaim, ClaimState, ClaimType
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


@dataclass
class GenerationConfig:
    """Configuration for Cohere API generation"""

    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.8
    top_k: int = 40
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    p: float = 0.75  # Cohere-specific parameter
    k: int = 0  # Cohere-specific parameter


class CohereProcessor:
    """Cohere API integration for claim processing and analysis"""

    def __init__(self, api_key: str, api_url: str = "https://api.cohere.ai/v1", model_name: str = "command"):
        if not api_key:
            raise ValueError("API key is required for Cohere integration")

        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_processing_time": 0.0,
        }

        # Initialize enhanced error handling
        self.error_handler = LLMErrorHandler(
            retry_config=RetryConfig(
                max_attempts=3,
                base_delay=1.0,
                max_delay=30.0,
                exponential_base=2.0,
                jitter=True
            )
        )

    @with_error_handling("generation")
    def _make_api_request(self, prompt: str, config: Optional[GenerationConfig] = None) -> Dict[str, Any]:
        """Make API request to Cohere with enhanced error handling"""
        if config is None:
            config = GenerationConfig()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        data = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "p": config.p,
            "k": config.k,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty,
            "truncate": "END"  # How to handle prompts longer than context
        }

        response = requests.post(
            f"{self.api_url}/generate",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    @with_error_handling("generation")
    def _make_chat_request(self, messages: List[Dict[str, str]], config: Optional[GenerationConfig] = None) -> Dict[str, Any]:
        """Make chat API request to Cohere with enhanced error handling"""
        if config is None:
            config = GenerationConfig()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        # Convert messages to Cohere chat format
        chat_history = []
        message = ""
        for msg in messages[:-1]:  # All except last message go to history
            chat_history.append({
                "role": msg["role"],
                "message": msg["content"]
            })
        
        # Last message is the current prompt
        if messages:
            message = messages[-1]["content"]

        data = {
            "model": self.model_name,
            "message": message,
            "chat_history": chat_history,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "p": config.p,
            "k": config.k,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty
        }

        response = requests.post(
            f"{self.api_url}/chat",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    def _extract_content(self, response: Dict[str, Any], is_chat: bool = False) -> str:
        """Extract content from Cohere response"""
        try:
            if is_chat:
                content = response.get("text", "")
            else:
                # Generate endpoint
                generations = response.get("generations", [])
                if not generations:
                    raise ValueError("No generations in response")
                content = generations[0].get("text", "")
            
            if not content:
                raise ValueError("No content in response")

            return content

        except (KeyError, IndexError) as e:
            raise ValueError(f"Failed to extract content from Cohere response: {e}")

    def _extract_usage_stats(self, response: Dict[str, Any]) -> Tuple[int, int]:
        """Extract token usage statistics from response"""
        # Cohere doesn't always provide detailed usage stats in all endpoints
            meta = response.get("meta", {})
        if meta:
            tokens = meta.get("billed_units", {})
            prompt_tokens = tokens.get("input_tokens", 0)
            completion_tokens = tokens.get("output_tokens", 0)
            total_tokens = prompt_tokens + completion_tokens
        else:
            # Fallback estimation
            content = self._extract_content(response)
            total_tokens = len(content) // 4  # Rough estimate
            completion_tokens = total_tokens
        
        return total_tokens, completion_tokens

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

    def generate_response(self, prompt: str, config: Optional[GenerationConfig] = None) -> LLMProcessingResult:
        """Generate a response from Cohere"""
        start_time = time.time()
        
        try:
            response = self._make_api_request(prompt, config)
            content = self._extract_content(response)
            total_tokens, completion_tokens = self._extract_usage_stats(response)
            
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
                model_used=self.model_name
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
                model_used=self.model_name
            )

    def process_claims(self, claims: List[BasicClaim], task: str = "analyze", 
                      config: Optional[GenerationConfig] = None, **kwargs) -> LLMProcessingResult:
        """Process claims using Cohere"""
        start_time = time.time()
        
        try:
            # Format claims for processing
            claims_text = "\n".join([self._format_claim_for_processing(claim) for claim in claims])
            
            # Create prompt based on task
            if task == "analyze":
                prompt = f"""You are an enterprise-grade AI assistant with advanced analytical capabilities. Analyze the following claims systematically and professionally.

Claims to analyze:
{claims_text}

For each claim, provide a comprehensive evaluation:

1. **State Classification**:
   - VERIFIED: Well-supported by reliable evidence
   - UNVERIFIED: Insufficient or mixed evidence available
   - DEBUNKED: Contradicted by available evidence

2. **Confidence Assessment** (0.0-1.0): Quantify your certainty in the evaluation

3. **Professional Analysis**:
   - Factual accuracy and reliability assessment
   - Logical consistency and coherence evaluation
   - Context consideration and relevance analysis
   - Risk assessment and implications
   - Areas requiring additional information

4. **Verification Framework**:
   - Supporting evidence or authoritative sources
   - Methodological considerations for verification
   - Potential biases or limitations
   - Recommendations for further investigation

Return your analysis as structured JSON:
{{
    "claims": [
        {{
            "claim_id": "claim_id_here",
            "state": "VERIFIED|UNVERIFIED|DEBUNKED",
            "confidence": 0.85,
            "analysis": "Professional analysis covering accuracy, consistency, context, and risk factors",
            "verification": "Supporting evidence, verification methods, and recommendations"
        }}
    ]
}}

Provide thorough, evidence-based analysis suitable for professional context."""
            elif task == "categorize":
                prompt = f"""As an enterprise-grade AI, categorize each claim by its primary communicative function and structure.

Claims to categorize:
{claims_text}

Categories:
- **ASSERTION**: Statement presented as factual claim or truth
- **HYPOTHESIS**: Proposed explanation or theory requiring investigation
- **PREDICTION**: Forecast or projection about future events
- **QUESTION**: Inquiry seeking information or clarification
- **OPINION**: Subjective viewpoint or personal assessment

Analyze linguistic patterns, semantic structure, and communicative intent to determine the most appropriate category.

Return your categorization as JSON:
{{
    "claims": [
        {{
            "claim_id": "claim_id_here",
            "claim_type": "ASSERTION|HYPOTHESIS|PREDICTION|QUESTION|OPINION",
            "confidence": 0.85,
            "reasoning": "Analysis of linguistic markers and communicative intent"
        }}
    ]
}}"""
            else:
                # Default to general processing
                prompt = f"""Please analyze and process the following claims using your enterprise-grade analytical capabilities:

{claims_text}

Provide comprehensive professional analysis and insights for each claim.

Return your analysis as structured JSON data."""

            response = self._make_api_request(prompt, config)
            content = self._extract_content(response)
            total_tokens, completion_tokens = self._extract_usage_stats(response)
            
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
                model_used=self.model_name
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
                model_used=self.model_name
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
        """Perform health check on the Cohere processor"""
        try:
            # Test with a simple prompt
            result = self.generate_response("Hello", GenerationConfig(max_tokens=10))
            
            return {
                "status": "healthy" if result.success else "unhealthy",
                "model": self.model_name,
                "last_check": datetime.now().isoformat(),
                "error": None if result.success else result.errors[0] if result.errors else "Unknown error"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "model": self.model_name,
                "last_check": datetime.now().isoformat(),
                "error": str(e)
            }