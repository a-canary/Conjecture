"""
Anthropic (Claude) API Integration for Conjecture LLM Processing
Implements LLM processing using Anthropic Claude API with advanced reasoning capabilities
"""

import json
import time
import requests
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime

from .error_handling import RetryConfig, CircuitBreakerConfig, with_error_handling
from .common import GenerationConfig, LLMProcessingResult


class AnthropicProcessor:
    """Anthropic Claude API integration for claim processing and analysis"""

    def __init__(
        self,
        api_key: str,
        api_url: str = "https://api.anthropic.com",
        model_name: str = "claude-3-haiku-20240307",
    ):
        if not api_key:
            raise ValueError("API key is required for Anthropic integration")

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
                jitter=True,
            )
        )

    @with_error_handling("generation")
    def _make_api_request(
        self, messages: List[Dict[str, str]], config: Optional[GenerationConfig] = None
    ) -> Dict[str, Any]:
        """Make API request to Anthropic with enhanced error handling"""
        if config is None:
            config = GenerationConfig()

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        # Convert OpenAI-style messages to Claude format
        claude_messages = self._convert_messages_to_claude_format(messages)

        data = {
            "model": self.model_name,
            "messages": claude_messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
        }

        response = requests.post(
            f"{self.api_url}/v1/messages", headers=headers, json=data, timeout=30
        )
        response.raise_for_status()
        return response.json()

    def _convert_messages_to_claude_format(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Convert OpenAI-style messages to Claude format"""
        claude_messages = []

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                # Claude uses system parameter separately, but we'll include it in the first user message if needed
                continue
            elif role == "assistant":
                claude_messages.append({"role": "assistant", "content": content})
            elif role == "user":
                claude_messages.append({"role": "user", "content": content})

        return claude_messages

    def _extract_content(self, response: Dict[str, Any]) -> str:
        """Extract content from Anthropic response"""
        try:
            content = response.get("content", [])
            if not content:
                raise ValueError("No content in response")

            if isinstance(content, list) and content:
                return content[0].get("text", "")
            elif isinstance(content, str):
                return content
            else:
                raise ValueError("Unexpected content format in response")

        except (KeyError, IndexError) as e:
            raise ValueError(f"Failed to extract content from Anthropic response: {e}")

    def _extract_usage_stats(self, response: Dict[str, Any]) -> Tuple[int, int]:
        """Extract token usage statistics from response"""
        usage = response.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = input_tokens + output_tokens
        return total_tokens, output_tokens

    def _format_claim_for_processing(self, claim: BasicClaim) -> str:
        """Format a claim for LLM processing"""
        claim_type_desc = (
            ClaimType(claim.claim_type).name if claim.claim_type else "UNKNOWN"
        )
        claim_state_desc = ClaimState(claim.state).name if claim.state else "UNKNOWN"

        return f"""
Claim ID: {claim.claim_id}
Type: {claim_type_desc}
State: {claim_state_desc}
Content: {claim.content}
Confidence: {claim.confidence}
Evidence: {"Available" if claim.evidence else "None"}
"""

    def _parse_claims_from_response(
        self, response_text: str, original_claims: List[BasicClaim]
    ) -> List[BasicClaim]:
        """Parse processed claims from LLM response"""
        processed_claims = []

        try:
            # Try to parse as JSON first
            if response_text.strip().startswith(
                "{"
            ) or response_text.strip().startswith("["):
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
                            original_claim.state = ClaimState[
                                claim_data["state"].upper()
                            ]
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
                        claim_id=claim_data.get(
                            "claim_id", f"generated_{len(processed_claims)}"
                        ),
                        content=claim_data.get("content", ""),
                        claim_type=claim_data.get(
                            "claim_type", ClaimType.ASSERTION.value
                        ),
                    )
                    processed_claims.append(new_claim)

        except Exception as e:
            # If parsing fails, return original claims unchanged
            processed_claims = original_claims.copy()

        return processed_claims

    def _parse_text_claims(self, text: str) -> List[Dict[str, Any]]:
        """Parse claims from text-based response"""
        claims = []
        lines = text.strip().split("\n")

        current_claim = {}
        for line in lines:
            line = line.strip()
            if not line:
                if current_claim:
                    claims.append(current_claim)
                    current_claim = {}
                continue

            if ":" in line:
                key, value = line.split(":", 1)
                current_claim[key.strip().lower()] = value.strip()

        if current_claim:
            claims.append(current_claim)

        return claims

    def generate_response(
        self, prompt: str, config: Optional[GenerationConfig] = None
    ) -> LLMProcessingResult:
        """Generate a response from Anthropic Claude"""
        start_time = time.time()

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self._make_api_request(messages, config)

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
                model_used=self.model_name,
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
                model_used=self.model_name,
            )

    def process_claims(
        self,
        claims: List[BasicClaim],
        task: str = "analyze",
        config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> LLMProcessingResult:
        """Process claims using Anthropic Claude"""
        start_time = time.time()

        try:
            # Format claims for processing
            claims_text = "\n".join(
                [self._format_claim_for_processing(claim) for claim in claims]
            )

            # Create prompt based on task
            if task == "analyze":
                prompt = f"""You are Claude, an AI assistant with advanced reasoning capabilities. Analyze the following claims with precision and intellectual honesty.

Claims to analyze:
{claims_text}

For each claim, provide a thorough analysis considering:

1. **State Classification**: 
   - VERIFIED: Strong evidence supports the claim
   - UNVERIFIED: Evidence is insufficient or mixed
   - DEBUNKED: Evidence contradicts the claim

2. **Confidence Assessment** (0.0-1.0): Reflect your certainty in the evaluation

3. **Comprehensive Analysis**: 
   - Logical coherence and internal consistency
   - Evidence quality and availability (explicit or implicit)
   - Plausibility based on established knowledge
   - Potential biases, assumptions, or logical fallacies
   - Contextual factors that might affect validity

4. **Verification Evidence**:
   - Specific supporting or contradicting evidence
   - Reasoning pathways that lead to your conclusion
   - Knowledge domains where the claim can be verified
   - Uncertainties or areas requiring further investigation

Return your analysis as structured JSON:
{{
    "claims": [
        {{
            "claim_id": "claim_id_here",
            "state": "VERIFIED|UNVERIFIED|DEBUNKED",
            "confidence": 0.85,
            "analysis": "Comprehensive analysis covering logical consistency, evidence assessment, plausibility, and potential biases",
            "verification": "Specific evidence, reasoning, or methodological approach that supports the conclusion"
        }}
    ]
}}

Be thorough but concise, focusing on the most relevant factors for each claim's evaluation."""
            elif task == "categorize":
                prompt = f"""As Claude, carefully categorize each claim based on its primary nature and communicative intent.

Claims to categorize:
{claims_text}

Categories:
- **ASSERTION**: Confident statement of fact, belief, or conclusion presented as true
- **HYPOTHESIS**: Proposed explanation or theoretical framework requiring testing
- **PREDICTION**: Statement about future events, outcomes, or developments
- **QUESTION**: Inquiry seeking information, clarification, or eliciting responses
- **OPINION**: Personal viewpoint, judgment, or subjective evaluation

Consider the claim's structure, linguistic markers, and apparent purpose. For borderline cases, select the most appropriate primary category.

Return your categorization as JSON:
{{
    "claims": [
        {{
            "claim_id": "claim_id_here",
            "claim_type": "ASSERTION|HYPOTHESIS|PREDICTION|QUESTION|OPINION",
            "confidence": 0.85,
            "reasoning": "Brief explanation for the categorization based on linguistic and contextual analysis"
        }}
    ]
}}"""
            else:
                # Default to general processing
                prompt = f"""Please analyze and process the following claims with your advanced reasoning capabilities:

{claims_text}

Provide comprehensive insights, logical analysis, and contextual understanding for each claim.

Return your analysis as structured JSON data."""

            messages = [{"role": "user", "content": prompt}]
            response = self._make_api_request(messages, config)

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
                model_used=self.model_name,
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
                model_used=self.model_name,
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = self.stats.copy()

        if stats["total_requests"] > 0:
            stats["success_rate"] = (
                stats["successful_requests"] / stats["total_requests"]
            )
            stats["average_processing_time"] = (
                stats["total_processing_time"] / stats["total_requests"]
            )
            stats["average_tokens_per_request"] = (
                stats["total_tokens"] / stats["total_requests"]
            )
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
        """Perform health check on the Anthropic processor"""
        try:
            # Test with a simple prompt
            result = self.generate_response("Hello", GenerationConfig(max_tokens=10))

            return {
                "status": "healthy" if result.success else "unhealthy",
                "model": self.model_name,
                "last_check": datetime.now().isoformat(),
                "error": None
                if result.success
                else result.errors[0]
                if result.errors
                else "Unknown error",
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "model": self.model_name,
                "last_check": datetime.now().isoformat(),
                "error": str(e),
            }
