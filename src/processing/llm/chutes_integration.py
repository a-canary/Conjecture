"""
Chutes.ai API Integration for Conjecture LLM Processing
Implements LLM processing using Chutes.ai API with response format adaptation
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
    """Configuration for Chutes.ai API generation"""

    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.8


class ChutesProcessor:
    """Chutes.ai API integration for claim processing and analysis"""

    def __init__(self, api_key: str, api_url: str = "https://llm.chutes.ai/v1", model_name: str = "zai-org/GLM-4.6"):
        if not api_key:
            raise ValueError("API key is required for Chutes.ai integration")

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
    def _make_api_request(self, messages: List[Dict[str, str]], config: Optional[GenerationConfig] = None) -> Dict[str, Any]:
        """Make API request to Chutes.ai with enhanced error handling"""
        if config is None:
            config = GenerationConfig()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p
        }

        response = requests.post(
            f"{self.api_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    def _extract_content(self, response: Dict[str, Any]) -> str:
        """Extract content from Chutes.ai response, handling their unique format"""
        try:
            # Chutes.ai uses 'reasoning_content' field instead of standard 'content'
            choices = response.get("choices", [])
            if not choices:
                return ""

            message = choices[0].get("message", {})
            
            # Try reasoning_content first (Chutes.ai specific)
            content = message.get("reasoning_content")
            if content:
                return content
            
            # Fallback to standard content field
            content = message.get("content")
            if content:
                return content
            
            return ""
        except Exception as e:
            print(f"Error extracting content from Chutes.ai response: {e}")
            return ""

    def _format_claim_for_processing(self, claim: BasicClaim) -> str:
        """Format claim according to Conjecture input format"""
        type_str = ", ".join(
            [t.value if hasattr(t, "value") else str(t) for t in claim.type]
        )

        return f"- [{claim.id}, {claim.confidence:.3f}, {type_str}, {claim.state.value}] {claim.content}"

    def _format_context_for_processing(
        self, claims: List[BasicClaim], include_supporting: bool = True
    ) -> str:
        """Format multiple claims into YAML-formatted context"""
        context_lines = ["# Conjecture Processing Context"]
        context_lines.append(f"# Generated: {datetime.now().isoformat()}")
        context_lines.append("# Claims for processing:")
        context_lines.append("")

        for claim in claims:
            claim_line = self._format_claim_for_processing(claim)
            context_lines.append(claim_line)

            # Add supporting relationships if available
            if include_supporting and claim.supported_by:
                context_lines.append(
                    f"  # Supported by: {', '.join(claim.supported_by)}"
                )

            if include_supporting and claim.supports:
                context_lines.append(f"  # Supports: {', '.join(claim.supports)}")

            context_lines.append("")

        return "\n".join(context_lines)

    def _parse_generated_claims(self, response_text: str) -> List[BasicClaim]:
        """Parse generated claims from Chutes.ai response"""
        claims = []

        # Split by lines and look for claim patterns
        lines = response_text.strip().split("\n")

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Parse claim in format: <claim type="" confidence="">content</claim>
            if line.startswith("<claim ") and line.endswith("</claim>"):
                try:
                    # Extract attributes from opening tag
                    opening_tag = line[:line.index(">") + 1]
                    closing_tag_index = line.index("</claim>")
                    content = line[line.index(">") + 1:closing_tag_index]

                    # Parse type attribute
                    type_start = opening_tag.find('type="') + 6
                    type_end = opening_tag.find('"', type_start)
                    claim_type = opening_tag[type_start:type_end]

                    # Parse confidence attribute
                    conf_start = opening_tag.find('confidence="') + 12
                    conf_end = opening_tag.find('"', conf_start)
                    confidence = float(opening_tag[conf_start:conf_end])

                    # Create BasicClaim object
                    claim = BasicClaim(
                        id=f"c{int(time.time() * 1000) % 10000000:07d}",
                        content=content.strip(),
                        confidence=confidence,
                        type=[ClaimType(claim_type)] if claim_type in [t.value for t in ClaimType] else [ClaimType.CONCEPT],
                        state=ClaimState.EXPLORE,
                        created_by="chutes_ai_processor",
                        created_at=datetime.utcnow()
                    )
                    claims.append(claim)

                except Exception as e:
                    print(f"Error parsing claim on line {line_num}: {e}")
                    continue

        return claims

    @with_error_handling("processing")
    def process_claims(
        self, 
        claims: List[BasicClaim], 
        task: str = "analyze",
        config: Optional[GenerationConfig] = None
    ) -> LLMProcessingResult:
        """Process claims using Chutes.ai API"""
        start_time = time.time()
        errors = []
        processed_claims = []

        try:
            # Format context
            context = self._format_context_for_processing(claims)

            # Create prompt
            prompt = f"""You are an expert knowledge analyst. Analyze the following claims and provide insights.

{context}

Task: {task}

Please respond with claims in this format:
<claim type="concept" confidence="0.8">Your analysis or new claim here</claim>
<claim type="thesis" confidence="0.7">Another insight here</claim>

Focus on providing accurate, well-reasoned claims with appropriate confidence scores."""

            # Make API request
            messages = [
                {"role": "user", "content": prompt}
            ]

            response = self._make_api_request(messages, config)
            
            # Extract content using Chutes.ai specific format
            content = self._extract_content(response)
            
            # Parse claims
            processed_claims = self._parse_generated_claims(content)

            # Update stats
            processing_time = time.time() - start_time
            tokens_used = response.get("usage", {}).get("total_tokens", 0)

            self.stats["total_requests"] += 1
            self.stats["successful_requests"] += 1
            self.stats["total_tokens"] += tokens_used
            self.stats["total_processing_time"] += processing_time

            return LLMProcessingResult(
                success=True,
                processed_claims=processed_claims,
                errors=errors,
                processing_time=processing_time,
                tokens_used=tokens_used,
                model_used=self.model_name
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Chutes.ai processing failed: {e}"
            errors.append(error_msg)

            self.stats["total_requests"] += 1
            self.stats["failed_requests"] += 1
            self.stats["total_processing_time"] += processing_time

            return LLMProcessingResult(
                success=False,
                processed_claims=processed_claims,
                errors=errors,
                processing_time=processing_time,
                tokens_used=0,
                model_used=self.model_name
            )

    @with_error_handling("generation")
    def generate_response(
        self, 
        prompt: str, 
        config: Optional[GenerationConfig] = None
    ) -> LLMProcessingResult:
        """Generate response from Chutes.ai API"""
        start_time = time.time()
        errors = []

        try:
            # Make API request
            messages = [
                {"role": "user", "content": prompt}
            ]

            response = self._make_api_request(messages, config)
            
            # Extract content using Chutes.ai specific format
            content = self._extract_content(response)

            # Update stats
            processing_time = time.time() - start_time
            tokens_used = response.get("usage", {}).get("total_tokens", 0)

            self.stats["total_requests"] += 1
            self.stats["successful_requests"] += 1
            self.stats["total_tokens"] += tokens_used
            self.stats["total_processing_time"] += processing_time

            # Return as a single claim for consistency
            synthetic_claim = BasicClaim(
                id=f"c{int(time.time() * 1000) % 10000000:07d}",
                content=content.strip(),
                confidence=0.8,
                type=[ClaimType.CONCEPT],
                state=ClaimState.EXPLORE,
                created_by="chutes_ai_processor",
                created_at=datetime.utcnow()
            )

            return LLMProcessingResult(
                success=True,
                processed_claims=[synthetic_claim],
                errors=errors,
                processing_time=processing_time,
                tokens_used=tokens_used,
                model_used=self.model_name
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Chutes.ai generation failed: {e}"
            errors.append(error_msg)

            self.stats["total_requests"] += 1
            self.stats["failed_requests"] += 1
            self.stats["total_processing_time"] += processing_time

            return LLMProcessingResult(
                success=False,
                processed_claims=[],
                errors=errors,
                processing_time=processing_time,
                tokens_used=0,
                model_used=self.model_name
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics including error handling"""
        if self.stats["total_requests"] > 0:
            avg_time = self.stats["total_processing_time"] / self.stats["total_requests"]
            success_rate = self.stats["successful_requests"] / self.stats["total_requests"]
        else:
            avg_time = 0.0
            success_rate = 0.0

        base_stats = {
            "total_requests": self.stats["total_requests"],
            "successful_requests": self.stats["successful_requests"],
            "failed_requests": self.stats["failed_requests"],
            "success_rate": success_rate,
            "total_tokens": self.stats["total_tokens"],
            "average_processing_time": avg_time,
            "total_processing_time": self.stats["total_processing_time"],
            "model": self.model_name,
            "provider": "chutes"
        }

        # Add error handling stats
        error_stats = self.error_handler.get_stats()
        base_stats["error_handling"] = error_stats

        return base_stats

    def reset_stats(self):
        """Reset processing statistics"""
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_processing_time": 0.0,
        }