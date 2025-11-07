"""
Gemini API Integration for Conjecture LLM Processing
Implements LLM processing using Google Gemini API as primary choice
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmBlockThreshold, HarmCategory

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print(
        "Google Generative AI library not available. Install with: pip install google-generativeai"
    )

from ...core.basic_models import BasicClaim, ClaimState, ClaimType


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
    """Configuration for Gemini API generation"""

    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 40
    max_output_tokens: int = 2048
    candidate_count: int = 1


class GeminiProcessor:
    """Gemini API integration for claim processing and analysis"""

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Google Generative AI library is required for Gemini integration"
            )

        if not api_key:
            raise ValueError("API key is required for Gemini integration")

        self.api_key = api_key
        self.model_name = model_name
        self.client = None
        self.model = None
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_processing_time": 0.0,
        }

        self._initialize_client()

    def _initialize_client(self):
        """Initialize Gemini client and model"""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 2048,
                    "candidate_count": 1,
                },
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                },
            )
            self.client = genai
            print(f"Gemini client initialized with model: {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini client: {e}")

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
        """Parse generated claims from Gemini response"""
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
                    # Extract attributes
                    import re

                    match = re.match(
                        r'<claim type="([^"]*)" confidence="([^"]*)"[^>]*>(.*?)</claim>',
                        line,
                    )
                    if not match:
                        continue

                    claim_type_str, confidence_str, content = match.groups()

                    # Parse confidence
                    try:
                        confidence = float(confidence_str)
                        if not (0.0 <= confidence <= 1.0):
                            confidence = 0.5  # Default if invalid
                    except ValueError:
                        confidence = 0.5

                    # Parse claim type
                    try:
                        claim_type = ClaimType(claim_type_str)
                    except ValueError:
                        claim_type = ClaimType.EXAMPLE  # Default type

                    # Generate unique ID
                    claim_id = f"generated_{int(time.time() * 1000)}_{line_num}"

                    # Create claim
                    claim = BasicClaim(
                        id=claim_id,
                        content=content.strip(),
                        confidence=confidence,
                        type=[claim_type],
                        tags=["generated", "gemini"],
                        state=ClaimState.EXPLORE,
                        created=datetime.now(),
                    )

                    claims.append(claim)

                except Exception as e:
                    print(f"Error parsing claim on line {line_num}: {e}")
                    continue

        return claims

    def process_exploration(
        self, context_claims: List[BasicClaim], query: str, max_new_claims: int = 5
    ) -> LLMProcessingResult:
        """Process exploration request to generate new claims"""
        start_time = time.time()
        processed_claims = []
        errors = []
        tokens_used = 0

        try:
            self.stats["total_requests"] += 1

            # Format context
            context = self._format_context_for_processing(context_claims)

            # Build prompt
            prompt = f"""You are Conjecture, an AI system that uses evidence-based reasoning to explore new claims based on existing context.

Context Analysis:
{context}

User Query: {query}

Task: Generate {max_new_claims} new claims that explore the user query based on the provided context. Each claim should:
1. Be directly relevant to the query
2. Build upon or extend the existing context
3. Have a realistic confidence score (0.0-1.0)
4. Be properly typed as one of: concept, reference, thesis, skill, example

Format each claim exactly as:
<claim type="TYPE" confidence="0.X">CLAIM CONTENT</claim>

Example format:
<claim type="concept" confidence="0.85">Quantum entanglement creates instantaneous correlations between particles</claim>

Respond only with the generated claims, one per line."""

            # Generate response
            response = self.model.generate_content(prompt)

            # Update stats
            if hasattr(response, "usage_metadata"):
                tokens_used = response.usage_metadata.total_token_count
            else:
                tokens_used = len(prompt) + len(response.text)  # Rough estimate

            # Parse generated claims
            processed_claims = self._parse_generated_claims(response.text)

            # Update stats
            self.stats["successful_requests"] += 1
            self.stats["total_tokens"] += tokens_used

            processing_time = time.time() - start_time
            self.stats["total_processing_time"] += processing_time

            return LLMProcessingResult(
                success=True,
                processed_claims=processed_claims,
                errors=errors,
                processing_time=processing_time,
                tokens_used=tokens_used,
                model_used=self.model_name,
            )

        except Exception as e:
            self.stats["failed_requests"] += 1
            errors.append(f"Processing failed: {e}")
            processing_time = time.time() - start_time

            return LLMProcessingResult(
                success=False,
                processed_claims=[],
                errors=errors,
                processing_time=processing_time,
                tokens_used=tokens_used,
                model_used=self.model_name,
            )

    def validate_claim(
        self, claim: BasicClaim, context_claims: List[BasicClaim]
    ) -> LLMProcessingResult:
        """Validate and potentially update an existing claim"""
        start_time = time.time()
        errors = []
        tokens_used = 0

        try:
            self.stats["total_requests"] += 1

            # Format context
            context = self._format_context_for_processing(context_claims)
            claim_line = self._format_claim_for_processing(claim)

            # Build validation prompt
            prompt = f"""You are Conjecture, an AI system that validates claims using evidence-based reasoning.

Context Analysis:
{context}

Claim to Validate:
{claim_line}

Task: Validate the claim based on the provided context. Consider:
1. Consistency with existing evidence
2. Logical coherence
3. Confidence score accuracy
4. Claim type appropriateness

If the claim is valid and confidence is appropriate, respond with:
<claim type="{claim.type[0].value}" confidence="{claim.confidence}">VALID</claim>

If the claim needs adjustments, respond with the corrected claim:
<claim type="TYPE" confidence="0.X">CORRECTED CONTENT</claim>

If the claim should be rejected, respond with:
<claim type="reference" confidence="0.0">REJECTED</claim>"""

            # Generate response
            response = self.model.generate_content(prompt)

            # Update stats
            if hasattr(response, "usage_metadata"):
                tokens_used = response.usage_metadata.total_token_count
            else:
                tokens_used = len(prompt) + len(response.text)

            # Parse response
            claims = self._parse_generated_claims(response.text)

            # Update stats
            self.stats["successful_requests"] += 1
            self.stats["total_tokens"] += tokens_used

            processing_time = time.time() - start_time
            self.stats["total_processing_time"] += processing_time

            if claims and "VALID" in claims[0].content:
                return LLMProcessingResult(
                    success=True,
                    processed_claims=[claim],  # Original claim is valid
                    errors=errors,
                    processing_time=processing_time,
                    tokens_used=tokens_used,
                    model_used=self.model_name,
                )
            elif claims:
                return LLMProcessingResult(
                    success=True,
                    processed_claims=claims,  # Corrected or rejected claim
                    errors=errors,
                    processing_time=processing_time,
                    tokens_used=tokens_used,
                    model_used=self.model_name,
                )
            else:
                errors.append("No valid response generated")
                return LLMProcessingResult(
                    success=False,
                    processed_claims=[],
                    errors=errors,
                    processing_time=processing_time,
                    tokens_used=tokens_used,
                    model_used=self.model_name,
                )

        except Exception as e:
            self.stats["failed_requests"] += 1
            errors.append(f"Validation failed: {e}")
            processing_time = time.time() - start_time

            return LLMProcessingResult(
                success=False,
                processed_claims=[],
                errors=errors,
                processing_time=processing_time,
                tokens_used=tokens_used,
                model_used=self.model_name,
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        success_rate = 0.0
        if self.stats["total_requests"] > 0:
            success_rate = (
                self.stats["successful_requests"] / self.stats["total_requests"]
            )

        avg_processing_time = 0.0
        if self.stats["successful_requests"] > 0:
            avg_processing_time = (
                self.stats["total_processing_time"] / self.stats["successful_requests"]
            )

        return {
            **self.stats,
            "success_rate": success_rate,
            "average_processing_time": avg_processing_time,
            "model_name": self.model_name,
            "api_available": True,
        }

    def test_connectivity(self) -> bool:
        """Test API connectivity"""
        try:
            # Simple test prompt
            response = self.model.generate_content("Respond with just: OK")
            return "OK" in response.text
        except Exception as e:
            print(f"Connectivity test failed: {e}")
            return False

    def close(self):
        """Cleanup resources"""
        # No specific cleanup needed for Gemini client
        pass


# Configuration and utility functions
def create_gemini_config(
    api_key: str,
    model_name: str = "gemini-1.5-flash",
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    """Create Gemini configuration dictionary"""
    return {
        "api_key": api_key,
        "model_name": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def test_gemini_availability() -> Tuple[bool, str]:
    """Test if Gemini API is available and configured"""
    if not GEMINI_AVAILABLE:
        return (
            False,
            "Google Generative AI library not installed. Install with: pip install google-generativeai",
        )

    # Check for API key (would be passed in config in real usage)
    return True, "Gemini API available. Configure with API key for processing."


# Fallback and error handling
class GeminiAPIError(Exception):
    """Custom exception for Gemini API errors"""

    pass


class GeminiRateLimitError(GeminiAPIError):
    """Raised when API rate limits are exceeded"""

    pass


class GeminiQuotaExceededError(GeminiAPIError):
    """Raised when API quota is exceeded"""

    pass
