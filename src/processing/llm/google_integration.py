"""
Google (Gemini) API Integration for Conjecture LLM Processing
Implements LLM processing using Google Gemini API with multimodal capabilities
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmBlockThreshold, HarmCategory

    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    print("Google Generative AI library not available. Install with: pip install google-generativeai")

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
    """Configuration for Google Gemini API generation"""

    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 40
    max_output_tokens: int = 2048
    candidate_count: int = 1


class GoogleProcessor:
    """Google Gemini API integration for claim processing and analysis"""

    def __init__(self, api_key: str, api_url: str = "https://generativelanguage.googleapis.com", model_name: str = "gemini-pro"):
        if not GOOGLE_AVAILABLE:
            raise ImportError("Google Generative AI library is required for Google integration")

        if not api_key:
            raise ValueError("API key is required for Google integration")

        self.api_key = api_key
        self.api_url = api_url
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
            print(f"Google Gemini client initialized with model: {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Google Gemini client: {e}")

    @with_error_handling("generation")
    def _make_api_request(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """Make API request to Google Gemini with enhanced error handling"""
        if config is None:
            config = GenerationConfig()

        # Update generation config if provided
        generation_config = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "max_output_tokens": config.max_output_tokens,
            "candidate_count": config.candidate_count,
        }

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            if response.text is None:
                raise ValueError("Empty response from Google Gemini")

            return response.text

        except Exception as e:
            raise RuntimeError(f"Google Gemini API error: {e}")

    def _extract_content(self, response_text: str) -> str:
        """Extract content from Gemini response"""
        if not response_text or not response_text.strip():
            raise ValueError("Empty content in Gemini response")
        return response_text.strip()

    def _extract_usage_stats(self, response) -> Tuple[int, int]:
        """Extract token usage statistics from response (Gemini doesn't provide detailed usage stats)"""
        # Gemini doesn't provide detailed token usage in the response
        # We'll estimate based on character count as a rough approximation
        if hasattr(response, 'text') and response.text:
            # Rough estimate: 1 token ≈ 4 characters
            estimated_tokens = len(response.text) // 4
            completion_tokens = estimated_tokens
            total_tokens = completion_tokens  # We can't separate prompt from completion easily
        else:
            total_tokens = 0
            completion_tokens = 0
        
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
        """Generate a response from Google Gemini"""
        start_time = time.time()
        
        try:
            response_text = self._make_api_request(prompt, config)
            content = self._extract_content(response_text)
            
            # Gemini doesn't provide detailed usage, so we'll estimate
            estimated_tokens = len(content) // 4
            processing_time = time.time() - start_time
            
            # Update stats
            self.stats["total_requests"] += 1
            self.stats["successful_requests"] += 1
            self.stats["total_tokens"] += estimated_tokens
            self.stats["total_processing_time"] += processing_time
            
            return LLMProcessingResult(
                success=True,
                processed_claims=[],
                errors=[],
                processing_time=processing_time,
                tokens_used=estimated_tokens,
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
        """Process claims using Google Gemini"""
        start_time = time.time()
        
        try:
            # Format claims for processing
            claims_text = "\n".join([self._format_claim_for_processing(claim) for claim in claims])
            
            # Create prompt based on task
            if task == "analyze":
                prompt = f"""You are Google Gemini, an AI with deep knowledge and analytical capabilities. Analyze the following claims with thoroughness and precision.

Claims to analyze:
{claims_text}

For each claim, provide a comprehensive analysis:

1. **State Classification**:
   - VERIFIED: Strong evidence supports the claim
   - UNVERIFIED: Evidence is insufficient or mixed
   - DEBUNKED: Evidence contradicts the claim

2. **Confidence Assessment** (0.0-1.0): Your certainty in the evaluation

3. **Detailed Analysis**:
   - Factual accuracy based on your knowledge base
   - Logical consistency and coherence
   - Context appropriateness
   - Potential biases or assumptions
   - Areas where more information would be helpful

4. **Verification Evidence**:
   - Supporting evidence or reasoning
   - Counterarguments or conflicting information
   - Sources or domains where verification would be possible
   - Limitations in current knowledge

Return your analysis as structured JSON:
{{
    "claims": [
        {{
            "claim_id": "claim_id_here",
            "state": "VERIFIED|UNVERIFIED|DEBUNKED",
            "confidence": 0.85,
            "analysis": "Comprehensive analysis covering factual accuracy, logical consistency, and contextual considerations",
            "verification": "Supporting evidence, counterarguments, and areas for further investigation"
        }}
    ]
}}

Be thorough and evidence-based in your analysis."""
            elif task == "categorize":
                prompt = f"""As Google Gemini, categorize each claim by its primary communicative purpose and nature.

Claims to categorize:
{claims_text}

Categories:
- **ASSERTION**: Statement presented as fact or truth claim
- **HYPOTHESIS**: Proposed explanation or theory needing investigation
- **PREDICTION**: Forecast about future developments or outcomes
- **QUESTION**: Inquiry seeking information or clarification
- **OPINION**: Subjective viewpoint or personal judgment

Analyze each claim's structure, language, and apparent intent to determine the most appropriate category.

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
                prompt = f"""Please analyze and process the following claims using your comprehensive knowledge base:

{claims_text}

Provide detailed insights, factual context, and analytical perspective for each claim.

Return your analysis as structured JSON data."""

            response_text = self._make_api_request(prompt, config)
            content = self._extract_content(response_text)
            
            # Estimate tokens (Gemini doesn't provide usage stats)
            estimated_tokens = len(content) // 4
            
            # Parse processed claims
            processed_claims = self._parse_claims_from_response(content, claims)
            
            processing_time = time.time() - start_time
            
            # Update stats
            self.stats["total_requests"] += 1
            self.stats["successful_requests"] += 1
            self.stats["total_tokens"] += estimated_tokens
            self.stats["total_processing_time"] += processing_time
            
            return LLMProcessingResult(
                success=True,
                processed_claims=processed_claims,
                errors=[],
                processing_time=processing_time,
                tokens_used=estimated_tokens,
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
        """Perform health check on the Google processor"""
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