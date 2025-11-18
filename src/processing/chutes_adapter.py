"""
Chutes.ai Adapter for LLM Bridge
Implements LLMProvider interface for Chutes.ai integration
Maintains clean separation with minimal complexity
"""

import os
import time
from typing import Any, Dict, List

from .llm_bridge import LLMProvider, LLMRequest, LLMResponse
from .llm.chutes_integration import ChutesProcessor, GenerationConfig
from core.unified_models import Claim, ClaimType, ClaimState
from config.simple_config import Config


class ChutesAdapter(LLMProvider):
    """
    Chutes.ai adapter implementing LLMProvider interface
    Provides clean bridge between standardized requests and Chutes.ai API
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processor = None
        self._stats = {
            "requests_processed": 0,
            "successful_requests": 0,
            "total_tokens": 0,
            "total_time": 0.0
        }
    
    def _initialize(self):
        """Initialize Chutes.ai processor with configuration"""
        try:
            api_key = self.config.get("api_key")
            api_url = self.config.get("api_url", "https://llm.chutes.ai/v1")
            model_name = self.config.get("model", "zai-org/GLM-4.6")
            
            if not api_key:
                raise ValueError("Chutes.ai API key required")
            
            self.processor = ChutesProcessor(
                api_key=api_key,
                api_url=api_url,
                model_name=model_name
            )
            
        except Exception as e:
            print(f"Failed to initialize Chutes.ai adapter: {e}")
            self.processor = None
    
    def process_request(self, request: LLMRequest) -> LLMResponse:
        """Process standardized request using Chutes.ai"""
        if not self.is_available():
            return LLMResponse(
                success=False,
                content="",
                generated_claims=[],
                metadata={},
                errors=["Chutes.ai not available"],
                processing_time=0.0,
                tokens_used=0
            )
        
        start_time = time.time()
        self._stats["requests_processed"] += 1
        
        try:
            # Convert standardized request to Chutes.ai format
            chutes_request = self._convert_request(request)
            
            # Process with Chutes.ai
            if request.task_type == "explore":
                result = self.processor.generate_response(
                    chutes_request["prompt"],
                    GenerationConfig(
                        temperature=request.temperature,
                        max_tokens=request.max_tokens
                    )
                )
            else:
                # For validation, analysis, etc.
                result = self.processor.process_claims(
                    request.context_claims or [],
                    task=request.task_type,
                    config=GenerationConfig(
                        temperature=request.temperature,
                        max_tokens=request.max_tokens
                    )
                )
            
            # Convert response to standardized format
            processing_time = time.time() - start_time
            self._stats["total_time"] += processing_time
            
            if result.success:
                self._stats["successful_requests"] += 1
                self._stats["total_tokens"] += result.tokens_used
                
                return LLMResponse(
                    success=True,
                    content=result.processed_claims[0].content if result.processed_claims else "",
                    generated_claims=self._convert_claims(result.processed_claims),
                    metadata={
                        "model": result.model_used,
                        "provider": "chutes",
                        "task_type": request.task_type
                    },
                    errors=result.errors,
                    processing_time=processing_time,
                    tokens_used=result.tokens_used
                )
            else:
                return LLMResponse(
                    success=False,
                    content="",
                    generated_claims=[],
                    metadata={"provider": "chutes"},
                    errors=result.errors,
                    processing_time=processing_time,
                    tokens_used=0
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            self._stats["total_time"] += processing_time
            
            return LLMResponse(
                success=False,
                content="",
                generated_claims=[],
                metadata={"provider": "chutes"},
                errors=[f"Chutes.ai processing error: {e}"],
                processing_time=processing_time,
                tokens_used=0
            )
    
    def _convert_request(self, request: LLMRequest) -> Dict[str, Any]:
        """Convert standardized request to Chutes.ai format"""
        # Build context from claims if provided
        context_parts = []
        if request.context_claims:
            context_parts.append("# Context Claims:")
            for claim in request.context_claims:
                type_str = ", ".join([t.value for t in claim.type])
                context_parts.append(f"- [{claim.confidence:.2f}, {type_str}] {claim.content}")
            context_parts.append("")
        
        # Build full prompt
        full_prompt = "\n".join(context_parts + [request.prompt])
        
        return {
            "prompt": full_prompt,
            "context_claims": request.context_claims
        }
    
    def _convert_claims(self, chutes_claims: List) -> List[Claim]:
        """Convert Chutes.ai claims to unified Claim model"""
        unified_claims = []
        
        for chutes_claim in chutes_claims:
            try:
                # Convert Chutes claim to unified model
                unified_claim = Claim(
                    id=chutes_claim.id,
                    content=chutes_claim.content,
                    confidence=chutes_claim.confidence,
                    type=chutes_claim.type,
                    state=chutes_claim.state,
                    tags=getattr(chutes_claim, 'tags', []),
                    created_by=getattr(chutes_claim, 'created_by', 'chutes_ai'),
                    created_at=getattr(chutes_claim, 'created_at', None),
                    updated_at=getattr(chutes_claim, 'updated_at', None)
                )
                unified_claims.append(unified_claim)
            except Exception as e:
                print(f"Error converting claim: {e}")
                continue
        
        return unified_claims
    
    def is_available(self) -> bool:
        """Check if Chutes.ai adapter is available"""
        return self.processor is not None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics"""
        base_stats = super().get_stats()
        
        if self._stats["requests_processed"] > 0:
            success_rate = self._stats["successful_requests"] / self._stats["requests_processed"]
            avg_time = self._stats["total_time"] / self._stats["requests_processed"]
        else:
            success_rate = 0.0
            avg_time = 0.0
        
        base_stats.update({
            "requests_processed": self._stats["requests_processed"],
            "successful_requests": self._stats["successful_requests"],
            "success_rate": success_rate,
            "total_tokens": self._stats["total_tokens"],
            "average_processing_time": avg_time,
            "total_processing_time": self._stats["total_time"]
        })
        
        # Add processor stats if available
        if self.processor:
            processor_stats = self.processor.get_stats()
            base_stats["processor_stats"] = processor_stats
        
        return base_stats


def create_chutes_adapter_from_config() -> ChutesAdapter:
    """
    Factory function to create ChutesAdapter from environment configuration
    Simplifies adapter creation with standard config
    """
    config = Config()
    
    chutes_config = {
        "api_key": getattr(config, 'llm_api_key', None) or os.getenv('CHUTES_API_KEY'),
        "api_url": getattr(config, 'llm_api_url', 'https://llm.chutes.ai/v1'),
        "model": getattr(config, 'llm_model', 'zai-org/GLM-4.6')
    }
    
    return ChutesAdapter(chutes_config)