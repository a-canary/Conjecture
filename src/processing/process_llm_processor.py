"""
ProcessLLMProcessor - LLM processing for Process Layer

This module implements the ProcessLLMProcessor class which handles
LLM processing logic for the Process Layer, integrating with
existing UnifiedLLMManager and following the interface defined
in research.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from ..core.models import Claim, ProcessingResult
from .unified_llm_manager import UnifiedLLMManager, get_unified_llm_manager

logger = logging.getLogger(__name__)


class ProcessLLMProcessor:
    """
    LLM processor for Process Layer operations.
    
    This class handles LLM processing for claims using the
    UnifiedLLMManager and provides a clean interface for
    the Process Layer to interact with LLM providers.
    """
    
    def __init__(self, llm_manager: Optional[UnifiedLLMManager] = None):
        """
        Initialize the ProcessLLMProcessor.
        
        Args:
            llm_manager: Optional UnifiedLLMManager instance
        """
        self.llm_manager = llm_manager or get_unified_llm_manager()
        self._initialized = False
        self._processing_stats = {
            "total_processed": 0,
            "successful_processed": 0,
            "failed_processed": 0,
            "average_processing_time": 0.0
        }
    
    async def initialize(self) -> None:
        """Initialize the LLM processor"""
        if not self.llm_manager:
            raise ValueError("LLM Manager is required")
        
        # Check if we have available providers
        available_providers = self.llm_manager.get_available_providers()
        if not available_providers:
            logger.warning("No LLM providers available")
        
        self._initialized = True
        logger.info(f"ProcessLLMProcessor initialized with {len(available_providers)} providers")
    
    async def process_claim_creation(
        self,
        context: Dict[str, Any],
        provider: Optional[str] = None
    ) -> ProcessingResult:
        """
        Process claim creation using LLM.
        
        This method takes a context dictionary and uses the LLM
        to process the claim creation request.
        
        Args:
            context: The context dictionary built by ProcessContextBuilder
            provider: Optional specific provider to use
            
        Returns:
            ProcessingResult with the outcome of the operation
        """
        if not self._initialized:
            raise RuntimeError("ProcessLLMProcessor not initialized")
        
        start_time = datetime.utcnow()
        errors = []
        
        try:
            # Build the prompt for claim creation
            prompt = self._build_claim_creation_prompt(context)
            
            # Generate response using LLM
            response = self.llm_manager.generate_response(
                prompt=prompt,
                provider=provider,
                max_tokens=1000,
                temperature=0.7
            )
            
            # Parse the response to extract claim data
            claim_data = self._parse_claim_creation_response(response)
            
            if claim_data:
                # Create Claim object
                claim = Claim(
                    id=claim_data.get("id", self._generate_claim_id()),
                    content=claim_data.get("content", context.get("input_content", "")),
                    confidence=claim_data.get("confidence", context.get("input_confidence", 0.5)),
                    tags=claim_data.get("tags", context.get("input_tags", [])),
                    state=claim_data.get("state", "Explore")
                )
                
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                result = ProcessingResult(
                    success=True,
                    processed_claims=1,
                    updated_claims=0,
                    errors=errors,
                    execution_time=execution_time,
                    message=f"Successfully processed claim creation: {claim.id}"
                )
                # Store additional data in a custom attribute
                result._metadata = {
                    "claim": claim.model_dump(),
                    "provider_used": self._get_provider_used(provider),
                    "context_stats": context.get("total_context_claims", 0)
                }
                
                self._update_stats(True, execution_time)
                logger.info(f"Successfully processed claim creation: {claim.id}")
                return result
            else:
                errors.append("Failed to parse LLM response for claim creation")
                
        except Exception as e:
            errors.append(f"Error processing claim creation: {str(e)}")
            logger.error(f"Claim creation processing failed: {e}")
        
        # If we reach here, processing failed
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        result = ProcessingResult(
            success=False,
            processed_claims=0,
            updated_claims=0,
            errors=errors,
            execution_time=execution_time,
            message="Claim creation processing failed"
        )
        
        self._update_stats(False, execution_time)
        return result
    
    async def process_claim_analysis(
        self,
        context: Dict[str, Any],
        provider: Optional[str] = None
    ) -> ProcessingResult:
        """
        Process claim analysis using LLM.
        
        This method takes a context dictionary and uses the LLM
        to analyze an existing claim.
        
        Args:
            context: The context dictionary built by ProcessContextBuilder
            provider: Optional specific provider to use
            
        Returns:
            ProcessingResult with the outcome of the operation
        """
        if not self._initialized:
            raise RuntimeError("ProcessLLMProcessor not initialized")
        
        start_time = datetime.utcnow()
        errors = []
        
        try:
            # Build the prompt for claim analysis
            prompt = self._build_claim_analysis_prompt(context)
            
            # Generate response using LLM
            response = self.llm_manager.generate_response(
                prompt=prompt,
                provider=provider,
                max_tokens=1500,
                temperature=0.3
            )
            
            # Parse the response to extract analysis
            analysis = self._parse_claim_analysis_response(response)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            result = ProcessingResult(
                success=True,
                processed_claims=1,
                updated_claims=0,
                errors=errors,
                execution_time=execution_time,
                message=f"Successfully analyzed claim: {context.get('target_claim', {}).get('id', 'unknown')}"
            )
            # Store additional data in a custom attribute
            result._metadata = {
                "analysis": analysis,
                "provider_used": self._get_provider_used(provider),
                "context_stats": context.get("total_context_claims", 0)
            }
            
            self._update_stats(True, execution_time)
            logger.info(f"Successfully analyzed claim: {context.get('target_claim', {}).get('id', 'unknown')}")
            return result
            
        except Exception as e:
            errors.append(f"Error processing claim analysis: {str(e)}")
            logger.error(f"Claim analysis processing failed: {e}")
        
        # If we reach here, processing failed
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        result = ProcessingResult(
            success=False,
            processed_claims=0,
            updated_claims=0,
            errors=errors,
            execution_time=execution_time,
            message="Claim analysis processing failed"
        )
        
        self._update_stats(False, execution_time)
        return result
    
    def _build_claim_creation_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for claim creation"""
        prompt_parts = [
            "You are an AI assistant helping to create evidence-based claims.",
            "",
            "TASK: Create a well-structured claim based on the following input:",
            f"Content: {context.get('input_content', '')}",
            f"Confidence: {context.get('input_confidence', 0.5)}",
            f"Tags: {', '.join(context.get('input_tags', []))}",
            ""
        ]
        
        # Add context claims if available
        similar_claims = context.get('similar_claims', [])
        recent_claims = context.get('recent_claims', [])
        
        if similar_claims or recent_claims:
            prompt_parts.append("CONTEXT: Here are some related existing claims for reference:")
            
            for claim in similar_claims[:3]:
                prompt_parts.append(f"- [c{claim['id']} | {claim['content']} | / {claim['confidence']:.2f}]")
            
            for claim in recent_claims[:2]:
                prompt_parts.append(f"- [c{claim['id']} | {claim['content']} | / {claim['confidence']:.2f}]")
            
            prompt_parts.append("")
        
        prompt_parts.extend([
            "Please provide a structured response with the following format:",
            "{",
            '  "content": "refined claim content",',
            '  "confidence": 0.85,',
            '  "tags": ["tag1", "tag2"],',
            '  "state": "Explore",',
            '  "reasoning": "brief explanation of any refinements made"',
            "}",
            "",
            "Focus on clarity, evidence-based reasoning, and appropriate confidence levels."
        ])
        
        return "\n".join(prompt_parts)
    
    def _build_claim_analysis_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for claim analysis"""
        target_claim = context.get('target_claim', {})
        
        prompt_parts = [
            "You are an AI assistant helping to analyze evidence-based claims.",
            "",
            "TASK: Analyze the following claim and provide insights:",
            f"Claim ID: {target_claim.get('id', 'unknown')}",
            f"Content: {target_claim.get('content', '')}",
            f"Confidence: {target_claim.get('confidence', 0.5)}",
            f"State: {target_claim.get('state', 'unknown')}",
            f"Tags: {', '.join(target_claim.get('tags', []))}",
            ""
        ]
        
        # Add context claims if available
        similar_claims = context.get('similar_claims', [])
        related_claims = context.get('related_claims', [])
        
        if similar_claims or related_claims:
            prompt_parts.append("CONTEXT: Here are some related claims:")
            
            for claim in similar_claims[:3]:
                prompt_parts.append(f"- SIMILAR: [c{claim['id']} | {claim['content']} | / {claim['confidence']:.2f}]")
            
            for claim in related_claims[:3]:
                prompt_parts.append(f"- RELATED: [c{claim['id']} | {claim['content']} | / {claim['confidence']:.2f}]")
            
            prompt_parts.append("")
        
        prompt_parts.extend([
            "Please provide a comprehensive analysis including:",
            "1. Strengths and weaknesses of the claim",
            "2. Confidence assessment and justification",
            "3. Suggested improvements or refinements",
            "4. Potential supporting evidence needed",
            "5. Relationships with other claims",
            "",
            "Provide your analysis in a structured, clear format."
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_claim_creation_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response for claim creation"""
        try:
            import json
            
            # Try to extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Fallback: extract basic information
                return {
                    "content": response.strip(),
                    "confidence": 0.7,
                    "tags": [],
                    "state": "Explore"
                }
                
        except Exception as e:
            logger.error(f"Failed to parse claim creation response: {e}")
            return None
    
    def _parse_claim_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response for claim analysis"""
        return {
            "analysis_text": response.strip(),
            "parsed_at": datetime.utcnow().isoformat()
        }
    
    def _generate_claim_id(self) -> str:
        """Generate a new claim ID"""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _get_provider_used(self, requested_provider: Optional[str]) -> str:
        """Get the actual provider used for processing"""
        if requested_provider:
            return requested_provider
        
        if self.llm_manager.primary_provider:
            return self.llm_manager.primary_provider
        
        return "auto-selected"
    
    def _update_stats(self, success: bool, execution_time: float) -> None:
        """Update processing statistics"""
        self._processing_stats["total_processed"] += 1
        
        if success:
            self._processing_stats["successful_processed"] += 1
        else:
            self._processing_stats["failed_processed"] += 1
        
        # Update average processing time
        total = self._processing_stats["total_processed"]
        current_avg = self._processing_stats["average_processing_time"]
        self._processing_stats["average_processing_time"] = (
            (current_avg * (total - 1) + execution_time) / total
        )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        stats = self._processing_stats.copy()
        stats["success_rate"] = (
            stats["successful_processed"] / stats["total_processed"]
            if stats["total_processed"] > 0 else 0.0
        )
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the processor"""
        if not self._initialized:
            return {"status": "uninitialized", "message": "Processor not initialized"}
        
        provider_health = self.llm_manager.health_check()
        
        return {
            "status": "healthy" if provider_health["overall_status"] == "healthy" else "degraded",
            "initialized": self._initialized,
            "provider_health": provider_health,
            "processing_stats": self._processing_stats
        }