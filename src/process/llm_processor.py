"""
Process LLM Processor

The ProcessLLMProcessor is responsible for claim evaluation and instruction
identification using language models within the Process Layer.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import asyncio
import logging
import json

from src.core.models import Claim
from src.processing.llm_bridge import LLMBridge
from .models import (
    ContextResult,
    ProcessingResult,
    Instruction,
    InstructionType,
    ProcessingStatus,
    ProcessingConfig,
    ProcessingRequest
)

logger = logging.getLogger(__name__)

class ProcessLLMProcessor:
    """
    Processes claims using language models to evaluate claims and identify
    instructions for subsequent actions.
    
    This class serves as the core LLM integration point in the Process Layer,
    handling claim evaluation, instruction identification, and processing
    workflow coordination.
    """
    
    def __init__(
        self,
        llm_bridge: LLMBridge,
        config: Optional[ProcessingConfig] = None
    ):
        """
        Initialize the ProcessLLMProcessor.
        
        Args:
            llm_bridge: Bridge for interacting with language models
            config: Processing configuration (optional)
        """
        self.llm_bridge = llm_bridge
        self.config = config or ProcessingConfig()
        self._processing_cache: Dict[str, ProcessingResult] = {}
        
    async def process_claim(
        self,
        request: ProcessingRequest,
        context: Optional[ContextResult] = None
    ) -> ProcessingResult:
        """
        Process a claim with evaluation and instruction identification.
        
        Args:
            request: Processing request containing claim and parameters
            context: Optional pre-built context for the claim
            
        Returns:
            ProcessingResult with evaluation and identified instructions
            
        Raises:
            ValueError: If request is invalid
            RuntimeError: If processing fails
        """
        start_time = datetime.utcnow()
        
        try:
            # Validate request
            if not request.claim_id:
                raise ValueError("Claim ID is required in processing request")
            
            # Check cache first
            cache_key = self._generate_cache_key(request)
            if cache_key in self._processing_cache:
                logger.info(f"Returning cached result for claim {request.claim_id}")
                return self._processing_cache[cache_key]
            
            # Update status to in progress
            result = ProcessingResult(
                claim_id=request.claim_id,
                status=ProcessingStatus.IN_PROGRESS,
                created_at=start_time
            )
            
            # Build evaluation prompt
            evaluation_prompt = await self._build_evaluation_prompt(
                request, 
                context
            )
            
            # Process with LLM
            llm_response = await self._evaluate_with_llm(evaluation_prompt)
            
            # Parse LLM response
            instructions = await self._parse_instructions(llm_response, request)
            evaluation_score = await self._extract_evaluation_score(llm_response)
            reasoning = await self._extract_reasoning(llm_response)
            
            # Update result with processing outcomes
            processing_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            result.status = ProcessingStatus.COMPLETED
            result.instructions = instructions
            result.evaluation_score = evaluation_score
            result.reasoning = reasoning
            result.processing_time_ms = processing_time_ms
            result.metadata = {
                "cache_key": cache_key,
                "llm_model": getattr(self.llm_bridge, 'model_name', 'unknown'),
                "prompt_tokens": len(evaluation_prompt.split()),
                "instruction_types_found": [inst.instruction_type for inst in instructions]
            }
            
            # Cache successful results
            if result.status == ProcessingStatus.COMPLETED:
                self._processing_cache[cache_key] = result
            
            logger.info(
                f"Processed claim {request.claim_id}: "
                f"score={evaluation_score:.2f}, "
                f"instructions={len(instructions)}, "
                f"time={processing_time_ms}ms"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process claim {request.claim_id}: {str(e)}")
            
            processing_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return ProcessingResult(
                claim_id=request.claim_id,
                status=ProcessingStatus.FAILED,
                error_message=str(e),
                processing_time_ms=processing_time_ms,
                created_at=start_time
            )
    
    async def _build_evaluation_prompt(
        self,
        request: ProcessingRequest,
        context: Optional[ContextResult]
    ) -> str:
        """
        Build the evaluation prompt for the LLM.
        
        Args:
            request: Processing request
            context: Optional context for the claim
            
        Returns:
            Formatted prompt string
        """
        # This is a skeleton implementation
        # In a full implementation, this would:
        # - Format the claim text
        # - Include relevant context claims
        # - Add context hints
        # - Specify instruction types to look for
        # - Include evaluation criteria
        
        prompt_parts = [
            "You are an AI assistant that evaluates claims and identifies instructions.",
            "Analyze the following claim and determine what actions should be taken.",
            "",
            f"Claim ID: {request.claim_id}",
        ]
        
        if context:
            prompt_parts.extend([
                "",
                "Context:",
                f"Related claims: {len(context.context_claims)}",
                f"Context size: {context.context_size} tokens"
            ])
        
        if request.instruction_types:
            types_str = ", ".join([t.value for t in request.instruction_types])
            prompt_parts.extend([
                "",
                f"Look for these instruction types: {types_str}"
            ])
        
        if request.context_hints:
            hints_str = ", ".join(request.context_hints)
            prompt_parts.extend([
                "",
                f"Context hints: {hints_str}"
            ])
        
        prompt_parts.extend([
            "",
            "Please provide:",
            "1. An evaluation score (0.0-1.0) for the claim",
            "2. Reasoning for your evaluation",
            "3. Any instructions that should be executed",
            "",
            "Respond in JSON format with keys: evaluation_score, reasoning, instructions"
        ])
        
        return "\n".join(prompt_parts)
    
    async def _evaluate_with_llm(self, prompt: str) -> str:
        """
        Send the evaluation prompt to the LLM and get response.
        
        Args:
            prompt: Evaluation prompt
            
        Returns:
            LLM response string
        """
        # This is a skeleton implementation
        # In a full implementation, this would:
        # - Use the LLMBridge to send the prompt
        # - Handle timeouts and retries
        # - Validate the response format
        
        try:
            response = await self.llm_bridge.generate_response(prompt)
            return response
        except Exception as e:
            logger.error(f"LLM evaluation failed: {str(e)}")
            raise RuntimeError(f"LLM evaluation failed: {str(e)}")
    
    async def _parse_instructions(
        self,
        llm_response: str,
        request: ProcessingRequest
    ) -> List[Instruction]:
        """
        Parse instructions from the LLM response.
        
        Args:
            llm_response: Response from the LLM
            request: Original processing request
            
        Returns:
            List of parsed instructions
        """
        # This is a skeleton implementation
        # In a full implementation, this would:
        # - Parse JSON response from LLM
        # - Validate instruction types
        # - Filter by confidence threshold
        # - Apply priority scoring
        
        instructions = []
        
        try:
            # Try to parse as JSON
            response_data = json.loads(llm_response)
            
            if "instructions" in response_data:
                for instr_data in response_data["instructions"]:
                    # Validate instruction type
                    instr_type = instr_data.get("type", "custom_action")
                    if instr_type not in [t.value for t in InstructionType]:
                        instr_type = "custom_action"
                    
                    instruction = Instruction(
                        instruction_type=InstructionType(instr_type),
                        description=instr_data.get("description", ""),
                        parameters=instr_data.get("parameters", {}),
                        confidence=instr_data.get("confidence", 0.0),
                        priority=instr_data.get("priority", 0),
                        source_claim_id=request.claim_id
                    )
                    
                    # Filter by confidence threshold
                    if instruction.confidence >= self.config.instruction_confidence_threshold:
                        instructions.append(instruction)
                        
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse instructions from LLM response: {str(e)}")
        
        return instructions
    
    async def _extract_evaluation_score(self, llm_response: str) -> Optional[float]:
        """
        Extract the evaluation score from the LLM response.
        
        Args:
            llm_response: Response from the LLM
            
        Returns:
            Evaluation score or None if not found
        """
        try:
            response_data = json.loads(llm_response)
            score = response_data.get("evaluation_score")
            if isinstance(score, (int, float)) and 0.0 <= score <= 1.0:
                return float(score)
        except (json.JSONDecodeError, ValueError):
            pass
        
        return None
    
    async def _extract_reasoning(self, llm_response: str) -> Optional[str]:
        """
        Extract reasoning from the LLM response.
        
        Args:
            llm_response: Response from the LLM
            
        Returns:
            Reasoning string or None if not found
        """
        try:
            response_data = json.loads(llm_response)
            reasoning = response_data.get("reasoning")
            if isinstance(reasoning, str):
                return reasoning
        except (json.JSONDecodeError, ValueError):
            pass
        
        return None
    
    def _generate_cache_key(self, request: ProcessingRequest) -> str:
        """
        Generate a cache key for the processing request.
        
        Args:
            request: Processing request
            
        Returns:
            Cache key string
        """
        # Create a deterministic key from request parameters
        key_parts = [
            request.claim_id,
            str(sorted(request.context_hints)),
            str(sorted([t.value for t in request.instruction_types]))
        ]
        return "|".join(key_parts)
    
    def clear_cache(self) -> None:
        """Clear the processing cache."""
        self._processing_cache.clear()
        logger.info("Processing cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the processing cache.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "cached_results": len(self._processing_cache),
            "cache_keys": list(self._processing_cache.keys())
        }
    
    async def process_batch(
        self,
        requests: List[ProcessingRequest]
    ) -> List[ProcessingResult]:
        """
        Process multiple claims in batch.
        
        Args:
            requests: List of processing requests
            
        Returns:
            List of processing results
        """
        if not self.config.enable_parallel_processing:
            # Sequential processing
            results = []
            for request in requests:
                result = await self.process_claim(request)
                results.append(result)
            return results
        
        # Parallel processing
        tasks = [self.process_claim(request) for request in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)