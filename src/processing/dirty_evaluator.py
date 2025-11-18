"""
Dirty Claim Evaluator for Conjecture
Handles batch processing and LLM evaluation of dirty claims with two-pass system
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from ..core.models import Claim, DirtyReason, ProcessingResult
from ..core.dirty_flag import DirtyFlagSystem
from .llm.llm_manager import LLMManager


class DirtyClaimBatch:
    """Batch of dirty claims for evaluation"""
    
    def __init__(
        self,
        claims: List[Claim],
        batch_id: str,
        priority_level: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.claims = claims
        self.batch_id = batch_id
        self.priority_level = priority_level
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
        self.processing_started_at: Optional[datetime] = None
        self.processing_completed_at: Optional[datetime] = None
        self.status = "pending"  # pending, processing, completed, failed
        self.results: List[Dict[str, Any]] = []
        self.errors: List[str] = []


class DirtyEvaluationConfig:
    """Configuration for dirty claim evaluation"""
    
    def __init__(
        self,
        batch_size: int = 5,
        max_parallel_batches: int = 3,
        confidence_threshold: float = 0.90,
        confidence_boost_factor: float = 0.10,
        enable_two_pass: bool = True,
        relationship_threshold: float = 0.7,
        timeout_seconds: int = 300,
        max_retries: int = 2
    ):
        self.batch_size = batch_size
        self.max_parallel_batches = max_parallel_batches
        self.confidence_threshold = confidence_threshold
        self.confidence_boost_factor = confidence_boost_factor
        self.enable_two_pass = enable_two_pass
        self.relationship_threshold = relationship_threshold
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries


class DirtyEvaluator:
    """Evaluates dirty claims using LLM with parallel batch processing"""

    def __init__(
        self,
        llm_manager: LLMManager,
        dirty_flag_system: DirtyFlagSystem,
        config: Optional[DirtyEvaluationConfig] = None
    ):
        """
        Initialize dirty evaluator
        
        Args:
            llm_manager: LLM manager for claim evaluation
            dirty_flag_system: Dirty flag system for claim management
            config: Evaluation configuration
        """
        self.llm_manager = llm_manager
        self.dirty_flag_system = dirty_flag_system
        self.config = config or DirtyEvaluationConfig()
        self.logger = logging.getLogger(__name__)
        self.processing_stats = {
            "total_processed": 0,
            "successful_evaluations": 0,
            "failed_evaluations": 0,
            "average_processing_time": 0.0,
            "total_batches_processed": 0
        }

    async def evaluate_dirty_claims(
        self,
        claims: Optional[List[Claim]] = None,
        priority_only: bool = True,
        max_claims: Optional[int] = None
    ) -> ProcessingResult:
        """
        Evaluate dirty claims with parallel batch processing
        
        Args:
            claims: List of claims to evaluate (gets dirty claims if None)
            priority_only: Whether to only evaluate priority claims
            max_claims: Maximum number of claims to evaluate
            
        Returns:
            ProcessingResult with evaluation details
        """
        start_time = datetime.utcnow()
        
        try:
            # Get dirty claims to evaluate
            if priority_only:
                dirty_claims = self.dirty_flag_system.get_priority_dirty_claims(
                    claims=claims,
                    confidence_threshold=self.config.confidence_threshold,
                    max_count=max_claims
                )
            else:
                dirty_claims = self.dirty_flag_system.get_dirty_claims(
                    claims=claims,
                    prioritize=True,
                    max_count=max_claims
                )

            if not dirty_claims:
                return ProcessingResult(
                    success=True,
                    processed_claims=0,
                    updated_claims=0,
                    message="No dirty claims to evaluate"
                )

            self.logger.info(f"Starting evaluation of {len(dirty_claims)} dirty claims")

            # Create batches
            batches = self._create_evaluation_batches(dirty_claims)
            
            # Process batches in parallel
            updated_claims = await self._process_batches_parallel(batches)

            # Clear dirty flags from successfully processed claims
            successfully_processed = [
                claim for claim in dirty_claims 
                if updated_claims and claim.id in [c.id for c in updated_claims]
            ]
            
            for claim in successfully_processed:
                claim.mark_clean()

            # Update statistics
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_processing_stats(len(dirty_claims), len(updated_claims), execution_time)

            return ProcessingResult(
                success=True,
                processed_claims=len(dirty_claims),
                updated_claims=len(updated_claims),
                execution_time=execution_time,
                message=f"Successfully evaluated {len(updated_claims)} claims"
            )

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.logger.error(f"Dirty claim evaluation failed: {e}")
            return ProcessingResult(
                success=False,
                processed_claims=0,
                updated_claims=0,
                errors=[str(e)],
                execution_time=execution_time,
                message="Evaluation failed"
            )

    def _create_evaluation_batches(self, claims: List[Claim]) -> List[DirtyClaimBatch]:
        """Create evaluation batches from claims"""
        batches = []
        
        for i in range(0, len(claims), self.config.batch_size):
            batch_claims = claims[i:i + self.config.batch_size]
            batch_id = f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{i // self.config.batch_size + 1}"
            
            # Calculate batch priority based on claims
            batch_priority = max([claim.dirty_priority for claim in batch_claims])
            
            batch = DirtyClaimBatch(
                claims=batch_claims,
                batch_id=batch_id,
                priority_level=batch_priority,
                metadata={
                    "claim_count": len(batch_claims),
                    "confidence_range": (
                        min([c.confidence for c in batch_claims]),
                        max([c.confidence for c in batch_claims])
                    )
                }
            )
            batches.append(batch)
        
        # Sort batches by priority (higher priority first)
        batches.sort(key=lambda b: b.priority_level, reverse=True)
        
        self.logger.info(f"Created {len(batches)} evaluation batches")
        return batches

    async def _process_batches_parallel(self, batches: List[DirtyClaimBatch]) -> List[Claim]:
        """Process multiple batches in parallel"""
        updated_claims = []
        
        # Process batches in parallel with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.config.max_parallel_batches)
        
        async def process_single_batch(batch: DirtyClaimBatch) -> List[Claim]:
            async with semaphore:
                return await self._process_single_batch(batch)
        
        # Execute all batches
        tasks = [process_single_batch(batch) for batch in batches]
        
        # Wait for all batches to complete
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        for result in batch_results:
            if isinstance(result, Exception):
                self.logger.error(f"Batch processing failed: {result}")
            else:
                updated_claims.extend(result)
        
        return updated_claims

    async def _process_single_batch(self, batch: DirtyClaimBatch) -> List[Claim]:
        """Process a single batch of dirty claims"""
        batch.processing_started_at = datetime.utcnow()
        batch.status = "processing"
        
        try:
            self.logger.debug(f"Processing batch {batch.batch_id} with {len(batch.claims)} claims")
            
            # Apply confidence boost for re-evaluated claims
            boosted_claims = self._apply_confidence_boost(batch.claims)
            
            if self.config.enable_two_pass:
                # Two-pass evaluation: claims first, then relationships
                updated_claims = await self._two_pass_evaluation(boosted_claims)
            else:
                # Single-pass evaluation
                updated_claims = await self._single_pass_evaluation(boosted_claims)
            
            batch.processing_completed_at = datetime.utcnow()
            batch.status = "completed"
            
            self.logger.debug(f"Batch {batch.batch_id} completed: {len(updated_claims)} claims updated")
            return updated_claims
            
        except Exception as e:
            batch.status = "failed"
            batch.errors.append(str(e))
            self.logger.error(f"Batch {batch.batch_id} failed: {e}")
            return []
        finally:
            # Update batch results
            execution_time = (
                batch.processing_completed_at - batch.processing_started_at
            ).total_seconds() if batch.processing_completed_at else 0
            
            batch.results.append({
                "execution_time": execution_time,
                "claim_count": len(batch.claims),
                "status": batch.status
            })

    def _apply_confidence_boost(self, claims: List[Claim]) -> List[Claim]:
        """Apply confidence boost to re-evaluated claims"""
        boosted_claims = []
        
        for claim in claims:
            # Create a copy or modify in place based on your data model
            boosted_claim = claim  # In real implementation, you might want to copy
            
            # Apply confidence boost for low-confidence claims
            if claim.confidence < self.config.confidence_threshold:
                boost = min(
                    self.config.confidence_boost_factor,
                    self.config.confidence_threshold - claim.confidence
                )
                boosted_claim.confidence += boost
                self.logger.debug(
                    f"Applied confidence boost +{boost:.2f} to claim {claim.id}"
                )
            
            boosted_claims.append(boosted_claim)
        
        return boosted_claims

    async def _two_pass_evaluation(self, claims: List[Claim]) -> List[Claim]:
        """Perform two-pass evaluation: claims first, then relationships"""
        updated_claims = []
        
        # First pass: Evaluate individual claims
        batch_results = await self._evaluate_claims_batch(claims, task="evaluate_claims")
        
        # Update claims with LLM results
        for claim, result in zip(claims, batch_results):
            if result.get("success"):
                updated_claim = self._update_claim_from_llm_result(claim, result)
                if updated_claim:
                    updated_claims.append(updated_claim)
        
        # Second pass: Evaluate relationships between claims
        if len(updated_claims) > 1:
            relationship_results = await self._evaluate_relationships_batch(updated_claims)
            
            # Update claim relationships based on LLM results
            for claim, rel_result in zip(updated_claims, relationship_results):
                if rel_result.get("success"):
                    self._update_claim_relationships(claim, rel_result)
        
        return updated_claims

    async def _single_pass_evaluation(self, claims: List[Claim]) -> List[Claim]:
        """Perform single-pass evaluation of claims"""
        # Evaluate claims with comprehensive analysis
        batch_results = await self._evaluate_claims_batch(claims, task="comprehensive_evaluation")
        
        updated_claims = []
        for claim, result in zip(claims, batch_results):
            if result.get("success"):
                updated_claim = self._update_claim_from_llm_result(claim, result)
                if updated_claim:
                    updated_claims.append(updated_claim)
        
        return updated_claims

    async def _evaluate_claims_batch(self, claims: List[Claim], task: str) -> List[Dict[str, Any]]:
        """Evaluate a batch of claims using LLM"""
        try:
            # Prepare claims for LLM processing
            basic_claims = [
                {
                    "id": claim.id,
                    "content": claim.content,
                    "confidence": claim.confidence,
                    "type": [t.value for t in claim.type],
                    "tags": claim.tags,
                    "context": claim.format_for_context()
                }
                for claim in claims
            ]
            
            # Use LLM manager to process claims
            llm_result = self.llm_manager.process_claims(
                basic_claims,
                task=task,
                batch_id=claims[0].id if claims else "batch",
                max_tokens=1000,
                temperature=0.3
            )
            
            if hasattr(llm_result, 'success') and llm_result.success:
                return self._parse_llm_response(llm_result, len(claims))
            else:
                self.logger.warning(f"LLM processing returned no results for batch")
                return [{"success": False, "error": "No LLM results"}] * len(claims)
                
        except Exception as e:
            self.logger.error(f"Batch evaluation failed: {e}")
            return [{"success": False, "error": str(e)}] * len(claims)

    async def _evaluate_relationships_batch(self, claims: List[Claim]) -> List[Dict[str, Any]]:
        """Evaluate relationships between claims"""
        try:
            # Create relationship evaluation prompt
            context = {
                "claims": [
                    {
                        "id": claim.id,
                        "content": claim.content,
                        "confidence": claim.confidence,
                        "supported_by": claim.supported_by,
                        "supports": claim.supports
                    }
                    for claim in claims
                ],
                "task": "evaluate_relationships",
                "threshold": self.config.relationship_threshold
            }
            
            prompt = self._create_relationship_prompt(context)
            
            # Generate response
            response = self.llm_manager.generate_response(
                prompt,
                max_tokens=800,
                temperature=0.2
            )
            
            if hasattr(response, 'success') and response.success:
                return self._parse_relationship_response(response, len(claims))
            else:
                return [{"success": False, "error": "Relationship evaluation failed"}] * len(claims)
                
        except Exception as e:
            self.logger.error(f"Relationship evaluation failed: {e}")
            return [{"success": False, "error": str(e)}] * len(claims)

    def _update_claim_from_llm_result(self, claim: Claim, result: Dict[str, Any]) -> Optional[Claim]:
        """Update claim based on LLM evaluation result"""
        try:
            if not result.get("success"):
                return None
            
            data = result.get("data", {})
            
            # Update confidence if provided
            if "confidence" in data:
                new_confidence = float(data["confidence"])
                if 0.0 <= new_confidence <= 1.0:
                    claim.update_confidence(new_confidence)
            
            # Update state if provided
            if "state" in data:
                from ..core.models import ClaimState
                try:
                    new_state = ClaimState(data["state"])
                    claim.state = new_state
                except ValueError:
                    self.logger.warning(f"Invalid state {data['state']} for claim {claim.id}")
            
            # Update tags if provided
            if "tags" in data and isinstance(data["tags"], list):
                claim.tags = [str(tag) for tag in data["tags"]]
            
            # Update timestamp
            claim.updated = datetime.utcnow()
            
            return claim
            
        except Exception as e:
            self.logger.error(f"Failed to update claim {claim.id} from LLM result: {e}")
            return None

    def _update_claim_relationships(self, claim: Claim, rel_result: Dict[str, Any]) -> None:
        """Update claim relationships based on relationship evaluation"""
        try:
            if not rel_result.get("success"):
                return
            
            data = rel_result.get("data", {})
            
            # Update supported_by relationships
            if "supported_by" in data and isinstance(data["supported_by"], list):
                claim.supported_by = [str(support_id) for support_id in data["supported_by"]]
            
            # Update supports relationships
            if "supports" in data and isinstance(data["supports"], list):
                claim.supports = [str(supported_id) for supported_id in data["supports"]]
            
            claim.updated = datetime.utcnow()
            
        except Exception as e:
            self.logger.error(f"Failed to update relationships for claim {claim.id}: {e}")

    def _parse_llm_response(self, llm_result, claim_count: int) -> List[Dict[str, Any]]:
        """Parse LLM response into individual claim results"""
        results = []
        
        try:
            # Extract response content
            if hasattr(llm_result, 'content'):
                content = llm_result.content
            else:
                content = str(llm_result)
            
            # Try to parse as JSON
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and "results" in data:
                    return data["results"]
                else:
                    # Single result for single claim
                    return [data] * claim_count
            except json.JSONDecodeError:
                # Fallback: create simple successful results
                return [{"success": True, "data": {}}] * claim_count
                
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            return [{"success": False, "error": "Response parsing failed"}] * claim_count

    def _parse_relationship_response(self, llm_result, claim_count: int) -> List[Dict[str, Any]]:
        """Parse relationship evaluation response"""
        return self._parse_llm_response(llm_result, claim_count)

    def _create_relationship_prompt(self, context: Dict[str, Any]) -> str:
        """Create relationship evaluation prompt"""
        claims_info = "\n".join([
            f"Claim {claim['id']}: {claim['content']} (confidence: {claim['confidence']})"
            for claim in context["claims"]
        ])
        
        prompt = f"""
Analyze the relationships between the following claims:

{claims_info}

For each claim, determine:
1. Which other claims it supports (based on logical consistency, evidence, etc.)
2. Which other claims support it
3. The strength of these relationships (considering confidence scores)

Relationship threshold: {context['threshold']}

Return results in JSON format:
{{
  "results": [
    {{
      "claim_id": "id",
      "supports": ["supported_claim_ids"],
      "supported_by": ["supporting_claim_ids"]
    }}
  ]
}}
"""
        return prompt

    def _update_processing_stats(
        self,
        processed_count: int,
        updated_count: int,
        execution_time: float
    ) -> None:
        """Update processing statistics"""
        self.processing_stats["total_processed"] += processed_count
        self.processing_stats["total_batches_processed"] += 1
        self.processing_stats["average_processing_time"] = (
            (self.processing_stats["average_processing_time"] * (self.processing_stats["total_batches_processed"] - 1) + execution_time) /
            self.processing_stats["total_batches_processed"]
        )
        
        if updated_count > 0:
            self.processing_stats["successful_evaluations"] += updated_count
        else:
            self.processing_stats["failed_evaluations"] += processed_count

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return {
            **self.processing_stats,
            "success_rate": (
                self.processing_stats["successful_evaluations"] / 
                max(self.processing_stats["total_processed"], 1)
            )
        }

    def reset_stats(self) -> None:
        """Reset processing statistics"""
        self.processing_stats = {
            "total_processed": 0,
            "successful_evaluations": 0,
            "failed_evaluations": 0,
            "average_processing_time": 0.0,
            "total_batches_processed": 0
        }