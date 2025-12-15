"""
Process Layer Models

This module defines the core data structures used by the Process Layer
for claim evaluation, instruction identification, and processing workflow.
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict, field_serializer

from src.core.models import Claim

class InstructionType(str, Enum):
    """Types of instructions that can be identified during claim processing."""
    
    CREATE_CLAIM = "create_claim"
    UPDATE_CLAIM = "update_claim"
    DELETE_CLAIM = "delete_claim"
    SEARCH_CLAIMS = "search_claims"
    ANALYZE_CLAIM = "analyze_claim"
    VALIDATE_CLAIM = "validate_claim"
    CONNECT_CLAIMS = "connect_claims"
    GENERATE_CONTEXT = "generate_context"
    CUSTOM_ACTION = "custom_action"

class ProcessingStatus(str, Enum):
    """Status of processing operations."""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class ContextResult(BaseModel):
    """Result of context building operations."""
    
    claim_id: str = Field(..., description="ID of the primary claim")
    context_claims: List[Claim] = Field(default_factory=list, description="Related claims for context")
    context_size: int = Field(default=0, description="Size of the context in tokens")
    traversal_depth: int = Field(default=0, description="Depth of claim graph traversal")
    build_time_ms: int = Field(default=0, description="Time taken to build context in milliseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context metadata")
    
    model_config = ConfigDict(protected_namespaces=())
    
    @field_serializer('context_claims')
    def serialize_context_claims(self, value: List[Claim]) -> List[Dict[str, Any]]:
        return [claim.to_dict() for claim in value]

class Instruction(BaseModel):
    """Instruction identified during claim processing."""
    
    instruction_type: InstructionType = Field(..., description="Type of instruction")
    description: str = Field(..., description="Human-readable description of the instruction")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the instruction")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in this instruction")
    priority: int = Field(default=0, description="Priority of this instruction (higher = more important)")
    source_claim_id: Optional[str] = Field(None, description="ID of claim that generated this instruction")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When this instruction was created")
    
    model_config = ConfigDict(protected_namespaces=())
    
    @field_serializer('created_at')
    def serialize_created_at(self, value: datetime) -> str:
        return value.isoformat()

class ProcessingResult(BaseModel):
    """Result of claim processing operations."""
    
    claim_id: str = Field(..., description="ID of the processed claim")
    status: ProcessingStatus = Field(..., description="Processing status")
    instructions: List[Instruction] = Field(default_factory=list, description="Identified instructions")
    evaluation_score: Optional[float] = Field(None, description="Evaluation score (0.0-1.0)")
    reasoning: Optional[str] = Field(None, description="Reasoning for the evaluation")
    processing_time_ms: int = Field(default=0, description="Total processing time in milliseconds")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional processing metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When processing was completed")
    
    model_config = ConfigDict(protected_namespaces=())
    
    @field_serializer('created_at')
    def serialize_created_at(self, value: datetime) -> str:
        return value.isoformat()
    
    @field_serializer('instructions')
    def serialize_instructions(self, value: List[Instruction]) -> List[Dict[str, Any]]:
        return [instruction.model_dump() for instruction in value]

class ProcessingConfig(BaseModel):
    """Configuration for processing operations."""
    
    max_context_size: int = Field(default=10, description="Maximum number of claims in context")
    max_traversal_depth: int = Field(default=3, description="Maximum depth for claim graph traversal")
    instruction_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum confidence for instructions")
    enable_parallel_processing: bool = Field(default=True, description="Enable parallel processing when possible")
    timeout_seconds: int = Field(default=300, description="Timeout for processing operations")
    retry_attempts: int = Field(default=3, description="Number of retry attempts for failed operations")
    
    model_config = ConfigDict(extra="allow")

class ProcessingRequest(BaseModel):
    """Request for claim processing."""
    
    claim_id: str = Field(..., description="ID of claim to process")
    config: Optional[ProcessingConfig] = Field(None, description="Processing configuration overrides")
    context_hints: List[str] = Field(default_factory=list, description="Hints for context building")
    instruction_types: List[InstructionType] = Field(default_factory=list, description="Specific instruction types to look for")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional request metadata")
    requested_at: datetime = Field(default_factory=datetime.utcnow, description="When the request was made")
    
    model_config = ConfigDict()
    
    @field_serializer('requested_at')
    def serialize_requested_at(self, value: datetime) -> str:
        return value.isoformat()
    
    @field_serializer('instruction_types')
    def serialize_instruction_types(self, value: List[InstructionType]) -> List[str]:
        return [instruction_type.value for instruction_type in value]

class ProcessingBatch(BaseModel):
    """Batch of processing requests for bulk operations."""
    
    requests: List[ProcessingRequest] = Field(..., description="List of processing requests")
    batch_id: str = Field(..., description="Unique identifier for this batch")
    config: Optional[ProcessingConfig] = Field(None, description="Batch-level configuration")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the batch was created")
    
    model_config = ConfigDict()
    
    @field_serializer('created_at')
    def serialize_created_at(self, value: datetime) -> str:
        return value.isoformat()
    
    @field_serializer('requests')
    def serialize_requests(self, value: List[ProcessingRequest]) -> List[Dict[str, Any]]:
        return [request.model_dump() for request in value]