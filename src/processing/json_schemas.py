"""
JSON Schemas for LLM Response Types

Defines comprehensive JSON schemas for all supported response types in the
JSON frontmatter format. Each schema includes validation rules and examples.
"""

from typing import Dict, Any, List
from enum import Enum
from pydantic import BaseModel, Field, field_validator
import json


class ResponseSchemaType(str, Enum):
    """Supported response schema types"""
    CLAIMS = "claims"
    ANALYSIS = "analysis" 
    VALIDATION = "validation"
    INSTRUCTION_SUPPORT = "instruction_support"
    RELATIONSHIP_ANALYSIS = "relationship_analysis"
    ERROR = "error"
    EXPLORATION = "exploration"
    RESEARCH = "research"


class BaseClaimSchema(BaseModel):
    """Base schema for all claim types"""
    id: str = Field(..., description="Unique claim identifier in format 'c<number>'")
    content: str = Field(..., min_length=5, max_length=2000, description="Claim content")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    type: str = Field(..., description="Claim type")
    tags: List[str] = Field(default_factory=list, description="Additional tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator('id')
    @classmethod
    def validate_claim_id(cls, v):
        if not v.startswith('c') or not v[1:].isdigit():
            raise ValueError(f"Claim ID must be in format 'c<number>', got: {v}")
        return v

    @field_validator('type')
    @classmethod
    def validate_claim_type(cls, v):
        valid_types = {
            'fact', 'concept', 'example', 'goal', 'reference', 
            'assertion', 'thesis', 'hypothesis', 'question', 'task'
        }
        if v.lower() not in valid_types:
            raise ValueError(f"Invalid claim type: {v}. Must be one of: {valid_types}")
        return v.lower()


class ClaimsResponseSchema(BaseModel):
    """Schema for claims generation responses"""
    type: str = Field(default="claims", description="Response type identifier")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence in claims")
    claims: List[BaseClaimSchema] = Field(..., min_items=1, description="List of generated claims")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    version: str = Field(default="1.0", description="Schema version")
    timestamp: str = Field(..., description="ISO timestamp")

    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        if v != "claims":
            raise ValueError(f"Invalid response type for claims schema: {v}")
        return v


class AnalysisResponseSchema(BaseModel):
    """Schema for analysis responses"""
    type: str = Field(default="analysis", description="Response type identifier")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence in analysis")
    analysis: Dict[str, Any] = Field(..., description="Analysis results")
    claims: List[BaseClaimSchema] = Field(default_factory=list, description="Claims derived from analysis")
    insights: List[str] = Field(default_factory=list, description="Key insights from analysis")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations based on analysis")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    version: str = Field(default="1.0", description="Schema version")
    timestamp: str = Field(..., description="ISO timestamp")

    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        if v != "analysis":
            raise ValueError(f"Invalid response type for analysis schema: {v}")
        return v


class ValidationResponseSchema(BaseModel):
    """Schema for claim validation responses"""
    type: str = Field(default="validation", description="Response type identifier")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in validation result")
    target_claim_id: str = Field(..., description="ID of claim being validated")
    validation_result: str = Field(..., description="Validation result: valid, invalid, needs_revision")
    revised_claim: BaseClaimSchema = Field(None, description="Revised claim if needed")
    validation_reasoning: str = Field(..., description="Reasoning for validation decision")
    confidence_adjustment: float = Field(None, ge=-1.0, le=1.0, description="Suggested confidence adjustment")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    version: str = Field(default="1.0", description="Schema version")
    timestamp: str = Field(..., description="ISO timestamp")

    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        if v != "validation":
            raise ValueError(f"Invalid response type for validation schema: {v}")
        return v

    @field_validator('validation_result')
    @classmethod
    def validate_result(cls, v):
        valid_results = {'valid', 'invalid', 'needs_revision'}
        if v.lower() not in valid_results:
            raise ValueError(f"Invalid validation result: {v}. Must be one of: {valid_results}")
        return v.lower()


class InstructionSupportSchema(BaseModel):
    """Schema for instruction support relationship responses"""
    type: str = Field(default="instruction_support", description="Response type identifier")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    instruction_claims: List[BaseClaimSchema] = Field(..., description="Identified instruction claims")
    relationships: List[Dict[str, Any]] = Field(..., description="Support relationships")
    analysis_summary: str = Field(..., description="Summary of instruction support analysis")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    version: str = Field(default="1.0", description="Schema version")
    timestamp: str = Field(..., description="ISO timestamp")

    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        if v != "instruction_support":
            raise ValueError(f"Invalid response type for instruction support schema: {v}")
        return v

    @field_validator('relationships')
    @classmethod
    def validate_relationships(cls, v):
        for rel in v:
            if 'instruction_claim_id' not in rel or 'target_claim_id' not in rel:
                raise ValueError("Each relationship must include 'instruction_claim_id' and 'target_claim_id'")
        return v


class RelationshipAnalysisSchema(BaseModel):
    """Schema for relationship analysis responses"""
    type: str = Field(default="relationship_analysis", description="Response type identifier")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    analyzed_relationships: List[Dict[str, Any]] = Field(..., description="Analyzed relationships")
    new_relationships: List[Dict[str, Any]] = Field(default_factory=list, description="Suggested new relationships")
    relationship_insights: List[str] = Field(default_factory=list, description="Insights about relationships")
    claims: List[BaseClaimSchema] = Field(default_factory=list, description="Claims related to relationships")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    version: str = Field(default="1.0", description="Schema version")
    timestamp: str = Field(..., description="ISO timestamp")

    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        if v != "relationship_analysis":
            raise ValueError(f"Invalid response type for relationship analysis schema: {v}")
        return v


class ExplorationResponseSchema(BaseModel):
    """Schema for exploration responses"""
    type: str = Field(default="exploration", description="Response type identifier")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence in exploration")
    exploration_claims: List[BaseClaimSchema] = Field(..., description="Claims from exploration")
    exploration_direction: str = Field(..., description="Direction of exploration")
    key_findings: List[str] = Field(default_factory=list, description="Key findings from exploration")
    next_steps: List[str] = Field(default_factory=list, description="Suggested next steps")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    version: str = Field(default="1.0", description="Schema version")
    timestamp: str = Field(..., description="ISO timestamp")

    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        if v != "exploration":
            raise ValueError(f"Invalid response type for exploration schema: {v}")
        return v


class ResearchResponseSchema(BaseModel):
    """Schema for research responses"""
    type: str = Field(default="research", description="Response type identifier")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence in research")
    research_claims: List[BaseClaimSchema] = Field(..., description="Claims from research")
    research_summary: str = Field(..., description="Summary of research findings")
    sources: List[Dict[str, str]] = Field(default_factory=list, description="Research sources")
    methodology: str = Field(..., description="Research methodology used")
    limitations: List[str] = Field(default_factory=list, description="Research limitations")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    version: str = Field(default="1.0", description="Schema version")
    timestamp: str = Field(..., description="ISO timestamp")

    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        if v != "research":
            raise ValueError(f"Invalid response type for research schema: {v}")
        return v


class ErrorResponseSchema(BaseModel):
    """Schema for error responses"""
    type: str = Field(default="error", description="Response type identifier")
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Error description")
    error_details: Dict[str, Any] = Field(default_factory=dict, description="Additional error details")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for resolution")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    version: str = Field(default="1.0", description="Schema version")
    timestamp: str = Field(..., description="ISO timestamp")

    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        if v != "error":
            raise ValueError(f"Invalid response type for error schema: {v}")
        return v


# Schema registry for easy access
SCHEMA_REGISTRY = {
    ResponseSchemaType.CLAIMS: ClaimsResponseSchema,
    ResponseSchemaType.ANALYSIS: AnalysisResponseSchema,
    ResponseSchemaType.VALIDATION: ValidationResponseSchema,
    ResponseSchemaType.INSTRUCTION_SUPPORT: InstructionSupportSchema,
    ResponseSchemaType.RELATIONSHIP_ANALYSIS: RelationshipAnalysisSchema,
    ResponseSchemaType.EXPLORATION: ExplorationResponseSchema,
    ResponseSchemaType.RESEARCH: ResearchResponseSchema,
    ResponseSchemaType.ERROR: ErrorResponseSchema,
}


def get_schema_examples() -> Dict[str, Dict[str, Any]]:
    """Get example data for each schema type"""
    from datetime import datetime
    
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    return {
        "claims": {
            "type": "claims",
            "confidence": 0.95,
            "claims": [
                {
                    "id": "c1",
                    "content": "The doctor lives in house 3 based on clue 1",
                    "confidence": 0.95,
                    "type": "fact",
                    "tags": ["deduction", "logic_puzzle"],
                    "metadata": {"source": "clue_1", "reasoning": "direct_statement"}
                },
                {
                    "id": "c2",
                    "content": "The engineer is not in house 2",
                    "confidence": 0.80,
                    "type": "inference",
                    "tags": ["elimination", "process_of_elimination"],
                    "metadata": {"reasoning": "elimination_based_on_other_clues"}
                }
            ],
            "metadata": {"puzzle_type": "logic_puzzle", "difficulty": "medium"},
            "version": "1.0",
            "timestamp": timestamp
        },
        
        "analysis": {
            "type": "analysis",
            "confidence": 0.90,
            "analysis": {
                "summary": "Analysis reveals strong evidence for the main claim",
                "key_factors": ["evidence_quality", "logical_consistency", "source_reliability"],
                "confidence_factors": {
                    "supporting_evidence": 0.95,
                    "logical_coherence": 0.85,
                    "source_credibility": 0.90
                }
            },
            "claims": [
                {
                    "id": "c1",
                    "content": "Main claim is well-supported by available evidence",
                    "confidence": 0.90,
                    "type": "conclusion"
                }
            ],
            "insights": [
                "Strong correlation between evidence quality and confidence",
                "Logical consistency is high across all supporting claims"
            ],
            "recommendations": [
                "Seek additional primary sources to strengthen confidence",
                "Consider alternative explanations for completeness"
            ],
            "version": "1.0",
            "timestamp": timestamp
        },
        
        "validation": {
            "type": "validation",
            "confidence": 0.95,
            "target_claim_id": "c123",
            "validation_result": "valid",
            "validation_reasoning": "Claim is well-supported by existing evidence and maintains logical consistency",
            "confidence_adjustment": 0.05,
            "version": "1.0",
            "timestamp": timestamp
        },
        
        "instruction_support": {
            "type": "instruction_support",
            "confidence": 0.85,
            "instruction_claims": [
                {
                    "id": "c1",
                    "content": "Follow the step-by-step process to implement the feature",
                    "confidence": 0.90,
                    "type": "instruction"
                }
            ],
            "relationships": [
                {
                    "instruction_claim_id": "c1",
                    "target_claim_id": "c456",
                    "relationship_type": "provides_guidance_for",
                    "confidence": 0.85
                }
            ],
            "analysis_summary": "Identified clear instructional content that provides guidance for implementation tasks",
            "version": "1.0",
            "timestamp": timestamp
        },
        
        "error": {
            "type": "error",
            "error_code": "INSUFFICIENT_CONTEXT",
            "error_message": "Not enough context provided to generate accurate claims",
            "error_details": {
                "missing_information": ["background_context", "specific_domain"],
                "provided_context_length": 50
            },
            "suggestions": [
                "Provide more background information about the topic",
                "Include specific domain or context details"
            ],
            "version": "1.0",
            "timestamp": timestamp
        }
    }


def validate_json_response(response_data: Dict[str, Any], response_type: ResponseSchemaType) -> tuple[bool, List[str]]:
    """
    Validate JSON response data against appropriate schema.
    
    Args:
        response_data: JSON data to validate
        response_type: Type of response to validate against
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    try:
        schema_class = SCHEMA_REGISTRY[response_type]
        schema_class(**response_data)
        return True, []
    except KeyError:
        return False, [f"Unknown response type: {response_type}"]
    except Exception as e:
        return False, [f"Validation error: {str(e)}"]


def get_schema_json(response_type: ResponseSchemaType) -> str:
    """Get JSON schema definition for response type"""
    schema_class = SCHEMA_REGISTRY[response_type]
    return json.dumps(schema_class.schema(), indent=2)


def create_prompt_template_for_type(response_type: ResponseSchemaType) -> str:
    """Create a prompt template that explains the required JSON format for a specific type"""
    examples = get_schema_examples()
    example = examples.get(response_type.value, examples["claims"])
    
    template = f"""Please format your response as JSON frontmatter for reliable parsing.

## REQUIRED FORMAT:
```json
---
{json.dumps(example, indent=2)}
---
```

## FORMAT REQUIREMENTS:
- JSON frontmatter must be at the very beginning
- Use valid JSON syntax
- All required fields must be included
- Confidence scores must be between 0.0 and 1.0
- Claim IDs must be in format 'c1', 'c2', etc.
- Timestamp must be ISO format

## RESPONSE TYPE: {response_type.value}
"""
    
    return template