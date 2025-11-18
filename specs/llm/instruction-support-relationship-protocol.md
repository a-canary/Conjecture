# LLM Integration Protocol for Instruction Support Relationships

**Last Updated:** November 11, 2025  
**Version:** 1.0  
**Author:** Design Documentation Writer

## Overview

The LLM Integration Protocol defines how Large Language Models interact with the Simple Universal Claim Architecture to identify instruction claims and create support relationships. This protocol establishes a clean separation between system responsibilities (data management, context building) and LLM responsibilities (semantic analysis, relationship intelligence).

The protocol follows a **request-response-apply** pattern where the system provides complete context, the LLM analyzes and suggests relationships, and the system validates and applies approved changes.

## Core Protocol Design

### Interaction Flow

```
System → LLM: Complete claim context + user request
LLM → System: Analysis results + suggested relationships
System → LLM: (Optional) Clarification requests or additional context
LLM → System: Refined analysis
System → Database: Validated relationship updates
```

### Protocol Responsibilities

| System Responsibility | LLM Responsibility |
|----------------------|--------------------|
| Build complete context with all relationships | Identify instruction claims from context |
| Validate data integrity | Create logical support relationships |
| Apply approved changes *transitionally | Ensure logical consistency |
| Handle error recovery | Provide confidence scores for suggestions |
| Maintain audit trail | Explain reasoning for relationships |

## LLM Prompt Templates

### 1. Primary Instruction Identification Prompt

```python
INSTRUCTION_IDENTIFICATION_TEMPLATE = """
You are analyzing a claim knowledge base to identify instruction claims and establish support relationships.

## CONTEXT
{context_with_relationships}

## USER REQUEST
{user_request}

## YOUR TASK
1. **Instruction Identification**: Identify which claims represent instructions, guidance, or actionable advice
2. **Support Relationship Analysis**: Determine which claims provide evidence/support for instructions
3. **Relationship Creation**: Suggest new support relationships if needed
4. **Quality Assessment**: Evaluate the strength and validity of relationships

## INSTRUCTION CRITERIA
A claim is an INSTRUCTION if it:
- Contains imperative verbs (e.g., "do", "use", "implement", "avoid")
- Provides specific guidance or methodology
- Defines steps or processes to follow
- Sets constraints or requirements for actions
- Offers recommendations or best practices

## ANALYSIS FORMAT
Return a JSON response with the following structure:

```json
{
  "analysis_summary": {
    "total_claims_reviewed": number,
    "instructions_identified": number,
    "relationships_suggested": number,
    "confidence_score": number
  },
  "instructions": [
    {
      "claim_id": "CLAIM_ID",
      "instruction_type": "guidance|action|constraint|process|recommendation",
      "description": "Brief explanation of what instruction represents",
      "confidence": 0.0-1.0,
      "supported_by": ["EVIDENCE_CLAIM_ID_1", "EVIDENCE_CLAIM_ID_2"],
      "missing_evidence": ["What additional evidence would strengthen this instruction"]
    }
  ],
  "new_relationships": [
    {
      "from_claim": "EVIDENCE_CLAIM_ID",
      "to_claim": "INSTRUCTION_CLAIM_ID", 
      "relationship_type": "supports",
      "confidence": 0.0-1.0,
      "reasoning": "Why this claim supports the instruction"
    }
  ],
  "quality_issues": [
    {
      "type": "contradiction|missing_evidence|weak_support|circular_reference",
      "description": "Description of the issue",
      "affected_claims": ["CLAIM_ID_1", "CLAIM_ID_2"],
      "suggested_resolution": "How to fix the issue"
    }
  ],
  "suggested_improvements": [
    {
      "claim_content": "Suggested new claim content",
      "claim_type": "concept|reference|example",
      "supports_instruction": "INSTRUCTION_CLAIM_ID",
      "reasoning": "Why this claim would improve the knowledge base"
    }
  ]
}
```

## EXAMPLES

### Example 1: Simple Guidance Instruction
**Context includes:**
- Claim A: "Always validate user input before processing"
- Claim B: "Input validation prevents security vulnerabilities" 
- Claim C: "Security vulnerabilities lead to data breaches"

**Expected Response:**
{
  "instructions": [
    {
      "claim_id": "A",
      "instruction_type": "guidance",
      "description": "Security guidance for input validation",
      "confidence": 0.9,
      "supported_by": ["B", "C"],
      "missing_evidence": []
    }
  ],
  "new_relationships": [
    {
      "from_claim": "B",
      "to_claim": "A",
      "relationship_type": "supports", 
      "confidence": 0.95,
      "reasoning": "Validation prevents vulnerabilities, directly supporting the instruction to validate"
    }
  ]
}

### Example 2: Process Instruction
**Context includes:**
- Claim X: "First analyze the requirements, then design the solution"
- Claim Y: "Requirements analysis identifies user needs"
- Claim Z: "Design phase translates needs to technical specifications"

**Expected Response:**
{
  "instructions": [
    {
      "claim_id": "X", 
      "instruction_type": "process",
      "description": "Sequential development process instruction",
      "confidence": 0.85,
      "supported_by": ["Y", "Z"],
      "missing_evidence": ["Evidence about why this order matters"]
    }
  ]
}

## IMPORTANT GUIDELINES
1. Be conservative in identifying instructions - only clear, actionable guidance
2. Require strong evidence for support relationships
3. Flag potential contradictions or logical inconsistencies
4. Provide clear reasoning for all suggestions
5. Consider the user's specific request when prioritizing analysis

Now analyze the provided context and return your analysis in the specified JSON format.
"""
```

### 2. Relationship Validation Prompt

```python
RELATIONSHIP_VALIDATION_TEMPLATE = """
You are validating suggested support relationships between claims.

## SUGGESTED RELATIONSHIPS
{suggested_relationships}

## RELEVANT CONTEXT CLAIMS
{relevant_claims_context}

## VALIDATION TASK
For each suggested relationship, evaluate:
1. **Logical Validity**: Does the evidence logically support the instruction?
2. **Strength**: How strong is the support relationship?
3. **Completeness**: Are there missing pieces in the support chain?
4. **Consistency**: Does this relationship conflict with others?

## VALIDATION RESPONSE FORMAT
```json
{
  "validated_relationships": [
    {
      "from_claim": "EVIDENCE_CLAIM_ID",
      "to_claim": "INSTRUCTION_CLAIM_ID",
      "validation_status": "approved|rejected|needs_revision",
      "updated_confidence": 0.0-1.0,
      "validation_reasoning": "Detailed explanation of validation decision",
      "concerns": ["Any concerns or issues identified"]
    }
  ],
  "rejected_relationships": [
    {
      "from_claim": "EVIDENCE_CLAIM_ID", 
      "to_claim": "INSTRUCTION_CLAIM_ID",
      "rejection_reason": "Why this relationship was rejected",
      "suggested_alternative": "Alternative approach if applicable"
    }
  ],
  "quality_adjustments": [
    {
      "claim_id": "CLAIM_ID",
      "adjustment_type": "confidence_adjustment|additional_evidence_needed",
      "old_value": number,
      "new_value": number,
      "reasoning": "Why adjustment is needed"
    }
  ]
}
```

## VALIDATION CRITERIA
- **Approve**: Strong logical connection, clear evidence, supports instruction
- **Reject**: No logical connection, contradicts evidence, creates circular dependency
- **Revise**: Partial connection, needs additional evidence, weak support

Now validate the suggested relationships and return your analysis.
"""
```

### 3. Quality Improvement Prompt

```python
QUALITY_IMPROVEMENT_TEMPLATE = """
You are improving a claim knowledge base by identifying quality issues and suggesting improvements.

## CURRENT STATE ANALYSIS
{current_state_summary}

## CONTEXT FOR IMPROVEMENT
{context_section}

## IMPROVEMENT TASK
1. **Identify Gaps**: What evidence or claims are missing?
2. **Strengthen Connections**: How can support relationships be improved?
3. **Resolve Inconsistencies**: Fix contradictions or conflicting claims
4. **Enhance Completeness**: Add missing but important claims

## IMPROVEMENT RESPONSE FORMAT
```json
{
  "improvement_summary": {
    "gaps_identified": number,
    "inconsistencies_found": number,
    "suggestions_made": number,
    "priority_areas": ["area1", "area2"]
  },
  "missing_claims": [
    {
      "suggested_content": "Claim statement content",
      "claim_type": "concept|reference|example|thesis|skill|goal",
      "would_support": ["INSTRUCTION_CLAIM_IDS"],
      "importance": "high|medium|low",
      "reasoning": "Why this claim is needed"
    }
  ],
  "relationship_improvements": [
    {
      "type": "add relationship|strengthen relationship|remove relationship",
      "from_claim": "CLAIM_ID",
      "to_claim": "CLAIM_ID",
      "confidence": 0.0-1.0,
      "improvement_reasoning": "Why this change improves the knowledge base"
    }
  ],
  "consistency_fixes": [
    {
      "conflict_type": "contradiction|duplicate_confidence|circular_reference",
      "conflicting_claims": ["CLAIM_ID_1", "CLAIM_ID_2"],
      "resolution_strategy": "merge|rephrase|adjust_confidence|remove_one",
      "resolution_details": "Specific steps to fix the conflict"
    }
  ]
}
```

Now analyze the current state and suggest improvements.
"""
```

## Processing Workflow Specifications

### 1. Complete Analysis Workflow

```python
async def process_with_instruction_support(
    context: str,
    user_request: str,
    validation_enabled: bool = True
) -> LLMResponse:
    """
    Complete workflow for instruction identification and relationship creation
    
    Args:
        context: Complete claim context with relationships
        user_request: Original user query/request
        validation_enabled: Whether to run validation step
        
    Returns:
        Structured LLM response with analysis and relationships
    """
    
    # Phase 1: Primary Instruction Identification
    primary_response = await call_llm_with_template(
        INSTRUCTION_IDENTIFICATION_TEMPLATE,
        context=context,
        user_request=user_request
    )
    
    # Phase 2: Relationship Validation (if enabled and relationships found)
    if (validation_enabled and 
        primary_response.new_relationships and
        len(primary_response.new_relationships) > 0):
        
        # Get context for relationship validation
        validation_context = build_validation_context(
            primary_response.new_relationships,
            context
        )
        
        validation_response = await call_llm_with_template(
            RELATIONSHIP_VALIDATION_TEMPLATE,
            suggested_relationships=primary_response.new_relationships,
            relevant_claims_context=validation_context
        )
        
        # Merge validation results
        return merge_analysis_results(primary_response, validation_response)
    
    return primary_response
```

### 2. Quality Improvement Workflow

```python
async def improve_context_quality(
    claim_network: Dict[str, Claim],
    focus_area: Optional[str] = None
) -> QualityImprovementResponse:
    """
    Analyze and suggest improvements for claim network quality
    
    Args:
        claim_network: Current claim network
        focus_area: Optional specific area to focus on
        
    Returns:
        Quality improvement suggestions
    """
    
    # Build current state summary
    current_state = analyze_network_quality(claim_network)
    
    # Build context for improvement analysis
    improvement_context = build_improvement_context(
        claim_network, 
        focus_area
    )
    
    # Get improvement suggestions
    improvement_response = await call_llm_with_template(
        QUALITY_IMPROVEMENT_TEMPLATE,
        current_state_summary=current_state,
        context_section=improvement_context
    )
    
    return improvement_response
```

## Data Structures

### LLM Response Models

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class InstructionType(str, Enum):
    GUIDANCE = "guidance"
    ACTION = "action" 
    CONSTRAINT = "constraint"
    PROCESS = "process"
    RECOMMENDATION = "recommendation"

class IdentifiedInstruction(BaseModel):
    claim_id: str = Field(..., description="ID of instruction claim")
    instruction_type: InstructionType = Field(..., description="Type of instruction")
    description: str = Field(..., description="What instruction represents")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in identification")
    supported_by: List[str] = Field(default_factory=list, description="Evidence claim IDs")
    missing_evidence: List[str] = Field(default_factory=list, description="Missing evidence")

class SuggestedRelationship(BaseModel):
    from_claim: str = Field(..., description="Evidence claim ID")
    to_claim: str = Field(..., description="Instruction claim ID")
    relationship_type: str = Field(default="supports", description="Relationship type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Relationship strength")
    reasoning: str = Field(..., description="Why this relationship exists")

class QualityIssue(BaseModel):
    type: str = Field(..., description="Type of quality issue")
    description: str = Field(..., description="Issue description")
    affected_claims: List[str] = Field(..., description="Claims affected by issue")
    suggested_resolution: str = Field(..., description="How to fix the issue")

class SuggestedImprovement(BaseModel):
    claim_content: str = Field(..., description="Suggested new claim content")
    claim_type: str = Field(..., description="Type of new claim")
    supports_instruction: str = Field(..., description="Instruction this would support")
    reasoning: str = Field(..., description="Why this improvement is needed")

class LLMResponse(BaseModel):
    analysis_summary: Dict[str, Any] = Field(..., description="Analysis overview")
    instructions: List[IdentifiedInstruction] = Field(default_factory=list)
    new_relationships: List[SuggestedRelationship] = Field(default_factory=list)  
    quality_issues: List[QualityIssue] = Field(default_factory=list)
    suggested_improvements: List[SuggestedImprovement] = Field(default_factory=list)
```

## Error Handling and Validation

### LLM Response Validation

```python
class LLMResponseValidator:
    """Validates LLM responses for correctness and completeness"""
    
    @staticmethod
    def validate_response_structure(response_data: Dict) -> List[str]:
        """Validate response has required structure"""
        errors = []
        
        required_fields = [
            "analysis_summary",
            "instructions", 
            "new_relationships",
            "quality_issues",
            "suggested_improvements"
        ]
        
        for field in required_fields:
            if field not in response_data:
                errors.append(f"Missing required field: {field}")
        
        # Validate analysis summary
        if "analysis_summary" in response_data:
            summary = response_data["analysis_summary"]
            summary_required = ["total_claims_reviewed", "confidence_score"]
            for field in summary_required:
                if field not in summary:
                    errors.append(f"Missing in analysis_summary: {field}")
        
        return errors
    
    @staticmethod
    def validate_relationship_consistency(
        relationships: List[SuggestedRelationship],
        available_claims: List[str]
    ) -> List[str]:
        """Validate relationships reference existing claims"""
        errors = []
        available_set = set(available_claims)
        
        for rel in relationships:
            if rel.from_claim not in available_set:
                errors.append(f"From claim {rel.from_claim} not found in context")
            if rel.to_claim not in available_set:
                errors.append(f"To claim {rel.to_claim} not found in context")
        
        return errors
    
    @staticmethod
    def validate_confidence_values(response: LLMResponse) -> List[str]:
        """Validate all confidence values are in valid range"""
        errors = []
        
        if response.analysis_summary.get("confidence_score", 0) not in [0, 1]:
            if not (0 <= response.analysis_summary["confidence_score"] <= 1):
                errors.append("Analysis summary confidence not in 0-1 range")
        
        for instruction in response.instructions:
            if not (0 <= instruction.confidence <= 1):
                errors.append(f"Instruction {instruction.claim_id} confidence invalid")
        
        for relationship in response.new_relationships:
            if not (0 <= relationship.confidence <= 1):
                errors.append(f"Relationship confidence invalid: {relationship}")
        
        return errors
```

### Error Recovery Strategies

```python
class LLMErrorHandler:
    """Handles errors in LLM interactions gracefully"""
    
    @staticmethod
    async def handle_json_parse_error(
        raw_response: str,
        context: str
    ) -> LLMResponse:
        """Handle when LLM response can't be parsed as JSON"""
        
        # Try to extract JSON with regex
        json_match = re.search(r'```json\\n(.*?)\\n```', raw_response, re.DOTALL)
        if json_match:
            try:
                return LLMResponse.parse_raw(json_match.group(1))
            except:
                pass
        
        # Fallback: Try to extract any valid JSON
        try:
            return LLMResponse.parse_raw(raw_response)
        except:
            # Last resort: Create minimal valid response
            return LLMResponse(
                analysis_summary={
                    "total_claims_reviewed": 0,
                    "instructions_identified": 0,
                    "relationships_suggested": 0,
                    "confidence_score": 0.0,
                    "parsing_error": "Failed to parse LLM response"
                }
            )
    
    @staticmethod
    async def handle_timeout_error(
        prompt: str,
        context: str,
        max_retries: int = 2
    ) -> LLMResponse:
        """Handle LLM timeout with shorter context retry"""
        
        for attempt in range(max_retries + 1):
            try:
                if attempt == 0:
                    # First attempt with full context
                    return await call_llm_with_timeout(prompt, context, timeout=30)
                else:
                    # Retry with reduced context
                    reduced_context = truncate_context(context, max_tokens=4000)
                    shortened_prompt = prompt.replace("{context}", "{reduced_context}")
                    return await call_llm_with_timeout(
                        shortened_prompt, 
                        reduced_context, 
                        timeout=20
                    )
            except TimeoutError:
                if attempt == max_retries:
                    # Return minimal response after all retries fail
                    return create_timeout_response()
                continue
```

## Performance Optimization

### Request Batching and Caching

```python
class LLMRequestOptimizer:
    """Optimizes LLM requests for better performance"""
    
    def __init__(self):
        self.response_cache = {}
        self.request_queue = asyncio.Queue()
        self.batch_processor_running = False
    
    async def batch_process_requests(self, requests: List[LLMRequest]) -> List[LLMResponse]:
        """Process multiple LLM requests in batch for efficiency"""
        
        # Group requests by similarity for potential batching
        similarity_groups = self.group_by_similarity(requests)
        
        # Process each group
        results = []
        for group in similarity_groups:
            if len(group) > 1 and self.can_batch_group(group):
                batch_result = await self.process_batch(group)
                results.extend(batch_result)
            else:
                # Process individually
                for request in group:
                    result = await self.process_single(request)
                    results.append(result)
        
        return results
    
    def group_by_similarity(self, requests: List[LLMRequest]) -> List[List[LLMRequest]]:
        """Group requests by prompt similarity for potential batching"""
        groups = []
        
        for request in requests:
            # Find similar group or create new one
            similar_group = None
            for group in groups:
                if self.requests_similar(request, group[0]):
                    similar_group = group
                    break
            
            if similar_group:
                similar_group.append(request)
            else:
                groups.append([request])
        
        return groups
    
    async def process_batch(self, requests: List[LLMRequest]) -> List[LLMResponse]:
        """Process a batch of similar requests efficiently"""
        
        # Combine contexts if possible
        combined_context = self.combine_contexts([r.context for r in requests])
        combined_prompt = self.create_batch_prompt(requests)
        
        # Single LLM call for the batch
        batch_response = await call_llm(combined_prompt, combined_context)
        
        # Split response back to individual requests
        return self.split_batch_response(batch_response, requests)
```

## Monitoring and Analytics

### LLM Performance Metrics

```python
class LLMMetrics:
    """Track LLM interaction performance and quality"""
    
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "average_tokens_used": 0,
            "identifications_per_request": 0.0,
            "relationships_per_request": 0.0,
            "confidence_distribution": {"high": 0, "medium": 0, "low": 0},
            "error_types": {}
        }
    
    def record_request(
        self,
        success: bool,
        response_time: float,
        tokens_used: int,
        response: Optional[LLMResponse] = None,
        error_type: Optional[str] = None
    ):
        """Record metrics for a single LLM request"""
        
        self.metrics["total_requests"] += 1
        
        if success:
            self.metrics["successful_requests"] += 1
            
            if response:
                # Update identification and relationship counts
                ident_count = len(response.instructions)
                rel_count = len(response.new_relationships)
                
                total = self.metrics["successful_requests"]
                self.metrics["identifications_per_request"] = (
                    (self.metrics["identifications_per_request"] * (total - 1) + ident_count) / total
                )
                self.metrics["relationships_per_request"] = (
                    (self.metrics["relationships_per_request"] * (total - 1) + rel_count) / total
                )
                
                # Update confidence distribution
                overall_conf = response.analysis_summary.get("confidence_score", 0)
                if overall_conf >= 0.8:
                    self.metrics["confidence_distribution"]["high"] += 1
                elif overall_conf >= 0.5:
                    self.metrics["confidence_distribution"]["medium"] += 1
                else:
                    self.metrics["confidence_distribution"]["low"] += 1
        else:
            self.metrics["failed_requests"] += 1
            if error_type:
                self.metrics["error_types"][error_type] = (
                    self.metrics["error_types"].get(error_type, 0) + 1
                )
        
        # Update response time and token averages
        total = self.metrics["total_requests"]
        self.metrics["average_response_time"] = (
            (self.metrics["average_response_time"] * (total - 1) + response_time) / total
        )
        self.metrics["average_tokens_used"] = (
            (self.metrics["average_tokens_used"] * (total - 1) + tokens_used) / total
        )
```

## Implementation Examples

### Example LLM Interaction

```python
# Example: Processing a user request about software development

user_request = "How should I structure code for a web application?"

# System builds complete context
context = build_complete_context("target_claim_123", max_tokens=8000)

# LLM processes the request
response = await process_with_instruction_support(
    context=context,
    user_request=user_request,
    validation_enabled=True
)

# Example response structure:
{
  "analysis_summary": {
    "total_claims_reviewed": 25,
    "instructions_identified": 3,
    "relationships_suggested": 7,
    "confidence_score": 0.87
  },
  "instructions": [
    {
      "claim_id": "claim_456",
      "instruction_type": "process",
      "description": "MVC architecture guidance for web applications",
      "confidence": 0.92,
      "supported_by": ["claim_789", "claim_101"],
      "missing_evidence": ["Specific framework examples"]
    }
  ],
  "new_relationships": [
    {
      "from_claim": "claim_789",
      "to_claim": "claim_456", 
      "relationship_type": "supports",
      "confidence": 0.89,
      "reasoning": "Separation of concerns principle supports MVC structure"
    }
  ],
  "quality_issues": [
    {
      "type": "missing_evidence",
      "description": "Instruction lacks specific framework examples",
      "affected_claims": ["claim_456"],
      "suggested_resolution": "Add framework-specific examples"
    }
  ]
}

# System applies validated relationships
validated_response = validate_and_apply_relationships(response)
```

The LLM Integration Protocol provides a robust, scalable approach to instruction identification and support relationship creation while maintaining system reliability and performance.