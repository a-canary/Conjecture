"""
Tests for JSON Schemas

Tests the validation and functionality of JSON schemas for different response types.
"""

import pytest
from datetime import datetime
import json

from src.processing.json_schemas import (
    ResponseSchemaType,
    ClaimsResponseSchema,
    AnalysisResponseSchema,
    ValidationResponseSchema,
    InstructionSupportSchema,
    RelationshipAnalysisSchema,
    ExplorationResponseSchema,
    ResearchResponseSchema,
    ErrorResponseSchema,
    validate_json_response,
    get_schema_json,
    create_prompt_template_for_type,
    get_schema_examples
)


class TestClaimsResponseSchema:
    """Test claims response schema"""

    def test_valid_claims_schema(self):
        """Test valid claims response schema"""
        schema_data = {
            "type": "claims",
            "confidence": 0.95,
            "claims": [
                {
                    "id": "c1",
                    "content": "Test claim content",
                    "confidence": 0.95,
                    "type": "fact",
                    "tags": ["test"],
                    "metadata": {"source": "test"}
                }
            ],
            "metadata": {"test": True},
            "version": "1.0",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        schema = ClaimsResponseSchema(**schema_data)
        
        assert schema.type == "claims"
        assert schema.confidence == 0.95
        assert len(schema.claims) == 1
        assert schema.claims[0].id == "c1"
        assert schema.claims[0].type == "fact"

    def test_invalid_type_in_claims_schema(self):
        """Test invalid type raises validation error"""
        schema_data = {
            "type": "invalid_type",
            "confidence": 0.95,
            "claims": [],
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        with pytest.raises(ValueError, match="Invalid response type for claims schema"):
            ClaimsResponseSchema(**schema_data)

    def test_missing_required_fields(self):
        """Test missing required fields raises validation error"""
        schema_data = {
            "type": "claims",
            "timestamp": datetime.utcnow().isoformat() + "Z"
            # Missing confidence and claims
        }
        
        with pytest.raises(ValueError):
            ClaimsResponseSchema(**schema_data)


class TestAnalysisResponseSchema:
    """Test analysis response schema"""

    def test_valid_analysis_schema(self):
        """Test valid analysis response schema"""
        schema_data = {
            "type": "analysis",
            "confidence": 0.90,
            "analysis": {
                "summary": "Analysis summary",
                "key_factors": ["evidence", "logic"]
            },
            "claims": [
                {
                    "id": "c1",
                    "content": "Analysis claim",
                    "confidence": 0.90,
                    "type": "conclusion"
                }
            ],
            "insights": ["Key insight"],
            "recommendations": ["Recommendation"],
            "version": "1.0",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        schema = AnalysisResponseSchema(**schema_data)
        
        assert schema.type == "analysis"
        assert schema.confidence == 0.90
        assert schema.analysis["summary"] == "Analysis summary"
        assert len(schema.insights) == 1
        assert len(schema.recommendations) == 1

    def test_invalid_analysis_type(self):
        """Test invalid type raises validation error"""
        schema_data = {
            "type": "claims",  # Wrong type
            "confidence": 0.90,
            "analysis": {},
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        with pytest.raises(ValueError, match="Invalid response type for analysis schema"):
            AnalysisResponseSchema(**schema_data)


class TestValidationResponseSchema:
    """Test validation response schema"""

    def test_valid_validation_schema(self):
        """Test valid validation response schema"""
        schema_data = {
            "type": "validation",
            "confidence": 0.95,
            "target_claim_id": "c123",
            "validation_result": "valid",
            "validation_reasoning": "Well-supported by evidence",
            "confidence_adjustment": 0.05,
            "version": "1.0",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        schema = ValidationResponseSchema(**schema_data)
        
        assert schema.type == "validation"
        assert schema.target_claim_id == "c123"
        assert schema.validation_result == "valid"
        assert schema.confidence_adjustment == 0.05

    def test_invalid_validation_result(self):
        """Test invalid validation result raises validation error"""
        schema_data = {
            "type": "validation",
            "confidence": 0.95,
            "target_claim_id": "c123",
            "validation_result": "invalid_result",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        with pytest.raises(ValueError, match="Invalid validation result"):
            ValidationResponseSchema(**schema_data)


class TestInstructionSupportSchema:
    """Test instruction support schema"""

    def test_valid_instruction_support_schema(self):
        """Test valid instruction support schema"""
        schema_data = {
            "type": "instruction_support",
            "confidence": 0.85,
            "instruction_claims": [
                {
                    "id": "c1",
                    "content": "Step-by-step instruction",
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
            "analysis_summary": "Identified clear instructional content",
            "version": "1.0",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        schema = InstructionSupportSchema(**schema_data)
        
        assert schema.type == "instruction_support"
        assert len(schema.instruction_claims) == 1
        assert len(schema.relationships) == 1
        assert schema.relationships[0]["instruction_claim_id"] == "c1"

    def test_invalid_relationship_structure(self):
        """Test invalid relationship structure raises validation error"""
        schema_data = {
            "type": "instruction_support",
            "confidence": 0.85,
            "instruction_claims": [],
            "relationships": [
                {
                    "instruction_claim_id": "c1"
                    # Missing target_claim_id
                }
            ],
            "analysis_summary": "Test",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        with pytest.raises(ValueError, match="Each relationship must include"):
            InstructionSupportSchema(**schema_data)


class TestSchemaRegistry:
    """Test schema registry functionality"""

    def test_get_schema_examples(self):
        """Test getting schema examples"""
        examples = get_schema_examples()
        
        assert "claims" in examples
        assert "analysis" in examples
        assert "validation" in examples
        assert "error" in examples
        
        # Check structure of claims example
        claims_example = examples["claims"]
        assert claims_example["type"] == "claims"
        assert "claims" in claims_example
        assert len(claims_example["claims"]) >= 1

    def test_validate_json_response_valid(self):
        """Test validation of valid JSON response"""
        valid_response = {
            "type": "claims",
            "confidence": 0.95,
            "claims": [
                {
                    "id": "c1",
                    "content": "Test claim",
                    "confidence": 0.95,
                    "type": "fact"
                }
            ],
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        is_valid, errors = validate_json_response(valid_response, ResponseSchemaType.CLAIMS)
        
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_json_response_invalid(self):
        """Test validation of invalid JSON response"""
        invalid_response = {
            "type": "invalid_type",
            "confidence": 0.95,
            "claims": []
        }
        
        is_valid, errors = validate_json_response(invalid_response, ResponseSchemaType.CLAIMS)
        
        assert is_valid is False
        assert len(errors) > 0

    def test_validate_json_response_unknown_type(self):
        """Test validation with unknown response type"""
        response_data = {"type": "claims", "confidence": 0.95, "claims": []}
        
        is_valid, errors = validate_json_response(response_data, "unknown_type")
        
        assert is_valid is False
        assert "Unknown response type" in errors[0]

    def test_get_schema_json(self):
        """Test getting schema JSON"""
        schema_json = get_schema_json(ResponseSchemaType.CLAIMS)
        
        assert isinstance(schema_json, str)
        schema_dict = json.loads(schema_json)
        assert "properties" in schema_dict
        assert "type" in schema_dict["properties"]

    def test_create_prompt_template_for_type(self):
        """Test creating prompt template for type"""
        template = create_prompt_template_for_type(ResponseSchemaType.CLAIMS)
        
        assert "JSON frontmatter" in template
        assert "claims" in template
        assert "REQUIRED FORMAT" in template
        assert "FORMAT REQUIREMENTS" in template


class TestAllSchemaTypes:
    """Test all schema types for basic validation"""

    def test_exploration_schema(self):
        """Test exploration response schema"""
        schema_data = {
            "type": "exploration",
            "confidence": 0.90,
            "exploration_claims": [
                {
                    "id": "c1",
                    "content": "Exploration finding",
                    "confidence": 0.90,
                    "type": "hypothesis"
                }
            ],
            "exploration_direction": "Testing hypothesis validation",
            "key_findings": ["Finding 1", "Finding 2"],
            "next_steps": ["Next step 1"],
            "version": "1.0",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        schema = ExplorationResponseSchema(**schema_data)
        assert schema.type == "exploration"
        assert len(schema.exploration_claims) == 1

    def test_research_schema(self):
        """Test research response schema"""
        schema_data = {
            "type": "research",
            "confidence": 0.85,
            "research_claims": [
                {
                    "id": "c1",
                    "content": "Research finding",
                    "confidence": 0.85,
                    "type": "fact"
                }
            ],
            "research_summary": "Summary of research",
            "sources": [
                {"title": "Source 1", "url": "https://example.com"},
                {"title": "Source 2", "url": "https://example2.com"}
            ],
            "methodology": "Systematic literature review",
            "limitations": ["Limited sample size"],
            "version": "1.0",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        schema = ResearchResponseSchema(**schema_data)
        assert schema.type == "research"
        assert len(schema.research_claims) == 1
        assert len(schema.sources) == 2

    def test_error_schema(self):
        """Test error response schema"""
        schema_data = {
            "type": "error",
            "error_code": "INSUFFICIENT_CONTEXT",
            "error_message": "Not enough context provided",
            "error_details": {
                "missing_information": ["background"],
                "provided_context_length": 50
            },
            "suggestions": [
                "Provide more background information"
            ],
            "version": "1.0",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        schema = ErrorResponseSchema(**schema_data)
        assert schema.type == "error"
        assert schema.error_code == "INSUFFICIENT_CONTEXT"
        assert len(schema.suggestions) == 1


class TestSchemaEdgeCases:
    """Test edge cases in schema validation"""

    def test_empty_claims_array(self):
        """Test empty claims array is valid"""
        schema_data = {
            "type": "claims",
            "confidence": 0.95,
            "claims": [],  # Empty array should be valid
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        schema = ClaimsResponseSchema(**schema_data)
        assert len(schema.claims) == 0

    def test_minimal_valid_claim(self):
        """Test minimal valid claim data"""
        claim_data = {
            "id": "c1",
            "content": "Valid claim content",
            "confidence": 0.5,
            "type": "concept"
        }
        
        claim = JSONClaimData(**claim_data)
        assert claim.id == "c1"
        assert claim.confidence == 0.5

    def test_maximum_confidence_values(self):
        """Test boundary confidence values"""
        claim_data = {
            "id": "c1",
            "content": "Test claim",
            "confidence": 1.0,  # Maximum valid value
            "type": "fact"
        }
        
        claim = JSONClaimData(**claim_data)
        assert claim.confidence == 1.0
        
        # Test minimum
        claim_data["confidence"] = 0.0  # Minimum valid value
        claim = JSONClaimData(**claim_data)
        assert claim.confidence == 0.0

    def test_invalid_claim_types(self):
        """Test invalid claim type validation"""
        claim_data = {
            "id": "c1",
            "content": "Test claim",
            "confidence": 0.95,
            "type": "invalid_type"
        }
        
        with pytest.raises(ValueError, match="Invalid claim type"):
            JSONClaimData(**claim_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])