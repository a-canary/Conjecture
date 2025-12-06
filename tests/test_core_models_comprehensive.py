#!/usr/bin/env python3
"""
Comprehensive tests for core models to maximize coverage
Tests all model validation, enums, and edge cases
"""

import pytest
from datetime import datetime
from typing import List, Dict, Any
import json

from src.core.models import (
    Claim, ClaimState, ClaimType, ClaimScope,
    Relationship, DirtyReason,
    ClaimFilter, ProcessingResult, DataConfig,
    RelationshipError, ClaimNotFoundError, InvalidClaimError, DataLayerError,
    create_claim, generate_claim_id, validate_claim_id, validate_confidence,
    create_claim_index, get_orphaned_claims, get_root_claims, get_leaf_claims,
    filter_claims_by_tags, filter_claims_by_confidence
)


class TestClaimState:
    """Test ClaimState enum functionality"""
    
    def test_claim_state_values(self):
        """Test all claim state values are present"""
        assert ClaimState.EXPLORE == "Explore"
        assert ClaimState.VALIDATED == "Validated"
        assert ClaimState.ORPHANED == "Orphaned"
        assert ClaimState.QUEUED == "Queued"
    
    def test_claim_state_iteration(self):
        """Test claim state can be iterated"""
        states = list(ClaimState)
        assert len(states) == 4
        assert ClaimState.EXPLORE in states
        assert ClaimState.VALIDATED in states
        assert ClaimState.ORPHANED in states
        assert ClaimState.QUEUED in states


class TestClaimType:
    """Test ClaimType enum functionality"""
    
    def test_claim_type_values(self):
        """Test all claim type values are present"""
        expected_types = [
            "fact", "concept", "example", "goal", "reference",
            "assertion", "thesis", "hypothesis", "question", "task"
        ]
        
        for expected in expected_types:
            assert expected in [t.value for t in ClaimType]
    
    def test_claim_type_iteration(self):
        """Test claim type can be iterated"""
        types = list(ClaimType)
        assert len(types) == 10
        
        # Test specific important types
        assert ClaimType.FACT in types
        assert ClaimType.HYPOTHESIS in types
        assert ClaimType.TASK in types


class TestClaimScope:
    """Test ClaimScope enum functionality"""
    
    def test_claim_scope_values(self):
        """Test all claim scope values are present"""
        assert ClaimScope.USER_WORKSPACE == "user-{workspace}"
        assert ClaimScope.TEAM_WORKSPACE == "team-{workspace}"
        assert ClaimScope.TEAM_WIDE == "team-wide"
        assert ClaimScope.PUBLIC == "public"
    
    def test_get_default(self):
        """Test default scope getter"""
        assert ClaimScope.get_default() == ClaimScope.USER_WORKSPACE.value
    
    def test_get_hierarchy(self):
        """Test scope hierarchy"""
        hierarchy = ClaimScope.get_hierarchy()
        assert len(hierarchy) == 4
        assert hierarchy[0] == ClaimScope.USER_WORKSPACE.value
        assert hierarchy[-1] == ClaimScope.PUBLIC.value
    
    def test_normalize_scope(self):
        """Test scope normalization with workspace"""
        # Test with workspace
        normalized = ClaimScope.normalize_scope("user-{workspace}", "myproject")
        assert normalized == "user-myproject"
        
        # Test without workspace
        normalized = ClaimScope.normalize_scope("public", "myproject")
        assert normalized == "public"
        
        # Test team workspace
        normalized = ClaimScope.normalize_scope("team-{workspace}", "myproject")
        assert normalized == "team-myproject"
    
    def test_is_valid_scope(self):
        """Test scope validation"""
        assert ClaimScope.is_valid_scope("user-{workspace}") is True
        assert ClaimScope.is_valid_scope("team-{workspace}") is True
        assert ClaimScope.is_valid_scope("team-wide") is True
        assert ClaimScope.is_valid_scope("public") is True
        assert ClaimScope.is_valid_scope("invalid") is False
    
    def test_get_scope_level(self):
        """Test scope level calculation"""
        assert ClaimScope.get_scope_level("user-{workspace}") == 0
        assert ClaimScope.get_scope_level("team-{workspace}") == 1
        assert ClaimScope.get_scope_level("team-wide") == 2
        assert ClaimScope.get_scope_level("public") == 3
        assert ClaimScope.get_scope_level("invalid") == -1


class TestClaimModel:
    """Test Claim model validation and functionality"""
    
    def test_minimal_claim_creation(self):
        """Test creating claim with minimal required fields"""
        claim = Claim(
            id="test-123",
            content="Test claim",
            confidence=0.8
        )
        
        assert claim.content == "Test claim"
        assert claim.confidence == 0.8
        assert claim.state == ClaimState.EXPLORE  # Default value
        assert claim.scope == ClaimScope.USER_WORKSPACE  # Default value
        assert claim.created is not None  # Should be auto-generated
        assert claim.updated is not None  # Should be auto-generated
    
    def test_full_claim_creation(self):
        """Test creating claim with all fields"""
        now = datetime.now()
        claim = Claim(
            id="test-123",
            content="Comprehensive test claim",
            confidence=0.95,
            state=ClaimState.VALIDATED,
            scope=ClaimScope.PUBLIC,
            supported_by=["support-1"],
            supports=["supported-1"],
            tags=["test", "comprehensive"],
            created=now,
            updated=now,
            is_dirty=False,
            dirty_reason=DirtyReason.MANUAL_MARK,
            dirty_priority=1
        )
        
        assert claim.content == "Comprehensive test claim"
        assert claim.confidence == 0.95
        assert claim.state == ClaimState.VALIDATED
        assert claim.scope == ClaimScope.PUBLIC
        assert claim.supported_by == ["support-1"]
        assert claim.supports == ["supported-1"]
        assert len(claim.tags) == 2
        assert claim.created == now
        assert claim.updated == now
        assert claim.is_dirty is False
        assert claim.dirty_reason == DirtyReason.MANUAL_MARK
        assert claim.dirty_priority == 1
    
    def test_claim_validation(self):
        """Test claim field validation"""
        # Test invalid confidence (too low)
        with pytest.raises(ValueError):
            Claim(id="test", content="Test claim content", confidence=-0.1)
        
        # Test invalid confidence (too high)
        with pytest.raises(ValueError):
            Claim(id="test", content="Test claim content", confidence=1.1)
        
        # Test empty claim text
        with pytest.raises(ValueError):
            Claim(id="test", content="", confidence=0.5)
        
        # Test valid boundary values
        claim_min = Claim(id="test", content="Test claim content", confidence=0.0)
        assert claim_min.confidence == 0.0
        
        claim_max = Claim(id="test", content="Test claim content", confidence=1.0)
        assert claim_max.confidence == 1.0
    
    def test_claim_tag_validation(self):
        """Test claim tag validation"""
        # Test valid tags
        claim = Claim(
            id="test",
            content="Test claim",
            confidence=0.8,
            tags=["valid", "tags"]
        )
        assert len(claim.tags) == 2
        
        # Test duplicate tag removal
        claim_dup = Claim(
            id="test2",
            content="Test claim with duplicates",
            confidence=0.8,
            tags=["tag1", "tag2", "tag1", "tag3"]
        )
        assert len(claim_dup.tags) == 3  # Duplicates removed
        assert claim_dup.tags == ["tag1", "tag2", "tag3"]
        
        # Test invalid tags (empty strings)
        with pytest.raises(ValueError):
            Claim(
                id="test3",
                content="Test claim with invalid tags",
                confidence=0.8,
                tags=["valid", "", "also_valid"]
            )
    
    def test_claim_serialization(self):
        """Test claim JSON serialization"""
        claim = Claim(
            id="test-123",
            content="Serialization test",
            confidence=0.75,
            tags=["serialization"]
        )
        
        # Test to_dict
        claim_dict = claim.model_dump()
        assert claim_dict["content"] == "Serialization test"
        assert claim_dict["confidence"] == 0.75
        
        # Test JSON serialization
        claim_json = claim.model_dump_json()
        assert "Serialization test" in claim_json
        
        # Test deserialization
        parsed_claim = Claim.model_validate_json(claim_json)
        assert parsed_claim.content == claim.content
        assert parsed_claim.confidence == claim.confidence
    
    def test_claim_update_methods(self):
        """Test claim update functionality"""
        claim = Claim(
            id="test-123",
            content="Original claim",
            confidence=0.5
        )
        original_updated = claim.updated
        
        # Update confidence
        claim.confidence = 0.8
        assert claim.confidence == 0.8
        # Note: updated field doesn't auto-update in this model
        
        # Update state
        claim.state = ClaimState.VALIDATED
        assert claim.state == ClaimState.VALIDATED
        
        # Add to supports
        claim.supports.append("supported-claim")
        assert len(claim.supports) == 1
        assert "supported-claim" in claim.supports
    
    def test_claim_formatting_methods(self):
        """Test claim formatting methods"""
        claim = Claim(
            id="test-123",
            content="Test formatting",
            confidence=0.85,
            tags=["format", "test"]
        )
        
        # Test format_for_context
        context_format = claim.format_for_context()
        expected = "[ctest-123 | Test formatting | / 0.85]"
        assert context_format == expected
        
        # Test format_for_output
        output_format = claim.format_for_output()
        assert output_format == expected
        
        # Test format_for_llm_analysis
        analysis_format = claim.format_for_llm_analysis()
        assert "Claim ID: test-123" in analysis_format
        assert "Content: Test formatting" in analysis_format
        assert "Confidence: 0.85" in analysis_format
        assert "Tags: format,test" in analysis_format
    
    def test_claim_chroma_integration(self):
        """Test claim ChromaDB integration methods"""
        claim = Claim(
            id="test-123",
            content="Chroma test claim",
            confidence=0.9,
            tags=["chroma", "test"]
        )
        
        # Test to_chroma_metadata
        metadata = claim.to_chroma_metadata()
        assert metadata["confidence"] == 0.9
        assert metadata["state"] == "Explore"
        assert metadata["tags"] == ["chroma", "test"]
        assert "created" in metadata
        assert "updated" in metadata
        
        # Test from_chroma_result
        recreated_claim = Claim.from_chroma_result(
            id="test-123",
            content="Chroma test claim",
            metadata=metadata
        )
        assert recreated_claim.id == "test-123"
        assert recreated_claim.content == "Chroma test claim"
        assert recreated_claim.confidence == 0.9
    
    def test_claim_properties(self):
        """Test claim properties"""
        claim = Claim(
            id="test-123",
            content="Property test",
            confidence=0.9
        )
        
        # Test backward compatibility properties
        assert claim.dirty == claim.is_dirty
        assert claim.created_at == claim.created
        assert claim.updated_at == claim.updated
        
        # Test confidence properties - these are properties, not methods
        effective_threshold = claim._get_default_threshold()
        assert claim.confidence >= effective_threshold  # Should be confident
        
        low_confidence_claim = Claim(
            id="test-456",
            content="Low confidence claim",
            confidence=0.3
        )
        assert low_confidence_claim.confidence < effective_threshold  # Should need evaluation
    
    def test_claim_hash_and_equality(self):
        """Test claim hash functionality"""
        claim1 = Claim(
            id="test-123",
            content="Hash test",
            confidence=0.8
        )
        claim2 = Claim(
            id="test-456",  # Different ID
            content="Hash test",
            confidence=0.8
        )
    
        # Test hash
        claim_set = {claim1, claim2}
        assert len(claim_set) == 2  # Different objects with different IDs
    
        # Test equality based on ID
        claim3 = Claim(
            id="test-123",
            content="Hash test",
            confidence=0.8
        )
    
        # Test equality (claims are not equal by default, only hashable)
        # The __eq__ method is not defined, so they are different objects
        # Test that they can both exist in a set due to different hash implementation
        claim_set = {claim1, claim3}
        assert len(claim_set) == 2  # Different objects even with same ID
        assert claim1 != claim2  # Different ID means not equal
        
        # Test hash consistency with equality
        claim_set_same = {claim1, claim3}
        assert len(claim_set_same) == 1  # Equal objects have same hash
        
        # Test that hash is based on ID
        assert hash(claim1) == hash(claim3)
        assert hash(claim1) != hash(claim2)
        
        # Test repr
        repr_str = repr(claim1)
        assert "Claim(id=test-123" in repr_str
        assert "confidence=0.8" in repr_str


class TestRelationshipModel:
    """Test Relationship model validation and functionality"""
    
    def test_minimal_relationship_creation(self):
        """Test creating relationship with minimal fields"""
        rel = Relationship(
            supporter_id="claim-1",
            supported_id="claim-2",
            relationship_type="supports"
        )
        
        assert rel.supporter_id == "claim-1"
        assert rel.supported_id == "claim-2"
        assert rel.relationship_type == "supports"
        assert rel.confidence == 1.0  # Default value
        assert rel.created is not None  # Should be auto-generated
    
    def test_full_relationship_creation(self):
        """Test creating relationship with all fields"""
        now = datetime.now()
        rel = Relationship(
            supporter_id="claim-1",
            supported_id="claim-2",
            relationship_type="contradicts",
            confidence=0.9,
            created_by="test-user",
            created=now
        )
        
        assert rel.supporter_id == "claim-1"
        assert rel.supported_id == "claim-2"
        assert rel.relationship_type == "contradicts"
        assert rel.confidence == 0.9
        assert rel.created_by == "test-user"
        assert rel.created == now
    
    def test_relationship_validation(self):
        """Test relationship field validation"""
        # Test invalid confidence
        with pytest.raises(ValueError):
            Relationship(
                supporter_id="claim-1",
                supported_id="claim-2",
                relationship_type="supports",
                confidence=1.5
            )
        
        # Test invalid relationship type
        with pytest.raises(ValueError):
            Relationship(
                supporter_id="claim-1",
                supported_id="claim-2",
                relationship_type="invalid_type"
            )
    
    def test_relationship_properties(self):
        """Test relationship backward compatibility properties"""
        rel = Relationship(
            supporter_id="claim-1",
            supported_id="claim-2",
            relationship_type="supports"
        )
        
        assert rel.supporter == "claim-1"  # Backward compatibility
        assert rel.supported == "claim-2"   # Backward compatibility
        assert rel.created_at == rel.created  # Backward compatibility


class TestClaimFilter:
    """Test ClaimFilter functionality"""
    
    def test_empty_filter(self):
        """Test empty filter creation"""
        filter_obj = ClaimFilter()
        assert filter_obj.tags is None
        assert filter_obj.states is None
        assert filter_obj.confidence_min is None
        assert filter_obj.confidence_max is None
        assert filter_obj.dirty_only is None
        assert filter_obj.content_contains is None
        assert filter_obj.limit == 100  # Default value
        assert filter_obj.offset == 0     # Default value
    
    def test_filter_with_criteria(self):
        """Test filter with various criteria"""
        filter_obj = ClaimFilter(
            tags=["important", "critical"],
            states=[ClaimState.VALIDATED],
            confidence_min=0.7,
            confidence_max=0.95,
            dirty_only=True,
            content_contains="test query",
            limit=10,
            offset=5
        )
        
        assert len(filter_obj.tags) == 2
        assert "important" in filter_obj.tags
        assert filter_obj.states == [ClaimState.VALIDATED]
        assert filter_obj.confidence_min == 0.7
        assert filter_obj.confidence_max == 0.95
        assert filter_obj.dirty_only is True
        assert filter_obj.content_contains == "test query"
        assert filter_obj.limit == 10
        assert filter_obj.offset == 5
    
    def test_filter_validation(self):
        """Test filter validation"""
        # Test invalid confidence range
        with pytest.raises(ValueError):
            ClaimFilter(
                confidence_min=0.8,
                confidence_max=0.6  # Less than min
            )
        
        # Test invalid limit
        with pytest.raises(ValueError):
            ClaimFilter(limit=0)
        
        # Test invalid offset
        with pytest.raises(ValueError):
            ClaimFilter(offset=-1)


class TestProcessingResult:
    """Test ProcessingResult functionality"""
    
    def test_successful_processing_result(self):
        """Test successful processing result"""
        result = ProcessingResult(
            success=True,
            operation_type="test_operation",
            processed_items=5,
            updated_items=3,
            message="Processing completed successfully",
            execution_time=1.5
        )
        
        assert result.success is True
        assert result.operation_type == "test_operation"
        assert result.processed_items == 5
        assert result.updated_items == 3
        assert result.message == "Processing completed successfully"
        assert result.execution_time == 1.5
        assert len(result.errors) == 0
        assert result.completed_at is not None  # Auto-set on success
    
    def test_failed_processing_result(self):
        """Test failed processing result"""
        result = ProcessingResult(
            success=False,
            operation_type="test_operation",
            processed_items=0,
            updated_items=0,
            message="Processing failed",
            errors=["Error 1", "Error 2"],
            execution_time=0.5
        )
        
        assert result.success is False
        assert result.operation_type == "test_operation"
        assert result.message == "Processing failed"
        assert len(result.errors) == 2
        assert result.errors[0] == "Error 1"
        assert result.processed_items == 0
        assert result.updated_items == 0
        assert result.completed_at is None  # Not set on failure


class TestDataConfig:
    """Test DataConfig functionality"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = DataConfig()
        
        assert config.sqlite_path == "./data/conjecture.db"
        assert config.chroma_path == "data/chroma"
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.max_tokens == 8000
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = DataConfig(
            sqlite_path="/custom/path.db",
            chroma_path="/custom/chroma",
            embedding_model="custom-model",
            max_tokens=4000
        )
        
        assert config.sqlite_path == "/custom/path.db"
        assert config.chroma_path == "/custom/chroma"
        assert config.embedding_model == "custom-model"
        assert config.max_tokens == 4000
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Test invalid max_tokens
        with pytest.raises(ValueError):
            DataConfig(max_tokens=500)  # Too low


class TestDirtyReason:
    """Test DirtyReason enum functionality"""
    
    def test_dirty_reason_values(self):
        """Test all dirty reason values are present"""
        expected_reasons = [
            "new_claim_added",
            "confidence_threshold",
            "supporting_claim_changed",
            "relationship_changed",
            "manual_mark",
            "batch_evaluation",
            "system_trigger"
        ]
        
        for expected in expected_reasons:
            assert expected in [r.value for r in DirtyReason]
    
    def test_dirty_reason_iteration(self):
        """Test dirty reason can be iterated"""
        reasons = list(DirtyReason)
        assert len(reasons) == 7
        assert DirtyReason.NEW_CLAIM_ADDED in reasons
        assert DirtyReason.MANUAL_MARK in reasons


class TestUtilityFunctions:
    """Test utility functions for claim management"""
    
    def test_generate_claim_id(self):
        """Test claim ID generation"""
        claim_id = generate_claim_id()
        assert claim_id.startswith("c")
        assert len(claim_id) == 8  # 'c' + 7 hex chars
        assert validate_claim_id(claim_id) is True
        
        # Test uniqueness
        claim_id2 = generate_claim_id()
        assert claim_id != claim_id2
    
    def test_validate_claim_id(self):
        """Test claim ID validation"""
        # Valid IDs
        assert validate_claim_id("c1234567") is True
        assert validate_claim_id("cabcdef1") is True
        
        # Invalid IDs
        assert validate_claim_id("1234567") is False  # Missing 'c'
        assert validate_claim_id("c123456") is False  # Too short
        assert validate_claim_id("c12345678") is False  # Too long
        assert validate_claim_id("cg123456") is False  # Invalid hex
    
    def test_validate_confidence(self):
        """Test confidence validation"""
        assert validate_confidence(0.0) is True
        assert validate_confidence(0.5) is True
        assert validate_confidence(1.0) is True
        
        assert validate_confidence(-0.1) is False
        assert validate_confidence(1.1) is False
    
    def test_create_claim(self):
        """Test claim creation factory function"""
        # Test basic claim creation
        claim = create_claim("Test claim", "fact", 0.8)
        assert claim.content == "Test claim"
        assert claim.confidence == 0.8
        assert "fact" in claim.tags
        
        # Test with custom tags
        claim_custom = create_claim(
            "Custom claim", 
            "concept", 
            0.9, 
            tags=["custom", "tag"]
        )
        assert claim_custom.tags == ["custom", "tag"]
        
        # Test instruction type
        claim_instruction = create_claim("Test instruction", "instruction", 0.7)
        assert "instruction" in claim_instruction.tags
        assert "guidance" in claim_instruction.tags
    
    def test_claim_index_functions(self):
        """Test claim indexing and filtering functions"""
        claims = [
            Claim(id="c1", content="Claim 1", confidence=0.9, tags=["a", "b"]),
            Claim(id="c2", content="Claim 2", confidence=0.7, tags=["b", "c"]),
            Claim(id="c3", content="Claim 3", confidence=0.8, tags=["a", "c"]),
            Claim(id="c4", content="Claim 4", confidence=0.6, supports=["c1"]),
            Claim(id="c5", content="Claim 5", confidence=0.5, supported_by=["c1", "c2"]),
        ]
        
        # Test create_claim_index
        index = create_claim_index(claims)
        assert len(index) == 5
        assert index["c1"].content == "Claim 1"
        assert index["c3"].confidence == 0.8
        
        # Test get_orphaned_claims
        orphaned = get_orphaned_claims(claims)
        orphaned_ids = [c.id for c in orphaned]
        assert "c1" in orphaned_ids  # No relationships
        assert "c2" in orphaned_ids  # No relationships
        assert "c3" in orphaned_ids  # No relationships
        assert "c4" not in orphaned_ids  # Has supports
        assert "c5" not in orphaned_ids  # Has supported_by
        
        # Test get_root_claims
        roots = get_root_claims(claims)
        root_ids = [c.id for c in roots]
        assert "c4" in root_ids  # Supports others but not supported
        
        # Test get_leaf_claims
        leaves = get_leaf_claims(claims)
        leaf_ids = [c.id for c in leaves]
        assert "c5" in leaf_ids  # Supported but doesn't support others
    
    def test_filter_functions(self):
        """Test claim filtering functions"""
        claims = [
            Claim(id="c1", content="High confidence", confidence=0.9, tags=["important"]),
            Claim(id="c2", content="Medium confidence", confidence=0.7, tags=["normal"]),
            Claim(id="c3", content="Low confidence", confidence=0.3, tags=["low"]),
            Claim(id="c4", content="High confidence 2", confidence=0.95, tags=["important", "critical"]),
        ]
        
        # Test filter_claims_by_tags
        important_claims = filter_claims_by_tags(claims, ["important"])
        assert len(important_claims) == 2
        assert all("important" in claim.tags for claim in important_claims)
        
        # Test multiple tags
        multi_claims = filter_claims_by_tags(claims, ["important", "low"])
        assert len(multi_claims) == 3  # c1, c3, c4
        
        # Test filter_claims_by_confidence
        high_confidence = filter_claims_by_confidence(claims, 0.8, 1.0)
        assert len(high_confidence) == 2  # c1, c4
        
        medium_confidence = filter_claims_by_confidence(claims, 0.6, 0.8)
        assert len(medium_confidence) == 1  # c2
        
        # Test full range
        all_claims = filter_claims_by_confidence(claims, 0.0, 1.0)
        assert len(all_claims) == 4


class TestModelIntegration:
    """Test integration between different models"""
    
    def test_claim_with_relationships(self):
        """Test claim model with relationship references"""
        claim = Claim(
            id="claim-1",
            content="Test claim with relationships",
            confidence=0.8
        )
        
        # Create relationships
        rel1 = Relationship(
            supporter_id=claim.id,
            supported_id="target-1",
            relationship_type="supports"
        )
        
        rel2 = Relationship(
            supporter_id="source-1",
            supported_id=claim.id,
            relationship_type="contradicts"
        )
        
        # Verify relationships reference claim
        assert rel1.supporter_id == claim.id
        assert rel2.supported_id == claim.id
    
    def test_filter_with_complex_criteria(self):
        """Test complex filtering scenarios"""
        filter_obj = ClaimFilter(
            tags=["critical", "reviewed"],
            states=[ClaimState.VALIDATED, ClaimState.EXPLORE],
            confidence_min=0.8,
            confidence_max=1.0,
            content_contains="important research",
            limit=50,
            offset=10
        )
        
        # Test all criteria are set
        assert len(filter_obj.tags) == 2
        assert len(filter_obj.states) == 2
        assert filter_obj.confidence_min == 0.8
        assert filter_obj.confidence_max == 1.0
        assert filter_obj.content_contains == "important research"
        assert filter_obj.limit == 50
        assert filter_obj.offset == 10


class TestExceptions:
    """Test custom exception classes"""
    
    def test_relationship_error(self):
        """Test RelationshipError exception"""
        with pytest.raises(RelationshipError):
            raise RelationshipError("Test relationship error")
    
    def test_claim_not_found_error(self):
        """Test ClaimNotFoundError exception"""
        with pytest.raises(ClaimNotFoundError):
            raise ClaimNotFoundError("Claim not found")
    
    def test_invalid_claim_error(self):
        """Test InvalidClaimError exception"""
        with pytest.raises(InvalidClaimError):
            raise InvalidClaimError("Invalid claim data")
    
    def test_data_layer_error(self):
        """Test DataLayerError exception"""
        with pytest.raises(DataLayerError):
            raise DataLayerError("Data layer operation failed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])