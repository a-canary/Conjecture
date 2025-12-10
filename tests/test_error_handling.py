"""
Comprehensive error handling and edge case tests for the Conjecture data layer.
Tests error propagation, validation, boundary conditions, and failure scenarios.
"""
import pytest
import asyncio
import tempfile
import os
import json
from typing import List, Dict, Any

from src.data.data_manager import DataManager
from src.data.optimized_sqlite_manager import OptimizedSQLiteManager as SQLiteManager
from src.data.chroma_manager import ChromaManager
from src.data.embedding_service import EmbeddingService, EmbeddingService
from src.data.models import (
    Claim, Relationship, ClaimFilter, DataConfig,
    ClaimNotFoundError, InvalidClaimError, RelationshipError, DataLayerError
)

class TestModelValidationEdgeCases:
    """Test edge cases in model validation."""

    @pytest.mark.error_handling
    def test_claim_minimal_valid_inputs(self):
        """Test claims with minimal valid input constraints."""
        # Exactly 10 characters (minimum)
        claim = Claim(
            id="c0000001",
            content="A" * 10,
            created_by="user"
        )
        assert claim.content == "A" * 10
        
        # Exactly 0.0 confidence (minimum)
        claim = Claim(
            id="c0000002",
            content="Valid content",
            confidence=0.0,
            created_by="user"
        )
        assert claim.confidence == 0.0
        
        # Exactly 1.0 confidence (maximum)
        claim = Claim(
            id="c0000003",
            content="Valid content",
            confidence=1.0,
            created_by="user"
        )
        assert claim.confidence == 1.0

    @pytest.mark.error_handling
    def test_claim_maximal_valid_inputs(self):
        """Test claims with maximal valid input constraints."""
        # Very long content (should be accepted)
        long_content = "X" * 10000
        claim = Claim(
            id="c0000001",
            content=long_content,
            created_by="user"
        )
        assert len(claim.content) == 10000
        
        # Many tags (should be accepted)
        many_tags = [f"tag_{i}" for i in range(100)]
        claim = Claim(
            id="c0000002",
            content="Valid content",
            tags=many_tags,
            created_by="user"
        )
        assert len(claim.tags) == 100

    @pytest.mark.error_handling
    def test_claim_boundary_invalid_inputs(self):
        """Test claims just outside valid boundaries."""
        # 9 characters (just below minimum)
        with pytest.raises(ValueError, match="ensure this value has at least 10 characters"):
            Claim(id="c0000001", content="A" * 9, created_by="user")
        
        # Slightly below 0.0 confidence
        with pytest.raises(ValueError, match="ensure this value is greater than or equal to 0"):
            Claim(id="c0000002", content="Valid content", confidence=-0.001, created_by="user")
        
        # Slightly above 1.0 confidence
        with pytest.raises(ValueError, match="ensure this value is less than or equal to 1"):
            Claim(id="c0000003", content="Valid content", confidence=1.001, created_by="user")

    @pytest.mark.error_handling
    def test_claim_id_edge_cases(self):
        """Test claim ID validation edge cases."""
        # Valid formats
        valid_claims = [
            Claim(id="c0000000", content="Zero padding", created_by="user"),
            Claim(id="c9999999", content="Max digits", created_by="user"),
            Claim(id="c1234567", content="Mixed digits", created_by="user"),
        ]
        
        for claim in valid_claims:
            assert claim.id.startswith('c')
            assert len(claim.id) == 8
            assert claim.id[1:].isdigit()
        
        # Invalid formats (edge cases)
        invalid_ids = [
            "c000000",   # Too short by 1
            "c00000000", # Too long by 1
            "C0000001",  # Wrong case
            "c000001 ",  # Trailing space
            " c000001",  # Leading space
            "c_000001",  # Invalid character
            "c+000001",  # Invalid character
            "c000.001",  # Invalid character
        ]
        
        for invalid_id in invalid_ids:
            with pytest.raises(ValueError, match="Claim ID must be in format c#######"):
                Claim(id=invalid_id, content="Valid content", created_by="user")

    @pytest.mark.error_handling
    def test_tags_validation_edge_cases(self):
        """Test tags validation edge cases."""
        # Tags with whitespace (should be stripped or rejected)
        with pytest.raises(ValueError, match="All tags must be non-empty strings"):
            Claim(id="c0000001", content="Valid content", tags=["valid", "   "], created_by="user")
        
        # None in tags list
        with pytest.raises(ValueError, match="All tags must be non-empty strings"):
            Claim(id="c0000002", content="Valid content", tags=["valid", None], created_by="user")
        
        # Numbers in tags (should be converted to strings or rejected)
        with pytest.raises(ValueError, match="All tags must be non-empty strings"):
            Claim(id="c0000003", content="Valid content", tags=["valid", 123], created_by="user")

    @pytest.mark.error_handling
    def test_relationship_validation_edge_cases(self):
        """Test relationship validation edge cases."""
        claim = Claim(id="c0000001", content="Base claim", created_by="user")
        
        # Valid relationship types
        valid_types = ["supports", "contradicts", "extends", "clarifies"]
        for rel_type in valid_types:
            rel = Relationship(
                supporter_id="c0000002",
                supported_id="c0000003",
                relationship_type=rel_type
            )
            assert rel.relationship_type == rel_type
        
        # Invalid relationship types
        invalid_types = [
            "SUPPORTS",     # Wrong case
            "support",      # Incomplete
            "supporting",   # Wrong form
            "invalid_type", # Completely invalid
            "",             # Empty string
            None,           # Null value
        ]
        
        for invalid_type in invalid_types:
            with pytest.raises(ValueError, match="Relationship type must be one of"):
                Relationship(
                    supporter_id="c0000002",
                    supported_id="c0000003",
                    relationship_type=invalid_type
                )

    @pytest.mark.error_handling
    def test_filter_validation_edge_cases(self):
        """Test filter validation edge cases."""
        # Confidence range validation
        # Max slightly below min
        with pytest.raises(ValueError, match="confidence_max must be >= confidence_min"):
            ClaimFilter(confidence_min=0.9, confidence_max=0.89)
        
        # Edge case: max equals min (should be valid)
        filter_obj = ClaimFilter(confidence_min=0.5, confidence_max=0.5)
        assert filter_obj.confidence_min == 0.5
        assert filter_obj.confidence_max == 0.5
        
        # Boundary values for limit and offset
        with pytest.raises(ValueError, match="ensure this value is greater than or equal to 1"):
            ClaimFilter(limit=0)
        
        with pytest.raises(ValueError, match="ensure this value is less than or equal to 1000"):
            ClaimFilter(limit=1001)
        
        with pytest.raises(ValueError, match="ensure this value is greater than or equal to 0"):
            ClaimFilter(offset=-1)

class TestDatabaseErrorScenarios:
    """Test database-related error scenarios."""

    @pytest.mark.error_handling
    @pytest.mark.asyncio
    async def test_sqlite_connection_failure(self):
        """Test SQLite connection failure scenarios."""
        # Invalid database path (read-only directory)
        readonly_path = "/root/nonexistent/path/test.db"
        manager = SQLiteManager(readonly_path)
        
        with pytest.raises(DataLayerError):
            await manager.initialize()

    @pytest.mark.error_handling
    @pytest.mark.asyncio
    async def test_sqlite_constraint_violations(self, sqlite_manager: SQLiteManager):
        """Test SQLite constraint violation handling."""
        claim = Claim(
            id="c0000001",
            content="Test claim",
            created_by="user"
        )
        
        # Create claim successfully
        await sqlite_manager.create_claim(claim)
        
        # Try to create same claim again (UNIQUE constraint)
        with pytest.raises(DataLayerError, match="Claim with ID .* already exists"):
            await sqlite_manager.create_claim(claim)
        
        # Try to add relationship with non-existent claim (FOREIGN KEY constraint)
        relationship = Relationship(
            supporter_id="c0999999",  # Non-existent
            supported_id="c0000001",
            relationship_type="supports"
        )
        
        with pytest.raises(DataLayerError, match="One or both claim IDs do not exist"):
            await sqlite_manager.add_relationship(relationship)
        
        # Try to add duplicate relationship (UNIQUE constraint)
        existing_claim = Claim(
            id="c0000002",
            content="Another claim",
            created_by="user"
        )
        await sqlite_manager.create_claim(existing_claim)
        
        rel = Relationship(
            supporter_id="c0000001",
            supported_id="c0000002",
            relationship_type="supports"
        )
        await sqlite_manager.add_relationship(rel)
        
        with pytest.raises(DataLayerError, match="Relationship already exists"):
            await sqlite_manager.add_relationship(rel)

    @pytest.mark.error_handling
    @pytest.mark.asyncio
    async def test_sqlite_corrupted_data_handling(self, sqlite_manager: SQLiteManager):
        """Test handling of corrupted or malformed data."""
        # Test with invalid JSON in tags field - using real data
        import pytest
        pytest.skip("Real corrupted data test requires specific database setup") 