"""
Simple Test Suite for Enhanced Conjecture
Basic functionality tests without complex dependencies
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.core.models import Claim, ClaimState, ClaimScope


class TestClaimModel:
    """Test the core claim model"""

    def test_claim_creation(self):
        """Test basic claim creation"""
        claim = Claim(
            id="test_001",
            content="This is a test claim with sufficient length",
            confidence=0.85,
            tags=["test", "claim"],
        )

        assert claim.id == "test_001"
        assert claim.confidence == 0.85
        assert "test" in claim.tags
        assert claim.state == ClaimState.EXPLORE

    def test_claim_validation(self):
        """Test claim validation"""
        # Test valid claim
        claim = Claim(
            id="valid_001",
            content="Valid claim content that meets minimum length requirements",
            confidence=0.75,
            tags=["reference"],
        )
        assert claim is not None

        # Test invalid confidence (too high)
        with pytest.raises(ValueError):
            Claim(
                id="invalid_001",
                content="Valid content with enough length",
                confidence=1.5,  # Invalid confidence
                tags=["concept"],
            )

        # Test short content
        with pytest.raises(ValueError):
            Claim(
                id="invalid_002",
                content="Too",  # Too short (min 5 chars)
                confidence=0.5,
                tags=["concept"],
            )

    def test_claim_relationships(self):
        """Test claim relationship fields"""
        claim1 = Claim(
            id="claim1",
            content="First claim with sufficient content",
            confidence=0.8,
            tags=["concept"],
            supports=["claim2"],
        )

        claim2 = Claim(
            id="claim2",
            content="Second claim with sufficient content",
            confidence=0.7,
            tags=["example"],
            supported_by=["claim1"],
        )

        # Test support relationships
        assert "claim2" in claim1.supports
        assert "claim1" in claim2.supported_by

    def test_claim_dirty_flags(self):
        """Test dirty flag functionality"""
        claim = Claim(
            id="dirty_test",
            content="Test claim for dirty flag testing",
            confidence=0.5,
            tags=["concept"],
        )

        # New claims should be dirty by default
        assert claim.is_dirty == True


class TestClaimStates:
    """Test claim state enumeration"""

    def test_claim_states(self):
        """Test claim state enumeration values"""
        assert ClaimState.EXPLORE == "Explore"
        assert ClaimState.VALIDATED == "Validated"
        assert ClaimState.ORPHANED == "Orphaned"
        assert ClaimState.QUEUED == "Queued"

    def test_claim_with_different_states(self):
        """Test creating claims with different states"""
        for state in ClaimState:
            claim = Claim(
                id=f"state_test_{state.value}",
                content=f"Test claim with state {state.value}",
                confidence=0.5,
                state=state,
            )
            assert claim.state == state


class TestClaimScopes:
    """Test claim scope enumeration"""

    def test_scope_hierarchy(self):
        """Test scope hierarchy"""
        hierarchy = ClaimScope.get_hierarchy()
        assert len(hierarchy) == 4
        assert ClaimScope.USER_WORKSPACE.value in hierarchy[0]

    def test_default_scope(self):
        """Test default scope is most restrictive"""
        default = ClaimScope.get_default()
        assert "user" in default.lower()


class TestBasicImports:
    """Test basic functionality without external dependencies"""

    def test_core_imports(self):
        """Test that core modules can be imported"""
        try:
            from src.core.models import Claim, ClaimState, ClaimScope
            from src.conjecture import Conjecture, ExplorationResult

            assert True
        except ImportError as e:
            pytest.fail(f"Core import failed: {e}")

    def test_config_import(self):
        """Test configuration import"""
        try:
            from src.config.config import get_config

            assert True
        except ImportError as e:
            pytest.skip(f"Config module not available: {e}")

    def test_cli_import(self):
        """Test CLI import"""
        try:
            from src.cli.modular_cli import app

            assert app is not None
        except ImportError as e:
            pytest.skip(f"CLI module not available: {e}")


class TestConjecture:
    """Test main Conjecture class"""

    def test_conjecture_import(self):
        """Test Conjecture class can be imported"""
        from src.conjecture import Conjecture

        assert Conjecture is not None

    def test_exploration_result_import(self):
        """Test ExplorationResult can be imported"""
        from src.conjecture import ExplorationResult

        assert ExplorationResult is not None


class TestDataModels:
    """Test data model serialization"""

    def test_claim_to_dict(self):
        """Test claim can be converted to dict"""
        claim = Claim(
            id="dict_test",
            content="Test claim for dictionary conversion",
            confidence=0.75,
            tags=["test", "serialization"],
        )

        claim_dict = claim.model_dump()
        assert claim_dict["id"] == "dict_test"
        assert claim_dict["confidence"] == 0.75
        assert "test" in claim_dict["tags"]

    def test_claim_to_chroma_metadata(self):
        """Test claim can be converted to ChromaDB metadata"""
        claim = Claim(
            id="chroma_test",
            content="Test claim for ChromaDB metadata",
            confidence=0.8,
            tags=["chroma", "test"],
        )

        metadata = claim.to_chroma_metadata()
        assert metadata["confidence"] == 0.8
        assert metadata["state"] == ClaimState.EXPLORE.value
        assert "chroma" in metadata["tags"]


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
