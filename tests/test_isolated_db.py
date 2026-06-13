# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Tests for Isolated Database Factory.

Ensures proper DB isolation for worktree testing.
"""

import pytest
import os
from pathlib import Path
from src.data.isolated_db import (
    IsolatedDBFactory,
    IsolatedClaimMemory,
    create_isolated_memory
)


class TestIsolatedDBFactory:
    """Test DB factory functionality."""

    def setup_method(self):
        self.factory = IsolatedDBFactory()

    def teardown_method(self):
        self.factory.cleanup_all()

    def test_factory_creates_db_dir(self):
        """Test DB directory is created."""
        assert self.factory.DB_DIR.exists()

    def test_get_experiment_db_path(self):
        """Test unique path generation."""
        path1 = self.factory.get_experiment_db_path("test1")
        path2 = self.factory.get_experiment_db_path("test2")

        assert path1 != path2
        assert "test1" in str(path1)
        assert "test2" in str(path2)

    def test_create_isolated_db(self):
        """Test isolated DB creation."""
        conn = self.factory.create_isolated_db("isolation_test")

        # Verify schema exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='claims'"
        )
        assert cursor.fetchone() is not None

        conn.close()

    def test_db_isolation_between_experiments(self):
        """Test that different experiments have isolated data."""
        # Create two isolated DBs
        conn1 = self.factory.create_isolated_db("exp1")
        conn2 = self.factory.create_isolated_db("exp2")

        # Add data to exp1
        conn1.execute(
            "INSERT INTO claims (content, is_correct) VALUES ('claim1', 1)"
        )
        conn1.commit()

        # Verify exp2 is empty
        cursor = conn2.execute("SELECT COUNT(*) FROM claims")
        assert cursor.fetchone()[0] == 0

        conn1.close()
        conn2.close()

    def test_cleanup_experiment(self):
        """Test experiment cleanup."""
        conn = self.factory.create_isolated_db("cleanup_test")
        db_path = self.factory.get_experiment_db_path("cleanup_test")
        conn.close()

        assert db_path.exists()
        self.factory.cleanup_experiment("cleanup_test")
        assert not db_path.exists()

    def test_list_experiment_dbs(self):
        """Test listing experiment databases."""
        self.factory.create_isolated_db("list_test1").close()
        self.factory.create_isolated_db("list_test2").close()

        dbs = self.factory.list_experiment_dbs()
        assert len(dbs) >= 2


class TestIsolatedClaimMemory:
    """Test isolated claim memory."""

    def test_add_and_retrieve_claims(self):
        """Test basic claim storage."""
        with create_isolated_memory("memory_test") as memory:
            memory.add_claim(
                content="Test claim",
                question="Test question?",
                confidence=0.9,
                is_correct=True,
                category="test"
            )

            claims = memory.get_claims()
            assert len(claims) == 1
            assert claims[0]["content"] == "Test claim"
            assert claims[0]["confidence"] == 0.9

    def test_correct_only_filter(self):
        """Test filtering for correct claims only."""
        with create_isolated_memory("filter_test") as memory:
            memory.add_claim("Correct", confidence=0.9, is_correct=True)
            memory.add_claim("Incorrect", confidence=0.9, is_correct=False)

            correct_only = memory.get_claims(correct_only=True)
            all_claims = memory.get_claims(correct_only=False)

            assert len(correct_only) == 1
            assert len(all_claims) == 2

    def test_confidence_threshold(self):
        """Test confidence filtering."""
        with create_isolated_memory("conf_test") as memory:
            memory.add_claim("Low", confidence=0.3, is_correct=True)
            memory.add_claim("High", confidence=0.9, is_correct=True)

            high_conf = memory.get_claims(min_confidence=0.8)
            assert len(high_conf) == 1
            assert high_conf[0]["content"] == "High"

    def test_stats(self):
        """Test memory statistics."""
        with create_isolated_memory("stats_test") as memory:
            memory.add_claim("C1", confidence=0.8, is_correct=True)
            memory.add_claim("C2", confidence=0.6, is_correct=False)

            stats = memory.get_stats()
            assert stats["total"] == 2
            assert stats["correct"] == 1
            assert stats["avg_confidence"] == 0.7

    def test_isolation_between_memories(self):
        """Test that different memory instances are isolated."""
        with create_isolated_memory("iso_test_a") as mem_a:
            mem_a.add_claim("Claim A", is_correct=True)

            with create_isolated_memory("iso_test_b") as mem_b:
                claims_b = mem_b.get_claims()
                assert len(claims_b) == 0  # B is isolated from A


class TestWorktreeIsolation:
    """Test worktree-specific isolation."""

    def test_worktree_id_generation(self):
        """Test worktree ID is generated."""
        factory = IsolatedDBFactory()
        worktree_id = factory.get_worktree_id()

        assert worktree_id is not None
        assert len(worktree_id) > 0

    def test_branch_name_detection(self):
        """Test branch name detection."""
        factory = IsolatedDBFactory()
        branch = factory.get_branch_name()

        assert branch is not None
        # Should be either actual branch name or 'default'
        assert len(branch) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
