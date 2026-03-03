"""
Tests for D-0007: Acyclic Graph Enforcement.

Verifies that:
  1. would_create_cycle() correctly detects cycles via BFS.
  2. assert_no_cycle() raises CycleDetectedError when a cycle would result.
  3. ClaimRepository.add_relationship() enforces the DAG constraint at the
     repository level:
       - A -> B then B -> A raises CycleDetectedError.
       - Longer cycles (A -> B -> C -> A) are also caught.
       - Valid DAG relationships are accepted.
       - Self-loops are rejected.
       - Duplicate relationships are silently ignored (idempotent).
       - Missing claims raise ValueError.
"""

import pytest
import asyncio

from src.core.graph_validator import (
    CycleDetectedError,
    would_create_cycle,
    assert_no_cycle,
)
from src.core.models import Claim
from src.data.repositories import ClaimRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_claim(claim_id: str, supers: list[str] | None = None) -> Claim:
    """Create a minimal Claim for testing."""
    return Claim(
        id=claim_id,
        content=f"Test claim {claim_id}",
        confidence=0.5,
        supers=supers or [],
    )


def get_supers_from_dict(claims: dict[str, Claim]):
    """Return a get_supers callable backed by a plain dict."""
    def _get_supers(claim_id: str) -> list[str]:
        claim = claims.get(claim_id)
        return claim.supers if claim else []
    return _get_supers


# ---------------------------------------------------------------------------
# Tests for would_create_cycle()
# ---------------------------------------------------------------------------

class TestWouldCreateCycle:
    """Unit tests for the pure cycle-detection function."""

    def test_no_cycle_simple_dag(self):
        """A -> B in an otherwise empty graph is fine."""
        claims = {
            "A": make_claim("A"),
            "B": make_claim("B"),
        }
        get_supers = get_supers_from_dict(claims)
        assert would_create_cycle("A", "B", get_supers) is False

    def test_direct_cycle_a_to_b_then_b_to_a(self):
        """After A -> B exists, adding B -> A creates a cycle."""
        # A -> B already in place: A.supers = ["B"]
        claims = {
            "A": make_claim("A", supers=["B"]),
            "B": make_claim("B"),
        }
        get_supers = get_supers_from_dict(claims)
        # Now checking if B -> A would cycle: traverse A's supers -> B -> B's supers -> ...
        # From target=A, we see A.supers=["B"] -> B -> B.supers=[] stop.
        # Wait — source=B, target=A.  We BFS from A upward and check if we hit B.
        # A.supers = ["B"] so immediately we reach B == source. Cycle detected.
        assert would_create_cycle("B", "A", get_supers) is True

    def test_three_node_cycle(self):
        """A -> B -> C already exists; adding C -> A creates a cycle."""
        claims = {
            "A": make_claim("A", supers=["B"]),
            "B": make_claim("B", supers=["C"]),
            "C": make_claim("C"),
        }
        get_supers = get_supers_from_dict(claims)
        # source=C, target=A.  BFS from A: A.supers=["B"] -> B.supers=["C"] -> reach C == source.
        assert would_create_cycle("C", "A", get_supers) is True

    def test_no_cycle_branching_dag(self):
        """Branching DAG without cycles is correctly identified as safe."""
        claims = {
            "A": make_claim("A", supers=["C"]),
            "B": make_claim("B", supers=["C"]),
            "C": make_claim("C"),
        }
        get_supers = get_supers_from_dict(claims)
        # Adding D -> A should be fine (D has no existing supers).
        claims["D"] = make_claim("D")
        get_supers = get_supers_from_dict(claims)
        assert would_create_cycle("D", "A", get_supers) is False

    def test_self_loop_is_cycle(self):
        """A -> A is always a cycle."""
        claims = {"A": make_claim("A")}
        get_supers = get_supers_from_dict(claims)
        assert would_create_cycle("A", "A", get_supers) is True

    def test_unrelated_claims_no_cycle(self):
        """Two disconnected claims can be safely connected."""
        claims = {
            "X": make_claim("X"),
            "Y": make_claim("Y"),
        }
        get_supers = get_supers_from_dict(claims)
        assert would_create_cycle("X", "Y", get_supers) is False

    def test_diamond_dag_cycle_detection_and_safe_edge(self):
        """Diamond shape (A->B, A->C, B->D, C->D) is a valid DAG.

        D->A would create a cycle (A->B->D->A), which is correctly detected.
        A brand-new claim E->A is safe because E has no ancestors.
        """
        claims = {
            "A": make_claim("A", supers=["B", "C"]),
            "B": make_claim("B", supers=["D"]),
            "C": make_claim("C", supers=["D"]),
            "D": make_claim("D"),
        }
        get_supers = get_supers_from_dict(claims)

        # D -> A would cycle: BFS from A reaches B, then D == source(D). Cycle!
        assert would_create_cycle("D", "A", get_supers) is True

        # E -> A is safe: E has no supers, BFS from A cannot reach E.
        claims["E"] = make_claim("E")
        get_supers = get_supers_from_dict(claims)
        assert would_create_cycle("E", "A", get_supers) is False


# ---------------------------------------------------------------------------
# Tests for assert_no_cycle()
# ---------------------------------------------------------------------------

class TestAssertNoCycle:
    """Tests for the assertion helper that raises CycleDetectedError."""

    def test_no_error_when_no_cycle(self):
        """assert_no_cycle does not raise for a safe relationship."""
        claims = {
            "A": make_claim("A"),
            "B": make_claim("B"),
        }
        get_supers = get_supers_from_dict(claims)
        # Should not raise.
        assert_no_cycle("A", "B", get_supers)

    def test_raises_cycle_detected_error(self):
        """assert_no_cycle raises CycleDetectedError when cycle would occur."""
        claims = {
            "A": make_claim("A", supers=["B"]),
            "B": make_claim("B"),
        }
        get_supers = get_supers_from_dict(claims)
        with pytest.raises(CycleDetectedError):
            assert_no_cycle("B", "A", get_supers)

    def test_cycle_detected_error_is_value_error(self):
        """CycleDetectedError is a subclass of ValueError for broad compatibility."""
        claims = {
            "A": make_claim("A", supers=["B"]),
            "B": make_claim("B"),
        }
        get_supers = get_supers_from_dict(claims)
        with pytest.raises(ValueError):
            assert_no_cycle("B", "A", get_supers)

    def test_error_contains_claim_ids(self):
        """CycleDetectedError message mentions the problematic claim IDs."""
        claims = {
            "A": make_claim("A", supers=["B"]),
            "B": make_claim("B"),
        }
        get_supers = get_supers_from_dict(claims)
        with pytest.raises(CycleDetectedError) as exc_info:
            assert_no_cycle("B", "A", get_supers)
        error = exc_info.value
        assert error.source_id == "B"
        assert error.target_id == "A"
        assert "B" in str(error)
        assert "A" in str(error)


# ---------------------------------------------------------------------------
# Tests for ClaimRepository.add_relationship()
# ---------------------------------------------------------------------------

class TestClaimRepositoryAddRelationship:
    """Integration tests for DAG enforcement at the repository level."""

    def _run(self, coro):
        """Run a coroutine synchronously (avoids asyncio fixture boilerplate)."""
        return asyncio.get_event_loop().run_until_complete(coro)

    async def _make_repo_with_claims(self, *claim_ids: str) -> ClaimRepository:
        """Create a fresh repository populated with claims for each given ID."""
        repo = ClaimRepository()
        await repo.initialize()
        for cid in claim_ids:
            await repo.create(make_claim(cid))
        return repo

    def test_valid_relationship_accepted(self):
        """A simple A -> B relationship in a fresh graph is accepted."""
        async def _test():
            repo = await self._make_repo_with_claims("A", "B")
            await repo.add_relationship("A", "B")
            a = await repo.get_by_id("A")
            b = await repo.get_by_id("B")
            assert "B" in a.supers
            assert "A" in b.subs

        self._run(_test())

    def test_gate_a_to_b_then_b_to_a_raises(self):
        """The gate test: adding A->B then B->A raises CycleDetectedError."""
        async def _test():
            repo = await self._make_repo_with_claims("A", "B")
            # First relationship is fine.
            await repo.add_relationship("A", "B")
            # Second would create a cycle.
            with pytest.raises(CycleDetectedError):
                await repo.add_relationship("B", "A")

        self._run(_test())

    def test_three_node_cycle_caught(self):
        """A->B, B->C, then C->A raises CycleDetectedError."""
        async def _test():
            repo = await self._make_repo_with_claims("A", "B", "C")
            await repo.add_relationship("A", "B")
            await repo.add_relationship("B", "C")
            with pytest.raises(CycleDetectedError):
                await repo.add_relationship("C", "A")

        self._run(_test())

    def test_self_loop_raises(self):
        """A -> A raises CycleDetectedError."""
        async def _test():
            repo = await self._make_repo_with_claims("A")
            with pytest.raises(CycleDetectedError):
                await repo.add_relationship("A", "A")

        self._run(_test())

    def test_missing_supporter_raises_value_error(self):
        """Attempting to add a relationship with an unknown supporter raises ValueError."""
        async def _test():
            repo = await self._make_repo_with_claims("B")
            with pytest.raises(ValueError):
                await repo.add_relationship("MISSING", "B")

        self._run(_test())

    def test_missing_supported_raises_value_error(self):
        """Attempting to add a relationship with an unknown supported claim raises ValueError."""
        async def _test():
            repo = await self._make_repo_with_claims("A")
            with pytest.raises(ValueError):
                await repo.add_relationship("A", "MISSING")

        self._run(_test())

    def test_duplicate_relationship_is_idempotent(self):
        """Adding the same relationship twice does not raise and does not duplicate entries."""
        async def _test():
            repo = await self._make_repo_with_claims("A", "B")
            await repo.add_relationship("A", "B")
            await repo.add_relationship("A", "B")  # Should not raise.
            a = await repo.get_by_id("A")
            assert a.supers.count("B") == 1  # No duplicate.

        self._run(_test())

    def test_valid_dag_chain_accepted(self):
        """A -> B -> C is a valid chain and all relationships are accepted."""
        async def _test():
            repo = await self._make_repo_with_claims("A", "B", "C")
            await repo.add_relationship("A", "B")
            await repo.add_relationship("B", "C")
            a = await repo.get_by_id("A")
            b = await repo.get_by_id("B")
            c = await repo.get_by_id("C")
            assert "B" in a.supers
            assert "A" in b.subs
            assert "C" in b.supers
            assert "B" in c.subs

        self._run(_test())

    def test_diamond_dag_accepted(self):
        """A->B, A->C, B->D, C->D is a valid diamond DAG."""
        async def _test():
            repo = await self._make_repo_with_claims("A", "B", "C", "D")
            await repo.add_relationship("A", "B")
            await repo.add_relationship("A", "C")
            await repo.add_relationship("B", "D")
            await repo.add_relationship("C", "D")
            d = await repo.get_by_id("D")
            assert "B" in d.subs
            assert "C" in d.subs

        self._run(_test())

    def test_cycle_error_is_also_value_error(self):
        """CycleDetectedError can be caught as ValueError for broad compatibility."""
        async def _test():
            repo = await self._make_repo_with_claims("A", "B")
            await repo.add_relationship("A", "B")
            with pytest.raises(ValueError):
                await repo.add_relationship("B", "A")

        self._run(_test())
