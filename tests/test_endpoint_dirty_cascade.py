"""
Tests for A-0011: Dirty flag cascade via ConjectureEndpoint.

Gate: Adding a sub-claim marks its supers dirty.

Scenario:
  1. Create claim B (a parent / super claim).
  2. Create claim A with B in supers (A provides evidence FOR B).
  3. Verify B is marked dirty (is_dirty=True) after A is created.

Also covers:
  - update_claim() cascades dirty flags to supers when content/confidence changes.
  - No cascade to subs (unidirectional toward root only).
  - Cascade is skipped when no supers are referenced.
"""

import asyncio
import tempfile
import os
import pytest

from src.endpoint.conjecture_endpoint import ConjectureEndpoint
from src.core.models import DirtyReason


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _make_endpoint(tmp_path: str) -> ConjectureEndpoint:
    """Create and initialize an endpoint backed by a temp SQLite DB."""
    db_path = os.path.join(tmp_path, "test_cascade.db")
    ep = ConjectureEndpoint(db_path=db_path, vector_path=":memory:")
    # Disable vector store so tests don't depend on FAISS
    ep._vector_store = None
    await ep._data_manager.initialize()
    ep._initialized = True
    return ep


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path):
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Test: Gate — creating a sub-claim marks its supers dirty
# ---------------------------------------------------------------------------

class TestCreateClaimCascadeDirty:
    """A-0011 gate: sub-claim creation cascades dirty flags to supers."""

    @pytest.mark.asyncio
    async def test_creating_sub_marks_super_dirty(self, tmp_dir):
        """Gate: Create A with B in supers -> B becomes dirty."""
        ep = await _make_endpoint(tmp_dir)

        # Step 1: Create claim B (parent/super) — starts with is_dirty=True by default
        # First mark it clean by noting its creation state, then verify the cascade
        resp_b = await ep.create_claim(
            content="B: parent claim that will be supported by A",
            confidence=0.8,
        )
        assert resp_b.success, f"Failed to create B: {resp_b.errors}"
        claim_b_id = resp_b.data["id"]

        # Manually mark B clean so we can detect the cascade dirtying it
        await ep._data_manager.update_claim(claim_b_id, {
            "is_dirty": False,
            "dirty_reason": None,
            "dirty_timestamp": None,
            "dirty_priority": 0,
        })

        # Confirm B is clean
        b_before = await ep._data_manager.get_claim(claim_b_id)
        assert b_before is not None
        assert b_before.is_dirty is False, "B should be clean before cascade"

        # Step 2: Create claim A with B in supers (A supports B, A is sub of B)
        resp_a = await ep.create_claim(
            content="A: sub-claim providing evidence for B",
            confidence=0.7,
            supers=[claim_b_id],
        )
        assert resp_a.success, f"Failed to create A: {resp_a.errors}"
        assert claim_b_id in resp_a.data["supers_marked_dirty"], (
            "create_claim should report that B was marked dirty"
        )

        # Step 3: Verify B is now dirty
        b_after = await ep._data_manager.get_claim(claim_b_id)
        assert b_after is not None
        assert b_after.is_dirty is True, (
            "B must be dirty after a sub-claim (A) was created that supports it"
        )
        assert b_after.dirty_reason == DirtyReason.SUPPORTING_CLAIM_CHANGED, (
            "dirty_reason should be SUPPORTING_CLAIM_CHANGED"
        )

    @pytest.mark.asyncio
    async def test_creating_sub_marks_multiple_supers_dirty(self, tmp_dir):
        """Creating a sub with multiple supers marks all of them dirty."""
        ep = await _make_endpoint(tmp_dir)

        # Create two super claims and clean them
        resp_b1 = await ep.create_claim(content="B1: first super claim", confidence=0.8)
        resp_b2 = await ep.create_claim(content="B2: second super claim", confidence=0.75)
        assert resp_b1.success and resp_b2.success
        b1_id = resp_b1.data["id"]
        b2_id = resp_b2.data["id"]

        clean_update = {"is_dirty": False, "dirty_reason": None, "dirty_timestamp": None, "dirty_priority": 0}
        await ep._data_manager.update_claim(b1_id, dict(clean_update))
        await ep._data_manager.update_claim(b2_id, dict(clean_update))

        # Create sub referencing both supers
        resp_a = await ep.create_claim(
            content="A: sub-claim supporting both B1 and B2",
            confidence=0.6,
            supers=[b1_id, b2_id],
        )
        assert resp_a.success
        assert set(resp_a.data["supers_marked_dirty"]) == {b1_id, b2_id}

        b1_after = await ep._data_manager.get_claim(b1_id)
        b2_after = await ep._data_manager.get_claim(b2_id)
        assert b1_after.is_dirty is True
        assert b2_after.is_dirty is True

    @pytest.mark.asyncio
    async def test_creating_claim_without_supers_no_cascade(self, tmp_dir):
        """Creating a claim with no supers does not cascade dirty flags."""
        ep = await _make_endpoint(tmp_dir)

        # Create an unrelated claim and clean it
        resp_x = await ep.create_claim(content="X: unrelated claim", confidence=0.9)
        assert resp_x.success
        x_id = resp_x.data["id"]
        await ep._data_manager.update_claim(x_id, {
            "is_dirty": False, "dirty_reason": None, "dirty_timestamp": None, "dirty_priority": 0
        })

        # Create a new claim with no supers
        resp_y = await ep.create_claim(
            content="Y: standalone claim with no supers",
            confidence=0.5,
        )
        assert resp_y.success
        assert resp_y.data["supers_marked_dirty"] == []

        # X must remain clean
        x_after = await ep._data_manager.get_claim(x_id)
        assert x_after.is_dirty is False, "Unrelated claim X must not be dirtied"

    @pytest.mark.asyncio
    async def test_cascade_does_not_touch_subs(self, tmp_dir):
        """A-0011: cascade is unidirectional; subs are never marked dirty."""
        ep = await _make_endpoint(tmp_dir)

        # Create a grandchild claim (C) that will be a sub of A
        resp_c = await ep.create_claim(content="C: grandchild claim", confidence=0.6)
        assert resp_c.success
        c_id = resp_c.data["id"]

        # Create super claim B and clean it
        resp_b = await ep.create_claim(content="B: super claim", confidence=0.85)
        assert resp_b.success
        b_id = resp_b.data["id"]

        clean = {"is_dirty": False, "dirty_reason": None, "dirty_timestamp": None, "dirty_priority": 0}
        await ep._data_manager.update_claim(b_id, dict(clean))
        await ep._data_manager.update_claim(c_id, dict(clean))

        # Create A: its super is B, its sub is C
        resp_a = await ep.create_claim(
            content="A: middle claim — super is B, sub is C",
            confidence=0.7,
            supers=[b_id],
            subs=[c_id],
        )
        assert resp_a.success

        # B should be dirty (cascade went upward)
        b_after = await ep._data_manager.get_claim(b_id)
        assert b_after.is_dirty is True

        # C must remain clean (no downward cascade)
        c_after = await ep._data_manager.get_claim(c_id)
        assert c_after.is_dirty is False, "Subs must never be marked dirty by cascade"


# ---------------------------------------------------------------------------
# Test: update_claim() cascade
# ---------------------------------------------------------------------------

class TestUpdateClaimCascadeDirty:
    """Dirty cascade when an existing claim is updated via endpoint."""

    @pytest.mark.asyncio
    async def test_update_content_marks_supers_dirty(self, tmp_dir):
        """Updating claim content cascades dirty to supers."""
        ep = await _make_endpoint(tmp_dir)

        # Create B (super) and A (sub supporting B)
        resp_b = await ep.create_claim(content="B: super claim to re-evaluate", confidence=0.8)
        assert resp_b.success
        b_id = resp_b.data["id"]

        resp_a = await ep.create_claim(
            content="A: original sub content",
            confidence=0.7,
            supers=[b_id],
        )
        assert resp_a.success
        a_id = resp_a.data["id"]

        # Clean both before update
        clean = {"is_dirty": False, "dirty_reason": None, "dirty_timestamp": None, "dirty_priority": 0}
        await ep._data_manager.update_claim(a_id, dict(clean))
        await ep._data_manager.update_claim(b_id, dict(clean))

        # Update A's content -> should cascade dirty to B
        update_resp = await ep.update_claim(a_id, {"content": "A: updated sub content"})
        assert update_resp.success, f"Update failed: {update_resp.errors}"
        assert b_id in update_resp.data["supers_marked_dirty"]

        b_after = await ep._data_manager.get_claim(b_id)
        assert b_after.is_dirty is True

    @pytest.mark.asyncio
    async def test_update_confidence_marks_supers_dirty(self, tmp_dir):
        """Updating claim confidence cascades dirty to supers."""
        ep = await _make_endpoint(tmp_dir)

        resp_b = await ep.create_claim(content="B: super that depends on A confidence", confidence=0.8)
        assert resp_b.success
        b_id = resp_b.data["id"]

        resp_a = await ep.create_claim(
            content="A: sub with variable confidence",
            confidence=0.5,
            supers=[b_id],
        )
        assert resp_a.success
        a_id = resp_a.data["id"]

        clean = {"is_dirty": False, "dirty_reason": None, "dirty_timestamp": None, "dirty_priority": 0}
        await ep._data_manager.update_claim(a_id, dict(clean))
        await ep._data_manager.update_claim(b_id, dict(clean))

        update_resp = await ep.update_claim(a_id, {"confidence": 0.9})
        assert update_resp.success
        assert b_id in update_resp.data["supers_marked_dirty"]

        b_after = await ep._data_manager.get_claim(b_id)
        assert b_after.is_dirty is True

    @pytest.mark.asyncio
    async def test_update_without_meaningful_change_no_cascade(self, tmp_dir):
        """Updating only tags (not content/confidence/supers) does not cascade."""
        ep = await _make_endpoint(tmp_dir)

        resp_b = await ep.create_claim(content="B: super claim", confidence=0.8)
        assert resp_b.success
        b_id = resp_b.data["id"]

        resp_a = await ep.create_claim(
            content="A: sub claim", confidence=0.7, supers=[b_id]
        )
        assert resp_a.success
        a_id = resp_a.data["id"]

        clean = {"is_dirty": False, "dirty_reason": None, "dirty_timestamp": None, "dirty_priority": 0}
        await ep._data_manager.update_claim(a_id, dict(clean))
        await ep._data_manager.update_claim(b_id, dict(clean))

        # Update only tags — neither content nor confidence changes
        update_resp = await ep.update_claim(a_id, {"tags": ["new-tag"]})
        assert update_resp.success
        assert update_resp.data["supers_marked_dirty"] == []

        b_after = await ep._data_manager.get_claim(b_id)
        assert b_after.is_dirty is False, "B must stay clean when only tags changed"

    @pytest.mark.asyncio
    async def test_update_nonexistent_claim_returns_error(self, tmp_dir):
        """Updating a non-existent claim returns a clear error response."""
        ep = await _make_endpoint(tmp_dir)

        update_resp = await ep.update_claim("c_does_not_exist", {"confidence": 0.9})
        assert update_resp.success is False
        assert "CLAIM_NOT_FOUND" in update_resp.errors
