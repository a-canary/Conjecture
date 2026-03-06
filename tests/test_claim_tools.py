"""
Tests for src/process/claim_tools.py

Gate requirement (Steps 19.1-19.2):
  ClaimToolExecutor can execute all three tools successfully.

Tests:
  - test_create_claim_tool
  - test_update_confidence_tool
  - test_respond_to_user_tool
  - test_tool_executor_routes_correctly
"""

import pytest
import pytest_asyncio

from src.process.claim_tools import (
    CLAIM_TOOLS,
    ClaimToolExecutor,
    ToolResult,
    _reset_counter,
)
from src.data.repositories import ClaimRepository
from src.core.models import Claim, ClaimType, ClaimState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _make_repo_with_claim(
    content: str = "An existing claim for testing purposes",
    confidence: float = 0.7,
    claim_id: str = "c0000001",
) -> ClaimRepository:
    """Return an initialised repository containing one pre-existing claim."""
    repo = ClaimRepository()
    await repo.initialize()
    existing = Claim(
        id=claim_id,
        content=content,
        confidence=confidence,
    )
    await repo.create(existing)
    return repo


# ---------------------------------------------------------------------------
# CLAIM_TOOLS schema tests
# ---------------------------------------------------------------------------


class TestClaimToolsSchema:
    """Validate the CLAIM_TOOLS list structure."""

    def test_claim_tools_is_list(self):
        assert isinstance(CLAIM_TOOLS, list)

    def test_claim_tools_has_four_entries(self):
        # explore_further was added in Phase 20 (A-0012 reasoning loop)
        assert len(CLAIM_TOOLS) == 4

    def test_tool_names(self):
        names = {t["function"]["name"] for t in CLAIM_TOOLS}
        assert names == {
            "create_claim",
            "update_confidence",
            "respond_to_user",
            "explore_further",
        }

    def test_all_tools_have_function_key(self):
        for tool in CLAIM_TOOLS:
            assert tool.get("type") == "function"
            assert "function" in tool

    def test_create_claim_required_params(self):
        schema = next(
            t for t in CLAIM_TOOLS if t["function"]["name"] == "create_claim"
        )
        required = schema["function"]["parameters"]["required"]
        assert "content" in required
        assert "type" in required
        assert "confidence" in required

    def test_update_confidence_required_params(self):
        schema = next(
            t for t in CLAIM_TOOLS if t["function"]["name"] == "update_confidence"
        )
        required = schema["function"]["parameters"]["required"]
        assert "claim_id" in required
        assert "new_confidence" in required
        assert "reason" in required

    def test_respond_to_user_required_params(self):
        schema = next(
            t for t in CLAIM_TOOLS if t["function"]["name"] == "respond_to_user"
        )
        required = schema["function"]["parameters"]["required"]
        assert "response" in required


# ---------------------------------------------------------------------------
# ToolResult dataclass tests
# ---------------------------------------------------------------------------


class TestToolResult:
    def test_success_result(self):
        result = ToolResult(success=True, result="ok", claim_ids=["c0000001"])
        assert result.success is True
        assert result.result == "ok"
        assert result.claim_ids == ["c0000001"]
        assert result.error is None

    def test_failure_result(self):
        result = ToolResult(success=False, result=None, error="something broke")
        assert result.success is False
        assert result.error == "something broke"
        assert result.claim_ids == []


# ---------------------------------------------------------------------------
# test_create_claim_tool
# ---------------------------------------------------------------------------


class TestCreateClaimTool:
    """Tests for the create_claim tool handler."""

    @pytest.mark.asyncio
    async def test_create_claim_basic(self):
        """Create a minimal claim and verify it is persisted."""
        _reset_counter()
        repo = ClaimRepository()
        await repo.initialize()
        executor = ClaimToolExecutor(repo)

        result = await executor.execute_tool(
            "create_claim",
            {
                "content": "Water is composed of hydrogen and oxygen molecules",
                "type": "assertion",
                "confidence": 0.95,
            },
        )

        assert result.success is True
        assert result.error is None
        assert len(result.claim_ids) == 1

        claim_id = result.claim_ids[0]
        stored = await repo.get_by_id(claim_id)
        assert stored is not None
        assert stored.content == "Water is composed of hydrogen and oxygen molecules"
        assert stored.confidence == 0.95
        assert ClaimType.ASSERTION in stored.type

    @pytest.mark.asyncio
    async def test_create_claim_all_types(self):
        """Each accepted type string maps to the correct ClaimType enum."""
        _reset_counter()
        type_map = {
            "goal": ClaimType.GOAL,
            "assertion": ClaimType.ASSERTION,
            "observation": ClaimType.OBSERVATION,
            "assumption": ClaimType.ASSUMPTION,
            "reference": ClaimType.REFERENCE,
            "impression": ClaimType.IMPRESSION,
            "conjecture": ClaimType.CONJECTURE,
            "concept": ClaimType.CONCEPT,
            "example": ClaimType.EXAMPLE,
        }
        repo = ClaimRepository()
        await repo.initialize()
        executor = ClaimToolExecutor(repo)

        for type_str, expected_enum in type_map.items():
            result = await executor.execute_tool(
                "create_claim",
                {
                    "content": f"Claim demonstrating type {type_str} for testing purposes",
                    "type": type_str,
                    "confidence": 0.5,
                },
            )
            assert result.success, f"Failed for type={type_str}: {result.error}"
            stored = await repo.get_by_id(result.claim_ids[0])
            assert expected_enum in stored.type, (
                f"Expected {expected_enum} for type_str={type_str}, "
                f"got {stored.type}"
            )

    @pytest.mark.asyncio
    async def test_create_claim_with_super_ids(self):
        """New claim wires bidirectional relationship to existing super claim.

        We use a hard-coded super claim ID (c9990001) that is far outside
        the sequential generator range so it can never collide with the
        auto-generated ID produced by _next_claim_id().
        """
        _reset_counter()
        # Use a distinct ID that will never match what _next_claim_id() generates.
        super_claim_id = "c9990001"
        repo = await _make_repo_with_claim(
            content="Super claim that will be supported by a new sub claim",
            claim_id=super_claim_id,
        )
        executor = ClaimToolExecutor(repo)

        result = await executor.execute_tool(
            "create_claim",
            {
                "content": "Evidence that supports the super claim above",
                "type": "observation",
                "confidence": 0.8,
                "super_ids": [super_claim_id],
            },
        )

        assert result.success is True, f"create_claim failed: {result.error}"
        new_id = result.claim_ids[0]

        # New claim's supers list contains the super id.
        new_claim = await repo.get_by_id(new_id)
        assert super_claim_id in new_claim.supers

        # Super claim's subs list was updated (reverse link).
        super_claim = await repo.get_by_id(super_claim_id)
        assert new_id in super_claim.subs

    @pytest.mark.asyncio
    async def test_create_claim_with_missing_super_id(self):
        """Missing super_id is silently skipped; claim is still created."""
        _reset_counter()
        repo = ClaimRepository()
        await repo.initialize()
        executor = ClaimToolExecutor(repo)

        result = await executor.execute_tool(
            "create_claim",
            {
                "content": "Claim that references a non-existent super claim id",
                "type": "assumption",
                "confidence": 0.5,
                "super_ids": ["c9999999"],  # does not exist
            },
        )

        # Claim is still created; missing super is logged but not fatal.
        assert result.success is True
        new_claim = await repo.get_by_id(result.claim_ids[0])
        assert new_claim is not None
        assert "c9999999" in new_claim.supers  # stored in claim's supers list

    @pytest.mark.asyncio
    async def test_create_claim_confidence_boundaries(self):
        """Confidence at the 0.0 and 1.0 extremes is accepted."""
        _reset_counter()
        repo = ClaimRepository()
        await repo.initialize()
        executor = ClaimToolExecutor(repo)

        for conf in (0.0, 1.0):
            result = await executor.execute_tool(
                "create_claim",
                {
                    "content": f"Boundary confidence claim with value {conf}",
                    "type": "observation",
                    "confidence": conf,
                },
            )
            assert result.success is True
            stored = await repo.get_by_id(result.claim_ids[0])
            assert stored.confidence == conf


# ---------------------------------------------------------------------------
# test_update_confidence_tool
# ---------------------------------------------------------------------------


class TestUpdateConfidenceTool:
    """Tests for the update_confidence tool handler."""

    @pytest.mark.asyncio
    async def test_update_confidence_basic(self):
        """Update confidence on an existing claim."""
        _reset_counter()
        repo = await _make_repo_with_claim(confidence=0.5, claim_id="c0000001")
        executor = ClaimToolExecutor(repo)

        result = await executor.execute_tool(
            "update_confidence",
            {
                "claim_id": "c0000001",
                "new_confidence": 0.9,
                "reason": "Verified by independent source",
            },
        )

        assert result.success is True
        assert result.error is None
        assert "c0000001" in result.claim_ids

        stored = await repo.get_by_id("c0000001")
        assert stored.confidence == 0.9

    @pytest.mark.asyncio
    async def test_update_confidence_result_payload(self):
        """Result dict contains old_confidence, new_confidence, and reason."""
        _reset_counter()
        repo = await _make_repo_with_claim(confidence=0.4, claim_id="c0000001")
        executor = ClaimToolExecutor(repo)

        result = await executor.execute_tool(
            "update_confidence",
            {
                "claim_id": "c0000001",
                "new_confidence": 0.7,
                "reason": "Additional evidence found",
            },
        )

        assert result.success is True
        payload = result.result
        assert payload["old_confidence"] == 0.4
        assert payload["new_confidence"] == 0.7
        assert payload["reason"] == "Additional evidence found"

    @pytest.mark.asyncio
    async def test_update_confidence_claim_not_found(self):
        """Returns failure ToolResult when claim id does not exist."""
        _reset_counter()
        repo = ClaimRepository()
        await repo.initialize()
        executor = ClaimToolExecutor(repo)

        result = await executor.execute_tool(
            "update_confidence",
            {
                "claim_id": "c9999999",
                "new_confidence": 0.5,
                "reason": "Testing missing claim",
            },
        )

        assert result.success is False
        assert "c9999999" in result.error

    @pytest.mark.asyncio
    async def test_update_confidence_invalid_range(self):
        """Returns failure ToolResult when new_confidence is out of range."""
        _reset_counter()
        repo = await _make_repo_with_claim(claim_id="c0000001")
        executor = ClaimToolExecutor(repo)

        for bad_value in (-0.1, 1.1):
            result = await executor.execute_tool(
                "update_confidence",
                {
                    "claim_id": "c0000001",
                    "new_confidence": bad_value,
                    "reason": "Testing invalid range",
                },
            )
            assert result.success is False, f"Should fail for confidence={bad_value}"

    @pytest.mark.asyncio
    async def test_update_confidence_boundary_values(self):
        """Confidence values 0.0 and 1.0 are valid."""
        _reset_counter()
        repo = await _make_repo_with_claim(confidence=0.5, claim_id="c0000001")
        executor = ClaimToolExecutor(repo)

        for boundary in (0.0, 1.0):
            result = await executor.execute_tool(
                "update_confidence",
                {
                    "claim_id": "c0000001",
                    "new_confidence": boundary,
                    "reason": f"Boundary test: {boundary}",
                },
            )
            assert result.success is True, f"Failed for boundary={boundary}"


# ---------------------------------------------------------------------------
# test_respond_to_user_tool
# ---------------------------------------------------------------------------


class TestRespondToUserTool:
    """Tests for the respond_to_user tool handler."""

    @pytest.mark.asyncio
    async def test_respond_to_user_basic(self):
        """Basic response with no supporting claims."""
        _reset_counter()
        repo = ClaimRepository()
        await repo.initialize()
        executor = ClaimToolExecutor(repo)

        result = await executor.execute_tool(
            "respond_to_user",
            {
                "response": "The answer to your question is 42.",
            },
        )

        assert result.success is True
        assert result.error is None
        assert result.result["response"] == "The answer to your question is 42."
        assert result.result["supporting_claims"] == []

    @pytest.mark.asyncio
    async def test_respond_to_user_with_supporting_claims(self):
        """Supporting claims are surfaced in claim_ids and result payload."""
        _reset_counter()
        repo = ClaimRepository()
        await repo.initialize()
        executor = ClaimToolExecutor(repo)

        result = await executor.execute_tool(
            "respond_to_user",
            {
                "response": "Based on the evidence, the hypothesis is confirmed.",
                "supporting_claims": ["c0000001", "c0000002"],
            },
        )

        assert result.success is True
        assert result.result["supporting_claims"] == ["c0000001", "c0000002"]
        assert "c0000001" in result.claim_ids
        assert "c0000002" in result.claim_ids

    @pytest.mark.asyncio
    async def test_respond_to_user_empty_response(self):
        """Empty response string is accepted (validation left to the caller)."""
        _reset_counter()
        repo = ClaimRepository()
        await repo.initialize()
        executor = ClaimToolExecutor(repo)

        result = await executor.execute_tool(
            "respond_to_user",
            {"response": ""},
        )

        assert result.success is True
        assert result.result["response"] == ""

    @pytest.mark.asyncio
    async def test_respond_to_user_does_not_create_claims(self):
        """respond_to_user does not persist anything to the repository."""
        _reset_counter()
        repo = ClaimRepository()
        await repo.initialize()
        executor = ClaimToolExecutor(repo)

        await executor.execute_tool(
            "respond_to_user",
            {
                "response": "Here is my answer to your query.",
                "supporting_claims": [],
            },
        )

        count = await repo.count()
        assert count == 0


# ---------------------------------------------------------------------------
# test_tool_executor_routes_correctly
# ---------------------------------------------------------------------------


class TestToolExecutorRouting:
    """Tests that execute_tool dispatches to the correct handler."""

    @pytest.mark.asyncio
    async def test_routes_create_claim(self):
        """execute_tool('create_claim', ...) returns a Claim in result."""
        _reset_counter()
        repo = ClaimRepository()
        await repo.initialize()
        executor = ClaimToolExecutor(repo)

        result = await executor.execute_tool(
            "create_claim",
            {
                "content": "Routing test claim content for create operation",
                "type": "observation",
                "confidence": 0.6,
            },
        )

        assert result.success is True
        from src.core.models import Claim
        assert isinstance(result.result, Claim)

    @pytest.mark.asyncio
    async def test_routes_update_confidence(self):
        """execute_tool('update_confidence', ...) returns a dict with claim."""
        _reset_counter()
        repo = await _make_repo_with_claim(claim_id="c0000001")
        executor = ClaimToolExecutor(repo)

        result = await executor.execute_tool(
            "update_confidence",
            {
                "claim_id": "c0000001",
                "new_confidence": 0.8,
                "reason": "Routing verification test",
            },
        )

        assert result.success is True
        assert isinstance(result.result, dict)
        assert "claim" in result.result

    @pytest.mark.asyncio
    async def test_routes_respond_to_user(self):
        """execute_tool('respond_to_user', ...) returns a dict with response key."""
        _reset_counter()
        repo = ClaimRepository()
        await repo.initialize()
        executor = ClaimToolExecutor(repo)

        result = await executor.execute_tool(
            "respond_to_user",
            {"response": "Routing test response for verification"},
        )

        assert result.success is True
        assert isinstance(result.result, dict)
        assert "response" in result.result

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_failure(self):
        """Unknown tool name returns a failure ToolResult without raising."""
        _reset_counter()
        repo = ClaimRepository()
        await repo.initialize()
        executor = ClaimToolExecutor(repo)

        result = await executor.execute_tool(
            "nonexistent_tool",
            {"foo": "bar"},
        )

        assert result.success is False
        assert result.error is not None
        assert "nonexistent_tool" in result.error

    @pytest.mark.asyncio
    async def test_all_three_tools_succeed_in_sequence(self):
        """Gate test: all three tools execute successfully end-to-end."""
        _reset_counter()
        repo = ClaimRepository()
        await repo.initialize()
        executor = ClaimToolExecutor(repo)

        # Step 1 — create a claim
        create_result = await executor.execute_tool(
            "create_claim",
            {
                "content": "Integration test: reasoning step one observation",
                "type": "observation",
                "confidence": 0.7,
            },
        )
        assert create_result.success, f"create_claim failed: {create_result.error}"

        claim_id = create_result.claim_ids[0]

        # Step 2 — update its confidence
        update_result = await executor.execute_tool(
            "update_confidence",
            {
                "claim_id": claim_id,
                "new_confidence": 0.9,
                "reason": "Confirmed by further analysis",
            },
        )
        assert update_result.success, f"update_confidence failed: {update_result.error}"

        # Verify the update was persisted.
        stored = await repo.get_by_id(claim_id)
        assert stored.confidence == 0.9

        # Step 3 — deliver final response citing the claim
        respond_result = await executor.execute_tool(
            "respond_to_user",
            {
                "response": "Based on the analysis, the observation is confirmed.",
                "supporting_claims": [claim_id],
            },
        )
        assert respond_result.success, f"respond_to_user failed: {respond_result.error}"
        assert respond_result.result["response"].startswith("Based on")
        assert claim_id in respond_result.result["supporting_claims"]
