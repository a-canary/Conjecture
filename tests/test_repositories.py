"""
Tests for Data Layer Repositories

Tests the ClaimRepository, RepositoryFactory, and get_data_manager
for claim storage and retrieval operations.
"""

import pytest
import pytest_asyncio
from datetime import datetime, timezone

from src.data.repositories import ClaimRepository, RepositoryFactory, get_data_manager
from src.core.models import Claim, ClaimState, ClaimType, DirtyReason


class TestClaimRepositoryInit:
    """Tests for ClaimRepository initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default config."""
        repo = ClaimRepository()

        assert repo._storage is None
        assert repo._claims == {}
        assert repo._initialized is False

    def test_init_with_storage(self):
        """Test initialization with custom storage."""
        mock_storage = {"type": "sqlite"}
        repo = ClaimRepository(storage=mock_storage)

        assert repo._storage == mock_storage

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test repository initialization."""
        repo = ClaimRepository()

        assert repo._initialized is False

        await repo.initialize()

        assert repo._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        """Test that initialize is idempotent."""
        repo = ClaimRepository()

        await repo.initialize()
        await repo.initialize()  # Should not fail

        assert repo._initialized is True


class TestClaimRepositoryCRUD:
    """Tests for ClaimRepository CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_claim(self):
        """Test creating a claim."""
        repo = ClaimRepository()
        claim = Claim(
            id="c001",
            content="Test claim content here",
            confidence=0.8,
        )

        result = await repo.create(claim)

        assert result.id == "c001"
        assert result.content == "Test claim content here"

    @pytest.mark.asyncio
    async def test_create_duplicate_fails(self):
        """Test creating duplicate claim fails."""
        repo = ClaimRepository()
        claim = Claim(id="c001", content="Test claim content", confidence=0.8)

        await repo.create(claim)

        with pytest.raises(ValueError, match="already exists"):
            await repo.create(claim)

    @pytest.mark.asyncio
    async def test_get_by_id_existing(self):
        """Test getting existing claim by ID."""
        repo = ClaimRepository()
        claim = Claim(id="c001", content="Test claim content", confidence=0.8)
        await repo.create(claim)

        result = await repo.get_by_id("c001")

        assert result is not None
        assert result.id == "c001"

    @pytest.mark.asyncio
    async def test_get_by_id_nonexistent(self):
        """Test getting non-existent claim returns None."""
        repo = ClaimRepository()

        result = await repo.get_by_id("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_update_existing(self):
        """Test updating existing claim."""
        repo = ClaimRepository()
        claim = Claim(id="c001", content="Original claim content", confidence=0.8)
        await repo.create(claim)

        claim.content = "Updated claim content"
        result = await repo.update(claim)

        assert result.content == "Updated claim content"
        assert result.updated is not None

    @pytest.mark.asyncio
    async def test_update_nonexistent_fails(self):
        """Test updating non-existent claim fails."""
        repo = ClaimRepository()
        claim = Claim(id="c001", content="Test claim content", confidence=0.8)

        with pytest.raises(ValueError, match="not found"):
            await repo.update(claim)

    @pytest.mark.asyncio
    async def test_delete_existing(self):
        """Test deleting existing claim."""
        repo = ClaimRepository()
        claim = Claim(id="c001", content="Test claim content", confidence=0.8)
        await repo.create(claim)

        result = await repo.delete("c001")

        assert result is True
        assert await repo.get_by_id("c001") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self):
        """Test deleting non-existent claim returns False."""
        repo = ClaimRepository()

        result = await repo.delete("nonexistent")

        assert result is False


class TestClaimRepositoryQueries:
    """Tests for ClaimRepository query operations."""

    @pytest.mark.asyncio
    async def test_count_empty(self):
        """Test count with empty repository."""
        repo = ClaimRepository()

        result = await repo.count()

        assert result == 0

    @pytest.mark.asyncio
    async def test_count_with_claims(self):
        """Test count with claims."""
        repo = ClaimRepository()
        await repo.create(Claim(id="c001", content="Claim A content", confidence=0.8))
        await repo.create(Claim(id="c002", content="Claim B content", confidence=0.7))

        result = await repo.count()

        assert result == 2

    @pytest.mark.asyncio
    async def test_list_all_empty(self):
        """Test list_all with empty repository."""
        repo = ClaimRepository()

        result = await repo.list_all()

        assert result == []

    @pytest.mark.asyncio
    async def test_list_all_with_claims(self):
        """Test list_all with claims."""
        repo = ClaimRepository()
        await repo.create(Claim(id="c001", content="Claim A content", confidence=0.8))
        await repo.create(Claim(id="c002", content="Claim B content", confidence=0.7))

        result = await repo.list_all()

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_all_respects_limit(self):
        """Test list_all respects limit."""
        repo = ClaimRepository()
        for i in range(10):
            await repo.create(Claim(id=f"c{i:03d}", content=f"Claim content {i}", confidence=0.8))

        result = await repo.list_all(limit=5)

        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_query_by_state(self):
        """Test query by claim state."""
        repo = ClaimRepository()
        await repo.create(Claim(id="c001", content="Claim A content", confidence=0.8, state=ClaimState.EXPLORE))
        await repo.create(Claim(id="c002", content="Claim B content", confidence=0.8, state=ClaimState.VALIDATED))
        await repo.create(Claim(id="c003", content="Claim C content", confidence=0.8, state=ClaimState.EXPLORE))

        result = await repo.query_by_state(ClaimState.EXPLORE)

        assert len(result) == 2
        assert all(c.state == ClaimState.EXPLORE for c in result)

    @pytest.mark.asyncio
    async def test_query_by_state_respects_limit(self):
        """Test query by state respects limit."""
        repo = ClaimRepository()
        for i in range(10):
            await repo.create(Claim(
                id=f"c{i:03d}",
                content=f"Claim content {i}",
                confidence=0.8,
                state=ClaimState.EXPLORE
            ))

        result = await repo.query_by_state(ClaimState.EXPLORE, limit=3)

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_query_by_tag(self):
        """Test query by tag."""
        repo = ClaimRepository()
        await repo.create(Claim(id="c001", content="Claim A content", confidence=0.8, tags=["math", "algebra"]))
        await repo.create(Claim(id="c002", content="Claim B content", confidence=0.8, tags=["science"]))
        await repo.create(Claim(id="c003", content="Claim C content", confidence=0.8, tags=["math", "geometry"]))

        result = await repo.query_by_tag("math")

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_query_by_tag_case_insensitive(self):
        """Test query by tag is case insensitive."""
        repo = ClaimRepository()
        await repo.create(Claim(id="c001", content="Claim A content", confidence=0.8, tags=["Math"]))
        await repo.create(Claim(id="c002", content="Claim B content", confidence=0.8, tags=["MATH"]))

        result = await repo.query_by_tag("math")

        assert len(result) == 2


class TestClaimRepositoryDirtyOperations:
    """Tests for ClaimRepository dirty flag operations."""

    @pytest.mark.asyncio
    async def test_mark_dirty(self):
        """Test marking a claim as dirty."""
        repo = ClaimRepository()
        claim = Claim(id="c001", content="Test claim content", confidence=0.8)
        await repo.create(claim)

        result = await repo.mark_dirty("c001", DirtyReason.CONTENT_UPDATE, priority=5)

        assert result is not None
        assert result.is_dirty is True
        assert result.dirty_reason == DirtyReason.CONTENT_UPDATE
        assert result.dirty_priority == 5

    @pytest.mark.asyncio
    async def test_mark_dirty_nonexistent(self):
        """Test marking non-existent claim returns None."""
        repo = ClaimRepository()

        result = await repo.mark_dirty("nonexistent", DirtyReason.MANUAL_FLAG)

        assert result is None

    @pytest.mark.asyncio
    async def test_mark_clean(self):
        """Test marking a claim as clean."""
        repo = ClaimRepository()
        claim = Claim(id="c001", content="Test claim content", confidence=0.8)
        await repo.create(claim)
        await repo.mark_dirty("c001", DirtyReason.CONTENT_UPDATE)

        result = await repo.mark_clean("c001")

        assert result is not None
        assert result.is_dirty is False
        assert result.dirty_reason is None

    @pytest.mark.asyncio
    async def test_mark_clean_nonexistent(self):
        """Test marking non-existent claim clean returns None."""
        repo = ClaimRepository()

        result = await repo.mark_clean("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_dirty_claims(self):
        """Test getting dirty claims."""
        repo = ClaimRepository()

        # Create claims - explicitly set is_dirty=False to start clean
        c1 = Claim(id="c001", content="Claim A content", confidence=0.8, is_dirty=False, dirty=False)
        c2 = Claim(id="c002", content="Claim B content", confidence=0.8, is_dirty=False, dirty=False)
        c3 = Claim(id="c003", content="Claim C content", confidence=0.8, is_dirty=False, dirty=False)
        await repo.create(c1)
        await repo.create(c2)
        await repo.create(c3)

        # Mark some dirty
        await repo.mark_dirty("c001", DirtyReason.CONTENT_UPDATE, priority=3)
        await repo.mark_dirty("c003", DirtyReason.MANUAL_FLAG, priority=5)

        result = await repo.get_dirty_claims()

        assert len(result) == 2
        # Sorted by priority descending
        assert result[0].id == "c003"  # Higher priority
        assert result[1].id == "c001"

    @pytest.mark.asyncio
    async def test_get_dirty_claims_priority_filter(self):
        """Test getting dirty claims with priority filter."""
        repo = ClaimRepository()

        c1 = Claim(id="c001", content="Claim A content", confidence=0.8)
        c2 = Claim(id="c002", content="Claim B content", confidence=0.8)
        await repo.create(c1)
        await repo.create(c2)

        await repo.mark_dirty("c001", DirtyReason.CONTENT_UPDATE, priority=3)
        await repo.mark_dirty("c002", DirtyReason.MANUAL_FLAG, priority=7)

        result = await repo.get_dirty_claims(priority_min=5)

        assert len(result) == 1
        assert result[0].id == "c002"

    @pytest.mark.asyncio
    async def test_get_dirty_claims_respects_limit(self):
        """Test getting dirty claims respects limit."""
        repo = ClaimRepository()

        for i in range(10):
            claim = Claim(id=f"c{i:03d}", content=f"Claim content {i}", confidence=0.8)
            await repo.create(claim)
            await repo.mark_dirty(claim.id, DirtyReason.MANUAL_FLAG, priority=i)

        result = await repo.get_dirty_claims(limit=3)

        assert len(result) == 3


class TestClaimRepositoryRelations:
    """Tests for ClaimRepository relation operations."""

    @pytest.mark.asyncio
    async def test_get_related_no_relations(self):
        """Test get_related with no relations."""
        repo = ClaimRepository()
        claim = Claim(id="c001", content="Test claim content", confidence=0.8)
        await repo.create(claim)

        result = await repo.get_related("c001")

        assert result == {"supers": [], "subs": []}

    @pytest.mark.asyncio
    async def test_get_related_nonexistent(self):
        """Test get_related with non-existent claim."""
        repo = ClaimRepository()

        result = await repo.get_related("nonexistent")

        assert result == {"supers": [], "subs": []}

    @pytest.mark.asyncio
    async def test_get_related_with_supers(self):
        """Test get_related with super claims."""
        repo = ClaimRepository()

        parent = Claim(id="c001", content="Parent claim content", confidence=0.9)
        child = Claim(id="c002", content="Child claim content", confidence=0.8, supers=["c001"])
        await repo.create(parent)
        await repo.create(child)

        result = await repo.get_related("c002")

        assert len(result["supers"]) == 1
        assert result["supers"][0].id == "c001"

    @pytest.mark.asyncio
    async def test_get_related_with_subs(self):
        """Test get_related with sub claims."""
        repo = ClaimRepository()

        parent = Claim(id="c001", content="Parent claim content", confidence=0.9, subs=["c002"])
        child = Claim(id="c002", content="Child claim content", confidence=0.8)
        await repo.create(parent)
        await repo.create(child)

        result = await repo.get_related("c001")

        assert len(result["subs"]) == 1
        assert result["subs"][0].id == "c002"


class TestRepositoryFactory:
    """Tests for RepositoryFactory."""

    def test_get_claim_repository_creates_instance(self):
        """Test factory creates repository instance."""
        RepositoryFactory.reset()  # Ensure clean state

        repo = RepositoryFactory.get_claim_repository()

        assert repo is not None
        assert isinstance(repo, ClaimRepository)

    def test_get_claim_repository_singleton(self):
        """Test factory returns same instance."""
        RepositoryFactory.reset()

        repo1 = RepositoryFactory.get_claim_repository()
        repo2 = RepositoryFactory.get_claim_repository()

        assert repo1 is repo2

    def test_reset_clears_instances(self):
        """Test reset clears cached instances."""
        repo1 = RepositoryFactory.get_claim_repository()

        RepositoryFactory.reset()

        repo2 = RepositoryFactory.get_claim_repository()
        assert repo1 is not repo2


class TestGetDataManager:
    """Tests for get_data_manager function."""

    def test_get_data_manager_returns_repo(self):
        """Test get_data_manager returns repository."""
        RepositoryFactory.reset()

        manager = get_data_manager()

        assert manager is not None

    def test_get_data_manager_caches(self):
        """Test get_data_manager caches result."""
        RepositoryFactory.reset()

        manager1 = get_data_manager(use_cache=True)
        manager2 = get_data_manager(use_cache=True)

        assert manager1 is manager2

    def test_get_data_manager_no_cache(self):
        """Test get_data_manager without cache."""
        RepositoryFactory.reset()

        # First call with cache creates the cache
        _ = get_data_manager(use_cache=True)

        # Second call without cache still returns cached if _data_manager is set
        manager = get_data_manager(use_cache=False)

        assert manager is not None
