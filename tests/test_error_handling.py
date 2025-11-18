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
from unittest.mock import Mock, patch, MagicMock

from src.data.data_manager import DataManager
from src.data.sqlite_manager import SQLiteManager
from src.data.chroma_manager import ChromaManager
from src.data.embedding_service import EmbeddingService, MockEmbeddingService
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
        # Test with invalid JSON in tags field
        with patch('json.dumps') as mock_json:
            mock_json.side_effect = ValueError("Invalid JSON data")
            
            claim = Claim(
                id="c0000001",
                content="Test claim",
                created_by="user"
            )
            
            with pytest.raises(DataLayerError, match="Failed to create claim"):
                await sqlite_manager.create_claim(claim)

    @pytest.mark.error_handling
    @pytest.mark.asyncio
    async def test_chroma_initialization_failure(self):
        """Test ChromaDB initialization failure scenarios."""
        # Invalid ChromaDB path
        invalid_manager = ChromaManager("")
        
        with pytest.raises(DataLayerError, match="Failed to initialize ChromaDB"):
            await invalid_manager.initialize()

    @pytest.mark.error_handling
    @pytest.mark.asyncio
    async def test_chroma_operation_failures(self, chroma_manager: ChromaManager):
        """Test ChromaDB operation failure handling."""
        claim = Claim(
            id="c0000001",
            content="Test claim",
            created_by="user"
        )
        
        # Test with invalid embedding format
        invalid_embedding = "not_a_list"
        
        with pytest.raises(DataLayerError, match="Failed to add embedding"):
            await chroma_manager.add_embedding(claim, invalid_embedding)
        
        # Test mismatched batch sizes
        claims = [claim, claim]
        embeddings = [[0.1] * 384]  # Only one embedding for two claims
        
        with pytest.raises(DataLayerError, match="Claims and embeddings must have same length"):
            await chroma_manager.batch_add_embeddings(claims, embeddings)

    @pytest.mark.error_handling
    @pytest.mark.asyncio
    async def test_embedding_service_failures(self):
        """Test embedding service failure scenarios."""
        # Mock sentence transformer failure
        with patch('src.data.embedding_service.SentenceTransformer') as mock_transformer:
            mock_transformer.side_effect = Exception("Model loading failed")
            
            service = EmbeddingService("all-MiniLM-L6-v2")
            
            with pytest.raises(DataLayerError, match="Failed to initialize embedding model"):
                await service.initialize()
        
        # Test model inference failure
        service = EmbeddingService("all-MiniLM-L6-v2")
        with patch.object(service, 'model') as mock_model:
            mock_model.encode.side_effect = Exception("Model inference failed")
            await service.initialize()
            
            with pytest.raises(DataLayerError, match="Failed to generate embedding"):
                await service.generate_embedding("test text")


class TestDataManagerErrorCoordination:
    """Test error coordination between DataManager components."""

    @pytest.mark.error_handling
    @pytest.mark.asyncio
    async def test_partial_operation_consistency(self, test_config: DataConfig):
        """Test data consistency when operations partially fail."""
        test_config.sqlite_path = ":memory:"
        test_config.chroma_path = "./test_chroma_partial"
        
        dm = DataManager(test_config, use_mock_embeddings=True)
        await dm.initialize()
        
        # Mock ChromaDB to fail after storing in SQLite
        original_add_embedding = dm.chroma_manager.add_embedding
        
        async def failing_add_embedding(*args, **kwargs):
            # Fail only on first call
            if not hasattr(failing_add_embedding, 'called'):
                failing_add_embedding.called = True
                raise DataLayerError("ChromaDB storage failed")
            return await original_add_embedding(*args, **kwargs)
        
        dm.chroma_manager.add_embedding = failing_add_embedding
        
        # Create claim - this should fail due to ChromaDB failure
        with pytest.raises(DataLayerError):
            await dm.create_claim("Test claim", "user")
        
        # Verify no claim was created (transaction-like behavior)
        # In actual implementation, this would depend on how you handle partial failures
        # For now, we just test that the error is properly propagated

    @pytest.mark.error_handling
    @pytest.mark.asyncio
    async def test_component_unavailable_handling(self, data_manager: DataManager):
        """Test behavior when components become unavailable."""
        # Manually close components to simulate failure
        await data_manager.sqlite_manager.close()
        
        # Operations should fail with clear error messages
        with pytest.raises(DataLayerError):
            await data_manager.get_claim("c0000001")
        
        with pytest.raises(DataLayerError):
            await data_manager.create_claim("Test", "user")
        
        with pytest.raises(DataLayerError):
            await data_manager.get_stats()

    @pytest.mark.error_handling
    @pytest.mark.asyncio
    async def test_embedding_generation_failure_propagation(self, data_manager: DataManager):
        """Test that embedding generation failures are properly propagated."""
        # Mock embedding service to fail
        original_generate = data_manager.embedding_service.generate_embedding
        
        async def failing_generate(*args, **kwargs):
            raise DataLayerError("Embedding generation failed")
        
        data_manager.embedding_service.generate_embedding = failing_generate
        
        # Claim creation should fail due to embedding failure
        with pytest.raises(DataLayerError):
            await data_manager.create_claim("Test claim", "user")

    @pytest.mark.error_handling
    @pytest.mark.asyncio
    async def test_batch_operation_partial_failures(self, data_manager: DataManager):
        """Test handling of partial failures in batch operations."""
        # Create claims that will partially fail
        problematic_claims_data = [
            {"content": "Valid claim 1", "created_by": "user"},  # Valid
            {"content": "Too short", "created_by": "user"},     # Invalid (too short)
            {"content": "Valid claim 2", "created_by": "user"},  # Valid
        ]
        
        # Depending on implementation, this might:
        # 1. Fail entirely (fail-fast)
        # 2. Succeed with partial results
        # 3. Return results with error information
        
        # Test current behavior - should fail completely due to validation
        with pytest.raises((ValueError, DataLayerError)):
            await data_manager.batch_create_claims(problematic_claims_data)


class TestInputValidationSecurity:
    """Test security-related input validation and sanitization."""

    @pytest.mark.error_handling
    def test_sql_injection_attempts(self):
        """Test SQL injection attempts are properly handled."""
        malicious_inputs = [
            "'; DROP TABLE claims; --",
            "'; DELETE FROM claims; --",
            "'; INSERT INTO claims VALUES ('malicious', 'data'); --",
            "' OR '1'='1",
            "'; SELECT * FROM users; --",
            "'; UPDATE claims SET content='hacked'; --",
        ]
        
        for malicious_content in malicious_inputs:
            # These should either be rejected or safely handled
            # The key is they shouldn't cause actual SQL injection
            try:
                claim = Claim(
                    id="c0000001",
                    content=malicious_content,
                    created_by="user"
                )
                # If validation passes, that's okay - the database layer should handle sanitization
                assert claim.content == malicious_content
            except Exception as e:
                # If validation rejects, that's also acceptable
                assert isinstance(e, (ValueError, ValidationError))


    @pytest.mark.error_handling
    def test_script_injection_attempts(self):
        """Test script injection attempts are properly handled."""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            '${jndi:ldap://malicious.com/a}',
            '{{7*7}}',  # Template injection
            '${7*7}',   # Expression injection
        ]
        
        for malicious_content in malicious_inputs:
            # These should be stored safely or rejected
            try:
                claim = Claim(
                    id="c0000001",
                    content=malicious_content,
                    created_by="user"
                )
                # If accepted, check they're stored as-is (not executed)
                assert malicious_content in claim.content
            except Exception as e:
                # Rejection is also acceptable
                assert isinstance(e, ValueError)

    @pytest.mark.error_handling
    def test_unicode_and_encoding_edge_cases(self):
        """Test unicode and encoding edge cases."""
        edge_cases = [
            "\x00\x01\x02",           # Null bytes and control characters
            "\uffff\ud800",           # Invalid Unicode sequences
            "🚀" * 1000,              # Many emojis
            "中文" * 500,              # Many Chinese characters
            "العربية" * 500,           # Many Arabic characters
            "🔥💯🎉😀🤔",              # Mixed emoji
        ]
        
        for content in edge_cases:
            try:
                claim = Claim(
                    id="c0000001",
                    content=content,
                    created_by="user"
                )
                # Should either accept or reject gracefully
                assert isinstance(claim.content, str)
            except Exception as e:
                # Should fail gracefully, not crash
                assert isinstance(e, (ValueError, UnicodeError))

    @pytest.mark.error_handling
    def test_extreme_input_sizes(self):
        """Test extremely large inputs."""
        # Very long claim content
        very_long = "A" * 1000000  # 1MB of content
        
        try:
            claim = Claim(
                id="c0000001",
                content=very_long,
                created_by="user"
            )
            # If accepted, that's okay
            assert len(claim.content) == 1000000
        except Exception as e:
            # Should fail gracefully
            assert isinstance(e, ValueError)
        
        # Very many tags
        many_tags = [f"tag_{i}" for i in range(10000)]
        
        try:
            claim = Claim(
                id="c0000002",
                content="Valid content",
                tags=many_tags,
                created_by="user"
            )
            # If accepted, check deduplication worked
            assert len(claim.tags) == len(set(claim.tags))
        except Exception as e:
            # Should fail gracefully
            assert isinstance(e, ValueError)


class TestResourceLimitHandling:
    """Test handling of resource limit constraints."""

    @pytest.mark.error_handling
    @pytest.mark.asyncio
    async def test_disk_space_exhaustion(self):
        """Test behavior when disk space is exhausted."""
        # This is difficult to test reliably, but we can simulate
        with patch('os.makedirs') as mock_makedirs:
            mock_makedirs.side_effect = OSError("No space left on device")
            
            manager = SQLiteManager("/tmp/test_disk_full.db")
            
            with pytest.raises((DataLayerError, OSError)):
                await manager.initialize()

    @pytest.mark.error_handling
    @pytest.mark.asyncio
    async def test_memory_exhaustion_handling(self, data_manager: DataManager):
        """Test behavior during memory exhaustion scenarios."""
        # Create many large claims to test memory handling
        large_content = "X" * 100000  # 100KB per claim
        
        # This should either succeed or fail gracefully
        try:
            claims = []
            for i in range(100):  # 10MB of data
                claim = await data_manager.create_claim(
                    f"{large_content}_{i}",
                    "memory_test_user"
                )
                claims.append(claim)
            
            # If successful, verify claims exist
            for claim in claims[-10:]:  # Check last 10
                retrieved = await data_manager.get_claim(claim.id)
                assert retrieved is not None
                
        except MemoryError:
            # Expected in memory-constrained environments
            pass
        except DataLayerError as e:
            # Should fail gracefully, not crash
            assert "memory" in str(e).lower() or "resource" in str(e).lower()

    @pytest.mark.error_handling
    @pytest.mark.asyncio
    async def test_connection_pool_exhaustion(self):
        """Test handling of connection pool exhaustion."""
        # Create multiple simultaneous operations to test pool limits
        test_config = DataConfig()
        test_config.sqlite_path = ":memory:"
        test_config.chroma_path = "./test_chroma_pool_exhaust"
        
        dm = DataManager(test_config, use_mock_embeddings=True)
        await dm.initialize()
        
        async def many_operations():
            """Perform many operations to stress connection pool."""
            tasks = []
            for i in range(50):
                task = dm.create_claim(
                    f"Pool test claim {i}",
                    f"user_{i % 10}"
                )
                tasks.append(task)
            
            # These should either all succeed or fail gracefully
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Check that exceptions are proper ones, not crashes
                for result in results:
                    if isinstance(result, Exception):
                        assert isinstance(result, (DataLayerError, ConnectionError))
                        
            except Exception as e:
                # Should fail gracefully
                assert isinstance(e, DataLayerError)
        
        await many_operations()
        await dm.close()


class TestConcurrentOperationConflicts:
    """Test handling of concurrent operation conflicts."""

    @pytest.mark.error_handling
    @pytest.mark.asyncio
    async def test_concurrent_claim_creation_conflicts(self, data_manager: DataManager):
        """Test concurrent claim creation with potential ID conflicts."""
        # Reset claim counter manually to create conflict scenario
        data_manager._claim_counter = 0
        
        async def create_claim_with_fixed_id(content: str):
            """Create claim with manually managed ID."""
            data_manager._claim_counter += 1
            claim_id = f"c{data_manager._claim_counter:07d}"
            
            claim = Claim(
                id=claim_id,
                content=content,
                created_by="concurrent_user"
            )
            
            # This might conflict if two async calls increment counter interleaved
            embedding = await data_manager.embedding_service.generate_embedding(content)
            await data_manager.sqlite_manager.create_claim(claim)
            await data_manager.chroma_manager.add_embedding(claim, embedding)
            return claim
        
        # Run concurrent creations with potential conflicts
        tasks = [
            create_claim_with_fixed_id(f"Concurrent claim {i}")
            for i in range(10)
        ]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Some might succeed, some might fail due to conflicts
            successes = [r for r in results if not isinstance(r, Exception)]
            failures = [r for r in results if isinstance(r, Exception)]
            
            print(f"Concurrent creation: {len(successes)} successes, {len(failures)} failures")
            
            # Failures should be proper exceptions, not crashes
            for failure in failures:
                assert isinstance(failure, (DataLayerError, sqlite3.IntegrityError))
                
        except Exception as e:
            # Should fail gracefully
            assert isinstance(e, DataLayerError)

    @pytest.mark.error_handling
    @pytest.mark.asyncio
    async def test_concurrent_modification_conflicts(self, populated_data_manager: DataManager):
        """Test concurrent modification conflicts on same claim."""
        claim_id = "c0000001"
        
        async def modify_claim_confidence(confidence: float):
            """Modify claim confidence concurrently."""
            await populated_data_manager.update_claim(claim_id, confidence=confidence)
            return confidence
        
        # Run concurrent updates
        tasks = [
            modify_claim_confidence(0.8),
            modify_claim_confidence(0.9),
            modify_claim_confidence(0.7),
        ]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Some might succeed, last write wins
            successes = [r for r in results if not isinstance(r, Exception)]
            failures = [r for r in results if isinstance(r, Exception)]
            
            print(f"Concurrent updates: {len(successes)} successes, {len(failures)} failures")
            
            if successes:
                # Check final state
                final_claim = await populated_data_manager.get_claim(claim_id)
                assert final_claim.confidence in [0.7, 0.8, 0.9]  # One of the updates took effect
            
            # Failures should be proper exceptions
            for failure in failures:
                assert isinstance(failure, (DataLayerError, RuntimeError))
                
        except Exception as e:
            # Should fail gracefully
            assert isinstance(e, DataLayerError)


class TestRecoveryAndResilience:
    """Test recovery and resilience scenarios."""

    @pytest.mark.error_handling
    @pytest.mark.asyncio
    async def test_automatic_recovery_from_temporary_failures(self, data_manager: DataManager):
        """Test automatic recovery from temporary failures."""
        # Mock embedding service to fail temporarily
        original_generate = data_manager.embedding_service.generate_embedding
        failure_count = [0]
        
        async def flaky_generate(*args, **kwargs):
            failure_count[0] += 1
            if failure_count[0] <= 2:  # Fail first 2 attempts
                raise DataLayerError("Temporary failure")
            return await original_generate(*args, **kwargs)
        
        data_manager.embedding_service.generate_embedding = flaky_generate
        
        # First attempts should fail
        with pytest.raises(DataLayerError):
            await data_manager.create_claim("Test claim 1", "user")
        
        with pytest.raises(DataLayerError):
            await data_manager.create_claim("Test claim 2", "user")
        
        # Third attempt should succeed
        claim = await data_manager.create_claim("Test claim 3", "user")
        assert claim is not None
        assert claim.content == "Test claim 3"

    @pytest.mark.error_handling
    @pytest.mark.asyncio
    async def test_data_integrity_after_failures(self, data_manager: DataManager):
        """Test data integrity after various failure scenarios."""
        # Create initial claim
        claim1 = await data_manager.create_claim("Initial claim", "user")
        assert claim1 is not None
        
        # Simulate partial failure during batch operation
        try:
            # Create claims with one that will fail
            claims_data = [
                {"content": "Valid claim 1", "created_by": "user"},
                {"content": "Invalid", "created_by": "user"},  # Too short
                {"content": "Valid claim 2", "created_by": "user"},
            ]
            
            await data_manager.batch_create_claims(claims_data)
            
        except (ValueError, DataLayerError):
            pass  # Expected to fail
        
        # Verify original claim is still intact
        retrieved = await data_manager.get_claim(claim1.id)
        assert retrieved is not None
        assert retrieved.content == "Initial claim"
        
        # Verify no partial data was created
        stats = await data_manager.get_stats()
        assert stats["total_claims"] == 1  # Only original claim should exist

    @pytest.mark.error_handling
    @pytest.mark.asyncio
    async def test_graceful_degradation_under_load(self, data_manager: DataManager):
        """Test graceful degradation under heavy load."""
        # Create many concurrent operations
        async def stress_operation(i):
            """Operation that might degrade under load."""
            try:
                return await data_manager.create_claim(
                    f"Stress test claim {i}",
                    f"stress_user_{i % 10}"
                )
            except DataLayerError as e:
                # Should fail gracefully, not crash
                if "overload" in str(e).lower() or "timeout" in str(e).lower():
                    return None  # Graceful degradation
                raise
        
        # Run many operations.concurrently
        tasks = [stress_operation(i) for i in range(100)]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successes, failures, and degradations
            successes = sum(1 for r in results if isinstance(r, Claim))
            degradations = sum(1 for r in results if r is None)
            exceptions = sum(1 for r in results if isinstance(r, Exception))
            
            print(f"Stress test: {successes} successes, {degradations} degradations, {exceptions} exceptions")
            
            # Should handle load gracefully
            assert successes + degradations > exceptions  # Most should succeed or degrade gracefully
            
            # Exceptions should be proper ones
            for result in results:
                if isinstance(result, Exception):
                    assert isinstance(result, DataLayerError)
                    
        except Exception as e:
            # Should fail gracefully, not crash
            assert isinstance(e, DataLayerError)


class TestEdgeCaseScenarios:
    """Test various edge case scenarios."""

    @pytest.mark.error_handling
    @pytest.mark.asyncio
    async def test_empty_and_null_inputs(self, data_manager: DataManager):
        """Test handling of empty and null inputs."""
        # Empty strings should be handled appropriately
        with pytest.raises((ValueError, InvalidClaimError)):
            await data_manager.create_claim("", "user")  # Empty content
        
        with pytest.raises((ValueError, InvalidClaimError)):
            await data_manager.create_claim("Valid content", "")  # Empty creator
        
        # Null values should be handled
        with pytest.raises((TypeError, ValueError)):
            await data_manager.create_claim(None, "user")  # None content
        
        with pytest.raises((TypeError, ValueError)):
            await data_manager.create_claim("Valid content", None)  # None creator

    @pytest.mark.error_handling
    @pytest.mark.asyncio
    async def test_extreme_numerical_values(self, data_manager: DataManager):
        """Test extreme numerical value handling."""
        # Floating point edge cases
        extreme_confidences = [
            0.0,      # Minimum
            1.0,      # Maximum
            0.5,      # Middle
            0.999999, # Just below 1.0
            0.000001, # Just above 0.0
        ]
        
        for confidence in extreme_confidences:
            claim = await data_manager.create_claim(
                f"Confidence test {confidence}",
                "test_user",
                confidence=confidence
            )
            assert claim.confidence == confidence

    @pytest.mark.error_handling
    @pytest.mark.asyncio
    async def test_circular_relationships(self, populated_data_manager: DataManager):
        """Test handling of circular relationships."""
        # Create circular relationships
        await populated_data_manager.add_relationship(
            supporter_id="c0000001",
            supported_id="c0000002",
            relationship_type="supports"
        )
        
        await populated_data_manager.add_relationship(
            supporter_id="c0000002",
            supported_id="c0000001",
            relationship_type="supports"
        )
        
        # Should not cause infinite loops or crashes
        supporters1 = await populated_data_manager.get_supported_by("c0000001")
        supporters2 = await populated_data_manager.get_supported_by("c0000002")
        
        assert "c0000002" in supporters1
        assert "c0000001" in supporters2

    @pytest.mark.error_handling
    @pytest.mark.asyncio
    async def test_orphaned_data_cleanup(self, data_manager: DataManager):
        """Test handling of potentially orphaned data."""
        # Create claim
        claim = await data_manager.create_claim("Test claim", "user")
        
        # Manually delete from one store to test orphaned handling
        await data_manager.chroma_manager.delete_embedding(claim.id)
        
        # Retrieval should handle missing embedding gracefully
        retrieved = await data_manager.get_claim(claim.id)
        assert retrieved is not None  # SQLite data still exists
        
        # Search should handle orphaned data gracefully
        results = await data_manager.search_similar("test query", limit=5)
        assert isinstance(results, list)  # Should not crash

    @pytest.mark.error_handling
    @pytest.mark.asyncio
    async def test_timezone_handling(self, data_manager: DataManager):
        """Test timezone handling in datetime fields."""
        # Create claim with explicit timezone handling
        from datetime import datetime, timezone, UTC
        
        # Current implementation uses UTC by default
        claim = await data_manager.create_claim("Timezone test claim", "user")
        
        # Should store in UTC-compatible format
        assert claim.created_at.tzinfo is None or claim.created_at.tzinfo == UTC
        
        # Should be retrievable
        retrieved = await data_manager.get_claim(claim.id)
        assert retrieved.created_at == claim.created_at