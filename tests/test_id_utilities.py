"""
Unit tests for ID generation and validation utilities
Tests utility functions without mocking
"""

import pytest
import re
import time
import random

from src.core.models import validate_claim_id, generate_claim_id, validate_confidence


class TestIdGeneration:
    """Test ID generation and validation utilities"""

    def test_generate_claim_id_format(self):
        """Test generated claim ID follows expected format"""
        claim_id = generate_claim_id()

        # Should start with 'c'
        assert claim_id.startswith("c")

        # Should be a string
        assert isinstance(claim_id, str)

        # Should match pattern: c + timestamp + underscore + uuid
        # Format: c{13 digit timestamp}_{8 char uuid}
        pattern = r"^c\d{13}_[a-f0-9]{8}$"
        assert re.match(pattern, claim_id) is not None

    def test_generate_claim_id_uniqueness(self):
        """Test generated claim IDs are unique"""
        ids = []
        for _ in range(100):
            claim_id = generate_claim_id()
            ids.append(claim_id)

        # All IDs should be unique (allowing for rare collisions)
        unique_count = len(set(ids))
        assert unique_count >= len(ids) * 0.95  # Allow at most 5% collisions

        # All should be valid format
        for claim_id in ids:
            assert validate_claim_id(claim_id)

    def test_generate_claim_id_timestamp_component(self):
        """Test claim ID contains timestamp component"""
        before_time = int(time.time() * 1000)
        claim_id = generate_claim_id()
        after_time = int(time.time() * 1000)

        # Extract timestamp part (remove 'c' prefix and split by underscore)
        # Format: c{timestamp}_{uuid}
        parts = claim_id[1:].split("_")
        timestamp_part = parts[0]

        # Should be within reasonable time range
        # Allow some tolerance for execution time
        assert before_time <= int(timestamp_part) <= after_time

    def test_generate_claim_id_random_component(self):
        """Test claim ID has random component"""
        # Generate IDs quickly to test randomness
        rapid_ids = []
        for _ in range(10):
            claim_id = generate_claim_id()
            rapid_ids.append(claim_id)
            time.sleep(0.001)  # Small delay

        # Even with rapid generation, should have some variation
        # due to random component
        unique_rapid = set(rapid_ids)
        assert len(unique_rapid) > 5  # At least half should be unique

    def test_validate_claim_id_valid(self):
        """Test validation of valid claim IDs"""
        valid_ids = [
            "c1234567890123",
            "c16409952000001",
            "c9999999999999",
            "c1234567890123456",  # Longer IDs also valid
            "c1234567890123_abc12345",  # New format with underscore and uuid
            "c1765948283086_405381a4",  # Actual generated format
        ]

        for claim_id in valid_ids:
            assert validate_claim_id(claim_id) is True

    def test_validate_claim_id_invalid_format(self):
        """Test validation of invalid claim ID formats"""
        invalid_ids = [
            "",  # Empty
            "c 1234567890123",  # Contains space
            "c@1234567890123",  # Contains special char
            "c#1234",  # Contains invalid special char
        ]

        for claim_id in invalid_ids:
            assert validate_claim_id(claim_id) is False

    def test_validate_claim_id_edge_cases(self):
        """Test validation edge cases"""
        # Non-string input - validate_claim_id checks isinstance(claim_id, str)
        assert validate_claim_id(123) is False
        assert validate_claim_id([]) is False
        assert validate_claim_id({}) is False
        assert validate_claim_id(None) is False

        # Only 'c' prefix is actually valid per the implementation
        # (it matches pattern r'^[a-zA-Z0-9_-]+$')
        assert validate_claim_id("c") is True

    def test_validate_confidence_valid(self):
        """Test confidence validation with valid values"""
        valid_confidences = [0.0, 0.1, 0.5, 0.9, 1.0]

        for confidence in valid_confidences:
            assert validate_confidence(confidence) is True

    def test_validate_confidence_invalid(self):
        """Test confidence validation with invalid values"""
        invalid_confidences = [-0.1, -1.0, 1.1, 2.0, float("inf"), float("-inf")]

        for confidence in invalid_confidences:
            assert validate_confidence(confidence) is False

    def test_validate_confidence_edge_cases(self):
        """Test confidence validation edge cases"""
        # Non-numeric input
        try:
            assert validate_confidence("0.5") is False
        except TypeError:
            # Expected behavior for non-numeric input
            pass

        try:
            assert validate_confidence([0.5]) is False
        except TypeError:
            # Expected behavior for non-numeric input
            pass

        try:
            assert validate_confidence({"confidence": 0.5}) is False
        except TypeError:
            # Expected behavior for non-numeric input
            pass

        # Very small positive number (should be valid)
        assert validate_confidence(0.0001) is True

        # Very close to boundaries
        assert validate_confidence(0.000001) is True
        assert validate_confidence(0.999999) is True

    def test_id_generation_performance(self):
        """Test ID generation performance"""
        import time

        start_time = time.time()

        # Generate many IDs
        for _ in range(1000):
            generate_claim_id()

        end_time = time.time()
        duration = end_time - start_time

        # Should be fast - less than 1 second for 1000 IDs
        assert duration < 1.0

    def test_id_validation_performance(self):
        """Test ID validation performance"""
        import time

        # Generate test data
        valid_ids = [generate_claim_id() for _ in range(500)]
        invalid_ids = [f"invalid{i}" for i in range(500)]

        start_time = time.time()

        # Validate all IDs
        for claim_id in valid_ids + invalid_ids:
            validate_claim_id(claim_id)

        end_time = time.time()
        duration = end_time - start_time

        # Should be fast - less than 0.5 seconds for 1000 validations
        assert duration < 0.5


class TestIdGeneratorUtilities:
    """Test ID generation utilities from src.utils.id_generator"""

    def test_generate_session_id_format(self):
        """Test generate_session_id returns expected format"""
        from src.utils.id_generator import generate_session_id

        session_id = generate_session_id()

        # Should be a string
        assert isinstance(session_id, str)

        # Should start with 'session_'
        assert session_id.startswith("session_")

        # Should contain timestamp and uuid components
        parts = session_id.split("_")
        assert len(parts) >= 3  # session, uuid_part, timestamp

    def test_generate_session_id_uniqueness(self):
        """Test generate_session_id produces unique IDs"""
        from src.utils.id_generator import generate_session_id

        ids = []
        for _ in range(10):
            session_id = generate_session_id()
            ids.append(session_id)

        # All IDs should be unique
        unique_count = len(set(ids))
        assert unique_count == len(ids)

    def test_generate_state_id_format(self):
        """Test generate_state_id returns expected format"""
        from src.utils.id_generator import generate_state_id

        state_id = generate_state_id()

        # Should be a string
        assert isinstance(state_id, str)

        # Should start with 'state_'
        assert state_id.startswith("state_")

        # Should contain uuid and timestamp components
        parts = state_id.split("_")
        assert len(parts) >= 3  # state, uuid_part, timestamp

    def test_generate_state_id_uniqueness(self):
        """Test generate_state_id produces unique IDs"""
        from src.utils.id_generator import generate_state_id

        ids = []
        for _ in range(10):
            state_id = generate_state_id()
            ids.append(state_id)

        # All IDs should be unique
        unique_count = len(set(ids))
        assert unique_count == len(ids)

    def test_generate_workflow_id_format(self):
        """Test generate_workflow_id returns expected format"""
        from src.utils.id_generator import generate_workflow_id

        workflow_id = generate_workflow_id()

        # Should be a string
        assert isinstance(workflow_id, str)

        # Should start with 'workflow_'
        assert workflow_id.startswith("workflow_")

        # Should contain uuid and timestamp components
        parts = workflow_id.split("_")
        assert len(parts) >= 3  # workflow, uuid_part, timestamp

    def test_generate_workflow_id_uniqueness(self):
        """Test generate_workflow_id produces unique IDs"""
        from src.utils.id_generator import generate_workflow_id

        ids = []
        for _ in range(10):
            workflow_id = generate_workflow_id()
            ids.append(workflow_id)

        # All IDs should be unique
        unique_count = len(set(ids))
        assert unique_count == len(ids)

    def test_generate_error_id_format(self):
        """Test generate_error_id returns expected format"""
        from src.utils.id_generator import generate_error_id

        error_id = generate_error_id()

        # Should be a string
        assert isinstance(error_id, str)

        # Should start with 'error_'
        assert error_id.startswith("error_")

        # Should contain uuid and timestamp components
        parts = error_id.split("_")
        assert len(parts) >= 3  # error, uuid_part, timestamp

    def test_generate_error_id_uniqueness(self):
        """Test generate_error_id produces unique IDs"""
        from src.utils.id_generator import generate_error_id

        ids = []
        for _ in range(10):
            error_id = generate_error_id()
            ids.append(error_id)

        # All IDs should be unique
        unique_count = len(set(ids))
        assert unique_count == len(ids)

    def test_generate_template_id_format(self):
        """Test generate_template_id returns expected format"""
        from src.utils.id_generator import generate_template_id

        template_id = generate_template_id()

        # Should be a string
        assert isinstance(template_id, str)

        # Should start with 'tpl_'
        assert template_id.startswith("tpl_")

        # Should contain uuid and random letter components
        parts = template_id.split("_")
        assert len(parts) >= 2  # tpl, uuid_part_letters

    def test_generate_template_id_uniqueness(self):
        """Test generate_template_id produces unique IDs"""
        from src.utils.id_generator import generate_template_id

        ids = []
        for _ in range(10):
            template_id = generate_template_id()
            ids.append(template_id)

        # All IDs should be unique
        unique_count = len(set(ids))
        assert unique_count == len(ids)

    def test_generate_context_id_format(self):
        """Test generate_context_id returns expected format"""
        from src.utils.id_generator import generate_context_id

        context_id = generate_context_id()

        # Should be a string
        assert isinstance(context_id, str)

        # Should start with 'ctx_'
        assert context_id.startswith("ctx_")

        # Should contain uuid and timestamp components
        parts = context_id.split("_")
        assert len(parts) >= 3  # ctx, uuid_part, timestamp

    def test_generate_context_id_uniqueness(self):
        """Test generate_context_id produces unique IDs"""
        from src.utils.id_generator import generate_context_id

        ids = []
        for _ in range(10):
            context_id = generate_context_id()
            ids.append(context_id)

        # All IDs should be unique
        unique_count = len(set(ids))
        assert unique_count == len(ids)

    def test_generate_short_id_default_length(self):
        """Test generate_short_id with default length"""
        from src.utils.id_generator import generate_short_id

        short_id = generate_short_id()

        # Should be a string
        assert isinstance(short_id, str)

        # Should be default length (8)
        assert len(short_id) == 8

        # Should contain only alphanumeric characters
        assert short_id.isalnum()

    def test_generate_short_id_custom_length(self):
        """Test generate_short_id with custom length"""
        from src.utils.id_generator import generate_short_id

        # Test different lengths
        for length in [4, 6, 10, 12]:
            short_id = generate_short_id(length)
            assert len(short_id) == length
            assert short_id.isalnum()

    def test_generate_short_id_uniqueness(self):
        """Test generate_short_id produces unique IDs"""
        from src.utils.id_generator import generate_short_id

        ids = []
        for _ in range(50):  # Generate more to test collision probability
            short_id = generate_short_id(12)  # Use longer length for uniqueness
            ids.append(short_id)

        # Most IDs should be unique (allow some collisions due to randomness)
        unique_count = len(set(ids))
        assert unique_count >= len(ids) * 0.9  # Allow at most 10% collisions

    def test_all_id_generators_return_strings(self):
        """Test all ID generator functions return strings"""
        from src.utils.id_generator import (
            generate_session_id,
            generate_state_id,
            generate_workflow_id,
            generate_error_id,
            generate_template_id,
            generate_context_id,
            generate_short_id,
        )

        functions = [
            generate_session_id,
            generate_state_id,
            generate_workflow_id,
            generate_error_id,
            generate_template_id,
            generate_context_id,
            generate_short_id,
        ]

        for func in functions:
            result = func()
            assert isinstance(result, str)
            assert len(result) > 0

    def test_id_generator_performance(self):
        """Test ID generator functions performance"""
        import time
        from src.utils.id_generator import generate_short_id

        start_time = time.time()

        # Generate many short IDs
        for _ in range(1000):
            generate_short_id()

        end_time = time.time()
        duration = end_time - start_time

        # Should be fast - less than 0.5 seconds for 1000 IDs
        assert duration < 0.5
