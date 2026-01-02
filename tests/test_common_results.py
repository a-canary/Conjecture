"""
Unit tests for src/core/common_results.py module
Tests ProcessingResult and BatchResult classes without mocking
"""

import pytest
from datetime import datetime, timedelta, timezone

from src.core.common_results import ProcessingResult, BatchResult


class TestProcessingResult:
    """Test ProcessingResult class"""

    def test_minimal_processing_result_success(self):
        """Test creating ProcessingResult with minimal required fields for success"""
        result = ProcessingResult(success=True, operation_type="test_operation")

        assert result.success is True
        assert result.operation_type == "test_operation"
        assert result.processed_items == 0  # Default value
        assert result.updated_items == 0  # Default value
        assert result.errors == []  # Default empty list
        assert result.warnings == []  # Default empty list
        assert result.execution_time is None  # Default value
        assert result.tokens_used is None  # Default value
        assert result.result_metadata == {}  # Default empty dict
        assert result.message == ""  # Default value
        assert result.started_at is None  # Default value
        assert result.completed_at is not None  # Set by __post_init__ for success

    def test_minimal_processing_result_failure(self):
        """Test creating ProcessingResult with minimal required fields for failure"""
        result = ProcessingResult(success=False, operation_type="failed_operation")

        assert result.success is False
        assert result.operation_type == "failed_operation"
        assert result.completed_at is None  # Not set for failure

    def test_full_processing_result(self):
        """Test creating ProcessingResult with all fields"""
        started_at = datetime.utcnow()
        completed_at = datetime.utcnow()

        result = ProcessingResult(
            success=True,
            operation_type="full_operation",
            processed_items=10,
            updated_items=5,
            errors=["error1", "error2"],
            warnings=["warning1"],
            execution_time=2.5,
            tokens_used=1000,
            result_metadata={"key1": "value1", "key2": 42},
            message="Operation completed",
            started_at=started_at,
            completed_at=completed_at,
        )

        assert result.success is True
        assert result.operation_type == "full_operation"
        assert result.processed_items == 10
        assert result.updated_items == 5
        assert result.errors == ["error1", "error2"]
        assert result.warnings == ["warning1"]
        assert result.execution_time == 2.5
        assert result.tokens_used == 1000
        assert result.result_metadata == {"key1": "value1", "key2": 42}
        assert result.message == "Operation completed"
        assert result.started_at == started_at
        assert result.completed_at == completed_at

    def test_add_error_method(self):
        """Test add_error method"""
        result = ProcessingResult(success=True, operation_type="test_operation")

        # Initially successful
        assert result.success is True
        assert result.errors == []

        # Add an error
        result.add_error("Test error message")

        assert result.success is False  # Should be marked as failed
        assert len(result.errors) == 1
        assert result.errors[0] == "Test error message"

        # Add another error
        result.add_error("Another error")

        assert result.success is False
        assert len(result.errors) == 2
        assert result.errors[1] == "Another error"

    def test_add_warning_method(self):
        """Test add_warning method"""
        result = ProcessingResult(success=True, operation_type="test_operation")

        # Initially no warnings
        assert result.warnings == []

        # Add a warning
        result.add_warning("Test warning message")

        assert len(result.warnings) == 1
        assert result.warnings[0] == "Test warning message"
        assert result.success is True  # Warnings don't affect success

        # Add another warning
        result.add_warning("Another warning")

        assert len(result.warnings) == 2
        assert result.warnings[1] == "Another warning"
        assert result.success is True

    def test_add_metadata_method(self):
        """Test add_metadata method"""
        result = ProcessingResult(success=True, operation_type="test_operation")

        # Initially empty metadata
        assert result.result_metadata == {}

        # Add metadata entries
        result.add_metadata("key1", "value1")
        result.add_metadata("key2", 42)
        result.add_metadata("key3", [1, 2, 3])

        assert result.result_metadata == {
            "key1": "value1",
            "key2": 42,
            "key3": [1, 2, 3],
        }

        # Overwrite existing key
        result.add_metadata("key1", "new_value")
        assert result.result_metadata["key1"] == "new_value"

    def test_get_summary_success(self):
        """Test get_summary method for successful operation"""
        result = ProcessingResult(
            success=True, operation_type="test_operation", processed_items=15
        )

        summary = result.get_summary()
        expected = "test_operation completed successfully: 15 items processed"
        assert summary == expected

    def test_get_summary_failure(self):
        """Test get_summary method for failed operation"""
        result = ProcessingResult(success=False, operation_type="failed_operation")
        result.add_error("Error 1")
        result.add_error("Error 2")

        summary = result.get_summary()
        expected = "failed_operation failed: 2 errors"
        assert summary == expected

    def test_get_summary_zero_items(self):
        """Test get_summary method with zero processed items"""
        result = ProcessingResult(
            success=True, operation_type="empty_operation", processed_items=0
        )

        summary = result.get_summary()
        expected = "empty_operation completed successfully: 0 items processed"
        assert summary == expected

    def test_to_dict_method(self):
        """Test to_dict method for serialization"""
        started_at = datetime(2023, 1, 1, 12, 0, 0)
        completed_at = datetime(2023, 1, 1, 12, 0, 5)

        result = ProcessingResult(
            success=True,
            operation_type="serialize_test",
            processed_items=10,
            updated_items=5,
            errors=["error1"],
            warnings=["warning1"],
            execution_time=2.5,
            tokens_used=1000,
            result_metadata={"key": "value"},
            message="Test message",
            started_at=started_at,
            completed_at=completed_at,
        )

        result_dict = result.to_dict()

        expected = {
            "success": True,
            "operation_type": "serialize_test",
            "processed_items": 10,
            "updated_items": 5,
            "errors": ["error1"],
            "warnings": ["warning1"],
            "execution_time": 2.5,
            "tokens_used": 1000,
            "metadata": {"key": "value"},
            "message": "Test message",
            "started_at": "2023-01-01T12:00:00",
            "completed_at": "2023-01-01T12:00:05",
        }

        assert result_dict == expected

    def test_to_dict_method_none_timestamps(self):
        """Test to_dict method with None timestamps"""
        result = ProcessingResult(success=False, operation_type="no_timestamps")

        result_dict = result.to_dict()

        assert result_dict["started_at"] is None
        assert result_dict["completed_at"] is None

    def test_processed_claims_property(self):
        """Test processed_claims property alias"""
        result = ProcessingResult(
            success=True, operation_type="test_operation", processed_items=25
        )

        assert result.processed_claims == 25
        assert result.processed_claims == result.processed_items

    def test_post_init_completion_timestamp(self):
        """Test __post_init__ sets completion timestamp for success"""
        before = datetime.utcnow()

        result = ProcessingResult(success=True, operation_type="auto_timestamp")

        after = datetime.utcnow()

        assert result.completed_at is not None
        assert before <= result.completed_at <= after

    def test_post_init_no_completion_timestamp_for_failure(self):
        """Test __post_init__ doesn't set completion timestamp for failure"""
        result = ProcessingResult(success=False, operation_type="failed_operation")

        assert result.completed_at is None

    def test_post_init_preserves_existing_timestamp(self):
        """Test __post_init__ preserves existing completion timestamp"""
        existing_time = datetime(2023, 1, 1, 12, 0, 0)

        result = ProcessingResult(
            success=True,
            operation_type="preserve_timestamp",
            completed_at=existing_time,
        )

        assert result.completed_at == existing_time

    def test_processing_result_immutability(self):
        """Test that ProcessingResult maintains its values"""
        result = ProcessingResult(
            success=True,
            operation_type="immutable_test",
            processed_items=100,
            updated_items=50,
        )

        # Values should remain unchanged
        assert result.success is True
        assert result.operation_type == "immutable_test"
        assert result.processed_items == 100
        assert result.updated_items == 50


class TestBatchResult:
    """Test BatchResult class"""

    def test_empty_batch_result(self):
        """Test creating empty BatchResult"""
        batch = BatchResult()

        assert batch.results == []
        assert batch.total_items == 0
        assert batch.successful_operations == 0
        assert batch.failed_operations == 0

    def test_batch_result_with_initial_values(self):
        """Test creating BatchResult with initial values"""
        batch = BatchResult(
            total_items=10, successful_operations=8, failed_operations=2
        )

        assert batch.results == []
        assert batch.total_items == 10
        assert batch.successful_operations == 8
        assert batch.failed_operations == 2

    def test_add_result_success(self):
        """Test add_result method with successful result"""
        batch = BatchResult()

        result = ProcessingResult(
            success=True, operation_type="success_op", processed_items=5
        )

        batch.add_result(result)

        assert len(batch.results) == 1
        assert batch.results[0] == result
        assert batch.total_items == 5
        assert batch.successful_operations == 1
        assert batch.failed_operations == 0

    def test_add_result_failure(self):
        """Test add_result method with failed result"""
        batch = BatchResult()

        result = ProcessingResult(
            success=False, operation_type="failed_op", processed_items=3
        )

        batch.add_result(result)

        assert len(batch.results) == 1
        assert batch.results[0] == result
        assert batch.total_items == 3
        assert batch.successful_operations == 0
        assert batch.failed_operations == 1

    def test_add_multiple_results(self):
        """Test adding multiple results"""
        batch = BatchResult()

        # Add successful result
        result1 = ProcessingResult(
            success=True, operation_type="op1", processed_items=5
        )
        batch.add_result(result1)

        # Add failed result
        result2 = ProcessingResult(
            success=False, operation_type="op2", processed_items=3
        )
        batch.add_result(result2)

        # Add another successful result
        result3 = ProcessingResult(
            success=True, operation_type="op3", processed_items=7
        )
        batch.add_result(result3)

        assert len(batch.results) == 3
        assert batch.total_items == 15  # 5 + 3 + 7
        assert batch.successful_operations == 2
        assert batch.failed_operations == 1

    def test_get_success_rate_empty(self):
        """Test get_success_rate with empty batch"""
        batch = BatchResult()

        success_rate = batch.get_success_rate()
        assert success_rate == 0.0

    def test_get_success_rate_all_success(self):
        """Test get_success_rate with all successful operations"""
        batch = BatchResult()

        for i in range(5):
            result = ProcessingResult(
                success=True, operation_type=f"op{i}", processed_items=1
            )
            batch.add_result(result)

        success_rate = batch.get_success_rate()
        assert success_rate == 100.0

    def test_get_success_rate_all_failure(self):
        """Test get_success_rate with all failed operations"""
        batch = BatchResult()

        for i in range(3):
            result = ProcessingResult(
                success=False, operation_type=f"op{i}", processed_items=1
            )
            batch.add_result(result)

        success_rate = batch.get_success_rate()
        assert success_rate == 0.0

    def test_get_success_rate_mixed(self):
        """Test get_success_rate with mixed success/failure"""
        batch = BatchResult()

        # Add 3 successful, 2 failed
        for i in range(3):
            result = ProcessingResult(
                success=True, operation_type=f"success{i}", processed_items=1
            )
            batch.add_result(result)

        for i in range(2):
            result = ProcessingResult(
                success=False, operation_type=f"failed{i}", processed_items=1
            )
            batch.add_result(result)

        success_rate = batch.get_success_rate()
        expected_rate = (3 / 5) * 100  # 60%
        assert success_rate == expected_rate

    def test_get_summary_method(self):
        """Test get_summary method"""
        batch = BatchResult()

        # Add some results
        for i in range(3):
            result = ProcessingResult(
                success=True, operation_type=f"op{i}", processed_items=1
            )
            batch.add_result(result)

        for i in range(2):
            result = ProcessingResult(
                success=False, operation_type=f"failed{i}", processed_items=1
            )
            batch.add_result(result)

        summary = batch.get_summary()
        expected = "Batch: 3/5 operations successful (60.0%)"
        assert summary == expected

    def test_get_summary_empty_batch(self):
        """Test get_summary with empty batch"""
        batch = BatchResult()

        summary = batch.get_summary()
        expected = "Batch: 0/0 operations successful (0.0%)"
        assert summary == expected

    def test_processed_claims_property(self):
        """Test processed_claims property alias"""
        batch = BatchResult()

        # Add results with different processed_items
        result1 = ProcessingResult(
            success=True, operation_type="op1", processed_items=10
        )
        result2 = ProcessingResult(
            success=False, operation_type="op2", processed_items=5
        )

        batch.add_result(result1)
        batch.add_result(result2)

        assert batch.processed_claims == 15  # 10 + 5
        assert batch.processed_claims == batch.total_items

    def test_success_property_all_success(self):
        """Test success property when all operations succeed"""
        batch = BatchResult()

        for i in range(3):
            result = ProcessingResult(
                success=True, operation_type=f"op{i}", processed_items=1
            )
            batch.add_result(result)

        assert batch.success is True

    def test_success_property_with_failures(self):
        """Test success property when there are failures"""
        batch = BatchResult()

        # Add successful operations
        for i in range(2):
            result = ProcessingResult(
                success=True, operation_type=f"success{i}", processed_items=1
            )
            batch.add_result(result)

        # Add one failed operation
        failed_result = ProcessingResult(
            success=False, operation_type="failed", processed_items=1
        )
        batch.add_result(failed_result)

        assert batch.success is False

    def test_success_property_empty_batch(self):
        """Test success property with empty batch"""
        batch = BatchResult()

        # Empty batch should be considered successful (no failures)
        assert batch.success is True

    def test_batch_result_immutability(self):
        """Test that BatchResult maintains its values"""
        batch = BatchResult(
            total_items=100, successful_operations=80, failed_operations=20
        )

        # Values should remain unchanged
        assert batch.total_items == 100
        assert batch.successful_operations == 80
        assert batch.failed_operations == 20

    def test_batch_result_with_zero_processed_items(self):
        """Test BatchResult with results that have zero processed items"""
        batch = BatchResult()

        result1 = ProcessingResult(
            success=True, operation_type="empty1", processed_items=0
        )
        result2 = ProcessingResult(
            success=False, operation_type="empty2", processed_items=0
        )

        batch.add_result(result1)
        batch.add_result(result2)

        assert batch.total_items == 0
        assert batch.successful_operations == 1
        assert batch.failed_operations == 1
        assert batch.processed_claims == 0

    def test_batch_result_comprehensive_workflow(self):
        """Test comprehensive workflow with BatchResult"""
        batch = BatchResult()

        # Simulate a batch processing workflow
        operations = [
            (True, "create_claim", 3),
            (True, "update_claim", 2),
            (False, "delete_claim", 1),
            (True, "search_claims", 5),
            (False, "analyze_claim", 1),
            (True, "export_claims", 10),
        ]

        for success, op_type, items in operations:
            result = ProcessingResult(
                success=success, operation_type=op_type, processed_items=items
            )

            # Add some warnings and errors
            if success and items > 5:
                result.add_warning(f"Large operation: {items} items")
            elif not success:
                result.add_error(f"Operation {op_type} failed")

            batch.add_result(result)

        # Verify final state
        assert len(batch.results) == 6
        assert batch.total_items == 22  # Sum of all items
        assert batch.successful_operations == 4
        assert batch.failed_operations == 2
        assert batch.get_success_rate() == (4 / 6) * 100  # 66.67%
        assert batch.success is False  # Has failures
        assert batch.processed_claims == 22

        # Check summary
        summary = batch.get_summary()
        assert "4/6 operations successful" in summary
        assert "66.7%" in summary
