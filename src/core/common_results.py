"""
Common result classes for processing operations
Consolidated ProcessingResult to reduce duplication across modules
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime

@dataclass
class ProcessingResult:
    """Unified result structure for processing operations"""

    # Core result information
    success: bool
    operation_type: str
    processed_items: int = 0
    updated_items: int = 0

    # Error and warning tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Performance metrics
    execution_time: Optional[float] = None  # in seconds
    tokens_used: Optional[int] = None

    # Additional metadata (renamed to avoid conflict)
    result_metadata: Dict[str, Any] = field(default_factory=dict)
    message: str = ""

    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def __post_init__(self):
        """Set completion timestamp if success"""
        if self.success and not self.completed_at:
            self.completed_at = datetime.utcnow()

    def add_error(self, error: str):
        """Add an error and mark as failed"""
        self.errors.append(error)
        self.success = False

    def add_warning(self, warning: str):
        """Add a warning message"""
        self.warnings.append(warning)

    def add_metadata(self, key: str, value: Any):
        """Add metadata entry"""
        self.result_metadata[key] = value

    def get_summary(self) -> str:
        """Get a human-readable summary"""
        if self.success:
            return f"{self.operation_type} completed successfully: {self.processed_items} items processed"
        else:
            return f"{self.operation_type} failed: {len(self.errors)} errors"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "success": self.success,
            "operation_type": self.operation_type,
            "processed_items": self.processed_items,
            "updated_items": self.updated_items,
            "errors": self.errors,
            "warnings": self.warnings,
            "execution_time": self.execution_time,
            "tokens_used": self.tokens_used,
            "metadata": self.result_metadata,
            "message": self.message,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
        }

    @property
    def processed_claims(self) -> int:
        """Alias for processed_items for test compatibility"""
        return self.processed_items

@dataclass
class BatchResult:
    """Result for batch processing operations"""

    results: List[ProcessingResult] = field(default_factory=list)
    total_items: int = 0
    successful_operations: int = 0
    failed_operations: int = 0

    def add_result(self, result: ProcessingResult):
        """Add a processing result to the batch"""
        self.results.append(result)
        self.total_items += result.processed_items

        if result.success:
            self.successful_operations += 1
        else:
            self.failed_operations += 1

    def get_success_rate(self) -> float:
        """Calculate success rate as percentage"""
        if not self.results:
            return 0.0
        return (self.successful_operations / len(self.results)) * 100

    def get_summary(self) -> str:
        """Get batch summary"""
        return f"Batch: {self.successful_operations}/{len(self.results)} operations successful ({self.get_success_rate():.1f}%)"

    @property
    def processed_claims(self) -> int:
        """Alias for total_items for test compatibility"""
        return self.total_items

    @property
    def success(self) -> bool:
        """Alias for success check for test compatibility"""
        return self.failed_operations == 0
