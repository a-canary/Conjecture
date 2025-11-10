"""
Data models for Agent Harness components
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SessionState(str, Enum):
    """Session state enumeration"""
    ACTIVE = "active"
    IDLE = "idle"
    SUSPENDED = "suspended"
    CLOSED = "closed"
    ERROR = "error"


class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ErrorSeverity(str, Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Session(BaseModel):
    """Represents a user session"""
    id: str = Field(..., description="Unique session identifier")
    user_id: str = Field(..., description="User identifier")
    state: SessionState = Field(default=SessionState.ACTIVE)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    workflow_history: List[str] = Field(default_factory=list)
    error_count: int = Field(default=0)
    
    def update_activity(self) -> None:
        """Update last activity timestamp"""
        self.last_activity = datetime.utcnow()
    
    def add_workflow_execution(self, workflow_id: str) -> None:
        """Add workflow execution to history"""
        self.workflow_history.append(workflow_id)
        self.update_activity()
    
    def increment_error_count(self) -> None:
        """Increment error counter"""
        self.error_count += 1
        self.update_activity()


class StateEntry(BaseModel):
    """Represents a state tracking entry"""
    id: str = Field(..., description="State entry ID")
    session_id: str = Field(..., description="Associated session ID")
    operation: str = Field(..., description="Operation name")
    state_data: Dict[str, Any] = Field(..., description="State data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowDefinition(BaseModel):
    """Defines a workflow template"""
    id: str = Field(..., description="Workflow identifier")
    name: str = Field(..., description="Workflow name")
    description: str = Field(..., description="Workflow description")
    steps: List[Dict[str, Any]] = Field(..., description="Workflow steps")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Expected parameters")
    timeout_seconds: int = Field(default=300, description="Workflow timeout in seconds")
    retry_policy: Dict[str, Any] = Field(default_factory=dict, description="Retry policy")
    
    def get_step(self, step_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow step by ID"""
        for step in self.steps:
            if step.get('id') == step_id:
                return step
        return None


class WorkflowExecution(BaseModel):
    """Represents a workflow execution instance"""
    id: str = Field(..., description="Execution identifier")
    workflow_id: str = Field(..., description="Workflow definition ID")
    session_id: str = Field(..., description="Associated session ID")
    status: WorkflowStatus = Field(default=WorkflowStatus.PENDING)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    current_step: Optional[str] = None
    completed_steps: List[str] = Field(default_factory=list)
    failed_steps: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    results: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    retry_count: int = Field(default=0)
    
    def mark_step_completed(self, step_id: str, result: Dict[str, Any]) -> None:
        """Mark a step as completed"""
        if step_id not in self.completed_steps:
            self.completed_steps.append(step_id)
            self.results[step_id] = result
            self.current_step = None
    
    def mark_step_failed(self, step_id: str, error_message: str) -> None:
        """Mark a step as failed"""
        if step_id not in self.failed_steps:
            self.failed_steps.append(step_id)
            self.error_message = error_message
    
    def start_step(self, step_id: str) -> None:
        """Start a new step"""
        self.current_step = step_id
    
    def complete(self) -> None:
        """Mark workflow as completed"""
        self.status = WorkflowStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.current_step = None
    
    def fail(self, error_message: str) -> None:
        """Mark workflow as failed"""
        self.status = WorkflowStatus.FAILED
        self.error_message = error_message
        self.completed_at = datetime.utcnow()
        self.current_step = None


class ErrorEntry(BaseModel):
    """Represents an error entry in the system"""
    id: str = Field(..., description="Error entry ID")
    session_id: Optional[str] = Field(None, description="Associated session ID")
    workflow_execution_id: Optional[str] = Field(None, description="Associated workflow execution")
    component: str = Field(..., description="Component where error occurred")
    error_type: str = Field(..., description="Error type/class")
    message: str = Field(..., description="Error message")
    severity: ErrorSeverity = Field(..., description="Error severity")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    stack_trace: Optional[str] = Field(None, description="Error stack trace")
    context: Dict[str, Any] = Field(default_factory=dict, description="Error context")
    resolved: bool = Field(default=False)
    resolution: Optional[str] = Field(None, description="Error resolution description")
    
    def resolve(self, resolution: str) -> None:
        """Mark error as resolved"""
        self.resolved = True
        self.resolution = resolution


class FallbackSolution(BaseModel):
    """Represents a fallback solution for failed operations"""
    operation: str = Field(..., description="Original operation")
    fallback_type: str = Field(..., description="Type of fallback solution")
    description: str = Field(..., description="Fallback description")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    success_probability: float = Field(default=0.5, description="Estimated success probability")
    implementation: Optional[str] = Field(None, description="Implementation details")


class WorkflowResult(BaseModel):
    """Result of workflow execution"""
    execution_id: str = Field(..., description="Workflow execution ID")
    status: WorkflowStatus = Field(..., description="Final status")
    results: Dict[str, Any] = Field(default_factory=dict, description="Step results")
    duration_seconds: float = Field(..., description="Total execution duration")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def is_success(self) -> bool:
        """Check if workflow completed successfully"""
        return self.status == WorkflowStatus.COMPLETED


class ErrorResult(BaseModel):
    """Result of error handling"""
    error_handled: bool = Field(..., description="Whether error was successfully handled")
    recovery_method: Optional[str] = Field(None, description="Method used for recovery")
    fallback_solution: Optional[FallbackSolution] = Field(None, description="Fallback solution if needed")
    user_message: str = Field(..., description="User-facing message")
    technical_details: Optional[str] = Field(None, description="Technical error details")
    should_retry: bool = Field(default=False, description="Whether operation should be retried")