"""
Data models for Skills Templates
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SkillType(str, Enum):
    """Skill template types"""
    RESEARCH = "research"
    WRITECODE = "writecode"
    TESTCODE = "testcode"
    ENDCLAIMEVAL = "endclaimeval"
    CUSTOM = "custom"


class GuidanceStep(BaseModel):
    """Single step in skill guidance"""
    step_number: int = Field(..., description="Step number")
    title: str = Field(..., description="Step title")
    description: str = Field(..., description="Detailed step description")
    instructions: List[str] = Field(default_factory=list, description="Specific instructions")
    tips: List[str] = Field(default_factory=list, description="Tips and best practices")
    expected_output: Optional[str] = Field(None, description="Expected output description")
    common_pitfalls: List[str] = Field(default_factory=list, description="Common pitfalls to avoid")


class SkillTemplate(BaseModel):
    """Template for a specific skill"""
    id: str = Field(..., description="Template ID")
    skill_type: SkillType = Field(..., description="Type of skill")
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    guidance_steps: List[GuidanceStep] = Field(..., description="Step-by-step guidance")
    prerequisites: List[str] = Field(default_factory=list, description="Prerequisites")
    success_criteria: List[str] = Field(default_factory=list, description="Success criteria")
    version: str = Field(default="1.0", description="Template version")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    usage_count: int = Field(default=0, description="Number of times used")
    success_rate: float = Field(default=1.0, description="Success rate")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SkillExecution(BaseModel):
    """Records skill execution and outcomes"""
    id: str = Field(..., description="Execution ID")
    template_id: str = Field(..., description="Template ID used")
    step_completed: int = Field(default=0, description="Number of steps completed")
    total_steps: int = Field(..., description="Total steps in template")
    user_input: str = Field(..., description="Original user input")
    execution_context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")
    outcomes: List[str] = Field(default_factory=list, description="Achieved outcomes")
    problems_encountered: List[str] = Field(default_factory=list, description="Problems faced")
    completion_status: str = Field(default="incomplete", description="Completion status")
    execution_time_seconds: float = Field(default=0.0, description="Total execution time")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class UserFeedback(BaseModel):
    """User feedback on skill template effectiveness"""
    template_id: str = Field(..., description="Template ID")
    execution_id: str = Field(..., description="Execution ID")
    clarity_score: int = Field(default=0, description="Clarity score (1-5)")
    helpfulness_score: int = Field(default=0, description="Helpfulness score (1-5)")
    completeness_score: int = Field(default=0, description="Completeness score (1-5)")
    feedback_text: Optional[str] = Field(None, description="Detailed feedback")
    suggested_improvements: List[str] = Field(default_factory=list, description="Suggested improvements")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SkillMetrics(BaseModel):
    """Performance metrics for skill templates"""
    template_id: str = Field(..., description="Template ID")
    total_executions: int = Field(default=0, description="Total number of executions")
    successful_executions: int = Field(default=0, description="Successful executions")
    average_completion_time: float = Field(default=0.0, description="Average completion time")
    average_steps_completed: float = Field(default=0.0, description="Average steps completed")
    user_satisfaction_score: float = Field(default=0.0, description="Average user satisfaction")
    common_problems: Dict[str, int] = Field(default_factory=dict, description="Common problems and their frequency")
    last_updated: datetime = Field(default_factory=datetime.utcnow)