"""
Skill-specific data models for the Conjecture skill-based agency system.
Extends base claim models with skill-specific functionality.
"""
from typing import Dict, List, Any, Optional, Union, Callable
from pydantic import Field, validator
from datetime import datetime
import inspect
import json

from ..data.models import Claim, ClaimType


class SkillParameter(BaseModel):
    """Represents a parameter for a skill claim."""
    name: str = Field(..., description="Parameter name")
    param_type: str = Field(..., description="Parameter type (str, int, float, bool, dict, list)")
    required: bool = Field(default=True, description="Whether parameter is required")
    default_value: Optional[Any] = Field(None, description="Default value if not required")
    description: Optional[str] = Field(None, description="Parameter description")
    
    @validator('param_type')
    def validate_type(cls, v):
        valid_types = ['str', 'int', 'float', 'bool', 'dict', 'list', 'any']
        if v not in valid_types:
            raise ValueError(f"param_type must be one of {valid_types}")
        return v
    
    def validate_value(self, value: Any) -> bool:
        """Validate a value against this parameter definition."""
        if self.param_type == 'str':
            return isinstance(value, str)
        elif self.param_type == 'int':
            return isinstance(value, int)
        elif self.param_type == 'float':
            return isinstance(value, (int, float))
        elif self.param_type == 'bool':
            return isinstance(value, bool)
        elif self.param_type == 'dict':
            return isinstance(value, dict)
        elif self.param_type == 'list':
            return isinstance(value, list)
        elif self.param_type == 'any':
            return True
        return False


class SkillClaim(Claim):
    """Extended claim model for skill-based functionality."""
    
    # Skill-specific fields
    function_name: str = Field(..., description="Name of the skill function")
    parameters: List[SkillParameter] = Field(default_factory=list, description="Skill parameters")
    return_type: Optional[str] = Field(None, description="Return type of the skill")
    execution_context: Optional[Dict[str, Any]] = Field(None, description="Execution context metadata")
    examples: List[str] = Field(default_factory=list, description="Example usage strings")
    
    # Skill metadata
    skill_category: Optional[str] = Field(None, description="Category of the skill")
    skill_version: str = Field(default="1.0.0", description="Version of the skill")
    execution_count: int = Field(default=0, description="Number of times this skill was executed")
    success_count: int = Field(default=0, description="Number of successful executions")
    
    @validator('tags', pre=True, always=True)
    def add_skill_tag(cls, v):
        """Ensure skill claims have the skill tag."""
        if not v:
            v = []
        if 'type.skill' not in v:
            v.append('type.skill')
        return v
    
    @validator('function_name')
    def validate_function_name(cls, v):
        """Validate function name format."""
        if not v or not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("function_name must contain only alphanumeric characters, underscores, and hyphens")
        return v
    
    def validate_parameters(self, params: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate provided parameters against skill definition.
        
        Args:
            params: Parameters to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required parameters
        param_names = {p.name for p in self.parameters}
        required_params = {p.name for p in self.parameters if p.required}
        
        # Missing required parameters
        missing = required_params - set(params.keys())
        for param in missing:
            errors.append(f"Missing required parameter: {param}")
        
        # Unknown parameters
        unknown = set(params.keys()) - param_names
        for param in unknown:
            errors.append(f"Unknown parameter: {param}")
        
        # Type validation
        for param_def in self.parameters:
            if param_def.name in params:
                value = params[param_def.name]
                if not param_def.validate_value(value):
                    errors.append(f"Parameter {param_def.name} must be of type {param_def.param_type}")
        
        return len(errors) == 0, errors
    
    def get_parameter_defaults(self) -> Dict[str, Any]:
        """Get default values for optional parameters."""
        defaults = {}
        for param in self.parameters:
            if not param.required and param.default_value is not None:
                defaults[param.name] = param.default_value
        return defaults
    
    def get_success_rate(self) -> float:
        """Calculate success rate of this skill."""
        if self.execution_count == 0:
            return 0.0
        return self.success_count / self.execution_count
    
    def update_execution_stats(self, success: bool) -> None:
        """Update execution statistics."""
        self.execution_count += 1
        if success:
            self.success_count += 1
        self.updated_at = datetime.utcnow()


class ExampleClaim(Claim):
    """Claim representing an example of skill usage."""
    
    # Example-specific fields
    skill_id: str = Field(..., description="ID of the skill this example demonstrates")
    input_parameters: Dict[str, Any] = Field(..., description="Input parameters used")
    output_result: Optional[Any] = Field(None, description="Output result from execution")
    execution_time_ms: Optional[int] = Field(None, description="Execution time in milliseconds")
    
    # Example metadata
    example_quality: float = Field(default=0.5, ge=0.0, le=1.0, description="Quality score of this example")
    usage_count: int = Field(default=0, description="How many times this example was referenced")
    
    @validator('tags', pre=True, always=True)
    def add_example_tag(cls, v):
        """Ensure example claims have the example tag."""
        if not v:
            v = []
        if 'type.example' not in v:
            v.append('type.example')
        return v
    
    @validator('skill_id')
    def validate_skill_id(cls, v):
        """Validate skill ID format."""
        if not v or not v.startswith('c'):
            raise ValueError("skill_id must be a valid claim ID starting with 'c'")
        return v


class ExecutionResult(BaseModel):
    """Result of skill execution."""
    
    # Execution outcome
    success: bool = Field(..., description="Whether execution was successful")
    result: Optional[Any] = Field(None, description="Execution result if successful")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    
    # Execution metadata
    execution_time_ms: int = Field(..., description="Execution time in milliseconds")
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB")
    stdout: Optional[str] = Field(None, description="Standard output from execution")
    stderr: Optional[str] = Field(None, description="Standard error from execution")
    
    # Execution context
    skill_id: str = Field(..., description="ID of the skill that was executed")
    parameters_used: Dict[str, Any] = Field(..., description="Parameters used in execution")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Execution timestamp")
    
    def to_example_data(self) -> Dict[str, Any]:
        """Convert execution result to example claim data."""
        return {
            'skill_id': self.skill_id,
            'input_parameters': self.parameters_used,
            'output_result': self.result if self.success else None,
            'execution_time_ms': self.execution_time_ms,
            'example_quality': 1.0 if self.success else 0.0,
            'content': f"Example execution of {self.skill_id}: {self.parameters_used} -> {self.result}",
            'confidence': 0.9 if self.success else 0.1,
            'tags': ['type.example', 'auto_generated'],
            'created_by': 'system'
        }


class ToolCall(BaseModel):
    """Represents a parsed tool call from LLM response."""
    
    name: str = Field(..., description="Name of the tool/skill to invoke")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the tool")
    call_id: Optional[str] = Field(None, description="Unique identifier for this call")
    
    def to_skill_execution_params(self) -> Dict[str, Any]:
        """Convert to parameters for skill execution."""
        return {
            'skill_name': self.name,
            'parameters': self.parameters,
            'call_id': self.call_id
        }


class ParsedResponse(BaseModel):
    """Parsed LLM response containing tool calls."""
    
    tool_calls: List[ToolCall] = Field(default_factory=list, description="Parsed tool calls")
    text_content: Optional[str] = Field(None, description="Non-tool-call text content")
    parsing_errors: List[str] = Field(default_factory=list, description="Parsing errors encountered")
    
    def has_tool_calls(self) -> bool:
        """Check if response contains any tool calls."""
        return len(self.tool_calls) > 0
    
    def get_tool_call_names(self) -> List[str]:
        """Get names of all tools called."""
        return [call.name for call in self.tool_calls]


class SkillRegistry(BaseModel):
    """Registry for managing available skills."""
    
    skills: Dict[str, SkillClaim] = Field(default_factory=dict, description="Registered skills by name")
    categories: Dict[str, List[str]] = Field(default_factory=dict, description="Skills by category")
    
    def register_skill(self, skill: SkillClaim) -> None:
        """Register a new skill."""
        self.skills[skill.function_name] = skill
        
        # Add to category
        category = skill.skill_category or 'uncategorized'
        if category not in self.categories:
            self.categories[category] = []
        if skill.function_name not in self.categories[category]:
            self.categories[category].append(skill.function_name)
    
    def get_skill(self, name: str) -> Optional[SkillClaim]:
        """Get a skill by name."""
        return self.skills.get(name)
    
    def find_skills_by_category(self, category: str) -> List[SkillClaim]:
        """Find all skills in a category."""
        skill_names = self.categories.get(category, [])
        return [self.skills[name] for name in skill_names if name in self.skills]
    
    def search_skills(self, query: str) -> List[SkillClaim]:
        """Search skills by name or description."""
        query_lower = query.lower()
        results = []
        
        for skill in self.skills.values():
            if (query_lower in skill.function_name.lower() or 
                query_lower in skill.content.lower() or
                any(query_lower in tag.lower() for tag in skill.tags)):
                results.append(skill)
        
        return results
    
    def get_skill_stats(self) -> Dict[str, Any]:
        """Get statistics about registered skills."""
        total_skills = len(self.skills)
        total_executions = sum(skill.execution_count for skill in self.skills.values())
        total_successes = sum(skill.success_count for skill in self.skills.values())
        
        avg_success_rate = (total_successes / total_executions 
                           if total_executions > 0 else 0.0)
        
        return {
            'total_skills': total_skills,
            'total_executions': total_executions,
            'total_successes': total_successes,
            'average_success_rate': avg_success_rate,
            'categories': len(self.categories),
            'most_used_skills': sorted(
                self.skills.values(), 
                key=lambda s: s.execution_count, 
                reverse=True
            )[:5]
        }


# Import BaseModel from pydantic
from pydantic import BaseModel