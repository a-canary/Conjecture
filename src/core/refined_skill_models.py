"""
Refined Skill Models for the Conjecture skill-based agency system.
Aligns with intended design: Tools=Python functions, Skills=procedures, Samples=XML/response.
"""
from typing import Dict, List, Any, Optional, Union
from pydantic import Field, validator
from datetime import datetime
import json
import re

from ..data.models import Claim


class ProcedureStep(BaseModel):
    """Represents a single step in a skill procedure."""
    step_number: int = Field(..., description="Step order number")
    instruction: str = Field(..., description="Instruction for LLM to follow")
    tool_name: Optional[str] = Field(None, description="Tool to use in this step")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Parameters for the tool")
    conditions: Optional[List[str]] = Field(None, description="Conditions for this step")
    expected_output: Optional[str] = Field(None, description="Expected output format")
    
    def to_llm_instruction(self) -> str:
        """Convert step to LLM instruction format."""
        instruction = f"Step {self.step_number}: {self.instruction}"
        
        if self.tool_name:
            instruction += f"\n  Tool: {self.tool_name}"
            if self.parameters:
                params_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
                instruction += f"({params_str})"
        
        if self.conditions:
            instruction += f"\n  Conditions: {'; '.join(self.conditions)}"
        
        if self.expected_output:
            instruction += f"\n  Expected: {self.expected_output}"
        
        return instruction


class SkillClaim(Claim):
    """
    Claim describing how to use a tool or logical procedure for LLM.
    NOT a Python function, but procedural instructions for LLM to follow.
    """
    
    # Skill-specific fields
    tool_name: Optional[str] = Field(None, description="Name of the tool this skill uses")
    procedure_steps: List[ProcedureStep] = Field(default_factory=list, description="Step-by-step procedure")
    prerequisites: List[str] = Field(default_factory=list, description="Prerequisites for using this skill")
    use_cases: List[str] = Field(default_factory=list, description="Common use cases")
    
    # Skill metadata
    skill_category: Optional[str] = Field(None, description="Category of the skill")
    complexity_level: int = Field(default=1, ge=1, le=5, description="Complexity level (1-5)")
    estimated_time: Optional[str] = Field(None, description="Estimated time to complete")
    
    @validator('tags', pre=True, always=True)
    def add_skill_tag(cls, v):
        """Ensure skill claims have the skill tag."""
        if not v:
            v = []
        if 'type.skill' not in v:
            v.append('type.skill')
        return v
    
    @validator('content')
    def validate_content_format(cls, v):
        """Validate that content describes a procedure."""
        if not v or len(v) < 20:
            raise ValueError("Skill content must be descriptive (min 20 characters)")
        
        # Check for procedural language patterns
        procedural_patterns = [
            r'\b(to|step|first|then|next|finally|after|before)\b',
            r'\b(use|call|invoke|execute|run)\b',
            r'\b(validate|check|verify|ensure)\b',
            r'\b(format|return|output|result)\b'
        ]
        
        has_procedural = any(re.search(pattern, v, re.IGNORECASE) for pattern in procedural_patterns)
        if not has_procedural:
            raise ValueError("Skill content should describe a procedure or process")
        
        return v
    
    def add_procedure_step(self, instruction: str, tool_name: Optional[str] = None,
                          parameters: Optional[Dict[str, Any]] = None,
                          conditions: Optional[List[str]] = None,
                          expected_output: Optional[str] = None) -> None:
        """Add a procedure step to this skill."""
        step_number = len(self.procedure_steps) + 1
        
        step = ProcedureStep(
            step_number=step_number,
            instruction=instruction,
            tool_name=tool_name,
            parameters=parameters,
            conditions=conditions,
            expected_output=expected_output
        )
        
        self.procedure_steps.append(step)
    
    def get_procedure_summary(self) -> str:
        """Get a summary of the procedure steps."""
        if not self.procedure_steps:
            return self.content
        
        summary = f"Procedure for {self.tool_name or 'this skill'}:\n"
        for step in self.procedure_steps:
            summary += f"  {step.to_llm_instruction()}\n"
        
        return summary
    
    def get_required_tools(self) -> List[str]:
        """Get list of tools required by this skill."""
        tools = set()
        for step in self.procedure_steps:
            if step.tool_name:
                tools.add(step.tool_name)
        return list(tools)
    
    def validate_procedure_completeness(self) -> tuple[bool, List[str]]:
        """Validate that the procedure is complete and logical."""
        errors = []
        
        if not self.procedure_steps:
            errors.append("Skill must have at least one procedure step")
        
        # Check step numbering
        for i, step in enumerate(self.procedure_steps):
            if step.step_number != i + 1:
                errors.append(f"Step {i+1} has incorrect number: {step.step_number}")
        
        # Check for tool references
        for step in self.procedure_steps:
            if step.tool_name and not step.parameters:
                errors.append(f"Step {step.step_number} references tool {step.tool_name} but has no parameters")
        
        # Check logical flow
        if self.procedure_steps:
            first_step = self.procedure_steps[0]
            if first_step.conditions:
                errors.append("First step should not have conditions (it's the entry point)")
        
        return len(errors) == 0, errors
    
    def to_llm_context(self) -> str:
        """Convert skill to LLM context format."""
        context = f"Skill: {self.content}\n"
        
        if self.tool_name:
            context += f"Tool: {self.tool_name}\n"
        
        if self.prerequisites:
            context += f"Prerequisites: {'; '.join(self.prerequisites)}\n"
        
        if self.procedure_steps:
            context += "Procedure:\n"
            for step in self.procedure_steps:
                context += f"  {step.to_llm_instruction()}\n"
        
        if self.use_cases:
            context += f"Use Cases: {'; '.join(self.use_cases)}\n"
        
        return context


class SampleClaim(Claim):
    """
    Claim recording exact LLM call XML and tool response.
    Used to teach LLM exact syntax and patterns, including failures.
    """
    
    # Sample-specific fields
    tool_name: str = Field(..., description="Name of the tool being sampled")
    llm_call_xml: str = Field(..., description="Exact XML call from LLM response")
    tool_response: Optional[Any] = Field(None, description="Raw response from tool")
    llm_summary: Optional[str] = Field(None, description="LLM-generated summary of response")
    
    # Sample metadata
    is_success: bool = Field(..., description="Whether the sample represents a successful call")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    execution_time_ms: Optional[int] = Field(None, description="Execution time in milliseconds")
    sample_quality: float = Field(default=0.5, ge=0.0, le=1.0, description="Quality score of this sample")
    
    # Usage tracking
    usage_count: int = Field(default=0, description="How many times this sample was referenced")
    helpfulness_score: float = Field(default=0.5, ge=0.0, le=1.0, description="How helpful this sample was")
    
    @validator('tags', pre=True, always=True)
    def add_sample_tag(cls, v):
        """Ensure sample claims have the sample tag."""
        if not v:
            v = []
        if 'type.sample' not in v:
            v.append('type.sample')
        if 'type.example' in v:  # Remove old tag
            v.remove('type.example')
        return v
    
    @validator('llm_call_xml')
    def validate_xml_format(cls, v):
        """Validate that llm_call_xml is properly formatted."""
        if not v:
            raise ValueError("LLM call XML cannot be empty")
        
        # Basic XML validation
        if not (v.strip().startswith('<') and v.strip().endswith('>')):
            raise ValueError("LLM call XML must be valid XML")
        
        # Check for required elements
        if '<invoke' not in v:
            raise ValueError("LLM call XML must contain an <invoke> element")
        
        if 'name=' not in v:
            raise ValueError("LLM call XML must specify tool name")
        
        return v
    
    def extract_tool_name_from_xml(self) -> str:
        """Extract tool name from the XML call."""
        # Use regex to find name attribute
        name_match = re.search(r'name=["\']([^"\']+)["\']', self.llm_call_xml)
        if name_match:
            return name_match.group(1)
        return self.tool_name  # Fallback
    
    def extract_parameters_from_xml(self) -> Dict[str, Any]:
        """Extract parameters from the XML call."""
        parameters = {}
        
        # Find parameter elements
        param_matches = re.findall(r'<parameter[^>]*name=["\']([^"\']+)["\'][^>]*>(.*?)</parameter>', 
                                  self.llm_call_xml, re.DOTALL)
        
        for param_name, param_value in param_matches:
            # Try to parse as JSON first
            try:
                parameters[param_name] = json.loads(param_value.strip())
            except json.JSONDecodeError:
                # Use as string
                parameters[param_name] = param_value.strip()
        
        return parameters
    
    def format_for_llm_context(self) -> str:
        """Format sample for LLM context."""
        context = f"Sample for {self.tool_name}:\n"
        
        if self.is_success:
            context += "Status: SUCCESS\n"
        else:
            context += f"Status: FAILED - {self.error_message}\n"
        
        context += f"XML Call:\n{self.llm_call_xml}\n"
        
        if self.tool_response is not None:
            if isinstance(self.tool_response, str):
                response_preview = self.tool_response[:200] + "..." if len(self.tool_response) > 200 else self.tool_response
            else:
                response_preview = str(self.tool_response)[:200] + "..." if len(str(self.tool_response)) > 200 else str(self.tool_response)
            context += f"Tool Response: {response_preview}\n"
        
        if self.llm_summary:
            context += f"Summary: {self.llm_summary}\n"
        
        if self.execution_time_ms:
            context += f"Execution Time: {self.execution_time_ms}ms\n"
        
        context += f"Quality Score: {self.sample_quality:.2f}\n"
        
        return context
    
    def update_usage_stats(self, helpfulness: Optional[float] = None) -> None:
        """Update usage statistics."""
        self.usage_count += 1
        if helpfulness is not None:
            # Update helpfulness with weighted average
            self.helpfulness_score = (self.helpfulness_score * (self.usage_count - 1) + helpfulness) / self.usage_count
    
    def is_similar_to(self, other: 'SampleClaim', threshold: float = 0.8) -> bool:
        """Check if this sample is similar to another sample."""
        if self.tool_name != other.tool_name:
            return False
        
        # Compare XML structure
        self_params = self.extract_parameters_from_xml()
        other_params = other.extract_parameters_from_xml()
        
        if not self_params or not other_params:
            return False
        
        # Simple parameter similarity check
        common_params = set(self_params.keys()) & set(other_params.keys())
        total_params = set(self_params.keys()) | set(other_params.keys())
        
        if not total_params:
            return False
        
        similarity = len(common_params) / len(total_params)
        return similarity >= threshold


class ToolCreationClaim(Claim):
    """
    Claim representing the creation of a new tool.
    Records the discovery process and tool creation details.
    """
    
    # Tool creation fields
    tool_name: str = Field(..., description="Name of the created tool")
    creation_method: str = Field(..., description="How the tool was discovered/created")
    websearch_query: Optional[str] = Field(None, description="Query used to discover tool method")
    discovery_source: Optional[str] = Field(None, description="Source of tool discovery")
    tool_code: str = Field(..., description="Python code for the tool")
    tool_file_path: Optional[str] = Field(None, description="Path to created tool file")
    
    # Creation metadata
    creation_reason: str = Field(..., description="Why this tool was needed")
    validation_status: str = Field(default="pending", description="Validation status")
    security_review: Optional[str] = Field(None, description="Security review notes")
    
    @validator('tags', pre=True, always=True)
    def add_tool_creation_tag(cls, v):
        """Ensure tool creation claims have the appropriate tag."""
        if not v:
            v = []
        if 'type.tool_creation' not in v:
            v.append('type.tool_creation')
        return v
    
    def to_creation_summary(self) -> str:
        """Get summary of tool creation process."""
        summary = f"Tool Creation: {self.tool_name}\n"
        summary += f"Reason: {self.creation_reason}\n"
        summary += f"Method: {self.creation_method}\n"
        
        if self.websearch_query:
            summary += f"Discovery Query: {self.websearch_query}\n"
        
        if self.discovery_source:
            summary += f"Source: {self.discovery_source}\n"
        
        summary += f"Status: {self.validation_status}\n"
        
        return summary


# Update imports for backward compatibility
from pydantic import BaseModel