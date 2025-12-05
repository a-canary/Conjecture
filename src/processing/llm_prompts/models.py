"""
Data models for LLM Prompt Management System
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator


class PromptTemplateStatus(str, Enum):
    """Prompt template status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DRAFT = "draft"
    DEPRECATED = "deprecated"


class PromptTemplateType(str, Enum):
    """Prompt template types"""
    RESEARCH = "research"
    CODING = "coding"
    TESTING = "testing"
    EVALUATION = "evaluation"
    GENERAL = "general"
    CUSTOM = "custom"
    ANALYSIS = "analysis"
    VALIDATION = "validation"
    SYNTHESIS = "synthesis"
    TASK_DECOMPOSITION = "task_decomposition"


class PromptVariable(BaseModel):
    """Represents a prompt template variable"""
    name: str = Field(..., description="Variable name")
    type: str = Field(default="string", description="Variable type")
    required: bool = Field(default=True, description="Whether variable is required")
    default_value: Optional[Any] = Field(None, description="Default value")
    description: str = Field("", description="Variable description")
    validation_rule: Optional[str] = Field(None, description="Validation rule")


class PromptTemplate(BaseModel):
    """Represents a prompt template"""
    id: str = Field(..., description="Template ID")
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    template_type: PromptTemplateType = Field(..., description="Template type")
    status: PromptTemplateStatus = Field(default=PromptTemplateStatus.ACTIVE)
    template_content: str = Field(..., description="Template content with placeholders")
    variables: List[PromptVariable] = Field(default_factory=list, description="Template variables")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    version: str = Field(default="1.0", description="Template version")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    usage_count: int = Field(default=0, description="Number of times used")
    success_rate: float = Field(default=1.0, description="Success rate")
    
    @validator('variables')
    def validate_content_placeholders(cls, v, values):
        """Ensure all placeholders in content have corresponding variables"""
        if 'template_content' in values:
            import re
            # Find {{variable}} placeholders
            placeholders = re.findall(r'\{\{(\w+)\}\}', values.get('template_content', ''))
            variable_names = [var.name for var in v]
            
            missing_vars = set(placeholders) - set(variable_names)
            if missing_vars:
                raise ValueError(f"Template contains undefined variables: {missing_vars}")
        
        return v
    
    def render(self, variables: Dict[str, Any]) -> str:
        """Render template with provided variables"""
        # Validate required variables
        for var in self.variables:
            if var.required and var.name not in variables:
                missing_var = var.name
                var_description = var.description or missing_var
                raise ValueError(f"Required variable '{missing_var}' ({var_description}) not provided")
        
        # Set default values
        resolved_vars = {}
        for var in self.variables:
            if var.name in variables:
                resolved_vars[var.name] = variables[var.name]
            elif var.default_value is not None:
                resolved_vars[var.name] = var.default_value
        
        # Render template
        rendered = self.template_content
        for name, value in resolved_vars.items():
            placeholder = f"{{{{{name}}}}}"
            rendered = rendered.replace(placeholder, str(value))
        
        return rendered
    
    def get_variables_summary(self) -> Dict[str, Any]:
        """Get summary of template variables"""
        required_vars = [var.name for var in self.variables if var.required]
        optional_vars = [var.name for var in self.variables if not var.required]
        
        return {
            'total_variables': len(self.variables),
            'required_variables': required_vars,
            'optional_variables': optional_vars,
            'variable_types': {var.name: var.type for var in self.variables}
        }
    
    def update_usage_stats(self, success: bool) -> None:
        """Update usage statistics"""
        self.usage_count += 1
        
        # Update success rate with exponential moving average
        if self.usage_count == 1:
            self.success_rate = 1.0 if success else 0.0
        else:
            alpha = 0.1  # Smoothing factor
            current_outcome = 1.0 if success else 0.0
            self.success_rate = (alpha * current_outcome) + ((1 - alpha) * self.success_rate)
        
        self.updated_at = datetime.utcnow()


class PromptValidationResult(BaseModel):
    """Result of prompt validation"""
    is_valid: bool = Field(default=False)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    rendered_preview: Optional[str] = Field(None)
    estimated_tokens: int = Field(default=0)


class IntegratedPrompt(BaseModel):
    """Represents an integrated prompt ready for LLM"""
    template_id: str = Field(..., description="Template ID used")
    rendered_prompt: str = Field(..., description="Rendered prompt content")
    variables_used: Dict[str, Any] = Field(..., description="Variables used in rendering")
    context_items_used: List[str] = Field(default_factory=list, description="Context item IDs used")
    token_count: int = Field(default=0, description="Total token count")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class OptimizedPrompt(BaseModel):
    """Represents an optimized prompt"""
    original_prompt: str = Field(..., description="Original prompt")
    optimized_prompt: str = Field(..., description="Optimized prompt")
    optimization_strategy: str = Field(..., description="Strategy used")
    token_reduction: int = Field(0, description="Tokens reduced")
    optimization_score: float = Field(0.0, description="Optimization quality score")
    changes_made: List[str] = Field(default_factory=list, description="Changes made")


class LLMResponse(BaseModel):
    """Represents an LLM response"""
    content: str = Field(..., description="Response content")
    model: str = Field(..., description="Model used")
    token_usage: Dict[str, int] = Field(..., description="Token usage information")
    response_time_ms: int = Field(0, description="Response time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ResponseSchema(BaseModel):
    """Schema for validating LLM responses"""
    schema_name: str = Field(..., description="Schema name")
    required_fields: List[str] = Field(default_factory=list)
    field_types: Dict[str, str] = Field(default_factory=dict)
    validation_rules: Dict[str, str] = Field(default_factory=dict)
    custom_validator: Optional[str] = Field(None, description="Custom validator name")


class ParsedResponse(BaseModel):
    """Represents a parsed and validated LLM response"""
    is_valid: bool = Field(default=False)
    parsed_data: Optional[Dict[str, Any]] = Field(None)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    confidence_score: float = Field(0.0, description="Confidence in parsing")
    extracted_claims: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FallbackResponse(BaseModel):
    """Represents a fallback response when parsing fails"""
    response_type: str = Field(..., description="Type of fallback")
    message: str = Field(..., description="Fallback message")
    original_response: str = Field(..., description="Original LLM response")
    parsing_errors: List[str] = Field(default_factory=list)
    should_retry: bool = Field(default=False, description="Whether to retry original request")
    retry_hints: List[str] = Field(default_factory=list)


class TokenUsage(BaseModel):
    """Token usage tracking"""
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)
    context_tokens: int = Field(default=0)


class PromptMetrics(BaseModel):
    """Metrics for prompt performance"""
    template_id: str = Field(..., description="Template ID")
    usage_count: int = Field(default=0)
    success_rate: float = Field(0.0)
    average_response_time_ms: float = Field(0.0)
    average_tokens_used: float = Field(0.0)
    last_used: Optional[datetime] = Field(None)
    error_rate: float = Field(0.0)
    user_satisfaction_score: Optional[float] = Field(None)