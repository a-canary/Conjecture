"""
Prompt Template Manager for LLM Prompt Management System
Template storage, versioning, and rendering
"""

import asyncio
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import logging
import json
import importlib.util

from .models import (
    PromptTemplate, PromptTemplateType, PromptTemplateStatus,
    PromptValidationResult, PromptMetrics
)
from ..utils.id_generator import generate_template_id


logger = logging.getLogger(__name__)


class PromptTemplateManager:
    """
    Manages prompt templates with versioning, rendering, and performance tracking
    """

    def __init__(self, storage_backend=None, max_templates: int = 1000):
        self.storage_backend = storage_backend
        self.max_templates = max_templates
        
        # Template storage
        self.templates: Dict[str, PromptTemplate] = {}
        self.template_versions: Dict[str, List[str]] = {}  # base_id -> list of version IDs
        
        # Performance tracking
        self.template_metrics: Dict[str, PromptMetrics] = {}
        
        # Template validators
        self.validators: Dict[str, callable] = {}
        
        # Built-in templates
        self._initialize_builtin_templates()

    def register_validator(self, validator_name: str, validator_func: callable) -> None:
        """
        Register a custom validator function
        
        Args:
            validator_name: Validator name
            validator_func: Validation function
        """
        self.validators[validator_name] = validator_func
        logger.info(f"Registered validator: {validator_name}")

    async def create_template(self, name: str, description: str,
                            template_content: str, template_type: PromptTemplateType,
                            variables: Optional[List[Dict[str, Any]]] = None,
                            metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new prompt template
        
        Args:
            name: Template name
            description: Template description
            template_content: Template content with {{variable}} placeholders
            template_type: Type of template
            variables: List of variable definitions
            metadata: Additional metadata
            
        Returns:
            Template ID
        """
        try:
            # Validate template content
            validation_result = await self.validate_template_content(
                template_content, variables or []
            )
            if not validation_result.is_valid:
                raise ValueError(f"Template validation failed: {validation_result.errors}")
            
            # Create template ID
            template_id = generate_template_id()
            
            # Convert variable definitions
            template_variables = []
            if variables:
                for var_def in variables:
                    template_variables.append({
                        'name': var_def['name'],
                        'type': var_def.get('type', 'string'),
                        'required': var_def.get('required', True),
                        'default_value': var_def.get('default_value'),
                        'description': var_def.get('description', ''),
                        'validation_rule': var_def.get('validation_rule')
                    })
            
            # Create template
            template = PromptTemplate(
                id=template_id,
                name=name,
                description=description,
                template_type=template_type,
                template_content=template_content,
                variables=template_variables,
                metadata=metadata or {}
            )
            
            # Store template
            self.templates[template_id] = template
            
            # Initialize metrics
            self.template_metrics[template_id] = PromptMetrics(
                template_id=template_id,
                usage_count=0,
                success_rate=1.0,
                average_response_time_ms=0.0,
                average_tokens_used=0.0,
                error_rate=0.0
            )
            
            logger.info(f"Created template: {template_id} ({name})")
            return template_id

        except Exception as e:
            logger.error(f"Failed to create template: {e}")
            raise

    async def get_template(self, template_id: str, version: Optional[str] = None) -> Optional[PromptTemplate]:
        """
        Get a template by ID (and optionally version)
        
        Args:
            template_id: Template ID
            version: Optional version identifier
            
        Returns:
            Template or None if not found
        """
        return self.templates.get(template_id)

    async def render_template(self, template_id: str, variables: Dict[str, Any],
                            version: Optional[str] = None) -> str:
        """
        Render a template with provided variables
        
        Args:
            template_id: Template ID
            variables: Variable values
            version: Optional version
            
        Returns:
            Rendered template content
            
        Raises:
            ValueError: If template not found or rendering fails
        """
        try:
            template = await self.get_template(template_id, version)
            if not template:
                raise ValueError(f"Template {template_id} not found")
            
            # Render template
            rendered = template.render(variables)
            
            # Update metrics
            metrics = self.template_metrics[template_id]
            metrics.usage_count += 1
            metrics.last_used = datetime.utcnow()
            
            logger.debug(f"Rendered template {template_id} with {len(variables)} variables")
            return rendered

        except Exception as e:
            logger.error(f"Failed to render template {template_id}: {e}")
            raise

    async def validate_template(self, template: PromptTemplate) -> PromptValidationResult:
        """
        Validate a prompt template
        
        Args:
            template: Template to validate
            
        Returns:
            Validation result
        """
        try:
            # Validate content placeholders
            validation_result = await self.validate_template_content(
                template.template_content, 
                template.variables
            )
            
            # Validate template variables
            for var in template.variables:
                if not var.name:
                    validation_result.errors.append("Variable name cannot be empty")
                
                # Check for duplicate variable names
                names = [v.name for v in template.variables]
                if names.count(var.name) > 1:
                    validation_result.errors.append(f"Duplicate variable name: {var.name}")
                
                # Validate variable type
                valid_types = ['string', 'number', 'boolean', 'object', 'array', 'any']
                if var.type not in valid_types:
                    validation_result.warnings.append(f"Unknown variable type: {var.type}")
            
            # Estimate tokens
            if not validation_result.errors:
                # Simple estimation (can be improved with actual tokenizer)
                validation_result.estimated_tokens = len(template.template_content) // 4
            
            return validation_result

        except Exception as e:
            return PromptValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"]
            )

    async def validate_template_content(self, content: str, 
                                      variables: List[Dict[str, Any]]) -> PromptValidationResult:
        """
        Validate template content and variables
        
        Args:
            content: Template content
            variables: Variable definitions
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        rendered_preview = None
        
        try:
            # Find all placeholders
            placeholders = re.findall(r'\{\{(\w+)\}\}', content)
            variable_names = [var['name'] for var in variables]
            
            # Check for undefined variables
            defined_vars = set(variable_names)
            used_vars = set(placeholders)
            
            undefined_vars = used_vars - defined_vars
            if undefined_vars:
                errors.append(f"Undefined variables: {', '.join(undefined_vars)}")
            
            # Check for unused variables
            unused_vars = defined_vars - used_vars
            if unused_vars:
                warnings.append(f"Unused variables: {', '.join(unused_vars)}")
            
            # Check for malformed placeholders
            malformed = re.findall(r'\{\{[^}]*$', content)
            if malformed:
                errors.append(f"Malformed placeholders found: {malformed}")
            
            # Create preview with dummy values
            if not errors:
                dummy_values = {}
                for var in variables:
                    var_name = var['name']
                    var_type = var.get('type', 'string')
                    
                    if var_type == 'string':
                        dummy_values[var_name] = f"[{var_name}]"
                    elif var_type in ['number', 'integer']:
                        dummy_values[var_name] = "0"
                    elif var_type == 'boolean':
                        dummy_values[var_name] = "true/false"
                    else:
                        dummy_values[var_name] = f"[{var_type}]"
                
                # Render preview
                rendered_preview = content
                for name, value in dummy_values.items():
                    placeholder = f"{{{{{name}}}}}"
                    rendered_preview = rendered_preview.replace(placeholder, str(value))
            
            is_valid = len(errors) == 0
            
            return PromptValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                rendered_preview=rendered_preview
            )

        except Exception as e:
            return PromptValidationResult(
                is_valid=False,
                errors=[f"Content validation error: {str(e)}"]
            )

    async def update_template(self, template_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a template
        
        Args:
            template_id: Template ID
            updates: Updates to apply
            
        Returns:
            True if update successful
        """
        try:
            template = self.templates.get(template_id)
            if not template:
                return False
            
            # Apply updates
            if 'name' in updates:
                template.name = updates['name']
            
            if 'description' in updates:
                template.description = updates['description']
            
            if 'template_content' in updates:
                # Validate new content
                validation_result = await self.validate_template_content(
                    updates['template_content'], 
                    [v.dict() for v in template.variables]
                )
                if not validation_result.is_valid:
                    raise ValueError(f"Template content validation failed: {validation_result.errors}")
                
                template.template_content = updates['template_content']
            
            if 'variables' in updates:
                # Validate new variables
                validation_result = await self.validate_template_content(
                    template.template_content,
                    updates['variables']
                )
                if not validation_result.is_valid:
                    raise ValueError(f"Variables validation failed: {validation_result.errors}")
                
                template.variables = [
                    PromptVariable(**var_def) for var_def in updates['variables']
                ]
            
            if 'metadata' in updates:
                template.metadata.update(updates['metadata'])
            
            if 'status' in updates:
                template.status = PromptTemplateStatus(updates['status'])
            
            # Update timestamp
            template.updated_at = datetime.utcnow()
            
            logger.info(f"Updated template: {template_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update template {template_id}: {e}")
            return False

    async def delete_template(self, template_id: str) -> bool:
        """
        Delete a template
        
        Args:
            template_id: Template ID
            
        Returns:
            True if deletion successful
        """
        try:
            if template_id in self.templates:
                del self.templates[template_id]
                del self.template_metrics[template_id]
                logger.info(f"Deleted template: {template_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to delete template {template_id}: {e}")
            return False

    async def list_templates(self, template_type: Optional[PromptTemplateType] = None,
                           status: Optional[PromptTemplateStatus] = None,
                           limit: Optional[int] = None) -> List[PromptTemplate]:
        """
        List templates with optional filtering
        
        Args:
            template_type: Filter by template type
            status: Filter by status
            limit: Maximum number of templates to return
            
        Returns:
            List of templates
        """
        try:
            templates = list(self.templates.values())
            
            # Apply filters
            if template_type:
                templates = [t for t in templates if t.template_type == template_type]
            
            if status:
                templates = [t for t in templates if t.status == status]
            
            # Sort by last used (most recent first)
            templates.sort(key=lambda t: (
                self.template_metrics[t.id].last_used or datetime.min
            ), reverse=True)
            
            # Apply limit
            if limit:
                templates = templates[:limit]
            
            return templates

        except Exception as e:
            logger.error(f"Failed to list templates: {e}")
            return []

    async def search_templates(self, query: str) -> List[PromptTemplate]:
        """
        Search templates by name, description, or content
        
        Args:
            query: Search query
            
        Returns:
            List of matching templates
        """
        try:
            query_lower = query.lower()
            matching_templates = []
            
            for template in self.templates.values():
                # Search in name and description
                if (query_lower in template.name.lower() or 
                    query_lower in template.description.lower()):
                    matching_templates.append(template)
                # Search in content
                elif query_lower in template.template_content.lower():
                    matching_templates.append(template)
            
            return matching_templates

        except Exception as e:
            logger.error(f"Failed to search templates: {e}")
            return []

    async def get_template_metrics(self, template_id: str) -> Optional[PromptMetrics]:
        """
        Get performance metrics for a template
        
        Args:
            template_id: Template ID
            
        Returns:
            Template metrics or None if not found
        """
        return self.template_metrics.get(template_id)

    async def update_template_performance(self, template_id: str, 
                                        response_time_ms: int,
                                        tokens_used: int,
                                        success: bool) -> None:
        """
        Update performance metrics for a template
        
        Args:
            template_id: Template ID
            response_time_ms: Response time in milliseconds
            tokens_used: Tokens used
            success: Whether the prompt was successful
        """
        try:
            template = self.templates.get(template_id)
            if not template:
                return
            
            # Update template stats
            template.update_usage_stats(success)
            
            # Update detailed metrics
            metrics = self.template_metrics[template_id]
            
            # Update averages
            if metrics.usage_count == 1:
                metrics.average_response_time_ms = response_time_ms
                metrics.average_tokens_used = tokens_used
            else:
                # Exponential moving average
                alpha = 0.1
                metrics.average_response_time_ms = (
                    alpha * response_time_ms + 
                    (1 - alpha) * metrics.average_response_time_ms
                )
                metrics.average_tokens_used = (
                    alpha * tokens_used + 
                    (1 - alpha) * metrics.average_tokens_used
                )
            
            # Update error rate
            if not success:
                error_outcome = 1.0
            else:
                error_outcome = 0.0
            
            if metrics.usage_count == 1:
                metrics.error_rate = error_outcome
            else:
                alpha = 0.1
                metrics.error_rate = (
                    alpha * error_outcome + 
                    (1 - alpha) * metrics.error_rate
                )

        except Exception as e:
            logger.error(f"Failed to update template performance: {e}")

    async def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics
        
        Returns:
            System statistics
        """
        try:
            total_templates = len(self.templates)
            
            # Count by type
            type_counts = {}
            for template in self.templates.values():
                type_name = template.template_type.value
                type_counts[type_name] = type_counts.get(type_name, 0) + 1
            
            # Count by status
            status_counts = {}
            for template in self.templates.values():
                status_name = template.status.value
                status_counts[status_name] = status_counts.get(status_name, 0) + 1
            
            # Performance summary
            total_usage = sum(t.usage_count for t in self.templates.values())
            avg_success_rate = sum(
                t.success_rate * t.usage_count for t in self.templates.values() if t.usage_count > 0
            ) / total_usage if total_usage > 0 else 0.0
            
            return {
                'total_templates': total_templates,
                'type_distribution': type_counts,
                'status_distribution': status_counts,
                'total_usage_count': total_usage,
                'average_success_rate': avg_success_rate,
                'active_validators': len(self.validators),
                'max_templates': self.max_templates
            }

        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {}

    def _initialize_builtin_templates(self) -> None:
        """Initialize built-in templates"""
        
        # Research template
        research_template = {
            'name': 'Research Skill Template',
            'description': 'Template for guiding research workflows',
            'template_content': '''You are a research assistant helping with the following query: {{user_query}}

RESEARCH GUIDANCE:
1. Information Gathering
   - Identify key concepts and terms in {{user_query}}
   - Determine what information needs to be collected
   - Plan search strategies

2. Source Evaluation
   - Evaluate credibility of sources
   - Check for bias and reliability
   - Verify information accuracy

3. Information Synthesis
   - Organize findings logically
   - Identify patterns and connections
   - Highlight gaps in knowledge

4. Claim Creation
   - Formulate clear, specific claims based on evidence
   - Assign confidence scores appropriately
   - Note uncertainty and limitations

Available Context:
{{relevant_context}}

Please provide step-by-step guidance for this research task.''',
            'template_type': 'research',
            'variables': [
                {'name': 'user_query', 'type': 'string', 'required': True, 'description': 'User research query'},
                {'name': 'relevant_context', 'type': 'string', 'required': False, 'description': 'Relevant context information'}
            ]
        }
        
        # Coding template
        coding_template = {
            'name': 'Code Development Template',
            'description': 'Template for guiding code development',
            'template_content': '''You are a coding assistant helping with: {{requirement}}

DEVELOPMENT GUIDANCE:
1. Requirements Analysis
   - Understand the functional requirements
   - Identify constraints and assumptions
   - Clarify ambiguous points

2. Design Planning
   - Plan the overall architecture
   - Define interfaces and abstractions
   - Consider edge cases

3. Implementation Approach
   - Choose appropriate algorithms and data structures
   - Follow coding best practices
   - Ensure code readability and maintainability

4. Testing Strategy
   - Plan comprehensive tests
   - Consider unit, integration, and system tests
   - Validate edge cases

Technical Context:
{{technical_context}}
{{available_tools}}

Please provide detailed guidance for implementing this solution.''',
            'template_type': 'coding',
            'variables': [
                {'name': 'requirement', 'type': 'string', 'required': True, 'description': 'Development requirement'},
                {'name': 'technical_context', 'type': 'string', 'required': False, 'description': 'Technical context'},
                {'name': 'available_tools', 'type': 'string', 'required': False, 'description': 'Available tools and libraries'}
            ]
        }
        
        # Testing template
        testing_template = {
            'name': 'Code Testing Template',
            'description': 'Template for guiding code testing',
            'template_content': '''You are a testing assistant for the following code: {{code_description}}

TESTING GUIDANCE:
1. Test Planning
   - Identify test requirements and objectives
   - Determine test coverage goals
   - Plan test data and scenarios

2. Test Types to Consider
   - Unit tests for individual components
   - Integration tests for component interactions
   - System tests for end-to-end functionality
   - Performance and load tests
   - Security tests if applicable

3. Test Design
   - Design test cases for happy paths
   - Include edge cases and error conditions
   - Consider boundary conditions
   - Plan for negative testing

4. Test Automation
   - Choose testing frameworks and tools
   - Structure test code organization
   - Implement test fixtures and utilities

Code Context:
{{code_context}}
{{available_testing_tools}}

Please provide comprehensive testing guidance.''',
            'template_type': 'testing',
            'variables': [
                {'name': 'code_description', 'type': 'string', 'required': True, 'description': 'Description of code to test'},
                {'name': 'code_context', 'type': 'string', 'required': False, 'description': 'Additional code context'},
                {'name': 'available_testing_tools', 'type': 'string', 'required': False, 'description': 'Available testing tools'}
            ]
        }
        
        # Create built-in templates
        builtin_templates = [research_template, coding_template, testing_template]
        
        for template_def in builtin_templates:
            try:
                template_id = asyncio.create_task(self._create_builtin_template(template_def))
                # In a real implementation, we'd need to handle this differently
                # For now, we'll store this for manual creation later
                logger.info(f"Initialized built-in template: {template_def['name']}")
            except Exception as e:
                logger.error(f"Failed to initialize built-in template {template_def['name']}: {e}")

    async def _create_builtin_template(self, template_def: Dict[str, Any]) -> str:
        """Create a built-in template"""
        return await self.create_template(
            name=template_def['name'],
            description=template_def['description'],
            template_content=template_def['template_content'],
            template_type=PromptTemplateType(template_def['template_type']),
            variables=template_def['variables'],
            metadata={'builtin': True}
        )

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on template manager"""
        try:
            # Test template creation
            test_template_id = await self.create_template(
                name="health_check",
                description="Test template for health check",
                template_content="Test: {{test_var}}",
                template_type=PromptTemplateType.GENERAL,
                variables=[{'name': 'test_var', 'type': 'string', 'required': True}]
            )
            
            # Test template rendering
            rendered = await self.render_template(
                test_template_id, 
                {'test_var': 'health_check_value'}
            )
            
            # Test template deletion
            await self.delete_template(test_template_id)
            
            return {
                'healthy': rendered == "Test: health_check_value",
                'total_templates': len(self.templates),
                'active_validators': len(self.validators),
                'test_render_works': rendered == "Test: health_check_value"
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'healthy': False,
                'error': str(e)
            }