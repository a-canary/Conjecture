"""
Skill Manager for the Conjecture skill-based agency system.
Handles skill registration, discovery, and execution coordination.
"""
import asyncio
import time
import psutil
import os
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import traceback
import logging

from ..core.skill_models import (
    SkillClaim, ExampleClaim, ExecutionResult, ToolCall, 
    SkillRegistry, SkillParameter
)
from ..data.data_manager import DataManager
from ..data.models import ClaimNotFoundError, InvalidClaimError


logger = logging.getLogger(__name__)


class SkillManager:
    """
    Manages skill claims, execution, and example generation.
    """
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.registry = SkillRegistry()
        self.execution_history: List[ExecutionResult] = []
        self.max_history_size = 1000
        
        # Built-in skill functions
        self.builtin_skills: Dict[str, Callable] = {}
        self._register_builtin_skills()
    
    async def initialize(self) -> None:
        """Initialize the skill manager by loading existing skills from database."""
        try:
            # Load existing skill claims from database
            skill_claims = await self.data_manager.filter_claims(
                filters=None  # Will be implemented to filter by tag
            )
            
            # Filter for skill claims and register them
            for claim_dict in skill_claims:
                if 'type.skill' in claim_dict.get('tags', []):
                    try:
                        skill_claim = SkillClaim(**claim_dict)
                        self.registry.register_skill(skill_claim)
                    except Exception as e:
                        logger.warning(f"Failed to load skill claim {claim_dict.get('id', 'unknown')}: {e}")
            
            logger.info(f"Loaded {len(self.registry.skills)} skills from database")
            
        except Exception as e:
            logger.error(f"Failed to initialize skill manager: {e}")
            raise
    
    def _register_builtin_skills(self) -> None:
        """Register built-in skill functions."""
        
        async def search_claims(query: str, limit: int = 10, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
            """Search for claims by content and tags."""
            try:
                # Use similarity search
                similar_claims = await self.data_manager.search_similar(query, limit)
                
                # Filter by tags if specified
                if tags:
                    similar_claims = [
                        claim for claim in similar_claims
                        if any(tag in claim.tags for tag in tags)
                    ]
                
                return [
                    {
                        'id': claim.id,
                        'content': claim.content,
                        'confidence': claim.confidence,
                        'tags': claim.tags
                    }
                    for claim in similar_claims
                ]
            except Exception as e:
                return [{'error': str(e)}]
        
        async def create_claim(content: str, confidence: float = 0.5, tags: Optional[List[str]] = None, created_by: str = "skill_execution") -> Dict[str, Any]:
            """Create a new claim."""
            try:
                claim = await self.data_manager.create_claim(
                    content=content,
                    confidence=confidence,
                    tags=tags or [],
                    created_by=created_by
                )
                return {
                    'id': claim.id,
                    'content': claim.content,
                    'confidence': claim.confidence,
                    'tags': claim.tags,
                    'created_at': claim.created_at.isoformat()
                }
            except Exception as e:
                return {'error': str(e)}
        
        async def get_claim(claim_id: str) -> Dict[str, Any]:
            """Get a claim by ID."""
            try:
                claim = await self.data_manager.get_claim(claim_id)
                if claim:
                    return {
                        'id': claim.id,
                        'content': claim.content,
                        'confidence': claim.confidence,
                        'tags': claim.tags,
                        'created_at': claim.created_at.isoformat(),
                        'created_by': claim.created_by
                    }
                else:
                    return {'error': f'Claim {claim_id} not found'}
            except Exception as e:
                return {'error': str(e)}
        
        async def create_relationship(supporter_id: str, supported_id: str, relationship_type: str = "supports", created_by: str = "skill_execution") -> Dict[str, Any]:
            """Create a relationship between two claims."""
            try:
                relationship_id = await self.data_manager.add_relationship(
                    supporter_id=supporter_id,
                    supported_id=supported_id,
                    relationship_type=relationship_type,
                    created_by=created_by
                )
                return {
                    'relationship_id': relationship_id,
                    'supporter_id': supporter_id,
                    'supported_id': supported_id,
                    'relationship_type': relationship_type
                }
            except Exception as e:
                return {'error': str(e)}
        
        async def get_relationships(claim_id: str) -> Dict[str, Any]:
            """Get relationships for a claim."""
            try:
                relationships = await self.data_manager.get_relationships(claim_id)
                return {
                    'claim_id': claim_id,
                    'relationships': [
                        {
                            'id': rel.id,
                            'supporter_id': rel.supporter_id,
                            'supported_id': rel.supported_id,
                            'relationship_type': rel.relationship_type,
                            'created_at': rel.created_at.isoformat()
                        }
                        for rel in relationships
                    ]
                }
            except Exception as e:
                return {'error': str(e)}
        
        async def get_stats() -> Dict[str, Any]:
            """Get system statistics."""
            try:
                stats = await self.data_manager.get_stats()
                skill_stats = self.registry.get_skill_stats()
                return {
                    'data_layer': stats,
                    'skills': skill_stats,
                    'execution_history_size': len(self.execution_history)
                }
            except Exception as e:
                return {'error': str(e)}
        
        # Register built-in skills
        self.builtin_skills.update({
            'search_claims': search_claims,
            'create_claim': create_claim,
            'get_claim': get_claim,
            'create_relationship': create_relationship,
            'get_relationships': get_relationships,
            'get_stats': get_stats
        })
    
    async def register_skill_claim(self, skill_claim: SkillClaim) -> bool:
        """
        Register a new skill claim in the system.
        
        Args:
            skill_claim: Skill claim to register
            
        Returns:
            True if registration successful
        """
        try:
            # Validate skill claim
            if not skill_claim.function_name:
                raise InvalidClaimError("Skill claim must have a function name")
            
            # Check if skill name already exists
            if skill_claim.function_name in self.registry.skills:
                logger.warning(f"Skill {skill_claim.function_name} already exists, updating...")
            
            # Save to database
            await self.data_manager.create_claim(
                content=skill_claim.content,
                confidence=skill_claim.confidence,
                tags=skill_claim.tags,
                created_by=skill_claim.created_by
            )
            
            # Register in local registry
            self.registry.register_skill(skill_claim)
            
            logger.info(f"Registered skill: {skill_claim.function_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register skill {skill_claim.function_name}: {e}")
            return False
    
    async def execute_skill(self, skill_name: str, parameters: Dict[str, Any], 
                          execution_context: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """
        Execute a skill with given parameters.
        
        Args:
            skill_name: Name of the skill to execute
            parameters: Parameters for the skill
            execution_context: Optional execution context
            
        Returns:
            ExecutionResult with outcome and metadata
        """
        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Get skill definition
            skill = self.registry.get_skill(skill_name)
            if not skill:
                return ExecutionResult(
                    success=False,
                    error_message=f"Skill '{skill_name}' not found",
                    execution_time_ms=int((time.time() - start_time) * 1000),
                    skill_id=skill.id if skill else "unknown",
                    parameters_used=parameters
                )
            
            # Validate parameters
            is_valid, errors = skill.validate_parameters(parameters)
            if not is_valid:
                return ExecutionResult(
                    success=False,
                    error_message=f"Parameter validation failed: {'; '.join(errors)}",
                    execution_time_ms=int((time.time() - start_time) * 1000),
                    skill_id=skill.id,
                    parameters_used=parameters
                )
            
            # Execute the skill
            result = await self._execute_skill_function(skill, parameters, execution_context)
            
            # Calculate execution metrics
            execution_time = int((time.time() - start_time) * 1000)
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = end_memory - start_memory
            
            # Create execution result
            execution_result = ExecutionResult(
                success=result.get('success', False),
                result=result.get('result'),
                error_message=result.get('error'),
                execution_time_ms=execution_time,
                memory_usage_mb=memory_usage,
                stdout=result.get('stdout'),
                stderr=result.get('stderr'),
                skill_id=skill.id,
                parameters_used=parameters
            )
            
            # Update skill statistics
            skill.update_execution_stats(execution_result.success)
            
            # Store in database
            await self.data_manager.update_claim(
                skill.id,
                execution_count=skill.execution_count,
                success_count=skill.success_count
            )
            
            # Add to execution history
            self._add_to_history(execution_result)
            
            # Generate example if successful
            if execution_result.success:
                await self._generate_example_from_execution(execution_result)
            
            return execution_result
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            logger.error(f"Error executing skill {skill_name}: {e}")
            
            return ExecutionResult(
                success=False,
                error_message=f"Execution error: {str(e)}",
                execution_time_ms=execution_time,
                skill_id=skill.id if skill else "unknown",
                parameters_used=parameters
            )
    
    async def _execute_skill_function(self, skill: SkillClaim, parameters: Dict[str, Any], 
                                    execution_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute the actual skill function."""
        try:
            # Check if it's a built-in skill
            if skill.function_name in self.builtin_skills:
                func = self.builtin_skills[skill.function_name]
                result = await func(**parameters)
                return {'success': True, 'result': result}
            
            # For custom skills, we would implement a sandboxed execution environment
            # For now, return a placeholder
            return {
                'success': False,
                'error': f"Custom skill execution not yet implemented for {skill.function_name}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Function execution error: {str(e)}",
                'traceback': traceback.format_exc()
            }
    
    async def _generate_example_from_execution(self, execution_result: ExecutionResult) -> None:
        """Generate an example claim from a successful execution."""
        try:
            if not execution_result.success:
                return
            
            example_data = execution_result.to_example_data()
            
            # Create example claim
            await self.data_manager.create_claim(
                content=example_data['content'],
                confidence=example_data['confidence'],
                tags=example_data['tags'],
                created_by=example_data['created_by']
            )
            
            logger.info(f"Generated example for skill {execution_result.skill_id}")
            
        except Exception as e:
            logger.error(f"Failed to generate example from execution: {e}")
    
    def _add_to_history(self, execution_result: ExecutionResult) -> None:
        """Add execution result to history with size limit."""
        self.execution_history.append(execution_result)
        
        # Maintain history size limit
        if len(self.execution_history) > self.max_history_size:
            self.execution_history = self.execution_history[-self.max_history_size:]
    
    async def find_relevant_skills(self, query: str, limit: int = 5) -> List[SkillClaim]:
        """
        Find skills relevant to a query.
        
        Args:
            query: Query to search for
            limit: Maximum number of results
            
        Returns:
            List of relevant skill claims
        """
        try:
            # Search in registry
            registry_results = self.registry.search_skills(query)
            
            # Also search in database for skill claims
            skill_claims = await self.data_manager.search_similar(query, limit * 2)
            skill_claims = [
                claim for claim in skill_claims 
                if 'type.skill' in claim.tags
            ]
            
            # Combine and deduplicate results
            all_skills = registry_results + skill_claims
            seen_names = set()
            unique_skills = []
            
            for skill in all_skills:
                skill_name = getattr(skill, 'function_name', skill.id)
                if skill_name not in seen_names:
                    seen_names.add(skill_name)
                    unique_skills.append(skill)
            
            return unique_skills[:limit]
            
        except Exception as e:
            logger.error(f"Error finding relevant skills: {e}")
            return []
    
    async def get_skill_examples(self, skill_id: str, limit: int = 5) -> List[ExampleClaim]:
        """
        Get examples for a specific skill.
        
        Args:
            skill_id: ID of the skill
            limit: Maximum number of examples
            
        Returns:
            List of example claims
        """
        try:
            # Search for example claims with this skill_id
            example_claims = await self.data_manager.filter_claims(
                filters=None  # Will be implemented to filter by skill_id
            )
            
            examples = []
            for claim_dict in example_claims:
                if ('type.example' in claim_dict.get('tags', []) and 
                    claim_dict.get('skill_id') == skill_id):
                    try:
                        example = ExampleClaim(**claim_dict)
                        examples.append(example)
                    except Exception as e:
                        logger.warning(f"Failed to load example claim {claim_dict.get('id', 'unknown')}: {e}")
            
            # Sort by quality and usage count
            examples.sort(key=lambda e: (e.example_quality, e.usage_count), reverse=True)
            
            return examples[:limit]
            
        except Exception as e:
            logger.error(f"Error getting skill examples: {e}")
            return []
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        if not self.execution_history:
            return {
                'total_executions': 0,
                'success_rate': 0.0,
                'average_execution_time_ms': 0.0,
                'most_used_skills': []
            }
        
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for result in self.execution_history if result.success)
        success_rate = successful_executions / total_executions
        
        avg_execution_time = sum(result.execution_time_ms for result in self.execution_history) / total_executions
        
        # Most used skills
        skill_counts = {}
        for result in self.execution_history:
            skill_counts[result.skill_id] = skill_counts.get(result.skill_id, 0) + 1
        
        most_used_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_executions': total_executions,
            'successful_executions': successful_executions,
            'success_rate': success_rate,
            'average_execution_time_ms': avg_execution_time,
            'most_used_skills': most_used_skills
        }