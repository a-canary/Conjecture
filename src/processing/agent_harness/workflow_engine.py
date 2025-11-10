"""
Workflow Engine for Agent Harness
Orchestrates end-to-end workflows and component coordination
"""

import asyncio
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
import logging

from .models import (
    WorkflowDefinition, WorkflowExecution, WorkflowStatus, WorkflowResult
)
from .state_tracker import StateTracker
from .session_manager import SessionManager
from ..utils.id_generator import generate_workflow_id


logger = logging.getLogger(__name__)


class StepType(str, Enum):
    """Workflow step types"""
    DATA_COLLECTION = "data_collection"
    CONTEXT_BUILDING = "context_building"
    PROMPT_GENERATION = "prompt_generation"
    LLM_INTERACTION = "llm_interaction"
    RESPONSE_PROCESSING = "response_processing"
    RESULT_PROCESSING = "result_processing"
    CUSTOM = "custom"


class WorkflowEngine:
    """
    Orchestrates workflow execution with support for parallel and sequential steps
    """

    def __init__(self, 
                 state_tracker: StateTracker,
                 session_manager: SessionManager,
                 max_concurrent_workflows: int = 50,
                 default_timeout_seconds: int = 300):
        
        self.state_tracker = state_tracker
        self.session_manager = session_manager
        self.max_concurrent_workflows = max_concurrent_workflows
        self.default_timeout_seconds = default_timeout_seconds
        
        # Workflow storage
        self.definitions: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        
        # Step handlers
        self.step_handlers: Dict[StepType, Callable] = {}
        self.custom_handlers: Dict[str, Callable] = {}
        
        # Registered components
        self.registered_components: Dict[str, Any] = {}
        
        # Initialize built-in handlers
        self._initialize_builtin_handlers()

    def register_component(self, name: str, component: Any) -> None:
        """
        Register a component that can be used in workflows
        
        Args:
            name: Component name
            component: Component instance
        """
        self.registered_components[name] = component
        logger.info(f"Registered component: {name}")

    def register_step_handler(self, step_type: StepType, handler: Callable) -> None:
        """
        Register a handler for a specific step type
        
        Args:
            step_type: Step type
            handler: Handler function
        """
        self.step_handlers[step_type] = handler
        logger.info(f"Registered step handler for: {step_type}")

    def register_custom_handler(self, step_name: str, handler: Callable) -> None:
        """
        Register a custom step handler
        
        Args:
            step_name: Step name
            handler: Handler function
        """
        self.custom_handlers[step_name] = handler
        logger.info(f"Registered custom handler: {step_name}")

    async def register_workflow(self, definition: WorkflowDefinition) -> bool:
        """
        Register a workflow definition
        
        Args:
            definition: Workflow definition
            
        Returns:
            True if registration successful
        """
        try:
            # Validate workflow definition
            validation_result = await self._validate_workflow_definition(definition)
            if not validation_result['valid']:
                logger.error(f"Invalid workflow definition: {validation_result['errors']}")
                return False

            self.definitions[definition.id] = definition
            logger.info(f"Registered workflow: {definition.id}")
            return True

        except Exception as e:
            logger.error(f"Failed to register workflow: {e}")
            return False

    async def execute_workflow(self, workflow_id: str, parameters: Dict[str, Any],
                             session_id: str) -> WorkflowResult:
        """
        Execute a workflow
        
        Args:
            workflow_id: Workflow definition ID
            parameters: Workflow parameters
            session_id: Session ID
            
        Returns:
            Workflow execution result
        """
        start_time = time.time()
        
        try:
            # Check workflow exists
            if workflow_id not in self.definitions:
                raise ValueError(f"Workflow {workflow_id} not found")

            # Check concurrent workflow limit
            active_workflows = [
                exec for exec in self.executions.values() 
                if exec.status in [WorkflowStatus.RUNNING, WorkflowStatus.PENDING]
            ]
            if len(active_workflows) >= self.max_concurrent_workflows:
                raise RuntimeError("Maximum concurrent workflows exceeded")

            # Get workflow definition
            definition = self.definitions[workflow_id]

            # Create execution instance
            execution = WorkflowExecution(
                id=generate_workflow_id(),
                workflow_id=workflow_id,
                session_id=session_id,
                parameters=parameters
            )

            # Store execution
            self.executions[execution.id] = execution

            # Track workflow start
            await self.state_tracker.track_state(
                session_id=session_id,
                operation=f"workflow_start_{workflow_id}",
                state_data={
                    "workflow_id": workflow_id,
                    "execution_id": execution.id,
                    "parameters": parameters,
                    "status": execution.status.value
                }
            )

            # Update session
            await self.session_manager.update_session(
                session_id, 
                {"context": {"current_workflow": execution.id}}
            )

            # Execute workflow
            logger.info(f"Starting workflow execution: {execution.id}")
            await self._execute_workflow_steps(definition, execution)

            # Calculate duration
            duration = time.time() - start_time

            # Create result
            result = WorkflowResult(
                execution_id=execution.id,
                status=execution.status,
                results=execution.results,
                duration_seconds=duration,
                metadata={
                    "workflow_id": workflow_id,
                    "parameters": parameters,
                    "completed_steps": execution.completed_steps,
                    "failed_steps": execution.failed_steps,
                    "retry_count": execution.retry_count
                }
            )

            # Track workflow completion
            await self.state_tracker.track_state(
                session_id=session_id,
                operation=f"workflow_complete_{workflow_id}",
                state_data={
                    "workflow_id": workflow_id,
                    "execution_id": execution.id,
                    "final_status": execution.status.value,
                    "duration_seconds": duration,
                    "result": result.dict()
                }
            )

            logger.info(f"Workflow execution completed: {execution.id} (status: {execution.status.value})")
            return result

        except Exception as e:
            duration = time.time() - start_time
            error_message = str(e)
            
            # Create error result
            result = WorkflowResult(
                execution_id=execution.id if 'execution' in locals() else "unknown",
                status=WorkflowStatus.FAILED,
                results={"error": error_message},
                duration_seconds=duration,
                metadata={"error": error_message}
            )
            
            logger.error(f"Workflow execution failed: {error_message}")
            return result

        finally:
            # Clean up execution if it exists
            if 'execution' in locals():
                # Keep completed executions for a while for inspection
                await asyncio.sleep(60)
                self.executions.pop(execution.id, None)

    async def get_workflow_status(self, workflow_execution_id: str) -> Optional[WorkflowStatus]:
        """
        Get the status of a workflow execution
        
        Args:
            workflow_execution_id: Workflow execution ID
            
        Returns:
            Workflow status or None if not found
        """
        execution = self.executions.get(workflow_execution_id)
        return execution.status if execution else None

    async def pause_workflow(self, workflow_execution_id: str) -> bool:
        """
        Pause a running workflow
        
        Args:
            workflow_execution_id: Workflow execution ID
            
        Returns:
            True if pause successful
        """
        try:
            execution = self.executions.get(workflow_execution_id)
            if not execution:
                return False

            if execution.status != WorkflowStatus.RUNNING:
                return False

            execution.status = WorkflowStatus.PAUSED
            
            await self.state_tracker.track_state(
                session_id=execution.session_id,
                operation="workflow_pause",
                state_data={
                    "workflow_execution_id": workflow_execution_id,
                    "current_step": execution.current_step
                }
            )
            
            logger.info(f"Paused workflow: {workflow_execution_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to pause workflow: {e}")
            return False

    async def resume_workflow(self, workflow_execution_id: str) -> bool:
        """
        Resume a paused workflow
        
        Args:
            workflow_execution_id: Workflow execution ID
            
        Returns:
            True if resume successful
        """
        try:
            execution = self.executions.get(workflow_execution_id)
            if not execution:
                return False

            if execution.status != WorkflowStatus.PAUSED:
                return False

            execution.status = WorkflowStatus.RUNNING
            
            # Continue execution
            definition = self.definitions[execution.workflow_id]
            asyncio.create_task(self._execute_workflow_steps(definition, execution))
            
            await self.state_tracker.track_state(
                session_id=execution.session_id,
                operation="workflow_resume",
                state_data={
                    "workflow_execution_id": workflow_execution_id,
                    "current_step": execution.current_step
                }
            )
            
            logger.info(f"Resumed workflow: {workflow_execution_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to resume workflow: {e}")
            return False

    async def cancel_workflow(self, workflow_execution_id: str) -> bool:
        """
        Cancel a workflow execution
        
        Args:
            workflow_execution_id: Workflow execution ID
            
        Returns:
            True if cancellation successful
        """
        try:
            execution = self.executions.get(workflow_execution_id)
            if not execution:
                return False

            if execution.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
                return False

            execution.status = WorkflowStatus.CANCELLED
            execution.completed_at = datetime.utcnow()
            
            await self.state_tracker.track_state(
                session_id=execution.session_id,
                operation="workflow_cancel",
                state_data={
                    "workflow_execution_id": workflow_execution_id,
                    "cancelled_at": execution.completed_at.isoformat()
                }
            )
            
            logger.info(f"Cancelled workflow: {workflow_execution_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel workflow: {e}")
            return False

    async def _execute_workflow_steps(self, definition: WorkflowDefinition, 
                                    execution: WorkflowExecution) -> None:
        """
        Execute workflow steps
        
        Args:
            definition: Workflow definition
            execution: Workflow execution instance
        """
        execution.status = WorkflowStatus.RUNNING

        for step in definition.steps:
            # Check workflow status (may have been paused/cancelled)
            if execution.status != WorkflowStatus.RUNNING:
                break

            try:
                # Start step
                step_id = step['id']
                execution.start_step(step_id)
                
                logger.debug(f"Executing step: {step_id}")
                
                # Execute step
                step_result = await self._execute_single_step(definition, execution, step)
                
                if step_result.get('success', False):
                    # Mark step completed
                    execution.mark_step_completed(step_id, step_result)
                    logger.debug(f"Completed step: {step_id}")
                else:
                    # Mark step failed
                    error_message = step_result.get('error', 'Unknown error')
                    execution.mark_step_failed(step_id, error_message)
                    
                    # Check retry policy
                    if await self._should_retry_step(definition, execution, step_id):
                        await self._retry_step(definition, execution, step)
                    else:
                        execution.fail(f"Step {step_id} failed: {error_message}")
                        break

            except Exception as e:
                error_message = str(e)
                execution.mark_step_failed(step_id, error_message)
                execution.fail(f"Step {step_id} failed with exception: {error_message}")
                break

        # Finalize workflow
        if execution.status == WorkflowStatus.RUNNING:
            execution.complete()

    async def _execute_single_step(self, definition: WorkflowDefinition,
                                 execution: WorkflowExecution, 
                                 step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single workflow step
        
        Args:
            definition: Workflow definition
            execution: Workflow execution instance
            step: Step definition
            
        Returns:
            Step execution result
        """
        try:
            step_id = step['id']
            step_type = StepType(step.get('type', 'custom'))
            step_config = step.get('config', {})
            step_dependencies = step.get('dependencies', [])
            step_timeout = step.get('timeout', definition.timeout_seconds)
            
            # Check dependencies
            for dep in step_dependencies:
                if dep not in execution.completed_steps:
                    return {
                        'success': False,
                        'error': f'Dependency {dep} not completed'
                    }
            
            # Get step handler
            if step_type == StepType.CUSTOM:
                handler = self.custom_handlers.get(step_id)
                if not handler:
                    return {
                        'success': False,
                        'error': f'No handler found for custom step: {step_id}'
                    }
            else:
                handler = self.step_handlers.get(step_type)
                if not handler:
                    return {
                        'success': False,
                        'error': f'No handler found for step type: {step_type}'
                    }
            
            # Prepare step context
            step_context = {
                'workflow_definition': definition,
                'execution': execution,
                'parameters': execution.parameters,
                'results': execution.results,
                'session_id': execution.session_id,
                'step_config': step_config,
                'registered_components': self.registered_components
            }
            
            # Execute with timeout
            try:
                result = await asyncio.wait_for(
                    handler(step_context),
                    timeout=step_timeout
                )
                
                if not isinstance(result, dict):
                    result = {'result': result}
                
                result['success'] = True
                return result

            except asyncio.TimeoutError:
                return {
                    'success': False,
                    'error': f'Step {step_id} timed out after {step_timeout} seconds'
                }

        except Exception as e:
            return {
                'success': False,
                'error': f'Step execution failed: {str(e)}'
            }

    async def _should_retry_step(self, definition: WorkflowDefinition,
                               execution: WorkflowExecution, step_id: str) -> bool:
        """
        Check if a step should be retried
        
        Args:
            definition: Workflow definition
            execution: Workflow execution instance
            step_id: Step ID
            
        Returns:
            True if step should be retried
        """
        retry_policy = definition.retry_policy
        if not retry_policy:
            return False

        step = definition.get_step(step_id)
        max_retries = step.get('max_retries', retry_policy.get('max_retries', 0))
        
        step_failed_count = execution.failed_steps.count(step_id)
        
        return step_failed_count < max_retries

    async def _retry_step(self, definition: WorkflowDefinition,
                         execution: WorkflowExecution, step: Dict[str, Any]) -> None:
        """
        Retry a failed step
        
        Args:
            definition: Workflow definition
            execution: Workflow execution instance 
            step: Step definition
        """
        retry_policy = definition.retry_policy
        retry_delay = step.get('retry_delay', retry_policy.get('delay_seconds', 5))
        
        execution.retry_count += 1
        
        # Remove from failed steps and retry
        if step['id'] in execution.failed_steps:
            execution.failed_steps.remove(step['id'])
        
        # Wait before retry
        await asyncio.sleep(retry_delay)
        
        # Execute step again
        step_result = await self._execute_single_step(definition, execution, step)
        
        if step_result.get('success', False):
            execution.mark_step_completed(step['id'], step_result)
        else:
            error_message = step_result.get('error', 'Unknown error')
            execution.mark_step_failed(step['id'], error_message)
            execution.fail(f"Step {step['id']} failed after {execution.retry_count} retries: {error_message}")

    async def _validate_workflow_definition(self, definition: WorkflowDefinition) -> Dict[str, Any]:
        """
        Validate a workflow definition
        
        Args:
            definition: Workflow definition to validate
            
        Returns:
            Validation result
        """
        errors = []
        
        # Check required fields
        if not definition.id:
            errors.append("Workflow ID is required")
        
        if not definition.name:
            errors.append("Workflow name is required")
        
        if not definition.steps:
            errors.append("Workflow must have at least one step")
        
        # Validate steps
        step_ids = set()
        for i, step in enumerate(definition.steps):
            step_id = step.get('id')
            if not step_id:
                errors.append(f"Step {i} missing ID")
                continue
            
            if step_id in step_ids:
                errors.append(f"Duplicate step ID: {step_id}")
            else:
                step_ids.add(step_id)
            
            step_type = step.get('type')
            if step_type not in [t.value for t in StepType] and step_type != 'custom':
                errors.append(f"Invalid step type: {step_type}")
            
            # Check dependencies
            dependencies = step.get('dependencies', [])
            for dep in dependencies:
                if dep not in step_ids and dep != step_id:  # Allow self-dependency for loops
                    errors.append(f"Step {step_id} depends on non-existent step: {dep}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }

    def _initialize_builtin_handlers(self) -> None:
        """Initialize built-in step handlers"""
        
        async def dummy_data_collection(context: Dict[str, Any]) -> Dict[str, Any]:
            """Dummy data collection handler"""
            return {
                'collected_data': [],
                'sources': [],
                'timestamp': datetime.utcnow().isoformat()
            }
        
        async def dummy_context_building(context: Dict[str, Any]) -> Dict[str, Any]:
            """Dummy context building handler"""
            return {
                'context_items': [],
                'context_string': '',
                'relevance_scores': {}
            }
        
        async def dummy_prompt_generation(context: Dict[str, Any]) -> Dict[str, Any]:
            """Dummy prompt generation handler"""
            return {
                'prompt': 'dummy prompt',
                'token_count': 100,
                'variables_used': []
            }
        
        async def dummy_llm_interaction(context: Dict[str, Any]) -> Dict[str, Any]:
            """Dummy LLM interaction handler"""
            return {
                'response': 'dummy response',
                'response_time_ms': 1000,
                'token_usage': {'input': 100, 'output': 50, 'total': 150}
            }
        
        async def dummy_response_processing(context: Dict[str, Any]) -> Dict[str, Any]:
            """Dummy response processing handler"""
            return {
                'parsed_result': {},
                'validation_status': 'success',
                'extracted_claims': []
            }
        
        async def dummy_result_processing(context: Dict[str, Any]) -> Dict[str, Any]:
            """Dummy result processing handler"""
            return {
                'final_result': {},
                'created_claims': [],
                'updated_relationships': []
            }
        
        # Register built-in handlers
        self.step_handlers[StepType.DATA_COLLECTION] = dummy_data_collection
        self.step_handlers[StepType.CONTEXT_BUILDING] = dummy_context_building
        self.step_handlers[StepType.PROMPT_GENERATION] = dummy_prompt_generation
        self.step_handlers[StepType.LLM_INTERACTION] = dummy_llm_interaction
        self.step_handlers[StepType.RESPONSE_PROCESSING] = dummy_response_processing
        self.step_handlers[StepType.RESULT_PROCESSING] = dummy_result_processing

    async def get_workflow_stats(self) -> Dict[str, Any]:
        """
        Get workflow execution statistics
        
        Returns:
            Statistics dictionary
        """
        try:
            total_executions = len(self.executions)
            status_counts = {}
            
            for execution in self.executions.values():
                status = execution.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            registered_workflows = len(self.definitions)
            
            return {
                'total_executions': total_executions,
                'status_distribution': status_counts,
                'registered_workflows': registered_workflows,
                'max_concurrent_workflows': self.max_concurrent_workflows,
                'registered_components': len(self.registered_components),
                'registered_handlers': len(self.step_handlers) + len(self.custom_handlers)
            }

        except Exception as e:
            logger.error(f"Failed to get workflow stats: {e}")
            return {}