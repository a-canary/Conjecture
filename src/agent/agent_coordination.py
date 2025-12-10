"""
Pure Agent Coordination System - The Core 3-Part Architecture Implementation
Pure functions for coordinating Claims → LLM → Tools → Claims flow.
"""
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..core.models import Claim, Claim, DirtyReason
from ..core.claim_operations import mark_dirty, mark_clean, find_dirty_claims
from ..processing.tool_registry import create_tool_registry, load_all_tools_from_directory
from ..processing.tool_execution import execute_tool_from_registry, create_execution_summary
from .llm_inference import coordinate_three_part_flow, LLMContext, LLMResponse

logger = logging.getLogger(__name__)

@dataclass
class AgentSession:
    """Pure data structure for agent session."""
    session_id: str
    user_request: str
    claims: List[Claim]
    tool_registry: Any
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CoordinationResult:
    """Pure data structure for coordination result."""
    success: bool
    session_id: str
    user_request: str
    llm_response: Optional[str] = None
    tool_results: List[Any] = field(default_factory=list)
    updated_claims: List[Claim] = field(default_factory=list)
    new_claims: List[Claim] = field(default_factory=list)
    execution_summary: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

# Pure Functions for Session Management

def create_agent_session(user_request: str, 
                        existing_claims: List[Claim],
                        tool_registry,
                        metadata: Dict[str, Any] = None) -> AgentSession:
    """Pure function to create an agent session."""
    return AgentSession(
        session_id=str(uuid.uuid4()),
        user_request=user_request,
        claims=existing_claims.copy(),
        tool_registry=tool_registry,
        metadata=metadata or {}
    )

def update_session_claims(session: AgentSession, claims: List[Claim]) -> AgentSession:
    """Pure function to update session claims."""
    return AgentSession(
        session_id=session.session_id,
        user_request=session.user_request,
        claims=claims.copy(),
        tool_registry=session.tool_registry,
        conversation_history=session.conversation_history.copy(),
        created_at=session.created_at,
        last_activity=datetime.utcnow(),
        metadata=session.metadata.copy()
    )

def add_conversation_item(session: AgentSession, 
                         user_msg: str, 
                         assistant_msg: str) -> AgentSession:
    """Pure function to add conversation item."""
    new_history = session.conversation_history.copy()
    new_history.append({
        "user": user_msg,
        "assistant": assistant_msg,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    return AgentSession(
        session_id=session.session_id,
        user_request=session.user_request,
        claims=session.claims.copy(),
        tool_registry=session.tool_registry,
        conversation_history=new_history,
        created_at=session.created_at,
        last_activity=datetime.utcnow(),
        metadata=session.metadata.copy()
    )

# Pure Functions for 3-Part Coordination

def process_user_request(user_request: str,
                        existing_claims: List[Claim],
                        tool_registry,
                        conversation_history: List[Dict[str, Any]] = None,
                        metadata: Dict[str, Any] = None) -> CoordinationResult:
    """Pure function to process user request through 3-part architecture."""
    
    session_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    
    try:
        # Create session
        session = create_agent_session(
            user_request=user_request,
            existing_claims=existing_claims,
            tool_registry=tool_registry,
            metadata=metadata
        )
        
        # Coordinate the 3-part flow
        coordination_result = coordinate_three_part_flow(
            session_id=session_id,
            user_request=user_request,
            all_claims=existing_claims,
            tool_registry=tool_registry,
            conversation_history=conversation_history
        )
        
        if not coordination_result["success"]:
            return CoordinationResult(
                success=False,
                session_id=session_id,
                user_request=user_request,
                errors=coordination_result["errors"]
            )
        
        # Extract results
        llm_response = coordination_result["llm_response"].response_text
        tool_results = coordination_result["tool_results"]
        updated_claims = coordination_result["updated_claims"]
        new_claims = coordination_result["new_claims"]
        
        # Create execution summary
        execution_summary = create_execution_summary(tool_results)
        
        execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        return CoordinationResult(
            success=True,
            session_id=session_id,
            user_request=user_request,
            llm_response=llm_response,
            tool_results=tool_results,
            updated_claims=updated_claims,
            new_claims=new_claims,
            execution_summary={
                **execution_summary,
                "execution_time_ms": execution_time
            },
            metadata={
                "claims_processed": len(existing_claims),
                "tool_calls_made": len(tool_results),
                "new_claims_created": len(new_claims)
            }
        )
    
    except Exception as e:
        execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        logger.error(f"Error processing user request: {e}")
        
        return CoordinationResult(
            success=False,
            session_id=session_id,
            user_request=user_request,
            errors=[f"Processing error: {str(e)}"],
            metadata={"execution_time_ms": execution_time}
        )

def process_research_request(query: str,
                            existing_claims: List[Claim],
                            tool_registry) -> CoordinationResult:
    """Pure function specialized for research requests."""
    user_request = f"Research: {query}"
    
    return process_user_request(
        user_request=user_request,
        existing_claims=existing_claims,
        tool_registry=tool_registry,
        metadata={"request_type": "research"}
    )

def process_code_request(description: str,
                        existing_claims: List[Claim],
                        tool_registry) -> CoordinationResult:
    """Pure function specialized for code requests."""
    user_request = f"Write code: {description}"
    
    return process_user_request(
        user_request=user_request,
        existing_claims=existing_claims,
        tool_registry=tool_registry,
        metadata={"request_type": "code"}
    )

def process_evaluation_request(claim_ids: List[str],
                             existing_claims: List[Claim],
                             tool_registry) -> CoordinationResult:
    """Pure function specialized for claim evaluation."""
    # Mark claims as dirty for re-evaluation
    dirty_claims = []
    for claim in existing_claims:
        if claim.id in claim_ids:
            dirty_claim = mark_dirty(claim, DirtyReason.MANUAL_MARK)
            dirty_claims.append(dirty_claim)
    
    user_request = f"Evaluate claims: {', '.join(claim_ids)}"
    
    return process_user_request(
        user_request=user_request,
        existing_claims=dirty_claims,
        tool_registry=tool_registry,
        metadata={"request_type": "evaluation", "claim_ids": claim_ids}
    )

# Pure Functions for Claim Management

def reconcile_claim_differences(original_claims: List[Claim],
                              updated_claims: List[Claim]) -> Tuple[List[Claim], List[Claim], List[Claim]]:
    """Pure function to reconcile differences between claim sets."""
    original_ids = {claim.id for claim in original_claims}
    updated_ids = {claim.id for claim in updated_claims}
    
    # Find changed claims
    changed_claims = []
    original_claim_map = {claim.id: claim for claim in original_claims}
    
    for updated_claim in updated_claims:
        if updated_claim.id in original_claim_map:
            original_claim = original_claim_map[updated_claim.id]
            if (updated_claim.content != original_claim.content or
                updated_claim.confidence != original_claim.confidence or
                updated_claim.state != original_claim.state):
                changed_claims.append(updated_claim)
    
    # Find new claims
    new_claims = [claim for claim in updated_claims if claim.id not in original_ids]
    
    # Find removed claims
    removed_claims = [claim for claim in original_claims if claim.id not in updated_ids]
    
    return changed_claims, new_claims, removed_claims

def apply_claim_updates(original_claims: List[Claim],
                       updates: List[Claim]) -> List[Claim]:
    """Pure function to apply claim updates."""
    updated_map = {claim.id: claim for claim in updates}
    result_claims = []
    
    # Keep original claims not being updated
    for original_claim in original_claims:
        if original_claim.id in updated_map:
            result_claims.append(updated_map[original_claim.id])
        else:
            result_claims.append(original_claim)
    
    # Add completely new claims
    existing_ids = {claim.id for claim in original_claims}
    for updated_claim in updates:
        if updated_claim.id not in existing_ids:
            result_claims.append(updated_claim)
    
    return result_claims

# Pure Functions for System Initialization

def initialize_agent_system(tools_directory: str = "tools",
                           execution_limits = None) -> Dict[str, Any]:
    """Pure function to initialize the agent system."""
    try:
        # Initialize tool registry
        tool_registry = create_tool_registry(
            tools_directory=tools_directory,
            execution_limits=execution_limits
        )
        
        # Load tools
        updated_registry, loaded_count = load_all_tools_from_directory(tool_registry)
        
        return {
            "success": True,
            "tool_registry": updated_registry,
            "tools_loaded": loaded_count,
            "tools_directory": tools_directory,
            "initialized_at": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to initialize agent system: {e}")
        return {
            "success": False,
            "error": str(e),
            "initialized_at": datetime.utcnow().isoformat()
        }

def get_system_status(tool_registry, claims: List[Claim]) -> Dict[str, Any]:
    """Pure function to get system status."""
    dirty_claims = find_dirty_claims(claims)
    
    return {
        "tool_count": len(tool_registry.tools),
        "claim_count": len(claims),
        "dirty_claims_count": len(dirty_claims),
        "claim_states": {
            state.value: len([c for c in claims if c.state == state])
            for state in ClaimState
        },
        "system_health": "healthy" if len(dirty_claims) == 0 else "needs_attention",
        "status_timestamp": datetime.utcnow().isoformat()
    }

# Async Functions (when needed for external tool execution)

async def process_user_request_async(user_request: str,
                                   existing_claims: List[Claim],
                                   tool_registry,
                                   conversation_history: List[Dict[str, Any]] = None,
                                   metadata: Dict[str, Any] = None) -> CoordinationResult:
    """Async version of user request processing."""
    # For now, just call the sync version
    # In a real implementation, this would handle async tool execution
    return process_user_request(
        user_request=user_request,
        existing_claims=existing_claims,
        tool_registry=tool_registry,
        conversation_history=conversation_history,
        metadata=metadata
    )
