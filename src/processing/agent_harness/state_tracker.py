"""
State Tracker for Agent Harness
Comprehensive state tracking, auditing, and history management
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import logging
import json
import threading

from .models import StateEntry, Session
from ..utils.id_generator import generate_state_id


logger = logging.getLogger(__name__)


class StateValidationRule:
    """Validation rule for state transitions"""
    
    def __init__(self, from_states: List[str], to_states: List[str], 
                 validator: Optional[Callable] = None):
        self.from_states = from_states
        self.to_states = to_states
        self.validator = validator
    
    def validate(self, current_state: str, target_state: str, 
                context: Optional[Dict[str, Any]] = None) -> bool:
        """Validate state transition"""
        # Check if current state is allowed
        if self.from_states and current_state not in self.from_states:
            return False
        
        # Check if target state is allowed
        if self.to_states and target_state not in self.to_states:
            return False
        
        # Run custom validator if provided
        if self.validator:
            return self.validator(current_state, target_state, context)
        
        return True


class StateTracker:
    """
    Tracks system and workflow states with comprehensive auditing and history
    """

    def __init__(self, max_history_entries: int = 10000, cleanup_interval_minutes: int = 60):
        self.max_history_entries = max_history_entries
        self.cleanup_interval_minutes = cleanup_interval_minutes
        
        # Storage structures
        self.state_history: List[StateEntry] = []
        self.current_states: Dict[str, Dict[str, Any]] = {}  # session_id -> state_data
        self.state_snapshots: Dict[str, List[StateEntry]] = {}  # session_id -> history snapshot
        
        # Validation rules
        self.validation_rules: Dict[str, List[StateValidationRule]] = {}
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Lock for thread safety
        self._lock = threading.Lock()

    async def initialize(self) -> None:
        """Initialize the state tracker and start cleanup task"""
        self._cleanup_task = asyncio.create_task(self._cleanup_old_entries())
        logger.info("State tracker initialized")

    async def shutdown(self) -> None:
        """Shutdown the state tracker and clean up resources"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("State tracker shutdown complete")

    def add_validation_rule(self, category: str, rule: StateValidationRule) -> None:
        """
        Add a state validation rule
        
        Args:
            category: Category of state (e.g., 'workflow', 'session', 'component')
            rule: Validation rule to add
        """
        if category not in self.validation_rules:
            self.validation_rules[category] = []
        
        self.validation_rules[category].append(rule)
        logger.debug(f"Added validation rule for category: {category}")

    async def track_state(self, session_id: str, operation: str, 
                         state_data: Dict[str, Any], 
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Track a state change
        
        Args:
            session_id: Associated session ID
            operation: Operation name
            state_data: State data
            metadata: Additional metadata
            
        Returns:
            State entry ID
        """
        try:
            with self._lock:
                # Validate state transition if rules exist
                if self._should_validate_state(operation, state_data):
                    current_state = self.current_states.get(session_id, {})
                    if not self._validate_state_transition(operation, current_state, state_data, metadata):
                        raise ValueError(f"Invalid state transition for {operation}")
                
                # Create state entry
                state_entry = StateEntry(
                    id=generate_state_id(),
                    session_id=session_id,
                    operation=operation,
                    state_data=state_data.copy(),
                    metadata=metadata or {}
                )
                
                # Store in history
                self.state_history.append(state_entry)
                
                # Update current state
                self.current_states[session_id] = {**state_data.copy(), 'timestamp': state_entry.timestamp}
                
                # Add to session snapshots
                if session_id not in self.state_snapshots:
                    self.state_snapshots[session_id] = []
                
                self.state_snapshots[session_id].append(state_entry)
                
                # Cleanup if needed
                if len(self.state_history) > self.max_history_entries:
                    self.state_history = self.state_history[-self.max_history_entries:]
                
                logger.debug(f"Tracked state for session {session_id}: {operation}")
                return state_entry.id

        except Exception as e:
            logger.error(f"Failed to track state: {e}")
            raise

    async def get_state_history(self, session_id: Optional[str] = None, 
                              operation: Optional[str] = None,
                              limit: Optional[int] = None,
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None) -> List[StateEntry]:
        """
        Get state history with filtering options
        
        Args:
            session_id: Optional session ID filter
            operation: Optional operation filter
            limit: Maximum number of entries to return
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            List of state entries
        """
        try:
            with self._lock:
                history = self.state_history.copy()
            
            # Apply filters
            if session_id:
                history = [entry for entry in history if entry.session_id == session_id]
            
            if operation:
                history = [entry for entry in history if entry.operation == operation]
            
            if start_time:
                history = [entry for entry in history if entry.timestamp >= start_time]
            
            if end_time:
                history = [entry for entry in history if entry.timestamp <= end_time]
            
            # Sort by timestamp (most recent first)
            history.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Apply limit
            if limit:
                history = history[:limit]
            
            return history

        except Exception as e:
            logger.error(f"Failed to get state history: {e}")
            return []

    async def get_current_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current state for a session
        
        Args:
            session_id: Session ID
            
        Returns:
            Current state data or None if not found
        """
        try:
            with self._lock:
                return self.current_states.get(session_id, {}).copy()
        except Exception as e:
            logger.error(f"Failed to get current state: {e}")
            return None

    async def rollback_to_state(self, session_id: str, state_id: str) -> bool:
        """
        Rollback session to a specific state
        
        Args:
            session_id: Session ID
            state_id: Target state ID
            
        Returns:
            True if rollback successful
        """
        try:
            # Find the target state
            target_state = None
            for entry in self.state_history:
                if entry.id == state_id and entry.session_id == session_id:
                    target_state = entry
                    break
            
            if not target_state:
                logger.warning(f"State {state_id} not found for session {session_id}")
                return False
            
            # Track rollback operation
            await self.track_state(
                session_id=session_id,
                operation="rollback",
                state_data=target_state.state_data,
                metadata={
                    "rollback_from_state_id": state_id,
                    "rollback_timestamp": datetime.utcnow().isoformat()
                }
            )
            
            # Update current state
            with self._lock:
                self.current_states[session_id] = {
                    **target_state.state_data.copy(),
                    'timestamp': datetime.utcnow(),
                    'rollback_from': state_id
                }
            
            logger.info(f"Rolled back session {session_id} to state {state_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to rollback session {session_id} to state {state_id}: {e}")
            return False

    async def validate_state_transition(self, category: str, current_state: str, 
                                      target_state: str, 
                                      context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate a state transition
        
        Args:
            category: State category
            current_state: Current state
            target_state: Target state
            context: Additional context
            
        Returns:
            True if transition is valid
        """
        try:
            if category not in self.validation_rules:
                return True  # No rules means any transition is allowed
            
            for rule in self.validation_rules[category]:
                if rule.validate(current_state, target_state, context):
                    return True
            
            return False

        except Exception as e:
            logger.error(f"Failed to validate state transition: {e}")
            return False

    async def create_state_snapshot(self, session_id: str, 
                                   description: Optional[str] = None) -> str:
        """
        Create a named state snapshot
        
        Args:
            session_id: Session ID
            description: Optional description
            
        Returns:
            Snapshot ID
        """
        try:
            with self._lock:
                current_state = self.current_states.get(session_id)
                if not current_state:
                    raise ValueError(f"No current state found for session {session_id}")
                
                snapshot_id = generate_state_id()
                
                # Create snapshot state entry
                snapshot_entry = StateEntry(
                    id=snapshot_id,
                    session_id=session_id,
                    operation="snapshot",
                    state_data=current_state.copy(),
                    metadata={
                        "description": description or f"Automatic snapshot at {datetime.utcnow()}",
                        "snapshot_type": "manual"
                    }
                )
                
                # Add to history and snapshots
                self.state_history.append(snapshot_entry)
                self.state_snapshots[session_id].append(snapshot_entry)
                
                logger.info(f"Created state snapshot {snapshot_id} for session {session_id}")
                return snapshot_id

        except Exception as e:
            logger.error(f"Failed to create state snapshot: {e}")
            raise

    async def get_state_snapshots(self, session_id: str) -> List[StateEntry]:
        """
        Get all state snapshots for a session
        
        Args:
            session_id: Session ID
            
        Returns:
            List of snapshot state entries
        """
        try:
            with self._lock:
                snapshots = self.state_snapshots.get(session_id, [])
                # Filter for snapshot operations
                return [entry for entry in snapshots if entry.operation == "snapshot"]

        except Exception as e:
            logger.error(f"Failed to get state snapshots: {e}")
            return []

    async def analyze_state_patterns(self, session_id: str, 
                                   time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Analyze state patterns for a session
        
        Args:
            session_id: Session ID
            time_window_hours: Time window to analyze in hours
            
        Returns:
            Analysis results
        """
        try:
            start_time = datetime.utcnow() - timedelta(hours=time_window_hours)
            history = await self.get_state_history(session_id=session_id, start_time=start_time)
            
            if not history:
                return {"session_id": session_id, "no_data": True}
            
            # Analyze operations
            operations = {}
            state_transitions = []
            
            for entry in history:
                # Count operations
                op_name = entry.operation
                operations[op_name] = operations.get(op_name, 0) + 1
                
                # Track state transitions
                if entry.state_data:
                    state_transitions.append({
                        'operation': op_name,
                        'timestamp': entry.timestamp,
                        'state': entry.state_data
                    })
            
            # Calculate metrics
            total_operations = len(history)
            operation_frequency = {op: count/total_operations for op, count in operations.items()}
            
            # Identify most frequent operations
            most_frequent_operations = sorted(operations.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                "session_id": session_id,
                "analysis_window_hours": time_window_hours,
                "total_operations": total_operations,
                "operation_counts": operations,
                "operation_frequency": operation_frequency,
                "most_frequent_operations": most_frequent_operations,
                "state_transitions": state_transitions[:10],  # Last 10 transitions
                "analysis_timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to analyze state patterns: {e}")
            return {"session_id": session_id, "error": str(e)}

    async def get_system_state_summary(self) -> Dict[str, Any]:
        """
        Get summary of system-wide state tracking
        
        Returns:
            System state summary
        """
        try:
            with self._lock:
                total_entries = len(self.state_history)
                active_sessions = len(self.current_states)
                total_snapshots = sum(len(snapshots) for snapshots in self.state_snapshots.values())
            
            # Recent activity
            recent_entries = await self.get_state_history(limit=100, start_time=datetime.utcnow() - timedelta(hours=1))
            
            # Operation distribution
            operation_counts = {}
            for entry in recent_entries:
                operation_counts[entry.operation] = operation_counts.get(entry.operation, 0) + 1
            
            return {
                "total_state_entries": total_entries,
                "active_sessions": active_sessions,
                "total_snapshots": total_snapshots,
                "recent_activity_1h": len(recent_entries),
                "recent_operation_distribution": operation_counts,
                "max_history_entries": self.max_history_entries,
                "cleanup_interval_minutes": self.cleanup_interval_minutes,
                "validation_rules": {category: len(rules) for category, rules in self.validation_rules.items()}
            }

        except Exception as e:
            logger.error(f"Failed to get system state summary: {e}")
            return {"error": str(e)}

    def _should_validate_state(self, operation: str, state_data: Dict[str, Any]) -> bool:
        """Check if state should undergo validation"""
        # Always validate if state has a 'state' field
        return 'state' in state_data or operation in self.validation_rules

    def _validate_state_transition(self, operation: str, current_state: Dict[str, Any], 
                                 new_state: Dict[str, Any], context: Optional[Dict[str, Any]]) -> bool:
        """Validate state transition using defined rules"""
        # Extract state values
        current_value = current_state.get('state')
        new_value = new_state.get('state')
        
        if not current_value or not new_value:
            return True  # No state to validate
        
        # Get validation rules for the operation
        rules = self.validation_rules.get(operation, [])
        
        if not rules:
            return True  # No rules means any transition is allowed
        
        # Check if any rule allows this transition
        for rule in rules:
            if rule.validate(current_value, new_value, context):
                return True
        
        return False

    async def _cleanup_old_entries(self) -> None:
        """Background task to clean up old state entries"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval_minutes * 60)
                
                with self._lock:
                    # Remove entries older than cleanup interval
                    cutoff_time = datetime.utcnow() - timedelta(minutes=self.cleanup_interval_minutes)
                    
                    # Clean up history
                    original_size = len(self.state_history)
                    self.state_history = [entry for entry in self.state_history 
                                         if entry.timestamp > cutoff_time]
                    cleaned_history = original_size - len(self.state_history)
                    
                    # Clean up snapshots
                    cleaned_snapshots = 0
                    for session_id, snapshots in self.state_snapshots.items():
                        original_size = len(snapshots)
                        self.state_snapshots[session_id] = [
                            entry for entry in snapshots if entry.timestamp > cutoff_time
                        ]
                        cleaned_snapshots += original_size - len(self.state_snapshots[session_id])
                    
                    # Clean up current states for inactive sessions
                    active_session_ids = set(entry.session_id for entry in self.state_history)
                    inactive_sessions = set(self.current_states.keys()) - active_session_ids
                    
                    for session_id in inactive_sessions:
                        del self.current_states[session_id]
                    
                    if cleaned_history > 0 or cleaned_snapshots > 0:
                        logger.info(f"Cleaned up {cleaned_history} history entries and {cleaned_snapshots} snapshots")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in state cleanup loop: {e}")

    async def export_state_data(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Export state data for backup or analysis
        
        Args:
            session_id: Optional session ID to export (exports all if None)
            
        Returns:
            Exported state data
        """
        try:
            with self._lock:
                if session_id:
                    history = [entry for entry in self.state_history if entry.session_id == session_id]
                    current_state = self.current_states.get(session_id)
                    snapshots = self.state_snapshots.get(session_id, [])
                else:
                    history = self.state_history.copy()
                    current_state = self.current_states.copy()
                    snapshots = self.state_snapshots.copy()
            
            return {
                'export_timestamp': datetime.utcnow().isoformat(),
                'session_id': session_id or 'all',
                'version': '1.0',
                'history': [entry.dict() for entry in history],
                'current_state': current_state,
                'snapshots': {sid: [entry.dict() for entry in snap] 
                             for sid, snap in snapshots.items()},
                'validation_rules': {category: [{'from_states': rule.from_states, 
                                               'to_states': rule.to_states} 
                                              for rule in rules] 
                                   for category, rules in self.validation_rules.items()}
            }

        except Exception as e:
            logger.error(f"Failed to export state data: {e}")
            return {"error": str(e)}