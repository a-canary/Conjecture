"""
Session Manager for Agent Harness
Handles user sessions, context, and lifecycle management
"""

import asyncio
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

from .models import Session, SessionState
from ..utils.id_generator import generate_session_id


logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages user sessions with isolation, persistence, and lifecycle handling
    """

    def __init__(self, session_timeout_minutes: int = 30, max_concurrent_sessions: int = 100):
        self.session_timeout_minutes = session_timeout_minutes
        self.max_concurrent_sessions = max_concurrent_sessions
        self.active_sessions: Dict[str, Session] = {}
        self.session_cleanup_interval_seconds = 300  # 5 minutes
        self._cleanup_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize the session manager and start cleanup task"""
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
        logger.info("Session manager initialized")

    async def shutdown(self) -> None:
        """Shutdown the session manager and clean up resources"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all active sessions
        for session in list(self.active_sessions.values()):
            await self.close_session(session.id)
        
        logger.info("Session manager shutdown complete")

    async def create_session(self, user_id: str, context: Optional[Dict[str, Any]] = None) -> Session:
        """
        Create a new session for a user
        
        Args:
            user_id: User identifier
            context: Initial session context
            
        Returns:
            Created session object
            
        Raises:
            RuntimeError: If maximum concurrent sessions exceeded
        """
        try:
            # Check concurrent session limit
            if len(self.active_sessions) >= self.max_concurrent_sessions:
                logger.warning(f"Maximum concurrent sessions ({self.max_concurrent_sessions}) reached")
                await self._cleanup_expired_sessions()
                
                # Check again after cleanup
                if len(self.active_sessions) >= self.max_concurrent_sessions:
                    raise RuntimeError("Maximum concurrent sessions exceeded")

            # Generate unique session ID
            session_id = generate_session_id()
            
            # Create session
            session = Session(
                id=session_id,
                user_id=user_id,
                context=context or {},
                metadata={
                    "client_info": {},  # Could be populated from request
                    "capabilities": []
                }
            )

            # Store session
            self.active_sessions[session_id] = session
            
            logger.info(f"Created session {session_id} for user {user_id}")
            return session

        except Exception as e:
            logger.error(f"Failed to create session for user {user_id}: {e}")
            raise

    async def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get a session by ID
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session object or None if not found
        """
        session = self.active_sessions.get(session_id)
        
        if session:
            # Check if session has expired
            if self._is_session_expired(session):
                logger.info(f"Session {session_id} expired and will be removed")
                await self.close_session(session_id)
                return None
            
            # Update activity
            session.update_activity()
            
        return session

    async def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update session data
        
        Args:
            session_id: Session identifier
            updates: Dictionary of updates to apply
            
        Returns:
            True if update successful, False if session not found
        """
        try:
            session = await self.get_session(session_id)
            if not session:
                return False

            # Update context
            if 'context' in updates and isinstance(updates['context'], dict):
                session.context.update(updates['context'])

            # Update metadata
            if 'metadata' in updates and isinstance(updates['metadata'], dict):
                session.metadata.update(updates['metadata'])

            # Update state if specified
            if 'state' in updates:
                try:
                    session.state = SessionState(updates['state'])
                except ValueError:
                    logger.warning(f"Invalid state value for session {session_id}: {updates['state']}")
                    return False

            # Update last activity
            session.update_activity()
            
            logger.debug(f"Updated session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update session {session_id}: {e}")
            return False

    async def close_session(self, session_id: str) -> bool:
        """
        Close a session and clean up resources
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session closed successfully, False if not found
        """
        try:
            session = self.active_sessions.pop(session_id, None)
            if session:
                session.state = SessionState.CLOSED
                logger.info(f"Closed session {session_id} for user {session.user_id}")
                return True
            else:
                logger.warning(f"Attempted to close non-existent session {session_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to close session {session_id}: {e}")
            return False

    async def list_active_sessions(self, user_id: Optional[str] = None) -> List[Session]:
        """
        List active sessions
        
        Args:
            user_id: Optional user ID to filter sessions
            
        Returns:
            List of active sessions
        """
        try:
            # Clean up expired sessions first
            await self._cleanup_expired_sessions()
            
            sessions = list(self.active_sessions.values())
            
            # Filter by user ID if specified
            if user_id:
                sessions = [s for s in sessions if s.user_id == user_id]
            
            # Sort by last activity (most recent first)
            sessions.sort(key=lambda s: s.last_activity, reverse=True)
            
            return sessions

        except Exception as e:
            logger.error(f"Failed to list active sessions: {e}")
            return []

    async def get_user_sessions(self, user_id: str) -> List[Session]:
        """
        Get all sessions for a specific user
        
        Args:
            user_id: User identifier
            
        Returns:
            List of user's sessions
        """
        return await self.list_active_sessions(user_id)

    async def find_sessions_by_context(self, context_filter: Dict[str, Any]) -> List[Session]:
        """
        Find sessions matching context criteria
        
        Args:
            context_filter: Dictionary of context key-value pairs to match
            
        Returns:
            List of matching sessions
        """
        try:
            matching_sessions = []
            
            for session in self.active_sessions.values():
                # Check if session matches all context filters
                matches = True
                for key, value in context_filter.items():
                    if session.context.get(key) != value:
                        matches = False
                        break
                
                if matches:
                    matching_sessions.append(session)
            
            return matching_sessions

        except Exception as e:
            logger.error(f"Failed to find sessions by context: {e}")
            return []

    async def suspend_session(self, session_id: str) -> bool:
        """
        Suspend a session (temporarily pause activity)
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if suspension successful
        """
        return await self.update_session(session_id, {'state': 'suspended'})

    async def resume_session(self, session_id: str) -> bool:
        """
        Resume a suspended session
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if resumption successful
        """
        return await self.update_session(session_id, {'state': 'active'})

    async def get_session_stats(self) -> Dict[str, Any]:
        """
        Get session statistics
        
        Returns:
            Dictionary with session statistics
        """
        try:
            current_time = datetime.utcnow()
            
            # Count sessions by state
            state_counts = {}
            for session in self.active_sessions.values():
                state = session.state.value
                state_counts[state] = state_counts.get(state, 0) + 1

            # Calculate session durations
            session_durations = []
            for session in self.active_sessions.values():
                duration = (current_time - session.created_at).total_seconds()
                session_durations.append(duration)

            # Active sessions by user
            user_sessions = {}
            for session in self.active_sessions.values():
                user_sessions[session.user_id] = user_sessions.get(session.user_id, 0) + 1

            return {
                'total_active_sessions': len(self.active_sessions),
                'session_state_counts': state_counts,
                'average_session_duration_seconds': sum(session_durations) / len(session_durations) if session_durations else 0,
                'sessions_by_user': user_sessions,
                'timeout_minutes': self.session_timeout_minutes,
                'max_concurrent_sessions': self.max_concurrent_sessions
            }

        except Exception as e:
            logger.error(f"Failed to get session stats: {e}")
            return {}

    def _is_session_expired(self, session: Session) -> bool:
        """Check if a session has expired due to inactivity"""
        inactive_time = datetime.utcnow() - session.last_activity
        return inactive_time > timedelta(minutes=self.session_timeout_minutes)

    async def _cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions"""
        try:
            expired_sessions = []
            
            for session_id, session in list(self.active_sessions.items()):
                if self._is_session_expired(session):
                    expired_sessions.append(session_id)

            for session_id in expired_sessions:
                await self.close_session(session_id)

            if expired_sessions:
                logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

        except Exception as e:
            logger.error(f"Error during session cleanup: {e}")

    async def _cleanup_expired_sessions(self) -> None:
        """
        Background task to clean up expired sessions
        """
        while True:
            try:
                await asyncio.sleep(self.session_cleanup_interval_seconds)
                await self._cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup loop: {e}")

    async def validate_session_access(self, session_id: str, user_id: str) -> bool:
        """
        Validate that a user can access a session
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            
        Returns:
            True if access is allowed
        """
        try:
            session = await self.get_session(session_id)
            if not session:
                return False

            # Check user ownership (in a real system, you might have admin access etc.)
            return session.user_id == user_id

        except Exception as e:
            logger.error(f"Failed to validate session access: {e}")
            return False

    async def export_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Export session data for backup or analysis
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data dictionary or None if session not found
        """
        try:
            session = await self.get_session(session_id)
            if not session:
                return None

            return {
                'session': session.dict(),
                'export_timestamp': datetime.utcnow().isoformat(),
                'version': '1.0'
            }

        except Exception as e:
            logger.error(f"Failed to export session data: {e}")
            return None

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on session manager
        
        Returns:
            Health check results
        """
        try:
            current_time = datetime.utcnow()
            
            # Check basic functionality
            test_session = await self.create_session("health_check_user", {"test": True})
            session_retrieved = await self.get_session(test_session.id)
            await self.close_session(test_session.id)
            
            # Check cleanup task
            cleanup_task_healthy = (
                self._cleanup_task is not None and 
                not self._cleanup_task.done()
            )

            return {
                'healthy': session_retrieved is not None and cleanup_task_healthy,
                'active_sessions': len(self.active_sessions),
                'cleanup_task_healthy': cleanup_task_healthy,
                'test_session_works': session_retrieved is not None,
                'timestamp': current_time.isoformat()
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'healthy': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }