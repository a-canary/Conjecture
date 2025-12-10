"""
Agent Harness - Core orchestration system for Conjecture AI agent.
Manages sessions, state, and workflow coordination.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..data.data_manager import DataManager
from ..core.models import Claim
from .support_systems import ContextBuilder, DataManager as SupportDataManager
from .prompt_system import PromptBuilder, ResponseParser

logger = logging.getLogger(__name__)

class SessionStatus(Enum):
    """Session status enumeration."""

    ACTIVE = "active"
    IDLE = "idle"
    ERROR = "error"
    TERMINATED = "terminated"

@dataclass
class Interaction:
    """Represents a single interaction in a session."""

    timestamp: datetime
    user_request: str
    llm_response: str
    parsed_response: Optional[Dict[str, Any]] = None
    execution_time_ms: int = 0
    error: Optional[str] = None

@dataclass
class SessionState:
    """Session state information."""

    session_id: str
    status: SessionStatus = SessionStatus.ACTIVE
    current_task: Optional[str] = None
    step_in_process: int = 0
    accumulated_context: Optional[Dict[str, Any]] = None
    error_count: int = 0
    last_error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)

@dataclass
class Session:
    """User session with state and history."""

    session_id: str
    state: SessionState
    interactions: List[Interaction] = field(default_factory=list)
    max_interactions: int = 100
    timeout_minutes: int = 30

    def add_interaction(self, interaction: Interaction) -> None:
        """Add an interaction to the session."""
        self.interactions.append(interaction)

        # Maintain interaction limit
        if len(self.interactions) > self.max_interactions:
            self.interactions = self.interactions[-self.max_interactions :]

        # Update last activity
        self.state.last_activity = interaction.timestamp

    def is_expired(self) -> bool:
        """Check if session has expired due to inactivity."""
        return datetime.utcnow() - self.state.last_activity > timedelta(
            minutes=self.timeout_minutes
        )

    def get_recent_interactions(self, count: int = 10) -> List[Interaction]:
        """Get recent interactions from the session."""
        return self.interactions[-count:] if self.interactions else []

class AgentHarness:
    """
    Main orchestration system for the AI agent.
    Manages sessions, coordinates components, and handles workflows.
    """

    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.sessions: Dict[str, Session] = {}
        self.context_builder = ContextBuilder(data_manager)
        self.prompt_builder = PromptBuilder()
        self.response_parser = ResponseParser()

        # Configuration
        self.max_sessions = 100
        self.cleanup_interval_minutes = 5
        self.session_timeout_minutes = 30

        # Start cleanup task
        self._cleanup_task = None

    async def initialize(self) -> None:
        """Initialize the agent harness."""
        try:
            # Initialize support systems
            await self.context_builder.initialize()

            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

            logger.info("Agent harness initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize agent harness: {e}")
            raise

    async def close(self) -> None:
        """Close the agent harness and cleanup resources."""
        try:
            # Cancel cleanup task
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass

            # Cleanup all sessions
            for session_id in list(self.sessions.keys()):
                await self.cleanup_session(session_id)

            logger.info("Agent harness closed successfully")

        except Exception as e:
            logger.error(f"Error closing agent harness: {e}")

    async def create_session(self) -> str:
        """
        Create a new session.

        Returns:
            Session ID for the new session
        """
        try:
            # Check session limit
            if len(self.sessions) >= self.max_sessions:
                # Cleanup expired sessions first
                await self._cleanup_expired_sessions()

                # Still at limit? Remove oldest idle session
                if len(self.sessions) >= self.max_sessions:
                    await self._remove_oldest_idle_session()

            # Generate session ID
            session_id = str(uuid.uuid4())

            # Create session state
            session_state = SessionState(
                session_id=session_id, timeout_minutes=self.session_timeout_minutes
            )

            # Create session
            session = Session(session_id=session_id, state=session_state)

            # Store session
            self.sessions[session_id] = session

            logger.info(f"Created new session: {session_id}")
            return session_id

        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise

    async def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get a session by ID.

        Args:
            session_id: Session ID to retrieve

        Returns:
            Session object if found, None otherwise
        """
        session = self.sessions.get(session_id)

        if session and session.is_expired():
            await self.cleanup_session(session_id)
            return None

        return session

    async def process_request(
        self, session_id: str, user_request: str
    ) -> Dict[str, Any]:
        """
        Process a user request within a session.

        Args:
            session_id: Session ID
            user_request: User's request text

        Returns:
            Response dictionary with results
        """
        start_time = datetime.utcnow()

        try:
            # Get session
            session = await self.get_session(session_id)
            if not session:
                return {
                    "success": False,
                    "error": f"Session {session_id} not found or expired",
                    "session_id": session_id,
                }

            # Update session state
            session.state.status = SessionStatus.ACTIVE
            session.state.current_task = user_request[:100]  # Truncate for storage
            session.state.last_activity = start_time

            # Build context
            context = await self.context_builder.build_context(session, user_request)

            # Build prompt
            prompt = self.prompt_builder.assemble_prompt(context, user_request)

            # For now, simulate LLM response (in real implementation, would call LLM)
            llm_response = await self._simulate_llm_response(prompt, user_request)

            # Parse response
            parsed_response = self.response_parser.parse_response(llm_response)

            # Create interaction record
            execution_time = int(
                (datetime.utcnow() - start_time).total_seconds() * 1000
            )
            interaction = Interaction(
                timestamp=start_time,
                user_request=user_request,
                llm_response=llm_response,
                parsed_response=parsed_response,
                execution_time_ms=execution_time,
            )

            # Add to session
            session.add_interaction(interaction)

            # Process any tool calls or claims
            await self._process_parsed_response(session, parsed_response)

            # Update session state
            session.state.status = SessionStatus.IDLE
            session.state.step_in_process = 0

            return {
                "success": True,
                "session_id": session_id,
                "response": llm_response,
                "parsed_response": parsed_response,
                "execution_time_ms": execution_time,
                "session_status": session.state.status.value,
            }

        except Exception as e:
            logger.error(f"Error processing request in session {session_id}: {e}")

            # Update session error state
            if session_id in self.sessions:
                session = self.sessions[session_id]
                session.state.status = SessionStatus.ERROR
                session.state.last_error = str(e)
                session.state.error_count += 1

            return {
                "success": False,
                "error": str(e),
                "session_id": session_id,
                "execution_time_ms": int(
                    (datetime.utcnow() - start_time).total_seconds() * 1000
                ),
            }

    async def cleanup_session(self, session_id: str) -> bool:
        """
        Clean up a session and free resources.

        Args:
            session_id: Session ID to cleanup

        Returns:
            True if cleanup successful
        """
        try:
            session = self.sessions.get(session_id)
            if session:
                # Persist session data if needed
                await self._persist_session(session)

                # Remove from active sessions
                del self.sessions[session_id]

                logger.info(f"Cleaned up session: {session_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {e}")
            return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all active sessions.

        Returns:
            List of session information dictionaries
        """
        sessions_info = []

        for session_id, session in self.sessions.items():
            sessions_info.append(
                {
                    "session_id": session_id,
                    "status": session.state.status.value,
                    "created_at": session.state.created_at.isoformat(),
                    "last_activity": session.state.last_activity.isoformat(),
                    "interaction_count": len(session.interactions),
                    "current_task": session.state.current_task,
                    "error_count": session.state.error_count,
                }
            )

        return sessions_info

    async def _simulate_llm_response(self, prompt: str, user_request: str) -> str:
        """
        Simulate LLM response for testing.
        In real implementation, this would call the actual LLM.
        """
        # Simple simulation based on request content
        if "research" in user_request.lower():
            return """I'll help you research this topic. Let me start by searching for relevant information and then create claims based on what I find.

<tool_calls>
  <invoke name="web_search">
    <parameter name="query">research topic information</parameter>
  </invoke>
</tool_calls>

Based on my research, I'll create claims to capture the key findings and support them with evidence."""

        elif "write code" in user_request.lower() or "code" in user_request.lower():
            return """I'll help you write code for this requirement. Let me follow the development process:

<tool_calls>
  <invoke name="write_code_file">
    <parameter name="filename">solution.py</parameter>
    <parameter name="code"># Solution code based on requirements
def main():
    print("Hello, World!")

if __name__ == "__main__":
    main()</parameter>
  </invoke>
</tool_calls>

I've created the code file and will now create claims about the solution."""

        elif "test" in user_request.lower():
            return """I'll help you test the code. Let me create test cases and validate the implementation:

<tool_calls>
  <invoke name="write_code_file">
    <parameter name="filename">test_solution.py</parameter>
    <parameter name="code"># Test cases for the solution
import unittest
from solution import main

class TestSolution(unittest.TestCase):
    def test_main(self):
        # Test the main function
        pass

if __name__ == "__main__":
    unittest.main()</parameter>
  </invoke>
</tool_calls>

I've created test cases and will run them to validate the code."""

        else:
            return """I understand your request. Let me help you work through this systematically using the available tools and skills to achieve the best outcome."""

    async def _process_parsed_response(
        self, session: Session, parsed_response: Dict[str, Any]
    ) -> None:
        """Process parsed LLM response for tool calls and claims."""
        try:
            # Process tool calls
            tool_calls = parsed_response.get("tool_calls", [])
            for tool_call in tool_calls:
                await self._execute_tool_call(tool_call)

            # Process claims
            claims = parsed_response.get("claims", [])
            for claim_data in claims:
                await self._create_claim(claim_data)

        except Exception as e:
            logger.error(f"Error processing parsed response: {e}")

    async def _execute_tool_call(self, tool_call: Dict[str, Any]) -> None:
        """Execute a tool call from parsed response."""
        # This would integrate with the actual tool execution system
        logger.info(f"Executing tool call: {tool_call}")

    async def _create_claim(self, claim_data: Dict[str, Any]) -> None:
        """Create a claim from parsed response."""
        try:
            await self.data_manager.create_claim(
                content=claim_data.get("content", ""),
                confidence=claim_data.get("confidence", 0.5),
                tags=claim_data.get("tags", []),
            )

            logger.info(f"Created claim from LLM response")

        except Exception as e:
            logger.error(f"Error creating claim: {e}")

    async def _persist_session(self, session: Session) -> None:
        """Persist session data for recovery."""
        # In a real implementation, this would save session state to database
        logger.debug(f"Persisting session {session.session_id}")

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop for expired sessions."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval_minutes * 60)
                await self._cleanup_expired_sessions()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def _cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions."""
        expired_sessions = [
            session_id
            for session_id, session in self.sessions.items()
            if session.is_expired()
        ]

        for session_id in expired_sessions:
            await self.cleanup_session(session_id)

        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

    async def _remove_oldest_idle_session(self) -> None:
        """Remove the oldest idle session to make room."""
        idle_sessions = [
            (session_id, session)
            for session_id, session in self.sessions.items()
            if session.state.status == SessionStatus.IDLE
        ]

        if idle_sessions:
            # Sort by last activity
            idle_sessions.sort(key=lambda x: x[1].state.last_activity)
            oldest_session_id = idle_sessions[0][0]

            await self.cleanup_session(oldest_session_id)
            logger.info(f"Removed oldest idle session: {oldest_session_id}")

    def get_harness_stats(self) -> Dict[str, Any]:
        """Get statistics about the agent harness."""
        total_sessions = len(self.sessions)
        active_sessions = sum(
            1
            for session in self.sessions.values()
            if session.state.status == SessionStatus.ACTIVE
        )
        idle_sessions = sum(
            1
            for session in self.sessions.values()
            if session.state.status == SessionStatus.IDLE
        )
        error_sessions = sum(
            1
            for session in self.sessions.values()
            if session.state.status == SessionStatus.ERROR
        )

        total_interactions = sum(
            len(session.interactions) for session in self.sessions.values()
        )

        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "idle_sessions": idle_sessions,
            "error_sessions": error_sessions,
            "total_interactions": total_interactions,
            "max_sessions": self.max_sessions,
            "session_timeout_minutes": self.session_timeout_minutes,
        }
