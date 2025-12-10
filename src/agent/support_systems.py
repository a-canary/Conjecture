"""
Support Systems - Data collection, context building, and persistence for Conjecture.
Handles the data layer and context management separate from core orchestration.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from ..data.data_manager import DataManager
from ..core.models import Claim

logger = logging.getLogger(__name__)

class Context:
    """Context information for LLM prompt building."""

    def __init__(self):
        self.relevant_claims: List[Claim] = []
        self.skill_templates: List[Dict[str, Any]] = []
        self.available_tools: List[Dict[str, Any]] = []
        self.session_history: List[Dict[str, Any]] = []
        self.current_focus: Optional[str] = None
        self.context_window_size: int = 8000  # tokens

    def add_claim(self, claim: Claim) -> None:
        """Add a claim to the context."""
        self.relevant_claims.append(claim)

    def add_skill_template(self, skill: Dict[str, Any]) -> None:
        """Add a skill template to the context."""
        self.skill_templates.append(skill)

    def add_tool(self, tool: Dict[str, Any]) -> None:
        """Add a tool to the context."""
        self.available_tools.append(tool)

    def estimate_size(self) -> int:
        """Estimate the context size in tokens."""
        # Rough estimation (could be made more accurate)
        size = 0

        # Claims
        for claim in self.relevant_claims:
            size += len(claim.content) // 4  # Rough token estimation

        # Skills
        for skill in self.skill_templates:
            size += len(str(skill)) // 4

        # Tools
        for tool in self.available_tools:
            size += len(str(tool)) // 4

        # History
        for interaction in self.session_history:
            size += len(str(interaction)) // 4

        return size

    def trim_to_fit(self) -> None:
        """Trim context to fit within window size."""
        while self.estimate_size() > self.context_window_size:
            # Remove oldest claims first
            if self.relevant_claims:
                self.relevant_claims.pop(0)
            elif self.session_history:
                self.session_history.pop(0)
            else:
                break

class ContextBuilder:
    """
    Builds context for LLM prompts by collecting relevant information.
    """

    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.skill_templates = self._load_skill_templates()
        self.available_tools = self._load_available_tools()
        self.context_cache = {}
        self.cache_ttl_minutes = 10

    async def initialize(self) -> None:
        """Initialize the context builder."""
        try:
            # Load any persistent data
            logger.info("Context builder initialized")
        except Exception as e:
            logger.error(f"Failed to initialize context builder: {e}")
            raise

    async def build_context(self, session, user_request: str) -> Context:
        """
        Build context for a user request.

        Args:
            session: User session
            user_request: User's request

        Returns:
            Context object with relevant information
        """
        try:
            # Check cache first
            cache_key = self._generate_cache_key(user_request, session.session_id)
            cached_context = self._get_from_cache(cache_key)
            if cached_context:
                return cached_context

            # Create new context
            context = Context()

            # Add relevant claims
            await self._add_relevant_claims(context, user_request)

            # Add relevant skill templates
            self._add_relevant_skills(context, user_request)

            # Add available tools
            self._add_available_tools(context)

            # Add session history
            self._add_session_history(context, session)

            # Set current focus
            context.current_focus = user_request[:100]  # Truncate for storage

            # Trim to fit context window
            context.trim_to_fit()

            # Cache the context
            self._add_to_cache(cache_key, context)

            return context

        except Exception as e:
            logger.error(f"Error building context: {e}")
            # Return minimal context on error
            return self._create_minimal_context(user_request)

    async def _add_relevant_claims(self, context: Context, user_request: str) -> None:
        """Add relevant claims to context."""
        try:
            # Search for claims similar to the request
            similar_claims = await self.data_manager.search_similar(
                user_request, limit=10
            )

            # Add claims to context
            for claim in similar_claims:
                context.add_claim(claim)

        except Exception as e:
            logger.error(f"Error adding relevant claims: {e}")

    def _add_relevant_skills(self, context: Context, user_request: str) -> None:
        """Add relevant skill templates to context."""
        try:
            # Simple keyword-based skill selection
            request_lower = user_request.lower()

            # Research skill
            if any(
                keyword in request_lower
                for keyword in ["research", "search", "find", "look up", "investigate"]
            ):
                context.add_skill_template(self.skill_templates["research"])

            # Write code skill
            if any(
                keyword in request_lower
                for keyword in ["write code", "code", "program", "implement", "develop"]
            ):
                context.add_skill_template(self.skill_templates["write_code"])

            # Test code skill
            if any(
                keyword in request_lower
                for keyword in ["test", "validate", "check", "verify", "test code"]
            ):
                context.add_skill_template(self.skill_templates["test_code"])

            # End claim evaluation skill
            if any(
                keyword in request_lower
                for keyword in ["evaluate", "assess", "review", "analyze claim"]
            ):
                context.add_skill_template(self.skill_templates["end_claim_eval"])

        except Exception as e:
            logger.error(f"Error adding relevant skills: {e}")

    def _add_available_tools(self, context: Context) -> None:
        """Add available tools to context."""
        try:
            for tool in self.available_tools:
                context.add_tool(tool)
        except Exception as e:
            logger.error(f"Error adding available tools: {e}")

    def _add_session_history(self, context: Context, session) -> None:
        """Add recent session history to context."""
        try:
            recent_interactions = session.get_recent_interactions(5)

            for interaction in recent_interactions:
                context.session_history.append(
                    {
                        "timestamp": interaction.timestamp.isoformat(),
                        "user_request": interaction.user_request,
                        "llm_response": interaction.llm_response[:200] + "..."
                        if len(interaction.llm_response) > 200
                        else interaction.llm_response,
                    }
                )

        except Exception as e:
            logger.error(f"Error adding session history: {e}")

    def _load_skill_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load skill templates."""
        return {
            "research": {
                "name": "Research",
                "description": "Guide for gathering information and creating claims",
                "steps": [
                    "Search web for relevant information",
                    "Read relevant files and documents",
                    "Create claims for key findings",
                    "Support claims with collected evidence",
                ],
                "suggested_tools": [
                    "WebSearch",
                    "ReadFiles",
                    "CreateClaim",
                    "ClaimSupport",
                ],
                "example_usage": "To research a topic: use WebSearch to find information, ReadFiles to examine documents, CreateClaim to capture findings, ClaimSupport to link evidence",
            },
            "write_code": {
                "name": "WriteCode",
                "description": "Guide for code development and testing",
                "steps": [
                    "Understand the requirements clearly",
                    "Design a solution approach",
                    "Write the code implementation",
                    "Test the code works correctly",
                    "Create claims about the solution",
                ],
                "suggested_tools": [
                    "ReadFiles",
                    "WriteCodeFile",
                    "TestCode",
                    "CreateClaim",
                ],
                "example_usage": "To write code: understand requirements, design solution, write implementation, test functionality, create claims about the solution",
            },
            "test_code": {
                "name": "TestCode",
                "description": "Guide for validation and quality assurance",
                "steps": [
                    "Write comprehensive test cases",
                    "Run the tests to validate functionality",
                    "Fix any issues that are found",
                    "Create claims about test results",
                ],
                "suggested_tools": ["WriteCodeFile", "ReadFiles", "CreateClaim"],
                "example_usage": "To test code: write test cases, run tests, fix failures, create claims about test results and quality",
            },
            "end_claim_eval": {
                "name": "EndClaimEval",
                "description": "Guide for knowledge assessment and evaluation",
                "steps": [
                    "Review all supporting evidence",
                    "Check for contradictions or gaps",
                    "Update confidence scores appropriately",
                    "Note areas needing more research",
                ],
                "suggested_tools": ["ReadFiles", "CreateClaim", "ClaimSupport"],
                "example_usage": "To evaluate claims: review evidence, check consistency, update confidence, identify knowledge gaps",
            },
        }

    def _load_available_tools(self) -> List[Dict[str, Any]]:
        """Load available tools."""
        return [
            {
                "name": "WebSearch",
                "description": "Search the web for information",
                "parameters": {"query": "string - search query"},
                "example": "WebSearch(query='python weather api')",
            },
            {
                "name": "ReadFiles",
                "description": "Read contents of files",
                "parameters": {"file_paths": "list - list of file paths to read"},
                "example": "ReadFiles(file_paths=['data.txt', 'config.json'])",
            },
            {
                "name": "WriteCodeFile",
                "description": "Write code to a file",
                "parameters": {
                    "filename": "string - name of the file",
                    "code": "string - code content to write",
                },
                "example": "WriteCodeFile(filename='solution.py', code='print(\"Hello\")')",
            },
            {
                "name": "CreateClaim",
                "description": "Create a new claim with confidence score",
                "parameters": {
                    "content": "string - claim content",
                    "confidence": "float - confidence score (0.0-1.0)",
                    "tags": "list - optional tags",
                },
                "example": "CreateClaim(content='Python is popular', confidence=0.9, tags=['programming'])",
            },
            {
                "name": "ClaimSupport",
                "description": "Link evidence to support a claim",
                "parameters": {
                    "supporter_id": "string - ID of supporting claim",
                    "supported_id": "string - ID of supported claim",
                },
                "example": "ClaimSupport(supporter_id='c0000001', supported_id='c0000002')",
            },
        ]

    def _create_minimal_context(self, user_request: str) -> Context:
        """Create a minimal context when errors occur."""
        context = Context()
        context.current_focus = user_request[:100]

        # Add basic tools
        self._add_available_tools(context)

        return context

    def _generate_cache_key(self, user_request: str, session_id: str) -> str:
        """Generate cache key for context."""
        import hashlib

        key_data = f"{user_request}:{session_id}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[Context]:
        """Get context from cache."""
        if cache_key in self.context_cache:
            cached_item = self.context_cache[cache_key]
            timestamp = cached_item["timestamp"]

            # Check if cache is still valid
            if datetime.utcnow() - timestamp < timedelta(
                minutes=self.cache_ttl_minutes
            ):
                return cached_item["context"]
            else:
                # Remove expired cache item
                del self.context_cache[cache_key]

        return None

    def _add_to_cache(self, cache_key: str, context: Context) -> None:
        """Add context to cache."""
        self.context_cache[cache_key] = {
            "timestamp": datetime.utcnow(),
            "context": context,
        }

        # Maintain cache size
        if len(self.context_cache) > 100:
            # Remove oldest items
            oldest_keys = sorted(
                self.context_cache.keys(),
                key=lambda k: self.context_cache[k]["timestamp"],
            )[:20]

            for key in oldest_keys:
                del self.context_cache[key]

    def clear_cache(self) -> None:
        """Clear the context cache."""
        self.context_cache.clear()
        logger.info("Context cache cleared")

    def get_context_stats(self) -> Dict[str, Any]:
        """Get statistics about context building."""
        return {
            "skill_templates_count": len(self.skill_templates),
            "available_tools_count": len(self.available_tools),
            "cache_size": len(self.context_cache),
            "cache_ttl_minutes": self.cache_ttl_minutes,
        }

class DataManager:
    """
    Enhanced data manager for support systems.
    Extends the base data manager with additional functionality.
    """

    def __init__(self, base_data_manager):
        self.base_data_manager = base_data_manager
        self.persistence_cache = {}

    async def persist_session_data(
        self, session_id: str, session_data: Dict[str, Any]
    ) -> bool:
        """Persist session data for recovery."""
        try:
            # In a real implementation, this would save to database
            self.persistence_cache[session_id] = {
                "data": session_data,
                "timestamp": datetime.utcnow(),
            }
            return True
        except Exception as e:
            logger.error(f"Error persisting session data: {e}")
            return False

    async def load_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load persisted session data."""
        try:
            if session_id in self.persistence_cache:
                return self.persistence_cache[session_id]["data"]
            return None
        except Exception as e:
            logger.error(f"Error loading session data: {e}")
            return None

    async def cleanup_expired_data(self, max_age_hours: int = 24) -> int:
        """Clean up expired persisted data."""
        try:
            expired_keys = []
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)

            for key, value in self.persistence_cache.items():
                if value["timestamp"] < cutoff_time:
                    expired_keys.append(key)

            for key in expired_keys:
                del self.persistence_cache[key]

            return len(expired_keys)
        except Exception as e:
            logger.error(f"Error cleaning up expired data: {e}")
            return 0

# Import required modules
from datetime import timedelta
