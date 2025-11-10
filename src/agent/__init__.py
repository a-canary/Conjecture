"""
Agent Package for Conjecture AI Agent System.

This package provides the orchestration layer that coordinates between the LLM, tools, 
skills, and data systems with clear separation of concerns.

Components:
- AgentHarness: Core orchestration and session management
- SupportSystems: Data collection, context building, and persistence
- PromptSystem: LLM prompt assembly and response parsing

Usage:
    from src.agent import AgentHarness
    from src.data import DataManager
    
    # Initialize components
    data_manager = DataManager()
    agent = AgentHarness(data_manager)
    await agent.initialize()
    
    # Create session and process request
    session_id = await agent.create_session()
    response = await agent.process_request(session_id, "Research Python weather APIs")
"""

from .agent_harness import AgentHarness, Session, SessionState, Interaction, SessionStatus
from .support_systems import ContextBuilder, Context, DataManager as SupportDataManager
from .prompt_system import PromptBuilder, ResponseParser

__version__ = "1.0.0"
__author__ = "Conjecture Team"

__all__ = [
    # Core orchestration
    "AgentHarness",
    "Session",
    "SessionState", 
    "Interaction",
    "SessionStatus",
    
    # Support systems
    "ContextBuilder",
    "Context",
    "SupportDataManager",
    
    # Prompt system
    "PromptBuilder",
    "ResponseParser",
]