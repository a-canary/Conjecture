# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Agent Package for Conjecture AI Agent System.

This package provides the orchestration layer that coordinates between the LLM, tools,
skills, and data systems with clear separation of concerns.

Components:
- AgentFramework: Core orchestration and session management
- SupportSystems: Data collection, context building, and persistence
- PromptSystem: LLM prompt assembly and response parsing

Usage:
    from src.agent import AgentFramework
    from src.data import DataManager

    # Initialize components
    data_manager = DataManager()
    agent = AgentFramework(data_manager)
    await agent.initialize()

    # Create session and process request
    session_id = await agent.create_session()
    response = await agent.process_request(session_id, "Research Python weather APIs")
"""

from .agent_framework import AgentFramework, Session, SessionState, Interaction, SessionStatus

# Backward compatibility alias (deprecated)
AgentHarness = AgentFramework
from .support_systems import ContextBuilder, Context, DataManager as SupportDataManager
from .prompt_system import PromptBuilder, ResponseParser

__version__ = "1.0.0"
__author__ = "Conjecture Team"

__all__ = [
    # Core orchestration
    "AgentFramework",
    "AgentHarness",  # Deprecated alias
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