#!/usr/bin/env python3
"""
Comprehensive Test for Core Tools System Integration
Tests the complete Core Tools system including registry, processor, and context builder
"""

import sys
import os
import json
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import Core Tools system components
from src.tools.registry import ToolRegistry, register_tool, get_tool_registry
from src.processing.simple_llm_processor import SimpleLLMProcessor
from src.context.complete_context_builder import CompleteContextBuilder
from src.core.models import Claim, ClaimType
from src.interfaces.llm_interface import LLMInterface

class SimpleLLMProcessor(LLMInterface):
    """Real implementation of LLM interface for testing"""
    
    def generate_response(self, prompt: str) -> str:
        """Generate test response with tool calls"""
        print(f"LLM received prompt ({len(prompt)} chars) - returning mock tool calls")
        