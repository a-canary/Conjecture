"""
Process Layer - Conjecture 4-Layer Architecture

The Process Layer is the third layer in the Conjecture 4-layer architecture,
responsible for claim evaluation, instruction identification, and processing
workflow coordination.

Architecture Layers:
1. Data Layer - Data models, repositories, and storage
2. Context Layer - Context building and management
3. Process Layer - Claim evaluation and instruction processing
4. Interface Layer - User interfaces and CLI

Key Components:
- ProcessContextBuilder: Traverses claim graphs and builds processing contexts
- ProcessLLMProcessor: Evaluates claims and identifies instructions
- Process Models: Data structures for processing results and instructions

Integration Points:
- Data Layer: Access to claim repositories and storage
- Context Layer: Utilizes context building capabilities
- LLM Layer: Interfaces with language models for evaluation
- Interface Layer: Provides processing capabilities to user interfaces
"""

from .context_builder import ProcessContextBuilder
from .llm_processor import ProcessLLMProcessor
from .models import (
    ContextResult,
    ProcessingResult,
    Instruction,
    InstructionType,
    ProcessingStatus
)

__all__ = [
    "ProcessContextBuilder",
    "ProcessLLMProcessor", 
    "ContextResult",
    "ProcessingResult",
    "Instruction",
    "InstructionType",
    "ProcessingStatus"
]

__version__ = "0.1.0"