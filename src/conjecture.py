"""
Conjecture: Async Evidence-Based AI Reasoning System
OPTIMIZED: Enhanced with comprehensive performance monitoring
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, AsyncGenerator, Callable
from pathlib import Path
import logging
from functools import lru_cache
import hashlib

from src.core.models import Claim, ClaimState, ClaimFilter

# Define ExplorationResult for backward compatibility
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class ExplorationResult:
    """Result of claim exploration"""
    
    success: bool
    claim_id: str
    original_confidence: float
    new_confidence: float
    state: ClaimState
    evaluation_summary: str
    supporting_evidence: List[str] = field(default_factory=list)
    counter_evidence: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exploration result to dictionary"""
        return {
            "success": self.success,
            "claim_id": self.claim_id,
            "original_confidence": self.original_confidence,
            "new_confidence": self.new_confidence,
            "state": self.state.value,
            "evaluation_summary": self.evaluation_summary,
            "supporting_evidence": self.supporting_evidence,
            "counter_evidence": self.counter_evidence,
            "recommendations": self.recommendations,
            "processing_time": self.processing_time,
            "metadata": self.metadata,
            "errors": self.errors,
        }
from src.config.unified_config import UnifiedConfig as Config
from src.processing.unified_bridge import UnifiedLLMBridge as LLMBridge, LLMRequest
from src.processing.simplified_llm_manager import get_simplified_llm_manager
from src.processing.enhanced_llm_router import get_enhanced_llm_router
from src.processing.async_eval import AsyncClaimEvaluationService
from src.processing.context_collector import ContextCollector
from src.processing.tool_manager import DynamicToolCreator
from src.data.repositories import get_data_manager, RepositoryFactory
from src.monitoring import get_performance_monitor, monitor_performance
from src.interfaces.processing_interface import (
    ProcessingInterface,
    EvaluationResult,
    ToolResult,
    Context as ProcessingContext,
    Session,
    SessionState,
    ProcessingEvent,
    EventType
)

class Conjecture(ProcessingInterface):
    """
    Enhanced Conjecture with Async Claim Evaluation and Dynamic Tool Creation
    Implements the full architecture described in the specifications
    Now implements ProcessingInterface for clean architecture separation
    """

    def __init__(self, config: Optional[Config] = None):
        """OPTIMIZED: Initialize Enhanced Conjecture with performance monitoring"""
        self.config = config or Config()

        # Initialize performance monitor
        self.performance_monitor = get_performance_monitor()

        # Initialize data layer with repository pattern
        self.data_manager = get_data_manager(use_mock_embeddings=False)