"""
Dirty Flag CLI Commands
Command-line interface for dirty flag system management
"""

import asyncio
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from typing import List, Optional
import json
import sys
import os
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.models import Claim, DirtyReason
from core.dirty_flag import DirtyFlagSystem
from processing.dirty_evaluator import DirtyEvaluator, DirtyEvaluationConfig
from processing.simplified_llm_manager import SimplifiedLLMManager as LLMManager
from config.dirty_flag_config import DirtyFlagConfig, DirtyFlagConfigManager

# Rich console for beautiful output
console = Console()
error_console = Console(stderr=True)

# Create Typer app for dirty flag commands
dirty_app = typer.Typer(
    name="dirty", help="Dirty flag system management commands", no_args_is_help=True
)

# Global instances
_dirty_flag_system: Optional[DirtyFlagSystem] = None
_dirty_evaluator: Optional[DirtyEvaluator] = None
_config_manager: Optional[DirtyFlagConfigManager] = None


def get_dirty_flag_system() -> DirtyFlagSystem:
    """Get or create dirty flag system instance."""
    global _dirty_flag_system
    if _dirty_flag_system is None:
        config = get_config_manager().get_config()
        _dirty_flag_system = DirtyFlagSystem(
            confidence_threshold=config.confidence_threshold,
            cascade_depth=config.cascade_depth,
        )
    return _dirty_flag_system


def get_dirty_evaluator() -> DirtyEvaluator:
    """Get or create dirty evaluator instance."""
    global _dirty_evaluator
    if _dirty_evaluator is None:
        config = get_config_manager().get_config()

        # Create evaluation config
        eval_config = DirtyEvaluationConfig(
            batch_size=config.batch_size,
            max_parallel_batches=config.max_parallel_batches,
            confidence_threshold=config.confidence_threshold,
            confidence_boost_factor=config.confidence_boost_factor,
            enable_two_pass=config.two_pass_evaluation,
            relationship_threshold=config.relationship_threshold,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
        )

        # Initialize LLM manager
        llm_manager = LLMManager()

        # Create evaluator
        dirty_flag_system = get_dirty_flag_system()
        _dirty_evaluator = DirtyEvaluator(llm_manager, dirty_flag_system, eval_config)

    return _dirty_evaluator


def get_config_manager() -> DirtyFlagConfigManager:
    """Get configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = DirtyFlagConfigManager()
    return _config_manager
