#!/usr/bin/env python3
"""
Base CLI functionality for Conjecture - Test Version
"""

import asyncio
import json
import os
import sqlite3
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config import get_config, validate_config


class BaseCLI(ABC):
    """Base class for all CLI implementations with common functionality."""

    def __init__(self, name: str = "conjecture", help_text: str = "Conjecture CLI"):
        self.name = name
        self.help_text = help_text
        self.console = Console()
        self.error_console = Console(stderr=True)
        self.embedding_model = None
        self.current_provider_config = None
        self.db_path = "data/conjecture.db"

    def _get_backend_type(self) -> str:
        """Get the backend type for this CLI."""
        return self.__class__.__name__.replace("CLI", "").lower()

    def _init_services(self):
        """Initialize services common to all backends."""
        # Validate configuration first
        result = validate_config()

        if not result:  # validate_config returns a bool
            self.console.print("[bold red]❌ Configuration Required[/bold red]")
            self.console.print(
                "Please configure at least one provider in your .env file"
            )
            raise SystemExit(1)

        # Get the best configured provider
        self.current_provider_config = self._get_configured_provider()

        # Create data directory
        os.makedirs("data", exist_ok=True)

        # Initialize sentence transformer
        if self.embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.console.print("[green][OK][/green] Model loaded successfully")
            except ImportError:
                self.console.print(
                    "[red]Error: sentence-transformers not installed[/red]"
                )
                raise SystemExit(1)

    def _init_database(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS claims (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                confidence REAL NOT NULL,
                user_id TEXT NOT NULL,
                embedding BLOB,
                metadata TEXT,
                tags TEXT,
                is_dirty BOOLEAN DEFAULT 1,
                dirty_reason TEXT,
                dirty_priority INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        return self.embedding_model.encode(text)

    def _save_claim(
        self,
        content: str,
        confidence: float,
        user_id: str,
        metadata: dict = None,
        tags: list = None,
    ) -> str:
        """Save claim to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Generate embedding
        embedding = self._generate_embedding(content)
        embedding_bytes = embedding.tobytes()

        # Generate ID
        claim_id = f"c{int(time.time() * 1000) % 1000000000:07d}"

        # Save claim
        cursor.execute(
            """
            INSERT INTO claims (id, content, confidence, user_id, embedding, metadata, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                claim_id,
                content,
                confidence,
                user_id,
                embedding_bytes,
                json.dumps(metadata) if metadata else None,
                json.dumps(tags) if tags else None,
            ),
        )

        conn.commit()
        conn.close()
        return claim_id

    # Abstract methods that must be implemented by backends
    @abstractmethod
    def create_claim(
        self,
        content: str,
        confidence: float,
        user_id: str,
        analyze: bool = False,
        tags: list = None,
        **kwargs,
    ) -> str:
        """Create a new claim."""
        pass

    @abstractmethod
    def get_claim(self, claim_id: str) -> Optional[dict]:
        """Get a claim by ID."""
        pass

    @abstractmethod
    def search_claims(self, query: str, limit: int = 10, **kwargs) -> List[dict]:
        """Search claims by content."""
        pass

    @abstractmethod
    def analyze_claim(self, claim_id: str, **kwargs) -> Dict[str, Any]:
        """Analyze a claim using backend-specific services."""
        pass

def process_prompt(
        self, prompt_text: str, confidence: float = 0.8, verbose: int = 0, **kwargs
    ) -> Dict[str, Any]:
        """Process user prompt as claim with dirty evaluation."""
        from enum import Enum
        
        class DirtyReason(str, Enum):
            NEW_CLAIM_ADDED = "new_claim_added"
            CONFIDENCE_THRESHOLD = "confidence_threshold"
            SUPPORTING_CLAIM_CHANGED = "supporting_claim_changed"
            RELATIONSHIP_CHANGED = "relationship_changed"
            MANUAL_MARK = "manual_mark"
            BATCH_EVALUATION = "batch_evaluation"
            SYSTEM_TRIGGER = "system_trigger"

        # Get workspace context from config
        config = get_config()
        workspace = config.workspace
        user = config.user
        team = config.team

        # Create tags with workspace context
        tags = ["user-prompt", f"workspace-{workspace}", f"user-{user}", f"team-{team}"]

        if verbose >= 1:
            self.console.print(
                f"[dim]🔧 Creating claim with context: {workspace}/{team}/{user}[/dim]"
            )
            self.console.print(f"[dim]🔧 Tags: {tags}[/dim]")

        # Create claim with user-prompt tags
        claim_id = self.create_claim(
            content=prompt_text, confidence=confidence, user_id=user, tags=tags
        )

        if verbose >= 1:
            self.console.print(
                f"[dim]🔧 Marking claim {claim_id} as dirty for evaluation[/dim]"
            )

        # Mark as dirty for evaluation (default priority=10)
        self.mark_claim_dirty(claim_id, DirtyReason.NEW_CLAIM_ADDED, priority=10)

        if verbose >= 1:
            self.console.print(f"[dim]🔧 Starting dirty evaluation...[/dim]")

        # Process dirty evaluation (mock for now)
        evaluation_result = self._mock_evaluate_claim(claim_id)

        if verbose >= 1:
            self.console.print(
                f"[dim]🔧 Evaluation complete, confidence: {evaluation_result['final_confidence']:.1%}[/dim]"
            )

        # Check confidence threshold for user response
        if evaluation_result["final_confidence"] > 0.90:
            if verbose >= 2:
                self.console.print(
                    f"[blue]📋 High-confidence claim achieved: {evaluation_result['final_confidence']:.1%}[/blue]"
                )
                self.console.print(
                    f"[blue]📋 Claim details: {evaluation_result['summary']}[/blue]"
                )

            if verbose >= 1:
                self.console.print(
                    f"[dim]🔧 Generating user response with TellUser tool...[/dim]"
                )

            user_response = self._generate_user_response(prompt_text, evaluation_result)

            # Always show final response (even at verbose=0)
            self.console.print(user_response)

            evaluation_result["user_response"] = user_response
        else:
            if verbose >= 1:
                self.console.print(
                    f"[dim]🔧 Confidence {evaluation_result['final_confidence']:.1%} below 90% threshold - no user response[/dim]"
                )

        return {
            "claim_id": claim_id,
            "original_prompt": prompt_text,
            "workspace_context": f"{workspace}/{team}/{user}",
            "tags": tags,
            "evaluation_result": evaluation_result,
            "final_status": "processed",
        }

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available and properly configured."""
        pass

    def _get_configured_provider(self) -> Optional[Dict[str, Any]]:
        """Get configured provider in legacy format."""
        # For now, return a simple mock provider config
        # In a real implementation, this would integrate with the provider system
        config = get_config()
        if config.provider_api_key or "localhost" in config.provider_api_url:
            return {
                "name": config.llm_provider,
                "type": "local" if "localhost" in config.provider_api_url else "cloud",
                "base_url": config.provider_api_url,
                "model": config.provider_model,
                "api_key": config.provider_api_key,
            }
        return None

    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the current backend."""
        return {
            "name": self._get_backend_type(),
            "configured": self.is_available(),
            "provider": self.current_provider_config.get("name")
            if self.current_provider_config
            else None,
            "type": self.current_provider_config.get("type")
            if self.current_provider_config
            else None,
            "model": self.current_provider_config.get("model")
            if self.current_provider_config
            else None,
        }

    def mark_claim_dirty(self, claim_id: str, reason, priority: int = 0):
        """Mark a claim as dirty in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE claims 
            SET is_dirty = 1, dirty_reason = ?, dirty_priority = ?
            WHERE id = ?
        """,
            (reason.value, priority, claim_id),
        )

        conn.commit()
        conn.close()

    def _mock_evaluate_claim(self, claim_id: str) -> Dict[str, Any]:
        """Mock evaluation for testing - replace with actual evaluation logic."""
        import random

        # Get the claim
        claim = self.get_claim(claim_id)
        if not claim:
            raise ValueError(f"Claim {claim_id} not found")

        # Mock evaluation - in real implementation this would use LLM
        final_confidence = random.uniform(0.85, 0.96)

        return {
            "claim_id": claim_id,
            "original_confidence": claim["confidence"],
            "final_confidence": final_confidence,
            "evaluation_steps": [
                "Analyzed claim content",
                "Checked supporting evidence",
                "Validated reasoning",
            ],
            "summary": f"Evaluated claim: {claim['content'][:50]}...",
            "status": "validated" if final_confidence > 0.90 else "needs_review",
        }

    def _generate_user_response(
        self, prompt_text: str, evaluation_result: Dict[str, Any]
    ) -> str:
        """Generate user response using TellUser tool format."""
        # Mock response - in real implementation this would use LLM
        confidence = evaluation_result["final_confidence"]

        response = f"[SUCCESS] Based on your prompt about '{prompt_text[:50]}...', "
        response += f"I've analyzed the requirements and validated the approach. "
        response += (
            f"The system confidence in this assessment is high at {confidence:.1%}. "
        )

        if confidence > 0.95:
            response += "This claim has been thoroughly validated and can be trusted."
        elif confidence > 0.90:
            response += (
                "This claim shows strong validation with good supporting evidence."
            )

        return response


class ClaimValidationError(Exception):
    """Exception raised for claim validation errors."""

    pass


class DatabaseError(Exception):
    """Exception raised for database operations."""

    pass


class BackendNotAvailableError(Exception):
    """Exception raised when backend is not available."""

    pass
