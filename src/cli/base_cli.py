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
from data.sqlite_manager import SQLiteManager
from core.models import Claim, ClaimScope, DirtyReason, generate_claim_id


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
        # Initialize SQLite database manager
        self.sqlite_manager = SQLiteManager(self.db_path)
        self._workspace = None

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
        """Initialize SQLite database with scope support."""
        # Initialize the SQLite database manager
        asyncio.create_task(self.sqlite_manager.initialize())

        # Set workspace context
        config = get_config()
        self._workspace = config.workspace

    def _get_claim(self, claim_id: str) -> Optional[dict]:
        """Get claim from database using SQLiteManager."""
        # Use asyncio to run the async method
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            claim_dict = loop.run_until_complete(
                self.sqlite_manager.get_claim(claim_id)
            )
            return claim_dict
        finally:
            loop.close()

    def _search_claims(self, query: str, limit: int = 10) -> List[dict]:
        """Search claims using vector similarity."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all claims with embeddings
        cursor.execute("""
            SELECT id, content, confidence, user_id, metadata, tags, scope, is_dirty, dirty_reason, dirty_priority, created_at, embedding
            FROM claims
        """)

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return []

        # Generate query embedding
        query_embedding = self._generate_embedding(query)

        # Calculate similarities
        results = []
        for row in rows:
            claim_embedding = np.frombuffer(row[10], dtype=np.float32)

            # Calculate cosine similarity
            similarity = np.dot(query_embedding, claim_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(claim_embedding)
            )

            results.append(
                {
                    "id": row[0],
                    "content": row[1],
                    "confidence": row[2],
                    "user_id": row[3],
                    "metadata": json.loads(row[4]) if row[4] else None,
                    "tags": json.loads(row[5]) if row[5] else [],
                    "scope": row[6],
                    "is_dirty": bool(row[7]),
                    "dirty_reason": row[8],
                    "dirty_priority": row[9],
                    "created_at": row[10],
                    "similarity": float(similarity),
                }
            )

        # Sort by similarity and return top results
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]

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
        scope: str = "global",
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
            INSERT INTO claims (id, content, confidence, user_id, embedding, metadata, tags, scope)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                claim_id,
                content,
                confidence,
                user_id,
                embedding_bytes,
                json.dumps(metadata) if metadata else None,
                json.dumps(tags) if tags else None,
                scope,
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
        scope: str = "global",
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
        """Process user prompt as claim with dirty evaluation using local database."""
        # Get workspace context from config
        config = get_config()
        workspace = config.workspace
        user = config.user
        team = config.team

        # Use simplified scope system - default to USER_WORKSPACE for security
        scope = ClaimScope.USER_WORKSPACE.value.format(workspace=workspace)

        # Create tags with user-prompt type only (scope handles context)
        tags = ["user-prompt"]

        if verbose >= 1:
            self.console.print(
                f"[dim]🔧 Creating claim with context: {workspace}/{team}/{user}[/dim]"
            )
            self.console.print(f"[dim]🔧 Scope: {scope}[/dim]")
            self.console.print(f"[dim]🔧 Tags: {tags}[/dim]")

        # Initialize database if not already done
        if not self.sqlite_manager.connection:
            asyncio.create_task(self.sqlite_manager.initialize())
            self._workspace = workspace

        # Create claim using SQLiteManager
        claim = Claim(
            id=generate_claim_id(),
            content=prompt_text,
            confidence=confidence,
            tags=tags,
            scope=ClaimScope.USER_WORKSPACE,  # Most restrictive by default
            is_dirty=True,
            dirty=True,  # Backward compatibility
            dirty_reason=DirtyReason.NEW_CLAIM_ADDED,
            dirty_priority=10,
        )

        # Save claim to SQLite database
        asyncio.create_task(self.sqlite_manager.create_claim(claim))
        claim_id = claim.id

        if verbose >= 1:
            self.console.print(
                f"[dim]🔧 Marking claim {claim_id} as dirty for evaluation[/dim]"
            )

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
            "scope": scope,
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
        asyncio.create_task(
            self.sqlite_manager.mark_claim_dirty(claim_id, reason, priority)
        )

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
