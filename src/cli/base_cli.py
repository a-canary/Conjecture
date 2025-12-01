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

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, TextColumn
from rich.table import Table

from config.config import get_config, validate_config
from data.sqlite_manager import SQLiteManager
from core.models import Claim, ClaimScope, DirtyReason, generate_claim_id
from llm.provider_manager import get_provider_manager, process_prompt_with_failover


class BaseCLI:
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
        # Initialize provider manager
        self.provider_manager = get_provider_manager()
        # Initialize services (embedding model, etc.)
        self._init_services()

    def _get_backend_type(self) -> str:
        """Get the backend type for this CLI."""
        return self.__class__.__name__.replace("CLI", "").lower()

    def _init_services(self):
        """Initialize services common to all backends."""
        # Initialize provider manager with failover support
        self.provider_manager = get_provider_manager()

        # Check if any providers are available
        if not self.provider_manager.providers:
            self.console.print(
                "[bold red][ERROR] No LLM Providers Configured[/bold red]"
            )
            self.console.print(
                "Please configure providers in ~/.conjecture/config.json or workspace/.conjecture/config.json"
            )
            raise SystemExit(1)

        # Get current provider
        current_provider = self.provider_manager.get_available_provider()
        if not current_provider:
            self.console.print(
                "[bold red][ERROR] No Available LLM Providers[/bold red]"
            )
            raise SystemExit(1)

        self.current_provider_config = {
            "name": current_provider.name,
            "url": current_provider.url,
            "model": current_provider.model,
        }

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
        # Initialize the SQLite database manager synchronously
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, schedule the task
                asyncio.ensure_future(self.sqlite_manager.initialize())
            else:
                # If no loop running, run synchronously
                loop.run_until_complete(self.sqlite_manager.initialize())
        except RuntimeError:
            # No event loop exists, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.sqlite_manager.initialize())
            finally:
                pass  # Keep loop for future use

        # Set workspace context
        config = get_config()
        self._workspace = config.workspace

    def _get_claim(self, claim_id: str) -> Optional[dict]:
        """Get claim from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, content, confidence, state, tags, scope, is_dirty, 
                   dirty_reason, dirty_priority, created, created_by
            FROM claims
            WHERE id = ?
        """,
            (claim_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return {
            "id": row[0],
            "content": row[1],
            "confidence": row[2],
            "state": row[3],
            "tags": json.loads(row[4]) if row[4] else [],
            "scope": row[5],
            "is_dirty": bool(row[6]),
            "dirty_reason": row[7],
            "dirty_priority": row[8],
            "created_at": row[9],
            "created_by": row[10],
        }

    def _search_claims(self, query: str, limit: int = 10) -> List[dict]:
        """Search claims using vector similarity."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all claims with embeddings (using correct schema)
        cursor.execute("""
            SELECT id, content, confidence, state, tags, scope, is_dirty, dirty_reason, dirty_priority, created, embedding, created_by
            FROM claims
            WHERE embedding IS NOT NULL
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
            try:
                # Embedding is stored as JSON array
                claim_embedding = np.array(json.loads(row[10]), dtype=np.float32)

                # Calculate cosine similarity
                norm_query = np.linalg.norm(query_embedding)
                norm_claim = np.linalg.norm(claim_embedding)

                if norm_query > 0 and norm_claim > 0:
                    similarity = np.dot(query_embedding, claim_embedding) / (
                        norm_query * norm_claim
                    )
                else:
                    similarity = 0.0

                results.append(
                    {
                        "id": row[0],
                        "content": row[1],
                        "confidence": row[2],
                        "state": row[3],
                        "tags": json.loads(row[4]) if row[4] else [],
                        "scope": row[5],
                        "is_dirty": bool(row[6]),
                        "dirty_reason": row[7],
                        "dirty_priority": row[8],
                        "created_at": row[9],
                        "created_by": row[11],
                        "similarity": float(similarity),
                    }
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                # Skip claims with invalid embeddings
                continue

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
        scope: str = "user-workspace",
    ) -> str:
        """Save claim to database using SQLiteManager schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Ensure table exists with correct schema
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS claims (
                id VARCHAR(20) PRIMARY KEY,
                content TEXT NOT NULL,
                confidence REAL NOT NULL,
                state VARCHAR(20) NOT NULL DEFAULT 'Explore',
                supported_by TEXT NOT NULL DEFAULT '[]',
                supports TEXT NOT NULL DEFAULT '[]',
                tags TEXT NOT NULL DEFAULT '[]',
                scope VARCHAR(50) NOT NULL DEFAULT 'user-workspace',
                created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                embedding TEXT,
                is_dirty BOOLEAN NOT NULL DEFAULT 1,
                dirty_reason VARCHAR(50),
                dirty_timestamp TIMESTAMP,
                dirty_priority INTEGER NOT NULL DEFAULT 0,
                created_by VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Generate embedding
        embedding = self._generate_embedding(content)
        embedding_list = embedding.tolist()

        # Generate ID
        claim_id = f"c{int(time.time() * 1000) % 100000000:08d}"

        # Save claim with correct schema
        cursor.execute(
            """
            INSERT INTO claims (id, content, confidence, state, tags, scope, embedding, is_dirty, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                claim_id,
                content,
                confidence,
                "Explore",
                json.dumps(tags) if tags else "[]",
                scope,
                json.dumps(embedding_list),
                1,  # is_dirty = True
                user_id,
            ),
        )

        conn.commit()
        conn.close()
        return claim_id

    def _create_claim_panel(
        self,
        claim_id: str,
        content: str,
        confidence: float,
        user_id: str,
        metadata: dict = None,
    ) -> Panel:
        """Create a Rich panel displaying claim information."""
        content_preview = content[:100] + "..." if len(content) > 100 else content
        metadata_str = json.dumps(metadata, indent=2) if metadata else "{}"

        panel_content = (
            f"[bold]ID:[/bold] {claim_id}\n"
            f"[bold]Content:[/bold] {content_preview}\n"
            f"[bold]Confidence:[/bold] {confidence:.2f}\n"
            f"[bold]Created By:[/bold] {user_id}\n"
            f"[bold]Metadata:[/bold] {metadata_str}"
        )

        return Panel(
            panel_content,
            title=f"[green]Claim Created: {claim_id}[/green]",
            border_style="green",
        )

    # Concrete implementations
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
        # Validate input
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

        if len(content) < 10:
            raise ValueError("Content must be at least 10 characters")

        # Save claim with metadata
        metadata = {
            "analyzed": analyze,
            "provider": self.current_provider_config.get("name")
            if self.current_provider_config
            else None,
            "backend": "unified",
            "workspace": self._workspace,
        }

        claim_id = self._save_claim(content, confidence, user_id, metadata, tags, scope)

        # Display result
        panel = self._create_claim_panel(
            claim_id, content, confidence, user_id, metadata
        )
        self.console.print(panel)

        # Analyze if requested and provider is available
        if analyze and self.current_provider_config:
            self.console.print(
                f"[yellow]Analyzing with {self.current_provider_config['name']}...[/yellow]"
            )
            # TODO: Implement LLM analysis
            self.console.print("[green]Analysis complete[/green]")

        return claim_id

    def get_claim(self, claim_id: str) -> Optional[dict]:
        """Get a claim by ID."""
        self._init_database()
        return self._get_claim(claim_id)

    def search_claims(self, query: str, limit: int = 10, **kwargs) -> List[dict]:
        """Search claims by content."""
        self._init_database()

        try:
            results = self._search_claims(query, limit)
            return results
        except Exception as e:
            self.error_console.print(f"[red]Error searching claims: {e}[/red]")
            raise

    def analyze_claim(self, claim_id: str, **kwargs) -> Dict[str, Any]:
        """Analyze a claim using LLM services."""
        claim = self.get_claim(claim_id)
        if not claim:
            raise ValueError(f"Claim {claim_id} not found")

        self.console.print(f"[blue][UNIFIED] Analyzing claim {claim_id}...[/blue]")

        # Mock analysis for now - would integrate with LLM APIs
        analysis = {
            "claim_id": claim_id,
            "backend": "unified",
            "provider": self.current_provider_config.get("name")
            if self.current_provider_config
            else None,
            "analysis_type": "semantic",
            "confidence_score": claim.get("confidence", 0.0),
            "sentiment": "neutral",
            "topics": ["general", "fact_checking"],
            "verification_status": "pending",
            "model_used": self.current_provider_config.get("model", "unknown")
            if self.current_provider_config
            else None,
        }

        self.console.print("[green][OK] Analysis complete[/green]")
        return analysis

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
                f"[dim][INFO] Creating claim with context: {workspace}/{team}/{user}[/dim]"
            )
            self.console.print(f"[dim][INFO] Scope: {scope}[/dim]")
            self.console.print(f"[dim][INFO] Tags: {tags}[/dim]")

        # Initialize database if not already done
        if not self.sqlite_manager.connection:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, schedule the task
                    asyncio.ensure_future(self.sqlite_manager.initialize())
                else:
                    # If no loop running, run synchronously
                    loop.run_until_complete(self.sqlite_manager.initialize())
            except RuntimeError:
                # No event loop exists, create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self.sqlite_manager.initialize())
                finally:
                    pass  # Keep loop for future use
            self._workspace = workspace

        # Create claim using SQLiteManager
        claim = Claim(
            id=generate_claim_id(),
            content=prompt_text,
            confidence=confidence,
            tags=tags,
            scope=ClaimScope.USER_WORKSPACE,  # Most restrictive by default
            is_dirty=True,
            dirty_reason=DirtyReason.NEW_CLAIM_ADDED,
            dirty_priority=10,
        )

        # Save claim to SQLite database
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, schedule the task
                asyncio.ensure_future(self.sqlite_manager.create_claim(claim))
            else:
                # If no loop running, run synchronously
                loop.run_until_complete(self.sqlite_manager.create_claim(claim))
        except RuntimeError:
            # No event loop exists, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.sqlite_manager.create_claim(claim))
            finally:
                pass  # Keep loop for future use
        claim_id = claim.id

        if verbose >= 1:
            self.console.print(
                f"[dim][INFO] Marking claim {claim_id} as dirty for evaluation[/dim]"
            )

        if verbose >= 1:
            self.console.print(f"[dim][INFO] Starting dirty evaluation...[/dim]")

        # Process dirty evaluation (mock for now)
        evaluation_result = self._mock_evaluate_claim(claim_id)

        if verbose >= 1:
            self.console.print(
                f"[dim][INFO] Evaluation complete, confidence: {evaluation_result['final_confidence']:.1%}[/dim]"
            )

        # Check confidence threshold for user response
        if evaluation_result["final_confidence"] > 0.90:
            if verbose >= 2:
                self.console.print(
                    f"[blue][OK] High-confidence claim achieved: {evaluation_result['final_confidence']:.1%}[/blue]"
                )
                self.console.print(
                    f"[blue][OK] Claim details: {evaluation_result['summary']}[/blue]"
                )

            if verbose >= 1:
                self.console.print(
                    f"[dim][INFO] Generating user response with TellUser tool...[/dim]"
                )

            user_response = self._generate_user_response(prompt_text, evaluation_result)

            # Always show final response (even at verbose=0)
            self.console.print(user_response)

            evaluation_result["user_response"] = user_response
        else:
            if verbose >= 1:
                self.console.print(
                    f"[dim][INFO] Confidence {evaluation_result['final_confidence']:.1%} below 90% threshold - no user response[/dim]"
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

    def is_available(self) -> bool:
        """Check if backend is available and properly configured."""
        return bool(self.provider_manager and self.provider_manager.providers)

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
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, schedule the task
                asyncio.ensure_future(
                    self.sqlite_manager.mark_claim_dirty(claim_id, reason, priority)
                )
            else:
                # If no loop running, run synchronously
                loop.run_until_complete(
                    self.sqlite_manager.mark_claim_dirty(claim_id, reason, priority)
                )
        except RuntimeError:
            # No event loop exists, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    self.sqlite_manager.mark_claim_dirty(claim_id, reason, priority)
                )
            finally:
                pass  # Keep loop for future use

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
