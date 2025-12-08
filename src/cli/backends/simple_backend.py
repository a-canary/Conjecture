#!/usr/bin/env python3
"""
Simple backend for Conjecture CLI
Minimal backend implementation for basic operations without LLM dependencies
"""

import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime

from ..base_cli import BaseCLI, DatabaseError, BackendNotAvailableError
from ...core.models import Claim, generate_claim_id


class SimpleBackend(BaseCLI):
    """Simple backend implementation for basic operations"""
    
    # Class-level storage for persistence across instances
    _claims = {}
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = "simple"
        # Use class-level storage for persistence
        
    def is_available(self) -> bool:
        """Simple backend is always available"""
        return True
    
    async def create_claim(
        self, 
        content: str, 
        confidence: float = 0.8,
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> Claim:
        """Create a new claim"""
        try:
            # Generate claim ID
            claim_id = generate_claim_id()
            
            # Create claim object
            claim = Claim(
                id=claim_id,
                content=content,
                confidence=confidence,
                tags=tags or [],
                created_by=kwargs.get("user", "user"),
            )
            
            # Store in class-level memory for persistence
            SimpleBackend._claims[claim_id] = claim
            
            return claim
            
        except Exception as e:
            raise DatabaseError(f"Failed to create claim: {e}")
    
    async def get_claim(self, claim_id: str) -> Optional[Claim]:
        """Get a claim by ID"""
        return SimpleBackend._claims.get(claim_id)
    
    async def search_claims(
        self, 
        query: str, 
        limit: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search claims by content"""
        results = []
        query_lower = query.lower()
        
        for claim_id, claim in SimpleBackend._claims.items():
            if query_lower in claim.content.lower():
                results.append({
                    "id": claim_id,
                    "content": claim.content,
                    "confidence": claim.confidence,
                    "similarity": 1.0,  # Simple matching
                    "created_by": claim.created_by if hasattr(claim, 'created_by') else "user"
                })
                if len(results) >= limit:
                    break
        
        return results
    
    async def analyze_claim(self, claim_id: str) -> Dict[str, Any]:
        """Simple claim analysis"""
        claim = SimpleBackend._claims.get(claim_id)
        if not claim:
            raise DatabaseError(f"Claim {claim_id} not found")
        
        return {
            "claim_id": claim_id,
            "backend": self.name,
            "analysis_type": "simple",
            "confidence_score": claim.confidence,
            "sentiment": "neutral",
            "topics": claim.tags,
            "verification_status": "validated" if claim.confidence > 0.8 else "pending"
        }
        
        async def process_prompt(
            self,
            prompt_text: str,
            confidence: float = 0.8,
            verbose: int = 0,
            **kwargs
        ) -> Dict[str, Any]:
            """Process a prompt as a claim with workspace context"""
            # Create a claim from the prompt
            claim = await self.create_claim(
                content=prompt_text,
                confidence=confidence,
                tags=["prompt"],
                **kwargs
            )
            
            # Return simple processing result
            return {
                "claim_id": claim.id,
                "prompt": prompt_text,
                "confidence": confidence,
                "status": "processed",
                "backend": self.name,
                "verbose_level": verbose
            }
        
    def get_backend_info(self) -> Dict[str, Any]:
        """Get backend information"""
        return {
            "name": self.name,
            "type": "simple",
            "configured": True,
            "provider": "simple",
            "model": "none"
        }
    
    def _get_backend_type(self) -> str:
        """Get backend type for stats"""
        return self.name