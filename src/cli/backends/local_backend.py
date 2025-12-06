#!/usr/bin/env python3
"""
Local backend for Conjecture CLI
Handles local LLM providers like Ollama and LM Studio
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
import aiohttp
import os

from ..base_cli import BaseCLI, DatabaseError, BackendNotAvailableError
from ...core.models import Claim, generate_claim_id


class LocalBackend(BaseCLI):
    """Local backend implementation for local LLM providers"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = "local"
        self.provider_url = self.get_config_value("url", "http://localhost:11434")
        self.model_name = self.get_config_value("model", "llama2")
        self.api_key = self.get_config_value("api_key", "")
        
    def is_available(self) -> bool:
        """Check if local backend is available"""
        try:
            # Simple health check - try to connect to the provider
            import requests
            response = requests.get(f"{self.provider_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    async def create_claim(
        self, 
        content: str, 
        confidence: float = 0.8,
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> Claim:
        """Create a new claim using local LLM"""
        try:
            # Generate claim ID
            claim_id = generate_claim_id()
            
            # Create claim object
            claim = Claim(
                id=claim_id,
                content=content,
                confidence=confidence,
                tags=tags or [],
                created_by=kwargs.get("created_by", "local_user")
            )
            
            # Validate claim
            await self.validate_claim(claim)
            
            # In a real implementation, this would save to database
            # For now, just return the claim
            return claim
            
        except Exception as e:
            raise DatabaseError(f"Failed to create claim: {e}")
    
    async def get_claim(self, claim_id: str) -> Optional[Claim]:
        """Retrieve a claim by ID"""
        try:
            # In a real implementation, this would query the database
            # For now, return None to indicate not found
            return None
        except Exception as e:
            raise DatabaseError(f"Failed to get claim {claim_id}: {e}")
    
    async def search_claims(
        self, 
        query: str, 
        limit: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for claims using local LLM"""
        try:
            # In a real implementation, this would search the database
            # For now, return empty results
            return []
        except Exception as e:
            raise DatabaseError(f"Failed to search claims: {e}")
    
    async def analyze_claim(self, claim_id: str) -> Dict[str, Any]:
        """Analyze a claim using local LLM"""
        try:
            # Get the claim first
            claim = await self.get_claim(claim_id)
            if not claim:
                raise DatabaseError(f"Claim {claim_id} not found")
            
            # In a real implementation, this would use the LLM to analyze
            # For now, return basic analysis
            return {
                "claim_id": claim_id,
                "analysis": "Basic analysis - local backend",
                "confidence_score": claim.confidence,
                "recommendations": ["Add supporting evidence", "Verify sources"]
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to analyze claim {claim_id}: {e}")
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get backend information"""
        return {
            "name": self.name,
            "type": "local",
            "provider_url": self.provider_url,
            "model": self.model_name,
            "available": self.is_available(),
            "description": "Local LLM backend (Ollama, LM Studio, etc.)"
        }
    
    async def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to the local LLM provider"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                
                url = f"{self.provider_url}{endpoint}"
                async with session.post(url, json=data, headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        raise BackendNotAvailableError(
                            f"Provider request failed: {response.status}"
                        )
        except Exception as e:
            raise BackendNotAvailableError(f"Failed to connect to provider: {e}")