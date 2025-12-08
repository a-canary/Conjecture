#!/usr/bin/env python3
"""
Base CLI module for Conjecture
Provides abstract base class and common exceptions for CLI implementations
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.core.models import Claim, ClaimFilter


class ClaimValidationError(Exception):
    """Exception raised for claim validation errors"""
    pass


class DatabaseError(Exception):
    """Exception raised for database operation errors"""
    pass


class BackendNotAvailableError(Exception):
    """Exception raised when a backend is not available"""
    pass


class BaseCLI(ABC):
    """Abstract base class for CLI implementations"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the CLI with optional configuration"""
        self.config = config or {}
        self._backend = None
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available"""
        pass
    
    @abstractmethod
    async def create_claim(
        self, 
        content: str, 
        confidence: float = 0.8,
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> Claim:
        """Create a new claim"""
        pass
    
    @abstractmethod
    async def get_claim(self, claim_id: str) -> Optional[Claim]:
        """Retrieve a claim by ID"""
        pass
    
    @abstractmethod
    async def search_claims(
        self, 
        query: str, 
        limit: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for claims"""
        pass
    
    @abstractmethod
    async def analyze_claim(self, claim_id: str) -> Dict[str, Any]:
        """Analyze a claim"""
        pass
    
    @abstractmethod
    def get_backend_info(self) -> Dict[str, Any]:
        """Get backend information"""
        pass
    
    async def validate_claim(self, claim: Claim) -> bool:
        """Validate a claim (default implementation)"""
        try:
            # Basic validation
            if not claim.content or len(claim.content.strip()) < 5:
                raise ClaimValidationError("Claim content must be at least 5 characters")
            
            if not 0.0 <= claim.confidence <= 1.0:
                raise ClaimValidationError("Confidence must be between 0.0 and 1.0")
            
            return True
        except Exception as e:
            raise ClaimValidationError(f"Claim validation failed: {e}")
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        return self.config.get(key, default)
    
    def set_config_value(self, key: str, value: Any) -> None:
        """Set a configuration value"""
        self.config[key] = value