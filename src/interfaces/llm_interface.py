"""
LLM Interface for Conjecture

A simple interface for LLM implementations to follow.
Provides the contract for LLM providers to generate responses.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMInterface(ABC):
    """
    Abstract interface for LLM implementations.
    
    All LLM providers should implement this interface to ensure
    compatibility with the Conjecture system.
    """
    
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response from the LLM for the given prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            str: The LLM's response as a string
        """
        pass
    
    def is_available(self) -> bool:
        """
        Check if the LLM service is available and ready to use.
        
        Returns:
            bool: True if the LLM is available, False otherwise
        """
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the LLM model.
        
        Returns:
            Dict[str, Any]: Dictionary containing model information
        """
        return {
            "model_name": "Unknown",
            "provider": "Unknown",
            "version": "1.0.0"
        }