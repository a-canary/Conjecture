#!/usr/bin/env python3
"""
Database Priming Script for Conjecture

This script adds foundational claims about fact-checking, programming, scientific method,
and critical thinking to improve system reasoning quality.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.conjecture import Conjecture
from src.core.models import Claim, ClaimState, ClaimType


async def prime_database_with_foundational_claims():
    """Prime the Conjecture database with foundational claims"""
    
    print("🚀 Priming database with foundational claims...")
    
    # Initialize Conjecture
    conjecture = Conjecture()
    
    try:
        # Start services
        await conjecture.start_services()
        
        # Foundational claims to prime the database
        foundational_claims = [
            {
                "content": "Fact-checking is the systematic process of verifying information accuracy, credibility, and relevance using evidence-based reasoning.",
                "confidence": 0.95,
                "tags": ["fact", "concept", "best_practice"],
                "state": ClaimState.VALIDATED
            },
            {
                "content": "Programming best practices involve writing clean, maintainable, and efficient code with proper documentation, testing, and version control.",
                "confidence": 0.9,
                "tags": ["fact", "concept", "best_practice"],
                "state": ClaimState.VALIDATED
            },
            {
                "content": "The scientific method is a systematic approach to investigating phenomena, acquiring new knowledge, and developing testable explanations through observation, experimentation, and peer review.",
                "confidence": 0.9,
                "tags": ["fact", "concept", "best_practice"],
                "state": ClaimState.VALIDATED
            },
            {
                "content": "Critical thinking involves questioning assumptions, seeking evidence, considering alternative perspectives, identifying biases, and making reasoned judgments based on available information.",
                "confidence": 0.9,
                "tags": ["fact", "concept", "best_practice"],
                "state": ClaimState.VALIDATED
            }
        ]
        
        # Add foundational claims to database
        for i, claim_data in enumerate(foundational_claims, 1):
            claim_id = f"prime_{i:03d}"
            
            print(f"Adding claim {i+1}/6: {claim_data['content'][:50]}...")
            
            await conjecture.add_claim(
                content=claim_data['content'],
                confidence=claim_data['confidence'],
                tags=claim_data['tags'],
                auto_evaluate=False  # Don't auto-evaluate priming claims
            )
        
        print(f"✅ Successfully primed database with {len(foundational_claims)} foundational claims")
        
        # Stop services
        await conjecture.stop_services()
        
        return True
    
    except Exception as e:
        print(f"❌ Failed to prime database: {e}")
        return False




if __name__ == "__main__":
    asyncio.run(prime_database_with_foundational_claims())