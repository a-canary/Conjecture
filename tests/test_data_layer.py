#!/usr/bin/env python3
"""
Simple test script for Conjecture data layer - Core functionality only.
"""
import pytest
import asyncio
import sys
import os
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_manager import DataManager
from src.core.models import DataConfig

@pytest.mark.asyncio
async def test_data_layer():
    """Test basic data layer functionality."""
    print("=== Conjecture Data Layer Test ===")
    
    # Create temporary directory for test
    temp_dir = tempfile.mkdtemp()
    print(f"Using temp directory: {temp_dir}")
    
    dm = None
    try:
        # Configure data manager
        config = DataConfig(
            sqlite_path=os.path.join(temp_dir, "test.db"),
            chroma_path=os.path.join(temp_dir, "chroma"),
            use_chroma=False,  # Disable ChromaDB for simpler testing
            use_embeddings=False  # Disable embeddings for simpler testing
        )
        
        # Initialize data manager
        dm = DataManager(config)
        await dm.initialize()
        
        print("✓ Data manager initialized successfully")
        
        # Test basic claim creation
        from src.core.models import Claim
        test_claim = Claim(
            id="c0000001",
            content="Test claim for data layer validation",
            confidence=0.8,
            created_by="test_user"
        )
        
        claim_id = await dm.create_claim(test_claim)
        assert claim_id == test_claim.id
        print("✓ Claim created successfully")
        
        # Test claim retrieval
        retrieved = await dm.get_claim(claim_id)
        assert retrieved is not None
        assert retrieved['id'] == claim_id
        assert retrieved['content'] == test_claim.content
        print("✓ Claim retrieved successfully")
        
        print("✓ All data layer tests passed!")
        
    except Exception as e:
        print(f"✗ Data layer test failed: {e}")
        raise
    finally:
        if dm:
            await dm.close()
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("✓ Cleanup completed")