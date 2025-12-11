"""
Experiment 6: Enhanced Claim Synthesis Test

Tests the advanced claim synthesis algorithms to fix
multi-modal integration issues from Experiment 5.
"""

import asyncio
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the enhanced multi-modal processor
from src.processing.multimodal_processor import MultiModalProcessor

async def test_enhanced_claim_synthesis():
    """Test enhanced claim synthesis with multi-modal evidence"""
    
    print("Experiment 6: Enhanced Claim Synthesis - Test")
    print("=" * 60)
    
    try:
        # Initialize enhanced processor
        print("\n1. Initializing Enhanced Multi-Modal Processor...")
        processor = MultiModalProcessor()
        print("   + Enhanced Multi-Modal Processor: READY")
        
        # Create comprehensive test data
        print("\n2. Creating Comprehensive Test Data...")
        
        test_text = "Analyze the performance metrics from the quarterly report and identify key trends"
        test_images = [b"fake_image_data"]
        
        # Process the multi-modal data
        print("\n3. Processing Multi-Modal Data...")
        result = await processor.process_multimodal_data(test_text, test_images)
        
        # Display results
        print("\n4. Results:")
        print(f"   + Claims Generated: {len(result.get('claims', []))}")
        print(f"   + Processing Time: {result.get('processing_time', 0):.2f}s")
        print(f"   + Success: {result.get('success', False)}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in enhanced claim synthesis test: {e}")
        print(f"\nERROR: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    asyncio.run(test_enhanced_claim_synthesis())