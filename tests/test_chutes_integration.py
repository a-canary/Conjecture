#!/usr/bin/env python3
"""
Test Chutes.ai LLM integration
"""
import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from processing.llm.chutes_integration import ChutesProcessor, GenerationConfig
from src.core.models import BasicClaim, ClaimType, ClaimState

def test_chutes_integration():
    """Test Chutes.ai integration"""
    print("=== Chutes.ai Integration Test ===")
    
    # Load environment
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("CHUTES_API_KEY")
    if not api_key:
        print("[ERROR] CHUTES_API_KEY not found in environment")
        return False
    
    try:
        # Initialize processor
        processor = ChutesProcessor(api_key=api_key)
        print(f"[OK] Chutes.ai processor initialized")
        
        # Test 1: Simple generation
        print("\n--- Test 1: Simple Generation ---")
        result = processor.generate_response("What is machine learning? Answer in one sentence.")
        
        if result.success:
            print(f"[OK] Generation successful")
            print(f"Content: {result.processed_claims[0].content}")
            print(f"Tokens used: {result.tokens_used}")
            print(f"Processing time: {result.processing_time:.2f}s")
        else:
            print(f"[ERROR] Generation failed: {result.errors}")
            return False
        
        # Test 2: Claim processing
        print("\n--- Test 2: Claim Processing ---")
        
        # Create test claims
        test_claims = [
            BasicClaim(
                id="c0000001",
                content="Machine learning is a subset of artificial intelligence",
                confidence=0.8,
                type=[ClaimType.CONCEPT],
                state=ClaimState.EXPLORE,
                created_by="test",
                created_at=datetime.now()
            ),
            BasicClaim(
                id="c0000002", 
                content="Deep learning uses neural networks",
                confidence=0.7,
                type=[ClaimType.CONCEPT],
                state=ClaimState.EXPLORE,
                created_by="test",
                created_at=datetime.now()
            )
        ]
        
        result = processor.process_claims(test_claims, task="analyze relationships")
        
        if result.success:
            print(f"[OK] Processing successful")
            print(f"Generated {len(result.processed_claims)} claims")
            for claim in result.processed_claims:
                print(f"  - {claim.content} (confidence: {claim.confidence})")
            print(f"Tokens used: {result.tokens_used}")
            print(f"Processing time: {result.processing_time:.2f}s")
        else:
            print(f"[ERROR] Processing failed: {result.errors}")
            return False
        
        # Test 3: Stats
        print("\n--- Test 3: Statistics ---")
        stats = processor.get_stats()
        print(f"Total requests: {stats['total_requests']}")
        print(f"Successful requests: {stats['successful_requests']}")
        print(f"Success rate: {stats['success_rate']:.2%}")
        print(f"Total tokens: {stats['total_tokens']}")
        print(f"Average processing time: {stats['average_processing_time']:.2f}s")
        
        print("\n=== All Tests Passed! ===")
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_chutes_integration()
    sys.exit(0 if success else 1)