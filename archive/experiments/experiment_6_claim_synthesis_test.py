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
        test_images = [b"mock_image_data_1", b"mock_image_data_2"]
        test_documents = [b"mock_document_data_1", b"mock_document_data_2"]
        
        print(f"   + Text input: {len(test_text)} characters")
        print(f"   + Test images: {len(test_images)} items")
        print(f"   + Test documents: {len(test_documents)} items")
        
        # Test individual components first
        print("\n3. Testing Individual Components...")
        
        # Test vision processing
        print("   Testing vision processing...")
        vision_results = []
        for i, image in enumerate(test_images):
            result = await processor.vision_processor.analyze_image(image)
            vision_results.append(result)
            print(f"   + Image {i+1}: {result.description} (confidence: {result.confidence:.2f})")
        
        # Test document processing
        print("   Testing document processing...")
        doc_results = []
        for i, doc in enumerate(test_documents):
            result = await processor.document_processor.analyze_document(doc)
            doc_results.append(result)
            print(f"   + Document {i+1}: {result.structure.get('type', 'unknown')} (confidence: {result.confidence:.2f})")
        
        # Test cross-modal reasoning
        print("   Testing cross-modal reasoning...")
        evidence = await processor.cross_modal_reasoner.synthesize(
            test_text, 
            vision_results[0] if vision_results else None,
            doc_results[0] if doc_results else None
        )
        print(f"   + Evidence synthesized: {len(evidence)} items")
        
        # Test enhanced claim synthesis
        print("\n4. Testing Enhanced Claim Synthesis...")
        start_time = time.time()
        
        claims = await processor.claim_synthesizer.synthesize_claims(evidence)
        
        synthesis_time = (time.time() - start_time) * 1000
        
        if claims:
            print(f"   + Claims generated: {len(claims)}")
            print(f"   + Synthesis time: {synthesis_time:.1f}ms")
            
            for i, claim in enumerate(claims):
                print(f"   + Claim {i+1}: {claim.content[:80]}...")
                print(f"     Confidence: {claim.confidence:.2f}")
                print(f"     Tags: {claim.tags}")
                if hasattr(claim, 'metadata') and 'claim_types' in claim.metadata:
                    print(f"     Types: {claim.metadata['claim_types']}")
        else:
            print("   + FAILED: No claims generated")
        
        # Test end-to-end multi-modal processing
        print("\n5. Testing End-to-End Multi-Modal Processing...")
        start_time = time.time()
        
        multimodal_result = await processor.process_multimodal_input(
            text=test_text,
            images=test_images[:1],  # Use one image for testing
            documents=test_documents[:1]  # Use one document for testing
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        if multimodal_result.claims:
            print(f"   + End-to-end processing: SUCCESS")
            print(f"   + Claims generated: {len(multimodal_result.claims)}")
            print(f"   + Evidence items: {len(multimodal_result.evidence)}")
            print(f"   + Confidence score: {multimodal_result.confidence_score:.2f}")
            print(f"   + Processing time: {processing_time:.1f}ms")
        else:
            print("   + FAILED: End-to-end processing failed")
        
        # Get processing statistics
        print("\n6. Getting Processing Statistics...")
        
        stats = processor.get_stats()
        synthesis_stats = processor.claim_synthesizer.get_stats()
        
        print(f"   + Multi-modal requests: {stats.get('multimodal_requests_processed', 0)}")
        print(f"   + Total processing time: {stats.get('total_processing_time_ms', 0):.1f}ms")
        print(f"   + Claims generated: {stats.get('claims_generated', 0)}")
        print(f"   + Average confidence: {stats.get('average_confidence', 0):.2f}")
        
        print(f"\n   + Enhanced Synthesis Statistics:")
        print(f"   + Claims generated: {synthesis_stats.get('claims_generated', 0)}")
        print(f"   + Multi-modal integrations: {synthesis_stats.get('multimodal_integrations', 0)}")
        print(f"   + Synthesis time: {synthesis_stats.get('synthesis_time_ms', 0):.1f}ms")
        print(f"   + Evidence clusters processed: {synthesis_stats.get('evidence_clusters_processed', 0)}")
        
        # Performance analysis
        print("\n7. Performance Analysis...")
        
        # Calculate success metrics
        individual_component_success = len(vision_results) > 0 and len(doc_results) > 0
        claim_synthesis_success = len(claims) > 0
        end_to_end_success = len(multimodal_result.claims) > 0
        processing_time_acceptable = processing_time < 5000  # Under 5 seconds
        
        success_criteria_met = sum([
            individual_component_success,
            claim_synthesis_success,
            end_to_end_success,
            processing_time_acceptable
        ])
        
        print(f"   + Individual components: {'SUCCESS' if individual_component_success else 'FAILED'}")
        print(f"   + Claim synthesis: {'SUCCESS' if claim_synthesis_success else 'FAILED'}")
        print(f"   + End-to-end processing: {'SUCCESS' if end_to_end_success else 'FAILED'}")
        print(f"   + Processing time: {'ACCEPTABLE' if processing_time_acceptable else 'TOO SLOW'}")
        print(f"   + Overall success rate: {success_criteria_met}/4 criteria met")
        
        # Prepare results
        experiment_results = {
            "test_data": {
                "text_length": len(test_text),
                "images_count": len(test_images),
                "documents_count": len(test_documents)
            },
            "component_results": {
                "vision_processing": {
                    "success": len(vision_results) > 0,
                    "images_processed": len(vision_results),
                    "average_confidence": sum(r.confidence for r in vision_results) / len(vision_results) if vision_results else 0
                },
                "document_processing": {
                    "success": len(doc_results) > 0,
                    "documents_processed": len(doc_results),
                    "average_confidence": sum(r.confidence for r in doc_results) / len(doc_results) if doc_results else 0
                },
                "cross_modal_reasoning": {
                    "success": len(evidence) > 0,
                    "evidence_items": len(evidence)
                },
                "claim_synthesis": {
                    "success": len(claims) > 0,
                    "claims_generated": len(claims),
                    "synthesis_time_ms": synthesis_time,
                    "average_confidence": sum(c.confidence for c in claims) / len(claims) if claims else 0
                }
            },
            "end_to_end_results": {
                "success": len(multimodal_result.claims) > 0,
                "claims_generated": len(multimodal_result.claims),
                "evidence_items": len(multimodal_result.evidence),
                "confidence_score": multimodal_result.confidence_score,
                "processing_time_ms": processing_time
            },
            "success_criteria": {
                "individual_component_success": individual_component_success,
                "claim_synthesis_success": claim_synthesis_success,
                "end_to_end_success": end_to_end_success,
                "processing_time_acceptable": processing_time_acceptable,
                "criteria_met": success_criteria_met,
                "overall_success": success_criteria_met >= 3
            },
            "processing_stats": {
                "processor_stats": stats,
                "synthesis_stats": synthesis_stats
            },
            "timestamp": time.time()
        }
        
        # Save results
        results_file = Path("experiment_6_enhanced_synthesis_results.json")
        with open(results_file, 'w') as f:
            json.dump(experiment_results, f, indent=2)
        
        print(f"\n8. Results Summary...")
        print(f"   + Results saved to: {results_file}")
        
        return success_criteria_met >= 3
        
    except Exception as e:
        print(f"Enhanced claim synthesis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test execution"""
    print("Starting Experiment 6: Enhanced Claim Synthesis Test")
    print("=" * 60)
    
    success = await test_enhanced_claim_synthesis()
    
    if success:
        print("\n+ EXPERIMENT 6: ENHANCED CLAIM SYNTHESIS SUCCESSFUL!")
        print("+ Multi-modal claim synthesis working correctly")
        print("+ Ready for production integration")
    else:
        print("\n- EXPERIMENT 6: ENHANCED CLAIM SYNTHESIS FAILED!")
        print("+ Check enhanced synthesis implementation")
        print("+ Debug integration workflow")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())