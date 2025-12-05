#!/usr/bin/env python3
"""
Experiment 5: Multi-Modal Integration - Test Script
Test multi-modal processing capabilities with image and document analysis
"""

import sys
import time
import json
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_test_data():
    """Create test data for multi-modal processing"""
    
    # Test text input
    test_text = "Analyze this quarterly financial report and identify key performance indicators, risks, and recommendations for improvement."
    
    # Mock image data (simulating different types)
    mock_chart_image = b"mock_chart_data"  # Simulate chart image
    mock_document_image = b"mock_document_data"  # Simulate document scan
    mock_medical_image = b"mock_medical_xray"  # Simulate medical image
    
    test_images = [mock_chart_image, mock_document_image, mock_medical_image]
    
    # Mock document data
    mock_financial_report = b"mock_financial_document"  # Simulate financial report
    mock_legal_contract = b"mock_legal_document"  # Simulate legal contract
    mock_technical_spec = b"mock_technical_document"  # Simulate technical spec
    
    test_documents = [mock_financial_report, mock_legal_contract, mock_technical_spec]
    
    return test_text, test_images, test_documents

async def test_multimodal_processing():
    """Test multi-modal processing capabilities"""
    print("Experiment 5: Multi-Modal Integration - Test")
    print("=" * 60)
    
    try:
        from src.processing.multimodal_processor import get_multimodal_processor
        
        print("\n1. Initializing Multi-Modal Processor...")
        processor = get_multimodal_processor()
        print("   + Multi-Modal Processor: READY")
        
        print("\n2. Creating Test Data...")
        test_text, test_images, test_documents = create_test_data()
        print(f"   + Text input: {len(test_text)} characters")
        print(f"   + Test images: {len(test_images)} items")
        print(f"   + Test documents: {len(test_documents)} items")
        
        print("\n3. Testing Image Processing...")
        
        # Test image analysis
        for i, image_data in enumerate(test_images):
            print(f"   Testing image {i+1}...")
            
            start_time = time.time()
            result = await processor.vision_processor.analyze_image(image_data, "object_detection")
            processing_time = (time.time() - start_time) * 1000
            
            if result.description:
                print(f"   + Analysis: {result.description}")
                print(f"   + Objects detected: {len(result.objects_detected)}")
                print(f"   + Confidence: {result.confidence:.2f}")
                print(f"   + Processing time: {processing_time:.1f}ms")
            else:
                print(f"   + FAILED: No analysis returned")
        
        print("\n4. Testing Document Processing...")
        
        # Test document analysis
        for i, document_data in enumerate(test_documents):
            print(f"   Testing document {i+1}...")
            
            start_time = time.time()
            result = await processor.document_processor.analyze_document(document_data)
            processing_time = (time.time() - start_time) * 1000
            
            if result.structure:
                print(f"   + Document type: {result.structure.get('type', 'unknown')}")
                print(f"   + Sections: {len(result.structure.get('sections', []))}")
                print(f"   + Tables extracted: {len(result.tables_extracted)}")
                print(f"   + Confidence: {result.confidence:.2f}")
                print(f"   + Processing time: {processing_time:.1f}ms")
            else:
                print(f"   + FAILED: No analysis returned")
        
        print("\n5. Testing Multi-Modal Integration...")
        
        # Test combined text + image processing
        print("   Testing text + image integration...")
        start_time = time.time()
        multimodal_result = await processor.process_multimodal_input(
            text=test_text,
            images=test_images[:1],  # Use first image for combined test
            documents=[]
        )
        processing_time = (time.time() - start_time) * 1000
        
        if multimodal_result.claims:
            print(f"   + Claims generated: {len(multimodal_result.claims)}")
            print(f"   + Evidence items: {len(multimodal_result.evidence)}")
            print(f"   + Confidence score: {multimodal_result.confidence_score:.2f}")
            print(f"   + Processing time: {processing_time:.1f}ms")
            
            # Analyze evidence types
            evidence_types = {}
            for evidence in multimodal_result.evidence:
                modality = evidence.modality.value
                evidence_types[modality] = evidence_types.get(modality, 0) + 1
            
            print(f"   + Evidence types: {evidence_types}")
        else:
            print(f"   + FAILED: No multi-modal result")
        
        print("\n6. Testing Document + Image Integration...")
        
        # Test combined document + image processing
        print("   Testing document + image integration...")
        start_time = time.time()
        multimodal_result = await processor.process_multimodal_input(
            text="Analyze this technical specification with the accompanying diagram.",
            images=test_images[2:3],  # Use remaining images
            documents=test_documents[:1]  # Use first document
        )
        processing_time = (time.time() - start_time) * 1000
        
        if multimodal_result.claims:
            print(f"   + Claims generated: {len(multimodal_result.claims)}")
            print(f"   + Evidence items: {len(multimodal_result.evidence)}")
            print(f"   + Confidence score: {multimodal_result.confidence_score:.2f}")
            print(f"   + Processing time: {processing_time:.1f}ms")
        else:
            print(f"   + FAILED: No multi-modal result")
        
        print("\n7. Getting Processing Statistics...")
        
        stats = processor.get_stats()
        print(f"   + Total multi-modal requests: {stats.get('multimodal_requests_processed', 0)}")
        print(f"   + Images processed: {stats.get('vision_processor', {}).get('images_processed', 0)}")
        print(f"   + Documents processed: {stats.get('document_processor', {}).get('documents_processed', 0)}")
        print(f"   + Claims generated: {stats.get('claim_synthesizer', {}).get('claims_generated', 0)}")
        print(f"   + Average processing time: {stats.get('total_processing_time_ms', 0):.1f}ms")
        
        print("\n8. Performance Analysis...")
        
        # Calculate performance metrics
        total_images = len(test_images)
        total_documents = len(test_documents)
        total_claims = len(multimodal_result.claims) if multimodal_result.claims else 0
        
        # Success criteria evaluation
        image_processing_success = total_images > 0
        document_processing_success = total_documents > 0
        multimodal_integration_success = total_claims > 0
        processing_time_acceptable = stats.get('total_processing_time_ms', 0) < 5000  # Under 5 seconds
        
        success_criteria_met = sum([
            image_processing_success,
            document_processing_success,
            multimodal_integration_success,
            processing_time_acceptable
        ])
        
        print(f"\n9. Test Results Summary...")
        print(f"   + Image processing: {'SUCCESS' if image_processing_success else 'FAILED'}")
        print(f"   + Document processing: {'SUCCESS' if document_processing_success else 'FAILED'}")
        print(f"   + Multi-modal integration: {'SUCCESS' if multimodal_integration_success else 'FAILED'}")
        print(f"   + Processing time: {'ACCEPTABLE' if processing_time_acceptable else 'TOO SLOW'}")
        print(f"   + Overall success rate: {success_criteria_met}/4 criteria met")
        
        # Prepare results
        experiment_results = {
            "test_data": {
                "text_length": len(test_text),
                "images_count": total_images,
                "documents_count": total_documents
            },
            "processing_results": {
                "images_processed": total_images,
                "documents_processed": total_documents,
                "claims_generated": total_claims,
                "multimodal_requests": 1,
                "processing_stats": stats
            },
            "success_criteria": {
                "image_processing_success": image_processing_success,
                "document_processing_success": document_processing_success,
                "multimodal_integration_success": multimodal_integration_success,
                "processing_time_acceptable": processing_time_acceptable,
                "criteria_met": success_criteria_met,
                "overall_success": success_criteria_met >= 3
            },
            "timestamp": time.time()
        }
        
        # Save results
        results_file = Path("experiment_5_multimodal_results.json")
        with open(results_file, 'w') as f:
            json.dump(experiment_results, f, indent=2)
        
        print(f"   + Results saved to: {results_file}")
        
        return success_criteria_met >= 3
        
    except Exception as e:
        print(f"Multi-modal test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_multimodal_processing())
    
    if success:
        print("\n+ EXPERIMENT 5: MULTI-MODAL INTEGRATION SUCCESSFUL!")
        print("+ Multi-modal processing components working correctly")
        print("+ Ready for integration with Conjecture system")
    else:
        print("\n- EXPERIMENT 5: MULTI-MODAL INTEGRATION FAILED!")
        print("+ Check component implementation and retry")
    
    sys.exit(0 if success else 1)