#!/usr/bin/env python3
"""
Test script to verify XML format optimization integration
"""

import sys
import os
sys.path.append('.')

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        try:
            locale.setlocale(locale.LC_ALL, 'C.UTF-8')
        except:
            pass

def test_xml_templates():
    """Test XML template manager"""
    print("Testing XML template manager...")
    
    try:
        from src.processing.llm_prompts.xml_optimized_templates import XMLOptimizedTemplateManager
        
        xml_manager = XMLOptimizedTemplateManager()
        template = xml_manager.get_template('research_enhanced_xml')
        
        if template:
            print(f"✓ XML template found: {template.name}")
            print(f"✓ Template description: {template.description}")
            return True
        else:
            print("X XML template not found")
            return False
            
    except Exception as e:
        print(f"X Error testing XML templates: {e}")
        return False

def test_xml_parser():
    """Test XML claim parser"""
    print("\nTesting XML claim parser...")
    
    try:
        from src.processing.unified_claim_parser import UnifiedClaimParser
        
        parser = UnifiedClaimParser()
        
        # Test XML parsing
        test_xml = '''
        <claim type="fact" confidence="0.9">This is a test fact claim</claim>
        <claim type="concept" confidence="0.8">This is a test concept claim</claim>
        <claim type="example" confidence="0.7">This is a test example claim</claim>
        '''
        
        claims = parser._parse_xml_format(test_xml)
        
        if claims:
            print(f"✓ XML parser working: {len(claims)} claims parsed")
            for i, claim in enumerate(claims):
                print(f"  Claim {i+1}: {claim.claim_type} - {claim.confidence} - {claim.content[:50]}...")
            return True
        else:
            print("X XML parser failed to parse claims")
            return False
            
    except Exception as e:
        print(f"X Error testing XML parser: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test full integration"""
    print("\nTesting full integration...")
    
    try:
        # Test template formatting
        from src.processing.llm_prompts.xml_optimized_templates import XMLOptimizedTemplateManager
        
        xml_manager = XMLOptimizedTemplateManager()
        template = xml_manager.get_template('research_enhanced_xml')
        
        if template:
            # Test template formatting with variables
            formatted_prompt = template.template_content.format(
                user_query="test query",
                relevant_context="test context"
            )
            
            if "test query" in formatted_prompt and "test context" in formatted_prompt:
                print("✓ Template formatting working")
            else:
                print("X Template formatting failed")
                return False
        
        # Test parser with formatted output
        from src.processing.unified_claim_parser import UnifiedClaimParser
        
        parser = UnifiedClaimParser()
        
        # Test the expected XML output format
        test_output = '''
        <claims>
          <claim type="fact" confidence="0.9">Test fact about machine learning</claim>
          <claim type="concept" confidence="0.8">Test concept about neural networks</claim>
        </claims>
        '''
        
        claims = parser.parse_claims_from_response(test_output)
        
        if claims:
            print(f"✓ Full integration working: {len(claims)} claims parsed")
            return True
        else:
            print("X Full integration failed")
            return False
            
    except Exception as e:
        print(f"X Error testing integration: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("=== XML Format Optimization Integration Test ===\n")
    
    tests = [
        test_xml_templates,
        test_xml_parser,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("+ All XML optimization integration tests PASSED")
        print("+ Ready for Experiment 1: XML Format Optimization")
        return True
    else:
        print("X Some tests FAILED")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)