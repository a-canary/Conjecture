#!/usr/bin/env python3
"""
Simple test to verify XML format optimization components
"""

import sys
import re

def test_xml_template_directly():
    """Test XML template content directly"""
    print("Testing XML template content...")
    
    # Read the XML template file directly
    try:
        with open('src/processing/llm_prompts/xml_optimized_templates.py', 'r') as f:
            content = f.read()
        
        # Check if XML template exists
        if 'research_enhanced_xml' in content and '<claim type=' in content:
            print("+ XML template content found")
            return True
        else:
            print("X XML template content not found")
            return False
            
    except Exception as e:
        print(f"X Error reading XML template: {e}")
        return False

def test_xml_parsing_directly():
    """Test XML parsing logic directly"""
    print("\nTesting XML parsing logic...")
    
    # Test XML pattern matching
    xml_pattern = r'<claim\s+type="([^"]*)"\s+confidence="([^"]*)"[^>]*>(.*?)</claim>'
    
    test_xml = '''
    <claim type="fact" confidence="0.9">This is a test fact claim</claim>
    <claim type="concept" confidence="0.8">This is a test concept claim</claim>
    '''
    
    try:
        matches = re.findall(xml_pattern, test_xml, re.MULTILINE | re.DOTALL)
        
        if matches:
            print(f"+ XML parsing working: {len(matches)} claims matched")
            for i, (claim_type, confidence, content) in enumerate(matches):
                print(f"  Claim {i+1}: {claim_type} - {confidence} - {content.strip()}")
            return True
        else:
            print("X XML parsing failed - no matches")
            return False
            
    except Exception as e:
        print(f"X Error in XML parsing: {e}")
        return False

def test_conjecture_integration():
    """Test if conjecture.py has XML integration"""
    print("\nTesting conjecture.py XML integration...")
    
    try:
        with open('src/conjecture.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for XML integration
        checks = [
            ('XML template manager import', 'XMLOptimizedTemplateManager' in content),
            ('XML template usage', 'research_enhanced_xml' in content),
            ('XML format in prompt', '<claim type=' in content),
        ]
        
        all_passed = True
        for check_name, passed in checks:
            if passed:
                print(f"+ {check_name}: Found")
            else:
                print(f"X {check_name}: Not found")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"X Error checking conjecture.py: {e}")
        return False

def test_parser_integration():
    """Test if parser has XML enhancements"""
    print("\nTesting parser XML integration...")
    
    try:
        with open('src/processing/unified_claim_parser.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for XML integration
        checks = [
            ('XML parsing method', '_parse_xml_format' in content),
            ('XML patterns', 'xml_pattern' in content),
            ('Enhanced XML patterns', 'enhanced_xml_patterns' in content),
            ('XML priority in parsing', 'xml' in content and 'bracket' in content),
        ]
        
        all_passed = True
        for check_name, passed in checks:
            if passed:
                print(f"+ {check_name}: Found")
            else:
                print(f"X {check_name}: Not found")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"X Error checking parser: {e}")
        return False

def main():
    """Main test function"""
    print("=== XML Format Optimization Simple Test ===\n")
    
    tests = [
        test_xml_template_directly,
        test_xml_parsing_directly,
        test_conjecture_integration,
        test_parser_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("+ All XML optimization tests PASSED")
        print("+ Ready for Experiment 1: XML Format Optimization")
        return True
    else:
        print("X Some tests FAILED")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)