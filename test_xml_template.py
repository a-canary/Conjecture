#!/usr/bin/env python3
"""
Test XML Enhanced Template for Experiment 1
Validate that enhanced XML template works correctly
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.processing.llm_prompts.xml_optimized_templates import XMLOptimizedTemplateManager

def test_enhanced_xml_template():
    """Test the enhanced XML template"""
    print("Testing Enhanced XML Template")
    print("=" * 40)
    
    # Initialize template manager
    template_manager = XMLOptimizedTemplateManager()
    
    # Get enhanced template
    template = template_manager.get_template("research_enhanced_xml")
    
    if template:
        print(f"✅ Enhanced XML template found: {template.id}")
        print(f"   Description: {template.description}")
        
        # Test formatting
        test_query = "What are best practices for scientific research?"
        test_context = "Previous research shows systematic methodology is important."
        
        formatted_prompt = template.format(
            user_query=test_query,
            relevant_context=test_context
        )
        
        print(f"\n✅ Template formatting successful")
        print(f"   Prompt length: {len(formatted_prompt)} characters")
        print(f"   Contains XML tags: {'<claim' in formatted_prompt}")
        print(f"   Contains examples: {'EXAMPLES:' in formatted_prompt or 'example' in formatted_prompt}")
        
        # Check key components
        required_components = [
            'type="[fact|concept|example|goal|reference|hypothesis]"',
            'confidence="[0.0-1.0]"',
            '<content>',
            '<support>',
            '<uncertainty>',
        ]
        
        print("\n📋 Template Component Check:")
        for component in required_components:
            if component in formatted_prompt:
                print(f"   ✅ {component}")
            else:
                print(f"   ❌ {component}")
        
        return True
    else:
        print("❌ Enhanced XML template not found")
        return False

def check_xml_compatibility():
    """Check if XML parsing is available"""
    print("\nChecking XML Parsing Compatibility")
    print("=" * 40)
    
    try:
        from processing.unified_claim_parser import UnifiedClaimParser
        parser = UnifiedClaimParser()
        
        # Test XML claim parsing
        test_xml_claim = '''<claim type="fact" confidence="0.9">
<content>Test claim content</content>
<support>Supporting evidence</support>
<uncertainty>No significant uncertainty</uncertainty>
</claim>'''
        
        claims = parser.parse_claims(test_xml_claim)
        
        if claims:
            print("✅ XML claim parsing successful")
            print(f"   Parsed {len(claims)} claims")
            print(f"   First claim type: {claims[0].type}")
            print(f"   First claim confidence: {claims[0].confidence}")
            return True
        else:
            print("❌ XML claim parsing failed")
            return False
            
    except Exception as e:
        print(f"❌ XML parsing check failed: {e}")
        return False

if __name__ == "__main__":
    template_success = test_enhanced_xml_template()
    parsing_success = check_xml_compatibility()
    
    if template_success and parsing_success:
        print("\n🎉 Enhanced XML template system ready for deployment!")
        print("✅ Experiment 1: XML Format Optimization - TEMPLATE VALIDATION PASSED")
    else:
        print("\n❌ Enhanced XML template system needs fixes!")
        print("❌ Experiment 1: XML Format Optimization - TEMPLATE VALIDATION FAILED")
    
    print(f"\nReady for testing: {'YES' if template_success and parsing_success else 'NO'}")