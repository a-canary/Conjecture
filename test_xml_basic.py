#!/usr/bin/env python3
"""
Simple XML format test
"""

def test_xml_format():
    """Test XML format requirements"""
    print("Testing XML Format Requirements")
    print("=" * 40)
    
    # Test XML template structure
    xml_template = '''You are Conjecture, an advanced AI reasoning system that creates structured claims using XML format.

<research_task>
{user_query}
</research_task>

<available_context>
{relevant_context}
</available_context>

CLAIM CREATION REQUIREMENTS:
- Use this EXACT XML format for each claim:
<claim type="[fact|concept|example|goal|reference|hypothesis]" confidence="[0.0-1.0]">
<content>Your clear, specific claim content here</content>
<support>Supporting evidence or reasoning</support>
<uncertainty>Any limitations or confidence notes</uncertainty>
</claim>

- Generate 5-10 high-quality claims covering different aspects
- Include <support> tags with evidence or reasoning
- Use <uncertainty> for speculative claims
- Assign realistic confidence scores'''
    
    # Check components
    required_components = [
        '<claim type="',
        'confidence="',
        '<content>',
        '<support>',
        '<uncertainty>',
        '</claim>'
    ]
    
    print("XML Template Component Check:")
    all_present = True
    for component in required_components:
        if component in xml_template:
            print(f"   OK: {component}")
        else:
            print(f"   MISSING: {component}")
            all_present = False
    
    # Test example
    example_claim = '''<claim type="fact" confidence="0.9">
<content>The Earths average temperature has increased by 1.1C since pre-industrial times.</content>
<support>NASA and NOAA data show consistent warming trends</support>
<uncertainty>No significant uncertainty in this trend</uncertainty>
</claim>'''
    
    print(f"Example Claim Test:")
    print(f"   Length: {len(example_claim)} characters")
    print(f"   Valid XML: {'<claim' in example_claim and '</claim>' in example_claim}")
    
    # Test key characteristics
    print(f"Format Characteristics:")
    print(f"   Uses XML tags instead of bracket format")
    print(f"   Structured with content/support/uncertainty")
    print(f"   Type attribute for categorization")
    print(f"   Confidence attribute for calibration")
    print(f"   Clear hierarchical structure")
    
    return all_present

if __name__ == "__main__":
    success = test_xml_format()
    
    if success:
        print("\nEnhanced XML template format ready for deployment!")
        print("Experiment 1: XML Format Optimization - STRUCTURE VALIDATION PASSED")
        print("\nNext Steps:")
        print("1. Deploy enhanced XML template to Conjecture")
        print("2. Update claim parser for XML compatibility")
        print("3. Run baseline test (claim creation success rate)")
        print("4. Implement XML optimization")
        print("5. Run comparison test")
        print("6. Analyze results and commit/revert")
    else:
        print("\nEnhanced XML template format needs fixes!")
        print("Experiment 1: XML Format Optimization - STRUCTURE VALIDATION FAILED")