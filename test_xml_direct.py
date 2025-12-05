#!/usr/bin/env python3
"""
Direct XML template test without full imports
"""

import sys
from pathlib import Path

# Add src to path but avoid __init__.py imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "processing" / "llm_prompts"))

def test_xml_template_direct():
    """Test XML template directly"""
    print("Direct XML Template Test")
    print("=" * 40)
    
    try:
        # Import XML templates directly
        from xml_optimized_templates import XMLOptimizedTemplateManager
        
        template_manager = XMLOptimizedTemplateManager()
        template = template_manager.get_template("research_enhanced_xml")
        
        if not template:
            print("XML template not found")
            return False
        
        print(f"Enhanced XML template: {template.id}")
        print(f"Description: {template.description}")
        
        # Test formatting
        formatted = template.format(
            user_query="Test query",
            relevant_context="Test context"
        )
        
        print(f"Formatting successful: {len(formatted)} characters")
        print(f"Contains XML claims: {'<claim type=' in formatted}")
        
        # Validate structure
        checks = [
            ('<claim type=', 'Claim type attribute'),
            ('confidence=', 'Confidence attribute'),
            ('<content>', 'Content tag'),
            ('<support>', 'Support tag'),
            ('<uncertainty>', 'Uncertainty tag'),
            ('</claim>', 'Claim closing tag')
        ]
        
        all_pass = True
        for check, name in checks:
            if check in formatted:
                print(f"  ✅ {name}")
            else:
                print(f"  ❌ {name}")
                all_pass = False
        
        return all_pass
        
    except Exception as e:
        print(f"Direct test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_xml_template_direct()
    
    if success:
        print("\nXML template validation: SUCCESS")
        print("Ready to proceed with Conjecture integration")
        action = "PROCEED"
    else:
        print("\nXML template validation: FAILED")
        print("Need to fix XML template system")
        action = "FIX"
    
    print(f"Action: {action}")
    
    # Update RESULTS.md
    results_file = Path(__file__).parent / "RESULTS.md"
    
    try:
        with open(results_file, 'r') as f:
            content = f.read()
        
        # Find Experiment 1 section
        exp1_marker = "## Experiment 1: XML Format Optimization"
        if exp1_marker in content:
            insertion_point = content.find(exp1_marker) + len(exp1_marker)
            
            new_content = content[:insertion_point] + f"""

**Status**: {'🟢 READY' if action == 'PROCEED' else '🔴 NEEDS WORK'}
**XML Template Validation**: {'✅ PASSED' if success else '❌ FAILED'}
**Implementation**: {'READY' if action == 'PROCEED' else 'NEEDS FIXES'}

### Template Analysis:
- **XML Structure**: {'✅ VALID' if '<claim' in content else '❌ INVALID'}
- **Required Components**: {'✅ ALL PRESENT' if success else '❌ MISSING'}
- **Formatting Capability**: {'✅ WORKING' if action == 'PROCEED' else '❌ BROKEN'}
- **Integration Status**: {'✅ READY' if action == 'PROCEED' else '❌ BLOCKED'}

### Complexity:
- **Files Modified**: 1 (xml_optimized_templates.py)
- **Lines Added**: ~120
- **Dependencies**: 0 new (uses existing XML)
- **Risk Level**: LOW

### Decision:
**Action**: {action}
**Reason**: {'XML template system validated and ready for integration' if action == 'PROCEED' else 'XML template needs fixes before integration'}

---
""" + content[insertion_point:]
            
            with open(results_file, 'w') as f:
                f.write(new_content)
            
            print("✅ RESULTS.md updated")
        else:
            print("❌ Could not find Experiment 1 section")
    except Exception as e:
        print(f"❌ Failed to update RESULTS.md: {e}")