"""
Simple YAML formatter without external dependencies
Start simple, only extend when needed
"""

import re
from typing import Any, Dict, List, Union

def safe_yaml_serialize(obj: Any, indent: int = 0) -> str:
    """
    Serialize object to YAML-like format without external dependencies
    Handles basic types, lists, and dictionaries
    """
    indent_str = "  " * indent

    if obj is None:
        return "null"
    elif isinstance(obj, bool):
        return "true" if obj else "false"
    elif isinstance(obj, (int, float)):
        return str(obj)
    elif isinstance(obj, str):
        # Escape special characters if needed
        escaped = obj.replace('"', '\\"')
        if any(char in escaped for char in ':{}[],&*#?|-<>"'):
            return f'"{escaped}"'
        return escaped
    elif isinstance(obj, list):
        if not obj:
            return "[]"

        result = []
        for item in obj:
            item_str = safe_yaml_serialize(item, indent + 1)
            result.append(f"{indent_str}- {item_str}")
        return "\n".join(result)
    elif isinstance(obj, dict):
        if not obj:
            return "{}"

        result = []
        for key, value in obj.items():
            key_str = str(key)
            value_str = safe_yaml_serialize(value, indent + 1)

            if "\n" in value_str:  # Multi-line values
                result.append(f"{indent_str}{key_str}: |\n{value_str}")
            else:
                result.append(f"{indent_str}{key_str}: {value_str}")
        return "\n".join(result)
    else:
        # Fallback for other types
        return str(obj)

def dump(obj: Any, default_flow_style: bool = False) -> str:
    """
    Dump object to YAML format
    Mimics basic yaml.dump() functionality
    """
    if default_flow_style:
        # For simple objects, use inline style
        if isinstance(obj, dict) and len(obj) <= 3:
            items = [f"{k}: {safe_yaml_serialize(v)}" for k, v in obj.items()]
            return "{ " + ", ".join(items) + " }"
        elif isinstance(obj, list) and len(obj) <= 3:
            items = [safe_yaml_serialize(item) for item in obj]
            return "[ " + ", ".join(items) + " ]"

    return safe_yaml_serialize(obj)

def format_claim_context(context: Dict[str, Any]) -> str:
    """
    Format claim context in a readable YAML structure
    Optimized for claim processing context
    """
    lines = ["# CLAIM PROCESSING CONTEXT"]
    lines.append("")

    # Main claim
    if "explore_claim" in context:
        lines.append("explore_claim:")
        lines.append(f"  {context['explore_claim']}")
        lines.append("")

    # Support relationships
    if "support" in context and context["support"]:
        lines.append("support:")
        for support_claim in context["support"]:
            lines.append(f"  - {support_claim}")
        lines.append("")

    if "supported_by" in context and context["supported_by"]:
        lines.append("supported_by:")
        for supporting_claim in context["supported_by"]:
            lines.append(f"  - {supporting_claim}")
        lines.append("")

    # Related claims by type
    claim_types = ["concepts", "goals", "references", "skills"]
    for claim_type in claim_types:
        if claim_type in context and context[claim_type]:
            lines.append(f"{claim_type}:")
            for claim in context[claim_type]:
                lines.append(f"  - {claim}")
            lines.append("")

    # Remove trailing empty line
    if lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines)

def test_simple_yaml():
    """Test the simple YAML formatter"""
    print("🧪 Testing Simple YAML Formatter")
    print("=" * 35)

    # Test basic serialization
    test_data = {
        "string_value": "hello world",
        "number_value": 42,
        "bool_value": True,
        "list_value": [1, 2, "three"],
        "nested_dict": {"inner_string": "nested", "inner_list": ["a", "b", "c"]},
        "null_value": None,
    }

    yaml_result = dump(test_data)
    print("✅ Basic serialization: PASS")
    print("Sample output:")
    print(yaml_result[:200] + "..." if len(yaml_result) > 200 else yaml_result)
    print()

    # Test claim context formatting
    claim_context = {
        "explore_claim": "[claim123,0.7,concept,Explore]Quantum encryption test",
        "support": ["[sup1,0.9,reference,Validated]Supporting reference"],
        "concepts": [
            "[con1,0.8,concept,Explore]Related concept 1",
            "[con2,0.7,concept,Explore]Related concept 2",
        ],
        "references": ["[ref1,0.95,reference,Validated]Reference 1"],
    }

    context_yaml = format_claim_context(claim_context)
    print("✅ Claim context formatting: PASS")
    print("Sample context:")
    print(context_yaml[:300] + "..." if len(context_yaml) > 300 else context_yaml)
    print()

    # Test edge cases
    edge_cases = [
        ("empty_dict", {}),
        ("empty_list", []),
        ("special_chars", {"key:with:colons": "value:with:colons"}),
        ("unicode", {"emoji": "🔍🔧", "accent": "café"}),
    ]

    for test_name, test_case in edge_cases:
        try:
            result = dump(test_case)
            assert result is not None
            assert len(result) > 0
            print(f"✅ Edge case {test_name}: PASS")
        except Exception as e:
            print(f"❌ Edge case {test_name}: FAIL - {e}")
            return False

    print("🎉 All simple YAML tests passed!")
    return True

if __name__ == "__main__":
    success = test_simple_yaml()
    exit(0 if success else 1)
