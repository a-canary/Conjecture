"""
Unit tests for ResponseParser component.
"""
import pytest
import xml.etree.ElementTree as ET
import json

from src.processing.response_parser import ResponseParser
from src.core.skill_models import ToolCall, ParsedResponse


class TestResponseParser:
    """Test cases for ResponseParser class."""

    def test_parser_initialization(self):
        """Test ResponseParser initialization."""
        parser = ResponseParser()
        
        assert len(parser.supported_formats) == 3
        assert 'xml' in parser.supported_formats
        assert 'json' in parser.supported_formats
        assert 'markdown' in parser.supported_formats
        assert parser.xml_namespaces['tc'] == 'http://conjecture.ai/tool_calls'

    def test_parse_empty_response(self, response_parser):
        """Test parsing empty response."""
        result = response_parser.parse_response("")
        
        assert isinstance(result, ParsedResponse)
        assert len(result.tool_calls) == 0
        assert result.text_content == ""
        assert len(result.parsing_errors) == 1
        assert "Empty response" in result.parsing_errors[0]

    def test_parse_whitespace_only_response(self, response_parser):
        """Test parsing response with only whitespace."""
        result = response_parser.parse_response("   \n\t   ")
        
        assert len(result.tool_calls) == 0
        assert len(result.parsing_errors) >= 1

    def test_parse_none_response(self, response_parser):
        """Test parsing None response."""
        result = response_parser.parse_response(None)
        
        assert len(result.tool_calls) == 0
        assert len(result.parsing_errors) >= 1

    def test_parse_plain_text_response(self, response_parser):
        """Test parsing plain text response without tool calls."""
        text = "This is just plain text with no tool calls."
        result = response_parser.parse_response(text)
        
        assert len(result.tool_calls) == 0
        assert result.text_content == text
        assert len(result.parsing_errors) == 1
        assert "No valid tool calls found" in result.parsing_errors[0]

    def test_supported_formats_priority(self, response_parser):
        """Test that parsing tries formats in priority order."""
        # XML should be tried first
        xml_response = """
        <tool_calls>
            <invoke name="test_tool">
                <parameter name="param">value</parameter>
            </invoke>
        </tool_calls>
        """
        
        result = response_parser.parse_response(xml_response)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "test_tool"

    # XML Parsing Tests
    def test_parse_xml_valid_single_tool(self, response_parser, xml_response_samples):
        """Test parsing XML with single valid tool call."""
        result = response_parser._parse_xml_format(xml_response_samples["valid_single"])
        
        tool_calls, text_content = result
        
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "search_claims"
        assert tool_calls[0].parameters["query"] == "test query"
        assert tool_calls[0].parameters["limit"] == 10
        assert text_content is None

    def test_parse_xml_valid_multiple_tools(self, response_parser, xml_response_samples):
        """Test parsing XML with multiple valid tool calls."""
        result = response_parser._parse_xml_format(xml_response_samples["valid_multiple"])
        
        tool_calls, text_content = result
        
        assert len(tool_calls) == 2
        assert tool_calls[0].name == "search_claims"
        assert tool_calls[1].name == "create_claim"
        assert tool_calls[1].parameters["confidence"] == 0.8

    def test_parse_xml_with_text_content(self, response_parser, xml_response_samples):
        """Test parsing XML with surrounding text content."""
        result = response_parser._parse_xml_format(xml_response_samples["with_text"])
        
        tool_calls, text_content = result
        
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "search_claims"
        assert "I'll help you search for claims." in text_content
        assert "Let me know the results." in text_content

    def test_parse_xml_no_tool_calls(self, response_parser):
        """Test parsing XML without tool_calls wrapper."""
        xml = "<some_other_tag>content</some_other_tag>"
        result = response_parser._parse_xml_format(xml)
        
        tool_calls, text_content = result
        
        assert len(tool_calls) == 0
        assert text_content == xml

    def test_parse_xml_malformed(self, response_parser, xml_response_samples):
        """Test parsing malformed XML."""
        with pytest.raises(Exception):
            response_parser._parse_xml_format(xml_response_samples["malformed"])

    def test_parse_xml_nested_parameters(self, response_parser):
        """Test parsing XML with nested parameters."""
        xml = """
        <tool_calls>
            <invoke name="complex_tool">
                <parameter name="nested">
                    <level1>
                        <level2>deep value</level2>
                    </level1>
                </parameter>
                <parameter name="simple">simple value</parameter>
            </invoke>
        </tool_calls>
        """
        
        result = response_parser._parse_xml_format(xml)
        tool_calls, _ = result
        
        assert len(tool_calls) == 1
        assert tool_calls[0].parameters["simple"] == "simple value"
        assert "level1" in tool_calls[0].parameters["nested"]

    def test_parse_xml_parameter_with_id(self, response_parser):
        """Test parsing XML tool call with call ID."""
        xml = """
        <tool_calls>
            <invoke name="test_tool" id="call123">
                <parameter name="param">value</parameter>
            </invoke>
        </tool_calls>
        """
        
        result = response_parser._parse_xml_format(xml)
        tool_calls, _ = result
        
        assert len(tool_calls) == 1
        assert tool_calls[0].call_id == "call123"

    def test_parse_xml_json_parameters(self, response_parser):
        """Test parsing XML with JSON-encoded parameters."""
        xml = """
        <tool_calls>
            <invoke name="json_tool">
                <parameter name="array_param">["item1", "item2"]</parameter>
                <parameter name="object_param">{"key": "value"}</parameter>
            </invoke>
        </tool_calls>
        """
        
        result = response_parser._parse_xml_format(xml)
        tool_calls, _ = result
        
        assert len(tool_calls) == 1
        assert tool_calls[0].parameters["array_param"] == ["item1", "item2"]
        assert tool_calls[0].parameters["object_param"] == {"key": "value"}

    def test_parse_xml_inline_parameters(self, response_parser):
        """Test parsing XML with inline parameters (not in <parameter> tags)."""
        xml = """
        <tool_calls>
            <invoke name="inline_tool">
                <param1>value1</param1>
                <param2>123</param2>
            </invoke>
        </tool_calls>
        """
        
        result = response_parser._parse_xml_format(xml)
        tool_calls, _ = result
        
        assert len(tool_calls) == 1
        assert tool_calls[0].parameters["param1"] == "value1"
        assert tool_calls[0].parameters["param2"] == 123

    # JSON Parsing Tests
    def test_parse_json_valid_single_tool(self, response_parser, json_response_samples):
        """Test parsing JSON with single valid tool call."""
        result = response_parser._parse_json_format(json_response_samples["valid_single"])
        
        tool_calls, text_content = result
        
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "search_claims"
        assert tool_calls[0].parameters["query"] == "test"
        assert tool_calls[0].parameters["limit"] == 10

    def test_parse_json_valid_multiple_tools(self, response_parser, json_response_samples):
        """Test parsing JSON with multiple valid tool calls."""
        result = response_parser._parse_json_format(json_response_samples["valid_multiple"])
        
        tool_calls, text_content = result
        
        assert len(tool_calls) == 2
        assert tool_calls[0].name == "search_claims"
        assert tool_calls[1].name == "create_claim"

    def test_parse_json_no_tool_calls(self, response_parser, json_response_samples):
        """Test parsing JSON without tool_calls."""
        result = response_parser._parse_json_format(json_response_samples["no_tool_calls"])
        
        tool_calls, text_content = result
        
        assert len(tool_calls) == 0
        assert text_content == json_response_samples["no_tool_calls"]

    def test_parse_json_invalid_json(self, response_parser, json_response_samples):
        """Test parsing invalid JSON."""
        with pytest.raises(json.JSONDecodeError):
            response_parser._parse_json_format(json_response_samples["invalid_json"])

    def test_parse_json_alternate_format(self, response_parser):
        """Test parsing JSON with alternate field names."""
        json_str = """
        {
            "calls": [
                {"name": "test_tool", "args": {"param": "value"}}
            ]
        }
        """
        
        result = response_parser._parse_json_format(json_str)
        tool_calls, _ = result
        
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "test_tool"
        assert tool_calls[0].parameters["param"] == "value"

    def test_parse_json_single_object(self, response_parser):
        """Test parsing JSON where entire object is a tool call."""
        json_str = """
        {
            "name": "single_tool",
            "function": "single_tool",
            "parameters": {"param": "value"}
        }
        """
        
        result = response_parser._parse_json_format(json_str)
        tool_calls, _ = result
        
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "single_tool"

    def test_parse_json_with_text_content(self, response_parser):
        """Test parsing JSON with surrounding text."""
        json_with_text = f"""Here's the result:
        {{"tool_calls": [{{"name": "test_tool", "parameters": {{}}}}]}}
        End of response."""
        
        result = response_parser._parse_json_format(json_with_text)
        tool_calls, text_content = result
        
        assert len(tool_calls) == 1
        assert "Here's the result:" in text_content
        assert "End of response." in text_content

    # Markdown Parsing Tests
    def test_parse_markdown_valid_single_tool(self, response_parser, markdown_response_samples):
        """Test parsing markdown with single valid tool call."""
        result = response_parser._parse_markdown_format(markdown_response_samples["valid_single"])
        
        tool_calls, text_content = result
        
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "search_claims"
        assert tool_calls[0].parameters["query"] == "test"
        assert tool_calls[0].parameters["limit"] == 10

    def test_parse_markdown_valid_simple_format(self, response_parser, markdown_response_samples):
        """Test parsing markdown with simple key=value format."""
        result = response_parser._parse_markdown_format(markdown_response_samples["valid_simple"])
        
        tool_calls, text_content = result
        
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "search_claims"
        assert tool_calls[0].parameters["query"] == "test"
        assert tool_calls[0].parameters["limit"] == 10

    def test_parse_markdown_with_text_content(self, response_parser, markdown_response_samples):
        """Test parsing markdown with surrounding text."""
        result = response_parser._parse_markdown_format(markdown_response_samples["with_text"])
        
        tool_calls, text_content = result
        
        assert len(tool_calls) == 1
        assert "I'll search for the information" in text_content
        assert "Let me find those results" in text_content

    def test_parse_markdown_no_tool_calls(self, response_parser, markdown_response_samples):
        """Test parsing markdown without tool calls."""
        result = response_parser._parse_markdown_format(markdown_response_samples["no_tool_calls"])
        
        tool_calls, text_content = result
        
        assert len(tool_calls) == 0
        assert text_content == markdown_response_samples["no_tool_calls"]

    def test_parse_markdown_various_languages(self, response_parser):
        """Test parsing markdown with different language identifiers."""
        formats = [
            "```tool_call\nname: test\nparam: value\n```",
            "```tool\nname: test\nparam: value\n```",
            "```call\nname: test\nparam: value\n```"
        ]
        
        for fmt in formats:
            result = response_parser._parse_markdown_format(fmt)
            tool_calls, _ = result
            assert len(tool_calls) == 1
            assert tool_calls[0].name == "test"

    def test_parse_markdown_mixed_case(self, response_parser):
        """Test parsing markdown with mixed case language identifier."""
        md = "```TOOL_CALL\nname: test\nparam: value\n```"
        
        result = response_parser._parse_markdown_format(md)
        tool_calls, _ = result
        
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "test"

    def test_parse_markdown_with_comments(self, response_parser):
        """Test parsing markdown with comments and extra lines."""
        md = """
        # Tool call example
        ```tool_call
        name: search_claims
        # This is a comment
        query: machine learning
        limit: 5
        ```
        """
        
        result = response_parser._parse_markdown_format(md)
        tool_calls, _ = result
        
        assert len(tool_calls) == 1
        assert tool_calls[0].parameters["query"] == "machine learning"
        assert tool_calls[0].parameters["limit"] == 5

    def test_parse_markdown_quoted_values(self, response_parser):
        """Test parsing markdown with quoted parameter values."""
        md = """
        ```tool_call
        name: search_claims
        query: "machine learning"
        description: 'A search for ML content'
        ```
        """
        
        result = response_parser._parse_markdown_format(md)
        tool_calls, _ = result
        
        assert len(tool_calls) == 1
        assert tool_calls[0].parameters["query"] == "machine learning"
        assert tool_calls[0].parameters["description"] == "A search for ML content"

    def test_parse_markdown_type_conversion(self, response_parser):
        """Test parsing markdown with automatic type conversion."""
        md = """
        ```tool_call
        name: test_tool
        string_param: hello
        int_param: 42
        float_param: 3.14
        bool_param: true
        ```
        """
        
        result = response_parser._parse_markdown_format(md)
        tool_calls, _ = result
        
        assert len(tool_calls) == 1
        assert tool_calls[0].parameters["string_param"] == "hello"
        assert tool_calls[0].parameters["int_param"] == 42
        assert tool_calls[0].parameters["float_param"] == 3.14
        assert tool_calls[0].parameters["bool_param"] is True

    # Tool Call Parsing Tests
    def test_parse_invoke_element_missing_name(self, response_parser):
        """Test parsing invoke element without name."""
        xml = """
        <invoke>
            <parameter name="param">value</parameter>
        </invoke>
        """
        element = ET.fromstring(xml)
        
        result = response_parser._parse_invoke_element(element)
        
        assert result is None

    def test_parse_invoke_element_with_name_attribute(self, response_parser):
        """Test parsing invoke element with name attribute."""
        xml = """
        <invoke name="test_tool">
            <parameter name="param">value</parameter>
        </invoke>
        """
        element = ET.fromstring(xml)
        
        result = response_parser._parse_invoke_element(element)
        
        assert result is not None
        assert result.name == "test_tool"
        assert result.parameters["param"] == "value"

    def test_parse_invoke_element_with_name_child(self, response_parser):
        """Test parsing invoke element with name child element."""
        xml = """
        <invoke>
            <name>test_tool</name>
            <parameter name="param">value</parameter>
        </invoke>
        """
        element = ET.fromstring(xml)
        
        result = response_parser._parse_invoke_element(element)
        
        assert result is not None
        assert result.name == "test_tool"
        assert result.parameters["param"] == "value"

    def test_parse_nested_elements_simple(self, response_parser):
        """Test parsing nested elements with simple structure."""
        xml = """
        <root>
            <key1>value1</key1>
            <key2>42</key2>
        </root>
        """
        elements = list(ET.fromstring(xml))
        
        result = response_parser._parse_nested_elements(elements)
        
        assert result["key1"] == "value1"
        assert result["key2"] == 42

    def test_parse_nested_elements_with_duplicates(self, response_parser):
        """Test parsing nested elements with duplicate tags."""
        xml = """
        <root>
            <item>first</item>
            <item>second</item>
            <item>third</item>
        </root>
        """
        elements = list(ET.fromstring(xml))
        
        result = response_parser._parse_nested_elements(elements)
        
        assert isinstance(result["item"], list)
        assert result["item"] == ["first", "second", "third"]

    # Response Validation Tests
    def test_validate_response_structure_valid(self, response_parser):
        """Test validating response with valid structure."""
        response = """
        <tool_calls>
            <invoke name="test_tool">
                <parameter name="param">value</parameter>
            </invoke>
        </tool_calls>
        """
        
        is_valid, errors = response_parser.validate_response_structure(response)
        
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_response_structure_empty(self, response_parser):
        """Test validating empty response."""
        is_valid, errors = response_parser.validate_response_structure("")
        
        assert is_valid is False
        assert any("empty" in error.lower() for error in errors)

    def test_validate_response_structure_no_tool_calls(self, response_parser):
        """Test validating response with no tool calls."""
        response = "Just plain text no tools here."
        
        is_valid, errors = response_parser.validate_response_structure(response)
        
        assert is_valid is False
        assert any("No tool calls found" in error for error in errors)

    def test_validate_response_structure_missing_name(self, response_parser):
        """Test validating response with tool calls missing names."""
        response = """
        <tool_calls>
            <invoke>
                <parameter name="param">value</parameter>
            </invoke>
        </tool_calls>
        """
        
        is_valid, errors = response_parser.validate_response_structure(response)
        
        assert is_valid is False
        assert any("missing name" in error.lower() for error in errors)

    def test_validate_response_structure_invalid_parameters(self, response_parser):
        """Test validating response with invalid parameters."""
        response = """
        {
            "tool_calls": [
                {
                    "name": "test_tool",
                    "parameters": "should be object"
                }
            ]
        }
        """
        
        is_valid, errors = response_parser.validate_response_structure(response)
        
        assert is_valid is False
        assert any("parameters must be a dictionary" in error for error in errors)

    def test_validate_response_structure_parsing_error(self, response_parser):
        """Test validating response that causes parsing error."""
        response = "This will cause a parsing error: {{invalid json}}"
        
        is_valid, errors = response_parser.validate_response_structure(response)
        
        assert is_valid is False
        assert len(errors) >= 1

    # Formatting Tests
    def test_format_tool_call_response_single(self, response_parser):
        """Test formatting single tool call to XML."""
        tool_calls = [
            ToolCall(name="test_tool", parameters={"param1": "value1", "param2": 123})
        ]
        
        result = response_parser.format_tool_call_response(tool_calls)
        
        assert "<tool_calls>" in result
        assert "</tool_calls>" in result
        assert 'name="test_tool"' in result
        assert 'param1' in result
        assert 'param2' in result
        assert '123' in result

    def test_format_tool_call_response_multiple(self, response_parser):
        """Test formatting multiple tool calls to XML."""
        tool_calls = [
            ToolCall(name="tool1", parameters={"param": "value1"}),
            ToolCall(name="tool2", parameters={"param": "value2"}, call_id="call123")
        ]
        
        result = response_parser.format_tool_call_response(tool_calls)
        
        assert result.count("<invoke>") == 2
        assert result.count("</invoke>") == 2
        assert 'name="tool1"' in result
        assert 'name="tool2"' in result
        assert 'id="call123"' in result

    def test_format_tool_call_response_empty(self, response_parser):
        """Test formatting empty tool call list."""
        result = response_parser.format_tool_call_response([])
        
        assert result == ""

    def test_format_tool_call_response_complex_parameters(self, response_parser):
        """Test formatting tool calls with complex parameters."""
        tool_calls = [
            ToolCall(name="json_tool", parameters={
                "array": [1, 2, 3],
                "object": {"key": "value"},
                "string": "test",
                "boolean": True
            })
        ]
        
        result = response_parser.format_tool_call_response(tool_calls)
        
        assert '"array": [1, 2, 3]' in result or '"array": [1, 2, 3]' in result
        assert '"object": {"key": "value"}' in result
        assert 'test' in result
        assert 'true' in result

    # Text Extraction Tests
    def test_extract_text_from_response_xml(self, response_parser):
        """Test extracting text from XML response."""
        response = """
        Some intro text.
        <tool_calls>
            <invoke name="test_tool">
                <parameter name="param">value</parameter>
            </invoke>
        </tool_calls>
        Some outro text.
        """
        
        result = response_parser.extract_text_from_response(response)
        
        assert "Some intro text." in result
        assert "Some outro text." in result
        assert "<tool_calls>" not in result

    def test_extract_text_from_response_json(self, response_parser):
        """Test extracting text from JSON response."""
        response = 'Here is some text before { "tool_calls": [] } and some text after.'
        
        result = response_parser.extract_text_from_response(response)
        
        assert "Here is some text before" in result
        assert "and some text after" in result
        assert '{ "tool_calls": [] }' not in result

    def test_extract_text_from_response_plain_text(self, response_parser):
        """Test extracting text from plain text response."""
        response = "Just plain text with no tool calls."
        
        result = response_parser.extract_text_from_response(response)
        
        assert result == response

    def test_extract_text_from_response_malformed(self, response_parser):
        """Test extracting text from malformed response."""
        response = "Some text with <invalid>xml that won't parse"
        
        result = response_parser.extract_text_from_response(response)
        
        # Should return the text as-is since parsing failed
        assert "Some text with" in result

    # XML Fixing Tests
    def test_fix_xml_issues_ampersands(self, response_parser):
        """Test fixing unescaped ampersands."""
        xml = "<param>A & B</param>"
        
        result = response_parser._fix_xml_issues(xml)
        
        assert "A &amp; B" in result

    def test_fix_xml_issues_self_closing(self, response_parser):
        """Test fixing self-closing tags."""
        xml = "<param/>content"
        
        result = response_parser._fix_xml_issues(xml)
        
        assert "<param></param>" in result

    def test_fix_xml_issues_invalid_chars(self, response_parser):
        """Test fixing invalid characters."""
        xml = "content\x00with\x08invalid\x1Fchars"
        
        result = response_parser._fix_xml_issues(xml)
        
        assert "\x00" not in result
        assert "\x08" not in result
        assert "\x1F" not in result

    # Simple Parameter Parsing Tests
    def test_parse_simple_parameters_basic(self, response_parser):
        """Test parsing simple key=value parameters."""
        param_str = "key1=value1, key2=42, key3=true"
        
        result = response_parser._parse_simple_parameters(param_str)
        
        assert result["key1"] == "value1"
        assert result["key2"] == 42
        assert result["key3"] is True

    def test_parse_simple_parameters_quoted(self, response_parser):
        """Test parsing parameters with quoted values."""
        param_str = 'key1="value with spaces", key2=\'single quoted\''
        
        result = response_parser._parse_simple_parameters(param_str)
        
        assert result["key1"] == "value with spaces"
        assert result["key2"] == "single quoted"

    def test_parse_simple_parameters_empty(self, response_parser):
        """Test parsing empty parameter string."""
        result = response_parser._parse_simple_parameters("")
        
        assert result == {}

    def test_parse_simple_parameters_malformed(self, response_parser):
        """Test parsing malformed parameter string."""
        param_str = "key1=value1, malformed, key2=value2"
        
        result = response_parser._parse_simple_parameters(param_str)
        
        # Should handle gracefully
        assert "key1" in result
        assert "key2" in result


class TestResponseParserIntegration:
    """Integration tests for ResponseParser."""

    def test_parse_response_integration_xml(self, response_parser, xml_response_samples):
        """Test complete XML parsing integration."""
        response = xml_response_samples["valid_multiple"]
        
        result = response_parser.parse_response(response)
        
        assert isinstance(result, ParsedResponse)
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "search_claims"
        assert result.tool_calls[1].name == "create_claim"
        assert len(result.parsing_errors) == 0

    def test_parse_response_integration_json(self, response_parser, json_response_samples):
        """Test complete JSON parsing integration."""
        response = json_response_samples["valid_single"]
        
        result = response_parser.parse_response(response)
        
        assert isinstance(result, ParsedResponse)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search_claims"
        assert len(result.parsing_errors) == 0

    def test_parse_response_integration_markdown(self, response_parser, markdown_response_samples):
        """Test complete markdown parsing integration."""
        response = markdown_response_samples["valid_simple"]
        
        result = response_parser.parse_response(response)
        
        assert isinstance(result, ParsedResponse)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search_claims"
        assert len(result.parsing_errors) == 0

    def test_parse_response_fallback_text(self, response_parser):
        """Test fallback to text content when no tool calls found."""
        response = "This is just text with no tool calls in any format."
        
        result = response_parser.parse_response(response)
        
        assert isinstance(result, ParsedResponse)
        assert len(result.tool_calls) == 0
        assert result.text_content == response
        assert len(result.parsing_errors) == 1

    def test_parse_response_multiple_format_fallback(self, response_parser):
        """Test trying multiple formats and falling back."""
        # Response that looks like it might be valid but isn't
        response = "Not XML {not valid JSON} ```no tool call```"
        
        result = response_parser.parse_response(response)
        
        assert isinstance(result, ParsedResponse)
        assert len(result.tool_calls) == 0
        assert result.text_content == response
        assert len(result.parsing_errors) >= 1


if __name__ == "__main__":
    pytest.main([__file__])