"""
LLM Response Parser for the Conjecture skill-based agency system.
Parses XML-like structured tool calls from LLM responses.
"""

import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Tuple
import json
import logging

from ..core.models import ToolCall, ParsedResponse

logger = logging.getLogger(__name__)

class ResponseParser:
    """
    Parses LLM responses to extract structured tool calls.

    Supports multiple formats:
    - XML-like: <tool_calls><invoke name="skill_name">...</invoke></tool_calls>
    - JSON: {"tool_calls": [{"name": "skill_name", "parameters": {...}}]}
    - Markdown: ```tool_call\nname: skill_name\nparameters: {...}\n```
    """

    def __init__(self):
        self.supported_formats = ["xml", "json", "markdown"]
        self.xml_namespaces = {"tc": "http://conjecture.ai/tool_calls"}

    def parse_response(self, response: str) -> ParsedResponse:
        """
        Parse an LLM response and extract tool calls.

        Args:
            response: Raw LLM response text

        Returns:
            ParsedResponse with tool calls and any errors
        """
        if not response or not response.strip():
            return ParsedResponse(
                tool_calls=[], text_content=response, parsing_errors=["Empty response"]
            )

        # Try different parsing formats in order of preference
        for format_type in self.supported_formats:
            try:
                if format_type == "xml":
                    tool_calls, text_content = self._parse_xml_format(response)
                elif format_type == "json":
                    tool_calls, text_content = self._parse_json_format(response)
                elif format_type == "markdown":
                    tool_calls, text_content = self._parse_markdown_format(response)

                if tool_calls:
                    return ParsedResponse(
                        tool_calls=tool_calls,
                        text_content=text_content,
                        parsing_errors=[],
                    )

            except Exception as e:
                logger.debug(f"Failed to parse as {format_type}: {e}")
                continue

        # If no tool calls found, return as text content
        return ParsedResponse(
            tool_calls=[],
            text_content=response,
            parsing_errors=["No valid tool calls found in response"],
        )

    def _parse_xml_format(self, response: str) -> Tuple[List[ToolCall], Optional[str]]:
        """Parse XML-like tool call format."""
        tool_calls = []
        text_content = None

        try:
            # Look for tool_calls element
            tool_calls_match = re.search(
                r"<tool_calls[^>]*>(.*?)</tool_calls>",
                response,
                re.DOTALL | re.IGNORECASE,
            )

            if not tool_calls_match:
                return [], response

            tool_calls_xml = tool_calls_match.group(1)

            # Parse the XML content
            try:
                root = ET.fromstring(f"<root>{tool_calls_xml}</root>")
            except ET.ParseError as e:
                # Try to fix common XML issues
                fixed_xml = self._fix_xml_issues(tool_calls_xml)
                root = ET.fromstring(f"<root>{fixed_xml}</root>")

            # Extract tool calls
            for invoke_elem in root.findall(".//invoke"):
                tool_call = self._parse_invoke_element(invoke_elem)
                if tool_call:
                    tool_calls.append(tool_call)

            # Extract text content outside tool_calls
            text_parts = []
            before = response[: tool_calls_match.start()]
            after = response[tool_calls_match.end() :]

            if before.strip():
                text_parts.append(before.strip())
            if after.strip():
                text_parts.append(after.strip())

            text_content = "\n".join(text_parts) if text_parts else None

        except Exception as e:
            logger.error(f"XML parsing error: {e}")
            raise

        return tool_calls, text_content

    def _parse_invoke_element(self, invoke_elem: ET.Element) -> Optional[ToolCall]:
        """Parse an <invoke> element to extract tool call information."""
        try:
            # Get tool name
            name = invoke_elem.get("name")
            if not name:
                name = invoke_elem.findtext("name")

            if not name:
                logger.warning("Tool call missing name attribute")
                return None

            # Extract parameters
            parameters = {}

            # Parse <parameter> elements
            for param_elem in invoke_elem.findall("parameter"):
                param_name = param_elem.get("name")
                if param_name:
                    # Handle different parameter value formats
                    if param_elem.text:
                        # Try to parse as JSON first
                        try:
                            value = json.loads(param_elem.text)
                        except json.JSONDecodeError:
                            value = param_elem.text.strip()
                        parameters[param_name] = value
                    else:
                        # Check for nested elements
                        children = list(param_elem)
                        if children:
                            parameters[param_name] = self._parse_nested_elements(
                                children
                            )
                        else:
                            parameters[param_name] = None

            # Parse inline parameters (e.g., <param_name>value</param_name>)
            for child in invoke_elem:
                if child.tag not in ["parameter", "name"]:
                    if child.text:
                        try:
                            value = json.loads(child.text)
                        except json.JSONDecodeError:
                            value = child.text.strip()
                        parameters[child.tag] = value

            # Generate call ID if not present
            call_id = invoke_elem.get("id")

            return ToolCall(name=name, parameters=parameters, call_id=call_id)

        except Exception as e:
            logger.error(f"Error parsing invoke element: {e}")
            return None

    def _parse_nested_elements(self, elements: List[ET.Element]) -> Dict[str, Any]:
        """Parse nested XML elements into a dictionary."""
        result = {}

        for elem in elements:
            if elem.text:
                try:
                    value = json.loads(elem.text)
                except json.JSONDecodeError:
                    value = elem.text.strip()
            else:
                # Check for nested children
                children = list(elem)
                if children:
                    value = self._parse_nested_elements(children)
                else:
                    value = None

            # Handle multiple elements with same tag
            if elem.tag in result:
                if not isinstance(result[elem.tag], list):
                    result[elem.tag] = [result[elem.tag]]
                result[elem.tag].append(value)
            else:
                result[elem.tag] = value

        return result

    def _fix_xml_issues(self, xml_content: str) -> str:
        """Fix common XML formatting issues."""
        # Escape unescaped ampersands
        xml_content = re.sub(r"&(?![a-zA-Z#])", "&amp;", xml_content)

        # Fix self-closing tags
        xml_content = re.sub(r"<(\w+)([^>]*)/>", r"<\1\2></\1>", xml_content)

        # Remove invalid characters
        xml_content = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", xml_content)

        return xml_content

    def _parse_json_format(self, response: str) -> Tuple[List[ToolCall], Optional[str]]:
        """Parse JSON tool call format."""
        tool_calls = []
        text_content = None

        try:
            # Look for JSON object with tool_calls
            json_match = re.search(r'\{[^{}]*"tool_calls"[^{}]*\}', response, re.DOTALL)

            if not json_match:
                # Try to find any JSON object
                json_match = re.search(r"\{.*\}", response, re.DOTALL)

            if not json_match:
                return [], response

            json_str = json_match.group(0)
            data = json.loads(json_str)

            # Extract tool calls
            if "tool_calls" in data:
                tool_calls_data = data["tool_calls"]
            elif "calls" in data:
                tool_calls_data = data["calls"]
            else:
                # Assume the entire object is a single tool call
                tool_calls_data = [data]

            for call_data in tool_calls_data:
                tool_call = self._parse_json_tool_call(call_data)
                if tool_call:
                    tool_calls.append(tool_call)

            # Extract text content outside JSON
            text_parts = []
            before = response[: json_match.start()]
            after = response[json_match.end() :]

            if before.strip():
                text_parts.append(before.strip())
            if after.strip():
                text_parts.append(after.strip())

            text_content = "\n".join(text_parts) if text_parts else None

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error parsing JSON format: {e}")
            raise

        return tool_calls, text_content

    def _parse_json_tool_call(self, call_data: Dict[str, Any]) -> Optional[ToolCall]:
        """Parse a single tool call from JSON data."""
        try:
            name = (
                call_data.get("name")
                or call_data.get("function")
                or call_data.get("tool")
            )
            if not name:
                logger.warning("JSON tool call missing name")
                return None

            parameters = (
                call_data.get("parameters", {}) or call_data.get("args", {}) or {}
            )
            call_id = call_data.get("id") or call_data.get("call_id")

            return ToolCall(name=name, parameters=parameters, call_id=call_id)

        except Exception as e:
            logger.error(f"Error parsing JSON tool call: {e}")
            return None

    def _parse_markdown_format(
        self, response: str
    ) -> Tuple[List[ToolCall], Optional[str]]:
        """Parse markdown tool call format."""
        tool_calls = []
        text_content = None

        try:
            # Look for code blocks with tool_call language
            code_block_pattern = r"```(?:tool_call|tool|call)\n(.*?)\n```"
            code_blocks = re.findall(
                code_block_pattern, response, re.DOTALL | re.IGNORECASE
            )

            for block in code_blocks:
                tool_call = self._parse_markdown_tool_call(block)
                if tool_call:
                    tool_calls.append(tool_call)

            # Extract text content outside code blocks
            text_parts = re.split(
                code_block_pattern, response, flags=re.DOTALL | re.IGNORECASE
            )
            text_parts = [part.strip() for part in text_parts if part.strip()]

            text_content = "\n".join(text_parts) if text_parts else None

        except Exception as e:
            logger.error(f"Error parsing markdown format: {e}")
            raise

        return tool_calls, text_content

    def _parse_markdown_tool_call(self, block: str) -> Optional[ToolCall]:
        """Parse a single tool call from markdown block."""
        try:
            lines = block.strip().split("\n")
            name = None
            parameters = {}
            call_id = None

            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Parse key-value pairs
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower()
                    value = value.strip()

                    if key in ["name", "function", "tool"]:
                        name = value
                    elif key in ["id", "call_id"]:
                        call_id = value
                    elif key == "parameters":
                        # Try to parse as JSON
                        try:
                            parameters = json.loads(value)
                        except json.JSONDecodeError:
                            # Parse as simple key=value format
                            parameters = self._parse_simple_parameters(value)
                    else:
                        # Add to parameters
                        try:
                            parameters[key] = json.loads(value)
                        except json.JSONDecodeError:
                            parameters[key] = value

            if not name:
                logger.warning("Markdown tool call missing name")
                return None

            return ToolCall(name=name, parameters=parameters, call_id=call_id)

        except Exception as e:
            logger.error(f"Error parsing markdown tool call: {e}")
            return None

    def _parse_simple_parameters(self, param_str: str) -> Dict[str, Any]:
        """Parse simple key=value parameter format."""
        parameters = {}

        # Split by comma, but handle quoted values
        parts = re.findall(r'([^=,]+(?:="[^"]*"|=\'[^\']*\'|=[^,]+))', param_str)

        for part in parts:
            if "=" in part:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()

                # Remove quotes if present
                if (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    value = value[1:-1]

                # Try to convert to appropriate type
                try:
                    if value.lower() in ["true", "false"]:
                        value = value.lower() == "true"
                    elif value.isdigit():
                        value = int(value)
                    elif "." in value and value.replace(".", "").isdigit():
                        value = float(value)
                except ValueError:
                    pass

                parameters[key] = value

        return parameters

    def validate_response_structure(self, response: str) -> Tuple[bool, List[str]]:
        """
        Validate if response contains properly structured tool calls.

        Args:
            response: Response to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if not response or not response.strip():
            errors.append("Response is empty")
            return False, errors

        try:
            parsed = self.parse_response(response)

            if not parsed.has_tool_calls():
                errors.append("No tool calls found in response")
                return False, errors

            # Validate each tool call
            for i, tool_call in enumerate(parsed.tool_calls):
                if not tool_call.name:
                    errors.append(f"Tool call {i + 1} missing name")

                if not isinstance(tool_call.parameters, dict):
                    errors.append(f"Tool call {i + 1} parameters must be a dictionary")

            return len(errors) == 0, errors

        except Exception as e:
            errors.append(f"Parsing error: {str(e)}")
            return False, errors

    def format_tool_call_response(self, tool_calls: List[ToolCall]) -> str:
        """
        Format tool calls into XML-like response string.

        Args:
            tool_calls: List of tool calls to format

        Returns:
            Formatted XML-like string
        """
        if not tool_calls:
            return ""

        xml_parts = ["<tool_calls>"]

        for tool_call in tool_calls:
            xml_parts.append(f'  <invoke name="{tool_call.name}"')
            if tool_call.call_id:
                xml_parts.append(f' id="{tool_call.call_id}"')
            xml_parts.append(">")

            for param_name, param_value in tool_call.parameters.items():
                if isinstance(param_value, (dict, list)):
                    param_str = json.dumps(param_value, ensure_ascii=False)
                elif isinstance(param_value, str):
                    param_str = param_value
                elif isinstance(param_value, bool):
                    param_str = str(param_value).lower()
                else:
                    param_str = str(param_value)

                xml_parts.append(
                    f'    <parameter name="{param_name}">{param_str}</parameter>'
                )

            xml_parts.append("  </invoke>")

        xml_parts.append("</tool_calls>")

        return "\n".join(xml_parts)

    def extract_text_from_response(self, response: str) -> str:
        """
        Extract only the text content from a response, removing tool calls.

        Args:
            response: Response to extract text from

        Returns:
            Text content without tool calls
        """
        try:
            parsed = self.parse_response(response)
            return parsed.text_content or ""
        except Exception:
            # If parsing fails, try to remove obvious tool call patterns
            # Remove XML tool calls
            text = re.sub(
                r"<tool_calls>.*?</tool_calls>",
                "",
                response,
                flags=re.DOTALL | re.IGNORECASE,
            )
            # Remove JSON tool calls
            text = re.sub(r'\{[^{}]*"tool_calls"[^{}]*\}', "", text, flags=re.DOTALL)
            # Remove markdown tool calls
            text = re.sub(
                r"```(?:tool_call|tool|call)\n.*?\n```",
                "",
                text,
                flags=re.DOTALL | re.IGNORECASE,
            )

            return text.strip()
