"""
Prompt System - Core LLM prompt assembly and response parsing for Conjecture.
Handles the communication layer between the agent and LLM.
"""
import re
import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Builds prompts for LLM by assembling context, skills, and tools.
    """
    
    def __init__(self):
        self.system_prompt = self._get_system_prompt()
        self.max_prompt_length = 8000  # tokens
    
    def assemble_prompt(self, context, user_request: str) -> str:
        """
        Assemble a complete prompt for the LLM.
        
        Args:
            context: Context object with relevant information
            user_request: User's request
            
        Returns:
            Complete prompt string
        """
        try:
            prompt_parts = []
            
            # System prompt
            prompt_parts.append(self.system_prompt)
            prompt_parts.append("")
            
            # Context section
            context_section = self._build_context_section(context)
            if context_section:
                prompt_parts.append("=== CONTEXT ===")
                prompt_parts.append(context_section)
                prompt_parts.append("")
            
            # User request
            prompt_parts.append("=== REQUEST ===")
            prompt_parts.append(user_request)
            prompt_parts.append("")
            
            # Instructions
            prompt_parts.append("=== INSTRUCTIONS ===")
            prompt_parts.append("Please respond to the request using the available tools and following the relevant skill guidance. Use tool calls when appropriate and create claims to capture important information.")
            
            # Join and trim if needed
            full_prompt = "\n".join(prompt_parts)
            
            if len(full_prompt) > self.max_prompt_length * 4:  # Rough token estimation
                full_prompt = self._trim_prompt(full_prompt)
            
            return full_prompt
            
        except Exception as e:
            logger.error(f"Error assembling prompt: {e}")
            return self._create_emergency_prompt(user_request)
    
    def _build_context_section(self, context) -> str:
        """Build the context section of the prompt."""
        context_parts = []
        
        # Available tools
        if context.available_tools:
            context_parts.append("AVAILABLE TOOLS:")
            for tool in context.available_tools:
                tool_desc = f"- {tool['name']}: {tool['description']}"
                if 'example' in tool:
                    tool_desc += f"\n  Example: {tool['example']}"
                context_parts.append(tool_desc)
            context_parts.append("")
        
        # Skill templates
        if context.skill_templates:
            context_parts.append("RELEVANT SKILLS:")
            for skill in context.skill_templates:
                context_parts.append(f"Skill: {skill['name']}")
                context_parts.append(f"Description: {skill['description']}")
                context_parts.append("Steps:")
                for i, step in enumerate(skill['steps'], 1):
                    context_parts.append(f"  {i}. {step}")
                if 'example_usage' in skill:
                    context_parts.append(f"Example: {skill['example_usage']}")
                context_parts.append("")
        
        # Relevant claims
        if context.relevant_claims:
            context_parts.append("RELEVANT CLAIMS:")
            for claim in context.relevant_claims:
                claim_info = f"- Claim {claim.id}: {claim.content}"
                claim_info += f" (confidence: {claim.confidence:.2f})"
                if claim.tags:
                    claim_info += f" [tags: {', '.join(claim.tags)}]"
                context_parts.append(claim_info)
            context_parts.append("")
        
        # Session history
        if context.session_history:
            context_parts.append("RECENT CONVERSATION:")
            for interaction in context.session_history[-3:]:  # Last 3 interactions
                context_parts.append(f"User: {interaction['user_request']}")
                context_parts.append(f"Assistant: {interaction['llm_response']}")
                context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _trim_prompt(self, prompt: str) -> str:
        """Trim prompt to fit within length limits."""
        # Simple truncation - could be made smarter
        lines = prompt.split('\n')
        
        # Keep system prompt and request, trim context
        system_lines = []
        request_lines = []
        context_lines = []
        
        current_section = None
        
        for line in lines:
            if line.startswith("=== SYSTEM ===") or line == self.system_prompt.split('\n')[0]:
                current_section = 'system'
            elif line.startswith("=== CONTEXT ==="):
                current_section = 'context'
            elif line.startswith("=== REQUEST ==="):
                current_section = 'request'
            elif line.startswith("=== INSTRUCTIONS ==="):
                current_section = 'instructions'
            
            if current_section == 'system':
                system_lines.append(line)
            elif current_section == 'context':
                context_lines.append(line)
            elif current_section == 'request':
                request_lines.append(line)
            elif current_section == 'instructions':
                request_lines.append(line)  # Include instructions with request
        
        # Reassemble with trimmed context
        result_lines = system_lines
        
        # Add as much context as will fit
        max_context_length = (self.max_prompt_length * 4) - len('\n'.join(system_lines + request_lines))
        context_text = '\n'.join(context_lines)
        
        if len(context_text) > max_context_length:
            # Truncate context from the middle
            context_text = context_text[:max_context_length//2] + "\n... [context truncated] ...\n" + context_text[-max_context_length//2:]
        
        result_lines.extend(context_text.split('\n'))
        result_lines.extend(request_lines)
        
        return '\n'.join(result_lines)
    
    def _create_emergency_prompt(self, user_request: str) -> str:
        """Create an emergency prompt when normal assembly fails."""
        return f"""You are a helpful AI assistant with access to tools for research, coding, and knowledge management.

Available tools:
- WebSearch: Search for information online
- ReadFiles: Read file contents
- WriteCodeFile: Write code to files
- CreateClaim: Create knowledge claims with confidence scores
- ClaimSupport: Link evidence to support claims

Please help with this request: {user_request}

Use tools when appropriate and create claims to capture important information."""
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM."""
        return """You are Conjecture, an AI assistant that helps with research, coding, and knowledge management. You have access to tools for gathering information and creating structured knowledge claims.

Your core approach is to:
1. Understand the user's request clearly
2. Use relevant skills to guide your thinking process
3. Use available tools to gather information and create solutions
4. Create claims to capture important knowledge with confidence scores
5. Support claims with evidence when possible

When you need to use tools, format your tool calls like this:
<tool_calls>
  <invoke name="ToolName">
    <parameter name="parameter_name">parameter_value</parameter>
  </invoke>
</tool_calls>

Always create claims for important information you discover or generate. Claims should have confidence scores between 0.0 and 1.0."""


class ResponseParser:
    """
    Parses LLM responses to extract tool calls, claims, and other structured information.
    """
    
    def __init__(self):
        self.tool_call_pattern = re.compile(r'<tool_calls>(.*?)</tool_calls>', re.DOTALL | re.IGNORECASE)
        self.invoke_pattern = re.compile(r'<invoke\s+name="([^"]+)"[^>]*>(.*?)</invoke>', re.DOTALL | re.IGNORECASE)
        self.parameter_pattern = re.compile(r'<parameter\s+name="([^"]+)"[^>]*>(.*?)</parameter>', re.DOTALL | re.IGNORECASE)
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response to extract structured information.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed response dictionary
        """
        try:
            parsed = {
                "raw_response": response,
                "tool_calls": [],
                "claims": [],
                "text_content": response,
                "errors": []
            }
            
            # Extract tool calls
            tool_calls = self._extract_tool_calls(response)
            parsed["tool_calls"] = tool_calls
            
            # Extract claims (simple pattern matching)
            claims = self._extract_claims(response)
            parsed["claims"] = claims
            
            # Extract text content (remove tool calls)
            text_content = self._extract_text_content(response)
            parsed["text_content"] = text_content
            
            return parsed
            
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return {
                "raw_response": response,
                "tool_calls": [],
                "claims": [],
                "text_content": response,
                "errors": [str(e)]
            }
    
    def _extract_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """Extract tool calls from response."""
        tool_calls = []
        
        try:
            # Find tool_calls section
            tool_calls_match = self.tool_call_pattern.search(response)
            if not tool_calls_match:
                return tool_calls
            
            tool_calls_xml = tool_calls_match.group(1)
            
            # Parse individual invoke elements
            invoke_matches = self.invoke_pattern.findall(tool_calls_xml)
            
            for tool_name, invoke_content in invoke_matches:
                tool_call = {
                    "name": tool_name.strip(),
                    "parameters": {}
                }
                
                # Extract parameters
                param_matches = self.parameter_pattern.findall(invoke_content)
                for param_name, param_value in param_matches:
                    # Try to parse as JSON first
                    try:
                        parsed_value = json.loads(param_value.strip())
                        tool_call["parameters"][param_name.strip()] = parsed_value
                    except json.JSONDecodeError:
                        # Use as string
                        tool_call["parameters"][param_name.strip()] = param_value.strip()
                
                tool_calls.append(tool_call)
            
        except Exception as e:
            logger.error(f"Error extracting tool calls: {e}")
        
        return tool_calls
    
    def _extract_claims(self, response: str) -> List[Dict[str, Any]]:
        """Extract claims from response using pattern matching."""
        claims = []
        
        try:
            # Look for claim indicators
            claim_patterns = [
                r'claim[:\s]+([^.\n]+)',
                r'i claim that ([^.\n]+)',
                r'it is true that ([^.\n]+)',
                r'evidence suggests ([^.\n]+)',
                r'based on.*?([^.\n]+)'
            ]
            
            for pattern in claim_patterns:
                matches = re.findall(pattern, response, re.IGNORECASE)
                for match in matches:
                    claim_text = match.strip()
                    if len(claim_text) > 10:  # Minimum length
                        # Estimate confidence based on language
                        confidence = self._estimate_confidence(claim_text, response)
                        
                        claims.append({
                            "content": claim_text,
                            "confidence": confidence,
                            "source": "llm_response",
                            "evidence": []
                        })
            
        except Exception as e:
            logger.error(f"Error extracting claims: {e}")
        
        return claims
    
    def _estimate_confidence(self, claim_text: str, full_response: str) -> float:
        """Estimate confidence score for a claim based on language."""
        confidence = 0.5  # Default
        
        # Look for confidence indicators
        high_confidence_words = ['certain', 'definitely', 'clearly', 'proven', 'established']
        medium_confidence_words = ['likely', 'probably', 'seems', 'appears', 'suggests']
        low_confidence_words = ['might', 'could', 'possibly', 'perhaps', 'uncertain']
        
        claim_lower = claim_text.lower()
        
        for word in high_confidence_words:
            if word in claim_lower:
                confidence = max(confidence, 0.8)
        
        for word in medium_confidence_words:
            if word in claim_lower:
                confidence = max(confidence, 0.6)
        
        for word in low_confidence_words:
            if word in claim_lower:
                confidence = min(confidence, 0.4)
        
        # Look for evidence mentions
        if any(word in full_response.lower() for word in ['evidence', 'data', 'source', 'study', 'research']):
            confidence = min(confidence + 0.1, 1.0)
        
        return round(confidence, 2)
    
    def _extract_text_content(self, response: str) -> str:
        """Extract text content without tool calls."""
        try:
            # Remove tool_calls sections
            text = self.tool_call_pattern.sub('', response)
            
            # Clean up extra whitespace
            text = re.sub(r'\n\s*\n', '\n\n', text)
            text = text.strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text content: {e}")
            return response
    
    def validate_tool_call(self, tool_call: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a tool call for correctness."""
        errors = []
        
        if not tool_call.get('name'):
            errors.append("Tool call missing name")
        
        if not isinstance(tool_call.get('parameters'), dict):
            errors.append("Tool call parameters must be a dictionary")
        
        return len(errors) == 0, errors
    
    def validate_claim(self, claim: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a claim for correctness."""
        errors = []
        
        if not claim.get('content'):
            errors.append("Claim missing content")
        elif len(claim['content']) < 10:
            errors.append("Claim content too short")
        
        confidence = claim.get('confidence')
        if confidence is None:
            errors.append("Claim missing confidence")
        elif not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            errors.append("Claim confidence must be between 0.0 and 1.0")
        
        return len(errors) == 0, errors
    
    def get_parsing_stats(self) -> Dict[str, Any]:
        """Get statistics about response parsing."""
        return {
            "tool_call_patterns": len(self.tool_call_pattern.pattern),
            "invoke_patterns": len(self.invoke_pattern.pattern),
            "parameter_patterns": len(self.parameter_pattern.pattern),
            "claim_patterns": 5  # Number of claim extraction patterns
        }