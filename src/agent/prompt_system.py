
    def _quick_self_critique(self, response: str, problem_type: str) -> Dict[str, Any]:
        """Lightweight self-critique layer for common reasoning errors"""
        critiques = []
        confidence_boost = 1.0

        # Mathematical critiques
        if "math" in problem_type or any(word in response.lower() for word in ["multiply", "add", "calculate", "×", "+"]):
            # Check for calculation consistency
            if "×" in response and "=" in response:
                # Look for inconsistent multiplication patterns
                lines = response.split('\n')
                for line in lines:
                    if "×" in line and "=" in line:
                        if "=>" not in line:  # Not a step-by-step explanation
                            # Check if calculation makes sense
                            if any(num in line for num in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]):
                                critiques.append("Consider showing calculation steps for clarity")
                                confidence_boost *= 0.95

        # Logical critiques
        if "logic" in problem_type or any(word in response.lower() for word in ["if", "then", "conclude"]):
            # Check for clear premise-conclusion structure
            if not any(marker in response.lower() for marker in ["premise", "conclusion", "therefore", "thus"]):
                critiques.append("Logic reasoning could benefit from clearer premise-conclusion structure")
                confidence_boost *= 0.9

        # General quality critiques
        if len(response) < 50:
            critiques.append("Response appears too brief - consider more detailed explanation")
            confidence_boost *= 0.85
        elif len(response) > 1000:
            critiques.append("Response is quite long - consider focusing on key points")
            confidence_boost *= 0.97

        # Quality scoring
        quality_score = confidence_boost
        if not critiques:
            quality_score *= 1.1  # Bonus for no issues found
        quality_score = min(1.0, quality_score)

        return {
            "critiques": critiques,
            "confidence_boost": confidence_boost,
            "quality_score": quality_score,
            "needs_revision": len(critiques) > 2
        }


    def _get_self_verification_prompt(self, problem_text: str, answer: str) -> str:
        """Get self-verification prompt for error detection and correction"""
        problem_lower = problem_text.lower()
        answer_lower = answer.lower()

        # Mathematical verification
        if any(word in problem_lower for word in ['calculate', 'multiply', 'add', 'subtract', 'divide', 'percent', 'what is', 'how many']):
            return f"""SELF-VERIFICATION CHECKLIST:
Please verify your answer to this mathematical problem.

ORIGINAL PROBLEM: {problem_text}
YOUR ANSWER: {answer}

VERIFICATION STEPS:
1. Calculation Check:
   - Recalculate the problem using a different method
   - Verify arithmetic operations step-by-step
   - Check for common calculation errors

2. Reasonableness Check:
   - Does the answer make sense in context?
   - Can you estimate to verify the magnitude?
   - Are units correct?

3. Completeness Check:
   - Did you answer exactly what was asked?
   - Are all parts of the problem addressed?
   - Is the final answer clearly stated?

4. Confidence Assessment:
   - Rate your confidence in this answer (0-100%)
   - What are the potential sources of error?
   - Would you like to revise your answer?

If you find any errors, please provide the corrected answer with explanation."""

        # Logical verification
        elif any(word in problem_lower for word in ['if', 'then', 'conclude', 'logic', 'premise', 'assume', 'yes or no']):
            return f"""SELF-VERIFICATION CHECKLIST:
Please verify your reasoning to this logical problem.

ORIGINAL PROBLEM: {problem_text}
YOUR ANSWER: {answer}

VERIFICATION STEPS:
1. Premise Analysis:
   - Did you correctly identify all given premises?
   - Are there any hidden assumptions you made?
   - Are the premises clearly stated and understood?

2. Logical Validity:
   - Does your conclusion necessarily follow from premises?
   - Are there any logical fallacies in your reasoning?
   - Can you think of counterexamples?

3. Completeness Check:
   - Did you address the exact question asked?
   - Is your reasoning fully explained?
   - Is the final answer clear and unambiguous?

4. Confidence Assessment:
   - Rate your confidence in this reasoning (0-100%)
   - What are the potential weak points?
   - Would you like to revise your answer?

If you find any issues, please provide the corrected reasoning with explanation."""

        # Default verification
        else:
            return f"""SELF-VERIFICATION CHECKLIST:
Please verify your answer to this problem.

ORIGINAL PROBLEM: {problem_text}
YOUR ANSWER: {answer}

VERIFICATION STEPS:
1. Understanding Check:
   - Did you correctly understand what was asked?
   - Are all parts of the question addressed?
   - Is there any ambiguity in interpretation?

2. Reasoning Quality:
   - Is your reasoning clear and logical?
   - Are your steps well-explained?
   - Can you identify any potential flaws?

3. Answer Appropriateness:
   - Does your answer directly address the question?
   - Is the answer complete and accurate?
   - Is the final answer clearly stated?

4. Confidence Assessment:
   - Rate your confidence in this answer (0-100%)
   - What are potential sources of error?
   - Would you like to revise your answer?

If you find any issues, please provide the corrected answer with explanation."""

    def _get_context_for_problem_type(self, problem_text: str) -> str:
        """Get problem-type-specific context scaffolding"""
        problem_lower = problem_text.lower()

        # Mathematical context
        if any(word in problem_lower for word in ['calculate', 'multiply', 'add', 'subtract', 'divide', 'percent', 'what is', 'how many']):
            return """MATHEMATICAL CONTEXT:
- Break down calculations into clear steps
- Write out intermediate results
- Double-check arithmetic operations
- Consider estimation to verify reasonableness
- Use standard mathematical notation

USEFUL FRAMEWORKS:
1. Identify the operation needed
2. Extract all numbers and values
3. Set up the calculation
4. Execute step-by-step
5. Verify the result makes sense"""

        # Logical context
        elif any(word in problem_lower for word in ['if', 'then', 'conclude', 'logic', 'premise', 'assume', 'yes or no']):
            return """LOGICAL CONTEXT:
- Identify premises and conclusions
- Check for hidden assumptions
- Consider counterexamples
- Distinguish between necessary and sufficient conditions
- Avoid logical fallacies

USEFUL FRAMEWORKS:
1. List all given premises
2. Identify what needs to be proven
3. Consider if the conclusion necessarily follows
4. Look for alternative interpretations
5. Provide clear logical justification"""

        # Default mixed context
        else:
            return """MIXED PROBLEM CONTEXT:
- Identify the dominant domain (math or logic)
- Apply appropriate reasoning strategies
- Break complex problems into simpler parts
- Consider multiple solution approaches
- Provide clear justification for conclusions"""

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
            
            # System prompt with context integration
            system_prompt = self.system_prompt
            # Use the user_request parameter directly for context detection
            if user_request:
                context_info = self._get_context_for_problem_type(user_request)
                system_prompt += f"\n\n{context_info}"
            prompt_parts.append(system_prompt)
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
        """Get the domain-adaptive system prompt for the LLM."""
        return """You are Conjecture, an adaptive AI system that matches reasoning approach to problem domain.

DOMAIN-ADAPTIVE APPROACH:
For MATHEMATICAL problems:
- Focus on calculation accuracy and step-by-step reasoning
- Use mathematical knowledge and problem-solving strategies
- Work through calculations systematically and verify results

For LOGICAL problems:
- Focus on premise analysis and valid inference
- Carefully examine what is explicitly stated vs implied
- Provide clear logical justification for conclusions

For MIXED problems:
- Identify which domain dominates the problem
- Apply appropriate reasoning strategies
- Maintain clarity and precision in your approach

CORE PRINCIPLES:
1. Match reasoning strategy to problem type
2. Use domain-specific knowledge effectively
3. Provide clear, accurate solutions
4. Avoid adding unnecessary complexity

When solving problems, identify the domain first, then apply the most appropriate reasoning approach."""


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

    def _get_context_for_problem_type(self, problem_text: str) -> str:
        """Get problem-type-specific context scaffolding"""
        problem_lower = problem_text.lower()

        # Mathematical context
        if any(word in problem_lower for word in ['calculate', 'multiply', 'add', 'subtract', 'divide', 'percent', 'what is', 'how many']):
            return """MATHEMATICAL CONTEXT:
- Break down calculations into clear steps
- Write out intermediate results
- Double-check arithmetic operations
- Consider estimation to verify reasonableness
- Use standard mathematical notation

USEFUL FRAMEWORKS:
1. Identify the operation needed
2. Extract all numbers and values
3. Set up the calculation
4. Execute step-by-step
5. Verify the result makes sense"""

        # Logical context
        elif any(word in problem_lower for word in ['if', 'then', 'conclude', 'logic', 'premise', 'assume', 'yes or no']):
            return """LOGICAL CONTEXT:
- Identify premises and conclusions
- Check for hidden assumptions
- Consider counterexamples
- Distinguish between necessary and sufficient conditions
- Avoid logical fallacies

USEFUL FRAMEWORKS:
1. List all given premises
2. Identify what needs to be proven
3. Consider if the conclusion necessarily follows
4. Look for alternative interpretations
5. Provide clear logical justification"""

        # Default mixed context
        else:
            return """MIXED PROBLEM CONTEXT:
- Identify the dominant domain (math or logic)
- Apply appropriate reasoning strategies
- Break complex problems into simpler parts
- Consider multiple solution approaches
- Provide clear justification for conclusions"""

# Alias for backward compatibility
ResponseParser = PromptBuilder