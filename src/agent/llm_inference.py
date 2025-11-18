"""
Pure LLM Inference Layer - The Bridge between Claims and Tools
Pure functions for LLM reasoning, context processing, and decision making.
"""
import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from ..core.models import Claim, ClaimState
from ..core.claim_operations import (
    should_prioritize, find_supporting_claims, find_supported_claims,
    calculate_support_strength, filter_claims_by_confidence,
    filter_claims_by_type, filter_claims_by_tags
)
from ..processing.tool_registry import ToolCall, ToolFunction, get_tool_function, list_tool_functions
from ..processing.tool_execution import execute_tool_from_registry, create_tool_call


logger = logging.getLogger(__name__)


@dataclass
class LLMContext:
    """Pure data structure for LLM context."""
    session_id: str
    user_request: str
    relevant_claims: List[Claim]
    available_tools: List[ToolFunction]
    conversation_history: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass
class LLMResponse:
    """Pure data structure for LLM response."""
    response_text: str
    tool_calls: List[ToolCall]
    claim_suggestions: List[Claim]
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class ProcessingPlan:
    """Pure data structure for processing plan."""
    planned_tool_calls: List[ToolCall]
    planned_claims: List[Claim]
    reasoning: str
    confidence: float


# Pure Functions for Context Building

def build_llm_context(session_id: str,
                     user_request: str,
                     all_claims: List[Claim],
                     tool_registry,
                     conversation_history: List[Dict[str, Any]] = None,
                     max_claims: int = 20,
                     metadata: Dict[str, Any] = None) -> LLMContext:
    """Pure function to build LLM context from claims and tools."""
    
    # Find relevant claims
    relevant_claims = find_relevant_claims(user_request, all_claims, max_claims)
    
    # Get available tools
    available_tools = list_tool_functions(tool_registry)
    
    return LLMContext(
        session_id=session_id,
        user_request=user_request,
        relevant_claims=relevant_claims,
        available_tools=available_tools,
        conversation_history=conversation_history or [],
        metadata=metadata or {}
    )


def find_relevant_claims(user_request: str, 
                        all_claims: List[Claim], 
                        max_claims: int = 20) -> List[Claim]:
    """Pure function to find relevant claims for user request."""
    if not all_claims:
        return []
    
    # Simple relevance scoring based on keywords in user request
    request_lower = user_request.lower()
    scored_claims = []
    
    for claim in all_claims:
        score = 0
        claim_content_lower = claim.content.lower()
        
        # Keyword matching
        request_words = request_lower.split()
        for word in request_words:
            if word in claim_content_lower:
                score += 1
        
        # Prioritize dirty claims
        if claim.is_dirty:
            score += 0.5
        
        # Prioritize high-confidence claims
        score += claim.confidence * 0.5
        
        # Consider claim state
        if claim.state == ClaimState.VALIDATED:
            score += 0.3
        elif claim.state == ClaimState.EXPLORE:
            score += 0.1
        
        scored_claims.append((claim, score))
    
    # Sort by score and return top claims
    scored_claims.sort(key=lambda x: x[1], reverse=True)
    return [claim for claim, score in scored_claims[:max_claims]]


def format_claims_for_llm(claims: List[Claim]) -> str:
    """Pure function to format claims for LLM consumption."""
    if not claims:
        return "No relevant claims available."
    
    formatted_sections = []
    
    # Group by state
    by_state = {}
    for claim in claims:
        state_key = claim.state.value
        if state_key not in by_state:
            by_state[state_key] = []
        by_state[state_key].append(claim)
    
    for state, state_claims in by_state.items():
        formatted_sections.append(f"## {state.upper()} CLAIMS:")
        for claim in state_claims:
            type_str = ",".join([t.value for t in claim.type])
            support_info = f" (supports: {len(claim.supports)}, supported_by: {len(claim.supported_by)})"
            dirty_info = " [DIRTY]" if claim.is_dirty else ""
            formatted_sections.append(
                f"- [{claim.id}] C:{claim.confidence:.2f} T:{type_str}{support_info}{dirty_info} {claim.content}"
            )
    
    return "\n\n".join(formatted_sections)


def format_tools_for_llm(tools: List[ToolFunction]) -> str:
    """Pure function to format tools for LLM consumption."""
    if not tools:
        return "No tools available."
    
    formatted_tools = ["## AVAILABLE TOOLS:"]
    for tool in tools:
        params = []
        for param_name, param_info in tool.parameters.items():
            param_str = param_name
            if not param_info['required']:
                param_str += f" (default: {param_info['default']})"
            param_str += f": {param_info['type_hint']}"
            params.append(param_str)
        
        formatted_tools.append(f"- {tool.name}({', '.join(params)})")
        if tool.description:
            formatted_tools.append(f"  Description: {tool.description}")
    
    return "\n".join(formatted_tools)


# Pure Functions for LLM Reasoning

def create_llm_prompt(context: LLMContext) -> str:
    """Pure function to create LLM prompt from context."""
    prompt_parts = [
        "You are an AI assistant with access to knowledge claims and tools.",
        "Your role is to reason about the user's request using available claims and decide which tools to use.",
        "",
        "## USER REQUEST:",
        context.user_request,
        ""
    ]
    
    # Add claims
    if context.relevant_claims:
        prompt_parts.extend([
            "## RELEVANT KNOWLEDGE CLAIMS:",
            format_claims_for_llm(context.relevant_claims),
            ""
        ])
    
    # Add tools
    if context.available_tools:
        prompt_parts.extend([
            format_tools_for_llm(context.available_tools),
            ""
        ])
    
    # Add conversation history if available
    if context.conversation_history:
        prompt_parts.extend([
            "## CONVERSATION HISTORY:",
            *[f"User: {item.get('user', '')}" for item in context.conversation_history[-3:]],
            *[f"Assistant: {item.get('assistant', '')}" for item in context.conversation_history[-3:]],
            ""
        ])
    
    # Add instructions
    prompt_parts.extend([
        "## INSTRUCTIONS:",
        "1. Analyze the user's request in the context of available knowledge claims.",
        "2. Consider which tools might help answer the request or gather new information.",
        "3. Provide a clear, helpful response.",
        "4. If appropriate, suggest new knowledge claims based on reasoning.",
        "",
        "## RESPONSE FORMAT:",
        "Provide your response in plain text. If you need to use tools, format tool calls like:",
        "<tool_calls>",
        "  <invoke name=\"tool_name\">",
        "    <parameter name=\"param1\">value1</parameter>",
        "    <parameter name=\"param2\">value2</parameter>",
        "  </invoke>",
        "</tool_calls>",
        "",
        "## YOUR RESPONSE:"
    ])
    
    return "\n".join(prompt_parts)


def parse_llm_response(response_text: str) -> LLMResponse:
    """Pure function to parse LLM response into structured format."""
    # Extract tool calls if present
    tool_calls = []
    claim_suggestions = []
    
    # Simple tool call parsing
    if "<tool_calls>" in response_text and "</tool_calls>" in response_text:
        tool_call_text = response_text.split("<tool_calls>")[1].split("</tool_calls>")[0]
        tool_calls = parse_tool_calls_from_text(tool_call_text)
    
    # Extract main response (remove tool calls)
    main_response = response_text
    if "<tool_calls>" in response_text:
        main_response = response_text.split("<tool_calls>")[0].strip()
    elif "tool_calls:" in response_text.lower():
        main_response = response_text.split("tool_calls:")[0].strip()
    
    return LLMResponse(
        response_text=main_response,
        tool_calls=tool_calls,
        claim_suggestions=claim_suggestions,
        confidence=0.8,  # Default confidence
        metadata={"raw_response": response_text}
    )


def parse_tool_calls_from_text(text: str) -> List[ToolCall]:
    """Pure function to parse tool calls from text."""
    tool_calls = []
    
    try:
        # Simple XML-style parsing
        import re
        
        # Find all invoke blocks
        invoke_pattern = r'<invoke name="([^"]+)">(.*?)</invoke>'
        invoke_matches = re.findall(invoke_pattern, text, re.DOTALL)
        
        for tool_name, params_text in invoke_matches:
            # Parse parameters
            param_pattern = r'<parameter name="([^"]+)">(.*?)</parameter>'
            param_matches = re.findall(param_pattern, params_text, re.DOTALL)
            
            parameters = {}
            for param_name, param_value in param_matches:
                parameters[param_name] = param_value.strip()
            
            tool_calls.append(create_tool_call(tool_name, parameters))
    
    except Exception as e:
        logger.error(f"Error parsing tool calls: {e}")
    
    return tool_calls


def simulate_llm_interaction(context: LLMContext) -> LLMResponse:
    """Pure function to simulate LLM interaction (for testing)."""
    user_request_lower = context.user_request.lower()
    
    # Simple simulation logic
    if "research" in user_request_lower and any("web" in tool.name.lower() for tool in context.available_tools):
        response_text = "I'll help you research this topic. Let me search for relevant information."
        tool_calls = []
        
        # Find web search tool
        for tool in context.available_tools:
            if "web" in tool.name.lower() or "search" in tool.name.lower():
                tool_calls.append(create_tool_call(
                    tool.name,
                    {"query": context.user_request[:100]}  # Limit query length
                ))
                break
        return LLMResponse(
            response_text=response_text,
            tool_calls=tool_calls,
            claim_suggestions=[],
            confidence=0.9,
            metadata={"simulated": True}
        )
    
    elif "write code" in user_request_lower or "code" in user_request_lower:
        response_text = "I'll help you write code. Based on your requirements, I can create a solution."
        
        # Find code generation tool
        tool_calls = []
        for tool in context.available_tools:
            if "write" in tool.name.lower() or "code" in tool.name.lower():
                tool_calls.append(create_tool_call(
                    tool.name,
                    {"code": "# Generated code based on requirements\ndef main():\n    pass"}
                ))
                break
        
        return LLMResponse(
            response_text=response_text,
            tool_calls=tool_calls,
            claim_suggestions=[],
            confidence=0.8,
            metadata={"simulated": True}
        )
    
    else:
        response_text = f"I understand your request: {context.user_request}. Let me help you with that using the available tools and knowledge."
        return LLMResponse(
            response_text=response_text,
            tool_calls=[],
            claim_suggestions=[],
            confidence=0.7,
            metadata={"simulated": True}
        )


# Pure Functions for Processing Planning

def create_processing_plan(response: LLMResponse, context: LLMContext) -> ProcessingPlan:
    """Pure function to create processing plan from LLM response."""
    return ProcessingPlan(
        planned_tool_calls=response.tool_calls,
        planned_claims=response.claim_suggestions,
        reasoning=response.response_text,
        confidence=response.confidence
    )


def validate_processing_plan(plan: ProcessingPlan, tool_registry) -> Tuple[bool, List[str]]:
    """Pure function to validate processing plan."""
    errors = []
    
    # Validate tool calls
    for tool_call in plan.planned_tool_calls:
        tool_func = get_tool_function(tool_registry, tool_call.name)
        if not tool_func:
            errors.append(f"Tool '{tool_call.name}' not found in registry")
        else:
            # Validate parameters (basic check)
            required_params = {name for name, info in tool_func.parameters.items() if info['required']}
            provided_params = set(tool_call.parameters.keys())
            missing = required_params - provided_params
            if missing:
                errors.append(f"Tool '{tool_call.name}' missing parameters: {missing}")
    
    return len(errors) == 0, errors


# Pure Functions for Result Processing

def process_tool_results(tool_results: List[Any], 
                        original_claims: List[Claim],
                        plan: ProcessingPlan) -> List[Claim]:
    """Pure function to process tool results and update claims."""
    updated_claims = original_claims.copy()
    
    # For now, return original claims - in real implementation would analyze results
    # and create/update claims based on tool outputs
    
    return updated_claims


def create_claims_from_results(tool_results: List[Any], 
                              context: LLMContext) -> List[Claim]:
    """Pure function to create new claims from tool results."""
    new_claims = []
    
    # Simple claim creation from tool results
    for i, result in enumerate(tool_results):
        if isinstance(result, dict) or (hasattr(result, 'success') and result.success):
            # Create a claim about the tool execution
            claim_content = f"Tool execution completed successfully: {plan.planned_tool_calls[i].name if i < len(plan.planned_tool_calls) else 'unknown tool'}"
            
            new_claim = Claim(
                id=f"tool_result_{uuid.uuid4().hex[:8]}",
                content=claim_content,
                confidence=0.8,
                state=ClaimState.VALIDATED,
                type=["example"],
                tags=["tool_result"]
            )
            new_claims.append(new_claim)
    
    return new_claims


# Pure Functions for Coordination

def coordinate_three_part_flow(session_id: str,
                             user_request: str,
                             all_claims: List[Claim], 
                             tool_registry,
                             conversation_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Pure function to coordinate the complete 3-part flow."""
    
    # 1. Claims Layer: Build context from existing claims
    context = build_llm_context(
        session_id=session_id,
        user_request=user_request,
        all_claims=all_claims,
        tool_registry=tool_registry,
        conversation_history=conversation_history
    )
    
    # 2. LLM Inference Layer: Reason about request and create plan
    llm_response = simulate_llm_interaction(context)  # In real implementation, call actual LLM
    processing_plan = create_processing_plan(llm_response, context)
    
    # Validate plan
    is_valid, errors = validate_processing_plan(processing_plan, tool_registry)
    if not is_valid:
        return {
            "success": False,
            "error": "Processing plan validation failed",
            "errors": errors,
            "context": context,
            "plan": processing_plan
        }
    
    # 3. Tools Layer: Execute tool calls
    tool_results = []
    for tool_call in processing_plan.planned_tool_calls:
        result = execute_tool_from_registry(tool_call, tool_registry)
        tool_results.append(result)
    
    # Process results and update claims
    updated_claims = process_tool_results(tool_results, all_claims, processing_plan)
    new_claims = create_claims_from_results(tool_results, context)
    
    return {
        "success": True,
        "context": context,
        "llm_response": llm_response,
        "processing_plan": processing_plan,
        "tool_results": tool_results,
        "updated_claims": updated_claims,
        "new_claims": new_claims
    }


# Import the reference to `plan` used in the functions above
from .llm_inference import ProcessingPlan