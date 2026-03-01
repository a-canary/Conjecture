"""
Interaction Tools for Conjecture
Provides tools for user interaction and reasoning tracking
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import json

# Import the registry system
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from src.tools.registry import register_tool

# Simple storage for reasoning steps and user interactions
_reasoning_log: List[Dict[str, Any]] = []
_user_messages: List[Dict[str, Any]] = []
_user_questions: List[Dict[str, Any]] = []

@register_tool(name="Reason", is_core=True)
def Reason(thought_process: str) -> Dict[str, Any]:
    """
    Record a reasoning step in the thought process.

    Args:
        thought_process: Description of the reasoning step or thought process

    Returns:
        Dictionary with reasoning metadata and log entry
    """
    # Input validation
    if not thought_process or not thought_process.strip():
        return {
            'success': False,
            'error': 'Thought process cannot be empty',
            'reasoning_id': None
        }

    if len(thought_process) > 5000:  # 5KB limit
        return {
            'success': False,
            'error': 'Thought process too long (max 5000 characters)',
            'reasoning_id': None
        }

    # Create reasoning entry
    reasoning_id = f"reason_{len(_reasoning_log) + 1}"
    timestamp = datetime.now().isoformat()

    reasoning_entry = {
        'id': reasoning_id,
        'thought_process': thought_process.strip(),
        'timestamp': timestamp,
        'step_number': len(_reasoning_log) + 1
    }

    _reasoning_log.append(reasoning_entry)

    return {
        'success': True,
        'reasoning_id': reasoning_id,
        'step_number': reasoning_entry['step_number'],
        'timestamp': timestamp,
        'total_reasoning_steps': len(_reasoning_log)
    }

@register_tool(name="TellUser", is_core=True)
def TellUser(message: str, message_type: str = "info") -> Dict[str, Any]:
    """
    Send a message to the user.

    Args:
        message: The message content to send to the user
        message_type: Type of message - "info", "warning", "error", "success" (default: "info")

    Returns:
        Dictionary with message metadata
    """
    # Input validation
    if not message or not message.strip():
        return {
            'success': False,
            'error': 'Message cannot be empty'
        }

    if len(message) > 2000:  # 2KB limit
        return {
            'success': False,
            'error': 'Message too long (max 2000 characters)'
        }

    valid_message_types = ["info", "warning", "error", "success", "debug"]
    if message_type not in valid_message_types:
        message_type = "info"

    # Create message entry
    message_id = f"message_{len(_user_messages) + 1}"
    timestamp = datetime.now().isoformat()

    message_entry = {
        'id': message_id,
        'message': message.strip(),
        'message_type': message_type,
        'timestamp': timestamp,
        'direction': 'to_user'
    }

    _user_messages.append(message_entry)

    # Print to console (in a real system, this would send through the UI)
    print(f"[{message_type.upper()}] {message}")

    return {
        'success': True,
        'message_id': message_id,
        'message_type': message_type,
        'timestamp': timestamp,
        'total_messages': len(_user_messages)
    }

@register_tool(name="AskUser", is_core=True)
def AskUser(question: str, options: Optional[List[str]] = None, required: bool = False) -> Dict[str, Any]:
    """
    Ask the user a question and optionally provide response options.

    Args:
        question: The question to ask the user
        options: List of response options (optional)
        required: Whether user response is required (default: False)

    Returns:
        Dictionary with question metadata and placeholder for response
    """
    # Input validation
    if not question or not question.strip():
        return {
            'success': False,
            'error': 'Question cannot be empty'
        }

    if len(question) > 1000:  # 1KB limit
        return {
            'success': False,
            'error': 'Question too long (max 1000 characters)'
        }

    if options is None:
        options = []

    if not isinstance(options, list):
        return {
            'success': False,
            'error': 'Options must be a list'
        }

    # Limit options
    if len(options) > 10:
        options = options[:10]

    # Create question entry
    question_id = f"question_{len(_user_questions) + 1}"
    timestamp = datetime.now().isoformat()

    question_entry = {
        'id': question_id,
        'question': question.strip(),
        'options': [opt.strip() for opt in options if opt.strip()],
        'required': bool(required),
        'timestamp': timestamp,
        'response': None,  # Would be populated when user responds
        'response_timestamp': None
    }

    _user_questions.append(question_entry)

    # Build question display text
    question_text = f"[QUESTION] {question}"
    if options:
        for i, option in enumerate(options, 1):
            question_text += f"\n  {i}. {option}"
    if required:
        question_text += " (Required)"

    # Print to console
    print(question_text)
    print("[Waiting for user response...]")

    return {
        'success': True,
        'question_id': question_id,
        'timestamp': timestamp,
        'has_options': len(options) > 0,
        'required': required,
        'pending_response': True
    }

@register_tool(name="GetInteractionHistory", is_core=False)
def GetInteractionHistory(interaction_type: str = "all", limit: int = 50) -> Dict[str, Any]:
    """
    Get the history of user interactions and reasoning steps.

    Args:
        interaction_type: Type to retrieve - "reasoning", "messages", "questions", or "all"
        limit: Maximum number of entries to return (default: 50)

    Returns:
        Dictionary with interaction history
    """
    valid_types = ["reasoning", "messages", "questions", "all"]
    if interaction_type not in valid_types:
        interaction_type = "all"

    if not isinstance(limit, int) or limit < 1:
        limit = 50

    result = {
        'success': True,
        'interaction_type': interaction_type,
        'limit': limit
    }

    # Get reasoning steps
    if interaction_type in ["reasoning", "all"]:
        reasoning_history = _reasoning_log[-limit:] if limit else _reasoning_log
        result['reasoning_steps'] = reasoning_history
        result['total_reasoning_steps'] = len(_reasoning_log)

    # Get user messages
    if interaction_type in ["messages", "all"]:
        message_history = _user_messages[-limit:] if limit else _user_messages
        result['user_messages'] = message_history
        result['total_messages'] = len(_user_messages)

    # Get user questions
    if interaction_type in ["questions", "all"]:
        question_history = _user_questions[-limit:] if limit else _user_questions
        result['user_questions'] = question_history
        result['total_questions'] = len(_user_questions)

    return result

@register_tool(name="RecordClaim", is_core=False)
def RecordClaim(claim_text: str, confidence: float = 0.8, source: str = "reasoning") -> Dict[str, Any]:
    """
    Record a claim made during reasoning or interaction.

    Args:
        claim_text: The claim text
        confidence: Confidence level (0.0 to 1.0, default: 0.8)
        source: Source of the claim (default: "reasoning")

    Returns:
        Dictionary with claim recording metadata
    """
    # Input validation
    if not claim_text or not claim_text.strip():
        return {
            'success': False,
            'error': 'Claim text cannot be empty'
        }

    if len(claim_text) > 1000:  # 1KB limit
        return {
            'success': False,
            'error': 'Claim text too long (max 1000 characters)'
        }

    if not isinstance(confidence, (int, float)) or confidence < 0.0 or confidence > 1.0:
        confidence = 0.8

    # Create claim entry
    claim_id = f"interaction_claim_{len(_user_messages) + len(_user_questions) + 1}"
    timestamp = datetime.now().isoformat()

    claim_entry = {
        'id': claim_id,
        'claim_text': claim_text.strip(),
        'confidence': float(confidence),
        'source': source,
        'timestamp': timestamp,
        'type': 'recorded_claim'
    }

    # Add to messages for tracking
    _user_messages.append(claim_entry)

    return {
        'success': True,
        'claim_id': claim_id,
        'confidence': confidence,
        'source': source,
        'timestamp': timestamp,
        'total_recorded_claims': len([m for m in _user_messages if m.get('type') == 'recorded_claim'])
    }

# Helper functions not exposed as tools
def _clear_interaction_log():
    """Clear interaction logs (for testing)."""
    global _reasoning_log, _user_messages, _user_questions
    _reasoning_log = []
    _user_messages = []
    _user_questions = []

def examples() -> List[str]:
    """
    Return example usage claims for LLM context
    These examples help the LLM understand when and how to use these tools
    """
    return [
        "Reason('I need to search for Rust game development tutorials first, then examine the existing code structure') records a reasoning step about the development approach",
        "TellUser('Found 5 relevant Rust TUI tutorials. I will now analyze the existing codebase.', message_type='info') informs the user about progress",
        "AskUser('Do you want me to create a minesweeper game or a different type of game?', options=['minesweeper', 'tic-tac-toe', 'chess']) asks user for project choice",
        "RecordClaim('TUI games in Rust typically use the ratatui crate', confidence=0.8, source='web_research') records a factual claim from research",
        "Reason('The existing code structure shows we need a board module and game logic module') records reasoning about code organization",
        "TellUser('Project setup complete. Ready to implement game logic.', message_type='success') informs about completion",
        "AskUser('What difficulty level should be the default?', required=True) asks for required user input",
        "GetInteractionHistory('reasoning', limit=10) retrieves the last 10 reasoning steps for review"
    ]

if __name__ == "__main__":
    # Test the interaction tools
    print("Testing interaction tools...")

    # Clear logs for testing
    _clear_interaction_log()

    # Test reasoning
    print("\n1. Recording reasoning:")
    result1 = Reason("I need to analyze the user requirements first")
    print(f"   Reasoning recorded: {result1['success']} (step {result1['step_number']})")

    # Test user message
    print("\n2. Sending message to user:")
    result2 = TellUser("Analysis complete. Ready to proceed.", "info")
    print(f"   Message sent: {result2['success']} (ID: {result2['message_id']})")

    # Test asking user
    print("\n3. Asking user question:")
    result3 = AskUser("Which programming language do you prefer?", ["Rust", "Python", "JavaScript"], required=True)
    print(f"   Question asked: {result3['success']} (ID: {result3['question_id']})")

    # Test recording claim
    print("\n4. Recording claim:")
    result4 = RecordClaim("Rust provides better performance for systems programming", 0.9, "expertise")
    print(f"   Claim recorded: {result4['success']} (ID: {result4['claim_id']})")

    # Test getting history
    print("\n5. Getting interaction history:")
    history = GetInteractionHistory("all", limit=10)
    print(f"   Retrieved: {history['total_reasoning_steps']} reasoning steps")
    print(f"   Retrieved: {history['total_messages']} messages")
    print(f"   Retrieved: {history['total_questions']} questions")

    print("\nExamples for LLM context:")
    for example in examples():
        print(f"- {example}")