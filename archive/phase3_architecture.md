> [!WARNING]
> This document is outdated and has been superseded by the architecture described in `specs/project.md`. It is retained for historical context only.

# Phase 3: Agent Harness Architecture Design

## Overview

The Agent Harness provides the orchestration layer that coordinates between the LLM, tools, skills, and data systems. The architecture emphasizes clear separation of concerns and maintainable design.

## Architecture Components

### 1. Agent Harness (Core Orchestration)

**Responsibilities:**
- Session management and lifecycle
- State tracking and persistence
- Workflow orchestration
- Error handling and recovery

**Key Classes:**
```python
class AgentHarness:
    - create_session() -> Session
    - get_session(session_id) -> Session
    - process_request(session_id, request) -> Response
    - cleanup_session(session_id)

class Session:
    - session_id: str
    - state: SessionState
    - context: Context
    - history: List[Interaction]
    - created_at: datetime
    - last_activity: datetime
```

### 2. Support Systems (Data & Context)

**Responsibilities:**
- Data collection and persistence
- Context building and management
- Resource optimization
- Background processing

**Key Classes:**
```python
class ContextBuilder:
    - build_context(session: Session, request: str) -> Context
    - collect_relevant_claims(query: str) -> List[Claim]
    - select_skill_templates(query: str) -> List[SkillTemplate]
    - get_available_tools() -> List[Tool]

class DataManager:
    - persist_session(session: Session)
    - load_session(session_id: str) -> Session
    - store_claims(claims: List[Claim])
    - query_claims(filters: Dict) -> List[Claim]
```

### 3. Core LLM Prompt (Communication Layer)

**Responsibilities:**
- Prompt assembly and formatting
- Response parsing and extraction
- Tool call identification
- Error detection and handling

**Key Classes:**
```python
class PromptBuilder:
    - assemble_prompt(context: Context, request: str) -> str
    - format_skill_templates(skills: List[SkillTemplate]) -> str
    - format_tools(tools: List[Tool]) -> str
    - format_claims(claims: List[Claim]) -> str

class ResponseParser:
    - parse_response(response: str) -> ParsedResponse
    - extract_tool_calls(response: str) -> List[ToolCall]
    - extract_claims(response: str) -> List[Claim]
    - detect_errors(response: str) -> List[Error]
```

### 4. Skills (Simple Guidance Templates)

**Responsibilities:**
- Provide step-by-step guidance
- Offer thinking patterns
- Suggest tool usage
- Enable systematic problem solving

**Structure:**
```python
class SkillTemplate:
    - name: str
    - description: str
    - steps: List[str]  # 4-step process
    - suggested_tools: List[str]
    - example_usage: str

# Example Skill Templates
RESEARCH_SKILL = SkillTemplate(
    name="Research",
    description="Guide for gathering information and creating claims",
    steps=[
        "Search web for relevant information",
        "Read relevant files and documents", 
        "Create claims for key findings",
        "Support claims with collected evidence"
    ],
    suggested_tools=["WebSearch", "ReadFiles", "CreateClaim", "ClaimSupport"],
    example_usage="To research a topic: use WebSearch to find information, ReadFiles to examine documents, CreateClaim to capture findings, ClaimSupport to link evidence"
)
```

## Data Flow Architecture

```
User Request
    ↓
Agent Harness (Session Management)
    ↓
Support Systems (Context Building)
    ├── Collect relevant claims
    ├── Select skill templates  
    ├── Get available tools
    └── Build context
    ↓
Core LLM Prompt (Assembly)
    ├── Format context
    ├── Add skill templates
    ├── Include tool info
    └── Create final prompt
    ↓
LLM Processing
    ↓
Response Parser (Extraction)
    ├── Parse tool calls
    ├── Extract claims
    ├── Detect errors
    └── Return structured response
    ↓
Agent Harness (Orchestration)
    ├── Execute tool calls
    ├── Store claims
    ├── Update session state
    └── Return response to user
```

## Component Interfaces

### AgentHarness Interface
```python
class IAgentHarness:
    def create_session(self) -> str: ...
    def get_session(self, session_id: str) -> Optional[Session]: ...
    def process_request(self, session_id: str, request: str) -> Response: ...
    def cleanup_session(self, session_id: str) -> bool: ...
    def list_sessions(self) -> List[Session]: ...
```

### Support Systems Interface
```python
class ISupportSystem:
    def build_context(self, session: Session, request: str) -> Context: ...
    def persist_data(self, data: Any) -> bool: ...
    def retrieve_data(self, query: Any) -> Any: ...
    def cleanup_resources(self) -> None: ...
```

### Prompt System Interface
```python
class IPromptSystem:
    def build_prompt(self, context: Context, request: str) -> str: ...
    def parse_response(self, response: str) -> ParsedResponse: ...
    def validate_prompt(self, prompt: str) -> bool: ...
    def optimize_prompt(self, prompt: str) -> str: ...
```

## State Management

### Session State
```python
@dataclass
class SessionState:
    session_id: str
    status: SessionStatus  # ACTIVE, IDLE, ERROR
    current_task: Optional[str]
    step_in_process: int
    accumulated_context: Context
    error_count: int
    last_error: Optional[str]
```

### Context Structure
```python
@dataclass
class Context:
    relevant_claims: List[Claim]
    skill_templates: List[SkillTemplate]
    available_tools: List[Tool]
    session_history: List[Interaction]
    current_focus: Optional[str]
    context_window_size: int
```

## Error Handling Strategy

### Error Categories
1. **Session Errors**: Session corruption, timeout, resource exhaustion
2. **Context Errors**: Context building failure, data retrieval issues
3. **Prompt Errors**: Prompt assembly failure, formatting issues
4. **Parsing Errors**: Response parsing failure, malformed data
5. **Tool Errors**: Tool execution failure, invalid parameters

### Recovery Mechanisms
```python
class ErrorHandler:
    def handle_session_error(self, error: SessionError) -> RecoveryAction:
        # Attempt session recovery from persistence
        # Fallback to new session if recovery fails
        
    def handle_context_error(self, error: ContextError) -> RecoveryAction:
        # Fallback to basic context
        # Use default skill templates
        
    def handle_prompt_error(self, error: PromptError) -> RecoveryAction:
        # Use emergency prompt template
        # Simplify context if needed
        
    def handle_parsing_error(self, error: ParseError) -> RecoveryAction:
        # Use fallback parsing strategy
        # Request clarification if needed
```

## Performance Considerations

### Optimization Strategies
1. **Context Caching**: Cache built contexts for similar queries
2. **Session Pooling**: Reuse session resources
3. **Lazy Loading**: Load skill templates and tools on demand
4. **Batch Operations**: Process multiple claims efficiently
5. **Memory Management**: Clean up unused resources

### Monitoring Metrics
- Session creation/deletion rate
- Context building time
- Prompt assembly time
- Response parsing time
- Error rates by category
- Resource usage patterns

## Security Considerations

### Session Security
- Session isolation and data protection
- Authentication and authorization
- Session timeout and cleanup
- Audit logging

### Data Security
- Input validation and sanitization
- Output encoding and escaping
- Secure data persistence
- Privacy protection

## Testing Strategy

### Unit Tests
- Component isolation testing
- Interface contract testing
- Error condition testing
- Performance benchmarking

### Integration Tests
- End-to-end workflow testing
- Component interaction testing
- Error propagation testing
- Resource management testing

### Scenario Tests
- Real-world usage patterns
- Complex problem solving
- Multi-session workflows
- Stress testing

This architecture provides a clean separation of concerns while maintaining the flexibility needed for sophisticated AI agent functionality.