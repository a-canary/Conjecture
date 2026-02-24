# Processing Subsystem

## Overview

The Processing subsystem provides the core execution and analysis capabilities for the Conjecture system. It handles tool execution, response parsing, example generation, and the integration between the agent's orchestration layer and the actual tools.

This subsystem implements a robust, secure, and extensible framework for executing tools, parsing LLM responses, and generating examples from successful executions.

## Key Components and Responsibilities

### Tool Registry and Management
- `tool_registry.py`: Central registry for all available tools with metadata
- `tool_manager.py`: Manages tool discovery, loading, and registration
- `tool_registry.py`: Defines ToolFunction and ToolCall data structures
- `tool_execution.py`: Core execution engine with safety features

### Tool Execution Engine
- `tool_executor.py`: Safe execution engine with resource limits
- `ExecutionLimits`: Configurable limits for execution time, memory, and recursion
- `SafeExecutor`: Wrapper for secure tool execution
- `execute_tool_from_registry`: Main entry point for executing tool calls
- `batch_execute_tool_calls`: Executes multiple tool calls in sequence

### Response Parsing
- `response_parser.py`: Parses LLM responses into structured tool calls
- Uses XML-style tags to identify tool calls in LLM output
- Extracts parameters from tool call tags
- Handles error cases and malformed responses

### Tool Creation System
- `tool_creator.py`: Dynamic tool creation system that allows LLM to discover needs and create tools
- `ToolValidator`: Validates tool code for security and functionality
- `DynamicToolCreator`: Main class for discovering tool needs and creating tools
- Uses LLM to analyze claims and determine if new tools are needed
- Searches for implementation methods and generates secure code
- Creates supporting skill claims and examples

### Support Systems
- `context.py`: Context building utilities
- `dirty_evaluator.py`: Evaluates claims for "dirty" status (needs re-evaluation)
- `exploration_engine.py`: Handles exploration of new ideas and claims
- `example_generator.py`: Generates examples from successful tool executions
- `response_parser.py`: Parses LLM responses for structured output
- `tool_registry.py`: Central registry for all tools
- `tool_manager.py`: Manages tool discovery and loading
- `tool_execution.py`: Core execution engine
- `tool_executor.py`: Safe execution wrapper

### Advanced Processing
- `async_eval.py`: Asynchronous evaluation utilities
- `bridge.py`: Bridge between agent and processing layers
- `chutes_adapter.py`: Adapter for Chutes API integration
- `context.py`: Context building utilities
- `example_generator.py`: Generates examples from successful executions
- `exploration_engine.py`: Handles exploration of new ideas
- `response_parser.py`: Parses LLM responses
- `tool_creator.py`: Dynamic tool creation
- `tool_execution.py`: Core execution engine
- `tool_executor.py`: Safe execution wrapper
- `tool_manager.py`: Manages tool discovery and loading
- `tool_registry.py`: Central registry for all tools

## Integration with the Rest of the System

The Processing subsystem integrates with other components through well-defined interfaces:

- **Agent Subsystem**: Receives tool calls from AgentHarness and returns execution results
- **LLM Subsystem**: Uses InstructionSupportProcessor to identify instruction claims and create support relationships
- **Local Subsystem**: Uses LocalServicesManager for local LLM inference and embeddings
- **Data Layer**: Uses DataManager to persist and retrieve claims and tool results

The processing subsystem acts as the execution engine for the agent, translating high-level requests into concrete actions.

## Example Usage

```python
from processing import ToolExecutor, ResponseParser
from agent import AgentHarness

# Initialize components
data_manager = DataManager()
agent = AgentHarness(data_manager)
executor = ToolExecutor()
parser = ResponseParser()

# Process LLM response
llm_response = "<tool_calls><invoke name='web_search'><parameter name='query'>Python weather APIs</parameter></invoke></tool_calls>"
parsed = parser.parse_response(llm_response)

# Execute tool calls
for tool_call in parsed.tool_calls:
    result = await executor.execute_tool_call(tool_call)
    print(f"Tool result: {result}")
```

## Configuration Requirements

- **Tool Directory**: The system requires a tools directory (default: "tools") containing executable tool modules
- **Execution Limits**: Configurable limits for:
  - Maximum execution time (default: 30 seconds)
  - Maximum memory usage (default: 512MB)
  - Maximum recursion depth (default: 10)
- **Security Settings**: Configuration for dangerous imports and functions to block
- **Tool Discovery**: Configuration for whether dynamic tool creation is enabled
- **Example Generation**: Configuration for whether examples are automatically generated from successful executions

All configuration is handled through the LocalConfig system in the config/ directory. The system defaults to secure settings with dynamic tool creation disabled by default for safety.