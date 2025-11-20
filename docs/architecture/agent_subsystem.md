# Agent Subsystem

## Overview

The Agent subsystem provides the core orchestration layer that coordinates between the LLM, tools, skills, and data systems with clear separation of concerns. It implements a pure 3-part architecture (Claims → LLM → Tools → Claims) with functional purity at its core.

The subsystem is designed around the concept of "claims" - structured pieces of knowledge that can be validated, supported, or refuted. The agent coordinates between these claims, the LLM for reasoning, and tools for external actions.

## Key Components and Responsibilities

### AgentHarness
- Core orchestration and session management
- Manages user sessions with state tracking and timeout handling
- Coordinates the interaction between context building, prompt assembly, and response parsing
- Provides session lifecycle management (creation, cleanup, expiration)

### AgentCoordination
- Pure functions implementing the 3-part architecture flow
- `process_user_request`: Main entry point that coordinates claims → LLM → tools → claims
- `coordinate_three_part_flow`: Orchestrates the complete workflow
- `create_agent_session`: Pure function to create session data structures
- `reconcile_claim_differences`: Pure function to handle claim updates

### SupportSystems
- `ContextBuilder`: Builds contextual information from claims and user requests
- `Context`: Data structure representing contextual information
- `SupportDataManager`: Manages persistent data storage for claims

### PromptSystem
- `PromptBuilder`: Assembles LLM prompts from context and user requests
- `ResponseParser`: Parses LLM responses into structured tool calls and claims

### LLMInference
- `LLMContext`: Data structure for LLM input context
- `LLMResponse`: Data structure for LLM output
- `coordinate_three_part_flow`: Core function that implements the 3-part architecture
- `build_llm_context`: Creates context from claims and tools
- `format_claims_for_llm`: Formats claims for LLM consumption
- `format_tools_for_llm`: Formats available tools for LLM consumption
- `create_llm_prompt`: Constructs the complete LLM prompt
- `parse_llm_response`: Parses LLM responses into structured format

## Integration with the Rest of the System

The Agent subsystem integrates with other components through well-defined interfaces:

- **Processing Subsystem**: Uses tool execution and tool registry from processing/ to execute tool calls
- **Data Layer**: Uses DataManager from data/ to persist and retrieve claims
- **LLM Subsystem**: Uses InstructionSupportProcessor from llm/ for instruction identification and support relationship creation
- **Local Subsystem**: Uses LocalServicesManager from local/ for local LLM inference and embeddings

The agent acts as the central coordinator, receiving user requests and orchestrating the workflow across these subsystems.

## Example Usage

```python
from agent import AgentHarness
from data import DataManager

# Initialize components
data_manager = DataManager()
agent = AgentHarness(data_manager)
await agent.initialize()

# Create session and process request
session_id = await agent.create_session()
response = await agent.process_request(session_id, "Research Python weather APIs")

# Process results
print(f"Response: {response['response']}")
print(f"Parsed response: {response['parsed_response']}")
```

## Configuration Requirements

- **Tool Directory**: The agent system requires a tools directory (default: "tools") containing executable tool modules
- **Claim Persistence**: Requires a data layer for claim storage (implemented by DataManager)
- **LLM Configuration**: Requires an LLM provider (local or external) to be configured in the system
- **Session Limits**: Configurable parameters for maximum sessions (default: 100) and session timeout (default: 30 minutes)
- **Cleanup Interval**: Configurable cleanup interval for expired sessions (default: 5 minutes)

The agent system is designed to be highly configurable, with defaults provided for common use cases. All configuration is handled through the LocalConfig system in the config/ directory.