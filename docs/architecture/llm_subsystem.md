# LLM Subsystem

## Overview

The LLM subsystem provides the intelligence layer for the Conjecture system, handling instruction identification, support relationship creation, and LLM integration. It implements a specialized processor for identifying instructional claims and creating support relationships between claims.

Unlike traditional LLM interfaces, this subsystem focuses on a specific, well-defined task: analyzing claims to identify instructional content and creating support relationships between claims.

## Key Components and Responsibilities

### InstructionSupportProcessor
- Core component for LLM-driven instruction identification
- Identifies instruction claims from a set of claims using LLM analysis
- Creates support relationships between instruction claims and target claims
- Processes complete context to identify instructional content
- Formats prompts for instruction identification
- Parses LLM responses and extracts relationships
- Validates and persists new relationships

### LLMProvider and LLMBridge
- `LLMProvider`: Abstract base class for LLM providers
- `LLMBridge`: Simple bridge between Conjecture API and LLM providers
- Provides clean interface with minimal complexity
- Supports primary and fallback providers for resilience
- Standardizes request and response formats

### Data Structures
- `LLMRequest`: Standardized LLM request structure with prompt, context claims, and configuration
- `LLMResponse`: Standardized LLM response structure with success status, content, generated claims, and metadata
- `InstructionIdentification`: Result of instruction identification from LLM
- `ProcessingResult`: Result of LLM instruction support processing

## Integration with the Rest of the System

The LLM subsystem integrates with other components through well-defined interfaces:

- **Agent Subsystem**: Provides instruction identification and support relationship creation to the agent
- **Processing Subsystem**: Uses the InstructionSupportProcessor to analyze claims and create relationships
- **Data Layer**: Uses UnifiedClaim and SupportRelationshipManager to persist claims and relationships
- **Local Subsystem**: Uses LocalServicesManager for local LLM inference

The LLM subsystem acts as a specialized intelligence layer that enhances the agent's ability to understand and create relationships between claims.

## Example Usage

```python
from llm import InstructionSupportProcessor
from core.unified_claim import UnifiedClaim

# Initialize processor with claims
claims = [UnifiedClaim(...), UnifiedClaim(...)]
processor = InstructionSupportProcessor(claims)

# Process a target claim with instruction support
result = processor.process_with_instruction_support(
    target_claim_id="claim-123",
    user_request="How should I implement this feature?"
)

# Access results
print(f"New instruction claims: {len(result.new_instruction_claims)}")
print(f"Created relationships: {len(result.created_relationships)}")
print(f"Success: {result.success}")
```

## Configuration Requirements

- **LLM Provider**: Requires an LLM provider (local or external) to be configured in the system
- **Confidence Threshold**: Minimum confidence threshold for identifying instruction claims (default: 0.6)
- **Maximum Instruction Length**: Maximum length for instruction claims (default: 500 characters)
- **Instruction Tags**: List of tags that identify instruction claims (default: ["instruction", "guidance", "method", "approach", "technique"])
- **Context Token Limit**: Maximum tokens for context building (default: 8000)

All configuration is handled through the LocalConfig system in the config/ directory. The system defaults to secure settings with appropriate thresholds for confidence and length to ensure quality results.