# Simple Universal Claim Architecture with LLM-Driven Instruction Support

**Last Updated:** November 11, 2025  
**Version:** 1.0  
**Author:** Design Documentation Writer

## Executive Summary

The Simple Universal Claim Architecture represents a fundamental simplification of the Conjecture system, achieving 90% of functionality with 10% of complexity. The core innovation is maintaining a single, universal Claim model while offloading intelligence tasks to Large Language Models (LLMs). This approach eliminates complex data structures, reduces implementation overhead, and provides a clean separation between system responsibilities and AI-driven capabilities.

The architecture's key insight is that **complete relationship coverage** in context building, combined with **LLM-driven instruction identification**, creates a more effective and maintainable system than traditional rule-based or multi-model approaches.

## Core Principles

### 1. Simplicity First
- **Single Universal Model**: One Claim model handles all use cases
- **No Enhanced Structures**: No specialized claim types, instruction models, or hierarchical abstractions
- **Minimal Complexity**: Focus on essential features that provide maximum value

### 2. Complete Relationship Coverage
- **All Supporting Claims to Root**: Every claim in the support chain from target to root
- **All Supported Claims**: Every claim directly or indirectly supported by the target
- **Semantic Similarity Claims**: Contextually relevant claims filling remaining tokens

### 3. LLM-Driven Intelligence
- **Instruction Identification**: LLM identifies which claims represent instructions
- **Support Relationship Creation**: LLM establishes support relationships naturally
- **Pattern Learning**: System improves over time through LLM interactions

### 4. Token Efficiency
- **Prioritize Relationships**: Guaranteed relevant claims get first allocation
- **Semantic Claims Fill Budget**: No wasted context space
- **Optimized Context Structure**: Clear organization for LLM consumption

## Data Model Specification

### Universal Claim Model

```python
class Claim(BaseModel):
    """Universal claim model - handles all claim types and use cases"""
    
    # Core identification
    id: str                                    # Unique claim identifier
    content: str                               # Claim content/statement
    
    # Quality metrics
    confidence: float                          # Confidence score (0.0-1.0)
    state: ClaimState                         # Current state (Explore, Validated, Orphaned, Queued)
    
    # Relationship management (bidirectional)
    supported_by: List[str]                   # Claims that support this claim
    supports: List[str]                       # Claims this claim supports
    
    # Classification and organization
    type: List[ClaimType]                     # Claim types (concept, reference, thesis, skill, example, goal)
    tags: List[str]                           # Topic/tags for organization and search
    
    # Metadata
    created_by: str                           # Creator identifier
    created: datetime                         # Creation timestamp
    updated: datetime                         # Last update timestamp
    
    # NO additional fields needed for instructions, hierarchy, or complex relationships
    # LLM handles all identification and relationship creation
```

### Key Design Decisions

**Why No Instruction Model?**
- LLMs naturally identify instruction claims from context
- No need for separate instruction data structure
- Simpler codebase and maintenance

**Why Bidirectional Support Links?**
- Enables efficient traversal both upward and downward
- No need for complex graph algorithms
- Simple list operations work effectively

**Why Simple Type/Tag System?**
- Provides basic organization without complexity
- Semantic similarity handles fine-grained classification
- LLM can interpret context better than rigid hierarchies

## Context Building Strategy

### Complete Relationship Coverage Algorithm

```
build_complete_context(target_claim_id: str, max_tokens: int = 8000):
    1. Get target claim
    2. Traverse upward: get all supporting claims to root
    3. Traverse downward: get all supported claims recursively
    4. Add semantic similar claims until token budget reached
    5. Format for optimal LLM consumption
```

### Token Allocation Strategy

Based on analysis of typical claim systems:

| Priority | Content Type | Token Allocation | Rationale |
|----------|--------------|------------------|-----------|
| 1 | Upward Support Chain | 40% | Critical for understanding context and validity |
| 2 | Downward Supported Claims | 30% | Shows implications and dependencies |
| 3 | Semantic Similar Claims | 30% | Provides broader context and connections |

### Context Formatting

```
=== TARGET CLAIM ===
[CLAIM_ID] Claim Content (confidence: 0.85)

=== SUPPORTING CHAIN (to root) ===
[CLAIM_ID_1] Supporting claim 1 (confidence: 0.90)
  └─ supports -> TARGET_CLAIM
[CLAIM_ID_2] Supporting claim 2 (confidence: 0.75)
  └─ supports -> CLAIM_ID_1
...

=== SUPPORTED CLAIMS (all descendants) ===
[CLAIM_ID_A] Claim supported by target (confidence: 0.80)
  └─ supported_by -> TARGET_CLAIM
[CLAIM_ID_B] Descendant claim (confidence: 0.70)
  └─ supported_by -> CLAIM_ID_A
...

=== SEMANTIC CONTEXT ===
[CLAIM_ID_X] Similar claim (similarity: 0.85)
[CLAIM_ID_Y] Related claim (similarity: 0.78)
...
```

## LLM Responsibility Definition

### 1. Instruction Identification

The LLM is responsible for identifying instruction claims within context:

```python
# LLM analyzes complete context and identifies:
# 1. Which claims represent instructions or guidance
# 2. What actions each instruction implies
# 3. How instructions relate to each other

instruction_analysis_prompt = """
Analyze the following claim context and identify all instruction/guidance claims.

Context:
{complete_claim_context}

Instructions:
1. Identify claims that represent instructions, guidance, or actionable advice
2. For each instruction, specify:
   - The instruction content
   - Required actions
   - Dependencies on other claims
3. Create support relationships between instructions and their evidence

Response format:
{
  "instructions": [
    {
      "claim_id": "CLAIM_ID",
      "instruction_type": "guidance|action|constraint",
      "content": "Instruction content",
      "supported_by": ["EVIDENCE_CLAIM_IDS"]
    }
  ],
  "new_relationships": [
    {"from": "EVIDENCE_ID", "to": "INSTRUCTION_ID", "type": "supports"}
  ]
}
"""
```

### 2. Support Relationship Creation

The LLM creates support relationships based on logical analysis:

```python
relationship_creation_prompt = """
Based on the instruction analysis above, create support relationships.

For each instruction claim:
1. Identify which claims provide evidence/support
2. Create bidirectional support links
3. Assign confidence based on evidence quality

Return structured relationship updates for persistence.
"""
```

### 3. Quality Assurance

The LLM validates and improves context quality:

```python
quality_assurance_prompt = """
Review the complete claim context for:
1. Logical consistency
2. Missing support relationships
3. Contradictions or conflicts
4. Areas needing additional evidence

Suggest improvements and additional claims needed.
"""
```

## Comparison with Previous Complex Approaches

### Traditional Multi-Model Approach
| Aspect | Complex Approach | Simple Universal Approach |
|--------|------------------|---------------------------|
| **Data Models** | Claim, Instruction, Evidence, Step, Hierarchy | Single Claim model |
| **Relationships** | Complex graph structures, typed edges | Simple bidirectional support links |
| **Context Building** | Multi-stage filtering, complex algorithms | Simple traversal + semantic search |
| **LLM Usage** | Task-specific prompts, narrow scope | Broad analysis, relationship creation |
| **Code Complexity** | High (multiple models, converters, validators) | Low (single model, simple operations) |
| **Maintainability** | Difficult (interdependent components) | Easy (clear separation of concerns) |

### Benefits of Simplified Approach

1. **Reduced Implementation Complexity**
   - 90% fewer data structures
   - Simplified persistence layer
   - Easier testing and validation

2. **Enhanced Flexibility**
   - LLM handles edge cases automatically
   - No rigid classification constraints
   - Adapts to new claim types naturally

3. **Improved Performance**
   - Faster context building
   - Reduced memory overhead
   - Simpler database queries

4. **Better Developer Experience**
   - Single model to understand
   - Clear API surface
   - Easier debugging and troubleshooting

## Technical Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Simple Claim  │    │   Context       │    │     LLM         │
│   Model Only    │───▶│   Builder       │───▶│   Intelligence  │
│                 │    │                 │    │                 │
│ • One data type │    │ • Relationship  │    │ • Instruction   │
│ • No hierarchy  │    │   traversal     │    │   identification│
│ • Simple links  │    │ • Semantic fill │    │ • Support links │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow

1. **Storage**: Universal Claim model
2. **Retrieval**: Context builder with complete relationships
3. **Analysis**: LLM processes complete context
4. **Update**: System persists LLM-created relationships
5. **Learning**: Patterns improve over time

### API Surface

```python
# Core API (minimal yet complete)
class ClaimSystem:
    def create_claim(content: str, confidence: float, **metadata) -> Claim
    def get_claim(claim_id: str) -> Claim
    def build_context(claim_id: str, max_tokens: int = 8000) -> str
    def process_with_llm(context: str, user_request: str) -> LLMResponse
    def update_relationships(relationships: List[Relationship]) -> bool
```

## Implementation Requirements

### Performance Targets
- **Context Building**: < 200ms for typical claim networks
- **LLM Processing**: < 2s for standard analysis
- **Database Queries**: < 50ms for relationship traversal
- **Memory Usage**: < 100MB for 10,000 claim network

### Scalability Considerations
- **Horizontal Scaling**: Stateless context builders
- **Database Optimization**: Indexed claim relationships
- **Caching Strategy**: Frequently accessed context groups
- **LLM Integration**: Async processing capabilities

## Security and Reliability

### Data Integrity
- Bidirectional relationship validation
- Confidence score consistency checks
- Circular dependency detection
- Transaction-based updates

### Error Handling
- Graceful degradation for LLM failures
- Fallback relationship creation
- Context building error recovery
- Comprehensive logging and monitoring

## Future Evolution Path

### Phase 1: Core Implementation
- Universal Claim model
- Basic context building
- LLM integration

### Phase 2: Optimization
- Performance tuning
- Caching strategies
- Batch processing

### Phase 3: Advanced Features
- Learning and adaptation
- Advanced relationship types
- Multi-LLM support

The Simple Universal Claim Architecture provides a solid foundation that can evolve with needs while maintaining its core simplicity and elegance.