# Conjecture System Design

**Last Updated:** November 9, 2025

## Overview

Conjecture is a simple, practical framework for LLM agents. The design focuses on clarity and usefulness over theoretical completeness.

## Clean Architecture

### Three Core Components

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Tools     │    │    Skills   │    │   Claims    │
│             │    │             │    │             │
│ Get data    │───▶│ Think about │───▶│ What we     │
│ in/out      │    │ problems    │    │ know        │
└─────────────┘    └─────────────┘    └─────────────┘
```

**Tools = How to get data in/out**
- Simple functions for external interaction
- Clear inputs, predictable outputs
- Independent operation

**Skills = How to think about problems**
- Step-by-step thinking templates
- Guide the LLM through logical processes
- Reusable patterns for different tasks

**Claims = What we know and how confident we are**
- Structured statements of knowledge
- Linked to supporting evidence
- Can be updated as we learn

## Tool Design

### Simple, Focused Functions

Each tool does one thing well:

```python
# WebSearch
results = WebSearch(query="machine learning basics")
# Returns: structured list of relevant sources

# ReadFiles  
content = ReadFiles(path="src/*.py")
# Returns: parsed file contents

# WriteCodeFile
WriteCodeFile(path="solution.py", code="def hello(): pass")
# Returns: success/failure status

# CreateClaim
claim = CreateClaim(statement="Python is popular", confidence=85)
# Returns: structured claim with metadata

# ClaimSupport
SupportClaim(claim_id=123, evidence="Stack Overflow survey data")
# Returns: supporting evidence link
```

### Tool Characteristics

- **Simple interfaces**: One clear purpose per tool
- **Consistent patterns**: Similar input/output style across tools
- **Error handling**: Graceful failure modes
- **No side effects**: Tools don't modify each other's state

## Skill Design

### Simple Templates

Skills are just text templates that guide thinking:

**Research Skill Template**
```
You are researching {topic}. Follow these steps:
1. Search the web for relevant information
2. Read any relevant local files
3. Create claims for key findings
4. Support each claim with evidence
Focus on practical, actionable insights.
```

**WriteCode Skill Template**  
```
You need to create {solution_type}. Follow these steps:
1. Understand the requirements clearly
2. Design a simple, clean solution
3. Write the code file
4. Test that it works
5. Create claims about what the solution accomplishes
Focus on working code over complex abstractions.
```

**TestCode Skill Template**
```
You need to test {code_to_test}. Follow these steps:
1. Write comprehensive test cases
2. Run the tests
3. Fix anything that fails
4. Create claims about test results
Focus on reliability and edge cases.
```

**EndClaimEval Skill Template**
```
You need to evaluate these claims {claims_list}. Follow these steps:
1. Review supporting evidence for each claim
2. Check for contradictions between claims
3. Update confidence scores based on evidence quality
4. Note any gaps or uncertain areas
Focus on accuracy and intellectual honesty.
```

## Claim System Design

### Simple Knowledge Representation

```json
{
  "claim_id": "unique_identifier",
  "statement": "Clear, testable statement",
  "confidence": 85,
  "evidence": [
    {
      "source": "tool_name",
      "content": "supporting data",
      "relevance": "high"
    }
  ],
  "created": "2025-11-09T10:30:00Z",
  "updated": "2025-11-09T11:15:00Z"
}
```

### Claim Lifecycle

1. **Create**: Tool generates new claim from evidence
2. **Support**: Evidence gets linked to claims
3. **Evaluate**: Skills review and update confidence
4. **Revise**: Claims updated as new evidence arrives

## Workflow Examples

### Research Workflow

```
1. Start with Research Skill template
2. Use WebSearch tool → find sources
3. Use ReadFiles tool → access local data  
4. Use CreateClaim tool → capture findings
5. Use ClaimSupport tool → link evidence
6. Use EndClaimEval skill → validate knowledge
```

### Code Development Workflow

```
1. Start with WriteCode Skill template
2. Use ReadFiles tool → understand requirements
3. Use WriteCodeFile tool → create solution
4. Use CreateClaim tool → document capabilities
5. Use ClaimSupport tool → link to tests
6. Use EndClaimEval skill → validate solution
```

## Design Principles

### Keep It Simple

- **One purpose per component**: Tools do one thing, skills guide one process
- **Clear separation**: Data, thinking, and knowledge are distinct
- **Minimal dependencies**: Components work independently

### Focus on Practical Value

- **Work first**: Prioritize getting things done
- **Learn gradually**: Build understanding through evidence
- **Stay flexible**: Adapt to different problems easily

### Avoid Over-Engineering

- **No complex abstractions**: Keep concepts concrete
- **No elaborate processes**: Use simple, linear workflows
- **No enterprise patterns**: This is a practical tool, not a framework

## Implementation Notes

### Starting Point

1. Implement basic tools first (WebSearch, ReadFiles)
2. Create skill templates as simple text files
3. Build simple claim storage (JSON files work)
4. Test with basic research and coding tasks

### Evolution Path

- Add tools as needed for new capabilities
- Refine skill templates based on usage
- Improve claim evaluation as confidence grows
- Keep the core architecture simple and clean

This design prioritizes clarity and usefulness over theoretical completeness. The goal is a system that helps LLMs think clearly and work effectively without getting in the way.