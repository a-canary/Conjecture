# Architecture Gap Analysis: Current vs Intended Design

## 🎯 Intended Design Vision

Based on your clarification, the intended architecture is:

### **Tools** 
- Secure Python functions with internal security and resource limitations
- Dynamically created by LLM when new capabilities are needed
- Stored in a tools directory as `.py` files
- Have standardized interfaces and error handling

### **Skill-Claims**
- Claims describing intended use of tools OR logical procedures
- NOT Python functions, but procedural instructions for LLM
- Multi-step processes and conditions for LLM to follow
- Example: "To get weather by zipcode, first validate zipcode format, then call weather_tool(zipcode), then format response"

### **Sample-Claims**
- Record exact LLM call XML and tool response
- Include both successful and failed examples
- Used to teach LLM exact syntax and patterns
- May include LLM-generated summaries of tool responses

### **Workflow Example**
1. LLM needs weather forecast by zipcode
2. LLM websearches for weather query methods
3. LLM writes `weather.py` tool file
4. LLM creates skill-claim describing weather tool usage
5. LLM creates sample-claim with exact XML call format
6. System calls weather tool and records sample-claim of result
7. Future weather-related claims retrieve weather skills/samples for context

## 🔍 Current Implementation Analysis

### What I Built vs What You Intended

| Component | Current Implementation | Intended Design | Gap |
|-----------|----------------------|-----------------|-----|
| **Tools** | Built-in functions in SkillManager | Dynamic Python files in tools directory | ❌ Major gap |
| **Skill-Claims** | Function signatures with parameters | Procedural instructions for LLM | ❌ Major gap |
| **Sample-Claims** | Auto-generated from execution results | Exact LLM XML + tool response recording | ❌ Major gap |
| **Tool Creation** | Manual registration | LLM-driven discovery and creation | ❌ Missing |
| **Context Collection** | Basic similarity search | Skill/sample retrieval for LLM context | ❌ Missing |

### Specific Issues

1. **Tool System**: I implemented built-in functions, but you want dynamic Python file creation
2. **Skill Claims**: I implemented function-like claims, but you want procedural instructions
3. **Sample Claims**: I auto-generate from execution, but you want exact LLM XML recording
4. **LLM Integration**: Missing the LLM-driven tool discovery and creation workflow
5. **Context Building**: Missing the skill/sample retrieval for LLM context

## 🛠️ Refactoring Plan

### Phase 2R: Refactor to Match Intended Design

#### 1. **Tool System Refactor**
```python
# Current: Built-in functions
self.builtin_skills['search_claims'] = search_claims_function

# Intended: Dynamic Python file loading
tools/
├── weather.py
├── search_claims.py
└── calculator.py
```

#### 2. **Skill-Claim Refactor**
```python
# Current: Function signature
SkillClaim(
    function_name="weather_by_zipcode",
    parameters=[zipcode: str]
)

# Intended: Procedural instructions
SkillClaim(
    content="To get weather by zipcode: 1) Validate zipcode format using regex, 2) Call weather_tool(zipcode), 3) Format response as 'Weather for {zipcode}: {temp}°F, {conditions}'",
    tool_name="weather_by_zipcode",
    procedure_steps=[...]
)
```

#### 3. **Sample-Claim Refactor**
```python
# Current: Auto-generated from execution
ExampleClaim(
    input_parameters={"zipcode": "90210"},
    output_result="Weather for 90210: 72°F, Sunny"
)

# Intended: Exact LLM XML + response
SampleClaim(
    llm_call_xml='<invoke name="weather_tool"><parameter name="zipcode">90210</parameter></invoke>',
    tool_response='{"temp": 72, "conditions": "Sunny"}',
    llm_summary="Weather for 90210: 72°F, Sunny"
)
```

#### 4. **Dynamic Tool Creation**
```python
# New: LLM-driven tool creation
class ToolCreator:
    async def discover_tool_need(self, claim: Claim) -> str
    async def websearch_tool_methods(self, need: str) -> List[str]
    async def create_tool_file(self, method: str, code: str) -> str
    async def validate_tool_security(self, tool_path: str) -> bool
```

#### 5. **Context Collector**
```python
# New: Skill/sample retrieval for LLM context
class ContextCollector:
    async def collect_relevant_skills(self, claim: Claim) -> List[SkillClaim]
    async def collect_relevant_samples(self, claim: Claim) -> List[SampleClaim]
    async def build_llm_context(self, claim: Claim) -> str
```

## 🎯 Implementation Strategy

### Step 1: Refactor Core Models
- Update SkillClaim to be procedural, not functional
- Update SampleClaim to store LLM XML and responses
- Create Tool model for dynamic Python functions

### Step 2: Create Tool Management System
- Dynamic tool loading from files
- Tool validation and security checking
- Tool registration and discovery

### Step 3: Implement LLM-Driven Creation
- Tool need detection
- Websearch integration for tool discovery
- Dynamic tool file creation

### Step 4: Build Context Collector
- Skill/sample similarity matching
- Context building for LLM
- Retrieval optimization

### Step 5: Integration Testing
- Weather example end-to-end
- Tool creation workflow
- Context collection validation

## 📋 Updated Architecture

```
LLM Processing Flow:
1. Claim Evaluation → Tool Need Detection
2. Websearch → Tool Method Discovery  
3. Tool Creation → Python File Generation
4. Skill-Claim Creation → Procedural Instructions
5. Sample-Claim Creation → XML Format Examples
6. Tool Execution → Secure Function Calls
7. Result Recording → Sample Claims with LLM Summaries
8. Context Building → Skill/Sample Retrieval for Future Claims

Tool System:
tools/
├── weather.py (dynamically created)
├── search_claims.py (built-in)
└── calculator.py (dynamically created)

Claim Types:
- Claim: Base knowledge assertions
- Skill-Claim: "How to use X tool for Y purpose"
- Sample-Claim: "Exact XML to call X tool and what it returns"
```

This refactoring will align the implementation with your intended design while maintaining the security and performance foundations already built.