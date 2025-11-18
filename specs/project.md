# Conjecture Project Specification

**Last Updated:** November 14, 2025

## Overview

Conjecture is a simple, practical framework for LLM agents that helps developers systematically expand knowledge while preventing and correcting AI hallucinations. The design focuses on clarity and usefulness over theoretical completeness.

## Clean Architecture

### Three-Layer Design

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Data Layer      │    │  Process Layer  │    │Presentation Layer│
│                 │    │                 │    │                 │
│ Claims + Tools  │───▶│ Context & LLM   │───▶│ CLI | TUI | GUI │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**Data Layer = What we know and how we interact**
- **Claims**: Structured statements of knowledge with confidence scores
- **Skills as Claims**: Knowledge about methodologies and problem-solving approaches
- **Tools**: Simple functions for external data interaction

- **Process Layer = How we think and orchestrate**
- Context building relevant to tasks
- LLM orchestration of tools and skill claims
- Response parsing to extract structured insights
- Async Claim Evaluation Service (continuous evaluation engine)
- Concurrency Controls (file locking, provider throttling, retry logic)

**Presentation Layer = How users interact**
- Command Line Interface (CLI)
- Terminal User Interface (TUI)
- Graphical User Interface (GUI)

## Problem Statement

In today's rapidly evolving software development landscape, developers face a fragmented ecosystem of tools and knowledge sources that impede productivity and innovation. While large language models (LLMs) have revolutionized how we approach coding tasks, they suffer from hallucinations that can misinformation, wasted effort, and flawed implementations. There remains a critical gap in systematically expanding knowledge while preventing and correcting AI hallucinations, verifying assumptions, conducting deep research, and building confidence from multiple diverse sources.

Developers need a system that can:
1. Systematically expand knowledge and capabilities through structured exploration
2. Prevent and correct LLM hallucinations through evidence-based validation
3. Rigorously verify assumptions from multiple independent sources
4. Conduct deep research that builds comprehensive understanding of complex topics
5. Aggregate and validate information from diverse sources to build confident claims
6. Maintain clear provenance of all claims, tracing evidence from source to conclusion
7. Decompose complex problems into smaller, manageable sub-problems with proper framing
8. Retain and manage significant context for each part of decomposed problems

The absence of such a system leads to superficial understanding, unverified assumptions, and knowledge that fails to accumulate across sessions. Developers waste time debugging and correcting hallucinated code, cannot effectively build on previous research, struggle with complex problem decomposition that loses context, and lack tools to systematically increase confidence in their understanding while mitigating AI-generated misinformation.

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
- **Evidence generation**: Tools produce structured output that can be used as evidence for claims

## Skill Design

### Skills as Methodological Claims

Skills are not separate components but rather specialized claims about methodologies, strategies, and approaches. They are retrieved and applied by the LLM based on contextual relevance.

**Research Skill Claim**
```json
{
  "claim_id": "skill_research_methodology",
  "statement": "To research effectively: 1) Search web for information, 2) Read relevant local files, 3) Create claims for key findings, 4) Support claims with evidence",
  "confidence": 90,
  "type": "skill",
  "template": "You are researching {topic}. Follow these steps:\n1. Search the web for relevant information\n2. Read any relevant local files\n3. Create claims for key findings\n4. Support each claim with evidence\nFocus on practical, actionable insights.",
  "applicable_contexts": ["research", "investigation", "knowledge_gathering"]
}
```

**WriteCode Skill Claim**
```json
{
  "claim_id": "skill_code_development",
  "statement": "To create effective code: 1) Understand requirements, 2) Design simple solution, 3) Write implementation, 4) Test functionality",
  "confidence": 85,
  "type": "skill",
  "template": "You need to create {solution_type}. Follow these steps:\n1. Understand the requirements clearly\n2. Design a simple, clean solution\n3. Write the code file\n4. Test that it works\n5. Create claims about what the solution accomplishes\nFocus on working code over complex abstractions.",
  "applicable_contexts": ["development", "programming", "implementation"]
}
```

**TestCode Skill Claim**
```json
**TestCode Skill Template**:
```python
{
  "claim_id": "skill_testing_methodology",
  "statement": "To test code effectively: 1) Write comprehensive tests, 2) Execute tests, 3) Fix failures, 4) Document results",
  "confidence": 90,
  "type": "skill",
  "template": "You need to test {code_to_test}. Follow these steps:\n1. Write comprehensive test cases\n2. Run the tests\n3. Fix anything that fails\n4. Create claims about test results\nFocus on reliability and edge cases.",
  "applicable_contexts": ["testing", "validation", "quality_assurance"]
}
```



### Skill Claim Characteristics

- **Template-based**: Each skill claim includes a template for applying the methodology
- **Context-aware**: Skills specify applicable contexts where they should be used
- **Evidence-supported**: Like all claims, skills have confidence scores and can be supported by evidence
- **Retrievable**: Skills are retrieved by relevance to the current task context
- **Evolvable**: Skills can be refined or replaced as better methodologies are discovered

## Claim System Design

### Simple Knowledge Representation

Claims are the core knowledge representation in Conjecture, capturing what we know, how confident we are, and what evidence supports our knowledge:

```json
{
  "claim_id": "unique_identifier",
  "statement": "Clear, testable statement",
  "confidence": 85,
  "dirty": false,  // Indicates if claim needs re-evaluation
  "evidence": [
    {
      "source": "tool_name",
      "content": "supporting data",
      "relevance": "high"
    }
  ],
  "scope": "session",  // Default scope for new claims
  "elevation_history": [
    {
      "from_scope": "session",
      "to_scope": "user",
      "elevated_by": "llm",
      "reasoning": "This claim represents user preference that will be useful across sessions",
      "timestamp": "2025-11-14T10:45:00Z"
    }
  ],
  "created": "2025-11-14T10:30:00Z",
  "updated": "2025-11-14T11:15:00Z"
}
```

### Specialized Claim Types

All knowledge in Conjecture is represented as claims, with specialized types for different purposes:

**Knowledge Claims**: Statements about facts, concepts, or findings
```json
{
  "claim_id": "python_popularity",
  "statement": "Python is one of the most popular programming languages in 2025",
  "confidence": 90,
  "type": "knowledge",
  "evidence": [
    {
      "source": "Stack Overflow Developer Survey",
      "content": "Python ranked as the #1 most wanted technology",
      "relevance": "high"
    }
  ]
}
```

**Skill Claims**: Methodological knowledge about how to approach problems (see Skill Design section)

**Tool Claims**: Knowledge about tool capabilities and appropriate usage contexts
```json
{
  "claim_id": "websearch_scope",
  "statement": "WebSearch tool is effective for finding current technical documentation and tutorials",
  "confidence": 85,
  "type": "tool_knowledge",
  "tool": "WebSearch",
  "best_practices": ["Use specific technical terms", "Include year for time-sensitive queries"]
}
```

### Claim Lifecycle

1. **Create**: Tools or LLM generate new claims from evidence (defaults to session scope, dirty=true)
2. **Support**: Evidence gets linked to claims through explicit relationships
3. **Evaluate**: Dirty claims are selected for evaluation until LLM sets confidence and marks as clean
4. **Revise**: Claims updated as new evidence arrives or contradictions are found (sets dirty=true)
5. **Connect**: Claims are linked to related claims (support, conflict, dependency)
6. **Scope Inheritance**: Claims inherit accessibility based on scope hierarchy
7. **Scope Elevation**: LLM can automatically elevate valuable claims to User/Project scope

### Claim Relationships

Claims are not isolated but form a network of knowledge:
- **Support**: When evidence or other claims strengthen a claim's confidence
- **Conflict**: When claims contradict each other, requiring resolution
- **Dependency**: When one claim relies on another being true
- **Context**: When claims are relevant only in specific contexts or domains

### Claim Scopes

All claims have a scope that determines their accessibility and sharing boundaries:

```python
class ClaimScope(Enum):
    GLOBAL = "global"      # Universal knowledge accessible to all
    TEAM = "team"          # Shared knowledge within a team
    PROJECT = "project"    # Knowledge specific to a project
    USER = "user"          # Personal knowledge and preferences
    SESSION = "session"    # Temporary claims during a single session
```

**Scope Hierarchy and Inheritance:**
- Sessions inherit from User scope
- User scope inherits from all associated Projects
- Project scope inherits from Team scope
- Team scope inherits from Global scope
- This creates a natural inheritance hierarchy for claim access

**Scope Characteristics:**

- **Global Claims**: Universal facts, fundamental methodologies, core tool knowledge
  ```json
  {
    "claim_id": "python_basics",
    "statement": "Python is an interpreted, high-level programming language",
    "scope": "global",
    "confidence": 95
  }
  ```

- **Team Claims**: Team-specific workflows, shared coding standards, internal tools
  ```json
  {
    "claim_id": "team_api_standards",
    "statement": "Our team uses RESTful API design with OpenAPI 3.0 specification",
    "scope": "team",
    "team_id": "dev_team_alpha"
  }
  ```

- **Project Claims**: Project-specific architectures, libraries, requirements
  ```json
  {
    "claim_id": "project_stack",
    "statement": "Project uses React frontend with Node.js backend and PostgreSQL database",
    "scope": "project",
    "project_id": "web_portal_v2"
  }
  ```

- **User Claims**: Personal preferences, learned shortcuts, individual workflows
  ```json
  {
    "claim_id": "user_vscode_preference",
    "statement": "User prefers Vim keybindings in VSCode",
    "scope": "user",
    "user_id": "user_123"
  }
  ```

- **Session Claims**: Temporary working hypotheses, debugging hypotheses, intermediate results
  ```json
  {
    "claim_id": "session_debug_hypothesis",
    "statement": "The authentication failure is caused by expired token format",
    "scope": "session",
    "session_id": "sess_abc789"
  }
  ```

**Claim Scope Management:**
- All new claims are created with session scope by default
- Claims can be manually promoted to broader scopes (user, project, team, global)
- Scope inheritance determines accessibility boundaries but doesn't affect context generation
- Context building considers all accessible claims regardless of their scope

**Performance Benefits:**
- Claim reuse across sessions at the User, Project, and Team levels
- Faster evaluation start with relevant pre-existing claims
- Reduced redundant evaluation of common knowledge
- Efficient knowledge sharing within teams and projects

### Claim Evaluation Process

```python
# Pseudocode for claim evaluation with confidence-driven continuation
def evaluate_claim(claim_id):
    claim_context = generate_claim_context(claim_id)
    claim = get_claim(claim_id)
    
    while claim.is_dirty:
        # Step 1: Send context to LLM with exploration encouragement
        llm_response = send_to_llm_with_prompt(
            context=claim_context,
            prompt=f"""
            Evaluate the following claim: "{claim.statement}"
            
            You are encouraged to explore this claim further by:
            1. Making tool calls to gather more evidence
            2. Creating new claims that support or refine this claim
            
            Continue exploring until you have sufficient confidence.
            If you believe you have enough information, set a confidence level
            and mark the evaluation as complete.
            """
        )
        
        # Step 2: Parse response for tool calls, new claims, and confidence updates
        tool_calls, new_claims, confidence_update = parse_llm_response(llm_response)
        
        # Step 3: Store new claims created by LLM
        for new_claim_data in new_claims:
            created_claim = create_claim(new_claim_data)
            # Check for scope elevation opportunity
            elevation_decision = llm_evaluate_scope_elevation(created_claim.id)
            if elevation_decision.should_elevate:
                elevate_claim_scope(
                    claim_id=created_claim.id,
                    target_scope=elevation_decision.target_scope,  # "user" or "project"
                    reason=elevation_decision.reasoning
                )
        
        # Step 4: Execute tool calls if any, otherwise update confidence
        if tool_calls:
            # Execute tools and continue loop
            tool_responses = []
            for tool_call in tool_calls:
                tool_result = execute_tool(tool_call.name, tool_call.parameters)
                tool_responses.append(tool_result)
            
            # Update context with tool responses and continue
            claim_context = update_context_with_tool_responses(claim_context, tool_responses)
        else:
            # LLM decided no more exploration needed
            update_claim(
                claim_id=claim_id, 
                confidence=confidence_update.value,
                dirty=False
            )
            claim.is_dirty = False
        
        # Optionally add safety limit to prevent infinite loops
        if claim_context.iteration_count > MAX_ITERATIONS:
            # Force completion with current confidence
            update_claim(claim_id=claim_id, dirty=False)
            claim.is_dirty = False
    
    return claim
```

### LLM-Initiated Scope Elevation

The LLM can automatically elevate claims during evaluation when it determines they have broader value:

#### Elevation Triggers

The LLM evaluates these factors when considering scope elevation:

1. **Cross-Session Value**: Claims that would be useful in future sessions
2. **Project Relevance**: Claims about project-specific knowledge
3. **User Preference**: Claims about user preferences, patterns, or habits
4. **Persistent Patterns**: Claims about recurring issues or solutions
5. **Configuration Knowledge**: Claims about configuration or setup decisions

#### Elevation Rules

**User Scope Elevation (Automatic)**:
```python
def should_elevate_to_user(claim):
    """LLM prompt template for user scope evaluation"""
    return f"""
    Analyze this claim for User scope elevation:
    Claim: "{claim.statement}"
    Evidence: {format_evidence(claim.evidence)}
    
    Elevate to User scope if:
    - This represents a user preference or habit
    - This is a personal workflow pattern
    - This is user-specific configuration
    - This would help the user in future sessions
    - This is NOT tied to any specific project
    
    Respond with JSON: {{"elevate": true/false, "reasoning": "why or why not"}}
    """
```

**Project Scope Elevation (Automatic)**:
```python
def should_elevate_to_project(claim):
    """LLM prompt template for project scope evaluation"""
    return f"""
    Analyze this claim for Project scope elevation:
    Claim: "{claim.statement}"
    Project: {get_current_project_details()}
    Evidence: {format_evidence(claim.evidence)}
    
    Elevate to Project scope if:
    - This relates to the current project's architecture
    - This is a project-specific tooling setup
    - This is a project convention or standard
    - This would help other project contributors
    - This is NOT generally applicable to all projects
    
    Respond with JSON: {{"elevate": true/false, "reasoning": "why or why not"}}
    """
```

#### Elevation Examples

**User Scope Elevation Example**:
```json
{
  "claim_id": "user_vscode_theme",
  "statement": "User prefers dark theme in VSCode with specific color contrast settings",
  "original_scope": "session",
  "new_scope": "user",
  "elevated_by": "llm",
  "llm_reasoning": "This represents a user preference that will persist across all sessions"
}
```

**Project Scope Elevation Example**:
```json
{
  "claim_id": "project_api_version",
  "statement": "Project uses REST API v2.1 with JWT authentication",
  "original_scope": "session",
  "new_scope": "project",
  "elevated_by": "llm",
  "llm_reasoning": "This is project-specific architectural knowledge needed by all contributors"
}
```

#### Manual Elevation (Required for Team/Global)

Team and Global scope elevations require manual approval:

```python
async def request_team_elevation(user_id: str, claim_id: str, justification: str):
    # Create elevation request for team administrator approval
    elevation_request = await create_elevation_request(
        claim_id=claim_id,
        target_scope="team",
        requested_by=user_id,
        justification=justification
    )
    
    # Notify team administrators for approval
    await notify_team_admins(elevation_request)
    
    return elevation_request.id
```

#### Elevation Audit Trail

All scope elevations are tracked with full audit history:

```json
{
  "elevation_history": [
    {
      "from_scope": "session",
      "to_scope": "user",
      "elevated_by": "llm",
      "reasoning": "Represents persistent user preference across sessions",
      "timestamp": "2025-11-14T10:45:00Z"
    },
    {
      "from_scope": "user",
      "to_scope": "project",
      "elevated_by": "user:john_doe",
      "reasoning": "This habit is team-wide standard",
      "timestamp": "2025-11-14T15:30:00Z"
    }
  ]
}
```

## Workflow Examples

### Research Workflow

The research workflow demonstrates how the LLM orchestrates tools and applies skill claims to systematically gather knowledge:

```
1. LLM retrieves Research Skill Claim based on task context
2. LLM follows Research Skill template steps:
   a. Uses WebSearch tool → find sources
   b. Uses ReadFiles tool → access local data  
   c. Uses CreateClaim tool → capture findings
   d. Uses ClaimSupport tool → link evidence
3. Each tool response is processed in the continuous evaluation loop
4. Claims are stored with relationships and confidence scores
```

### Code Development Workflow

The code development workflow shows how skill claims guide the implementation process:

```
1. LLM retrieves WriteCode Skill Claim based on development task
2. LLM follows WriteCode template steps:
   a. Uses ReadFiles tool → understand requirements
   b. Designs simple, clean solution internally
   c. Uses WriteCodeFile tool → create solution
   d. Uses CreateClaim tool → document capabilities
3. LLM retrieves TestCode Skill Claim for validation
4. LLM applies TestCode template in the continuous evaluation loop:
   a. Writes comprehensive test cases
   b. Runs tests using appropriate tools
   c. Fixes anything that fails
   d. Creates claims about test results
5. Claims are linked to show the development relationship
```

### Knowledge Validation Workflow

When dealing with existing knowledge claims, the validation workflow ensures accuracy:

```
1. LLM identifies claims relevant to the task
2. LLM reviews existing claims in the continuous evaluation loop:
   a. Reviews supporting evidence for each claim
   b. Checks for contradictions between claims
   c. Updates confidence scores based on evidence quality
   d. Notes any gaps or uncertain areas
3. LLM creates new claims about validation results
4. If conflicts or uncertainties are found:
   a. LLM retrieves appropriate skill claims to address issues
   b. Applies additional tools to gather more evidence
   c. Updates existing claims with new information
```

### Key Workflow Patterns

Across all workflows, consistent patterns emerge:

1. **Skill Retrieval**: LLM retrieves relevant skill claims based on task context
2. **Template Application**: LLM applies skill claim templates to guide the process
3. **Tool Orchestration**: LLM orchestrates tools in sequence suggested by the skill
4. **Confidence-Driven Evaluation**: LLM autonomously decides when exploration is sufficient and sets confidence
5. **Tool Response Loop**: When LLM calls tools, responses trigger continued evaluation
6. **Claim Generation**: Tools and LLM generate new claims as they explore
7. **Evidence Linking**: Claims are linked to their supporting evidence

## Design Principles

### Keep It Simple

- **One purpose per component**: Tools do one thing, skills guide one process
- **Clear separation**: Data, process, and presentation are distinct layers
- **Minimal dependencies**: Components work independently
- **Unified knowledge model**: All knowledge, including methodologies, is represented as claims
- **Stateless evaluations**: Each claim evaluation uses claims as context, not conversation history

### Focus on Practical Value

- **Work first**: Prioritize getting things done
- **Learn gradually**: Build understanding through evidence
- **Stay flexible**: Adapt to different problems easily
- **Evidence-based**: All knowledge is traceable to sources and can be validated

### Avoid Over-Engineering

- **No complex abstractions**: Keep concepts concrete
- **No elaborate processes**: Use simple, linear workflows
- **No enterprise patterns**: This is a practical tool, not a framework
- **No forced workflows**: LLM orchestrates based on context, not rigid procedures

## What to Exclude

**Complexity we're removing:**
- Sophisticated rubrics and quality gates
- Detailed sub-claim generation strategies
- Complex iteration cycles and feedback loops
- Enterprise-style framework documentation
- Over-engineered architectural patterns

**Focus on practical value:**
- Simple, clear guidance for LLM
- Basic tools for data manipulation
- Context that helps LLM think step-by-step
- Maintainable and flexible system

## Relevant Principles

### Simplicity and Elegance

Conjecture follows the principle of "simple architecture" with a unified API that eliminates over-engineering. The system prioritizes:

- Clean separation of concerns through a three-part architecture: Data (Claims + Tools), Process (Context building, LLM, response Parsing), and Presentation (CLI, TUI, GUI)
- One primary API class `Conjecture` that provides access to all functionality
- Minimal abstractions that don't distract from the core purpose
- Claims can be created before being proven true, serving as hypotheses to be explored and validated
- Claims are queried by relevance, allowing exploration of knowledge through connections and context

### Flexible Integration

The system must accommodate diverse environments and preferences:

- Multiple LLM providers (OpenAI, Anthropic, local models, etc.)
- Both local and cloud-based processing options
- Extensible tool ecosystem for specialized capabilities
- Multiple interface implementations (CLI, TUI, GUI)

### Verifiable Transparency

Users should be able to understand and verify the system's reasoning:

- Clear exposure of the reasoning process
- Visible evidence supporting each conclusion
- Explicit tracking of assumptions and their sources
- Capable of being audited and refined by human oversight

## Solution Strategy

Conjecture implements a Data-Process-Presentation architecture that elegantly addresses these challenges:

At its core, the system recognizes that skills are not separate components but rather specialized claims themselves—knowledge about methodologies, analytical frameworks, and strategies for using tools effectively. This insight allows for a more unified architecture where all knowledge, including knowledge about how to think, is represented as claims within the data layer. The LLM retrieves and applies skill claims based on task context, orchestrating tools in sequences suggested by methodological claims.

### Implementation Notes

#### Starting Point

1. Implement basic tools first (WebSearch, ReadFiles, WriteCodeFile, CreateClaim, FileLock)
2. Create skill claims with templates for basic methodologies
3. Build simple claim storage (JSON files work initially)
4. Implement basic Async Claim Evaluation Service with simple priority queue
5. Test with basic research and coding tasks to validate evaluation workflow

#### Evolution Path

- Add tools as needed for new capabilities
- Create and refine skill claims based on usage patterns
- Improve claim evaluation logic and priority algorithms
- Enhance concurrency controls and error handling
- Add performance monitoring and optimization
- Keep the evaluation service simple and reliable

### 1. Unified API Design

At the core of the solution is a single `Conjecture` class that provides access to all functionality:

```python
from contextflow import Conjecture

# One class for all functionality
cf = Conjecture()
result = cf.explore("machine learning")
claim = cf.add_claim("content", 0.85, "concept")
stats = cf.get_statistics()
```

This unified API eliminates the complexity of juggling multiple interfaces while providing clear, predictable functionality that works consistently across all use cases.

### 2. Three-Part Architecture

The system is organized into three distinct but interconnected components:

#### Data Layer: Claims + Tools
**Claims**: Unified knowledge representation with scope-based organization
- **Knowledge Claims**: Statements about facts, concepts, or findings
- **Skill Claims**: Methodological knowledge about how to approach problems
**Tool Claims**: Knowledge about tool capabilities and appropriate usage
- **Evaluation Claims**: Status and metadata about claim evaluation processes
- All claims include confidence scores, evidence links, relationships, scope, and dirty flag
- Claims form the persistent context that replaces conversation history
- **Dirty Flag**: Indicates claims needing re-evaluation, drives the evaluation queue
- **Scopes**: Global, Team, Project, User, and Session - enabling knowledge reuse while maintaining boundaries

**Tools**: Data collection and production mechanisms
- WebSearch: Find information on the web
- ReadFiles: Access local files and data
- WriteCodeFile: Create and modify code files
- CreateClaim: Make structured statements about what we know
- ClaimSupport: Link evidence to claims
- FileLock: Manage concurrent access to shared files
- Tools produce structured output that serves as evidence for claims
- Tool calls generate evaluation events for UI updates

#### Process Layer: Context Building & LLM Interaction
**Context Building**: Assembling relevant knowledge for claim evaluation
- Retrieval of pertinent existing claims (not conversation history)
- Integration of relevant skill claims (methodologies and strategies)
- Composition of evaluation-specific context for each claim
- Note: Claim scope doesn't affect context selection - all accessible claims are considered

**Async Claim Evaluation Service**: Confidence-driven claim processing engine
- Maintains priority queue of dirty claims to evaluate
- Evaluates top N dirtiest claims sequentially using scope-aware context building
- Implements confidence-driven evaluation: LLM autonomously decides when to stop exploring
- Continues evaluation loop only while LLM makes tool calls or creates new claims
- Marks claims as clean (not dirty) when LLM sets confidence and completes evaluation
- Manages concurrency controls (file locking, provider throttling)
- Handles retry logic with exponential backoff for failed evaluations
- Emits scope-decorated events for UI updates during evaluation process
- Operates independently but prioritizes claims based on scope hierarchy and dirtiness
- Enables claim sharing and reuse across sessions at appropriate scope levels

**Response Parsing**: Extracting structured insights from LLM outputs
- Identification of new claims and confidence levels
- Extraction of evidence relationships from tool outputs
- Priority assignment for new claims based on multiple factors
- Creation of evaluation status claims for tracking reasoning quality

#### Presentation Layer: User Interfaces
- CLI: Quick, lightweight access for power users
- TUI: Interactive terminal interface with rich claim visualizations
- GUI: Feature-rich graphical interface for complex workflows
- API/MCP: Integration with other tools and systems

### 3. Async Claim Evaluation Service

The system uses an Async Claim Evaluation Service as the operational core that processes claims continuously:

Instead of using conversation history, the system operates through:
- Independent evaluation of claims using existing claims as context
- Continuous processing of top N highest-priority claims
- Event-driven updates to maintain UI awareness without affecting evaluations
- Sophisticated concurrency controls for file access and provider throttling
- Retry logic with exponential backoff for failed evaluations
- Priority management that considers user importance, dependencies, and confidence gaps

The evaluation process:
1. Each evaluation fetches relevant existing claims as context
2. LLM receives context with encouragement to explore further
3. LLM decides whether to:
   - Make tool calls to gather evidence (continue loop)
   - Create new supporting claims (continue loop)
   - Set confidence and complete evaluation (end loop)
4. If tools are called, responses are added to context and loop continues
5. If no tools are called, confidence is updated, claim marked as clean (not dirty)
6. New claims are evaluated for scope elevation opportunities
7. Events are emitted for UI updates
8. Priority is reassessed for any newly created claims
9. The service continues with the next dirtiest claim in the queue

### 4. Multiple Interface Implementations

All interfaces follow the same pattern but cater to different user preferences:

- CLI: Quick, lightweight access for power users
- TUI: Interactive terminal interface with rich visualizations
- GUI: Feature-rich graphical interface for complex workflows
- API/MCP: Integration with other tools and systems

### 5. Progressive Plan Disclosure

For complex tasks, the system generates and discloses its execution plan:

- Step-by-step visualization of proposed actions
- Interactive approval/rejection of individual steps
- Progress tracking during execution
- Plan modification capabilities based on feedback

### 6. Implementation Phases

The solution strategy follows a phased implementation approach:

#### Phase 1: Core Foundation (Current: ~70% Complete)
- Claim model system with validation
- Basic tool implementations
- Vector similarity and search
- Unified API design

#### Phase 2: Async Claim Evaluation Service (Next Priority)
- Priority queue management for claim evaluations
- Context building from existing claims (not conversation history)
- Basic concurrency controls (file locking, simple rate limiting)
- Event system for UI updates
- Retry logic with exponential backoff

#### Phase 3: Enhanced Evaluation Capabilities (Following Priority)
- Advanced dependency tracking between claims
- Sophisticated provider throttling and backoff strategies
- Parallel evaluation queue management
- Evaluation priority algorithms
- Performance monitoring and optimization

#### Phase 4: Interface Development (Future)
- Terminal User Interface (TUI) with real-time event display
- Web User Interface (WebUI) with claim visualization
- Advanced CLI features for evaluation monitoring
- Model Context Protocol (MCP) Interface for external integrations

## Session Management and User Interaction

### Session Lifecycle

The system manages user sessions that interface with the Async Claim Evaluation Service:

1. **Session Creation**: When a user initiates a new session
2. **Prompt Processing**: User prompts are decomposed into initial claims
3. **Claim Queuing**: Initial claims are added to the evaluation queue with high priority
4. **Evaluation Monitoring**: Session tracks the progress of relevant claim evaluations
5. **Event Streaming**: Session receives evaluation events for UI updates
6. **Session Completion**: When all relevant claims reach sufficient confidence or user ends session

### Session-Claim Interaction

```python
class SessionManager:
    async def create_session(self, user_id: str, team_id: Optional[str], project_id: Optional[str]) -> Session
    async def process_prompt(self, session_id: str, prompt: str) -> List[str]
    async def track_claim_progress(self, session_id: str, claim_ids: List[str]) -> None
    async def stream_evaluation_events(self, session_id: str) -> AsyncIterator[EvaluationEvent]
    async def promote_claim_scope(self, session_id: str, claim_id: str, target_scope: ClaimScope) -> bool
    async def get_session_summary(self, session_id: str) -> SessionSummary
```

### Priority Management for User Claims

User-initiated claims receive special priority handling:

1. **Initial Boost**: Claims from user prompts get a temporary priority boost (start at session scope)
2. **Scope-Aware Spreading**: When a user claim spawns new claims, they inherit session scope by default
3. **Dependency Spreading**: Dependencies consider scope hierarchy when propagating priority
4. **Decay Function**: User claim priority gradually decreases as other claims enter the system
5. **Re-boost on Interaction**: If user interactively requests updates, related claims are re-boosted
6. **Elevation Priority**: Claims auto-elevated to User/Project scope by LLM receive priority boost in those contexts
7. **Manual Promotion**: Users can manually promote session claims to broader scopes

### Event-Driven UI Updates

Since evaluations don't use conversation history, UI updates come through an event streaming system:

```python
# Evaluation events that update UI
EvaluationEvent = Union[
    ClaimStartedEvent(claim_id: str, priority: int),
    ConfidenceUpdatedEvent(claim_id: str, old_confidence: float, new_confidence: float),
    ToolCalledEvent(claim_id: str, tool_name: str, params: Dict),
    ToolResponseEvent(claim_id: str, tool_name: str, response: Any),
    ClaimCompletedEvent(claim_id: str, final_confidence: float),
    ErrorEvent(claim_id: str, error: Exception, retry_count: int)
]
```

### Session Evaluation Controls

Users can influence the evaluation process through session controls:

- **Pause Evaluation**: Temporarily stop evaluating claims for this session
- **Adjust Claim Priority**: Manually increase or decrease priority of specific claims
- **Add Constraints**: Restrict which tools or providers can be used for session claims
- **Set Confidence Thresholds**: Define when claims are considered "complete"
- **Request Summary**: Get a condensed report of session progress

## Claim Scope Management

### Scope-Based Access Control

The system implements hierarchical access control based on claim scopes:

```python
class ScopeManager:
    async def get_accessible_claims(self, session_id: str, target_scope: ClaimScope) -> List[Claim]
    async def can_access_claim(self, session_id: str, claim: Claim) -> bool
    async def promote_claim_scope(self, claim_id: str, current_scope: ClaimScope, target_scope: ClaimScope, user_id: str) -> bool
    async def get_scope_statistics(self, scope: ClaimScope, scope_id: str) -> ScopeStatistics
```

### Scope Inheritance Rules

 1. **Session Scope**: 
   - Can be accessed by the current session only
   - Claims exist only for duration of session (temporary)
   - Can be auto-elevated to User/Project scope by LLM
   - Manual elevation required for Team/Global scope
   - Affects claim accessibility but NOT context generation

 2. **User Scope**:
   - Claims are accessible to the specific user across all their sessions
   - Cannot be accessed by other users
   - Often auto-elevated from Session scope by LLM for persistent patterns
   - Manual elevation required for Team/Global scope
   - Affects claim accessibility but NOT context generation

 3. **Project Scope**:
   - Claims are accessible to all contributors of the project
   - Contributors can access project claims across their sessions
   - Often auto-elevated from Session scope by LLM for project-specific knowledge
   - Manual elevation required for Team/Global scope by project leads
   - Affects claim accessibility but NOT context generation

 4. **Team Scope**:
   - Claims are accessible to all team members
   - MANUAL elevation only (not automatic) - team administrators must approve
   - Contains team-wide standards and methodologies
   - Team administrators can promote claims to Global scope
   - Affects claim accessibility but NOT context generation

 5. **Global Scope**:
   - Claims are accessible to all users and sessions
   - MANUAL elevation only (not automatic) - requires system administrator approval
   - Contains universal knowledge and core methodologies
   - Affects claim accessibility but NOT context generation

**Scope Elevation Matrix**:
| From/To | Session | User | Project | Team | Global |
|---------|---------|------|---------|------|--------|
| Session | - | ✓ (LLM) | ✓ (LLM) | ✓ (Manual) | ✓ (Manual) |
| User | ✓ | - | ✓ (Manual) | ✓ (Manual) | ✓ (Manual) |
| Project | ✓ | ✓ | - | ✓ (Manual) | ✓ (Manual) |
| Team | ✓ | ✓ | ✓ | - | ✓ (Manual) |
| Global | X | X | X | X | - |

**Important Note**: Claim scope ONLY determines which claims can be accessed by which users/sessions. It does NOT affect which claims are included in context generation - all accessible claims are considered for context regardless of their scope.

### Scope Promotion Workflow

```python
# Example of promoting a valuable session claim to project scope
async def promote_to_project(session_id: str, claim_id: str, justification: str):
    # 1. Check if user has permission to promote to project scope
    user_can_promote = await scope_manager.can_promote_to_project(session_id)
    
    if user_can_promote:
        # 2. Verify claim quality and community value
        quality_score = await evaluate_claim_quality(claim_id)
        if quality_score > PROMOTION_THRESHOLD:
            # 3. Create new claim at project scope
            new_claim = await copy_claim_with_new_scope(claim_id, "project", session_id)
            
            # 4. Record promotion for tracking
            await record_scope_promotion(session_id, claim_id, new_claim.id, justification)
            
            # 5. Emit event for UI updates
            await emit_scope_promotion_event(new_claim.id, "session", "project")
            
            return new_claim.id
```

### Scope-Aware Caching Strategy

The system implements intelligent caching based on claim scopes:

```python
class ScopeAwareCache:
    def __init__(self):
        self.global_cache = LRUCache(maxsize=10000)    # Largest, longest-lived
        self.team_caches = defaultdict(LRUCache)       # Per-team caches
        self.project_caches = defaultdict(LRUCache)    # Per-project caches
        self.user_caches = defaultdict(LRUCache)       # Per-user caches
        
    async def get_claim(self, claim_id: str, session_scope: ClaimScope) -> Optional[Claim]:
        # Check caches in order from most specific to most general
        for scope_type in self.get_cache_hierarchy(session_scope):
            cache = self.get_cache_for_scope(scope_type)
            claim = cache.get(claim_id)
            if claim and self.can_access_claim(claim, session_scope):
                return claim
        return None
```

### Cross-Scope Claim Resolution

When claims in different scopes conflict, the system implements resolution rules:

1. **Specificity Rule**: More specific scope (session) takes precedence over general (global)
2. **Freshness Rule**: More recently updated claims take precedence
3. **Confidence Rule**: Higher confidence claims take precedence
4. **Source Rule**: Claims with stronger evidence chains take precedence

```python
async def resolve_claim_conflicts(claims: List[Claim]) -> Claim:
    # Sort by resolution criteria
    sorted_claims = sorted(claims, key=lambda c: (
        get_scope_specificity(c.scope),
        c.updated.timestamp(),
        c.confidence,
        count_supporting_evidence(c)
    ), reverse=True)
    
    return sorted_claims[0]  # Return highest-ranked claim
```

## Concurrency Controls

### File Locking System

The File Locking System ensures safe concurrent access to shared resources during claim evaluations:

```python
class FileLockManager:
    async def acquire_lock(self, resource_path: str, timeout: int = 30) -> Optional[Lock]:
        """Acquire a lock on a file resource with timeout"""
        
    async def release_lock(self, lock: Lock) -> bool:
        """Release a previously acquired lock"""
        
    async def with_lock(self, resource_path: str, operation: Callable, timeout: int = 30) -> Any:
        """Execute a function while holding a lock"""
        
    async def is_locked(self, resource_path: str) -> bool:
        """Check if a resource is currently locked"""
        
    async def get_lock_info(self, resource_path: str) -> Optional[LockInfo]:
        """Get information about a current lock holder"""
```

### Provider Rate Limiting

The Provider Rate Limiting system manages interactions with LLM providers to prevent throttling:

```python
class ProviderRateLimiter:
    def __init__(self, provider_configs: Dict[str, ProviderConfig]):
        """Initialize with provider-specific rate limits"""
        
    async def check_rate_limit(self, provider: str, operation: str) -> bool:
        """Check if a provider can accept a request"""
        
    async def get_backoff_time(self, provider: str, attempt: int, error: Optional[Exception] = None) -> int:
        """Calculate backoff time with jitter"""
        
    async def wait_for_slot(self, provider: str, operation: str) -> None:
        """Wait until the provider is available for requests"""
        
    async def record_request(self, provider: str, operation: str, duration: float) -> None:
        """Record a completed request for adaptive throttling"""
        
    async def get_provider_status(self, provider: str) -> ProviderStatus:
        """Get current status and availability of a provider"""
```

### Claim Evaluation Retry Logic

The Claim Retry Manager handles retries for failed evaluations with exponential backoff:

```python
class ClaimRetryManager:
    async def schedule_retry(self, claim_id: str, error: Exception, original_evaluation_id: str) -> None:
        """Schedule a claim for retry with appropriate delay"""
        
    async def should_retry(self, claim_id: str) -> Tuple[bool, Optional[str]]:
        """Check if a claim should be retried and return retry reason"""
        
    async def update_retry_count(self, claim_id: str, attempt_id: str) -> None:
        """Update retry statistics for monitoring"""
        
    async def get_retry_backoff(self, claim_id: str, error_type: str) -> int:
        """Calculate exponential backoff with jitter and error-specific adjustments"""
        
    async def cancel_retries(self, claim_id: str, reason: str) -> None:
        """Cancel pending retries for a claim"""
```

### Error Recovery Strategies

The system implements multiple recovery strategies for different error types:

1. **Transient Errors** (network issues, temporary provider unavailability):
   - Automatic retry with exponential backoff
   - Provider switching if available
   - Reduced complexity retry (simplified evaluation)

2. **Resource Conflicts** (file locks, concurrent modifications):
   - Gentle retry with increasing intervals
   - Conflict resolution with last-writer-wins policy
   - Operation decomposition to smaller atomic steps

3. **Permanent Errors** (invalid claims, unreachable resources):
   - Mark claim as failed with explanatory evidence
   - Create new claims with alternative approaches
   - Notify user through session events

4. **Cascading Failures** (multiple dependent claims failing):
   - Dependency graph analysis to isolate root causes
   - Strategic claim reordering to prevent future cascades
   - Temporary reduction in evaluation concurrency

### Performance Monitoring

The Concurrency Control system includes comprehensive monitoring:

```python
class ConcurrencyMonitor:
    async def get_lock_statistics(self) -> LockStatistics:
        """Get statistics about lock usage and contention"""
        
    async def get_provider_metrics(self, provider: str) -> ProviderMetrics:
        """Get performance metrics for a specific provider"""
        
    async def get_retry_statistics(self) -> RetryStatistics:
        """Get statistics about retry patterns and success rates"""
        
    async def get_evaluation_throughput(self, time_window: int) -> ThroughputMetrics:
        """Get metrics about evaluation throughput and queue depth"""
        
    async def identify_bottlenecks(self) -> List[BottleneckReport]:
        """Identify performance bottlenecks in the evaluation system"""
```

## Performance Targets and Optimization

### Performance Targets

The Async Claim Evaluation Service is designed with specific performance targets:

- **Claim_evaluation_start_time**: <50ms from queue to execution
- **Context_retrieval_time**: <100ms for typical claim contexts
- **LLM_provider_interaction**: <2 seconds for standard evaluations
- **Tool_execution_time**: <1 second for local tools, <5 seconds for web tools
- **Evaluation_throughput**: >100 claims/minute per evaluation node
- **File_lock_acquire_time**: <10ms for uncontested files
- **Provider_backoff_max**: <5 minutes for severely throttled providers

### Scalability Features

- **Horizontal Evaluation Scaling**: Multiple evaluation nodes processing claims independently
- **Intelligent Claim Routing**: Direct claims to evaluation nodes with relevant cached context
- **Connection Pooling**: Reused connections for tool and provider interactions
- **Adaptive Concurrency**: Dynamic adjustment of parallel evaluations based on system load
- **Resource Monitoring**: Real-time tracking of CPU, memory, and I/O utilization

### Optimization Strategies

#### Context Optimization

- **Claim Relevance Scoring**: Advanced algorithms to identify only the most relevant claims
- **Context Compression**: Intelligent summarization of claim clusters for dense knowledge areas
- **Fragmented Loading**: Load context in chunks as needed during evaluation
- **Context Caching**: Maintain most-frequently-used claim contexts in memory

#### Evaluation Optimization

- **Claim Batching**: Evaluate related claims in batches for improved provider efficiency
- **Dependency Grouping**: Group dependent claims for shared context loading
- **Priority Boosting**: Temporarily boost priority of claims that unlock many dependent claims
- **Evaluation Short-circuiting**: Skip evaluations when existing answers suffice

#### Resource Optimization

- **Smart Provider Selection**: Choose optimal provider based on claim type and current load
- **Tool Caching": Cache tool results when inputs are deterministic
- **Background Processing**: Move non-critical evaluations to background processing
- **Resource-aware Scheduling**: Schedule resource-intensive evaluations during low load periods

## Success Criteria

The Conjecture project succeeds when:
1. Users can understand the entire system in under 30 minutes
2. New skills can be added in a few lines of text
3. Claims clearly represent what we know and how confident we are
4. Tools work reliably without complex configuration
5. The system helps solve real problems without getting in the way

## Success Metrics

The project will be considered successful when it achieves:

1. End-to-end functionality for complex requests like "make a minesweeper in rust"
2. Consistent performance across all supported LLM providers
3. Transparent reasoning processes through traceable claim networks
4. Successful integration of tools and skill claims into coherent workflows
5. Interfaces that provide the same core functionality with different user experiences
6. Demonstrable reduction in hallucinations through evidence-based validation
7. Performance targets met for claim evaluation throughput and latency
8. Reliable concurrency handling under load with minimal resource conflicts

## Conclusion

Conjecture bridges the gap between AI capabilities and practical development workflows through a revolutionary architecture that replaces conversation history with an Async Claim Evaluation Service. The three-layer design of Data (Claims + Tools), Process (Context, LLM, and Async Evaluation), and Presentation (UI) provides a comprehensive framework for building knowledge systematically while maintaining transparency.

The claim-centric evaluation approach represents a fundamental shift from traditional LLM systems. Instead of maintaining an ever-growing context window, each evaluation is self-contained, using existing claims as context. This eliminates the context window bottleneck while enabling distributed evaluation and graceful error recovery. The event-driven UI updates ensure users stay informed without affecting the evaluation process.

The Async Claim Evaluation Service serves as the operational core, continuously processing claims based on priority while managing concurrency controls, provider throttling, and retry logic. This enables complex multi-step workflows that can span hours or days, with resilient error handling and graceful degradation.

With a focus on evidence-based validation and progressive confidence assessment, Conjecture creates a trustworthy foundation for AI-assisted development. The system's emphasis on traceability and explicit relationships between claims, combined with its sophisticated concurrency controls and performance optimization, enables reliable operation at scale.

With its current foundation approximately 70% complete, the project is well-positioned to deliver on its vision of making AI-assisted development more structured, transparent, and effective for complex real-world workflows.





