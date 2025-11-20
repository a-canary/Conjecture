# Conjecture System Requirements

## Document Overview
**Version**: 1.0  
**Date**: November 14, 2025  
**Purpose**: This document provides a comprehensive list of all expectations, features, and requirements for the Conjecture system as specified in the project design document.

## 1. System Architecture

### 1.1 Architecture Overview
1.1.1 The system shall implement a three-layer architecture comprising Data Layer, Process Layer, and Presentation Layer
1.1.2 The system shall follow a claim-centric approach where all knowledge, including methodologies, is represented as claims
1.1.3 The system shall operate without conversation history context, using claims as persistent context
1.1.4 The system shall implement an Async Claim Evaluation Service as the operational core
1.1.5 The system shall maintain clean separation of concerns with minimal dependencies between components

### 1.2 Data Layer Requirements
1.2.1 The system shall store structured claims as the primary knowledge representation
1.2.2 The system shall implement specialized claim types: Knowledge Claims, Skill Claims, Tool Claims, and Evaluation Claims
1.2.3 The system shall include tool mechanisms for data collection and production
1.2.4 The system shall maintain claim scopes: Global, Team, Project, User, and Session
1.2.5 The system shall implement a dirty flag mechanism to identify claims requiring re-evaluation

### 1.3 Process Layer Requirements
1.3.1 The system shall implement context building from existing claims (not conversation history)
1.3.2 The system shall provide Async Claim Evaluation Service with continuous processing
1.3.3 The system shall implement LLM interaction orchestration with tool calls and claim creation
1.3.4 The system shall provide response parsing without including history in future context
1.3.5 The system shall implement scope-aware context building
1.3.6 The system shall implement confidence-driven evaluation continuation

### 1.4 Presentation Layer Requirements
1.4.1 The system shall provide Command Line Interface (CLI) implementation
1.4.2 The system shall provide Terminal User Interface (TUI) with rich claim visualizations
1.4.3 The system shall provide Graphical User Interface (GUI) for complex workflows
1.4.4 The system shall provide API/Model Context Protocol (MCP) for external integrations
1.4.5 The system shall implement event-driven UI updates independent of evaluation context

## 2. Claim System Requirements

### 2.1 Claim Structure and Management
2.1.1 Claims shall include unique identifiers, statements, confidence scores, and dirty flags
2.1.2 Claims shall maintain list of claims that support this claim, collected for context generation
2.1.2 Claims shall maintain list of claims that are supported_by this claim, to flag dirty when this claim is updated
2.1.3 Claims shall include scope information (session, user, project, team, global)
2.1.4 Claims shall maintain creation and modification timestamps
.1.6 New claims shall default to session scope with dirty flag set to true

### 2.2 Claim Lifecycle Management
2.2.1 The system shall create claims only from LLM responses
2.2.2 The system shall evaluate dirty claims until LLM sets confidence and marks as clean
2.2.3 The system shall revise claims when new evidence arrives or contradictions are found (sets dirty=true)
2.2.4 The system shall establish relationships between claims: supporting or supported_by
2.2.5 the system shall have a configurable maximum database size, when the database size exceeds the configured limit, the system shall delete 10% of claims that are [oldest and lowest confidence] claims 

### 2.3 Claim Scopes and Access Control
2.2.5 The system shall implement scope inheritance for claim accessibility
2.2.6 The system shall enable LLM-initiated scope elevation to User/Project scopes
2.2.7 The system shall require manual elevation for Team/Global scopes
2.3.1	Session scoped claims shall be accessible only to the current session
2.3.2	User scoped claims shall be accessible to the specific user across all sessions
2.3.3	Project scoped claims shall be accessible to all project contributors
2.3.4	Team scoped claims shall be accessible to all team members
2.3.5	Global scoped claims shall be accessible to all users and sessions
2.3.6	The system shall implement scope-specific caching strategies
2.3.7	The system shall resolve cross-scope claim conflicts using specificity, freshness, confidence, and source rules

## 3. Async Claim Evaluation Service Requirements

### 3.1 Evaluation Engine
3.1.1 The system shall maintain a priority queue of dirty claims to evaluate
3.1.2 The system shall evaluate top N dirtiest claims sequentially using scope-aware context building
3.1.3 The system shall implement confidence-driven evaluation: LLM autonomously decides when to stop exploring
3.1.4 The system shall process one tool call at a time to manage response sizes and context limits
3.1.5 The system shall mark claims as clean (not dirty) when LLM sets confidence and completes evaluation
3.1.6 The system shall implement iteration limits to prevent infinite evaluation loops

### 3.2 Evaluation Process
3.2.1 The system shall generate claim context from existing claims (not conversation history)
3.2.2 The system shall send context to LLM with encouragement to explore further (max one tool call)
3.2.3 The system shall parse LLM responses for tool calls, new claims, and confidence updates
3.2.4 The system shall store new claims created by LLM with session scope by default
3.2.5 The system shall evaluate new claims for scope elevation opportunities
3.2.6 The system shall execute single tool calls and update context with individual responses
3.2.7 The system shall update claim confidence and mark as clean when no more tools are needed

### 3.3 Scope Elevation
3.3.1 The system shall leverage LLM evaluation and responses to determine scope elevation 
3.3.2 The system shall merge similar claims and elevate to highest scope level needed
3.3.3 The system shall request user approval to evelvate to Global

## 4. Tools and Tool Management

### 4.1 Core Tool Requirements
4.1.1 The system shall provide WebSearch tool for finding information on the web
4.1.2 The system shall provide ReadFiles tool for accessing local files and data
4.1.3 The system shall provide WriteCodeFile tool for creating and modifying code files
4.1.4 The system shall provide CreateClaim tool for making structured statements
4.1.5 The system shall provide ClaimSupport tool for linking evidence to claims
4.1.6 The system shall provide FileLock tool for managing concurrent access

### 4.2 Tool Characteristics
4.2.1 Tools shall have simple interfaces with one clear purpose per tool
4.2.2 Tools shall implement consistent input/output patterns
4.2.3 Tools shall implement error handling with graceful failure modes
4.2.4 Tools shall not modify other tools' state
4.2.5 Tools shall produce structured or raw output for easy LLM consumption
4.2.6 Tools shall generate evaluation events for UI updates

## 5. Skills and Skill Management

### 5.1 Skill Representation
5.1.1 Skills shall be represented as claims about methodologies and approaches
5.1.3 Skill claims shall specify applicable contexts where they should be used
5.1.4 Skills shall have confidence scores and can be supported by claims of examples or past uses (failure or success)
5.1.5 Skills shall be retrievable by relevance to the current task context
5.1.6 Skills shall be evolvable and can be refined as better methodologies are discovered

### 5.2 Claim Priming
5.2.1 Tool implemenations shall inject claim examples of how they are used (or not used)
5.2.2 The system shall inject claims for externally provided skills, project best practices and project principles

## 6. Session Management Requirements

### 6.1 Session Lifecycle
6.1.1 The system shall create user sessions with unique identifiers
6.1.2 The system shall maintain session state and history
6.1.3 The system shall process user prompts by creating a claim to be evaluated
6.1.4 The system shall track claim progress for each session
6.1.5 The system shall stream evaluation events to sessions for UI updates
6.1.6 The system shall provide session summaries showing progress and results

### 6.2 Session-Claim Interaction
6.2.1 The system shall add user-generated claims to evaluation queue
6.2.2 The system shall provide temporary priority boosts for claims from user prompts
6.2.3 The system shall implement scope-aware spreading of priority from user claims
6.2.4 The system shall implement priority decay functions as other claims enter the system
6.2.5 The system shall re-boost related claims on user interaction
6.2.6 The system shall enable claim scope promotion from sessions

### 6.3 Session Controls
6.3.1 The system shall provide options to pause evaluation for sessions
6.3.2 The system shall allow manual adjustment of claim priorities
6.3.3 The system shall provide constraint options for tool and provider usage
6.3.4 The system shall allow configuration of confidence thresholds
6.3.5 The system shall provide summary reports of session progress

## 7. Concurrency Control Requirements

### 7.1 File Locking System
7.1.1 The system shall implement file locking for shared resources during claim evaluations
7.1.2 The system shall provide lock acquisition with timeout capabilities
7.1.3 The system shall implement lock release mechanisms
7.1.4 The system shall provide lock status checking capabilities
7.1.5 The system shall implement context execution while holding locks

### 7.2 Provider Rate Limiting
7.2.1 The system shall manage interactions with LLM providers to prevent throttling
7.2.2 The system shall implement provider-specific rate limit configurations
7.2.3 The system shall calculate backoff times with jitter for rate limit compliance
7.2.4 The system shall provide slot waiting mechanisms for provider availability
7.2.5 The system shall record request metrics for adaptive throttling
7.2.6 The system shall provide provider status monitoring

### 7.3 Claim Retry Logic
7.3.1 The system shall handle retries for failed evaluations with exponential backoff
7.3.2 The system shall schedule retries based on error type and attempt count
7.3.3 The system shall track retry statistics for monitoring and optimization
7.3.4 The system shall calculate appropriate backoff times with error-specific adjustments
7.3.5 The system shall provide mechanisms to cancel pending retries

## 8. Error Handling and Recovery

### 8.1 Error Classification
8.1.1 The system shall categorize transient errors (network issues, temporary unavailability)
8.1.2 The system shall categorize resource conflicts (file locks, concurrent modifications)
8.1.3 The system shall categorize permanent errors (invalid claims, unreachable resources)
8.1.4 The system shall categorize cascading failures (multiple dependent claim failures)

### 8.2 Recovery Strategies
8.2.1 The system shall implement automatic retry with exponential backoff for transient errors
8.2.2 The system shall provide provider switching mechanisms when available
8.2.3 The system shall implement gentle retry with increasing intervals for resource conflicts
8.2.4 The system shall mark claims as failed with explanatory evidence for permanent errors
8.2.5 The system shall analyze dependency graphs to isolate cascading failure root causes
8.2.6 The system shall implement temporary reduction in evaluation concurrency during issues

## 9. Performance Requirements

### 9.1 Performance Targets
9.1.1 Claim evaluation start time shall be less than 50ms from queue to execution
9.1.2 Context retrieval time shall be less than 100ms for typical claim contexts
9.1.3 LLM provider interaction shall be less than 2 seconds for standard evaluations
9.1.4 Tool execution time shall be less than 1 second for local tools, 5 seconds for web tools
9.1.5 Evaluation throughput shall exceed 100 claims/minute per evaluation node
9.1.6 File lock acquire time shall be less than 10ms for uncontested files
9.1.7 Provider backoff maximum shall be less than 5 minutes for severely throttled providers

### 9.2 Scalability Features
9.2.1 The system shall support horizontal evaluation scaling across multiple nodes
9.2.2 The system shall implement intelligent claim routing to evaluation nodes with relevant cached context
9.2.3 The system shall provide connection pooling for tool and provider interactions
9.2.4 The system shall implement adaptive concurrency adjustment based on system load
9.2.5 The system shall provide real-time monitoring of CPU, memory, and I/O utilization

### 9.3 Optimization Strategies
9.3.1 The system shall implement claim relevance scoring for context optimization
9.3.2 The system shall provide context compression for dense knowledge areas
9.3.3 The system shall implement fragmented loading of context chunks as needed
9.3.4 The system shall maintain most-frequently-used claim contexts in memory cache
9.3.5 The system shall implement claim batching for improved provider efficiency
9.3.6 The system shall group dependent claims for shared context loading

## 10. Event System Requirements

### 10.1 Event Types
10.1.1 The system shall emit ClaimStartedEvent when claim evaluation begins
10.1.2 The system shall emit ConfidenceUpdatedEvent when claim confidence changes
10.1.3 The system shall emit ToolCalledEvent when tools are invoked
10.1.4 The system shall emit ToolResponseEvent when tools return results
10.1.5 The system shall emit ClaimCompletedEvent when evaluation finishes
10.1.6 The system shall emit ErrorEvent when errors occur during evaluation
10.1.7 The system shall emit ScopePromotionEvent when claim scope changes

### 10.2 Event Management
10.2.1 Events shall be emitted independently of claim evaluation context
10.2.2 Events shall include claim IDs and relevant metadata
10.2.3 Events shall be scope-decorated for UI filtering
10.2.4 Events shall provide real-time updates without affecting evaluation performance
10.2.5 Events shall be streamed to appropriate sessions and interfaces

## 11. Performance Monitoring

### 11.1 Monitoring Capabilities
11.1.1 The system shall provide lock statistics including usage and contention metrics
11.1.2 The system shall provide performance metrics for specific providers
11.1.3 The system shall provide retry statistics showing patterns and success rates
11.1.4 The system shall provide evaluation throughput measuring queue depth and processing speed
11.1.5 The system shall identify performance bottlenecks across the evaluation system

## 12. Quality Assurance Requirements

### 12.1 Success Metrics
12.1.1 The system shall enable end-to-end functionality for complex requests like "make a minesweeper in rust"
12.1.2 The system shall provide consistent performance across all supported LLM providers
12.1.3 The system shall provide transparent reasoning processes through traceable claim networks
12.1.4 The system shall provide successful integration of tools and skill claims into coherent workflows
12.1.5 The system shall provide interfaces that deliver the same core functionality with different user experiences
12.1.6 The system shall demonstrate reduction in hallucinations through evidence-based validation
12.1.7 The system shall meet performance targets for claim evaluation throughput and latency
12.1.8 The system shall provide reliable concurrency handling under load with minimal resource conflicts

## 13. Implementation Phases

### 13.1 Phase 1: Core Foundation
13.1.1 Implement basic tools including WebSearch, ReadFiles, WriteCodeFile, CreateClaim, and FileLock
13.1.2 Create skill claims with templates for basic methodologies
13.1.3 Build simple claim storage using JSON files initially
13.1.4 Implement basic Async Claim Evaluation Service with simple priority queue
13.1.5 Test with basic research and coding tasks to validate evaluation workflow

### 13.2 Phase 2: Async Claim Evaluation Service
13.2.1 Implement priority queue management for claim evaluations
13.2.2 Implement context building from existing claims (not conversation history)
13.2.3 Implement basic concurrency controls including file locking and simple rate limiting
13.2.4 Implement event system for UI updates
13.2.5 Implement retry logic with exponential backoff

### 13.3 Phase 3: Enhanced Evaluation Capabilities
13.3.1 Implement advanced dependency tracking between claims
13.3.2 Implement sophisticated provider throttling and backoff strategies
13.3.3 Implement parallel evaluation queue management
13.3.4 Implement evaluation priority algorithms
13.3.5 Implement performance monitoring and optimization

### 13.4 Phase 4: Interface Development
13.4.1 Implement Terminal User Interface (TUI) with real-time event display
13.4.2 Implement Web User Interface (WebUI) with claim visualization
13.4.3 Implement advanced CLI features for evaluation monitoring
13.4.4 Implement Model Context Protocol (MCP) Interface for external integrations

## 14. User Interface Requirements

### 14.1 General Principles
14.1.1 The system should provide clean, intuitive, and responsive user interfaces (CLI, TUI, GUI).
14.1.2 Process layer events shall trigger UI updates with smooth animations to provide continuous and clear feedback.
14.1.3 Interfaces should provide clear and constant feedback on asynchronous operations, ensuring the user understands the state of the system.
14.1.4 The UI should organize and present data effectively, explicitly showing:
    *   Claim dependencies and relations.
    *   Active evaluations.
    *   Confidence levels for claims.
14.1.5 The design should prioritize clarity and ease of use, making complex system interactions manageable for the user.
