# Conjecture Project Requirements

**Last Updated:** November 9, 2025

## Overview

Conjecture is a simple framework for LLM agents to collect, process, and validate information through practical tools and clear thinking patterns.

## Core Architecture

### Simple Tool/Skill Distinction

**Tools**: External data collection and production
- WebSearch: Find information on the web
- ReadFiles: Access local files and data
- WriteCodeFile: Create and modify code files
- CreateClaim: Make structured statements about what we know
- ClaimSupport: Link evidence to claims

**Skills**: Simple context templates for LLM guidance
- Research: How to investigate and learn
- WriteCode: How to create working code
- TestCode: How to validate code works
- EndClaimEval: How to assess what we know

### Clean Architecture Principle

- **Tools** = How to get data in/out
- **Skills** = How to think about problems
- **Claims** = What we know and how confident we are

## Essential Requirements

### 1. Tool Requirements

All tools must:
- Be simple and focused on single responsibilities
- Accept clear inputs and produce structured outputs
- Handle errors gracefully
- Work independently without complex dependencies

**Specific Tool Requirements:**
- WebSearch: Return relevant, structured search results
- ReadFiles: Access and parse file contents safely
- WriteCodeFile: Create valid code with proper formatting
- CreateClaim: Generate structured claims with confidence scores
- ClaimSupport: Link evidence to claims with clear relationships

### 2. Skill Requirements

All skills must:
- Be simple, repeatable thinking templates
- Provide step-by-step guidance
- Focus on practical outcomes
- Avoid over-engineered complexity

**Core Skills as Simple Templates:**

**Research Skill:**
```
To research: 
1) Search web for information
2) Read relevant files  
3) Create claims for key findings
4) Support claims with evidence
```

**WriteCode Skill:**
```
To write code:
1) Understand requirements
2) Design solution
3) Write code file
4) Test it works
5) Create claims about the solution
```

**TestCode Skill:**
```
To test code:
1) Write test cases
2) Run tests
3) Fix what fails
4) Create claims about test results
```

**EndClaimEval Skill:**
```
To evaluate claims:
1) Review supporting evidence
2) Check for contradictions
3) Update confidence scores
4) Note gaps
```

### 3. Claim System Requirements

Claims must:
- Represent clear statements about what we know
- Include confidence scores (0-100%)
- Link to supporting evidence
- Allow for updates and revisions

### 4. Practical Value Requirements

The system must be:
- **Easy to understand**: New team members can grasp it quickly
- **Maintainable**: Simple to modify and extend
- **Flexible**: Can adapt to different problem domains
- **Focused**: Prioritizes getting work done over theoretical perfection

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

## Success Criteria

The Conjecture project succeeds when:
1. Users can understand the entire system in under 30 minutes
2. New skills can be added in a few lines of text
3. Claims clearly represent what we know and how confident we are
4. Tools work reliably without complex configuration
5. The system helps solve real problems without getting in the way