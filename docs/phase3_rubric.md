# Phase 3: Basic Skills Templates - Implementation Rubric

## Overview

**Phase Goal**: Create simple thinking templates for core workflows with proper Agent Harness architecture.

**Success Threshold**: 8.0/10 (Production Ready)  
**Excellent Threshold**: 9.0/10 (Exceeds Expectations)

---

## Evaluation Categories

### 1. Agent Harness Architecture (Weight: 30%)

#### 1.1 Separation of Concerns (10/10)
- **Agent Harness**: Orchestration, session management, state tracking
- **Support Systems**: Data collection, context building, persistence  
- **Core LLM Prompt**: The actual prompt sent to LLM with context
- **Skills**: Simple guidance templates

#### 1.2 Session Management (10/10)
- Session creation and isolation
- State persistence and recovery
- Multi-session support
- Resource cleanup

#### 1.3 Context Building (10/10)
- Relevant claim collection
- Skill template integration
- Tool availability context
- Prompt assembly

### 2. Skills Implementation (Weight: 25%)

#### 2.1 Research Skill (10/10)
- Clear 4-step process template
- Integration with WebSearch and ReadFiles tools
- Claim creation guidance
- Evidence linking instructions

#### 2.2 WriteCode Skill (10/10)
- Clear development lifecycle template
- Integration with WriteCodeFile tool
- Testing guidance
- Claim creation about solutions

#### 2.3 TestCode Skill (10/10)
- Clear testing process template
- Validation guidance
- Error handling instructions
- Claim creation about test results

#### 2.4 EndClaimEval Skill (10/10)
- Clear evaluation process template
- Evidence review guidance
- Confidence updating instructions
- Gap identification guidance

### 3. Core LLM Prompt System (Weight: 20%)

#### 3.1 Prompt Assembly (10/10)
- Dynamic prompt construction
- Context integration
- Skill template inclusion
- Tool availability information

#### 3.2 Prompt Quality (10/10)
- Clear, concise instructions
- Proper context formatting
- Effective skill guidance
- Tool usage examples

#### 3.3 Response Parsing (10/10)
- Tool call extraction
- Claim identification
- Skill application detection
- Error handling

### 4. Integration with Existing Systems (Weight: 15%)

#### 4.1 Tool Integration (10/10)
- Seamless tool access
- Tool result processing
- Error handling
- Performance optimization

#### 4.2 Claim System Integration (10/10)
- Claim creation from LLM responses
- Evidence linking
- Confidence scoring
- Storage integration

#### 4.3 Data Layer Integration (10/10)
- Data persistence
- Retrieval operations
- Query optimization
- Error handling

### 5. Testing and Quality Assurance (Weight: 10%)

#### 5.1 Unit Tests (10/10)
- All components tested
- Edge cases covered
- Mock dependencies
- High code coverage (>90%)

#### 5.2 Integration Tests (10/10)
- End-to-end workflows
- Component interaction
- Error scenarios
- Performance validation

#### 5.3 Scenario Tests (10/10)
- Real-world usage patterns
- Complex problem solving
- Multi-step workflows
- Quality validation

---

## Success Criteria

### Minimum Viable (7.0/10)
- [ ] Basic Agent Harness with session management
- [ ] Four core skill templates implemented
- [ ] Integration with existing tools and claims
- [ ] Basic testing coverage
- [ ] Simple prompt assembly

### Production Ready (8.0/10)
- [ ] Complete separation of concerns
- [ ] Robust session management with persistence
- [ ] High-quality skill templates
- [ ] Comprehensive integration
- [ ] Full test coverage with integration tests
- [ ] Performance benchmarks met

### Excellent (9.0/10)
- [ ] Advanced Agent Harness features
- [ ] Optimized context building
- [ ] Sophisticated prompt engineering
- [ ] Extensive testing including edge cases
- [ ] Performance exceeds benchmarks
- [ ] Documentation and examples

---

## Performance Benchmarks

### Response Time Targets
- **Session Initialization**: <100ms
- **Context Building**: <200ms
- **Prompt Assembly**: <50ms
- **Response Parsing**: <100ms

### Resource Usage
- **Memory per Session**: <50MB
- **CPU Usage**: <10% during normal operation
- **Storage Overhead**: <10MB for 100 sessions

### Quality Metrics
- **Skill Template Effectiveness**: >80% successful application
- **Prompt Clarity**: >90% user comprehension
- **Integration Reliability**: >95% success rate
- **Test Coverage**: >90%

---

## Evaluation Process

### 1. Implementation Review
- Code architecture assessment
- Design pattern validation
- Best practices compliance

### 2. Functional Testing
- Unit test execution
- Integration test validation
- Scenario test completion

### 3. Performance Testing
- Benchmark execution
- Resource usage monitoring
- Stress testing

### 4. Quality Assessment
- Code review
- Documentation review
- User experience validation

---

## Deliverables

### Core Components
- [ ] Agent Harness system
- [ ] Session management
- [ ] Context builder
- [ ] Prompt assembler
- [ ] Response parser

### Skill Templates
- [ ] Research skill template
- [ ] WriteCode skill template
- [ ] TestCode skill template
- [ ] EndClaimEval skill template

### Integration Layer
- [ ] Tool integration
- [ ] Claim system integration
- [ ] Data layer integration

### Testing Suite
- [ ] Unit tests
- [ ] Integration tests
- [ ] Scenario tests
- [ ] Performance tests

### Documentation
- [ ] Architecture documentation
- [ ] API documentation
- [ ] Usage examples
- [ ] Testing guide

---

## Failure Modes and Recovery

### Critical Failures
- Session corruption → Automatic recovery from persistence
- Context building failure → Fallback to basic context
- Tool integration failure → Graceful degradation
- Prompt assembly failure → Emergency prompt template

### Warning Conditions
- Performance degradation → Optimization triggers
- Resource usage high → Cleanup procedures
- Error rate increase → Fallback mechanisms
- Quality metrics drop → Review and refinement

---

## Success Metrics

### Quantitative
- All benchmarks met or exceeded
- Test coverage >90%
- Error rate <5%
- Performance targets achieved

### Qualitative
- Clean, maintainable code
- Clear separation of concerns
- Effective skill templates
- Smooth user experience

### Validation
- Independent review passes
- User testing successful
- Performance validation complete
- Quality assurance approved