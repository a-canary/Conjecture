# Conjecture Design Documentation

**Last Updated:** November 11, 2025  
**Version:** 1.0  

This directory contains comprehensive design documentation for the Simple Universal Claim Architecture with LLM-Driven Instruction Support. The documentation is organized into logical sections covering architecture, implementation details, and migration strategies.

## Documentation Structure

### 📋 Architecture Overview
**[Architectural Design](./architecture/simple-universal-claim-architecture.md)**  
Core architectural design document outlining the Simple Universal Claim Architecture approach.

- Executive summary and design principles
- Universal Claim model specifications  
- Context building strategy and LLM responsibilities
- Performance considerations and comparison with complex approaches

### 🔧 Context Building System
**[Context Builder Specification](./context/complete-relationship-context-builder.md)**  
Detailed specification for building comprehensive claim contexts with complete relationship coverage.

- Complete relationship traversal algorithms
- Token management and optimization strategies
- Context formatting for LLM consumption
- Performance monitoring and caching strategies

### 🤖 LLM Integration Protocol
**[Instruction Support Relationship Protocol](./llm/instruction-support-relationship-protocol.md)**  
Protocol for LLM-driven instruction identification and support relationship creation.

- Comprehensive prompt templates
- Processing workflows and data structures
- Error handling and validation strategies
- Performance optimization and monitoring

### 🛠️ Implementation Guide
**[Unified Claim System Implementation](./implementation/unified-claim-system-implementation.md)**  
Complete implementation roadmap and technical specifications.

- Step-by-step implementation phases
- API specifications and database schemas
- Testing strategies and validation approaches
- Configuration management and deployment considerations

### 🔄 Migration Strategy
**[Simplified Architecture Migration](./migration/simplified-architecture-migration.md)**  
Comprehensive migration plan from complex to simplified architecture.

- Phase-wise migration approach
- Data model unification and validation
- Risk mitigation and rollback procedures
- Success metrics and timeline

## Key Design Principles

### 1. Simplicity First
- **Single Universal Model**: One Claim model handles all use cases
- **No Enhanced Structures**: Elimination of complex data hierarchies
- **Minimal Dependencies**: 90% of functionality with 10% complexity

### 2. Complete Relationship Coverage
- **All Supporting Claims**: Complete traversal to root claims
- **All Supported Claims**: Full descendant relationship coverage
- **Semantic Context**: Intelligent claim selection for remaining tokens

### 3. LLM-Driven Intelligence
- **Instruction Identification**: Natural language recognition of instruction claims
- **Support Relationships**: LLM-creation of logical support connections
- **Quality Assurance**: Automated validation and improvement suggestions

### 4. Performance Optimization
- **Token Efficiency**: Prioritized allocation to relationship claims
- **Fast Context Building**: Optimized traversal and caching
- **Scalable Architecture**: Horizontal scaling capabilities

## Implementation Quick Reference

### Universal Claim Model
```python
class Claim(BaseModel):
    id: str
    content: str
    confidence: float
    state: ClaimState
    supported_by: List[str]
    supports: List[str]
    type: List[ClaimType]
    tags: List[str]
    created_by: str
    created: datetime
    updated: datetime
    # No additional fields needed - LLM handles intelligence
```

### Core API Endpoints
```python
# Claim Management
POST   /api/claims                    # Create claim
GET    /api/claims/{id}               # Get claim
PUT    /api/claims/{id}/confidence    # Update confidence

# Context Building
POST   /api/context/build             # Build complete context
GET    /api/context/summary/{id}      # Get context statistics

# LLM Analysis
POST   /api/analysis/instructions     # Analyze instructions
POST   /api/analysis/improve          # Improve context quality
```

### Token Allocation Strategy
| Priority | Content Type | Allocation | Rationale |
|----------|--------------|------------|-----------|
| 1 | Upward Support Chain | 40% | Critical for understanding context and validity |
| 2 | Downward Supported Claims | 30% | Shows implications and dependencies |
| 3 | Semantic Similar Claims | 30% | Provides broader context and connections |

## Development Workflow

### 1. Setup Development Environment
```bash
# Clone repository
git clone <repository-url>
cd Conjecture

# Install dependencies
pip install -r requirements.txt

# Setup configuration
cp .env.example .env
# Edit .env with your configuration
```

### 2. Initialize Database
```bash
# Run database migrations
python -m src.core.migration migrate

# Run initial data seeding (optional)
python -m scripts.seed_data
```

### 3. Start Development Server
```bash
# Start API server
python -m src.api.main

# Or using Docker
docker-compose up -d
```

### 4. Run Tests
```bash
# Run all tests
pytest

# Run specific test suites
pytest tests/unit/
pytest tests/integration/

# Run with coverage
pytest --cov=src tests/
```

## Getting Started Examples

### Creating Claims
```python
from src.core.models import Claim, ClaimType, ClaimState
from src.data.data_manager import get_data_manager

# Create a basic claim
data_manager = get_data_manager()

claim = Claim(
    id="example_001",
    content="Always validate user input before processing",
    confidence=0.95,
    state=ClaimState.EXPLORE,
    supported_by=["security_principle_001"],
    supports=[],
    type=[ClaimType.CONCEPT],
    tags=["security", "validation", "best-practice"],
    created_by="developer",
    created=datetime.utcnow(),
    updated=datetime.utcnow()
)

saved_claim = await data_manager.save_claim(claim)
```

### Building Context
```python
from src.context.context_builder import get_context_builder

# Build complete context for analysis
context_builder = get_context_builder()

context = await context_builder.build_complete_context(
    target_claim_id="example_001",
    max_tokens=8000
)

print(f"Context includes {len(context['upward_chain'])} supporting claims")
print(f"Context includes {len(context['downward_claims'])} supported claims")
print(f"Context includes {len(context['semantic_claims'])} semantic claims")
```

### LLM Analysis
```python
from src.llm.llm_processor import get_llm_processor

# Analyze context for instructions and relationships
llm_processor = get_llm_processor()

analysis = await llm_processor.process_with_instruction_support(
    context=context["formatted_context"],
    user_request="How should I handle security in my application?",
    validation_enabled=True
)

print(f"Identified {len(analysis.instructions)} instruction claims")
print(f"Suggested {len(analysis.new_relationships)} new relationships")
```

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Context Building Time | < 200ms | Average time for typical claim networks |
| LLM Response Time | < 2s | Standard analysis with validation |
| Memory Usage | < 100MB | For 10,000 claim networks |
| Database Query Time | < 50ms | Relationship traversal operations |
| API Response Time | < 500ms | Complete request processing |
| Cache Hit Rate | > 70% | Frequently accessed contexts |

## Migration Timeline

### Phase 1: Data Model Unification (Weeks 1-3)
- Schema analysis and conversion scripts
- Data migration with validation
- API compatibility layer implementation

### Phase 2: Context Builder Migration (Weeks 4-5)
- Algorithm migration and performance testing
- Complete rollout with monitoring

### Phase 3: LLM Integration (Weeks 6-8)
- LLM protocol implementation
- Gradual rollout and comparison testing
- Complete migration and optimization

### Phase 4: Cleanup and Optimization (Weeks 9-10)
- Legacy code removal
- System optimization and documentation
- Performance validation

## Support and Contributing

### Documentation Maintenance
- Keep all design documents updated with implementation changes
- Add examples and use cases as they are discovered
- Maintain API documentation in sync with code

### Code Standards
- Follow the established code structure and naming conventions
- Write comprehensive tests for all new features
- Update documentation for any architectural changes

### Issue Reporting
- Use the issue tracker for bugs and feature requests
- Include detailed reproduction steps and system information
- Tag issues with the appropriate documentation section

## Related Documents

### Historical Documents
- [Original System Design](./design.md) - Pre-simplification architecture
- [Requirements Specification](./requirements.md) - Original system requirements
- [Phase Planning](./phases.md) - Historical development phases

### Supporting Documents
- [Agent Harness Architecture](./agent_harness_architecture.md) - Related agent components
- [Interface Design](./interface_design.md) - User interface specifications
- [Sophisticated Skill Frameworks](./sophisticated_skill_frameworks_summary.md) - Skill system components

---

**Note**: This documentation represents the current state of the Simple Universal Claim Architecture. As implementation progresses, documents will be updated to reflect actual implementation decisions and lessons learned.

For questions or contributions, please refer to the project's contribution guidelines or contact the architecture team.