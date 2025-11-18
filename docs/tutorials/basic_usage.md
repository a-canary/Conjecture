# Conjecture Examples

This document provides practical examples of how to use Conjecture for various tasks.

## 1. Basic Exploration and Claim Creation

```python
from src.enhanced_conjecture import EnhancedConjecture, ExplorationResult
from core.unified_models import Claim, ClaimType

# Initialize Conjecture
cf = EnhancedConjecture()

# Explore a topic
print("🔍 Exploring machine learning...")
result = cf.explore("machine learning algorithms", max_claims=5)
print(result.summary())

# Create individual claims
claim1 = cf.add_claim(
    content="Supervised learning requires labeled training data",
    confidence=0.9,
    claim_type="concept",
    tags=["ml", "supervised", "data"]
)

claim2 = cf.add_claim(
    content="Neural networks can model complex non-linear relationships",
    confidence=0.85,
    claim_type="thesis",
    tags=["ml", "neural-networks", "math"]
)

print(f"✅ Created claims: {claim1.id}, {claim2.id}")
```

## 2. Building Claim Networks

```python
# Create a network of related claims
cf = EnhancedConjecture()

# Core concept
core_claim = cf.add_claim(
    content="Gradient descent is an optimization algorithm used in machine learning",
    confidence=0.95,
    claim_type="concept",
    tags=["ml", "optimization", "math"]
)

# Supporting claims
support1 = cf.add_claim(
    content="Gradient descent minimizes a cost function by iteratively moving in the direction of steepest descent",
    confidence=0.9,
    claim_type="concept",
    tags=["optimization", "math", "calculus"]
)

support2 = cf.add_claim(
    content="Common variants include stochastic gradient descent and Adam optimizer",
    confidence=0.85,
    claim_type="example",
    tags=["optimization", "variants", "sgd", "adam"]
)

# Build relationships
core_claim.add_support(support1.id)
core_claim.add_support(support2.id)
support1.add_supports(core_claim.id)
support2.add_supports(core_claim.id)

print("✅ Built claim network with relationships")
```

## 3. Research Workflow

```python
import asyncio

async def research_workflow():
    """Complete research workflow example"""
    async with EnhancedConjecture() as cf:
        # Step 1: Initial exploration
        print("📚 Step 1: Exploring renewable energy sources...")
        exploration = await cf.explore("renewable energy sources", max_claims=10)
        print(f"Found {len(exploration.claims)} initial claims")
        
        # Step 2: Deep dive on specific topics
        print("\n🔬 Step 2: Deep diving on solar energy...")
        solar_exploration = await cf.explore("solar panel efficiency factors", max_claims=8)
        
        # Step 3: Create synthesized claims
        print("\n⚡ Step 3: Creating synthesized insights...")
        synthesis = await cf.add_claim(
            content="Solar panel efficiency is primarily determined by semiconductor material quality and environmental factors",
            confidence=0.8,
            claim_type="thesis",
            tags=["solar", "efficiency", "materials", "environment"]
        )
        
        # Step 4: Wait for evaluation
        print("\n⏱️ Step 4: Waiting for claim evaluation...")
        evaluation = await cf.wait_for_evaluation(synthesis.id, timeout=60)
        print(f"Evaluation result: {evaluation}")
        
        # Step 5: Review statistics
        print("\n📊 Step 5: System statistics...")
        stats = cf.get_statistics()
        print(f"Claims processed: {stats['claims_processed']}")
        print(f"Tools created: {stats['tools_created']}")

# Run the workflow
# asyncio.run(research_workflow())
```

## 4. Performance Testing Example

```python
from tests.performance_benchmarks_final import PerformanceBenchmark

def performance_demo():
    """Demonstrate performance capabilities"""
    print("🚀 Running performance demonstration...")
    
    benchmark = PerformanceBenchmark()
    
    # Test claim creation speed
    creation_result = benchmark.benchmark_claim_creation(1000)
    print(f"Claim creation: {creation_result['claims_per_second']:.0f} claims/sec")
    
    # Test relationship operations
    relationship_result = benchmark.benchmark_claim_relationships(500)
    print(f"Relationship ops: {relationship_result['operations_per_second']:.0f} ops/sec")
    
    # Test concurrent operations
    concurrent_result = benchmark.benchmark_concurrent_operations(4, 250)
    print(f"Concurrent ops: {concurrent_result['ops_per_second']:.0f} ops/sec")

# Run performance demo
# performance_demo()
```

## 5. Tool Validation Example

```python
from tests.test_tool_validator_simple import ToolValidator

def tool_validation_demo():
    """Demonstrate tool validation capabilities"""
    validator = ToolValidator()
    
    # Safe code example
    safe_code = '''
def execute(param: str) -> dict:
    """Execute a safe mathematical operation"""
    import math
    result = math.sqrt(len(param))
    return {
        "success": True,
        "result": result,
        "length": len(param)
    }
'''
    
    is_valid, issues = validator.validate_tool_code(safe_code)
    print(f"Safe code validation: {is_valid}")
    if issues:
        print(f"Issues: {issues}")
    
    # Unsafe code example
    unsafe_code = '''
import os
def execute(param: str) -> dict:
    """Dangerous file system operation"""
    os.system("rm -rf /")  # Very dangerous!
    return {"success": True}
'''
    
    is_valid, issues = validator.validate_tool_code(unsafe_code)
    print(f"Unsafe code validation: {is_valid}")
    if issues:
        print(f"Security issues found: {issues}")

# Run tool validation demo
# tool_validation_demo()
```

## 6. Context Building Example

```python
def context_building_demo():
    """Demonstrate context building capabilities"""
    from core.unified_models import Claim, ClaimType
    
    # Create context claims
    context_claims = [
        Claim(
            id="ctx_001",
            content="Machine learning algorithms learn patterns from data",
            confidence=0.9,
            type=[ClaimType.CONCEPT],
            tags=["ml", "algorithms", "learning"]
        ),
        Claim(
            id="ctx_002",
            content="Neural networks are inspired by biological brain structure",
            confidence=0.85,
            type=[ClaimType.CONCEPT],
            tags=["ml", "neural-networks", "biology"]
        ),
        Claim(
            id="ctx_003",
            content="Deep learning uses multiple layers of neural networks",
            confidence=0.9,
            type=[ClaimType.CONCEPT],
            tags=["ml", "deep-learning", "neural-networks"]
        )
    ]
    
    # Format for context
    formatted_context = []
    for claim in context_claims:
        formatted = claim.format_for_context()
        formatted_context.append(formatted)
        print(f"Context: {formatted}")

# Run context building demo
# context_building_demo()
```

## 7. Integration Testing Example

```python
def integration_test_demo():
    """Demonstrate complete system integration"""
    print("🧪 Running integration test...")
    
    try:
        # Test core functionality
        from core.unified_models import Claim, ClaimType, ClaimState
        from config.simple_config import Config
        
        # Test claim creation
        claim = Claim(
            id="integration_test_001",
            content="Integration testing ensures components work together",
            confidence=0.95,
            type=[ClaimType.CONCEPT],
            tags=["testing", "integration", "quality"]
        )
        
        print(f"✅ Claim creation: {claim.id}")
        print(f"✅ Claim validation: {claim.confidence}")
        print(f"✅ Claim type: {claim.type[0].value}")
        
        # Test configuration
        config = Config()
        print(f"✅ Configuration loaded: {config.database_type}")
        
        # Test relationships
        claim.add_support("support_001")
        claim.add_supports("supported_001")
        print(f"✅ Relationship management: {len(claim.supported_by)} supports, {len(claim.supports)} supported")
        
        print("🎉 Integration test completed successfully!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        raise

# Run integration test
# integration_test_demo()
```

## 8. Real-world Application: Research Assistant

```python
class ResearchAssistant:
    """Practical research assistant using Conjecture"""
    
    def __init__(self):
        self.cf = EnhancedConjecture()
        self.research_topics = []
    
    async def research_topic(self, topic: str, max_claims: int = 10):
        """Research a topic and build knowledge base"""
        print(f"🔬 Researching: {topic}")
        
        # Explore topic
        result = await self.cf.explore(topic, max_claims=max_claims)
        
        # Store for later reference
        self.research_topics.append({
            "topic": topic,
            "claims": result.claims,
            "timestamp": result.timestamp
        })
        
        return result
    
    async def synthesize_findings(self, topic1: str, topic2: str):
        """Find connections between two research topics"""
        print(f"🔗 Synthesizing: {topic1} ↔ {topic2}")
        
        # Create synthesis claim
        synthesis = await self.cf.add_claim(
            content=f"Connections exist between {topic1} and {topic2} in their underlying principles",
            confidence=0.7,
            claim_type="thesis",
            tags=[topic1.replace(" ", "-"), topic2.replace(" ", "-"), "synthesis"]
        )
        
        return synthesis

# Example usage:
async def research_demo():
    """Demonstrate research assistant"""
    assistant = ResearchAssistant()
    
    # Research multiple topics
    ml_result = await assistant.research_topic("machine learning", 5)
    ai_result = await assistant.research_topic("artificial intelligence", 5)
    
    # Find connections
    synthesis = await assistant.synthesize_findings("machine learning", "artificial intelligence")
    
    print(f"📊 Research complete - ML claims: {len(ml_result.claims)}, AI claims: {len(ai_result.claims)}")
    print(f"🔗 Synthesis claim: {synthesis.id}")

# Run research demo
# asyncio.run(research_demo())
```

These examples demonstrate the core capabilities of Conjecture:
- Basic exploration and claim creation
- Building networks of related claims
- Complete research workflows
- Performance testing
- Tool validation
- Context building
- System integration
- Real-world applications

Each example can be run independently to understand how different components work together.