# Conjecture User Guide

Welcome to Conjecture, an evidence-based AI reasoning system that helps you explore knowledge, create claims, and build understanding systematically.

## Table of Contents
1. [What is Conjecture?](#what-is-conjecture)
2. [Key Concepts](#key-concepts)
3. [Getting Started](#getting-started)
4. [Basic Usage](#basic-usage)
5. [Advanced Features](#advanced-features)
6. [Performance](#performance)
7. [Troubleshooting](#troubleshooting)

## What is Conjecture?

Conjecture is like giving your AI assistant a "magic notebook" where it writes down everything it knows as structured "claims" with confidence scores and evidence. Instead of just remembering conversations, Conjecture builds a knowledge base of verified information.

### Key Benefits:
- **Evidence-Based**: Every claim is backed by evidence
- **Transparent**: You can see confidence levels and sources
- **Systematic**: Structured approach to knowledge building
- **Safe**: Built-in validation and security checks
- **Scalable**: Handles complex reasoning tasks

## Key Concepts

### Claims
Claims are the fundamental units of knowledge in Conjecture. Each claim has:
- **Content**: What is being claimed
- **Confidence**: How certain we are (0.0-1.0)
- **Type**: Concept, Reference, Thesis, Example, or Goal
- **Tags**: For categorization and search
- **Supporting Evidence**: Links to other claims or sources

### Claim Types
- **Concept**: Fundamental ideas or definitions
- **Reference**: Citations or sources
- **Thesis**: Analytical insights or arguments
- **Example**: Concrete illustrations
- **Goal**: Desired outcomes or objectives

### Claim States
- **Explore**: Needs further investigation
- **Validated**: Confirmed as accurate
- **Orphaned**: No supporting evidence
- **Queued**: Waiting for evaluation

## Getting Started

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/conjecture.git
cd conjecture

# Install dependencies
pip install -r requirements.txt
```

### Quick Start
```python
from src.enhanced_conjecture import EnhancedConjecture

# Initialize Conjecture
cf = EnhancedConjecture()

# Explore a topic
result = cf.explore("machine learning applications")
print(result.summary())

# Create a claim
claim = cf.add_claim(
    content="Machine learning requires substantial training data for optimal performance",
    confidence=0.85,
    claim_type="concept",
    tags=["ml", "data", "performance"]
)
print(f"Created claim: {claim.id}")
```

## Basic Usage

### 1. Exploring Topics
```python
# Basic exploration
result = cf.explore("quantum computing")
print(result.summary())

# With parameters
result = cf.explore(
    query="renewable energy technologies",
    max_claims=15,
    confidence_threshold=0.7
)
```

### 2. Creating Claims
```python
# Simple claim creation
claim = cf.add_claim(
    content="Solar panels convert sunlight to electricity",
    confidence=0.95,
    claim_type="concept"
)

# With relationships
claim = cf.add_claim(
    content="Solar energy is more sustainable than fossil fuels",
    confidence=0.8,
    claim_type="thesis",
    tags=["solar", "sustainability", "comparison"],
    auto_evaluate=True  # Automatically validate with LLM
)
```

### 3. Working with Claim Relationships
```python
# Add supporting relationships
claim1.add_supports(claim2.id)  # claim1 supports claim2
claim2.add_support(claim1.id)   # claim2 is supported by claim1

# Update confidence
claim.update_confidence(0.9)
```

## Advanced Features

### Async Claim Evaluation
Conjecture automatically evaluates claims in the background:
```python
# Check evaluation status
status = cf.get_evaluation_status(claim.id)
print(status)

# Wait for evaluation to complete
result = cf.wait_for_evaluation(claim.id, timeout=30)
```

### Dynamic Tool Creation
Conjecture can discover needs and create tools automatically:
```python
# The system automatically detects when new tools are needed
# and creates them with proper security validation
```

### Context Building
Intelligent context collection for better reasoning:
```python
# Context is automatically built from related claims
# to provide relevant background information
```

## Performance

Conjecture is highly optimized for performance:
- **Claim Creation**: ~30,000 claims/second
- **Relationship Operations**: ~250,000 ops/second
- **Concurrent Processing**: Multi-threaded operations
- **Memory Efficient**: ~1.5KB per claim average

### Performance Tips
1. Use batch operations when possible
2. Set appropriate confidence thresholds
3. Limit claim relationships to reduce complexity
4. Use concurrent operations for large datasets

## Troubleshooting

### Common Issues

**Import Errors**: Make sure you're running from the project root directory.

**LLM Connection Issues**: Check your configuration and API keys.

**Memory Issues**: Reduce batch sizes or use smaller claim sets.

### Getting Help
- Check the documentation
- Run the test suite: `python -m pytest tests/`
- Review performance benchmarks: `python tests/performance_benchmarks_final.py`

## Examples

### Research Assistant
```python
# Research a complex topic
cf = EnhancedConjecture()
result = cf.explore("climate change mitigation strategies", max_claims=20)

# Review findings
for claim in result.claims:
    print(f"[{claim.confidence:.2f}] {claim.content}")

# Add your own insights
cf.add_claim(
    content="Carbon pricing is an effective market-based climate solution",
    confidence=0.85,
    claim_type="thesis",
    tags=["climate", "policy", "economics"]
)
```

### Code Development Helper
```python
# Get coding assistance
result = cf.explore("python web scraping best practices")

# Create implementation claims
cf.add_claim(
    content="Use requests library with proper error handling for web scraping",
    confidence=0.9,
    claim_type="concept",
    tags=["python", "web", "scraping"]
)
```

### Knowledge Base Builder
```python
# Build a personal knowledge base
knowledge_base = EnhancedConjecture()

# Add your learning
knowledge_base.add_claim(
    content="Understanding neural networks requires knowledge of linear algebra",
    confidence=0.8,
    claim_type="concept",
    tags=["ml", "math", "learning"]
)

# Review statistics
stats = knowledge_base.get_statistics()
print(f"Knowledge base contains {stats['claims_processed']} claims")
```

## Next Steps

1. **Explore the API**: Try different exploration parameters
2. **Build Relationships**: Create networks of connected claims
3. **Monitor Performance**: Run the benchmark suite
4. **Extend Functionality**: Add custom tools and skills
5. **Contribute**: Help improve the system

Conjecture is designed to grow with your needs. Start simple and gradually explore more advanced features as you become comfortable with the system.