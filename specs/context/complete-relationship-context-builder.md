# Complete Relationship Context Builder Specification

**Last Updated:** November 11, 2025  
**Version:** 1.0  
**Author:** Design Documentation Writer

## Overview

The Complete Relationship Context Builder is responsible for creating comprehensive LLM contexts that include all support relationships and semantic connections for any given target claim. This specification defines the algorithm, data structures, performance considerations, and optimization strategies for building optimal contexts that enable LLMs to identify instructions and create support relationships effectively.

## Core Algorithm

### Context Building Workflow

```
build_complete_context(target_claim_id: str, max_tokens: int = 8000) -> str:
    """
    Build comprehensive context with complete relationship coverage
    
    Args:
        target_claim_id: ID of the claim to build context for
        max_tokens: Maximum tokens for the entire context
        
    Returns:
        Formatted context string optimized for LLM processing
    """
    
    # Phase 1: Initialize and validate
    target_claim = get_claim(target_claim_id)
    if not target_claim:
        raise ValueError(f"Claim {target_claim_id} not found")
    
    # Phase 2: Gather relationship claims (priority 1: guaranteed relevance)
    upward_chain = traverse_upward_to_root(target_claim_id)
    downward_claims = traverse_downward_all_descendants(target_claim_id)
    
    # Phase 3: Add semantic claims (priority 2: contextual relevance)
    semantic_claims = get_semantic_similar_claims(
        target_claim, 
        remaining_token_budget(upward_chain + downward_claims, max_tokens)
    )
    
    # Phase 4: Format and optimize for LLM
    return format_context_for_llm(
        target_claim,
        upward_chain,
        downward_claims,
        semantic_claims
    )
```

### Detailed Algorithm Steps

#### Step 1: Target Claim Validation
```python
def validate_target_claim(claim_id: str) -> Claim:
    """Validate target claim exists and is accessible"""
    claim = data_manager.get_claim(claim_id)
    if not claim:
        raise ClaimNotFoundError(f"Claim {claim_id} not found")
    if claim.state == ClaimState.ORPHANED:
        logger.warning(f"Building context for orphaned claim: {claim_id}")
    return claim
```

#### Step 2: Upward Support Chain Traversal
```python
def traverse_upward_to_root(target_claim_id: str) -> List[Claim]:
    """
    Get all claims in the support chain from target to root
    
    Returns claims in order from target upwards (closest to target first)
    """
    visited = set()
    chain = []
    queue = [(target_claim_id, 0)]  # (claim_id, depth)
    
    while queue:
        claim_id, depth = queue.pop(0)
        
        if claim_id in visited or depth > MAX_UPWARD_DEPTH:
            continue
            
        visited.add(claim_id)
        claim = data_manager.get_claim(claim_id)
        if not claim:
            continue
            
        chain.append(claim)
        
        # Add supporting claims to queue (breadth-first traversal)
        for supporting_id in claim.supported_by:
            if supporting_id not in visited:
                queue.append((supporting_id, depth + 1))
    
    # Remove target claim (will be added separately)
    return [c for c in chain if c.id != target_claim_id]
```

#### Step 3: Downward Descendant Traversal
```python
def traverse_downward_all_descendants(target_claim_id: str) -> List[Claim]:
    """
    Get all claims directly or indirectly supported by target
    
    Returns claims in hierarchy order (direct children first)
    """
    visited = set()
    descendants = []
    queue = [(target_claim_id, 0)]  # (claim_id, depth)
    
    while queue:
        claim_id, depth = queue.pop(0)
        
        if claim_id in visited or depth > MAX_DOWNWARD_DEPTH:
            continue
            
        visited.add(claim_id)
        claim = data_manager.get_claim(claim_id)
        if not claim:
            continue
            
        descendants.append(claim)
        
        # Add supported claims to queue
        for supported_id in claim.supports:
            if supported_id not in visited:
                queue.append((supported_id, depth + 1))
    
    # Remove target claim (will be added separately)
    return [c for c in descendants if c.id != target_claim_id]
```

#### Step 4: Semantic Similarity Selection
```python
def get_semantic_similar_claims(
    target_claim: Claim, 
    token_budget: int
) -> List[Claim]:
    """
    Get semantically similar claims within token budget
    
    Uses vector similarity and excludes already included claims
    """
    if token_budget <= 0:
        return []
    
    # Get existing claim IDs to exclude
    existing_ids = {target_claim.id}
    existing_ids.update(c.id for c in upward_chain)
    existing_ids.update(c.id for c in downward_claims)
    
    # Search for similar claims using embedding similarity
    similar_claims = data_manager.search_similar_claims(
        target_claim.embedding,
        exclude_ids=existing_ids,
        limit=100  # Get more than needed for filtering
    )
    
    # Select claims based on token budget and relevance
    selected_claims = []
    used_tokens = 0
    
    for claim in similar_claims:
        claim_tokens = estimate_token_count(claim.content)
        if used_tokens + claim_tokens <= token_budget:
            selected_claims.append(claim)
            used_tokens += claim_tokens
        else:
            break
    
    return selected_claims
```

## Context Formatting Strategy

### Section Organization

The context is organized into clear sections for optimal LLM processing:

```python
def format_context_for_llm(
    target_claim: Claim,
    upward_chain: List[Claim],
    downward_claims: List[Claim],
    semantic_claims: List[Claim]
) -> str:
    """Format all claims into structured LLM context"""
    
    context_parts = [
        format_header("TARGET CLAIM"),
        format_target_claim(target_claim),
        format_header("SUPPORTING CHAIN (to root)"),
        format_upward_chain(upward_chain),
        format_header("SUPPORTED CLAIMS (all descendants)"),
        format_downward_claims(downward_claims),
        format_header("SEMANTIC CONTEXT"),
        format_semantic_claims(semantic_claims),
        format_header("CONTEXT SUMMARY"),
        format_context_summary(target_claim, upward_chain, downward_claims, semantic_claims)
    ]
    
    return "\n\n".join(filter(None, context_parts))
```

### Claim Formatting Standards

```python
def format_claim_with_metadata(claim: Claim, relationship_info: str = "") -> str:
    """Standard claim formatting for LLM context"""
    
    claim_lines = [
        f"[{claim.id}] {claim.content}",
        f"  Confidence: {claim.confidence:.2f}, State: {claim.state.value}",
        f"  Type: {', '.join(t.value for t in claim.type)}",
        f"  Tags: {', '.join(claim.tags) if claim.tags else 'none'}",
    ]
    
    if relationship_info:
        claim_lines.append(f"  Relationship: {relationship_info}")
    
    # Add support chain indicators for clarity
    if claim.supported_by:
        claim_lines.append(f"  Supported by: {', '.join(claim.supported_by)}")
    if claim.supports:
        claim_lines.append(f"  Supports: {', '.join(claim.supports)}")
    
    return "\n".join(claim_lines)
```

### Relationship Visualization

```python
def format_upward_chain(upward_chain: List[Claim]) -> str:
    """Format upward support chain with hierarchy visualization"""
    
    if not upward_chain:
        return "No supporting claims found."
    
    sections = []
    for i, claim in enumerate(upward_chain):
        # Calculate indentation based on position
        indent = "  " * i
        relationship = "supports → TARGET" if i == 0 else f"supports → {upward_chain[i-1].id}"
        
        sections.append(
            f"{indent}{format_claim_with_metadata(claim, relationship)}"
        )
    
    return "\n\n".join(sections)

def format_downward_claims(downward_claims: List[Claim]) -> str:
    """Format downward claims with dependency visualization"""
    
    if not downward_claims:
        return "No supported claims found."
    
    # Group by depth for better visualization
    depth_groups = group_by_depth(downward_claims)
    
    sections = []
    for depth, claims in depth_groups.items():
        if depth == 0:
            sections.append("=== Directly Supported Claims ===")
        else:
            sections.append(f"=== Claims at Depth {depth} ===")
        
        for claim in claims:
            indent = "  " * depth
            relationship = f"supported by → {'parent_claim' if depth == 0 else 'ancestor'}"
            sections.append(f"{indent}{format_claim_with_metadata(claim, relationship)}")
    
    return "\n\n".join(sections)
```

## Token Management Strategy

### Token Allocation Algorithm

```python
class TokenBudgetManager:
    """Manages token allocation for different context sections"""
    
    DEFAULT_ALLOCATIONS = {
        "target": 100,           # Fixed allocation for target claim
        "upward": 0.40,          # 40% for supporting chain
        "downward": 0.30,        # 30% for supported claims
        "semantic": 0.30,        # 30% for semantic context
        "overhead": 200          # Headers and formatting
    }
    
    def __init__(self, max_tokens: int = 8000):
        self.max_tokens = max_tokens
        self.allocations = self.calculate_allocations()
    
    def calculate_allocations(self) -> Dict[str, int]:
        """Calculate token allocations for each section"""
        available = self.max_tokens - self.DEFAULT_ALLOCATIONS["overhead"]
        
        return {
            "target": self.DEFAULT_ALLOCATIONS["target"],
            "upward": int(available * self.DEFAULT_ALLOCATIONS["upward"]),
            "downward": int(available * self.DEFAULT_ALLOCATIONS["downward"]),
            "semantic": int(available * self.DEFAULT_ALLOCATIONS["semantic"])
        }
    
    def check_section_tokens(self, section: str, claims: List[Claim]) -> int:
        """Calculate tokens needed for a section and warn if over budget"""
        needed = sum(estimate_token_count(claim.content) for claim in claims)
        budget = self.allocations[section]
        
        if needed > budget:
            logger.warning(f"{section} section exceeds budget: {needed} > {budget}")
        
        return needed
```

### Dynamic Token Adjustment

```python
def adjust_token_allocations(
    upward_needed: int,
    downward_needed: int,
    total_available: int
) -> Dict[str, int]:
    """
    Dynamically adjust token allocations based on actual needs
    
    Prioritizes relationship claims over semantic claims
    """
    relationship_total = upward_needed + downward_needed
    
    if relationship_total <= total_available * 0.7:
        # Relationships fit easily, give semantic more space
        upward_min = upward_needed
        downward_min = downward_needed
        semantic_budget = total_available - relationship_total - 300  # overhead
    else:
        # Relationships need more space, reduce semantic
        scale_factor = (total_available * 0.7) / relationship_total
        upward_min = int(upward_needed * scale_factor)
        downward_min = int(downward_needed * scale_factor)
        semantic_budget = total_available * 0.2  # Reduced semantic budget
    
    return {
        "upward": upward_min,
        "downward": downward_min,
        "semantic": int(semantic_budget)
    }
```

## Performance Considerations

### Caching Strategy

```python
class ContextCache:
    """Intelligent caching for frequently accessed contexts"""
    
    def __init__(self, cache_size: int = 1000):
        self.cache = {}
        self.cache_size = cache_size
        self.access_times = {}
    
    def get_context(self, claim_id: str, snapshot_time: datetime) -> Optional[str]:
        """Get cached context if still valid"""
        cache_key = f"{claim_id}:{snapshot_time.timestamp()}"
        
        if cache_key in self.cache:
            self.access_times[cache_key] = datetime.utcnow()
            return self.cache[cache_key]
        
        return None
    
    def cache_context(self, claim_id: str, snapshot_time: datetime, context: str):
        """Cache context with LRU eviction"""
        cache_key = f"{claim_id}:{snapshot_time.timestamp()}"
        
        # Evict oldest if cache is full
        if len(self.cache) >= self.cache_size:
            oldest_key = min(self.access_times.keys(), 
                           key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[cache_key] = context
        self.access_times[cache_key] = datetime.utcnow()
```

### Database Optimization

```python
# Optimized database queries for relationship traversal
UPWARD_CHAIN_QUERY = """
WITH RECURSIVE support_chain AS (
    SELECT id, content, confidence, supported_by, supports, 0 as depth
    FROM claims WHERE id = ?
    UNION ALL
    SELECT c.id, c.content, c.confidence, c.supported_by, c.supports, sc.depth + 1
    FROM claims c
    JOIN support_chain sc ON c.id = ANY(sc.supported_by)
    WHERE sc.depth < ? AND c.id NOT IN (SELECT id FROM support_chain)
)
SELECT * FROM support_chain ORDER BY depth;
"""

DOWNWARD_TRAVERSAL_QUERY = """
WITH RECURSIVE descendants AS (
    SELECT id, content, confidence, supported_by, supports, 0 as depth
    FROM claims WHERE id = ?
    UNION ALL
    SELECT c.id, c.content, c.confidence, c.supported_by, c.supports, d.depth + 1
    FROM claims c
    JOIN descendants d ON c.id = ANY(d.supports)
    WHERE d.depth < ? AND c.id NOT IN (SELECT id FROM descendants)
)
SELECT * FROM descendants ORDER BY depth;
"""
```

### Async Processing for Large Contexts

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def build_large_context_async(
    target_claim_id: str,
    max_tokens: int = 16000
) -> str:
    """Build context asynchronously for large claim networks"""
    
    # Parallel processing of different sections
    loop = asyncio.get_event_loop()
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Run traversals in parallel
        upward_task = loop.run_in_executor(
            executor, traverse_upward_to_root, target_claim_id
        )
        downward_task = loop.run_in_executor(
            executor, traverse_downward_all_descendants, target_claim_id
        )
        
        # Get target claim while traversals run
        target_claim = await loop.run_in_executor(
            executor, data_manager.get_claim, target_claim_id
        )
        
        # Wait for traversals to complete
        upward_chain, downward_claims = await asyncio.gather(
            upward_task, downward_task
        )
    
    # Process semantic claims (can be async with vector search)
    semantic_claims = await get_semantic_similar_claims_async(
        target_claim, max_tokens
    )
    
    # Format and return context
    return await loop.run_in_executor(
        executor, 
        format_context_for_llm,
        target_claim, upward_chain, downward_claims, semantic_claims
    )
```

## Quality Assurance and Validation

### Context Completeness Validation

```python
def validate_context_completeness(context_data: Dict) -> List[str]:
    """Validate that context includes all necessary information"""
    
    errors = []
    
    # Check target claim presence
    if not context_data.get("target_claim"):
        errors.append("Target claim missing from context")
    
    # Check relationship coverage
    if not context_data.get("upward_chain"):
        errors.append("No supporting claims found - context may be incomplete")
    
    # Check for circular references
    if detect_circular_dependencies(context_data):
        errors.append("Circular dependencies detected in support chain")
    
    # Check claim consistency
    if not validate_claim_consistency(context_data):
        errors.append("Inconsistent claim data found")
    
    return errors

def validate_claim_consistency(context_data: Dict) -> bool:
    """Validate that all referenced claims exist and relationships are bidirectional"""
    
    all_claims = {}
    
    # Collect all claims from different sections
    for section in ["target_claim", "upward_chain", "downward_claims", "semantic_claims"]:
        claims = context_data.get(section, [])
        if isinstance(claims, list):
            for claim in claims:
                all_claims[claim.id] = claim
        else:
            # Single claim (target)
            all_claims[claims.id] = claims
    
    # Validate all relationships point to existing claims
    for claim in all_claims.values():
        for supported_id in claim.supported_by:
            if supported_id not in all_claims:
                return False
        
        for supported_id in claim.supports:
            if supported_id not in all_claims:
                return False
    
    return True
```

### Performance Monitoring

```python
class ContextBuilderMetrics:
    """Monitor context building performance and quality"""
    
    def __init__(self):
        self.metrics = {
            "contexts_built": 0,
            "total_claims_processed": 0,
            "average_build_time": 0.0,
            "cache_hit_rate": 0.0,
            "token_utilization": 0.0,
            "relationship_coverage": 0.0
        }
    
    def record_build(
        self,
        build_time: float,
        claim_count: int,
        tokens_used: int,
        tokens_total: int,
        relationship_count: int,
        cache_hit: bool
    ):
        """Record metrics for a context build"""
        
        self.metrics["contexts_built"] += 1
        self.metrics["total_claims_processed"] += claim_count
        
        # Update average build time
        current_avg = self.metrics["average_build_time"]
        n = self.metrics["contexts_built"]
        self.metrics["average_build_time"] = (current_avg * (n-1) + build_time) / n
        
        # Update cache hit rate
        if cache_hit:
            self.metrics["cache_hit_rate"] = (
                (self.metrics["cache_hit_rate"] * (n-1) + 1.0) / n
            )
        else:
            self.metrics["cache_hit_rate"] = (
                (self.metrics["cache_hit_rate"] * (n-1) + 0.0) / n
            )
        
        # Update token utilization
        self.metrics["token_utilization"] = tokens_used / tokens_total
        
        # Update relationship coverage (assuming max 1000 claims in full network)
        self.metrics["relationship_coverage"] = min(relationship_count / 1000, 1.0)
```

## Error Handling and Edge Cases

### Common Error Scenarios

```python
class ContextBuilderError(Exception):
    """Base exception for context building errors"""
    pass

class ClaimNotFound(ContextBuilderError):
    """Target claim not found"""
    pass

class CircularDependencyDetected(ContextBuilderError):
    """Circular dependency in support chain"""
    pass

class TokenBudgetExceeded(ContextBuilderError):
    """Context requires more tokens than available"""
    pass

def handle_context_errors(func):
    """Decorator for graceful error handling in context building"""
    
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ClaimNotFound as e:
            logger.error(f"Claim not found: {e}")
            return build_empty_context_with_error(str(e))
        except CircularDependencyDetected as e:
            logger.warning(f"Circular dependency detected: {e}")
            return build_context_without_relationships(args[0])
        except TokenBudgetExceeded as e:
            logger.warning(f"Token budget exceeded: {e}")
            return build_minimal_context(args[0], max_tokens=4000)
        except Exception as e:
            logger.error(f"Unexpected error in context building: {e}")
            return build_emergency_context(args[0])
    
    return wrapper
```

## Testing Strategy

### Unit Tests for Core Algorithms

```python
class TestContextBuilder:
    """Comprehensive test suite for context building"""
    
    def test_upward_traversal_simple_chain(self):
        """Test upward traversal with simple linear support chain"""
        # Setup: A <- B <- C (C is target)
        # Expected: Should return B, then A
        
    def test_upward_traversalbranched_support(self):
        """Test upward traversal with multiple supporting claims"""
        # Setup: A, B <- C <- D (D is target)
        # Expected: Should return C, then A and B
        
    def test_downward_traversal_multiple_levels(self):
        """Test downward traversal with nested supported claims"""
        # Setup: A -> B -> C, D (A is target)
        # Expected: Should return B, then C and D
        
    def test_token_budget_enforcement(self):
        """Test that context respects token limits"""
        # Setup: Large claim network
        # Expected: Should truncate intelligently
        
    def test_semantic_claim_selection(self):
        """Test semantic claim selection and exclusion"""
        # Setup: Target claim and similar claims
        # Expected: Should exclude already included claims
        
    def test_circular_dependency_detection(self):
        """Test detection and handling of circular dependencies"""
        # Setup: A -> B -> A (circular)
        # Expected: Should detect and handle gracefully
```

### Integration Tests with Real Data

```python
class TestContextBuilderIntegration:
    """Integration tests with real claim data"""
    
    def test_large_claim_network(self):
        """Test performance with 1000+ claim network"""
        
    def test_frequent_access_patterns(self):
        """Test caching effectiveness with common access patterns"""
        
    def test_concurrent_context_building(self):
        """Test thread safety and performance with concurrent requests"""
        
    def test_llm_context_compatibility(self):
        """Test that generated context works effectively with LLMs"""
```

## Implementation Guidelines

### Configuration Management

```python
class ContextBuilderConfig:
    """Configuration for context building behavior"""
    
    # Traversal limits
    MAX_UPWARD_DEPTH = 10
    MAX_DOWNWARD_DEPTH = 8
    
    # Token allocation
    DEFAULT_MAX_TOKENS = 8000
    MIN_TOKENS_PER_SECTION = 50
    
    # Semantic similarity
    SIMILARITY_THRESHOLD = 0.7
    MAX_SEMANTIC_CLAIMS = 20
    
    # Caching
    ENABLE_CACHING = True
    CACHE_SIZE = 1000
    CACHE_TTL_MINUTES = 30
    
    # Performance
    ASYNC_THRESHOLD = 50  # Number of claims to trigger async processing
    TIMEOUT_SECONDS = 30
    
    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        config = cls()
        config.DEFAULT_MAX_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", 8000))
        config.ENABLE_CACHING = os.getenv("ENABLE_CONTEXT_CACHE", "true").lower() == "true"
        return config
```

### Logging and Monitoring

```python
import structlog

logger = structlog.get_logger()

def log_context_build_start(target_claim_id: str, max_tokens: int):
    logger.info(
        "context_build_started",
        target_claim_id=target_claim_id,
        max_tokens=max_tokens
    )

def log_context_build_complete(
    target_claim_id: str,
    upward_count: int,
    downward_count: int,
    semantic_count: int,
    tokens_used: int,
    build_time: float
):
    logger.info(
        "context_build_completed",
        target_claim_id=target_claim_id,
        upward_claims=upward_count,
        downward_claims=downward_count,
        semantic_claims=semantic_count,
        tokens_used=tokens_used,
        build_time_seconds=build_time
    )
```

The Complete Relationship Context Builder provides the foundation for successful LLM-driven instruction identification and support relationship creation by ensuring comprehensive, optimized context delivery.