# Conjecture Design Specification

## Conjecture Architecture Overview

### System Architecture Diagram
```
┌─────────────────────────────────────────────────────────────────┐
│                        Conjecture System                        │
├─────────────────────────────────────────────────────────────────┤
│  Interface Layer                                               │
│  ┌──────────┬──────────┬──────────┬──────────────────────────┐ │
│  │   TUI    │   CLI    │   MCP    │         WebUI            │ │
│  └──────────┴──────────┴──────────┴──────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  Application Layer                                             │
│  ┌─────────────────────┬─────────────────────┬───────────────┐ │
│  │   Claim Processor   │  Evidence Manager   │ Goal Tracker │ │
│  └─────────────────────┴─────────────────────┴───────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  Domain Layer                                                  │
│  ┌───────────────┬───────────────┬───────────────┬───────────┐ │
│  │    Claim      │   Evidence    │   Vector      │  Query   │ │
│  │  Management   │  Management   │ Similarity    │ Engine   │ │
│  └───────────────┴───────────────┴───────────────┴───────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  Infrastructure Layer                                          │
│  ┌───────────────┬───────────────┬───────────────┬───────────┐ │
│  │   Database    │   Storage     │   Embedding   │   Auth    │ │
│  │   Manager     │   Manager     │   Service     │ Manager  │ │
│  └───────────────┴───────────────┴───────────────┴───────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  External Services                                             │
│  ┌───────────────┬───────────────┬───────────────┬───────────┐ │
│  │   Vector DB   │  LLM Service  │   External    │   File    │ │
│  │               │               │   Sources     │  System   │ │
│  └───────────────┴───────────────┴───────────────┴───────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Architectural Principles

1. **Separation of Concerns**
   - Clear boundaries between UI, application, domain, and infrastructure
   - Interface isolation for different user interaction patterns
   - Domain logic independent of technical implementation

2. **Multi-Modal Interface Design**
   - Shared business logic across all interfaces
   - Interface-specific adapters for different interaction patterns
   - Consistent data models regardless of access method

3. **Scalable Knowledge Representation**
   - Graph-based relationship modeling
   - Vector similarity for semantic connections
   - Efficient storage and retrieval of complex knowledge structures

4. **Extensibility and Integration**
   - Plugin architecture for new interfaces
   - API-first approach for external integrations
   - Configurable components for different deployment scenarios

## Core Components

### Claim Manager
Manages the lifecycle of claims, including creation, modification, validation, and deletion.

### Skill-Based Agency System
Enables LLMs to perform specific actions through skill claims and example claims:
- **Skill Claims** (`type.skill`): Instruct LLM how to perform specific actions
- **Example Claims** (`type.example`): Show proper tool response formatting
- **Tool Call Reflection**: Automatic creation of example claims from successful tool executions
- **Skill Injection**: During evaluation, not just session initialization
- **LLM Response Parsing**: XML-like structured response parsing for Python function execution

### Enhanced Session Manager
Manages multi-session evaluation with adaptive context window management:
- **Session Isolation**: Separate contexts with shared persistent database
- **Adaptive Context Window**: Dynamic sizing based on 30% model token limit
- **Claim Selection Algorithm**: Heap sorted by similarity to root claim with confidence boosting
- **Minimum Functional Claims**: 4 skills, 3 concepts, 3 principles per session
- **Progressive Claim Addition**: incremental context building based on token availability
- **Fresh Context Building**: Per evaluation cycle, not cached sessions

### Trustworthiness Validation System
Manages source validation and trust assessment:
- **Web Content Author Trustworthiness**: Author credibility claims and validation chains
- **Multi-Level Source Validation**: Hierarchical trust assessment across source dependencies
- **Persistent Trust Claims**: One-time validation with monthly confidence decay instead of trust decay
- **Trust Propagation**: Confidence inheritance through validated source chains

### Contradiction Detection and Merging Engine
Handles claim conflicts and intelligent merging:
- **Confidence-Based Merging**: Preserve higher confidence claim content
- **Union of Support Relationships**: Merge supporting evidence during claim combination
- **Contradiction Detection**: Identify opposing claims through semantic analysis
- **Mark Dirty for Re-evaluation**: Automatic reprocessing after merging operations
- **Heritage Chain Preservation**: Maintain claim lineage through merge operations

### Relationship Manager
Processes relationships between claims, tracking support, contradiction, and dependency connections.

### Vector Similarity Engine
Provides semantic search capabilities through embedding-based similarity matching.

### Query Engine
Handles complex queries across the knowledge graph, combining structural and semantic search.

### Confidence Manager
Manages confidence scoring, inheritance, and aggregation for claims and their relationships.

## Enhanced Session Flow Architecture

### Session Management Overview
Conjecture implements a sophisticated session management system that isolates evaluation contexts while maintaining shared persistent storage. Each session operates independently with adaptive context window management and intelligent claim selection algorithms.

### Multi-Session Isolation Architecture

```python
class SessionManager:
    def __init__(self, persistent_db: DatabaseManager):
        self.persistent_db = persistent_db
        self.active_sessions: Dict[str, Session] = {}
        self.shared_databases: Dict[str, DatabaseManager] = {}
        
    async def create_session(self, session_config: SessionConfig) -> str:
        """Create new isolated session with shared persistent database"""
        session_id = generate_session_id()
        
        # Configure database instance for this session
        db_instance = await self._configure_session_db(session_config)
        self.shared_databases[session_id] = db_instance
        
        # Create session with adaptive context management
        session = Session(
            session_id=session_id,
            db_instance=db_instance,
            context_window=AdaptiveContextWindow(),
            claim_selector=ClaimSelectionHeap(),
            skill_injector=SkillInjector(db_instance)
        )
        
        self.active_sessions[session_id] = session
        return session_id
    
    async def get_session(self, session_id: str) -> Session:
        """Retrieve active session or recreate from persistent state"""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Recreate session from persistent storage
        return await self._recreate_session(session_id)
```

### Adaptive Context Window Management

```python
class AdaptiveContextWindow:
    def __init__(self, model_token_limit: int):
        self.model_token_limit = model_token_limit
        self.max_context_size = int(model_token_limit * 0.3)  # 30% rule
        self.current_context: List[Claim] = []
        self.skill_claims: List[Claim] = []
        self.concept_claims: List[Claim] = []
        self.principle_claims: List[Claim] = []
    
    async def build_context(self, root_claim_id: str, db_manager: DatabaseManager) -> List[Claim]:
        """Build adaptive context with minimum functional claims"""
        
        # Start with minimum functional claims
        await self._ensure_minimum_functional_claims(root_claim_id, db_manager)
        
        # Build context progressively
        context = []
        token_count = 0
        
        # 1. Add skill claims (highest priority)
        for skill_claim in self.skill_claims:
            claim_tokens = estimate_tokens(skill_claim.content)
            if token_count + claim_tokens <= self.max_context_size:
                context.append(skill_claim)
                token_count += claim_tokens
        
        # 2. Add concept claims
        for concept_claim in self.concept_claims:
            claim_tokens = estimate_tokens(concept_claim.content)
            if token_count + claim_tokens <= self.max_context_size:
                context.append(concept_claim)
                token_count += claim_tokens
        
        # 3. Add principle claims
        for principle_claim in self.principle_claims:
            claim_tokens = estimate_tokens(principle_claim.content)
            if token_count + claim_tokens <= self.max_context_size:
                context.append(principle_claim)
                token_count += claim_tokens
        
        # 4. Add relevant claims based on similarity
        additional_claims = await self._select_additional_claims(
            root_claim_id, token_count, db_manager
        )
        context.extend(additional_claims)
        
        self.current_context = context
        return context
    
    async def _ensure_minimum_functional_claims(self, root_claim_id: str, db_manager: DatabaseManager):
        """Ensure minimum functional claims: 4 skills, 3 concepts, 3 principles"""
        
        # Try to get from database first
        self.skill_claims = await db_manager.get_claims_by_tag("skill", limit=4)
        self.concept_claims = await db_manager.get_claims_by_tag("concept", limit=3)
        self.principle_claims = await db_manager.get_claims_by_tag("principle", limit=3)
        
        # If insufficient, create default functional claims
        if len(self.skill_claims) < 4:
            await self._create_default_skill_claims(db_manager)
        if len(self.concept_claims) < 3:
            await self._create_default_concept_claims(db_manager)
        if len(self.principle_claims) < 3:
            await self._create_default_principle_claims(db_manager)
```

### Claim Selection Heap Algorithm

```python
class ClaimSelectionHeap:
    def __init__(self):
        self.heap: List[Tuple[float, Claim]] = []
        self.root_embedding: Optional[List[float]] = None
    
    async def get_next_evaluation_batch(self, root_claim_id: str, db_manager: DatabaseManager, batch_size: int = 4) -> List[Claim]:
        """Get top N dirty claims by relevance to root claim with confidence boost"""
        
        # Get root claim embedding if not already cached
        if not self.root_embedding:
            root_claim = await db_manager.get_claim(root_claim_id)
            self.root_embedding = root_claim.embedding
        
        # Query dirty claims with adjusted similarity (confidence boost for low confidence)
        query = """
        SELECT c.*, 
               (c.embedding <-> $root_embedding) - 
               (CASE WHEN c.confidence < 0.90 THEN 0.30 ELSE 0 END) as adjusted_similarity
        FROM claims c 
        WHERE c.dirty = true 
        ORDER BY adjusted_similarity ASC 
        LIMIT $batch_size
        """
        
        claims = await db_manager.execute_query(query, {
            'root_embedding': self.root_embedding,
            'batch_size': batch_size
        })
        
        # Build heap for maintaining priority order
        for claim in claims:
            score = self._calculate_priority_score(claim, root_claim_id)
            heapq.heappush(self.heap, (-score, claim))  # Negative for max heap
        
        # Extract top claims
        selected_claims = []
        for _ in range(min(batch_size, len(self.heap))):
            _, claim = heapq.heappop(self.heap)
            selected_claims.append(claim)
        
        return selected_claims
    
    def _calculate_priority_score(self, claim: Claim, root_claim_id: str) -> float:
        """Calculate priority score based on similarity and confidence"""
        similarity = 1 - cosine_similarity(claim.embedding, self.root_embedding)
        
        # Confidence boost: lower confidence gets higher priority
        confidence_boost = 1.0 - claim.confidence
        
        # Weighted combination
        priority_score = (0.7 * similarity) + (0.3 * confidence_boost)
        
        return priority_score
```

### Skill Injection During Evaluation

```python
class SkillInjector:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.skill_cache: Dict[str, Claim] = {}
        
    async def inject_relevant_skills(self, evaluation_context: Dict[str, Any]) -> List[Claim]:
        """Inject relevant skill claims during evaluation, not just session init"""
        
        root_claim_id = evaluation_context.get('root_claim_id')
        current_context = evaluation_context.get('current_context', [])
        
        if not root_claim_id:
            return []
        
        # Find skill claims relevant to current evaluation
        relevant_skills = await self._find_relevant_skills(root_claim_id, current_context)
        
        # Check if skill examples are available
        enhanced_skills = []
        for skill in relevant_skills:
            example_claims = await self.db_manager.get_claims_by_tags(
                ["example", skill.id], limit=2
            )
            
            if example_claims:
                # Add skill with examples
                enhanced_skills.extend([skill] + example_claims)
            else:
                # Add skill and schedule example creation
                enhanced_skills.append(skill)
                await self._schedule_example_creation(skill)
        
        return enhanced_skills
    
    async def _find_relevant_skills(self, root_claim_id: str, current_context: List[Claim]) -> List[Claim]:
        """Find skills relevant to the current evaluation task"""
        
        # Get root claim to understand evaluation context
        root_claim = await self.db_manager.get_claim(root_claim_id)
        if not root_claim:
            return []
        
        # Search for skills by semantic similarity
        all_skill_claims = await self.db_manager.get_claims_by_tag("skill")
        
        # Calculate relevance scores
        relevant_skills = []
        for skill in all_skill_claims:
            relevance = self._calculate_skill_relevance(skill, root_claim, current_context)
            if relevance > 0.5:  # Threshold for relevance
                relevant_skills.append((relevance, skill))
        
        # Sort by relevance
        relevant_skills.sort(key=lambda x: x[0], reverse=True)
        
        return [skill for _, skill in relevant_skills[:5]]  # Top 5 relevant skills
    
    def _calculate_skill_relevance(self, skill: Claim, root_claim: Claim, context: List[Claim]) -> float:
        """Calculate how relevant a skill is to the current evaluation"""
        
        # Semantic similarity to root claim
        root_similarity = 1 - cosine_similarity(skill.embedding, root_claim.embedding)
        
        # Semantic similarity to context claims
        context_similarity = 0
        if context:
            context_embeddings = [claim.embedding for claim in context]
            avg_similarity = sum(
                1 - cosine_similarity(skill.embedding, ctx_emb) 
                for ctx_emb in context_embeddings
            ) / len(context_embeddings)
            context_similarity = avg_similarity
        
        # Check skill content for keywords
        skill_content_lower = skill.content.lower()
        root_content_lower = root_claim.content.lower()
        
        keyword_match = 0
        if any(word in skill_content_lower for word in root_content_lower.split()):
            keyword_match = 0.2
        
        # Weighted combination
        relevance = (0.5 * root_similarity) + (0.3 * context_similarity) + keyword_match
        
        return min(1.0, relevance)
```

### Session Evaluation Loop

```python
class Session:
    def __init__(self, session_id: str, db_instance: DatabaseManager, 
                 context_window: AdaptiveContextWindow, claim_selector: ClaimSelectionHeap,
                 skill_injector: SkillInjector):
        self.session_id = session_id
        self.db_instance = db_instance
        self.context_window = context_window
        self.claim_selector = claim_selector
        self.skill_injector = skill_injector
        self.evaluation_count = 0
        
    async def evaluate_root_claim(self, root_claim_id: str, max_iterations: int = 50):
        """Complete evaluation loop with enhanced session flow"""
        
        iteration = 0
        while iteration < max_iterations:
            # Build fresh context each iteration
            context_claims = await self.context_window.build_context(root_claim_id, self.db_instance)
            
            # Get batch of most relevant dirty claims
            batch = await self.claim_selector.get_next_evaluation_batch(
                root_claim_id, self.db_instance, batch_size=4
            )
            
            if not batch:
                print("Evaluation complete - no more dirty claims")
                break
            
            # Inject relevant skills for this batch
            evaluation_context = {
                'root_claim_id': root_claim_id,
                'current_context': context_claims,
                'batch_claims': batch
            }
            relevant_skills = await self.skill_injector.inject_relevant_skills(evaluation_context)
            
            # Combine context with skills (skills get priority)
            full_context = relevant_skills + context_claims
            
            # Process batch with enhanced context
            results = await self._process_evaluation_batch(batch, full_context)
            
            # Check completion condition
            root_claim = await self.db_instance.get_claim(root_claim_id)
            print(f"Iteration {iteration}: Root confidence = {root_claim.confidence:.2f}")
            
            if root_claim.confidence >= 0.9:
                print("Evaluation complete - root claim confident")
                break
                
            iteration += 1
            self.evaluation_count += 1
    
    async def _process_evaluation_batch(self, batch: List[Claim], context: List[Claim]) -> List[Dict]:
        """Process batch of claims with context and skill injection"""
        
        tasks = []
        for claim in batch:
            task = self._evaluate_single_claim(claim, context)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle any exceptions
        successful_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error evaluating claim: {result}")
            else:
                successful_results.append(result)
        
        return successful_results
```

### Multi-Session Deployment Architecture

```python
class MultiSessionDeployment:
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.session_containers: Dict[str, DockerContainer] = {}
        self.session_databases: Dict[str, DatabaseManager] = {}
        
    async def deploy_isolated_session(self, session_config: SessionConfig) -> str:
        """Deploy session in isolated Docker environment"""
        
        session_id = session_config.session_id
        
        # Create Docker container for session isolation
        container_config = {
            'image': 'conjecture:latest',
            'environment': {
                'SESSION_ID': session_id,
                'DB_CONFIG': json.dumps(session_config.database_config),
                'MODEL_CONFIG': json.dumps(session_config.model_config)
            },
            'volumes': {
                self.base_config['shared_data_path']: '/data/shared',
                f'/data/session_{session_id}': '/data/session'
            },
            'network': 'conjecture_network'
        }
        
        container = await self._create_container(container_config)
        self.session_containers[session_id] = container
        
        # Initialize shared database connection
        shared_db = await self._initialize_shared_database(session_config)
        self.session_databases[session_id] = shared_db
        
        # Start session runtime
        await self._start_session_runtime(container, session_id)
        
        return session_id
    
    async def initialize_from_community_checkpoint(self, session_id: str, checkpoint_url: str):
        """Initialize session from community checkpoint"""
        
        # Download checkpoint
        checkpoint_data = await self._download_checkpoint(checkpoint_url)
        
        # Load into session database
        db_manager = self.session_databases[session_id]
        await self._load_checkpoint_data(db_manager, checkpoint_data)
        
        # Initialize session with community knowledge
        session = await self._get_session(session_id)
        await session._initialize_from_checkpoint(checkpoint_data)
```

## Performance Optimization Architecture

### Claim Selection Heap Optimization
```python
class OptimizedClaimSelectionHeap:
    def __init__(self):
        self.similarity_cache: Dict[str, float] = {}
        self.confidence_threshold = 0.90
        self.cache_ttl = 300  # 5 minutes
        
    async def get_optimized_batch(self, root_claim_id: str, db_manager: DatabaseManager) -> List[Claim]:
        """Optimized batch selection with similarity caching"""
        
        # Check cache first
        cache_key = f"batch_{root_claim_id}"
        if cache_key in self.similarity_cache:
            cached_batch = self.similarity_cache[cache_key]
            if time.time() - cached_batch['timestamp'] < self.cache_ttl:
                return cached_batch['claims']
        
        # Optimized query with pre-calculated embeddings
        batch = await self._fetch_optimized_batch(root_claim_id, db_manager)
        
        # Cache result
        self.similarity_cache[cache_key] = {
            'claims': batch,
            'timestamp': time.time()
        }
        
        return batch
    
    async def _fetch_optimized_batch(self, root_claim_id: str, db_manager: DatabaseManager) -> List[Claim]:
        """Fetch batch with database-level similarity calculation"""
        
        # This query is optimized at the database level
        query = """
        WITH target_embedding AS (
            SELECT embedding FROM claims WHERE id = $root_claim_id
        )
        SELECT c.*, 
               CASE 
                 WHEN c.confidence < 0.90 
                 THEN (c.embedding <-> (SELECT embedding FROM target_embedding)) - 0.30
                 ELSE (c.embedding <-> (SELECT embedding FROM target_embedding))
               END as adjusted_similarity
        FROM claims c 
        WHERE c.dirty = true 
        ORDER BY adjusted_similarity ASC 
        LIMIT 4
        """
        
        return await db_manager.execute_query(query, {'root_claim_id': root_claim_id})
```

## Data Models

### Unified Claim Data Structure
```python
class Claim:
    id: str  # Unique identifier (format: c#######, e.g., "c0000001")
    content: str  # Claim text (50 words for concepts, 500 for thesis)
    confidence: float  # Confidence score (0.0-1.0)
    dirty: boolean  # Flag indicating claim needs re-evaluation
    tags: List[str]  # Flexible categorization replacing rigid types
    created_at: datetime  # Creation timestamp (ISO 8601)
    created_by: str  # User ID of creator
    embedding: Optional[List[float]]  # Vector embedding
    # Note: support relationships are stored in junction table
```
    
# Goals are implemented as claims with specific content patterns
# Goals: claims stating what exists/is done, with confidence based on supporting evidence
# Examples:
# - Content: "FeatureX implementation with proper error handling and unit tests"
#   - Confidence: 0.3 (low - only implementation planned, no actual tests yet)
# - Content: "Quantum entanglement research completed with reproducible experiments"
#   - Confidence: 0.7 (moderate - some experimental evidence, but needs more validation)
# - Tags: ["implementation", "featurex", "tested"] or ["research", "quantum", "entanglement"]
# - supported_by relationships to evidence/research supporting completion
# - supports relationships to further work that depends on this completion
# - No special handling - just regular claims evaluated by LLM based on evidence
```

### Specialized Claim Types through Tagging System
```
Primary Claim Types (through tags):
- concept: Building blocks of understanding (≤50 words)
- thesis: Comprehensive explanations (≤500 words)
- reference: Source provenance and citations
- implementation: Code/feature implementation claims
- research: Research findings and discoveries

Processing Patterns (through tags):
- task/todo: Action items and deliverables
- test: Testing and validation claims
- example: Illustrative examples and demonstrations
- result: Experimental results or outcomes

Domain/Context Tags:
- User-defined categories for organization and filtering
- Examples: "ai", "database", "frontend", "backend", "security"
```

### Simplified Relationship System
```python
# Relationships are implemented as parent-child connections between claims

class RelationshipDirection(Enum):
    # Forward relationship: this claim supports other claims
    SUPPORTS = "supports"  
    # Backward relationship: other claims support this claim
    SUPPORTED_BY = "supported_by"  # Inferred from reverse direction of supports

# Example: Implementation tracking through relationships
# Implementation claim
#   - children: test claims that validate the implementation
#   - parents: requirement claims that this implementation fulfills
#   - confidence: certainty that implementation is complete/correct
```

### Goal-Layer Implementation (Claims with Specialized Tags)
```
# Goals are claims with the "goal" tag
# Progress is tracked through evidence-based confidence (not completion percentage)
# Status is maintained through additional tags (active, paused, completed)

Example Goal Claim:
{
    "id": "c0000001",
    "content": "User authentication system complete with OAuth and JWT support",
    "confidence": 0.3,  # Low confidence - only OAuth implemented, JWT not yet
    "supported_by": ["c0000005"],  # Evidence: OAuth implementation works
    "supports": ["c0000010"],     # This enables user management features
    "tags": ["goal", "learning", "research", "active"],  # Type, domain, status
    "created_at": "2025-06-18T12:00:00Z",
    "created_by": "user123"
}

Tag Patterns for Goals:
- Type: "goal"
- Domain: "learning", "research", "development", etc.
- Status: "active", "paused", "completed", "cancelled"
- Priority: "low", "medium", "high", "critical" (if needed)
```

### Query Data Structure
```python
class Query:
    id: str  # Unique identifier
    text: str  # Query text
    query_type: QueryType  # Type of query
    filters: Dict[str, Any]  # Query filters
    created_at: datetime  # Creation timestamp
    created_by: str  # User ID of creator
    results: Optional[List[QueryResult]]  # Query results
    
class QueryType(Enum):
    EXACT = "exact"  # Exact text match
    SIMILARITY = "similarity"  # Semantic similarity
    STRUCTURAL = "structural"  # Graph structure
    HYBRID = "hybrid"  # Combination of methods
    
class QueryResult:
    claim_id: str  # Result claim ID
    relevance_score: float  # Relevance to query
    explanation: Optional[str]  # Explanation of relevance
```

## Database Schema

### Unified Knowledge Graph Structure
```
(Claim) -[CLAIM_RELATIONSHIPS]-> (Claim)
    |
   +-[TAGGED_AS]-> (Tag)
    |
   +-[CREATED_BY]-> (User)

# Specialized Claim Patterns:
# - Goal claims connect to Objective claims through supported_by relationships
# - Goal claims connect to Task claims through supports relationships
# - All claims connect through symmetric supporter/supported relationships
```

### Indexes for Performance
1. Claim content full-text index
2. Claim embedding vector index (for similarity search)
3. Claim relationship junction table indexes (both directions)
4. Tag-based lookup indexes
5. Dirty flag index for efficient evaluation queuing
6. Confidence score range index
7. User activity timestamp index

## Data Access Layer

### Database Manager Interface
```python
class DatabaseManager:
    # Claim operations
    async def create_claim(self, claim: Claim) -> str
    async def get_claim(self, claim_id: str) -> Optional[Claim]
    async def update_claim(self, claim_id: str, updates: Dict[str, Any]) -> bool
    async def delete_claim(self, claim_id: str) -> bool
    async def search_claims(self, query: Query) -> List[Claim]
    
    # Relationship operations (supports/supported_by)
    async def add_supports_relationship(self, supporter_id: str, supported_id: str, user_id: str = None) -> str
    async def remove_supports_relationship(self, supporter_id: str, supported_id: str) -> bool
    async def get_claim_supports(self, claim_id: str) -> List[str]  # Claims this claim supports
    async def get_claim_supported_by(self, claim_id: str) -> List[str]  # Claims that support this claim
    async def get_bidirectional_relationships(self, claim_id: str) -> Dict[str, List[str]]
    
    # Graph operations
    async def get_related_claims(self, claim_id: str, depth: int = 1) -> List[Claim]
    async def get_knowledge_subgraph(self, claim_ids: List[str]) -> Dict
    
    # Tag operations
    async def get_claims_by_tag(self, tag: str) -> List[Claim]
    async def get_claims_by_tags(self, tags: List[str], match_all: bool = False) -> List[Claim]
    
    # Dirty claim operations
    async def get_dirty_claims_by_relevance(self, root_embedding: List[float], batch_size: int = 4) -> List[Claim]
    async def mark_claims_dirty(self, claim_ids: List[str]) -> bool
    async def mark_claim_clean(self, claim_id: str) -> bool
    async def mark_related_claims_dirty(self, claim_id: str) -> bool
    
    # Goal-specific operations (tag-based)
    async def get_user_goals(self, user_id: str) -> List[Claim]  # Claims with "goal" tag
```

### Vector Database Interface
```python
class VectorDatabase:
    async def add_embedding(self, claim_id: str, embedding: List[float]) -> bool
    async def search_similar(self, embedding: List[float], count: int = 10) -> List[Tuple[str, float]]
    async def update_embedding(self, claim_id: str, embedding: List[float]) -> bool
    async def delete_embedding(self, claim_id: str) -> bool
    async def find_similar_claims(self, claim_id: str, count: int = 10) -> List[Tuple[str, float]]

class ClaimEvaluationQueue:
    """Manages prioritized evaluation of dirty claims"""
    
    async def get_next_evaluation_batch(self, root_claim_id: str, batch_size: int = 4) -> List[Claim]:
        """
        Get top N dirty claims by relevance to root claim with confidence boost
        
        Claims with confidence < 0.90 receive priority boost in selection
        """
        query = """
        SELECT c.*, 
               (c.embedding <-> $root_embedding) - 
               (CASE WHEN c.confidence < 0.90 THEN 0.30 ELSE 0 END) as adjusted_similarity
        FROM claims c 
        WHERE c.dirty = true 
        ORDER BY adjusted_similarity ASC 
        LIMIT $batch_size
        """
        
        root_claim = await self.db_manager.get_claim(root_claim_id)
        
        return await self.db_manager.execute_query(query, {
            'root_embedding': root_claim.embedding,
            'batch_size': batch_size
        })
    
    async def process_evaluation_batch(self, root_claim_id: str, batch_size: int = 4):
        """Process a batch of dirty claims in parallel"""
        
        # Get batch of most relevant dirty claims
        batch = await self.get_next_evaluation_batch(root_claim_id, batch_size)
        
        if not batch:
            return False  # No more dirty claims
        
        # Process all claims in parallel
        tasks = [self.claim_processor.process_claim(claim.id) for claim in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check completion condition
        root_claim = await self.db_manager.get_claim(root_claim_id)
        return root_claim.confidence < 0.9  # Continue if not confident enough
    
    async def evaluate_root_claim(self, root_claim_id: str, max_iterations=50):
        """Complete evaluation loop with batch processing"""
        
        iteration = 0
        while iteration < max_iterations:
            # Get batch of dirty claims
            has_more = await self.process_evaluation_batch(root_claim_id)
            
            if not has_more:
                print("Evaluation complete - no more dirty claims")
                break
                
            # Check root confidence
            root_claim = await self.db_manager.get_claim(root_claim_id)
            print(f"Iteration {iteration}: Root confidence = {root_claim.confidence:.2f}")
            
            if root_claim.confidence >= 0.9:
                print("Evaluation complete - root claim confident")
                break
                
            iteration += 1
```

## Service Layer

### Unified Claim Processing Service
```python
class ClaimProcessor:
    def __init__(self, db_manager: DatabaseManager, embedding_service: EmbeddingService):
        self.db_manager = db_manager
        self.embedding_service = embedding_service
        
    async def create_claim(self, content: str, tags: List[str], parents: List[str], user_id: str) -> str:
        # Create embedding for claim
        embedding = await self.embedding_service.generate_embedding(content)
        
        # Determine initial confidence based on claim type and sources
        initial_confidence = self._calculate_initial_confidence(tags, parents)
        
        # Create claim object
        claim = Claim(
            content=content,
            tags=tags,
            embedding=embedding,
            created_by=user_id,
            confidence=initial_confidence,
            dirty=True  # All claims start dirty, requiring evaluation
        )
        
        # Store claim in database
        claim_id = await self.db_manager.create_claim(claim)
        
        # Update claim relationships
        await self._update_claim_relationships(claim_id, parents, user_id)
        
        # Mark related claims as dirty (this claim might affect them)
        await self._mark_related_claims_dirty(claim_id)
        
        # Validate claim (async)
        asyncio.create_task(self._validate_claim(claim_id))
        
        return claim_id

    def _calculate_initial_confidence(self, tags: List[str], initial_confidence: float = None, created_by: str = None) -> float:
        # User claims start with no confidence (no evidence)
        # LLM claims use the confidence provided in response
        if initial_confidence is not None:
            return initial_confidence
        elif created_by != 'llm':  # User-created claims
            return 0.0
        else:
            return 0.5  # LLM default when no confidence specified
    
    async def _validate_claim(self, claim_id: str) -> None:
        # Get claim details
        claim = await self.db_manager.get_claim(claim_id)
        if not claim:
            return
            
        # Build evaluation context
        context = await self._build_evaluation_context(claim_id)
        
        # LLM evaluates claim and sets confidence based on evidence
        result = await self._evaluate_with_llm(claim, context)
        
        # Update claim confidence and mark as clean
        await self.db_manager.update_claim(claim_id, {
            "confidence": result["confidence"],
            "dirty": False
        })
    
    async def _validate_goal_progress(self, goal_claim: Claim) -> None:
        # Get child tasks
        # Goals are now handled like regular claims
        # No special confidence calculation needed
        # LLM evaluates based on supporting evidence
        return
    
    async def _mark_related_claims_dirty(self, claim_id: str) -> None:
        """Mark claims that might be affected by this claim as dirty"""
        # Find semantically related claims
        similar_claims = await self.db_manager.find_similar_claims(claim_id, count=100)
        
        # Mark them as dirty for re-evaluation
        claim_ids = [c[0] for c in similar_claims if c[0] != claim_id]
        if claim_ids:
            await self.db_manager.mark_claims_dirty(claim_ids)
    
    async def process_claim(self, claim_id: str) -> Dict:
        """Process a dirty claim to update its confidence and create related claims"""
        # Get claim details
        claim = await self.db_manager.get_claim(claim_id)
        if not claim or not claim.dirty:
            return {"status": "skipped", "reason": "not dirty"}
        
        # Build context for evaluation
        context = await self._build_evaluation_context(claim_id)
        
        # Process claim with LLM
        result = await self._evaluate_with_llm(claim, context)
        
        # Update claim confidence
        await self.db_manager.update_claim(claim_id, {
            "confidence": result["confidence"],
            "dirty": False  # Mark as clean
        })
        
        # Create new claims as needed
        new_claim_ids = []
        for new_claim_data in result.get("new_claims", []):
            new_id = await self.create_claim(
                new_claim_data["content"],
                new_claim_data["tags"],
                new_claim_data.get("parents", [claim_id]),
                new_claim_data.get("created_by", claim.created_by)
            )
            new_claim_ids.append(new_id)
        
        # Create relationships
        for relationship in result.get("relationships", []):
            await self.db_manager.add_supports_relationship(
                relationship["supporter_id"],
                relationship["supported_id"],
                claim.created_by
            )
        
        # Mark dependent claims as dirty (confidence chain)
        await self._mark_dependent_claims_dirty(claim_id)
        
        return {
            "status": "processed",
            "claim_id": claim_id,
            "new_confidence": result["confidence"],
            "new_claims": new_claim_ids
        }
    
    async def _mark_dependent_claims_dirty(self, claim_id: str) -> None:
        """Mark claims that depend on this claim as dirty"""
        # Get supported claims (claims this one supports)
        supported_claims = await self.db_manager.get_claim_supports(claim_id)
        
        if supported_claims:
            await self.db_manager.mark_claims_dirty(supported_claims)
    
    async def _build_evaluation_context(self, claim_id: str) -> Dict:
        """Build context for claim evaluation"""
        claim = await self.db_manager.get_claim(claim_id)
        
        # Get similar claims
        similar_claims = await self.db_manager.find_similar_claims(claim_id, count=20)
        
        # Get supporting claims (supported_by)
        supporting_claims = await self.db_manager.get_claim_supported_by(claim_id)
        
        # Get supported claims (supports)
        supported_claims = await self.db_manager.get_claim_supports(claim_id)
        
        return {
            "current_claim": claim,
            "similar_claims": [await self.db_manager.get_claim(c[0]) for c in similar_claims],
            "supporting_claims": supporting_claims,
            "supported_claims": supported_claims
        }
    
    async def _evaluate_with_llm(self, claim: Claim, context: Dict) -> Dict:
        """LLM evaluates claim and sets confidence based on available evidence"""
        prompt = f"""
        Evaluate this claim and set confidence based ONLY on available evidence:
        
        CLAIM: {claim.content}
        CURRENT CONFIDENCE: {claim.confidence}
        
        AVAILABLE EVIDENCE:
        - Supporting claims: {[c.content for c in context['supporting_claims']]}
        - Similar claims: {[c.content for c in context['similar_claims'][:10]]}
        
        CONFIDENCE GUIDELINES:
        - 0.0: No evidence (user claims, new ideas)
        - 0.3: Weak evidence (single source, theoretical)
        - 0.5: Moderate evidence (multiple sources, some examples)
        - 0.7: Strong evidence (well-documented, tested examples)
        - 0.9: Very strong evidence (industry standards, comprehensive testing)
        - 1.0: Definitive (implemented, tested, verified)
        
        CONSIDER:
        - Quality of supporting sources
        - Number of independent sources
        - Presence of examples/tests
        - Contradictory evidence (lower confidence)
        
        RESPONSE FORMAT:
        {{
            "confidence": <0.0-1.0 based on evidence>,
            "reasoning": "<explanation of confidence score>",
            "missing_evidence": ["<what would increase confidence>"],
            "new_claims": [
                {{
                    "content": "<claim for missing evidence>",
                    "tags": ["<tags>"],
                    "confidence": <0.0-1.0>,
                    "supported_by": ["<supporting_claim_ids>"]
                }}
            ],
            "relationships": [
                {{
                    "supporter_id": "<supporting_claim_id>",
                    "supported_id": "<supported_claim_id>"
                }}
            ]
        }}
        """
        
        # Call LLM service
        result = await self.llm_service.process(prompt)
        
        # Parse and validate result
        return self._parse_llm_response(result)
```

### Relationship Management Service
```python
class RelationshipManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        
    async def add_supports_relationship(
        self, 
        parent_claim_id: str, 
        child_claim_id: str,
        user_id: str
    ) -> str:
        # Create relationship as parent-child connection
        await self.db_manager.add_child_to_parent(parent_claim_id, child_claim_id)
        
        # Update claim confidence based on new relationship
        asyncio.create_task(self._update_claim_confidence_from_relationships(child_claim_id))
        
        # Mark child claim as dirty for re-evaluation
        await self.db_manager.update_claim(child_claim_id, {"dirty": True})
        await self._mark_children_dirty(child_claim_id)
        
        return f"{parent_claim_id}-> Supports -> {child_claim_id}"
    
    async def _update_claim_confidence_from_relationships(self, claim_id: str) -> None:
        # Get claim details
        claim = await self.db_manager.get_claim(claim_id)
        if not claim:
            return
            
        # Get parent claims
        parent_claims = await self.db_manager.get_parent_claims(claim_id)

        if not parent_claims:
            return  # No parents to inherit from
            
        # Calculate confidence from parents (with inheritance penalty)
        parent_confidences = [p.confidence for p in parent_claims]
        avg_parent_confidence = sum(parent_confidences) / len(parent_confidences)
        
        # Apply inheritance penalty (typically 0.05-0.10 decrease per generation)
        inheritance_penalty = 0.07  # Default inheritance penalty
        new_confidence = max(0.0, avg_parent_confidence - inheritance_penalty)
        
        # Consider multiple supporting sources can boost confidence
        if len(parent_claims) > 1:
            boost = min(0.1, len(parent_claims) * 0.02)
            new_confidence = min(1.0, new_confidence + boost)
        
        # Update claim
        await self.db_manager.update_claim(claim_id, {"confidence": new_confidence})
    
        async def _update_orphaned_status(self, claim: Claim) -> None:
            """Update orphaned status based on confidence and child dependencies"""
            # If confidence < 0.95, definitely not orphaned
            if claim.confidence < 0.95:
                if claim.orphaned:
                    claim.orphaned = False
                    await self.db_manager.update_claim(claim.id, {"orphaned": False})
                return
            
            # Check if any active children need this claim
            active_children_count = await self.db_manager.count_claims({
                'parents': claim.id,
                'confidence': {'$lt': 0.95},
                'orphaned': False
            })
        
            # Update orphaned status
            was_orphaned = claim.orphaned
            claim.orphaned = active_children_count == 0
        
            # Only save if status changed
            if was_orphaned != claim.orphaned:
                await self.db_manager.update_claim(claim.id, {"orphaned": claim.orphaned})
    
        async def _mark_children_dirty(self, claim_id: str):
            """Recursively mark child claims as dirty"""
            children = await self.db_manager.get_child_claims(claim_id)
        
            for child in children:
                if not child.orphaned:
                    await self.db_manager.update_claim(child.id, {"dirty": True})
                    # Recursively mark grandchildren
                    await self._mark_children_dirty(child.id)
```



## Interface Components

### Unified Claim Service Interface
```python
class ConjectureService:
    def __init__(
        self,
        db_manager: DatabaseManager,
        claim_processor: ClaimProcessor,
        relationship_manager: RelationshipManager,
        query_engine: QueryEngine
    ):
        self.db_manager = db_manager
        self.claim_processor = claim_processor
        self.relationship_manager = relationship_manager
        self.query_engine = query_engine
        
    # Common operations available to all interfaces
    async def create_claim(self, content: str, tags: List[str], parents: List[str], user_id: str) -> str:
        return await self.claim_processor.create_claim(content, tags, parents, user_id)
        
    async def create_implementation_claim(self, description: str, tags: List[str], parents: List[str], user_id: str) -> str:
        """Create an implementation/feature claim"""
        impl_tags = ["implementation"] + tags
        return await self.claim_processor.create_claim(description, impl_tags, parents, user_id)
        
    async def search_claims(self, query_text: str, filters: Dict[str, Any]) -> List[Claim]:
        query = Query(text=query_text, query_type=QueryType.HYBRID, filters=filters)
        return await self.db_manager.search_claims(query)
        
    async def add_supports_relationship(
        self,
        parent_claim_id: str,
        child_claim_id: str,
        user_id: str
    ) -> str:
        return await self.relationship_manager.add_supports_relationship(
            parent_claim_id, child_claim_id, user_id
        )
        
    async def get_knowledge_graph(self, claim_id: str, depth: int = 1) -> Dict:
        return await self.db_manager.get_knowledge_subgraph([claim_id])
        
    async def evaluate_claim_batch(self, root_claim_id: str, batch_size: int = 4):
        """Evaluate a batch of prioritized dirty claims"""
        
        # Get most relevant dirty claims with confidence boost
        query = """
        SELECT c.*, 
               (c.embedding <-> $root_embedding) - 
               (CASE WHEN c.confidence < 0.90 THEN 0.30 ELSE 0 END) as adjusted_similarity
        FROM claims c 
        WHERE c.dirty = true AND c.orphaned = false
        ORDER BY adjusted_similarity ASC 
        LIMIT $batch_size
        """
        
        root_claim = await self.db_manager.get_claim(root_claim_id)
goal_id)
        if not goal or "goal" not in goal.tags:
            return {"error": "Goal not found"}
            
        # Get child tasks
        child_tasks = await self.db_manager.get_child_claims(goal_id)
        
        return {
            "goal_id": goal_id,
            "title": goal.content,
            "confidence": goal.confidence,  # Evidence-based confidence, not progress
            "status": next((tag for tag in goal.tags if tag in ["active", "paused", "completed"]), "unknown"),
            "supporting_evidence": await get_claim_supported_by(goal_id),
                for task in child_tasks
            ]
        }
```

### Interface Adapters

#### TUI Adapter
```python
class TUIAdapter:
    def __init__(self, service: ConjectureService):
        self.service = service
        
    def create_claim_from_input(self, content: str, tags: List[str], user_id: str, parents: List[str] = None) -> str:
        if parents is None:
            parents = []
            
        # Create claim through service
        return asyncio.run(self.service.create_claim(content, tags, parents, user_id))
        
    def search_and_display(self, query_text: str, filters: Dict[str, Any]) -> List[Dict]:
        # Search claims through service
        claims = asyncio.run(self.service.search_claims(query_text, filters))
        
        # Convert to display format
        return [
            {
                "id": claim.id,
                "content": claim.content,
                "confidence": claim.confidence,
                "tags": claim.tags
            }
            for claim in claims
        ]
```

#### CLI Adapter
```python
class CLIAdapter:
    def __init__(self, service: ConjectureService):
        self.service = service
        
    async def claim_command(self, statement: str, tags: List[str], user_id: str, parents: List[str] = None) -> Dict:
        if parents is None:
            parents = []
            
        claim_id = await self.service.create_claim(statement, tags, parents, user_id)
        claim = await self.service.db_manager.get_claim(claim_id)
        
        return {
            "claim_id": claim_id,
            "content": claim.content,
            "confidence": claim.confidence,
            "tags": claim.tags
        }
        
    async def inspect_command(self, query: str, count: int, user_id: str) -> List[Dict]:
        claims = await self.service.search_claims(query, {"limit": count})
        
        return [
            {
                "claim_id": claim.id,
                "content": claim.content,
                "confidence": claim.confidence,
                "tags": claim.tags
            }
            for claim in claims
        ]
    
    async def goal_command(self, title: str, tags: List[str], user_id: str, parents: List[str] = None) -> Dict:
        # Goals are specialized claims with the "goal" tag
        goal_tags = ["goal"] + tags
        return await self.claim_command(title, goal_tags, user_id, parents)
```

#### MCP Adapter
```python
class MCPAdapter:
    def __init__(self, service: ConjectureService):
        self.service = service
        
    async def claim_action(self, statement: str, user_id: str) -> List[str]:
        # Auto-determine claim tags (simplified for example)
        tags = ["fact"] if "." in statement[-2:] else ["hypothesis"]
        
        claim_id = await self.service.create_claim(statement, tags, [], user_id)
        return [claim_id]
        
    async def prompt_action(self, question: str, user_id: str) -> str:
        # Search for relevant claims
        claims = await self.service.search_claims(question, {})
        
        if not claims:
            return f"No relevant information found for: {question}"
            
        # Generate response based on top claim (simplified for example)
        top_claim = claims[0]
        return f"{top_claim.content} (confidence: {top_claim.confidence:.2f})"
        
    async def inspect_action(self, query: str, count: int, user_id: str) -> List[Dict]:
        claims = await self.service.search_claims(query, {"limit": count})
        
        return [
            {
                "id": claim.id,
                "content": claim.content,
                "confidence": claim.confidence,
                "tags": claim.tags
            }
            for claim in claims
        ]
```

#### WebUI Adapter
```python
class WebUIAdapter:
    def __init__(self, service: ConjectureService):
        self.service = service
        
    async def create_claim_with_metadata(
        self, 
        content: str, 
        tags: List[str],
        user_id: str,
        parents: List[str] = None
    ) -> Dict:
        if parents is None:
            parents = []
            
        claim_id = await self.service.create_claim(content, tags, parents, user_id)
        claim = await self.service.db_manager.get_claim(claim_id)
        
        return {
            "claim_id": claim_id,
            "content": claim.content,
            "confidence": claim.confidence,
            "tags": claim.tags
        }
        
    async def create_goal_with_metadata(
        self, 
        title: str, 
        tags: List[str],
        user_id: str,
        parents: List[str] = None
    ) -> Dict:
        if parents is None:
            parents = []
            
        # Goals are just specialized claims with the "goal" tag
        goal_tags = ["goal"] + tags
        return await self.create_claim_with_metadata(title, goal_tags, user_id, parents)
        
    async def get_knowledge_graph_data(self, claim_ids: List[str]) -> Dict:
        return await self.service.get_knowledge_graph(claim_ids[0], depth=2)
        
    async def collaborate_on_claim(
        self, 
        claim_id: str, 
        action: str, 
        data: Dict,
        user_id: str
    ) -> Dict:
        if action == "add_relationship":
            return await self.service.add_supports_relationship(
                claim_id,
                data["target_claim_id"],
                user_id
            )
        elif action == "add_task_to_goal":
            # Create task claim with goal as parent
            task_tags = ["task", "todo"]
            return await self.service.create_claim(
                data["task_content"],
                task_tags,
                user_id
            )
            
            # Add supports relationship
            await self.service.add_supports_relationship(
                task_result, claim_id, user_id
            )
        elif action == "complete_goal_task":
            # Update task confidence to indicate completion
            success = await self.service.db_manager.update_claim(
                data["task_id"], 
                {"confidence": 1.0}
            )
            
            if success:
                # Recalculate goal progress
                goal_claim = await self.service.db_manager.get_claim(claim_id)
                if goal_claim and "goal" in goal_claim.tags:
                    # Goals now use evidence-based confidence
                    # No task completion calculation needed
                    # Confidence is set by LLM based on supporting evidence
                    pass
                    
            return {"success": success}
        elif action == "edit":
            return await self.service.db_manager.update_claim(claim_id, data)
            
        return {"status": "error", "message": "Unknown action"}
```

## Data Storage Strategy

### Primary Database
- **Technology**: PostgreSQL with Apache AGE for graph capabilities
- **Rationale**: ACID compliance, complex query support, graph extensions
- **Storage**: Claims, evidence relationships, goals, user data

### Vector Database
- **Technology**: Qdrant or Weaviate
- **Rationale**: Optimized for similarity search, metadata filtering
- **Storage**: Claim embeddings for semantic search

### File Storage
- **Technology**: Local filesystem or cloud storage (S3, MinIO)
- **Rationale**: Large content, attachments, exports
- **Storage**: Document references, exported knowledge bases

### Caching Layer
- **Technology**: Redis
- **Rationale**: Fast access to frequently used claims, session management
- **Cache**: Active claims, user sessions, query results

### Backup Strategy
- **Primary**: Regular point-in-time snapshots
- **Secondary**: Incremental backups with versioning
- **Export**: Periodic knowledge base exports in standard formats

## Data Wrappers

### Claim Wrapper
```python
class ClaimWrapper:
    def __init__(self, claim: Claim, db_manager: DatabaseManager):
        self.claim = claim
        self.db_manager = db_manager
        
    @property
    def content(self) -> str:
        return self.claim.content
        
    @property
    def confidence(self) -> float:
        return self.claim.confidence
        
    async def get_supporting_relationships(self) -> List[str]:
        """Get claims that support this claim"""
        return await self.db_manager.get_claim_supported_by(self.claim.id)
        
    async def get_supported_relationships(self) -> List[str]:
        """Get claims that this claim supports"""
        return await self.db_manager.get_claim_supports(self.claim.id)
        
    async def update_content(self, new_content: str, user_id: str) -> bool:
        # Generate new embedding
        embedding_service = EmbeddingService()
        new_embedding = await embedding_service.generate_embedding(new_content)
        
        # Update claim
        success = await self.db_manager.update_claim(self.claim.id, {
            "content": new_content,
            "embedding": new_embedding,
            "modified_at": datetime.now()
        })
        
        if success:
            self.claim.content = new_content
            self.claim.embedding = new_embedding
            self.claim.modified_at = datetime.now()
            
        return success
```

### Evidence Wrapper
```python
# EvidenceWrapper removed - relationships now handled through junction table
        
# RelationshipWrapper class for bidirectional relationship access
class RelationshipWrapper:
    def __init__(self, supporter_id: str, supported_id: str, db_manager: DatabaseManager):
        self.supporter_id = supporter_id
        self.supported_id = supported_id
        self.db_manager = db_manager
        
    @property
    def supporter_claim(self) -> ClaimWrapper:
        claim = asyncio.run(self.db_manager.get_claim(self.supporter_id))
        return ClaimWrapper(claim, self.db_manager)
        
    @property
    def supported_claim(self) -> ClaimWrapper:
        claim = asyncio.run(self.db_manager.get_claim(self.supported_id))
        return ClaimWrapper(claim, self.db_manager)
        
    async def update_strength(self, new_strength: float, user_id: str) -> bool:
        success = await self.db_manager.update_evidence(self.evidence.id, {
            "strength": new_strength
        })
        
        if success:
            self.evidence.strength = new_strength
            # Update source claim confidence
            claim_processor = ClaimProcessor(self.db_manager, None)
            asyncio.create_task(claim_processor._update_claim_confidence(self.evidence.source_claim_id))
            
        return success
```

### Goal Claim Wrapper (Specialized Claim Wrapper)
```python
class GoalClaimWrapper:
    def __init__(self, goal_claim: Claim, db_manager: DatabaseManager):
        self.claim = goal_claim
        self.db_manager = db_manager
        
    @property
    def confidence(self) -> float:
        # For goals, confidence represents evidence strength
        return self.claim.confidence
        
    @property
    def title(self) -> str:
        return self.claim.content
        
    def get_status(self) -> str:
        # Extract status from tags
        status_tags = ["active", "paused", "completed", "cancelled"]
        for tag in status_tags:
            if tag in self.claim.tags:
                return tag
        return "unknown"
        
    async def get_tasks(self) -> List[ClaimWrapper]:
        # Get supported task claims
        task_ids = await self.db_manager.get_claim_supports(self.claim.id)
        task_claims = []
        for task_id in task_ids:
            task = await self.db_manager.get_claim(task_id)
            if task and "task" in task.tags:
                task_claims.append(task)
        return [ClaimWrapper(task, self.db_manager) for task in task_claims]
        
    async def add_task(self, task_content: str, user_id: str) -> str:
        # Create task claim with this goal as parent
        task_tags = ["task", "todo"]
        task_id = await self.db_manager.create_claim(
            content=task_content,
            tags=task_tags,
            created_by=user_id
        )
        
        # Add supports relationship to goal
        await self.db_manager.add_supports_relationship(task_id, self.claim.id, user_id)
        
        return task_id
        
        # Mark goal as dirty due to new task
        await self.db_manager.mark_claim_dirty(self.claim.id)
        
        return task_id
        
    async def update_progress(self) -> float:
        # Get child tasks
        child_tasks = await self.db_manager.get_child_claims(self.claim.id)
        
        if not child_tasks:
            return self.claim.confidence
            
        # Goals no longer calculate progress from tasks
        # Confidence is based on supporting evidence quality
        # Update confidence marker for re-evaluation
        await self.db_manager.update_claim(self.claim.id, {
            "dirty": True,
            "dirty": True  # Mark as dirty for re-evaluation
        })
        
        # Update local claim object
        self.claim.confidence = progress
        self.claim.dirty = True
        
        return progress
        
    async def complete_task(self, task_id: str, user_id: str) -> bool:
        # Update task confidence to indicate completion
        success = await self.db_manager.update_claim(task_id, {
            "confidence": 1.0,
            "dirty": True  # Mark as dirty for re-evaluation
        })
        
        if success:
            # Mark goal as dirty due to task completion
            await self.db_manager.mark_claim_dirty(self.claim.id)
            
        return success
```

## Security Architecture

### Authentication
- JWT-based token authentication
- Multi-factor authentication for sensitive operations
- Session management with expiration

### Authorization
- Role-based access control (RBAC)
- Claim-specific permissions (view, edit, delete)
- Team-based knowledge sharing controls

### Data Protection
- Encryption at rest for sensitive claims
- Encryption in transit for all API communications
- Data anonymization for collaborative knowledge sharing

### Audit Trail
- Comprehensive logging of all claim modifications
- Evidence relationship tracking
- Knowledge base version control

## Performance Optimization

### Claim Selection Heap Optimization
```python
class OptimizedClaimSelectionHeap:
    def __init__(self):
        self.similarity_cache: Dict[str, float] = {}
        self.confidence_threshold = 0.90
        self.cache_ttl = 300  # 5 minutes
        
    async def get_optimized_batch(self, root_claim_id: str, db_manager: DatabaseManager) -> List[Claim]:
        """Optimized batch selection with similarity caching"""
        
        # Check cache first
        cache_key = f"batch_{root_claim_id}"
        if cache_key in self.similarity_cache:
            cached_batch = self.similarity_cache[cache_key]
            if time.time() - cached_batch['timestamp'] < self.cache_ttl:
                return cached_batch['claims']
        
        # Optimized query with pre-calculated embeddings
        batch = await self._fetch_optimized_batch(root_claim_id, db_manager)
        
        # Cache result
        self.similarity_cache[cache_key] = {
            'claims': batch,
            'timestamp': time.time()
        }
        
        return batch
    
    async def _fetch_optimized_batch(self, root_claim_id: str, db_manager: DatabaseManager) -> List[Claim]:
        """Fetch batch with database-level similarity calculation"""
        
        # This query is optimized at the database level
        query = """
        WITH target_embedding AS (
            SELECT embedding FROM claims WHERE id = $root_claim_id
        )
        SELECT c.*, 
               CASE 
                 WHEN c.confidence < 0.90 
                 THEN (c.embedding <-> (SELECT embedding FROM target_embedding)) - 0.30
                 ELSE (c.embedding <-> (SELECT embedding FROM target_embedding))
               END as adjusted_similarity
        FROM claims c 
        WHERE c.dirty = true 
        ORDER BY adjusted_similarity ASC 
        LIMIT 4
        """
        
        return await db_manager.execute_query(query, {'root_claim_id': root_claim_id})
```

### Similarity Caching Strategy (DB Level Only)
```python
class DatabaseLevelSimilarityCache:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.cache_table = "similarity_cache"
        
    async def initialize_cache_tables(self):
        """Initialize similarity cache tables in database"""
        
        await self.db_manager.execute_query(f"""
            CREATE TABLE IF NOT EXISTS {self.cache_table} (
                claim_id_1 VARCHAR(20),
                claim_id_2 VARCHAR(20),
                similarity FLOAT,
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (claim_id_1, claim_id_2),
                INDEX idx_similarity_value (similarity),
                INDEX idx_calculated_at (calculated_at)
            )
        """)
        
        await self.db_manager.execute_query(f"""
            CREATE OR REPLACE FUNCTION get_cached_similarity(
                claim_id_1 VARCHAR, claim_id_2 VARCHAR
            ) RETURNS FLOAT AS $$
                DECLARE
                    cached_sim FLOAT;
                    recalc_sim FLOAT;
                BEGIN
                    -- Try to get cached similarity
                    SELECT similarity INTO cached_sim 
                    FROM {self.cache_table} 
                    WHERE claim_id_1 = claim_id_1 AND claim_id_2 = claim_id_2;
                    
                    IF cached_sim IS NOT NULL AND 
                       calculated_at > CURRENT_TIMESTAMP - INTERVAL '1 day' THEN
                        RETURN cached_sim;
                    END IF;
                    
                    -- Calculate new similarity
                    SELECT 1 - (c1.embedding <-> c2.embedding) INTO recalc_sim
                    FROM claims c1, claims c2 
                    WHERE c1.id = claim_id_1 AND c2.id = claim_id_2;
                    
                    -- Cache the result
                    INSERT INTO {self.cache_table} (claim_id_1, claim_id_2, similarity)
                    VALUES (claim_id_1, claim_id_2, recalc_sim)
                    ON CONFLICT (claim_id_1, claim_id_2) 
                    DO UPDATE SET similarity = recalc_sim, calculated_at = CURRENT_TIMESTAMP;
                    
                    RETURN recalc_sim;
                END;
            $$ LANGUAGE plpgsql;
        """)
    
    async def invalidate_cache_for_claim(self, claim_id: str):
        """Invalidate similarity cache when claim embedding changes"""
        
        await self.db_manager.execute_query(f"""
            DELETE FROM {self.cache_table} 
            WHERE claim_id_1 = $claim_id OR claim_id_2 = $claim_id
        """, {'claim_id': claim_id})
```

### Fresh Context Building Per Evaluation
```python
class FreshContextBuilder:
    def __init__(self, db_manager: DatabaseManager, context_window: AdaptiveContextWindow):
        self.db_manager = db_manager
        self.context_window = context_window
        
    async def build_fresh_context(self, root_claim_id: str, evaluation_round: int) -> EvaluationContext:
        """Build fresh context for each evaluation round (no persistent caching)"""
        
        # Get current state of root claim
        root_claim = await self.db_manager.get_claim(root_claim_id)
        
        # Always rebuild from scratch to ensure freshness
        context_claims = await self._assemble_fresh_claims(root_claim, evaluation_round)
        
        # Calculate token usage for this context
        token_count = self._calculate_token_usage(context_claims)
        
        # Build session-specific context (not cached across sessions)
        evaluation_context = EvaluationContext(
            root_claim=root_claim,
            context_claims=context_claims,
            token_usage=token_count,
            evaluation_round=evaluation_round,
            built_at=datetime.now(),
            fresh_build=True
        )
        
        return evaluation_context
    
    async def _assemble_fresh_claims(self, root_claim: Claim, evaluation_round: int) -> List[Claim]:
        """Assemble claims fresh from database state"""
        
        claim_ids = set()
        assembled_claims = []
        
        # Start with minimum functional claims (always fresh)
        functional_claims = await self._get_minimum_functional_claims()
        assembled_claims.extend(functional_claims)
        claim_ids.update([claim.id for claim in functional_claims])
        
        # Get currently relevant claims based on round
        if evaluation_round == 1:
            # First round: get most similar claims
            similar_claims = await self.db_manager.find_similar_claims(
                root_claim.content, count=15, threshold=0.6
            )
        else:
            # Subsequent rounds: get claims marked dirty in this round
            dirty_claims = await self.db_manager.get_dirty_claims(limit=20)
            similar_claims = [(claim.id, 1.0) for claim in dirty_claims]
        
        # Add similar claims (avoiding duplicates)
        for claim_id, similarity in similar_claims:
            if claim_id not in claim_ids and claim_id != root_claim.id:
                claim = await self.db_manager.get_claim(claim_id)
                if claim:
                    assembled_claims.append(claim)
                    claim_ids.add(claim_id)
        
        return assembled_claims
```

### Adaptive Context Sizing Based on Token Limits
```python
class AdaptiveContextSizer:
    def __init__(self, model_token_limit: int):
        self.model_token_limit = model_token_limit
        self.max_context_size = int(model_token_limit * 0.3)  # 30% rule
        self.min_context_size = 500  # Minimum tokens for meaningful context
        
    async def optimize_context_size(self, claims: List[Claim], root_claim: Claim) -> List[Claim]:
        """Adaptively size context based on token limits and claim importance"""
        
        if not claims:
            return claims
        
        # Calculate initial token usage
        total_tokens = sum(self._estimate_claim_tokens(claim) for claim in claims)
        
        if total_tokens <= self.max_context_size:
            return claims
        
        # Need to reduce context size - prioritize claims
        scored_claims = []
        
        for claim in claims:
            importance_score = await self._calculate_claim_importance(claim, root_claim)
            token_cost = self._estimate_claim_tokens(claim)
            efficiency = importance_score / token_cost
            
            scored_claims.append((efficiency, claim, token_cost))
        
        # Sort by efficiency (highest first)
        scored_claims.sort(key=lambda x: x[0], reverse=True)
        
        # Select claims until we hit the token limit
        selected_claims = []
        current_tokens = 0
        
        for efficiency, claim, token_cost in scored_claims:
            if current_tokens + token_cost <= self.max_context_size:
                selected_claims.append(claim)
                current_tokens += token_cost
            elif current_tokens < self.min_context_size:
                # Ensure minimum context size even if we exceed limits
                selected_claims.append(claim)
                current_tokens += token_cost
            else:
                break
        
        return selected_claims
    
    async def _calculate_claim_importance(self, claim: Claim, root_claim: Claim) -> float:
        """Calculate importance score for claim context selection"""
        
        # Semantic similarity to root claim
        similarity = 1 - cosine_similarity(claim.embedding, root_claim.embedding)
        
        # Confidence boost (more confident claims are more important)
        confidence_boost = claim.confidence
        
        # Type-based importance
        type_multiplier = 1.0
        if 'skill' in claim.tags:
            type_multiplier = 1.5  # Skills are very important
        elif 'concept' in claim.tags:
            type_multiplier = 1.2  # Concepts are important
        elif 'example' in claim.tags:
            type_multiplier = 0.8  # Examples less critical
        
        # Relationship-based importance
        relationship_boost = await self._calculate_relationship_importance(claim)
        
        importance = similarity * confidence_boost * type_multiplier + relationship_boost
        
        return min(1.0, importance)
    
    def _estimate_claim_tokens(self, claim: Claim) -> int:
        """Estimate token count for a claim"""
        
        # Rough estimation: 1 token ≈ 4 characters
        content_tokens = len(claim.content) // 4
        
        # Add overhead for metadata
        metadata_tokens = len(str(claim.tags)) // 4 + 10  # Tags + overhead
        
        return content_tokens + metadata_tokens
```

### Query Optimization with Database-Level Indexing
```python
class OptimizedQueryEngine:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        
    async def initialize_optimized_indexes(self):
        """Initialize database indexes for optimal query performance"""
        
        # Composite index for claim selection with similarity
        await self.db_manager.execute_query("""
            CREATE INDEX IF NOT EXISTS idx_claims_dirty_confidence 
            ON claims (dirty, confidence) 
            WHERE dirty = true
        """)
        
        # Index for tag-based queries with claim count
        await self.db_manager.execute_query("""
            CREATE INDEX IF NOT EXISTS idx_claims_tags_gin 
            ON claims USING GIN (tags)
        """)
        
        # Index for relationship queries (both directions)
        await self.db_manager.execute_query("""
            CREATE INDEX IF NOT EXISTS idx_relationships_supporter_composite 
            ON claim_relationships (supporter_id, relationship_type, created_at)
        """)
        
        await self.db_manager.execute_query("""
            CREATE INDEX IF NOT EXISTS idx_relationships_supported_composite 
            ON claim_relationships (supported_id, relationship_type, created_at)
        """)
        
        # Vector similarity index for pgvector
        await self.db_manager.execute_query("""
            CREATE INDEX IF NOT EXISTS idx_claims_embedding_vector 
            ON claims USING hnsw (embedding vector_cosine_ops)
        """)
    
    async def optimized_similar_claims_query(self, claim_id: str, limit: int = 10, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Optimized similarity query using database vector index"""
        
        query = """
        SELECT c.id, 1 - (c.embedding <-> target.embedding) as similarity
        FROM claims c, (SELECT embedding FROM claims WHERE id = $claim_id) target
        WHERE c.id != $claim_id
          AND 1 - (c.embedding <-> target.embedding) >= $threshold
        ORDER BY similarity DESC
        LIMIT $limit
        """
        
        return await self.db_manager.execute_query(query, {
            'claim_id': claim_id,
            'threshold': threshold,
            'limit': limit
        })
    
    async def optimized_dirty_claims_with_similarity(self, root_claim_id: str, batch_size: int = 4) -> List[Claim]:
        """Optimized query for dirty claims with pre-calculated similarity ranking"""
        
        query = """
        WITH target_embedding AS (
            SELECT embedding FROM claims WHERE id = $root_claim_id
        ),
        dirty_claims_ranked AS (
            SELECT c.*, 
                   (c.embedding <-> target.embedding) - 
                   CASE WHEN c.confidence < 0.90 THEN 0.30 ELSE 0 END as adjusted_similarity
            FROM claims c, target_embedding
            WHERE c.dirty = true AND c.id != $root_claim_id
            ORDER BY adjusted_similarity ASC
            LIMIT $batch_size
        )
        SELECT * FROM dirty_claims_ranked
        """
        
        return await self.db_manager.execute_query(query, {
            'root_claim_id': root_claim_id,
            'batch_size': batch_size
        })
```

### Batch Processing and Parallelization
```python
class PerformanceOptimizedProcessor:
    def __init__(self, db_manager: DatabaseManager, max_concurrent_tasks: int = 8):
        self.db_manager = db_manager
        self.max_concurrent_tasks = max_concurrent_tasks
        self.processing_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
    async def process_evaluation_batch_optimized(self, batch: List[Claim], context: EvaluationContext) -> List[ProcessingResult]:
        """Process batch with optimized parallelization"""
        
        # Create semaphore-controlled tasks
        tasks = []
        for claim in batch:
            task = self._process_single_claim_optimized(claim, context)
            tasks.append(task)
        
        # Execute with controlled concurrency
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter and categorize results
        successful_results = []
        failed_results = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_results.append({
                    'claim_id': batch[i].id,
                    'error': str(result),
                    'claim': batch[i]
                })
            else:
                successful_results.append(result)
        
        # Batch process successful results
        if successful_results:
            await self._batch_update_database(successful_results)
        
        return ProcessingResult(
            successful=successful_results,
            failed=failed_results,
            batch_size=len(batch),
            processing_time=None  # Would be set by caller
        )
    
    async def _process_single_claim_optimized(self, claim: Claim, context: EvaluationContext) -> ProcessingResult:
        """Process single claim with resource optimization"""
        
        async with self.processing_semaphore:
            start_time = time.time()
            
            try:
                # Optimized claim processing
                result = await self._evaluate_claim_with_context(claim, context)
                
                processing_time = time.time() - start_time
                
                return ProcessingResult(
                    claim_id=claim.id,
                    status='success',
                    result=result,
                    processing_time=processing_time
                )
                
            except Exception as e:
                processing_time = time.time() - start_time
                
                return ProcessingResult(
                    claim_id=claim.id,
                    status='error',
                    error=str(e),
                    processing_time=processing_time
                )
    
    async def _batch_update_database(self, results: List[ProcessingResult]):
        """Batch database updates for better performance"""
        
        updates = []
        relationship_creates = []
        
        for result in results:
            if result.status == 'success' and result.result:
                # Collect claim updates
                if 'claim_updates' in result.result:
                    updates.extend(result.result['claim_updates'])
                
                # Collect relationship creations
                if 'relationships' in result.result:
                    relationship_creates.extend(result.result['relationships'])
        
        # Batch execute updates
        if updates:
            await self.db_manager.batch_update_claims(updates)
        
        if relationship_creates:
            await self.db_manager.batch_create_relationships(relationship_creates)
```

### Scalability Patterns

#### Horizontal Scaling for Web Interface
- **Session-Based Load Balancing**: Distribute sessions across multiple server instances
- **Database Connection Pooling**: Shared connection pools with automatic failover
- **Vector Database Sharding**: Distribute vector embeddings across multiple nodes

#### Memory Optimization
- **Claim Streaming**: Stream large claim sets instead of loading all into memory
- **Embedding Compression**: Compress stored embeddings to reduce memory footprint
- **Context Window Management**: Strict token limits to prevent memory bloat

#### Asynchronous Processing
- **Background Evaluation Queue**: Process claim evaluations asynchronously
- **Non-blocking I/O**: All database and external service calls are async
- **Timeout Management**: Prevent hanging operations with appropriate timeouts

### Caching Strategy
- Claim content and metadata caching
- Query result caching with TTL
- Graph traversal result caching
- **Similarity caching at DB level only** (not in memory)

### Query Optimization
- Database query optimization
- Vector search index tuning
- Hybrid query result ordering
- **Composite indexes for complex query patterns**

### Error Handling Strategy

#### Error Types
1. **Validation Errors**: Invalid input data
2. **Authorization Errors**: Insufficient permissions
3. **Resource Errors**: Resource not found or unavailable
4. **System Errors**: Infrastructure failures

#### Error Response Format
```json
{
  "error": {
    "code": "CLAIM_NOT_FOUND",
    "message": "Claim with ID 'c0000001' not found",
    "details": {
      "claim_id": "c0000001",
      "timestamp": "2025-06-18T12:34:56Z"
    }
  }
}
```

### Error Recovery
- Automatic retry for transient failures
- Graceful degradation when dependencies unavailable
- User-friendly error messages with actionable guidance

## LLM Response Processing

### Enhanced XML-Like Structured Response Parsing

```python
class LLMResponseProcessor:
    def __init__(self, db_manager: DatabaseManager, skill_manager: SkillManager):
        self.db_manager = db_manager
        self.skill_manager = skill_manager
        self.tool_executor = ToolExecutor()
        self.example_generator = ExampleClaimGenerator()
    
    async def process_structured_response(self, llm_response: str, context: Dict) -> Dict:
        """Process XML-like structured LLM response with tool calls and claim creation"""
        
        # Parse XML-like response structure
        parsed_response = self._parse_xml_response(llm_response)
        
        # Process tool calls first
        tool_results = await self._process_tool_calls(parsed_response.get('tool_calls', []))
        
        # Generate example claims from successful tool executions
        example_claims = await self._generate_example_claims(tool_results, context)
        
        # Process claim creation with tool result integration
        claim_results = await self._process_claim_creation(parsed_response, tool_results, context)
        
        # Process relationships with example claims
        relationship_results = await self._process_relationships(
            claim_results['nc_mapping'], 
            parsed_response.get('relationships', []),
            example_claims
        )
        
        return {
            'tool_results': tool_results,
            'example_claims': example_claims,
            'claim_results': claim_results,
            'relationship_results': relationship_results
        }
    
    def _parse_xml_response(self, llm_response: str) -> Dict:
        """Parse XML-like structured response from LLM"""
        
        import xml.etree.ElementTree as ET
        
        try:
            root = ET.fromstring(f"<response>{llm_response}</response>")
            
            parsed = {
                'tool_calls': [],
                'new_claims': [],
                'relationships': []
            }
            
            # Parse tool calls
            for tool_elem in root.findall('.//tool_call'):
                tool_call = {
                    'id': tool_elem.get('id'),
                    'function': tool_elem.find('function').text,
                    'parameters': self._parse_parameters(tool_elem.find('parameters'))
                }
                parsed['tool_calls'].append(tool_call)
            
            # Parse new claims
            for claim_elem in root.findall('.//claim'):
                claim = {
                    'content': claim_elem.find('content').text,
                    'confidence': float(claim_elem.find('confidence').text),
                    'tags': [tag.text for tag in claim_elem.findall('tags/tag')],
                    'supported_by': [ref.text for ref in claim_elem.findall('supported_by/ref')],
                    'supports': [ref.text for ref in claim_elem.findall('supports/ref')]
                }
                parsed['new_claims'].append(claim)
            
            # Parse relationships
            for rel_elem in root.findall('.//relationship'):
                relationship = {
                    'supporter': rel_elem.get('supporter'),
                    'supported': rel_elem.get('supported'),
                    'type': rel_elem.get('type', 'supports')
                }
                parsed['relationships'].append(relationship)
            
            return parsed
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {'tool_calls': [], 'new_claims': [], 'relationships': []}
    
    async def _process_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        """Execute tool calls and return results"""
        
        results = []
        
        for tool_call in tool_calls:
            try:
                # Execute tool (Python function)
                result = await self.tool_executor.execute_function(
                    tool_call['function'],
                    tool_call['parameters']
                )
                
                results.append({
                    'tool_call_id': tool_call['id'],
                    'status': 'success',
                    'result': result,
                    'function': tool_call['function'],
                    'parameters': tool_call['parameters']
                })
                
            except Exception as e:
                results.append({
                    'tool_call_id': tool_call['id'],
                    'status': 'error',
                    'error': str(e),
                    'function': tool_call['function'],
                    'parameters': tool_call['parameters']
                })
        
        return results
    
    async def _generate_example_claims(self, tool_results: List[Dict], context: Dict) -> List[Claim]:
        """Generate example claims from successful tool executions"""
        
        example_claims = []
        
        for result in tool_results:
            if result['status'] == 'success':
                # Find relevant skill claims
                relevant_skills = await self.skill_manager.find_skills_for_function(result['function'])
                
                for skill in relevant_skills:
                    # Create example claim showing proper tool response format
                    example_content = self._format_example_content(result, skill)
                    
                    example_claim = Claim(
                        content=example_content,
                        tags=['example', skill.id, f"function:{result['function']}"],
                        confidence=1.0,  # Example claims are fully confident
                        created_by='system',
                        dirty=False  # Examples don't need evaluation
                    )
                    
                    # Store example claim
                    claim_id = await self.db_manager.create_claim(example_claim)
                    example_claim.id = claim_id
                    
                    # Link example to skill
                    await self.db_manager.add_supports_relationship(skill.id, claim_id, 'system')
                    
                    example_claims.append(example_claim)
        
        return example_claims
    
    async def _process_claim_creation(self, parsed_response: Dict, tool_results: List[Dict], context: Dict) -> Dict:
        """Process claim creation with tool result integration"""
        
        nc_to_c_map = {}
        created_claims = []
        
        for i, claim_data in enumerate(parsed_response.get('new_claims', [])):
            # Check for similar existing claims to avoid duplicates
            content = claim_data['content']
            tags = claim_data.get('tags', [])
            
            # Find high similarity claim (if exists)
            similar_claim_id = await self._find_high_similarity_claim(content, threshold=0.95)
            
            if similar_claim_id:
                # Use existing claim instead of creating duplicate
                claim_id = similar_claim_id
                logger.info(f"Using existing claim {claim_id} instead of duplicate")
            else:
                # Apply tool result insights to claim content
                enhanced_content = await self._apply_tool_insights(content, tool_results)
                
                # Create new claim with initial confidence from LLM
                claim = Claim(
                    content=enhanced_content,
                    tags=tags,
                    confidence=claim_data.get('confidence', 0.5),
                    created_by=context.get('user_id', 'llm'),
                    dirty=True  # All LLM claims start dirty for validation
                )
                
                claim_id = await self.db_manager.create_claim(claim)
                claim.id = claim_id
                created_claims.append(claim)
            
            # Map contextual reference to global ID
            nc_ref = f"nc{i+1:03d}"
            nc_to_c_map[nc_ref] = claim_id
        
        # Process claim relationships
        relationship_errors = await self._process_claim_relationships(
            parsed_response.get('new_claims', []),
            nc_to_c_map,
            context.get('user_id', 'llm')
        )
        
        return {
            'nc_mapping': nc_to_c_map,
            'created_claims': created_claims,
            'relationship_errors': relationship_errors
        }
    
    async def _apply_tool_insights(self, content: str, tool_results: List[Dict]) -> str:
        """Apply insights from tool results to claim content"""
        
        enhanced_content = content
        
        # Add tool result evidence to claim
        successful_results = [r for r in tool_results if r['status'] == 'success']
        
        if successful_results:
            evidence_snippets = []
            for result in successful_results:
                if isinstance(result['result'], dict):
                    # Add key findings from structured results
                    for key, value in result['result'].items():
                        if isinstance(value, (str, int, float)):
                            evidence_snippets.append(f"{key}: {value}")
                elif isinstance(result['result'], str):
                    # Add string results directly
                    evidence_snippets.append(result['result'][:200])  # Limit to 200 chars
            
            if evidence_snippets:
                evidence_text = "; ".join(evidence_snippets[:3])  # Top 3 evidence items
                enhanced_content = f"{content} [Evidence: {evidence_text}]"
        
        return enhanced_content
    
    async def _find_high_similarity_claim(self, content: str, threshold: float = 0.95) -> Optional[str]:
        """Find existing claim with high similarity to prevent duplicates"""
        # Search vector database for similar claims
        similar_claims = await self.db_manager.find_similar_claims(content, limit=1)
        
        if similar_claims and similar_claims[0][1] >= threshold:
            return similar_claims[0][0]  # Return claim ID
        
        return None

### Tool Execution and Reflection System

```python
class ToolExecutor:
    def __init__(self):
        self.available_functions = {
            'search_web': self._search_web,
            'execute_python': self._execute_python,
            'read_file': self._read_file,
            'create_file': self._create_file,
            'get_current_time': self._get_current_time
        }
    
    async def execute_function(self, function_name: str, parameters: Dict) -> Any:
        """Execute Python function with safety checks"""
        
        if function_name not in self.available_functions:
            raise ValueError(f"Unknown function: {function_name}")
        
        function = self.available_functions[function_name]
        
        # Validate parameters
        validated_params = self._validate_parameters(function_name, parameters)
        
        # Execute function with timeout and resource limits
        try:
            result = await asyncio.wait_for(
                function(**validated_params),
                timeout=30.0  # 30 second timeout
            )
            
            return result
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"Function {function_name} timed out")
        except Exception as e:
            raise RuntimeError(f"Function {function_name} failed: {str(e)}")
    
    async def _execute_python(self, code: str) -> Dict:
        """Safely execute Python code with limited permissions"""
        
        # Create restricted execution environment
        restricted_globals = {
            '__builtins__': {
                'print': print,
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'list': list,
                'dict': dict,
                'set': set,
                'range': range,
                'enumerate': enumerate,
                'sum': sum,
                'max': max,
                'min': min,
                'abs': abs,
                'round': round,
            },
            'math': __import__('math'),
            'json': __import__('json'),
            'datetime': __import__('datetime'),
        }
        
        # Capture output
        import io
        import sys
        from contextlib import redirect_stdout, redirect_stderr
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Execute the code
                exec(code, restricted_globals)
            
            return {
                'status': 'success',
                'stdout': stdout_capture.getvalue(),
                'stderr': stderr_capture.getvalue(),
                'variables': {k: v for k, v in restricted_globals.items() 
                             if not k.startswith('__') and k not in ['math', 'json', 'datetime']}
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'stdout': stdout_capture.getvalue(),
                'stderr': stderr_capture.getvalue()
            }
    
    def _validate_parameters(self, function_name: str, parameters: Dict) -> Dict:
        """Validate and sanitize function parameters"""
        
        validation_rules = {
            'execute_python': {
                'code': {'type': str, 'max_length': 10000}
            },
            'search_web': {
                'query': {'type': str, 'max_length': 500},
                'max_results': {'type': int, 'min': 1, 'max': 10}
            },
            'read_file': {
                'path': {'type': str, 'allowed_extensions': ['.txt', '.md', '.json', '.csv']}
            }
        }
        
        if function_name in validation_rules:
            rules = validation_rules[function_name]
            
            for param_name, rule in rules.items():
                if param_name in parameters:
                    value = parameters[param_name]
                    
                    # Type validation
                    if 'type' in rule and not isinstance(value, rule['type']):
                        raise ValueError(f"Parameter {param_name} must be of type {rule['type'].__name__}")
                    
                    # Length validation
                    if 'max_length' in rule and len(str(value)) > rule['max_length']:
                        raise ValueError(f"Parameter {param_name} exceeds maximum length")
                    
                    # Value range validation
                    if 'min' in rule and value < rule['min']:
                        raise ValueError(f"Parameter {param_name} below minimum value")
                    if 'max' in rule and value > rule['max']:
                        raise ValueError(f"Parameter {param_name} above maximum value")
                    
                    # File extension validation
                    if 'allowed_extensions' in rule:
                        import os
                        ext = os.path.splitext(value)[1].lower()
                        if ext not in rule['allowed_extensions']:
                            raise ValueError(f"File extension {ext} not allowed")
        
        return parameters
```

### Two-Pass Processing System (Enhanced)

```python
def process_make_claims(llm_response: Dict) -> Dict[str, str]:
    """Upsert claims in database, auto-resolves high similarity claim merging
    
    Returns mapping of contextual references (nc###) to global claim IDs (c#######)
    """
    
    nc_to_c_map = {}
    
    # Process each new claim from LLM response
    for i, claim_data in enumerate(llm_response.get('new_claims', [])):
        # Check for similar existing claims to avoid duplicates
        content = claim_data['content']
        tags = claim_data.get('tags', [])
        
        # Find high similarity claim (if exists)
        similar_claim_id = find_high_similarity_claim(content, threshold=0.95)
        
        if similar_claim_id:
            # Use existing claim instead of creating duplicate
            claim_id = similar_claim_id
        else:
            # Create new claim
            claim_id = create_claim(
                content=content,
                tags=tags,
                created_by=llm_response.get('user_id', 'system')
            )
        
        # Map contextual reference to global ID
        nc_ref = f"nc{i+1:03d}"
        nc_to_c_map[nc_ref] = claim_id
    
    return nc_to_c_map

def process_add_supports(llm_response: Dict, nc_to_c_map: Dict[str, str]) -> List[str]:
    """Append claim support/supported_by relationships using only real claim IDs
    
    Returns list of any errors in relationship processing
    """
    
    relationship_errors = []
    
    for i, claim_data in enumerate(llm_response.get('new_claims', [])):
        nc_ref = f"nc{i+1:03d}"
        claim_id = nc_to_c_map[nc_ref]
        
        # Process supported_by relationships
        for ref in claim_data.get('supported_by', []):
            supporter_id = resolve_claim_reference(ref, nc_to_c_map)
            
            if supporter_id:
                add_supports_relationship(supporter_id, claim_id, llm_response.get('user_id'))
            else:
                relationship_errors.append(f"Unknown reference: {ref}")
        
        # Process supports relationships  
        for ref in claim_data.get('supports', []):
            supported_id = resolve_claim_reference(ref, nc_to_c_map)
            
            if supported_id:
                add_supports_relationship(claim_id, supported_id, llm_response.get('user_id'))
            else:
                relationship_errors.append(f"Unknown reference: {ref}")
    
    return relationship_errors

def resolve_claim_reference(ref: str, nc_to_c_map: Dict[str, str]) -> Optional[str]:
    """Convert nc### or c####### references to valid global claim IDs"""
    if ref.startswith('nc'):
        return nc_to_c_map.get(ref)  # Convert contextual to global
    elif ref.startswith('c') and ref[1:].isdigit():
        return ref  # Already global ID
    else:
        return None  # Invalid reference format

def find_high_similarity_claim(content: str, threshold: float = 0.95) -> Optional[str]:
    """Find existing claim with high similarity to prevent duplicates"""
    # Search vector database for similar claims
    similar_claims = vector_search(content, limit=1)
    
    if similar_claims and similar_claims[0][1] >= threshold:
        return similar_claims[0][0]  # Return claim ID
    
    return None
```

## Trustworthiness Validation System

### Web Content Author Trustworthiness Assessment

```python
class TrustworthinessValidator:
    def __init__(self, db_manager: DatabaseManager, web_scanner: WebContentScanner):
        self.db_manager = db_manager
        self.web_scanner = web_scanner
        self.author_cache: Dict[str, AuthorTrustProfile] = {}
        
    async def validate_source_trustworthiness(self, claim: Claim) -> TrustValidationResult:
        """Validate web content author trustworthiness for claim sources"""
        
        # Extract URLs and source references from claim content
        sources = await self._extract_sources_from_claim(claim)
        
        if not sources:
            return TrustValidationResult(
                claim_id=claim.id,
                trust_score=0.5,  # Neutral for claims without web sources
                validation_chain=[],
                missing_evidence=["No web sources found for trustworthiness validation"]
            )
        
        validation_chain = []
        total_trust = 0.0
        source_count = 0
        
        for source in sources:
            # Get or create author trust profile
            author_profile = await self._get_author_trust_profile(source)
            
            # Validate source content
            source_validation = await self._validate_source_content(source, author_profile)
            validation_chain.append(source_validation)
            
            total_trust += source_validation.trust_score
            source_count += 1
        
        # Calculate average trust score
        avg_trust = total_trust / source_count if source_count > 0 else 0.5
        
        # Create persistent trust claim for this claim
        trust_claim = await self._create_trust_claim(claim.id, avg_trust, validation_chain)
        
        return TrustValidationResult(
            claim_id=claim.id,
            trust_score=avg_trust,
            trust_claim_id=trust_claim.id,
            validation_chain=validation_chain,
            persistent=True  # Trust claims are persistent until revalidated
        )
    
    async def _get_author_trust_profile(self, source: WebSource) -> AuthorTrustProfile:
        """Get or create author trustworthiness profile"""
        
        # Check cache first
        if source.author_domain in self.author_cache:
            cached_profile = self.author_cache[source.author_domain]
            if time.time() - cached_profile.last_updated < 86400:  # 24 hour cache
                return cached_profile
        
        # Create new profile or update existing
        profile = await self._build_author_profile(source)
        self.author_cache[source.author_domain] = profile
        
        # Persist profile to database
        await self.db_manager.store_author_trust_profile(profile)
        
        return profile
    
    async def _build_author_profile(self, source: WebSource) -> AuthorTrustProfile:
        """Build comprehensive author trustworthiness profile"""
        
        # Scan author's web presence
        author_metrics = await self.web_scanner.scan_author_metrics(source.author_domain)
        
        # Calculate trust factors
        domain_authority = author_metrics.get('domain_authority', 0)
        publication_frequency = author_metrics.get('publication_frequency', 0)
        citation_count = author_metrics.get('citation_count', 0)
        peer_review_score = author_metrics.get('peer_review_score', 0)
        institutional_affiliation = author_metrics.get('institutional_affiliation_score', 0)
        
        # Expertise relevance to claim domain
        expertise_score = await self._calculate_expertise_relevance(source, author_metrics)
        
        # Historical accuracy (if we have data)
        historical_accuracy = await self._get_historical_accuracy(source.author_domain)
        
        # Weighted trust score calculation
        trust_factors = {
            'domain_authority': (domain_authority / 100) * 0.15,
            'publication_frequency': min(publication_frequency / 30, 1.0) * 0.10,  # Monthly publications
            'citation_count': min(citation_count / 1000, 1.0) * 0.20,
            'peer_review_score': peer_review_score * 0.25,
            'institutional_affiliation': institutional_affiliation * 0.15,
            'expertise_relevance': expertise_score * 0.10,
            'historical_accuracy': historical_accuracy * 0.05
        }
        
        overall_trust = sum(trust_factors.values())
        
        return AuthorTrustProfile(
            author_domain=source.author_domain,
            overall_trust_score=overall_trust,
            trust_factors=trust_factors,
            expertise_areas=author_metrics.get('expertise_areas', []),
            last_updated=time.time(),
            validation_sources=author_metrics.get('validation_sources', [])
        )
    
    async def _create_trust_claim(self, claim_id: str, trust_score: float, validation_chain: List[SourceValidation]) -> Claim:
        """Create persistent trust claim for a source claim"""
        
        trust_content = f"Source trustworthiness validation for claim {claim_id}: {trust_score:.2f}"
        
        # Add evidence tags
        trust_tags = ['trustworthiness', 'source_validation']
        
        # Add domain-specific tags based on validation
        for validation in validation_chain:
            if validation.domain_relevance > 0.8:
                trust_tags.append(f"domain:{validation.domain}")
        
        trust_claim = Claim(
            content=trust_content,
            tags=trust_tags,
            confidence=trust_score,
            created_by='trust_validator',
            dirty=False,  # Trust claims don't need further evaluation
            metadata={
                'validated_claim_id': claim_id,
                'trust_score': trust_score,
                'validation_date': datetime.now().isoformat(),
                'validation_chain': [v.to_dict() for v in validation_chain]
            }
        )
        
        # Create relationship: trust claim supports the original claim
        trust_claim_id = await self.db_manager.create_claim(trust_claim)
        await self.db_manager.add_supports_relationship(trust_claim_id, claim_id, 'trust_validator')
        
        trust_claim.id = trust_claim_id
        return trust_claim
```

### Multi-Level Source Validation Chains

```python
class SourceValidationChain:
    def __init__(self, trust_validator: TrustworthinessValidator):
        self.trust_validator = trust_validator
        self.validation_depth = 3  # Maximum depth for source validation
        
    async def build_validation_chain(self, claim: Claim) -> List[SourceValidation]:
        """Build multi-level validation chain for claim sources"""
        
        validation_chain = []
        visited_sources = set()
        
        await self._build_chain_recursive(claim, validation_chain, visited_sources, depth=0)
        
        return validation_chain
    
    async def _build_chain_recursive(self, claim: Claim, chain: List[SourceValidation], 
                                   visited_sources: Set[str], depth: int):
        """Recursively build validation chain"""
        
        if depth >= self.validation_depth:
            return
        
        # Extract sources from current claim
        sources = await self.trust_validator._extract_sources_from_claim(claim)
        
        for source in sources:
            if source.url in visited_sources:
                continue
                
            visited_sources.add(source.url)
            
            # Validate current source
            author_profile = await self.trust_validator._get_author_trust_profile(source)
            source_validation = await self.trust_validator._validate_source_content(source, author_profile)
            chain.append(source_validation)
            
            # Find sources cited by this source (next level)
            secondary_sources = await self._find_secondary_sources(source)
            
            for secondary_source in secondary_sources:
                secondary_claim = await self._create_source_claim(secondary_source)
                await self._build_chain_recursive(secondary_claim, chain, visited_sources, depth + 1)
    
    async def _validate_source_content(self, source: WebSource, author_profile: AuthorTrustProfile) -> SourceValidation:
        """Validate individual source content"""
        
        # Fetch and analyze source content
        content_analysis = await self.web_scanner.analyze_content(source.url)
        
        # Content quality metrics
        factual_accuracy = content_analysis.get('factual_accuracy', 0.5)
        bias_score = content_analysis.get('bias_score', 0.5)  # 0.0 = unbiased, 1.0 = highly biased
        citation_density = content_analysis.get('citation_density', 0)
        methodology_rigor = content_analysis.get('methodology_rigor', 0)
        
        # Calculate domain relevance
        domain_relevance = await self._calculate_domain_relevance(source, author_profile)
        
        # Content freshness (for time-sensitive domains)
        content_age = content_analysis.get('content_age_days', float('inf'))
        freshness_score = max(0, 1 - (content_age / 365))  # Decay over a year
        
        # Combined validation score
        content_factors = {
            'author_trust': author_profile.overall_trust_score * 0.30,
            'factual_accuracy': factual_accuracy * 0.25,
            'low_bias': (1 - bias_score) * 0.15,
            'citation_density': min(citation_density / 10, 1.0) * 0.10,
            'methodology_rigor': methodology_rigor * 0.15,
            'domain_relevance': domain_relevance * 0.05
        }
        
        trust_score = sum(content_factors.values())
        
        return SourceValidation(
            source_url=source.url,
            author_domain=source.author_domain,
            trust_score=trust_score,
            content_factors=content_factors,
            domain_relevance=domain_relevance,
            freshness_score=freshness_score,
            validation_timestamp=datetime.now(),
            evidence_claims=await self._extract_evidence_claims(content_analysis)
        )
```

### Persistent Trust Claims with Monthly Confidence Decay

```python
class PersistentTrustManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.decay_scheduler = MonthlyDecayScheduler()
        
    async def create_persistent_trust_claim(self, claim_id: str, validation_result: TrustValidationResult) -> Claim:
        """Create persistent trust claim that doesn't decay like normal claims"""
        
        trust_content = f"Persistent trust validation for claim {claim_id}: {validation_result.trust_score:.2f}"
        
        trust_claim = Claim(
            content=trust_content,
            tags=['persistent_trust', 'source_validation'],
            confidence=validation_result.trust_score,
            created_by='trust_validator',
            dirty=False,
            metadata={
                'validated_claim_id': claim_id,
                'validation_chain': [v.to_dict() for v in validation_result.validation_chain],
                'persistent': True,
                'original_validation_date': datetime.now().isoformat()
            }
        )
        
        trust_id = await self.db_manager.create_claim(trust_claim)
        trust_claim.id = trust_id
        
        # Create supporting relationship
        await self.db_manager.add_supports_relationship(trust_id, claim_id, 'trust_validator')
        
        # Schedule monthly revalidation
        await self.decay_scheduler.schedule_revalidation(trust_id)
        
        return trust_claim
    
    async def apply_monthly_confidence_decay(self):
        """Apply monthly confidence decay to persistent trust claims"""
        
        # Get all persistent trust claims
        trust_claims = await self.db_manager.get_claims_by_tag('persistent_trust')
        
        for trust_claim in trust_claims:
            # Check if revalidation is needed
            last_validation = datetime.fromisoformat(
                trust_claim.metadata.get('original_validation_date', trust_claim.created_at)
            )
            
            days_since_validation = (datetime.now() - last_validation).days
            
            if days_since_validation >= 30:  # Monthly decay
                # Apply decay factor
                decay_factor = 0.95  # 5% decay per month
                new_confidence = trust_claim.confidence * decay_factor
                
                # Mark for revalidation if confidence drops below threshold
                if new_confidence < 0.7:
                    await self.db_manager.update_claim(trust_claim.id, {
                        'confidence': new_confidence,
                        'dirty': True  # Mark for revalidation
                    })
                    
                    # Trigger revalidation process
                    await self._schedule_revalidation(trust_claim)
                else:
                    # Just update confidence
                    await self.db_manager.update_claim(trust_claim.id, {
                        'confidence': new_confidence
                    })
    
    async def _schedule_revalidation(self, trust_claim: Claim):
        """Schedule revalidation of persistent trust claim"""
        
        # Get original validated claim
        validated_claim_id = trust_claim.metadata.get('validated_claim_id')
        if not validated_claim_id:
            return
        
        validated_claim = await self.db_manager.get_claim(validated_claim_id)
        if not validated_claim:
            return
        
        # Re-run trust validation
        trust_validator = TrustworthinessValidator(self.db_manager, WebContentScanner())
        new_validation = await trust_validator.validate_source_trustworthiness(validated_claim)
        
        # Update trust claim if score improved
        if new_validation.trust_score > trust_claim.confidence:
            await self.db_manager.update_claim(trust_claim.id, {
                'confidence': new_validation.trust_score,
                'metadata': {
                    **trust_claim.metadata,
                    'validation_chain': [v.to_dict() for v in new_validation.validation_chain],
                    'original_validation_date': datetime.now().isoformat(),
                    'last_revalidation_date': datetime.now().isoformat()
                },
                'dirty': False
            })
        else:
            # Keep current score but update timestamp
            await self.db_manager.update_claim(trust_claim.id, {
                'metadata': {
                    **trust_claim.metadata,
                    'last_revalidation_date': datetime.now().isoformat()
                },
                'dirty': False
            })

class MonthlyDecayScheduler:
    def __init__(self):
        self.revalidation_queue = asyncio.Queue()
        self.running = False
    
    async def schedule_revalidation(self, trust_claim_id: str):
        """Schedule monthly revalidation for trust claim"""
        
        revalidation_task = RevalidationTask(
            trust_claim_id=trust_claim_id,
            scheduled_date=datetime.now() + timedelta(days=30)
        )
        
        await self.revalidation_queue.put(revalidation_task)
        
        if not self.running:
            asyncio.create_task(self._process_revalidations())
            self.running = True
    
    async def _process_revalidations(self):
        """Process scheduled revalidations"""
        
        while True:
            try:
                # Wait for next revalidation task
                task = await asyncio.wait_for(self.revalidation_queue.get(), timeout=3600)  # 1 hour
                
                # Check if it's time to revalidate
                if datetime.now() >= task.scheduled_date:
                    await self._execute_revalidation(task)
                else:
                    # Put it back and wait
                    await self.revalidation_queue.put(task)
                    await asyncio.sleep(3600)  # Check again in an hour
                    
            except asyncio.TimeoutError:
                continue  # No tasks, continue waiting
            except Exception as e:
                logger.error(f"Error in revalidation scheduler: {e}")
    
    async def _execute_revalidation(self, task: RevalidationTask):
        """Execute scheduled revalidation"""
        
        try:
            # Get persistent trust manager
            trust_manager = PersistentTrustManager(db_manager)
            
            # Get the trust claim
            trust_claim = await db_manager.get_claim(task.trust_claim_id)
            if trust_claim and 'persistent_trust' in trust_claim.tags:
                await trust_manager._schedule_revalidation(trust_claim)
                
        except Exception as e:
            logger.error(f"Error executing revalidation for {task.trust_claim_id}: {e}")
```

## Contradiction Detection and Merging Engine

### Confidence-Based Claim Merging

```python
class ContradictionDetector:
    def __init__(self, db_manager: DatabaseManager, embedding_service: EmbeddingService):
        self.db_manager = db_manager
        self.embedding_service = embedding_service
        self.contradiction_threshold = 0.8
        self.merging_threshold = 0.95
        
    async def detect_and_process_contradictions(self, claim_id: str) -> List[ContradictionResult]:
        """Detect contradictions and process merging opportunities"""
        
        target_claim = await self.db_manager.get_claim(claim_id)
        if not target_claim:
            return []
        
        # Find semantically similar claims
        similar_claims = await self.db_manager.find_similar_claims(
            target_claim.content, 
            count=20, 
            threshold=0.7
        )
        
        contradiction_results = []
        
        for similar_claim_id, similarity in similar_claims:
            if similar_claim_id == claim_id:
                continue
            
            similar_claim = await self.db_manager.get_claim(similar_claim_id)
            
            # Check for contradiction
            contradiction_score = await self._analyze_contradiction(target_claim, similar_claim)
            
            if contradiction_score >= self.contradiction_threshold:
                contradiction_result = await self._process_contradiction(
                    target_claim, similar_claim, contradiction_score
                )
                contradiction_results.append(contradiction_result)
            
            # Check for merging opportunity (very similar content)
            elif similarity >= self.merging_threshold:
                merge_result = await self._process_merge_opportunity(
                    target_claim, similar_claim, similarity
                )
                if merge_result:
                    contradiction_results.append(merge_result)
        
        return contradiction_results
    
    async def _analyze_contradiction(self, claim1: Claim, claim2: Claim) -> float:
        """Analyze semantic contradiction between two claims"""
        
        # Semantic opposition analysis
        embedding_similarity = 1 - cosine_similarity(claim1.embedding, claim2.embedding)
        
        # Extract contradiction keywords
        contradiction_keywords = [
            'not', 'never', 'cannot', 'impossible', 'false', 'incorrect',
            'however', 'but', 'although', 'despite', 'contrary', 'opposite'
        ]
        
        content1_lower = claim1.content.lower()
        content2_lower = claim2.content.lower()
        
        # Count contradiction indicators
        contradiction_indicators = 0
        for keyword in contradiction_keywords:
            if keyword in content1_lower or keyword in content2_lower:
                contradiction_indicators += 1
        
        # Factor scores
        semantic_opposition = embedding_similarity
        linguistic_contradiction = min(contradiction_indicators / 3, 1.0)
        
        # Confidence difference factor (higher confidence contradictions are more significant)
        confidence_difference = abs(claim1.confidence - claim2.confidence)
        confidence_factor = confidence_difference  # 0.0 to 1.0
        
        # Combined contradiction score
        contradiction_score = (
            0.6 * semantic_opposition + 
            0.3 * linguistic_contradiction + 
            0.1 * confidence_factor
        )
        
        return min(1.0, contradiction_score)
    
    async def _process_contradiction(self, claim1: Claim, claim2: Claim, score: float) -> ContradictionResult:
        """Process detected contradiction"""
        
        # Determine higher confidence claim
        if claim1.confidence > claim2.confidence:
            dominant_claim = claim1
            subordinate_claim = claim2
        else:
            dominant_claim = claim2
            subordinate_claim = claim1
        
        # Create contradiction detection claim
        contradiction_content = f"Contradiction detected between {claim1.id} and {claim2.id} (score: {score:.2f})"
        
        contradiction_claim = Claim(
            content=contradiction_content,
            tags=['contradiction', 'validation_needed'],
            confidence=score,  # Confidence based on contradiction detection certainty
            created_by='contradiction_detector',
            dirty=True  # Needs human evaluation
        )
        
        contradiction_id = await self.db_manager.create_claim(contradiction_claim)
        
        # Mark both claims as needing re-evaluation
        await self.db_manager.mark_claims_dirty([claim1.id, claim2.id])
        
        # Create relationship tracking
        await self.db_manager.add_supports_relationship(contradiction_id, claim1.id, 'contradiction_detector')
        await self.db_manager.add_supports_relationship(contradiction_id, claim2.id, 'contradiction_detector')
        
        return ContradictionResult(
            contradiction_id=contradiction_id,
            claim1_id=claim1.id,
            claim2_id=claim2.id,
            contradiction_score=score,
            dominant_claim_id=dominant_claim.id,
            action_required='human_evaluation'
        )
    
    async def _process_merge_opportunity(self, claim1: Claim, claim2: Claim, similarity: float) -> Optional[MergeResult]:
        """Process high similarity claims for merging"""
        
        # Don't merge if both have high confidence (likely distinct but similar topics)
        if claim1.confidence >= 0.8 and claim2.confidence >= 0.8:
            return None
        
        # Select higher confidence claim as the base
        if claim1.confidence >= claim2.confidence:
            base_claim = claim1
            merged_claim = claim2
        else:
            base_claim = claim2
            merged_claim = claim1
        
        # Create merged claim with higher confidence content
        merged_content = await self._create_merged_content(base_claim, merged_claim)
        
        # Union of support relationships
        support_sources = await self._union_support_relationships(base_claim, merged_claim)
        
        # Update the base claim with merged content and relationships
        await self.db_manager.update_claim(base_claim.id, {
            'content': merged_content,
            'confidence': max(base_claim.confidence, merged_claim.confidence),
            'dirty': True  # Mark for re-evaluation after merge
        })
        
        # Mark merged claim for deletion (or archiving)
        await self.db_manager.update_claim(merged_claim.id, {
            'tags': merged_claim.tags + ['merged_into:' + base_claim.id],
            'dirty': True
        })
        
        # Report the merge
        return MergeResult(
            surviving_claim_id=base_claim.id,
            merged_claim_id=merged_claim.id,
            similarity_score=similarity,
            support_sources_count=len(support_sources)
        )
    
    async def _create_merged_content(self, base_claim: Claim, merged_claim: Claim) -> str:
        """Create merged claim content preserving higher confidence elements"""
        
        # Start with base claim content
        merged_content = base_claim.content
        
        # Add unique elements from merged claim that aren't already in base
        await self._add_unique_elements(merged_content, merged_claim)
        
        # Preserve higher confidence evidence and citations
        if merged_claim.confidence > base_claim.confidence:
            # Extract evidence from merged claim
            evidence_elements = await self._extract_evidence_elements(merged_claim.content)
            for evidence in evidence_elements:
                if evidence not in merged_content:
                    merged_content += f" [{evidence}]"
        
        return merged_content
    
    async def _union_support_relationships(self, claim1: Claim, claim2: Claim) -> List[str]:
        """Create union of support relationships for merged claims"""
        
        # Get support relationships for both claims
        claim1_supports = await self.db_manager.get_claim_supports(claim1.id)
        claim2_supports = await self.db_manager.get_claim_supports(claim2.id)
        
        claim1_supported_by = await self.db_manager.get_claim_supported_by(claim1.id)
        claim2_supported_by = await self.db_manager.get_claim_supported_by(claim2.id)
        
        # Union of unique relationships
        unique_supports = list(set(claim1_supports + claim2_supports))
        unique_supported_by = list(set(claim1_supported_by + claim2_supported_by))
        
        # Update relationships on surviving claim
        surviving_id = claim1.id if claim1.confidence >= claim2.confidence else claim2.id
        
        # Add new relationships (avoiding duplicates)
        for support_id in unique_supports:
            if support_id != surviving_id:  # Don't create self-references
                await self.db_manager.add_supports_relationship(surviving_id, support_id, 'merge_processor')
        
        return unique_supports + unique_supported_by
    
    async def mark_claims_dirty_for_re_evaluation(self, affected_claim_ids: List[str]):
        """Mark related claims dirty after merge operations"""
        
        # For each affected claim, mark its dependents dirty
        for claim_id in affected_claim_ids:
            # Get claims that depend on this claim
            dependent_claims = await self.db_manager.get_claim_supported_by(claim_id)
            
            if dependent_claims:
                await self.db_manager.mark_claims_dirty(dependent_claims)
                
                # Recursively mark their dependents
                await self.mark_claims_dirty_for_re_evaluation(dependent_claims)
```

### Heritage Chain Preservation During Merge

```python
class HeritageChainManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        
    async def preserve_heritage_chain(self, merged_claim: Claim, original_claims: List[Claim]):
        """Preserve claim heritage chain during merge operations"""
        
        for original_claim in original_claims:
            # Create heritage preservation claim
            heritage_content = f"Heritage chain: {original_claim.id} merged into {merged_claim.id}"
            
            heritage_claim = Claim(
                content=heritage_content,
                tags=['heritage', 'merge_record', f'original:{original_claim.id}'],
                confidence=1.0,  # Heritage records are fully confident
                created_by='heritage_manager',
                dirty=False,
                metadata={
                    'original_claim_id': original_claim.id,
                    'merged_claim_id': merged_claim.id,
                    'merge_date': datetime.now().isoformat(),
                    'original_content': original_claim.content,
                    'original_confidence': original_claim.confidence,
                    'original_relationships': await self._get_claim_relationships(original_claim.id)
                }
            )
            
            heritage_id = await self.db_manager.create_claim(heritage_claim)
            
            # Link heritage claim to merged claim
            await self.db_manager.add_supports_relationship(heritage_id, merged_claim.id, 'heritage_manager')
            
            # Preserve original relationships as heritage
            await self._preserve_relationship_heritage(original_claim, merged_claim, heritage_id)
    
    async def _preserve_relationship_heritage(self, original_claim: Claim, merged_claim: Claim, heritage_id: str):
        """Preserve original claim relationships in heritage"""
        
        # Get original relationships
        original_supports = await self.db_manager.get_claim_supports(original_claim.id)
        original_supported_by = await self.db_manager.get_claim_supported_by(original_claim.id)
        
        # Create heritage relationship records
        for supporter_id in original_supported_by:
            if supporter_id != merged_claim.id:  # Avoid circular references
                heritage_rel_content = f"Original relationship: {supporter_id} supported {original_claim.id}"
                
                heritage_rel = Claim(
                    content=heritage_rel_content,
                    tags=['heritage_relationship', 'original_support'],
                    confidence=1.0,
                    created_by='heritage_manager',
                    dirty=False,
                    metadata={
                        'original_supporter': supporter_id,
                        'original_supported': original_claim.id,
                        'current_merged': merged_claim.id
                    }
                )
                
                heritage_rel_id = await self.db_manager.create_claim(heritage_rel)
                await self.db_manager.add_supports_relationship(heritage_rel_id, heritage_id, 'heritage_manager')
    
    async def trace_heritage_chain(self, claim_id: str) -> HeritageChain:
        """Trace complete heritage chain for a claim"""
        
        heritage_chain = HeritageChain(current_claim_id=claim_id)
        
        await self._trace_heritage_recursive(claim_id, heritage_chain, visited=set())
        
        return heritage_chain
    
    async def _trace_heritage_recursive(self, claim_id: str, chain: HeritageChain, visited: Set[str]):
        """Recursively trace heritage chain"""
        
        if claim_id in visited:
            return
        
        visited.add(claim_id)
        
        # Get heritage claims for this claim
        heritage_claims = await self.db_manager.execute_query("""
            SELECT c.* FROM claims c
            JOIN claim_relationships cr ON c.id = cr.supporter_id
            WHERE cr.supported_id = $claim_id AND 'heritage' = ANY(c.tags)
        """, {'claim_id': claim_id})
        
        for heritage_claim in heritage_claims:
            original_id = heritage_claim.metadata.get('original_claim_id')
            if original_id and original_id not in visited:
                chain.add_heritage_step(HeritageStep(
                    original_claim_id=original_id,
                    heritage_claim_id=heritage_claim.id,
                    merge_date=heritage_claim.metadata.get('merge_date'),
                    original_content=heritage_claim.metadata.get('original_content'),
                    merged_into=heritage_claim.metadata.get('merged_claim_id')
                ))
                
                # Continue tracing the original claim
                await self._trace_heritage_recursive(original_id, chain, visited)
```

### Contradiction Resolution Workflow

```python
class ContradictionResolutionWorkflow:
    def __init__(self, db_manager: DatabaseManager, contradiction_detector: ContradictionDetector):
        self.db_manager = db_manager
        self.contradiction_detector = contradiction_detector
        
    async def process_contradiction_batch(self, max_batch_size: int = 10) -> List[ResolutionResult]:
        """Process batch of contradictions for resolution"""
        
        # Get unresolved contradictions
        contradiction_claims = await self.db_manager.get_claims_by_tag('contradiction')
        
        resolution_results = []
        
        for contradiction_claim in contradiction_claims[:max_batch_size]:
            resolution = await self._resolve_single_contradiction(contradiction_claim)
            resolution_results.append(resolution)
        
        return resolution_results
    
    async def _resolve_single_contradiction(self, contradiction_claim: Claim) -> ResolutionResult:
        """Resolve single contradiction through structured workflow"""
        
        # Extract the two contradictory claims
        related_claims = await self.db_manager.get_bidirectional_relationships(contradiction_claim.id)
        claim_ids = [cid for cid in related_claims.get('supports', []) if cid != contradiction_claim.id]
        
        if len(claim_ids) < 2:
            return ResolutionResult(
                contradiction_id=contradiction_claim.id,
                status='insufficient_data',
                message='Cannot find both contradictory claims'
            )
        
        claim1 = await self.db_manager.get_claim(claim_ids[0])
        claim2 = await self.db_manager.get_claim(claim_ids[1])
        
        # Automated resolution attempts
        
        # 1. Check for temporal resolution (one claim may be outdated)
        temporal_resolution = await self._attempt_temporal_resolution(claim1, claim2)
        if temporal_resolution:
            return await self._apply_temporal_resolution(contradiction_claim, claim1, claim2, temporal_resolution)
        
        # 2. Check for contextual resolution (claims may apply to different contexts)
        contextual_resolution = await self._attempt_contextual_resolution(claim1, claim2)
        if contextual_resolution:
            return await self._apply_contextual_resolution(contradiction_claim, claim1, claim2, contextual_resolution)
        
        # 3. Check for evidence-weighted resolution
        evidence_resolution = await self._attempt_evidence_resolution(claim1, claim2)
        if evidence_resolution:
            return await self._apply_evidence_resolution(contradiction_claim, claim1, claim2, evidence_resolution)
        
        # If no automated resolution possible, mark for human review
        return await self._escalate_for_human_review(contradiction_claim, claim1, claim2)
    
    async def _attempt_temporal_resolution(self, claim1: Claim, claim2: Claim) -> Optional[TemporalResolution]:
        """Attempt to resolve contradiction based on temporal information"""
        
        # Extract temporal indicators from claims
        claim1_time = await self._extract_temporal_info(claim1)
        claim2_time = await self._extract_temporal_info(claim2)
        
        if claim1_time and claim2_time:
            # If one claim is significantly more recent and both discuss time-sensitive topics
            time_difference = abs((claim1_time - claim2_time).days)
            
            if time_difference > 365:  # More than a year difference
                # More recent claim gets priority for time-sensitive topics
                recent_claim = claim1 if claim1_time > claim2_time else claim2
                older_claim = claim2 if claim1_time > claim2_time else claim1
                
                return TemporalResolution(
                    recent_claim_id=recent_claim.id,
                    older_claim_id=older_claim.id,
                    reasoning=f"More recent information from {recent_claim_time}",
                    confidence=0.8
                )
        
        return None
    
    async def _apply_evidence_resolution(self, contradiction_claim: Claim, claim1: Claim, 
                                       claim2: Claim, resolution: EvidenceResolution) -> ResolutionResult:
        """Apply evidence-weighted resolution"""
        
        # Update higher-evidence claim confidence
        winning_claim = claim1 if claim1.confidence > claim2.confidence else claim2
        losing_claim = claim2 if claim1.confidence > claim2.confidence else claim1
        
        # Update contradiction claim with resolution
        await self.db_manager.update_claim(contradiction_claim.id, {
            'tags': contradiction_claim.tags + ['resolved'],
            'metadata': {
                **contradiction_claim.metadata,
                'resolution_type': 'evidence_weighted',
                'winning_claim_id': winning_claim.id,
                'losing_claim_id': losing_claim.id,
                'resolution_confidence': resolution.confidence,
                'resolution_reasoning': resolution.reasoning,
                'resolved_date': datetime.now().isoformat()
            }
        })
        
        # Update losing claim to reflect contradiction
        updated_content = f"{losing_claim.content} [Contradicted by evidence in {winning_claim.id}]"
        await self.db_manager.update_claim(losing_claim.id, {
            'content': updated_content,
            'tags': losing_claim.tags + ['contradicted'],
            'dirty': False  # No longer needs evaluation as contradiction is resolved
        })
        
        # Mark dependent claims dirty for re-evaluation
        await self.contradiction_detector.mark_claims_dirty_for_re_evaluation([winning_claim.id, losing_claim.id])
        
        return ResolutionResult(
            contradiction_id=contradiction_claim.id,
            status='resolved',
            resolution_type='evidence_weighted',
            winning_claim_id=winning_claim.id,
            reasoning=resolution.reasoning
        )
```

## Database Schema

### Claims Table
```sql
CREATE TABLE claims (
    id VARCHAR(20) PRIMARY KEY,  -- c####### format
    content TEXT NOT NULL,
    confidence FLOAT NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    dirty BOOLEAN NOT NULL DEFAULT true,
    tags TEXT[] NOT NULL DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(50),
    embedding vector(1536),  -- For pgvector
    
    -- Indexes for performance
    INDEX idx_claims_confidence (confidence),
    INDEX idx_claims_dirty (dirty),
    INDEX idx_claims_created_at (created_at),
    INDEX idx_claims_tags USING GIN (tags)
);

-- Full-text search index
CREATE INDEX idx_claims_content_fts ON claims USING GIN (to_tsvector('english', content));
```

### Claim Relationships Junction Table
```sql
CREATE TABLE claim_relationships (
    id SERIAL PRIMARY KEY,
    supporter_id VARCHAR(20) NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    supported_id VARCHAR(20) NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    relationship_type VARCHAR(20) NOT NULL DEFAULT 'supports',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(50),
    
    -- Ensure same relationship doesn't exist twice
    UNIQUE(supporter_id, supported_id, relationship_type),
    
    -- Indexes for efficient querying both directions
    INDEX idx_relationship_supporter ON claim_relationships(supporter_id),
    INDEX idx_relationship_supported ON claim_relationships(supported_id)
);
```

## Interface Components

### Unified Claim Service Interface
