# Conjecture Interface Layer Design Specification

**Version**: 1.0  
**Date**: 2025-06-18  
**Status**: Phase 3 Ready  
**Extends**: specs/design.md, specs/requirements.md, specs/phases.md  

---

## Executive Summary

This specification defines the unified architecture for Conjecture's interface layer, enabling consistent interactions across Terminal User Interface (TUI), Command Line Interface (CLI), Model Context Protocol (MCP), and Web User Interface (WebUI) implementations. The design follows the established principle of "maximum power through minimum complexity" through a shared interface component model.

### Key Design Principles

1. **Unified Claims Centerpiece**: All interfaces center around claim exploration and manipulation
2. **Dirty Flag Visualization**: All interfaces clearly display claim evaluation status
3. **Confidence-Based Prioritization**: Interfaces surface high-priority claims automatically
4. **Component Reusability**: Shared business logic across all interface types
5. **Performance First**: Sub-100ms response for all claim operations
6. **Real-Time Updates**: Immediate visibility of claim evaluation changes

---

## Interface Architecture Overview

### Multi-Modal Interface Model

```
┌─────────────────────────────────────────────────────────────┐
│                    Conjecture Interface Layer                │
├─────────────────────────────────────────────────────────────┤
│                      Shared Components                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │  Claim Manager  │  │  Confidence     │  │  Dirty Flag  │ │
│  │  Component      │  │  Manager        │  │  Tracker     │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                     Interface Adapters                     │
│                                                             │
│  ┌───────────────┐  ┌───────────────┐  ┌─────────────────┐  │
│  │   TUI Layer   │  │   CLI Layer   │  │   MCP Layer     │  │
│  │  (Rich UI)    │  │ (Commands)    │  │   (Actions)     │  │
│  └───────────────┘  └───────────────┘  └─────────────────┘  │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                WebUI Layer (Browser)                    │  │
│  └─────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                     Backend Services                       │
│    ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐   │
│    │   Claim     │  │ Relationship │  │  LLM Processing  │   │
│    │  Manager    │  │  Manager     │  │     Service      │   │
│    └─────────────┘  └─────────────┘  └──────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                     Data Layer                              │
│       ┌─────────────────┐      ┌─────────────────────┐      │
│       │ Vector Database  │      │  SQL Database       │      │
│       │  (Similarity)    │      │  (Relationships)    │      │
│       └─────────────────┘      └─────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

```
User Input → Interface Adapter → Shared Components → Backend Services → Data Layer
     ↓               ↓                    ↓                 ↓              ↓
  Interface →    Command/Action →    Business Logic →  Claim/Relationship → Storage
  Response      Processing           (Validation)     Processing        Operations
     ↑               ↑                    ↑                 ↑              ↑
  Real-Time ←    Event Bus ←     Component State ←   Dirty Flag ←    Database
  Notifications   Updates         Management          Updates         Triggers
```

---

## Multi-Session Management Architecture

### Session Isolation and Management

Conjecture supports multiple concurrent sessions with proper isolation while maintaining shared persistent storage. Each session operates independently with its own context, claims, and evaluation state.

### Session Manager Component

**Purpose**: Centralized session lifecycle management across all interfaces with isolation guarantees.

**Key Responsibilities**:
- Session creation, configuration, and termination
- Isolated context window management per session
- Session state persistence and recovery
- Multi-session resource allocation and monitoring
- Cross-session communication and data sharing controls

**Interface**:
```python
class SessionManager:
    async def create_session(self, config: SessionConfig) -> Session
    async def get_session(self, session_id: str) -> Session
    async def list_active_sessions(self, user_id: str) -> List[Session]
    async def terminate_session(self, session_id: str) -> bool
    async def switch_session(self, interface_adapter, session_id: str) -> bool
    async def get_session_metrics(self, session_id: str) -> SessionMetrics
    async def cleanup_inactive_sessions(self, max_idle_hours: int = 24) -> int
```

### Session Configuration and Isolation

```python
class SessionConfig:
    def __init__(self):
        self.session_id: str = generate_session_id()
        self.user_id: str = None
        self.interface_type: InterfaceType = InterfaceType.TUI  # TUI, CLI, MCP, WEBUI
        self.database_config: DatabaseConfig = DatabaseConfig()
        self.model_config: ModelConfig = ModelConfig()
        self.context_limit: int = 0.3  # 30% of model token limit
        self.isolation_level: IsolationLevel = IsolationLevel.FULL  # FULL, SHARED_DB, COMMUNITY
        self.checkpoint_url: Optional[str] = None  # For community initialization
        self.resource_limits: ResourceLimits = ResourceLimits()

class IsolationLevel(Enum):
    FULL = "full"          # Complete isolation with private database
    SHARED_DB = "shared_db" # Shared persistent DB, isolated session context
    COMMUNITY = "community" # Shared DB with community checkpoint initialization

class ResourceLimits:
    def __init__(self):
        self.max_claims: int = 10000
        self.max_context_tokens: int = 4096
        self.max_concurrent_evaluations: int = 4
        self.session_timeout_minutes: int = 120
        self.memory_limit_mb: int = 512
```

### Session State Management

```python
class Session:
    def __init__(self, config: SessionConfig):
        self.session_id = config.session_id
        self.config = config
        self.context_window = AdaptiveContextWindow(config.context_limit)
        self.claim_selector = ClaimSelectionHeap()
        self.skill_injector = SkillInjector()
        self.active_root_claims: Set[str] = set()
        self.evaluation_history: List[EvaluationRecord] = []
        self.created_at = datetime.now()
        self.last_activity = created_at
        self.status = SessionStatus.ACTIVE
        
    async def switch_to_session(self, interface_adapter: InterfaceAdapter) -> bool:
        """Switch interface adapter to use this session"""
        try:
            # Reconfigure interface adapter for this session
            await interface_adapter.reconfigure_for_session(self)
            
            # Load session state
            await self._load_session_state()
            
            # Update last activity
            self.last_activity = datetime.now()
            
            # Notify session switch
            await self._notify_session_switch(interface_adapter)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch to session {self.session_id}: {e}")
            return False
    
    async def get_session_summary(self) -> SessionSummary:
        """Get comprehensive session state summary"""
        
        return SessionSummary(
            session_id=self.session_id,
            status=self.status,
            uptime=(datetime.now() - self.created_at).total_seconds(),
            last_activity=self.last_activity,
            active_root_claims=len(self.active_root_claims),
            context_usage=self.context_window.get_usage_percentage(),
            evaluation_count=len(self.evaluation_history),
            interface_type=self.config.interface_type,
            isolation_level=self.config.isolation_level,
            resource_usage=await self._get_resource_usage()
        )
```

### Multi-Session Support Patterns

#### Session Switching Interface Pattern

```python
class SessionSwitcher:
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        
    async def display_available_sessions(self, interface_adapter: InterfaceAdapter) -> List[SessionSummary]:
        """Display available sessions in interface-appropriate format"""
        
        user_sessions = await self.session_manager.list_active_sessions(
            interface_adapter.user_id
        )
        
        # Format based on interface type
        if isinstance(interface_adapter, TUIAdapter):
            return self._format_sessions_for_tui(user_sessions)
        elif isinstance(interface_adapter, CLIAdapter):
            return self._format_sessions_for_cli(user_sessions)
        elif isinstance(interface_adapter, WebUIAdapter):
            return self._format_sessions_for_webui(user_sessions)
        
        return user_sessions
    
    async def handle_session_switch_command(self, interface_adapter: InterfaceAdapter, session_id: str) -> bool:
        """Handle session switch user command"""
        
        session = await self.session_manager.get_session(session_id)
        if not session:
            await interface_adapter.display_error(f"Session {session_id} not found")
            return False
        
        if session.user_id != interface_adapter.user_id:
            await interface_adapter.display_error("Access denied to session")
            return False
        
        success = await session.switch_to_session(interface_adapter)
        if success:
            await interface_adapter.display_success(f"Switched to session {session_id}")
            await interface_adapter.refresh_interface_state()
        else:
            await interface_adapter.display_error("Failed to switch session")
        
        return success

# TUI Session Management Integration
class TUISessionManager:
    def __init__(self, tui_app: TUIApplication):
        self.tui_app = tui_app
        self.session_switcher = SessionSwitcher(tui_app.session_manager)
        
    def add_session_shortcuts(self):
        """Add keyboard shortcuts for session management"""
        self.tui_app.bind_keys({
            'ctrl+s': self._show_session_menu,
            'ctrl+1': self._switch_to_session_1,
            'ctrl+2': self._switch_to_session_2,
            'ctrl+n': self._create_new_session,
            'ctrl+d': self._delete_current_session
        })
    
    async def _show_session_menu(self):
        """Show session selection menu"""
        sessions = await self.session_switcher.display_available_sessions(
            self.tui_app.interface_adapter
        )
        
        # Display TUI session menu
        selected_session = await self._display_session_menu(sessions)
        if selected_session:
            await self.session_switcher.handle_session_switch_command(
                self.tui_app.interface_adapter,
                selected_session.session_id
            )
```

#### Session Comparison and Analysis Tools

```python
class SessionComparator:
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        
    async def compare_sessions(self, session_ids: List[str]) -> SessionComparison:
        """Compare multiple sessions across various dimensions"""
        
        sessions = [await self.session_manager.get_session(sid) for sid in session_ids]
        sessions = [s for s in sessions if s]  # Filter out None
        
        comparison = SessionComparison(
            session_ids=session_ids,
            comparison_timestamp=datetime.now()
        )
        
        # Compare claim counts and growth
        comparison.claim_counts = {
            s.session_id: await self._count_session_claims(s) for s in sessions
        }
        
        # Compare confidence distributions
        comparison.confidence_distributions = {
            s.session_id: await self._get_confidence_distribution(s) for s in sessions
        }
        
        # Compare evaluation patterns
        comparison.evaluation_patterns = {
            s.session_id: self._analyze_evaluation_patterns(s) for s in sessions
        }
        
        # Compare context usage efficiency
        comparison.context_efficiency = {
            s.session_id: self._calculate_context_efficiency(s) for s in sessions
        }
        
        # Find overlapping claims between sessions
        comparison.overlapping_claims = await self._find_overlapping_claims(sessions)
        
        return comparison
    
    async def merge_sessions(self, source_session_id: str, target_session_id: str, 
                           merge_strategy: MergeStrategy) -> MergeResult:
        """Merge sessions with configurable strategy"""
        
        source_session = await self.session_manager.get_session(source_session_id)
        target_session = await self.session_manager.get_session(target_session_id)
        
        if not source_session or not target_session:
            raise ValueError("Invalid session IDs")
        
        merger = SessionMerger()
        result = await merger.merge_sessions(source_session, target_session, merge_strategy)
        
        if result.success:
            # Clean up source session if desired
            if merge_strategy.delete_source_after_merge:
                await self.session_manager.terminate_session(source_session_id)
        
        return result
```

### Session Status Visualization

```python
class SessionStatusDisplay:
    def __init__(self, interface_adapter: InterfaceAdapter):
        self.interface_adapter = interface_adapter
        
    async def display_session_status_bar(self, session: Session):
        """Display current session status in interface-appropriate format"""
        
        status_color = self._get_status_color(session.status)
        session_info = f"Session: {session.session_id[:8]} "
        
        # Add context usage indicator
        context_percent = session.context_window.get_usage_percentage()
        context_color = "green" if context_percent < 70 else "yellow" if context_percent < 90 else "red"
        
        # Add evaluation activity indicator
        recent_evaluations = len([
            e for e in session.evaluation_history 
            if (datetime.now() - e.timestamp).total_seconds() < 300  # Last 5 minutes
        ])
        
        evaluation_indicator = "🔄" if recent_evaluations > 0 else "✓"
        
        if isinstance(self.interface_adapter, TUIAdapter):
            await self._display_tui_status_bar(
                session_info, status_color, context_percent, 
                context_color, evaluation_indicator
            )
        elif isinstance(self.interface_adapter, WebUIAdapter):
            await self._display_webui_status_bar(
                session, context_percent, evaluation_indicator
            )
    
    async def display_session_metrics(self, session: Session):
        """Display detailed session metrics"""
        
        metrics = await session.get_session_summary()
        
        # Format metrics for display
        metrics_data = {
            "Session ID": metrics.session_id,
            "Status": metrics.status.value,
            "Uptime": f"{metrics.uptime / 3600:.1f} hours",
            "Active Evaluations": metrics.active_root_claims,
            "Context Usage": f"{metrics.context_usage:.1f}%",
            "Total Evaluations": metrics.evaluation_count,
            "Memory Usage": f"{metrics.resource_usage.memory_mb:.1f} MB",
            "Cache Hit Rate": f"{metrics.resource_usage.cache_hit_rate:.1f}%",
            "Last Activity": metrics.last_activity.strftime("%H:%M:%S")
        }
        
        if isinstance(self.interface_adapter, TUIAdapter):
            await self._display_tui_metrics_table(metrics_data)
        elif isinstance(self.interface_adapter, CLIAdapter):
            await self._display_cli_metrics(metrics_data)
        elif isinstance(self.interface_adapter, WebUIAdapter):
            await self._display_webui_metrics(metrics_data)
```

### Cross-Session Event Isolation

```python
class SessionEventBus:
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self.session_buses: Dict[str, EventBus] = {}
        self.global_bus = EventBus()
        
    async def publish_to_session(self, session_id: str, event: Event):
        """Publish event to specific session only"""
        
        if session_id not in self.session_buses:
            self.session_buses[session_id] = EventBus()
        
        await self.session_buses[session_id].publish(event)
        
        # Also publish to global bus with session context
        global_event = SessionEvent(
            original_event=event,
            session_id=session_id,
            timestamp=datetime.now()
        )
        await self.global_bus.publish(global_event)
    
    async def publish_to_all_sessions(self, event: Event):
        """Publish event to all active sessions"""
        
        active_sessions = await self.session_manager.list_active_sessions()
        
        for session in active_sessions:
            await self.publish_to_session(session.session_id, event)
    
    async def subscribe_to_session_events(self, session_id: str, callback: Callable):
        """Subscribe to events for specific session"""
        
        if session_id not in self.session_buses:
            self.session_buses[session_id] = EventBus()
        
        self.session_buses[session_id].subscribe("session.*", callback)
    
    async def cleanup_session_events(self, session_id: str):
        """Clean up event bus for terminated session"""
        
        if session_id in self.session_buses:
            # Publish session termination event
            await self.session_buses[session_id].publish(
                SessionTerminatedEvent(session_id=session_id)
            )
            
            # Clean up event bus
            del self.session_buses[session_id]
```

---

## Shared Interface Components

### Claim Manager Component

**Purpose**: Centralized claim operations and state management across all interfaces with enhanced skill and example claim support.

**Key Responsibilities**:
- Claim creation, retrieval, and modification
- Skill claim management and execution
- Example claim generation and display
- Claim similarity search and discovery
- Claim relationship management
- Real-time claim status updates

**Interface**:
```python
class ClaimManager:
    # Basic claim operations
    async def create_claim(self, content: str, tags: List[str]) -> Claim
    async def get_claim(self, claim_id: str) -> Claim
    async def search_claims(self, query: str, max_results: int = 10) -> List[Claim]
    async def update_claim(self, claim_id: str, content: str) -> Claim
    async def delete_claim(self, claim_id: str) -> bool
    async def get_similar_claims(self, claim_id: str, threshold: float = 0.8) -> List[Claim]
    
    # Skill claim operations
    async def create_skill_claim(self, content: str, function_signature: str, 
                               parameters: List[str]) -> Claim
    async def get_skill_claims(self, function_name: Optional[str] = None) -> List[Claim]
    async def execute_skill_claim(self, skill_id: str, parameters: Dict) -> SkillExecutionResult
    async def update_skill_examples(self, skill_id: str, examples: List[Claim]) -> bool
    
    # Example claim operations
    async def create_example_claim(self, skill_id: str, input_data: Dict, 
                                 output_data: Dict, execution_metadata: Dict) -> Claim
    async def get_skill_examples(self, skill_id: str, limit: int = 5) -> List[Claim]
    async def auto_generate_examples(self, skill_id: str, count: int = 3) -> List[Claim]
    
    # Enhanced search and discovery
    async def search_claims_by_type(self, claim_type: str, query: str = "", 
                                  limit: int = 10) -> List[Claim]
    async def find_relevant_skills(self, context: str, limit: int = 5) -> List[Claim]
    async def get_claim_examples_for_function(self, function_name: str) -> List[Claim]
    
    def subscribe_to_updates(self, callback: Callable) -> None
```

**State Management**:
- Local claim cache for fast access
- Dirty flag tracking for evaluation status
- Skill execution caching and monitoring
- Example claim generation tracking
- Confidence-based sorting and filtering
- Real-time update subscriptions

### Skill Management Component

**Purpose**: Specialized component for managing skill claims and their execution.

**Key Responsibilities**:
- Skill claim lifecycle management
- Skill execution monitoring and validation
- Example claim creation and management
- Skill-performance tracking and optimization

**Interface**:
```python
class SkillManager:
    async def register_skill(self, skill_claim: Claim) -> bool
    async def execute_skill(self, skill_id: str, parameters: Dict, context: Dict) -> SkillResult
    async def validate_skill_execution(self, execution_result: SkillResult) -> ValidationResult
    async def create_example_from_execution(self, skill_id: str, execution: SkillResult) -> Claim
    async def get_skill_performance_metrics(self, skill_id: str) -> SkillMetrics
    async def optimize_skill_execution(self, skill_id: str) -> OptimizationResult
    
    # Skill discovery and relevance
    async def find_skills_for_context(self, context: str, limit: int = 5) -> List[Claim]
    async def calculate_skill_relevance(self, skill: Claim, context: str) -> float
    async def get_skill_dependencies(self, skill_id: str) -> List[Claim]
```

### Example Claim Management Component

**Purpose**: Specialized component for managing example claims that demonstrate skill execution.

**Key Responsibilities**:
- Example claim creation from successful skill executions
- Example claim search and retrieval
- Example-claim-based learning and improvement
- Example quality assessment and ranking

**Interface**:
```python
class ExampleManager:
    async def create_example_claim(self, skill_id: str, execution_result: SkillResult) -> Claim
    async def get_examples_for_skill(self, skill_id: str, limit: int = 5) -> List[Claim]
    async def search_examples_by_pattern(self, pattern: Dict, limit: int = 10) -> List[Claim]
    async def assess_example_quality(self, example_claim: Claim) -> QualityScore
    async def rank_examples_by_relevance(self, examples: List[Claim], context: str) -> List[Claim]
    async def generate_code_examples(self, skill_id: str, variations: int = 3) -> List[Claim]
    
    # Learning and improvement
    async def learn_from_examples(self, skill_id: str) -> LearningResult
    async def suggest_example_improvements(self, example_id: str) -> List[Suggestion]
    async def detect_example_anomalies(self, examples: List[Claim]) -> List[Anomaly]
```

### Confidence Manager Component

**Purpose**: Centralized confidence evaluation and tracking across all interfaces.

**Key Responsibilities**:
- Confidence score calculation and tracking
- Priority-based claim evaluation
- Confidence history visualization
- Dirty flag management for claims needing evaluation

**Interface**:
```python
class ConfidenceManager:
    async def evaluate_claim(self, claim_id: str, force: bool = False) -> float
    async def get_dirty_claims(self, count: int = 10) -> List[Claim]
    async def mark_claim_dirty(self, claim_id: str) -> None
    async def get_claim_history(self, claim_id: str) -> List[ConfidenceHistory]
    def is_being_evaluated(self, claim_id: str) -> bool
    def get_evaluation_priority(self, claim: Claim) -> float
```

### Relationship Manager Component

**Purpose**: Centralized relationship management across all interfaces.

**Key Responsibilities**:
- Support relationship creation and management
- Claim dependency tracking
- Evidence aggregation
- Relationship impact analysis

**Interface**:
```python
class RelationshipManager:
    async def add_supports_relationship(self, supporter_id: str, supported_id: str) -> bool
    async def remove_supports_relationship(self, supporter_id: str, supported_id: str) -> bool
    async def get_supporting_claims(self, claim_id: str) -> List[Claim]
    async def get_supported_claims(self, claim_id: str) -> List[Claim]
    async def get_related_claims(self, claim_id: str, depth: int = 2) -> List[Claim]
    async def get_relationship_impact(self, claim_id: str) -> Dict[str, int]
```

---

## Terminal User Interface (TUI) Design

### Layout Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Conjecture v1.0 │ Knowledge: 1,247 claims │ Eval: 87/123 │
├─────────────────────────────────────────────────────────────┤
│ Input: [Search query or claim text                    │GO] │
├─────────┬───────────────────────────────────────┬───────────┤
│ Claims   │ Claim Details                            │ Similar  │
│(123 total)│                                         │ Claims   │
├─────────┤ ID: c0001234                           ├───────────┤
│ ⚬ c00123 │ Content: Quantum entanglement...       │ ⚬ c000567 │
│ ? c00124 │ Confidence: 0.73 [███████░░]          │ ⚬ c000812 │
│ * c00125 │ Status: ✓ Evaluated                    │ ⚬ c001456 │
│ ⚬ c00126 │ Tags: [physics, quantum]               │ ⚬ c000999 │
│ ⚬ c00127 │                                         │           │
│ ? c00128 │ Relationships (3):                      │ ⚬ c001234 │
│ ⚬ c00129 │   • supports: c000456                  │ ⚬ c001567 │
│ * c00130 │   • supports: c000789                  │           │
│ ⚬ c00131 │   • supported_by: c000234             ├───────────┤
│ ⚬ c00132 │                                         │ Goals     │
│ ? c00133 │ Evidence (2):                           │           │
│ ⚬ c00134 │   • [0.83] Research paper... (c000456) │ ⚬ g00001 │
│ ⚬ c00135 │   • [0.71] Experiment... (c000789)    │ ⚬ g00002 │
├─────────┴───────────────────────────────────────┴───────────┤
│ Status: Ready │ Claims: 1,247 │ Dirty: 36/123 │ Eval: 87/123 │
└─────────────────────────────────────────────────────────────┘
```

### Key Interactions

1. **Keyboard Navigation**
   - Arrow keys to navigate between panels
   - Tab to cycle through focused elements
   - Enter to select/expand items
   - `/` to focus search input
   - `c` to create new claim
   - `g` to view goals
   - `e` to force evaluation of dirty claims

2. **Real-Time Features**
   - Dirty flag indicators (question mark icon)
   - Live confidence bars showing evaluation progress
   - Auto-refresh of claim list when evaluations complete
   - Visual priority highlighting of dirty claims

3. **Panel Interactions**
   - Claims panel shows status icons and confidence bars
   - Details panel expands with relationships and evidence
   - Similar claims update based on selected claim
   - Goals panel tracks progress on claim-based objectives

### Component Hierarchy

```python
class TUIApplication:
    - Header: ApplicationHeader
    - InputPanel: CommandInput
    - MainLayout: SplitLayout
        - ClaimsPanel: ClaimListPanel
        - DetailsPanel: ClaimDetailsPanel
        - SidePanel: VerticalLayout
            - SimilarPanel: ClaimListPanel
            - GoalsPanel: GoalListPanel
    - StatusBar: StatusBar
```

### State Management Pattern

```python
class TUIState:
    def __init__(self):
        self.selected_claim_id: Optional[str] = None
        self.search_query: str = ""
        self.claim_filter: ClaimFilter = ClaimFilter.ALL
        self.dirty_claim_ids: Set[str] = set()
        self.evaluation_in_progress: Set[str] = set()
        self.side_panel_focus: PanelType = PanelType.SIMILAR
    
    def select_claim(self, claim_id: str) -> None:
        self.selected_claim_id = claim_id
        # Trigger side panels update
    
    def update_dirty_claims(self, dirty_ids: Set[str]) -> None:
        self.dirty_claim_ids = dirty_ids
        # Update UI components
    
    def mark_evaluation_started(self, claim_id: str) -> None:
        self.evaluation_in_progress.add(claim_id)
        # Update UI components
```

---

## Command Line Interface (CLI) Design

### Command Structure

```bash
# Primary claim operations
conjecture create "Quantum entanglement enables faster-than-light communication" --tags="physics,quantum"
conjecture search "quantum entanglement" --limit=5 --format=json
conjecture get c0001234 --format=yaml
conjecture update c0001234 "Quantum entanglement enables quantum communication"

# Evidence and relationships
conjecture support c0001234 --supports=c000456
conjecture support c0001234 --supported-by=c000789
conjecture evidence c0001234 --show

# Goal management
conjecture goal "Complete quantum entanglement research" --tags="research,physics"
conjecture goals --status=incomplete
conjecture goal g000001 --tasks

# Evaluation system
conjecture evaluate --priority --limit=10
conjecture evaluate c0001234 --force
conjecture dirty --count=15
conjecture status --show-evaluation

# Configuration and utilities
conjecture config get llm.provider
conjecture config set ui.theme dark
conjecture export --format=json --output=claims.json
```

### Output Formatting

1. **Human-Readable Output** (default)
```
┌───────────────────────────────────────────────┐
│ c0001234 │ 0.73 ████████░ │ quantum, physics │
├───────────────────────────────────────────────┤
│ Quantum entanglement enables quantum...       │
│                                               │
│ Evidence:                                     │
│   • [0.83] Research paper... (c000456)       │
│   • [0.71] Experiment... (c000789)           │
└───────────────────────────────────────────────┘
```

2. **JSON Output** (for integration)
```json
{
  "claim": {
    "id": "c0001234",
    "content": "Quantum entanglement enables quantum communication",
    "confidence": 0.73,
    "tags": ["quantum", "physics"],
    "dirty": false,
    "evidence": [
      {
        "claim_id": "c000456",
        "content": "Research paper...",
        "confidence": 0.83
      }
    ]
  }
}
```

3. **YAML Output** (for configuration)
```yaml
claim:
  id: c0001234
  content: Quantum entanglement enables quantum communication
  confidence: 0.73
  tags:
  - quantum
  - physics
  dirty: false
  evidence:
  - claim_id: c000456
    content: Research paper...
    confidence: 0.83
```

### Command Processing Pipeline

```python
class CLIProcessor:
    def __init__(self, claim_manager, confidence_manager, relationship_manager):
        self.claim_manager = claim_manager
        self.confidence_manager = confidence_manager
        self.relationship_manager = relationship_manager
    
    async def process_command(self, args) -> str:
        if args.command == "create":
            claim = await self.claim_manager.create_claim(args.statement, args.tags)
            return self._format_claim(claim, args.format)
        
        elif args.command == "evaluate":
            if args.priority:
                claims = await self.confidence_manager.get_dirty_claims(args.limit)
                results = []
                for claim in claims:
                    confidence = await self.confidence_manager.evaluate_claim(claim.id, force=True)
                    results.append((claim, confidence))
                return self._format_evaluation_results(results, args.format)
            
        # ... other command implementations
```

---

## Model Context Protocol (MCP) Design

### Protocol Actions

```python
class MCPActions:
    async def claim(self, statement: str) -> List[Claim]:
        """Create claim(s) from statement using LLM processing"""
        return await self.claim_manager.create_claims_from_text(statement)
    
    async def prompt(self, question: str) -> List[Claim]:
        """Ask question about knowledge base"""
        claims = await self.claim_manager.search_claims(question)
        response = await self.llm_processor.generate_response(question, claims)
        return response.claims
    
    async def inspect(self, query: str, count: int = 5) -> List[Claim]:
        """Return N claims relevant to query"""
        return await self.claim_manager.search_claims(query, max_results=count)
    
    async def evaluate(self, claim_id: str) -> float:
        """Force evaluation of a specific claim"""
        return await self.confidence_manager.evaluate_claim(claim_id, force=True)
    
    async def status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "total_claims": await self.claim_manager.get_count(),
            "dirty_claims": len(await self.confidence_manager.get_dirty_claims()),
            "average_confidence": await self.confidence_manager.get_average_confidence(),
            "evaluation_progress": await self.confidence_manager.get_evaluation_progress()
        }
```

### Context-Aware Claim Generation

```python
class ContextAwareProcessor:
    async def process_contextual_claim(self, statement: str, context: Dict) -> List[Claim]:
        """Process claims with contextual reference support"""
        # Extract contextual references like nc001, nc002, etc.
        contextual_refs = self._extract_contextual_references(statement)
        
        # Map contextual references to global claim IDs
        ref_mapping = {}
        for ref in contextual_refs:
            global_id = await self._resolve_contextual_reference(ref, context)
            if global_id:
                ref_mapping[ref] = global_id
        
        # Process with mapped references
        return await self._process_with_mapped_references(statement, ref_mapping)
```

### Real-Time Synchronization

```python
class MCPEventStreamer:
    def __init__(self, event_bus):
        self.event_bus = event_bus
    
    async def stream_evaluation_progress(self, connection):
        """Stream real-time evaluation updates to AI assistant"""
        async for event in self.event_bus.subscribe("evaluation.progress"):
            await connection.send(json.dumps({
                "type": "evaluation.update",
                "claim_id": event.claim_id,
                "confidence": event.confidence,
                "status": event.status
            }))
    
    async def stream_claim_updates(self, connection):
        """Stream new claim additions to AI assistant"""
        async for event in self.event_bus.subscribe("claim.created"):
            await connection.send(json.dumps({
                "type": "claim.created",
                "claim": event.claim.to_dict()
            }))
```

---

## Web User Interface (WebUI) Design

### Enhanced Component Architecture with Multi-Session Support

```javascript
// React.js component structure with session management
const App = () => {
  const [currentSession, setCurrentSession] = useState(null);
  const [availableSessions, setAvailableSessions] = useState([]);
  const [sessionManager] = useState(() => new SessionManager());
  
  return (
    <SessionProvider sessionManager={sessionManager}>
      <EventBusProvider>
        <SessionHeader 
          currentSession={currentSession}
          onSessionSwitch={setCurrentSession}
        />
        <SplitLayout>
          <SessionPanel 
            currentSession={currentSession}
            onCreateSession={sessionManager.createSession}
            onSessionDelete={sessionManager.deleteSession}
          />
          <ClaimsPanel currentSession={currentSession} />
          <DetailsPanel currentSession={currentSession} />
          <SidePanel>
            <SkillExecutionPanel currentSession={currentSession} />
            <ExampleClaimsPanel currentSession={currentSession} />
            <SimilarClaimsPanel currentSession={currentSession} />
            <GoalsPanel currentSession={currentSession} />
          </SidePanel>
        </SplitLayout>
        <SessionStatusBar currentSession={currentSession} />
      </EventBusProvider>
    </SessionProvider>
  );
};

// Session Management Context
const SessionContext = createContext();

export const SessionProvider = ({ children, sessionManager }) => {
  const [currentSession, setCurrentSession] = useState(null);
  const [sessions, setSessions] = useState([]);
  
  useEffect(() => {
    // Load available sessions
    sessionManager.listSessions().then(setSessions);
  }, [sessionManager]);
  
  const switchSession = async (sessionId) => {
    const session = await sessionManager.switchToSession(sessionId);
    setCurrentSession(session);
    return session;
  };
  
  return (
    <SessionContext.Provider value={{
      currentSession,
      sessions,
      switchSession,
      createSession: sessionManager.createSession,
      deleteSession: sessionManager.deleteSession
    }}>
      {children}
    </SessionContext.Provider>
  );
};
```

### Multi-Session UI Components

#### Session Switcher Component
```javascript
const SessionSwitcher = () => {
  const { currentSession, sessions, switchSession } = useContext(SessionContext);
  const [isOpen, setIsOpen] = useState(false);
  
  return (
    <div className="session-switcher">
      <button 
        className="session-current-button" 
        onClick={() => setIsOpen(!isOpen)}
      >
        <SessionIcon />
        <span>{currentSession?.name || 'No Session'}</span>
        <StatusIndicator status={currentSession?.status} />
      </button>
      
      {isOpen && (
        <div className="session-dropdown">
          <div className="session-header">
            <h3>Active Sessions</h3>
            <button onClick={() => setIsOpen(false)}>×</button>
          </div>
          <div className="session-list">
            {sessions.map(session => (
              <SessionItem
                key={session.id}
                session={session}
                isActive={session.id === currentSession?.id}
                onSelect={() => {
                  switchSession(session.id);
                  setIsOpen(false);
                }}
              />
            ))}
          </div>
          <div className="session-footer">
            <button onClick={handleNewSession}>
              <PlusIcon /> New Session
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

const SessionItem = ({ session, isActive, onSelect }) => (
  <div 
    className={`session-item ${isActive ? 'active' : ''}`}
    onClick={onSelect}
  >
    <div className="session-info">
      <span className="session-name">{session.name}</span>
      <span className="session-stats">
        {session.claimCount} claims • {session.evaluationCount} evaluations
      </span>
    </div>
    <div className="session-indicators">
      <ConfidenceBar confidence={session.confidence} />
      <ActivityIndicator lastActivity={session.lastActivity} />
    </div>
  </div>
);
```

#### Session Comparison Interface
```javascript
const SessionComparator = () => {
  const { sessions } = useContext(SessionContext);
  const [selectedSessions, setSelectedSessions] = useState([]);
  const [comparison, setComparison] = useState(null);
  
  const handleCompare = async () => {
    if (selectedSessions.length >= 2) {
      const result = await sessionComparator.compare(selectedSessions);
      setComparison(result);
    }
  };
  
  return (
    <div className="session-comparator">
      <div className="comparison-selector">
        <h3>Select Sessions to Compare</h3>
        <div className="session-checkboxes">
          {sessions.map(session => (
            <label key={session.id} className="session-checkbox">
              <input
                type="checkbox"
                checked={selectedSessions.includes(session.id)}
                onChange={(e) => {
                  if (e.target.checked) {
                    setSelectedSessions([...selectedSessions, session.id]);
                  } else {
                    setSelectedSessions(selectedSessions.filter(id => id !== session.id));
                  }
                }}
              />
              <div className="session-preview">
                <span className="session-name">{session.name}</span>
                <span className="session-uptime">{formatUptime(session.uptime)}</span>
              </div>
            </label>
          ))}
        </div>
      </div>
      
      {selectedSessions.length >= 2 && (
        <button className="compare-button" onClick={handleCompare}>
          Compare Sessions
        </button>
      )}
      
      {comparison && (
        <div className="comparison-results">
          <h4>Comparison Results</h4>
          <SessionComparisonChart data={comparison} />
          <ComparisonMetrics metrics={comparison.metrics} />
          <OverlapAnalysis overlaps={comparison.overlapping_claims} />
        </div>
      )}
    </div>
  );
};
```

### Session Management Workflow Integration

#### Session Creation Dialog
```javascript
const CreateSessionDialog = ({ isOpen, onClose, onCreate }) => {
  const [sessionConfig, setSessionConfig] = useState({
    name: '',
    isolation_level: 'shared_db',
    import_from_session: null,
    checkpoint_url: '',
    resource_limits: {
      max_claims: 10000,
      max_context_tokens: 4096
    }
  });
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    const newSession = await onCreate(sessionConfig);
    if (newSession) {
      onClose();
    }
  };
  
  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <form onSubmit={handleSubmit} className="create-session-form">
        <h2>Create New Session</h2>
        
        <FormField label="Session Name">
          <input
            type="text"
            value={sessionConfig.name}
            onChange={(e) => setSessionConfig({
              ...sessionConfig,
              name: e.target.value
            })}
            required
          />
        </FormField>
        
        <FormField label="Isolation Level">
          <select
            value={sessionConfig.isolation_level}
            onChange={(e) => setSessionConfig({
              ...sessionConfig,
              isolation_level: e.target.value
            })}
          >
            <option value="shared_db">Shared Database (Recommended)</option>
            <option value="full">Full Isolation</option>
            <option value="community">Community Checkpoint</option>
          </select>
        </FormField>
        
        {sessionConfig.isolation_level === 'community' && (
          <FormField label="Community Checkpoint URL">
            <input
              type="url"
              value={sessionConfig.checkpoint_url}
              onChange={(e) => setSessionConfig({
                ...sessionConfig,
                checkpoint_url: e.target.value
              })}
              placeholder="https://example.com/checkpoint.json"
            />
          </FormField>
        )}
        
        <FormField label="Import from Existing Session">
          <select
            value={sessionConfig.import_from_session || ''}
            onChange={(e) => setSessionConfig({
              ...sessionConfig,
              import_from_session: e.target.value || null
            })}
          >
            <option value="">Start Fresh</option>
            {/* Dynamically populate with available sessions */}
          </select>
        </FormField>
        
        <ResourceLimitForm 
          limits={sessionConfig.resource_limits}
          onChange={(limits) => setSessionConfig({
            ...sessionConfig,
            resource_limits: limits
          })}
        />
        
        <div className="form-actions">
          <button type="button" onClick={onClose}>Cancel</button>
          <button type="submit" className="primary">Create Session</button>
        </div>
      </form>
    </Modal>
  );
};
```

#### Session Merging Interface
```javascript
const SessionMerger = () => {
  const [sessions, setSessions] = useState([]);
  const [sourceSession, setSourceSession] = useState(null);
  const [targetSession, setTargetSession] = useState(null);
  const [mergeStrategy, setMergeStrategy] = useState('preserve_both');
  const [mergePreview, setMergePreview] = useState(null);
  
  const generatePreview = async () => {
    if (sourceSession && targetSession) {
      const preview = await sessionMerger.previewMerge(
        sourceSession.id,
        targetSession.id,
        mergeStrategy
      );
      setMergePreview(preview);
    }
  };
  
  const executeMerge = async () => {
    try {
      const result = await sessionMerger.executeMerge(
        sourceSession.id,
        targetSession.id,
        mergeStrategy
      );
      
      // Show success and update session list
      showSuccess('Session merge completed successfully');
      await refreshSessions();
    } catch (error) {
      showError('Merge failed: ' + error.message);
    }
  };
  
  return (
    <div className="session-merger">
      <h2>Merge Sessions</h2>
      
      <div className="merge-selector">
        <div className="session-selector">
          <label>Source Session (to merge from)</label>
          <SessionDropdown 
            sessions={sessions}
            selected={sourceSession}
            onSelect={setSourceSession}
            exclude={targetSession?.id}
          />
        </div>
        
        <div className="merge-arrow">→</div>
        
        <div className="session-selector">
          <label>Target Session (to merge into)</label>
          <SessionDropdown 
            sessions={sessions}
            selected={targetSession}
            onSelect={setTargetSession}
            exclude={sourceSession?.id}
          />
        </div>
      </div>
      
      <div className="merge-strategy">
        <label>Merge Strategy</label>
        <select
          value={mergeStrategy}
          onChange={(e) => setMergeStrategy(e.target.value)}
        >
          <option value="preserve_both">Preserve Both Sessions</option>
          <option value="delete_source">Delete Source After Merge</option>
          <option value="intelligent">Intelligent Merging</option>
        </select>
      </div>
      
      <button 
        onClick={generatePreview}
        disabled={!sourceSession || !targetSession}
        className="preview-button"
      >
        Preview Merge
      </button>
      
      {mergePreview && (
        <div className="merge-preview">
          <h3>Merge Preview</h3>
          <MergeConflicts conflicts={mergePreview.conflicts} />
          <MergeStatistics stats={mergePreview.statistics} />
          <ConflictResolution 
            conflicts={mergePreview.conflicts}
            onResolve={setResolvedConflicts}
          />
          
          <div className="merge-actions">
            <button onClick={executeMerge} className="execute-merge">
              Execute Merge
            </button>
            <button onClick={() => setMergePreview(null)}>
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
```

### Real-Time Features

1. **WebSocket Integration with Tool Call Support**
```javascript
const useRealTimeUpdates = () => {
  const [dirtyClaims, setDirtyClaims] = useState([]);
  const [evaluationInProgress, setEvaluationInProgress] = useState(new Set());
  const [activeToolCalls, setActiveToolCalls] = useState(new Map());
  const [skillExecutions, setSkillExecutions] = useState(new Map());
  
  useEffect(() => {
    const ws = new WebSocket('/api/ws');
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      switch (data.type) {
        case 'claims.dirty':
          setDirtyClaims(data.claim_ids);
          break;
        
        case 'evaluation.started':
          setEvaluationInProgress(prev => new Set([...prev, data.claim_id]));
          break;
        
        case 'evaluation.completed':
          setEvaluationInProgress(prev => {
            const next = new Set(prev);
            next.delete(data.claim_id);
            return next;
          });
          break;
          
        case 'tool_call.started':
          setActiveToolCalls(prev => new Map(prev.set(data.tool_call_id, {
            ...data,
            status: 'running',
            start_time: new Date()
          })));
          break;
          
        case 'tool_call.completed':
          setActiveToolCalls(prev => {
            const updated = new Map(prev);
            const existing = updated.get(data.tool_call_id);
            if (existing) {
              updated.set(data.tool_call_id, {
                ...existing,
                status: 'completed',
                result: data.result,
                end_time: new Date()
              });
            }
            return updated;
          });
          break;
          
        case 'skill.execution.started':
          setSkillExecutions(prev => new Map(prev.set(data.skill_id, {
            ...data,
            status: 'executing',
            start_time: new Date()
          })));
          break;
          
        case 'skill.execution.completed':
          setSkillExecutions(prev => {
            const updated = new Map(prev);
            const existing = updated.get(data.skill_id);
            if (existing) {
              updated.set(data.skill_id, {
                ...existing,
                status: 'completed',
                examples_created: data.examples_created,
                end_time: new Date()
              });
            }
            return updated;
          });
          break;
          
        case 'example.claim.created':
          // Handle real-time example claim creation
          break;
      }
    };
    
    return () => ws.close();
  }, []);
  
  return { 
    dirtyClaims, 
    evaluationInProgress, 
    activeToolCalls, 
    skillExecutions 
  };
};
```

2. **Real-Time Tool Call Visualization Component**
```javascript
const ToolCallMonitor = () => {
  const { activeToolCalls } = useRealTimeUpdates();
  
  return (
    <div className="tool-call-monitor">
      <h3>Active Tool Calls</h3>
      <div className="tool-call-list">
        {Array.from(activeToolCalls.values()).map(toolCall => (
          <ToolCallItem key={toolCall.tool_call_id} toolCall={toolCall} />
        ))}
      </div>
    </div>
  );
};

const ToolCallItem = ({ toolCall }) => {
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);
  
  useEffect(() => {
    if (toolCall.status === 'running') {
      const interval = setInterval(() => {
        const elapsed = Date.now() - new Date(toolCall.start_time).getTime();
        setDuration(elapsed);
        // Simulate progress based on duration
        setProgress(Math.min((elapsed / 10000) * 100, 95)); // Assume 10s max
      }, 100);
      
      return () => clearInterval(interval);
    } else if (toolCall.status === 'completed') {
      setProgress(100);
      setDuration(new Date(toolCall.end_time).getTime() - new Date(toolCall.start_time).getTime());
    }
  }, [toolCall]);
  
  return (
    <div className={`tool-call-item ${toolCall.status}`}>
      <div className="tool-call-header">
        <span className="function-name">{toolCall.function}</span>
        <span className="status-badge">{toolCall.status}</span>
        <span className="duration">{formatDuration(duration)}</span>
      </div>
      
      <div className="tool-call-progress">
        <ProgressBar progress={progress} />
      </div>
      
      <div className="tool-call-details">
        <details>
          <summary>Parameters</summary>
          <pre className="parameters">{JSON.stringify(toolCall.parameters, null, 2)}</pre>
        </details>
        
        {toolCall.status === 'completed' && toolCall.result && (
          <details>
            <summary>Result</summary>
            <pre className="result">{JSON.stringify(toolCall.result, null, 2)}</pre>
          </details>
        )}
      </div>
    </div>
  );
};
```

3. **Skill Execution Monitoring Component**
```javascript
const SkillExecutionMonitor = () => {
  const { skillExecutions, activeToolCalls } = useRealTimeUpdates();
  
  return (
    <div className="skill-execution-monitor">
      <h3>Skill Executions</h3>
      <div className="skill-execution-list">
        {Array.from(skillExecutions.values()).map(execution => (
          <SkillExecutionItem 
            key={execution.skill_id} 
            execution={execution}
            relatedToolCalls={Array.from(activeToolCalls.values()).filter(
              tc => tc.skill_id === execution.skill_id
            )}
          />
        ))}
      </div>
    </div>
  );
};

const SkillExecutionItem = ({ execution, relatedToolCalls }) => {
  return (
    <div className={`skill-execution-item ${execution.status}`}>
      <div className="skill-execution-header">
        <SkillIcon />
        <span className="skill-name">{execution.skill_name}</span>
        <StatusIndicator status={execution.status} />
        <span className="execution-time">
          {formatDuration(
            new Date(execution.end_time || Date.now()).getTime() - 
            new Date(execution.start_time).getTime()
          )}
        </span>
      </div>
      
      <div className="execution-details">
        <div className="related-tool-calls">
          <h4>Tool Calls ({relatedToolCalls.length})</h4>
          <div className="tool-call-mini-list">
            {relatedToolCalls.map(toolCall => (
              <ToolCallMini key={toolCall.tool_call_id} toolCall={toolCall} />
            ))}
          </div>
        </div>
        
        {execution.examples_created && execution.examples_created.length > 0 && (
          <div className="examples-created">
            <h4>Example Claims Created</h4>
            <div className="example-list">
              {execution.examples_created.map(example => (
                <ExampleCard key={example.id} example={example} />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const ToolCallMini = ({ toolCall }) => (
  <div className={`tool-call-mini ${toolCall.status}`}>
    <span className="function-name">{toolCall.function}</span>
    <StatusIndicator status={toolCall.status} size="small" />
  </div>
);
```

4. **Example Claim Creation Visualization**
```javascript
const ExampleCreationStream = () => {
  const [examples, setExamples] = useState([]);
  const socket = useWebSocket();
  
  useEffect(() => {
    if (socket) {
      socket.subscribe('example.claim.created', (exampleData) => {
        setExamples(prev => [exampleData, ...prev.slice(0, 9)]); // Keep last 10
      });
    }
  }, [socket]);
  
  return (
    <div className="example-creation-stream">
      <h3>Recent Example Claims</h3>
      <div className="example-stream">
        {examples.map(example => (
          <ExampleStreamItem key={example.id} example={example} />
        ))}
      </div>
    </div>
  );
};

const ExampleStreamItem = ({ example }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  
  return (
    <div className="example-stream-item">
      <div className="example-header" onClick={() => setIsExpanded(!isExpanded)}>
        <span className="example-id">{example.id}</span>
        <span className="skill-id">Skill: {example.skill_id}</span>
        <span className="creation-time">{formatTime(example.created_at)}</span>
      </div>
      
      {isExpanded && (
        <div className="example-details">
          <div className="example-content">
            <h4>Content</h4>
            <p>{example.content}</p>
          </div>
          
          <div className="example-metadata">
            <h4>Execution Metadata</h4>
            <div className="metadata-grid">
              <div className="metadata-item">
                <label>Function:</label>
                <span>{example.function_name}</span>
              </div>
              <div className="metadata-item">
                <label>Input:</label>
                <pre>{JSON.stringify(example.input_data, null, 2)}</pre>
              </div>
              <div className="metadata-item">
                <label>Output:</label>
                <pre>{JSON.stringify(example.output_data, null, 2)}</pre>
              </div>
            </div>
          </div>
          
          <div className="example-actions">
            <button onClick={() => viewClaimDetails(example.id)}>
              View Full Details
            </button>
            <button onClick={() => useExampleAsTemplate(example)}>
              Use as Template
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
```

5. **Live Evaluation Dashboard**
```javascript
const LiveEvaluationDashboard = () => {
  const { evaluationInProgress, activeToolCalls, skillExecutions } = useRealTimeUpdates();
  
  return (
    <div className="live-evaluation-dashboard">
      <DashboardHeader />
      
      <div className="dashboard-grid">
        <div className="dashboard-section">
          <h3>Evaluation Status</h3>
          <div className="status-grid">
            <StatusCard
              title="Claims Evaluating"
              count={evaluationInProgress.size}
              color="blue"
            />
            <StatusCard
              title="Tool Calls Active"
              count={activeToolCalls.size}
              color="green"
            />
            <StatusCard
              title="Skills Executing"
              count={skillExecutions.size}
              color="purple"
            />
          </div>
        </div>
        
        <div className="dashboard-section">
          <h3>Current Activity</h3>
          <ActivityFeed />
        </div>
        
        <div className="dashboard-section">
          <h3>Performance Metrics</h3>
          <PerformanceMetrics />
        </div>
        
        <div className="dashboard-section">
          <h3>Resource Usage</h3>
          <ResourceUsageWidget />
        </div>
      </div>
    </div>
  );
};

const ActivityFeed = () => {
  const [activities, setActivities] = useState([]);
  const socket = useWebSocket();
  
  useEffect(() => {
    if (socket) {
      socket.subscribe('evaluation.*', (data) => {
        setActivities(prev => [data, ...prev.slice(0, 49)]); // Keep last 50
      });
    }
  }, [socket]);
  
  return (
    <div className="activity-feed">
      {activities.map(activity => (
        <ActivityItem key={activity.timestamp} activity={activity} />
      ))}
    </div>
  );
};

const ActivityItem = ({ activity }) => {
  return (
    <div className="activity-item">
      <ActivityIcon type={activity.type} />
      <div className="activity-content">
        <span className="activity-message">{formatActivityMessage(activity)}</span>
        <span className="activity-time">{formatRelativeTime(activity.timestamp)}</span>
      </div>
    </div>
  );
};
```

2. **Visual Indicators**
```javascript
const ClaimListItem = ({ claim, isSelected }) => {
  const { evaluationInProgress } = useRealTimeUpdates();
  const isEvaluating = evaluationInProgress.has(claim.id);
  
  return (
    <ClaimItem selected={isSelected}>
      <StatusIcon>
        {claim.dirty && <DirtyIndicator />}
        {isEvaluating && <Spinner size="small" />}
      </StatusIcon>
      <ClaimContent>{claim.content}</ClaimContent>
      <ConfidenceBar confidence={claim.confidence} />
    </ClaimItem>
  );
};
```

### Collaboration Features

1. **Real-Time Collaborative Editing**
```javascript
const CollaborativeClaimEditor = ({ claimId }) => {
  const [claim, setClaim] = useState(null);
  const [collaborators, setCollaborators] = useState([]);
  
  useEffect(() => {
    const socket = io(`/claim/${claimId}`);
    
    socket.on('claim.updated', (updatedClaim) => {
      setClaim(updatedClaim);
    });
    
    socket.on('user.joined', (user) => {
      setCollaborators(prev => [...prev, user]);
    });
    
    socket.on('user.left', (userId) => {
      setCollaborators(prev => prev.filter(u => u.id !== userId));
    });
    
    return () => socket.disconnect();
  }, [claimId]);
  
  const handleClaimUpdate = (newContent) => {
    socket.emit('claim.update', { content: newContent });
  };
  
  // ... component JSX
};
```

2. **Relationship Visualization**
```javascript
const RelationshipGraph = ({ claimId }) => {
  const [graphData, setGraphData] = useState(null);
  
  useEffect(() => {
    api.getRelationshipGraph(claimId).then(setGraphData);
  }, [claimId]);
  
  return (
    <ForceGraph
      graphData={graphData}
      nodeLabel="content"
      nodeColor="confidence"
      linkLabel="supports"
      onNodeClick={handleNodeClick}
    />
  );
};
```

---

## Cross-Interface Event System

### Event Bus Architecture

```python
# Core event bus implementation
class EventBus:
    def __init__(self):
        self._subscribers = defaultdict(list)
        self._event_queue = asyncio.Queue()
    
    def subscribe(self, event_type: str, callback: Callable) -> None:
        self._subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        self._subscribers[event_type].remove(callback)
    
    async def publish(self, event: Event) -> None:
        await self._event_queue.put(event)
    
    async def process_events(self) -> None:
        while True:
            event = await self._event_queue.get()
            
            # Notify subscribers
            for callback in self._subscribers[event.type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Error in event subscriber: {e}")
            
            self._event_queue.task_done()
```

### Event Types

```python
# Claim-related events
class ClaimCreatedEvent(Event):
    type = "claim.created"
    def __init__(self, claim: Claim):
        self.claim = claim
        self.timestamp = datetime.now()

class ClaimUpdatedEvent(Event):
    type = "claim.updated"
    def __init__(self, claim_id: str, changes: Dict):
        self.claim_id = claim_id
        self.changes = changes
        self.timestamp = datetime.now()

# Evaluation-related events
class ClaimsMarkedDirtyEvent(Event):
    type = "claims.dirty"
    def __init__(self, claim_ids: List[str]):
        self.claim_ids = claim_ids
        self.timestamp = datetime.now()

class EvaluationStartedEvent(Event):
    type = "evaluation.started"
    def __init__(self, claim_id: str):
        self.claim_id = claim_id
        self.timestamp = datetime.now()

class EvaluationCompletedEvent(Event):
    type = "evaluation.completed"
    def __init__(self, claim_id: str, confidence: float):
        self.claim_id = claim_id
        self.confidence = confidence
        self.timestamp = datetime.now()

# Relationship events
class RelationshipCreatedEvent(Event):
    type = "relationship.created"
    def __init__(self, supporter_id: str, supported_id: str):
        self.supporter_id = supporter_id
        self.supported_id = supported_id
        self.timestamp = datetime.now()
```

### Interface Integration

```python
# TUI event integration
class TUIEventAdapter:
    def __init__(self, app, event_bus):
        self.app = app
        self.event_bus = event_bus
        self._setup_subscriptions()
    
    def _setup_subscriptions(self):
        self.event_bus.subscribe("claims.dirty", self._handle_claims_dirty)
        self.event_bus.subscribe("evaluation.completed", self._handle_evaluation_completed)
    
    def _handle_claims_dirty(self, event: ClaimsMarkedDirtyEvent):
        # Update claim list highlighting
        self.app.claims_panel.highlight_dirty_claims(event.claim_ids)
        
        # Update status bar
        self.app.status_bar.update_dirty_count(len(event.claim_ids))
    
    def _handle_evaluation_completed(self, event: EvaluationCompletedEvent):
        # Update claim confidence display
        self.app.claims_panel.update_claim_confidence(event.claim_id, event.confidence)
        
        # Update detail panel if this claim is selected
        if self.app.selected_claim_id == event.claim_id:
            self.app.details_panel.refresh()

# WebUI event integration (WebSocket)
class WebUIEventAdapter:
    def __init__(self, event_bus, websocket_manager):
        self.event_bus = event_bus
        self.websocket_manager = websocket_manager
        self._setup_subscriptions()
    
    def _setup_subscriptions(self):
        self.event_bus.subscribe("claims.dirty", self._handle_claims_dirty)
        self.event_bus.subscribe("evaluation.completed", self._handle_evaluation_completed)
    
    def _handle_claims_dirty(self, event: ClaimsMarkedDirtyEvent):
        self.websocket_manager.broadcast({
            "type": "claims.dirty",
            "claim_ids": event.claim_ids
        })
    
    def _handle_evaluation_completed(self, event: EvaluationCompletedEvent):
        self.websocket_manager.broadcast({
            "type": "evaluation.completed",
            "claim_id": event.claim_id,
            "confidence": event.confidence
        })
```

---

## Performance Optimization Strategies

### Lazy Loading and Caching

```python
# Claim caching with invalidation
class ClaimCacheManager:
    def __init__(self, claim_manager, event_bus):
        self.claim_manager = claim_manager
        self.event_bus = event_bus
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Setup cache invalidation
        self._setup_invalidation()
    
    async def get_claim(self, claim_id: str) -> Claim:
        if claim_id in self._cache:
            self._cache_hits += 1
            return self._cache[claim_id]
        
        self._cache_misses += 1
        claim = await self.claim_manager.get_claim(claim_id)
        self._cache[claim_id] = claim
        return claim
    
    def _setup_invalidation(self):
        self.event_bus.subscribe("claim.updated", self._invalidate_claim)
        self.event_bus.subscribe("evaluation.completed", self._invalidate_claim)
    
    def _invalidate_claim(self, event):
        if isinstance(event, ClaimUpdatedEvent):
            claim_id = event.claim_id
        elif isinstance(event, EvaluationCompletedEvent):
            claim_id = event.claim_id
        else:
            return
        
        if claim_id in self._cache:
            del self._cache[claim_id]
```

### Batch Processing for Performance

```python
# Batch evaluation for UI responsiveness
class BatchEvaluationProcessor:
    def __init__(self, confidence_manager, event_bus):
        self.confidence_manager = confidence_manager
        self.event_bus = event_bus
        self._batch_size = 5
        self._evaluation_semaphore = asyncio.Semaphore(self._batch_size)
        self._evaluation_queue = asyncio.Queue()
        self._processing = False
    
    async def start_processing(self):
        self._processing = True
        while self._processing:
            # Wait for dirty claims to evaluate
            dirty_claims = await self.confidence_manager.get_dirty_claims(self._batch_size)
            
            if not dirty_claims:
                await asyncio.sleep(1)
                continue
            
            # Process batch with concurrency control
            tasks = []
            for claim in dirty_claims:
                task = self._evaluate_with_semaphore(claim)
                tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks)
    
    async def _evaluate_with_semaphore(self, claim):
        async with self._evaluation_semaphore:
            await self._evaluate_single_claim(claim)
    
    async def _evaluate_single_claim(self, claim):
        try:
            # Notify evaluation start
            await self.event_bus.publish(EvaluationStartedEvent(claim.id))
            
            # Perform evaluation
            confidence = await self.confidence_manager.evaluate_claim(claim.id)
            
            # Notify completion
            await self.event_bus.publish(EvaluationCompletedEvent(claim.id, confidence))
            
        except Exception as e:
            logger.error(f"Error evaluating claim {claim.id}: {e}")
```

### Performance Monitoring

```python
# Performance metrics collection
class PerformanceMonitor:
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self._metrics = {
            "claim_search": [],
            "claim_evaluation": [],
            "relationship_query": []
        }
        
        self._setup_monitoring()
    
    def _setup_monitoring(self):
        # Track claim search performance
        self.event_bus.subscribe("claim.search.start", self._record_search_start)
        self.event_bus.subscribe("claim.search.complete", self._record_search_complete)
        
        # Track evaluation performance
        self.event_bus.subscribe("evaluation.started", self._record_evaluation_start)
        self.event_bus.subscribe("evaluation.completed", self._record_evaluation_complete)
    
    def _record_search_start(self, event):
        if event.request_id not in self._active_searches:
            self._active_searches[event.request_id] = time.time()
    
    def _record_search_complete(self, event):
        if event.request_id in self._active_searches:
            duration = time.time() - self._active_searches[event.request_id]
            self._metrics["claim_search"].append(duration)
            del self._active_searches[event.request_id]
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        stats = {}
        
        for metric_name, values in self._metrics.items():
            if values:
                stats[metric_name] = {
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "p50": sorted(values)[len(values) // 2],
                    "p95": sorted(values)[int(len(values) * 0.95)],
                    "p99": sorted(values)[int(len(values) * 0.99)]
                }
            else:
                stats[metric_name] = {
                    "average": 0,
                    "min": 0,
                    "max": 0,
                    "p50": 0,
                    "p95": 0,
                    "p99": 0
                }
        
        return stats
```

---

## Testing Strategy

### Unit Testing

```python
# Claim Manager testing
class TestClaimManager:
    @pytest.fixture
    def claim_manager(self):
        # Create test dependency with mocks
        mock_db = MockDatabase()
        mock_vector_db = MockVectorDB()
        return ClaimManager(mock_db, mock_vector_db)
    
    @pytest.mark.asyncio
    async def test_create_claim(self, claim_manager):
        claim = await claim_manager.create_claim(
            "Test claim content",
            ["test", "unit"]
        )
        
        assert claim.content == "Test claim content"
        assert "test" in claim.tags
        assert "unit" in claim.tags
        assert claim.confidence == 0.0  # New claims start at 0.0
    
    @pytest.mark.asyncio
    async def test_search_claims(self, claim_manager):
        # Create test claims
        await claim_manager.create_claim("Quantum entanglement research", ["physics"])
        await claim_manager.create_claim("Quantum computing applications", ["computing"])
        
        # Search for quantum claims
        results = await claim_manager.search_claims("quantum", max_results=10)
        
        assert len(results) == 2
        assert "quantum" in results[0].content.lower()
        assert "quantum" in results[1].content.lower()
```

### Integration Testing

```python
# Interface integration testing
class TestInterfaceIntegration:
    @pytest.fixture
    async def full_system(self):
        # Setup complete system with test database
        test_db = await create_test_database()
        claim_manager = ClaimManager(test_db, test_vector_db)
        confidence_manager = ConfidenceManager(test_db, claim_manager)
        relationship_manager = RelationshipManager(test_db)
        event_bus = EventBus()
        
        # Start event processing
        event_processing = asyncio.create_task(event_bus.process_events())
        
        yield {
            "claim_manager": claim_manager,
            "confidence_manager": confidence_manager,
            "relationship_manager": relationship_manager,
            "event_bus": event_bus,
            "event_processing": event_processing
        }
        
        # Cleanup
        event_processing.cancel()
        await test_db.close()
    
    @pytest.mark.asyncio
    async def test_dirty_flag_propagation(self, full_system):
        # Create claims and relationship
        claim1 = await full_system["claim_manager"].create_claim("Test claim", ["test"])
        claim2 = await full_system["claim_manager"].create_claim("Supporting evidence", ["evidence"])
        
        # Add relationship
        await full_system["relationship_manager"].add_supports_relationship(claim2.id, claim1.id)
        
        # Mark claim2 dirty
        claim2_id = claim2.id
        
        # Track events
        dirtied_claims = []
        
        def track_dirtied_claims(event):
            if event.type == "claims.dirty":
                dirtied_claims.extend(event.claim_ids)
        
        full_system["event_bus"].subscribe("claims.dirty", track_dirtied_claims)
        
        # Manually mark claim dirty (simulating external change)
        await full_system["confidence_manager"].mark_claim_dirty(claim2_id)
        
        # Wait for async event processing
        await asyncio.sleep(0.1)
        
        # Check that dependent claim was marked dirty
        assert claim1.id in dirtied_claims
        assert claim2_id in dirtied_claims
```

### Performance Testing

```python
# Interface performance testing
class TestInterfacePerformance:
    @pytest.mark.asyncio
    async def test_claim_search_performance(self, claim_manager):
        # Create large set of test claims
        for i in range(1000):
            await claim_manager.create_claim(
                f"Test claim {i} about various topics",
                ["test", "topic"]
            )
        
        # Measure search performance
        start_time = time.time()
        results = await claim_manager.search_claims("topics", max_results=10)
        duration = time.time() - start_time
        
        # Verify performance requirements
        assert duration < 0.1  # 100ms target
        assert len(results) == 10
    
    @pytest.mark.asyncio
    async def test_evaluation_batch_performance(self, confidence_manager, claim_manager):
        # Create test claims
        claims = []
        for i in range(20):
            claim = await claim_manager.create_claim(
                f"Evaluation test claim {i}",
                ["test", "evaluation"]
            )
            claims.append(claim)
            await confidence_manager.mark_claim_dirty(claim.id)
        
        # Measure batch evaluation performance
        start_time = time.time()
        
        dirty_claims = await confidence_manager.get_dirty_claims(20)
        for claim in dirty_claims:
            await confidence_manager.evaluate_claim(claim.id)
        
        duration = time.time() - start_time
        
        # Verify batch evaluation is reasonable (should be parallelizable)
        assert duration < 5.0  # 250ms per claim average
```

### User Interface Testing

```python
# TUI testing with Textual test framework
class TestTUI:
    @pytest.fixture
    def tui_app(self, mock_services):
        return TUIApplication(
            claim_manager=mock_services["claim_manager"],
            confidence_manager=mock_services["confidence_manager"],
            relationship_manager=mock_services["relationship_manager"]
        )
    
    async def test_search_functionality(self, tui_app):
        async with tui_app.run_test() as pilot:
            # Focus search input
            await pilot.press("/")
            
            # Type search query
            await pilot.type("quantum")
            
            # Submit search
            await pilot.press("enter")
            
            # Verify search results appear
            claim_list = tui_app.query_one(ClaimListPanel)
            assert "quantum" in claim_list.nodes[0].renderable.lower()
    
    async def test_claim_selection(self, tui_app):
        async with tui_app.run_test() as pilot:
            # Wait for initial claims to load
            await pilot.pause()
            
            # Select first claim
            claim_list = tui_app.query_one(ClaimListPanel)
            await claim_list.node.action_select()
            
            # Verify details panel updates
            details_panel = tui_app.query_one(ClaimDetailsPanel)
            assert details_panel.claim_id == claim_list.nodes[0].claim_id
```

---

## Implementation Roadmap

### Phase 3.1: Core Interface Infrastructure (Week 1)

**Objectives**:
- Implement shared interface components
- Build event bus infrastructure
- Create basic TUI framework layout

**Key Deliverables**:
1. **Shared Components**
   - ClaimManager component with basic CRUD operations
   - ConfidenceManager with dirty flag tracking
   - RelationshipManager for claim relationships

2. **Event System**
   - EventBus implementation with publishing/subscribing
   - Core event types for claims and evaluations
   - Performance metrics collection framework

3. **TUI Foundation**
   - Basic Textual application structure
   - Multi-panel layout framework
   - Keyboard navigation system

### Phase 3.2: TUI Implementation (Week 2)

**Objectives**:
- Complete TUI implementation with real-time updates
- Integrate visual indicators for dirty claims
- Implement claim exploration workflows

**Key Deliverables**:
1. **TUI Panels**
   - ClaimsPanel with status indicators and confidence bars
   - DetailsPanel with relationships and evidence
   - SimilarClaimsPanel for related claims
   - GoalsPanel for claim-based objectives

2. **Real-Time Features**
   - Dirty flag indicators and highlighting
   - Live evaluation progress visualization
   - Auto-refresh on claim changes
   - Confidence-based priority sorting

3. **User Interactions**
   - Keyboard navigation between panels
   - Claim creation and editing
   - Relationship management
   - Evaluation triggering controls

### Phase 3.3: CLI Implementation (Week 3)

**Objectives**:
- Implement comprehensive CLI commands
- Add batch processing capabilities
- Create various output formatting options

**Key Deliverables**:
1. **Command Set**
   - claim create/search/get/update/delete
   - support add/remove/evidence
   - goal create/list/status
   - evaluate batch/individual/status
   - config get/set/export

2. **Output Formats**
   - Human-readable table format
   - JSON for integration
   - YAML for configuration
   - CSV for data analysis

3. **Automation Features**
   - Shell completion scripts
   - Batch processing commands
   - Configuration management
   - Status monitoring tools

### Phase 3.4: MCP Implementation (Week 4)

**Objectives**:
- Implement MCP protocol for AI assistant integration
- Add contextual claim reference support
- Create bidirectional synchronization

**Key Deliverables**:
1. **MCP Actions**
   - claim() for claim creation from statements
   - prompt() for knowledge base questions
   - inspect() for relevant claim retrieval
   - evaluate() for forced claim evaluation
   - status() for system status reporting

2. **Context Processing**
   - Contextual reference resolution (nc001 -> c0001234)
   - Two-pass processing support
   - LLM response formatting
   - Error handling and recovery

3. **Real-Time Sync**
   - WebSocket integration for live updates
   - Event streaming for AI assistants
   - Bidirectional synchronization
   - Progress reporting during evaluations

---

## Maintenance and Extensibility

### Plugin Architecture

```python
# Interface plugin system
class InterfacePlugin:
    def __init__(self, name, version):
        self.name = name
        self.version = version
    
    def install(self, app):
        """Install plugin into application"""
        pass
    
    def uninstall(self, app):
        """Uninstall plugin from application"""
        pass

class PluginManager:
    def __init__(self):
        self.plugins = {}
    
    def register_plugin(self, plugin):
        self.plugins[plugin.name] = plugin
    
    def install_plugin(self, plugin_name, app):
        if plugin_name in self.plugins:
            self.plugins[plugin_name].install(app)
            return True
        return False
    
    def list_plugins(self):
        return list(self.plugins.keys())
```

### Configuration Management

```python
# Unified configuration across interfaces
class ConfigurationManager:
    def __init__(self, config_file="config.yaml"):
        self.config_file = config_file
        self._config = self._load_config()
        self._watchers = []
    
    def _load_config(self):
        try:
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._default_config()
    
    def _default_config(self):
        return {
            "ui": {
                "theme": "dark",
                "claims_per_page": 20,
                "auto_refresh": True,
                "refresh_interval": 5
            },
            "llm": {
                "provider": "gemini",
                "api_key": "",
                "model": "gemini-pro",
                "temperature": 0.1
            },
            "evaluation": {
                "batch_size": 5,
                "priority_threshold": 0.8,
                "auto_evaluate": True
            },
            "database": {
                "vector_provider": "chroma",
                "connection_string": "./data/vector_db",
                "cache_size": 1000
            }
        }
    
    def get(self, key_path, default=None):
        """Get configuration value using dot notation (e.g., 'ui.theme')"""
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path, value):
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config = self._config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
        self._save_config()
        self._notify_watchers(key_path, value)
    
    def watch(self, callback):
        """Watch for configuration changes"""
        self._watchers.append(callback)
    
    def _notify_watchers(self, key_path, value):
        for callback in self._watchers:
            try:
                callback(key_path, value)
            except Exception as e:
                logger.error(f"Error in config watcher: {e}")
```

### Update and Migration System

```python
# System update management
class UpdateManager:
    def __init__(self, current_version):
        self.current_version = current_version
        self.migrations = {}
    
    def register_migration(self, version, migration_func):
        self.migrations[version] = migration_func
    
    async def migrate(self, target_version=None):
        """Run migrations from current version to target"""
        target_version = target_version or self._get_latest_version()
        
        migration_path = self._get_migration_path(self.current_version, target_version)
        
        for version in migration_path:
            if version in self.migrations:
                await self.migrations[version]()
                self.current_version = version
                logger.info(f"Migrated to version {version}")
    
    def _get_migration_path(self, from_version, to_version):
        """Calculate upgrade path from version to version"""
        # Simplified for example - real implementation would be more sophisticated
        return sorted(
            [v for v in self.migrations.keys() if v > from_version and v <= to_version]
        )

# Example migration for interface changes
async def migrate_to_v1_1():
    """Add default evaluation configuration"""
    config = ConfigurationManager()
    if config.get("evaluation.batch_size") is None:
        config.set("evaluation.batch_size", 5)
        config.set("evaluation.priority_threshold", 0.8)
        config.set("evaluation.auto_evaluate", True)

# Register migration
update_manager = UpdateManager("1.0")
update_manager.register_migration("1.1", migrate_to_v1_1)
```

---

## Conclusion

This interface layer design specification provides a comprehensive framework for implementing Conjecture's multi-modal interface approach unified through shared components and real-time event processing. The design maintains the project's core principles of "maximum power through minimum complexity" while enabling rich user experiences across TUI, CLI, MCP, and WebUI interfaces.

### Key Benefits

1. **Unified Architecture**: Shared components reduce code duplication and ensure consistent behavior
2. **Real-Time Responsiveness**: Event-driven architecture provides immediate feedback on claim changes
3. **Dirty Flag Visualization**: All interfaces clearly show claim evaluation status
4. **Performance Optimization**: Lazy loading, caching, and batch processing ensure responsive interfaces
5. **Extensible Design**: Plugin architecture and configuration management enable future enhancements

### Implementation Priority

1. **Phase 3.1**: Core shared components and event system (Week 1)
2. **Phase 3.2**: Full TUI implementation with real-time updates (Week 2)
3. **Phase 3.3**: Comprehensive CLI with automation features (Week 3)
4. **Phase 3.4**: MCP integration with AI assistants (Week 4)

This design positions Conjecture for successful Phase 3 implementation while establishing patterns that will support future enhancements and integration with external systems.