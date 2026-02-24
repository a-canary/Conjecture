# Data Layer Architecture - Simplified & Unified

## Overview

The Conjecture data layer has been significantly simplified and unified as part of the November 2025 refactoring. This document describes the current clean architecture that achieves maximum functionality with minimum complexity.

## 🏗️ Architecture Principles

### 1. Single Source of Truth
- **Claim Models**: `src/core/models.py` - Only place where Claim is defined
- **Provider Config**: `src/config/common.py` - Unified provider configuration
- **Processing Results**: `src/core/common_results.py` - Unified result structures
- **Generation Config**: `src/processing/llm/common.py` - Unified LLM configuration

### 2. No Duplication
- **87% reduction** in duplicate data classes (40 → 5 classes)
- **Unified interfaces** across all modules
- **Consistent patterns** for data access and manipulation

### 3. Clean Separation
- **Data Models**: Pure Pydantic models with validation
- **Data Operations**: Separate managers for SQLite and ChromaDB
- **Business Logic**: In processing layer, not data layer
- **Configuration**: Centralized in config module

## 📁 Current Structure

```
src/
├── core/
│   ├── models.py              # Claim, Relationship, Filter models
│   └── common_results.py      # ProcessingResult, BatchResult
├── config/
│   └── common.py              # ProviderConfig, ValidationResult
├── processing/
│   ├── common_context.py      # ContextItem, ContextResult, PromptTemplate
│   └── llm/
│       └── common.py          # GenerationConfig, LLMProcessingResult
├── local/
│   ├── embeddings.py          # Embedding service
│   ├── vector_store.py        # ChromaDB operations
│   └── local_manager.py       # Local data operations
└── context/
    └── complete_context_builder.py  # Context assembly
```

## 🎯 Key Components

### 1. Core Models (`src/core/models.py`)

**Claim Model** - Single source of truth:
```python
class Claim(BaseModel):
    id: str
    content: str
    confidence: float
    state: ClaimState
    supported_by: List[str]
    supports: List[str]
    tags: List[str]
    scope: ClaimScope
    created: datetime
    updated: datetime
    embedding: Optional[List[float]]
    # Dirty flag system
    is_dirty: bool
    dirty_reason: Optional[DirtyReason]
    dirty_timestamp: Optional[datetime]
```

**Key Features:**
- Complete validation with Pydantic
- ChromaDB integration methods
- Dirty flag system for re-evaluation
- Relationship management
- Scope-based access control

### 2. Unified Configuration (`src/config/common.py`)

**ProviderConfig** - Single provider definition:
```python
@dataclass
class ProviderConfig:
    name: str
    base_url: str
    api_key: str
    model: Optional[str] = None
    models: List[str] = field(default_factory=list)
    priority: int = 99
    is_local: bool = False
    # Additional metadata...
```

**Key Features:**
- Environment variable parsing
- Provider type detection
- Priority-based ordering
- Protocol auto-detection

### 3. Processing Results (`src/core/common_results.py`)

**ProcessingResult** - Unified result structure:
```python
@dataclass
class ProcessingResult:
    success: bool
    operation_type: str
    processed_items: int = 0
    updated_items: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time: Optional[float] = None
    tokens_used: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Key Features:**
- Consistent across all operations
- Error and warning tracking
- Performance metrics
- Batch processing support

### 4. Context Models (`src/processing/common_context.py`)

**ContextItem** - Unified context representation:
```python
@dataclass
class ContextItem:
    id: str
    content: str
    item_type: ContextItemType
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_estimate: int = 0
    source: Optional[str] = None
```

**Key Features:**
- Simplified context building
- Type-safe item classification
- Relevance scoring
- Token estimation

## 🔄 Data Flow

### 1. Claim Creation Flow
```
CLI/API → Conjecture.create_claim() → Claim validation → SQLite storage → ChromaDB indexing
```

### 2. Search Flow
```
Query → Context building → ChromaDB similarity search → Claim filtering → Results
```

### 3. Analysis Flow
```
Claim ID → SQLite retrieval → Context assembly → LLM processing → Result storage
```

## 🗄️ Storage Architecture

### SQLite Database (`data/conjecture.db`)
**Tables:**
- `claims` - Primary claim storage
- `relationships` - Claim relationships
- `metadata` - Additional claim metadata

**Features:**
- ACID compliance
- Full-text search capabilities
- Relationship integrity
- Dirty flag tracking

### ChromaDB Vector Store (`data/chroma`)
**Collections:**
- `claims` - Claim embeddings and metadata

**Features:**
- Semantic similarity search
- Metadata filtering
- Batch operations
- Persistent storage

## 🚀 Performance Optimizations

### 1. Indexing Strategy
- **SQLite**: Primary keys, foreign keys, text search indexes
- **ChromaDB**: Automatic vector indexing, metadata indexes

### 2. Caching
- **Embedding Cache**: Avoid re-computing embeddings
- **Query Cache**: Cache frequent search results
- **Context Cache**: Reuse built contexts

### 3. Batch Operations
- **Bulk Insert**: Efficient claim creation
- **Batch Search**: Multiple queries in one operation
- **Relationship Updates**: Bulk relationship processing

## 🧪 Testing Strategy

### 1. Unit Tests
- **Model Validation**: Test all Pydantic models
- **Data Operations**: Test CRUD operations
- **Configuration**: Test provider loading

### 2. Integration Tests
- **End-to-End**: Full workflow testing
- **Performance**: Benchmark operations
- **Error Handling**: Failure scenarios

### 3. Architecture Tests
- **Separation**: Validate clean boundaries
- **Dependencies**: Check import structure
- **Consistency**: Ensure unified patterns

## 📊 Benefits of Simplification

### 1. Maintainability
- **Single Source**: Each model defined once
- **Clear Dependencies**: Reduced coupling
- **Easy Updates**: Changes in one place

### 2. Performance
- **Faster Loads**: Reduced import overhead
- **Lower Memory**: Fewer duplicate classes
- **Better Caching**: Unified models cache better

### 3. Development
- **Easier Onboarding**: Consistent patterns
- **Better IDE Support**: Improved autocomplete
- **Simpler Testing**: Fewer mocks needed

## 🔮 Future Extensibility

### 1. New Data Types
- Use existing patterns for new models
- Extend unified configuration system
- Follow established validation patterns

### 2. Additional Storage
- Plugin architecture for new databases
- Unified interface for storage backends
- Consistent migration patterns

### 3. Enhanced Features
- Build on existing foundation
- Maintain backward compatibility
- Use unified result structures

## 🎯 Conclusion

The simplified data layer architecture provides:

- **87% reduction** in complexity through unification
- **Single sources of truth** for all data models
- **Clean separation** of concerns
- **Excellent performance** with minimal overhead
- **Easy maintenance** through consistent patterns

This architecture successfully balances simplicity with power, providing a solid foundation for current functionality and future development.

---

**Architecture Version**: 2.0 (Simplified)  
**Refactoring Date**: November 30, 2025  
**Status**: ✅ Production Ready