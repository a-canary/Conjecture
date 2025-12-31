"""
Simplified context models for LLM processing
Consolidated from multiple context-related classes
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum

class ContextItemType(str, Enum):
    """Simplified context item types"""
    CLAIM = "claim"
    DATA = "data"
    KNOWLEDGE = "knowledge"
    EXAMPLE = "example"
    TOOL = "tool"

@dataclass
class ContextItem:
    """Unified context item for LLM consumption"""
    
    id: str
    content: str
    item_type: ContextItemType
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_estimate: int = 0
    source: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "content": self.content,
            "type": self.item_type.value,
            "relevance_score": self.relevance_score,
            "metadata": self.metadata,
            "token_estimate": self.token_estimate,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
        }

@dataclass
class ContextResult:
    """Result of context building operation"""
    
    query: str
    items: List[ContextItem] = field(default_factory=list)
    total_tokens: int = 0
    processing_time_ms: int = 0
    collection_method: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_item(self, item: ContextItem):
        """Add a context item"""
        self.items.append(item)
        self.total_tokens += item.token_estimate
    
    def get_summary(self) -> str:
        """Get human-readable summary"""
        return f"Collected {len(self.items)} items ({self.total_tokens} tokens) for query: '{self.query[:50]}...'"
    
    def filter_by_relevance(self, min_score: float = 0.5) -> "ContextResult":
        """Filter items by relevance score"""
        filtered_items = [item for item in self.items if item.relevance_score >= min_score]
        filtered_tokens = sum(item.token_estimate for item in filtered_items)
        
        return ContextResult(
            query=self.query,
            items=filtered_items,
            total_tokens=filtered_tokens,
            processing_time_ms=self.processing_time_ms,
            collection_method=self.collection_method,
            metadata={**self.metadata, "filtered": True, "min_score": min_score}
        )

@dataclass
class PromptTemplate:
    """Simplified prompt template"""
    
    id: str
    name: str
    template: str
    variables: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def render(self, **kwargs) -> str:
        """Render template with provided variables"""
        rendered = self.template
        for key, value in kwargs.items():
            placeholder = "{{" + key + "}}"
            rendered = rendered.replace(placeholder, str(value))
        return rendered
    
    def validate_variables(self, **kwargs) -> List[str]:
        """Validate that all required variables are provided"""
        import re
        required_vars = set(re.findall(r'\{\{(\w+)\}\}', self.template))
        provided_vars = set(kwargs.keys())
        missing_vars = required_vars - provided_vars
        return list(missing_vars)