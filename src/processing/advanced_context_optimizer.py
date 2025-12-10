"""
Advanced Context Window Optimizer for Tiny LLM Enhancement

Implements intelligent context optimization strategies specifically designed
to maximize tiny LLM performance on complex tasks through:

1. Semantic Information Density Maximization
2. Dynamic Context Allocation based on Task Requirements
3. Multi-layer Compression with Quality Preservation
4. Tiny Model-specific Optimization Strategies
5. Performance-aware Context Engineering

Key Innovation: Information-Theoretic Context Optimization (ITCO)
"""

import json
import math
import re
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

from src.core.models import Claim
from src.utils.logging import get_logger

logger = get_logger(__name__)

class TaskType(Enum):
    """Task types requiring different context optimization strategies"""
    REASONING = "reasoning"           # Complex logical reasoning
    SYNTHESIS = "synthesis"          # Information synthesis
    ANALYSIS = "analysis"            # Detailed analysis
    DECISION = "decision"            # Decision making
    CREATION = "creation"            # Content creation
    COMPARISON = "comparison"        # Comparative analysis

class InformationCategory(Enum):
    """Information categories for intelligent prioritization"""
    CRITICAL = "critical"            # Essential for task completion
    HIGH = "high"                    # Strongly relevant
    MEDIUM = "medium"                # Moderately relevant
    LOW = "low"                      # Minimally relevant
    REDUNDANT = "redundant"          # Duplicate information

@dataclass
class ContextMetrics:
    """Metrics for context optimization evaluation"""
    semantic_density: float = 0.0    # Information tokens / total tokens
    relevance_score: float = 0.0     # Task-specific relevance
    complexity_score: float = 0.0    # Information complexity
    novelty_score: float = 0.0       # Novel information ratio
    compression_quality: float = 0.0 # Quality preservation score
    token_efficiency: float = 0.0    # Tokens per unit information

@dataclass
class OptimizationTarget:
    """Optimization target for specific model and task combination"""
    model_name: str
    task_type: TaskType
    max_tokens: int
    target_density: float
    min_quality: float
    compression_strategies: List[str] = field(default_factory=list)
    priority_weights: Dict[str, float] = field(default_factory=dict)

@dataclass
class ContextChunk:
    """Chunk of context with optimization metadata"""
    content: str
    tokens: int
    category: InformationCategory
    relevance_score: float
    semantic_density: float
    keywords: Set[str]
    dependencies: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

class InformationTheoreticOptimizer:
    """
    Core optimizer implementing information-theoretic compression
    and semantic density maximization for tiny LLMs.
    """

    def __init__(self):
        self.stop_words = self._load_stop_words()
        self.semantic_patterns = self._load_semantic_patterns()

    def _load_stop_words(self) -> Set[str]:
        """Load common stop words for density calculation"""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }

    def _load_semantic_patterns(self) -> Dict[str, float]:
        """Load semantic patterns with importance weights"""
        return {
            # Critical reasoning patterns
            r'\b(because|since|therefore|thus|consequently|hence)\b': 0.9,
            r'\b(however|nevertheless|nonetheless|although|despite)\b': 0.85,
            r'\b(first|second|third|finally|additionally|furthermore)\b': 0.8,

            # Claim and evidence patterns
            r'\b(claim|assertion|argument|evidence|proof|demonstrates)\b': 0.95,
            r'\b(proves|establishes|confirms|validates|supports)\b': 0.9,
            r'\b(refutes|contradicts|disproves|challenges)\b': 0.9,

            # Quantitative patterns
            r'\b(\d+%|\d+\.\d+%|\d+ percent)\b': 0.85,
            r'\b(\d+\.\d+|\d+,*\d+)\b': 0.7,

            # Causal patterns
            r'\b(causes|leads to|results in|produces|generates)\b': 0.9,
            r'\b(due to|because of|as a result of|owing to)\b': 0.85,
        }

    def calculate_semantic_density(self, text: str) -> float:
        """
        Calculate semantic information density of text.
        Higher density = more information per token.
        """
        if not text or not text.strip():
            return 0.0

        # Token approximation (rough)
        tokens = len(text.split())
        words = text.lower().split()

        # Non-stop word ratio (information content)
        content_words = [w for w in words if w not in self.stop_words]
        content_ratio = len(content_words) / max(len(words), 1)

        # Semantic pattern matches
        pattern_score = 0.0
        for pattern, weight in self.semantic_patterns.items():
            matches = len(re.findall(pattern, text.lower()))
            pattern_score += matches * weight

        # Unique content ratio (novelty)
        unique_words = len(set(content_words))
        novelty_ratio = unique_words / max(len(content_words), 1)

        # Combined density score
        density = (
            content_ratio * 0.3 +           # Information content
            min(pattern_score / tokens, 1.0) * 0.4 +  # Semantic patterns
            novelty_ratio * 0.3             # Information diversity
        )

        return min(density, 1.0)

    def extract_key_entities(self, text: str) -> Set[str]:
        """Extract key entities and concepts from text"""
        # Simple entity extraction (could be enhanced with NLP)
        entities = set()

        # Capitalized phrases (potential entities)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.update([c.lower() for c in capitalized])

        # Numbers and measurements
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', text)
        entities.update(numbers)

        # Technical terms (simplified)
        technical = re.findall(r'\b[a-z]+(?:_[a-z]+|[A-Z][a-z]*)*\b', text)
        entities.update([t for t in technical if len(t) > 6])

        return entities

    def calculate_relevance(self, text: str, task_keywords: Set[str],
                          context_keywords: Set[str]) -> float:
        """Calculate relevance score based on keyword matching and semantic similarity"""
        text_lower = text.lower()
        text_entities = self.extract_key_entities(text_lower)

        # Direct keyword matches
        direct_matches = len(text_entities & task_keywords)
        context_matches = len(text_entities & context_keywords)

        # Semantic pattern relevance
        semantic_score = 0.0
        for pattern, weight in self.semantic_patterns.items():
            if re.search(pattern, text_lower):
                semantic_score += weight

        # Combined relevance
        total_entities = max(len(text_entities), 1)
        relevance = (
            (direct_matches / total_entities) * 0.5 +
            (context_matches / total_entities) * 0.3 +
            min(semantic_score, 1.0) * 0.2
        )

        return min(relevance, 1.0)

class TinyLLMContextOptimizer:
    """
    Advanced context optimizer specifically designed for tiny LLMs.
    Implements multi-layer optimization with information-theoretic approaches.
    """

    def __init__(self, model_name: str = "ibm/granite-4-h-tiny"):
        self.model_name = model_name
        self.information_optimizer = InformationTheoreticOptimizer()
        self.optimization_targets = self._load_optimization_targets()
        self.compression_cache = {}

    def _load_optimization_targets(self) -> Dict[TaskType, OptimizationTarget]:
        """Load optimization targets for different task types"""
        return {
            TaskType.REASONING: OptimizationTarget(
                model_name=self.model_name,
                task_type=TaskType.REASONING,
                max_tokens=2048,  # Conservative for complex reasoning
                target_density=0.85,
                min_quality=0.9,
                compression_strategies=["semantic", "redundancy", "hierarchical"],
                priority_weights={
                    "critical": 1.0,
                    "high": 0.8,
                    "medium": 0.5,
                    "low": 0.2
                }
            ),
            TaskType.SYNTHESIS: OptimizationTarget(
                model_name=self.model_name,
                task_type=TaskType.SYNTHESIS,
                max_tokens=1536,  # Smaller for synthesis tasks
                target_density=0.8,
                min_quality=0.85,
                compression_strategies=["density", "relevance", "abstraction"],
                priority_weights={
                    "critical": 1.0,
                    "high": 0.7,
                    "medium": 0.4,
                    "low": 0.1
                }
            ),
            TaskType.ANALYSIS: OptimizationTarget(
                model_name=self.model_name,
                task_type=TaskType.ANALYSIS,
                max_tokens=2560,  # Medium for detailed analysis
                target_density=0.75,
                min_quality=0.8,
                compression_strategies=["selective", "summarization", "clustering"],
                priority_weights={
                    "critical": 1.0,
                    "high": 0.9,
                    "medium": 0.6,
                    "low": 0.3
                }
            )
        }

    def analyze_context(self, context_text: str, task_type: TaskType,
                       task_keywords: Set[str] = None) -> List[ContextChunk]:
        """
        Analyze context and break into chunks with optimization metadata
        """
        if task_keywords is None:
            task_keywords = set()

        # Split context into sentences
        sentences = re.split(r'(?<=[.!?])\s+', context_text.strip())
        chunks = []

        # Extract context keywords from all text
        all_text = context_text.lower()
        context_keywords = self.information_optimizer.extract_key_entities(all_text)

        for sentence in sentences:
            if not sentence.strip():
                continue

            # Calculate metrics for this chunk
            tokens = len(sentence.split()) * 1.3  # Rough token estimate
            semantic_density = self.information_optimizer.calculate_semantic_density(sentence)
            relevance = self.information_optimizer.calculate_relevance(
                sentence, task_keywords, context_keywords
            )

            # Categorize information
            category = self._categorize_information(relevance, semantic_density)

            chunk = ContextChunk(
                content=sentence.strip(),
                tokens=int(tokens),
                category=category,
                relevance_score=relevance,
                semantic_density=semantic_density,
                keywords=self.information_optimizer.extract_key_entities(sentence.lower()),
                metadata={
                    "length": len(sentence),
                    "complexity": self._calculate_complexity(sentence),
                    "novelty": self._calculate_novelty(sentence, chunks)
                }
            )

            chunks.append(chunk)

        return chunks

    def _categorize_information(self, relevance: float, density: float) -> InformationCategory:
        """Categorize information based on relevance and density"""
        if relevance >= 0.8 and density >= 0.7:
            return InformationCategory.CRITICAL
        elif relevance >= 0.6 and density >= 0.5:
            return InformationCategory.HIGH
        elif relevance >= 0.4 and density >= 0.3:
            return InformationCategory.MEDIUM
        elif relevance >= 0.2:
            return InformationCategory.LOW
        else:
            return InformationCategory.REDUNDANT

    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score"""
        # Simple complexity metrics
        avg_word_length = sum(len(word) for word in text.split()) / max(len(text.split()), 1)
        sentence_count = len(re.split(r'[.!?]', text))
        clause_count = len(re.findall(r'[,;:]', text))

        complexity = (
            min(avg_word_length / 10, 1.0) * 0.3 +
            min(clause_count / max(sentence_count, 1) / 5, 1.0) * 0.4 +
            min(len(text) / 200, 1.0) * 0.3
        )

        return complexity

    def _calculate_novelty(self, text: str, existing_chunks: List[ContextChunk]) -> float:
        """Calculate novelty compared to existing chunks"""
        text_words = set(text.lower().split())

        if not existing_chunks:
            return 1.0

        # Compare with previous chunks
        overlap_scores = []
        for chunk in existing_chunks:
            chunk_words = set(chunk.content.lower().split())
            if chunk_words:
                overlap = len(text_words & chunk_words) / len(text_words | chunk_words)
                overlap_scores.append(overlap)

        avg_overlap = sum(overlap_scores) / max(len(overlap_scores), 1)
        novelty = 1.0 - avg_overlap

        return max(novelty, 0.0)

    def optimize_context(self, context_text: str, task_type: TaskType,
                        task_keywords: Set[str] = None) -> Tuple[str, ContextMetrics]:
        """
        Main optimization function that produces optimized context for tiny LLMs
        """
        target = self.optimization_targets.get(task_type)
        if not target:
            raise ValueError(f"No optimization target for task type: {task_type}")

        # Analyze context
        chunks = self.analyze_context(context_text, task_type, task_keywords)

        # Apply multi-stage optimization
        optimized_chunks = self._apply_priority_filtering(chunks, target)
        optimized_chunks = self._apply_semantic_compression(optimized_chunks, target)
        optimized_chunks = self._apply_density_optimization(optimized_chunks, target)
        optimized_chunks = self._apply_token_budgeting(optimized_chunks, target)

        # Assemble final context
        final_context = " ".join(chunk.content for chunk in optimized_chunks)

        # Calculate metrics
        metrics = self._calculate_optimization_metrics(
            original_text=context_text,
            optimized_text=final_context,
            chunks=optimized_chunks,
            target=target
        )

        return final_context, metrics

    def _apply_priority_filtering(self, chunks: List[ContextChunk],
                                 target: OptimizationTarget) -> List[ContextChunk]:
        """Apply priority-based filtering based on information category"""
        min_priority = min(target.priority_weights.values())

        # Filter by category priority
        filtered_chunks = []
        for chunk in chunks:
            category_weight = target.priority_weights.get(chunk.category.value, min_priority)

            # Keep chunk if it meets priority threshold
            if category_weight >= 0.2:  # Minimum threshold
                chunk.metadata["priority_weight"] = category_weight
                filtered_chunks.append(chunk)

        return filtered_chunks

    def _apply_semantic_compression(self, chunks: List[ContextChunk],
                                   target: OptimizationTarget) -> List[ContextChunk]:
        """Apply semantic compression while preserving meaning"""
        compressed_chunks = []

        for chunk in chunks:
            # Skip compression for critical information
            if chunk.category == InformationCategory.CRITICAL:
                compressed_chunks.append(chunk)
                continue

            # Apply compression based on density and relevance
            if chunk.semantic_density < 0.3 or chunk.relevance_score < 0.4:
                compressed_content = self._compress_chunk_semantic(chunk.content)
                if compressed_content:
                    chunk.content = compressed_content
                    chunk.tokens = len(compressed_content.split()) * 1.3
                    compressed_chunks.append(chunk)
            else:
                compressed_chunks.append(chunk)

        return compressed_chunks

    def _compress_chunk_semantic(self, text: str) -> Optional[str]:
        """Compress text while preserving semantic meaning"""
        # Remove redundant phrases
        redundant_patterns = [
            r'\b(in order to|so as to)\b',  # Replace with "to"
            r'\b(due to the fact that|owing to the fact that)\b',  # Replace with "because"
            r'\b(at this point in time|at the present time)\b',  # Replace with "now"
            r'\b(despite the fact that|in spite of the fact that)\b',  # Replace with "despite"
        ]

        compressed = text
        for pattern in replacement in [
            (r'\b(in order to|so as to)\b', 'to'),
            (r'\b(due to the fact that|owing to the fact that)\b', 'because'),
            (r'\b(at this point in time|at the present time)\b', 'now'),
            (r'\b(despite the fact that|in spite of the fact that)\b', 'despite'),
        ]:
            compressed = re.sub(pattern, replacement, compressed, flags=re.IGNORECASE)

        # Remove filler words and phrases
        filler_patterns = [
            r'\b(basically|essentially|actually|really|virtually)\b',
            r'\b(i think|i believe|it seems to me)\b',
            r"\b(as far as i\'m concerned|from my perspective)\b",
        ]

        for pattern in filler_patterns:
            compressed = re.sub(pattern, '', compressed, flags=re.IGNORECASE)

        # Clean up extra spaces
        compressed = re.sub(r'\s+', ' ', compressed).strip()

        return compressed if compressed and len(compressed) > 10 else None

    def _apply_density_optimization(self, chunks: List[ContextChunk],
                                   target: OptimizationTarget) -> List[ContextChunk]:
        """Optimize information density through intelligent reorganization"""
        # Sort chunks by combined density score
        for chunk in chunks:
            chunk.metadata["combined_score"] = (
                chunk.semantic_density * 0.4 +
                chunk.relevance_score * 0.4 +
                chunk.metadata["novelty"] * 0.2
            )

        # Sort by combined score (highest first)
        chunks.sort(key=lambda x: x.metadata["combined_score"], reverse=True)

        return chunks

    def _apply_token_budgeting(self, chunks: List[ContextChunk],
                             target: OptimizationTarget) -> List[ContextChunk]:
        """Apply token budget constraints"""
        current_tokens = sum(chunk.tokens for chunk in chunks)

        if current_tokens <= target.max_tokens:
            return chunks

        # Greedy selection until budget is met
        selected_chunks = []
        used_tokens = 0

        for chunk in chunks:
            if used_tokens + chunk.tokens <= target.max_tokens:
                selected_chunks.append(chunk)
                used_tokens += chunk.tokens
            else:
                # Try to fit partial chunk
                remaining_tokens = target.max_tokens - used_tokens
                if remaining_tokens > 20:  # Minimum meaningful chunk
                    partial_content = self._truncate_chunk(
                        chunk.content, remaining_tokens
                    )
                    if partial_content:
                        chunk.content = partial_content
                        chunk.tokens = remaining_tokens
                        selected_chunks.append(chunk)
                break

        return selected_chunks

    def _truncate_chunk(self, text: str, max_tokens: int) -> Optional[str]:
        """Intelligently truncate text to fit token budget"""
        words = text.split()
        max_words = int(max_tokens / 1.3)  # Rough token to word conversion

        if len(words) <= max_words:
            return text

        # Find best truncation point (end of sentence)
        truncated = " ".join(words[:max_words])

        # Try to end at sentence boundary
        last_period = truncated.rfind('.')
        last_exclamation = truncated.rfind('!')
        last_question = truncated.rfind('?')

        last_sentence_end = max(last_period, last_exclamation, last_question)

        if last_sentence_end > len(truncated) * 0.7:  # If we can keep 70%+ of content
            return truncated[:last_sentence_end + 1]
        else:
            return truncated + "..."  # Indicate truncation

    def _calculate_optimization_metrics(self, original_text: str, optimized_text: str,
                                      chunks: List[ContextChunk],
                                      target: OptimizationTarget) -> ContextMetrics:
        """Calculate comprehensive optimization metrics"""
        original_tokens = len(original_text.split()) * 1.3
        optimized_tokens = len(optimized_text.split()) * 1.3

        # Calculate densities
        avg_semantic_density = sum(chunk.semantic_density for chunk in chunks) / max(len(chunks), 1)
        avg_relevance = sum(chunk.relevance_score for chunk in chunks) / max(len(chunks), 1)

        # Information preservation estimate
        critical_chunks = [c for c in chunks if c.category == InformationCategory.CRITICAL]
        high_chunks = [c for c in chunks if c.category == InformationCategory.HIGH]

        preservation_score = (
            len(critical_chunks) / max(len([c for c in chunks]), 1) * 0.6 +
            len(high_chunks) / max(len([c for c in chunks]), 1) * 0.4
        )

        return ContextMetrics(
            semantic_density=avg_semantic_density,
            relevance_score=avg_relevance,
            complexity_score=sum(chunk.metadata["complexity"] for chunk in chunks) / max(len(chunks), 1),
            novelty_score=sum(chunk.metadata["novelty"] for chunk in chunks) / max(len(chunks), 1),
            compression_quality=preservation_score,
            token_efficiency=(original_tokens - optimized_tokens) / max(original_tokens, 1)
        )

class ContextPerformanceEvaluator:
    """
    Evaluates context optimization performance and provides tuning recommendations
    """

    def __init__(self):
        self.benchmark_results = []

    def evaluate_optimization(self, original_context: str, optimized_context: str,
                            task_performance: Dict[str, float],
                            metrics: ContextMetrics) -> Dict[str, Any]:
        """Evaluate optimization quality and performance impact"""

        evaluation = {
            "compression_metrics": {
                "original_tokens": len(original_context.split()),
                "optimized_tokens": len(optimized_context.split()),
                "compression_ratio": len(optimized_context.split()) / max(len(original_context.split()), 1),
                "tokens_saved": len(original_context.split()) - len(optimized_context.split())
            },
            "quality_metrics": {
                "semantic_density": metrics.semantic_density,
                "relevance_score": metrics.relevance_score,
                "compression_quality": metrics.compression_quality,
                "information_preservation": metrics.compression_quality
            },
            "performance_metrics": task_performance,
            "efficiency_score": self._calculate_efficiency_score(metrics, task_performance),
            "recommendations": self._generate_recommendations(metrics, task_performance)
        }

        return evaluation

    def _calculate_efficiency_score(self, metrics: ContextMetrics,
                                  task_performance: Dict[str, float]) -> float:
        """Calculate overall efficiency score"""
        # Weight components
        density_weight = 0.3
        quality_weight = 0.3
        performance_weight = 0.4

        # Average performance score
        avg_performance = sum(task_performance.values()) / max(len(task_performance), 1)

        efficiency = (
            metrics.semantic_density * density_weight +
            metrics.compression_quality * quality_weight +
            avg_performance * performance_weight
        )

        return efficiency

    def _generate_recommendations(self, metrics: ContextMetrics,
                                task_performance: Dict[str, float]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        if metrics.semantic_density < 0.6:
            recommendations.append(
                "Consider increasing semantic density through better information selection"
            )

        if metrics.compression_quality < 0.8:
            recommendations.append(
                "Compression may be losing important information - reduce compression ratio"
            )

        if metrics.token_efficiency < 0.2:
            recommendations.append(
                "Low token efficiency - consider more aggressive compression strategies"
            )

        avg_performance = sum(task_performance.values()) / max(len(task_performance), 1)
        if avg_performance < 0.7:
            recommendations.append(
                "Task performance degraded - review information categorization"
            )

        return recommendations

# Factory function for easy instantiation
def create_tiny_llm_optimizer(model_name: str = "ibm/granite-4-h-tiny") -> TinyLLMContextOptimizer:
    """Create optimized context optimizer for tiny LLMs"""
    return TinyLLMContextOptimizer(model_name=model_name)