"""
Research-Optimized Claim Selector

Implements all research findings for improved claim accumulation:
1. Position Primacy: Claims at START of prompts (Lost in Middle research)
2. Strict Gating: Only 0.8+ confidence claims
3. Windowing: Recent 20 claims maximum
4. Semantic Filtering: Category-matched claims
5. Limited Count: Max 3 claims

Based on R&D experiments 2026-03-01.
"""

import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class OptimizedClaim:
    """Claim with optimized metadata for selection"""
    content: str
    question: str  # Original question that generated this claim
    confidence: float
    is_correct: bool
    category: str
    embedding: List[float] = field(default_factory=list)
    created_at: int = 0  # Sequence number for windowing


class ResearchOptimizedSelector:
    """
    Claim selector implementing all research-backed optimizations.

    Research basis:
    - Lost in the Middle (Liu et al. 2023): Primacy bias
    - The Few-shot Dilemma: Over-prompting degrades performance
    - Context Rot: Larger contexts reduce effectiveness
    - Cluster-based Adaptive Retrieval: Semantic selection
    """

    # Configurable thresholds based on R&D findings
    WINDOW_SIZE = 20          # Only recent N claims (context rot mitigation)
    CONFIDENCE_THRESHOLD = 0.8  # Strict gating (noise reduction)
    MAX_CLAIMS = 3            # Limit count (prompt dilution mitigation)
    EMBEDDING_DIM = 64        # Simple hash-based embedding dimension

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        max_claims: int = MAX_CLAIMS
    ):
        self.claims: List[OptimizedClaim] = []
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.max_claims = max_claims
        self.sequence = 0

    def add_claim(
        self,
        content: str,
        question: str,
        confidence: float,
        is_correct: bool,
        category: str = ""
    ) -> None:
        """Add a claim to memory with auto-generated embedding."""
        embedding = self._embed(question)
        category = category or self._detect_category(question)

        claim = OptimizedClaim(
            content=content[:200],  # Truncate for efficiency
            question=question[:100],
            confidence=confidence,
            is_correct=is_correct,
            category=category,
            embedding=embedding,
            created_at=self.sequence
        )

        self.claims.append(claim)
        self.sequence += 1

    def select_claims(
        self,
        current_question: str,
        current_category: str = ""
    ) -> List[OptimizedClaim]:
        """
        Select optimal claims for current question.

        Applies all research optimizations:
        1. Window to recent claims
        2. Filter to correct + high confidence
        3. Score by semantic similarity + category match
        4. Return top N
        """
        # 1. Windowing: Only recent claims
        windowed = self._window_claims()

        # 2. Strict gating: Correct + high confidence only
        gated = self._gate_claims(windowed)

        if not gated:
            return []

        # 3. Semantic scoring with category bonus
        current_category = current_category or self._detect_category(current_question)
        current_embedding = self._embed(current_question)

        scored = self._score_claims(gated, current_embedding, current_category)

        # 4. Top N claims
        return sorted(scored, key=lambda x: x[1], reverse=True)[:self.max_claims]

    def build_prompt(
        self,
        question: str,
        category: str = ""
    ) -> str:
        """
        Build optimized prompt with claims at START (primacy bias).

        Research finding: Claims at beginning of prompt get more attention
        due to primacy bias in transformer attention mechanisms.
        """
        selected = self.select_claims(question, category)

        if selected:
            # Claims at START (primacy bias)
            hints = "KEY PATTERNS FROM SIMILAR PROBLEMS:\n"
            for claim, score in selected:
                hints += f"• {claim.content[:80]}\n"
            hints += "\n"

            return f"""{hints}Problem: {question}

Answer:"""
        else:
            return f"""Problem: {question}

Answer:"""

    def _window_claims(self) -> List[OptimizedClaim]:
        """Apply windowing to get recent claims only."""
        if len(self.claims) <= self.window_size:
            return self.claims
        return self.claims[-self.window_size:]

    def _gate_claims(self, claims: List[OptimizedClaim]) -> List[OptimizedClaim]:
        """Filter to correct + high confidence claims."""
        return [
            c for c in claims
            if c.is_correct and c.confidence >= self.confidence_threshold
        ]

    def _score_claims(
        self,
        claims: List[OptimizedClaim],
        query_embedding: List[float],
        query_category: str
    ) -> List[Tuple[OptimizedClaim, float]]:
        """Score claims by relevance."""
        scored = []
        for claim in claims:
            # Category match bonus (1.5x if same category)
            cat_bonus = 1.5 if claim.category == query_category else 1.0

            # Semantic similarity
            sem_sim = self._cosine_sim(query_embedding, claim.embedding)

            # Combined score
            score = claim.confidence * cat_bonus * (0.5 + sem_sim)
            scored.append((claim, score))

        return scored

    def _embed(self, text: str) -> List[float]:
        """Simple hash-based embedding (no external dependencies)."""
        words = text.lower().split()
        vec = [0.0] * self.EMBEDDING_DIM

        for word in words:
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vec[h % self.EMBEDDING_DIM] += 1.0

        # L2 normalize
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]

        return vec

    def _cosine_sim(self, a: List[float], b: List[float]) -> float:
        """Cosine similarity between embeddings."""
        if not a or not b:
            return 0.5
        return sum(x * y for x, y in zip(a, b))

    def _detect_category(self, question: str) -> str:
        """Simple keyword-based category detection."""
        q = question.lower()

        if any(x in q for x in ["sell", "buy", "price", "cost", "revenue", "store", "profit"]):
            return "sales"
        elif any(x in q for x in ["travel", "speed", "mph", "distance", "hour", "km", "mile"]):
            return "distance"
        elif any(x in q for x in ["percent", "%", "fraction", "ratio"]):
            return "percentage"
        elif any(x in q for x in ["divide", "share", "equally", "each", "split", "per"]):
            return "division"
        elif any(x in q for x in ["area", "perimeter", "triangle", "circle", "square"]):
            return "geometry"
        elif any(x in q for x in ["if", "then", "therefore", "conclude"]):
            return "logic"

        return "general"

    def get_stats(self) -> Dict:
        """Get memory statistics."""
        windowed = self._window_claims()
        gated = self._gate_claims(windowed)

        return {
            "total_claims": len(self.claims),
            "windowed_claims": len(windowed),
            "gated_claims": len(gated),
            "window_size": self.window_size,
            "confidence_threshold": self.confidence_threshold,
            "max_claims": self.max_claims
        }


# Convenience factory
def create_optimized_selector(**kwargs) -> ResearchOptimizedSelector:
    """Create selector with optional custom config."""
    return ResearchOptimizedSelector(**kwargs)
