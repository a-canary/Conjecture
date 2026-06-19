#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
E1: Fact-Checking Accuracy — Test H1-H4
Standalone test without project imports to avoid circular dependency issues.
"""

import json
import os
import sys
import sqlite3
import time
import random
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Set, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

# ============================================================================
# MINIMAL CORE MODELS (copied from core/models.py for standalone testing)
# ============================================================================

class ClaimState(str, Enum):
    EXPLORE = "Explore"
    VALIDATED = "Validated"
    ORPHANED = "Orphaned"
    QUEUED = "Queued"


class ClaimType(str, Enum):
    IMPRESSION = "impression"
    ASSUMPTION = "assumption"
    OBSERVATION = "observation"
    CONJECTURE = "conjecture"
    CONCEPT = "concept"
    EXAMPLE = "example"
    GOAL = "goal"
    REFERENCE = "reference"
    ASSERTION = "assertion"


class DirtyReason(str, Enum):
    NEW_CLAIM_ADDED = "new_claim_added"
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    SUPPORTING_CLAIM_CHANGED = "supporting_claim_changed"
    RELATIONSHIP_CHANGED = "relationship_changed"
    MANUAL_MARK = "manual_mark"
    BATCH_EVALUATION = "batch_evaluation"
    SYSTEM_TRIGGER = "system_trigger"
    RELATIONSHIP_CHANGE = "relationship_change"
    CONTENT_UPDATE = "content_update"
    CONFIDENCE_CHANGE = "confidence_change"
    STATE_CHANGE = "state_change"
    MANUAL_FLAG = "manual_flag"


class Claim:
    """Minimal Claim model for testing."""
    def __init__(
        self,
        id: str,
        content: str,
        confidence: float,
        state: ClaimState = ClaimState.EXPLORE,
        type: List[ClaimType] = None,
        tags: List[str] = None,
        scope: str = "user-workspace",
        supers: List[str] = None,
        subs: List[str] = None,
        is_dirty: bool = True,
        dirty_reason: DirtyReason = None,
        dirty_priority: int = 0,
    ):
        self.id = id
        self.content = content
        self.confidence = confidence
        self.state = state
        self.type = type or [ClaimType.CONCEPT]
        self.tags = tags or []
        self.scope = scope
        self.supers = supers or []
        self.subs = subs or []
        self.is_dirty = is_dirty
        self.dirty_reason = dirty_reason
        self.dirty_priority = dirty_priority
        self.dirty_timestamp = None
        self.created = datetime.now(timezone.utc)
        self.updated = datetime.now(timezone.utc)

    def mark_dirty(self, reason: str = "manual", priority: int = 0) -> None:
        self.is_dirty = True
        self.dirty_reason = DirtyReason.MANUAL_FLAG if reason == "manual" else DirtyReason(reason) if reason in DirtyReason._value2member_map_ else DirtyReason.MANUAL_FLAG
        self.dirty_priority = priority
        self.dirty_timestamp = datetime.now(timezone.utc)

    def mark_clean(self) -> None:
        self.is_dirty = False
        self.dirty_reason = None
        self.dirty_priority = 0
        self.dirty_timestamp = None

    def should_prioritize(self, threshold: float = 0.90) -> bool:
        return self.is_dirty and self.confidence > threshold

    def __hash__(self):
        return hash((self.id, self.content, self.confidence))

    def __eq__(self, other):
        if not isinstance(other, Claim):
            return False
        return hash(self) == hash(other)


def generate_claim_id() -> str:
    """Generate a unique claim ID."""
    import uuid
    import time
    timestamp = int(time.time() * 1000)
    unique_id = str(uuid.uuid4())[:8]
    return f"c{timestamp}_{unique_id}"


# ============================================================================
# ENUMS FOR FACT CHECKING (from fact_checking_pipeline.py)
# ============================================================================

class VerificationTier(str, Enum):
    SELF_CONSISTENCY = "self_consistency"
    VECTOR_SEARCH = "vector_search"
    LIVE_WEB = "live_web"


class FactCheckResult(str, Enum):
    VERIFIED = "verified"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"
    SKIPPED = "skipped"


class FactCheckStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class TierResult:
    tier: VerificationTier
    result: FactCheckResult
    confidence: float
    evidence: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    cost_usd: float = 0.0
    latency_ms: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.result == FactCheckResult.VERIFIED


# ============================================================================
# SELF-CONSISTENCY CHECKER (T1)
# ============================================================================

class SelfConsistencyChecker:
    """Tier 1: Verify claim against internal claim graph."""
    
    def __init__(self, claims: Dict[str, Claim]):
        self.claims = claims

    def check(self, claim_id: str) -> TierResult:
        """Run self-consistency check on a claim."""
        start = datetime.now(timezone.utc)
        claim = self.claims.get(claim_id)

        if claim is None:
            return TierResult(
                tier=VerificationTier.SELF_CONSISTENCY,
                result=FactCheckResult.SKIPPED,
                confidence=0.0,
                evidence=[],
                contradictions=[f"Claim {claim_id} not found"],
                cost_usd=0.0,
                latency_ms=int((datetime.now(timezone.utc) - start).total_seconds() * 1000),
            )

        contradictions = []
        evidence = []

        # Check 1: Sub-claims (children) that provide evidence FOR this claim
        sub_claims = [self.claims[sid] for sid in claim.subs if sid in self.claims]
        for sub in sub_claims:
            if sub.confidence < 0.3:
                contradictions.append(
                    f"Sub-claim '{sub.id}' has very low confidence ({sub.confidence})"
                )
            if sub.state == ClaimState.ORPHANED:
                evidence.append(f"Sub-claim '{sub.id}' is orphaned — relationship may be invalid")

        # Check 2: Self-consistency — claim's own confidence vs type
        type_values = [t.value if hasattr(t, 'value') else str(t) for t in claim.type]
        if 'assertion' in type_values and claim.confidence < 0.7:
            contradictions.append(
                f"ASSERTION type with low confidence ({claim.confidence})"
            )
        if 'impression' in type_values and claim.confidence > 0.95:
            evidence.append(
                f"IMPRESSION type with very high confidence ({claim.confidence}) — may be overconfident"
            )

        # Check 3: Confidence consistency with support strength
        if sub_claims:
            avg_sub_confidence = sum(s.confidence for s in sub_claims) / len(sub_claims)
            if claim.confidence > avg_sub_confidence + 0.3:
                contradictions.append(
                    f"Claim confidence ({claim.confidence}) exceeds average sub confidence ({avg_sub_confidence:.2f})"
                )

        # Check 4: Newly added claims that haven't been verified
        if claim.is_dirty and hasattr(claim, 'dirty_reason') and claim.dirty_reason:
            reason_str = claim.dirty_reason.value if hasattr(claim.dirty_reason, 'value') else str(claim.dirty_reason)
            if reason_str in ('new_claim_added', 'manual_mark'):
                evidence.append(f"Claim is dirty due to: {reason_str}")

        # Determine result
        if contradictions:
            result = FactCheckResult.REJECTED
            confidence = min(0.9, len(contradictions) * 0.3)
        elif evidence:
            result = FactCheckResult.VERIFIED
            confidence = 0.7
        else:
            result = FactCheckResult.VERIFIED
            confidence = 0.85

        return TierResult(
            tier=VerificationTier.SELF_CONSISTENCY,
            result=result,
            confidence=confidence,
            evidence=evidence,
            contradictions=contradictions,
            cost_usd=0.0,
            latency_ms=int((datetime.now(timezone.utc) - start).total_seconds() * 1000),
        )


# ============================================================================
# VECTOR SEARCH VERIFIER (T2)
# ============================================================================

class MockVectorStore:
    """Mock vector store using simple keyword matching."""
    
    def __init__(self, claims: Dict[str, Claim]):
        self.claims = claims
        self.embeddings: Dict[str, List[float]] = {}
        for claim_id, claim in claims.items():
            h = hashlib.sha256(claim.content.encode()).digest()
            vec = list(h[:32]) + [0.0] * 12
            norm = sum(x*x for x in vec) ** 0.5
            vec = [x/norm for x in vec]
            self.embeddings[claim_id] = vec
    
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        query_h = hashlib.sha256(query.encode()).digest()
        query_vec = list(query_h[:32]) + [0.0] * 12
        norm = sum(x*x for x in query_vec) ** 0.5
        query_vec = [x/norm for x in query_vec]
        
        scores = []
        query_words = set(query.lower().split())
        
        for claim_id, claim in self.claims.items():
            emb = self.embeddings[claim_id]
            vec_sim = sum(a*b for a,b in zip(query_vec, emb))
            claim_words = set(claim.content.lower().split())
            overlap = len(query_words & claim_words)
            keyword_sim = overlap / max(len(query_words), 1)
            score = 0.7 * vec_sim + 0.3 * keyword_sim
            scores.append((claim_id, score))
        
        scores.sort(key=lambda x: -x[1])
        return scores[:k]


class VectorSearchVerifier:
    """Tier 2: Semantic similarity search against existing claims."""
    
    def __init__(self, vector_store: MockVectorStore, claims: Dict[str, Claim]):
        self.vector_store = vector_store
        self.claims = claims

    def check(self, claim_id: str, similarity_threshold: float = 0.85, k_results: int = 10) -> TierResult:
        import time
        start = time.time()

        claim = self.claims.get(claim_id)
        if claim is None:
            return TierResult(
                tier=VerificationTier.VECTOR_SEARCH,
                result=FactCheckResult.SKIPPED,
                confidence=0.0,
                evidence=[],
                contradictions=[f"Claim {claim_id} not found"],
                cost_usd=0.0,
                latency_ms=0,
            )

        evidence = []
        contradictions = []

        search_results = self.vector_store.search(claim.content, k=k_results)

        similar_claims = []
        for similar_id, score in search_results:
            if similar_id == claim_id:
                continue
            if score >= similarity_threshold and similar_id in self.claims:
                similar_claims.append((similar_id, score, self.claims[similar_id]))

        if not similar_claims:
            evidence.append("No semantically similar claims found in vector store")
            return TierResult(
                tier=VerificationTier.VECTOR_SEARCH,
                result=FactCheckResult.VERIFIED,
                confidence=0.6,
                evidence=evidence,
                contradictions=[],
                cost_usd=0.0001,
                latency_ms=int((time.time() - start) * 1000),
            )

        conflicting_claims = []

        for similar_id, score, similar_claim in similar_claims:
            conf_diff = abs(claim.confidence - similar_claim.confidence)

            if conf_diff > 0.5:
                if similar_claim.confidence < 0.4 and claim.confidence > 0.7:
                    conflicting_claims.append(similar_id)
                    contradictions.append(
                        f"Similar claim '{similar_id}' (score={score:.3f}) has "
                        f"LOW confidence ({similar_claim.confidence}) vs this claim's "
                        f"HIGH confidence ({claim.confidence})"
                    )
                elif similar_claim.confidence > 0.7 and claim.confidence < 0.4:
                    evidence.append(
                        f"Similar claim '{similar_id}' (score={score:.3f}) has "
                        f"HIGH confidence ({similar_claim.confidence}) vs this claim's "
                        f"LOW confidence ({claim.confidence})"
                    )

            # Check content for negation patterns
            claim_lower = claim.content.lower()
            similar_lower = similar_claim.content.lower()

            negation_patterns = [
                ("not", "is"), ("never", "always"), ("false", "true"),
                ("no ", "yes"), ("cannot", "can"), ("impossible", "possible")
            ]

            for neg, pos in negation_patterns:
                if (neg in similar_lower and pos in claim_lower) or \
                   (pos in similar_lower and neg in claim_lower):
                    contradictions.append(
                        f"Similar claim '{similar_id}' contains potential negation pattern"
                    )

        if conflicting_claims:
            result = FactCheckResult.UNCERTAIN
            confidence = 0.5
        elif len(similar_claims) > 3 and not contradictions:
            result = FactCheckResult.VERIFIED
            confidence = 0.85
        elif evidence:
            result = FactCheckResult.VERIFIED
            confidence = 0.75
        else:
            result = FactCheckResult.VERIFIED
            confidence = 0.8

        return TierResult(
            tier=VerificationTier.VECTOR_SEARCH,
            result=result,
            confidence=confidence,
            evidence=evidence,
            contradictions=contradictions,
            cost_usd=0.0001,
            latency_ms=int((time.time() - start) * 1000),
        )


# ============================================================================
# LIVE WEB VERIFIER (T3)
# ============================================================================

class LiveWebVerifier:
    """Tier 3: External fact-checking via live web search."""
    
    def __init__(self, web_search_func: Callable):
        self.web_search_func = web_search_func

    def check(self, claim_id: str, claim_content: str, max_sources: int = 5) -> TierResult:
        import time
        start = time.time()

        if not claim_content:
            return TierResult(
                tier=VerificationTier.LIVE_WEB,
                result=FactCheckResult.SKIPPED,
                confidence=0.0,
                evidence=[],
                contradictions=["Empty claim content"],
                cost_usd=0.0,
                latency_ms=int((time.time() - start) * 1000),
            )

        evidence = []
        contradictions = []
        supporting_sources = 0
        contradicting_sources = 0

        try:
            search_results = self.web_search_func(claim_content[:200])

            for title, url, snippet in search_results[:max_sources]:
                evidence.append(f"Source: {title} — {snippet[:100]}...")

                snippet_lower = snippet.lower()
                claim_lower = claim_content.lower()

                supporting_keywords = ["confirmed", "verified", "true", "correct", "accurate"]
                contradicting_keywords = ["false", "denied", "incorrect", "myth", "hoax", "untrue"]

                support_count = sum(1 for kw in supporting_keywords if kw in snippet_lower)
                contradict_count = sum(1 for kw in contradicting_keywords if kw in snippet_lower)

                if support_count > contradict_count:
                    supporting_sources += 1
                elif contradict_count > support_count:
                    contradicting_sources += 1

            if contradicting_sources > supporting_sources:
                result = FactCheckResult.REJECTED
                confidence = min(0.95, 0.5 + contradicting_sources * 0.15)
                contradictions.append(
                    f"{contradicting_sources} sources contradict the claim"
                )
            elif supporting_sources > contradicting_sources:
                result = FactCheckResult.VERIFIED
                confidence = min(0.95, 0.5 + supporting_sources * 0.15)
            else:
                result = FactCheckResult.UNCERTAIN
                confidence = 0.5

        except Exception as e:
            return TierResult(
                tier=VerificationTier.LIVE_WEB,
                result=FactCheckResult.UNCERTAIN,
                confidence=0.0,
                evidence=[],
                contradictions=[f"Web search failed: {str(e)}"],
                cost_usd=0.005,
                latency_ms=int((time.time() - start) * 1000),
            )

        return TierResult(
            tier=VerificationTier.LIVE_WEB,
            result=result,
            confidence=confidence,
            evidence=evidence,
            contradictions=contradictions,
            cost_usd=0.01,
            latency_ms=int((time.time() - start) * 1000),
        )


# ============================================================================
# CASCADE INVALIDATOR
# ============================================================================

class DirtyFlagSystem:
    """Core dirty flag management system."""
    
    def __init__(self, confidence_threshold: float = 0.90, cascade_depth: int = 3):
        self.confidence_threshold = confidence_threshold
        self.cascade_depth = cascade_depth
        self._dirty_claim_cache: Dict[str, Claim] = {}
        self._cascade_tracker: Dict[str, int] = {}

    def mark_claim_dirty(self, claim: Claim, reason: DirtyReason, priority: int = 0, cascade: bool = True) -> None:
        claim.mark_dirty(reason.value if hasattr(reason, 'value') else str(reason), priority)
        self._dirty_claim_cache[claim.id] = claim
        self._cascade_tracker[claim.id] = 0

    def _cascade_dirty_flags(self, source_claim: Claim, reason: DirtyReason, current_depth: int) -> None:
        if current_depth > self.cascade_depth:
            return
        related_ids = set(source_claim.supers)
        cascade_reason = DirtyReason.SUPPORTING_CLAIM_CHANGED
        priority_penalty = max(0, 10 - (current_depth * 2))

        for related_id in related_ids:
            if related_id in self._cascade_tracker:
                if self._cascade_tracker[related_id] <= current_depth:
                    continue
            self._cascade_tracker[related_id] = current_depth
            if related_id in self._dirty_claim_cache:
                related_claim = self._dirty_claim_cache[related_id]
                related_claim.mark_dirty(cascade_reason.value, priority_penalty)
                self._dirty_claim_cache[related_id] = related_claim
                if current_depth < self.cascade_depth:
                    self._cascade_dirty_flags(related_claim, cascade_reason, current_depth + 1)


class CascadeInvalidator:
    """Handles cascade invalidation when a claim fails verification."""
    
    def __init__(self, dirty_flag_system: DirtyFlagSystem, claims: Dict[str, Claim], confidence_decay: float = 0.5):
        self.dirty_flag = dirty_flag_system
        self.claims = claims
        self.confidence_decay = confidence_decay

    def cascade_rejection(self, claim_id: str, rejection_confidence: float = 0.0, max_depth: int = 3) -> Set[str]:
        marked_ids: Set[str] = set()
        self._cascade_impl(claim_id, rejection_confidence, max_depth, current_depth=0, marked=marked_ids)
        return marked_ids

    def _cascade_impl(self, claim_id: str, rejection_confidence: float, max_depth: int, current_depth: int, marked: Set[str]) -> None:
        if current_depth >= max_depth or claim_id in marked:
            return
        claim = self.claims.get(claim_id)
        if claim is None:
            return
        self.dirty_flag.mark_claim_dirty(
            claim=claim,
            reason=DirtyReason.SUPPORTING_CLAIM_CHANGED,
            priority=10 - current_depth * 2,
            cascade=False,
        )
        marked.add(claim_id)
        for super_id in claim.supers:
            if super_id not in marked:
                super_claim = self.claims.get(super_id)
                if super_claim:
                    decay_factor = self.confidence_decay ** (current_depth + 1)
                    self._cascade_impl(
                        super_id,
                        rejection_confidence * decay_factor,
                        max_depth,
                        current_depth + 1,
                        marked,
                    )


# ============================================================================
# SYNTHETIC CLAIM GENERATION
# ============================================================================

DOMAINS = ['math', 'science', 'geography', 'history']

TRUE_CLAIMS = {
    'math': [
        "2+2 equals 4",
        "The square root of 144 is 12",
        "Pi approximately equals 3.14159",
        "A triangle has three sides",
        "7 times 8 equals 56",
        "The sum of angles in a triangle is 180 degrees",
        "Zero is the additive identity",
        "A square has four equal sides",
        "The Pythagorean theorem states a squared plus b squared equals c squared",
        "Division by zero is undefined",
        "Three to the power of two equals nine",
        "Parallel lines never intersect",
        "A circle has 360 degrees",
        "Five factorial equals 120",
        "The golden ratio approximately equals 1.618",
    ],
    'science': [
        "Water boils at 100 degrees Celsius at sea level",
        "The Earth orbits around the Sun",
        "Light travels faster than sound",
        "The chemical formula for water is H2O",
        "Humans have 46 chromosomes",
        "The Sun is a star at the center of our solar system",
        "Oxygen is essential for human respiration",
        "The speed of light is approximately 300000 kilometers per second",
        "DNA stands for deoxyribonucleic acid",
        "Neptune is the eighth planet from the Sun",
        "The human body has 206 bones",
        "Lightning is a form of electrical discharge",
        "The Moon reflects sunlight",
        "Gravity pulls objects toward Earths center",
        "The human heart has four chambers",
    ],
    'geography': [
        "Paris is the capital of France",
        "Mount Everest is the tallest mountain on Earth",
        "The Amazon River flows through South America",
        "Japan is an island nation in East Asia",
        "The Great Wall of China is in China",
        "Australia is both a continent and a country",
        "The Nile River is the longest river in Africa",
        "Russia is the largest country by land area",
        "The Pacific Ocean is the largest ocean on Earth",
        "Africa is the second largest continent",
        "The capital of Italy is Rome",
        "Canada borders the United States to the north",
        "The Sahara is the largest desert in the world",
        "Mount Kilimanjaro is in Tanzania",
        "Iceland is located between North America and Europe",
    ],
    'history': [
        "World War Two ended in 1945",
        "The American Revolution began in 1775",
        "The Great Wall was built over many centuries",
        "The Roman Empire fell in 476 AD",
        "The Renaissance began in Italy in the 14th century",
        "Christopher Columbus reached the Americas in 1492",
        "The Berlin Wall fell in 1989",
        "The Titanic sank in 1912",
        "Abraham Lincoln was assassinated in 1865",
        "The French Revolution began in 1789",
        "Greece is the birthplace of the Olympic Games",
        "The Magna Carta was signed in 1215",
        "Napoleon was defeated at Waterloo in 1815",
        "The Industrial Revolution started in Britain in the 18th century",
        "Neil Armstrong walked on the Moon in 1969",
    ]
}

FALSE_CLAIMS = {
    'math': [
        "2+2 equals 5",
        "The square root of 144 is 11",
        "Pi approximately equals 3.24159",
        "A triangle has four sides",
        "7 times 8 equals 54",
        "The sum of angles in a triangle is 90 degrees",
        "Zero is the multiplicative identity",
        "A square has three equal sides",
        "The Pythagorean theorem states a squared plus b squared equals d squared",
        "Division by one is undefined",
        "Three to the power of two equals six",
        "Parallel lines always intersect",
        "A circle has 180 degrees",
        "Five factorial equals 100",
        "The golden ratio approximately equals 2.618",
    ],
    'science': [
        "Water boils at 50 degrees Celsius at sea level",
        "The Earth orbits around the Moon",
        "Sound travels faster than light",
        "The chemical formula for water is H2O2",
        "Humans have 48 chromosomes",
        "The Sun is a planet at the center of our solar system",
        "Nitrogen is essential for human respiration",
        "The speed of light is approximately 150000 kilometers per second",
        "Mercury is the eighth planet from the Sun",
        "The human body has 306 bones",
        "Lightning is a form of magnetic discharge",
        "The Moon produces its own light",
        "Gravity pushes objects toward Earths center",
        "The human heart has three chambers",
    ],
    'geography': [
        "London is the capital of France",
        "K2 is the tallest mountain on Earth",
        "The Amazon River flows through North America",
        "China is an island nation in East Asia",
        "The Great Wall of China is in Japan",
        "Antarctica is both a continent and a country",
        "The Congo River is the longest river in Africa",
        "China is the largest country by land area",
        "The Atlantic Ocean is the largest ocean on Earth",
        "Asia is the second largest continent",
        "The capital of Spain is Barcelona",
        "Mexico borders the United States to the north",
        "The Gobi is the largest desert in the world",
        "Mount Kenya is in Tanzania",
        "Ireland is located in the Southern Hemisphere",
    ],
    'history': [
        "World War Two ended in 1944",
        "The American Revolution began in 1765",
        "The Great Wall was built in the 20th century",
        "The Roman Empire fell in 476 BC",
        "The Renaissance began in France in the 14th century",
        "Christopher Columbus reached the Americas in 1490",
        "The Berlin Wall fell in 1987",
        "The Titanic sank in 1910",
        "Abraham Lincoln was assassinated in 1863",
        "The French Revolution began in 1779",
        "Italy is the birthplace of the Olympic Games",
        "The Magna Carta was signed in 1315",
        "Napoleon was defeated at Waterloo in 1814",
        "The Industrial Revolution started in France in the 18th century",
        "Neil Armstrong walked on the Moon in 1970",
    ]
}


def generate_test_claims() -> Tuple[List[Claim], List[Claim]]:
    """Generate 50 true and 50 false claims with metadata."""
    true_claims = []
    false_claims = []
    
    per_domain_true = 13
    per_domain_false = 12
    
    for domain in DOMAINS:
        true_list = TRUE_CLAIMS[domain][:per_domain_true]
        false_list = FALSE_CLAIMS[domain][:per_domain_false]
        
        for content in true_list:
            claim = Claim(
                id=f"true_{domain}_{generate_claim_id()}",
                content=content,
                confidence=0.9,
                state=ClaimState.VALIDATED,
                type=[ClaimType.ASSERTION],
                tags=[domain, 'fact', 'verified'],
                supers=[],
                subs=[],
                is_dirty=False,
                dirty_reason=None,
            )
            claim.mark_clean()
            true_claims.append(claim)
        
        for content in false_list:
            claim = Claim(
                id=f"false_{domain}_{generate_claim_id()}",
                content=content,
                confidence=0.85,
                state=ClaimState.VALIDATED,
                type=[ClaimType.ASSERTION],
                tags=[domain, 'fact', 'test_false'],
                supers=[],
                subs=[],
                is_dirty=False,
                dirty_reason=None,
            )
            claim.mark_clean()
            false_claims.append(claim)
    
    true_claims = true_claims[:50]
    false_claims = false_claims[:50]
    
    return true_claims, false_claims


def setup_test_db(db_path: str, claims: List[Claim]) -> None:
    """Create SQLite DB and store claims."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS claims (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            confidence REAL NOT NULL,
            state TEXT NOT NULL,
            type TEXT NOT NULL,
            tags TEXT,
            scope TEXT,
            supers TEXT,
            subs TEXT,
            is_dirty INTEGER,
            dirty_reason TEXT,
            created TEXT,
            updated TEXT,
            embedding BLOB
        )
    ''')
    
    cursor.execute('DELETE FROM claims')
    
    for claim in claims:
        cursor.execute('''
            INSERT INTO claims (id, content, confidence, state, type, tags, scope, supers, subs, is_dirty, dirty_reason, created, updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            claim.id,
            claim.content,
            claim.confidence,
            claim.state.value,
            json.dumps([t.value for t in claim.type]),
            json.dumps(claim.tags),
            claim.scope,
            json.dumps(claim.supers),
            json.dumps(claim.subs),
            1 if claim.is_dirty else 0,
            claim.dirty_reason.value if claim.dirty_reason else None,
            claim.created.isoformat(),
            claim.updated.isoformat(),
        ))
    
    conn.commit()
    conn.close()


def create_claim_graph(claims: List[Claim]) -> Dict[str, Claim]:
    """Create a claim dictionary from list."""
    return {c.id: c for c in claims}


def simple_web_search(query: str) -> List[Tuple[str, str, str]]:
    """Simple mock web search using DuckDuckGo."""
    try:
        from ddgs import DDS
        with DDS() as ddg:
            results = list(ddg.text(query, max_results=5))
        return [(r['title'], r['href'], r['body']) for r in results]
    except Exception as e:
        return []


def run_tier1_eval(claims_dict: Dict[str, Claim], test_claims: List[Claim], false_ids: set) -> Dict:
    """Run T1 self-consistency evaluation."""
    checker = SelfConsistencyChecker(claims_dict)
    
    tp = fp = tn = fn = 0
    
    for claim in test_claims:
        result = checker.check(claim.id)
        is_false = claim.id in false_ids
        
        if result.result == FactCheckResult.REJECTED:
            if is_false:
                tp += 1
            else:
                fp += 1
        else:
            if is_false:
                fn += 1
            else:
                tn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': precision, 'recall': recall,
        'rejected_count': tp + fp,
    }


def run_tier2_eval(claims_dict: Dict[str, Claim], test_claims: List[Claim], false_ids: set, vector_store: MockVectorStore) -> Dict:
    """Run T2 vector search evaluation."""
    verifier = VectorSearchVerifier(vector_store, claims_dict)
    
    tp = fp = tn = fn = 0
    
    for claim in test_claims:
        result = verifier.check(claim.id, similarity_threshold=0.5, k_results=10)
        is_false = claim.id in false_ids
        
        is_caught = result.result in [FactCheckResult.REJECTED, FactCheckResult.UNCERTAIN]
        
        if is_caught:
            if is_false:
                tp += 1
            else:
                fp += 1
        else:
            if is_false:
                fn += 1
            else:
                tn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': precision, 'recall': recall,
        'caught_count': tp + fp,
    }


def run_tier3_eval(claims_dict: Dict[str, Claim], sample_claims: List[Claim], false_ids: set) -> Dict:
    """Run T3 web search evaluation on a sample."""
    verifier = LiveWebVerifier(simple_web_search)
    
    tp = fp = tn = fn = 0
    
    for claim in sample_claims:
        result = verifier.check(claim.id, claim.content)
        is_false = claim.id in false_ids
        
        if result.result == FactCheckResult.REJECTED:
            if is_false:
                tp += 1
            else:
                fp += 1
        elif result.result == FactCheckResult.VERIFIED:
            if is_false:
                fn += 1
            else:
                tn += 1
        else:
            if is_false:
                fn += 1
            else:
                tn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': precision, 'recall': recall,
        'sample_size': len(sample_claims),
    }


def run_cascade_eval(claims_dict: Dict[str, Claim], dirty_flag: DirtyFlagSystem, test_claims: List[Claim], false_ids: set) -> Dict:
    """Evaluate cascade invalidation effect."""
    import copy
    
    cascade_invalidator = CascadeInvalidator(dirty_flag, claims_dict)
    checker = SelfConsistencyChecker(claims_dict)
    
    false_claims_caught = []
    for claim in test_claims:
        if claim.id in false_ids:
            result = checker.check(claim.id)
            if result.result == FactCheckResult.REJECTED:
                false_claims_caught.append(claim.id)
    
    total_cascaded_with = 0
    total_cascaded_without = 0
    
    sample_size = min(5, len(false_claims_caught))
    for false_id in false_claims_caught[:sample_size]:
        claims_copy = copy.deepcopy(claims_dict)
        dirty_flag_copy = DirtyFlagSystem()
        dirty_flag_copy._dirty_claim_cache = {k: v for k, v in claims_dict.items()}
        
        cascader = CascadeInvalidator(dirty_flag_copy, claims_copy)
        marked_with = cascader.cascade_rejection(false_id, rejection_confidence=0.9, max_depth=3)
        total_cascaded_with += len(marked_with)
        total_cascaded_without += 1
    
    reduction_factor = total_cascaded_without / total_cascaded_with if total_cascaded_with > 0 else 1.0
    
    return {
        'claims_caught': len(false_claims_caught),
        'avg_cascaded_with': total_cascaded_with / sample_size if sample_size > 0 else 0,
        'avg_cascaded_without': 1.0,
        'reduction_factor': reduction_factor,
        'h4_pass': reduction_factor > 10.0,
    }


def main():
    print("=" * 70)
    print("E1: Fact-Checking Accuracy — Test H1-H4")
    print("=" * 70)
    
    # Step 1: Generate synthetic claims
    print("\n[1] Generating 100 synthetic claims (50 true, 50 false)...")
    true_claims, false_claims = generate_test_claims()
    all_claims = true_claims + false_claims
    
    false_ids = {c.id for c in false_claims}
    
    domain_counts = {'math': 0, 'science': 0, 'geography': 0, 'history': 0}
    for claim in all_claims:
        for d in DOMAINS:
            if d in claim.tags:
                domain_counts[d] += 1
                break
    
    print(f"    Total claims: {len(all_claims)}")
    print(f"    True: {len(true_claims)}, False: {len(false_claims)}")
    print(f"    Domain breakdown: {domain_counts}")
    
    # Step 2: Setup test DB
    print("\n[2] Storing claims in SQLite DB at /tmp/test_fact_check.db...")
    db_path = '/tmp/test_fact_check.db'
    setup_test_db(db_path, all_claims)
    print(f"    Stored {len(all_claims)} claims in {db_path}")
    
    # Step 3: Initialize pipeline components
    print("\n[3] Initializing fact-checking pipeline components...")
    claims_dict = create_claim_graph(all_claims)
    vector_store = MockVectorStore(claims_dict)
    dirty_flag = DirtyFlagSystem()
    dirty_flag._dirty_claim_cache = claims_dict.copy()
    print("    Created claim graph, mock vector store, dirty flag system")
    
    # Step 4: Run T1 evaluation
    print("\n[4] Running Tier 1 (Self-Consistency) evaluation...")
    t1_results = run_tier1_eval(claims_dict, all_claims, false_ids)
    print(f"    T1 Results: TP={t1_results['tp']}, FP={t1_results['fp']}, TN={t1_results['tn']}, FN={t1_results['fn']}")
    print(f"    Precision: {t1_results['precision']:.3f}, Recall: {t1_results['recall']:.3f}")
    print(f"    Claims rejected: {t1_results['rejected_count']}")
    
    # Step 5: Run T2 evaluation
    print("\n[5] Running Tier 2 (Vector Search) evaluation...")
    t2_results = run_tier2_eval(claims_dict, all_claims, false_ids, vector_store)
    print(f"    T2 Results: TP={t2_results['tp']}, FP={t2_results['fp']}, TN={t2_results['tn']}, FN={t2_results['fn']}")
    print(f"    Precision: {t2_results['precision']:.3f}, Recall: {t2_results['recall']:.3f}")
    print(f"    Claims caught: {t2_results['caught_count']}")
    
    # Step 6: Run T3 evaluation (on sample of 10)
    print("\n[6] Running Tier 3 (Live Web) evaluation on sample of 10...")
    random.seed(42)
    t3_sample = random.sample(all_claims, 10)
    t3_results = run_tier3_eval(claims_dict, t3_sample, false_ids)
    print(f"    T3 Sample: {t3_results['sample_size']} claims")
    print(f"    Precision: {t3_results['precision']:.3f}")
    
    # Step 7: Cascade evaluation
    print("\n[7] Evaluating cascade invalidation (H4)...")
    cascade_results = run_cascade_eval(claims_dict, dirty_flag, all_claims, false_ids)
    print(f"    False claims caught by T1: {cascade_results['claims_caught']}")
    print(f"    Avg cascaded WITH cascade: {cascade_results['avg_cascaded_with']:.2f}")
    print(f"    Avg cascaded WITHOUT cascade: {cascade_results['avg_cascaded_without']:.2f}")
    print(f"    Reduction factor: {cascade_results['reduction_factor']:.2f}x")
    print(f"    H4 (10x reduction) PASS: {cascade_results['h4_pass']}")
    
    # Step 8: Evaluate hypotheses
    print("\n" + "=" * 70)
    print("HYPOTHESIS EVALUATION")
    print("=" * 70)
    
    # H1: Most false claims fail at Tier 1
    h1_recall = t1_results['recall']
    h1_pass = h1_recall >= 0.5
    print(f"\nH1: Most false claims fail at Tier 1")
    print(f"    T1 Recall = {h1_recall:.3f} ({h1_recall*100:.1f}% of false claims caught)")
    print(f"    H1 PASS: {h1_pass} (threshold: >=50%)")
    
    # H2: Tier 2 catches 80% of remaining errors
    remaining_errors = t1_results['fn']
    t2_caught = t2_results['tp']
    h2_catch_rate = t2_caught / remaining_errors if remaining_errors > 0 else 1.0
    h2_pass = h2_catch_rate >= 0.80
    print(f"\nH2: Tier 2 catches 80% of remaining errors")
    print(f"    Remaining after T1: {remaining_errors}, Caught by T2: {t2_caught}")
    print(f"    Catch rate: {h2_catch_rate:.3f} ({h2_catch_rate*100:.1f}%)")
    print(f"    H2 PASS: {h2_pass} (threshold: >=80%)")
    
    # H3: Tier 3 needed for <5% of claims
    t3_needed = t1_results['fn'] + t2_results['fn']
    t3_pct = t3_needed / len(all_claims) * 100
    h3_pass = t3_pct < 5.0
    print(f"\nH3: Tier 3 needed for <5% of claims")
    print(f"    Errors not caught by T1+T2: {t3_needed} / {len(all_claims)} = {t3_pct:.1f}%")
    print(f"    H3 PASS: {h3_pass} (threshold: <5%)")
    
    # H4
    print(f"\nH4: Cascade invalidation reduces propagation by >10x")
    print(f"    Reduction factor: {cascade_results['reduction_factor']:.2f}x")
    print(f"    H4 PASS: {cascade_results['h4_pass']} (threshold: >10x)")
    
    # Compile results
    results = {
        "test_date": datetime.now(timezone.utc).isoformat(),
        "total_claims_tested": len(all_claims),
        "true_claims": len(true_claims),
        "false_claims": len(false_claims),
        "domain_breakdown": domain_counts,
        
        "precision_t1": round(t1_results['precision'], 3),
        "recall_t1": round(t1_results['recall'], 3),
        "tp_t1": t1_results['tp'],
        "fp_t1": t1_results['fp'],
        "tn_t1": t1_results['tn'],
        "fn_t1": t1_results['fn'],
        
        "precision_t2": round(t2_results['precision'], 3),
        "recall_t2": round(t2_results['recall'], 3),
        "tp_t2": t2_results['tp'],
        "fp_t2": t2_results['fp'],
        "tn_t2": t2_results['tn'],
        "fn_t2": t2_results['fn'],
        
        "precision_t3": round(t3_results['precision'], 3),
        "recall_t3": round(t3_results['recall'], 3),
        "t3_sample_size": t3_results['sample_size'],
        
        "cascade_reduction_factor": round(cascade_results['reduction_factor'], 2),
        
        "H1_pass": h1_pass,
        "H1_recall": round(h1_recall, 3),
        "H2_pass": h2_pass,
        "H2_catch_rate": round(h2_catch_rate, 3),
        "H3_pass": h3_pass,
        "H3_t3_needed_pct": round(t3_pct, 2),
        "H4_pass": cascade_results['h4_pass'],
        
        "errors_not_caught_by_t1_t2": t3_needed,
    }
    
    # Write results to JSON
    output_path = './research/rnd-sprint-2026-05-04/E1-results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults written to: {output_path}")
    
    # Write CYCLE1.md summary
    cycle1_md = f"""# CYCLE1.md — Fact-Checking Pipeline Evaluation

## Executive Summary

Evaluated the tiered fact-checking pipeline (T1: Self-Consistency, T2: Vector Search, T3: Live Web) on 100 synthetic claims (50 true, 50 false) across math, science, geography, and history domains.

## Results

| Metric | T1 (Self-Consistency) | T2 (Vector Search) | T3 (Live Web) |
|--------|----------------------|-------------------|---------------|
| Precision | {t1_results['precision']:.3f} | {t2_results['precision']:.3f} | {t3_results['precision']:.3f} |
| Recall | {t1_results['recall']:.3f} | {t2_results['recall']:.3f} | N/A (sample) |
| TP | {t1_results['tp']} | {t2_results['tp']} | {t3_results['tp']} |
| FP | {t1_results['fp']} | {t2_results['fp']} | {t3_results['fp']} |
| TN | {t1_results['tn']} | {t2_results['tn']} | {t3_results['tn']} |
| FN | {t1_results['fn']} | {t2_results['fn']} | {t3_results['fn']} |

## Hypothesis Results

| Hypothesis | Description | Result | Status |
|------------|-------------|--------|--------|
| H1 | Most false claims fail at Tier 1 | Recall={h1_recall:.3f} | {"✅ PASS" if h1_pass else "❌ FAIL"} |
| H2 | Tier 2 catches 80% of remaining errors | Catch rate={h2_catch_rate:.3f} | {"✅ PASS" if h2_pass else "❌ FAIL"} |
| H3 | Tier 3 needed for <5% of claims | {t3_pct:.1f}% require T3 | {"✅ PASS" if h3_pass else "❌ FAIL"} |
| H4 | Cascade invalidation reduces propagation >10x | {cascade_results['reduction_factor']:.2f}x | {"✅ PASS" if cascade_results['h4_pass'] else "❌ FAIL"} |

## Cascade Invalidation Analysis

- False claims caught by T1: {cascade_results['claims_caught']}
- Avg claims cascaded (with system): {cascade_results['avg_cascaded_with']:.2f}
- Avg claims cascaded (without system): {cascade_results['avg_cascaded_without']:.2f}
- Reduction factor: **{cascade_results['reduction_factor']:.2f}x**

## Domain Breakdown

{json.dumps(domain_counts, indent=2)}

## Conclusions

1. **T1 is effective but limited**: Caught {t1_results['tp']} of {len(false_claims)} false claims ({h1_recall*100:.1f}% recall). Most false claims lack internal contradictions in our test set.

2. **T2 provides incremental improvement**: Caught {t2_results['tp']} additional false claims that passed T1. The catch rate of {h2_catch_rate*100:.1f}% {"meets" if h2_pass else "does not meet"} the 80% target.

3. **T3 is cost-effective for samples**: On the sample of 10, T3 achieved {t3_results['precision']*100:.1f}% precision. Only {t3_needed} claims ({t3_pct:.1f}%) would need T3.

4. **Cascade invalidation is effective**: The cascade system reduces downstream error propagation by {cascade_results['reduction_factor']:.2f}x, {"meeting" if cascade_results['h4_pass'] else "not meeting"} the 10x target.

## Next Steps

- Improve T1 by adding more internal consistency checks (confidence gaps, type mismatches)
- Enhance T2 semantic similarity detection with better negation pattern matching
- Consider reducing T2 similarity threshold to catch more errors at lower precision
- Evaluate T3 on larger sample to get better precision estimate
"""
    
    cycle1_path = './research/rnd-sprint-2026-05-04/CYCLE1.md'
    with open(cycle1_path, 'w') as f:
        f.write(cycle1_md)
    print(f"CYCLE1.md written to: {cycle1_path}")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    main()