"""
Configuration settings for Conjecture system
"""

import os
from pathlib import Path
from typing import List

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_PATH = str(DATA_DIR / "chroma_db")

# ChromaDB settings
CHROMA_COLLECTION_NAME = "claims"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000

# Claim processing settings
VALIDATION_THRESHOLD = 0.95
SIMILARITY_THRESHOLD = 0.7
MAX_CONTEXT_CONCEPTS = 10
MAX_CONTEXT_REFERENCES = 8
MAX_CONTEXT_SKILLS = 5
MAX_CONTEXT_GOALS = 3
EXPLORATION_BATCH_SIZE = 10

# LLM settings
LLM_MODEL = "gpt-3.5-turbo"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 2000
LLM_TIMEOUT = 30

# Performance settings
QUERY_TIMEOUT = 100  # ms
BATCH_SIZE = 50
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Claim type definitions
CLAIM_TYPES = ["concept", "reference", "thesis", "example", "goal"]

# Claim state definitions
CLAIM_STATES = ["Explore", "Validated", "Orphaned", "Queued"]

# Claim type confidence ranges
TYPE_CONFIDENCE_RANGES = {
    "concept": [0.1, 0.95],
    "reference": [0.3, 0.95],
    "thesis": [0.2, 0.95],
    
    "example": [0.6, 0.95],
    "goal": [0.1, 0.95],
}

# Context query priorities
CONTEXT_PRIORITIES = {
    "concept": (0.3, 0.7),  # (similarity_weight, confidence_weight)
    "reference": (0.2, 0.8),
    
    "goal": (0.4, 0.6),
    "thesis": (0.6, 0.4),
}

# Validation rules
MIN_CONTENT_LENGTH = 5
MAX_CONTENT_LENGTH = 2000
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0

# Create data directory if it doesn't exist
DATA_DIR.mkdir(exist_ok=True)

# Dirty Flag System Settings
DIRTY_FLAG_CONFIDENCE_THRESHOLD = 0.90
DIRTY_FLAG_CASCADE_DEPTH = 3
DIRTY_FLAG_BATCH_SIZE = 5
DIRTY_FLAG_MAX_PARALLEL_BATCHES = 3
DIRTY_FLAG_CONFIDENCE_BOOST_FACTOR = 0.10
DIRTY_FLAG_TWO_PASS_EVALUATION = True
DIRTY_FLAG_RELATIONSHIP_THRESHOLD = 0.7
DIRTY_FLAG_TIMEOUT_SECONDS = 300
DIRTY_FLAG_MAX_RETRIES = 2
DIRTY_FLAG_AUTO_EVALUATION_ENABLED = True
DIRTY_FLAG_EVALUATION_INTERVAL_MINUTES = 30
DIRTY_FLAG_SIMILARITY_THRESHOLD = 0.7

# Dirty flag priority settings
DIRTY_FLAG_PRIORITY_WEIGHTS = {
    "confidence_gap": 10.0,  # Weight for confidence below threshold
    "new_claim": 5.0,        # Priority for new claim additions
    "confidence_threshold": 15.0,  # Priority when below threshold
    "supporting_claim_changed": 8.0,
    "relationship_changed": 6.0,
    "manual_mark": 20.0,     # Highest priority for manual marks
    "batch_evaluation": 3.0,
    "system_trigger": 4.0
}

# Dirty flag evaluation settings
DIRTY_FLAG_ENABLE_CASCADE = True
DIRTY_FLAG_MAX_CLAIMS_PER_EVALUATION = 50
DIRTY_FLAG_MIN_DIRTY_CLAIMS_BATCH = 2
DIRTY_FLAG_CACHE_INVALIDATION_MINUTES = 60

# Environment overrides
CHROMA_PATH = os.getenv("CHROMA_PATH", CHROMA_PATH)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL)
LLM_MODEL = os.getenv("LLM_MODEL", LLM_MODEL)
LOG_LEVEL = os.getenv("LOG_LEVEL", LOG_LEVEL)

# Dirty flag environment overrides
DIRTY_FLAG_CONFIDENCE_THRESHOLD = float(os.getenv("DIRTY_FLAG_CONFIDENCE_THRESHOLD", DIRTY_FLAG_CONFIDENCE_THRESHOLD))
DIRTY_FLAG_CASCADE_DEPTH = int(os.getenv("DIRTY_FLAG_CASCADE_DEPTH", DIRTY_FLAG_CASCADE_DEPTH))
DIRTY_FLAG_BATCH_SIZE = int(os.getenv("DIRTY_FLAG_BATCH_SIZE", DIRTY_FLAG_BATCH_SIZE))
DIRTY_FLAG_MAX_PARALLEL_BATCHES = int(os.getenv("DIRTY_FLAG_MAX_PARALLEL_BATCHES", DIRTY_FLAG_MAX_PARALLEL_BATCHES))
DIRTY_FLAG_CONFIDENCE_BOOST_FACTOR = float(os.getenv("DIRTY_FLAG_CONFIDENCE_BOOST_FACTOR", DIRTY_FLAG_CONFIDENCE_BOOST_FACTOR))
DIRTY_FLAG_TWO_PASS_EVALUATION = os.getenv("DIRTY_FLAG_TWO_PASS_EVALUATION", str(DIRTY_FLAG_TWO_PASS_EVALUATION)).lower() == "true"
DIRTY_FLAG_AUTO_EVALUATION_ENABLED = os.getenv("DIRTY_FLAG_AUTO_EVALUATION_ENABLED", str(DIRTY_FLAG_AUTO_EVALUATION_ENABLED)).lower() == "true"
DIRTY_FLAG_EVALUATION_INTERVAL_MINUTES = int(os.getenv("DIRTY_FLAG_EVALUATION_INTERVAL_MINUTES", DIRTY_FLAG_EVALUATION_INTERVAL_MINUTES))
