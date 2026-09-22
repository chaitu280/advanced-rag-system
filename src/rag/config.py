from pathlib import Path
import os


# ==================================================
# BASE DIRECTORY
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[2]


# ==================================================
# DIRECTORIES
# ==================================================

DOCUMENTS_DIR = BASE_DIR / "data" / "documents"

VECTORSTORE_DIR = BASE_DIR / "vectorstore"

EVALUATION_DIR = BASE_DIR / "evaluation"


# ==================================================
# CHUNKING
# ==================================================

CHUNK_SIZE = 800

CHUNK_OVERLAP = 150


# ==================================================
# RETRIEVAL
# ==================================================

DENSE_TOP_K = 10

BM25_TOP_K = 10

RRF_K = 60

RERANK_TOP_K = 5

RERANK_SCORE_THRESHOLD = 0.0


# ==================================================
# V4 — CITATION VALIDATION
# ==================================================

MAX_GENERATION_RETRIES = 2

REQUIRE_CITATION = True


# ==================================================
# V5 — EVALUATION
# ==================================================

# Number of retrieved chunks used for
# retrieval evaluation.
EVALUATION_TOP_K = 5

# Enable / disable evaluation
ENABLE_EVALUATION = True

# Save detailed evaluation results
SAVE_EVALUATION_RESULTS = True


# ==================================================
# V5 — MONITORING
# ==================================================

# Enable latency measurement
ENABLE_MONITORING = True


# ==================================================
# MODELS
# ==================================================

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

RERANKER_MODEL = (
    "BAAI/bge-reranker-base"
)


# ==================================================
# LLM — GEMINI
# ==================================================

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "gemini-3.8-flash"
)

TEMPERATURE = float(
    os.getenv(
        "TEMPERATURE",
        "0"
    )
)


# ==================================================
# GEMINI API
# ==================================================

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)