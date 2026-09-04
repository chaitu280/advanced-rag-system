from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]


# --------------------------------------------------
# DIRECTORIES
# --------------------------------------------------

DOCUMENTS_DIR = BASE_DIR / "data" / "documents"

VECTORSTORE_DIR = BASE_DIR / "vectorstore"


# --------------------------------------------------
# CHUNKING
# --------------------------------------------------

CHUNK_SIZE = 800

CHUNK_OVERLAP = 150


# --------------------------------------------------
# RETRIEVAL
# --------------------------------------------------

DENSE_TOP_K = 10

BM25_TOP_K = 10

FINAL_TOP_K = 5

RRF_K = 60


# --------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# LLM
# --------------------------------------------------

LLM_MODEL = "gemini-2.5-flash"

TEMPERATURE = 0