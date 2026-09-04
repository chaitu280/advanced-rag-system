from pathlib import Path


# Project root
BASE_DIR = Path(__file__).resolve().parents[2]


# Data
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"


# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


# Retrieval
TOP_K = 5


# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# LLM
LLM_MODEL = "gemini-2.5-flash"
TEMPERATURE = 0