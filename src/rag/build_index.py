from pathlib import Path

from rag.config import (
    DOCUMENTS_DIR,
    VECTORSTORE_DIR,
)

from rag.ingestion import load_documents
from rag.chunking import split_documents
from rag.embeddings import get_embeddings
from rag.vectorstore import create_vectorstore
from rag.bm25 import BM25Retriever


def build_index():

    print("=" * 60)
    print("BUILDING HYBRID RAG INDEX")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load documents
    # --------------------------------------------------

    print("\n[1/5] Loading documents...")

    documents = load_documents(
        DOCUMENTS_DIR
    )

    print(
        f"Loaded {len(documents)} documents/pages."
    )


    # --------------------------------------------------
    # 2. Split documents into chunks
    # --------------------------------------------------

    print("\n[2/5] Creating chunks...")

    chunks = split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )


    # --------------------------------------------------
    # 3. Create embeddings
    # --------------------------------------------------

    print("\n[3/5] Creating embeddings...")

    embeddings = get_embeddings()


    # --------------------------------------------------
    # 4. Create FAISS vector store
    # --------------------------------------------------

    print("\n[4/5] Creating FAISS index...")

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    create_vectorstore(
        chunks,
        embeddings,
        VECTORSTORE_DIR
    )


    # --------------------------------------------------
    # 5. Create BM25 index
    # --------------------------------------------------

    print("\n[5/5] Creating BM25 index...")

    bm25 = BM25Retriever(
        chunks
    )

    bm25_path = (
        VECTORSTORE_DIR / "bm25.pkl"
    )

    bm25.save(
        bm25_path
    )

    print(
        f"BM25 index saved to: {bm25_path}"
    )


    # --------------------------------------------------
    # Complete
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("HYBRID RAG INDEX BUILD COMPLETE")
    print("=" * 60)

    print("\nCreated:")

    print(
        f"FAISS index: {VECTORSTORE_DIR}"
    )

    print(
        f"BM25 index:  {bm25_path}"
    )

    print(
        f"Total chunks: {len(chunks)}"
    )


if __name__ == "__main__":

    build_index()