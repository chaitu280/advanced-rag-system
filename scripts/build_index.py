from src.rag.config import (
    DOCUMENTS_DIR,
    VECTORSTORE_DIR
)

from src.rag.ingestion import (
    load_documents
)

from src.rag.chunking import (
    split_documents
)

from src.rag.embeddings import (
    get_embeddings
)

from src.rag.vectorstore import (
    create_vectorstore
)

from src.rag.bm25 import (
    BM25Retriever
)


def main():

    print("=" * 60)
    print("BUILDING HYBRID RAG INDEX")
    print("=" * 60)

    # ---------------------------------------------
    # 1. Load documents
    # ---------------------------------------------

    documents = load_documents(
        DOCUMENTS_DIR
    )

    print(
        f"Loaded {len(documents)} document pages."
    )

    # ---------------------------------------------
    # 2. Chunk documents
    # ---------------------------------------------

    chunks = split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    # ---------------------------------------------
    # 3. Embeddings
    # ---------------------------------------------

    embeddings = get_embeddings()

    # ---------------------------------------------
    # 4. FAISS
    # ---------------------------------------------

    print(
        "Creating FAISS vector store..."
    )

    create_vectorstore(
        chunks,
        embeddings,
        VECTORSTORE_DIR
    )

    print(
        f"FAISS vector store saved to "
        f"{VECTORSTORE_DIR}"
    )

    # ---------------------------------------------
    # 5. BM25
    # ---------------------------------------------

    print(
        "Creating BM25 index..."
    )

    bm25 = BM25Retriever(
        chunks
    )

    bm25_path = (
        VECTORSTORE_DIR /
        "bm25.pkl"
    )

    bm25.save(
        bm25_path
    )

    print(
        f"BM25 index saved to: "
        f"{bm25_path}"
    )

    # ---------------------------------------------
    # 6. Verify BM25 file
    # ---------------------------------------------

    if bm25_path.exists():

        print(
            "SUCCESS: bm25.pkl created!"
        )

    else:

        raise FileNotFoundError(
            f"BM25 file was not created: {bm25_path}"
        )

    print(
        "\nHybrid indexing completed!"
    )


if __name__ == "__main__":
    main()