from src.rag.config import (
    DOCUMENTS_DIR,
    VECTORSTORE_DIR,
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


def main():

    print("=" * 60)
    print("BUILDING RAG INDEX")
    print("=" * 60)

    # 1. Load documents
    documents = load_documents(
        DOCUMENTS_DIR
    )

    # 2. Chunk documents
    chunks = split_documents(
        documents
    )

    # 3. Load embedding model
    embeddings = get_embeddings()

    # 4. Create vector store
    create_vectorstore(
        chunks,
        embeddings,
        VECTORSTORE_DIR
    )

    print("\nIndexing completed!")


if __name__ == "__main__":
    main()