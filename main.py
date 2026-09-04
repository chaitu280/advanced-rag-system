import os

from dotenv import load_dotenv

from src.rag.config import (
    VECTORSTORE_DIR
)

from src.rag.embeddings import (
    get_embeddings
)

from src.rag.vectorstore import (
    load_vectorstore
)

from src.rag.retriever import (
    Retriever
)

from src.rag.generator import (
    Generator
)

from src.rag.pipeline import (
    RAGPipeline
)


def main():

    # Load variables from .env
    load_dotenv()

    # Check Gemini API key
    if not os.getenv(
        "GOOGLE_API_KEY"
    ):

        raise ValueError(
            "GOOGLE_API_KEY not found in .env"
        )

    print("=" * 60)
    print("NORMAL RAG SYSTEM - GEMINI")
    print("=" * 60)

    # Embedding model
    embeddings = get_embeddings()

    # Load FAISS
    vectorstore = load_vectorstore(
        embeddings,
        VECTORSTORE_DIR
    )

    # Retriever
    retriever = Retriever(
        vectorstore
    )

    # Generator
    generator = Generator()

    # RAG pipeline
    rag = RAGPipeline(
        retriever,
        generator
    )

    print("\nRAG system ready!")
    print("Type 'exit' to quit.\n")

    while True:

        question = input(
            "You: "
        ).strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        result = rag.run(
            question
        )

        print("\nAssistant:")
        print(result["answer"])

        print("\nRetrieved Sources:")

        for i, document in enumerate(
            result["documents"],
            start=1
        ):

            print(
                f"{i}. "
                f"{document.metadata.get('source', 'Unknown')} "
                f"| Page: "
                f"{document.metadata.get('page', 'Unknown')}"
            )

        print()


if __name__ == "__main__":
    main()