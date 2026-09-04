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

from src.rag.bm25 import (
    BM25Retriever
)

from src.rag.retriever import (
    HybridRetriever
)

from src.rag.generator import (
    Generator
)

from src.rag.pipeline import (
    RAGPipeline
)


def main():

    # Load environment variables
    load_dotenv()

    # Check Gemini API key
    if not os.getenv(
        "GOOGLE_API_KEY"
    ):

        raise ValueError(
            "GOOGLE_API_KEY not found in .env"
        )

    print("=" * 60)

    print("HYBRID RAG SYSTEM - GEMINI")

    print("=" * 60)


    # -----------------------------------------
    # Embeddings
    # -----------------------------------------

    embeddings = get_embeddings()


    # -----------------------------------------
    # FAISS
    # -----------------------------------------

    vectorstore = load_vectorstore(
        embeddings,
        VECTORSTORE_DIR
    )


    # -----------------------------------------
    # BM25
    # -----------------------------------------

    bm25 = BM25Retriever.load(
        VECTORSTORE_DIR / "bm25.pkl"
    )


    # -----------------------------------------
    # Hybrid Retriever
    # -----------------------------------------

    retriever = HybridRetriever(
        vectorstore,
        bm25
    )


    # -----------------------------------------
    # Generator - Gemini
    # -----------------------------------------

    generator = Generator()


    # -----------------------------------------
    # RAG Pipeline
    # -----------------------------------------

    rag = RAGPipeline(
        retriever,
        generator
    )


    print(
        "\nHybrid RAG system ready!"
    )

    print(
        "Type 'exit' to quit.\n"
    )


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

        print(
            result["answer"]
        )


        print(
            "\nRetrieved Chunks:"
        )


        for i, item in enumerate(
            result["results"],
            start=1
        ):

            document = item["document"]

            score = item["score"]

            chunk_id = document.metadata.get(
                "chunk_id",
                "Unknown"
            )

            source = document.metadata.get(
                "source",
                "Unknown"
            )

            page = document.metadata.get(
                "page",
                "Unknown"
            )


            print(
                f"{i}. "
                f"{chunk_id} | "
                f"{source} | "
                f"Page: {page} | "
                f"RRF: {score:.4f}"
            )


        print()


if __name__ == "__main__":

    main()