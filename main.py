import os

from dotenv import load_dotenv

from src.rag.config import VECTORSTORE_DIR
from src.rag.embeddings import get_embeddings
from src.rag.vectorstore import load_vectorstore
from src.rag.bm25 import BM25Retriever
from src.rag.reranker import CrossEncoderReranker
from src.rag.retriever import HybridRerankRetriever
from src.rag.generator import Generator
from src.rag.pipeline import RAGPipeline


def create_pipeline():

    # =================================================
    # Load Embedding Model
    # =================================================

    print("\nLoading embedding model...")

    embeddings = get_embeddings()


    # =================================================
    # Load FAISS Vector Store
    # =================================================

    print("Loading FAISS vector store...")

    vectorstore = load_vectorstore(
        embeddings,
        VECTORSTORE_DIR
    )


    # =================================================
    # Load BM25
    # =================================================

    print("Loading BM25 index...")

    bm25 = BM25Retriever.load(
        VECTORSTORE_DIR / "bm25.pkl"
    )


    # =================================================
    # Load Cross Encoder
    # =================================================

    print(
        "Loading cross-encoder reranker..."
    )

    reranker = CrossEncoderReranker()


    # =================================================
    # Create Hybrid Retriever
    # =================================================

    retriever = HybridRerankRetriever(
        vectorstore=vectorstore,
        bm25_retriever=bm25,
        reranker=reranker
    )


    # =================================================
    # Gemini Generator
    # =================================================

    print(
        "Loading Gemini generator..."
    )

    generator = Generator()


    # =================================================
    # Create RAG Pipeline
    # =================================================

    rag = RAGPipeline(
        retriever=retriever,
        generator=generator
    )

    return rag


def main():

    # =================================================
    # Load environment variables
    # =================================================

    load_dotenv()


    # =================================================
    # Check Gemini API Key
    # =================================================

    if not os.getenv("GOOGLE_API_KEY"):

        raise ValueError(
            "GOOGLE_API_KEY not found in .env"
        )


    # =================================================
    # Header
    # =================================================

    print("=" * 70)

    print(
        "V5 — ADVANCED RAG + "
        "RERANKING + CITATION + EVALUATION"
    )

    print("=" * 70)


    # =================================================
    # Create Pipeline
    # =================================================

    rag = create_pipeline()


    # =================================================
    # System Ready
    # =================================================

    print("\n" + "=" * 70)

    print("RAG SYSTEM READY")

    print("=" * 70)

    print(
        "\nFeatures:"
    )

    print(
        "✓ Dense Retrieval (FAISS)"
    )

    print(
        "✓ Sparse Retrieval (BM25)"
    )

    print(
        "✓ Reciprocal Rank Fusion (RRF)"
    )

    print(
        "✓ Cross-Encoder Reranking"
    )

    print(
        "✓ Top-5 Evidence Selection"
    )

    print(
        "✓ Citation-Aware Gemini Generation"
    )

    print(
        "✓ Structured Claims"
    )

    print(
        "✓ Citation Validation"
    )

    print(
        "✓ Citation Entailment"
    )

    print(
        "✓ Automatic Retry"
    )

    print(
        "✓ V5 Evaluation Support"
    )

    print(
        "\nType 'exit' to quit."
    )


    # =================================================
    # Chat Loop
    # =================================================

    while True:

        print("\n" + "-" * 70)

        question = input(
            "You: "
        ).strip()


        # -------------------------------------------------
        # Exit
        # -------------------------------------------------

        if question.lower() == "exit":

            print(
                "\nExiting RAG system..."
            )

            break


        # -------------------------------------------------
        # Empty question
        # -------------------------------------------------

        if not question:

            continue


        # =================================================
        # Run RAG Pipeline
        # =================================================

        try:

            result = rag.run(
                question
            )

        except Exception as e:

            print(
                "\nError while running RAG pipeline:"
            )

            print(
                f"{type(e).__name__}: {e}"
            )

            continue


        # =================================================
        # Assistant Answer
        # =================================================

        print(
            "\nAssistant:"
        )

        print(
            result.get(
                "answer",
                "No answer generated."
            )
        )


        # =================================================
        # Claims + Citation IDs
        # =================================================

        claims = result.get(
            "claims",
            []
        )


        if claims:

            print(
                "\nClaims + Citations:"
            )

            for index, claim in enumerate(
                claims,
                start=1
            ):

                claim_text = claim.get(
                    "claim",
                    ""
                )

                citations = claim.get(
                    "citations",
                    []
                )

                print(
                    f"\n{index}. {claim_text}"
                )

                print(
                    "   Citations: "
                    f"{', '.join(citations)}"
                )


        # =================================================
        # Formatted Citations
        # =================================================

        citations = result.get(
            "citations",
            []
        )


        if citations:

            print(
                "\nSources:"
            )

            for citation in citations:

                print(
                    f"  • {citation}"
                )


        # =================================================
        # Validation Status
        # =================================================

        validation_status = result.get(
            "validation_status",
            "unknown"
        )


        print(
            "\nCitation Validation:"
        )

        print(
            f"  Status: {validation_status}"
        )


        # =================================================
        # Validation Errors
        # =================================================

        validation_errors = result.get(
            "validation_errors",
            []
        )


        if validation_errors:

            print(
                "\nValidation Errors:"
            )

            for error in validation_errors:

                print(
                    f"  • {error}"
                )


        # =================================================
        # V5 — Claim Validation
        # =================================================

        claim_validation = result.get(
            "claim_validation",
            []
        )


        if claim_validation:

            print(
                "\nClaim Entailment:"
            )

            for index, supported in enumerate(
                claim_validation,
                start=1
            ):

                status = (
                    "SUPPORTED"
                    if supported
                    else "NOT SUPPORTED"
                )

                print(
                    f"  Claim {index}: "
                    f"{status}"
                )


        # =================================================
        # Final Reranked Chunks
        # =================================================

        retrieved_results = result.get(
            "retrieved_results",
            []
        )


        if retrieved_results:

            print(
                "\nFinal Reranked Chunks:"
            )


            for i, item in enumerate(
                retrieved_results,
                start=1
            ):

                document = item[
                    "document"
                ]


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


                rrf_score = item.get(
                    "rrf_score",
                    0.0
                )


                rerank_score = item.get(
                    "rerank_score",
                    0.0
                )


                print(
                    f"{i}. "
                    f"{chunk_id} | "
                    f"{source} | "
                    f"Page: {page} | "
                    f"RRF: {rrf_score:.4f} | "
                    f"Rerank: {rerank_score:.4f}"
                )


        print()


# =====================================================
# Entry Point
# =====================================================

if __name__ == "__main__":

    main()