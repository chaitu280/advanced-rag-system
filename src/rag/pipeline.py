from .retriever import HybridRetriever
from .generator import Generator

from .config import FINAL_TOP_K


class RAGPipeline:

    def __init__(
        self,
        retriever: HybridRetriever,
        generator: Generator
    ):

        self.retriever = retriever

        self.generator = generator


    def build_context(
        self,
        results
    ):

        context_parts = []

        for result in results:

            document = result["document"]

            source = document.metadata.get(
                "source",
                "Unknown"
            )

            page = document.metadata.get(
                "page",
                "Unknown"
            )

            chunk_id = document.metadata.get(
                "chunk_id",
                "Unknown"
            )

            content = document.page_content

            context_parts.append(
                f"""
Chunk ID: {chunk_id}
Source: {source}
Page: {page}

Content:
{content}
"""
            )

        return "\n".join(
            context_parts
        )


    def run(
        self,
        question: str
    ):

        # -----------------------------------------
        # Hybrid retrieval
        # -----------------------------------------

        results = self.retriever.retrieve(
            question
        )


        # -----------------------------------------
        # Select final chunks
        # -----------------------------------------

        results = results[:FINAL_TOP_K]


        # -----------------------------------------
        # Build context
        # -----------------------------------------

        context = self.build_context(
            results
        )


        # -----------------------------------------
        # Generate answer
        # -----------------------------------------

        answer = self.generator.generate(
            question,
            context
        )


        return {
            "answer": answer,
            "results": results
        }