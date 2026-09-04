from .retriever import Retriever
from .generator import Generator


class RAGPipeline:

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator
    ):

        self.retriever = retriever
        self.generator = generator

    def build_context(
        self,
        documents
    ):

        context_parts = []

        for document in documents:

            source = document.metadata.get(
                "source",
                "Unknown"
            )

            page = document.metadata.get(
                "page",
                "Unknown"
            )

            content = document.page_content

            context_parts.append(
                f"""
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

        # 1. Retrieve
        documents = self.retriever.retrieve(
            question
        )

        # 2. Build context
        context = self.build_context(
            documents
        )

        # 3. Generate
        answer = self.generator.generate(
            question,
            context
        )

        return {
            "answer": answer,
            "documents": documents
        }