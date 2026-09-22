
from typing import List

from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate

from .config import LLM_MODEL, TEMPERATURE


class Claim(BaseModel):
    claim: str = Field(
        description="A factual claim made in the answer."
    )

    citations: List[str] = Field(
        description="Chunk IDs that directly support this claim."
    )


class RAGResponse(BaseModel):
    answer: str = Field(
        description="Final grounded answer to the user."
    )

    claims: List[Claim] = Field(
        description="Every factual claim with supporting chunk IDs."
    )


SYSTEM_PROMPT = """
You are a strictly grounded RAG assistant.

Use ONLY the information present in CONTEXT.

Rules:
1. Never use outside knowledge.
2. Never guess or invent information.
3. Every factual statement must be supported by the context.
4. Every factual claim must have at least one citation.
5. Citations must be valid Chunk IDs from the context.
6. Never invent a Chunk ID.
7. If the context does not contain enough information, answer exactly:
"I don't know based on the provided documents."
8. Keep the answer concise.

CONTEXT:
----------------
{context}
----------------

QUESTION:
{question}
"""


class Generator:

    def __init__(self):

        print(f"Initializing Gemini model: {LLM_MODEL}")

        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            max_retries=2
        )

        self.structured_llm = self.llm.with_structured_output(
            RAGResponse
        )

        self.prompt = ChatPromptTemplate.from_template(
            SYSTEM_PROMPT
        )

    def generate(self, question: str, context: str) -> dict:

        print("\n[GENERATOR] Question:", question)
        print("[GENERATOR] Context length:", len(context))

        messages = self.prompt.format_messages(
            question=question,
            context=context
        )

        try:

            response = self.structured_llm.invoke(messages)

            print("[GENERATOR] Gemini response received")

            print("[GENERATOR] Answer:", response.answer)

            return response.model_dump()

        except Exception as e:

            print("\n[GENERATOR ERROR]")
            print(type(e).__name__)
            print(str(e))

            raise

