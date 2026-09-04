from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from .config import (
    LLM_MODEL,
    TEMPERATURE,
)

# Load environment variables from .env
load_dotenv()


SYSTEM_PROMPT = """
You are a helpful question-answering assistant.

Answer the user's question using ONLY
the provided context.

Do not use outside knowledge.

If the answer cannot be found in the
provided context, say:

"I don't know based on the provided documents."

Do not invent information.

Context:
----------------
{context}
----------------

Question:
{question}
"""


class Generator:

    def __init__(self):

        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=TEMPERATURE
        )

        self.prompt = ChatPromptTemplate.from_template(
            SYSTEM_PROMPT
        )

    def generate(
        self,
        question: str,
        context: str
    ):

        messages = self.prompt.format_messages(
            question=question,
            context=context
        )

        response = self.llm.invoke(
            messages
        )

        return response.content