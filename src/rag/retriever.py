from langchain_community.vectorstores import FAISS

from .config import TOP_K


class Retriever:

    def __init__(
        self,
        vectorstore: FAISS
    ):

        self.vectorstore = vectorstore

    def retrieve(
        self,
        query: str
    ):

        documents = self.vectorstore.similarity_search(
            query,
            k=TOP_K
        )

        return documents