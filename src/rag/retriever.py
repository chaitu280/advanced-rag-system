from langchain_community.vectorstores import FAISS

from .bm25 import BM25Retriever

from .config import (
    DENSE_TOP_K,
    BM25_TOP_K
)

from .fusion import (
    reciprocal_rank_fusion
)


class HybridRetriever:

    def __init__(
        self,
        vectorstore: FAISS,
        bm25_retriever: BM25Retriever
    ):

        self.vectorstore = vectorstore

        self.bm25_retriever = bm25_retriever


    def retrieve(
        self,
        query: str
    ):

        # -----------------------------------------
        # Dense retrieval
        # -----------------------------------------

        dense_results = (
            self.vectorstore.similarity_search(
                query,
                k=DENSE_TOP_K
            )
        )


        # -----------------------------------------
        # BM25 retrieval
        # -----------------------------------------

        bm25_results = (
            self.bm25_retriever.retrieve(
                query,
                top_k=BM25_TOP_K
            )
        )


        # -----------------------------------------
        # RRF fusion
        # -----------------------------------------

        fused_results = (
            reciprocal_rank_fusion(
                dense_results,
                bm25_results
            )
        )


        return fused_results