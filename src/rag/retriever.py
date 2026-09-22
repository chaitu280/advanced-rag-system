from langchain_community.vectorstores import FAISS

from .bm25 import BM25Retriever

from .fusion import reciprocal_rank_fusion

from .reranker import CrossEncoderReranker

from .config import (
    DENSE_TOP_K,
    BM25_TOP_K,
    RERANK_TOP_K,
    RERANK_SCORE_THRESHOLD
)


class HybridRerankRetriever:

    def __init__(
        self,
        vectorstore: FAISS,
        bm25_retriever: BM25Retriever,
        reranker: CrossEncoderReranker
    ):

        self.vectorstore = vectorstore

        self.bm25_retriever = bm25_retriever

        self.reranker = reranker


    def retrieve(
        self,
        query: str
    ):

        # ==================================================
        # STEP 1 — Dense Retrieval
        # ==================================================

        dense_results = (
            self.vectorstore.similarity_search(
                query,
                k=DENSE_TOP_K
            )
        )


        # ==================================================
        # STEP 2 — BM25 Retrieval
        # ==================================================

        bm25_results = (
            self.bm25_retriever.retrieve(
                query,
                top_k=BM25_TOP_K
            )
        )


        # ==================================================
        # STEP 3 — RRF Fusion
        # ==================================================

        fused_results = (
            reciprocal_rank_fusion(
                dense_results,
                bm25_results
            )
        )


        # ==================================================
        # STEP 4 — Cross-Encoder Reranking
        # ==================================================

        reranked_results = (
            self.reranker.rerank(
                query,
                fused_results,
                top_k=RERANK_TOP_K,
                threshold=RERANK_SCORE_THRESHOLD
            )
        )


        return reranked_results