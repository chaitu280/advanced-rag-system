from sentence_transformers import CrossEncoder

from .config import RERANKER_MODEL


class CrossEncoderReranker:

    def __init__(self):

        self.model = CrossEncoder(
            RERANKER_MODEL
        )

    def rerank(
        self,
        query: str,
        documents,
        top_k: int,
        threshold: float
    ):

        if not documents:
            return []

        pairs = [
            (
                query,
                item["document"].page_content
            )
            for item in documents
        ]

        scores = self.model.predict(pairs)

        reranked = []

        for item, score in zip(
            documents,
            scores
        ):

            score = float(score)

            if score >= threshold:

                reranked.append(
                    {
                        "document": item["document"],
                        "rrf_score": item["score"],
                        "rerank_score": score
                    }
                )

        reranked.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked[:top_k]