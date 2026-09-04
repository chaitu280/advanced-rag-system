import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, documents):

        self.documents = documents

        tokenized_documents = [
            self.tokenize(doc.page_content)
            for doc in documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

    @staticmethod
    def tokenize(text):

        return re.findall(
            r"\b\w+\b",
            text.lower()
        )

    def retrieve(
        self,
        query,
        top_k=10
    ):

        query_tokens = self.tokenize(
            query
        )

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []

        for index in ranked_indices[:top_k]:

            results.append({
                "document": self.documents[index],
                "score": float(scores[index]),
                "rank": len(results) + 1
            })

        return results

    def save(self, path: Path):

        with open(path, "wb") as file:

            pickle.dump(
                self,
                file
            )

    @staticmethod
    def load(path: Path):

        with open(path, "rb") as file:

            return pickle.load(file)