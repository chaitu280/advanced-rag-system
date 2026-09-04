from collections import defaultdict

from .config import RRF_K


def reciprocal_rank_fusion(
    dense_results,
    bm25_results
):

    scores = defaultdict(float)

    documents = {}

    # ---------------------------------------------
    # Dense results
    # ---------------------------------------------

    for rank, document in enumerate(
        dense_results,
        start=1
    ):

        chunk_id = get_chunk_id(
            document
        )

        scores[chunk_id] += (
            1 / (RRF_K + rank)
        )

        documents[chunk_id] = document


    # ---------------------------------------------
    # BM25 results
    # ---------------------------------------------

    for item in bm25_results:

        document = item["document"]

        rank = item["rank"]

        chunk_id = get_chunk_id(
            document
        )

        scores[chunk_id] += (
            1 / (RRF_K + rank)
        )

        documents[chunk_id] = document


    # ---------------------------------------------
    # Sort
    # ---------------------------------------------

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for chunk_id, score in ranked:

        results.append({
            "document": documents[chunk_id],
            "score": score
        })

    return results


def get_chunk_id(document):

    return document.metadata.get(
        "chunk_id",
        document.metadata.get(
            "source",
            ""
        )
        + "_"
        + str(
            document.metadata.get(
                "page",
                0
            )
        )
        + "_"
        + str(
            document.metadata.get(
                "start_index",
                0
            )
        )
    )