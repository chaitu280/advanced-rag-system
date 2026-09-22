def recall_at_k(
    retrieved_ids,
    expected_ids,
    k
):
    """
    Measures how much of the expected evidence
    was retrieved within top-k.
    """

    retrieved_top_k = set(
        retrieved_ids[:k]
    )

    expected = set(
        expected_ids
    )

    if not expected:
        return 0.0

    hits = retrieved_top_k.intersection(
        expected
    )

    return len(hits) / len(expected)


def precision_at_k(
    retrieved_ids,
    expected_ids,
    k
):
    """
    Measures how much of the retrieved
    top-k evidence is relevant.
    """

    retrieved_top_k = retrieved_ids[:k]

    expected = set(
        expected_ids
    )

    if not retrieved_top_k:
        return 0.0

    hits = sum(
        1
        for chunk_id in retrieved_top_k
        if chunk_id in expected
    )

    return hits / len(
        retrieved_top_k
    )


def reciprocal_rank(
    retrieved_ids,
    expected_ids
):
    """
    Reciprocal Rank for a single query.

    Returns:
        1 / rank of first relevant chunk
    """

    expected = set(
        expected_ids
    )

    for rank, chunk_id in enumerate(
        retrieved_ids,
        start=1
    ):

        if chunk_id in expected:

            return 1 / rank

    return 0.0


def citation_accuracy(
    claims,
    retrieved_ids
):
    """
    Measures whether citations refer
    to retrieved evidence.
    """

    retrieved_ids = set(
        retrieved_ids
    )

    total_citations = 0
    valid_citations = 0

    for claim in claims:

        citations = claim.get(
            "citations",
            []
        )

        for citation in citations:

            total_citations += 1

            if citation in retrieved_ids:

                valid_citations += 1


    if total_citations == 0:

        return 0.0


    return (
        valid_citations
        / total_citations
    )


def citation_coverage(
    claims
):
    """
    Measures the percentage of factual
    claims that have at least one citation.
    """

    if not claims:

        return 0.0


    cited_claims = sum(
        1
        for claim in claims
        if claim.get(
            "citations",
            []
        )
    )


    return (
        cited_claims
        / len(claims)
    )


def faithfulness(
    claim_validation_results
):
    """
    Measures the percentage of claims
    supported by their cited evidence.

    Example:

        [True, True, False]

    Faithfulness:

        2 / 3 = 0.667
    """

    if not claim_validation_results:

        return 0.0


    supported_claims = sum(
        1
        for result in claim_validation_results
        if result is True
    )


    return (
        supported_claims
        / len(claim_validation_results)
    )


def average_metric(
    values
):
    """
    Calculates the average of a metric
    across evaluation questions.
    """

    if not values:

        return 0.0


    return sum(values) / len(values)